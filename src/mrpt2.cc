
/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "forte_options.h"
#include "mrpt2.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

namespace psi {
namespace forte {

void set_PT2_options(ForteOptions& foptions) {
    /*- Maximum size of the determinant hash (GB)-*/
    foptions.add_double("PT2_MAX_MEM", 1.0, " Maximum size of the determinant hash (GB)");
}

MRPT2::MRPT2(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
             std::shared_ptr<MOSpaceInfo> mo_space_info, DeterminantHashVec& reference,
             SharedMatrix evecs, SharedVector evals)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info), reference_(reference),
      evecs_(evecs), evals_(evals) {
    shallow_copy(ref_wfn);
    //    print_method_banner(
    //        {"Deterministic MR-PT2", "Jeff Schriber"});
    startup();
}

MRPT2::~MRPT2() {}

void MRPT2::startup() {
    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");

    // Define the correlated space
    auto active_mo = mo_space_info_->get_corr_abs_mo("ACTIVE");

    fci_ints_ = std::make_shared<FCIIntegrals>(ints_, active_mo,
                                               mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));

    // Set the integrals
    ambit::Tensor tei_active_aa = ints_->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab = ints_->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb = ints_->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);

    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);

    fci_ints_->compute_restricted_one_body_operator();

    nroot_ = options_.get_int("NROOT");
    multiplicity_ = options_.get_int("MULTIPLICITY");
    screen_thresh_ = options_.get_double("ACI_PRESCREEN_THRESHOLD");
}

double MRPT2::compute_energy() {
    outfile->Printf("\n\n  Computing PT2 correction from %zu reference determinants",
                    reference_.size());

    Timer en;
    double pt2_energy = compute_pt2_energy();
    //  double scalar = fci_ints_->scalar_energy() + molecule_->nuclear_repulsion_energy();
    //  double energy = pt2_energy + scalar + evals_->get(0);
    outfile->Printf("\n  PT2 computation took %1.6f s", en.get());
    outfile->Printf("\n  PT2 energy:  %1.12f", pt2_energy);
    //  outfile->Printf("\n  Total energy:  %1.12f", energy);

    return pt2_energy;
}

double MRPT2::compute_pt2_energy() {
    double energy = 0.0;
    const size_t n_dets = reference_.size();
    int nmo = fci_ints_->nmo();
    double max_mem = options_.get_double("PT2_MAX_MEM");

    size_t guess_size = n_dets * nmo * nmo;
    double nbyte = (1073741824 * max_mem) / (sizeof(double));

    int nbin = static_cast<int>(std::ceil(guess_size / (nbyte)));

#pragma omp parallel reduction(+ : energy)
    {
        int tid = omp_get_thread_num();
        int ntds = omp_get_num_threads();

        if ((ntds > nbin)) {
            nbin = ntds;
        }

        if (tid == 0) {
            outfile->Printf("\n  Number of bins for exitation space:  %d", nbin);
            outfile->Printf("\n  Number of threads: %d", ntds);
        }
        int batch_size = nbin / ntds;
        batch_size += (tid < (nbin % ntds)) ? 1 : 0;

        int start_idx = (tid < (nbin % ntds))
                            ? tid * batch_size
                            : (nbin % ntds) * (batch_size + 1) + (tid - (nbin % ntds)) * batch_size;
        int end_idx = start_idx + batch_size;

        for (int bin = start_idx; bin < end_idx; ++bin) {
            energy += energy_kernel(bin, nbin);
        }
    }
    return energy;
}

double MRPT2::energy_kernel(int bin, int nbin) {
    double E_0 = evals_->get(0);
    double energy = 0.0;
    const size_t n_dets = reference_.size();
    const det_hashvec& dets = reference_.wfn_hash();
    det_hash<double> A_I;
    for (size_t I = 0; I < n_dets; ++I) {
        double c_I = evecs_->get(I, 0);
        const Determinant& det = dets[I];
        std::vector<int> aocc = det.get_alfa_occ();
        std::vector<int> bocc = det.get_beta_occ();
        std::vector<int> avir = det.get_alfa_vir();
        std::vector<int> bvir = det.get_beta_vir();

        int noalpha = aocc.size();
        int nobeta = bocc.size();
        int nvalpha = avir.size();
        int nvbeta = bvir.size();
        Determinant new_det(det);

        // Generate alpha excitations
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    new_det = det;
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    if (reference_.has_det(new_det))
                        continue;

                    // Check if the determinant goes in this bin
                    size_t hash_val = Determinant::Hash()(new_det);
                    if ((hash_val % nbin) == bin) {
                        double coupling =
                            fci_ints_->slater_rules_single_alpha(new_det, ii, aa) * c_I;
                        if (A_I.find(new_det) != A_I.end()) {
                            coupling += A_I[new_det];
                        }
                        A_I[new_det] = coupling;
                    }
                }
            }
        }
        // Generate beta excitations
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    new_det = det;
                    new_det.set_beta_bit(ii, false);
                    new_det.set_beta_bit(aa, true);
                    if (reference_.has_det(new_det))
                        continue;
                    // Check if the determinant goes in this bin
                    size_t hash_val = Determinant::Hash()(new_det);
                    if ((hash_val % nbin) == bin) {
                        double coupling =
                            fci_ints_->slater_rules_single_beta(new_det, ii, aa) * c_I;
                        if (A_I.find(new_det) != A_I.end()) {
                            coupling += A_I[new_det];
                        }
                        A_I[new_det] = coupling;
                    }
                }
            }
        }
        // Generate ab excitations
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int j = 0; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    for (int b = 0; b < nvbeta; ++b) {
                        int bb = bvir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                             mo_symmetry_[bb]) == 0) {
                            new_det = det;
                            double sign = new_det.double_excitation_ab(ii, jj, aa, bb);
                            if (reference_.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = Determinant::Hash()(new_det);
                            if ((hash_val % nbin) == bin) {

                                double coupling = sign * c_I * fci_ints_->tei_ab(ii, jj, aa, bb);
                                if (A_I.find(new_det) != A_I.end()) {
                                    coupling += A_I[new_det];
                                }
                                A_I[new_det] = coupling;
                            }
                        }
                    }
                }
            }
        }
        // Generate aa excitations
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int j = i + 1; j < noalpha; ++j) {
                int jj = aocc[j];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    for (int b = a + 1; b < nvalpha; ++b) {
                        int bb = avir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                             mo_symmetry_[bb]) == 0) {
                            new_det = det;
                            double sign = new_det.double_excitation_aa(ii, jj, aa, bb);
                            if (reference_.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = Determinant::Hash()(new_det);
                            if ((hash_val % nbin) == bin) {
                                double coupling = sign * c_I * fci_ints_->tei_aa(ii, jj, aa, bb);
                                if (A_I.find(new_det) != A_I.end()) {
                                    coupling += A_I[new_det];
                                }
                                A_I[new_det] = coupling;
                            }
                        }
                    }
                }
            }
        }
        // Generate bb excitations
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int j = i + 1; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    for (int b = a + 1; b < nvbeta; ++b) {
                        int bb = bvir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                             mo_symmetry_[bb]) == 0) {
                            new_det = det;
                            double sign = new_det.double_excitation_bb(ii, jj, aa, bb);
                            if (reference_.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = Determinant::Hash()(new_det);
                            if ((hash_val % nbin) == bin) {
                                double coupling = sign * c_I * fci_ints_->tei_bb(ii, jj, aa, bb);
                                if (A_I.find(new_det) != A_I.end()) {
                                    coupling += A_I[new_det];
                                }
                                A_I[new_det] = coupling;
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto& det : A_I) {
        energy += (det.second * det.second) / (E_0 - fci_ints_->energy(det.first));
    }
    return energy;
}
}
}
