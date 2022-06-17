
/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"

#include "base_classes/forte_options.h"
#include "mrpt2.h"
#include "helpers/timer.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

using namespace psi;

namespace forte {

MRPT2::MRPT2(std::shared_ptr<ForteOptions> options, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
             std::shared_ptr<MOSpaceInfo> mo_space_info, DeterminantHashVec& reference,
             psi::SharedMatrix evecs, psi::SharedVector evals, int nroot)
    : reference_(reference), options_(options), as_ints_(as_ints), mo_space_info_(mo_space_info),
      evecs_(evecs), evals_(evals), nroot_(nroot) {
    outfile->Printf("\n  ==> Full EN-MRPT2 correction  <==");
    //    print_method_banner(
    //        {"Deterministic MR-PT2", "Jeff Schriber"});
    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");
}

MRPT2::~MRPT2() {}

std::vector<double> MRPT2::compute_energy() {
    outfile->Printf("\n\n  Computing PT2 correction from %zu reference determinants",
                    reference_.size());

    std::vector<double> pt2_en;

    local_timer en;
    for (int n = 0; n < nroot_; ++n) {
        pt2_en.push_back(compute_pt2_energy(n));
        outfile->Printf("\n  Root %d PT2 energy:  %1.12f", n, pt2_en[n]);
    }
    //  double scalar = as_ints_->scalar_energy() + molecule_->nuclear_repulsion_energy();
    //  double energy = pt2_energy + scalar + evals_->get(0);
    outfile->Printf("\n  Full PT2 computation took %1.6f s", en.get());
    //  outfile->Printf("\n  Total energy:  %1.12f", energy);

    return pt2_en;
}

double MRPT2::compute_pt2_energy(int root) {
    double energy = 0.0;
    const size_t n_dets = reference_.size();
    int nmo = as_ints_->nmo();
    double max_mem = options_->get_double("PT2_MAX_MEM");

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
            energy += energy_kernel(bin, nbin, root);
        }
    }
    return energy;
}

double MRPT2::energy_kernel(int bin, int nbin, int root) {
    size_t nact = mo_space_info_->size("ACTIVE");
    double E_0 = evals_->get(root);
    double energy = 0.0;
    const size_t n_dets = reference_.size();
    const det_hashvec& dets = reference_.wfn_hash();
    det_hash<double> A_I;
    for (size_t I = 0; I < n_dets; ++I) {
        double c_I = evecs_->get(I, root);
        const Determinant& det = dets[I];
        std::vector<int> aocc = det.get_alfa_occ(nact);
        std::vector<int> bocc = det.get_beta_occ(nact);
        std::vector<int> avir = det.get_alfa_vir(nact);
        std::vector<int> bvir = det.get_beta_vir(nact);

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
                            as_ints_->slater_rules_single_alpha(new_det, ii, aa) * c_I;
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
                        double coupling = as_ints_->slater_rules_single_beta(new_det, ii, aa) * c_I;
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

                                double coupling = sign * c_I * as_ints_->tei_ab(ii, jj, aa, bb);
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
                                double coupling = sign * c_I * as_ints_->tei_aa(ii, jj, aa, bb);
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
                                double coupling = sign * c_I * as_ints_->tei_bb(ii, jj, aa, bb);
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
        energy += (det.second * det.second) / (E_0 - as_ints_->energy(det.first));
    }
    return energy;
}
} // namespace forte
