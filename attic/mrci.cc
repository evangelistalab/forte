
/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"


#include "helpers/printing.h"
#include "mrci.h"
#include "helpers/timer.h"

using namespace psi;

namespace forte {

MRCI::MRCI(psi::SharedWavefunction ref_wfn, std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info, DeterminantHashVec& reference)
    : Wavefunction(options), ints_(ints),reference_(reference), mo_space_info_(mo_space_info) {
    shallow_copy(ref_wfn);
    ref_wfn_ = ref_wfn;
    print_method_banner({"Uncontracted MR-CISD", "Jeff Schriber"});
    startup();
}

MRCI::~MRCI() {}

void MRCI::startup() {
    mo_symmetry_ = mo_space_info_->symmetry("GENERALIZED PARTICLE");

    // Define the correlated space
    auto correlated_mo = mo_space_info_->corr_absolute_mo("GENERALIZED PARTICLE");
    std::sort(correlated_mo.begin(), correlated_mo.end());

    fci_ints_ = std::make_shared<ActiveSpaceIntegrals>(ints_, correlated_mo,
                                               mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC"));

    // Set the integrals
    ambit::Tensor tei_active_aa =
        ints_->aptei_aa_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo);
    ambit::Tensor tei_active_ab =
        ints_->aptei_ab_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo);
    ambit::Tensor tei_active_bb =
        ints_->aptei_bb_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo);

    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);

    fci_ints_->compute_restricted_one_body_operator();

    nroot_ = options_.get_int("NROOT");
    multiplicity_ = options_.get_int("MULTIPLICITY");

    diag_method_ = DLSolver;
}

double MRCI::compute_energy() {

    upcast_reference();
    WFNOperator op(mo_symmetry_, fci_ints_);

    outfile->Printf("\n  Adding single and double excitations ...");
    local_timer add;
    get_excited_determinants();
    outfile->Printf("\n  Excitations took %1.5f s", add.get());
    outfile->Printf("\n  psi::Dimension of model space: %zu", reference_.size());

    std::string sigma_alg = options_.get_str("SIGMA_BUILD_TYPE");

    if (sigma_alg == "HZ") {
        op.op_lists(reference_);
        op.tp_lists(reference_);
    } else {
        op.build_strings(reference_);
        op.op_s_lists(reference_);
        op.tp_s_lists(reference_);
    }

    // Diagonalize MR-CISD Hamiltonian
    psi::SharedMatrix evecs;
    psi::SharedVector evals;

    SparseCISolver sparse_solver(fci_ints_);

    // set options
    sparse_solver.set_sigma_method(sigma_alg);
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(true);
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_spin_project_full(false);

    sparse_solver.diagonalize_hamiltonian_map(reference_, op, evals, evecs, nroot_, multiplicity_,
                                              diag_method_);

    std::vector<double> energy(nroot_);
    double scalar = fci_ints_->scalar_energy() +
                    molecule_->nuclear_repulsion_energy(ref_wfn_->get_dipole_field_strength());

    outfile->Printf("\n");
    for (int n = 0; n < nroot_; ++n) {
        energy[n] = scalar + evals->get(n);
        outfile->Printf("\n  MR-CISD energy root %d: %1.13f Eh", n, energy[n]);
    }

    psi::Process::environment.globals["MRCISD ENERGY"] = energy[0];

    return energy[0];
}

void MRCI::get_excited_determinants() {
    // Only excite into the restricted uocc

    auto external_mo = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    size_t nact = mo_space_info_->size("ACTIVE");

    DeterminantHashVec external;
    external.clear();

    int n_ext = external_mo.size();

    const auto& internal = reference_.wfn_hash();
    for (const auto& det : internal) {

        std::vector<int> aocc = det.get_alfa_occ(nact + n_ext); // <- TODO: check if correct
        std::vector<int> bocc = det.get_beta_occ(nact + n_ext); // <- TODO: check if correct

        int noalfa = aocc.size();
        int nobeta = bocc.size();

        Determinant new_det(det);

        // Single Alpha
        for (int i = 0; i < noalfa; ++i) {
            int ii = aocc[i];
            for (int a = 0; a < n_ext; ++a) {
                int aa = external_mo[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    new_det = det;
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    external.add(new_det);
                }
            }
        }
        // Single Beta
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0; a < n_ext; ++a) {
                int aa = external_mo[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    new_det = det;
                    new_det.set_beta_bit(ii, false);
                    new_det.set_beta_bit(aa, true);
                    external.add(new_det);
                }
            }
        }
        // Double Alpha
        for (int i = 0; i < noalfa; ++i) {
            int ii = aocc[i];
            for (int j = i + 1; j < noalfa; ++j) {
                int jj = aocc[j];
                for (int a = 0; a < n_ext; ++a) {
                    int aa = external_mo[a];
                    for (int b = a + 1; b < n_ext; ++b) {
                        int bb = external_mo[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa] ^ mo_symmetry_[jj] ^
                             mo_symmetry_[bb]) == 0) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(jj, false);
                            new_det.set_alfa_bit(aa, true);
                            new_det.set_alfa_bit(bb, true);
                            external.add(new_det);
                        }
                    }
                }
            }
        }
        // Double Beta
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int j = i + 1; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < n_ext; ++a) {
                    int aa = external_mo[a];
                    for (int b = a + 1; b < n_ext; ++b) {
                        int bb = external_mo[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa] ^ mo_symmetry_[jj] ^
                             mo_symmetry_[bb]) == 0) {
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(jj, false);
                            new_det.set_beta_bit(aa, true);
                            new_det.set_beta_bit(bb, true);
                            external.add(new_det);
                        }
                    }
                }
            }
        }
        // Alpha/Beta
        for (int i = 0; i < noalfa; ++i) {
            int ii = aocc[i];
            for (int j = 0; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < n_ext; ++a) {
                    int aa = external_mo[a];
                    for (int b = 0; b < n_ext; ++b) {
                        int bb = external_mo[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa] ^ mo_symmetry_[jj] ^
                             mo_symmetry_[bb]) == 0) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_beta_bit(jj, false);
                            new_det.set_alfa_bit(aa, true);
                            new_det.set_beta_bit(bb, true);
                            external.add(new_det);
                        }
                    }
                }
            }
        }
    }

    // const auto dets = external.determinants();
    //    for( auto& det : dets) outfile->Printf("\n  %s", det.str().c_str());

    outfile->Printf("\n  Added %zu determinants from external space", external.size());
    reference_.merge(external);
}

void MRCI::upcast_reference() {
    //    auto mo_sym = mo_space_info_->symmetry("GENERALIZED PARTICLE");

    //    psi::Dimension old_dim = mo_space_info_->dimension("ACTIVE");
    //    psi::Dimension new_dim = mo_space_info_->dimension("GENERALIZED PARTICLE");
    //    size_t nact = mo_space_info_->size("ACTIVE");
    //    size_t ncorr = mo_space_info_->size("GENERALIZED PARTICLE");
    //    int n_irrep = old_dim.n();

    //    det_hashvec ref_dets;
    //    ref_dets.swap(reference_.wfn_hash());
    //    reference_.clear();

    //    // Compute shifts
    //    std::vector<int> shift(n_irrep, 0);
    //    if (n_irrep > 1) {
    //        for (int n = 1; n < n_irrep; ++n) {
    //            shift[n] += new_dim[n - 1] - old_dim[n - 1] + shift[n - 1];
    //        }
    //    }
    //    int b_shift = ncorr - nact;

    //    for (size_t I = 0, max = ref_dets.size(); I < max; ++I) {
    //        Determinant det(ref_dets[I]);

    //        // First beta
    //        for (int n = n_irrep - 1; n >= 0; --n) {
    //            int min = 0;
    //            for (int m = 0; m < n; ++m) {
    //                min += old_dim[m];
    //            }
    //            for (int pos = nact + min + old_dim[n] - 1; pos >= min + nact; --pos) {
    //                det.set_beta_bit(pos + b_shift + shift[n], );
    //                det.bits_[pos + b_shift + shift[n]] = det.bits_[pos];
    //                det.bits_[pos] = 0;
    //            }
    //        }
    //        // Then alpha
    //        for (int n = n_irrep - 1; n >= 0; --n) {
    //            int min = 0;
    //            for (int m = 0; m < n; ++m) {
    //                min += old_dim[m];
    //            }
    //            for (int pos = min + old_dim[n] - 1; pos >= min; --pos) {
    //                det.bits_[pos + shift[n]] = det.bits_[pos];

    //                if (n > 0)
    //                    det.bits_[pos] = 0;
    //            }
    //        }

    //        reference_.add(det);
    //    }
}
}
