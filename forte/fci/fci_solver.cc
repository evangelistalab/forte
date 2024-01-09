/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <numeric>

#include "psi4/libpsi4util/process.h"

#include "integrals/active_space_integrals.h"
#include "sparse_ci/ci_spin_adaptation.h"
#include "helpers/davidson_liu_solver.h"

#include "fci_string_lists.h"
#include "helpers/printing.h"
#include "fci_string_address.h"
#include "fci_vector.h"

#include "fci_solver.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "psi4/psi4-dec.h"

namespace forte {

class MOSpaceInfo;

FCISolver::FCISolver(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints),
      active_dim_(mo_space_info->dimension("ACTIVE")), nirrep_(as_ints->ints()->nirrep()),
      symmetry_(state.irrep()) {
    // TODO: read this info from the base class
    na_ = state.na() - core_mo_.size() - mo_space_info->size("FROZEN_DOCC");
    nb_ = state.nb() - core_mo_.size() - mo_space_info->size("FROZEN_DOCC");
}

void FCISolver::set_maxiter_davidson(int value) { maxiter_davidson_ = value; }

void FCISolver::set_ndets_per_guess_state(size_t value) { ndets_per_guess_ = value; }

void FCISolver::set_guess_per_root(int value) { guess_per_root_ = value; }

void FCISolver::set_collapse_per_root(int value) { collapse_per_root_ = value; }

void FCISolver::set_subspace_per_root(int value) { subspace_per_root_ = value; }

void FCISolver::set_spin_adapt(bool value) { spin_adapt_ = value; }

void FCISolver::set_spin_adapt_full_preconditioner(bool value) {
    spin_adapt_full_preconditioner_ = value;
}

void FCISolver::set_test_rdms(bool value) { test_rdms_ = value; }

void FCISolver::set_print_no(bool value) { print_no_ = value; }

void FCISolver::copy_state_into_fci_vector(int root, std::shared_ptr<FCIVector> C) {
    // grab a row of the eigenvector matrix and put it into the FCIVector
    std::shared_ptr<psi::Vector> psi_vector;
    if (spin_adapt_) {
        auto eig = eigen_vecs_->get_row(0, root);
        psi_vector = std::make_shared<psi::Vector>(spin_adapter_->ndet());
        spin_adapter_->csf_C_to_det_C(eig, psi_vector);
    } else {
        psi_vector = eigen_vecs_->get_row(0, root);
    }
    C->copy(psi_vector);
}

std::shared_ptr<psi::Matrix> FCISolver::evecs() { return eigen_vecs_; }

std::shared_ptr<FCIStringLists> FCISolver::lists() { return lists_; }

int FCISolver::symmetry() { return symmetry_; }

void FCISolver::startup() {
    // Create the string lists

#if USE_GAS_LISTS
    std::vector<int> gas_size;
    std::vector<int> gas_min;
    std::vector<int> gas_max;
    for (size_t gas_count = 0; gas_count < 6; gas_count++) {
        std::string space = "GAS" + std::to_string(gas_count + 1);
        int orbital_maximum = mo_space_info_->size(space);
        gas_size.push_back(orbital_maximum);
    }
    for (auto n : state_.gas_min()) {
        gas_min.push_back(n);
    }
    for (auto n : state_.gas_max()) {
        gas_max.push_back(n);
    }
    lists_ = std::make_shared<FCIStringLists>(mo_space_info_, na_, nb_, print_, gas_size, gas_min,
                                              gas_max);
#else
    lists_ = std::make_shared<FCIStringLists>(active_dim_, core_mo_, active_mo_, na_, nb_, print_);
#endif

    nfci_dets_ = 0;
    for (int h = 0; h < nirrep_; ++h) {
        size_t nastr = lists_->alfa_address()->strpcls(h);
        size_t nbstr = lists_->beta_address()->strpcls(h ^ symmetry_);
        nfci_dets_ += nastr * nbstr;
    }

    // Create the spin adapter
    if (spin_adapt_) {
        spin_adapter_ = std::make_shared<SpinAdapter>(state().multiplicity() - 1,
                                                      state().twice_ms(), lists_->ncmo());
        dets_ = lists_->make_determinants(symmetry_);
        spin_adapter_->prepare_couplings(dets_);
    }

    if (print_ >= PrintLevel::Default) {
        table_printer printer;
        printer.add_int_data({{"Number of determinants", nfci_dets_},
                              {"Symmetry", symmetry_},
                              {"Multiplicity", state().multiplicity()},
                              {"Number of roots", nroot_},
                              {"Target root", root_}});
        printer.add_bool_data({{"Spin adapt", spin_adapt_}});
        std::string table = printer.get_table("FCI Solver");
        psi::outfile->Printf("%s", table.c_str());
    }
}

void FCISolver::set_options(std::shared_ptr<ForteOptions> options) {
    set_e_convergence(options->get_double("E_CONVERGENCE"));
    set_r_convergence(options->get_double("R_CONVERGENCE"));
    set_spin_adapt(options->get_bool("CI_SPIN_ADAPT"));
    set_spin_adapt_full_preconditioner(options->get_bool("CI_SPIN_ADAPT_FULL_PRECONDITIONER"));
    set_test_rdms(options->get_bool("FCI_TEST_RDMS"));

    set_root(options->get_int("ROOT"));

    set_guess_per_root(options->get_int("DL_GUESS_PER_ROOT"));
    set_ndets_per_guess_state(options->get_int("DL_DETS_PER_GUESS"));
    set_collapse_per_root(options->get_int("DL_COLLAPSE_PER_ROOT"));
    set_subspace_per_root(options->get_int("DL_SUBSPACE_PER_ROOT"));
    set_maxiter_davidson(options->get_int("DL_MAXITER"));

    set_print(int_to_print_level(options->get_int("PRINT")));
}

/*
 * See Appendix A in J. Comput. Chem. 2001 vol. 22 (13) pp. 1574-1589
 */
double FCISolver::compute_energy() {
    local_timer t;
    startup();

    FCIVector::allocate_temp_space(lists_, print_);

    C_ = std::make_shared<FCIVector>(lists_, symmetry_);
    T_ = std::make_shared<FCIVector>(lists_, symmetry_);
    C_->set_print(print_);

    // Compute the size of the determinant space and the basis used by the Davidson solver
    size_t det_size = C_->size();
    size_t basis_size = spin_adapt_ ? spin_adapter_->ncsf() : det_size;

    // Create the vectors that stores the b and sigma vectors in the determinant basis
    auto b = std::make_shared<psi::Vector>("b", det_size);
    auto sigma = std::make_shared<psi::Vector>("sigma", det_size);

    // Optionally create the vectors that stores the b and sigma vectors in the CSF basis
    auto b_basis = b;
    auto sigma_basis = sigma;

    if (spin_adapt_) {
        b_basis = std::make_shared<psi::Vector>("b", basis_size);
        sigma_basis = std::make_shared<psi::Vector>("sigma", basis_size);
    }

    // if not allocate, create the DL solver
    bool first_run = false;
    if (dl_solver_ == nullptr) {
        dl_solver_ = std::make_shared<DavidsonLiuSolver>(basis_size, nroot_, collapse_per_root_,
                                                         subspace_per_root_);
        dl_solver_->set_e_convergence(e_convergence_);
        dl_solver_->set_r_convergence(r_convergence_);
        dl_solver_->set_print_level(print_);
        dl_solver_->set_maxiter(maxiter_davidson_);
        first_run = true;
    }

    // determine the number of guess vectors
    const size_t num_guess_states = std::min(guess_per_root_ * nroot_, basis_size);

    auto Hdiag_vec =
        spin_adapt_ ? form_Hdiag_csf(as_ints_, spin_adapter_) : form_Hdiag_det(as_ints_);
    dl_solver_->add_h_diag(Hdiag_vec);

    // The first time we run Form the diagonal of the Hamiltonian and the initial guess
    if (spin_adapt_) {
        if (first_run) {
            auto guesses = initial_guess_csf(Hdiag_vec, num_guess_states);
            dl_solver_->add_guesses(guesses);
        }
    } else {
        bool use_initial_guess = (num_guess_states * ndets_per_guess_ >= det_size);
        if (first_run or use_initial_guess) {
            dl_solver_->reset();
            auto [guesses, bad_roots] = initial_guess_det(Hdiag_vec, num_guess_states, as_ints_);
            dl_solver_->add_guesses(guesses);
            dl_solver_->add_project_out_vectors(bad_roots);
        }
    }

    // Print the initial guess
    auto sigma_builder = [this, &b_basis, &b, &sigma, &sigma_basis](std::span<double> b_span,
                                                                    std::span<double> sigma_span) {
        // copy the b vector
        size_t basis_size = b_span.size();
        for (size_t I = 0; I < basis_size; ++I) {
            b_basis->set(I, b_span[I]);
        }
        if (spin_adapt_) {
            // Compute sigma in the CSF basis and convert it to the determinant basis
            spin_adapter_->csf_C_to_det_C(b_basis, b);
            C_->copy(b);
            C_->Hamiltonian(*T_, as_ints_);
            T_->copy_to(sigma);
            spin_adapter_->det_C_to_csf_C(sigma, sigma_basis);
        } else {
            // Compute sigma in the determinant basis
            C_->copy(b_basis);
            C_->Hamiltonian(*T_, as_ints_);
            T_->copy_to(sigma_basis);
        }
        for (size_t I = 0; I < basis_size; ++I) {
            sigma_span[I] = sigma_basis->get(I);
        }
    };

    // Run the Davidson-Liu solver
    dl_solver_->add_sigma_builder(sigma_builder);

    auto converged = dl_solver_->solve();
    if (not converged) {
        throw std::runtime_error(
            "Davidson-Liu solver did not converge.\nPlease try to increase the number of "
            "Davidson-Liu iterations (DL_MAXITER). You can also try to increase:\n - the "
            "maximum "
            "size of the subspace (DL_SUBSPACE_PER_ROOT)"
            "\n - the number of guess states (DL_GUESS_PER_ROOT)");
        return false;
    }

    // Copy eigenvalues and eigenvectors from the Davidson-Liu solver
    evals_ = dl_solver_->eigenvalues();
    energies_ = std::vector<double>(nroot_, 0.0);
    spin2_ = std::vector<double>(nroot_, 0.0);
    for (size_t r = 0; r < nroot_; r++) {
        energies_[r] = evals_->get(r);
        b_basis = dl_solver_->eigenvector(r);
        if (spin_adapt_) {
            spin_adapter_->csf_C_to_det_C(b_basis, b);
        } else {
            b = b_basis;
        }
        C_->copy(b);
        spin2_[r] = C_->compute_spin2();
    }
    eigen_vecs_ = dl_solver_->eigenvectors();

    // Print determinants
    if (print_ >= PrintLevel::Default) {
        print_solutions(100, b, b_basis, dl_solver_);
    }

    // Optionally, test the RDMs
    if (test_rdms_) {
        test_rdms(b, b_basis, dl_solver_);
    }

    energy_ = dl_solver_->eigenvalues()->get(root_);
    psi::Process::environment.globals["CURRENT ENERGY"] = energy_;
    psi::Process::environment.globals["FCI ENERGY"] = energy_;

    if (print_ >= PrintLevel::Default) {
        psi::outfile->Printf("\n    Time for FCI: %20.12f", t.get());
    }
    return energy_;
}

void FCISolver::print_solutions(size_t sample_size, std::shared_ptr<psi::Vector> b,
                                std::shared_ptr<psi::Vector> b_basis,
                                std::shared_ptr<DavidsonLiuSolver> dls) {
    for (size_t r = 0; r < nroot_; ++r) {
        psi::outfile->Printf("\n\n  ==> Root No. %d <==\n", r);

        b_basis = dls->eigenvector(r);
        if (spin_adapt_) {
            spin_adapter_->csf_C_to_det_C(b_basis, b);
        } else {
            b = b_basis;
        }
        C_->copy(b);
        std::vector<std::tuple<double, double, size_t, size_t, size_t>> dets_config =
            C_->max_abs_elements(sample_size);

        for (auto& det_config : dets_config) {
            double ci_abs, ci;
            size_t h, add_Ia, add_Ib;
            std::tie(ci_abs, ci, h, add_Ia, add_Ib) = det_config;

            if (ci_abs < 0.01)
                continue;

            auto Ia_v = lists_->alfa_str(h, add_Ia);
            auto Ib_v = lists_->beta_str(h ^ symmetry_, add_Ib);

            psi::outfile->Printf("\n    ");
            size_t offset = 0;
            for (int h = 0; h < nirrep_; ++h) {
                for (int k = 0; k < active_dim_[h]; ++k) {
                    size_t i = k + offset;
                    bool a = Ia_v[i];
                    bool b = Ib_v[i];
                    if (a == b) {
                        psi::outfile->Printf("%d", a ? 2 : 0);
                    } else {
                        psi::outfile->Printf("%c", a ? 'a' : 'b');
                    }
                }
                if (active_dim_[h] != 0)
                    psi::outfile->Printf(" ");
                offset += active_dim_[h];
            }
            psi::outfile->Printf("%15.8f", ci);
        }

        double root_energy = dl_solver_->eigenvalues()->get(r);

        psi::outfile->Printf("\n\n    Total Energy: %20.12f, <S^2>: %8.6f", root_energy, spin2_[r]);
    }
}

void FCISolver::test_rdms(std::shared_ptr<psi::Vector> b, std::shared_ptr<psi::Vector> b_basis,
                          std::shared_ptr<DavidsonLiuSolver> dls) {
    b_basis = dls->eigenvector(root_);
    if (spin_adapt_) {
        spin_adapter_->csf_C_to_det_C(b_basis, b);
    } else {
        b = b_basis;
    }
    C_->copy(b);
    if (print_ >= PrintLevel::Verbose) {
        std::string title_rdm = "Computing RDMs for Root No. " + std::to_string(root_);
        print_h2(title_rdm);
    }
    int max_rdm_level = 3;
    auto rdms = C_->compute_rdms(*C_, *C_, max_rdm_level, RDMsType::spin_dependent);
    C_->test_rdms(*C_, *C_, max_rdm_level, RDMsType::spin_dependent, rdms);
}

std::shared_ptr<psi::Matrix> FCISolver::ci_wave_functions() {
    if (eigen_vecs_ == nullptr)
        return std::make_shared<psi::Matrix>();

    auto evecs = std::make_shared<psi::Matrix>("FCI Eigenvectors", eigen_vecs_->ncol(), nroot_);
    for (int i = 0, size = static_cast<int>(nroot_); i < size; ++i) {
        evecs->set_column(0, i, eigen_vecs_->get_row(0, i));
    }
    return evecs;
}

} // namespace forte
