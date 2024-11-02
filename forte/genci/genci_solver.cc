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

#include "base_classes/forte_options.h"

#include "integrals/active_space_integrals.h"
#include "sparse_ci/ci_spin_adaptation.h"
#include "helpers/davidson_liu_solver.h"

#include "genci_solver.h"
#include "genci_string_lists.h"
#include "helpers/printing.h"
#include "genci_string_address.h"

#include "genci_vector.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "psi4/psi4-dec.h"

namespace forte {

class MOSpaceInfo;

GenCISolver::GenCISolver(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
                         std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints),
      active_dim_(mo_space_info->dimension("ACTIVE")), nirrep_(as_ints->ints()->nirrep()),
      symmetry_(state.irrep()) {
    // TODO: read this info from the base class
    na_ = state.na() - mo_space_info->size("INACTIVE_DOCC");
    nb_ = state.nb() - mo_space_info->size("INACTIVE_DOCC");
}

void GenCISolver::set_maxiter_davidson(int value) { maxiter_davidson_ = value; }

void GenCISolver::set_ndets_per_guess_state(size_t value) { ndets_per_guess_ = value; }

void GenCISolver::set_guess_per_root(int value) { guess_per_root_ = value; }

void GenCISolver::set_collapse_per_root(int value) { collapse_per_root_ = value; }

void GenCISolver::set_subspace_per_root(int value) { subspace_per_root_ = value; }

void GenCISolver::set_spin_adapt(bool value) { spin_adapt_ = value; }

void GenCISolver::set_spin_adapt_full_preconditioner(bool value) {
    spin_adapt_full_preconditioner_ = value;
}

void GenCISolver::set_test_rdms(bool value) { test_rdms_ = value; }

void GenCISolver::set_print_no(bool value) { print_no_ = value; }

void GenCISolver::copy_state_into_fci_vector(int root, std::shared_ptr<GenCIVector> C) {
    // grab a row of the eigenvector matrix and put it into the GenCIVector
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

std::shared_ptr<psi::Matrix> GenCISolver::evecs() { return eigen_vecs_; }

std::shared_ptr<GenCIStringLists> GenCISolver::lists() { return lists_; }

int GenCISolver::symmetry() { return symmetry_; }

void GenCISolver::startup() {
    // Create the string lists

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
    lists_ = std::make_shared<GenCIStringLists>(mo_space_info_, na_, nb_, symmetry_, print_,
                                                gas_min, gas_max);

    nfci_dets_ = 0;
    for (const auto& [_, class_Ia, class_Ib] : lists_->determinant_classes()) {
        auto size_alfa = lists_->alfa_address()->strpcls(class_Ia);
        auto size_beta = lists_->beta_address()->strpcls(class_Ib);
        auto detpcls = size_alfa * size_beta;
        nfci_dets_ += detpcls;
    }

    // Create the spin adapter
    if (spin_adapt_) {
        spin_adapter_ = std::make_shared<SpinAdapter>(state().multiplicity() - 1,
                                                      state().twice_ms(), lists_->ncmo());
        dets_ = lists_->make_determinants();
        spin_adapter_->prepare_couplings(dets_);
    }

    if (print_ >= PrintLevel::Brief) {
        table_printer printer;
        printer.add_int_data({{"Number of determinants", nfci_dets_},
                              {"Symmetry", symmetry_},
                              {"Multiplicity", state().multiplicity()},
                              {"Number of roots", nroot_},
                              {"Target root", root_}});
        printer.add_bool_data({{"Spin adapt", spin_adapt_}});
        printer.add_string_data({{"Print level", to_string(print_)}});
        std::string table = printer.get_table("String-based CI Solver");
        psi::outfile->Printf("%s", table.c_str());
    }
}

void GenCISolver::set_options(std::shared_ptr<ForteOptions> options) {
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
double GenCISolver::compute_energy() {
    local_timer t;
    startup();

    GenCIVector::allocate_temp_space(lists_, print_);

    C_ = std::make_shared<GenCIVector>(lists_);
    T_ = std::make_shared<GenCIVector>(lists_);
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

    if (print_ >= PrintLevel::Default) {
        print_timing("CI", t.get());
    }

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
    psi::Process::environment.globals["CI ENERGY"] = energy_;

    // GenCIVector::release_temp_space();
    return energy_;
}

void GenCISolver::print_solutions(size_t sample_size, std::shared_ptr<psi::Vector> b,
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
        std::vector<std::tuple<double, double, int, int, size_t, size_t>> dets_config =
            C_->max_abs_elements(sample_size);

        for (const auto& [ci_abs, ci, class_Ia, class_Ib, add_Ia, add_Ib] : dets_config) {
            if (ci_abs < 0.1)
                continue;

            auto Ia_v = lists_->alfa_str(class_Ia, add_Ia);
            auto Ib_v = lists_->beta_str(class_Ib, add_Ib);

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

void GenCISolver::test_rdms(std::shared_ptr<psi::Vector> b, std::shared_ptr<psi::Vector> b_basis,
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

std::shared_ptr<psi::Matrix> GenCISolver::ci_wave_functions() {
    if (eigen_vecs_ == nullptr)
        return std::make_shared<psi::Matrix>();

    auto evecs = std::make_shared<psi::Matrix>("FCI Eigenvectors", eigen_vecs_->ncol(), nroot_);
    for (int i = 0, size = static_cast<int>(nroot_); i < size; ++i) {
        evecs->set_column(0, i, eigen_vecs_->get_row(0, i));
    }
    return evecs;
}

} // namespace forte
