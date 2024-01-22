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

#include <cmath>
#include <algorithm>
#include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "forte-def.h"

#include "base_classes/forte_options.h"

#include "helpers/davidson_liu_solver.h"
#include "helpers/timer.h"
#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "helpers/determinant_helpers.h"

#include "ci_spin_adaptation.h"
#include "sigma_vector_dynamic.h"
#include "determinant_functions.hpp"
#include "sparse_initial_guess.h"

#include "sparse_ci_solver.h"

using namespace psi;

namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

SparseCISolver::SparseCISolver() {}

void SparseCISolver::set_spin_project(bool value) { spin_project_ = value; }

void SparseCISolver::set_e_convergence(double value) { e_convergence_ = value; }

void SparseCISolver::set_r_convergence(double value) { r_convergence_ = value; }

void SparseCISolver::set_maxiter_davidson(int value) { maxiter_davidson_ = value; }

void SparseCISolver::set_guess_per_root(int value) { guess_per_root_ = value; }

void SparseCISolver::set_collapse_per_root(int value) { collapse_per_root_ = value; }

void SparseCISolver::set_subspace_per_root(int value) { subspace_per_root_ = value; }

void SparseCISolver::set_spin_project_full(bool value) { spin_project_full_ = value; }

void SparseCISolver::set_spin_adapt(bool value) { spin_adapt_ = value; }

void SparseCISolver::set_print(PrintLevel value) { print_ = value; }

void SparseCISolver::set_spin_adapt_full_preconditioner(bool value) {
    spin_adapt_full_preconditioner_ = value;
}

void SparseCISolver::set_force_diag(bool value) { force_diag_ = value; }

void SparseCISolver::add_bad_states(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SparseCISolver::set_root_project(bool value) { root_project_ = value; }

void SparseCISolver::set_options(std::shared_ptr<ForteOptions> options) {
    set_parallel(true);
    set_force_diag(options->get_bool("FORCE_DIAG_METHOD"));
    set_e_convergence(options->get_double("E_CONVERGENCE"));
    set_r_convergence(options->get_double("R_CONVERGENCE"));

    set_guess_per_root(options->get_int("DL_GUESS_PER_ROOT"));
    set_ndets_per_guess_state(options->get_int("DL_DETS_PER_GUESS"));
    set_collapse_per_root(options->get_int("DL_COLLAPSE_PER_ROOT"));
    set_subspace_per_root(options->get_int("DL_SUBSPACE_PER_ROOT"));
    set_maxiter_davidson(options->get_int("DL_MAXITER"));

    set_spin_project(options->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS"));
    set_spin_project_full(options->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS"));

    set_print(int_to_print_level(options->get_int("PRINT")));

    set_spin_adapt(options->get_bool("CI_SPIN_ADAPT"));
    set_spin_adapt_full_preconditioner(options->get_bool("CI_SPIN_ADAPT_FULL_PRECONDITIONER"));
}

void SparseCISolver::set_initial_guess(
    const std::vector<std::vector<std::pair<size_t, double>>>& guess) {
    user_guess_ = guess;
}

void SparseCISolver::reset_initial_guess() { user_guess_.clear(); }

std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
SparseCISolver::diagonalize_hamiltonian(const DeterminantHashVec& space,
                                        std::shared_ptr<SigmaVector> sigma_vector, int nroot,
                                        int multiplicity) {
    if (print_details_) {
        outfile->Printf("\n\n  Davidson-Liu solver algorithm using %s sigma algorithm",
                        sigma_vector->label().c_str());
    }

    if ((!force_diag_ and (space.size() <= 200)) or
        sigma_vector->sigma_vector_type() == SigmaVectorType::Full) {
        outfile->Printf("\n\n  Performing full diagonalization of the H matrix");
        const std::vector<Determinant> dets = space.determinants();
        return diagonalize_hamiltonian_full(dets, sigma_vector->as_ints(), nroot, multiplicity);
    }

    size_t dim_space = space.size();
    auto evecs = std::make_shared<psi::Matrix>("U", dim_space, nroot);
    auto evals = std::make_shared<psi::Vector>("e", nroot);

    sigma_vector->add_bad_roots(bad_states_);
    davidson_liu_solver(space, sigma_vector, evals, evecs, nroot, multiplicity);

    return std::make_pair(evals, evecs);
}

std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
SparseCISolver::diagonalize_hamiltonian(const std::vector<Determinant>& space,
                                        std::shared_ptr<SigmaVector> sigma_vector, int nroot,
                                        int multiplicity) {
    DeterminantHashVec dhv(space);
    return diagonalize_hamiltonian(dhv, sigma_vector, nroot, multiplicity);
}

std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
SparseCISolver::diagonalize_hamiltonian_full(const std::vector<Determinant>& space,
                                             std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                             int nroot, int multiplicity) {
    // Diagonalize the full Hamiltonian
    size_t dim_space = space.size();
    auto evecs = std::make_shared<psi::Matrix>("U", dim_space, nroot);
    auto evals = std::make_shared<psi::Vector>("e", nroot);

    // Build the Hamiltonian
    auto H = build_full_hamiltonian(space, as_ints);

    // Build the S^2 matrix
    auto S2 = make_s2_matrix(space);

    const double target_S = 0.5 * (static_cast<double>(multiplicity) - 1.0);

    // First, we check if this space is spin complete by looking at how much the
    // eigenvalue of S^2 deviate from their exact values
    auto S2vals = std::make_shared<psi::Vector>("S^2 Eigen Values", dim_space);
    auto S2vecs = std::make_shared<psi::Matrix>("S^2 Eigen Vectors", dim_space, dim_space);

    S2->diagonalize(S2vecs, S2vals);

    bool spin_complete = true;
    double Stollerance = 1.0e-4;
    std::map<int, std::vector<int>> multi_list;
    for (size_t i = 0; i < dim_space; ++i) {
        double multi = std::sqrt(1.0 + 4.0 * S2vals->get(i));
        double error = std::round(multi) - multi;
        if (std::fabs(error) < Stollerance) {
            int multi_round = std::llround(multi);
            multi_list[multi_round].push_back(i);
        } else {
            spin_complete = false;
            if (print_details_) {
                outfile->Printf("\n  Spin multiplicity of root %zu not close to integer (%.4f) ", i,
                                multi);
            }
        }
    }

    if (spin_complete) {
        // Test S^2 eigen values
        int nfound = 0;
        for (const auto& mi : multi_list) {
            int multi = mi.first;
            size_t multi_size = mi.second.size();
            std::string mark = " *";
            if (multi == multiplicity) {
                nfound = static_cast<int>(multi_size);
            } else {
                mark = "";
            }
            if (print_details_) {
                outfile->Printf("\n  Found %zu roots with 2S+1 = %d%s", multi_size, multi,
                                mark.c_str());
            }
        }
        if (nfound < nroot) {
            outfile->Printf("\n  Error: ask for %d roots with 2S+1 = %d but only "
                            "%d were found!",
                            nroot, multiplicity, nfound);
            throw std::runtime_error(
                "Too many roots of interest in full diag. of sparce_ci_solver.");
        }

        // Select sub eigen vectors of S^2 with correct multiplicity
        auto S2vecs_sub =
            std::make_shared<psi::Matrix>("Spin Selected S^2 Eigen Vectors", dim_space, nfound);
        for (int i = 0; i < nfound; ++i) {
            auto vec = S2vecs->get_column(0, multi_list[multiplicity][i]);
            S2vecs_sub->set_column(0, i, vec);
        }

        // Build spin selected Hamiltonian
        auto H = build_full_hamiltonian(space, as_ints);
        auto Hss = psi::linalg::triplet(S2vecs_sub, H, S2vecs_sub, true, false, false);
        Hss->set_name("Hss");

        // Obtain spin selected eigen values and vectors
        auto Hss_vals = std::make_shared<psi::Vector>("Hss Eigen Values", nfound);
        auto Hss_vecs = std::make_shared<psi::Matrix>("Hss Eigen Vectors", nfound, nfound);
        Hss->diagonalize(Hss_vecs, Hss_vals);

        // Project Hss_vecs back to original manifold
        auto H_vecs = psi::linalg::doublet(S2vecs_sub, Hss_vecs);
        H_vecs->set_name("H Eigen Vectors");

        // Fill in results
        energies_.clear();
        spin_.clear();
        for (int i = 0; i < nroot; ++i) {
            evals->set(i, Hss_vals->get(i));
            evecs->set_column(0, i, H_vecs->get_column(0, i));
            spin_.push_back(target_S * (target_S + 1.0));
            energies_.push_back(Hss_vals->get(i));
        }
    } else {

        auto full_evecs = std::make_shared<psi::Matrix>("U", dim_space, dim_space);
        auto full_evals = std::make_shared<psi::Vector>("e", dim_space);

        // Diagonalize H
        H->diagonalize(full_evecs, full_evals);

        // Compute (C)^+ S^2 C
        auto CtSC = psi::linalg::triplet(full_evecs, S2, full_evecs, true, false, false);

        // Find how each solution deviates from the target multiplicity
        std::vector<std::tuple<double, double, size_t, double>> sorted_evals(dim_space);
        std::map<int, std::vector<std::pair<double, size_t>>> S_vals_sorted;

        outfile->Printf("\n  Seeking %d roots with <S^2> = %f", nroot, target_S * (target_S + 1.0));

        outfile->Printf("\n     Root           Energy         <S^2>");
        outfile->Printf("\n    -------------------------------------");
        for (size_t I = 0; I < dim_space; ++I) {
            double avg_S2 = CtSC->get(I, I);
            double energy = full_evals->get(I);
            double S = 0.5 * (std::sqrt(1.0 + 4.0 * avg_S2) - 1.0);
            double S_rounded = 0.5 * std::lround(2.0 * S);
            double twoSp1_rounded = std::lround(2.0 * S + 1.0);
            sorted_evals[I] = std::make_tuple(std::fabs(S_rounded - target_S), energy, I, S);
            S_vals_sorted[twoSp1_rounded].emplace_back(S - S_rounded, I);
            if (I < static_cast<size_t>(std::max(2 * nroot, 10))) {
                outfile->Printf("\n     %3d   %20.12f %9.6f", I,
                                energy + as_ints->nuclear_repulsion_energy(), avg_S2);
            }
        }
        outfile->Printf("\n    -------------------------------------");
        std::sort(begin(sorted_evals), end(sorted_evals));

        outfile->Printf("\n\n    2S + 1   Roots");
        outfile->Printf("\n    --------------");
        for (const auto& k_v : S_vals_sorted) {
            outfile->Printf("\n     %3d   %5zu", k_v.first, k_v.second.size());
        }
        outfile->Printf("\n    --------------");

        const auto& target_S_vals = S_vals_sorted[std::llround(2 * target_S + 1)];
        if (target_S_vals.size() < static_cast<size_t>(nroot)) {
            outfile->Printf("\n  Error: requested for %d roots with 2S+1 = %d but only "
                            "%zu were found!",
                            nroot, multiplicity, target_S_vals.size());
            throw std::runtime_error(
                "Too few roots of interest with correct value of 2S + 1 in full "
                "diag. of sparce_ci_solver.");
        }
        double max_S_error = 0.0;
        for (int n = 0; n < nroot; n++) {
            max_S_error = std::max(std::fabs(target_S_vals[n].first), max_S_error);
        }
        outfile->Printf("\n  Largest deviation from target S value: %.6f\n", max_S_error);
        if (max_S_error > 0.25) {
            outfile->Printf("\n\n  Warning: The CI solutions are heavily spin contaminated.\n");
        }

        // Fill in results
        energies_.clear();
        spin_.clear();
        for (int i = 0; i < nroot; ++i) {
            double energy = std::get<1>(sorted_evals[i]);
            size_t I = std::get<2>(sorted_evals[i]);
            double S = std::get<3>(sorted_evals[i]);
            energies_.push_back(energy);
            evals->set(i, energy);
            spin_.push_back(S * (S + 1.0));
            for (size_t J = 0; J < dim_space; ++J) {
                double C = full_evecs->get(J, I);
                evecs->set(J, i, C);
            }
        }
    }

    return std::make_pair(evals, evecs);
}

std::shared_ptr<psi::Matrix>
SparseCISolver::build_full_hamiltonian(const std::vector<Determinant>& space,
                                       std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    // Build the H matrix
    auto H = make_hamiltonian_matrix(space, as_ints);

    if (root_project_) {
        // Form the projection matrix
        size_t dim_space = space.size();
        for (int n = 0, max_n = bad_states_.size(); n < max_n; ++n) {
            auto P = std::make_shared<psi::Matrix>("P", dim_space, dim_space);
            P->identity();
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            for (size_t det1 = 0, ndet = bad_state.size(); det1 < ndet; ++det1) {
                for (size_t det2 = 0; det2 < ndet; ++det2) {
                    size_t& I = bad_state[det1].first;
                    size_t& J = bad_state[det2].first;
                    double& el1 = bad_state[det1].second;
                    double& el2 = bad_state[det2].second;
                    P->set(I, J, P->get(I, J) - el1 * el2);
                }
            }
            H->transform(P);
        }
    }
    return H;
}

std::vector<Determinant>
SparseCISolver::initial_guess_generate_dets(const DeterminantHashVec& space,
                                            const std::shared_ptr<SigmaVector> sigma_vector,
                                            const size_t num_guess_states) const {
    size_t ndets = space.size();
    size_t num_guess_dets = std::min(num_guess_states * ndets_per_guess_, ndets);

    // Find the ntrial lowest diagonals
    std::vector<std::pair<double, Determinant>> smallest;
    const det_hashvec& detmap = space.wfn_hash();

    auto as_ints = sigma_vector->as_ints();
    size_t nmo = as_ints->nmo();

    for (const Determinant& det : detmap) {
        smallest.emplace_back(as_ints->energy(det), det);
    }
    std::sort(smallest.begin(), smallest.end());
    std::vector<Determinant> guess_dets(num_guess_dets);
    for (size_t i = 0; i < num_guess_dets; i++) {
        guess_dets[i] = smallest[i].second;
    }

    if (spin_project_) {
        outfile->Printf(
            "\n\n  Spin-adaptation of the initial guess based on minimum spin-complete subset");
        auto spin_complete_guess_det = find_minimum_spin_complete(guess_dets, nmo);
        outfile->Printf("\n  Guess determinants after screening: %zu",
                        spin_complete_guess_det.size());
        guess_dets = spin_complete_guess_det;
        // Update the number of guess determinants
    } else {
        outfile->Printf("\n\n  Skipping spin-adaptation of the initial guess");
    }
    return guess_dets;
}

auto SparseCISolver::initial_guess_det(const DeterminantHashVec& space,
                                       std::shared_ptr<SigmaVector> sigma_vector,
                                       size_t num_guess_states, int multiplicity,
                                       bool do_spin_project) {

    auto guess_dets = initial_guess_generate_dets(space, sigma_vector, num_guess_states);
    size_t num_guess_dets = guess_dets.size();

    std::vector<size_t> guess_dets_pos(num_guess_dets);
    for (size_t I = 0; I < num_guess_dets; ++I) {
        guess_dets_pos[I] = space.get_idx(guess_dets[I]);
    }

    return find_initial_guess_det(guess_dets, guess_dets_pos, num_guess_states,
                                  sigma_vector->as_ints(), multiplicity, do_spin_project,
                                  print_ >= PrintLevel::Default, user_guess_);
}

auto SparseCISolver::initial_guess_csf(std::shared_ptr<psi::Vector> diag, size_t num_guess_states,
                                       int multiplicity) {

    return find_initial_guess_csf(diag, num_guess_states, multiplicity,
                                  print_ >= PrintLevel::Default);
}

std::shared_ptr<psi::Vector>
SparseCISolver::form_Hdiag_csf(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                               std::shared_ptr<SpinAdapter> spin_adapter) {
    auto ncsf = spin_adapter->ncsf();
    auto Hdiag_csf = std::make_shared<psi::Vector>(ncsf);
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    double E0 = fci_ints->nuclear_repulsion_energy() + fci_ints->scalar_energy();
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    if (spin_adapt_full_preconditioner_) {
        for (size_t i = 0; i < ncsf; ++i) {
            double energy = E0;
            int I = 0;
            for (const auto& [det_add_I, c_I] : spin_adapter_->csf(i)) {
                int J = 0;
                for (const auto& [det_add_J, c_J] : spin_adapter_->csf(i)) {
                    if (I == J) {
                        energy += c_I * c_J * fci_ints->energy(dets_[det_add_I]);
                    } else if (I < J) {
                        if (c_I * c_J != 0.0) {
                            energy += 2.0 * c_I * c_J *
                                      fci_ints->slater_rules(dets_[det_add_I], dets_[det_add_J]);
                        }
                    }
                    J++;
                }
                I++;
            }
            Hdiag_csf->set(i, energy);
        }
    } else {
        for (size_t i = 0; i < ncsf; ++i) {
            double energy = E0;
            for (const auto& [det_add_I, c_I] : spin_adapter_->csf(i)) {
                energy += c_I * c_I * fci_ints->energy(dets_[det_add_I]);
            }
            Hdiag_csf->set(i, energy);
        }
    }
    return Hdiag_csf;
}

bool SparseCISolver::davidson_liu_solver(const DeterminantHashVec& space,
                                         std::shared_ptr<SigmaVector> sigma_vector,
                                         std::shared_ptr<psi::Vector> Eigenvalues,
                                         std::shared_ptr<psi::Matrix> Eigenvectors, int nroot,
                                         int multiplicity) {
    local_timer dl;

    // Create the spin adapter
    if (spin_adapt_) {
        auto nmo = sigma_vector->as_ints()->nmo();
        auto twice_ms = space[0].count_alfa() - space[0].count_beta();
        spin_adapter_ = std::make_shared<SpinAdapter>(multiplicity - 1, twice_ms, nmo);
        dets_ = space.determinants();
        spin_adapter_->prepare_couplings(dets_);
    }

    // Compute the size of the determinant space and the basis used by the Davidson solver
    size_t fci_size = sigma_vector->size();
    size_t basis_size = spin_adapt_ ? spin_adapter_->ncsf() : fci_size;

    // if the DL solver is not allocated or if the basis size changed create a new one
    if ((dl_solver_ == nullptr) or (dl_solver_->size() != basis_size)) {
        dl_solver_ = std::make_shared<DavidsonLiuSolver>(basis_size, nroot, collapse_per_root_,
                                                         subspace_per_root_);
        dl_solver_->set_e_convergence(e_convergence_);
        dl_solver_->set_r_convergence(r_convergence_);
        dl_solver_->set_print_level(print_);
        dl_solver_->set_maxiter(maxiter_davidson_);
    } else {
        dl_solver_->reset();
    }

    // allocate vectors
    auto b = std::make_shared<psi::Vector>("b", fci_size);
    auto sigma = std::make_shared<psi::Vector>("sigma", fci_size);

    // Optionally create the vectors that stores the b and sigma vectors in the CSF basis
    std::shared_ptr<psi::Vector> b_basis = b;
    std::shared_ptr<psi::Vector> sigma_basis = sigma;

    if (spin_adapt_) {
        b_basis = std::make_shared<psi::Vector>("b", basis_size);
        sigma_basis = std::make_shared<psi::Vector>("sigma", basis_size);
    }

    const size_t num_guess_states = std::min(guess_per_root_ * nroot, basis_size);

    // Form the diagonal of the Hamiltonian and the initial guess
    if (spin_adapt_) {
        auto Hdiag_vec = form_Hdiag_csf(sigma_vector->as_ints(), spin_adapter_);
        dl_solver_->add_h_diag(Hdiag_vec);
        auto guesses = initial_guess_csf(Hdiag_vec, num_guess_states, multiplicity);
        dl_solver_->add_guesses(guesses);
    } else {
        sigma_vector->get_diagonal(*sigma);
        dl_solver_->add_h_diag(sigma);
        auto [guesses, bad_roots] =
            initial_guess_det(space, sigma_vector, num_guess_states, multiplicity, spin_project_);
        dl_solver_->add_guesses(guesses);
        dl_solver_->add_project_out_vectors(bad_roots);
    }

    // Setup the sigma builder
    auto sigma_builder = [this, &b_basis, &b, &sigma, &sigma_basis,
                          &sigma_vector](std::span<double> b_span, std::span<double> sigma_span) {
        // copy the b vector
        size_t basis_size = b_span.size();
        for (size_t I = 0; I < basis_size; ++I) {
            b_basis->set(I, b_span[I]);
        }
        if (spin_adapt_) {
            // Compute sigma in the CSF basis and convert it to the determinant basis
            spin_adapter_->csf_C_to_det_C(b_basis, b);
            sigma_vector->compute_sigma(sigma, b);
            spin_adapter_->det_C_to_csf_C(sigma, sigma_basis);
        } else {
            // Compute sigma in the determinant basis
            sigma_vector->compute_sigma(sigma_basis, b_basis);
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
            "Davidson-Liu iterations (DL_MAXITER). You can also try to increase:\n - the maximum "
            "size of the subspace (DL_SUBSPACE_PER_ROOT)"
            "\n - the number of guess states (DL_GUESS_PER_ROOT)");
        return false;
    }

    // Copy eigenvalues and eigenvectors from the Davidson-Liu solver
    spin_.clear();
    auto evals = dl_solver_->eigenvalues();
    auto evecs = dl_solver_->eigenvectors();

    for (int r = 0; r < nroot; ++r) {
        Eigenvalues->set(r, evals->get(r));
        energies_.push_back(evals->get(r));
        b_basis = dl_solver_->eigenvector(r);
        std::vector<double> c(sigma_vector->size());
        if (spin_adapt_) {
            spin_adapter_->csf_C_to_det_C(b_basis, b);
            for (size_t I = 0; I < fci_size; ++I) {
                Eigenvectors->set(I, r, b->get(I));
                c[I] = b->get(I);
            }
        } else {
            b = b_basis;
            for (size_t I = 0; I < fci_size; ++I) {
                Eigenvectors->set(I, r, evecs->get(r, I));
                c[I] = evecs->get(r, I);
            }
        }
        double s2 = sigma_vector->compute_spin(c);
        spin_.push_back(s2);
    }

    if (print_details_) {
        outfile->Printf("\n  Davidson-Liu procedure took  %1.6f s", dl.get());
    }

    return true;
}
} // namespace forte
