/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <filesystem>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/mo_space_info.h"

#include "helpers/timer.h"
#include "helpers/disk_io.h"
#include "helpers/printing.h"

#include "helpers/iterative_solvers.h"

#define PRINT_VARS(msg)                                                                            \
    // std::vector<std::pair<size_t, std::string>> v = {{collapse_size_, "collapse_size_"},           \
    //                                                  {subspace_size_, "subspace_size_"},           \
    //                                                  {basis_size_, "basis_size_"},                 \
    //                                                  {sigma_size_, "sigma_size_"},                 \
    //                                                  {nroot_, "nroot_"}};                          \
    // outfile->Printf("\n\n => %s <=", msg);                                                         \
    // for (auto vk : v) {                                                                            \
    //     outfile->Printf("\n    %-30s  %zu", vk.second.c_str(), vk.first);                          \
    // }

using namespace psi;

namespace forte {

DavidsonLiuSolver::DavidsonLiuSolver(size_t size, size_t nroot) : size_(size), nroot_(nroot) {
    if (size_ == 0)
        throw std::runtime_error("DavidsonLiuSolver called with space of dimension zero.");
    residual_.resize(nroot, 0.0);
}

size_t DavidsonLiuSolver::startup(std::shared_ptr<psi::Vector> diagonal) {
    // set space size
    collapse_size_ = std::min(collapse_per_root_ * nroot_, size_);
    subspace_size_ = std::min(subspace_per_root_ * nroot_, size_);

    if (collapse_size_ > subspace_size_) {
        std::string msg =
            "DavidsonLiuSolver: collapse space size (" + std::to_string(collapse_size_) +
            ") must be less or equal to subspace size (" + std::to_string(subspace_size_) + ")";
        throw std::runtime_error(msg);
    }

    basis_size_ = 0; // start with no vectors
    sigma_size_ = 0; // start with no sigma vectors
    iter_ = 0;
    converged_ = 0;

    // store the basis vectors (each row is a vector)
    b_ = std::make_shared<psi::Matrix>("b", subspace_size_, size_);
    b_->zero();
    bnew = std::make_shared<psi::Matrix>("bnew", subspace_size_, size_);
    f = std::make_shared<psi::Matrix>("f", subspace_size_, size_);
    sigma_ = std::make_shared<psi::Matrix>("sigma", size_, subspace_size_);
    sigma_->zero();

    G = std::make_shared<psi::Matrix>("G", subspace_size_, subspace_size_);
    S = std::make_shared<psi::Matrix>("S", subspace_size_, subspace_size_);
    alpha = std::make_shared<psi::Matrix>("alpha", subspace_size_, subspace_size_);

    lambda = std::make_shared<psi::Vector>("lambda", subspace_size_);
    lambda_old = std::make_shared<psi::Vector>("lambda_old", subspace_size_);
    h_diag = std::make_shared<psi::Vector>("h_diag", size_);

    h_diag->copy(*diagonal);

    basis_size_ = load_state();

    // Print a summary of the calculation options
    table_printer printer;
    printer.add_double_data({{"Energy convergence threshold", e_convergence_},
                             {"Residual convergence threshold", r_convergence_},
                             {"Schmidt orthogonality threshold", schmidt_orthogonality_threshold_},
                             {"Schmidt discard threshold", schmidt_discard_threshold_}});

    printer.add_int_data({{"Size of the space", size_},
                          {"Number of roots", nroot_},
                          {"Collapse subspace size", collapse_size_},
                          {"Maximum subspace size", subspace_size_},
                          {"States read from file", basis_size_}});

    printer.add_bool_data({{"Save state at destruction", save_state_at_destruction_}});

    std::string table = printer.get_table("Davidson-Liu Solver");
    outfile->Printf("%s", table.c_str());

    return basis_size_;
}

DavidsonLiuSolver::~DavidsonLiuSolver() {
    if (save_state_at_destruction_) {
        save_state();
    }
}

void DavidsonLiuSolver::set_print_level(size_t n) { print_level_ = n; }

void DavidsonLiuSolver::set_e_convergence(double value) { e_convergence_ = value; }

void DavidsonLiuSolver::set_r_convergence(double value) { r_convergence_ = value; }

double DavidsonLiuSolver::get_e_convergence() const { return e_convergence_; }

double DavidsonLiuSolver::get_r_convergence() const { return r_convergence_; }

void DavidsonLiuSolver::set_collapse_per_root(int value) { collapse_per_root_ = value; }

void DavidsonLiuSolver::set_subspace_per_root(int value) { subspace_per_root_ = value; }

void DavidsonLiuSolver::set_save_state_at_destruction(bool value) {
    save_state_at_destruction_ = value;
}

size_t DavidsonLiuSolver::collapse_size() const { return collapse_size_; }

void DavidsonLiuSolver::add_guess(std::shared_ptr<psi::Vector> vec) {
    if (basis_size_ >= subspace_size_) {
        throw std::runtime_error("DavidsonLiuSolver: subspace is full.");
    }
    // Give the next b that does not have a sigma
    for (size_t j = 0; j < size_; ++j) {
        b_->set(basis_size_, j, vec->get(j));
    }
    basis_size_++;
}

void DavidsonLiuSolver::get_b(std::shared_ptr<psi::Vector> vec) {
    PRINT_VARS("get_b")
    // Give the next b that does not have a sigma
    for (size_t j = 0; j < size_; ++j) {
        vec->set(j, b_->get(sigma_size_, j));
    }
}

bool DavidsonLiuSolver::add_sigma(std::shared_ptr<psi::Vector> vec) {
    PRINT_VARS("add_sigma")
    if (sigma_size_ >= subspace_size_) {
        throw std::runtime_error("DavidsonLiuSolver: sigma subspace is full.");
    }
    // Place the new sigma vector at the end
    for (size_t j = 0; j < size_; ++j) {
        sigma_->set(j, sigma_size_, vec->get(j));
    }
    sigma_size_++;
    return (sigma_size_ < basis_size_);
}

void DavidsonLiuSolver::set_project_out(std::vector<sparse_vec> project_out) {
    project_out_ = project_out;
}

size_t DavidsonLiuSolver::size() const { return size_; }

std::shared_ptr<psi::Vector> DavidsonLiuSolver::eigenvalues() const { return lambda; }

std::shared_ptr<psi::Matrix> DavidsonLiuSolver::eigenvectors() const { return bnew; }

std::shared_ptr<psi::Vector> DavidsonLiuSolver::eigenvector(size_t n) const {
    double** v = bnew->pointer();
    auto evec = std::make_shared<psi::Vector>("V", size_);
    for (size_t I = 0; I < size_; I++) {
        evec->set(I, v[n][I]);
    }
    return evec;
}

std::vector<double> DavidsonLiuSolver::residuals() const { return residual_; }

SolverStatus DavidsonLiuSolver::update() {
    // If converged or exceeded the maximum number of iterations return true
    // if ((converged_ >= nroot_) or (iter_ > maxiter_)) return
    // SolverStatus::Converged;
    if ((converged_ >= nroot_)) {
        get_results();
        return SolverStatus::Converged;
    }

    PRINT_VARS("update")

    local_timer t_davidson;

    // form and diagonalize mini-matrix
    G->zero();
    G->gemm(false, false, 1.0, b_, sigma_, 0.0);
    G->hermitivitize();
    G->diagonalize(alpha, lambda);

    bool is_energy_converged = false;
    bool is_residual_converged = false;
    if (not last_update_collapsed_) {
        auto converged = check_convergence();
        is_energy_converged = converged.first;
        is_residual_converged = converged.second;
        if (is_energy_converged and is_residual_converged) {
            get_results();
            return SolverStatus::Converged;
        }
    }

    if (size_ == 1) {
        get_results();
        return SolverStatus::Converged;
    }

    // outfile->Printf("\n  Iteration %d:  %d converged roots", iter_, basis_size_);
    check_orthogonality();

    // Do subspace collapse
    if (subspace_collapse()) {
        last_update_collapsed_ = true;
        return SolverStatus::Collapse;
    } else {
        last_update_collapsed_ = false;
    }

    // Step #3: Build the Correction Vectors
    // form preconditioned residue vectors
    form_correction_vectors();

    // Step #3b: Project out undesired roots
    project_out_roots(f);

    // Step #4: Normalize the Correction Vectors
    auto f_norm = normalize_vectors(f, nroot_);

    // schmidt orthogonalize the f[k] against the set of b[i] and add new
    // vectors
    size_t num_added = 0;
    for (size_t k = 0; k < nroot_; k++) {
        if (basis_size_ < subspace_size_) {
            // Schmidt-orthogonalize the correction vector
            if (schmidt_add(b_, basis_size_, size_, f, k)) {
                basis_size_++;
                num_added += 1;
            } else {
                outfile->Printf("\n  Rejected new correction vector %d with norm: %f", k,
                                f_norm[k]);
            }
        }
    }

    // outfile->Printf("\n 2 Iteration %d:  %d converged roots", iter_, basis_size_);
    check_orthogonality();

    // if we do not add any new vector then we are in trouble and we better finish the computation
    if ((num_added == 0) and is_energy_converged) {
        outfile->Printf("\n  Davidson-Liu solver:  No new vectors added, but energy converged. "
                        "Finishing computation.");
        get_results();
        return SolverStatus::Converged;
    }

    iter_++;

    timing_ += t_davidson.get();

    return SolverStatus::NotConverged;
}

void DavidsonLiuSolver::form_correction_vectors() {
    f->zero();
    double* lambda_p = lambda->pointer();
    double* Adiag_p = h_diag->pointer();
    double** b_p = b_->pointer();
    double** f_p = f->pointer();
    double** alpha_p = alpha->pointer();
    double** sigma_p = sigma_->pointer();

    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        residual_[k] = 0.0;
        for (size_t I = 0; I < size_; I++) { // loop over elements
            for (size_t i = 0; i < basis_size_; i++) {
                f_p[k][I] += alpha_p[i][k] * (sigma_p[I][i] - lambda_p[k] * b_p[i][I]);
            }
            residual_[k] += std::pow(f_p[k][I], 2.0);

            double denom = lambda_p[k] - Adiag_p[I];
            if (std::fabs(denom) > 1.0e-6) {
                f_p[k][I] /= denom;
            } else {
                f_p[k][I] = 0.0;
            }
        }
        residual_[k] = std::sqrt(residual_[k]);
    }
}

void DavidsonLiuSolver::compute_residual_norm() {
    double* lambda_p = lambda->pointer();
    double** b_p = b_->pointer();
    double** alpha_p = alpha->pointer();
    double** sigma_p = sigma_->pointer();

    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        residual_[k] = 0.0;
        for (size_t I = 0; I < size_; I++) { // loop over elements
            double r = 0.0;
            for (size_t i = 0; i < basis_size_; i++) {
                r += alpha_p[i][k] * (sigma_p[I][i] - lambda_p[k] * b_p[i][I]);
            }
            residual_[k] += std::pow(r, 2.0);
        }
        residual_[k] = std::sqrt(residual_[k]);
    }
}

void DavidsonLiuSolver::project_out_roots(std::shared_ptr<psi::Matrix> v) {
    double** v_p = v->pointer();
    for (size_t k = 0; k < nroot_; k++) {
        for (auto& bad_root : project_out_) {
            double overlap = 0.0;
            for (auto& I_CI : bad_root) {
                size_t I = I_CI.first;
                double CI = I_CI.second;
                overlap += v_p[k][I] * CI;
            }
            for (auto& I_CI : bad_root) {
                size_t I = I_CI.first;
                double CI = I_CI.second;
                v_p[k][I] -= overlap * CI;
            }
        }
    }
}

std::vector<double> DavidsonLiuSolver::normalize_vectors(std::shared_ptr<psi::Matrix> v, size_t n) {
    // normalize each residual
    std::vector<double> v_norm;
    for (size_t k = 0; k < n; k++) {
        double* v_k = v->pointer()[k];
        double norm = C_DDOT(size_, v_k, 1, v_k, 1);
        norm = std::sqrt(norm);
        for (size_t I = 0; I < size_; I++) {
            v_k[I] /= norm;
        }
        v_norm.push_back(norm);
    }
    return v_norm;
}

bool DavidsonLiuSolver::subspace_collapse() {
    if (collapse_size_ + nroot_ > subspace_size_) { // in this case I will never
                                                    // be able to add new
                                                    // vectors
        auto collapsable_size = std::min(collapse_size_, basis_size_);

        // collapse vectors
        collapse_vectors(collapsable_size);

        // normalize new vectors
        normalize_vectors(bnew, collapsable_size);

        // Copy them into place
        b_->zero();
        sigma_->zero();
        basis_size_ = 0;
        sigma_size_ = 0;
        for (size_t k = 0; k < collapsable_size; k++) {
            if (schmidt_add(b_, basis_size_, size_, bnew, k)) {
                basis_size_++; // <- Increase L if we add one more basis vector
            }
        }
        // outfile->Printf("\n  Subspace collapse1: %d vectors left", basis_size_);
        // S->gemm(false, true, 1.0, b_, b_, 0.0);
        // S->print();
        return false;
    }

    // If L is close to maxdim, collapse to one guess per root */
    if (nroot_ + basis_size_ > subspace_size_) { // this means that next
                                                 // iteration I cannot add more
                                                 // roots so I need to collapse
        if (print_level_ > 1) {
            outfile->Printf("\nSubspace too large: max subspace size = %d, "
                            "basis size = %d\n",
                            subspace_size_, basis_size_);
            outfile->Printf("Collapsing eigenvectors.\n");
        }
        auto collapsable_size = std::min(collapse_size_, basis_size_);

        // collapse vectors
        collapse_vectors(collapsable_size);

        // normalize new vectors
        normalize_vectors(bnew, collapsable_size);

        // Copy them into place
        b_->zero();
        sigma_->zero();
        basis_size_ = 0;
        sigma_size_ = 0;
        for (size_t k = 0; k < collapsable_size; k++) {
            if (schmidt_add(b_, basis_size_, size_, bnew, k)) {
                basis_size_++; // <- Increase L if we add one more basis vector
            }
        }

        // outfile->Printf("\n  Subspace collapse2: %d vectors left\n", basis_size_);
        // S->gemm(false, true, 1.0, b_, b_, 0.0);
        // S->print();

        /// Need new sigma vectors to continue, so return control to caller
        return true;
    }
    return false;
}

void DavidsonLiuSolver::collapse_vectors(size_t collapsable_size) {
    bnew->zero();
    double** alpha_p = alpha->pointer();
    double** b_p = b_->pointer();
    double** bnew_p = bnew->pointer();
    for (size_t i = 0; i < collapsable_size; i++) {
        for (size_t j = 0; j < basis_size_; j++) {
            for (size_t k = 0; k < size_; k++) {
                bnew_p[i][k] += alpha_p[j][i] * b_p[j][k];
            }
        }
    }
}

/// add a new vector and orthogonalize it against the existing ones
bool DavidsonLiuSolver::schmidt_add(std::shared_ptr<psi::Matrix> Amat, size_t rows, size_t cols,
                                    std::shared_ptr<psi::Matrix> vvec, int l) {
    double** A = Amat->pointer();
    double* v = vvec->pointer()[l];

    int max_orthogonalization_cycles = 10;

    // here we do the schmidt orthogonalization several times. Often, one step is enough
    // but sometimes it takes more than one step to guarantee orthogonality to within
    // a tight tolerance. The option that controls this is schmidt_orthogonality_threshold_

    for (int cycle = 0; cycle < max_orthogonalization_cycles; cycle++) {
        // schmidt orthogonalize the v vector against the set of A[i]
        for (size_t i = 0; i < rows; i++) {
            const auto dotval = C_DDOT(cols, A[i], 1, v, 1);
            for (size_t I = 0; I < cols; I++)
                v[I] -= dotval * A[i][I];
        }
        // compute the norm of the vector
        const auto normval = std::sqrt(C_DDOT(cols, v, 1, v, 1));

        // if the norm is small, discard the vector
        if (normval < schmidt_discard_threshold_)
            return false;
        for (size_t I = 0; I < cols; I++)
            v[I] /= normval;

        // check the overlap with the previous vectors
        double max_overlap = 0.0;
        for (size_t i = 0; i < rows; i++) {
            max_overlap = std::max(max_overlap, std::fabs(C_DDOT(cols, A[i], 1, v, 1)));
        }
        double norm = C_DDOT(cols, v, 1, v, 1);

        // outfile->Printf(
        //     "\n  Schmidt orthogonalization cycle %d: max_overlap = %20.16f, norm = %20.16f",
        //     cycle, max_overlap, norm);

        if ((max_overlap < schmidt_orthogonality_threshold_) and
            (std::fabs(norm - 1.0) < schmidt_orthogonality_threshold_)) {
            // add the new vector to the set of A[i]
            for (size_t I = 0; I < cols; I++)
                A[rows][I] = v[I];
            return true;
        }
    }

    // if we did not converge, return false
    return false;
}

std::pair<bool, bool> DavidsonLiuSolver::check_convergence() {
    compute_residual_norm();
    // check convergence on all roots
    size_t num_converged_energy = 0;
    size_t num_converged_residual = 0;
    converged_ = 0;
    if (print_level_ > 1) {
        outfile->Printf("\n  Root      Eigenvalue        Delta   Converged?\n");
        outfile->Printf("  ---- -------------------- --------- ----------\n");
    }
    for (size_t k = 0; k < nroot_; k++) {
        // check if the energy converged
        double diff = std::fabs(lambda->get(k) - lambda_old->get(k));
        bool this_energy_converged = (diff < e_convergence_);
        // check if the residual converged
        bool this_residual_converged = (residual_[k] < r_convergence_);
        // update counters
        num_converged_energy += this_energy_converged;
        num_converged_residual += this_residual_converged;

        if (this_energy_converged and this_residual_converged) {
            converged_++;
        }
        // update the old eigenvalue
        lambda_old->set(k, lambda->get(k));

        if (print_level_ > 1) {
            bool this_converged = (this_energy_converged and this_residual_converged);
            outfile->Printf("  %3d  %20.14f %4.3e      %1s\n", k, lambda->get(k), diff,
                            this_converged ? "Y" : "N");
        }
    }

    bool is_energy_converged = (num_converged_energy == nroot_);
    bool is_residual_converged = (num_converged_residual == nroot_);
    return std::make_pair(is_energy_converged, is_residual_converged);
}

void DavidsonLiuSolver::get_results() {
    /* generate final eigenvalues and eigenvectors */
    double** alpha_p = alpha->pointer();
    double** b_p = b_->pointer();
    double* eps = lambda_old->pointer();
    double** v = bnew->pointer();
    bnew->zero();

    for (size_t i = 0; i < nroot_; i++) {
        eps[i] = lambda->get(i);
        for (size_t j = 0; j < basis_size_; j++) {
            for (size_t I = 0; I < size_; I++) {
                v[i][I] += alpha_p[j][i] * b_p[j][I];
            }
        }
        // Normalize v
        double norm = 0.0;
        for (size_t I = 0; I < size_; I++) {
            norm += v[i][I] * v[i][I];
        }
        norm = std::sqrt(norm);
        for (size_t I = 0; I < size_; I++) {
            v[i][I] /= norm;
        }
    }
}

void DavidsonLiuSolver::check_orthogonality() {
    // here we use a looser threshold than the one used in the schmidt orthogonalization
    double orthogonality_threshold = schmidt_orthogonality_threshold_ * 3.0;

    // Compute the overlap matrix
    S->gemm(false, true, 1.0, b_, b_, 0.0);

    // Check for normalization
    double maxdiag = 0.0;
    for (size_t i = 0; i < basis_size_; ++i) {
        maxdiag = std::max(maxdiag, std::fabs(S->get(i, i) - 1.0));
    }
    if (maxdiag > orthogonality_threshold) {
        outfile->Printf("\n\nDavidsonLiuSolver::check_orthogonality(): eigenvectors are not "
                        "normalized!\nMaximum absolute deviation from normalization: %e",
                        maxdiag);
        S->print();
        std::string msg =
            "DavidsonLiuSolver::check_orthogonality(): eigenvectors are not normalized!";
        throw std::runtime_error(msg);
    }

    // Check for orthogonality
    double maxoffdiag = 0.0;
    for (size_t i = 0; i < basis_size_; ++i) {
        for (size_t j = 0; j < basis_size_; ++j) {
            if (i != j) {
                maxoffdiag = std::max(maxoffdiag, std::fabs(S->get(i, j)));
            }
        }
    }
    if (maxoffdiag > orthogonality_threshold) {
        outfile->Printf("\n\nDavidsonLiuSolver::check_orthogonality(): eigenvectors are not "
                        "normalized!\nMaximum absolute off-diagonal element of the overlap: %e",
                        maxoffdiag);
        S->print();
        std::string msg =
            "DavidsonLiuSolver::check_orthogonality(): eigenvectors are not orthogonal!";
        throw std::runtime_error(msg);
    }
}

void DavidsonLiuSolver::save_state() const {
    // save the current state of the solver
    std::filesystem::path filepath = "./dl.b";

    write_psi_matrix(filepath.c_str(), *b_, true);
    psi::outfile->Printf("\n\n  Storing the DL solver to file: %s", filepath.c_str());
}

size_t DavidsonLiuSolver::load_state() {
    // restore the state of the solver
    std::filesystem::path filepath = "./dl.b";

    if (not std::filesystem::exists(filepath)) {
        return 0;
    }

    psi::outfile->Printf("\n  Restoring DL solver from file: %s", filepath.c_str());

    read_psi_matrix("dl.b", *b_);

    // Compute the inner product matrix
    S->gemm(false, true, 1.0, b_, b_, 0.0);

    // Determine the number of non-zero vectors
    size_t nnonzero = 0;
    for (size_t i = 0; i < subspace_size_; i++) {
        if (std::fabs(S->get(i, i) - 1.0) < schmidt_orthogonality_threshold_ * 3.0) {
            nnonzero++;
        }
    }
    // Check for orthogonality
    try {
        check_orthogonality();
    } catch (const std::exception& e) {
        psi::outfile->Printf("\n  Orthogonality check failed: %s", e.what());
        return 0;
    }

    if (nnonzero < nroot_) {
        outfile->Printf("\n  Not enought valid initial guess vectors. Using initial guess.");
        return false;
    }
    outfile->Printf("\n  Restored %d vectors from file.", nnonzero);

    return nnonzero;
}

} // namespace forte
