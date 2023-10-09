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

#include <random>

#include "helpers/printing.h"

#include "helpers/davidson_liu_solver.h"

// Global debug flag
bool global_debug_flag = false;

// Wrapper function
template <typename Func> void debug(Func func) {
    if (global_debug_flag) {
        func();
    }
}

namespace forte {

DavidsonLiuSolver2::DavidsonLiuSolver2(size_t size, size_t nroot, size_t collapse_per_root,
                                       size_t subspace_per_root)
    : size_(size), nroot_(nroot), collapse_per_root_(collapse_per_root),
      subspace_per_root_(subspace_per_root) {

    startup();
}

void DavidsonLiuSolver2::startup() {
    // sanity checks
    // collapse_per_root_ must greater than or equal to one. This guarantees that we have at least
    // one vector per root after the collapse
    if (size_ == 0)
        throw std::runtime_error("DavidsonLiuSolver2 called with space of dimension zero.");
    if (nroot_ == 0)
        throw std::runtime_error("DavidsonLiuSolver2 called with zero roots.");
    if (collapse_per_root_ < 1) {
        std::string msg =
            "DavidsonLiuSolver2: collapse_per_root_ (" + std::to_string(collapse_per_root_) +
            ") must be greater or equal to the number of roots (" + std::to_string(nroot_) + ")";
        throw std::runtime_error(msg);
    }
    // subspace_per_root_ should be at least one greater than collapse_per_root_
    if (subspace_per_root_ < collapse_per_root_ + 1) {
        std::string msg = "DavidsonLiuSolver2: subspace_per_root_ (" +
                          std::to_string(subspace_per_root_) +
                          ") must be greater or equal to collapse_per_root_ + 1 (" +
                          std::to_string(collapse_per_root_ + 1) + ")";
        throw std::runtime_error(msg);
    }

    // set space size
    collapse_size_ = std::min(collapse_per_root_ * nroot_, size_);
    subspace_size_ = std::min(subspace_per_root_ * nroot_, size_);

    basis_size_ = 0; // start with no vectors
    sigma_size_ = 0; // start with no vectors

    // Vectors (here we store the vectors as row vectors)
    temp_ = std::make_shared<psi::Matrix>("temp", subspace_size_, size_);
    b_ = std::make_shared<psi::Matrix>("b", subspace_size_, size_);
    r_ = std::make_shared<psi::Matrix>("r", subspace_size_, size_);
    sigma_ = std::make_shared<psi::Matrix>("sigma", subspace_size_, size_);

    // Subspace matrices/vector
    G_ = std::make_shared<psi::Matrix>("G", subspace_size_, subspace_size_);
    S_ = std::make_shared<psi::Matrix>("S", subspace_size_, subspace_size_);
    alpha_ = std::make_shared<psi::Matrix>("alpha", subspace_size_, subspace_size_);

    lambda_ = std::make_shared<psi::Vector>("lambda", subspace_size_);
    lambda_old_ = std::make_shared<psi::Vector>("lambda_old", subspace_size_);

    residual_.resize(nroot_, 0.0);

    // Print a summary of the calculation options
    table_printer printer;
    printer.add_double_data({{"Energy convergence threshold", e_convergence_},
                             {"Residual convergence threshold", r_convergence_},
                             {"Schmidt orthogonality threshold", schmidt_orthogonality_threshold_},
                             {"Schmidt discard threshold", schmidt_discard_threshold_}});

    printer.add_int_data({{"Size of the space", size_},
                          {"Number of roots", nroot_},
                          {"Maximum number of iterations", max_iter_},
                          {"Collapse subspace size", collapse_size_},
                          {"Maximum subspace size", subspace_size_},
                          {"States read from file", basis_size_},
                          {"Print level", print_level_}});

    std::string table = printer.get_table("Davidson-Liu Solver");
    psi::outfile->Printf("%s", table.c_str());
}

void DavidsonLiuSolver2::add_h_diag(std::shared_ptr<psi::Vector>& h_diag) {
    if (h_diag->dim() != size_) {
        std::string msg = "DavidsonLiuSolver2: h_diag vector size (" +
                          std::to_string(h_diag_->dim()) + ") must be equal to space size (" +
                          std::to_string(size_) + ")";
        throw std::runtime_error(msg);
    }
    h_diag_ = h_diag;
}

void DavidsonLiuSolver2::add_guesses(const std::vector<sparse_vec>& guesses) { guesses_ = guesses; }

void DavidsonLiuSolver2::add_project_out_vectors(
    const std::vector<sparse_vec>& project_out_vectors) {
    project_out_vectors_ = project_out_vectors;
}

void DavidsonLiuSolver2::add_sigma_builder(
    std::function<void(std::span<double>, std::span<double>)> sigma_builder) {
    sigma_builder_ = sigma_builder;
}

void DavidsonLiuSolver2::reset() {
    sigma_size_ = 0;
    sigma_->zero();
}

void DavidsonLiuSolver2::set_print_level(size_t n) { print_level_ = n; }

void DavidsonLiuSolver2::set_e_convergence(double value) { e_convergence_ = value; }

void DavidsonLiuSolver2::set_r_convergence(double value) { r_convergence_ = value; }

std::shared_ptr<psi::Vector> DavidsonLiuSolver2::eigenvalues() const { return lambda_; }

std::shared_ptr<psi::Matrix> DavidsonLiuSolver2::eigenvectors() const { return b_; }

std::shared_ptr<psi::Vector> DavidsonLiuSolver2::eigenvector(size_t n) const {
    double** v = b_->pointer();
    auto evec = std::make_shared<psi::Vector>("V", size_);
    for (size_t I = 0; I < size_; I++) {
        evec->set(I, v[n][I]);
    }
    return evec;
}

bool DavidsonLiuSolver2::solve() {
    psi::outfile->Printf(
        "\n  Iteration     Average Energy            max(∆E)            max(Residual)");
    psi::outfile->Printf(
        "\n  ------------------------------------------------------------------------");

    // check that the sigma builder has been set
    if (sigma_builder_ == nullptr) {
        std::string msg = "DavidsonLiuSolver2: sigma builder has not been set";
        throw std::runtime_error(msg);
    }

    // ensure that h_diag was set
    if (h_diag_ == nullptr) {
        std::string msg = "DavidsonLiuSolver2: h_diag has not been set";
        throw std::runtime_error(msg);
    }

    // Add the initial guess to the basis and orthonormalize it
    if (basis_size_ == 0) {
        // add the guesses to temp
        if ((guesses_.size() >= nroot_) and (guesses_.size() <= subspace_size_)) {
            // guesses [copy] -> temp [orthonormalize] -> b
            set_vector(temp_, guesses_);

        } else if (guesses_.size() == 0) {
            // add random vectors
            temp_->zero();
            auto added = add_random_vectors(temp_, 0, nroot_);
        } else {
            std::string msg = "DavidsonLiuSolver2: number of guess vectors (" +
                              std::to_string(guesses_.size()) +
                              ") must be between 0 and the subspace size (" +
                              std::to_string(subspace_size_) + ")";
            throw std::runtime_error(msg);
        }
        auto should_be_added = std::max(nroot_, guesses_.size());

        // project out the unwanted roots
        project_out_roots(temp_);

        // orthonormalize what is left
        auto added = add_rows_and_orthonormalize(b_, 0, temp_, should_be_added);
        if (added != should_be_added) {
            std::string msg = "DavidsonLiuSolver2: guess vectors are zero or linearly dependent";
            throw std::runtime_error(msg);
        }
        basis_size_ += added;
    }

    // Print the initial basis

    for (size_t iter = 0; iter < max_iter_; iter++) {
        // ensure that the DL basis is orthonormal
        check_orthogonality();

        // 1. Compute the matrix-vector product (sigma) for the basis vectors that have not been
        // computed yet
        compute_sigma();

        // 2. Form and diagonalize mini-matrix
        form_and_diagonalize_effective_hamiltonian();

        // 3. Form preconditioned residue vectors
        form_correction_vectors();

        // 4. Project out undesired roots from the correction vectors
        project_out_roots(r_);

        // 5. Print iteration summary
        print_iteration(iter);

        // 6. Check for convergence
        auto [is_energy_converged, is_residual_converged] = check_convergence();
        bool is_converged = (is_energy_converged and is_residual_converged);
        // Edge case: if the basis is orthogonal (as it should) is the same size as the
        // subspace, we are done
        bool is_edge_case = (basis_size_ == size_);
        if (is_converged or is_edge_case) {
            get_results();
            return true;
        }

        // 7. Check if we need to collapse the subspace
        if (basis_size_ + nroot_ > subspace_size_) {
            subspace_collapse();
        }

        // 8. Add the correction vectors to the basis (optionally collapsed) and orthonormalize
        // it we add one vector per root, up to the subspace size
        auto num_to_add = std::min(nroot_, subspace_size_ - basis_size_);
        auto added = add_rows_and_orthonormalize(b_, basis_size_, r_, num_to_add);
        basis_size_ += added;
        auto missing = num_to_add - added;

        // if we did not add as many vectors as we wanted, we will add orthogonal random vectors
        // to get ourself unstuck
        auto random_added = add_random_vectors(b_, basis_size_, missing);
        basis_size_ += random_added;
        added += random_added;

        // if we do not add any new vector then we are in trouble and we better finish the
        // computation
        if (added == 0) {
            if (is_residual_converged) {
                psi::outfile->Printf(
                    "\n\n  Davidson-Liu solver:  No new vectors added, but residual converged. "
                    "Finishing computation.");
                get_results();
                return true;
            } else {
                psi::outfile->Printf("\n\n  Davidson-Liu solver:  No new vectors added, and "
                                     "energy not converged. "
                                     "Finishing computation.");
                return false;
            }
        }
    }

    psi::outfile->Printf("\n\n  Davidson-Liu solver:  Maximum number of iterations reached.");
    return false;
}

void DavidsonLiuSolver2::print_iteration(size_t iter) {
    auto e_diff = lambda_->clone();
    e_diff.axpy(-1.0, *lambda_old_);

    double average_energy = 0.0;
    double max_residual = 0.0;
    double max_e_diff = 0.0;
    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        max_residual = std::max(max_residual, residual_[k]);
        max_e_diff = std::max(max_e_diff, std::fabs(e_diff.get(k)));
        average_energy += lambda_->get(k) / static_cast<double>(nroot_);
    }
    psi::outfile->Printf("\n  %6d  %20.12f  %20.12f  %20.12f %d/%d", iter, average_energy,
                         max_e_diff, max_residual, basis_size_, subspace_size_);
}

void DavidsonLiuSolver2::compute_sigma() {
    for (size_t j = sigma_size_; j < basis_size_; j++) {
        auto bj = b_->pointer()[j];
        auto sigmaj = sigma_->pointer()[j];
        sigma_builder_(std::span(bj, size_), std::span(sigmaj, size_));
    }
    // update the number of sigma vectors
    sigma_size_ = basis_size_;
    debug([&]() { sigma_->print(); });
}

void DavidsonLiuSolver2::form_and_diagonalize_effective_hamiltonian() {
    G_->zero();
    G_->gemm(false, true, 1.0, b_, sigma_, 0.0);
    G_->hermitivitize();
    // Here we need to copy the matrix to a new one because the diagonalize function will
    // otherwise include zero eigenvalues, which we do not want
    auto Gb_ = std::make_shared<psi::Matrix>("G", basis_size_, basis_size_);
    auto alpha_b_ = std::make_shared<psi::Matrix>("alpha", subspace_size_, subspace_size_);
    auto lambda_b_ = std::make_shared<psi::Vector>("lambda", subspace_size_);
    // gere
    for (size_t i = 0; i < basis_size_; i++) {
        for (size_t j = 0; j < basis_size_; j++) {
            Gb_->set(i, j, G_->get(i, j));
        }
    }
    Gb_->diagonalize(alpha_b_, lambda_b_);
    alpha_->zero();
    for (size_t i = 0; i < basis_size_; i++) {
        for (size_t j = 0; j < basis_size_; j++) {
            alpha_->set(i, j, alpha_b_->get(i, j));
        }
    }
    lambda_->zero();
    for (size_t i = 0; i < basis_size_; i++) {
        lambda_->set(i, lambda_b_->get(i));
    }
    lambda_old_->copy(*lambda_);
}

void DavidsonLiuSolver2::form_correction_vectors() {
    r_->zero();
    double* lambda_p = lambda_->pointer();
    double* Adiag_p = h_diag_->pointer();
    double** b_p = b_->pointer();
    double** f_p = r_->pointer();
    double** alpha_p = alpha_->pointer();
    double** sigma_p = sigma_->pointer();

    debug([&]() { h_diag_->print(); });
    debug([&]() { alpha_->print(); });

    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        residual_[k] = 0.0;
        for (size_t I = 0; I < size_; I++) { // loop over elements
            for (size_t i = 0; i < basis_size_; i++) {
                f_p[k][I] += alpha_p[i][k] * (sigma_p[i][I] - lambda_p[k] * b_p[i][I]);
            }
            residual_[k] += std::pow(f_p[k][I], 2.0);

            double denom = lambda_p[k] - Adiag_p[I];
            if (std::fabs(denom) > 1.0e-6) {
                f_p[k][I] /= denom;
            } else {
                // if the denominator is too small, we set the element of the correction vector
                // to 1 or -1 depending on the sign
                f_p[k][I] = f_p[k][I] * denom > 0.0 ? 1.0 : -1.0;
            }
        }
        residual_[k] = std::sqrt(residual_[k]);
    }
    debug([&]() { r_->print(); });
}

void DavidsonLiuSolver2::compute_residual_norm() {
    double* lambda_p = lambda_->pointer();
    double** b_p = b_->pointer();
    double** alpha_p = alpha_->pointer();
    double** sigma_p = sigma_->pointer();

    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        residual_[k] = 0.0;
        for (size_t I = 0; I < size_; I++) { // loop over elements
            double r = 0.0;
            for (size_t i = 0; i < basis_size_; i++) {
                r += alpha_p[i][k] * (sigma_p[i][I] - lambda_p[k] * b_p[i][I]);
            }
            residual_[k] += std::pow(r, 2.0);
        }
        residual_[k] = std::sqrt(residual_[k]);
    }
}

std::pair<bool, bool> DavidsonLiuSolver2::check_convergence() {
    compute_residual_norm();
    // check convergence on all roots
    size_t num_converged_energy = 0;
    size_t num_converged_residual = 0;
    int converged_ = 0;
    if (print_level_ > 1) {
        psi::outfile->Printf("\n  Root      Eigenvalue        Delta   Converged?\n");
        psi::outfile->Printf("  ---- -------------------- --------- ----------\n");
    }
    for (size_t k = 0; k < nroot_; k++) {
        // check if the energy converged
        double diff = std::fabs(lambda_->get(k) - lambda_old_->get(k));
        bool this_energy_converged = (diff < e_convergence_);
        // check if the residual converged
        bool this_residual_converged = (residual_[k] < r_convergence_);
        // update counters
        num_converged_energy += this_energy_converged;
        num_converged_residual += this_residual_converged;

        // update the old eigenvalue
        lambda_old_->set(k, lambda_->get(k));

        if (print_level_ > 1) {
            bool this_converged = (this_energy_converged and this_residual_converged);
            psi::outfile->Printf("  %3d  %20.14f %4.3e      %1s\n", k, lambda_->get(k), diff,
                                 this_converged ? "Y" : "N");
        }
    }

    bool is_energy_converged = (num_converged_energy == nroot_);
    bool is_residual_converged = (num_converged_residual == nroot_);
    return std::make_pair(is_energy_converged, is_residual_converged);
}

void DavidsonLiuSolver2::get_results() {
    /* generate final eigenvalues and eigenvectors */
    temp_->zero();
    double** alpha_p = alpha_->pointer();
    double** b_p = b_->pointer();
    double* eps = lambda_old_->pointer();
    double** v = temp_->pointer();

    for (size_t i = 0; i < nroot_; i++) {
        eps[i] = lambda_->get(i);
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
    b_->copy(*temp_);
}

void DavidsonLiuSolver2::set_vector(std::shared_ptr<psi::Matrix> M,
                                    const std::vector<sparse_vec>& vecs) {
    // check that we were passed less vectors than the subspace size
    if (vecs.size() > static_cast<size_t>(M->nrow())) {
        std::string msg = "DavidsonLiuSolver2: size of vecs (" + std::to_string(vecs.size()) +
                          ") must be less or equal to matrix size (" + std::to_string(M->nrow()) +
                          ")";
        throw std::runtime_error(msg);
    }
    M->zero();
    for (size_t k = 0; const auto& vec : vecs) {
        for (const auto& [I, CI] : vec) {
            M->set(k, I, CI);
        }
        k++;
    }
}

size_t DavidsonLiuSolver2::add_random_vectors(std::shared_ptr<psi::Matrix> A, size_t rowsA,
                                              size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    // generate n random vectors of size size_ and add them to the matrix A
    temp_->zero();
    for (size_t j = 0; j < n; j++) {
        auto v = temp_->pointer()[j];
        for (size_t I = 0; I < size_; I++) {
            v[I] = dist(gen); // Random number between -1 and 1
        }
    }
    auto added = add_rows_and_orthonormalize(A, rowsA, temp_, n);
    return added;
}

void DavidsonLiuSolver2::subspace_collapse() {
    debug([&]() {
        psi::outfile->Printf("\n  Subspace collapse: %d -> %d", basis_size_, collapse_size_);
    });
    // Collapse the subspace to a smaller size. Here we enforce that:
    // 1. we need to collapse to at least one vector per root (max)
    // 2. we need to collapse to at most the number of vectors in the basis (min)
    // 3. if the basis is smaller than the number of the collapse space, we pass
    auto collapsable_size = std::min(std::max(collapse_size_, nroot_), basis_size_);
    if (collapsable_size <= basis_size_) {
        return;
    }

    // collapse basis and sigma vectors (stored in bnew_ and sigma_)
    auto added = collapse_vectors(collapsable_size);

    // ensure that we have added as many vectors as we wanted
    if (added != collapsable_size) {
        std::string msg = "DavidsonLiuSolver2: could not add " + std::to_string(collapsable_size) +
                          " vectors to the basis. Only " + std::to_string(added) + " were added.";
        throw std::runtime_error(msg);
    }
    debug([&]() { b_->print(); });
    debug([&]() { sigma_->print(); });
}

size_t DavidsonLiuSolver2::collapse_vectors(size_t collapsable_size) {
    double** alpha_p = alpha_->pointer();
    double** b_p = b_->pointer();
    double** temp_p = temp_->pointer();
    double** sigma_p = sigma_->pointer();

    // collapse the basis vectors
    temp_->zero();
    for (size_t i = 0; i < collapsable_size; i++) {
        for (size_t j = 0; j < basis_size_; j++) {
            for (size_t k = 0; k < size_; k++) {
                temp_p[i][k] += alpha_p[j][i] * b_p[j][k];
            }
        }
    }
    b_->zero();
    auto added = add_rows_and_orthonormalize(b_, 0, temp_, collapsable_size);

    // collapse the sigma vectors
    temp_->zero();
    for (size_t i = 0; i < collapsable_size; i++) {
        for (size_t j = 0; j < basis_size_; j++) {
            for (size_t k = 0; k < size_; k++) {
                temp_p[i][k] += alpha_p[j][i] * sigma_p[j][k];
            }
        }
    }
    sigma_->copy(*temp_);

    basis_size_ = collapsable_size;
    sigma_size_ = collapsable_size;

    return added;
}

void DavidsonLiuSolver2::project_out_roots(std::shared_ptr<psi::Matrix> v) {
    for (size_t k = 0; k < nroot_; k++) {
        auto v_k = v->pointer()[k];
        for (auto& bad_root : project_out_vectors_) {
            double overlap = 0.0;
            for (const auto& [I, CI] : bad_root) {
                overlap += v_k[I] * CI;
            }
            for (const auto& [I, CI] : bad_root) {
                v_k[I] -= overlap * CI;
            }
        }
    }
}

/// add a new vector and orthogonalize it against the existing ones
size_t DavidsonLiuSolver2::add_rows_and_orthonormalize(std::shared_ptr<psi::Matrix> A, size_t rowsA,
                                                       std::shared_ptr<psi::Matrix> B,
                                                       size_t rowsB) {
    // sanity checks
    // rowsA + rowsB must be less than the number of rows of A
    if (rowsA + rowsB > static_cast<size_t>(A->nrow())) {
        std::string msg = "DavidsonLiuSolver2: rowsA + rowsB (" + std::to_string(rowsA + rowsB) +
                          ") must be less or equal to matrix size (" + std::to_string(A->nrow()) +
                          ")";
        throw std::runtime_error(msg);
    }
    // rowsB must be less than or equal to the number of rows of B
    if (rowsB > static_cast<size_t>(B->nrow())) {
        std::string msg = "DavidsonLiuSolver2: rowsB (" + std::to_string(rowsB) +
                          ") must be less or equal to matrix size (" + std::to_string(B->nrow()) +
                          ")";
        throw std::runtime_error(msg);
    }

    size_t added = 0;
    for (size_t j = 0; j < rowsB; j++) {
        auto success = add_row_and_orthonormalize(A, rowsA + added, B, j);
        if (success) {
            added++;
        }
    }
    return added;
}

/// add a new vector and orthogonalize it against the existing ones
bool DavidsonLiuSolver2::add_row_and_orthonormalize(std::shared_ptr<psi::Matrix> A, size_t rowsA,
                                                    std::shared_ptr<psi::Matrix> B, size_t rowB) {
    // Assume that A is a matrix with num_A orthonormal rows
    size_t ncols = A->ncol();

    // the new vector is the row rowB of B
    auto b = B->pointer()[rowB];
    // copy the b into the rowsA + 1 row of A. Call this vector v to keep it nice and short
    auto v = A->pointer()[rowsA];
    for (size_t I = 0; I < ncols; I++)
        v[I] = b[I];

    // here we do the schmidt orthogonalization several times. Often, one step is enough
    // but sometimes it takes more than one step to guarantee orthogonality to within
    // a tight tolerance. The option that controls this is schmidt_orthogonality_threshold_
    int max_orthogonalization_cycles = 10;
    for (int cycle = 0; cycle < max_orthogonalization_cycles; cycle++) {
        // schmidt orthogonalize the j-th row of rowsA + j row of A against the rows of A
        for (size_t i = 0; i < rowsA; i++) {
            auto Ai = A->pointer()[i];
            const auto dotval = psi::C_DDOT(ncols, Ai, 1, v, 1);
            for (size_t I = 0; I < ncols; I++)
                v[I] -= dotval * Ai[I];
        }
        // compute the norm of the vector
        const auto normval = std::sqrt(psi::C_DDOT(ncols, v, 1, v, 1));

        // if the norm is small, discard the vector
        if (normval < schmidt_discard_threshold_)
            return false;
        // normalize the vector
        for (size_t I = 0; I < ncols; I++)
            v[I] *= 1. / normval;

        // check the overlap with the previous vectors
        double max_overlap = 0.0;
        for (size_t i = 0; i < rowsA; i++) {
            auto Ai = A->pointer()[i];
            max_overlap = std::max(max_overlap, std::fabs(psi::C_DDOT(ncols, Ai, 1, v, 1)));
        }
        // compute the norm of the vector (again)
        double norm = psi::C_DDOT(ncols, v, 1, v, 1);

        // if the vector is orthogonal to the previous ones, and it is normalized, we're
        // done
        if ((max_overlap < schmidt_orthogonality_threshold_) and
            (std::fabs(norm - 1.0) < schmidt_orthogonality_threshold_)) {
            return true;
        }
    }

    // if we did not converge, return false
    return false;
}

void DavidsonLiuSolver2::check_orthogonality() {
    // here we use a looser threshold than the one used in the schmidt orthogonalization
    double orthogonality_threshold = schmidt_orthogonality_threshold_ * 3.0;

    // Compute the overlap matrix
    S_->gemm(false, true, 1.0, b_, b_, 0.0);

    // Check for normalization
    double maxdiag = 0.0;
    for (size_t i = 0; i < basis_size_; ++i) {
        maxdiag = std::max(maxdiag, std::fabs(S_->get(i, i) - 1.0));
    }
    if (maxdiag > orthogonality_threshold) {
        psi::outfile->Printf("\n\nDavidsonLiuSolver::check_orthogonality(): eigenvectors are not "
                             "normalized!\nMaximum absolute deviation from normalization: %e",
                             maxdiag);
        S_->print();
        std::string msg =
            "DavidsonLiuSolver::check_orthogonality(): eigenvectors are not normalized!";
        throw std::runtime_error(msg);
    }

    // Check for orthogonality
    double maxoffdiag = 0.0;
    for (size_t i = 0; i < basis_size_; ++i) {
        for (size_t j = 0; j < basis_size_; ++j) {
            if (i != j) {
                maxoffdiag = std::max(maxoffdiag, std::fabs(S_->get(i, j)));
            }
        }
    }
    if (maxoffdiag > orthogonality_threshold) {
        psi::outfile->Printf(
            "\n\nDavidsonLiuSolver::check_orthogonality(): eigenvectors are not "
            "normalized!\nMaximum absolute off-diagonal element of the overlap: %e",
            maxoffdiag);
        S_->print();
        std::string msg =
            "DavidsonLiuSolver::check_orthogonality(): eigenvectors are not orthogonal!";
        throw std::runtime_error(msg);
    }
}

} // namespace forte