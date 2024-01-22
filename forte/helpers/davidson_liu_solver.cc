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

#include <random>

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

DavidsonLiuSolver::DavidsonLiuSolver(size_t size, size_t nroot, size_t collapse_per_root,
                                     size_t subspace_per_root)
    : size_(size), nroot_(nroot), collapse_per_root_(collapse_per_root),
      subspace_per_root_(subspace_per_root) {

    startup();
}

size_t DavidsonLiuSolver::size() const { return size_; }

void DavidsonLiuSolver::startup() {
    // sanity checks
    // collapse_per_root_ must greater than or equal to one. This guarantees that we have at least
    // one vector per root after the collapse
    if (size_ == 0)
        throw std::runtime_error("DavidsonLiuSolver called with space of dimension zero.");
    if (nroot_ == 0)
        throw std::runtime_error("DavidsonLiuSolver called with zero roots.");
    if (collapse_per_root_ < 1) {
        std::string msg =
            "DavidsonLiuSolver: collapse_per_root_ (" + std::to_string(collapse_per_root_) +
            ") must be greater or equal to the number of roots (" + std::to_string(nroot_) + ")";
        throw std::runtime_error(msg);
    }
    // subspace_per_root_ should be at least one greater than collapse_per_root_
    if (subspace_per_root_ < collapse_per_root_ + 1) {
        std::string msg = "DavidsonLiuSolver: subspace_per_root_ (" +
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

    residual_2norm_.resize(nroot_, 0.0);
}

void DavidsonLiuSolver::print_table() {
    if (print_ < PrintLevel::Default)
        return;

    // Print a summary of the calculation options
    table_printer printer;
    printer.add_double_data({{"Energy convergence threshold", e_convergence_},
                             {"Residual convergence threshold", r_convergence_},
                             {"Schmidt orthogonality threshold", schmidt_orthogonality_threshold_},
                             {"Schmidt discard threshold", schmidt_discard_threshold_}});

    printer.add_int_data({
        {"Size of the space", size_},
        {"Number of roots", nroot_},
        {"Maximum number of iterations", max_iter_},
        {"Collapse subspace size", collapse_size_},
        {"Maximum subspace size", subspace_size_},
    });

    printer.add_string_data({{"Print level", to_string(print_)}});

    std::string table = printer.get_table("Davidson-Liu Solver");
    psi::outfile->Printf("%s", table.c_str());
}

void DavidsonLiuSolver::add_h_diag(std::shared_ptr<psi::Vector> h_diag) {
    if (static_cast<size_t>(h_diag->dim()) != size_) {
        std::string msg = "DavidsonLiuSolver: h_diag vector size (" +
                          std::to_string(h_diag_->dim()) + ") must be equal to space size (" +
                          std::to_string(size_) + ")";
        throw std::runtime_error(msg);
    }
    if (h_diag_ == nullptr) {
        h_diag_ = std::make_shared<psi::Vector>("h_diag", size_);
    }
    h_diag_->copy(*h_diag);
}

void DavidsonLiuSolver::add_guesses(const std::vector<sparse_vec>& guesses) { guesses_ = guesses; }

void DavidsonLiuSolver::add_project_out_vectors(
    const std::vector<sparse_vec>& project_out_vectors) {
    project_out_vectors_ = project_out_vectors;
}

void DavidsonLiuSolver::add_sigma_builder(
    std::function<void(std::span<double>, std::span<double>)> sigma_builder) {
    sigma_builder_ = sigma_builder;
}

void DavidsonLiuSolver::reset() {
    basis_size_ = 0;
    sigma_size_ = 0;
    b_->zero();
    sigma_->zero();
}

void DavidsonLiuSolver::set_print_level(PrintLevel level) { print_ = level; }

void DavidsonLiuSolver::set_e_convergence(double value) { e_convergence_ = value; }

void DavidsonLiuSolver::set_r_convergence(double value) { r_convergence_ = value; }

void DavidsonLiuSolver::set_maxiter(size_t n) { max_iter_ = n; }

std::shared_ptr<psi::Vector> DavidsonLiuSolver::eigenvalues() const { return lambda_; }

std::shared_ptr<psi::Matrix> DavidsonLiuSolver::eigenvectors() const { return b_; }

std::shared_ptr<psi::Vector> DavidsonLiuSolver::eigenvector(size_t n) const {
    const auto v_n = b_->pointer()[n];
    auto evec = std::make_shared<psi::Vector>("V", size_);
    for (size_t I = 0; I < size_; I++) {
        evec->set(I, v_n[I]);
    }
    return evec;
}

bool DavidsonLiuSolver::solve() {
    print_table();

    setup_guesses();

    preiteration_sanity_checks();

    print_header();

    for (size_t iter = 0; iter < max_iter_; iter++) {
        // ensure that the basis is orthonormal
        check_orthonormality();

        // 1. Compute the matrix-vector product (sigma) for the basis vectors that have not been
        // computed yet
        compute_sigma();

        // 2. Form and diagonalize mini-matrix
        form_and_diagonalize_effective_hamiltonian();

        // 3. Form preconditioned residue vectors
        form_residual_vectors();
        compute_residual_norm();
        form_correction_vectors();

        // 4. Project out undesired roots from the correction vectors
        project_out_roots(r_);

        normalize_vectors(r_, nroot_);

        // 5. Print iteration summary
        print_iteration(iter);

        // 6. Check for convergence
        auto [is_energy_converged, is_residual_converged] = check_convergence();
        bool is_converged = (is_energy_converged and is_residual_converged);
        // Edge case: if the basis is the same size as the subspace, we are done
        bool is_edge_case = (basis_size_ == size_);
        if (is_converged or is_edge_case) {
            print_footer();
            get_results();
            return true;
        }

        // 7. Check if we need to collapse the subspace
        if (basis_size_ + nroot_ > subspace_size_) {
            subspace_collapse();
        }

        // 8. Add the correction vectors to the basis (optionally collapsed) and orthonormalize
        // it. We add one vector per root, up to the subspace size
        auto num_to_add = std::min(nroot_, subspace_size_ - basis_size_);
        auto added = add_rows_and_orthonormalize(b_, basis_size_, r_, num_to_add);
        basis_size_ += added;
        auto missing = num_to_add - added;

        // if we did not add as many vectors as we wanted, we will add orthogonal random vectors
        // to get ourself unstuck
        if (missing > 0) {
            psi::outfile->Printf(" <- added %d random vector%s", missing, missing > 1 ? "s" : "");
            temp_->zero();
            add_random_vectors(temp_, 0, missing);
            project_out_roots(temp_);
            auto random_added = add_rows_and_orthonormalize(b_, basis_size_, temp_, missing);
            basis_size_ += random_added;
            added += random_added;
        }

        // if we do not add any new vector then we are in trouble and we better finish the
        // computation
        if (added == 0) {
            print_footer();
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
                get_results();
                return false;
            }
        }
        lambda_old_->copy(*lambda_);
    }

    print_footer();
    psi::outfile->Printf("\n\n  Davidson-Liu solver:  Maximum number of iterations reached.");
    get_results();
    return false;
}

void DavidsonLiuSolver::setup_guesses() {
    // Add the initial guess to the basis and orthonormalize it
    if (basis_size_ == 0) {
        // add the guesses to temp
        if ((guesses_.size() >= nroot_) and (guesses_.size() <= subspace_size_)) {
            // guesses [copy] -> temp [orthonormalize] -> b
            if (print_ >= PrintLevel::Default) {
                psi::outfile->Printf("\n\n  Davidson-Liu solver: adding %d guess vectors",
                                     guesses_.size());
            }
            set_vector(temp_, guesses_);

        } else if (guesses_.size() == 0) {
            // add random vectors
            temp_->zero();
            add_random_vectors(temp_, 0, nroot_);
            if (print_ >= PrintLevel::Default) {
                psi::outfile->Printf("\n\n  Davidson-Liu solver: adding %d random vectors", nroot_);
            }
        } else {
            std::string msg = "DavidsonLiuSolver: number of guess vectors (" +
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
            std::string msg = "DavidsonLiuSolver: guess vectors are zero or linearly dependent";
            throw std::runtime_error(msg);
        }
        basis_size_ += added;
    } else if (basis_size_ >= nroot_) {
        psi::outfile->Printf("\n\n  Davidson-Liu solver: restarting from previous calculation.");
        basis_size_ = nroot_; // first nroot_ vectors from the previous calculation
        sigma_size_ = 0;      // trigger computation of all sigma vectors
    } else {
        std::string msg =
            "DavidsonLiuSolver: number of guess vectors (" + std::to_string(guesses_.size()) +
            ") must be between 0 and the subspace size (" + std::to_string(subspace_size_) + ")";
        throw std::runtime_error(msg);
    }
}

void DavidsonLiuSolver::preiteration_sanity_checks() {
    // check that the sigma builder has been set
    if (sigma_builder_ == nullptr) {
        std::string msg = "DavidsonLiuSolver: sigma builder has not been set";
        throw std::runtime_error(msg);
    }

    // ensure that h_diag was set
    if (h_diag_ == nullptr) {
        std::string msg = "DavidsonLiuSolver: h_diag has not been set";
        throw std::runtime_error(msg);
    }
}

void DavidsonLiuSolver::print_header() {
    if (print_ < PrintLevel::Default)
        return;
    psi::outfile->Printf(
        "\n  Iteration     Average Energy            max(∆E)            max(Residual)  Vectors");
    psi::outfile->Printf(
        "\n  ---------------------------------------------------------------------------------");
}

void DavidsonLiuSolver::print_footer() {
    if (print_ < PrintLevel::Default)
        return;
    psi::outfile->Printf(
        "\n  ---------------------------------------------------------------------------------");
}

void DavidsonLiuSolver::print_iteration(size_t iter) {
    if (print_ < PrintLevel::Default)
        return;

    auto e_diff = lambda_->clone();
    e_diff.axpy(-1.0, *lambda_old_);

    double average_energy = 0.0;
    double max_residual = 0.0;
    double max_e_diff = 0.0;
    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        max_residual = std::max(max_residual, residual_2norm_[k]);
        max_e_diff = std::max(max_e_diff, std::fabs(e_diff.get(k)));
        average_energy += lambda_->get(k) / static_cast<double>(nroot_);
    }
    psi::outfile->Printf("\n  %6d  %20.12f  %20.12f  %20.12f %6d", iter, average_energy, max_e_diff,
                         max_residual, basis_size_);
}

void DavidsonLiuSolver::compute_sigma() {
    for (size_t j = sigma_size_; j < basis_size_; j++) {
        auto bj = b_->pointer()[j];
        auto sigmaj = sigma_->pointer()[j];
        sigma_builder_(std::span(bj, size_), std::span(sigmaj, size_));
    }
    // update the number of sigma vectors
    sigma_size_ = basis_size_;
    debug([&]() { sigma_->print(); });
}

void DavidsonLiuSolver::form_and_diagonalize_effective_hamiltonian() {
    G_->gemm(false, true, 1.0, b_, sigma_, 0.0);
    G_->hermitivitize();
    // Here we need to copy the matrix to a new one because the diagonalize function will
    // otherwise include zero eigenvalues, which we do not want
    auto Gb_ = std::make_shared<psi::Matrix>("G", basis_size_, basis_size_);
    auto alpha_b_ = std::make_shared<psi::Matrix>("alpha", basis_size_, basis_size_);
    auto lambda_b_ = std::make_shared<psi::Vector>("lambda", basis_size_);
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
}

void DavidsonLiuSolver::form_residual_vectors() {
    r_->zero();

    debug([&]() { h_diag_->print(); });
    debug([&]() { alpha_->print(); });

    r_->gemm(true, false, 1.0, alpha_, sigma_, 0.0);
    temp_->gemm(true, false, 1.0, alpha_, b_, 0.0);

    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        const auto lambda_k = lambda_->get(k);
        const auto temp_k = temp_->pointer()[k];
        auto r_k = r_->pointer()[k];
        for (size_t I = 0; I < size_; I++) { // loop over elements
            r_k[I] -= lambda_k * temp_k[I];
        }
    }
    debug([&]() { r_->print(); });
}

void DavidsonLiuSolver::form_correction_vectors() {
    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        auto r_k = r_->pointer()[k];
        const auto lambda_k = lambda_->get(k);
        for (size_t I = 0; I < size_; I++) { // loop over elements
            double denom = lambda_k - h_diag_->get(I);
            if (std::fabs(denom) > 1.0e-6) {
                r_k[I] /= denom;
            } else {
                // if the denominator is too small, we set the element of the correction vector
                // to 1 or -1 depending on the sign
                r_k[I] = 0.0; // r_k[I] * denom > 0.0 ? 1.0 : -1.0;
            }
        }
    }
    debug([&]() { r_->print(); });
}

void DavidsonLiuSolver::compute_residual_norm() {
    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        auto r_k = r_->pointer()[k];
        residual_2norm_[k] = std::sqrt(psi::C_DDOT(size_, r_k, 1, r_k, 1));
    }
}

void DavidsonLiuSolver::normalize_vectors(std::shared_ptr<psi::Matrix> M, size_t n) {
    for (size_t k = 0; k < n; k++) { // loop over roots
        auto v_k = M->pointer()[k];
        double norm = std::sqrt(psi::C_DDOT(size_, v_k, 1, v_k, 1));
        for (size_t I = 0; I < size_; I++) { // loop over elements
            v_k[I] /= norm;
        }
    }
}

std::pair<bool, bool> DavidsonLiuSolver::check_convergence() {
    // compute_residual_norm();
    // check convergence on all roots
    size_t num_converged_energy = 0;
    size_t num_converged_residual = 0;
    for (size_t k = 0; k < nroot_; k++) {
        // check if the energy converged
        double diff = std::fabs(lambda_->get(k) - lambda_old_->get(k));
        bool this_energy_converged = (diff < e_convergence_);
        // check if the residual converged
        bool this_residual_converged = (residual_2norm_[k] < r_convergence_);
        // update counters
        num_converged_energy += this_energy_converged;
        num_converged_residual += this_residual_converged;
    }

    bool is_energy_converged = (num_converged_energy == nroot_);
    bool is_residual_converged = (num_converged_residual == nroot_);
    return std::make_pair(is_energy_converged, is_residual_converged);
}

void DavidsonLiuSolver::get_results() {
    // copy the eigenvalues
    lambda_old_->copy(*lambda_);
    // generate final eigenvectors
    temp_->gemm(true, false, 1.0, alpha_, b_, 0.0);
    b_->zero();
    auto added = add_rows_and_orthonormalize(b_, 0, temp_, nroot_);
    if (added != nroot_) {
        std::string msg = "DavidsonLiuSolver: get_results generated less vectors (" +
                          std::to_string(added) + ") than expected (" + std::to_string(nroot_) +
                          ")";
        throw std::runtime_error(msg);
    }
}

void DavidsonLiuSolver::set_vector(std::shared_ptr<psi::Matrix> M,
                                   const std::vector<sparse_vec>& vecs) {
    // check that we were passed less vectors than the subspace size
    if (vecs.size() > static_cast<size_t>(M->nrow())) {
        std::string msg = "DavidsonLiuSolver: size of vecs (" + std::to_string(vecs.size()) +
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

size_t DavidsonLiuSolver::add_random_vectors(std::shared_ptr<psi::Matrix> A, size_t rowsA,
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

void DavidsonLiuSolver::subspace_collapse() {
    debug([&]() {
        psi::outfile->Printf("\n  Subspace collapse: %d -> %d", basis_size_, collapse_size_);
    });
    // Collapse the subspace to a smaller size. Here we enforce that:
    // 1. we need to collapse to at least one vector per root (max)
    // 2. we need to collapse to at most the number of vectors in the basis (min)
    // 3. if the basis is smaller than the number of the collapse space, we pass
    auto collapsable_size = std::min(std::max(collapse_size_, nroot_), basis_size_);
    debug([&]() { psi::outfile->Printf("\n  Collapsing to %d vectors", collapsable_size); });
    // I think the line below is not needed or should be changed to >=
    // if (collapsable_size <= basis_size_) {
    //     return;
    // }

    // collapse basis and sigma vectors (stored in bnew_ and sigma_)
    auto added = collapse_vectors(collapsable_size);

    // ensure that we have added as many vectors as we wanted
    if (added != collapsable_size) {
        std::string msg = "DavidsonLiuSolver: could not add " + std::to_string(collapsable_size) +
                          " vectors to the basis. Only " + std::to_string(added) + " were added.";
        throw std::runtime_error(msg);
    }
    debug([&]() { b_->print(); });
    debug([&]() { sigma_->print(); });
}

size_t DavidsonLiuSolver::collapse_vectors(size_t collapsable_size) {
    // collapse the basis vectors
    temp_->gemm(true, false, 1.0, alpha_, b_, 0.0);
    b_->zero();
    auto added = add_rows_and_orthonormalize(b_, 0, temp_, collapsable_size);

    // collapse the sigma vectors
    temp_->gemm(true, false, 1.0, alpha_, sigma_, 0.0);
    sigma_->copy(*temp_);

    if (added != collapsable_size) {
        std::string msg = "DavidsonLiuSolver: collapse_vectors generated less vectors (" +
                          std::to_string(added) + ") than expected (" +
                          std::to_string(collapsable_size) + ")";

        throw std::runtime_error(msg);
    }

    basis_size_ = collapsable_size;
    sigma_size_ = collapsable_size;

    return added;
}

void DavidsonLiuSolver::project_out_roots(std::shared_ptr<psi::Matrix> v) {
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

size_t DavidsonLiuSolver::add_rows_and_orthonormalize(std::shared_ptr<psi::Matrix> A, size_t rowsA,
                                                      std::shared_ptr<psi::Matrix> B,
                                                      size_t rowsB) {
    // sanity checks
    // rowsA + rowsB must be less than the number of rows of A
    if (rowsA + rowsB > static_cast<size_t>(A->nrow())) {
        std::string msg = "DavidsonLiuSolver: rowsA + rowsB (" + std::to_string(rowsA + rowsB) +
                          ") must be less or equal to matrix size (" + std::to_string(A->nrow()) +
                          ")";
        throw std::runtime_error(msg);
    }
    // rowsB must be less than or equal to the number of rows of B
    if (rowsB > static_cast<size_t>(B->nrow())) {
        std::string msg = "DavidsonLiuSolver: rowsB (" + std::to_string(rowsB) +
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

bool DavidsonLiuSolver::add_row_and_orthonormalize(std::shared_ptr<psi::Matrix> A, size_t rowsA,
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

void DavidsonLiuSolver::check_orthonormality() {
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