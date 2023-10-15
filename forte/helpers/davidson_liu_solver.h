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

#pragma once

#include <functional>
#include <vector>
#include <stdexcept>
#include <span>

#include "psi4/libqt/qt.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"

namespace psi {
class Vector;
class Matrix;
} // namespace psi

namespace forte {

/// @brief A class to solve the symmetric eigenvalue problem using the Davidson-Liu algorithm
/// @details This class implements the Davidson-Liu algorithm to solve the symmetric eigenvalue
/// problem.
class DavidsonLiuSolver {
    using sparse_vec = std::vector<std::pair<size_t, double>>;

  public:
    DavidsonLiuSolver(size_t size, size_t nroot, size_t set_collapse_per_root = 1,
                      size_t set_subspace_per_root = 5);

    /// Setup the solver
    void add_sigma_builder(std::function<void(std::span<double>, std::span<double>)> sigma_builder);
    void add_h_diag(std::shared_ptr<psi::Vector> h_diag);
    void add_guesses(const std::vector<sparse_vec>& guesses);
    void add_project_out_vectors(const std::vector<sparse_vec>& project_out_vectors);

    /// Set the print level
    void set_print_level(size_t n);
    /// Set the energy convergence
    void set_e_convergence(double value);
    /// Set the residual convergence
    void set_r_convergence(double value);

    /// Function to reset the solver
    void reset();

    /// @brief The main solver function
    /// @return true if the solver converged
    bool solve();

    size_t size() const;

    /// Return the eigenvalues
    std::shared_ptr<psi::Vector> eigenvalues() const;
    /// Return the eigenvectors
    std::shared_ptr<psi::Matrix> eigenvectors() const;
    /// Return the n-th eigenvector
    std::shared_ptr<psi::Vector> eigenvector(size_t n) const;

  private:
    // ==> Class Private Data <==

    // Passed in by the user at construction

    /// The dimension of the vectors
    const size_t size_;
    /// The number of roots requested
    const size_t nroot_;
    /// The number of collapse vectors for each root
    const size_t collapse_per_root_;
    /// The maximum subspace size for each root
    const size_t subspace_per_root_;

    // Passed in by the user at setup
    /// The sigma builder function
    std::function<void(std::span<double>, std::span<double>)> sigma_builder_;
    /// Diagonal elements of the Hamiltonian
    std::shared_ptr<psi::Vector> h_diag_;
    /// The initial guess
    std::vector<sparse_vec> guesses_;
    /// The vectors to project out
    std::vector<sparse_vec> project_out_vectors_;

    /// The print level
    size_t print_level_ = 1;
    /// The maximum number of iterations
    size_t max_iter_ = 50;
    /// Eigenvalue convergence threshold
    double e_convergence_ = 1.0e-12;
    /// Residual convergence threshold
    double r_convergence_ = 1.0e-6;
    /// The threshold used to discard correction vectors
    double schmidt_discard_threshold_ = 1.0e-7;
    /// The threshold used to guarantee orthogonality among the roots
    double schmidt_orthogonality_threshold_ = 1.0e-12;
    /// The number of vectors to retain after collapse
    size_t collapse_size_;
    /// The maximum subspace size
    size_t subspace_size_;
    /// The number of basis vectors currently stored
    size_t basis_size_;
    /// The number of sigma vectors currently stored
    size_t sigma_size_;

    /// A matrix to store temporary results
    std::shared_ptr<psi::Matrix> temp_;
    /// Current set of basis vectors stored by row
    std::shared_ptr<psi::Matrix> b_;
    /// Residual eigenvectors, stored by row
    std::shared_ptr<psi::Matrix> r_;
    /// Sigma vectors, stored by row
    std::shared_ptr<psi::Matrix> sigma_;
    /// Davidson-Liu mini-Hamitonian
    std::shared_ptr<psi::Matrix> G_;
    /// Davidson-Liu mini-metric
    std::shared_ptr<psi::Matrix> S_;
    /// Eigenvectors of the Davidson mini-Hamitonian
    std::shared_ptr<psi::Matrix> alpha_;
    /// Eigenvalues of the Davidson mini-Hamitonian
    std::shared_ptr<psi::Vector> lambda_;
    /// Old eigenvalues of the Davidson mini-Hamitonian
    std::shared_ptr<psi::Vector> lambda_old_;
    /// 2-Norm of the residuals
    std::vector<double> residual_2norm_;

    /// Allocate memory for the solver
    void startup();

    /// Print the solver variables
    void print_table();

    /// Set the initial guesses
    void setup_guesses();

    /// Run sanity checks before the iteration starts
    void preiteration_sanity_checks();

    /// Print the header of the iteration table
    void print_header();
    /// Print the current iteration
    void print_iteration(size_t iter);
    /// Print the footer of the iteration table
    void print_footer();

    /// Compute the sigma vectors for the new basis vectors
    void compute_sigma();

    /// Form the effective Hamiltonian and diagonalize it
    void form_and_diagonalize_effective_hamiltonian();

    /// Collapse the subspace
    void subspace_collapse();

    /// Check if the the iterative procedure has converged
    /// @return a pair of boolean (is_energy_converged,is_residual_converged)
    std::pair<bool, bool> check_convergence();

    /// Form the residual vectors r = (Hc - Ec)
    void form_residual_vectors();

    /// Compute the 2-norm of the residual
    void compute_residual_norm();

    /// Form the correction vectors c = r / preconditioner
    void form_correction_vectors();

    /// Normalize the first n rows of a matrix
    /// @param M the matrix to normalize
    /// @param n the number of rows to normalize
    void normalize_vectors(std::shared_ptr<psi::Matrix> M, size_t n);

    /// Perform an update step that saves the final results in the class variables
    void get_results();

    /// Perform the actual collapse
    size_t collapse_vectors(size_t collapsable_size);

    /// Add random rows to a matrix
    /// @param A the matrix to add the rows to
    /// @param rowsA the rows of A where we can add the new rows
    /// @param n the number of rows to add
    size_t add_random_vectors(std::shared_ptr<psi::Matrix> A, size_t rowsA, size_t n);

    /// Check that the eigenvectors are orthonormal. Here we throw if the check fails
    void check_orthonormality();

    /// Project out undesired roots from a matrix
    void project_out_roots(std::shared_ptr<psi::Matrix> v);

    /// @brief Add rows to a matrix and orthonormalize them
    /// @param A the matrix to add the rows to
    /// @param rowsA the number of rows in A (assumed to be orthonormal)
    /// @param B the matrix containing the rows to add
    /// @param rowsB the number of rows in B to add
    /// @return the number of rows added to A
    size_t add_rows_and_orthonormalize(std::shared_ptr<psi::Matrix> A, size_t rowsA,
                                       std::shared_ptr<psi::Matrix> B, size_t rowsB);

    /// @brief Add one row to a matrix and orthonormalize it with respect to the other rows
    /// @param A the matrix to add the row to
    /// @param rowsA the number of rows in A (assumed to be orthonormal)
    /// @param B the matrix containing the rows to add
    /// @param rowB the row in B to add
    /// @return the if this row was added to A
    bool add_row_and_orthonormalize(std::shared_ptr<psi::Matrix> A, size_t rowsA,
                                    std::shared_ptr<psi::Matrix> B, size_t rowB);

    /// Set a dense matrix from a vector of sparse vectors
    void set_vector(std::shared_ptr<psi::Matrix> M, const std::vector<sparse_vec>& vecs);
};

} // namespace forte
