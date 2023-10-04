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

#ifndef _iterative_solvers_h_
#define _iterative_solvers_h_

#include "psi4/libqt/qt.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"

namespace forte {

/// Result of the update step
enum class SolverStatus { Converged, NotConverged, Collapse };

/**
 * @brief The DavidsonLiuSolver class
 * This class diagonalizes the Hamiltonian in a basis
 * of determinants.
 *
 * Example use:
 *
 *     DavidsonLiuSolver dls(nbasis,nroots);      // create solver
 *     dls.startup(diagonal);                     // initialize solver with
 * diagonal of Matrix
 *     size_t guess_size = dls.collapse_size();   // get the size of the guess
 *
 *     for (size_t n = 0; n < guess_size; ++n){
 *         dls.add_b(guess[n]);                   // add guess vectors
 *     }
 *
 *     bool converged = false;
 *     for (int cycle = 0; cycle < max_cycle; ++cycle){
 *         do{
 *             dls.get_b(b);                      // solver provides a vector b
 *             ...                                // code to compute sigma = Hb
 *             add_sigma = dls.add_sigma(sigma);  // return sigma vector to solver
 *         } while (add_sigma);
 *         converged = dls.update();              // check convergence
 *         if (converged == Converged) break;
 *     }
 */
class DavidsonLiuSolver {
    using sparse_vec = std::vector<std::pair<size_t, double>>;

  public:
    // ==> Class Constructor and Destructor <==

    /// Constructor
    DavidsonLiuSolver(size_t size, size_t nroot);

    /// Destructor
    ~DavidsonLiuSolver();

    // ==> Class Interface <==

    /// Set the print level
    void set_print_level(size_t n);
    /// Set the energy convergence
    void set_e_convergence(double value);
    /// Set the residual convergence
    void set_r_convergence(double value);
    /// Get the energy convergence
    double get_e_convergence() const;
    /// Get the residual convergence
    double get_r_convergence() const;
    /// Set the number of collapse vectors for each root
    void set_collapse_per_root(int value);
    /// Set the maximum subspace size for each root
    void set_subspace_per_root(int value);
    /// Save the state of the solver to a file at distruction?
    void set_save_state_at_destruction(bool value);

    /// Return the size of the collapse vectors
    size_t collapse_size() const;

    /// Add a guess basis vector
    void add_guess(std::shared_ptr<psi::Vector> vec);
    /// Get a basis vector
    void get_b(std::shared_ptr<psi::Vector> vec);
    /// Add a sigma vector
    bool add_sigma(std::shared_ptr<psi::Vector> vec);

    void set_project_out(std::vector<sparse_vec> project_out);

    /// Return the eigenvalues
    std::shared_ptr<psi::Vector> eigenvalues() const;
    /// Return the eigenvectors
    std::shared_ptr<psi::Matrix> eigenvectors() const;
    /// Return the n-th eigenvector
    std::shared_ptr<psi::Vector> eigenvector(size_t n) const;

    /// Initialize the object
    size_t startup(std::shared_ptr<psi::Vector> diagonal);

    /// Return the size of the subspace
    size_t size() const;

    /// Perform an update step
    SolverStatus update();

    /// Produce the final eigenvectors and eigenvalues
    void get_results();

    /// A vector with the 2-norm of the residual for each root
    std::vector<double> residuals() const;

    /// Save the state of the solver to a file
    void save_state() const;

    /// Load the state of the solver from a file and return the size of the subspace
    size_t load_state();

  private:
    // ==> Class Private Functions <==

    /// Check that the eigenvectors are orthogonal. Here we use a  Throws if fails
    void check_orthogonality();
    /// Check if the the iterative procedure has converged
    /// @return a pair of boolean (is_energy_converged,is_residual_converged)
    std::pair<bool, bool> check_convergence();
    /// Build the correction vectors
    void form_correction_vectors();
    /// Compute the 2-norm of the residual
    void compute_residual_norm();
    /// Project out undesired roots
    void project_out_roots(std::shared_ptr<psi::Matrix> v);
    /// Perform the Schmidt orthogonalization (add a new vector to the subspace)
    bool schmidt_add(std::shared_ptr<psi::Matrix> Amat, size_t rows, size_t cols,
                     std::shared_ptr<psi::Matrix> vvec, int l);
    /// Normalize the correction vectors and return the norm of the vectors before they were
    /// normalized
    std::vector<double> normalize_vectors(std::shared_ptr<psi::Matrix> v, size_t n);
    /// Perform subspace collapse
    bool subspace_collapse();
    /// Collapse the vectors
    void collapse_vectors(size_t collapsable_size);

    // ==> Class Private Data <==

    /// The print level
    size_t print_level_ = 1;
    /// Eigenvalue convergence threshold
    double e_convergence_ = 1.0e-12;
    /// Residual convergence threshold
    double r_convergence_ = 1.0e-6;
    /// The threshold used to discard correction vectors
    double schmidt_discard_threshold_ = 1.0e-7;
    /// The threshold used to guarantee orthogonality among the roots
    double schmidt_orthogonality_threshold_ = 1.0e-12;
    /// The dimension of the vectors
    size_t size_;
    /// The number of roots requested
    size_t nroot_;
    /// The number of collapse vectors for each root
    size_t collapse_per_root_ = 2;
    /// The maximum subspace size for each root
    size_t subspace_per_root_ = 5;
    /// The number of vectors to retain after collapse
    size_t collapse_size_;
    /// The maximum subspace size
    size_t subspace_size_;

    /// The number of iterations performed
    int iter_ = 0;
    /// The number of basis vectors
    size_t basis_size_;
    /// The number of sigma vectors
    size_t sigma_size_;
    /// The number of converged roots
    size_t converged_ = 0;
    /// Timing information
    double timing_ = 0.0;
    /// Did we collapse the subspace in the last update?
    bool last_update_collapsed_ = false;
    /// Should we save the state of the solver to a file at distruction?
    bool save_state_at_destruction_ = false;

    /// Current set of basis vectors stored by row
    std::shared_ptr<psi::Matrix> b_;
    /// Guess vectors formed from old vectors, stored by row
    std::shared_ptr<psi::Matrix> bnew;
    /// Residual eigenvectors, stored by row
    std::shared_ptr<psi::Matrix> f;
    /// Sigma vectors, stored by column
    std::shared_ptr<psi::Matrix> sigma_;
    /// Davidson-Liu mini-Hamitonian
    std::shared_ptr<psi::Matrix> G;
    /// Davidson-Liu mini-metric
    std::shared_ptr<psi::Matrix> S;
    /// Eigenvectors of the Davidson mini-Hamitonian
    std::shared_ptr<psi::Matrix> alpha;

    /// Eigenvalues of the Davidson mini-Hamitonian
    std::shared_ptr<psi::Vector> lambda;
    /// Old eigenvalues of the Davidson mini-Hamitonian
    std::shared_ptr<psi::Vector> lambda_old;
    /// Diagonal elements of the Hamiltonian
    std::shared_ptr<psi::Vector> h_diag;
    /// 2-Norm of the residuals
    std::vector<double> residual_;

    /// Approximate eigenstates to project out
    std::vector<sparse_vec> project_out_;
};
} // namespace forte

#endif // _iterative_solvers_h_
