/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

    /// Return the size of the collapse vectors
    size_t collapse_size() const;

    /// Add a guess basis vector
    void add_guess(psi::SharedVector vec);
    /// Get a basis vector
    void get_b(psi::SharedVector vec);
    /// Add a sigma vector
    bool add_sigma(psi::SharedVector vec);

    void set_project_out(std::vector<sparse_vec> project_out);

    /// Return the eigenvalues
    psi::SharedVector eigenvalues() const;
    /// Return the eigenvectors
    psi::SharedMatrix eigenvectors() const;
    /// Return the n-th eigenvector
    psi::SharedVector eigenvector(size_t n) const;

    /// Initialize the object
    void startup(psi::SharedVector diagonal);

    /// Perform an update step
    SolverStatus update();

    /// Produce the final eigenvectors and eigenvalues
    void get_results();

    /// A vector with the 2-norm of the residual for each root
    std::vector<double> residuals() const;

  private:
    // ==> Class Private Functions <==

    /// Check that the eigenvectors are orthogonal
    bool check_orthogonality();
    /// Check if the the iterative procedure has converged
    /// @return a pair of boolean (is_energy_converged,is_residual_converged)
    std::pair<bool, bool> check_convergence();
    /// Build the correction vectors
    void form_correction_vectors();
    /// Compute the 2-norm of the residual
    void compute_residual_norm();
    /// Project out undesired roots
    void project_out_roots(psi::SharedMatrix v);
    /// Normalize the correction vectors and return the norm of the vectors before they were
    /// normalized
    std::vector<double> normalize_vectors(psi::SharedMatrix v, size_t n);
    /// Perform subspace collapse
    bool subspace_collapse();
    /// Collapse the vectors
    void collapse_vectors();

    // ==> Class Private Data <==

    /// The print level
    size_t print_level_ = 1;
    /// Eigenvalue convergence threshold
    double e_convergence_ = 1.0e-12;
    /// Residual convergence threshold
    double r_convergence_ = 1.0e-6;
    /// The threshold used to discard correction vectors
    double schmidt_threshold_ = 1.0e-8;
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

    int iter_ = 0;
    size_t basis_size_;
    /// The size
    size_t sigma_size_;
    size_t converged_ = 0;
    double timing_ = 0.0;
    bool last_update_collapsed_ = false;

    /// Current set of basis vectors stored by row
    psi::SharedMatrix b_;
    /// Guess vectors formed from old vectors, stored by row
    psi::SharedMatrix bnew;
    /// Residual eigenvectors, stored by row
    psi::SharedMatrix f;
    /// Sigma vectors, stored by column
    psi::SharedMatrix sigma_;
    /// Davidson-Liu mini-Hamitonian
    psi::SharedMatrix G;
    /// Davidson-Liu mini-metric
    psi::SharedMatrix S;
    /// Eigenvectors of the Davidson mini-Hamitonian
    psi::SharedMatrix alpha;

    /// Eigenvalues of the Davidson mini-Hamitonian
    psi::SharedVector lambda;
    /// Old eigenvalues of the Davidson mini-Hamitonian
    psi::SharedVector lambda_old;
    /// Diagonal elements of the Hamiltonian
    psi::SharedVector h_diag;
    /// 2-Norm of the residuals
    std::vector<double> residual_;

    /// Approximate eigenstates to project out
    std::vector<sparse_vec> project_out_;
};
} // namespace forte

#endif // _iterative_solvers_h_
