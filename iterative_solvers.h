/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _iterative_solvers_h_
#define _iterative_solvers_h_

#include <libqt/qt.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>

namespace psi{ namespace libadaptive{

/**
 * @brief The DavidsonLiuSolver class
 * This class diagonalizes the Hamiltonian in a basis
 * of determinants.
 */
class DavidsonLiuSolver
{
public:

    // ==> Class Constructor and Destructor <==

    /// Constructor
    DavidsonLiuSolver(size_t size,size_t nroot);

    /// Destructor
    ~DavidsonLiuSolver();

    // ==> Class Interface <==

    void set_print_level(size_t n) {print_level_ = n;}

    /// Set the energy convergence
    void set_e_convergence(double value) {e_convergence_ = value;}
    /// Set the residual convergence
    void set_r_convergence(double value) {r_convergence_ = value;}

    size_t collapse_size() const {return collapse_size_;}

    void add_b(SharedVector vec);
    void get_b(SharedVector vec);
    bool add_sigma(SharedVector vec);

    SharedVector eigenvalues() const {return lambda;}
    SharedMatrix eigenvectors() const {return b_;}
    SharedVector eigenvector(size_t n);

    /// Initialize the object
    void startup(SharedVector diagonal);

    /// Add a new vector
    bool update();

    ///
    void get_results();

private:

    // ==> Class Private Functions <==

    /// Check that the eigenvectors are orthogonal
    bool check_orthogonality();
    /// Check if the the iterative procedure has converged
    bool check_convergence();

    // ==> Class Private Data <==

    /// The maximum number of iterations
    int maxiter_ = 35;
    /// The print level
    size_t print_level_ = 1;
    /// Eigenvalue convergence threshold
    double e_convergence_ = 1.0e-15;
    /// Residual convergence threshold
    double r_convergence_ = 1.0e-6;
    /// The dimension of the vectors
    size_t size_;
    /// The number of roots requested
    size_t nroot_;
    /// The number of vectors to retain for each root
    size_t collapse_per_root_ = 2;
    /// The maximum subspace size for each root
    size_t subspace_per_root_ = 4;
    /// The number of vectors to retain after collapse
    size_t collapse_size_;
    /// The maximum subspace size
    size_t subspace_size_;

    size_t iter_ = 0;
    size_t basis_size_;
    /// The size
    size_t sigma_size_;
    size_t converged_ = 0;
    double timing_ = 0.0;

    /// Current set of guess vectors stored by row
    SharedMatrix b_;
    /// Guess vectors formed from old vectors, stored by row
    SharedMatrix bnew;
    /// Residual eigenvectors, stored by row
    SharedMatrix f;
    /// Sigma vectors, stored by column
    SharedMatrix sigma_;
    /// Davidson-Liu mini-Hamitonian
    SharedMatrix G;
    /// Davidson-Liu mini-metric
    SharedMatrix S;
    /// Eigenvectors of the Davidson mini-Hamitonian
    SharedMatrix alpha;

    /// Eigenvalues of the Davidson mini-Hamitonian
    SharedVector lambda;
    /// Old eigenvalues of the Davidson mini-Hamitonian
    SharedVector lambda_old;
    /// Diagonal elements of the Hamiltonian
    SharedVector h_diag;
};

}}

#endif // _iterative_solvers_h_
