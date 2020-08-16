/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _lbfgs_h_
#define _lbfgs_h_

#include <functional>
#include <vector>

#include "psi4/libmints/vector.h"

#include "helpers/lbfgs/lbfgs_param.h"

namespace forte {

class LBFGS {
  public:
    /**
     * @brief Constructor of the Limited-BFGS class
     * @param dim: The dimension of the problem
     * @param m: The max number of vectors kept for L-BFGS
     * @param descent: Minimize the function if true, otherwise maximize
     *
     * Implementation notes:
     *   See Wikipedia https://en.wikipedia.org/wiki/Limited-memory_BFGS
     */
    LBFGS(int dim, int m = 10, bool descent = true);

    /**
     * @brief Constructor of the Limited-BFGS class
     * @param dim: The dimension of the problem
     * @param param: The LBFGS_PARAM object for L-BFGS parameters
     *
     * Implementation notes:
     *   See Wikipedia https://en.wikipedia.org/wiki/Limited-memory_BFGS
     *   and <Numerical Optimization> 2nd Ed. by Jorge Nocedal and Stephen J. Wright
     */
    LBFGS(const LBFGS_PARAM& param);

    /**
     * @brief The minimization for the target function
     * @param foo: Target class that should have the following methods:
     *             fx = foo.evaluate(x, g) where gradient g is modified by the function,
     *             and fx is the function return value.
     *             If diagonal Hessian is specified, foo.hess_diag(x, h0) should be available.
     * @param x: The initial value of x as input, the final value of x as output.
     *
     * @return the function value of at optimized x
     */
    template <class Foo> double minimize(Foo& foo, psi::SharedVector x);

    /// Generate correction vector
    psi::SharedVector compute_correction(psi::SharedVector x, psi::SharedVector g);

    /// Set the max size of vectors
    void set_size(int m);

    /// Set the diagonal Hessian
    void set_hess_diag(psi::SharedVector hess_diag);

    /// Reset the L-BFGS space
    void reset();

    /// Return the final gradient norm
    double g_norm() { return g_->norm(); }

    /// Return the final number of iterations
    int iter() { return iter_; }

  private:
    /// The psi4 Dimension object
    psi::Dimension dimpi_;

    /// The number of irreps
    int nirrep_;

    /// The current iteration number
    int iter_;

    /// The dimension of the problem
    int dim_;

    /// Parameters of L-BFGS
    LBFGS_PARAM param_;

    /// Max size of the vectors stored
    int m_;

    /// Diagonal elements of Hessian
    psi::SharedVector h0_;

    /// Gradient difference vectors
    std::vector<psi::SharedVector> y_;

    /// Variable difference vectors
    std::vector<psi::SharedVector> s_;

    /// The rho vectors
    std::vector<double> rho_;

    /// The alpha vector
    std::vector<double> alpha_;

    /// Moving direction vector
    psi::SharedVector p_;

    /// The current gradient vector
    psi::SharedVector g_;

    /// The last gradient vector
    psi::SharedVector g_last_;

    /// The last solution vector
    psi::SharedVector x_last_;

    /// Guess the inverse of diagonal Hessian
    void guess_h0();
    /// Compute gamma that can be used as inverse of diagonal Hessian
    double gamma();
    /// Apply h0_ to some vector
    void apply_h0(psi::SharedVector q);

    void guess_h0_new();
    double gamma_new();
    void apply_h0_new(psi::SharedVector q);

    void update();

    template <class Foo> void line_search(Foo& foo, psi::SharedVector x, double& fx, double& step);
    template <class Foo> void line_search_backtracking(Foo& foo, psi::SharedVector x, double& fx, double& step);
    template <class Foo> void line_search_bracketing_zoom(Foo& foo, psi::SharedVector x, double& fx, double& step);

    /// Update direction
    bool descent_;

    /// Counter
    int counter_;

    /// Check dimension matches or not
    void check_dim(psi::SharedVector a, psi::SharedVector b, const std::string error_msg);

    /// Resize all vectors uisng m_
    void resize(int m);
};
} // namespace forte
#endif // LBFGS_H
