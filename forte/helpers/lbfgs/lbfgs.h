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
     * @param param: The LBFGS_PARAM object for L-BFGS parameters
     *
     * Implementation notes:
     *   See Wikipedia https://en.wikipedia.org/wiki/Limited-memory_BFGS
     *   and <Numerical Optimization> 2nd Ed. by Jorge Nocedal and Stephen J. Wright
     */
    LBFGS(std::shared_ptr<LBFGS_PARAM> param);

    /**
     * @brief The minimization for the target function
     * @param foo: Target class that should have the following methods:
     *             fx = foo.evaluate(x, g, do_g=true) where gradient g is modified by the function,
     *             fx is the function return value, and g is computed when do_g is true.
     *             If diagonal Hessian is specified, foo.hess_diag(x, h0) should be available.
     * @param x: The initial value of x as input, the final value of x as output.
     *
     * @return the function value of at optimized x
     */
    template <class Foo> double minimize(Foo& foo, psi::SharedVector x);

    /// Reset the L-BFGS space
    void reset();

    /// Return the current / final gradient vector
    psi::SharedVector g() { return g_; }

    /// Return the final number of iterations
    int iter() { return iter_; }

    /// Return true if minimization converged
    bool converged() { return converged_; }

  private:
    /// The dimension of x
    psi::Dimension dimpi_;

    /// The number of irreps of x
    int nirrep_;

    /// The current iteration number
    int iter_;

    /// Parameters of L-BFGS
    std::shared_ptr<LBFGS_PARAM> param_;

    /// Minimization procedure converged or not
    bool converged_;

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

    /// The correction (moving direction) vector
    psi::SharedVector p_;

    /// The current gradient vector
    psi::SharedVector g_;

    /// The last gradient vector
    psi::SharedVector g_last_;

    /// The last solution vector
    psi::SharedVector x_last_;

    /// Compute gamma that can be used as inverse of diagonal Hessian
    double compute_gamma();

    /// Apply h0_ to some vector
    void apply_h0(psi::SharedVector q);

    /// Generate correction (direction) vector
    void update();

    /// Determine step length
    template <class Foo> void next_step(Foo& foo, psi::SharedVector x, double& fx, double& step);

    /// Determine step length using max value of direction vector
    template <class Foo>
    void scale_direction_vector(Foo& foo, psi::SharedVector x, double& fx, double& step);

    /// Line search using backtracking to determine step length
    template <class Foo>
    void line_search_backtracking(Foo& foo, psi::SharedVector x, double& fx, double& step);

    /// Line search using bracketing and zoom  to determine step length
    /// See (Algorithm 3.5) of <Numerical Optimization> 2nd Ed. by Nocedal and Wright
    template <class Foo>
    void line_search_bracketing_zoom(Foo& foo, psi::SharedVector x, double& fx, double& step);

    /// Resize all vectors uisng m_
    void resize(int m);
};
} // namespace forte
#endif // LBFGS_H
