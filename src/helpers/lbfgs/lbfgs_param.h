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

#ifndef _lbfgs_param_h_
#define _lbfgs_param_h_

#include <stdexcept>

namespace forte {
class LBFGS_PARAM {
  public:
    /// Default constructor of the Limited-BFGS parameter class
    LBFGS_PARAM();

    /// Check if the parameters make sense
    void check_param() {
        if (m <= 0)
            throw std::runtime_error("Size of L-BFGS history (m) must > 0");

        if (epsilon <= 0)
            throw std::runtime_error("Convergence threshold (epsilon) must > 0");

        if (maxiter <= 0)
            throw std::runtime_error("Max number of iterations (maxiter) must > 0");

        if (maxiter_linesearch <= 0) {
            auto msg = "Max iterations for line search (maxiter_linesearch) must > 0";
            throw std::runtime_error(msg);
        }

        if (min_step < 0)
            throw std::runtime_error("Minimum step length (min_step) must > 0");

        if (max_step < min_step)
            throw std::runtime_error("Maximum step length (max_step) must > min_step");

        if (c1 <= 0 || c1 >= 0.5)
            throw std::runtime_error("Parameter c1 must lie in (0, 0.5)");

        if (c2 <= c1 || c2 >= 1)
            throw std::runtime_error("Parameter c2 must lie in (c1, 1.0)");
    }

    // => L-BFGS parameters <=

    /// The number of vectors kept
    int m = 8;

    /// Convergence threshold to terminate the minimization: |g| < ε * max(1, |x|)
    double epsilon = 1.0e-4;

    /// Max number of iterations
    int maxiter = 20;

    /// Max number of trials for line search of optimal step length
    int maxiter_linesearch = 10;

    /// Minimal step length allowed
    double min_step = 1.0e-15;

    /// Maximal step length allowed
    double max_step = 1.0e15;

    /// Parameter for Armijo condition
    double c1 = 1.0e-4;

    /// Parameter for Wolfe curvature condition
    double c2 = 0.9;

    /// Automatically guess diagonal Hessian (true) or user provided (false)
    bool auto_hess = false;

    /// Printing level
    int print = 1;
};
} // namespace forte

#endif // lbfgs_param_h
