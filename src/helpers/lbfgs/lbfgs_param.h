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
    /// Default constructor of the Limited-BFGS parameter class set defaut
    LBFGS_PARAM();

    /// Check if the parameters make sense
    void check_param();

    // => L-BFGS parameters <=

    /// The number of vectors kept
    int m;

    /// Convergence threshold to terminate the minimization: |g| < ε * max(1, |x|)
    double epsilon;

    /// Max number of iterations
    int maxiter;

    /// Max number of trials for line search of optimal step length
    int maxiter_linesearch;

    /// Minimal step length allowed
    double min_step;

    /// Maximal step length allowed
    double max_step;

    /// Parameter for Armijo condition
    double c1;

    /// Parameter for Wolfe curvature condition
    double c2;

    /// Automatically guess diagonal Hessian (true) or user provided (false)
    bool auto_hess;

    /// Printing level
    int print;
};
} // namespace forte

#endif // lbfgs_param_h
