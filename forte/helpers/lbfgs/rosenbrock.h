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

#ifndef _rosenbrock_h_
#define _rosenbrock_h_

#include "psi4/libmints/vector.h"

namespace forte {

class ROSENBROCK {
  public:
    /**
     * @brief Constructor of the Rosenbrock function
     * @param n: the size of the problem (must be even)
     * @param do_h0: compute diagonal Hessian or not
     *
     * Implementation notes:
     *   See Wikipedia https://en.wikipedia.org/wiki/Rosenbrock_function
     */
    ROSENBROCK(int n);

    /// Compute the function value and gradients
    double evaluate(psi::SharedVector x, psi::SharedVector g, bool do_g = true);

    /// Compute the diagonal Hessian
    void hess_diag(psi::SharedVector x, psi::SharedVector h0);

  private:
    /// Size of the problem
    int n_;

    /// Error message printing
    void check_dim(psi::SharedVector x);
};
/// Test L-BFGS on Rosenbrock function
double test_lbfgs_rosenbrock(int n, int h0_freq = 0);
} // namespace forte
#endif // _rosenbrock_h_
