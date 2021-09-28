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

#include <stdexcept>
#include "helpers/lbfgs/lbfgs_param.h"

namespace forte {

LBFGS_PARAM::LBFGS_PARAM() {
    // set dedfault values
    print = 1;
    m = 6;
    epsilon = 1.0e-5;
    maxiter = 20;
    h0_freq = 0; // only compute the diagonal Hessian for once
    maxiter_linesearch = 5;
    max_dir = 1.0e15;

    // the following are for experts
    line_search_condition = LINE_SEARCH_CONDITION::STRONG_WOLFE;
    step_length_method = STEP_LENGTH_METHOD::LINE_BRACKETING_ZOOM;
    min_step = 1.0e-15;
    max_step = 1.0e15;
    c1 = 1.0e-4;
    c2 = 0.9;
}

void LBFGS_PARAM::check_param() {
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

    if (max_dir < 0)
        throw std::runtime_error("Max absolute value in direction vector (max_dir) must > 0");

    if (min_step < 0)
        throw std::runtime_error("Minimum step length (min_step) must > 0");

    if (max_step < min_step)
        throw std::runtime_error("Maximum step length (max_step) must > min_step");

    if (c1 <= 0 || c1 >= 0.5)
        throw std::runtime_error("Parameter c1 must lie in (0, 0.5)");

    if (c2 <= c1 || c2 >= 1)
        throw std::runtime_error("Parameter c2 must lie in (c1, 1.0)");
}
} // namespace forte
