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

#ifndef _sparse_exp_h_
#define _sparse_exp_h_

#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"

#include "sparse_ci/sparse_state_vector.h"
#include "sparse_ci/sparse_operator.h"

#include "sparse_ci/determinant_hashvector.h"

namespace forte {

class SparseExp {
  public:
    SparseExp();
    StateVector compute(const SparseOperator& sop, const StateVector& state,
                        double scaling_factor = 1.0, int maxk = 19, double screen_thresh = 1.0e-12);
    StateVector compute_on_the_fly(const SparseOperator& sop, const StateVector& state,
                                   double scaling_factor = 1.0, int maxk = 19,
                                   double screen_thresh = 1.0e-12);
    std::map<std::string, double> time() const;

  private:
    StateVector apply_exp_operator(const SparseOperator& sop, const StateVector& state0,
                                   double scaling_factor, int maxk, double screen_thresh);
    StateVector apply_exp_operator2(const SparseOperator& sop, const StateVector& state0,
                                    double scaling_factor, int maxk, double screen_thresh);
    StateVector apply_operator(const SparseOperator& sop, const StateVector& state0,
                               double screen_thresh);
    StateVector apply_operator2(const SparseOperator& sop, const StateVector& state0,
                                double screen_thresh);
    StateVector apply_operator_cached(const SparseOperator& sop, const StateVector& state0,
                                      double screen_thresh);

    double time_ = 0.0;
    double couplings_time_ = 0.0;
    double exp_time_ = 0.0;
    double on_the_fly_time_ = 0.0;
    DeterminantHashVec exp_hash_;

    // map Determinant -> [(operator, new determinant, factor),...]
    det_hash<std::vector<std::tuple<size_t, Determinant, double>>> couplings_;
};

} // namespace forte

#endif // _sparse_exp_h_
