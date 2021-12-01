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

#ifndef _sparse_exp_h_
#define _sparse_exp_h_

#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"

#include "sparse_ci/sparse_state_vector.h"
#include "sparse_ci/sparse_operator.h"

#include "sparse_ci/determinant_hashvector.h"

namespace forte {

/**
 * @brief The SparseFactExp class
 * This class implements an algorithm to apply the exponential of an operator to a state
 * 
 *    |state> -> exp(op) |state>
 * 
 */
class SparseExp {
    enum class Algorithm { Cached, OnTheFlySorted, OnTheFlyStd };

  public:

    /// @brief Compute the exponential applied to a state via a Taylor expansion
    ///
    ///             exp(op) |state>
    ///
    /// This algorithms is useful when applying the exponential repeatedly
    /// to the same state or in an iterative procedure
    /// This function applies only those elements of the operator that satisfy the condition:
    ///     |t * C_I| > screen_threshold
    /// where C_I is the coefficient of a determinant
    ///
    /// @param sop the operator. Each term in this operator is applied in the order provided
    /// @param state the state to which the factorized exponential will be applied
    /// @param algorithm the algorithm used to compute the exponential. If algorithm = "onthefly"
    /// this function will compute the factorized exponential using an on-the-fly implementation (slow).
    /// If algorithm = "cached", a cached approach is used.
    /// @param scaling_factor A scalar factor that multiplies the operator exponentiated. If set to -1.0
    /// it allows to compute the inverse of the exponential exp(-op)
    /// @param maxk the maximum power of op used in the Taylor expansion of exp(op)
    /// @param screen_thresh a threshold to select which elements of the operator applied to the state.
    /// An operator in the form exp(t ...), where t is an amplitude, will be applied to a determinant
    /// Phi_I with coefficient C_I if the product |t * C_I| > screen_threshold
    StateVector compute(const SparseOperator& sop, const StateVector& state,
                        const std::string& algorithm = "cached", double scaling_factor = 1.0,
                        int maxk = 19, double screen_thresh = 1.0e-12);
    /// @return timings for this class
    std::map<std::string, double> timings() const;

  private:
    StateVector apply_exp_operator(const SparseOperator& sop, const StateVector& state0,
                                   double scaling_factor, int maxk, double screen_thresh,
                                   Algorithm alg);
    StateVector apply_operator_cached(const SparseOperator& sop, const StateVector& state0,
                                      double screen_thresh);
    StateVector apply_operator_sorted(const SparseOperator& sop, const StateVector& state0,
                                      double screen_thresh);
    StateVector apply_operator_std(const SparseOperator& sop, const StateVector& state0,
                                   double screen_thresh);


    std::map<std::string, double> timings_;
    DeterminantHashVec exp_hash_;
    // map Determinant -> [(operator, new determinant, factor),...]
    det_hash<std::vector<std::tuple<size_t, Determinant, double>>> couplings_;
    det_hash<std::vector<std::tuple<size_t, Determinant, double>>> couplings_dexc_;
};

} // namespace forte

#endif // _sparse_exp_h_
