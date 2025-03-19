/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#pragma once

#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"

#include "sparse_ci/sparse_state.h"
#include "sparse_ci/sparse_operator.h"

#include "sparse_ci/determinant_hashvector.h"

namespace forte {

/// @brief This class implements an algorithm to apply a factorized exponential operator to a state

class SparseFactExp {
  public:
    /// @brief Constructor
    /// @param screen_thresh a threshold to select which elements of the operator applied to the
    /// state. An operator in the form exp(t ...), where t is an amplitude, will be applied to a
    /// determinant Phi_I with coefficient C_I if the product |t * C_I| > screen_threshold
    SparseFactExp(double screen_thresh = 1.0e-12);

    /// @brief Compute the factorized exponential applied to a state using an exact algorithm
    ///
    ///             ... exp(op2) exp(op1) |state>
    ///
    /// This algorithm is useful when applying the factorized exponential repeatedly
    /// to the same state or in an iterative procedure
    /// This function applies only those elements of the operator that satisfy the condition:
    ///     |t * C_I| > screen_threshold
    /// where C_I is the coefficient of a determinant
    ///
    /// @param sop the operator. Each term in this operator is applied in the order provided
    /// @param state the state to which the factorized exponential will be applied
    /// @param inverse If true, compute the inverse of the factorized exponential:
    ///
    ///             exp(-op1) exp(-op2) ... |state>
    ///
    /// @param reverse If true, apply the operator in reverse order:
    ///             ... exp(opN-1) exp(opN) |state>
    SparseState apply_op(const SparseOperatorList& sop, const SparseState& state, bool inverse,
                         bool reverse = false);

    /// @brief Compute the factorized exponential applied to a state using an exact algorithm
    ///
    ///             ... exp(op2 - op2^dagger) exp(op1 - op1^dagger) |state>
    ///
    /// This algorithm is useful when applying the factorized exponential repeatedly
    /// to the same state or in an iterative procedure
    /// This function applies only those elements of the operator that satisfy the condition:
    ///     |t * C_I| > screen_threshold
    /// where C_I is the coefficient of a determinant
    ///
    /// @param sop the operator. Each term in this operator is applied in the order provided
    /// @param state the state to which the factorized exponential will be applied
    /// @param inverse If true, compute the inverse of the factorized exponential
    ///
    ///             exp(-op1 + op1^dagger) exp(-op2 + op2^dagger) ... |state>
    ///
    /// @param reverse If true, apply the operator in reverse order:
    ///             ... exp(opN-1) exp(opN) |state>
    SparseState apply_antiherm(const SparseOperatorList& sop, const SparseState& state,
                               bool inverse, bool reverse = false);

    std::pair<SparseState, SparseState>
    antiherm_deriv(const SQOperatorString& sqop, const sparse_scalar_t t, const SparseState& state);

    void set_screen_thresh(double screen_thresh) { screen_thresh_ = screen_thresh; }

    double sinc_taylor(double x) const {
        int taylor_order = 5;
        double taylor_thresh = 1.0e-3;
        if (std::abs(x) < taylor_thresh) {
            double result = 1.0;
            double x_squared = x * x;
            double term = 1.0;
            for (int i = 1; i < taylor_order; i++) {
                term *= -x_squared / (2 * i * (2 * i + 1));
                result += term;
            }
            return result;
        }
        return std::sin(x) / x;
    }

  private:
    double screen_thresh_;
};

} // namespace forte