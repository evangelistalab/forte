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

/**
 * @brief The SparseFactExp class
 * This class implements an algorithm to apply the exponential of an operator to a state
 *
 *    |state> -> exp(op) |state>
 *
 */
class SparseExp {
    enum class OperatorType { Excitation, Antihermitian };

  public:
    /// @brief Constructor
    /// @param maxk the maximum power of op used in the Taylor expansion of exp(op)
    /// @param screen_thresh a threshold to select which elements of the operator applied to the
    /// state. An operator in the form exp(t ...), where t is an amplitude, will be applied to a
    /// determinant Phi_I with coefficient C_I if the product |t * C_I| > screen_threshold
    SparseExp(int maxk, double screen_thresh);

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
    /// @param scaling_factor A scalar factor that multiplies the operator exponentiated. If set to
    /// -1.0 it allows to compute the inverse of the exponential exp(-op)
    SparseState apply_op(const SparseOperator& sop, const SparseState& state,
                         double scaling_factor = 1.0);

    SparseState apply_op(const SparseOperatorList& sop, const SparseState& state,
                         double scaling_factor = 1.0);

    /// @brief Compute the exponential of the antihermitian of an operator applied to a state via a
    /// Taylor expansion
    ///
    ///             exp(op - op^dagger) |state>
    ///
    /// This algorithms is useful when applying the exponential repeatedly
    /// to the same state or in an iterative procedure
    /// This function applies only those elements of the operator that satisfy the condition:
    ///     |t * C_I| > screen_threshold
    /// where C_I is the coefficient of a determinant
    ///
    /// @param sop the operator. Each term in this operator is applied in the order provided
    /// @param state the state to which the factorized exponential will be applied
    /// @param scaling_factor A scalar factor that multiplies the operator exponentiated. If set to
    /// -1.0 it allows to compute the inverse of the exponential exp(-op)
    SparseState apply_antiherm(const SparseOperator& sop, const SparseState& state,
                               double scaling_factor = 1.0);

    SparseState apply_antiherm(const SparseOperatorList& sop, const SparseState& state,
                               double scaling_factor = 1.0);

    void set_screen_thresh(double screen_thresh) { screen_thresh_ = screen_thresh; }

    void set_maxk(int maxk) { maxk_ = maxk; }

  private:
    int maxk_ = 19;
    double screen_thresh_ = 1e-12;

    SparseState apply_exp_operator(OperatorType op_type, const SparseOperator& sop,
                                   const SparseState& state, double scaling_factor);
};

} // namespace forte