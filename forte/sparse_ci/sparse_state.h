/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <vector>
#include <unordered_map>

#include "sparse_ci/sparse.h"
#include "sparse_ci/determinant.h"
#include "sparse_ci/sparse_operator.h"

namespace forte {

/// @brief A class to represent general Fock space states
class SparseState
    : public VectorSpace<SparseState, Determinant, sparse_scalar_t, Determinant::Hash> {
  public:
    /// @return a string representation of the object
    /// @param n the number of spatial orbitals to print
    std::string str(int n = 0) const;
};

// Functions to apply operators to a state
/// @brief Apply an operator to a state
/// @param op the operator to apply
/// @param state the state to apply the operator to
/// @param screen_thresh the threshold to screen the operator
/// @return the new state
SparseState apply_operator_lin(const SparseOperator& op, const SparseState& state,
                               double screen_thresh = 1.0e-12);

/// @brief Apply the antihermitian combination of an operator to a state
/// @param op the operator to apply
/// @param state the state to apply the operator to
/// @param screen_thresh the threshold to screen the operator
/// @return the new state
SparseState apply_operator_antiherm(const SparseOperator& op, const SparseState& state,
                                    double screen_thresh = 1.0e-12);

/// compute the projection  <state0 | op | ref>, for each operator op in gop
std::vector<sparse_scalar_t> get_projection(const SparseOperatorList& sop, const SparseState& ref,
                                            const SparseState& state0);

/// apply the number projection operator P^alpha_na P^beta_nb |state>
SparseState apply_number_projector(int na, int nb, const SparseState& state);

/// compute the overlap value <left_state|right_state>
sparse_scalar_t overlap(const SparseState& left_state, const SparseState& right_state);

/// compute the S^2 expectation value
sparse_scalar_t spin2(const SparseState& left_state, const SparseState& right_state);

/// Return the normalized state
SparseState normalize(const SparseState& state);

} // namespace forte
