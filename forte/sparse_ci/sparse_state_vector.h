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

#ifndef _sparse_state_vector_h_
#define _sparse_state_vector_h_

#include <vector>
#include <unordered_map>

#include "sparse_ci/determinant.h"
#include "sparse_ci/sparse_operator.h"

namespace forte {

class ActiveSpaceIntegrals;

class StateVector {
  public:
    /// Constructor
    StateVector() = default;
    /// Constructor from a map/dictionary (python friendly)
    StateVector(const det_hash<double>& state_vec);

    /// @return the map that holds the determinants
    det_hash<double>& map() { return state_vec_; }
    /// @return true if the two states are identical
    bool operator==(const StateVector& lhs) const;

    /// @return a string representation of the object
    std::string str(int n = 0) const;
    /// @return the number of elements (determinants)
    auto size() const { return state_vec_.size(); }
    /// @brief reset this state
    void clear() { state_vec_.clear(); }
    /// @brief find and return the element corresponding to a determinant
    /// @param d the determinant to search for
    /// @return the element found
    auto find(const Determinant& d) const { return state_vec_.find(d); }

    /// @return the beginning of the map
    auto begin() { return state_vec_.begin(); }
    /// @return the beginning of the map (const)
    auto begin() const { return state_vec_.begin(); }
    /// @return the end of the map
    auto end() { return state_vec_.end(); }
    /// @return the end of the map (const)
    auto end() const { return state_vec_.end(); }
    /// @return the coefficient corresponding to a determinant
    /// @param d the determinant to search
    double& operator[](const Determinant& d) { return state_vec_[d]; }

  private:
    /// Holds an unordered map Determinant -> double
    det_hash<double> state_vec_;
};

// Functions to apply operators, gop |state>

/// apply the number projection operator P^alpha_na P^beta_nb |state>
StateVector apply_number_projector(int na, int nb, StateVector& state);

/// compute the projection  <state0 | op | ref>, for each operator op in gop
std::vector<double> get_projection(SparseOperator& sop, const StateVector& ref,
                                   const StateVector& state0);

/// compute the overlap value <left_state|right_state>
double overlap(StateVector& left_state, StateVector& right_state);

/// safe implementation of apply operator
StateVector apply_operator_safe(SparseOperator& sop, const StateVector& state);
/// fast implementation of apply operator based on sorting
StateVector apply_operator(SparseOperator& sop, const StateVector& state0,
                           double screen_thresh = 1.0e-12);

} // namespace forte

#endif // _sparse_state_vector_h_
