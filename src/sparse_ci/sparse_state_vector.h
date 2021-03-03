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

#ifndef _sparse_state_vector_h_
#define _sparse_state_vector_h_

#include <vector>
#include <unordered_map>

#include "sparse_ci/determinant.h"
#include "sparse_ci/general_operator.h"
#include "sparse_ci/sparse_operator.h"

namespace forte {

class ActiveSpaceIntegrals;

class StateVector {
  public:
    StateVector();
    StateVector(const det_hash<double>& state_vec);
    det_hash<double>& map() { return state_vec_; }

    std::string str(int n = 0) const;

    auto size() const { return state_vec_.size(); }
    void clear() { state_vec_.clear(); }
    auto find(const Determinant& d) const { return state_vec_.find(d); }

    auto begin() { return state_vec_.begin(); }
    auto end() { return state_vec_.end(); }

    auto begin() const { return state_vec_.begin(); }
    auto end() const { return state_vec_.end(); }

    double& operator[](const Determinant& d) { return state_vec_[d]; }

  private:
    det_hash<double> state_vec_;
};

// Functions to apply operators, gop |state>

/// apply the Hamiltonian operator H|state>
StateVector apply_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                              const StateVector& state0, double screen_thresh = 1.0e-12);

/// apply the number projection operator P^alpha_na P^beta_nb |state>
StateVector apply_number_projector(int na, int nb, StateVector& state);

/// compute the projection  <state0 | op | ref>, for each operator op in gop
std::vector<double> get_projection(SparseOperator& sop, const StateVector& ref,
                                   const StateVector& state0);

/// compute the expectation value <left_state|H|right_state>
double hamiltonian_matrix_element(StateVector& left_state, StateVector& right_state,
                                  std::shared_ptr<ActiveSpaceIntegrals> as_ints);

/// compute the overlap value <left_state|right_state>
double overlap(StateVector& left_state, StateVector& right_state);

/// safe implementation of apply operator
StateVector apply_operator_safe(SparseOperator& sop, const StateVector& state);
/// fast implementation of apply operator based on sorting
StateVector apply_operator(SparseOperator& sop, const StateVector& state0,
                           double screen_thresh = 1.0e-12);
/// fast implementation of apply operator
StateVector apply_operator_2(SparseOperator& sop, const StateVector& state0,
                             double screen_thresh = 1.0e-12);

// Functions to apply the exponential of an operator, exp(gop) |state0>
/// fast implementation of exp operator based on sorting
StateVector apply_exp_operator(SparseOperator& sop, const StateVector& state0,
                               double scaling_factor = 1.0, int maxk = 20,
                               double screen_thresh = 1.0e-12);

StateVector apply_exp_operator_2(SparseOperator& sop, const StateVector& state0,
                                 double scaling_factor = 1.0, int maxk = 20,
                                 double screen_thresh = 1.0e-12);

// Functions to apply the product of exponentials of anti-hermitian operators
// ... exp(gop_3) exp(gop_2) exp(gop_1) |state0>
/// safe implementation
StateVector apply_exp_ah_factorized_safe(SparseOperator& sop, const StateVector& state);
/// fast implementation of apply operator based on exact exponentiation
StateVector apply_exp_ah_factorized(SparseOperator& sop, const StateVector& state0,
                                    bool inverse = false);

/// safe implementation of apply operator
StateVector apply_operator_safe(GeneralOperator& gop, const StateVector& state);
/// fast implementation of apply operator based on sorting
StateVector apply_operator(GeneralOperator& gop, const StateVector& state0,
                           double screen_thresh = 1.0e-12);
/// fast implementation of apply operator
StateVector apply_operator_2(GeneralOperator& gop, const StateVector& state0,
                             double screen_thresh = 1.0e-12);

// Functions to apply the exponential of an operator, exp(gop) |state0>
/// fast implementation of exp operator based on sorting
StateVector apply_exp_operator(GeneralOperator& gop, const StateVector& state0,
                               double scaling_factor = 1.0, int maxk = 20,
                               double screen_thresh = 1.0e-12);

StateVector apply_exp_operator_2(GeneralOperator& gop, const StateVector& state0,
                                 double scaling_factor = 1.0, int maxk = 20,
                                 double screen_thresh = 1.0e-12);

// Functions to apply the product of exponentials of anti-hermitian operators
// ... exp(gop_3) exp(gop_2) exp(gop_1) |state0>
/// safe implementation
StateVector apply_exp_ah_factorized_safe(GeneralOperator& gop, const StateVector& state);
/// fast implementation of apply operator based on exact exponentiation
StateVector apply_exp_ah_factorized(GeneralOperator& gop, const StateVector& state0,
                                    bool inverse = false);

} // namespace forte

#endif // _sparse_state_vector_h_
