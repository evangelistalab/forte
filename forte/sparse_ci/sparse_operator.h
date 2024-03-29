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
#include <helpers/math_structures.h>

#include "sparse_ci/sq_operator_string.h"

namespace forte {

class ActiveSpaceIntegrals;

/**
 * @brief The SparseOperator class
 * Base class for second quantized operators.
 *
 * An operator is a linear combination of terms, where each term is a numerical factor
 * times a product of second quantized operators (a SQOperator object)
 *
 * For example:
 *   0.1 * [2a+ 0a-] - 0.5 * [2a+ 0a-] + ...
 *       Term 0            Term 1
 *
 *        This class stores operators in each term in the following canonical form
 *            a+_p1 a+_p2 ...  a+_P1 a+_P2 ...   ... a-_Q2 a-_Q1   ... a-_q2 a-_q1
 *            alpha creation   beta creation    beta annihilation  alpha annihilation
 *
 *        with indices sorted as
 *
 *            (p1 < p2 < ...) (P1 < P2 < ...)  (... > Q2 > Q1) (... > q2 > q1)
 */
class SparseOperator
    : public VectorSpace<SparseOperator, SQOperatorString, double, SQOperatorString::Hash> {
  public:
    SparseOperator() = default;

    /// @brief add a term to this operator
    /// @param str a string that defines the product of operators in the format [... q_2 q_1 q_0]
    /// @param coefficient a coefficient that multiplies the product of second quantized operators
    /// @param allow_reordering if true, the operator will be reordered to canonical form
    /// @details The operator is stored in canonical form
    ///
    ///     coefficient * [... q_2 q_1 q_0]
    ///
    /// where q_0, q_1, ... are second quantized operators. These operators are
    /// passed as string
    ///
    ///     '[... q_2 q_1 q_0]'
    ///
    /// where q_i = <orbital_i><spin_i><type_i> and the quantities in <> are
    ///
    ///     orbital_i: int
    ///     spin_i: 'a' (alpha) or 'b' (beta)
    ///     type_i: '+' (creation) or '-' (annihilation)
    ///
    /// For example, '[0a+ 1b+ 12b- 0a-]'
    ///
    void add_term_from_str(std::string str, double coefficient, bool allow_reordering = false);

    /// @return a string representation of this operator
    std::vector<std::string> str() const;

    /// @return a latex representation of this operator
    std::string latex() const;
};

class SparseOperatorList : public VectorSpaceList<SQOperatorString, double> {
  public:
    SparseOperatorList() = default;

    void add(const SQOperatorString& op, double coefficient);

    double operator[](size_t i) const { return elements_[i].second; }
    double& operator[](size_t i) { return elements_[i].second; }

  private:
    std::vector<std::pair<SQOperatorString, double>> elements_;
};

// double norm(const SparseOperator& op);

// /// @return The product of two second quantized operators
// SparseOperator operator+(SparseOperator lhs, const SparseOperator& rhs);

// /// @return The product of two second quantized operators
// SparseOperator operator-(SparseOperator lhs, const SparseOperator& rhs);

/// @return The product of two second quantized operators
SparseOperator operator*(const SparseOperator& lhs, const SparseOperator& rhs);

// /// @return The product of a second quantized operator and a numerical factor
// SparseOperator operator*(double scalar, const SparseOperator& op);

// /// @return A second quantized operator divided by a numerical factor
// SparseOperator operator/(double scalar, const SparseOperator& op);

/// @return The commutator of two second quantized operators
SparseOperator commutator(const SparseOperator& lhs, const SparseOperator& rhs);

// void similarity_transform_test(SparseOperator& op, const SQOperatorString& A, double theta);

void sim_trans_fact_op(SparseOperator& O, const SparseOperatorList& T, bool reverse = false,
                       double screen_threshold = 1e-12);

void sim_trans_fact_antiherm(SparseOperator& O, const SparseOperatorList& T, bool reverse = false,
                             double screen_threshold = 1e-12);

// class NormalOrderedOperator {
//   public:
//     NormalOrderedOperator() = default;
//     NormalOrderedOperator(const SparseOperator& op, const MOSpaceInfo& mo_space_info,
//                           const RDMs& rdm);

//   private:
// };

} // namespace forte
