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

#include "sparse_ci/determinant.h"

namespace forte {

/**
 * A data structure used to represent a second quantized operator string like
 *
 * ... op(2) op(1) op(0), where op(i) = a^\dagger_(orb_i, spin_i) or a_(orb_i, spin_i)
 *
 * like, for example
 *
 *  a^\dagger_{2,\alpha} a^\dagger_{3,\beta} a_{1,\beta} a_{0,\alpha}
 *
 * This operator is stored as
 *
 *  [(false,true,0),(false,false,1),(true,false,3),(true,true,2)]
 *
 * The data format is
 *
 *  [(creation_0, spin_0, orb_0), (creation_1, spin_1, orb_1), ...]
 *
 * where the operators are arranged as
 *
 * where
 *  creation_i  : bool (true = creation, false = annihilation)
 *  spin_i      : bool (true = alpha, false = beta)
 *  orb_i       : int  (the index of the mo)
 *
 */
using op_tuple_t = std::vector<std::tuple<bool, bool, int>>;

/**
 * @brief A class to represent a second quantized operator.
 *
 * This class stores operators in the following canonical form
 *     a+_p1 a+_p2 ...  a+_P1 a+_P2 ...   ... a-_Q2 a-_Q1   ... a-_q2 a-_q1
 *     alpha creation   beta creation    beta annihilation  alpha annihilation
 *
 * with indices sorted as
 *
 *     (p1 < p2 < ...) (P1 < P2 < ...)  (... > Q2 > Q1) (... > q2 > q1)
 *
 * The creation and annihilation operators are stored separately as bit arrays
 * using the Determinant class
 */
class SQOperatorString {
  public:
    /// default constructor
    SQOperatorString();
    /// constructor from a pair of Determinant objects
    SQOperatorString(const Determinant& cre, const Determinant& ann);
    /// @return a Determinant object that represents the creation operators
    const Determinant& cre() const;
    /// @return a Determinant object that represents the annihilation operators
    const Determinant& ann() const;
    /// @return true if this operator is a number operator (i.e. it contains no creation or
    /// annihilation  operators)
    bool is_number() const;
    /// @return the number of creation + annihilation operators in this operator
    int count() const;
    /// @return compare this operator with another operator
    bool operator==(const SQOperatorString& other) const;
    /// @return compare this operator with another operator
    bool operator<(const SQOperatorString& other) const;
    /// @return a string representation of this operator
    std::string str() const;
    /// @return a latex representation of this operator
    std::string latex() const;
    /// @return a sq_operator that is the adjoint of this operator
    SQOperatorString adjoint() const;

    struct Hash {
        std::size_t operator()(const SQOperatorString& sqop_str) const {
            std::uint64_t seed = Determinant::Hash()(sqop_str.cre());
            std::uint64_t w = Determinant::Hash()(sqop_str.ann());
            hash_combine_uint64(seed, w);
            return seed;
        }
    };

    /// a Determinant that represents the creation operators
    Determinant cre_;
    /// a Determinant that represents the annihilation operators
    Determinant ann_;
};

std::vector<std::pair<SQOperatorString, double>> operator*(const SQOperatorString& lhs,
                                                           const SQOperatorString& rhs);

std::vector<std::pair<SQOperatorString, double>> commutator(const SQOperatorString& lhs,
                                                            const SQOperatorString& rhs);

/// @return a SQOperatorString from a string and the corresponding phase (+1 or -1) due to
/// reordering, if reordering is allowed
std::pair<SQOperatorString, double> make_sq_operator_string(const std::string& s,
                                                            bool allow_reordering);

std::pair<SQOperatorString, double> make_sq_operator_string_from_list(const op_tuple_t& ops,

                                                                      bool allow_reordering);

template <size_t N> double apply_op(DeterminantImpl<N>& d, const SQOperatorString& sqop) {
    return apply_op(d, sqop.cre(), sqop.ann());
}

template <size_t N> double apply_op_safe(DeterminantImpl<N>& d, const SQOperatorString& sqop) {
    return apply_op_safe(d, sqop.cre(), sqop.ann());
}

} // namespace forte
