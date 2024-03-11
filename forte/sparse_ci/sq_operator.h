// /*
//  * @BEGIN LICENSE
//  *
//  * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
//  * that implements a variety of quantum chemistry methods for strongly
//  * correlated electrons.
//  *
//  * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
//  *
//  * The copyrights for code used from other parties are included in
//  * the corresponding files.
//  *
//  * This program is free software: you can redistribute it and/or modify
//  * it under the terms of the GNU Lesser General Public License as published by
//  * the Free Software Foundation, either version 3 of the License, or
//  * (at your option) any later version.
//  *
//  * This program is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU Lesser General Public License for more details.
//  *
//  * You should have received a copy of the GNU Lesser General Public License
//  * along with this program.  If not, see http://www.gnu.org/licenses/.
//  *
//  * @END LICENSE
//  */

// #pragma once

// #include <vector>

// #include "sparse_ci/determinant.h"
// #include "sparse_ci/sq_operator_string.h"

// namespace forte {

// /**
//  * A data structure used to represent a second quantized operator string like
//  *
//  * ... op(2) op(1) op(0), where op(i) = a^\dagger_(orb_i, spin_i) or a_(orb_i, spin_i)
//  *
//  * like, for example
//  *
//  *  a^\dagger_{2,\alpha} a^\dagger_{3,\beta} a_{1,\beta} a_{0,\alpha}
//  *
//  * This operator is stored as
//  *
//  *  [(false,true,0),(false,false,1),(true,false,3),(true,true,2)]
//  *
//  * The data format is
//  *
//  *  [(creation_0, spin_0, orb_0), (creation_1, spin_1, orb_1), ...]
//  *
//  * where the operators are arranged as
//  *
//  * where
//  *  creation_i  : bool (true = creation, false = annihilation)
//  *  spin_i      : bool (true = alpha, false = beta)
//  *  orb_i       : int  (the index of the mo)
//  *
//  */
// using op_tuple_t = std::vector<std::tuple<bool, bool, int>>;

// class SQOperator {
//   public:
//     SQOperator(double coefficient, const SQOperatorString& sqop_str);

//     /**
//      * @brief Create a second quantized operator
//      *
//      * @param ops a vector of triplets (is_creation, is_alpha, orb) that specify
//      *        the second quantized operators
//      * @param coefficient of the coefficient associated with this operator
//      * @param allow_reordering if true this function will reorder all terms and put them in
//      * canonical order adjusting the coefficient to account for the number of permutations. If set
//      * to false, this function will only accept operators that are already in the canonical order
//      */
//     /// @return the numerical coefficient associated with this operator
//     double coefficient() const;
//     /// @return the string of creation and annihilation operators associated with this operator
//     const SQOperatorString& sqop_str() const;
//     /// @return a Determinant object that represents the creation operators
//     const Determinant& cre() const;
//     /// @return a Determinant object that represents the annihilation operators
//     const Determinant& ann() const;
//     /// @return compare this operator with another operator
//     bool operator==(const SQOperator& other) const;
//     /// @return compare this operator with another operator
//     bool operator<(const SQOperator& other) const;
//     /// @return true if this operator is a number operator (i.e. it contains no creation or
//     bool is_number() const;
//     /// @return the number of creation + annihilation operators in this operator
//     int count() const;
//     /// @param value set the coefficient associated with this operator
//     void set_coefficient(double& value);
//     /// @return a string representation of this operator
//     std::string str() const;
//     /// @return a latex representation of this operator
//     std::string latex() const;
//     /// @return a sq_operator that is the adjoint of this operator
//     SQOperator adjoint() const;

//   private:
//     /// a numerical coefficient associated with this product of sq operators
//     double coefficient_;
//     /// a string representation of the product of creation and annihilation operators
//     SQOperatorString sqop_str_;
// };

// /// @return The product of two second quantized operators
// std::vector<SQOperator> operator*(const SQOperator& lhs, const SQOperator& rhs);

// /// @return The product of a second quantized operator and a numerical factor
// std::vector<SQOperator> operator*(const double factor, const SQOperator& sqop);

// /// @return The commutator of two second quantized operators
// std::vector<SQOperator> commutator(const SQOperator& lhs, const SQOperator& rhs);

// SQOperator make_sq_operator(const std::string& s, double coefficient = 1.0,
//                             bool allow_reordering = false);

// SQOperator make_sq_operator(const op_tuple_t& ops, double coefficient = 1.0,
//                             bool allow_reordering = false);

// } // namespace forte
