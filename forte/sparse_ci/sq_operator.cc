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

// #include <algorithm>
// #include <numeric>

// #include "helpers/combinatorial.h"
// #include "helpers/string_algorithms.h"

// #include "sparse_ci/sq_operator.h"

// namespace forte {

// SQOperator::SQOperator(double coefficient, const SQOperatorString& sqop_str)
//     : coefficient_(coefficient), sqop_str_(sqop_str) {}

// bool SQOperator::operator==(const SQOperator& other) const {
//     return (coefficient() == other.coefficient()) and (sqop_str() == other.sqop_str());
// }

// bool SQOperator::operator<(const SQOperator& other) const {
//     if (coefficient() != other.coefficient()) {
//         return coefficient() < other.coefficient();
//     }
//     return sqop_str() < other.sqop_str();
// }

// bool SQOperator::is_number() const { return sqop_str().is_number(); }

// int SQOperator::count() const { return sqop_str().count(); }

// double SQOperator::coefficient() const { return coefficient_; }

// const SQOperatorString& SQOperator::sqop_str() const { return sqop_str_; }

// const Determinant& SQOperator::cre() const { return sqop_str().cre(); }

// const Determinant& SQOperator::ann() const { return sqop_str().ann(); }

// void SQOperator::set_coefficient(double& value) { coefficient_ = value; }

// std::string SQOperator::str() const {
//     return to_string_with_precision(coefficient(), 12) + " * " + sqop_str().str();
// }

// std::string SQOperator::latex() const {
//     return double_to_string_latex(coefficient()) + "\\;" + sqop_str().latex();
// }

// SQOperator SQOperator::adjoint() const { return SQOperator(coefficient_, sqop_str_.adjoint()); }

// SQOperator make_sq_operator(const std::string& s, double coefficient, bool allow_reordering) {
//     auto [phase, sqop_str] = make_sq_operator_string(s, allow_reordering);
//     return SQOperator(phase * coefficient, sqop_str);
// }

// SQOperator make_sq_operator(const op_tuple_t& ops, double coefficient, bool allow_reordering) {
//     auto [phase, sqop_str] = make_sq_operator_string_from_list(ops, allow_reordering);
//     return SQOperator(phase * coefficient, sqop_str);
// }

// std::vector<SQOperator> operator*(const double factor, const SQOperator& sqop) {
//     return {SQOperator(factor * sqop.coefficient(), sqop.sqop_str())};
// }

// std::vector<SQOperator> operator*(const SQOperator& lhs, const SQOperator& rhs) {
//     auto prod = lhs.sqop_str() * rhs.sqop_str();
//     auto coefficient = lhs.coefficient() * rhs.coefficient();

//     // aggregate the terms
//     std::unordered_map<SQOperatorString, double, SQOperatorString::Hash> result_map;
//     for (const auto& [c, sqop_str] : prod) {
//         result_map[sqop_str] += coefficient * c;
//     }

//     std::vector<SQOperator> result;
//     result.reserve(result_map.size());

//     for (const auto& [sqop_str, c] : result_map) {
//         result.emplace_back(c, sqop_str);
//     }
//     return result;
// }

// std::vector<SQOperator> commutator(const SQOperator& lhs, const SQOperator& rhs) {
//     const auto common_l_cre_r_cre =
//     lhs.sqop_str().cre().fast_a_and_b_count(rhs.sqop_str().cre()); const auto common_l_ann_r_ann
//     = lhs.sqop_str().ann().fast_a_and_b_count(rhs.sqop_str().ann()); const auto
//     common_l_ann_r_cre = lhs.sqop_str().ann().fast_a_and_b_count(rhs.sqop_str().cre()); const
//     auto nl = lhs.sqop_str().count(); const auto nr = rhs.sqop_str().count(); if
//     (common_l_cre_r_cre == 0 and common_l_ann_r_ann == 0 and common_l_ann_r_cre == 0) {
//         if (nl * nr % 2 == 0) {
//             return {};
//         }
//         const auto prod = lhs.sqop_str() * rhs.sqop_str();
//         const auto coefficient = 2.0 * prod[0].first * lhs.coefficient() * rhs.coefficient();
//         return {SQOperator(coefficient, prod[0].second)};
//     }

//     const auto lr_prod = lhs.sqop_str() * rhs.sqop_str();
//     const auto rl_prod = rhs.sqop_str() * lhs.sqop_str();
//     const auto coefficient = lhs.coefficient() * rhs.coefficient();

//     // aggregate the terms
//     std::unordered_map<SQOperatorString, double, SQOperatorString::Hash> result_map;
//     for (const auto& [c, sqop_str] : lr_prod) {
//         result_map[sqop_str] += coefficient * c;
//     }
//     for (const auto& [c, sqop_str] : rl_prod) {
//         result_map[sqop_str] -= coefficient * c;
//     }

//     std::vector<SQOperator> result;
//     result.reserve(result_map.size());

//     for (const auto& [sqop_str, c] : result_map) {
//         if (c != 0.0) {
//             result.emplace_back(c, sqop_str);
//         }
//     }
//     return result;
// }

// } // namespace forte
