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

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <numeric>

#include "helpers/combinatorial.h"
#include "helpers/timer.h"
#include "helpers/string_algorithms.h"

#include "sparse_operator.h"

namespace forte {

std::string format_term_in_sum(sparse_scalar_t coefficient, const std::string& term) {
    // if (term == "[ ]") {
    //     if (coefficient == 0.0) {
    //         return "";
    //     } else {
    //         return (coefficient > 0.0 ? "+ " : "") + to_string_with_precision(coefficient, 12);
    //     }
    // }
    // if (coefficient == 0.0) {
    //     return "";
    // } else if (coefficient == 1.0) {
    //     return "+ " + term;
    // } else if (coefficient == -1.0) {
    //     return "- " + term;
    // } else if (coefficient == static_cast<int>(coefficient)) {
    //     return to_string_with_precision(coefficient, 12) + " * " + term;
    // } else {
    //     std::string s = to_string_with_precision(coefficient, 12);
    //     s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    //     return s + " * " + term;
    // }
    // return "";
    // if constexpr (std::is_same_v<sparse_scalar_t, std::complex<double>>) {
    // }
    // if constexpr (std::is_same_v<sparse_scalar_t, double>) {
    //     return std::format("{} * {}", coefficient, term);
    // }
    return std::format("({} + {}i) * {}", std::real(coefficient), std::imag(coefficient), term);
}

std::vector<std::string> SparseOperator::str() const {
    std::vector<std::string> v;
    for (const auto& [sqop, c] : this->elements()) {
        if (std::abs(c) < 1.0e-12)
            continue;
        v.push_back(format_term_in_sum(c, sqop.str()));
    }
    // sort v to guarantee a consistent order
    std::sort(v.begin(), v.end());
    return v;
}

std::string SparseOperator::latex() const {
    std::vector<std::string> v;
    for (const auto& [sqop, c] : this->elements()) {
        const std::string s = to_string_latex(c) + "\\;" + sqop.latex();
        v.push_back(s);
    }
    // sort v to guarantee a consistent order
    std::sort(v.begin(), v.end());
    return join(v, " ");
}

void SparseOperator::add_term_from_str(const std::string& s, sparse_scalar_t coefficient,
                                       bool allow_reordering) {
    auto [sqop, phase] = make_sq_operator_string(s, allow_reordering);
    add(sqop, phase * coefficient);
}

SparseOperator SparseOperatorList::to_operator() const {
    SparseOperator op;
    for (const auto& [sqop, c] : elements()) {
        op.add(sqop, c);
    }
    return op;
}

SparseOperator operator*(const SparseOperator& lhs, const SparseOperator& rhs) {
    SparseOperator result;
    for (const auto& [sqop_lhs, c_lhs] : lhs.elements()) {
        for (const auto& [sqop_rhs, c_rhs] : rhs.elements()) {
            const auto prod = sqop_lhs * sqop_rhs;
            for (const auto& [sqop, c] : prod) {
                if (c * c_lhs * c_rhs != 0.0) {
                    result[sqop] += c * c_lhs * c_rhs;
                }
            }
        }
    }
    return result;
}

SparseOperator product(const SparseOperator& lhs, const SparseOperator& rhs) {
    SQOperatorProductComputer computer;
    SparseOperator C;
    for (const auto& [lhs_op, lhs_c] : lhs.elements()) {
        for (const auto& [rhs_op, rhs_c] : rhs.elements()) {
            computer.product(
                lhs_op, rhs_op, lhs_c * rhs_c,
                [&C](const SQOperatorString& sqop, const sparse_scalar_t c) { C.add(sqop, c); });
        }
    }
    return C;
}

SparseOperator commutator(const SparseOperator& lhs, const SparseOperator& rhs) {
    // place the elements in a map to avoid duplicates and to simplify the addition
    SQOperatorProductComputer computer;
    SparseOperator C;
    for (const auto& [lhs_op, lhs_c] : lhs.elements()) {
        for (const auto& [rhs_op, rhs_c] : rhs.elements()) {
            computer.commutator(
                lhs_op, rhs_op, lhs_c * rhs_c,
                [&C](const SQOperatorString& sqop, const sparse_scalar_t c) { C[sqop] += c; });
        }
    }
    return C;
}

void SparseOperatorList::add_term_from_str(std::string str, sparse_scalar_t coefficient,
                                           bool allow_reordering) {
    auto [sqop, phase] = make_sq_operator_string(str, allow_reordering);
    add(sqop, phase * coefficient);
}

std::vector<std::string> SparseOperatorList::str() const {
    std::vector<std::string> v;
    for (const auto& [sqop, c] : this->elements()) {
        if (std::abs(c) < 1.0e-12)
            continue;
        v.push_back(format_term_in_sum(c, sqop.str()));
    }
    return v;
}

} // namespace forte
