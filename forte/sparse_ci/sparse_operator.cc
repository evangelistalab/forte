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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <regex>

#include "helpers/combinatorial.h"
#include "helpers/timer.h"
#include "helpers/string_algorithms.h"

#include "sparse_operator.h"

namespace forte {

SparseOperator::SparseOperator(
    bool antihermitian,
    const std::unordered_map<SQOperatorString, double, SQOperatorString::Hash>& op_map)
    : antihermitian_(antihermitian), op_map_(op_map) {
    // populate the insertion list in the order of the map
    for (const auto& [sqop_str, _] : op_map_) {
        op_insertion_list_.push_back(sqop_str);
    }
}

void SparseOperator::add_term(const std::vector<std::tuple<bool, bool, int>>& op_list,
                              double coefficient, bool allow_reordering) {
    auto [phase, sqop_str] = make_sq_operator_string_from_list(op_list, allow_reordering);
    add_term(sqop_str, phase * coefficient);
}

void SparseOperator::add_term(const SQOperatorString& sqop, double c) {
    auto result = op_map_.insert({sqop, c});
    if (!result.second) {
        // the element already exists, so add c to the existing value
        result.first->second += c;
    } else {
        // this is a new element, so add sqop to the insertion order tracking list
        op_insertion_list_.push_back(sqop);
    }
}

std::pair<SQOperatorString, double> SparseOperator::term(size_t n) const {
    if (n >= op_insertion_list_.size()) {
        throw std::out_of_range("Index out of range");
    }
    const SQOperatorString& key = op_insertion_list_[n];
    double value = op_map_.at(key); // This will throw std::out_of_range if key is not found, which
                                    // should never happen if the class is correctly maintained.
    return {key, value};
}

const SQOperatorString& SparseOperator::term_operator(size_t n) const {
    if (n >= op_insertion_list_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return op_insertion_list_[n];
}

size_t SparseOperator::size() const { return op_insertion_list_.size(); }

void SparseOperator::add_term_from_str(std::string str, double coefficient, bool allow_reordering) {
    // the regex to parse the entries
    std::regex re("\\s*(\\[[0-9ab\\+\\-\\s]*\\])");
    // the match object
    std::smatch m;

    // here we match terms of the form [<orb><a/b><+/-> ...], then parse the operator part
    // and translate it into a term that is added to the operator.
    //
    if (std::regex_match(str, m, re)) {
        if (m.ready()) {
            auto [phase, sqop_str] = make_sq_operator_string(m[1], allow_reordering);
            if (is_antihermitian()) {
                if (sqop_str.is_number()) {
                    throw std::runtime_error("SparseOperator: the operator " + str +
                                             " contains a number component.\nThis is not allowed"
                                             "for anti-Hermitian operators");
                }
            }
            add_term(sqop_str, phase * coefficient);
        }
    } else {
        std::string msg =
            "add_term_from_str(std::string str, double value) could not parse the string " + str;
        throw std::runtime_error(msg);
    }
}

std::vector<double> SparseOperator::coefficients() const {
    std::vector<double> v;
    v.reserve(op_insertion_list_.size());
    for (const auto& sqop_str : op_insertion_list_) {
        v.push_back(op_map_.at(sqop_str));
    }
    return v;
}

void SparseOperator::set_coefficients(const std::vector<double>& values) {
    if (values.size() != op_insertion_list_.size()) {
        throw std::invalid_argument("Mismatch in sizes of values and operator list.");
    }
    for (size_t n = 0, nmax = values.size(); n < nmax; ++n) {
        op_map_[op_insertion_list_[n]] = values[n];
    }
}

void SparseOperator::set_coefficient(size_t n, double value) {
    op_map_[op_insertion_list_[n]] = value;
}

double SparseOperator::coefficient(size_t n) const { return op_map_.at(op_insertion_list_[n]); }

void SparseOperator::pop_term() {
    if (!op_insertion_list_.empty()) {
        const auto& last_sqop_str = op_insertion_list_.back();
        op_map_.erase(last_sqop_str);
        op_insertion_list_.pop_back();
    }
}

std::vector<std::string> SparseOperator::str() const {
    std::vector<std::string> v;
    for (const auto& sqop_str : op_insertion_list_) {
        double coefficient = op_map_.at(sqop_str);
        if (std::fabs(coefficient) < 1.0e-12)
            continue;

        const std::string s = to_string_with_precision(coefficient, 12) + " * " + sqop_str.str();

        v.push_back(s);

        if (is_antihermitian()) {
            const std::string s =
                to_string_with_precision(-coefficient, 12) + " * " + sqop_str.adjoint().str();
            v.push_back(s);
        }
    }
    return v;
}

std::string SparseOperator::latex() const {
    std::vector<std::string> v;
    for (const auto& sqop_str : op_insertion_list_) {
        double coefficient = op_map_.at(sqop_str);
        const std::string s = double_to_string_latex(coefficient) + "\\;" + sqop_str.latex();
        v.push_back(s);
        if (is_antihermitian()) {
            const std::string s =
                double_to_string_latex(-coefficient) + "\\;" + sqop_str.adjoint().latex();
            v.push_back(s);
        }
    }
    return join(v, " ");
}

SparseOperator SparseOperator::adjoint() const {
    SparseOperator adjoint_operator(antihermitian_);
    for (const auto& sqop_str : op_insertion_list_) {
        // Retrieve the original coefficient
        auto coefficient = op_map_.at(sqop_str);

        adjoint_operator.add_term(sqop_str.adjoint(), coefficient);
    }
    return adjoint_operator;
}

SparseOperator& SparseOperator::operator+=(const SparseOperator& other) {
    for (const auto& [sqop_str, c] : other.op_map()) {
        this->add_term(sqop_str, c);
    }
    return *this;
}

SparseOperator operator*(const SparseOperator& lhs, const SparseOperator& rhs) {
    // this implementation only works for non-antihermitian operators
    if (lhs.is_antihermitian() or rhs.is_antihermitian()) {
        throw std::runtime_error("SparseOperator: operator* is not implemented for antihermitian "
                                 "operators");
    }

    std::unordered_map<SQOperatorString, double, SQOperatorString::Hash> result_map;

    for (const auto& [sqop_lhs, c_lhs] : lhs.op_map()) {
        for (const auto& [sqop_rhs, c_rhs] : rhs.op_map()) {
            const auto prod = sqop_lhs * sqop_rhs;
            for (const auto& [c, sqop_str] : prod) {
                result_map[sqop_str] += c * c_lhs * c_rhs;
            }
        }
    }
    return SparseOperator(lhs.is_antihermitian(), result_map);
}

} // namespace forte
