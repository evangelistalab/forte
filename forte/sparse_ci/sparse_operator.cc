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

void SparseOperator::add_term(const std::vector<std::tuple<bool, bool, int>>& op_list,
                              double coefficient, bool allow_reordering) {
    op_list_.push_back(SQOperator(op_list, coefficient, allow_reordering));
}

void SparseOperator::add_term(const SQOperator& sqop) { op_list_.push_back(sqop); }

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
            auto sq_operator = make_sq_operator(m[1], coefficient, allow_reordering);
            if (is_antihermitian()) {
                if (sq_operator.is_number()) {
                    throw std::runtime_error("SparseOperator: the operator " + str +
                                             " contains a number component.\nThis is not allowed"
                                             "for anti-Hermitian operators");
                }
            }
            add_term(sq_operator);
        }
    } else {
        std::string msg =
            "add_term_from_str(std::string str, double value) could not parse the string " + str;
        throw std::runtime_error(msg);
    }
}

const SQOperator& SparseOperator::term(size_t n) const { return op_list_[n]; }

std::vector<double> SparseOperator::coefficients() const {
    std::vector<double> v;
    for (const SQOperator& sqop : op_list_) {
        v.push_back(sqop.coefficient());
    }
    return v;
}

void SparseOperator::set_coefficients(std::vector<double>& values) {
    for (size_t n = 0, nmax = values.size(); n < nmax; ++n) {
        op_list_[n].set_coefficient(values[n]);
    }
}

void SparseOperator::pop_term() {
    if (size() > 0) {
        op_list_.pop_back();
    }
}

std::vector<std::string> SparseOperator::str() const {
    std::vector<std::string> v;
    for (const SQOperator& sqop : op_list_) {
        if (std::fabs(sqop.coefficient()) < 1.0e-12)
            continue;
        v.push_back(sqop.str());
        if (is_antihermitian()) {
            auto sqop_dagger = SQOperator(-sqop.coefficient(), sqop.ann(), sqop.cre());
            v.push_back(sqop_dagger.str());
        }
    }
    return v;
}

std::string SparseOperator::latex() const {
    std::vector<std::string> v;
    for (const SQOperator& sqop : op_list_) {
        v.push_back(sqop.latex());
        if (is_antihermitian()) {
            auto sqop_dagger = SQOperator(-sqop.coefficient(), sqop.ann(), sqop.cre());
            v.push_back(sqop_dagger.latex());
        }
    }
    return join(v, " ");
}

SparseOperator SparseOperator::adjoint() const {
    auto adjoint_operator = SparseOperator(antihermitian_);
    for (const SQOperator& sqop : op_list_) {
        adjoint_operator.add_term(sqop.adjoint());
    }
    return adjoint_operator;
}
} // namespace forte
