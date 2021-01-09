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

#include <algorithm>
#include <numeric>
#include <regex>

#include "helpers/combinatorial.h"
#include "helpers/timer.h"
#include "helpers/string_algorithms.h"

#include "sparse_operator.h"

namespace forte {

// SQOperator creation_alpha_orb_vec_to_SQOperator(
//    const std::vector<std::tuple<bool, bool, int>>& creation_alpha_orb_vec, double factor) {

//    Determinant cre, ann;

//    bool is_sorted =
//        std::is_sorted(creation_alpha_orb_vec.begin(), creation_alpha_orb_vec.end(), compare_ops);

//    // if not sorted, compute the permutation factor
//    if (not is_sorted) {
//        // We first sort the operators so that they are ordered in the following way
//        // [last](alpha cre. ascending) (beta cre. ascending) (beta ann. descending) (alpha ann.
//        // descending)[first] and keep track of the sign. We sort the operators using a set of
//        // auxiliary indices so that we can keep track of the permutation of the operators and
//        their
//        // sign
//        std::vector<size_t> idx(creation_alpha_orb_vec.size());
//        std::iota(idx.begin(), idx.end(), 0);
//        std::stable_sort(idx.begin(), idx.end(), [&creation_alpha_orb_vec](size_t i1, size_t i2) {
//            return compare_ops(creation_alpha_orb_vec[i1], creation_alpha_orb_vec[i2]);
//        });
//        auto parity = permutation_parity(idx);
//        // set the factor including the parity of the permutation
//        factor *= 1.0 - 2.0 * parity;
//    }

//    // set the bitarray part of the operator (the order does not matter)
//    for (auto creation_alpha_orb : creation_alpha_orb_vec) {
//        bool creation = std::get<0>(creation_alpha_orb);
//        bool alpha = std::get<1>(creation_alpha_orb);
//        int orb = std::get<2>(creation_alpha_orb);
//        if (creation) {
//            if (alpha) {
//                cre.set_alfa_bit(orb, true);
//            } else {
//                cre.set_beta_bit(orb, true);
//            }
//        } else {
//            if (alpha) {
//                ann.set_alfa_bit(orb, true);
//            } else {
//                ann.set_beta_bit(orb, true);
//            }
//        }
//    }
//    return SQOperator(factor, cre, ann);
//}

void SparseOperator::add_term(const std::vector<std::tuple<bool, bool, int>>& op_list,
                              double coefficient) {
    op_list_.push_back(SQOperator(op_list, coefficient));
}

void SparseOperator::add_term(const SQOperator& sqop) { op_list_.push_back(sqop); }

std::vector<std::tuple<bool, bool, int>> sparse_parse_ops(const std::string& s) {
    // reverse the operator order
    auto clean_s = s.substr(1, s.size() - 2);

    auto ops_str = split_string(clean_s, " ");
    std::reverse(ops_str.begin(), ops_str.end());

    std::vector<std::tuple<bool, bool, int>> ops_vec_tuple;
    for (auto op_str : ops_str) {
        size_t len = op_str.size();
        bool creation = op_str[len - 1] == '+' ? true : false;
        bool alpha = op_str[len - 2] == 'a' ? true : false;
        int orb = stoi(op_str.substr(0, len - 2));
        ops_vec_tuple.push_back(std::make_tuple(creation, alpha, orb));
    }
    return ops_vec_tuple;
}

void SparseOperator::add_term_from_str(std::string str, double value) {
    // the regex to parse the entries
    std::regex re("\\s*(\\[[0-9ab\\+\\-\\s]*\\])");
    // the match object
    std::smatch m;

    // here we match onethe terms of the form [<orb><a/b><+/-> ...]
    // in the middle of this code we parse the operator part and store it as a
    // std::vector<std::tuple<bool, bool, int>>  (in parsed_ops)
    // then we call creation_alpha_orb_vec_to_SQOperator to get a SQOperator object
    if (std::regex_match(str, m, re)) {
        if (m.ready()) {
            auto ops_vec_tuple = sparse_parse_ops(m[1]);
            add_term(ops_vec_tuple, value);
        }
    }
}

const SQOperator& SparseOperator::get_term(size_t n) const { return op_list_[n]; }

std::vector<double> SparseOperator::coefficients() {
    std::vector<double> v;
    for (const SQOperator& sqop : op_list_) {
        v.push_back(sqop.factor());
    }
    return v;
}

void SparseOperator::set_coefficients(std::vector<double>& values) {
    for (size_t n = 0, nmax = values.size(); n < nmax; ++n) {
        op_list_[n].set_factor(values[n]);
    }
}

void SparseOperator::pop_term() {
    if (nterms() > 0) {
        op_list_.pop_back();
    }
}

std::vector<std::string> SparseOperator::str() const {
    std::vector<std::string> v;
    for (const SQOperator& sqop : op_list_) {
        v.push_back(sqop.str());
    }
    return v;
}

std::string SparseOperator::latex() const {
    std::vector<std::string> v;
    for (const SQOperator& sqop : op_list_) {
        v.push_back(sqop.latex());
    }
    return to_string(v, " + ");
}
} // namespace forte
