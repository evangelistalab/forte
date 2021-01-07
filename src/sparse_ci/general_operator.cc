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

#include "general_operator.h"

namespace forte {

SingleOperator::SingleOperator(double factor, const Determinant& cre, const Determinant& ann)
    : factor_(factor), cre_(cre), ann_(ann) {}

double SingleOperator::factor() const { return factor_; }
const Determinant& SingleOperator::cre() const { return cre_; }
const Determinant& SingleOperator::ann() const { return ann_; }

std::tuple<bool, bool, int> flip_spin(const std::tuple<bool, bool, int>& t) {
    return std::make_tuple(std::get<0>(t), not std::get<1>(t), std::get<2>(t));
}

// a comparison function used to sort second quantized operators in the order
//  alpha+ beta+ beta- alpha-
bool compare_ops(const std::tuple<bool, bool, int>& lhs, const std::tuple<bool, bool, int>& rhs) {
    const auto& l_cre = std::get<0>(lhs);
    const auto& r_cre = std::get<0>(rhs);
    if ((l_cre == true) and (r_cre == true)) {
        return flip_spin(lhs) > flip_spin(rhs);
    }
    return flip_spin(lhs) < flip_spin(rhs);
}

SingleOperator op_t_to_SingleOperator(const op_t& op) {
    const std::vector<std::tuple<bool, bool, int>>& creation_alpha_orb_vec = op.second;

    Determinant cre, ann;
    // set the factor including the parity of the permutation
    double factor = op.first;

    bool is_sorted =
        std::is_sorted(creation_alpha_orb_vec.begin(), creation_alpha_orb_vec.end(), compare_ops);

    // if not sorted, compute the permutation factor
    if (not is_sorted) {
        // We first sort the operators so that they are ordered in the following way
        // [last](alpha cre. ascending) (beta cre. ascending) (beta ann. descending) (alpha ann.
        // descending)[first] and keep track of the sign. We sort the operators using a set of
        // auxiliary indices so that we can keep track of the permutation of the operators and their
        // sign
        std::vector<size_t> idx(creation_alpha_orb_vec.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&creation_alpha_orb_vec](size_t i1, size_t i2) {
            return compare_ops(creation_alpha_orb_vec[i1], creation_alpha_orb_vec[i2]);
        });
        auto parity = permutation_parity(idx);
        // set the factor including the parity of the permutation
        factor *= 1.0 - 2.0 * parity;
    }

    // set the bitarray part of the operator (the order does not matter)
    for (auto creation_alpha_orb : creation_alpha_orb_vec) {
        bool creation = std::get<0>(creation_alpha_orb);
        bool alpha = std::get<1>(creation_alpha_orb);
        int orb = std::get<2>(creation_alpha_orb);
        if (creation) {
            if (alpha) {
                cre.set_alfa_bit(orb, true);
            } else {
                cre.set_beta_bit(orb, true);
            }
        } else {
            if (alpha) {
                ann.set_alfa_bit(orb, true);
            } else {
                ann.set_beta_bit(orb, true);
            }
        }
    }
    return SingleOperator(factor, cre, ann);
}

double parse_sign(const std::string& s) {
    if (s == "-") {
        return -1.0;
    }
    return 1.0;
}

double parse_factor(const std::string& s) {
    if (s == "") {
        return 1.0;
    }
    return stod(s);
}

std::vector<std::tuple<bool, bool, int>> parse_ops(const std::string& s) {
    // reverse the operator order
    auto clean_s = s.substr(1, s.size() - 2);

    auto ops_str = split_string(clean_s, " ");
    std::reverse(ops_str.begin(), ops_str.end());

    std::vector<std::tuple<bool, bool, int>> ops_tuple;
    for (auto op_str : ops_str) {
        size_t len = op_str.size();
        bool creation = op_str[len - 1] == '+' ? true : false;
        bool alpha = op_str[len - 2] == 'a' ? true : false;
        int orb = stoi(op_str.substr(0, len - 2));
        ops_tuple.push_back(std::make_tuple(creation, alpha, orb));
    }
    return ops_tuple;
}
// void GeneralOperator::add_operator(const std::vector<op_t>& op_list, double value) {
//    coefficients_.push_back(value);
//    size_t start = op_list_.size();
//    size_t end = start + op_list.size();
//    op_indices_.push_back(std::make_pair(start, end));
//    // transform each term in the input into a SingleOperator object
//    for (const op_t& op : op_list) {
//        // Form a single operator object
//        op_list_.push_back(op_t_to_SingleOperator(op));
//    }
//}

void GeneralOperator::add_term(const std::vector<SingleOperator>& ops, double value) {
    coefficients_.push_back(value);
    size_t start = op_list_.size();
    size_t end = start + ops.size();
    op_indices_.push_back(std::make_pair(start, end));
    // transform each term in the input into a SingleOperator object
    for (const SingleOperator& op : ops) {
        op_list_.push_back(op);
    }
}

void GeneralOperator::add_term_from_str(std::string str, double value) {
    std::vector<SingleOperator> ops;

    // the regex to parse the entries
    std::regex re("\\s?([\\+\\-])?\\s*(\\d*\\.?\\d*)?\\s*\\*?\\s*(\\[[0-9ab\\+\\-\\s]*\\])");
    // the match object
    std::smatch m;

    // here we match all the terms that look like +/- factor [<orb><a/b><+/-> ...]
    // in the middle of this code we parse the operator part and store it as a
    // std::vector<std::tuple<bool, bool, int>>  (in parsed_ops)
    // then we call op_t_to_SingleOperator to get a SingleOperator object
    while (std::regex_search(str, m, re)) {
        if (m.ready()) {
            double sign = parse_sign(m[1]);
            double factor = parse_factor(m[2]);
            auto op = parse_ops(m[3]);
            op_t parsed_ops = std::make_pair(sign * factor, op);
            ops.push_back(op_t_to_SingleOperator(parsed_ops));
        }
        str = m.suffix().str();
    }
    add_term(ops, value);
}

std::pair<std::vector<SingleOperator>, double> GeneralOperator::get_term(size_t n) {
    size_t begin = op_indices_[n].first;
    size_t end = op_indices_[n].second;
    std::vector<SingleOperator> ops(op_list_.begin() + begin, op_list_.begin() + end);
    return std::make_pair(ops, coefficients_[n]);
}

void GeneralOperator::pop_term() {
    if (nterms() > 0) {
        coefficients_.pop_back();
        auto start_end = op_indices_.back();
        op_indices_.pop_back();
        size_t start = start_end.first;
        size_t end = start_end.second;
        for (; start < end; ++start) {
            op_list_.pop_back();
        }
    }
}

std::vector<std::string> GeneralOperator::str() {
    std::vector<std::string> result;
    size_t nterms = coefficients_.size();
    for (size_t n = 0; n < nterms; n++) {
        std::string s = std::to_string(coefficients_[n]) + " * ( ";
        size_t begin = op_indices_[n].first;
        size_t end = op_indices_[n].second;
        for (size_t j = begin; j < end; j++) {
            const double factor = op_list_[j].factor();
            const auto& ann = op_list_[j].ann();
            const auto& cre = op_list_[j].cre();
            if (j != begin) {
                s += (factor < 0.0) ? " " : " +";
            }
            s += std::to_string(factor) + " * [ ";
            auto acre = cre.get_alfa_occ(cre.norb());
            auto bcre = cre.get_beta_occ(cre.norb());
            auto aann = ann.get_alfa_occ(ann.norb());
            auto bann = ann.get_beta_occ(ann.norb());
            std::reverse(aann.begin(), aann.end());
            std::reverse(bann.begin(), bann.end());
            for (auto p : acre) {
                s += std::to_string(p) + "a+ ";
            }
            for (auto p : bcre) {
                s += std::to_string(p) + "b+ ";
            }
            for (auto p : bann) {
                s += std::to_string(p) + "b- ";
            }
            for (auto p : aann) {
                s += std::to_string(p) + "a- ";
            }
            s += "]";
        }
        s += " )";
        result.push_back(s);
    }
    return result;
}

StateVector::StateVector() { /* std::cout << "Created a StateVector object" << std::endl; */
}

StateVector::StateVector(const det_hash<double>& state_vec) : state_vec_(state_vec) {}

} // namespace forte
