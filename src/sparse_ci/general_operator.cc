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
#include "sparse_operator.h"

namespace forte {

void GeneralOperator::add_term(const std::vector<std::pair<double, op_tuple_t>>& op_list,
                               double value) {
    coefficients_.push_back(value);
    size_t start = op_list_.size();
    size_t end = start + op_list.size();
    op_indices_.push_back(std::make_pair(start, end));
    // transform each term in the input into a SQOperator object
    for (const std::pair<double, op_tuple_t>& factor_op : op_list) {
        // Form a single operator object
        op_list_.push_back(SQOperator(factor_op.second, factor_op.first));
    }
}

void GeneralOperator::add_term(const std::vector<SQOperator>& ops, double value) {
    coefficients_.push_back(value);
    size_t start = op_list_.size();
    size_t end = start + ops.size();
    op_indices_.push_back(std::make_pair(start, end));
    // transform each term in the input into a SQOperator object
    for (const SQOperator& op : ops) {
        op_list_.push_back(op);
    }
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

void GeneralOperator::add_term_from_str(std::string str, double value) {
    std::vector<SQOperator> ops;

    // the regex to parse the entries
    std::regex re("\\s?([\\+\\-])?\\s*(\\d*\\.?\\d*)?\\s*\\*?\\s*(\\[[0-9ab\\+\\-\\s]*\\])");
    // the match object
    std::smatch m;

    // here we match all the terms that look like +/- factor [<orb><a/b><+/-> ...]
    // in the middle of this code we parse the operator part and store it as a
    // std::vector<std::tuple<bool, bool, int>>  (in parsed_ops)
    // then we call op_t_to_SQOperator to get a SQOperator object
    while (std::regex_search(str, m, re)) {
        if (m.ready()) {
            double sign = parse_sign(m[1]);
            double factor = parse_factor(m[2]);
            auto op = parse_ops(m[3]);
            ops.push_back(SQOperator(op, sign * factor));
        }
        str = m.suffix().str();
    }
    add_term(ops, value);
}

std::pair<std::vector<SQOperator>, double> GeneralOperator::get_term(size_t n) {
    size_t begin = op_indices_[n].first;
    size_t end = op_indices_[n].second;
    std::vector<SQOperator> ops(op_list_.begin() + begin, op_list_.begin() + end);
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

std::vector<std::string> GeneralOperator::str() const {
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

std::string latex_sqops(const Determinant& cre, const Determinant& ann) {
    auto acre = cre.get_alfa_occ(cre.norb());
    auto bcre = cre.get_beta_occ(cre.norb());
    auto aann = ann.get_alfa_occ(ann.norb());
    auto bann = ann.get_beta_occ(ann.norb());
    std::reverse(aann.begin(), aann.end());
    std::reverse(bann.begin(), bann.end());

    std::string s;
    for (auto p : acre) {
        s += "\\hat{a}_{" + std::to_string(p) + " \\alpha}^\\dagger";
    }
    for (auto p : bcre) {
        s += "\\hat{a}_{" + std::to_string(p) + " \\beta}^\\dagger";
    }
    for (auto p : bann) {
        s += "\\hat{a}_{" + std::to_string(p) + " \\beta}";
    }
    for (auto p : aann) {
        s += "\\hat{a}_{" + std::to_string(p) + " \\alpha}";
    }
    return s;
}

std::string double_to_string(double value) {
    if (value == -1.0) {
        return "-";
    }
    if (value != 1.0) {
        return std::to_string(value);
    }
    return "";
}

std::string GeneralOperator::latex() const {
    std::vector<std::string> result;
    size_t nterms = coefficients_.size();
    // if we are printing only one term, distribute the coefficient
    if (nterms == 1) {
        std::string s;
        size_t begin = op_indices_[0].first;
        size_t end = op_indices_[0].second;
        for (size_t j = begin; j < end; j++) {
            const double factor = op_list_[j].factor();
            s += double_to_string(coefficients_[0] * factor) + "\\;";

            const auto& ann = op_list_[j].ann();
            const auto& cre = op_list_[j].cre();
            s += latex_sqops(cre, ann);
        }
        result.push_back(s);
    } else {
        for (size_t n = 0; n < nterms; n++) {
            std::string s = std::to_string(coefficients_[n]) + " * ( ";
            size_t begin = op_indices_[n].first;
            size_t end = op_indices_[n].second;
            for (size_t j = begin; j < end; j++) {
                const double factor = op_list_[j].factor();
                if (j != begin) {
                    s += (factor < 0.0) ? " " : " +";
                }
                const auto& ann = op_list_[j].ann();
                const auto& cre = op_list_[j].cre();
                s += std::to_string(factor) + "\\; " + latex_sqops(cre, ann);
            }
            s += " )";
            result.push_back(s);
        }
    }
    return to_string(result, " + ");
}

} // namespace forte
