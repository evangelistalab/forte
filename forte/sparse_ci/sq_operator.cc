/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "helpers/combinatorial.h"
#include "helpers/string_algorithms.h"

#include "sparse_ci/sq_operator.h"

namespace forte {

SQOperator::SQOperator(double coefficient, const Determinant& cre, const Determinant& ann)
    : coefficient_(coefficient), cre_(cre), ann_(ann) {}

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

SQOperator::SQOperator(const op_tuple_t& ops, double coefficient, bool allow_reordering) {
    const std::vector<std::tuple<bool, bool, int>>& creation_alpha_orb_vec = ops;
    coefficient_ = coefficient;

    bool is_sorted =
        std::is_sorted(creation_alpha_orb_vec.begin(), creation_alpha_orb_vec.end(), compare_ops);

    // if not sorted, compute the permutation coefficient
    if (not is_sorted) {
        if (not allow_reordering) {
            throw std::runtime_error(
                "Trying to initialize a SQOperator object with a product of\n"
                "operators that are not arranged in the canonical form\n\n"
                "    a+_p1 a+_p2 ...  a+_P1 a+_P2 ...   ... a-_Q2 a-_Q1   ... a-_q2 a-_q1\n"
                "    alpha creation   beta creation    beta annihilation  alpha annihilation\n\n"
                "with indices sorted as\n\n"
                "    (p1 < p2 < ...) (P1 < P2 < ...)  (... > Q2 > Q1) (... > q2 > q1)\n");
        }
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
        // set the coefficient including the parity of the permutation
        coefficient_ *= 1.0 - 2.0 * parity;
    }

    // set the bitarray part of the operator (the order does not matter)
    bool contains_duplicates = false;
    for (auto creation_alpha_orb : creation_alpha_orb_vec) {
        bool creation = std::get<0>(creation_alpha_orb);
        bool alpha = std::get<1>(creation_alpha_orb);
        int orb = std::get<2>(creation_alpha_orb);
        if (creation) {
            if (alpha) {
                if (cre_.get_alfa_bit(orb))
                    contains_duplicates = true;
                cre_.set_alfa_bit(orb, true);
            } else {
                if (cre_.get_beta_bit(orb))
                    contains_duplicates = true;
                cre_.set_beta_bit(orb, true);
            }
        } else {
            if (alpha) {
                if (ann_.get_alfa_bit(orb))
                    contains_duplicates = true;
                ann_.set_alfa_bit(orb, true);
            } else {
                if (ann_.get_beta_bit(orb))
                    contains_duplicates = true;
                ann_.set_beta_bit(orb, true);
            }
        }
    }
    if (contains_duplicates) {
        throw std::runtime_error("Trying to initialize a SQOperator object with a product of\n"
                                 "operators that contains repeated operators.\n");
    }
}

double SQOperator::coefficient() const { return coefficient_; }
const Determinant& SQOperator::cre() const { return cre_; }
const Determinant& SQOperator::ann() const { return ann_; }
void SQOperator::set_coefficient(double& value) { coefficient_ = value; }

std::string SQOperator::str() const {
    std::string s = to_string_with_precision(coefficient(), 12) + " * [ ";
    // std::string s = std::to_string(coefficient()) + " * [ ";
    auto acre = cre_.get_alfa_occ(cre_.norb());
    auto bcre = cre_.get_beta_occ(cre_.norb());
    auto aann = ann_.get_alfa_occ(ann_.norb());
    auto bann = ann_.get_beta_occ(ann_.norb());
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

    return s;
}

std::string sq_double_to_string(double value) {
    if (value == -1.0) {
        return "-";
    }
    if (value == 1.0) {
        return "+";
    }
    return (value > 0.0 ? "+" : "") + std::to_string(value);
}

std::string SQOperator::latex() const {
    std::string s = sq_double_to_string(coefficient()) + "\\;";

    auto acre = cre_.get_alfa_occ(cre_.norb());
    auto bcre = cre_.get_beta_occ(cre_.norb());
    auto aann = ann_.get_alfa_occ(ann_.norb());
    auto bann = ann_.get_beta_occ(ann_.norb());
    std::reverse(aann.begin(), aann.end());
    std::reverse(bann.begin(), bann.end());

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

} // namespace forte
