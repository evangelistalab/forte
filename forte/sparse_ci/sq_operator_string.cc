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
#include <numeric>

#include "helpers/combinatorial.h"
#include "helpers/string_algorithms.h"

#include "sparse_ci/sq_operator_string.h"

namespace forte {

void generate_wick_contractions(const SQOperatorString& lhs, const SQOperatorString& rhs,
                                std::vector<std::pair<SQOperatorString, double>>& result,
                                const double sign);

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

SQOperatorString::SQOperatorString() {}

SQOperatorString::SQOperatorString(const Determinant& cre, const Determinant& ann)
    : cre_(cre), ann_(ann) {}

const Determinant& SQOperatorString::cre() const { return cre_; }

const Determinant& SQOperatorString::ann() const { return ann_; }

bool SQOperatorString::is_number() const { return (cre().count() == 0) and (ann().count() == 0); }

int SQOperatorString::count() const { return cre().count() + ann().count(); }

bool SQOperatorString::operator==(const SQOperatorString& other) const {
    return (cre() == other.cre()) and (ann() == other.ann());
}

bool SQOperatorString::operator<(const SQOperatorString& other) const {
    if (cre() != other.cre()) {
        return cre() < other.cre();
    }
    return ann() < other.ann();
}

SQOperatorString SQOperatorString::adjoint() const { return SQOperatorString(ann(), cre()); }

std::string SQOperatorString::str() const {
    std::string s = "[ ";
    auto acre = cre().get_alfa_occ(cre().norb());
    auto bcre = cre().get_beta_occ(cre().norb());
    auto aann = ann().get_alfa_occ(ann().norb());
    auto bann = ann().get_beta_occ(ann().norb());
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

std::string SQOperatorString::latex() const {
    auto acre = cre().get_alfa_occ(cre().norb());
    auto bcre = cre().get_beta_occ(cre().norb());
    auto aann = ann().get_alfa_occ(ann().norb());
    auto bann = ann().get_beta_occ(ann().norb());
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

std::vector<std::tuple<bool, bool, int>> parse_sq_operator(const std::string& s) {
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

std::pair<SQOperatorString, double> make_sq_operator_string_from_list(const op_tuple_t& ops,
                                                                      bool allow_reordering) {
    double coefficient = 1.0;
    const std::vector<std::tuple<bool, bool, int>>& creation_alpha_orb_vec = ops;

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
        coefficient *= 1.0 - 2.0 * parity;
    }

    Determinant cre, ann;
    // set the bitarray part of the operator (the order does not matter)
    bool contains_duplicates = false;
    for (auto creation_alpha_orb : creation_alpha_orb_vec) {
        bool creation = std::get<0>(creation_alpha_orb);
        bool alpha = std::get<1>(creation_alpha_orb);
        int orb = std::get<2>(creation_alpha_orb);
        if (creation) {
            if (alpha) {
                if (cre.get_alfa_bit(orb))
                    contains_duplicates = true;
                cre.set_alfa_bit(orb, true);
            } else {
                if (cre.get_beta_bit(orb))
                    contains_duplicates = true;
                cre.set_beta_bit(orb, true);
            }
        } else {
            if (alpha) {
                if (ann.get_alfa_bit(orb))
                    contains_duplicates = true;
                ann.set_alfa_bit(orb, true);
            } else {
                if (ann.get_beta_bit(orb))
                    contains_duplicates = true;
                ann.set_beta_bit(orb, true);
            }
        }
    }
    if (contains_duplicates) {
        throw std::runtime_error("Trying to initialize a SQOperator object with a product of\n"
                                 "operators that contains repeated operators.\n");
    }
    return std::make_pair(SQOperatorString(cre, ann), coefficient);
}

std::pair<SQOperatorString, double> make_sq_operator_string(const std::string& s,
                                                            bool allow_reordering) {
    auto ops_vec_tuple = parse_sq_operator(s);
    return make_sq_operator_string_from_list(ops_vec_tuple, allow_reordering);
}

void process_cre(const SQOperatorString& lhs, const SQOperatorString& rhs,
                 std::vector<std::pair<SQOperatorString, double>>& result, const double sign) {
    //    Left   |   Right
    // cre | ann | cre | ann |
    //  ^     ^     ^
    //  3     2     1
    //
    // 1. take the first creation operator on the right
    // 2. contract it with the corresponding annihilation operator on the left
    // 3. move the operator to the left creation part

    // find the first right alpha creation operator to move
    auto i = rhs.cre().find_first_one();
    // remove the operator from the right
    // if a corresponding left annihilation operator exists, permute the operators and
    // introduce a contraction
    if (lhs.ann().get_bit(i) == true) {
        auto new_lhs_ann = lhs.ann();
        new_lhs_ann.set_bit(i, false);
        const auto coefficient = lhs.ann().slater_sign(i);
        SQOperatorString new_lhs(lhs.cre(), new_lhs_ann);

        auto new_rhs_cre = rhs.cre();
        new_rhs_cre.set_bit(i, false);
        SQOperatorString new_rhs(new_rhs_cre, rhs.ann());

        generate_wick_contractions(new_lhs, new_rhs, result, sign * coefficient);
    }
    // if the left creation operator does not exist, move the operator in place
    // otherwise we get a collision and the operator is removed
    if (lhs.cre().get_bit(i) == false) {
        auto new_lhs_cre = lhs.cre();
        new_lhs_cre.set_bit(i, true);
        const auto coefficient =
            (lhs.count()) % 2 == 0 ? lhs.cre().slater_sign(i) : -lhs.cre().slater_sign(i);
        SQOperatorString new_lhs(new_lhs_cre, lhs.ann());

        auto new_rhs_cre = rhs.cre();
        new_rhs_cre.set_bit(i, false);
        SQOperatorString new_rhs(new_rhs_cre, rhs.ann());

        generate_wick_contractions(new_lhs, new_rhs, result, sign * coefficient);
    }
}

void process_ann(const SQOperatorString& lhs, const SQOperatorString& rhs,
                 std::vector<std::pair<SQOperatorString, double>>& result, const double sign) {
    // Here we assume that the operators are in the canonical form
    // and that the right alpha and beta creation have been alredy removed
    //    Left   |   Right
    // cre | ann | ann |
    //        ^     ^
    //        2     1
    //
    // 1. take the last annihilation operator on the right
    // 2. move the operator to the left annihilation part

    // find the last right annihilation operator to move
    auto i = rhs.ann().find_last_one();
    // if the left annihilation operator does not exist, move the operator in place
    // otherwise we get a collision and the operator is removed
    if (lhs.ann().get_bit(i) == false) {
        auto new_lhs_ann = lhs.ann();
        new_lhs_ann.set_bit(i, true);
        const auto coefficient = lhs.ann().slater_sign(i);
        SQOperatorString new_lhs(lhs.cre(), new_lhs_ann);

        auto new_rhs_ann = rhs.ann();
        new_rhs_ann.set_bit(i, false);
        SQOperatorString new_rhs(rhs.cre(), new_rhs_ann);

        generate_wick_contractions(new_lhs, new_rhs, result, sign * coefficient);
    }
}

void generate_wick_contractions(const SQOperatorString& lhs, const SQOperatorString& rhs,
                                std::vector<std::pair<SQOperatorString, double>>& result,
                                double sign) {
    // if there are no operators on the right then we return
    if (rhs.count() == 0) {
        result.push_back({lhs, sign});
        return;
    }
    if (rhs.cre().count() > 0) {
        process_cre(lhs, rhs, result, sign);
    } else if (rhs.ann().count() > 0) {
        process_ann(lhs, rhs, result, sign);
    }
}

std::vector<std::pair<SQOperatorString, double>> operator*(const SQOperatorString& lhs,
                                                           const SQOperatorString& rhs) {
    std::vector<std::pair<SQOperatorString, double>> result;
    generate_wick_contractions(lhs, rhs, result, 1.0);
    return result;
}

std::vector<std::pair<SQOperatorString, double>> commutator(const SQOperatorString& lhs,
                                                            const SQOperatorString& rhs) {
    const auto common_l_cre_r_cre = lhs.cre().fast_a_and_b_count(rhs.cre());
    const auto common_l_ann_r_ann = lhs.ann().fast_a_and_b_count(rhs.ann());
    const auto common_l_ann_r_cre = lhs.ann().fast_a_and_b_count(rhs.cre());
    const auto nl = lhs.count();
    const auto nr = rhs.count();
    // if the operators do not have any common creation or annihilation operators
    if (common_l_cre_r_cre == 0 and common_l_ann_r_ann == 0 and common_l_ann_r_cre == 0) {
        // if the number of operators is even, the commutator is zero
        if (nl * nr % 2 == 0) {
            return {};
        }
        // if the number of operators is odd, the commutator is the product of the operators
        // and we get a factor of two because both terms in the commutator are the same
        const auto prod = lhs * rhs;
        return {{prod[0].first, 2.0 * prod[0].second}};
    }

    const auto lr_prod = lhs * rhs;
    const auto rl_prod = rhs * lhs;

    // aggregate the terms
    std::unordered_map<SQOperatorString, double, SQOperatorString::Hash> result_map;
    for (const auto& [sqop_str, c] : lr_prod) {
        result_map[sqop_str] += c;
    }
    for (const auto& [sqop_str, c] : rl_prod) {
        result_map[sqop_str] -= c;
    }

    std::vector<std::pair<SQOperatorString, double>> result;
    // result.reserve(result_map.size());
    for (const auto& [sqop_str, c] : result_map) {
        if (c != 0.0) {
            result.emplace_back(sqop_str, c);
        }
    }
    return result;
}

} // namespace forte

// void process_alfa_cre(const SQOperator& lhs, const SQOperator& rhs,
//                       std::vector<SQOperator>& result) {
//     //              Left             |              Right
//     // a cre | b cre | b ann | a ann | a cre | b cre | b ann | a ann |
//     //    ^                       ^       ^
//     //    3                       2       1
//     //
//     // 1. take the first alpha creation operator on the right
//     // 2. contract it with the first alpha annihilation operator on the left
//     // 3. move the operator to the left

//     // find the first right alpha creation operator to move
//     auto i = rhs.cre().find_first_one_alfa();
//     // remove the operator from the right
//     // if a corresponding left alpha annihilation operator exists, permute the operators and
//     // introduce a contraction
//     if (lhs.ann().get_alfa_bit(i) == true) {
//         auto new_lhs_ann = lhs.ann();
//         new_lhs_ann.set_alfa_bit(i, false);
//         // compute the sign of the transposition
//         // count all the alpha annihilators on the left before the one we are creating
//         auto sign = lhs.ann().slater_sign_a(i);
//         auto coefficient = sign * lhs.coefficient();
//         SQOperator new_lhs(coefficient, lhs.cre(), new_lhs_ann);
//         auto new_rhs_cre = rhs.cre();
//         new_rhs_cre.set_alfa_bit(i, false);
//         SQOperator new_rhs(rhs.coefficient(), new_rhs_cre, rhs.ann());
//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
//     // if the left alpha creation operator does not exist, move the operator in place
//     // otherwise we get a collision and the operator is removed
//     if (lhs.cre().get_alfa_bit(i) == false) {
//         auto new_lhs_cre = lhs.cre();
//         new_lhs_cre.set_alfa_bit(i, true);
//         // compute the sign of the permutation to the very left
//         auto sign = (lhs.count()) % 2 == 0 ? 1.0 : -1.0;
//         // correct the sign for the permutation in place
//         sign *= lhs.cre().slater_sign_a(i);
//         auto coefficient = sign * lhs.coefficient();
//         SQOperator new_lhs(coefficient, new_lhs_cre, lhs.ann());
//         auto new_rhs_cre = rhs.cre();
//         new_rhs_cre.set_alfa_bit(i, false);
//         SQOperator new_rhs(rhs.coefficient(), new_rhs_cre, rhs.ann());
//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
// }

// void process_beta_cre(const SQOperator& lhs, const SQOperator& rhs,
//                       std::vector<SQOperator>& result) {
//     //              Left             |         Right
//     // a cre | b cre | b ann | a ann | b cre | b ann | a ann |
//     //           ^        ^               ^
//     //           3        2               1
//     //
//     // 1. take the first beta creation operator on the right
//     // 2. contract it with the first beta annihilation operator on the left
//     // 3. move the operator to the left

//     // find the first right beta creation operator to move
//     auto i = rhs.cre().find_first_one_beta();
//     // remove the operator from the right
//     // if a corresponding left beta annihilation operator exists, permute the operators and
//     // introduce a contraction
//     if (lhs.ann().get_beta_bit(i) == true) {
//         auto new_lhs_ann = lhs.ann();
//         new_lhs_ann.set_beta_bit(i, false);
//         // compute the sign of the transposition
//         auto sign = lhs.ann().slater_sign_b(i);
//         std::cout << "\nsign b c: " << sign << std::endl;
//         auto coefficient = sign * lhs.coefficient();
//         SQOperator new_lhs(coefficient, lhs.cre(), new_lhs_ann);
//         auto new_rhs_cre = rhs.cre();
//         new_rhs_cre.set_beta_bit(i, false);
//         SQOperator new_rhs(rhs.coefficient(), new_rhs_cre, rhs.ann());
//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
//     // if the left beta creation operator does not exist, move the operator in place
//     // otherwise we get a collision and the operator is removed
//     if (lhs.cre().get_beta_bit(i) == false) {
//         auto new_lhs_cre = lhs.cre();
//         new_lhs_cre.set_beta_bit(i, true);
//         // compute the sign of the permutation to the very left
//         auto sign = (lhs.count()) % 2 == 0 ? 1.0 : -1.0;
//         // correct the sign for the permutation in place
//         sign *= lhs.cre().slater_sign_b(i);
//         auto coefficient = sign * lhs.coefficient();
//         SQOperator new_lhs(coefficient, new_lhs_cre, lhs.ann());
//         auto new_rhs_cre = rhs.cre();
//         new_rhs_cre.set_beta_bit(i, false);
//         SQOperator new_rhs(rhs.coefficient(), new_rhs_cre, rhs.ann());
//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
// }

// void process_beta_ann(const SQOperator& lhs, const SQOperator& rhs,
//                       std::vector<SQOperator>& result) {
//     // Here we assume that the operators are in the canonical form
//     // and that the right alpha and beta creation have been alredy removed
//     //              Left             |     Right
//     // a cre | b cre | b ann | a ann | b ann | a ann |
//     //                    ^               ^
//     //                    2               1
//     //
//     // 1. take the last beta annihilation operator on the right
//     // 2. move the operator to the left

//     // find the last right beta annihilation operator to move
//     // (note that annihilation operators are in reversed order)
//     auto i = rhs.ann().find_last_one_beta();
//     // if the left beta annihilation operator does not exist, move the operator in place
//     // otherwise we get a collision and the operator is removed
//     if (lhs.ann().get_beta_bit(i) == false) {
//         auto new_lhs_ann = lhs.ann();
//         new_lhs_ann.set_beta_bit(i, true);
//         // correct the sign for the permutation in place
//         auto sign = lhs.ann().slater_sign_b(i);
//         auto coefficient = sign * lhs.coefficient();
//         SQOperator new_lhs(coefficient, lhs.cre(), new_lhs_ann);
//         auto new_rhs_ann = rhs.ann();
//         new_rhs_ann.set_beta_bit(i, false);
//         SQOperator new_rhs(rhs.coefficient(), rhs.cre(), new_rhs_ann);
//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
// }

// void process_alfa_ann(const SQOperator& lhs, const SQOperator& rhs,
//                       std::vector<SQOperator>& result) {
//     // Here we assume that the operators are in the canonical form
//     // and that the right alpha and beta creation have been alredy removed
//     //              Left             | Right
//     // a cre | b cre | b ann | a ann | a ann |
//     //                            ^       ^
//     //                            2       1
//     //
//     // 1. take the last alpha annihilation operator on the right
//     // 2. move the operator to the left

//     // find the last right alpha annihilation operator to move
//     // (note that annihilation operators are in reversed order)
//     auto i = rhs.ann().find_last_one_alfa();
//     // if the left alfa annihilation operator does not exist, move the operator in place
//     // otherwise we get a collision and the operator is removed
//     if (lhs.ann().get_alfa_bit(i) == false) {
//         auto new_lhs_ann = lhs.ann();
//         new_lhs_ann.set_alfa_bit(i, true);
//         // correct the sign for the permutation in place
//         auto sign = lhs.ann().slater_sign_a(i);
//         auto coefficient = sign * lhs.coefficient();
//         SQOperator new_lhs(coefficient, lhs.cre(), new_lhs_ann);
//         auto new_rhs_ann = rhs.ann();
//         new_rhs_ann.set_alfa_bit(i, false);
//         SQOperator new_rhs(rhs.coefficient(), rhs.cre(), new_rhs_ann);
//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
// }

// void process_cre(const SQOperator& lhs, const SQOperator& rhs, std::vector<SQOperator>& result) {
//     //    Left   |   Right
//     // cre | ann | cre | ann |
//     //  ^     ^     ^
//     //  3     2     1
//     //
//     // 1. take the first creation operator on the right
//     // 2. contract it with the corresponding annihilation operator on the left
//     // 3. move the operator to the left creation part

//     // find the first right alpha creation operator to move
//     auto i = rhs.cre().find_first_one();
//     // remove the operator from the right
//     // if a corresponding left annihilation operator exists, permute the operators and
//     // introduce a contraction
//     if (lhs.ann().get_bit(i) == true) {
//         auto new_lhs_ann = lhs.ann();
//         new_lhs_ann.set_bit(i, false);
//         const auto coefficient = lhs.ann().slater_sign(i) * lhs.coefficient();
//         SQOperatorString new_lhs_opstr(lhs.cre(), new_lhs_ann);
//         SQOperator new_lhs(coefficient, new_lhs_opstr);

//         auto new_rhs_cre = rhs.cre();
//         new_rhs_cre.set_bit(i, false);
//         SQOperatorString new_rhs_opstr(new_rhs_cre, rhs.ann());
//         SQOperator new_rhs(rhs.coefficient(), new_rhs_opstr);

//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
//     // if the left creation operator does not exist, move the operator in place
//     // otherwise we get a collision and the operator is removed
//     if (lhs.cre().get_bit(i) == false) {
//         auto new_lhs_cre = lhs.cre();
//         new_lhs_cre.set_bit(i, true);
//         const auto sign =
//             (lhs.count()) % 2 == 0 ? lhs.cre().slater_sign(i) : -lhs.cre().slater_sign(i);
//         const auto coefficient = sign * lhs.coefficient();
//         SQOperatorString new_lhs_opstr(new_lhs_cre, lhs.ann());
//         SQOperator new_lhs(coefficient, new_lhs_opstr);

//         auto new_rhs_cre = rhs.cre();
//         new_rhs_cre.set_bit(i, false);
//         SQOperatorString new_rhs_opstr(new_rhs_cre, rhs.ann());
//         SQOperator new_rhs(rhs.coefficient(), new_rhs_opstr);

//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
// }

// void process_ann(const SQOperator& lhs, const SQOperator& rhs, std::vector<SQOperator>& result) {
//     // Here we assume that the operators are in the canonical form
//     // and that the right alpha and beta creation have been alredy removed
//     //    Left   |   Right
//     // cre | ann | ann |
//     //        ^     ^
//     //        2     1
//     //
//     // 1. take the last annihilation operator on the right
//     // 2. move the operator to the left annihilation part

//     // find the last right annihilation operator to move
//     auto i = rhs.ann().find_last_one();
//     // if the left annihilation operator does not exist, move the operator in place
//     // otherwise we get a collision and the operator is removed
//     if (lhs.ann().get_bit(i) == false) {
//         auto new_lhs_ann = lhs.ann();
//         new_lhs_ann.set_bit(i, true);
//         const auto coefficient = lhs.ann().slater_sign(i) * lhs.coefficient();
//         SQOperatorString new_lhs_opstr(lhs.cre(), new_lhs_ann);
//         SQOperator new_lhs(coefficient, new_lhs_opstr);

//         auto new_rhs_ann = rhs.ann();
//         new_rhs_ann.set_bit(i, false);
//         SQOperatorString new_rhs_opstr(rhs.cre(), new_rhs_ann);
//         SQOperator new_rhs(rhs.coefficient(), new_rhs_opstr);

//         generate_wick_contractions(new_lhs, new_rhs, result);
//     }
// }
