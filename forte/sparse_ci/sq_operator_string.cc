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
#include <regex>

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

SQOperatorString::SQOperatorString(const std::initializer_list<size_t> acre,
                                   const std::initializer_list<size_t> bcre,
                                   const std::initializer_list<size_t> aann,
                                   const std::initializer_list<size_t> bann) {
    cre_ = Determinant(acre, bcre);
    ann_ = Determinant(aann, bann);
}

const Determinant& SQOperatorString::cre() const { return cre_; }

const Determinant& SQOperatorString::ann() const { return ann_; }

Determinant& SQOperatorString::cre_mod() { return cre_; }

Determinant& SQOperatorString::ann_mod() { return ann_; }

bool SQOperatorString::is_identity() const { return (cre().count() == 0) and (ann().count() == 0); }

bool SQOperatorString::is_nilpotent() const {
    // here we test that op != op^dagger
    return (cre() != ann());
}

bool SQOperatorString::is_self_adjoint() const { return this->cre() == this->ann(); }

SQOperatorString SQOperatorString::number_component() const {
    const Determinant number = this->cre() & this->ann();
    return SQOperatorString(number, number);
}

SQOperatorString SQOperatorString::non_number_component() const {
    return SQOperatorString(this->cre() - this->ann(), this->ann() - this->cre());
}

int SQOperatorString::count() const { return cre().count_all() + ann().count_all(); }

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

SQOperatorString SQOperatorString::spin_flip() const {
    return SQOperatorString(cre().spin_flip(), ann().spin_flip());
}

std::string SQOperatorString::str() const {
    auto acre = cre().get_alfa_occ(cre().norb());
    auto bcre = cre().get_beta_occ(cre().norb());
    auto aann = ann().get_alfa_occ(ann().norb());
    auto bann = ann().get_beta_occ(ann().norb());
    std::reverse(aann.begin(), aann.end());
    std::reverse(bann.begin(), bann.end());
    std::vector<std::string> terms;
    for (auto p : acre) {
        terms.push_back(std::to_string(p) + "a+");
    }
    for (auto p : bcre) {
        terms.push_back(std::to_string(p) + "b+");
    }
    for (auto p : bann) {
        terms.push_back(std::to_string(p) + "b-");
    }
    for (auto p : aann) {
        terms.push_back(std::to_string(p) + "a-");
    }
    std::string s = "[" + join(terms, " ") + "]";
    return s;
}

op_tuple_t SQOperatorString::op_tuple() const {
    auto acre = cre().get_alfa_occ(cre().norb());
    auto bcre = cre().get_beta_occ(cre().norb());
    auto aann = ann().get_alfa_occ(ann().norb());
    auto bann = ann().get_beta_occ(ann().norb());
    std::reverse(aann.begin(), aann.end());
    std::reverse(bann.begin(), bann.end());
    op_tuple_t terms;
    for (auto p : acre) {
        terms.emplace_back(true, true, p);
    }
    for (auto p : bcre) {
        terms.emplace_back(true, false, p);
    }
    for (auto p : bann) {
        terms.emplace_back(false, false, p);
    }
    for (auto p : aann) {
        terms.emplace_back(false, true, p);
    }
    return terms;
}

// implement the << operator for SQOperatorString
std::ostream& operator<<(std::ostream& os, const SQOperatorString& sqop) {
    os << sqop.str();
    return os;
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

std::string SQOperatorString::latex_compact() const {
    auto acre = cre().get_alfa_occ(cre().norb());
    auto bcre = cre().get_beta_occ(cre().norb());
    auto aann = ann().get_alfa_occ(ann().norb());
    auto bann = ann().get_beta_occ(ann().norb());
    std::string s;
    s += "\\hat{a}^{";
    std::vector<std::string> terms;
    for (auto p : acre) {
        terms.push_back(std::to_string(p) + "_{\\alpha}");
    }
    for (auto p : bcre) {
        terms.push_back(std::to_string(p) + "_{\\beta}");
    }
    s += join(terms, " ");
    s += "}_{";
    terms.clear();
    for (auto p : aann) {
        terms.push_back(std::to_string(p) + "_{\\alpha}");
    }
    for (auto p : bann) {
        terms.push_back(std::to_string(p) + "_{\\beta}");
    }
    s += join(terms, " ");
    s += "}";
    return s;
}

std::vector<std::tuple<bool, bool, int>> parse_sq_operator(const std::string& s) {
    // the regex to verify the validity of the string
    std::regex validity(R"(\[?(\d+[ab][\+\-]\s*)*\]?)");
    std::smatch m;
    if (not std::regex_match(s, m, validity)) {
        std::string msg = "parse_sq_operator could not parse the string " + s;
        throw std::runtime_error(msg);
    }

    std::regex pattern(R"(\b(\d+)([ab])([\+\-]))");
    auto begin = std::sregex_iterator(s.begin(), s.end(), pattern);
    auto end = std::sregex_iterator();
    std::vector<std::tuple<bool, bool, int>> ops_vec_tuple;
    for (std::sregex_iterator i = begin; i != end; ++i) {
        std::smatch match = *i;
        int orb = std::stoi(match[1].str());               // Convert captured integer part to int
        bool alpha = match[2].str() == "a" ? true : false; // Capture 'a' or 'b'
        char creation = match[3].str() == "+" ? true : false; // Capture '+' or '-'
        ops_vec_tuple.push_back(std::make_tuple(creation, alpha, orb));
    }
    std::reverse(ops_vec_tuple.begin(), ops_vec_tuple.end());
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
            // print the operators in the error
            std::string ops_str;
            for (const auto& op : creation_alpha_orb_vec) {
                ops_str += std::to_string(std::get<2>(op)) + (std::get<1>(op) ? "a" : "b") +
                           (std::get<0>(op) ? "+" : "-") + " ";
            }
            throw std::runtime_error(
                "Trying to initialize a SQOperator object with a product of\n"
                "operators that are not arranged in the canonical form\n\n"
                "    a+_p1 a+_p2 ...  a+_P1 a+_P2 ...   ... a-_Q2 a-_Q1   ... a-_q2 a-_q1\n"
                "    alpha creation   beta creation    beta annihilation  alpha annihilation\n\n"
                "with indices sorted as\n\n"
                "    (p1 < p2 < ...) (P1 < P2 < ...)  (... > Q2 > Q1) (... > q2 > q1)\n"
                "The operators are: " +
                ops_str);
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

CommutatorType commutator_type(const SQOperatorString& lhs, const SQOperatorString& rhs) {
    // Find the number of operators in common between the two operator strings
    const auto common_l_ann_r_cre = lhs.ann().fast_a_and_b_count(rhs.cre());
    const auto common_l_cre_r_cre = lhs.cre().fast_a_and_b_count(rhs.cre());
    const auto common_l_ann_r_ann = lhs.ann().fast_a_and_b_count(rhs.ann());
    const auto common_l_cre_r_ann = lhs.cre().fast_a_and_b_count(rhs.ann());
    // if there are no indices in common
    if (common_l_cre_r_cre == 0 and common_l_ann_r_ann == 0 and common_l_ann_r_cre == 0 and
        common_l_cre_r_ann == 0) {
        const auto nl = lhs.count();
        const auto nr = rhs.count();
        // even number of operator permutations
        if ((nl * nr) % 2 == 0) {
            return CommutatorType::Commute;
        }
        // odd number of operator permutations
        return CommutatorType::AntiCommute;
    }
    return CommutatorType::MayNotCommute;
}

bool do_ops_commute(const SQOperatorString& lhs, const SQOperatorString& rhs) {
    const auto common_l_cre_r_cre = lhs.cre().fast_a_and_b_count(rhs.cre());
    const auto common_l_cre_r_ann = lhs.cre().fast_a_and_b_count(rhs.ann());
    const auto common_l_ann_r_ann = lhs.ann().fast_a_and_b_count(rhs.ann());
    const auto common_l_ann_r_cre = lhs.ann().fast_a_and_b_count(rhs.cre());
    if (common_l_cre_r_cre == 0 and common_l_ann_r_ann == 0 and common_l_ann_r_cre == 0 and
        common_l_cre_r_ann == 0) {
        // if the number of operators is even, the commutator is zero
        const auto nl = lhs.count();
        const auto nr = rhs.count();
        // if the operators do not have any common creation or annihilation operators
        if ((nl * nr) % 2 == 0) {
            return true;
        }
        return false;
    }
    return false;
}

std::vector<std::pair<SQOperatorString, double>> commutator_fast(const SQOperatorString& lhs,
                                                                 const SQOperatorString& rhs) {
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

std::vector<std::pair<SQOperatorString, double>> commutator(const SQOperatorString& lhs,
                                                            const SQOperatorString& rhs) {

    // const auto common_l_cre_r_cre = lhs.cre().fast_a_and_b_count(rhs.cre());
    // const auto common_l_cre_r_ann = lhs.cre().fast_a_and_b_count(rhs.ann());
    // const auto common_l_ann_r_ann = lhs.ann().fast_a_and_b_count(rhs.ann());
    // const auto common_l_ann_r_cre = lhs.ann().fast_a_and_b_count(rhs.cre());
    // const auto nl = lhs.count();
    // const auto nr = rhs.count();

    // if the operators do not have any common creation or annihilation operators
    // if (common_l_cre_r_cre == 0 and common_l_ann_r_ann == 0 and common_l_ann_r_cre == 0 and
    // common_l_cre_r_ann == 0) {
    // if (common_l_ann_r_cre == 0 and common_l_cre_r_ann == 0) {
    //     // std::cout << "\nno common operators" << std::endl;

    //     // std::cout << "rhs: " << rhs.str() << std::endl;
    //     // std::cout << "lhs: " << lhs.str() << std::endl;
    //     // std::cout << "common_l_cre_r_cre: " << common_l_cre_r_cre << std::endl;
    //     // std::cout << "common_l_ann_r_ann: " << common_l_ann_r_ann << std::endl;
    //     // std::cout << "common_l_ann_r_cre: " << common_l_ann_r_cre << std::endl;

    //     // if the number of operators is even, the commutator is zero
    //     if ((nl * nr) % 2 == 0) {
    //         return {};
    //     }
    //     // if the number of operators is odd, the commutator is the product of the operators
    //     // and we get a factor of two because both terms in the commutator are the same
    //     const auto prod = lhs * rhs;
    //     return {{prod[0].first, 2.0 * prod[0].second}};
    // }

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
    // if (is_zero) {
    //     std::cout << "\nrhs: " << rhs.str() << std::endl;
    //     std::cout << "lhs: " << lhs.str() << std::endl;
    //     std::cout << "common_l_cre_r_cre: " << common_l_cre_r_cre << std::endl;
    //     std::cout << "common_l_cre_r_ann: " << common_l_cre_r_ann << std::endl;
    //     std::cout << "common_l_ann_r_ann: " << common_l_ann_r_ann << std::endl;
    //     std::cout << "common_l_ann_r_cre: " << common_l_ann_r_cre << std::endl;
    //     std::cout << "nl: " << nl << std::endl;
    //     std::cout << "nr: " << nr << std::endl;
    //     std::cout << "common_l_cre_r_cre + common_l_ann_r_ann - nl: "
    //               << common_l_cre_r_cre + common_l_ann_r_ann - nl << ", nl - nr: " << nl - nr
    //               << std::endl;
    // }

    return result;
}

#define debug_print(x) ; // std::cout << #x << ": " << x << std::endl;

void SQOperatorProductComputer::product(
    const SQOperatorString& lhs, const SQOperatorString& rhs, sparse_scalar_t factor,
    std::function<void(const SQOperatorString&, const sparse_scalar_t)> func) {

    // apologies to those who read this code, it's meant to be fast, not readable.

    // determine the right creation operators that can be moved to the left without contraction
    ucon_rhs_cre_ = rhs.cre() - lhs.ann();
    // if there are common lhs creation ops and uncontracted rhs creation ops then we get zero
    if (not lhs.cre().fast_a_and_b_eq_zero(ucon_rhs_cre_)) {
        return;
    }
    // find the right creation ops that will need to be contracted with the left annihilation ops
    con_rhs_cre_ = rhs.cre() - ucon_rhs_cre_;
    // determine the right annihilation ops that can be moved to the left without contracting them
    // with the rhs creation ops
    ucon_rhs_ann_ = rhs.ann() - con_rhs_cre_;
    // if there are common lhs annihilation ops and uncontracted rhs annihilation ops then we get
    // zero, so we return
    if (not lhs.ann().fast_a_and_b_eq_zero(ucon_rhs_ann_)) {
        return;
    }

    // initialize the phase and the bitarray operators
    phase_ = factor;
    rhs_cre_ = rhs.cre();
    rhs_ann_ = rhs.ann();
    lhs_cre_ = lhs.cre();
    lhs_ann_ = lhs.ann();

    // Step 1. Move the uncontracted rhs creation operators to the left
    // 1.a phase adjustment due to permutation of the operator with left annihilation ops
    if (const auto ucon_rhs_cre_count = ucon_rhs_cre_.count_all(); ucon_rhs_cre_count > 0) {
        phase_ *= ((lhs_ann_.count_all() * ucon_rhs_cre_.count_all()) % 2) == 0 ? 1.0 : -1.0;
        // 1.b move the uncontracted rhs creation operators to the left creation ops
        // double cre_perm_phase = (lhs_cre_.count_all() % 2) == 0 ? 1.0 : -1.0;
        for (size_t i = ucon_rhs_cre_.fast_find_and_clear_first_one(0); i != ~0ULL;
             i = ucon_rhs_cre_.fast_find_and_clear_first_one(i)) {
            // remove op i and find the sign for permuting it to the left of the right creation ops
            rhs_cre_.set_bit(i, false);
            phase_ *= rhs_cre_.slater_sign(i);
            // add the op to the left and find the computing the phase
            lhs_cre_.set_bit(i, true);
            phase_ *= lhs_cre_.slater_sign_reverse(i); // * cre_perm_phase;
        }
    }

    // Step 2. Move the uncontracted rhs annihilation operators to the left
    // 2.a phase adjustment due to permutation of the operator with right creation ops
    if (const auto ucon_rhs_ann_count = ucon_rhs_ann_.count_all(); ucon_rhs_ann_count > 0) {
        phase_ *= ((rhs_cre_.count_all() * ucon_rhs_ann_count) % 2) == 0 ? 1.0 : -1.0;
        for (size_t i = ucon_rhs_ann_.fast_find_and_clear_first_one(0); i != ~0ULL;
             i = ucon_rhs_ann_.fast_find_and_clear_first_one(i)) {
            // remove op i and find the sign for permuting it with the right creation ops
            rhs_ann_.set_bit(i, false);
            phase_ *= rhs_ann_.slater_sign_reverse(i);
            // add the op to the left computing the phase
            lhs_ann_.set_bit(i, true);
            phase_ *= lhs_ann_.slater_sign(i);
        }
    }

    // Step 3. Find the number component operators on the right that can be treated trivially
    // E.g.   ([1-]) ([1+ 1-]) = ([1-]) (1 - [1-1+]) = [1-] - [1-1-1+] = [1-]
    // find the operators in common that can be trivially contracted
    auto rhs_comm_trivial_ = rhs_cre_ & rhs_ann_ & lhs_ann_;
    if (rhs_comm_trivial_.count_all() != 0) {
        // remove the trivially contracted operators from the right creation ops
        rhs_cre_ -= rhs_comm_trivial_;
        rhs_ann_ -= rhs_comm_trivial_; // remove the trivially contracted operators from the right
        ucon_rhs_cre_ = rhs_cre_;      // now this holds the operators that need to be contracted
        // adjust the phase due to the trivial contractions
        for (size_t i = rhs_comm_trivial_.fast_find_and_clear_first_one(0); i != ~0ULL;
             i = rhs_comm_trivial_.fast_find_and_clear_first_one(i)) {
            phase_ *= rhs_cre_.slater_sign_reverse(i) * rhs_ann_.slater_sign_reverse(i);
        }
    }

    // Step 4. Find the anti-number component operators on the right that can be treated trivially
    // E.g.   ([1+ 1-]) ([1+]) = (1 - [1- 1+]) [1+] = [1+] - [1- 1+ 1+] = [1+]
    // find the operators in common that can be trivially contracted
    auto lhs_comm_trivial_ = lhs_cre_ & lhs_ann_ & rhs_cre_;
    if (lhs_comm_trivial_.count_all() != 0) {
        // remove the trivially contracted operators from the right creation/left annihilation ops
        rhs_cre_ -= lhs_comm_trivial_;
        lhs_ann_ -= lhs_comm_trivial_;
        // ops adjust the phase due to the trivial contractions
        for (size_t i = lhs_comm_trivial_.fast_find_and_clear_first_one(0); i != ~0ULL;
             i = lhs_comm_trivial_.fast_find_and_clear_first_one(i)) {
            phase_ *= lhs_ann_.slater_sign(i) * rhs_cre_.slater_sign(i);
        }
    }

    // At this point we have moved all the uncontracted rhs creation and annihilation operators
    // to the left and we can now compute the product of the operators that don't commute.
    // These are the left annihilations-right creations
    // One thing we know is that each of these will give us 2^n terms where n is the number of
    // operators that can be contracted.
    auto ncontr = rhs_cre_.count_all();
    if (ncontr == 0) {
        func(SQOperatorString(lhs_cre_, lhs_ann_), phase_);
        return;
    }

    // loop over the right creation operators that can be contracted and compute the phase
    // adjustment to move them to the left and form pairs with the left annihilation operators
    ucon_rhs_cre_ = rhs_cre_; // now this holds the operators that need to be contracted
    // NOTE: this was not checked yet
    for (size_t i = ucon_rhs_cre_.fast_find_and_clear_first_one(0); i != ~0ULL;
         i = ucon_rhs_cre_.fast_find_and_clear_first_one(i)) {
        phase_ *= lhs_ann_.slater_sign(i) * rhs_cre_.slater_sign(i);
    }
    // find the set bits of the operators that can be contracted and store it in set_bits_
    rhs_cre_.find_set_bits(set_bits_, ncontr);
    // now this holds the left annihilation ops not paired with creation ops
    rhs_cre_ = lhs_ann_ - rhs_cre_;
    // precompute signs (store the sign as a 0 or 1 in a temporary array)
    for (size_t i = 0; i < ncontr; i++) {
        sign_[i] = (rhs_cre_.slater_sign_reverse(set_bits_[i]) *
                        lhs_cre_.slater_sign_reverse(set_bits_[i]) >
                    0.0)
                       ? 0
                       : 1;
    }
    for (size_t i = 0; i < (1ULL << ncontr); i++) {
        double contraction_phase = 1.0;
        auto new_lhs_cre = lhs_cre_;
        auto new_lhs_ann = lhs_ann_;
        for (size_t j = 0; j < ncontr; j++) {
            if ((i >> j) & 1) { // if bit j of i is set
                // this is the swapped operator term
                if (sign_[j]) {
                    contraction_phase *= -1.0;
                }
                new_lhs_cre.set_bit(set_bits_[j], true);
                new_lhs_ann.set_bit(set_bits_[j], true);
                contraction_phase *= -1.0;
            } else {
                // this is the identity term
                new_lhs_ann.set_bit(set_bits_[j], false);
            }
        }
        func(SQOperatorString(new_lhs_cre, new_lhs_ann), phase_ * contraction_phase);
    }
}

void SQOperatorProductComputer::commutator(
    const SQOperatorString& lhs, const SQOperatorString& rhs, sparse_scalar_t factor,
    std::function<void(const SQOperatorString&, const sparse_scalar_t)> func) {
    product(lhs, rhs, factor, func);
    product(rhs, lhs, -factor, func);
}

Determinant compute_sign_mask(const Determinant& cre, const Determinant& ann) {
    Determinant sign_mask;
    Determinant idx(ann); // temp is for looping over the operators
    Determinant temp;
    for (size_t i = idx.fast_find_and_clear_first_one(0); i != ~0ULL;
         i = idx.fast_find_and_clear_first_one(i)) {
        temp.fill_up_to(i);
        sign_mask ^= temp;
    }
    idx = cre;
    for (size_t i = idx.fast_find_and_clear_first_one(0); i != ~0ULL;
         i = idx.fast_find_and_clear_first_one(i)) {
        temp.fill_up_to(i);
        sign_mask ^= temp;
    }
    return sign_mask;
}

void compute_sign_mask(const Determinant& cre, const Determinant& ann, Determinant& sign_mask,
                       Determinant& idx) {
    sign_mask.zero();
    idx = ann; // temp is for looping over the operators
    for (size_t i = idx.fast_find_and_clear_first_one(0); i != ~0ULL;
         i = idx.fast_find_and_clear_first_one(i)) {
        sign_mask.xor_up_to(i);
    }
    idx = cre;
    for (size_t i = idx.fast_find_and_clear_first_one(0); i != ~0ULL;
         i = idx.fast_find_and_clear_first_one(i)) {
        sign_mask.xor_up_to(i);
    }
}

} // namespace forte
