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

void sim_trans_exc_impl(SparseOperator& O, const SQOperatorString& T_op, double theta,
                        double screen_threshold);

void sim_trans_antiherm_impl(SparseOperator& O, const SQOperatorString& T_op, double theta,
                             double screen_threshold);

SparseOperator::SparseOperator(
    const std::unordered_map<SQOperatorString, double, SQOperatorString::Hash>& op_map)
    : op_map_(op_map) {
    // populate the insertion list in the order of the map since we don't have a specific order
    for (const auto& [sqop, _] : op_map_) {
        op_insertion_list_.push_back(sqop);
    }
}

void SparseOperator::add_term(const std::vector<std::tuple<bool, bool, int>>& op_list,
                              double coefficient, bool allow_reordering) {
    auto [sqop, phase] = make_sq_operator_string_from_list(op_list, allow_reordering);
    // add the term only if the operator is not antihermitian or if the operator is
    // antihermitian and the term is antihermitian compatible
    add_term(sqop, phase * coefficient);
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

void SparseOperator::copy(const SparseOperator& other) { *this = other; }

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
            auto [sqop, phase] = make_sq_operator_string(m[1], allow_reordering);
            add_term(sqop, phase * coefficient);
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
    for (const auto& sqop : op_insertion_list_) {
        v.push_back(op_map_.at(sqop));
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
        const auto& last_sqop = op_insertion_list_.back();
        op_map_.erase(last_sqop);
        op_insertion_list_.pop_back();
    }
}

std::string format_term_in_sum(double coefficient, const std::string& term) {
    if (term == "[ ]") {
        if (coefficient == 0.0) {
            return "";
        } else {
            return (coefficient > 0.0 ? "+ " : "") + to_string_with_precision(coefficient, 12);
        }
    }
    if (coefficient == 0.0) {
        return "";
    } else if (coefficient == 1.0) {
        return "+ " + term;
    } else if (coefficient == -1.0) {
        return "- " + term;
    } else if (coefficient == static_cast<int>(coefficient)) {
        return to_string_with_precision(coefficient, 12) + " * " + term;
    } else {
        std::string s = to_string_with_precision(coefficient, 12);
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        return s + " * " + term;
    }
    return "";
}
std::vector<std::string> SparseOperator::str() const {
    std::vector<std::string> v;
    for (const auto& sqop : op_insertion_list_) {
        double coefficient = op_map_.at(sqop);
        if (std::fabs(coefficient) < 1.0e-12)
            continue;
        v.push_back(format_term_in_sum(coefficient, sqop.str()));
    }
    return v;
}

std::string SparseOperator::latex() const {
    std::vector<std::string> v;
    for (const auto& sqop : op_insertion_list_) {
        double coefficient = op_map_.at(sqop);
        const std::string s = double_to_string_latex(coefficient) + "\\;" + sqop.latex();
        v.push_back(s);
    }
    return join(v, " ");
}

SparseOperator SparseOperator::adjoint() const {
    SparseOperator adjoint_operator;
    for (const auto& sqop : op_insertion_list_) {
        // Retrieve the original coefficient
        auto coefficient = op_map_.at(sqop);

        adjoint_operator.add_term(sqop.adjoint(), coefficient);
    }
    return adjoint_operator;
}

SparseOperator SparseOperator::operator*(double scalar) const {
    SparseOperator result;
    for (const auto& [sqop, c] : op_map()) {
        result.add_term(sqop, scalar * c);
    }
    return result;
}

SparseOperator SparseOperator::operator/(double scalar) const { return *this * (1.0 / scalar); }

SparseOperator& SparseOperator::operator+=(const SparseOperator& other) {
    for (const auto& [sqop, c] : other.op_map()) {
        this->add_term(sqop, c);
    }
    return *this;
}

SparseOperator& SparseOperator::operator-=(const SparseOperator& other) {
    for (const auto& [sqop, c] : other.op_map()) {
        this->add_term(sqop, -c);
    }
    return *this;
}

SparseOperator& SparseOperator::operator*=(double factor) {
    for (auto& [_, c] : op_map_) {
        c *= factor;
    }
    return *this;
}

SparseOperator& SparseOperator::operator/=(double factor) {
    *this *= 1.0 / factor;
    return *this;
}

bool SparseOperator::operator==(const SparseOperator& other) const {
    // Check if the two operators have the number of terms
    if (op_insertion_list_.size() != other.op_insertion_list_.size()) {
        return false;
    }
    // Check if the two operators have the same terms
    for (const auto& sqop : op_insertion_list_) {
        if (other.op_map_.find(sqop) == other.op_map_.end()) {
            return false;
        }
        if (std::fabs(op_map_.at(sqop) - other.op_map_.at(sqop)) > 1.0e-14) {
            return false;
        }
    }
    return true;
}

SparseOperator operator+(SparseOperator lhs, const SparseOperator& rhs) {
    lhs += rhs;
    return lhs;
}

SparseOperator operator-(SparseOperator lhs, const SparseOperator& rhs) {
    lhs -= rhs;
    return lhs;
}

SparseOperator operator*(double scalar, const SparseOperator& op) {
    return op * scalar; // Reuse the member operator* for scalar multiplication
}

SparseOperator operator/(double scalar, const SparseOperator& op) {
    return op * (1.0 / scalar); // Reuse the member operator* for scalar multiplication
}

double norm(const SparseOperator& op) {
    double norm = 0.0;
    for (const auto& [_, c] : op.op_map()) {
        norm += c * c;
    }
    return std::sqrt(norm);
}

SparseOperator operator*(const SparseOperator& lhs, const SparseOperator& rhs) {
    std::unordered_map<SQOperatorString, double, SQOperatorString::Hash> result_map;
    for (const auto& [sqop_lhs, c_lhs] : lhs.op_map()) {
        for (const auto& [sqop_rhs, c_rhs] : rhs.op_map()) {
            const auto prod = sqop_lhs * sqop_rhs;
            for (const auto& [sqop, c] : prod) {
                if (c * c_lhs * c_rhs != 0.0) {
                    result_map[sqop] += c * c_lhs * c_rhs;
                }
            }
        }
    }
    return SparseOperator(result_map);
}

SparseOperator commutator(const SparseOperator& lhs, const SparseOperator& rhs) {
    std::unordered_map<SQOperatorString, double, SQOperatorString::Hash> result_map;
    // this works when one or none of the operators is antihermitian
    for (const auto& [sqop_lhs, c_lhs] : lhs.op_map()) {
        for (const auto& [sqop_rhs, c_rhs] : rhs.op_map()) {
            const auto prod = commutator(sqop_lhs, sqop_rhs);
            for (const auto& [sqop, c] : prod) {
                if (c * c_lhs * c_rhs != 0.0) {
                    result_map[sqop] += c * c_lhs * c_rhs;
                }
            }
        }
    }
    return SparseOperator(result_map);
}

void similarity_transform_test(SparseOperator& op, const SQOperatorString& sqop, double theta) {
    SparseOperator A;
    SparseOperator T;
    SparseOperator Td;
    A.add_term(sqop, 1.0);
    A.add_term(sqop.adjoint(), -1.0);
    T.add_term(sqop, 1.0);
    Td.add_term(sqop.adjoint(), 1.0);
    auto cOA = commutator(op, A);
    auto cOT = commutator(op, T);
    auto cOTd = commutator(op, Td);
    auto cOAA = commutator(cOA, A);
    auto N = -1.0 * A * A;
    auto aNO = N * op + op * N;
    auto aNcOA = N * cOA + cOA * N;
    auto NON = N * op * N;
    auto AOA = A * op * A;
    auto NOA_AON = N * op * A - A * op * N;
    auto TOT = T * op * T;
    auto TdOTd = Td * op * Td;

    auto Td_cOT_T = Td * cOT * T;
    auto Td_cOT_Td = Td * cOT * Td;
    auto T_cOT_Td = T * cOT * Td;

    auto T_cOTd_T = T * cOTd * T;
    auto T_cOTd_Td = T * cOTd * Td;
    auto Td_cOTd_T = Td * cOTd * T;

    op += std::sin(theta) * cOA;
    op += -std::pow(std::sin(theta), 2.0) * AOA;
    op += (std::cos(theta) - 1) * aNO;
    op += std::pow(std::cos(theta) - 1, 2.0) * NON;

    // op += std::sin(theta) * (std::cos(theta) - 1) * NOA_AON;
    op += -std::sin(theta) * (std::cos(theta) - 1) * Td_cOT_T;
    op += -std::sin(theta) * (std::cos(theta) - 1) * T_cOT_Td;
    op += +std::sin(theta) * (std::cos(theta) - 1) * Td_cOT_Td;
    op += -std::sin(theta) * (std::cos(theta) - 1) * T_cOTd_T;
    op += +std::sin(theta) * (std::cos(theta) - 1) * Td_cOTd_T;
    op += +std::sin(theta) * (std::cos(theta) - 1) * T_cOTd_Td;
}

void sim_trans_exc(SparseOperator& O, const SparseOperator& T, bool reverse,
                   double screen_threshold) {
    for (size_t m = 0, mmax = T.size(); m < mmax; ++m) {
        auto n = reverse ? mmax - m - 1 : m;
        sim_trans_exc_impl(O, T.term_operator(n), T.coefficient(n), screen_threshold);
    }
}

void sim_trans_exc_impl(SparseOperator& O, const SQOperatorString& T_op, double theta,
                        double screen_threshold) {
    // sanity check to make sure all indices are distinct
    if (not T_op.cre().fast_a_and_b_eq_zero(T_op.ann())) {
        throw std::runtime_error("similarity_transform_fast: the operator " + T_op.str() +
                                 " contains repeated indices.\nThis is not allowed for the "
                                 "similarity transformation.");
    }
    // (1 - T) O(1 + T) = O + [ O, T ] - T O T
    SparseOperator T;
    for (const auto& [O_op, O_c] : O.op_map()) {
        if (std::fabs(O_c * theta) < screen_threshold) {
            continue;
        }
        const bool do_O_T_commute = do_ops_commute(O_op, T_op);
        // if the commutator is zero, then we can skip this term
        if (do_O_T_commute) {
            continue;
        }
        auto cOT = commutator_fast(O_op, T_op);
        for (const auto& [cOT_op, cOT_c] : cOT) {
            // + [O, T]
            if (std::fabs(theta * cOT_c * O_c) > screen_threshold) {
                T.add_term(cOT_op, theta * cOT_c * O_c);
            }
            // - T [O,T]
            auto T_cOT = T_op * cOT_op;
            for (const auto& [T_cOT_op, T_cOT_c] : T_cOT) {
                if (std::fabs(theta * theta * T_cOT_c * cOT_c * O_c) > screen_threshold) {
                    T.add_term(T_cOT_op, -theta * theta * T_cOT_c * cOT_c * O_c);
                }
            }
        }
    }
    O += T;
}

void sim_trans_antiherm(SparseOperator& O, const SparseOperator& T, bool reverse,
                        double screen_threshold) {
    for (size_t m = 0, mmax = T.size(); m < mmax; ++m) {
        auto n = reverse ? mmax - m - 1 : m;
        sim_trans_antiherm_impl(O, T.term_operator(n), T.coefficient(n), screen_threshold);
    }
}

void sim_trans_antiherm_impl(SparseOperator& O, const SQOperatorString& T_op, double theta,
                             double screen_threshold) {
    // sanity check to make sure all indices are distinct
    if (not T_op.cre().fast_a_and_b_eq_zero(T_op.ann())) {
        throw std::runtime_error("similarity_transform_fast: the operator " + T_op.str() +
                                 " contains repeated indices.\nThis is not allowed for the "
                                 "similarity transformation.");
    }
    // Td = T^dagger
    auto Td_op = T_op.adjoint();
    // TTd = T T^dagger (gives a number operator if there are no repeated indices)
    auto TTd = T_op * Td_op;
    // TdT = T^dagger T (gives a number operator if there are no repeated indices)
    auto TdT = Td_op * T_op;

    SparseOperator T;
    auto sin_theta = std::sin(theta);
    auto sin_theta_2 = std::pow(std::sin(theta), 2.0);
    auto sin_theta_half_4 = std::pow(std::sin(0.5 * theta), 4.0);
    auto cos_theta_minus_1_2 = std::pow(std::cos(theta) - 1.0, 2.0);
    auto sin_theta_cos_theta_minus_1 = std::sin(theta) * (std::cos(theta) - 1.0);

    for (const auto& [O_op, O_c] : O.op_map()) {
        const bool do_O_T_commute = do_ops_commute(O_op, T_op);
        const bool do_O_Td_commute = do_ops_commute(O_op, Td_op);
        // if both commutators are zero, then we can skip this term
        if (do_O_T_commute and do_O_Td_commute) {
            continue;
        }
        if (not do_O_T_commute) {
            // [O, T]
            auto cOT = commutator_fast(O_op, T_op);
            for (const auto& [cOT_op, cOT_c] : cOT) {
                // sin(theta) [O, T]
                T.add_term(cOT_op, sin_theta * cOT_c * O_c);
                // -sin(theta)^2 T [O,T]
                auto T_cOT = T_op * cOT_op;
                for (const auto& [T_cOT_op, T_cOT_c] : T_cOT) {
                    T.add_term(T_cOT_op, -sin_theta_2 * T_cOT_c * cOT_c * O_c);
                }
                // +1/2 sin(theta)^2 [T^dagger, [O,T]]
                auto cTdcOT = commutator_fast(Td_op, cOT_op);
                for (const auto& [cTdcOT_op, cTdcOT_c] : cTdcOT) {
                    T.add_term(cTdcOT_op, 0.5 * sin_theta_2 * cTdcOT_c * cOT_c * O_c);
                }

                auto TdcOT = Td_op * cOT_op;
                for (const auto& [TdcOT_op, TdcOT_c] : TdcOT) {
                    // -sin(theta) (cos(theta) - 1) T^dagger [O,T] T
                    auto TdcOTT = TdcOT_op * T_op;
                    for (const auto& [TdcOTT_op, TdcOTT_c] : TdcOTT) {
                        T.add_term(TdcOTT_op,
                                   -sin_theta_cos_theta_minus_1 * TdcOTT_c * TdcOT_c * cOT_c * O_c);
                    }
                    // sin(theta) (cos(theta) - 1) T^dagger [O,T] T^dagger
                    auto TdcOTd = TdcOT_op * Td_op;
                    for (const auto& [TdcOTTd_op, TdcOTTd_c] : TdcOTd) {
                        T.add_term(TdcOTTd_op,
                                   sin_theta_cos_theta_minus_1 * TdcOTTd_c * TdcOT_c * cOT_c * O_c);
                    }
                }
                // -sin(theta) (cos(theta) - 1) T [O,T] T^dagger
                for (const auto& [TcOT_op, TcOT_c] : T_cOT) {
                    auto TcOTTd = TcOT_op * Td_op;
                    for (const auto& [TcOTTd_op, TcOTTd_c] : TcOTTd) {
                        T.add_term(TcOTTd_op,
                                   -sin_theta_cos_theta_minus_1 * TcOTTd_c * TcOT_c * cOT_c * O_c);
                    }
                }
            }
        }
        if (not do_O_Td_commute) {
            // [O, T^dagger]
            auto cOTd = commutator_fast(O_op, Td_op);
            for (const auto& [cOTd_op, cOTd_c] : cOTd) {
                // -sin(theta)[O, T^dagger]
                T.add_term(cOTd_op, -sin_theta * cOTd_c * O_c);
                // -sin(theta)^2 T^dagger [O,T^dagger]
                auto TdcOTd = Td_op * cOTd_op;
                for (const auto& [TdcOTd_op, TdcOTd_c] : TdcOTd) {
                    T.add_term(TdcOTd_op, -sin_theta_2 * TdcOTd_c * cOTd_c * O_c);
                }
                // +1/2 sin(theta)^2 [T, [O,T^dagger]]
                auto cTcOTd = commutator_fast(T_op, cOTd_op);
                for (const auto& [cTcOTd_op, cTcOTd_c] : cTcOTd) {
                    T.add_term(cTcOTd_op, 0.5 * sin_theta_2 * cTcOTd_c * cOTd_c * O_c);
                }

                auto TcOTd = T_op * cOTd_op;
                for (const auto& [TcOTd_op, TcOTd_c] : TcOTd) {
                    // -sin(theta) (cos(theta) - 1) T [O,T^dagger] T
                    auto TcOTdT = TcOTd_op * T_op;
                    for (const auto& [TcOTdT_op, TcOTdT_c] : TcOTdT) {
                        T.add_term(TcOTdT_op, -sin_theta_cos_theta_minus_1 * TcOTdT_c * TcOTd_c *
                                                  cOTd_c * O_c);
                    }
                    // +sin(theta) (cos(theta) - 1) T [O,T^dagger] T^dagger
                    auto TcOTdTd = TcOTd_op * Td_op;
                    for (const auto& [TcOTdTd_op, TcOTdTd_c] : TcOTdTd) {
                        T.add_term(TcOTdTd_op, sin_theta_cos_theta_minus_1 * TcOTdTd_c * TcOTd_c *
                                                   cOTd_c * O_c);
                    }
                }

                for (const auto& [TdcOTd_op, TdcOTd_c] : TdcOTd) {
                    // +sin(theta) (cos(theta) - 1) T^dagger [O,T^dagger] T
                    auto TdcOTdT = TdcOTd_op * T_op;
                    for (const auto& [TdcOTdT_op, TdcOTdT_c] : TdcOTdT) {
                        T.add_term(TdcOTdT_op, sin_theta_cos_theta_minus_1 * TdcOTdT_c * TdcOTd_c *
                                                   cOTd_c * O_c);
                    }
                }
            }
        }
        for (const auto& [TdT_op, TdT_c] : TdT) {
            // + 1/2 sin(theta)^2 T^dagger T O
            auto TdTO = TdT_op * O_op;
            for (const auto& [TdTO_op, TdTO_c] : TdTO) {
                T.add_term(TdTO_op, -2.0 * sin_theta_half_4 * TdTO_c * TdT_c * O_c);

                // + (cos(theta) - 1)^2 T^dagger T O T^dagger T
                for (const auto& [TdT_op2, TdT_c2] : TdT) {
                    auto TdTOTdT = TdTO_op * TdT_op2;
                    for (const auto& [TdTOTdT_op, TdTOTdT_c] : TdTOTdT) {
                        T.add_term(TdTOTdT_op,
                                   cos_theta_minus_1_2 * TdTOTdT_c * TdT_c2 * TdTO_c * TdT_c * O_c);
                    }
                }
                // + (cos(theta) - 1)^2 T^dagger T O T T^dagger
                for (const auto& [TTd_op, TTd_c] : TTd) {
                    auto TdTOTTd = TdTO_op * TTd_op;
                    for (const auto& [TdTOTTd_op, TdTOTTd_c] : TdTOTTd) {
                        T.add_term(TdTOTTd_op,
                                   cos_theta_minus_1_2 * TdTOTTd_c * TTd_c * TdTO_c * TdT_c * O_c);
                    }
                }
            }
            // + 1/2 sin(theta)^2 O T^dagger T
            auto OTdT = O_op * TdT_op;
            for (const auto& [OTdT_op, OTdT_c] : OTdT) {
                T.add_term(OTdT_op, -2.0 * sin_theta_half_4 * OTdT_c * TdT_c * O_c);
            }
        }
        for (const auto& [TTd_op, TTd_c] : TTd) {
            // sin(theta)^2 T T^dagger O
            auto TTdO = TTd_op * O_op;
            for (const auto& [TTdO_op, TTdO_c] : TTdO) {
                T.add_term(TTdO_op, -2.0 * sin_theta_half_4 * TTdO_c * TTd_c * O_c);
                // + (cos(theta) - 1)^2 T T^dagger O T^dagger T
                for (const auto& [TdT_op, TdT_c] : TdT) {
                    auto TTdOTdT = TTdO_op * TdT_op;
                    for (const auto& [TTdOTdT_op, TTdOTdT_c] : TTdOTdT) {
                        T.add_term(TTdOTdT_op,
                                   cos_theta_minus_1_2 * TTdOTdT_c * TdT_c * TTdO_c * TTd_c * O_c);
                    }
                }
                // + (cos(theta) - 1)^2 T T^dagger O T T^dagger
                for (const auto& [TTd_op2, TTd_c2] : TTd) {
                    auto TTdOTTd = TTdO_op * TTd_op2;
                    for (const auto& [TTdOTTd_op, TTdOTTd_c] : TTdOTTd) {
                        T.add_term(TTdOTTd_op,
                                   cos_theta_minus_1_2 * TTdOTTd_c * TTd_c2 * TTdO_c * TTd_c * O_c);
                    }
                }
            }
            // sin(theta)^2 O T T^dagger
            auto OTTd = O_op * TTd_op;
            for (const auto& [OTTd_op, OTTd_c] : OTTd) {
                T.add_term(OTTd_op, -2.0 * sin_theta_half_4 * OTTd_c * TTd_c * O_c);
            }
        }
    }
    O += T;
}

} // namespace forte
