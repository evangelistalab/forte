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

#include "helpers/combinatorial.h"
#include "helpers/timer.h"
#include "helpers/string_algorithms.h"

#include "sparse_operator.h"

namespace forte {

void sim_trans_op_impl(SparseOperator& O, const SQOperatorString& T_op, double theta,
                       double screen_threshold);

void sim_trans_antiherm_impl(SparseOperator& O, const SQOperatorString& T_op, double theta,
                             double screen_threshold);

void sim_trans_antiherm_impl_grad(SparseOperator& O, const SQOperatorString& T_op, double theta,
                                  double screen_threshold);

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
        const std::string s = double_to_string_latex(c) + "\\;" + sqop.latex();
        v.push_back(s);
    }
    // sort v to guarantee a consistent order
    std::sort(v.begin(), v.end());
    return join(v, " ");
}

void SparseOperator::add_term_from_str(const std::string& s, double coefficient,
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
                [&C](const SQOperatorString& sqop, const double c) { C.add(sqop, c); });
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
                [&C](const SQOperatorString& sqop, const double c) { C[sqop] += c; });
        }
    }
    return C;
}

void sim_trans_fact_op(SparseOperator& O, const SparseOperatorList& T, bool reverse,
                       double screen_threshold) {
    const auto& elements = T.elements();
    auto operation = [&](const auto& iter) {
        const auto& [sqop, c] = *iter;
        sim_trans_op_impl(O, sqop, c, screen_threshold);
    };
    if (reverse) {
        for (auto it = elements.rbegin(), end = elements.rend(); it != end; ++it) {
            operation(it);
        }
    } else {
        for (auto it = elements.begin(), end = elements.end(); it != end; ++it) {
            operation(it);
        }
    }
}

void sim_trans_op_impl(SparseOperator& O, const SQOperatorString& T_op, double theta,
                       double screen_threshold) {
    // sanity check to make sure all indices are distinct
    if (T_op.cre().fast_a_xor_b_count(T_op.ann()) == 0) {
        throw std::runtime_error("sim_trans_op_impl: the operator " + T_op.str() +
                                 " contains repeated indices.\nThis is not allowed for the "
                                 "similarity transformation.");
    }
    // (1 - T) O(1 + T) = O + [ O, T ] - T O T
    SparseOperator T;
    for (const auto& [O_op, O_c] : O.elements()) {
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
                T.add(cOT_op, theta * cOT_c * O_c);
            }
            // - T [O,T]
            auto T_cOT = T_op * cOT_op;
            for (const auto& [T_cOT_op, T_cOT_c] : T_cOT) {
                if (std::fabs(theta * theta * T_cOT_c * cOT_c * O_c) > screen_threshold) {
                    T.add(T_cOT_op, -theta * theta * T_cOT_c * cOT_c * O_c);
                }
            }
        }
    }
    O += T;
}

void sim_trans_fact_antiherm(SparseOperator& O, const SparseOperatorList& T, bool reverse,
                             double screen_threshold) {
    const auto& elements = T.elements();
    auto operation = [&](const auto& iter) {
        const auto& [sqop, c] = *iter;
        sim_trans_antiherm_impl(O, sqop, c, screen_threshold);
    };
    if (reverse) {
        for (auto it = elements.rbegin(), end = elements.rend(); it != end; ++it) {
            operation(it);
        }
    } else {
        for (auto it = elements.begin(), end = elements.end(); it != end; ++it) {
            operation(it);
        }
    }
}

void sim_trans_fact_antiherm_grad(SparseOperator& O, const SparseOperatorList& T, size_t n,
                                  bool reverse, double screen_threshold) {
    const auto& elements = T.elements();
    auto operation = [&](const auto& iter, const auto& index, const auto& grad_index) {
        const auto& [sqop, c] = *iter;
        if (index == grad_index) {
            sim_trans_antiherm_impl_grad(O, sqop, c, screen_threshold);
        } else {
            sim_trans_antiherm_impl(O, sqop, c, screen_threshold);
        }
    };
    if (reverse) {
        auto size = elements.size();
        for (auto it = elements.rbegin(), end = elements.rend(); it != end; ++it) {
            size_t index = size - 1 - std::distance(elements.rbegin(), it);
            operation(it, index, n);
        }
    } else {
        for (auto it = elements.begin(), end = elements.end(); it != end; ++it) {
            size_t index = std::distance(elements.begin(), it);
            operation(it, index, n);
        }
    }
}

void sim_trans_antiherm_impl(SparseOperator& O, const SQOperatorString& T_op, double theta,
                             double screen_threshold) {
    // sanity check to make sure all indices are distinct
    if (T_op.cre().fast_a_xor_b_count(T_op.ann()) == 0) {
        throw std::runtime_error("sim_trans_antiherm_impl: the operator " + T_op.str() +
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
    auto sin_theta_cos_theta_minus_1 = std::sin(theta) * (std::cos(theta) - 1.0);
    auto sin_theta_half_4 = std::pow(std::sin(0.5 * theta), 4.0);
    auto cos_theta_minus_1_2 = std::pow(std::cos(theta) - 1.0, 2.0);

    for (const auto& [O_op, O_c] : O.elements()) {
        const bool do_O_T_commute = do_ops_commute(O_op, T_op);
        const bool do_O_Td_commute = do_ops_commute(O_op, Td_op);
        // if both commutators are zero, then we can skip this term
        if (do_O_T_commute and do_O_Td_commute) {
            continue;
        }

        // check which terms survive the screening
        const auto do_sin_theta = std::fabs(sin_theta * O_c) > screen_threshold;
        const auto do_sin_theta_2 = std::fabs(sin_theta_2 * O_c) > screen_threshold;
        const auto do_sin_theta_cos_theta_minus_1 =
            std::fabs(sin_theta_cos_theta_minus_1 * O_c) > screen_threshold;
        const auto do_sin_theta_half_4 = std::fabs(2.0 * sin_theta_half_4 * O_c) > screen_threshold;
        const auto do_cos_theta_minus_1_2 = std::fabs(cos_theta_minus_1_2 * O_c) > screen_threshold;

        if (not do_O_T_commute) {
            // [O, T]
            auto cOT = commutator_fast(O_op, T_op);
            for (const auto& [cOT_op, cOT_c] : cOT) {
                // sin(theta) [O, T]
                if (do_sin_theta) {
                    T.add(cOT_op, sin_theta * cOT_c * O_c);
                }
                if (do_sin_theta_2) {
                    // -sin(theta)^2 T [O,T]
                    auto T_cOT = T_op * cOT_op;
                    for (const auto& [T_cOT_op, T_cOT_c] : T_cOT) {
                        T.add(T_cOT_op, -sin_theta_2 * T_cOT_c * cOT_c * O_c);
                    }
                    // +1/2 sin(theta)^2 [T^dagger, [O,T]]
                    auto cTdcOT = commutator_fast(Td_op, cOT_op);
                    for (const auto& [cTdcOT_op, cTdcOT_c] : cTdcOT) {
                        T.add(cTdcOT_op, 0.5 * sin_theta_2 * cTdcOT_c * cOT_c * O_c);
                    }
                    // for small angles, we can use the small angle approximation
                    // sin(theta) = theta, sin(theta)^2 = theta^2, and sin(theta) (cos(theta) -
                    // 1) = theta^2/2 which guarantees that |sin(theta) (cos(theta) - 1)| <
                    // |sin(theta)^2|. That's why we can nest this check here.
                    if (do_sin_theta_cos_theta_minus_1) {
                        auto TdcOT = Td_op * cOT_op;
                        for (const auto& [TdcOT_op, TdcOT_c] : TdcOT) {
                            // -sin(theta) (cos(theta) - 1) T^dagger [O,T] T
                            auto TdcOTT = TdcOT_op * T_op;
                            for (const auto& [TdcOTT_op, TdcOTT_c] : TdcOTT) {
                                T.add(TdcOTT_op, -sin_theta_cos_theta_minus_1 * TdcOTT_c * TdcOT_c *
                                                     cOT_c * O_c);
                            }
                            // sin(theta) (cos(theta) - 1) T^dagger [O,T] T^dagger
                            auto TdcOTd = TdcOT_op * Td_op;
                            for (const auto& [TdcOTTd_op, TdcOTTd_c] : TdcOTd) {
                                T.add(TdcOTTd_op, sin_theta_cos_theta_minus_1 * TdcOTTd_c *
                                                      TdcOT_c * cOT_c * O_c);
                            }
                        }
                        // -sin(theta) (cos(theta) - 1) T [O,T] T^dagger
                        for (const auto& [TcOT_op, TcOT_c] : T_cOT) {
                            auto TcOTTd = TcOT_op * Td_op;
                            for (const auto& [TcOTTd_op, TcOTTd_c] : TcOTTd) {
                                T.add(TcOTTd_op, -sin_theta_cos_theta_minus_1 * TcOTTd_c * TcOT_c *
                                                     cOT_c * O_c);
                            }
                        }
                    }
                }
            }
        }
        if (not do_O_Td_commute) {
            // [O, T^dagger]
            auto cOTd = commutator_fast(O_op, Td_op);
            for (const auto& [cOTd_op, cOTd_c] : cOTd) {
                // -sin(theta)[O, T^dagger]
                if (do_sin_theta) {
                    T.add(cOTd_op, -sin_theta * cOTd_c * O_c);
                }
                if (do_sin_theta_2) {
                    // -sin(theta)^2 T^dagger [O,T^dagger]
                    auto TdcOTd = Td_op * cOTd_op;
                    for (const auto& [TdcOTd_op, TdcOTd_c] : TdcOTd) {
                        T.add(TdcOTd_op, -sin_theta_2 * TdcOTd_c * cOTd_c * O_c);
                    }
                    // +1/2 sin(theta)^2 [T, [O,T^dagger]]
                    auto cTcOTd = commutator_fast(T_op, cOTd_op);
                    for (const auto& [cTcOTd_op, cTcOTd_c] : cTcOTd) {
                        T.add(cTcOTd_op, 0.5 * sin_theta_2 * cTcOTd_c * cOTd_c * O_c);
                    }
                    if (do_sin_theta_cos_theta_minus_1) {
                        auto TcOTd = T_op * cOTd_op;
                        for (const auto& [TcOTd_op, TcOTd_c] : TcOTd) {
                            // -sin(theta) (cos(theta) - 1) T [O,T^dagger] T
                            auto TcOTdT = TcOTd_op * T_op;
                            for (const auto& [TcOTdT_op, TcOTdT_c] : TcOTdT) {
                                T.add(TcOTdT_op, -sin_theta_cos_theta_minus_1 * TcOTdT_c * TcOTd_c *
                                                     cOTd_c * O_c);
                            }
                            // +sin(theta) (cos(theta) - 1) T [O,T^dagger] T^dagger
                            auto TcOTdTd = TcOTd_op * Td_op;
                            for (const auto& [TcOTdTd_op, TcOTdTd_c] : TcOTdTd) {
                                T.add(TcOTdTd_op, sin_theta_cos_theta_minus_1 * TcOTdTd_c *
                                                      TcOTd_c * cOTd_c * O_c);
                            }
                        }

                        for (const auto& [TdcOTd_op, TdcOTd_c] : TdcOTd) {
                            // +sin(theta) (cos(theta) - 1) T^dagger [O,T^dagger] T
                            auto TdcOTdT = TdcOTd_op * T_op;
                            for (const auto& [TdcOTdT_op, TdcOTdT_c] : TdcOTdT) {
                                T.add(TdcOTdT_op, sin_theta_cos_theta_minus_1 * TdcOTdT_c *
                                                      TdcOTd_c * cOTd_c * O_c);
                            }
                        }
                    }
                }
            }
        }
        if (do_sin_theta_half_4) {
            for (const auto& [TdT_op, TdT_c] : TdT) {
                auto TdTO = TdT_op * O_op;
                for (const auto& [TdTO_op, TdTO_c] : TdTO) {
                    // -2 sin(theta/2)^4 T^dagger T O
                    T.add(TdTO_op, -2.0 * sin_theta_half_4 * TdTO_c * TdT_c * O_c);
                    if (do_cos_theta_minus_1_2) {
                        // + (cos(theta) - 1)^2 T^dagger T O T^dagger T
                        for (const auto& [TdT_op2, TdT_c2] : TdT) {
                            auto TdTOTdT = TdTO_op * TdT_op2;
                            for (const auto& [TdTOTdT_op, TdTOTdT_c] : TdTOTdT) {
                                T.add(TdTOTdT_op, cos_theta_minus_1_2 * TdTOTdT_c * TdT_c2 *
                                                      TdTO_c * TdT_c * O_c);
                            }
                        }
                        // + (cos(theta) - 1)^2 T^dagger T O T T^dagger
                        for (const auto& [TTd_op, TTd_c] : TTd) {
                            auto TdTOTTd = TdTO_op * TTd_op;
                            for (const auto& [TdTOTTd_op, TdTOTTd_c] : TdTOTTd) {
                                T.add(TdTOTTd_op, cos_theta_minus_1_2 * TdTOTTd_c * TTd_c * TdTO_c *
                                                      TdT_c * O_c);
                            }
                        }
                    }
                }
                // -2 sin(theta/2)^4 O T^dagger T
                auto OTdT = O_op * TdT_op;
                for (const auto& [OTdT_op, OTdT_c] : OTdT) {
                    T.add(OTdT_op, -2.0 * sin_theta_half_4 * OTdT_c * TdT_c * O_c);
                }
            }
            for (const auto& [TTd_op, TTd_c] : TTd) {
                // -2 sin(theta/2)^4 T T^dagger O
                auto TTdO = TTd_op * O_op;
                for (const auto& [TTdO_op, TTdO_c] : TTdO) {
                    T.add(TTdO_op, -2.0 * sin_theta_half_4 * TTdO_c * TTd_c * O_c);
                    if (do_cos_theta_minus_1_2) {
                        // + (cos(theta) - 1)^2 T T^dagger O T^dagger T
                        for (const auto& [TdT_op, TdT_c] : TdT) {
                            auto TTdOTdT = TTdO_op * TdT_op;
                            for (const auto& [TTdOTdT_op, TTdOTdT_c] : TTdOTdT) {
                                T.add(TTdOTdT_op, cos_theta_minus_1_2 * TTdOTdT_c * TdT_c * TTdO_c *
                                                      TTd_c * O_c);
                            }
                        }
                        // + (cos(theta) - 1)^2 T T^dagger O T T^dagger
                        for (const auto& [TTd_op2, TTd_c2] : TTd) {
                            auto TTdOTTd = TTdO_op * TTd_op2;
                            for (const auto& [TTdOTTd_op, TTdOTTd_c] : TTdOTTd) {
                                T.add(TTdOTTd_op, cos_theta_minus_1_2 * TTdOTTd_c * TTd_c2 *
                                                      TTdO_c * TTd_c * O_c);
                            }
                        }
                    }
                }
                // -2 sin(theta/2)^4 O T T^dagger
                auto OTTd = O_op * TTd_op;
                for (const auto& [OTTd_op, OTTd_c] : OTTd) {
                    T.add(OTTd_op, -2.0 * sin_theta_half_4 * OTTd_c * TTd_c * O_c);
                }
            }
        }
    }
    O += T;
}

void sim_trans_antiherm_impl_grad(SparseOperator& O, const SQOperatorString& T_op, double theta,
                                  double screen_threshold) {
    // sanity check to make sure all indices are distinct
    if (T_op.cre().fast_a_xor_b_count(T_op.ann()) == 0) {
        throw std::runtime_error("sim_trans_antiherm_impl: the operator " + T_op.str() +
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
    auto dsin_theta = std::cos(theta);
    auto dsin_theta_2 = std::sin(2.0 * theta);
    auto dsin_theta_cos_theta_minus_1 = std::cos(2.0 * theta) - std::cos(theta);
    auto dsin_theta_half_4 = std::pow(std::sin(0.5 * theta), 2.0) * std::sin(theta);
    auto dcos_theta_minus_1_2 = 2.0 * (1.0 - std::cos(theta)) * std::sin(theta);

    for (const auto& [O_op, O_c] : O.elements()) {
        const bool do_O_T_commute = do_ops_commute(O_op, T_op);
        const bool do_O_Td_commute = do_ops_commute(O_op, Td_op);
        // if both commutators are zero, then we can skip this term
        if (do_O_T_commute and do_O_Td_commute) {
            continue;
        }

        // check which terms survive the screening
        const auto do_dsin_theta = std::fabs(dsin_theta * O_c) > screen_threshold;
        const auto do_dsin_theta_2 = std::fabs(dsin_theta_2 * O_c) > screen_threshold;
        const auto do_dsin_theta_cos_theta_minus_1 =
            std::fabs(dsin_theta_cos_theta_minus_1 * O_c) > screen_threshold;
        const auto do_dsin_theta_half_4 =
            std::fabs(2.0 * dsin_theta_half_4 * O_c) > screen_threshold;
        const auto do_dcos_theta_minus_1_2 =
            std::fabs(dcos_theta_minus_1_2 * O_c) > screen_threshold;

        if (not do_O_T_commute) {
            // [O, T]
            auto cOT = commutator_fast(O_op, T_op);
            for (const auto& [cOT_op, cOT_c] : cOT) {
                // sin(theta) [O, T]
                if (do_dsin_theta) {
                    T.add(cOT_op, dsin_theta * cOT_c * O_c);
                }
                if (do_dsin_theta_2 or do_dsin_theta_cos_theta_minus_1) {
                    // -sin(theta)^2 T [O,T]
                    auto T_cOT = T_op * cOT_op;
                    if (do_dsin_theta_2) {
                        for (const auto& [T_cOT_op, T_cOT_c] : T_cOT) {
                            T.add(T_cOT_op, -dsin_theta_2 * T_cOT_c * cOT_c * O_c);
                        }
                        // +1/2 sin(theta)^2 [T^dagger, [O,T]]
                        auto cTdcOT = commutator_fast(Td_op, cOT_op);
                        for (const auto& [cTdcOT_op, cTdcOT_c] : cTdcOT) {
                            T.add(cTdcOT_op, 0.5 * dsin_theta_2 * cTdcOT_c * cOT_c * O_c);
                        }
                    }
                    if (do_dsin_theta_cos_theta_minus_1) {
                        auto TdcOT = Td_op * cOT_op;
                        for (const auto& [TdcOT_op, TdcOT_c] : TdcOT) {
                            // -sin(theta) (cos(theta) - 1) T^dagger [O,T] T
                            auto TdcOTT = TdcOT_op * T_op;
                            for (const auto& [TdcOTT_op, TdcOTT_c] : TdcOTT) {
                                T.add(TdcOTT_op, -dsin_theta_cos_theta_minus_1 * TdcOTT_c *
                                                     TdcOT_c * cOT_c * O_c);
                            }
                            // sin(theta) (cos(theta) - 1) T^dagger [O,T] T^dagger
                            auto TdcOTd = TdcOT_op * Td_op;
                            for (const auto& [TdcOTTd_op, TdcOTTd_c] : TdcOTd) {
                                T.add(TdcOTTd_op, dsin_theta_cos_theta_minus_1 * TdcOTTd_c *
                                                      TdcOT_c * cOT_c * O_c);
                            }
                        }
                        // -sin(theta) (cos(theta) - 1) T [O,T] T^dagger
                        for (const auto& [TcOT_op, TcOT_c] : T_cOT) {
                            auto TcOTTd = TcOT_op * Td_op;
                            for (const auto& [TcOTTd_op, TcOTTd_c] : TcOTTd) {
                                T.add(TcOTTd_op, -dsin_theta_cos_theta_minus_1 * TcOTTd_c * TcOT_c *
                                                     cOT_c * O_c);
                            }
                        }
                    }
                }
            }
        }
        if (not do_O_Td_commute) {
            // [O, T^dagger]
            auto cOTd = commutator_fast(O_op, Td_op);
            for (const auto& [cOTd_op, cOTd_c] : cOTd) {
                // -sin(theta)[O, T^dagger]
                if (do_dsin_theta) {
                    T.add(cOTd_op, -dsin_theta * cOTd_c * O_c);
                }
                if (do_dsin_theta_2 or do_dsin_theta_cos_theta_minus_1) {
                    // -sin(theta)^2 T^dagger [O,T^dagger]
                    auto TdcOTd = Td_op * cOTd_op;
                    if (do_dsin_theta_2) {
                        for (const auto& [TdcOTd_op, TdcOTd_c] : TdcOTd) {
                            T.add(TdcOTd_op, -dsin_theta_2 * TdcOTd_c * cOTd_c * O_c);
                        }
                        // +1/2 sin(theta)^2 [T, [O,T^dagger]]
                        auto cTcOTd = commutator_fast(T_op, cOTd_op);
                        for (const auto& [cTcOTd_op, cTcOTd_c] : cTcOTd) {
                            T.add(cTcOTd_op, 0.5 * dsin_theta_2 * cTcOTd_c * cOTd_c * O_c);
                        }
                    }
                    if (do_dsin_theta_cos_theta_minus_1) {
                        auto TcOTd = T_op * cOTd_op;
                        for (const auto& [TcOTd_op, TcOTd_c] : TcOTd) {
                            // -sin(theta) (cos(theta) - 1) T [O,T^dagger] T
                            auto TcOTdT = TcOTd_op * T_op;
                            for (const auto& [TcOTdT_op, TcOTdT_c] : TcOTdT) {
                                T.add(TcOTdT_op, -dsin_theta_cos_theta_minus_1 * TcOTdT_c *
                                                     TcOTd_c * cOTd_c * O_c);
                            }
                            // +sin(theta) (cos(theta) - 1) T [O,T^dagger] T^dagger
                            auto TcOTdTd = TcOTd_op * Td_op;
                            for (const auto& [TcOTdTd_op, TcOTdTd_c] : TcOTdTd) {
                                T.add(TcOTdTd_op, dsin_theta_cos_theta_minus_1 * TcOTdTd_c *
                                                      TcOTd_c * cOTd_c * O_c);
                            }
                        }

                        for (const auto& [TdcOTd_op, TdcOTd_c] : TdcOTd) {
                            // +sin(theta) (cos(theta) - 1) T^dagger [O,T^dagger] T
                            auto TdcOTdT = TdcOTd_op * T_op;
                            for (const auto& [TdcOTdT_op, TdcOTdT_c] : TdcOTdT) {
                                T.add(TdcOTdT_op, dsin_theta_cos_theta_minus_1 * TdcOTdT_c *
                                                      TdcOTd_c * cOTd_c * O_c);
                            }
                        }
                    }
                }
            }
        }
        if (do_dcos_theta_minus_1_2) {
            for (const auto& [TdT_op, TdT_c] : TdT) {
                auto TdTO = TdT_op * O_op;
                for (const auto& [TdTO_op, TdTO_c] : TdTO) {
                    // + (cos(theta) - 1)^2 T^dagger T O T^dagger T
                    for (const auto& [TdT_op2, TdT_c2] : TdT) {
                        auto TdTOTdT = TdTO_op * TdT_op2;
                        for (const auto& [TdTOTdT_op, TdTOTdT_c] : TdTOTdT) {
                            T.add(TdTOTdT_op,
                                  dcos_theta_minus_1_2 * TdTOTdT_c * TdT_c2 * TdTO_c * TdT_c * O_c);
                        }
                    }
                    // + (cos(theta) - 1)^2 T^dagger T O T T^dagger
                    for (const auto& [TTd_op, TTd_c] : TTd) {
                        auto TdTOTTd = TdTO_op * TTd_op;
                        for (const auto& [TdTOTTd_op, TdTOTTd_c] : TdTOTTd) {
                            T.add(TdTOTTd_op,
                                  dcos_theta_minus_1_2 * TdTOTTd_c * TTd_c * TdTO_c * TdT_c * O_c);
                        }
                    }
                }
                if (do_dsin_theta_half_4) {
                    // -2 sin(theta/2)^4 T^dagger T O
                    for (const auto& [TdTO_op, TdTO_c] : TdTO) {
                        if (do_dsin_theta_half_4)
                            T.add(TdTO_op, -2.0 * dsin_theta_half_4 * TdTO_c * TdT_c * O_c);
                    }
                    // -2 sin(theta/2)^4 O T^dagger T
                    auto OTdT = O_op * TdT_op;
                    for (const auto& [OTdT_op, OTdT_c] : OTdT) {
                        T.add(OTdT_op, -2.0 * dsin_theta_half_4 * OTdT_c * TdT_c * O_c);
                    }
                }
            }
            for (const auto& [TTd_op, TTd_c] : TTd) {
                auto TTdO = TTd_op * O_op;
                for (const auto& [TTdO_op, TTdO_c] : TTdO) {
                    // + (cos(theta) - 1)^2 T T^dagger O T^dagger T
                    for (const auto& [TdT_op, TdT_c] : TdT) {
                        auto TTdOTdT = TTdO_op * TdT_op;
                        for (const auto& [TTdOTdT_op, TTdOTdT_c] : TTdOTdT) {
                            T.add(TTdOTdT_op,
                                  dcos_theta_minus_1_2 * TTdOTdT_c * TdT_c * TTdO_c * TTd_c * O_c);
                        }
                    }
                    // + (cos(theta) - 1)^2 T T^dagger O T T^dagger
                    for (const auto& [TTd_op2, TTd_c2] : TTd) {
                        auto TTdOTTd = TTdO_op * TTd_op2;
                        for (const auto& [TTdOTTd_op, TTdOTTd_c] : TTdOTTd) {
                            T.add(TTdOTTd_op,
                                  dcos_theta_minus_1_2 * TTdOTTd_c * TTd_c2 * TTdO_c * TTd_c * O_c);
                        }
                    }
                }
                if (do_dsin_theta_half_4) {
                    // -2 sin(theta/2)^4 T T^dagger O
                    for (const auto& [TTdO_op, TTdO_c] : TTdO) {
                        T.add(TTdO_op, -2.0 * dsin_theta_half_4 * TTdO_c * TTd_c * O_c);
                    }
                    // -2 sin(theta/2)^4 O T T^dagger
                    auto OTTd = O_op * TTd_op;
                    for (const auto& [OTTd_op, OTTd_c] : OTTd) {
                        T.add(OTTd_op, -2.0 * dsin_theta_half_4 * OTTd_c * TTd_c * O_c);
                    }
                }
            }
        }
    }
    O = T;
}

void SparseOperatorList::add_term_from_str(std::string str, double coefficient,
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
    // sort v to guarantee a consistent order
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace forte
