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
#include <complex>
#include <numeric>

#define FMT_HEADER_ONLY
#include "lib/fmt/core.h"

#include "helpers/combinatorial.h"
#include "helpers/timer.h"
#include "helpers/string_algorithms.h"

#include "sparse_operator.h"

namespace forte {

void sim_trans_op_impl(SparseOperator& O, const SQOperatorString& T_op, sparse_scalar_t theta,
                       double screen_threshold);

void sim_trans_imagherm_impl(SparseOperator& O, const SQOperatorString& T_op, sparse_scalar_t theta,
                             double screen_threshold);

void sim_trans_imagherm_impl2(SparseOperator& O, const SQOperatorString& T_op,
                              sparse_scalar_t theta, double screen_threshold);

std::string format_term_in_sum(sparse_scalar_t coefficient, const std::string& term) {
    // if (term == "[ ]") {
    //     if (coefficient == 0.0) {
    //         return "";
    //     } else {
    //         return (coefficient > 0.0 ? "+ " : "") + to_string_with_precision(coefficient, 12);
    //     }
    // }
    // if (coefficient == 0.0) {
    //     return "";
    // } else if (coefficient == 1.0) {
    //     return "+ " + term;
    // } else if (coefficient == -1.0) {
    //     return "- " + term;
    // } else if (coefficient == static_cast<int>(coefficient)) {
    //     return to_string_with_precision(coefficient, 12) + " * " + term;
    // } else {
    //     std::string s = to_string_with_precision(coefficient, 12);
    //     s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    //     return s + " * " + term;
    // }
    // return "";
    // if constexpr (std::is_same_v<sparse_scalar_t, std::complex<double>>) {
    // }
    // if constexpr (std::is_same_v<sparse_scalar_t, double>) {
    //     return fmt::format("{} * {}", coefficient, term);
    // }
    return fmt::format("({} + {}i) * {}", std::real(coefficient), std::imag(coefficient), term);
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
        const std::string s = to_string_latex(c) + "\\;" + sqop.latex();
        v.push_back(s);
    }
    // sort v to guarantee a consistent order
    std::sort(v.begin(), v.end());
    return join(v, " ");
}

void SparseOperator::add_term_from_str(const std::string& s, sparse_scalar_t coefficient,
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
                [&C](const SQOperatorString& sqop, const sparse_scalar_t c) { C.add(sqop, c); });
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
                [&C](const SQOperatorString& sqop, const sparse_scalar_t c) { C[sqop] += c; });
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

void sim_trans_op_impl(SparseOperator& O, const SQOperatorString& T_op, sparse_scalar_t theta,
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
        if (std::abs(O_c * theta) < screen_threshold) {
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
            if (std::abs(theta * cOT_c * O_c) > screen_threshold) {
                T.add(cOT_op, theta * cOT_c * O_c);
            }
            // - T [O,T]
            auto T_cOT = T_op * cOT_op;
            for (const auto& [T_cOT_op, T_cOT_c] : T_cOT) {
                if (std::abs(theta * theta * T_cOT_c * cOT_c * O_c) > screen_threshold) {
                    T.add(T_cOT_op, -theta * theta * T_cOT_c * cOT_c * O_c);
                }
            }
        }
    }
    O += T;
}

void sim_trans_fact_imagherm(SparseOperator& O, const SparseOperatorList& T, bool reverse,
                             double screen_threshold) {
    const auto& elements = T.elements();
    auto operation = [&](const auto& iter) {
        const auto& [sqop, c] = *iter;
        sim_trans_imagherm_impl2(O, sqop, c, screen_threshold);
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

void sim_trans_imagherm_impl2(SparseOperator& O, const SQOperatorString& T_op,
                              sparse_scalar_t theta, double screen_threshold) {
    // This function evaluates exp(i theta (T + T^dagger)) O exp(-i theta (T + T^dagger))
    // sanity check that theta is real
    if (std::abs(std::imag(theta)) > 1.0e-12) {
        throw std::runtime_error("sim_trans_imagherm_impl: the angle theta must be real.");
    }

    // consider the case where the operator is a number operator
    if (T_op.cre().fast_a_xor_b_count(T_op.ann()) == 0) {
        SparseOperator T;
        double two_theta = 2.0 * std::real(theta);
        sparse_scalar_t four_sin_theta_2 = 4.0 * std::pow(std::sin(theta), 2.0);
        std::complex<double> exp_phase_min_one = std::polar(1.0, two_theta) - 1.0;
        std::complex<double> exp_min_phase_min_one = std::polar(1.0, -two_theta) - 1.0;
        for (const auto& [O_op, O_c] : O.elements()) {
            auto NO = T_op * O_op;
            for (const auto& [NO_op, NO_c] : NO) {
                T.add(NO_op, exp_phase_min_one * NO_c * O_c);
                auto NON = NO_op * T_op;
                for (const auto& [NON_op, NON_c] : NON) {
                    T.add(NON_op, four_sin_theta_2 * NON_c * NO_c * O_c);
                }
            }
            auto ON = O_op * T_op;
            for (const auto& [ON_op, ON_c] : ON) {
                T.add(ON_op, exp_min_phase_min_one * ON_c * O_c);
            }
        }
        O += T;
        return;
    }

    // Td = T^dagger
    auto Td_op = T_op.adjoint();
    // TTd = T T^dagger (gives a number operator if there are no repeated indices)
    auto TTd = T_op * Td_op;
    // TdT = T^dagger T (gives a number operator if there are no repeated indices)
    auto TdT = Td_op * T_op;

    SparseOperator T;
    std::complex<double> imag1 = std::complex<double>(0.0, 1.0);
    sparse_scalar_t min_i_sin_theta = -imag1 * std::sin(theta);
    sparse_scalar_t cos_theta_minus_1 = std::cos(theta) - 1.0;
    sparse_scalar_t sin_theta_2 = std::pow(std::sin(theta), 2.0);
    sparse_scalar_t i_sin_theta_cos_theta_minus_1 =
        imag1 * std::sin(theta) * (std::cos(theta) - 1.0);
    sparse_scalar_t cos_theta_minus_1_2 = std::pow(std::cos(theta) - 1.0, 2.0);

    for (const auto& [O_op, O_c] : O.elements()) {
        if (std::abs(O_c) < screen_threshold) {
            continue;
        }
        const bool do_O_T_commute = do_ops_commute(O_op, T_op);
        const bool do_O_Td_commute = do_ops_commute(O_op, Td_op);
        // if both commutators are zero, then we can skip this term
        if (do_O_T_commute and do_O_Td_commute) {
            continue;
        }

        // check which terms survive the screening
        const auto do_min_i_sin_theta = std::abs(min_i_sin_theta * O_c) > screen_threshold;
        const auto do_sin_theta_2 = std::abs(sin_theta_2 * O_c) > screen_threshold;
        const auto do_i_sin_theta_cos_theta_minus_1 =
            std::abs(i_sin_theta_cos_theta_minus_1 * O_c) > screen_threshold;
        const auto do_cos_theta_minus_1_2 = std::abs(cos_theta_minus_1_2 * O_c) > screen_threshold;

        // -i sin(theta) [O, S] (all)
        // +sin(theta)^2 SOS (all)
        auto OT = O_op * T_op;
        for (const auto& [OT_op, OT_c] : OT) {
            T.add(OT_op, +min_i_sin_theta * OT_c * O_c);
            auto TOT = T_op * OT_op;
            // +sin(theta)^2 TOT
            for (const auto& [TOT_op, TOT_c] : TOT) {
                T.add(TOT_op, sin_theta_2 * TOT_c * OT_c * O_c);
            }
            auto TdOT = Td_op * OT_op;
            // +sin(theta)^2 TdOT
            for (const auto& [TdOT_op, TdOT_c] : TdOT) {
                T.add(TdOT_op, sin_theta_2 * TdOT_c * OT_c * O_c);
            }
        }
        auto TO = T_op * O_op;
        for (const auto& [TO_op, TO_c] : TO) {
            T.add(TO_op, -min_i_sin_theta * TO_c * O_c);
        }
        auto OTd = O_op * Td_op;
        for (const auto& [OTd_op, OTd_c] : OTd) {
            T.add(OTd_op, +min_i_sin_theta * OTd_c * O_c);
            // +sin(theta)^2 TOTd
            auto TOTd = T_op * OTd_op;
            for (const auto& [TOTd_op, TOTd_c] : TOTd) {
                T.add(TOTd_op, sin_theta_2 * TOTd_c * OTd_c * O_c);
            }
            // +sin(theta)^2 TdOTd
            auto TdOTd = Td_op * OTd_op;
            for (const auto& [TdOTd_op, TdOTd_c] : TdOTd) {
                T.add(TdOTd_op, sin_theta_2 * TdOTd_c * OTd_c * O_c);
            }
        }
        auto TdO = Td_op * O_op;
        for (const auto& [TdO_op, TdO_c] : TdO) {
            T.add(TdO_op, -min_i_sin_theta * TdO_c * O_c);
        }

        // // (cos(theta) - 1) {O, SS} (all)
        for (const auto& [TTd_op, TTd_c] : TTd) {
            auto OTTd = O_op * TTd_op;
            for (const auto& [OTTd_op, OTTd_c] : OTTd) {
                // (cos(theta) - 1) OTTd
                T.add(OTTd_op, cos_theta_minus_1 * OTTd_c * TTd_c * O_c); // OK
                // + i sin(theta) (cos(theta) - 1) TOTTd
                auto TOTTd = T_op * OTTd_op;
                for (const auto& [TOTTd_op, TOTTd_c] : TOTTd) {
                    T.add(TOTTd_op, i_sin_theta_cos_theta_minus_1 * TOTTd_c * OTTd_c * TTd_c * O_c);
                }
                // + i sin(theta) (cos(theta) - 1) TdOTTd
                auto TdOTTd = Td_op * OTTd_op;
                for (const auto& [TdOTTd_op, TdOTTd_c] : TdOTTd) {
                    T.add(TdOTTd_op,
                          i_sin_theta_cos_theta_minus_1 * TdOTTd_c * OTTd_c * TTd_c * O_c);
                }
                // + (cos(theta) - 1)^2 TTdOTTd
                for (const auto& [TTd_op2, TTd_c2] : TTd) {
                    auto TTdOTTd = TTd_op2 * OTTd_op;
                    for (const auto& [TTdOTTd_op, TTdOTTd_c] : TTdOTTd) {
                        T.add(TTdOTTd_op,
                              cos_theta_minus_1_2 * TTdOTTd_c * TTd_c2 * OTTd_c * TTd_c * O_c);
                    }
                }
                // + (cos(theta) - 1)^2 TdTOTTd
                for (const auto& [TdT_op, TdT_c] : TdT) {
                    auto TdTOTTd = TdT_op * OTTd_op;
                    for (const auto& [TdTOTTd_op, TdTOTTd_c] : TdTOTTd) {
                        T.add(TdTOTTd_op,
                              cos_theta_minus_1_2 * TdTOTTd_c * TdT_c * OTTd_c * TTd_c * O_c);
                    }
                }
            }
            auto TTdO = TTd_op * O_op;
            for (const auto& [TTdO_op, TTdO_c] : TTdO) {
                // (cos(theta) - 1) TTdO
                T.add(TTdO_op, cos_theta_minus_1 * TTdO_c * TTd_c * O_c);
                // - i sin(theta) (cos(theta) - 1) TTdOT
                auto TTdOT = TTdO_op * T_op;
                for (const auto& [TTdOT_op, TTdOT_c] : TTdOT) {
                    T.add(TTdOT_op,
                          -i_sin_theta_cos_theta_minus_1 * TTdOT_c * TTdO_c * TTd_c * O_c);
                }
                // - i sin(theta) (cos(theta) - 1) TTdOTd
                auto TTdOTd = TTdO_op * Td_op;
                for (const auto& [TTdOTd_op, TTdOTd_c] : TTdOTd) {
                    T.add(TTdOTd_op,
                          -i_sin_theta_cos_theta_minus_1 * TTdOTd_c * TTdO_c * TTd_c * O_c);
                }
            }
        }
        for (const auto& [TdT_op, TdT_c] : TdT) {
            auto OTdT = O_op * TdT_op;
            for (const auto& [OTdT_op, OTdT_c] : OTdT) {
                // (cos(theta) - 1) OTdT
                T.add(OTdT_op, cos_theta_minus_1 * OTdT_c * TdT_c * O_c);
                // + i sin(theta) (cos(theta) - 1) TOTdT
                auto TOTdT = T_op * OTdT_op;
                for (const auto& [TOTdT_op, TOTdT_c] : TOTdT) {
                    T.add(TOTdT_op, i_sin_theta_cos_theta_minus_1 * TOTdT_c * OTdT_c * TdT_c * O_c);
                }
                // + i sin(theta) (cos(theta) - 1) TdOTdT
                auto TdOTdT = Td_op * OTdT_op;
                for (const auto& [TdOTdT_op, TdOTdT_c] : TdOTdT) {
                    T.add(TdOTdT_op,
                          i_sin_theta_cos_theta_minus_1 * TdOTdT_c * OTdT_c * TdT_c * O_c);
                }
                // + (cos(theta) - 1)^2 TTdOTdT
                for (const auto& [TTd_op, TTd_c] : TTd) {
                    auto TTdOTdT = TTd_op * OTdT_op;
                    for (const auto& [TTdOTdT_op, TTdOTdT_c] : TTdOTdT) {
                        T.add(TTdOTdT_op,
                              cos_theta_minus_1_2 * TTdOTdT_c * TTd_c * OTdT_c * TdT_c * O_c);
                    }
                }
                // + (cos(theta) - 1)^2 TdTOTdT
                for (const auto& [TdT_op2, TdT_c2] : TdT) {
                    auto TdTOTdT = TdT_op2 * OTdT_op;
                    for (const auto& [TdTOTdT_op, TdTOTdT_c] : TdTOTdT) {
                        T.add(TdTOTdT_op,
                              cos_theta_minus_1_2 * TdTOTdT_c * TdT_c2 * OTdT_c * TdT_c * O_c);
                    }
                }
            }
            auto TdTO = TdT_op * O_op;
            for (const auto& [TdTO_op, TdTO_c] : TdTO) {
                // (cos(theta) - 1) OTdT
                T.add(TdTO_op, cos_theta_minus_1 * TdTO_c * TdT_c * O_c);
                // - i sin(theta) (cos(theta) - 1) TdTOT
                auto TdTOT = TdTO_op * T_op;
                for (const auto& [TdTOT_op, TdTOT_c] : TdTOT) {
                    T.add(TdTOT_op,
                          -i_sin_theta_cos_theta_minus_1 * TdTOT_c * TdTO_c * TdT_c * O_c);
                }
                // - i sin(theta) (cos(theta) - 1) TdTOTd
                auto TdTOTd = TdTO_op * Td_op;
                for (const auto& [TdTOTd_op, TdTOTd_c] : TdTOTd) {
                    T.add(TdTOTd_op,
                          -i_sin_theta_cos_theta_minus_1 * TdTOTd_c * TdTO_c * TdT_c * O_c);
                }
            }
        }
    }
    O += T;
}

void SparseOperatorList::add_term_from_str(std::string str, sparse_scalar_t coefficient,
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
    return v;
}

} // namespace forte
