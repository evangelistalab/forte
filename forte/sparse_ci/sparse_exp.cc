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

#include "sparse_ci/sparse_exp.h"

namespace forte {

size_t num_attempts_ = 0;
size_t num_success_ = 0;

StateVector SparseExp::apply_op(const SparseOperator& sop, const StateVector& state0,
                                const std::string& algorithm, double scaling_factor, int maxk,
                                double screen_thresh) {
    return compute(OperatorType::Excitation, sop, state0, algorithm, scaling_factor, maxk,
                   screen_thresh);
}

StateVector SparseExp::apply_antiherm(const SparseOperator& sop, const StateVector& state0,
                                      const std::string& algorithm, double scaling_factor, int maxk,
                                      double screen_thresh) {
    return compute(OperatorType::Antihermitian, sop, state0, algorithm, scaling_factor, maxk,
                   screen_thresh);
}

StateVector SparseExp::compute(OperatorType op_type, const SparseOperator& sop,
                               const StateVector& state0, const std::string& algorithm,
                               double scaling_factor, int maxk, double screen_thresh) {
    local_timer t;
    Algorithm alg = Algorithm::OnTheFlyStd;
    // if (algorithm == "onthefly") {
    //     alg = Algorithm::OnTheFlySorted;
    // } else if (algorithm == "ontheflystd") {
    //     alg = Algorithm::OnTheFlyStd;
    // }

    auto state = apply_exp_operator(op_type, sop, state0, scaling_factor, maxk, screen_thresh, alg);

    timings_["total"] = t.get();
    return state;
}

StateVector SparseExp::apply_exp_operator(OperatorType op_type, const SparseOperator& sop,
                                          const StateVector& state0, double scaling_factor,
                                          int maxk, double screen_thresh, Algorithm alg) {
    double convergence_threshold_ = screen_thresh;

    StateVector exp_state(state0);
    StateVector state(state0);
    StateVector new_terms;

    double factor = 1.0;
    for (int k = 1; k <= maxk; k++) {
        factor *= scaling_factor / static_cast<double>(k);
        if (alg == Algorithm::Cached) {
            new_terms = apply_operator_cached(op_type, sop, state, screen_thresh);
        } else if (alg == Algorithm::OnTheFlyStd) {
            new_terms = apply_operator_std(op_type, sop, state, screen_thresh);
        } else if (alg == Algorithm::OnTheFlySorted) {
            new_terms = apply_operator_sorted(op_type, sop, state, screen_thresh);
        }
        double norm = 0.0;
        double inf_norm = 0.0;
        for (const auto& det_c : new_terms) {
            const double delta_exp = factor * det_c.second;
            exp_state[det_c.first] += delta_exp;
            norm += std::pow(delta_exp, 2.0);
            inf_norm = std::max(inf_norm, std::fabs(delta_exp));
        }
        norm = std::sqrt(norm);
        if (inf_norm < convergence_threshold_) {
            break;
        }
        state = new_terms;
    }
    return exp_state;
}

StateVector SparseExp::apply_operator_cached(OperatorType op_type, const SparseOperator& sop,
                                             const StateVector& state0, double screen_thresh) {
    // make a copy of the state
    // std::vector<std::tuple<double, Determinant>> state_sorted;
    // state_sorted.reserve(state0.size());
    // for (const auto& [det, c] : state0) {
    //     state_sorted.emplace_back(c, det);
    // }
    // std::sort(state_sorted.rbegin(), state_sorted.rend(), [](const auto& lhs, const auto& rhs) {
    //     return std::abs(lhs.first) > std::abs(rhs.first);
    // });

    StateVector new_terms;
    // std::vector<std::tuple<double, double, Determinant>> state_sorted;

    // size_t k = 0;
    // for (const auto& det_c : state0) {
    //     const Determinant& d = det_c.first;
    //     const double c = det_c.second;
    //     state_sorted[k] = std::make_tuple(std::fabs(c), c, d);
    //     ++k;
    // }
    // std::sort(state_sorted.rbegin(), state_sorted.rend());

    // Determinant d_new;

    // // loop over all determinants
    // for (const auto& absc_c_det : state_sorted) {
    //     const double absc = std::get<0>(absc_c_det);
    //     if (absc > screen_thresh) {
    //         const double c = std::get<1>(absc_c_det);
    //         const Determinant& d = std::get<2>(absc_c_det);

    //         auto search = couplings_.find(d);

    //         if (search == couplings_.end()) {
    //             local_timer t_couplings;
    //             // we have to build the coupling list for this determinant
    //             std::vector<std::tuple<size_t, Determinant, double>> d_couplings;
    //             // loop over all the operators
    //             for (size_t n = 0; const auto& [sqop, _] : sop.elements()) {
    //                 // create a mask for screening determinants according to the creation
    //                 operators
    //                 // This mask looks only at creation operators that are not preceeded by
    //                 // annihilation operators
    //                 const Determinant ucre = sqop.cre() - sqop.ann();
    //                 // check if this operator can be applied
    //                 if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
    //                     d_new = d;
    //                     double value = apply_operator_to_det_fast(d_new, sqop.cre(), sqop.ann());
    //                     d_couplings.push_back(std::make_tuple(n, d_new, value));
    //                 }
    //                 n++;
    //             }
    //             couplings_[d] = d_couplings;
    //             timings_["couplings"] += t_couplings.get();
    //         }
    //         local_timer t_sum;
    //         // apply the operator
    //         const auto& d_couplings = couplings_[d];
    //         for (const auto& [op, d, f] : d_couplings) {
    //             const double value = sop.coefficient(op) * f * c;
    //             if (std::fabs(value) > screen_thresh)
    //                 new_terms[d] += value;
    //         }
    //         timings_["exp"] += t_sum.get();
    //     } else {
    //         break;
    //     }
    // }

    // if (op_type == OperatorType::Antihermitian) {
    //     for (const auto& absc_c_det : state_sorted) {
    //         const double absc = std::get<0>(absc_c_det);
    //         if (absc > screen_thresh) {
    //             const double c = std::get<1>(absc_c_det);
    //             const Determinant& d = std::get<2>(absc_c_det);

    //             auto search = couplings_dexc_.find(d);

    //             if (search == couplings_dexc_.end()) {
    //                 local_timer t_couplings;
    //                 // we have to build the coupling list for this determinant
    //                 std::vector<std::tuple<size_t, Determinant, double>> d_couplings;
    //                 // loop over all the operators
    //                 for (size_t n = 0, maxn = sop.size(); n < maxn; n++) {
    //                     const SQOperatorString& sqop = sop.term_operator(n);
    //                     // create a mask for screening determinants according to the annihilation
    //                     // operators. This mask looks only at annihilation operators that are not
    //                     // preceeded by creation operators
    //                     const Determinant ucre = sqop.ann() - sqop.cre();
    //                     // check if this operator can be applied
    //                     if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(ucre))
    //                     {
    //                         d_new = d;
    //                         double value =
    //                             apply_operator_to_det_fast(d_new, sqop.ann(), sqop.cre());
    //                         d_couplings.push_back(std::make_tuple(n, d_new, value));
    //                     }
    //                 }
    //                 couplings_dexc_[d] = d_couplings;
    //                 timings_["couplings"] += t_couplings.get();
    //             }
    //             local_timer t_sum;
    //             // apply the operator
    //             const auto& d_couplings = couplings_dexc_[d];
    //             for (const auto& [op, d, f] : d_couplings) {
    //                 const double value = sop.coefficient(op) * f * c;
    //                 if (std::fabs(value) > screen_thresh)
    //                     new_terms[d] -= value;
    //             }
    //             timings_["exp"] += t_sum.get();
    //         } else {
    //             break;
    //         }
    //     }
    // }
    return new_terms;
}

StateVector SparseExp::apply_operator_sorted(OperatorType op_type, const SparseOperator& sop,
                                             const StateVector& state0, double screen_thresh) {
    throw std::runtime_error("apply_operator_sorted Not implemented");
    // // make a copy of the state
    // std::vector<std::tuple<double, double, Determinant>> state_sorted(state0.size());
    // size_t k = 0;
    // for (const auto& det_c : state0) {
    //     const Determinant& d = det_c.first;
    //     const double c = det_c.second;
    //     state_sorted[k] = std::make_tuple(std::fabs(c), c, d);
    //     ++k;
    // }
    // std::sort(state_sorted.rbegin(), state_sorted.rend());

    StateVector new_terms;
    // Determinant d;

    // // loop over all the operators
    // for (size_t n = 0, maxn = sop.size(); n < maxn; n++) {
    //     const auto [sqop, coefficient] = sop.term(n);
    //     if (coefficient == 0.0)
    //         continue;
    //     // create a mask for screening determinants according to the creation operators
    //     // this mask looks only at creation operators that are not preceeded by annihilation
    //     // operators
    //     const Determinant ucre = sqop.cre() - sqop.ann();
    //     // loop over all determinants
    //     for (const auto& [absc, c, det] : state_sorted) {
    //         num_attempts_++;
    //         // screen according to the product tau * c
    //         if (std::fabs(coefficient * c) > screen_thresh) {
    //             // check if this operator can be applied
    //             if (det.fast_a_and_b_equal_b(sqop.ann()) and det.fast_a_and_b_eq_zero(ucre)) {
    //                 d = det;
    //                 const double value =
    //                     apply_operator_to_det_fast(d, sqop.cre(), sqop.ann()) * coefficient * c;
    //                 new_terms[d] += value;
    //                 num_success_++;
    //             }
    //         } else {
    //             break;
    //         }
    //     }
    // }

    // if (op_type == OperatorType::Antihermitian) {
    //     // loop over all the operators
    //     for (size_t n = 0, maxn = sop.size(); n < maxn; n++) {
    //         const auto [sqop, coefficient] = sop.term(n);
    //         if (coefficient == 0.0)
    //             continue;
    //         // create a mask for screening determinants according to the creation operators
    //         // this mask looks only at creation operators that are not preceeded by annihilation
    //         // operators
    //         const Determinant ucre = sqop.ann() - sqop.cre();
    //         // loop over all determinants
    //         for (const auto& [absc, c, det] : state_sorted) {
    //             num_attempts_++;

    //             // screen according to the product tau * c
    //             if (std::fabs(coefficient * c) > screen_thresh) {
    //                 // check if this operator can be applied
    //                 if (det.fast_a_and_b_equal_b(sqop.cre()) and det.fast_a_and_b_eq_zero(ucre))
    //                 {
    //                     d = det;
    //                     double value =
    //                         apply_operator_to_det_fast(d, sqop.ann(), sqop.cre()) * coefficient *
    //                         c;
    //                     new_terms[d] -= value;
    //                     num_success_++;
    //                 }
    //             } else {
    //                 break;
    //             }
    //         }
    //     }
    // }
    return new_terms;
}

StateVector SparseExp::apply_operator_std(OperatorType op_type, const SparseOperator& sop,
                                          const StateVector& state0, double screen_thresh) {
    local_timer t;

    StateVector new_terms;

    Determinant new_d;
    // loop over all the operators
    for (const auto& [sqop, coefficient] : sop.elements()) {
        if (coefficient == 0.0)
            continue;
        // create a mask for screening determinants according to the creation operators
        // this mask looks only at creation operators that are not preceeded by annihilation
        // operators
        const Determinant ucre = sqop.cre() - sqop.ann();
        // loop over all determinants
        for (const auto& [d, c] : state0) {
            // test if we can apply this operator to this determinant
            // screen according to the product tau * c
            if (std::fabs(coefficient * c) > screen_thresh) {
                // check if this operator can be applied
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    new_d = d;
                    double value =
                        apply_operator_to_det_fast(new_d, sqop.cre(), sqop.ann()) * coefficient * c;
                    new_terms[new_d] += value;
                }
            }
        }
    }

    if (op_type == OperatorType::Antihermitian) {
        // loop over all the operators
        for (const auto& [sqop, coefficient] : sop.elements()) {
            if (coefficient == 0.0)
                continue;
            // create a mask for screening determinants according to the creation operators
            // this mask looks only at creation operators that are not preceeded by annihilation
            // operators
            const Determinant ucre = sqop.ann() - sqop.cre();
            // loop over all determinants
            for (const auto& [d, c] : state0) {
                // test if we can apply this operator to this determinant
                // screen according to the product tau * c
                if (std::fabs(coefficient * c) > screen_thresh) {
                    // check if this operator can be applied
                    if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(ucre)) {
                        new_d = d;
                        double value = apply_operator_to_det_fast(new_d, sqop.ann(), sqop.cre()) *
                                       coefficient * c;
                        new_terms[new_d] -= value;
                    }
                }
            }
        }
    }
    return new_terms;
}

std::map<std::string, double> SparseExp::timings() const { return timings_; }

} // namespace forte
