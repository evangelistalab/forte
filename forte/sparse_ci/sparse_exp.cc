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

StateVector SparseExp::apply_op(const SparseOperatorList& sop_list, const StateVector& state0,
                                const std::string& algorithm, double scaling_factor, int maxk,
                                double screen_thresh) {
    SparseOperator sop = sop_list.to_operator();
    return apply_op(sop, state0, algorithm, scaling_factor, maxk, screen_thresh);
}

StateVector SparseExp::apply_antiherm(const SparseOperator& sop, const StateVector& state0,
                                      const std::string& algorithm, double scaling_factor, int maxk,
                                      double screen_thresh) {
    return compute(OperatorType::Antihermitian, sop, state0, algorithm, scaling_factor, maxk,
                   screen_thresh);
}

StateVector SparseExp::apply_antiherm(const SparseOperatorList& sop_list, const StateVector& state0,
                                      const std::string& algorithm, double scaling_factor, int maxk,
                                      double screen_thresh) {
    SparseOperator sop = sop_list.to_operator();
    return apply_antiherm(sop, state0, algorithm, scaling_factor, maxk, screen_thresh);
}

StateVector SparseExp::compute(OperatorType op_type, const SparseOperator& sop,
                               const StateVector& state0, const std::string& algorithm,
                               double scaling_factor, int maxk, double screen_thresh) {
    local_timer t;
    auto alg = Algorithm::OnTheFlyStd;
    if (algorithm == "onthefly") {
        alg = Algorithm::OnTheFlySorted;
    } else if (algorithm == "ontheflystd") {
        alg = Algorithm::OnTheFlyStd;
    } else if (algorithm == "cached" or algorithm == "default") {
        alg = Algorithm::Cached;
    } else {
        throw std::runtime_error("Unknown algorithm for SparseExp: " + algorithm +
                                 ". Options are onthefly, ontheflystd, or cached (default).");
    }

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
        state *= scaling_factor / static_cast<double>(k);
        if (alg == Algorithm::OnTheFlyStd) {
            new_terms = apply_operator_std(op_type, sop, state, screen_thresh);
        } else if (alg == Algorithm::OnTheFlySorted) {
            new_terms = apply_operator_sorted(op_type, sop, state, screen_thresh);
        } else if (alg == Algorithm::Cached) {
            new_terms = apply_operator_cached(op_type, sop, state, screen_thresh);
        }
        double norm = 0.0;
        double inf_norm = 0.0;
        exp_state += new_terms;
        for (const auto& [det, c] : new_terms) {
            norm += std::pow(c, 2.0);
            inf_norm = std::max(inf_norm, std::abs(c));
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
    StateVector new_terms;
    std::vector<std::tuple<double, double, Determinant>> state_sorted(state0.size());

    for (size_t k = 0; const auto& det_c : state0) {
        const Determinant& d = det_c.first;
        const double c = det_c.second;
        state_sorted[k] = std::make_tuple(std::fabs(c), c, d);
        ++k;
    }
    std::sort(state_sorted.rbegin(), state_sorted.rend());

    double max_t = 0.0;
    for (const auto& [_, t] : sop.elements()) {
        max_t = std::max(max_t, std::abs(t));
    }

    Determinant new_d;

    // loop over all determinants
    for (const auto& [absc, c, d] : state_sorted) {
        if (absc * max_t < screen_thresh)
            break;
        // search for the couplings of this determinant
        auto search = couplings_.find(d);

        if (search == couplings_.end()) {
            local_timer t_couplings;
            // we have to build the coupling list for this determinant
            std::vector<std::tuple<SQOperatorString, Determinant, double>> d_couplings;
            // loop over all the operators
            for (const auto& [sqop, _] : sop.elements()) {
                // create a mask for screening determinants according to the creation
                // operators This mask looks only at creation operators that are not
                // preceeded by annihilation operators
                const Determinant ucre = sqop.cre() - sqop.ann();
                // check if this operator can be applied
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    new_d = d;
                    double value = apply_operator_to_det_fast(new_d, sqop.cre(), sqop.ann());
                    d_couplings.push_back(std::make_tuple(sqop, new_d, value));
                }
            }
            couplings_[d] = d_couplings;
            timings_["couplings"] += t_couplings.get();
        }
        local_timer t_sum;
        // apply the operator
        for (const auto& [sqop, d2, f] : couplings_[d]) {
            const double value = sop[sqop] * f * c;
            if (std::abs(value) > screen_thresh)
                new_terms[d2] += value;
        }
        timings_["exp"] += t_sum.get();
    }

    if (op_type != OperatorType::Antihermitian) {
        return new_terms;
    }

    for (const auto& [absc, c, d] : state_sorted) {
        if (absc * max_t < screen_thresh)
            break;

        auto search = couplings_dexc_.find(d);

        if (search == couplings_dexc_.end()) {
            local_timer t_couplings;
            // we have to build the coupling list for this determinant
            std::vector<std::tuple<SQOperatorString, Determinant, double>> d_couplings;
            // loop over all the operators
            for (const auto& [sqop, _] : sop.elements()) {
                // create a mask for screening determinants according to the
                // annihilation operators. This mask looks only at annihilation
                // operators that are not preceeded by creation operators
                const Determinant ucre = sqop.ann() - sqop.cre();
                // check if this operator can be applied
                if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(ucre)) {
                    new_d = d;
                    double value = apply_operator_to_det_fast(new_d, sqop.ann(), sqop.cre());
                    d_couplings.push_back(std::make_tuple(sqop, new_d, value));
                }
            }
            couplings_dexc_[d] = d_couplings;
            timings_["couplings"] += t_couplings.get();
        }
        local_timer t_sum;
        // apply the operator
        const auto& d_couplings = couplings_dexc_[d];
        for (const auto& [sqop, d2, f] : d_couplings) {
            const double value = sop[sqop] * f * c;
            if (std::fabs(value) > screen_thresh)
                new_terms[d2] -= value;
        }
        timings_["exp"] += t_sum.get();
    }
    return new_terms;
}

StateVector SparseExp::apply_operator_sorted(OperatorType op_type, const SparseOperator& sop,
                                             const StateVector& state0, double screen_thresh) {
    // make a copy of the state
    std::vector<std::tuple<double, double, Determinant>> state_sorted(state0.size());
    double max_c = 0.0;
    for (size_t k = 0; const auto& [d, c] : state0) {
        state_sorted[k] = std::make_tuple(std::abs(c), c, d);
        max_c = std::max(max_c, std::abs(c));
        ++k;
    }
    std::sort(state_sorted.rbegin(), state_sorted.rend());

    StateVector new_terms;
    Determinant d;

    // loop over all the operators
    for (const auto& [sqop, coefficient] : sop.elements()) {
        if (std::abs(coefficient * max_c) < screen_thresh)
            continue;
        // create a mask for screening determinants according to the
        // creation operators this mask looks only at creation operators
        // that are not preceeded by annihilation operators
        const Determinant ucre = sqop.cre() - sqop.ann();
        // loop over all determinants
        for (const auto& [absc, c, det] : state_sorted) {
            const double factor = coefficient * c;
            num_attempts_++;
            // screen according to the product tau * c
            if (std::fabs(factor) > screen_thresh) {
                // check if this operator can be applied
                if (det.fast_a_and_b_equal_b(sqop.ann()) and det.fast_a_and_b_eq_zero(ucre)) {
                    d = det;
                    const double value = apply_operator_to_det_fast(d, sqop.cre(), sqop.ann());
                    new_terms[d] += value * factor;
                    num_success_++;
                }
            } else {
                break;
            }
        }
    }

    if (op_type != OperatorType::Antihermitian) {
        return new_terms;
    }

    // loop over all the operators
    for (const auto& [sqop, coefficient] : sop.elements()) {
        if (std::abs(coefficient * max_c) < screen_thresh)
            continue;
        // create a mask for screening determinants according to the
        // creation operators this mask looks only at creation operators
        // that are not preceeded by annihilation operators
        const Determinant ucre = sqop.ann() - sqop.cre();
        // loop over all determinants
        for (const auto& [absc, c, det] : state_sorted) {
            const double factor = coefficient * c;
            num_attempts_++;
            // screen according to the product tau * c
            if (std::fabs(factor) > screen_thresh) {
                // check if this operator can be applied
                if (det.fast_a_and_b_equal_b(sqop.cre()) and det.fast_a_and_b_eq_zero(ucre)) {
                    d = det;
                    double value = apply_operator_to_det_fast(d, sqop.ann(), sqop.cre());
                    new_terms[d] -= value * factor;
                    num_success_++;
                }
            } else {
                break;
            }
        }
    }
    return new_terms;
}

StateVector SparseExp::apply_operator_std(OperatorType op_type, const SparseOperator& sop,
                                          const StateVector& state0, double screen_thresh) {
    local_timer t;
    StateVector new_terms;
    Determinant new_d;

    double max_c = 0.0;
    for (const auto& [_, c] : state0) {
        max_c = std::max(max_c, std::fabs(c));
    }

    // loop over all the operators
    for (const auto& [sqop, coefficient] : sop.elements()) {
        if (std::abs(coefficient * max_c) < screen_thresh)
            continue;
        // create a mask for screening determinants according to the
        // creation operators this mask looks only at creation operators
        // that are not preceeded by annihilation operators
        const Determinant ucre = sqop.cre() - sqop.ann();
        // loop over all determinants
        for (const auto& [d, c] : state0) {
            const double factor = coefficient * c;
            // test if we can apply this operator to this determinant
            if (std::fabs(factor) > screen_thresh) {
                // check if this operator can be applied
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    new_d = d;
                    const double value = apply_operator_to_det_fast(new_d, sqop.cre(), sqop.ann());
                    new_terms[new_d] += value * factor;
                }
            }
        }
    }

    if (op_type != OperatorType::Antihermitian) {
        return new_terms;
    }

    // loop over all the operators
    for (const auto& [sqop, coefficient] : sop.elements()) {
        if (std::abs(coefficient * max_c) < screen_thresh)
            continue;
        // create a mask for screening determinants according to the
        // creation operators this mask looks only at creation operators
        // that are not preceeded by annihilation operators
        const Determinant ucre = sqop.ann() - sqop.cre();
        // loop over all determinants
        for (const auto& [d, c] : state0) {
            // test if we can apply this operator to this determinant
            // screen according to the product tau * c
            if (std::fabs(coefficient * c) > screen_thresh) {
                // check if this operator can be applied
                if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(ucre)) {
                    new_d = d;
                    const double value = apply_operator_to_det_fast(new_d, sqop.ann(), sqop.cre());
                    new_terms[new_d] -= value * coefficient * c;
                }
            }
        }
    }
    return new_terms;
}

std::map<std::string, double> SparseExp::timings() const { return timings_; }

} // namespace forte
