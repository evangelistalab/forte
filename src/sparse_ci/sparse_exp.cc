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

#include "sparse_ci/sparse_exp.h"

namespace forte {

size_t num_attempts_ = 0;
size_t num_success_ = 0;

SparseExp::SparseExp() {}

StateVector SparseExp::compute(const SparseOperator& sop, const StateVector& state0,
                               double scaling_factor, int maxk, double screen_thresh) {
    //    for (const auto& det_c : state0) {
    //        const Determinant& det = det_c.first;
    //        exp_hash_.add(det);
    //    }
    return compute_on_the_fly(sop, state0, scaling_factor, maxk, screen_thresh);
}

StateVector SparseExp::compute_on_the_fly(const SparseOperator& sop, const StateVector& state0,
                                          double scaling_factor, int maxk, double screen_thresh) {
    local_timer t;

    auto state = apply_exp_operator(sop, state0, scaling_factor, maxk, screen_thresh);
    on_the_fly_time_ += t.get();
    return state;
}

StateVector SparseExp::apply_exp_operator(const SparseOperator& sop, const StateVector& state0,
                                          double scaling_factor, int maxk, double screen_thresh) {
    double convergence_threshold_ = screen_thresh;

    local_timer t;
    StateVector exp_state(state0);
    StateVector state(state0);
    double factor = 1.0;
    for (int k = 1; k <= maxk; k++) {
        factor *= scaling_factor / static_cast<double>(k);
        StateVector new_terms = apply_operator_cached(sop, state, screen_thresh);
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

StateVector SparseExp::apply_exp_operator2(const SparseOperator& sop, const StateVector& state0,
                                           double scaling_factor, int maxk, double screen_thresh) {
    double convergence_threshold_ = screen_thresh;

    local_timer t;
    StateVector exp_state(state0);
    StateVector state(state0);
    double factor = 1.0;
    for (int k = 1; k <= maxk; k++) {
        factor *= scaling_factor / static_cast<double>(k);
        StateVector new_terms = apply_operator2(sop, state, screen_thresh);
        double norm = 0.0;
        double inf_norm = 0.0;
        for (const auto& det_c : new_terms) {
            exp_state[det_c.first] += factor * det_c.second;
            norm += std::pow(factor * det_c.second, 2.0);
            inf_norm = std::max(inf_norm, std::fabs(factor * det_c.second));
        }
        if (inf_norm < convergence_threshold_)
            break;
        state = new_terms;
    }
    return exp_state;
}

StateVector SparseExp::apply_operator_cached(const SparseOperator& sop, const StateVector& state0,
                                             double screen_thresh) {

    local_timer t;
    // make a copy of the state
    std::vector<std::tuple<double, double, Determinant>> state_sorted(state0.size());
    size_t k = 0;
    for (const auto& det_c : state0) {
        const Determinant& d = det_c.first;
        const double c = det_c.second;
        state_sorted[k] = std::make_tuple(std::fabs(c), c, d);
        ++k;
    }
    std::sort(state_sorted.rbegin(), state_sorted.rend());

    const auto& op_list = sop.op_list();

    StateVector new_terms;

    Determinant d_new;

    // loop over all determinants
    for (const auto& absc_c_det : state_sorted) {
        const double absc = std::get<0>(absc_c_det);
        if (absc > screen_thresh) {
            const double c = std::get<1>(absc_c_det);
            const Determinant& d = std::get<2>(absc_c_det);

            auto search = couplings_.find(d);

            if (search == couplings_.end()) {
                local_timer t_couplings;
                // we have to build the coupling list for this determinant
                std::vector<std::tuple<size_t, Determinant, double>> d_couplings;
                // loop over all the operators
                for (size_t n = 0, maxn = op_list.size(); n < maxn; n++) {
                    const SQOperator& sqop = op_list[n];
                    // create a mask for screening determinants according to the creation operators
                    // This mask looks only at creation operators that are not preceeded by
                    // annihilation operators
                    const Determinant ucre = sqop.cre() - sqop.ann();
                    // check if this operator can be applied
                    if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                        d_new = d;
                        double value = apply_op_safe(d_new, sqop.cre(), sqop.ann());
                        d_couplings.push_back(std::make_tuple(n, d_new, value));
                    }
                }
                couplings_[d] = d_couplings;
                couplings_time_ += t_couplings.get();
            }
            local_timer t_sum;
            // apply the operator
            const auto& d_couplings = couplings_[d];
            for (const auto& op_d_f : d_couplings) {
                const double value =
                    op_list[std::get<0>(op_d_f)].factor() * std::get<2>(op_d_f) * c;
                if (std::fabs(value) > screen_thresh)
                    new_terms[std::get<1>(op_d_f)] += value;
            }
            exp_time_ += t_sum.get();
        } else {
            break;
        }
    }

    if (sop.is_antihermitian()) {
        for (const auto& absc_c_det : state_sorted) {
            const double absc = std::get<0>(absc_c_det);
            if (absc > screen_thresh) {
                const double c = std::get<1>(absc_c_det);
                const Determinant& d = std::get<2>(absc_c_det);

                auto search = couplings_dexc_.find(d);

                if (search == couplings_dexc_.end()) {
                    local_timer t_couplings;
                    // we have to build the coupling list for this determinant
                    std::vector<std::tuple<size_t, Determinant, double>> d_couplings;
                    // loop over all the operators
                    for (size_t n = 0, maxn = op_list.size(); n < maxn; n++) {
                        const SQOperator& sqop = op_list[n];
                        // create a mask for screening determinants according to the annihilation
                        // operators. This mask looks only at annihilation operators that are not
                        // preceeded by creation operators
                        const Determinant ucre = sqop.ann() - sqop.cre();
                        // check if this operator can be applied
                        if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(ucre)) {
                            d_new = d;
                            double value = apply_op_safe(d_new, sqop.ann(), sqop.cre());
                            d_couplings.push_back(std::make_tuple(n, d_new, value));
                        }
                    }
                    couplings_dexc_[d] = d_couplings;
                    couplings_time_ += t_couplings.get();
                }
                local_timer t_sum;
                // apply the operator
                const auto& d_couplings = couplings_dexc_[d];
                for (const auto& op_d_f : d_couplings) {
                    const double value =
                        op_list[std::get<0>(op_d_f)].factor() * std::get<2>(op_d_f) * c;
                    if (std::fabs(value) > screen_thresh)
                        new_terms[std::get<1>(op_d_f)] -= value;
                }
                exp_time_ += t_sum.get();
            } else {
                break;
            }
        }
    }
    time_ += t.get();
    return new_terms;
}

StateVector SparseExp::apply_operator(const SparseOperator& sop, const StateVector& state0,
                                      double screen_thresh) {
    // make a copy of the state
    std::vector<std::tuple<double, double, Determinant>> state_sorted(state0.size());
    size_t k = 0;
    for (const auto& det_c : state0) {
        const Determinant& d = det_c.first;
        const double c = det_c.second;
        state_sorted[k] = std::make_tuple(std::fabs(c), c, d);
        ++k;
    }
    std::sort(state_sorted.rbegin(), state_sorted.rend());

    const auto& op_list = sop.op_list();

    StateVector new_terms;

    Determinant d;
    double c;
    double absc;

    // loop over all the operators
    for (const SQOperator& sqop : op_list) {
        if (sqop.factor() == 0.0)
            continue;
        // create a mask for screening determinants according to the creation operators
        // this mask looks only at creation operators that are not preceeded by annihilation
        // operators
        const Determinant ucre = sqop.cre() - sqop.ann();
        // loop over all determinants
        for (const auto& absc_c_det : state_sorted) {
            num_attempts_++;
            std::tie(absc, c, d) = absc_c_det;
            // screen according to the product tau * c
            if (std::fabs(sqop.factor() * c) > screen_thresh) {
                // check if this operator can be applied
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    const double value =
                        apply_op_safe(d, sqop.cre(), sqop.ann()) * sqop.factor() * c;
                    new_terms[d] += value;
                    num_success_++;
                }
            } else {
                break;
            }
        }
    }

    if (sop.is_antihermitian()) {
        // loop over all the operators
        for (const SQOperator& sqop : op_list) {
            if (sqop.factor() == 0.0)
                continue;
            // create a mask for screening determinants according to the creation operators
            // this mask looks only at creation operators that are not preceeded by annihilation
            // operators
            const Determinant ucre = sqop.ann() - sqop.cre();
            // loop over all determinants
            for (const auto& absc_c_det : state_sorted) {
                num_attempts_++;

                std::tie(absc, c, d) = absc_c_det;
                // screen according to the product tau * c
                if (std::fabs(sqop.factor() * c) > screen_thresh) {
                    // check if this operator can be applied
                    if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(ucre)) {
                        double value = apply_op_safe(d, sqop.ann(), sqop.cre()) * sqop.factor() * c;
                        new_terms[d] -= value;
                        num_success_++;
                    }
                } else {
                    break;
                }
            }
        }
    }
    return new_terms;
}

StateVector SparseExp::apply_operator2(const SparseOperator& sop, const StateVector& state0,
                                       double screen_thresh) {
    local_timer t;
    const auto& op_list = sop.op_list();

    StateVector new_terms;

    // loop over all the operators
    for (const SQOperator& sqop : op_list) {
        if (sqop.factor() == 0.0)
            continue;
        // create a mask for screening determinants according to the creation operators
        // this mask looks only at creation operators that are not preceeded by annihilation
        // operators
        const Determinant ucre = sqop.cre() - sqop.ann();
        Determinant new_d;
        // loop over all determinants
        for (const auto& det_c : state0) {
            const Determinant& d = det_c.first;
            const double c = det_c.second;
            // test if we can apply this operator to this determinant
            // screen according to the product tau * c
            if (std::fabs(sqop.factor() * c) > screen_thresh) {
                // check if this operator can be applied
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    new_d = d;
                    double value = apply_op_safe(new_d, sqop.cre(), sqop.ann()) * sqop.factor() * c;
                    new_terms[new_d] += value;
                }
            }
        }
    }

    if (sop.is_antihermitian()) {
        // loop over all the operators
        for (const SQOperator& sqop : op_list) {
            if (sqop.factor() == 0.0)
                continue;
            // create a mask for screening determinants according to the creation operators
            // this mask looks only at creation operators that are not preceeded by annihilation
            // operators
            const Determinant ucre = sqop.ann() - sqop.cre();
            Determinant new_d;
            // loop over all determinants
            for (const auto& det_c : state0) {
                const Determinant& d = det_c.first;
                const double c = det_c.second;
                // test if we can apply this operator to this determinant
                // screen according to the product tau * c
                if (std::fabs(sqop.factor() * c) > screen_thresh) {
                    // check if this operator can be applied
                    if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(ucre)) {
                        new_d = d;
                        double value =
                            apply_op_safe(new_d, sqop.ann(), sqop.cre()) * sqop.factor() * c;
                        new_terms[new_d] -= value;
                    }
                }
            }
        }
    }
    return new_terms;
}

std::map<std::string, double> SparseExp::time() const {
    std::map<std::string, double> t;
    t["time"] = time_;
    t["couplings_time"] = couplings_time_;
    t["exp_time"] = exp_time_;
    t["on_the_fly_time"] = on_the_fly_time_;
    t["num_attempts"] = static_cast<double>(num_attempts_);
    t["num_success"] = static_cast<double>(num_success_);
    return t;
}

} // namespace forte
