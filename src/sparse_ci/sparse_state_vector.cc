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
#include <cmath>
#include <numeric>

#include "helpers/timer.h"
#include "helpers/string_algorithms.h"
#include "integrals/active_space_integrals.h"

#include "sparse_ci/sparse_state_vector.h"

namespace forte {

StateVector::StateVector() { /* std::cout << "Created a StateVector object" << std::endl; */
}

StateVector::StateVector(const det_hash<double>& state_vec) : state_vec_(state_vec) {}

std::string StateVector::str(int n) const {
    if (n == 0) {
        n = Determinant::get_nbits_half();
    }
    std::string s;
    for (const auto& c_d : state_vec_) {
        s += forte::str(c_d.first, n) + " * " + std::to_string(c_d.second) + "\n";
    }
    return s;
}

 StateVector apply_operator_safe(SparseOperator& sop, const StateVector& state) {
    StateVector new_state;
    const auto& op_list = sop.op_list();
    Determinant d;
    for (const auto& det_c : state) {
        const double c = det_c.second;
        for (const SQOperator& sqop : op_list) {
            d = det_c.first;
            double sign = apply_op(d, sqop.cre(), sqop.ann());
            if (sign != 0.0) {
                new_state[d] += sqop.factor() * sign * c;
            }
        }
    }
    return new_state;
}

StateVector apply_operator(SparseOperator& sop, const StateVector& state0, double screen_thresh) {
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
            std::tie(absc, c, d) = absc_c_det;
            // screen according to the product tau * c
            if (std::fabs(sqop.factor() * c) > screen_thresh) {
                // check if this operator can be applied
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    double value = apply_op_safe(d, sqop.cre(), sqop.ann()) * sqop.factor() * c;
                    new_terms[d] += value;
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
                std::tie(absc, c, d) = absc_c_det;
                // screen according to the product tau * c
                if (std::fabs(sqop.factor() * c) > screen_thresh) {
                    // check if this operator can be applied
                    if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(ucre)) {
                        double value = apply_op_safe(d, sqop.ann(), sqop.cre()) * sqop.factor() * c;
                        new_terms[d] -= value;
                    }
                } else {
                    break;
                }
            }
        }
    }
    return new_terms;
}

std::vector<double> get_projection(SparseOperator& sop, const StateVector& ref,
                                   const StateVector& state0) {
    local_timer t;
    std::vector<double> proj(sop.nterms(), 0.0);

    const auto& op_list = sop.op_list();

    Determinant d;

    // loop over all the operators
    for (size_t n = 0, nterms = sop.nterms(); n < nterms; n++) {
        double value = 0.0;
        const SQOperator& sqop = op_list[n];

        // apply the operator op_n
        for (const auto& det_c : ref) {
            const double c = det_c.second;
            d = det_c.first;

            const double sign = apply_op(d, sqop.cre(), sqop.ann());
            if (sign != 0.0) {
                auto search = state0.find(d);
                if (search != state0.end()) {
                    value += sign * c * search->second;
                }
            }
        }

        proj[n] = value;
    }
    return proj;
}

StateVector apply_number_projector(int na, int nb, StateVector& state) {
    StateVector new_state;
    for (const auto& det_c : state) {
        if ((det_c.first.count_alfa() == na) and (det_c.first.count_beta() == nb) and
            (std::fabs(det_c.second) > 1.0e-12)) {
            new_state[det_c.first] = det_c.second;
        }
    }
    return new_state;
}

double overlap(StateVector& left_state, StateVector& right_state) {
    double overlap = 0.0;
    for (const auto& det_c_r : right_state) {
        auto it = left_state.find(det_c_r.first);
        if (it != left_state.end()) {
            overlap += it->second * det_c_r.second;
        }
    }
    return overlap;
}

} // namespace forte
