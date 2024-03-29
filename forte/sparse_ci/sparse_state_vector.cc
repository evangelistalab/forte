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

#include "helpers/helpers.h"
#include "helpers/timer.h"
#include "helpers/string_algorithms.h"
#include "integrals/active_space_integrals.h"

#include "sparse_ci/sparse_state_vector.h"

namespace forte {

StateVector apply_operator_impl(bool is_antihermitian, const SparseOperator& sop,
                                const StateVector& state, double screen_thresh);

StateVector::StateVector(const det_hash<double>& state_vec) : state_vec_(state_vec) {}

bool StateVector::operator==(const StateVector& lhs) const {
    return compare_hashes(map(), lhs.map());
}

std::string StateVector::str(int n) const {
    if (n == 0) {
        n = Determinant::norb();
    }
    std::string s;
    for (const auto& [det, c] : state_vec_) {
        if (std::fabs(c) > 1.0e-12) {
            s += forte::str(det, n) + " * " + std::to_string(c) + "\n";
        }
    }
    return s;
}

StateVector apply_operator_lin(const SparseOperator& sop, const StateVector& state,
                               double screen_thresh) {
    return apply_operator_impl(false, sop, state, screen_thresh);
}

StateVector apply_operator_antiherm(const SparseOperator& sop, const StateVector& state,
                                    double screen_thresh) {
    return apply_operator_impl(true, sop, state, screen_thresh);
}

StateVector apply_operator_impl(bool is_antihermitian, const SparseOperator& sop,
                                const StateVector& state, double screen_thresh) {
    if (screen_thresh < 0) {
        throw std::invalid_argument("apply_operator_impl:screen_thresh must be non-negative");
    }
    // make a copy of the state
    std::vector<std::tuple<double, double, Determinant>> state_sorted(state.size());
    for (size_t k = 0; const auto& [d, c] : state) {
        state_sorted[k] = std::make_tuple(std::fabs(c), c, d);
        ++k;
    }
    std::sort(state_sorted.rbegin(), state_sorted.rend());

    StateVector new_terms;

    Determinant d;
    // loop over all the operators (order does not matter)
    for (const auto& [sqop, coefficient] : sop.elements()) {
        if (coefficient == 0.0)
            continue;
        // create a mask for screening determinants according to the creation operators
        // this mask looks only at creation operators that are not preceeded by annihilation
        // operators
        const Determinant ucre = sqop.cre() - sqop.ann();
        // loop over all determinants
        for (const auto& [absc, c, det] : state_sorted) {
            // screen according to the product tau * c. Since the list is sorted, if the
            // coefficient is below the threshold, we can break the loop
            if (std::fabs(coefficient * c) > screen_thresh) {
                // check if this operator can be applied
                if (det.fast_a_and_b_equal_b(sqop.ann()) and det.fast_a_and_b_eq_zero(ucre)) {
                    d = det;
                    double value =
                        apply_operator_to_det_fast(d, sqop.cre(), sqop.ann()) * coefficient * c;
                    new_terms[d] += value;
                }
            } else {
                break;
            }
        }
    }

    if (is_antihermitian) {
        // loop over all the operators
        for (const auto& [sqop, coefficient] : sop.elements()) {
            if (coefficient == 0.0)
                continue;
            // create a mask for screening determinants according to the creation operators
            // this mask looks only at creation operators that are not preceeded by annihilation
            // operators
            const Determinant ucre = sqop.ann() - sqop.cre();
            // loop over all determinants
            for (const auto& [absc, c, det] : state_sorted) {
                // screen according to the product tau * c. Since the list is sorted, if the
                // coefficient is below the threshold, we can break the loop
                if (std::fabs(coefficient * c) > screen_thresh) {
                    // check if this operator can be applied
                    if (det.fast_a_and_b_equal_b(sqop.cre()) and det.fast_a_and_b_eq_zero(ucre)) {
                        d = det;
                        double value =
                            apply_operator_to_det_fast(d, sqop.ann(), sqop.cre()) * coefficient * c;
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

std::vector<double> get_projection(const SparseOperator& sop, const StateVector& ref,
                                   const StateVector& state) {
    local_timer t;
    std::vector<double> proj(sop.size(), 0.0);

    Determinant d;

    // loop over all the operators
    for (size_t n = 0; const auto& [sqop, coefficient] : sop.elements()) {
        double value = 0.0;
        // apply the operator op_n
        for (const auto& [det, c] : ref) {
            d = det;
            const double sign = apply_operator_to_det(d, sqop);
            if (sign != 0.0) {
                auto search = state.find(d);
                if (search != state.end()) {
                    value += sign * c * search->second;
                }
            }
        }
        proj[n] = value;
        n++;
    }
    return proj;
}

StateVector apply_number_projector(int na, int nb, const StateVector& state) {
    StateVector new_state;
    for (const auto& [det, c] : state) {
        if ((det.count_alfa() == na) and (det.count_beta() == nb) and (std::fabs(c) > 1.0e-12)) {
            new_state[det] = c;
        }
    }
    return new_state;
}

double overlap(const StateVector& left_state, const StateVector& right_state) {
    double overlap = 0.0;
    const auto& small_state = left_state.size() < right_state.size() ? left_state : right_state;
    const auto& large_state = left_state.size() < right_state.size() ? right_state : left_state;
    for (const auto& [det, c] : small_state) {
        auto it = large_state.find(det);
        if (it != large_state.end()) {
            overlap += it->second * c;
        }
    }
    return overlap;
}

} // namespace forte
