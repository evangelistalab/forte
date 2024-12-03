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
#include "sparse_ci/determinant.hpp"
#include "sparse_ci/sparse_state.h"

namespace forte {

double SparseState::norm() const {
    double n = 0.0;
    for (const auto& [det, c] : elements()) {
        n += std::norm(c);
    }
    return std::sqrt(n);
}

SparseState apply_operator_impl(bool is_antihermitian, const SparseOperator& sop,
                                const SparseState& state, double screen_thresh);

std::string SparseState::str(int n) const {
    if (n == 0) {
        n = Determinant::norb();
    }
    std::string s;
    for (const auto& [det, c] : elements()) {
        if (std::abs(c) > 1.0e-8) {
            s += forte::str(det, n) + " * " + to_string_with_precision(c, 8) + "\n";
        }
    }
    return s;
}

SparseState apply_operator_lin(const SparseOperator& sop, const SparseState& state,
                               double screen_thresh) {
    return apply_operator_impl(false, sop, state, screen_thresh);
}

SparseState apply_operator_antiherm(const SparseOperator& sop, const SparseState& state,
                                    double screen_thresh) {
    return apply_operator_impl(true, sop, state, screen_thresh);
}

SparseState apply_operator_impl(bool is_antihermitian, const SparseOperator& sop,
                                const SparseState& state, double screen_thresh) {
    if (screen_thresh < 0) {
        throw std::invalid_argument("apply_operator_impl:screen_thresh must be non-negative");
    }
    // make a copy of the state
    std::vector<std::pair<sparse_scalar_t, Determinant>> state_sorted;
    state_sorted.reserve(state.size());
    std::transform(state.begin(), state.end(), std::back_inserter(state_sorted),
                   [](const auto& pair) { return std::make_pair(pair.second, pair.first); });

    // Sorting the vector based on the decreasing absolute value of the sparse_scalar_t
    std::sort(state_sorted.begin(), state_sorted.end(),
              [](const std::pair<sparse_scalar_t, Determinant>& a,
                 const std::pair<sparse_scalar_t, Determinant>& b) {
                  return std::abs(a.first) > std::abs(b.first);
              });

    // Find the largest coefficient in absolute value
    auto max_c = state_sorted.size() > 0 ? std::abs(state_sorted[0].first) : 0.0;

    // make a copy of the operator and sort it according to decreasing values of |t|
    std::vector<std::pair<sparse_scalar_t, SQOperatorString>> op_sorted;
    for (const auto& [sqop, t] : sop.elements()) {
        if (std::abs(t * max_c) > screen_thresh)
            op_sorted.push_back(std::make_pair(t, sqop));
    }
    // Sorting the vector based on the decreasing absolute value of the sparse_scalar_t
    std::sort(op_sorted.begin(), op_sorted.end(),
              [](const std::pair<sparse_scalar_t, SQOperatorString>& a,
                 const std::pair<sparse_scalar_t, SQOperatorString>& b) {
                  return std::abs(a.first) > std::abs(b.first);
              });

    SparseState new_terms;
    Determinant new_det;

    for (const auto& [t, sqop] : op_sorted) {
        // mask for screening determinants according to the uncontracted creation operators
        const Determinant ucre = sqop.cre() - sqop.ann();
        const Determinant sign_mask = compute_sign_mask(sqop.cre(), sqop.ann());
        const auto screen_thresh_div_t = screen_thresh / std::abs(t);
        // Find the first determinant below the threshold using bisection
        auto last = std::lower_bound(
            state_sorted.begin(), state_sorted.end(), screen_thresh_div_t,
            [](const auto& pair, double threshold) { return std::abs(pair.first) > threshold; });
        for (auto it = state_sorted.begin(); it != last; ++it) {
            const auto& [c, det] = *it;
            if (det.fast_can_apply_operator(sqop.ann(), ucre)) {
                auto value =
                    faster_apply_operator_to_det(det, new_det, sqop.cre(), sqop.ann(), sign_mask);
                new_terms[new_det] += value * t * c;
            }
        }
    }
    if (not is_antihermitian) {
        return new_terms;
    }
    for (const auto& [t, sqop] : op_sorted) {
        // mask for screening determinants according to the uncontracted annihilation operators
        const Determinant uann = sqop.ann() - sqop.cre();
        const Determinant sign_mask = compute_sign_mask(sqop.ann(), sqop.cre());
        const auto screen_thresh_div_t = screen_thresh / std::abs(t);
        // Find the first determinant below the threshold using bisection
        auto last = std::lower_bound(
            state_sorted.begin(), state_sorted.end(), screen_thresh_div_t,
            [](const auto& pair, double threshold) { return std::abs(pair.first) > threshold; });
        for (auto it = state_sorted.begin(); it != last; ++it) {
            const auto& [c, det] = *it;
            if (det.fast_can_apply_operator(sqop.cre(), uann)) {
                auto value =
                    faster_apply_operator_to_det(det, new_det, sqop.ann(), sqop.cre(), sign_mask);
                new_terms[new_det] -= value * t * c;
            }
        }
    }
    return new_terms;
}

std::vector<sparse_scalar_t> get_projection(const SparseOperatorList& sop, const SparseState& ref,
                                            const SparseState& state) {
    local_timer t;
    std::vector<sparse_scalar_t> proj(sop.size(), 0.0);

    Determinant d;

    // loop over all the operators
    for (size_t n = 0; const auto& [sqop, coefficient] : sop.elements()) {
        sparse_scalar_t value = 0.0;
        // apply the operator op_n
        for (const auto& [det, c] : ref) {
            d = det;
            const auto sign = apply_operator_to_det(d, sqop);
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

SparseState apply_number_projector(int na, int nb, const SparseState& state) {
    SparseState new_state;
    for (const auto& [det, c] : state) {
        if ((det.count_alfa() == na) and (det.count_beta() == nb) and (std::abs(c) > 1.0e-12)) {
            new_state[det] = c;
        }
    }
    return new_state;
}

sparse_scalar_t overlap(const SparseState& left_state, const SparseState& right_state) {
    return left_state.dot(right_state);
}

sparse_scalar_t spin2_sparse(const SparseState& left_state, const SparseState& right_state) {
    sparse_scalar_t s2 = 0.0;
    for (const auto& [deti, ci] : left_state) {
        for (const auto& [detj, cj] : right_state) {
            s2 += ci * cj * spin2(deti, detj);
        }
    }
    return s2;
}

SparseState normalize(const SparseState& state) {
    SparseState new_state;
    const auto n = state.norm();
    for (const auto& [det, c] : state) {
        new_state[det] = c / n;
    }
    return new_state;
}


} // namespace forte

// SparseState apply_operator_impl_original(bool is_antihermitian, const SparseOperator& sop,
//                                          const SparseState& state, double screen_thresh) {
//     if (screen_thresh < 0) {
//         throw std::invalid_argument("apply_operator_impl:screen_thresh must be non-negative");
//     }
//     // make a copy of the state
//     std::vector<std::pair<double, Determinant>> state_sorted;
//     state_sorted.reserve(state.size());
//     std::transform(state.begin(), state.end(), std::back_inserter(state_sorted),
//                    [](const auto& pair) { return std::make_pair(pair.second, pair.first); });

//     // Sorting the vector based on the decreasing absolute value of the double
//     std::sort(state_sorted.begin(), state_sorted.end(),
//               [](const std::pair<double, Determinant>& a, const std::pair<double, Determinant>&
//               b) {
//                   return std::abs(a.first) > std::abs(b.first);
//               });

//     // Find the largest coefficient in absolute value
//     auto max_c = state_sorted.size() > 0 ? std::abs(state_sorted[0].first) : 0.0;

//     // make a copy of the operator and sort it according to decreasing values of |t|
//     std::vector<std::pair<double, SQOperatorString>> op_sorted;
//     for (const auto& [sqop, t] : sop.elements()) {
//         if (std::abs(t * max_c) > screen_thresh)
//             op_sorted.push_back(std::make_pair(t, sqop));
//     }
//     // Sorting the vector based on the decreasing absolute value of the double
//     std::sort(op_sorted.begin(), op_sorted.end(),
//               [](const std::pair<double, SQOperatorString>& a,
//                  const std::pair<double, SQOperatorString>& b) {
//                   return std::abs(a.first) > std::abs(b.first);
//               });

//     SparseState new_terms;
//     Determinant new_det;
//     Determinant ucre;

//     for (const auto& [t, sqop] : op_sorted) {
//         // mask for screening determinants according to the uncontracted creation operators
//         const Determinant ucre = sqop.cre() - sqop.ann();
//         // const Determinant sign_mask = compute_sign_mask(sqop.cre(), sqop.ann());
//         const auto screen_thresh_div_t = screen_thresh / std::abs(t);
//         // Find the first determinant below the threshold using bisection
//         auto last =
//             std::lower_bound(state_sorted.begin(), state_sorted.end(), screen_thresh_div_t,
//                              [](const std::pair<double, Determinant>& pair, double threshold) {
//                                  return std::abs(pair.first) > threshold;
//                              });
//         for (auto it = state_sorted.begin(); it != last; ++it) {
//             const auto& [c, det] = *it;
//             if (det.fast_can_apply_operator(sqop.ann(), ucre)) {
//                 new_det = det;
//                 auto value = fast_apply_operator_to_det(new_det, sqop.cre(), sqop.ann());
//                 new_terms[new_det] += value * t * c;
//             }
//         }
//     }
//     if (not is_antihermitian) {
//         return new_terms;
//     }
//     for (const auto& [t, sqop] : op_sorted) {
//         // mask for screening determinants according to the uncontracted annihilation operators
//         const Determinant ucre = sqop.ann() - sqop.cre();
//         const auto screen_thresh_div_t = screen_thresh / std::abs(t);
//         // Find the first determinant below the threshold using bisection
//         auto last =
//             std::lower_bound(state_sorted.begin(), state_sorted.end(), screen_thresh_div_t,
//                              [](const std::pair<double, Determinant>& pair, double threshold) {
//                                  return std::abs(pair.first) > threshold;
//                              });
//         for (auto it = state_sorted.begin(); it != last; ++it) {
//             const auto& [c, det] = *it;
//             if (det.fast_can_apply_operator(sqop.cre(), ucre)) {
//                 new_det = det;
//                 auto value = fast_apply_operator_to_det(new_det, sqop.ann(), sqop.cre());
//                 new_terms[new_det] -= value * t * c;
//             }
//         }
//     }
//     return new_terms;
// }