/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <thread>
#include <future>

#include "helpers/helpers.h"
#include "helpers/timer.h"
#include "helpers/string_algorithms.h"
#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.hpp"
#include "sparse_ci/sparse_state.h"

namespace forte {

std::vector<SparseState> split_state(const SparseState& state, size_t num_chunks);

// This is a naive implementation of the operator application that is used for testing
SparseState apply_operator_impl_naive(bool is_antihermitian, const SparseOperator& sop,
                                      const SparseState& state, double screen_thresh);

// This is the grouped implementation of the operator application. Fast, but scaling is not optimal.
SparseState apply_operator_impl_grouped(bool is_antihermitian, const SparseOperator& sop,
                                        const SparseState& state, double screen_thresh);

// The default implementation is the grouped implementation with grouping into alfa strings
SparseState apply_operator_impl_grouped_string(bool is_antihermitian, const SparseOperator& sop,
                                               const SparseState& state, double screen_thresh);

SparseState apply_operator_lin(const SparseOperator& sop, const SparseState& state,
                               double screen_thresh) {
    size_t num_threads = std::thread::hardware_concurrency();
    size_t nops = sop.size() * state.size();
    size_t nops_thresh = 10000;
    // hardware_concurrency() returns 0 if not well defined
    if (num_threads <= 1 || nops < nops_thresh){
        return apply_operator_impl_grouped_string(false, sop, state, screen_thresh);
    }
    auto chunks = split_state(state, num_threads);
    std::vector<std::future<SparseState>> futures;
    futures.reserve(num_threads);
    for (auto& chunk : chunks) {
        futures.emplace_back(std::async(std::launch::async, [&]() {
            return apply_operator_impl_grouped_string(false, sop, chunk, screen_thresh);
        }));
    }
    SparseState result;
    for (auto& future : futures) {
        result += future.get();
    }
    return result;
}

SparseState apply_operator_antiherm(const SparseOperator& sop, const SparseState& state,
                                    double screen_thresh) {
    size_t num_threads = std::thread::hardware_concurrency();
    size_t nops = sop.size() * state.size();
    size_t nops_thresh = 10000;
    // hardware_concurrency() returns 0 if not well defined
    if (num_threads <= 1 || nops < nops_thresh){
        return apply_operator_impl_grouped_string(true, sop, state, screen_thresh);
    }
    auto chunks = split_state(state, num_threads);
    std::vector<std::future<SparseState>> futures;
    futures.reserve(num_threads);
    for (auto& chunk : chunks) {
        futures.emplace_back(std::async(std::launch::async, [&]() {
            return apply_operator_impl_grouped_string(true, sop, chunk, screen_thresh);
        }));
    }
    SparseState result;
    for (auto& future : futures) {
        result += future.get();
    }
    return result;
}

std::vector<SparseState> split_state(const SparseState& state, size_t num_chunks) {
    if (num_chunks == 0 || state.size() == 0) {
        return {};
    }
    const size_t total_elements =state.size();
    const size_t chunk_size = total_elements / num_chunks;
    const size_t remainder = total_elements % num_chunks;

    std::vector<SparseState> chunks;
    chunks.reserve(num_chunks);
    auto it = state.elements().begin();
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t this_chunk_size = chunk_size + (chunk_idx < remainder);
        SparseState chunk;
        for (size_t j = 0; j < this_chunk_size; ++j, ++it) {
            chunk.insert(it->first, it->second);
        }
        chunks.emplace_back(std::move(chunk));
    }
    return chunks;
}

// This is a naive implementation of the operator application that is used for testing
// It has a cost complexity of O(M * N) where M is the number of terms in the operator
// and N is the number of terms in the state.
SparseState apply_operator_impl_naive(bool is_antihermitian, const SparseOperator& sop,
                                      const SparseState& state, double screen_thresh) {
    if (screen_thresh < 0) {
        throw std::invalid_argument("apply_operator_impl:screen_thresh must be non-negative");
    }
    SparseState new_terms; // the new state
    Determinant new_det;   // a temporary determinant to store the result of the operator
    Determinant sign_mask; // a temporary determinant to store the sign mask
    Determinant idx;       // a temporary determinant to store the index of the determinant
    for (const auto& [sqop, t] : sop) {
        compute_sign_mask(sqop.cre(), sqop.ann(), sign_mask, idx);
        for (const auto& [det, c] : state) {
            if (det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
                auto value =
                    faster_apply_operator_to_det(det, new_det, sqop.cre(), sqop.ann(), sign_mask);
                new_terms[new_det] += value * t * c;
            }
        }
    }

    if (not is_antihermitian) {
        return new_terms;
    }

    for (const auto& [sqop, t] : sop) {
        compute_sign_mask(sqop.ann(), sqop.cre(), sign_mask, idx);
        for (const auto& [det, c] : state) {
            if (det.faster_can_apply_operator(sqop.ann(), sqop.cre())) {
                auto value =
                    faster_apply_operator_to_det(det, new_det, sqop.ann(), sqop.cre(), sign_mask);
                new_terms[new_det] -= value * t * c;
            }
        }
    }
    return new_terms;
}

// This is a kernel that applies the operator to the state using a grouped approach
// It has a lower cost complexity
// It assumes that the operator is grouped by the annihilation operators and that these are prepared
// in another function calling this kernel
template <bool positive>
void apply_operator_kernel(const auto& sop_groups, const auto& state_sorted,
                           const auto& screen_thresh, auto& new_terms) {
    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
    for (const auto& [sqop_ann, sqop_group] : sop_groups) {
        for (const auto& [det, c] : state_sorted) {
            if (det.fast_a_and_b_equal_b(sqop_ann)) {
                // loop over the creation operators in this group
                for (const auto& [sqop_cre, t] : sqop_group) {
                    if (det.fast_a_and_b_minus_c_eq_zero(sqop_cre, sqop_ann)) {
                        if (std::abs(c * t) > screen_thresh) {
                            compute_sign_mask(sqop_cre, sqop_ann, sign_mask, idx);
                            const auto value = faster_apply_operator_to_det(det, new_det, sqop_cre,
                                                                            sqop_ann, sign_mask);
                            if constexpr (positive) {
                                new_terms[new_det] += value * t * c;
                            } else {
                                new_terms[new_det] -= value * t * c;
                            }
                        }
                    }
                }
            }
        }
    }
}

// This is the grouped implementation of the operator application. It mostly prepares the operator
// and state and then calls the kernel to apply the operator
SparseState apply_operator_impl_grouped(bool is_antihermitian, const SparseOperator& sop,
                                        const SparseState& state, double screen_thresh) {
    if (screen_thresh < 0) {
        throw std::invalid_argument(
            "apply_operator_impl_grouped:screen_thresh must be non-negative");
    }
    SparseState new_terms;

    // make a copy of the state with the determinants sorted
    // somehow this makes the code faster
    std::vector<std::pair<Determinant, sparse_scalar_t>> state_sorted(state.begin(), state.end());
    std::sort(state_sorted.begin(), state_sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Group the operators by common annihilation strings
    std::unordered_map<Determinant, std::vector<std::pair<Determinant, sparse_scalar_t>>,
                       Determinant::Hash>
        sop_groups;
    for (const auto& [sqop, t] : sop.elements()) {
        sop_groups[sqop.ann()].emplace_back(sqop.cre(), t);
    }

    // Call the kernel to apply the operator (adding the result)
    apply_operator_kernel<true>(sop_groups, state_sorted, screen_thresh, new_terms);

    if (not is_antihermitian) {
        return new_terms;
    }

    // Group the operators by common creation strings
    // Here we swap the annihilation and creation operators for the antihermitian case
    sop_groups.clear();
    for (const auto& [sqop, t] : sop.elements()) {
        sop_groups[sqop.cre()].emplace_back(sqop.ann(), t);
    }

    // Call the kernel to apply the operator (subtracting the result)
    apply_operator_kernel<false>(sop_groups, state_sorted, screen_thresh, new_terms);

    return new_terms;
}

// This is a kernel that applies the operator to the state using a grouped approach
// It has a lower cost complexity
// It assumes that the operator is grouped by the annihilation operators and that these are prepared
// in another function calling this kernel
template <bool positive>
void apply_operator_kernel_string(const auto& sop_groups, const auto& state_groups,
                                  const auto& screen_thresh, auto& new_terms) {
    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
    for (const auto& [sqop_ann_a, sqop_group] : sop_groups) {
        for (const auto& [det_a, state_group] : state_groups) {
            // can we annihilate the alfa string?
            if (det_a.fast_a_and_b_equal_b(sqop_ann_a)) {
                // loop over the creation operators in this group
                for (const auto& [sqop_ann, sqop_cre, t] : sqop_group) {
                    for (const auto& [det, c] : state_group) {
                        if (det.faster_can_apply_operator(sqop_cre, sqop_ann)) {
                            if (std::abs(c * t) > screen_thresh) {
                                compute_sign_mask(sqop_cre, sqop_ann, sign_mask, idx);
                                const auto value = faster_apply_operator_to_det(
                                    det, new_det, sqop_cre, sqop_ann, sign_mask);
                                if constexpr (positive) {
                                    new_terms[new_det] += value * t * c;
                                } else {
                                    new_terms[new_det] -= value * std::conj(t) * c;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// This is the grouped implementation of the operator application. It mostly prepares the
// operator and state and then calls the kernel to apply the operator
SparseState apply_operator_impl_grouped_string(bool is_antihermitian, const SparseOperator& sop,
                                               const SparseState& state, double screen_thresh) {
    if (screen_thresh < 0) {
        throw std::invalid_argument(
            "apply_operator_impl_grouped:screen_thresh must be non-negative");
    }
    SparseState new_terms;

    // Group the determinants by common alfa strings
    std::unordered_map<String, std::vector<std::pair<Determinant, sparse_scalar_t>>, String::Hash>
        state_groups;
    for (const auto& [det, c] : state) {
        state_groups[det.get_alfa_bits()].emplace_back(det, c);
    }

    // Group the operators by common alfa annihilation strings
    std::unordered_map<String, std::vector<std::tuple<Determinant, Determinant, sparse_scalar_t>>,
                       String::Hash>
        sop_groups;
    for (const auto& [sqop, t] : sop.elements()) {
        sop_groups[sqop.ann().get_alfa_bits()].emplace_back(sqop.ann(), sqop.cre(), t);
    }

    // Call the kernel to apply the operator (adding the result)
    apply_operator_kernel_string<true>(sop_groups, state_groups, screen_thresh, new_terms);

    if (not is_antihermitian) {
        return new_terms;
    }

    // Group the operators by common alfa creation strings
    // Here we swap the annihilation and creation operators for the antihermitian case
    sop_groups.clear();
    for (const auto& [sqop, t] : sop.elements()) {
        sop_groups[sqop.cre().get_alfa_bits()].emplace_back(sqop.cre(), sqop.ann(), t);
    }

    // Call the kernel to apply the operator (subtracting the result)
    apply_operator_kernel_string<false>(sop_groups, state_groups, screen_thresh, new_terms);

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

sparse_scalar_t spin2(const SparseState& left_state, const SparseState& right_state) {
    sparse_scalar_t s2 = 0.0;
    for (const auto& [deti, ci] : left_state) {
        for (const auto& [detj, cj] : right_state) {
            s2 += conjugate(ci) * cj * spin2(deti, detj);
        }
    }
    return s2;
}

SparseState normalize(const SparseState& state) {
    SparseState new_state(state);
    const auto n = state.norm();
    new_state /= n;
    return new_state;
}

} // namespace forte
