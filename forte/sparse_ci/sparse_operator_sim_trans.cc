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

#include <future>
#include <vector>
#include <thread>
#include <random>
#include <algorithm> // For std::shuffle

#include "sparse_operator.h"

namespace forte {

std::vector<SparseOperator> split_operator(const SparseOperator& O, size_t num_chunks);

SparseOperator process_chunk(SparseOperator& chunk, const SQOperatorString& T_op,
                             std::pair<sparse_scalar_t, sparse_scalar_t> c1_pair,
                             std::pair<sparse_scalar_t, sparse_scalar_t> c2_pair,
                             sparse_scalar_t sigma, bool add, double screen_threshold);

/// Parallel implementation of the similarity transformation
void parallel_sim_trans_impl(SparseOperator& O, const SQOperatorString& T_op,
                             std::pair<sparse_scalar_t, sparse_scalar_t> c1_pair,
                             std::pair<sparse_scalar_t, sparse_scalar_t> c2_pair,
                             sparse_scalar_t sigma, bool add, double screen_threshold);

/// @brief Implementation of the linearized similarity transformation [(1 - T) O (1 + T)]
void sim_trans_lin_impl(SparseOperator& O, const SQOperatorString& T_op, sparse_scalar_t theta,
                        double screen_threshold);

/// General implementation of the similarity transformation
void sim_trans_impl(SparseOperator& O, const SQOperatorString& T_op,
                    std::pair<sparse_scalar_t, sparse_scalar_t> c1_pair,
                    std::pair<sparse_scalar_t, sparse_scalar_t> c2_pair, sparse_scalar_t sigma,
                    bool add, double screen_threshold);

/// General implementation of the similarity transformation
void serial_sim_trans_impl(SparseOperator& O, const SQOperatorString& T_op,
                           std::pair<sparse_scalar_t, sparse_scalar_t> c1_pair,
                           std::pair<sparse_scalar_t, sparse_scalar_t> c2_pair,
                           sparse_scalar_t sigma, bool add, double screen_threshold);

/// @brief Wrapper function to apply a similarity transformation to an operator
void sim_trans_impl(SparseOperator& O, const SQOperatorString& T_op,
                    std::pair<sparse_scalar_t, sparse_scalar_t> c1_pair,
                    std::pair<sparse_scalar_t, sparse_scalar_t> c2_pair, sparse_scalar_t sigma,
                    bool add, double screen_threshold) {
    parallel_sim_trans_impl(O, T_op, c1_pair, c2_pair, sigma, add, screen_threshold);
}

/// @brief Implementation of the similarity transformation for multiple threads
void parallel_sim_trans_impl(SparseOperator& O, const SQOperatorString& T_op,
                             std::pair<sparse_scalar_t, sparse_scalar_t> c1_pair,
                             std::pair<sparse_scalar_t, sparse_scalar_t> c2_pair,
                             sparse_scalar_t sigma, bool add, double screen_threshold) {
    // Determine the number of threads to use
    size_t num_threads = std::thread::hardware_concurrency();

    // Split O into chunks
    auto chunks = split_operator(O, num_threads);

    // Vector to store futures
    std::vector<std::future<SparseOperator>> futures;

    // Launch async tasks for each chunk
    for (auto& chunk : chunks) {
        futures.push_back(std::async(std::launch::async, [=, &T_op]() mutable {
            return process_chunk(chunk, T_op, c1_pair, c2_pair, sigma, add, screen_threshold);
        }));
    }

    SparseOperator result;
    for (auto& fut : futures) {
        result += fut.get(); // Retrieve and accumulate the processed chunks
    }

    O = std::move(result);
}

/// @brief Split an operator into chunks and return a vector of the chunks
std::vector<SparseOperator> split_operator(const SparseOperator& O, size_t num_chunks) {
    if (num_chunks == 0 || O.size() == 0) {
        return {};
    }

    const size_t total_elements = O.size();
    const size_t chunk_size = total_elements / num_chunks;
    const size_t remainder = total_elements % num_chunks;

    std::vector<SparseOperator> chunks;
    chunks.reserve(num_chunks);
    auto it = O.elements().begin();
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t this_chunk_size = chunk_size + (chunk_idx < remainder);
        SparseOperator chunk;
        for (size_t j = 0; j < this_chunk_size; ++j, ++it) {
            chunk.insert(it->first, it->second);
        }
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

/// @brief Process a chunk of the operator using the similarity transformation
SparseOperator process_chunk(SparseOperator& chunk, const SQOperatorString& T_op,
                             std::pair<sparse_scalar_t, sparse_scalar_t> c1_pair,
                             std::pair<sparse_scalar_t, sparse_scalar_t> c2_pair,
                             sparse_scalar_t sigma, bool add, double screen_threshold) {
    // Process the chunk using sim_trans_impl
    serial_sim_trans_impl(chunk, T_op, c1_pair, c2_pair, sigma, add, screen_threshold);
    return chunk;
}

/// @brief  Implementation of the similarity transformation for  a single thread
/// This function is used to apply the similarity transformation to all the terms in the operator
/// This function modifies the operator O in place
void serial_sim_trans_impl(SparseOperator& O, const SQOperatorString& T_op,
                           std::pair<sparse_scalar_t, sparse_scalar_t> c1_pair,
                           std::pair<sparse_scalar_t, sparse_scalar_t> c2_pair,
                           sparse_scalar_t sigma, bool add, double screen_threshold) {
    // Td = T^dagger
    auto Td_op = T_op.adjoint();
    // TTd = T T^dagger (gives a number operator if there are no repeated indices)
    auto TTd = T_op * Td_op;
    // TdT = T^dagger T (gives a number operator if there are no repeated indices)
    auto TdT = Td_op * T_op;
    // T_n = T number component, T_nn = T non-number component
    auto T_n = T_op.number_component();
    auto T_nn = T_op.non_number_component();
    // Td_n = T^dagger number component, Td_nn = T^dagger non-number component
    auto Td_n = Td_op.number_component();
    auto Td_nn = Td_op.non_number_component();

    SparseOperator R;
    sparse_scalar_t c1, c2;
    Determinant nn_Op_cre;
    Determinant nn_Op_ann;

    for (const auto& [O_op, O_c] : O.elements()) {
        const auto O_T_commutator_type = commutator_type(O_op, T_op);
        const auto O_Td_commutator_type = commutator_type(O_op, Td_op);
        // if both commutators are zero, then we can skip this term
        if (O_T_commutator_type == CommutatorType::Commute and
            O_Td_commutator_type == CommutatorType::Commute) {
            continue;
        }

        // // Check if TOTd or TdOT are nonzero
        auto O_n = O_op.number_component();
        auto O_nn = O_op.non_number_component();
        int alpha = 1;

        // Simplified checks using lambda expressions
        auto check_ABA_case = [&](const auto& A, const auto& B) {
            return O_nn.cre().fast_a_and_b_equal_b(A.cre()) &&
                   O_nn.ann().fast_a_and_b_equal_b(A.ann()) &&
                   O_nn.cre().fast_a_and_b_eq_zero(B.cre()) &&
                   O_nn.ann().fast_a_and_b_eq_zero(B.cre());
        };

        auto check_ABAd_case = [&](const auto& A, const auto& B, const auto& C) {
            return O_nn.cre().fast_a_and_b_eq_zero(A.cre()) &&
                   O_nn.ann().fast_a_and_b_eq_zero(A.ann()) &&
                   O_nn.cre().fast_a_and_b_eq_zero(B.cre()) &&
                   O_nn.ann().fast_a_and_b_eq_zero(B.ann()) &&
                   O_n.cre().fast_a_and_b_eq_zero(B.cre()) &&
                   O_nn.cre().fast_a_and_b_eq_zero(C.cre()) &&
                   O_nn.ann().fast_a_and_b_eq_zero(C.cre());
        };

        // Check conditions and set alpha accordingly
        if (check_ABA_case(Td_nn, T_n)) {
            // Check if TOT != 0
            alpha = 4;
        } else if (check_ABA_case(T_nn, Td_n)) {
            // Check if TdOTd != 0
            alpha = 4;
        } else if (check_ABAd_case(T_nn, Td_nn, T_n)) {
            // Check if TOTd != 0
            alpha = 4;
        } else if (check_ABAd_case(Td_nn, T_nn, Td_n)) {
            // Check if TdOT != 0
            alpha = 4;
        }

        if (alpha == 4) {
            c1 = c1_pair.second * O_c;
            c2 = c2_pair.second * O_c;
        } else {
            c1 = c1_pair.first * O_c;
            c2 = c2_pair.first * O_c;
        }

        // // check which terms survive the screening
        const auto do_c1 = std::abs(c1) > screen_threshold;
        const auto do_c2 = std::abs(c2) > screen_threshold;

        if (O_T_commutator_type != CommutatorType::Commute) {
            // [O, T]
            auto cOT = commutator_fast(O_op, T_op);
            for (const auto& [cOT_op, cOT_c] : cOT) {
                // + c1 [O, T]
                if (do_c1) {
                    R.add(cOT_op, cOT_c * c1);
                }
                if (do_c2) {
                    auto ccOTT = commutator_fast(cOT_op, T_op);
                    for (const auto& [ccOTT_op, ccOTT_c] : ccOTT) {
                        // + c2 [[O, T], T]
                        R.add(ccOTT_op, ccOTT_c * cOT_c * c2);
                    }
                    auto ccOTTd = commutator_fast(cOT_op, Td_op);
                    for (const auto& [ccOTTd_op, ccOTTd_c] : ccOTTd) {
                        // sigma * c2 [[O, T], T^dagger]
                        R.add(ccOTTd_op, sigma * ccOTTd_c * cOT_c * c2);
                    }
                }
            }
        }
        if (O_Td_commutator_type != CommutatorType::Commute) {
            // [O, T^dagger]
            auto cOTd = commutator_fast(O_op, Td_op);
            for (const auto& [cOTd_op, cOTd_c] : cOTd) {
                // sigma * c1 [O, T^dagger]
                if (do_c1) {
                    R.add(cOTd_op, sigma * cOTd_c * c1);
                }
                if (do_c2) {
                    auto ccOTdT = commutator_fast(cOTd_op, T_op);
                    for (const auto& [ccOTdT_op, ccOTdT_c] : ccOTdT) {
                        // sigma * c2 [[O, T^dagger], T]
                        R.add(ccOTdT_op, sigma * ccOTdT_c * cOTd_c * c2);
                    }
                    auto ccOTdTd = commutator_fast(cOTd_op, Td_op);
                    for (const auto& [ccOTdTd_op, ccOTdTd_c] : ccOTdTd) {
                        // +c2 [[O, T^dagger], T^dagger]
                        R.add(ccOTdTd_op, ccOTdTd_c * cOTd_c * c2);
                    }
                }
            }
        }
    }
    if (add) {
        O += R;
    } else {
        O = std::move(R);
    }
}

void fact_unitary_trans_antiherm(SparseOperator& O, const SparseOperatorList& T, bool reverse,
                                 double screen_threshold) {
    const auto& elements = T.elements();
    auto operation = [&](const auto& iter) {
        const auto& [sqop, theta] = *iter;

        // sanity check that theta is real
        if (std::abs(std::imag(theta)) > 1.0e-12) {
            throw std::runtime_error("sim_trans_antiherm_impl: the angle theta must be real.");
        }

        // skip transformation if theta is zero
        if (std::abs(theta) < screen_threshold) {
            return;
        }

        // if T = T^dagger, then the transformation is trivial, so we can skip it
        if (sqop.is_self_adjoint()) {
            return;
        }

        // compute the coefficients for the similarity transformation
        const auto sin_theta = std::sin(theta);
        const auto one_minus_cos_theta = 1.0 - std::cos(theta);
        const auto half_sin_2_theta = 0.5 * std::sin(2.0 * theta);
        const auto half_sin_theta_2 = 0.5 * std::pow(std::sin(theta), 2.0);
        const std::pair c1_pair{sin_theta, half_sin_2_theta};
        const std::pair c2_pair{one_minus_cos_theta, half_sin_theta_2};
        sim_trans_impl(O, sqop, c1_pair, c2_pair, -1.0, true, screen_threshold);
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

void fact_unitary_trans_antiherm_grad(SparseOperator& O, const SparseOperatorList& T, size_t n,
                                      bool reverse, double screen_threshold) {
    const auto& elements = T.elements();
    auto operation = [&](const auto& iter, const auto& index, const auto& grad_index) {
        const auto& [sqop, theta] = *iter;

        // sanity check that theta is real
        if (std::abs(std::imag(theta)) > 1.0e-12) {
            throw std::runtime_error("sim_trans_antiherm_impl: the angle theta must be real.");
        }

        // skip transformation if theta is zero, except if we are taking the gradient
        if ((std::abs(theta) < screen_threshold) and index != grad_index) {
            return;
        }

        // if T = T^dagger, then the transformation is trivial, so we can skip it
        if (sqop.is_self_adjoint()) {
            // if we are taking the gradient of this operator then the result is zero
            if (index == grad_index) {
                O = SparseOperator();
            }
            return;
        }

        if (index == grad_index) {
            // compute the coefficients for the similarity transformation
            const auto sin_theta = std::sin(theta);
            const auto cos_theta = std::cos(theta);
            const auto cos_2_theta = std::cos(2.0 * theta);
            const auto half_sin_2_theta = 0.5 * std::sin(2.0 * theta);
            const std::pair c1_pair{cos_theta, cos_2_theta};
            const std::pair c2_pair{sin_theta, half_sin_2_theta};
            sim_trans_impl(O, sqop, c1_pair, c2_pair, -1.0, false, screen_threshold);
        } else {
            // compute the coefficients for the similarity transformation
            const auto sin_theta = std::sin(theta);
            const auto one_minus_cos_theta = 1.0 - std::cos(theta);
            const auto half_sin_2_theta = 0.5 * std::sin(2.0 * theta);
            const auto half_sin_theta_2 = 0.5 * std::pow(std::sin(theta), 2.0);
            const std::pair c1_pair{sin_theta, half_sin_2_theta};
            const std::pair c2_pair{one_minus_cos_theta, half_sin_theta_2};
            sim_trans_impl(O, sqop, c1_pair, c2_pair, -1.0, true, screen_threshold);
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

void fact_unitary_trans_imagherm(SparseOperator& O, const SparseOperatorList& T, bool reverse,
                                 double screen_threshold) {
    std::complex<double> imag1 = std::complex<double>(0.0, 1.0);
    const auto& elements = T.elements();
    auto operation = [&](const auto& iter) {
        const auto& [sqop, theta] = *iter;

        // sanity check that theta is real
        if (std::abs(std::imag(theta)) > 1.0e-12) {
            throw std::runtime_error("sim_trans_antiherm_impl: the angle theta must be real.");
        }

        // skip transformation if theta is zero
        if (std::abs(theta) < screen_threshold) {
            return;
        }

        // consider the case where the operator is a number operator
        if (sqop.cre().fast_a_xor_b_count(sqop.ann()) == 0) {
            SparseOperator R;
            double two_theta = 2.0 * std::real(theta);
            sparse_scalar_t four_sin_theta_2 = 4.0 * std::pow(std::sin(theta), 2.0);
            std::complex<double> exp_phase_min_one = std::polar(1.0, two_theta) - 1.0;
            std::complex<double> exp_min_phase_min_one = std::polar(1.0, -two_theta) - 1.0;
            for (const auto& [O_op, O_c] : O.elements()) {
                auto NO = sqop * O_op;
                for (const auto& [NO_op, NO_c] : NO) {
                    R.add(NO_op, exp_phase_min_one * NO_c * O_c);
                    auto NON = NO_op * sqop;
                    for (const auto& [NON_op, NON_c] : NON) {
                        R.add(NON_op, four_sin_theta_2 * NON_c * NO_c * O_c);
                    }
                }
                auto ON = O_op * sqop;
                for (const auto& [ON_op, ON_c] : ON) {
                    R.add(ON_op, exp_min_phase_min_one * ON_c * O_c);
                }
            }
            O += R;
            return;
        }

        // compute the coefficients for the similarity transformation
        const auto min_i_sin_theta = -imag1 * std::sin(theta);
        const auto cos_theta_minus_1 = std::cos(theta) - 1.0;
        const auto min_i_half_sin_2_theta = -imag1 * 0.5 * std::sin(2.0 * theta);
        const auto min_half_sin_theta_2 = -0.5 * std::pow(std::sin(theta), 2.0);

        const std::pair c1_pair{min_i_sin_theta, min_i_half_sin_2_theta};
        const std::pair c2_pair{cos_theta_minus_1, min_half_sin_theta_2};
        sim_trans_impl(O, sqop, c1_pair, c2_pair, 1.0, true, screen_threshold);
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

void fact_trans_lin(SparseOperator& O, const SparseOperatorList& T, bool reverse,
                    double screen_threshold) {
    const auto& elements = T.elements();
    auto operation = [&](const auto& iter) {
        const auto& [sqop, c] = *iter;
        sim_trans_lin_impl(O, sqop, c, screen_threshold);
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

void sim_trans_lin_impl(SparseOperator& O, const SQOperatorString& T_op, sparse_scalar_t theta,
                        double screen_threshold) {
    // sanity check to make sure all indices are distinct
    if (T_op.cre().fast_a_xor_b_count(T_op.ann()) == 0) {
        throw std::runtime_error("sim_trans_lin_impl: the operator " + T_op.str() +
                                 " contains repeated indices.\nThis is not allowed for the "
                                 "similarity transformation.");
    }
    // (1 - T) O(1 + T) = O + [ O, T ] - T O T
    SparseOperator R;
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
                R.add(cOT_op, theta * cOT_c * O_c);
            }
            // - T [O,T]
            auto T_cOT = T_op * cOT_op;
            for (const auto& [T_cOT_op, T_cOT_c] : T_cOT) {
                if (std::abs(theta * theta * T_cOT_c * cOT_c * O_c) > screen_threshold) {
                    R.add(T_cOT_op, -theta * theta * T_cOT_c * cOT_c * O_c);
                }
            }
        }
    }
    O += R;
}

} // namespace forte
