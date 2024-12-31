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

#pragma once

#include <vector>
#include <unordered_map>

#include "sparse_ci/sparse.h"
#include "sparse_ci/determinant.h"
#include "sparse_ci/sparse_operator.h"

namespace forte {

/// @brief A class to represent general Fock space states
class SparseState
    : public VectorSpace<SparseState, Determinant, sparse_scalar_t, Determinant::Hash> {
  public:
    /// @return a string representation of the object
    /// @param n the number of spatial orbitals to print
    std::string str(int n = 0) const;
};

// Functions to apply operators to a state
/// @brief Apply an operator to a state
/// @param op the operator to apply
/// @param state the state to apply the operator to
/// @param screen_thresh the threshold to screen the operator
/// @return the new state
SparseState apply_operator_lin(const SparseOperator& op, const SparseState& state,
                               double screen_thresh = 1.0e-12);

/// @brief Apply the antihermitian combination of an operator to a state
/// @param op the operator to apply
/// @param state the state to apply the operator to
/// @param screen_thresh the threshold to screen the operator
/// @return the new state
SparseState apply_operator_antiherm(const SparseOperator& op, const SparseState& state,
                                    double screen_thresh = 1.0e-12);

/// compute the projection  <state0 | op | ref>, for each operator op in gop
std::vector<sparse_scalar_t> get_projection(const SparseOperatorList& sop, const SparseState& ref,
                                            const SparseState& state0);

/// apply the number projection operator P^alpha_na P^beta_nb |state>
SparseState apply_number_projector(int na, int nb, const SparseState& state);

/// compute the overlap value <left_state|right_state>
sparse_scalar_t overlap(const SparseState& left_state, const SparseState& right_state);

/// compute the S^2 expectation value
sparse_scalar_t spin2(const SparseState& left_state, const SparseState& right_state);

/// Return the normalized state
SparseState normalize(const SparseState& state);

// Example: Define the size of your bitstring
constexpr size_t BitStringSize = Determinant::nbits;
using BitString = Determinant;

// class BitwiseTrie {
//   public:
//     struct Node {
//         std::unique_ptr<Node> zero = nullptr;
//         std::unique_ptr<Node> one = nullptr;
//         bool is_end = false; // Marks the end of a bitstring
//     };

//     // Insert a bitstring into the trie
//     void insert(const BitString& bitstring) {
//         Node* current = &root;
//         for (size_t i = 0; i < BitStringSize; ++i) {
//             if (bitstring[BitStringSize - 1 - i]) { // Traverse from MSB to LSB
//                 if (!current->one) {
//                     current->one = std::make_unique<Node>();
//                 }
//                 current = current->one.get();
//             } else {
//                 if (!current->zero) {
//                     current->zero = std::make_unique<Node>();
//                 }
//                 current = current->zero.get();
//             }
//         }
//         current->is_end = true;
//     }

//     // Find all bitstrings that satisfy (a & b) == b
//     std::vector<BitString> query(const BitString& b) const {
//         std::vector<BitString> result;
//         BitString current_result;
//         query_helper(&root, b, 0, current_result, result);
//         return result;
//     }

//   private:
//     Node root;

//     void query_helper(const Node* node, const BitString& b, size_t depth, BitString&
//     current_result,
//                       std::vector<BitString>& result) const {
//         if (!node)
//             return;

//         if (depth == BitStringSize) {
//             if (node->is_end) {
//                 result.push_back(current_result);
//             }
//             return;
//         }

//         // Check the current bit of b
//         if (b[BitStringSize - 1 - depth]) { // If b requires this bit to be 1
//             if (node->one) {
//                 current_result.set(BitStringSize - 1 - depth);
//                 query_helper(node->one.get(), b, depth + 1, current_result, result);
//                 current_result.reset(BitStringSize - 1 - depth);
//             }
//         } else { // If b requires nothing for this bit
//             // Can follow both branches
//             if (node->zero) {
//                 query_helper(node->zero.get(), b, depth + 1, current_result, result);
//             }
//             if (node->one) {
//                 current_result.set(BitStringSize - 1 - depth);
//                 query_helper(node->one.get(), b, depth + 1, current_result, result);
//                 current_result.reset(BitStringSize - 1 - depth);
//             }
//         }
//     }
// };

constexpr size_t DeterminantSize = Determinant::nbits;

// class BitwiseTrieMap {
//   public:
//     struct Node {
//         std::unique_ptr<Node> zero = nullptr;
//         std::unique_ptr<Node> one = nullptr;
//         bool is_end = false;     // Marks the end of a Determinant
//         Determinant determinant; // Only valid if is_end == true
//         sparse_scalar_t value;   // Associated sparse_scalar_t value
//     };

//     // Insert a Determinant and its associated value into the trie
//     void insert(const Determinant& determinant, sparse_scalar_t value) {
//         Node* current = &root;
//         for (size_t i = 0; i < DeterminantSize; ++i) {
//             if (determinant[DeterminantSize - 1 - i]) { // Traverse from MSB to LSB
//                 if (!current->one) {
//                     current->one = std::make_unique<Node>();
//                 }
//                 current = current->one.get();
//             } else {
//                 if (!current->zero) {
//                     current->zero = std::make_unique<Node>();
//                 }
//                 current = current->zero.get();
//             }
//         }
//         current->is_end = true;
//         current->determinant = determinant;
//         current->value = value;
//     }

//     // Query the trie for all Determinants that satisfy (a & b) == b
//     // Calls the provided lambda `callback` for each valid result
//     void query(const Determinant& b,
//                const std::function<void(const Determinant&, sparse_scalar_t)>& callback) const {
//         Determinant current_result;
//         query_helper(&root, b, 0, current_result, callback);
//     }

//   private:
//     Node root;

//     void
//     query_helper(const Node* node, const Determinant& b, size_t depth, Determinant&
//     current_result,
//                  const std::function<void(const Determinant&, sparse_scalar_t)>& callback) const
//                  {
//         if (!node)
//             return;

//         if (depth == DeterminantSize) {
//             if (node->is_end) {
//                 // Call the lambda with the determinant and its associated value
//                 callback(node->determinant, node->value);
//             }
//             return;
//         }

//         // Check the current bit of b
//         if (b[DeterminantSize - 1 - depth]) { // If b requires this bit to be 1
//             if (node->one) {
//                 current_result.set(DeterminantSize - 1 - depth);
//                 query_helper(node->one.get(), b, depth + 1, current_result, callback);
//                 current_result.reset(DeterminantSize - 1 - depth);
//             }
//         } else { // If b requires nothing for this bit
//             // Can follow both branches
//             if (node->zero) {
//                 query_helper(node->zero.get(), b, depth + 1, current_result, callback);
//             }
//             if (node->one) {
//                 current_result.set(DeterminantSize - 1 - depth);
//                 query_helper(node->one.get(), b, depth + 1, current_result, callback);
//                 current_result.reset(DeterminantSize - 1 - depth);
//             }
//         }
//     }
// };

class BitwiseTrieVector {
  public:
    struct Node {
        int64_t zero = -1;    // Index of the '0' child (-1 means nullptr)
        int64_t one = -1;     // Index of the '1' child (-1 means nullptr)
        int64_t det_idx = -1; // Index of the node in the vector (!= -1 means valid node)
    };

    // Constructor: Build a trie from sorted determinants
    BitwiseTrieVector(const std::vector<std::pair<Determinant, sparse_scalar_t>>& determinants) {
        nodes_.emplace_back(); // Root node
        build_trie(determinants);
    }

    // Query with two masks: one_mask and zero_mask
    void query(const Determinant& one_mask, const Determinant& zero_mask,
               const std::function<void(size_t)>& callback) const {
        Determinant current_result;
        query_helper_two_masks(0, one_mask, zero_mask, 0, current_result,
                               callback); // Start from the root node
    }

    void query_stack_based(const Determinant& one_mask, const Determinant& zero_mask,
                           const std::function<void(size_t)>& callback) const {
        // Stack contains tuples: (node_idx, depth, current_result)
        std::stack<std::tuple<int64_t, size_t, Determinant>> stack;
        stack.emplace(0, 0, Determinant()); // Start from the root node

        while (!stack.empty()) {
            auto [node_idx, depth, current_result] = stack.top();
            stack.pop();

            if (node_idx == -1)
                continue;

            const Node& node = nodes_[node_idx];

            if (depth == DeterminantSize) {
                if (node.det_idx != -1) {
                    callback(node.det_idx);
                }
                continue;
            }

            // Determine the required bit conditions
            bool must_be_one = one_mask.get_bit(DeterminantSize - 1 - depth);
            bool must_be_zero = zero_mask.get_bit(DeterminantSize - 1 - depth);

            if (must_be_one) { // This bit must be 1
                if (node.one != -1) {
                    current_result.set(DeterminantSize - 1 - depth);
                    stack.emplace(node.one, depth + 1, current_result);
                }
            } else if (must_be_zero) { // This bit must be 0
                if (node.zero != -1) {
                    stack.emplace(node.zero, depth + 1, current_result);
                }
            } else { // This bit is unconstrained
                if (node.zero != -1) {
                    stack.emplace(node.zero, depth + 1, current_result);
                }
                if (node.one != -1) {
                    current_result.set(DeterminantSize - 1 - depth);
                    stack.emplace(node.one, depth + 1, current_result);
                }
            }
        }
    }

  private:
    std::vector<Node> nodes_; // Vector of nodes

    void build_trie(const std::vector<std::pair<Determinant, sparse_scalar_t>>& determinants) {
        int64_t det_idx = 0;
        for (const auto& [det, coeff] : determinants) {
            insert(det, det_idx);
            ++det_idx;
        }
        std::cout << "Built trie with " << nodes_.size() << " nodes" << std::endl;
    }

    void insert(const Determinant& det, int64_t det_idx) {
        int64_t current_idx = 0; // Start at the root
        for (size_t i = 0; i < DeterminantSize; ++i) {
            bool bit = det.get_bit(DeterminantSize - 1 - i); // Traverse MSB to LSB

            auto child_idx = bit ? nodes_[current_idx].one : nodes_[current_idx].zero;

            if (child_idx == -1) {
                child_idx = nodes_.size();
                nodes_.emplace_back(); // Create a new node
                if (bit) {
                    nodes_[current_idx].one = child_idx;
                } else {
                    nodes_[current_idx].zero = child_idx;
                }
            }
            current_idx = child_idx;
        }
        nodes_[current_idx].det_idx = det_idx;
    }

    void query_helper_two_masks(int64_t node_idx, const Determinant& one_mask,
                                const Determinant& zero_mask, size_t depth,
                                Determinant& current_result,
                                const std::function<void(size_t)>& callback) const {
        if (node_idx == -1)
            return;

        const Node& node = nodes_[node_idx];

        if (depth == DeterminantSize) {
            if (node.det_idx != -1) {
                callback(node.det_idx);
            }
            return;
        }

        // Traverse based on one_mask and zero_mask
        bool must_be_one = one_mask.get_bit(DeterminantSize - 1 - depth);
        bool must_be_zero = zero_mask.get_bit(DeterminantSize - 1 - depth);

        if (must_be_one) { // This bit must be 1
            if (node.one != -1) {
                current_result.set(DeterminantSize - 1 - depth);
                query_helper_two_masks(node.one, one_mask, zero_mask, depth + 1, current_result,
                                       callback);
                current_result.reset(DeterminantSize - 1 - depth);
            }
        } else if (must_be_zero) { // This bit must be 0
            if (node.zero != -1) {
                query_helper_two_masks(node.zero, one_mask, zero_mask, depth + 1, current_result,
                                       callback);
            }
        } else { // This bit is unconstrained
            if (node.zero != -1) {
                query_helper_two_masks(node.zero, one_mask, zero_mask, depth + 1, current_result,
                                       callback);
            }
            if (node.one != -1) {
                current_result.set(DeterminantSize - 1 - depth);
                query_helper_two_masks(node.one, one_mask, zero_mask, depth + 1, current_result,
                                       callback);
                current_result.reset(DeterminantSize - 1 - depth);
            }
        }
    }
};
} // namespace forte
