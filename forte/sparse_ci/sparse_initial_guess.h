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
// #include "sparse_ci/determinant.h"

namespace forte {

using sparse_vec = std::vector<std::pair<size_t, double>>;
using sparse_mat = std::vector<std::vector<std::pair<size_t, double>>>;

class DavidsonLiuSolver;
class ActiveSpaceIntegrals;

/// @brief Return the Hamiltonian matrix transformed into the S^2 eigenbasis
/// @param guess_dets A vector of guess determinants
/// @param as_ints The active space integrals object
/// @return A tuple of the form (S^2 transformed Hamiltonian matrix, S^2 eigenvalues, S^2
/// eigenvectors)
std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
compute_s2_transformed_hamiltonian_matrix(const std::vector<Determinant>& dets,
                                          std::shared_ptr<ActiveSpaceIntegrals> as_ints);

/// @brief Generate initial guess vectors for the Davidson-Liu solver starting from a set of guess
/// determinants
/// @param guess_dets A vector of guess determinants
/// @param guess_dets_pos A vector of the positions of the guess determinants in the CI vector
/// @param num_guess_states The number of guess states to generate
/// @param as_ints The active space integrals object
/// @param dls The Davidson-Liu solver object
/// @param multiplicity The desired multiplicity of the guess states
/// @param do_spin_project Whether or not to project out guess states with the wrong multiplicity
/// @param print Whether or not to print information about the guess procedure
/// @param user_guess A vector of vectors of pairs of the form (determinant index, coefficient)
/// passed in by the user. If this vector is not empty, then we will use the user guess instead.
/// Spin projection can be still applied.
std::pair<sparse_mat, sparse_mat>
find_initial_guess_det(const std::vector<Determinant>& guess_dets,
                       const std::vector<size_t>& guess_dets_pos, size_t num_guess_states,
                       const std::shared_ptr<ActiveSpaceIntegrals>& as_ints, int multiplicity,
                       bool do_spin_project, bool print,
                       const std::vector<std::vector<std::pair<size_t, double>>>& user_guess,
                       bool core_guess = false);

/// @brief Generate initial guess vectors for the Davidson-Liu solver starting from a set of guess
/// configurations
/// @param diag A vector of guess configurations energies
/// @param num_guess_states The number of guess states to generate
/// @param dls The Davidson-Liu solver object
/// @param multiplicity The desired multiplicity of the guess states
/// @param temp A temporary vector
/// @param print Whether or not to print information about the guess procedure
sparse_mat find_initial_guess_csf(std::shared_ptr<psi::Vector> diag, size_t num_guess_states,
                                  size_t multiplicity, bool print);

} // namespace forte