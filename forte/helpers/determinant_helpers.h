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

#include "sparse_ci/determinant.h"

namespace psi {
class Matrix;
} // namespace psi

namespace forte {

class ActiveSpaceIntegrals;

/// @brief Build the S^2 operator matrix in the given basis of determinants (multithreaded)
/// @param dets A vector of determinants
/// @return A matrix of size (num_dets, num_dets) with the S^2 operator matrix
std::shared_ptr<psi::Matrix> make_s2_matrix(const std::vector<Determinant>& dets);

/// @brief Build the Hamiltonian operator matrix in the given basis of determinants (multithreaded)
/// @param dets A vector of determinants
/// @param as_ints A pointer to the ActiveSpaceIntegrals object
/// @return A matrix of size (num_dets, num_dets) with the Hamiltonian operator matrix
std::shared_ptr<psi::Matrix> make_hamiltonian_matrix(const std::vector<Determinant>& dets,
                                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints);

/// @brief Generate all strings of n orbitals and k electrons in each irrep
/// @param n The number of orbitals
/// @param k The number of electrons
/// @param nirrep The number of irreps
/// @param mo_symmetry The symmetry of the MOs
/// @return A vector of vectors of strings, one vector for each irrep
std::vector<std::vector<String>> make_strings(int n, int k, size_t nirrep,
                                              const std::vector<int>& mo_symmetry);

/// @brief Generate the Hilbert space for a given number of electrons and orbitals
/// @param nmo The number of orbitals
/// @param na The number of alpha electrons
/// @param nb The number of beta electrons
/// @param nirrep The number of irreps
/// @param mo_symmetry The symmetry of the MOs
/// @param symmetry The symmetry of the determinants
/// @return A vector of determinants
std::vector<Determinant> make_hilbert_space(size_t nmo, size_t na, size_t nb, size_t nirrep,
                                            const std::vector<int>& mo_symmetry, int symmetry);

} // namespace forte
