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

#pragma once

#include <memory>

#include "sparse_ci/sparse_operator.h"
#include "helpers/ndarray/ndarray.hpp"

namespace forte {

class ActiveSpaceIntegrals;

/// @brief Generate the a SparseOperator representation of the Hamiltonian using integrals from an
/// ActiveSpaceIntegrals object
/// @param as_ints the ActiveSpaceIntegrals object containing the integrals
/// @param screen_thresh the threshold to screen the integrals
SparseOperator sparse_operator_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                           double screen_thresh = 1e-12);

template <typename T>
SparseOperator sparse_operator_hamiltonian(double scalar, ndarray<T>& oei_a, ndarray<T>& oei_b,
                                           ndarray<T>& tei_aa, ndarray<T>& tei_ab,
                                           ndarray<T>& tei_bb, double screen_thresh);

} // namespace forte
