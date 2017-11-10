/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#ifndef _determinant_h_
#define _determinant_h_

#include <unordered_map>

#ifdef SMALL_BITSET
#include "ui64_determinant.h"
#else
#include "stl_bitset_determinant.h"
#endif

namespace psi {
namespace forte {

#ifdef SMALL_BITSET
using Determinant = UI64Determinant;
#else
using Determinant = STLBitsetDeterminant;
#endif

using det_vec = std::vector<Determinant>;
template <typename T = double>
using det_hash = std::unordered_map<Determinant, T, Determinant::Hash>;
using det_hash_it = std::unordered_map<Determinant, double, Determinant::Hash>::iterator;
}
}
#endif // _determinant_h_
