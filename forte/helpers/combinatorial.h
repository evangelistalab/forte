/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _combinatorial_h_
#define _combinatorial_h_

#include <vector>

namespace forte {

/// Return the parity of a permutation (0 = even, 1 = odd).
/// For example, the input vector is {1, 4, 3, 2, 0, 5, 7, 8, 6}.
/// In cycle notation, the permutation is (0 1 4)(2 3)(5)(6 7 8) and the parity is odd.
/// Only even-length cycles can change the permutation parity.
/// In the above example, only (2 3) is even-lengthed, which is 2.
/// IMPORTANT: the input vector index starts from 0!
///
/// For details, check the following
/// https://en.wikipedia.org/wiki/Parity_of_a_permutation
/// https://math.stackexchange.com/questions/65923/how-does-one-compute-the-sign-of-a-permutation
// of the integers 0,1,2,...,n - 1
int permutation_parity(const std::vector<size_t>& p);

} // namespace forte

#endif // _combinatorial_h_
