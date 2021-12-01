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

#include "combinatorial.h"

namespace forte {

int permutation_parity(const std::vector<size_t>& p) {
    auto n = static_cast<int>(p.size());

    // vector of elements visited
    std::vector<bool> visited(n, false);

    int total_parity = 0;
    // loop over all the elements
    for (int i = 0; i < n; i++) {
        // if an element was not visited start following its cycle
        if (visited[i] == false) {
            int cycle_size = 0;
            int next = i;
            for (int j = 0; j < n; j++) {
                next = p[next];
                // mark the next element as visited
                visited[next] = true;
                // increase cycle size
                cycle_size += 1;
                // if the next element is the same as the one we
                // started from, we reached the end of the cycle
                if (next == i)
                    break;
            }
            total_parity += (cycle_size - 1) % 2;
        }
    }
    return total_parity % 2;
}
} // namespace forte
