/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <stdexcept>
#include <utility>

namespace forte {

std::pair<size_t, size_t> thread_range(size_t n, size_t num_thread, size_t tid) {
    if (tid >= num_thread) {
        throw std::invalid_argument("thread_range: Thread ID cannot exceed number of threads.");
    }

    if (tid >= n) {
        // For tid >= n, return the same start and end indices (no work)
        return {n, n};
    }

    size_t base_size = n / num_thread;
    size_t leftover = n % num_thread;

    // threads with tid smaller than leftover will start one unit of work later
    size_t start_idx = tid * base_size + std::min(tid, leftover);
    size_t end_idx = start_idx + base_size + (tid < leftover);

    return {start_idx, end_idx};
}
} // namespace forte