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

#pragma once

#include <tuple>
#include <array>
#include <vector>
#include <string>

namespace forte {

const size_t max_active_spaces = 6;
using occupation_t = std::array<int, max_active_spaces>;

std::tuple<size_t, std::vector<std::array<int, 6>>, std::vector<std::array<int, 6>>,
           std::vector<std::pair<size_t, size_t>>>
get_gas_occupation(size_t na, size_t nb, const std::vector<int>& gas_min,
                   const std::vector<int>& gas_max, const std::vector<int>& gas_size);

std::tuple<size_t, std::vector<std::array<int, 6>>, std::vector<std::array<int, 6>>,
           std::vector<std::pair<size_t, size_t>>>
get_ormas_occupation(size_t na, size_t nb, const std::vector<int>& gas_min,
                     const std::vector<int>& gas_max, const std::vector<int>& gas_size);

std::tuple<std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
generate_gas_occupations(int na, int nb, const std::vector<int>& gas_min_el,
                         const std::vector<int>& gas_max_el, const std::vector<int>& gas_size,
                         size_t num_gas_spaces);

std::vector<std::array<int, 6>>
generate_1h_occupations(const std::vector<std::array<int, 6>>& gas_occupations);

std::string occupation_table(size_t num_spaces,
                             const std::vector<std::array<int, 6>>& alfa_occupation,
                             const std::vector<std::array<int, 6>>& beta_occupation,
                             const std::vector<std::pair<size_t, size_t>>& occupation_pairs);

} // namespace forte