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

#ifndef _string_address_h_
#define _string_address_h_

#include <vector>
#include <unordered_map>

#include "sparse_ci/determinant.h"

namespace forte {

/**
 * @brief The StringAddress class
 * This class helps compute the address of a string
 */
class StringAddress {
  public:
    // ==> Class Constructor and Destructor <==
    StringAddress(const std::vector<int>& irrep_size);
    ~StringAddress() = default;

    // ==> Class Interface <==
    size_t rel_add(const String& s);
    int sym(const String& s) const;
    size_t strpi(int h) const;

  private:
    // ==> Class Data <==
    std::vector<int> symmetry_; // symmetry of each orbital
    std::vector<size_t> strpi_; // number of strings in each irrep
    std::unordered_map<String, std::pair<int32_t, int32_t>, String::Hash> address_;
};
} // namespace forte

#endif // _string_address_h_
