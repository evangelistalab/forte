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
 * This class computes the address of a string
 */
class StringAddress {
  public:
    // ==> Class Constructor and Destructor <==
    /// @brief Default constructor
    StringAddress(const std::vector<std::vector<String>>& strings);
    /// @brief Default destructor
    ~StringAddress() = default;

    // ==> Class Interface <==
    /// @brief Add a string with a given irrep
    void push_back(const String& s, int irrep);
    /// @brief Return the address of a string within an irrep
    size_t add(const String& s) const;
    /// @brief Return the irrep of a string
    int sym(const String& s) const;
    /// @brief Return the number of strings in an irrep
    size_t strpi(int h) const;

  private:
    // ==> Class Data <==
    /// number of irreps
    int nirrep_;
    /// number of strings
    size_t nstr_;
    /// number of strings in each irrep
    std::vector<size_t> strpi_;
    /// Map from string to address and irrep
    std::unordered_map<String, std::pair<int32_t, int32_t>, String::Hash> address_;
};
} // namespace forte

#endif // _string_address_h_
