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

#include <vector>
#include <unordered_map>
#include <map>

#include "sparse_ci/determinant.h"
#include "base_classes/state_info.h"

namespace forte {
/**
 * @brief The FCIStringAddress class
 * This class computes the address of a string
 */
class FCIStringAddress {
  public:
    // ==> Class Constructor and Destructor <==
    /// @brief Default constructor
    /// @param strings a vector of vectors of strings.
    /// Each vector collects the strings of a given symmetry
    FCIStringAddress(int nmo, int ne, const std::vector<std::vector<String>>& strings);

    /// @brief Default destructor
    ~FCIStringAddress() = default;

    // ==> Class Interface <==
    /// @brief Add a string with a given irrep
    void push_back(const String& s, int irrep);
    /// @brief Return the address of a string within an irrep
    size_t add(const String& s) const;
    /// @brief Return the irrep of a string
    int sym(const String& s) const;
    /// @brief Return the address and irrep of a string
    const auto& address_and_class(const String& s) const;
    /// @brief Return the number of strings in an irrep
    size_t strpcls(int h) const;
    /// @brief Return the number of classes
    size_t num_classes() const;
    /// @brief Return the number of bits in the string
    int nbits() const;
    /// @brief Return the number of 1s in the string
    int nones() const;

  private:
    // ==> Class Data <==
    /// number of string classes
    int nclasses_;
    /// number of strings
    size_t nstr_;
    /// number of strings in each class
    std::vector<size_t> strpcls_;
    /// Map from string to address and class
    std::unordered_map<String, std::pair<uint32_t, uint32_t>, String::Hash> address_;
    /// the number of orbitals
    int nbits_; // number of orbitals
    /// the number of electrons
    int nones_; // number of 1s
};

class FCIStringClass {
  public:
    FCIStringClass(const std::vector<int>& mopi);

    /// @brief Return the symmetry of a string
    size_t symmetry(const String& s) const;

  private:
    /// The number of irreps
    size_t nirrep_;
    /// The symmetry of each MO
    std::vector<int> mo_sym_;
};

} // namespace forte
