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

#include "string_address.h"

namespace forte {

// StringAddress::StringAddress(const std::vector<int>& irrep_size) : strpi_(irrep_size.size(), 0) {
//     size_t nirreps = irrep_size.size();
//     for (size_t h = 0; h < nirreps; h++) {
//         fill_n(back_inserter(symmetry_), irrep_size[h], h); // insert h irrep_size[h] times
//     }
// }

StringAddress::StringAddress(const std::vector<std::vector<String>>& strings)
    : nirrep_(strings.size()), nstr_(0), strpi_(strings.size(), 0) {
    for (int h = 0; h < nirrep_; h++) {
        const auto& strings_h = strings[h];
        for (const auto& s : strings_h) {
            push_back(s, h);
        }
    }
}

void StringAddress::push_back(const String& s, int irrep) {
    size_t add = strpi_[irrep];
    address_[s] = std::pair(add, irrep);
    strpi_[irrep] += 1;
    nstr_++;
}

size_t StringAddress::add(const String& s) {
    auto it = address_.find(s);
    if (it == address_.end()) {
        // int symm = s.symmetry(symmetry_);
        // size_t add = strpi_[symm];
        // address_[s] = std::pair(add, symm);
        // strpi_[symm] += 1;
        // nstr_++;
        // return add;
        return 0;
    }
    return it->second.first;
}

size_t StringAddress::rel_add(bool* b) {
    tmp_.zero();
    for (int i = 0; i < nbits_; i++) {
        tmp_.set_bit(i, b[i]);
    }
    return add(tmp_);
}

int StringAddress::sym(const String& s) const { return s.symmetry(symmetry_); }

int StringAddress::sym(bool* b) {
    tmp_.zero();
    for (int i = 0; i < nbits_; i++) {
        tmp_.set_bit(i, b[i]);
    }
    return tmp_.symmetry(symmetry_);
}

// size_t StringAddress::strpi(int h) const { return strpi_[h]; }

// int StringAddress::nbits() const { return nbits_; }

// int StringAddress::nones() const { return nones_; }

// size_t StringAddress::nstr() const { return nstr_; }

} // namespace forte
