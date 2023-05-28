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

#include "base_classes/mo_space_info.h"

namespace forte {

StringAddress::StringAddress(int nmo, int ne, const std::vector<std::vector<String>>& strings)
    : nirrep_(strings.size()), nstr_(0), strpi_(strings.size(), 0), nbits_(nmo), nones_(ne) {
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

size_t StringAddress::add(const String& s) const { return address_.at(s).first; }

int StringAddress::sym(const String& s) const { return address_.at(s).second; }

size_t StringAddress::strpi(int h) const { return strpi_[h]; }

StringClass::StringClass(std::vector<int> mopi, StringClassType type)
    : type_(type), nirrep_(mopi.size()) {
    for (size_t h = 0; h < nirrep_; h++) {
        fill_n(back_inserter(mo_sym_), mopi[h], h); // insert h for mopi[h] times
    }
}

size_t StringClass::symmetry(const String& s) const {
    if (type_ == StringClassType::FCI) {
        return s.symmetry(mo_sym_);
    }
    return 0;
}

size_t StringClass::nclasses() const {
    if (type_ == StringClassType::FCI) {
        return nirrep_;
    }
    throw std::runtime_error("StringClass::nclasses() not implemented for types other than FCI");
    return 0;
}

} // namespace forte
