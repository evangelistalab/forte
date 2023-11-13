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

#include <memory>

#include "fci_string_address.h"

#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"

namespace forte {

FCIStringAddress::FCIStringAddress(int nmo, int ne, const std::vector<std::vector<String>>& strings)
    : nclasses_(strings.size()), nstr_(0), strpcls_(strings.size(), 0), nbits_(nmo), nones_(ne) {
    for (int h = 0; h < nclasses_; h++) {
        const auto& strings_h = strings[h];
        for (const auto& s : strings_h) {
            push_back(s, h);
        }
    }
}

void FCIStringAddress::push_back(const String& s, int string_class) {
    size_t add = strpcls_[string_class];
    address_[s] = std::pair(add, string_class);
    strpcls_[string_class] += 1;
    nstr_++;
}

int FCIStringAddress::nones() const { return nones_; }

int FCIStringAddress::nbits() const { return nbits_; }

size_t FCIStringAddress::add(const String& s) const { return address_.at(s).first; }

int FCIStringAddress::sym(const String& s) const { return address_.at(s).second; }

const auto& FCIStringAddress::address_and_class(const String& s) const { return address_.at(s); }

size_t FCIStringAddress::num_classes() const { return nclasses_; }

size_t FCIStringAddress::strpcls(int h) const { return strpcls_[h]; }

FCIStringClass::FCIStringClass(const std::vector<int>& mopi) : nirrep_(mopi.size()) {
    for (size_t h = 0; h < nirrep_; h++) {
        fill_n(back_inserter(mo_sym_), mopi[h], h); // insert h for mopi[h] times
    }
}

size_t FCIStringClass::symmetry(const String& s) const { return s.symmetry(mo_sym_); }

} // namespace forte
