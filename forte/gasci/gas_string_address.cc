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

#include "gas_string_address.h"

#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"

namespace forte {

StringAddress::StringAddress(const std::vector<int>& gas_size, int ne,
                             const std::vector<std::vector<String>>& strings)
    : nclasses_(strings.size()), nstr_(0), strpcls_(strings.size(), 0), nones_(ne),
      gas_size_(gas_size) {
    for (int h = 0; h < nclasses_; h++) {
        const auto& strings_h = strings[h];
        for (const auto& s : strings_h) {
            push_back(s, h);
        }
    }
}

void StringAddress::push_back(const String& s, int string_class) {
    size_t add = strpcls_[string_class];
    address_[s] = std::pair(add, string_class);
    strpcls_[string_class] += 1;
    nstr_++;
}

int StringAddress::nones() const { return nones_; }

int StringAddress::nbits() const { return math::sum(gas_size_); }

size_t StringAddress::add(const String& s) const { return address_.at(s).first; }

int StringAddress::sym(const String& s) const { return address_.at(s).second; }

const std::pair<uint32_t, uint32_t>& StringAddress::address_and_class(const String& s) const {
    return address_.at(s);
}

size_t StringAddress::strpcls(int h) const { return strpcls_[h]; }

StringClass::StringClass(const std::vector<int>& mopi,
                         const std::vector<std::vector<size_t>>& gas_mos,
                         const std::vector<std::array<int, 6>>& alfa_occupation,
                         const std::vector<std::array<int, 6>>& beta_occupation,
                         const std::vector<std::pair<size_t, size_t>>& occupations)
    : nirrep_(mopi.size()) {
    for (size_t h = 0; h < nirrep_; h++) {
        fill_n(back_inserter(mo_sym_), mopi[h], h); // insert h for mopi[h] times
    }
    for (size_t i = 0; i < alfa_occupation.size(); i++) {
        alfa_occupation_group_[alfa_occupation[i]] = i;
    }
    for (size_t i = 0; i < beta_occupation.size(); i++) {
        beta_occupation_group_[beta_occupation[i]] = i;
    }
    for (size_t n = 0; n < gas_mos.size(); n++) {
        gas_masks_[n].zero();
        for (const auto& mo : gas_mos[n]) {
            gas_masks_[n].set_bit(mo, true);
        }
    }
    occupations_ = occupations;
}

size_t StringClass::num_alfa_classes() const { return nirrep_ * alfa_occupation_group_.size(); }

size_t StringClass::num_beta_classes() const { return nirrep_ * beta_occupation_group_.size(); }

size_t StringClass::symmetry(const String& s) const { return s.symmetry(mo_sym_); }

size_t StringClass::alfa_string_class(const String& s) const {
    std::array<int, 6> occupation;
    for (size_t n = 0; n < gas_masks_.size(); n++) {
        occupation[n] = s.fast_a_and_b_count(gas_masks_[n]);
    }
    return alfa_occupation_group_.at(occupation) * nirrep_ + symmetry(s);
}

size_t StringClass::beta_string_class(const String& s) const {
    std::array<int, 6> occupation;
    for (size_t n = 0; n < gas_masks_.size(); n++) {
        occupation[n] = s.fast_a_and_b_count(gas_masks_[n]);
    }
    return beta_occupation_group_.at(occupation) * nirrep_ + symmetry(s);
}

} // namespace forte
