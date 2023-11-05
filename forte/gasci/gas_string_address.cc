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
    psi::outfile->Printf("String classes: %d\n", nclasses_);
    for (int h = 0; h < nclasses_; h++) {
        const auto& strings_h = strings[h];
        for (const auto& s : strings_h) {
            push_back(s, h);
        }
    }
    // print strpcls_
    for (int h = 0; h < nclasses_; h++) {
        psi::outfile->Printf("String class %d: %zu\n", h, strpcls_[h]);
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

std::unordered_map<String, std::pair<uint32_t, uint32_t>, String::Hash>::const_iterator
StringAddress::find(const String& s) const {
    return address_.find(s);
}

std::unordered_map<String, std::pair<uint32_t, uint32_t>, String::Hash>::const_iterator
StringAddress::end() const {
    return address_.end();
}

const std::pair<uint32_t, uint32_t>& StringAddress::address_and_class(const String& s) const {
    return address_.at(s);
}

int StringAddress::nclasses() const { return nclasses_; }

size_t StringAddress::strpcls(int h) const { return strpcls_[h]; }

StringClass::StringClass(size_t symmetry, const std::vector<int>& mopi,
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

    // Build the alpha string classes as the cartesian product of the alpha occupation groups and
    // the irreps. For each alpha occupation group, we have all the irreps next to each other.
    for (size_t n = 0, j = 0; n < alfa_occupation.size(); n++) {
        for (size_t h = 0; h < nirrep_; h++) {
            alfa_string_classes_.emplace_back(n, h);
            alfa_string_classes_map_[std::make_pair(n, h)] = j;
            j++;
        }
    }
    // Build the beta string classes as the cartesian product of the alpha occupation groups and
    // the irreps. For each alpha occupation group, we have all the irreps next to each other.
    for (size_t n = 0, j = 0; n < beta_occupation.size(); n++) {
        for (size_t h = 0; h < nirrep_; h++) {
            beta_string_classes_.emplace_back(n, h);
            beta_string_classes_map_[std::make_pair(n, h)] = j;
            j++;
        }
    }
    // Build the product of alpha and beta string classes as the cartesian product of the alpha and
    // beta string classes
    for (const auto& [aocc_idx, bocc_idx] : occupations) {
        for (size_t h_Ia = 0; h_Ia < nirrep_; h_Ia++) {
            auto h_Ib = h_Ia ^ symmetry;
            auto aocc_h_Ia = alfa_string_classes_map_.at(std::make_tuple(aocc_idx, h_Ia));
            auto bocc_h_Ib = beta_string_classes_map_.at(std::make_tuple(bocc_idx, h_Ib));
            block_index_[std::make_pair(aocc_h_Ia, bocc_h_Ib)] = determinant_classes_.size();
            determinant_classes_.emplace_back(determinant_classes_.size(), aocc_h_Ia, bocc_h_Ib);
        }
    }
}

size_t StringClass::num_alfa_classes() const { return nirrep_ * alfa_occupation_group_.size(); }

size_t StringClass::num_beta_classes() const { return nirrep_ * beta_occupation_group_.size(); }

size_t StringClass::symmetry(const String& s) const { return s.symmetry(mo_sym_); }

const std::vector<std::pair<size_t, size_t>>& StringClass::alfa_string_classes() const {
    return alfa_string_classes_;
}

const std::vector<std::pair<size_t, size_t>>& StringClass::beta_string_classes() const {
    return beta_string_classes_;
}

const std::vector<std::tuple<size_t, size_t, size_t>>& StringClass::determinant_classes() const {
    return determinant_classes_;
}

int StringClass::block_index(int class_Ia, int class_Ib) const {
    // check if the block index exists. If not, throw an exception
    if (block_index_.count(std::make_pair(class_Ia, class_Ib)) == 0) {
        throw std::runtime_error("Block index not found");
    }
    return block_index_.at(std::make_pair(class_Ia, class_Ib));
}

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
