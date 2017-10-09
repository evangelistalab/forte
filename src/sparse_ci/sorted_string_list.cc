/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "sorted_string_list.h"

namespace psi {
namespace forte {

SortedStringList::SortedStringList(const DeterminantHashVec& space,
                                   std::shared_ptr<FCIIntegrals> fci_ints, bool flip_spin) {
    nmo_ = fci_ints->nmo();
    // Copy and sort the determinants
    sorted_dets_ = space.determinants();
    if (flip_spin) {
        for (auto& d : sorted_dets_) {
            d.spin_flip();
        }
    }
    num_dets_ = sorted_dets_.size();
    std::sort(sorted_dets_.begin(), sorted_dets_.end());

    STLBitsetDeterminant half_bit_mask;
    for (int i = 0; i < nmo_; i++)
        half_bit_mask.set_alfa_bit(i, true);

    // Find the unique strings and their range
    STLBitsetDeterminant first_string;
    STLBitsetDeterminant last_first_string(sorted_dets_[0].bits() & half_bit_mask.bits(), nmo_);

    first_string_range_[last_first_string] = std::make_pair(0, 0);
    for (size_t i = 1; i < num_dets_; i++) {
        first_string.set_bits(sorted_dets_[i].bits() & half_bit_mask.bits());
        if (not(first_string == last_first_string)) {
            first_string_range_[last_first_string].second = i;
            first_string_range_[first_string] = std::make_pair(i, 0);
            last_first_string = first_string;
        }
    }
    first_string_range_[last_first_string].second = num_dets_;
}

const std::vector<STLBitsetDeterminant>& SortedStringList::sorted_dets() { return sorted_dets_; }
const det_hash<std::pair<size_t, size_t>>& SortedStringList::first_string_range() {
    return first_string_range_;
}
}
}
