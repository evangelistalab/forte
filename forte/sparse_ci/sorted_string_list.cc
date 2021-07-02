/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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
#include <algorithm>

#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"
#include "sorted_string_list.h"

namespace forte {

SortedStringList::SortedStringList() {}

SortedStringList::SortedStringList(const DeterminantHashVec& space,
                                             std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                             DetSpinType sorted_string_spin) {
    nmo_ = fci_ints->nmo();
    // Copy and sort the determinants
    auto dets = space.determinants();
    num_dets_ = dets.size();
    sorted_dets_.reserve(num_dets_);
    for (const auto& d : dets) { // TODO: this appears redundant now (Francesco)
        sorted_dets_.push_back(d);
    }
    if (sorted_string_spin == DetSpinType::Alpha) {
        map_to_hashdets_ = sort_permutation(sorted_dets_, Determinant::reverse_less_than);
        apply_permutation_in_place(sorted_dets_, map_to_hashdets_);
        //        std::sort(sorted_dets_.begin(), sorted_dets_.end(),
        //        UI64Determinant::reverse_less_then);
    } else {
        map_to_hashdets_ = sort_permutation(sorted_dets_, Determinant::less_than);
        apply_permutation_in_place(sorted_dets_, map_to_hashdets_);
        //        std::sort(sorted_dets_.begin(), sorted_dets_.end());
    }

    //    outfile->Printf("\n\n Sorted determinants (%zu,%s)\n", num_dets_,
    //                    sorted_string_spin == DetSpinType::Alpha ? "Alpha" : "Beta");
    // Find the unique strings and their range

    sorted_spin_type_ =
        sorted_string_spin == DetSpinType::Alpha ? DetSpinType::Alpha : DetSpinType::Beta;
    String first_string = sorted_dets_[0].get_bits(sorted_spin_type_);
    String old_first_string = first_string;

    first_string_range_[old_first_string] = std::make_pair(0, 0);
    sorted_half_dets_.push_back(old_first_string);

    //    outfile->Printf("\n %6d %s", 0, sorted_dets_[0].str2().c_str());
    size_t min_per_string = std::numeric_limits<std::size_t>::max();
    size_t max_per_string = 0;
    for (size_t i = 1; i < num_dets_; i++) {
        //        outfile->Printf("\n %6d %s", i, sorted_dets_[i].str2().c_str());
        first_string = sorted_dets_[i].get_bits(sorted_spin_type_);

        //        first_string.set_bits(sorted_dets_[i].bits() & half_bit_mask.bits());
        if (not(first_string == old_first_string)) {
            first_string_range_[old_first_string].second = i;
            first_string_range_[first_string] = std::make_pair(i, 0);
            //            outfile->Printf(" <- new determinant (%zu -> %zu)",
            //                            first_string_range_[old_first_string].first,
            //                            first_string_range_[old_first_string].second);
            old_first_string = first_string;
            sorted_half_dets_.push_back(first_string);
        }
    }
    first_string_range_[old_first_string].second = num_dets_;

    for (const auto& k_v : first_string_range_) {
        size_t range = k_v.second.second - k_v.second.first;
        min_per_string = std::min(min_per_string, range);
        max_per_string = std::max(max_per_string, range);
    }

    //   outfile->Printf("\n\n  SortedStringList Summary:");
    //   outfile->Printf("\n    Number of determinants: %zu", num_dets_);
    //   outfile->Printf("\n    Number of strings:      %zu (%.2f %%)", sorted_half_dets_.size(),
    //                   100.0 * double(sorted_half_dets_.size()) / double(num_dets_));
    //   outfile->Printf("\n    Max block size:         %zu", max_per_string);
    //   outfile->Printf("\n    Min block size:         %zu", min_per_string);
    //   outfile->Printf("\n    Avg block size:         %0.f\n",
    //                   double(num_dets_) / double(sorted_half_dets_.size()));
}

SortedStringList::~SortedStringList() {}

const std::vector<Determinant>& SortedStringList::sorted_dets() const {
    return sorted_dets_;
}

const std::vector<String>& SortedStringList::sorted_half_dets() const {
    return sorted_half_dets_;
}

const std::pair<size_t, size_t>& SortedStringList::range(const String& d) const {
    return first_string_range_.at(d);
}

size_t SortedStringList::add(size_t pos) const { return map_to_hashdets_[pos]; }
} // namespace forte
