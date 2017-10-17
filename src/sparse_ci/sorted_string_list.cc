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

#include "../helpers.h"
#include "sorted_string_list.h"

namespace psi {
namespace forte {

SortedStringList::SortedStringList(const DeterminantHashVec& space,
                                   std::shared_ptr<FCIIntegrals> fci_ints,
                                   STLBitsetDeterminant::SpinType sorted_string_spin) {
    nmo_ = fci_ints->nmo();
    // Copy and sort the determinants
    sorted_dets_ = space.determinants();
    num_dets_ = sorted_dets_.size();

    if (sorted_string_spin == STLBitsetDeterminant::SpinType::AlphaSpin) {
        std::sort(sorted_dets_.begin(), sorted_dets_.end(),
                  STLBitsetDeterminant::reverse_less_then);
    } else {
        std::sort(sorted_dets_.begin(), sorted_dets_.end());
    }

    outfile->Printf("\n\n Sorted determinants (%zu,%s)\n", num_dets_,
                    sorted_string_spin == STLBitsetDeterminant::SpinType::AlphaSpin ? "Alpha"
                                                                                    : "Beta");
    // Find the unique strings and their range
    STLBitsetDeterminant first_string = sorted_dets_[0];
    STLBitsetDeterminant last_first_string = sorted_dets_[0];
    zero_spin_type_ = sorted_string_spin == STLBitsetDeterminant::SpinType::AlphaSpin
                          ? STLBitsetDeterminant::SpinType::BetaSpin
                          : STLBitsetDeterminant::SpinType::AlphaSpin;
    last_first_string.zero_spin(zero_spin_type_);

    first_string_range_[last_first_string] = std::make_pair(0, 0);
    sorted_half_dets_.push_back(last_first_string);

    outfile->Printf("\n %6d %s", 0, sorted_dets_[0].str2().c_str());
    for (size_t i = 1; i < num_dets_; i++) {
        outfile->Printf("\n %6d %s", i, sorted_dets_[i].str2().c_str());
        first_string = sorted_dets_[i];
        first_string.zero_spin(zero_spin_type_);

        //        first_string.set_bits(sorted_dets_[i].bits() & half_bit_mask.bits());
        if (not(first_string == last_first_string)) {
            first_string_range_[last_first_string].second = i;
            first_string_range_[first_string] = std::make_pair(i, 0);
            outfile->Printf(" <- new determinant (%zu -> %zu)",
                            first_string_range_[last_first_string].first,
                            first_string_range_[last_first_string].second);
            last_first_string = first_string;
            sorted_half_dets_.push_back(first_string);
        }
    }
    first_string_range_[last_first_string].second = num_dets_;
    outfile->Printf(" <- last determinant (%zu -> %zu)",
                    first_string_range_[last_first_string].first,
                    first_string_range_[last_first_string].second);

    outfile->Printf("\n\n  Determinand ranges");
    for (const auto& d : sorted_half_dets_) {
        outfile->Printf("\n %s : %6zu -> %6zu", d.str2().c_str(), first_string_range_[d].first,
                        first_string_range_[d].second);
    }
}

const std::vector<STLBitsetDeterminant>& SortedStringList::sorted_dets() const {
    return sorted_dets_;
}

const std::vector<STLBitsetDeterminant>& SortedStringList::sorted_half_dets() const {
    return sorted_half_dets_;
}

const std::pair<size_t, size_t>& SortedStringList::range(STLBitsetDeterminant d) const {
    d.zero_spin(zero_spin_type_);
    return first_string_range_.at(d);
}

size_t SortedStringList::find(const STLBitsetDeterminant& d, size_t& first, size_t& last) const {
    for (size_t pos = first; pos < last; ++pos) {
        if (not STLBitsetDeterminant::less_than(d, sorted_dets_[pos])) {
            if (d == sorted_dets_[pos]) {
                // Case I. Found the determinant d at position pos.
                // return add(d) = pos
                // set first = pos + 1 for next search
                first = pos + 1;
                return pos;
            }
        } else {
            // Case II. Not found d,
            // but we found another determinant d' with lex(d') > lex(d).
            // return num_dets_ to indicate failure
            // set first = pos for next search
            first = pos;
            return num_dets_;
        }
    }
    // Case III. Not found d and we reached the end of the range.
    // return num_dets_ to indicate failure and set first = last so to skip the next search
    // The determinant
    first = last;
    return num_dets_;
}

SortedStringList_UI64::SortedStringList_UI64(const DeterminantHashVec& space,
                                             std::shared_ptr<FCIIntegrals> fci_ints,
                                             STLBitsetDeterminant::SpinType sorted_string_spin) {
    nmo_ = fci_ints->nmo();
    // Copy and sort the determinants
    auto dets = space.determinants();
    num_dets_ = dets.size();
    sorted_dets_.reserve(num_dets_);
    for (const auto& d : dets) {
        sorted_dets_.push_back(UI64Determinant(d));
    }
    if (sorted_string_spin == STLBitsetDeterminant::SpinType::AlphaSpin) {
        map_to_hashdets_ = sort_permutation(sorted_dets_, UI64Determinant::reverse_less_than);
        apply_permutation_in_place(sorted_dets_, map_to_hashdets_);
        //        std::sort(sorted_dets_.begin(), sorted_dets_.end(),
        //        UI64Determinant::reverse_less_then);
    } else {
        map_to_hashdets_ = sort_permutation(sorted_dets_, UI64Determinant::less_than);
        apply_permutation_in_place(sorted_dets_, map_to_hashdets_);
        //        std::sort(sorted_dets_.begin(), sorted_dets_.end());
    }

    outfile->Printf("\n\n Sorted determinants (%zu,%s)\n", num_dets_,
                    sorted_string_spin == STLBitsetDeterminant::SpinType::AlphaSpin ? "Alpha"
                                                                                    : "Beta");
    // Find the unique strings and their range

    sorted_spin_type_ = sorted_string_spin == STLBitsetDeterminant::SpinType::AlphaSpin
                            ? STLBitsetDeterminant::SpinType::AlphaSpin
                            : STLBitsetDeterminant::SpinType::BetaSpin;
    UI64Determinant::bit_t first_string = sorted_dets_[0].get_bits(sorted_spin_type_);
    UI64Determinant::bit_t old_first_string = first_string;

    first_string_range_[old_first_string] = std::make_pair(0, 0);
    sorted_half_dets_.push_back(old_first_string);

    //    outfile->Printf("\n %6d %s", 0, sorted_dets_[0].str2().c_str());
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
    //    outfile->Printf(" <- last determinant (%zu -> %zu)",
    //                    first_string_range_[old_first_string].first,
    //                    first_string_range_[old_first_string].second);

    //    outfile->Printf("\n\n  Determinand ranges");
    //    for (const auto& d : sorted_half_dets_) {
    //        outfile->Printf("\n %s : %6zu -> %6zu", d.str2().c_str(),
    //        first_string_range_[d].first,
    //                        first_string_range_[d].second);
    //    }
}

const std::vector<UI64Determinant>& SortedStringList_UI64::sorted_dets() const {
    return sorted_dets_;
}

const std::vector<UI64Determinant::bit_t>& SortedStringList_UI64::sorted_half_dets() const {
    return sorted_half_dets_;
}

const std::pair<size_t, size_t>&
SortedStringList_UI64::range(const UI64Determinant::bit_t& d) const {
    return first_string_range_.at(d);
}

// UI64Determinant::UI64Determinant() {}
// UI64Determinant::UI64Determinant(const STLBitsetDeterminant& d) {
//    for (int i = 0; i < 64; ++i) {
//        set_alfa_bit(i, d.get_alfa_bit(i));
//        set_beta_bit(i, d.get_beta_bit(i));
//    }
//}

// bool UI64Determinant::get_alfa_bit(bit_t n) const { return (0 != (a_ & (bit_t(1) << n))); }

// bool UI64Determinant::get_beta_bit(bit_t n) const { return (0 != (b_ & (bit_t(1) << n))); }

///// Set the value of an alpha bit
// void UI64Determinant::set_alfa_bit(bit_t n, bool v) {
//    if (v) {
//        a_ |= (bit_t(1) << n);
//    } else {
//        a_ &= ~(bit_t(1) << n);
//    }
//}
////            alfa_bits_ ^= (-bit_t(v) ^ alfa_bits_) & (1 << n);}
///// Set the value of a beta bit
// void UI64Determinant::set_beta_bit(bit_t n, bool v) {
//    if (v) {
//        b_ |= (bit_t(1) << n);
//    } else {
//        b_ &= ~(bit_t(1) << n);
//    }
//}

// UI64Determinant::bit_t UI64Determinant::get_bits(STLBitsetDeterminant::SpinType spin_type) const
// {
//    return (spin_type == STLBitsetDeterminant::SpinType::AlphaSpin) ? a_ : b_;
//}

// void UI64Determinant::zero_spin(STLBitsetDeterminant::SpinType spin_type) {
//    if (spin_type == STLBitsetDeterminant::SpinType::AlphaSpin) {
//        a_ = bit_t(0);
//    } else {
//        b_ = bit_t(0);
//    }
//}

// bool UI64Determinant::less_than(const UI64Determinant& rhs, const UI64Determinant& lhs) {
//    if (rhs.b_ < lhs.b_) {
//        return true;
//    } else if (rhs.b_ > lhs.b_) {
//        return false;
//    }
//    return rhs.a_ < lhs.a_;
//}

// bool UI64Determinant::reverse_less_then(const UI64Determinant& rhs, const UI64Determinant& lhs) {
//    if (rhs.a_ < lhs.a_) {
//        return true;
//    } else if (rhs.a_ > lhs.a_) {
//        return false;
//    }
//    return rhs.b_ < lhs.b_;
//}

// bool UI64Determinant::operator==(const UI64Determinant& lhs) const {
//    return ((a_ == lhs.a_) and (b_ == lhs.b_));
//}

// bool UI64Determinant::operator<(const UI64Determinant& lhs) const {
//    if (b_ < lhs.b_) {
//        return true;
//    } else if (b_ > lhs.b_) {
//        return false;
//    }
//    return a_ < lhs.a_;
//}

size_t SortedStringList_UI64::add(size_t pos) const { return map_to_hashdets_[pos]; }
}
}
