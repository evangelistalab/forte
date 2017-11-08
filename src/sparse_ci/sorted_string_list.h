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

#ifndef _sorted_string_list_h_
#define _sorted_string_list_h_

#include "stl_bitset_string.h"
#include "stl_bitset_determinant.h"
#include "ui64_determinant.h"
#include "../determinant_hashvector.h"
#include "../fci/fci_integrals.h"

namespace psi {
namespace forte {

/**
 * @brief The SortedStringList class
 * Stores determinants as a sorted string list.
 */
class SortedStringList {
  public:
    SortedStringList(const DeterminantHashVec& space, std::shared_ptr<FCIIntegrals> fci_ints,
                     DetSpinType sorted_string_spin);

    const std::vector<STLBitsetDeterminant>& sorted_dets() const;
    const std::vector<STLBitsetDeterminant>& sorted_half_dets() const;

    const std::pair<size_t, size_t>& range(STLBitsetDeterminant d) const;

    /// Find the determinant d in the range first and last.
    /// If d is found at position pos, then first = pos + 1 and the function returns pos
    /// If d is not found at position pos, then we return num_dets_ (out of range) and first is set
    /// to the position of the determinant that has lexicographic value greather than d
    size_t find(const STLBitsetDeterminant& d, size_t& first, size_t& last) const;

  protected:
    int nmo_ = 0;
    size_t num_dets_ = 0;
    DetSpinType zero_spin_type_;
    std::vector<STLBitsetDeterminant> sorted_half_dets_;
    std::vector<STLBitsetDeterminant> sorted_dets_;
    det_hash<std::pair<size_t, size_t>> first_string_range_;
};

/**
 * @brief The SortedStringList class
 * Stores determinants as a sorted string list.
 */
class SortedStringList_UI64 {
  public:
    SortedStringList_UI64(const DeterminantHashVec& space, std::shared_ptr<FCIIntegrals> fci_ints,
                          DetSpinType sorted_string_spin);

    const std::vector<UI64Determinant>& sorted_dets() const;
    const std::vector<UI64Determinant::bit_t>& sorted_half_dets() const;

    const std::pair<size_t, size_t>& range(const UI64Determinant::bit_t& d) const;
    size_t add(size_t pos) const;

  protected:
    int nmo_ = 0;
    size_t num_dets_ = 0;
    DetSpinType sorted_spin_type_;
    std::vector<UI64Determinant::bit_t> sorted_half_dets_;
    std::vector<UI64Determinant> sorted_dets_;
    std::vector<size_t> map_to_hashdets_;
    std::unordered_map<UI64Determinant::bit_t, std::pair<size_t, size_t>> first_string_range_;
};
}
}

#endif // _sigma_vector_direct_h_
