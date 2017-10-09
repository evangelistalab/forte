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
                     bool flip_spin);

    const std::vector<STLBitsetDeterminant>& sorted_dets();
    const det_hash<std::pair<size_t, size_t>>& first_string_range();

  protected:
    int nmo_ = 0;
    size_t num_dets_ = 0;
    std::vector<STLBitsetDeterminant> sorted_dets_;
    det_hash<std::pair<size_t, size_t>> first_string_range_;
};
}
}

#endif // _sigma_vector_direct_h_
