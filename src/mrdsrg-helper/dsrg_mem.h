/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _dsrg_mem_h_
#define _dsrg_mem_h_

#include <vector>
#include <string>
#include <map>

#include "base_classes/mo_space_info.h"

namespace forte {

class DSRG_MEM {
  public:
    /// Default constructor
    DSRG_MEM();

    /// Constructor using a map from MO labels to their sizes
    DSRG_MEM(int64_t mem_avai, std::map<char, size_t> label_to_size);

    /// Set memory currently available
    void set_mem_avai(int64_t mem_avai) { mem_avai_ = mem_avai; }

    /// Set the map from MO label to the corresponding size
    void set_label_to_size(std::map<char, size_t> label_to_size) { label_to_size_ = label_to_size; }

    /// Add entry using description and its number of elements
    void add_entry(const std::string& des, const size_t& mem_use, bool subtract = true);

    /// Add entry using the description and the block labels
    void add_entry(const std::string& des, const std::string& labels, int multiple,
                   bool subtract = true);

    /// Print the current data
    void print(const std::string& name);

  private:
    /// Memory currently available
    int64_t mem_avai_;

    /// Map from description to the number of elements
    std::map<std::string, size_t> data_;

    /// Map from MO labels to sizes
    std::map<char, size_t> label_to_size_;
};
} // namespace forte

#endif // _dsrg_mem_h_
