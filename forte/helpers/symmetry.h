/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _symmetry_h_
#define _symmetry_h_

#include <map>
#include <string>
#include <vector>

namespace forte {

/**
 * @brief The Symmetry class provides symmetry information for a given point group
 */
class Symmetry {
  public:
    /// Constructor
    /// @param point_group the molecular point group label
    Symmetry(std::string point_group);

    /// @return the label of this point group
    const std::string& point_group_label() const;
    /// @return a vector of irrep labels
    const std::vector<std::string>& irrep_labels() const;
    /// @return the label of irrep h
    const std::string& irrep_label(size_t h) const;
    /// @return the label of irrep h
    size_t irrep_label_to_index(const std::string& label) const;
    /// @return the number of irreps
    size_t nirrep() const;
    /// @return the product of irreps h and g
    static size_t irrep_product(size_t h, size_t g);

  private:
    /// the point group label
    std::string pg_;
    /// the point group labels
    const std::vector<std::string> __point_groups{"C1",  "CS",  "CI", "C2",
                                                  "C2H", "C2V", "D2", "D2H"};
    /// a map point group -> irrep labels (in Cotton order)
    const std::map<std::string, std::vector<std::string>> __pg_to_irrep_labels{
        {"C1", {"A"}},
        {"CS", {"Ap", "App"}},
        {"CI", {"Ag", "Au"}},
        {"C2", {"A", "B"}},
        {"C2H", {"Ag", "Bg", "Au", "Bu"}},
        {"C2V", {"A1", "A2", "B1", "B2"}},
        {"D2", {"A", "B1", "B2", "B3"}},
        {"D2H", {"Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"}}};
};

} // namespace forte
#endif
