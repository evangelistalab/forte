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

#ifndef _symmetry_h_
#define _symmetry_h_

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "helpers/string_algorithms.h"

namespace forte {

enum class PointGroup { C1, Cs, Ci, C2, C2h, C2v, D2, D2h };

class Symmetry {
  public:
    Symmetry(std::string point_group) {
        to_upper_string(point_group); // capitalize
        auto search = __str_to_pg.find(point_group);
        if (search != __str_to_pg.end()) {
            pg_ = __str_to_pg.at(point_group);
        } else {
            std::string msg = "Point group " + point_group + " not found or supported by forte.\n";
            throw std::runtime_error(msg);
        }
    }
    Symmetry(PointGroup pg) : pg_(pg) {}

    const std::vector<std::string>& irrep_labels() const { return __irrep_labels.at(pg_); }
    const std::string& irrep_label(size_t h) const { return __irrep_labels.at(pg_)[h]; }
    size_t nirrep() const { return __irrep_labels.at(pg_).size(); }
    size_t irrep_product(size_t h, size_t g) const { return h ^ g; }
    std::string point_group_label() const {
        for (const auto& p : __str_to_pg) {
            if (p.second == pg_) {
                return p.first;
            }
        }
        return std::string("");
    }

  private:
    PointGroup pg_;
    const std::map<std::string, PointGroup> __str_to_pg{
        {"C1", PointGroup::C1}, {"CS", PointGroup::Cs},   {"CI", PointGroup::Ci},
        {"C2", PointGroup::C2}, {"C2H", PointGroup::C2h}, {"C2V", PointGroup::C2v},
        {"D2", PointGroup::D2}, {"D2H", PointGroup::D2h}};

    const std::map<PointGroup, std::vector<std::string>> __irrep_labels{
        {PointGroup::C1, {"A"}},
        {PointGroup::Cs, {"Ap", "App"}},
        {PointGroup::Ci, {"Ag", "Au"}},
        {PointGroup::C2, {"A", "B"}},
        {PointGroup::C2h, {"Ag", "Bg", "Au", "Bu"}},
        {PointGroup::C2v, {"A1", "A2", "B1", "B2"}},
        {PointGroup::D2, {"A", "B1", "B2", "B3"}},
        {PointGroup::D2h, {"Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"}}};
};

} // namespace forte
#endif
