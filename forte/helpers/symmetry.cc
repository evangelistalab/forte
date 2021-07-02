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

#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "helpers/string_algorithms.h"
#include "helpers/symmetry.h"

namespace forte {

Symmetry::Symmetry(std::string point_group) {
    to_upper_string(point_group); // capitalize
    // check if this is a valid point group label
    auto search = std::find(__point_groups.cbegin(), __point_groups.cend(), point_group);
    if (search == __point_groups.end()) {
        std::string msg = "Point group " + point_group + " not found or supported by forte.\n";
        throw std::runtime_error(msg);
    }
    pg_ = point_group;
}

const std::string& Symmetry::point_group_label() const { return pg_; }

const std::vector<std::string>& Symmetry::irrep_labels() const {
    return __pg_to_irrep_labels.at(pg_);
}

const std::string& Symmetry::irrep_label(size_t h) const {
    if (h >= nirrep()) {
        std::string msg = "(Symmetry) irrep " + std::to_string(h) +
                          " is not valid (should be less than) " + std::to_string(nirrep()) +
                          " .\n";
        throw std::runtime_error(msg);
    }
    return __pg_to_irrep_labels.at(pg_)[h];
}

size_t Symmetry::irrep_label_to_index(const std::string& label) const {
    const auto& irrep_labels = __pg_to_irrep_labels.at(pg_);
    auto search = find_case_insensitive(label, irrep_labels);
    if (search == irrep_labels.end()) {
        std::string msg = "\n  Irrep label " + label + " not found in point group " + pg_ + ".\n" +
                          "  Allowed values are: " + join(irrep_labels, ", ") + "\n";
        throw std::runtime_error(msg);
    }
    return std::distance(irrep_labels.begin(), search);
}

size_t Symmetry::nirrep() const { return __pg_to_irrep_labels.at(pg_).size(); }

size_t Symmetry::irrep_product(size_t h, size_t g) { return h ^ g; }

} // namespace forte
