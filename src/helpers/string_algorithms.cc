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

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

#include "string_algorithms.h"

namespace forte {

std::vector<std::string> split_string(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos) {
        if (pos != prev) { // avoid repetitions
            strings.push_back(str.substr(prev, pos - prev));
        }
        prev = pos + 1;
    }
    // To get the last substring (or only, if delimiter is not found)
    if (str.substr(prev).size() > 0)
        strings.push_back(str.substr(prev));

    return strings;
}

void to_upper_string(std::string& s) { std::transform(s.begin(), s.end(), s.begin(), ::toupper); }

std::string upper_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

} // namespace forte
