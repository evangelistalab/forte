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

void to_lower_string(std::string& s) { std::transform(s.begin(), s.end(), s.begin(), ::tolower); }

std::string lower_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

std::string join(const std::vector<std::string>& vec_str, const std::string& sep) {
    if (vec_str.size() == 0)
        return std::string();

    std::string ss;

    std::for_each(vec_str.begin(), vec_str.end() - 1, [&](const std::string& s) { ss += s + sep; });
    ss += vec_str.back();

    return ss;
}

std::vector<std::string>::const_iterator
find_case_insensitive(const std::string& str, const std::vector<std::string>& vec) {
    auto ret = std::find_if(vec.cbegin(), vec.cend(), [&str](const std::string& s) {
        if (s.size() != str.size())
            return false;
        return std::equal(s.cbegin(), s.cend(), str.cbegin(), str.cend(),
                          [](auto c1, auto c2) { return std::toupper(c1) == std::toupper(c2); });
    });
    return ret;
}
} // namespace forte
