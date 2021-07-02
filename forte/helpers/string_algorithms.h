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

#ifndef _string_algorithms_h_
#define _string_algorithms_h_

#include <sstream>
#include <string>
#include <vector>

namespace forte {

std::vector<std::string> split_string(const std::string& str, const std::string& delimiter);

/// Convert all lower case letters in a string to the upper case
void to_upper_string(std::string& s);
std::string upper_string(std::string s);

/// Convert all lower case letters in a string to the upper case
void to_lower_string(std::string& s);
std::string lower_string(std::string s);

/// Join a vector of strings using a separator
std::string join(const std::vector<std::string>& vec_str, const std::string& sep = ",");

/// Convert a number to string with a given precision
template <typename T> std::string to_string_with_precision(const T val, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << val;
    return out.str();
}

/// Find a string in a vector of strings in a case insensitive
std::vector<std::string>::const_iterator find_case_insensitive(const std::string& str,
                                                               const std::vector<std::string>& vec);

} // namespace forte

#endif // _string_algorithms_h_
