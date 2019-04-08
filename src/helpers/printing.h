/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _printing_h_
#define _printing_h_

namespace forte {

enum class PrintLevel { Quiet = 0, Mini = 1, Default = 2, Debug = 3 };

/**
 * @brief print_method_banner Print a banner for a method
 * @param text A vector of strings to print in the banner. Each string is a line.
 * @param separator A string The separator used in the banner (defalut = "-").
 */
void print_method_banner(const std::vector<std::string>& text, const std::string& separator = "-");

/**
 * @brief print_timing Print the string "Timing for <text>: <padding> X.XXX s." to the output file
 * @param text The text that comes after "Timing for"
 * @param seconds The timing in seconds
 */
void print_timing(const std::string& text, double seconds);

} // namespace forte

#endif // _helpers_h_
