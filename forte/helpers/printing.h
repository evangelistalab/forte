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

#ifndef _printing_h_
#define _printing_h_

#include <vector>
#include <string>

namespace forte {

enum class PrintLevel { Quiet = 0, Mini = 1, Default = 2, Debug = 3 };

/**
 * @brief print_h1 Print a header
 * @param text The string to print in the header.
 * @param left_separator The left separator (default = "-")
 * @param right_separator The right separator (default = "-")
 */
void print_h1(const std::string& text, bool centerd = true, const std::string& left_filler = "-",
              const std::string& right_filler = "-");

/**
 * @brief print_h2 Print a header
 * @param text The string to print in the header.
 * @param left_separator The left separator (default = "==>")
 * @param right_separator The right separator (default = "<==")
 */
void print_h2(const std::string& text, const std::string& left_separator = "==>",
              const std::string& right_separator = "<==");

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

/**
 * @brief print_selected_options Print options summary
 * @param title The header of this printing block
 * @param info_string The options of string type
 * @param info_bool The options of Boolean type
 * @param info_double The options of double type
 * @param info_int The options of integer type
 */
void print_selected_options(const std::string& title,
                            const std::vector<std::pair<std::string, std::string>>& info_string,
                            const std::vector<std::pair<std::string, bool>>& info_bool,
                            const std::vector<std::pair<std::string, double>>& info_double,
                            const std::vector<std::pair<std::string, int>>& info_int);
} // namespace forte

#endif // _helpers_h_
