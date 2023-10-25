/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <utility>
#include <iomanip> // for std::setw

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

class table_printer {
  public:
    void add_string_data(const std::vector<std::pair<std::string, std::string>>& data) {
        info_string.insert(info_string.end(), data.begin(), data.end());
    }

    void add_bool_data(const std::vector<std::pair<std::string, bool>>& data) {
        info_bool.insert(info_bool.end(), data.begin(), data.end());
    }

    void add_double_data(const std::vector<std::pair<std::string, double>>& data) {
        info_double.insert(info_double.end(), data.begin(), data.end());
    }

    void add_int_data(const std::vector<std::pair<std::string, int>>& data) {
        info_int.insert(info_int.end(), data.begin(), data.end());
    }

    void add_timing_data(const std::vector<std::pair<std::string, double>>& data) {
        info_timing.insert(info_timing.end(), data.begin(), data.end());
    }

    std::string get_table(const std::string& title) const {
        std::ostringstream oss;
        print_h2(oss, title);
        print_data(oss, info_string, [](const std::string& val) { return val; });
        print_data(oss, info_bool,
                   [](bool val) { return val ? std::string("TRUE") : std::string("FALSE"); });
        print_data(oss, info_double, [](double val) { return format_double(val, "%15.3e"); });
        print_data(oss, info_int, [](int val) {
            char buffer[16];
            std::snprintf(buffer, sizeof(buffer), "%15d", val);
            return std::string(buffer);
        });
        print_data(oss, info_timing,
                   [](double val) { return format_double(val, "%15.3f") + " s"; });
        oss << "\n";
        return oss.str();
    }

  private:
    std::vector<std::pair<std::string, std::string>> info_string;
    std::vector<std::pair<std::string, bool>> info_bool;
    std::vector<std::pair<std::string, double>> info_double;
    std::vector<std::pair<std::string, int>> info_int;
    std::vector<std::pair<std::string, double>> info_timing;

    static void print_h2(std::ostream& os, const std::string& title) {
        os << "\n\n  ==> " << title << " <==\n";
    }

    static std::string format_double(double val, const char* format) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), format, val);
        return std::string(buffer);
    }

    template <typename T, typename Func>
    static void print_data(std::ostream& os, const std::vector<std::pair<std::string, T>>& data,
                           Func formatter) {
        for (const auto& item : data) {
            os << "\n    " << std::left << std::setw(40) << item.first << " " << std::right
               << std::setw(15) << formatter(item.second);
        }
    }
};

/// @brief Return the label for a spin state
/// @param twiceS Twice the spin quantum number S (multiplicity - 1)
const std::string& s2_label(int twiceS);

} // namespace forte
