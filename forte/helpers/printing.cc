/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <format>
#include <vector>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"

#include "printing.h"

using namespace psi;

std::vector<std::string> __s2_labels{
    "singlet", "doublet", "triplet", "quartet", "quintet", "sextet", "septet", "octet",
    "nonet",   "decet",   "11-et",   "12-et",   "13-et",   "14-et",  "15-et",  "16-et",
    "17-et",   "18-et",   "19-et",   "20-et",   "21-et",   "22-et",  "23-et",  "24-et"};

namespace forte {

PrintLevel int_to_print_level(int level) {
    switch (level) {
    case 0:
        return PrintLevel::Quiet;
    case 1:
        return PrintLevel::Brief;
    case 2:
        return PrintLevel::Default;
    case 3:
        return PrintLevel::Verbose;
    case 4:
        return PrintLevel::Debug;
    default:
        throw std::out_of_range("Invalid PrintLevel value");
    }
}

/// @brief Return the string representation of a PrintLevel
std::string to_string(PrintLevel level) {
    switch (level) {
    case PrintLevel::Quiet:
        return "Quiet";
    case PrintLevel::Brief:
        return "Brief";
    case PrintLevel::Default:
        return "Default";
    case PrintLevel::Verbose:
        return "Verbose";
    case PrintLevel::Debug:
        return "Debug";
    default:
        throw std::out_of_range("Invalid PrintLevel value");
    }
}

bool operator>(PrintLevel a, PrintLevel b) { return static_cast<int>(a) > static_cast<int>(b); }

bool operator<(PrintLevel a, PrintLevel b) { return static_cast<int>(a) < static_cast<int>(b); }

void print_h1(const std::string& text, bool centerd, const std::string& left_filler,
              const std::string& right_filler) {
    int text_width = static_cast<int>(text.size());
    int margin_width = 78 - text_width - 2;
    int left_margin_width = centerd ? margin_width / 2 : 0;
    int right_margin_width = margin_width - left_margin_width;
    outfile->Printf("\n\n\n  %s %s %s", std::string(left_margin_width, left_filler[0]).c_str(),
                    text.c_str(), std::string(right_margin_width, right_filler[0]).c_str());
}

void print_h2(const std::string& text, const std::string& left_separator,
              const std::string& right_separator) {
    outfile->Printf("\n\n  %s %s %s\n", left_separator.c_str(), text.c_str(),
                    right_separator.c_str());
}

void print_method_banner(const std::vector<std::string>& text, const std::string& separator) {
    size_t max_width = 80;

    size_t width = 0;
    for (auto& line : text) {
        width = std::max(width, line.size());
    }

    std::string tab((max_width - width - 4) / 2, ' ');
    std::string header(width + 4, char(separator[0]));

    outfile->Printf("\n\n%s%s\n", tab.c_str(), header.c_str());
    for (auto& line : text) {
        size_t padding = 2 + (width - line.size()) / 2;
        std::string padding_str(padding, ' ');
        outfile->Printf("%s%s%s\n", tab.c_str(), padding_str.c_str(), line.c_str());
    }
    outfile->Printf("%s%s\n", tab.c_str(), header.c_str());
}

void print_timing(const std::string& text, double seconds) {
    int nspaces = 43 - static_cast<int>(text.size());
    if (nspaces >= 0) {
        std::string padding(nspaces, ' ');
        outfile->Printf("\n  Timing for %s: %s%9.3f s.", text.c_str(), padding.c_str(), seconds);
    } else {
        outfile->Printf("\n  void print_timing(...): Cannot fit the following string\n%s\n",
                        text.c_str());
        exit(1);
    }
}

const std::string& s2_label(int twiceS) { return __s2_labels.at(twiceS); }

std::string matrix_to_string(const psi::Matrix& mat) {
    std::string str = mat.name() + ":\n";
    auto nirrep = mat.nirrep();
    auto nsopi = mat.rowspi();
    auto nmopi = mat.colspi();
    for (int h = 0; h < nirrep; ++h) {
        str += std::format("  irrep {}:\n", h);
        for (int mu = 0; mu < nsopi.get(h); ++mu) {
            str += std::format("    {:>3}  ", mu);
            for (int nu = 0; nu < nmopi.get(h); ++nu) {
                str += std::format("{:10.6f} ", mat.get(h, mu, nu));
            }
            str += "\n";
        }
    }
    return str;
}

} // namespace forte