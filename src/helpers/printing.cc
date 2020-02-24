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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "printing.h"

using namespace psi;

namespace forte {

void print_h1(const std::string& text, bool centerd, const std::string& left_separator,
              const std::string& right_separator) {
    int text_width = static_cast<int>(text.size());
    int margin_width = 78 - text_width - 2;
    int left_margin_width = centerd ? margin_width / 2 : 0;
    int right_margin_width = margin_width - left_margin_width;
    outfile->Printf("\n\n  %s %s %s\n", std::string(left_margin_width, '-').c_str(), text.c_str(),
                    std::string(right_margin_width, '-').c_str());
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

} // namespace forte
