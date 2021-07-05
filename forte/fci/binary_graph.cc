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

#include "binary_graph.hpp"

#ifdef BIN_GRAPH_TEST
#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

bool test_string(std::vector<bool>& string, int nones, int nbits) {
    if (string.size() != nbits) {
        psi::outfile->Printf(
            "\n  size_t abs_add(std::vector<bool>& string) called with string of %zu bits",
            string.size());
        exit(1);
    }
    int nos = 0;
    for (auto b : string) {
        if (b)
            nos += 1;
    }
    if (nos != nones) {
        psi::outfile->Printf(
            "\n  size_t abs_add(std::vector<bool>& string) called with string with on %d bits",
            nos);
        exit(1);
    }
    return true;
}

bool test_string(bool* string, int nones, int nbits) {
    int nos = 0;
    for (int n = 0; n < nbits; ++n) {
        if (string[n])
            nos += 1;
    }
    if (nos != nones) {
        psi::outfile->Printf(
            "\n  size_t abs_add(std::vector<bool>& string) called with string with on %d bits ",
            nos);
        exit(1);
    }
    return true;
}
#endif
