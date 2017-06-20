/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <libmoinfo/libmoinfo.h>

#include "excitation_determinant.h"

using namespace std;
using namespace psi;

#include <psi4-dec.h>

namespace psi {
namespace forte {

ExcitationDeterminant::ExcitationDeterminant() : naex_(0), nbex_(0) {}

ExcitationDeterminant::~ExcitationDeterminant() {}

ExcitationDeterminant::ExcitationDeterminant(const ExcitationDeterminant& det)
    : naex_(det.naex_), nbex_(det.nbex_), alpha_ops_(det.alpha_ops_), beta_ops_(det.beta_ops_) {}

ExcitationDeterminant& ExcitationDeterminant::operator=(const ExcitationDeterminant& rhs) {
    naex_ = rhs.naex_;
    nbex_ = rhs.nbex_;
    alpha_ops_ = rhs.alpha_ops_;
    beta_ops_ = rhs.beta_ops_;
    return *this;
}

/**
 * Print the determinant
 */
void ExcitationDeterminant::print() {
    outfile->Printf("\n  {");
    for (int p = 0; p < nbex_; ++p) {
        outfile->Printf(" %d", bann(p));
    }
    outfile->Printf("->");
    for (int p = 0; p < nbex_; ++p) {
        outfile->Printf(" %d", bcre(p));
    }
    outfile->Printf("}{");
    for (int p = 0; p < naex_; ++p) {
        outfile->Printf(" %d", aann(p));
    }
    outfile->Printf("->");
    for (int p = 0; p < naex_; ++p) {
        outfile->Printf(" %d", acre(p));
    }
    outfile->Printf("}");
    
}

void ExcitationDeterminant::to_pitzer(const std::vector<int>& qt_to_pitzer) {
    for (int p = 0; p < naex_; ++p) {
        alpha_ops_[2 * p] = qt_to_pitzer[aann(p)];
        alpha_ops_[2 * p + 1] = qt_to_pitzer[acre(p)];
    }
    for (int p = 0; p < nbex_; ++p) {
        beta_ops_[2 * p] = qt_to_pitzer[bann(p)];
        beta_ops_[2 * p + 1] = qt_to_pitzer[bcre(p)];
    }
}
}
} // End Namespaces
