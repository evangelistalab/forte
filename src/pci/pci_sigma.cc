/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "pci_sigma.h"

namespace forte {

PCISigmaVector::PCISigmaVector(det_hashvec &dets_hashvec, std::vector<double> &C, double spawning_threshold)
    : SigmaVector(dets_hashvec.size()), dets_(dets_hashvec), C_(C), spawning_threshold_(spawning_threshold)
{

}

void PCISigmaVector::compute_sigma(psi::SharedVector sigma, psi::SharedVector b) {

}

void PCISigmaVector::get_diagonal(psi::Vector& diag) {

}

void PCISigmaVector::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states) {

}

}
