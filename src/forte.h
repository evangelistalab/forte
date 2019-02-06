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

#include "base_classes/forte_options.h"

namespace forte {

class ForteIntegrals;

void forte_options(ForteOptions& options);

std::pair<int, int> startup();
void banner();
void cleanup();

void read_options(ForteOptions& options);
psi::SharedWavefunction run_forte(psi::SharedWavefunction ref_wfn, psi::Options& options);

std::shared_ptr<MOSpaceInfo> make_mo_space_info(psi::SharedWavefunction ref_wfn,
                                                std::shared_ptr<ForteOptions> options);

psi::SharedMatrix make_aosubspace_projector(psi::SharedWavefunction ref_wfn, psi::Options& options);

void make_ci_nos(psi::SharedWavefunction ref_wfn, psi::Options& options,
                 std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

double forte_old_methods(psi::SharedWavefunction ref_wfn, psi::Options& options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info);

void forte_old_options(ForteOptions& options);
} // namespace forte
