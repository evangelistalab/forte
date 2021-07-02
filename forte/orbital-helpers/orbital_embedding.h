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

#ifndef _orbital_embedding_h_
#define _orbital_embedding_h_

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"


#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"

namespace forte {

void make_avas(psi::SharedWavefunction ref_wfn, std::shared_ptr<ForteOptions> options, psi::SharedMatrix Ps);

std::shared_ptr<MOSpaceInfo> make_embedding(psi::SharedWavefunction ref_wfn, std::shared_ptr<ForteOptions> options,
                                            psi::SharedMatrix Pf, int nbf_A,
                                            std::shared_ptr<MOSpaceInfo> mo_space_info);

} // namespace forte

#endif // _orbital_embedding_h_
