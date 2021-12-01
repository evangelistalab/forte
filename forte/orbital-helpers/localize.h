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

#ifndef _localize_h_
#define _localize_h_


#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/local.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "integrals/integrals.h"
#include "base_classes/forte_options.h"
#include "base_classes/rdms.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/scf_info.h"

#include "base_classes/orbital_transform.h"

namespace forte {

class Localize : public OrbitalTransform {
  public:
    Localize(std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
             std::shared_ptr<MOSpaceInfo> mo_space_info);

    // Compute the rotation matrices
    void compute_transformation();

    // Return the matrices
    psi::SharedMatrix get_Ua();
    psi::SharedMatrix get_Ub();

    // Sets the orbitals to localize by first/last indices
    void set_orbital_space(std::vector<int>& orbital_spaces);

    // Sets the orbitals to localize by space label from MOSpaceInfo
    void set_orbital_space(std::vector<std::string>& labels);

  private:
    psi::SharedMatrix Ua_;
    psi::SharedMatrix Ub_;

    // orbitals to localize
    std::vector<int> orbital_spaces_;

    // Pipek-Mezey or Boys
    std::string local_method_;
};
} // namespace forte

#endif
