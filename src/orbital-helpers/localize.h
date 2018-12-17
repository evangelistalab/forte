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

#ifndef _localize_h_
#define _localize_h_

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/local.h"

#include "integrals/integrals.h"
#include "base_classes/reference.h"
#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"

#include "base_classes/orbital_transform.h"

namespace forte {

class LOCALIZE : public OrbitalTransform {
  public:
    LOCALIZE(StateInfo state, std::shared_ptr<SCFInfo> scf_info,
             std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
             std::shared_ptr<MOSpaceInfo> mo_space_info);

    psi::SharedMatrix get_Ua();
    psi::SharedMatrix get_Ub();


    // Sets the orbitals to localize, bool to split localize
    void set_orbital_space( std::vector<int>& orbital_spaces );

    // Call to localize, class handles options
    void localize();

    // Returns unitary matrix that transforms
    // RHF -> local basis
    psi::SharedMatrix get_U();

    void compute_transformation();

  private:
    std::shared_ptr<SCFInfo> scf_info_;
    std::shared_ptr<ForteOptions> options_;
    std::shared_ptr<ForteIntegrals> ints_;

    psi::SharedMatrix Ua_;
    psi::SharedMatrix Ub_;

    // orbitals to localize
    std::vector<int> orbital_spaces_;

    // Pipek-Mezey or Boys
    std::string local_method_;
};
} // namespace forte

#endif
