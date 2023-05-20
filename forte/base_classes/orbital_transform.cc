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

#include "ambit/tensor.h"

#include "base_classes/orbital_transform.h"
#include "base_classes/rdms.h"

#include "integrals/integrals.h"

#include "orbital-helpers/localize.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/mrpt2_nos.h"
#include "orbital-helpers/ci-no/ci-no.h"
#include "orbital-helpers/ci-no/mrci-no.h"
#include "orbital-helpers/semi_canonicalize.h"

namespace forte {

OrbitalTransform::OrbitalTransform(std::shared_ptr<ForteIntegrals> ints,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ints_(ints), mo_space_info_(mo_space_info) {}

std::unique_ptr<OrbitalTransform>
make_orbital_transformation(const std::string& type, std::shared_ptr<SCFInfo> scf_info,
                            std::shared_ptr<ForteOptions> options,
                            std::shared_ptr<ForteIntegrals> ints,
                            std::shared_ptr<MOSpaceInfo> mo_space_info) {

    std::unique_ptr<OrbitalTransform> orb_t;

    if (type == "LOCAL") {
        orb_t = std::make_unique<Localize>(options, ints, mo_space_info);
    } else if (type == "MP2NO") {
        orb_t = std::make_unique<MP2_NOS>(scf_info, options, ints, mo_space_info);
    } else if (type == "CINO") {
        orb_t = std::make_unique<CINO>(options, ints, mo_space_info);
    } else if (type == "MRCINO") {
        orb_t = std::make_unique<MRCINO>(scf_info, options, ints, mo_space_info);
    } else if (type == "MRPT2NO") {
        // perform reference CI calculation
        auto as_type = options->get_str("ACTIVE_SPACE_SOLVER");
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {"RESTRICTED_DOCC"});
        auto state_weights_map = make_state_weights_map(options, mo_space_info);
        auto state_nroots_map = to_state_nroots_map(state_weights_map);
        auto as_solver = make_active_space_solver(as_type, state_nroots_map, scf_info,
                                                  mo_space_info, as_ints, options);
        auto state_energies = as_solver->compute_energy();

        // compute RDMs
        auto rdm_level = (options->get_str("THREEPDC") == "ZERO" ? 2 : 3);
        auto rdms =
            as_solver->compute_average_rdms(state_weights_map, rdm_level, RDMsType::spin_free);

        // initialize
        orb_t = std::make_unique<MRPT2_NOS>(rdms, scf_info, options, ints, mo_space_info);
    } else {
        throw std::runtime_error("Orbital type " + type + " is not supported!");
    }

    return orb_t;
}

} // namespace forte
