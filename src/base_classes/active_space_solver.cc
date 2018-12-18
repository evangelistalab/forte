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

#include "psi4/libmints/wavefunction.h"

#include "base_classes/state_info.h"
#include "helpers/mo_space_info.h"
#include "integrals/integrals.h"

#include "base_classes/active_space_solver.h"
#include "fci/fci.h"

namespace forte {

ActiveSpaceSolver::ActiveSpaceSolver(StateInfo state, std::shared_ptr<ForteIntegrals> ints,
                                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : states_weights_({{state, 1.0}}), ints_(ints), mo_space_info_(mo_space_info) {}

ActiveSpaceSolver::ActiveSpaceSolver(
    const std::vector<std::pair<StateInfo, double>>& states_weights,
    std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : states_weights_(states_weights), ints_(ints), mo_space_info_(mo_space_info) {}

std::shared_ptr<ActiveSpaceSolver>
make_active_space_solver(const std::string& type, StateInfo state, std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info) {
    if (type == "FCI") {
        return std::make_shared<FCI>(state, options, ints, mo_space_info);
    }
    throw psi::PSIEXCEPTION("make_active_space_solver: type = " + type + " was not recognized");
    return std::shared_ptr<ActiveSpaceSolver>();
}

} // namespace forte
