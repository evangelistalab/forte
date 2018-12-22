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
#include "base_classes/scf_info.h"
#include "helpers/mo_space_info.h"
#include "integrals/integrals.h"

#include "base_classes/active_space_solver.h"
#include "fci/fci_solver.h"
#include "sci/aci.h"
#include "sci/asci.h"
#include "sci/fci_mo.h"

namespace forte {

ActiveSpaceSolver::ActiveSpaceSolver(StateInfo state, std::shared_ptr<MOSpaceInfo> mo_space_info,
                                     std::shared_ptr<ForteIntegrals> ints)
    : states_weights_({{state, 1.0}}), mo_space_info_(mo_space_info), ints_(ints) {
    // Allocate and compute active space integrals
    make_active_space_ints();
}

ActiveSpaceSolver::ActiveSpaceSolver(
    const std::vector<std::pair<StateInfo, double>>& states_weights,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints)
    : states_weights_(states_weights), mo_space_info_(mo_space_info), ints_(ints) {
    // Allocate and compute active space integrals
    make_active_space_ints();
}

void ActiveSpaceSolver::make_active_space_ints() {
    // get the active/core vectors
    active_mo_ = mo_space_info_->get_corr_abs_mo(active_mo_space_);
    core_mo_ = mo_space_info_->get_corr_abs_mo(core_mo_space_);

    // allocate the active space integral object
    as_ints_ = std::make_shared<ActiveSpaceIntegrals>(ints_, active_mo_, core_mo_);

    // grab the integrals from the ForteIntegrals object
    ambit::Tensor tei_active_aa =
        ints_->aptei_aa_block(active_mo_, active_mo_, active_mo_, active_mo_);
    ambit::Tensor tei_active_ab =
        ints_->aptei_ab_block(active_mo_, active_mo_, active_mo_, active_mo_);
    ambit::Tensor tei_active_bb =
        ints_->aptei_bb_block(active_mo_, active_mo_, active_mo_, active_mo_);
    as_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    as_ints_->compute_restricted_one_body_operator();
}

std::shared_ptr<ActiveSpaceSolver> make_active_space_solver(
    const std::string& type, StateInfo state, std::shared_ptr<SCFInfo> scf_info,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints,
    std::shared_ptr<ForteOptions> options) {

    std::shared_ptr<ActiveSpaceSolver> solver;
    if (type == "FCI") {
        solver = std::make_shared<FCISolver>(state, mo_space_info, ints);
    } else if (type == "ACI") {
        solver = std::make_shared<AdaptiveCI>(std::make_shared<StateInfo>(state), scf_info, options,
                                              ints, mo_space_info);
    } else if (type == "CAS") {
        solver = std::make_shared<FCI_MO>(scf_info, options, ints, mo_space_info);
    } else if (type == "ASCI") {
        solver = std::make_shared<ASCI>(std::make_shared<StateInfo>(state), scf_info, options, ints,
                                        mo_space_info);
    } else {
        throw psi::PSIEXCEPTION("make_active_space_solver: type = " + type + " was not recognized");
    }
    solver->set_options(options);
    return solver;
}

} // namespace forte
