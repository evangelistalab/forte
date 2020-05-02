/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <psi4/libmints/wavefunction.h>

#include "base_classes/mo_space_info.h"
#include "base_classes/forte_options.h"
#include "fci/fci_solver.h"
#include "casscf/casscf.h"
#include "sci/aci.h"
#include "sci/asci.h"
#include "sci/fci_mo.h"
#include "pci/pci.h"
#include "ci_ex_states/excited_state_solver.h"

#include "base_classes/active_space_method.h"

namespace forte {

ActiveSpaceMethod::ActiveSpaceMethod(StateInfo state, size_t nroot,
                                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : state_(state), nroot_(nroot), mo_space_info_(mo_space_info), as_ints_(as_ints) {
    active_mo_ = as_ints_->active_mo();
    core_mo_ = as_ints_->restricted_docc_mo();
}

void ActiveSpaceMethod::set_active_space_integrals(std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    as_ints_ = as_ints;
}

psi::SharedVector ActiveSpaceMethod::evals() { return evals_; }

const std::vector<double>& ActiveSpaceMethod::energies() const { return energies_; }

void ActiveSpaceMethod::set_e_convergence(double value) { e_convergence_ = value; }

void ActiveSpaceMethod::set_r_convergence(double value) { r_convergence_ = value; }

void ActiveSpaceMethod::set_root(int value) { root_ = value; }

void ActiveSpaceMethod::set_print(int level) { print_ = level; }

std::unique_ptr<ActiveSpaceMethod> make_active_space_method(
    const std::string& type, StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
    std::shared_ptr<ForteOptions> options) {

    std::unique_ptr<ActiveSpaceMethod> solver;
    if (type == "FCI") {
        solver = std::make_unique<FCISolver>(state, nroot, mo_space_info, as_ints);
    } else if (type == "ACI") {
        //        solver =
        //            std::make_unique<AdaptiveCI>(state, nroot, scf_info, options, mo_space_info,
        //            as_ints);
        solver = std::make_unique<ExcitedStateSolver>(
            state, nroot, mo_space_info, as_ints,
            std::make_unique<AdaptiveCI>(state, nroot, scf_info, options, mo_space_info, as_ints));
    } else if (type == "CAS") {
        solver = std::make_unique<FCI_MO>(state, nroot, scf_info, options, mo_space_info, as_ints);
    } else if (type == "ASCI") {
        solver = std::make_unique<ExcitedStateSolver>(
            state, nroot, mo_space_info, as_ints,
            std::make_unique<ASCI>(state, nroot, scf_info, options, mo_space_info, as_ints));
    } else if (type == "CASSCF") {
        solver = std::make_unique<CASSCF>(state, nroot, scf_info, options, mo_space_info, as_ints);
    } else if (type == "PCI") {
        solver = std::make_unique<ExcitedStateSolver>(
            state, nroot, mo_space_info, as_ints,
            std::make_unique<ProjectorCI>(state, nroot, scf_info, mo_space_info, as_ints));
    } else {
        throw psi::PSIEXCEPTION("make_active_space_method: type = " + type + " was not recognized");
    }
    // read options
    solver->set_options(options);
    return solver;
}

} // namespace forte
