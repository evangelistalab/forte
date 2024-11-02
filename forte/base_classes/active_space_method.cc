/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <memory>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/physconst.h"

#include "base_classes/active_space_method.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/forte_options.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "integrals/one_body_integrals.h"
#include "fci/fci_solver.h"
#include "genci/genci_solver.h"
#include "sci/aci.h"
#include "sci/asci.h"
#include "sci/detci.h"
#include "pci/pci.h"
#include "ci_ex_states/excited_state_solver.h"
#include "external/external_active_space_method.h"
#ifdef HAVE_CHEMPS2
#include "dmrg/dmrgsolver.h"
#endif

#ifdef HAVE_BLOCK2
#include "dmrg/block2_dmrg_solver.h"
#endif

namespace forte {

ActiveSpaceMethod::ActiveSpaceMethod(StateInfo state, size_t nroot,
                                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : state_(state), nroot_(nroot), mo_space_info_(mo_space_info), as_ints_(as_ints) {
    active_mo_ = as_ints_->active_mo();
}

void ActiveSpaceMethod::set_active_space_integrals(std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    as_ints_ = as_ints;
}

std::shared_ptr<psi::Vector> ActiveSpaceMethod::evals() { return evals_; }

const std::vector<double>& ActiveSpaceMethod::energies() const { return energies_; }

const std::vector<double>& ActiveSpaceMethod::spin2() const { return spin2_; }

void ActiveSpaceMethod::set_e_convergence(double value) { e_convergence_ = value; }

void ActiveSpaceMethod::set_r_convergence(double value) { r_convergence_ = value; }

void ActiveSpaceMethod::set_maxiter(size_t value) { maxiter_ = value; }

void ActiveSpaceMethod::set_read_wfn_guess(bool read) { read_wfn_guess_ = read; }

void ActiveSpaceMethod::set_dump_wfn(bool dump) { dump_wfn_ = dump; }

void ActiveSpaceMethod::set_dump_trdm(bool dump) { dump_trdm_ = dump; }

void ActiveSpaceMethod::set_wfn_filename(const std::string& name) { wfn_filename_ = name; }

void ActiveSpaceMethod::set_root(int value) { root_ = value; }

void ActiveSpaceMethod::set_print(PrintLevel level) { print_ = level; }

void ActiveSpaceMethod::set_quiet_mode() { set_print(PrintLevel::Quiet); }

DeterminantHashVec ActiveSpaceMethod::get_PQ_space() { return final_wfn_; }

std::shared_ptr<psi::Matrix> ActiveSpaceMethod::get_PQ_evecs() { return evecs_; }

void ActiveSpaceMethod::save_transition_rdms(
    const std::vector<std::shared_ptr<RDMs>>& rdms,
    const std::vector<std::pair<size_t, size_t>>& root_list,
    std::shared_ptr<ActiveSpaceMethod> method2) {

    std::string multi_label = state_.multiplicity_label();

    auto irrep1 = state_.irrep_label();
    auto gasmax1 = std::to_string(state_.gas_max()[0]);

    const auto& state2 = method2->state();
    auto irrep2 = state2.irrep_label();
    auto gasmax2 = std::to_string(state2.gas_max()[0]);

    for (size_t i = 0, size = root_list.size(); i < size; ++i) {
        std::string name1 = std::to_string(root_list[i].first) + irrep1 + "_" + gasmax1;
        std::string name2 = std::to_string(root_list[i].second) + irrep2 + "_" + gasmax2;

        std::string prefix = join({"trdm", multi_label, name1, name2}, ".");

        rdms[i]->dump_to_disk(prefix);
        rdms[i]->save_SF_G1(prefix + ".txt");
    }
}

std::shared_ptr<ActiveSpaceMethod> make_active_space_method(
    const std::string& type, StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
    std::shared_ptr<ForteOptions> options) {

    std::shared_ptr<ActiveSpaceMethod> method;
    if (type == "FCI") {
        method = std::make_unique<FCISolver>(state, nroot, mo_space_info, as_ints);
    } else if (type == "GENCI") {
        method = std::make_unique<GenCISolver>(state, nroot, mo_space_info, as_ints);
    } else if (type == "ACI") {
        method = std::make_unique<ExcitedStateSolver>(
            state, nroot, mo_space_info, as_ints,
            std::make_unique<AdaptiveCI>(state, nroot, scf_info, options, mo_space_info, as_ints));
    } else if (type == "DETCI") {
        method = std::make_unique<DETCI>(state, nroot, scf_info, options, mo_space_info, as_ints);
    } else if (type == "ASCI") {
        method = std::make_unique<ExcitedStateSolver>(
            state, nroot, mo_space_info, as_ints,
            std::make_unique<ASCI>(state, nroot, scf_info, options, mo_space_info, as_ints));
    } else if (type == "PCI") {
        method = std::make_unique<ExcitedStateSolver>(
            state, nroot, mo_space_info, as_ints,
            std::make_unique<ProjectorCI>(state, nroot, scf_info, options, mo_space_info, as_ints));
    } else if (type == "EXTERNAL") {
        method = std::make_unique<ExternalActiveSpaceMethod>(state, nroot, mo_space_info, as_ints);
    } else if (type == "DMRG") {
#ifdef HAVE_CHEMPS2
        method =
            std::make_unique<DMRGSolver>(state, nroot, scf_info, options, mo_space_info, as_ints);
#else
        throw std::runtime_error("DMRG is not available! Please compile with ENABLE_CHEMPS2=ON.");
#endif
    } else if (type == "BLOCK2") {
#ifdef HAVE_BLOCK2
        method = std::make_unique<Block2DMRGSolver>(state, nroot, scf_info, options, mo_space_info,
                                                    as_ints);
#else
        throw std::runtime_error("BLOCK2 is not available! Please compile with ENABLE_block2=ON.");
#endif
    } else {
        std::string msg = "make_active_space_method: type = " + type + " was not recognized";
        if (type == "") {
            msg += "\nPlease specify the active space method via the appropriate option.";
        }
        throw psi::PSIEXCEPTION(msg);
    }

    // read options
    method->set_options(options);

    // set default file name if dump wave function to disk
    auto nactv = mo_space_info->size("ACTIVE");
    std::string prefix = "forte." + lower_string(type) + ".o" + std::to_string(nactv) + ".";
    std::string state_str = method->state().str_short();
    method->set_wfn_filename(prefix + state_str + ".txt");

    return method;
}
} // namespace forte
