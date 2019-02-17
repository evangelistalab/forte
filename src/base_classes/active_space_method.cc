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

#include "psi4/libmints/wavefunction.h"

#include "base_classes/mo_space_info.h"
#include "base_classes/forte_options.h"
#include "fci/fci_solver.h"
#include "casscf/casscf.h"
#include "sci/aci.h"
#include "sci/asci.h"
#include "sci/fci_mo.h"
#include "pci/pci.h"
#include "pci/pci_hashvec.h"
#include "pci/ewci.h"
#include "pci/pci_simple.h"

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

void ActiveSpaceMethod::set_root(int value) { root_ = value; }

void ActiveSpaceMethod::set_print(int level) { print_ = level; }

std::vector<std::string> ActiveSpaceMethod::generate_rdm_file_names(int rdm_level, int root1,
                                                                    int root2,
                                                                    const StateInfo& state2) {
    std::vector<std::string> out, spins;
    out.reserve(rdm_level + 1);

    if (rdm_level == 1) {
        spins = std::vector<std::string>{"a", "b"};
    } else if (rdm_level == 2) {
        spins = std::vector<std::string>{"aa", "ab", "bb"};
    } else if (rdm_level == 3) {
        spins = std::vector<std::string>{"aaa", "aab", "abb", "bbb"};
    } else {
        throw psi::PSIEXCEPTION("RDM level > 3 is not supported.");
    }

    std::string path0 = psi::PSIOManager::shared_object()->get_default_path() + "psi." +
                        std::to_string(getpid()) + "." +
                        psi::Process::environment.molecule()->name();
    std::string name0 = (root1 == root2 and state_ == state2) ? std::to_string(rdm_level) + "RDMs"
                                                              : std::to_string(rdm_level) + "TrDMs";

    for (const std::string& spin : spins) {
        std::string name = name0 + spin;
        std::string path = path0;
        std::vector<std::string> components = {name,
                                               std::to_string(root1),
                                               std::to_string(state_.multiplicity()),
                                               std::to_string(state_.irrep()),
                                               std::to_string(root2),
                                               std::to_string(state2.multiplicity()),
                                               std::to_string(state2.irrep()),
                                               "bin"};
        for (const std::string& str : components) {
            path += "." + str;
        }
        out.push_back(path);
    }

    return out;
}

bool ActiveSpaceMethod::check_density_files(int rdm_level, int root1, int root2,
                                            const StateInfo& state2) {
    auto filenames = generate_rdm_file_names(rdm_level, root1, root2, state2);

    // only one of these file names is needed because
    // they are always pushed/deleted together to/from density_files_
    return density_files_.find(filenames[0]) != density_files_.end();
}

void ActiveSpaceMethod::remove_density_files(int rdm_level, int root1, int root2,
                                             const StateInfo& state2) {
    auto fullnames = generate_rdm_file_names(rdm_level, root1, root2, state2);
    for (const std::string& filename : fullnames) {
        density_files_.erase(filename);
        if (remove(filename.c_str()) != 0) {
            std::stringstream ss;
            ss << "Error deleting file " << filename << ": No such file or directory";
            throw psi::PSIEXCEPTION(ss.str());
        }
    }

    std::string level = std::to_string(rdm_level);
    std::string name = (root1 == root2 and state_ == state2) ? level + "RDMs" : level + "TrDMs";
    psi::outfile->Printf("\n  Deleted files from disk for %s (%d %s %s - %d %s %s).", name.c_str(),
                         root1, state_.multiplicity_label().c_str(), state_.irrep_label().c_str(),
                         root2, state2.multiplicity_label().c_str(), state2.irrep_label().c_str());
}

std::unique_ptr<ActiveSpaceMethod> make_active_space_method(
    const std::string& type, StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
    std::shared_ptr<ForteOptions> options) {

    std::unique_ptr<ActiveSpaceMethod> solver;
    if (type == "FCI") {
        solver = std::make_unique<FCISolver>(state, nroot, mo_space_info, as_ints);
    } else if (type == "ACI") {
        solver =
            std::make_unique<AdaptiveCI>(state, nroot, scf_info, options, mo_space_info, as_ints);
    } else if (type == "CAS") {
        solver = std::make_unique<FCI_MO>(state, nroot, scf_info, options, mo_space_info, as_ints);
    } else if (type == "ASCI") {
        solver = std::make_unique<ASCI>(state, nroot, scf_info, options, mo_space_info, as_ints);
    } else if (type == "CASSCF") {
        solver = std::make_unique<CASSCF>(state, nroot, scf_info, options, mo_space_info, as_ints);
    } else if (type == "PCI") {
        // TODO modify pci code to compute multiple roots under new framework
        solver =
            std::make_unique<ProjectorCI>(state, nroot, scf_info, options, mo_space_info, as_ints);
    } else if (type == "PCI_SIMPLE") {
        // TODO modify pci code to compute multiple roots under new framework
        solver = std::make_unique<ProjectorCI_Simple>(state, nroot, scf_info, options,
                                                      mo_space_info, as_ints);
    } else if (type == "PCI_HASHVEC") {
        // TODO modify pci code to compute multiple roots under new framework
        solver = std::make_unique<ProjectorCI_HashVec>(state, nroot, scf_info, options,
                                                       mo_space_info, as_ints);
    } else if (type == "EWCI") {
        // TODO modify pci code to compute multiple roots under new framework
        solver = std::make_unique<ElementwiseCI>(state, nroot, scf_info, options, mo_space_info,
                                                 as_ints);
    } else {
        throw psi::PSIEXCEPTION("make_active_space_method: type = " + type + " was not recognized");
    }
    // read options
    solver->set_options(options);
    return solver;
}

} // namespace forte
