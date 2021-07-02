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

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/physconst.h"

#include "base_classes/active_space_method.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/forte_options.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "fci/fci_solver.h"
#include "casscf/casscf.h"
#include "sci/aci.h"
#include "sci/asci.h"
#include "sci/fci_mo.h"
#include "sci/detci.h"
#include "pci/pci.h"
#include "ci_ex_states/excited_state_solver.h"

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

const std::vector<double>& ActiveSpaceMethod::spin2() const { return spin2_; }

void ActiveSpaceMethod::set_e_convergence(double value) { e_convergence_ = value; }

void ActiveSpaceMethod::set_r_convergence(double value) { r_convergence_ = value; }

void ActiveSpaceMethod::set_read_wfn_guess(bool read) { read_wfn_guess_ = read; }

void ActiveSpaceMethod::set_dump_wfn(bool dump) { dump_wfn_ = dump; }

void ActiveSpaceMethod::set_wfn_filename(const std::string& name) { wfn_filename_ = name; }

void ActiveSpaceMethod::set_root(int value) { root_ = value; }

void ActiveSpaceMethod::set_print(int level) { print_ = level; }

void ActiveSpaceMethod::set_quite_mode(bool quiet) { quiet_ = quiet; }

std::vector<double> ActiveSpaceMethod::compute_oscillator_strength_same_orbs(
    const std::vector<std::pair<size_t, size_t>>& root_list,
    std::shared_ptr<ActiveSpaceMethod> method2) {

    // compute transition dipole moments
    auto trans_dipoles = compute_transition_dipole_same_orbs(root_list, method2);

    const auto& state2 = method2->state();
    std::string title = state_.str_minimum() + " -> " + state2.str_minimum();
    print_h2("Transitions for " + title);

    // compute oscillator strength
    std::vector<double> out;
    auto energies2 = method2->energies();

    psi::outfile->Printf("\n    %6s %6s %14s %14s %14s", "Init.", "Final", "Energy [a.u.]",
                         "Energy [eV]", "Osc. [a.u.]");
    std::string dash(58, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    auto& globals = psi::Process::environment.globals;

    for (size_t i = 0, size = root_list.size(); i < size; ++i) {
        auto root1 = root_list[i].first;
        auto root2 = root_list[i].second;
        double e_diff = energies2[root2] - energies_[root1];
        double osc = 2.0 / 3.0 * std::fabs(e_diff) * trans_dipoles[i][3] * trans_dipoles[i][3];
        out.push_back(osc);

        std::string multi_label = upper_string(state_.multiplicity_label());
        std::string name1 = std::to_string(root1) + upper_string(state_.irrep_label());
        std::string name2 = std::to_string(root2) + upper_string(state2.irrep_label());

        psi::outfile->Printf("\n    %6s %6s", name1.c_str(), name2.c_str());
        psi::outfile->Printf("%15.8f%15.8f%15.8f", e_diff, e_diff * pc_hartree2ev, osc);

        // push to Psi4 environment
        std::string name = "OSC. " + multi_label + " " + name1 + " -> " + name2;

        // try to fix states with different gas_min and gas_max
        if (globals.find(name) != globals.end()) {
            if (globals.find(name + " ENTRY 0") == globals.end()) {
                globals[name + " ENTRY 0"] = globals[name];
            }

            int n = 1;
            std::string suffix = " ENTRY 1";
            while (globals.find(name + suffix) != globals.end()) {
                suffix = " ENTRY " + std::to_string(++n);
            }

            globals[name + suffix] = osc;
        }

        globals[name] = osc;
    }
    psi::outfile->Printf("\n    %s", dash.c_str());

    return out;
}

std::vector<std::vector<double>> ActiveSpaceMethod::compute_transition_dipole_same_orbs(
    const std::vector<std::pair<size_t, size_t>>& root_list,
    std::shared_ptr<ActiveSpaceMethod> method2) {

    const auto& state2 = method2->state();
    std::string title = state_.str_minimum() + " -> " + state2.str_minimum();
    print_h2("Transition Dipole Moments [e a0] for " + title);

    auto ints = as_ints_->ints();
    auto nactv = mo_space_info_->size("ACTIVE");
    auto actv_in_mo = mo_space_info_->pos_in_space("ACTIVE", "ALL");

    // grab MO dipole moment integrals
    auto mo_dipole_ints = ints->mo_dipole_ints(true, true); // just take alpha spin

    std::vector<ambit::Tensor> dipole_ints(3);
    std::vector<std::string> dirs{"X", "Y", "Z"};
    for (int i = 0; i < 3; ++i) {
        auto dm_ints = mo_dipole_ints[i];
        auto dm = ambit::Tensor::build(ambit::CoreTensor, "Dipole " + dirs[i], {nactv, nactv});
        dm.iterate([&](const std::vector<size_t>& i, double& value) {
            value = dm_ints->get(actv_in_mo[i[0]], actv_in_mo[i[1]]);
        });
        dipole_ints[i] = dm;
    }

    // compute transition 1-RDMs
    auto rdms = transition_rdms(root_list, method2, 1);

    psi::outfile->Printf("\n    %6s %6s %14s %14s %14s %14s", "Bra", "Ket", "DM_X", "DM_Y", "DM_Z",
                         "|DM|");
    std::string dash(73, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    // compute transition dipole
    std::vector<std::vector<double>> dipoles_out;
    for (size_t i = 0, size = root_list.size(); i < size; ++i) {
        auto root1 = root_list[i].first;
        auto root2 = root_list[i].second;

        std::string name = std::to_string(root1) + " -> " + std::to_string(root2);
        auto Dt = rdms[i].g1a().clone();
        Dt("pq") += rdms[i].g1b()("pq");

        std::vector<double> dipole(4, 0.0);
        for (int z = 0; z < 3; ++z) {
            dipole[z] = Dt("pq") * dipole_ints[z]("pq");
            dipole[3] += dipole[z] * dipole[z];
        }
        dipole[3] = std::sqrt(dipole[3]);
        dipoles_out.push_back(dipole);

        // printing
        std::string name1 = std::to_string(root1) + upper_string(state_.irrep_label());
        std::string name2 = std::to_string(root2) + upper_string(state2.irrep_label());

        psi::outfile->Printf("\n    %6s %6s", name1.c_str(), name2.c_str());
        psi::outfile->Printf("%15.8f%15.8f%15.8f%15.8f", dipole[0], dipole[1], dipole[2],
                             dipole[3]);

        // push to Psi4 environment
        auto& globals = psi::Process::environment.globals;
        std::string prefix = "TRANS " + upper_string(state_.multiplicity_label());

        std::vector<std::string> keys{
            " <" + name1 + "|DM_X|" + name2 + ">", " <" + name1 + "|DM_Y|" + name2 + ">",
            " <" + name1 + "|DM_Z|" + name2 + ">", " |<" + name1 + "|DM|" + name2 + ">|"};

        // try to fix states with different gas_min and gas_max
        std::string label = prefix + keys[3];
        if (globals.find(label) != globals.end()) {
            if (globals.find(label + " ENTRY 0") == globals.end()) {
                std::string suffix = " ENTRY 0";
                for (int i = 0; i < 4; ++i) {
                    globals[prefix + keys[i] + suffix] = globals[prefix + keys[i]];
                }
            }

            int n = 1;
            std::string suffix = " ENTRY 1";
            while (globals.find(label + suffix) != globals.end()) {
                suffix = " ENTRY " + std::to_string(++n);
            }

            for (int i = 0; i < 4; ++i) {
                globals[prefix + keys[i] + suffix] = dipole[i];
            }
        }

        for (int i = 0; i < 4; ++i) {
            globals[prefix + keys[i]] = dipole[i];
        }
    }
    psi::outfile->Printf("\n    %s", dash.c_str());

    return dipoles_out;
}

std::unique_ptr<ActiveSpaceMethod> make_active_space_method(
    const std::string& type, StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
    std::shared_ptr<ForteOptions> options) {

    std::unique_ptr<ActiveSpaceMethod> method;
    if (type == "FCI") {
        method = std::make_unique<FCISolver>(state, nroot, mo_space_info, as_ints);
    } else if (type == "ACI") {
        method = std::make_unique<ExcitedStateSolver>(
            state, nroot, mo_space_info, as_ints,
            std::make_unique<AdaptiveCI>(state, nroot, scf_info, options, mo_space_info, as_ints));
    } else if (type == "CAS") {
        method = std::make_unique<FCI_MO>(state, nroot, scf_info, options, mo_space_info, as_ints);
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
    } else {
        throw psi::PSIEXCEPTION("make_active_space_method: type = " + type + " was not recognized");
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
