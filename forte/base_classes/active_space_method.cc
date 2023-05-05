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
#include "casscf/casscf.h"
#include "sci/aci.h"
#include "sci/asci.h"
#include "sci/fci_mo.h"
#include "sci/detci.h"
#include "pci/pci.h"
#include "ci_ex_states/excited_state_solver.h"
#include "external/external_active_space_method.h"
#ifdef HAVE_CHEMPS2
#include "dmrg/dmrgsolver.h"
#endif

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

void ActiveSpaceMethod::set_dump_trdm(bool dump) { dump_trdm_ = dump; }

void ActiveSpaceMethod::set_wfn_filename(const std::string& name) { wfn_filename_ = name; }

void ActiveSpaceMethod::set_root(int value) { root_ = value; }

void ActiveSpaceMethod::set_print(int level) { print_ = level; }

void ActiveSpaceMethod::set_quiet_mode(bool quiet) { quiet_ = quiet; }

DeterminantHashVec ActiveSpaceMethod::get_PQ_space() { return final_wfn_; }

psi::SharedMatrix ActiveSpaceMethod::get_PQ_evecs() { return evecs_; }

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

std::vector<psi::SharedVector> ActiveSpaceMethod::compute_permanent_quadrupole(
    std::shared_ptr<ActiveMultipoleIntegrals> ampints,
    const std::vector<std::pair<size_t, size_t>>& root_list) {
    // print title
    auto multi_label = state_.multiplicity_label();
    auto multi_label_upper = upper_string(multi_label);
    auto ms_label = get_ms_string(state_.twice_ms());
    auto irrep_label = state_.irrep_label();

    std::string state_label = multi_label + " (Ms = " + ms_label + ") " + irrep_label;
    std::string prefix = ampints->qp_name().empty() ? "" : ampints->qp_name() + " ";
    print_h2(prefix + "Quadrupole Moments [e a0^2] (Nuclear + Electronic) for " + state_label);

    // nuclear contributions
    auto quadrupole_nuc = ampints->nuclear_quadrupole();

    // prepare RDMs
    auto rdms_vec = rdms(root_list, ampints->qp_many_body_level(), RDMsType::spin_free);

    // print table header
    psi::outfile->Printf("\n    %8s %14s %14s %14s %14s %14s %14s", "State", "QM_XX", "QM_XY",
                         "QM_XZ", "QM_YY", "QM_YZ", "QM_ZZ");
    std::string dash(98, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    // compute quadrupole
    std::vector<psi::SharedVector> out(root_list.size());
    for (size_t i = 0, size = root_list.size(); i < size; ++i) {
        const auto& [root1, root2] = root_list[i];
        if (root1 != root2)
            continue;
        std::string name = std::to_string(root1) + upper_string(irrep_label);

        auto quadrupole = ampints->compute_electronic_quadrupole(rdms_vec[i]);
        quadrupole->add(*quadrupole_nuc);
        out[i] = quadrupole;

        auto xx = quadrupole->get(0);
        auto xy = quadrupole->get(1);
        auto xz = quadrupole->get(2);
        auto yy = quadrupole->get(3);
        auto yz = quadrupole->get(4);
        auto zz = quadrupole->get(5);
        psi::outfile->Printf("\n    %8s%15.8f%15.8f%15.8f%15.8f%15.8f%15.8f", name.c_str(), xx, xy,
                             xz, yy, yz, zz);

        // push to Psi4 global environment
        push_to_psi4_env_globals(xx, multi_label_upper + " <" + name + "|QM_XX|" + name + ">");
        push_to_psi4_env_globals(xy, multi_label_upper + " <" + name + "|QM_XY|" + name + ">");
        push_to_psi4_env_globals(xz, multi_label_upper + " <" + name + "|QM_XZ|" + name + ">");
        push_to_psi4_env_globals(yy, multi_label_upper + " <" + name + "|QM_YY|" + name + ">");
        push_to_psi4_env_globals(yz, multi_label_upper + " <" + name + "|QM_YZ|" + name + ">");
        push_to_psi4_env_globals(zz, multi_label_upper + " <" + name + "|QM_ZZ|" + name + ">");
    }
    psi::outfile->Printf("\n    %s", dash.c_str());

    // print nuclear contribution
    psi::outfile->Printf("\n    %8s%15.8f%15.8f%15.8f%15.8f%15.8f%15.8f", "Nuclear",
                         quadrupole_nuc->get(0), quadrupole_nuc->get(1), quadrupole_nuc->get(2),
                         quadrupole_nuc->get(3), quadrupole_nuc->get(4), quadrupole_nuc->get(5));
    psi::outfile->Printf("\n    %s", dash.c_str());

    return out;
}

std::vector<psi::SharedVector>
ActiveSpaceMethod::compute_permanent_dipole(std::shared_ptr<ActiveMultipoleIntegrals> ampints,
                                            std::vector<std::pair<size_t, size_t>>& root_list) {
    // print title
    auto multi_label = state_.multiplicity_label();
    auto multi_label_upper = upper_string(multi_label);
    auto ms_label = get_ms_string(state_.twice_ms());
    auto irrep_label = state_.irrep_label();

    std::string state_label = multi_label + " (Ms = " + ms_label + ") " + irrep_label;
    std::string prefix = ampints->dp_name().empty() ? "" : ampints->dp_name() + " ";
    print_h2(prefix + "Dipole Moments [e a0] (Nuclear + Electronic) for " + state_label);

    // nuclear contributions
    auto dipole_nuc = ampints->nuclear_dipole();

    // prepare RDMs
    auto rdms_vec = rdms(root_list, ampints->dp_many_body_level(), RDMsType::spin_free);

    // print table header
    psi::outfile->Printf("\n    %8s %14s %14s %14s %14s", "State", "DM_X", "DM_Y", "DM_Z", "|DM|");
    std::string dash(68, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    // compute dipole
    std::vector<psi::SharedVector> dipoles_out(root_list.size());
    for (size_t i = 0, size = root_list.size(); i < size; ++i) {
        const auto& [root1, root2] = root_list[i];
        if (root1 != root2)
            continue;
        std::string name = std::to_string(root1) + upper_string(irrep_label);

        auto dipole = ampints->compute_electronic_dipole(rdms_vec[i]);
        dipole->add(*dipole_nuc);
        dipoles_out[i] = dipole;

        auto dx = dipole->get(0);
        auto dy = dipole->get(1);
        auto dz = dipole->get(2);
        auto dm = dipole->norm();
        psi::outfile->Printf("\n    %8s%15.8f%15.8f%15.8f%15.8f", name.c_str(), dx, dy, dz, dm);

        // push to Psi4 global environment
        push_to_psi4_env_globals(dx, multi_label_upper + " <" + name + "|DM_X|" + name + ">");
        push_to_psi4_env_globals(dy, multi_label_upper + " <" + name + "|DM_Y|" + name + ">");
        push_to_psi4_env_globals(dz, multi_label_upper + " <" + name + "|DM_Z|" + name + ">");
        push_to_psi4_env_globals(dm, multi_label_upper + " |<" + name + "|DM|" + name + ">|");
    }
    psi::outfile->Printf("\n    %s", dash.c_str());

    // print nuclear contribution
    psi::outfile->Printf("\n    %8s%15.8f%15.8f%15.8f%15.8f", "Nuclear", dipole_nuc->get(0),
                         dipole_nuc->get(1), dipole_nuc->get(2), dipole_nuc->norm());
    psi::outfile->Printf("\n    %s", dash.c_str());

    return dipoles_out;
}

std::vector<double> ActiveSpaceMethod::compute_oscillator_strength_same_orbs(
    std::shared_ptr<ActiveMultipoleIntegrals> ampints,
    const std::vector<std::pair<size_t, size_t>>& root_list,
    std::shared_ptr<ActiveSpaceMethod> method2) {

    // compute transition dipole moments
    auto trans_dipoles = compute_transition_dipole_same_orbs(ampints, root_list, method2);

    // print title
    std::string multi_label = upper_string(state_.multiplicity_label());
    const auto& state2 = method2->state();
    std::string title = state_.str_minimum() + " -> " + state2.str_minimum();
    print_h2("Transitions for " + title);
    psi::outfile->Printf("\n    %6s %6s %14s %14s %14s", "Init.", "Final", "Energy [a.u.]",
                         "Energy [eV]", "Osc. [a.u.]");
    std::string dash(58, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    // compute oscillator strength
    std::vector<double> out(root_list.size());
    auto energies2 = method2->energies();
    for (size_t i = 0, size = root_list.size(); i < size; ++i) {
        const auto& [root1, root2] = root_list[i];
        double e_diff = energies2[root2] - energies_[root1];
        double dm2 = trans_dipoles[i]->sum_of_squares();
        out[i] = 2.0 / 3.0 * std::fabs(e_diff) * dm2;

        // printing
        std::string name1 = std::to_string(root1) + upper_string(state_.irrep_label());
        std::string name2 = std::to_string(root2) + upper_string(state2.irrep_label());
        psi::outfile->Printf("\n    %6s %6s", name1.c_str(), name2.c_str());
        psi::outfile->Printf("%15.8f%15.8f%15.8f", e_diff, e_diff * pc_hartree2ev, out[i]);

        // push to psi4 environment globals
        std::string name_env = "OSC. " + multi_label + " " + name1 + " -> " + name2;
        push_to_psi4_env_globals(out[i], name_env);
    }
    psi::outfile->Printf("\n    %s", dash.c_str());

    return out;
}

std::vector<psi::SharedVector> ActiveSpaceMethod::compute_transition_dipole_same_orbs(
    std::shared_ptr<ActiveMultipoleIntegrals> ampints,
    const std::vector<std::pair<size_t, size_t>>& root_list,
    std::shared_ptr<ActiveSpaceMethod> method2) {
    // print title
    const auto& state2 = method2->state();
    std::string title = state_.str_minimum() + " -> " + state2.str_minimum();
    std::string prefix = ampints->dp_name().empty() ? "" : ampints->dp_name() + " ";
    print_h2(prefix + "Transition Dipole Moments [e a0] for " + title);

    // prepare transition RDMs
    auto rdm_level = ampints->dp_many_body_level();
    auto rdms = transition_rdms(root_list, method2, rdm_level, RDMsType::spin_free);

    // print table header
    psi::outfile->Printf("\n    %6s %6s %14s %14s %14s %14s", "Bra", "Ket", "DM_X", "DM_Y", "DM_Z",
                         "|DM|");
    std::string dash(73, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    // compute transition dipole
    std::vector<psi::SharedVector> trans_dipoles(root_list.size());
    for (size_t i = 0, size = root_list.size(); i < size; ++i) {
        auto td = ampints->compute_electronic_dipole(rdms[i], true);
        trans_dipoles[i] = td;

        // print
        const auto& [root1, root2] = root_list[i];
        std::string name1 = std::to_string(root1) + upper_string(state_.irrep_label());
        std::string name2 = std::to_string(root2) + upper_string(state2.irrep_label());
        psi::outfile->Printf("\n    %6s %6s", name1.c_str(), name2.c_str());

        auto dx = td->get(0);
        auto dy = td->get(1);
        auto dz = td->get(2);
        auto dm = td->norm();
        psi::outfile->Printf("%15.8f%15.8f%15.8f%15.8f", dx, dy, dz, dm);

        // push to psi4 environment globals
        std::string prefix = "TRANS " + upper_string(state_.multiplicity_label());
        push_to_psi4_env_globals(dx, prefix + " <" + name1 + "|DM_X|" + name2 + ">");
        push_to_psi4_env_globals(dy, prefix + " <" + name1 + "|DM_Y|" + name2 + ">");
        push_to_psi4_env_globals(dz, prefix + " <" + name1 + "|DM_Z|" + name2 + ">");
        push_to_psi4_env_globals(dm, prefix + " |<" + name1 + "|DM|" + name2 + ">|");
    }
    psi::outfile->Printf("\n    %s", dash.c_str());

    // Doing SVD for transition reduced density matrix
    auto nactv = mo_space_info_->size("ACTIVE");
    auto U = std::make_shared<psi::Matrix>("U", nactv, nactv);
    auto VT = std::make_shared<psi::Matrix>("VT", nactv, nactv);
    auto S = std::make_shared<psi::Vector>("S", nactv);

    print_h2("Transition Reduced Density Matrix Analysis for " + title);
    for (size_t i = 0, size = root_list.size(); i < size; ++i) {
        auto root1 = root_list[i].first;
        auto root2 = root_list[i].second;
        std::string name = std::to_string(root1) + " -> " + std::to_string(root2);
        auto Dt_matrix = rdms[i]->SF_G1mat();

        // Dump transition reduced matrices for each transition
        if (dump_trdm_) {
            // Change name that contains GAS info so it does not duplicate
            std::string gas_name1 = std::to_string(root1) + upper_string(state_.irrep_label()) +
                                    "_" + std::to_string(state_.gas_max()[0]);
            std::string gas_name2 = std::to_string(root2) + upper_string(state2.irrep_label()) +
                                    "_" + std::to_string(state2.gas_max()[0]);
            std::string dt_filename = "trdm_" + gas_name1 + "_" + gas_name2 + ".txt";
            Dt_matrix->save(dt_filename, false, false, true);
        }

        Dt_matrix->svd_a(U, S, VT);

        // Print the major components of transitions and major components to
        // the natural transition orbitals in active space
        std::string name1 = std::to_string(root1) + upper_string(state_.irrep_label());
        std::string name2 = std::to_string(root2) + upper_string(state2.irrep_label());
        psi::outfile->Printf("\n    Transition from State %4s  to State %4s :", name1.c_str(),
                             name2.c_str());
        double maxS = S->get(0);

        // push to Psi4 environment
        auto& globals = psi::Process::environment.globals;
        std::string prefix = "TRANS " + upper_string(state_.multiplicity_label());
        std::string key = " S_MAX " + name1 + " -> " + name2;
        push_to_psi4_env_globals(maxS, prefix + key);

        for (size_t comp = 0; comp < nactv; comp++) {
            double Svalue = S->get(comp);
            if (Svalue / maxS > 0.1) {
                // Print the components larger than 10% of the strongest occupation
                psi::outfile->Printf("\n      Component %2zu with value of W = %6.4f", comp + 1,
                                     Svalue);
                psi::outfile->Printf("\n        Init. Orbital:");
                for (size_t j = 0; j < nactv; j++) {
                    double coeff_i = U->get(j, comp);
                    if (coeff_i * coeff_i > 0.10) {
                        // Print the components with more than 0.1 amplitude from original orbitals
                        psi::outfile->Printf(" %6.4f Orb. %2zu", coeff_i * coeff_i, j);
                    }
                }
                psi::outfile->Printf("\n        Final Orbital:");
                for (size_t j = 0; j < nactv; j++) {
                    double coeff_f = VT->get(comp, j);
                    if (coeff_f * coeff_f > 0.10) {
                        // Print the components with more than 0.1 amplitude from final orbitals
                        psi::outfile->Printf(" %6.4f Orb. %2zu", coeff_f * coeff_f, j);
                    }
                }
            }
        }
        psi::outfile->Printf("\n");
    }

    return trans_dipoles;
}

std::shared_ptr<ActiveSpaceMethod> make_active_space_method(
    const std::string& type, StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
    std::shared_ptr<ForteOptions> options) {

    std::shared_ptr<ActiveSpaceMethod> method;
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
    } else if (type == "EXTERNAL") {
        method = std::make_unique<ExternalActiveSpaceMethod>(state, nroot, mo_space_info, as_ints);
    } else if (type == "DMRG") {
#ifdef HAVE_CHEMPS2
        method =
            std::make_unique<DMRGSolver>(state, nroot, scf_info, options, mo_space_info, as_ints);
#else
        throw std::runtime_error("DMRG is not available! Please compile with ENABLE_CHEMPS2=ON.");
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
