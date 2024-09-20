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

#include <algorithm>
#include <numeric>
#include <tuple>

#include "ambit/blocked_tensor.h"
#include "ambit/tensor.h"

#include "psi4/psi4-dec.h"
#include "psi4/physconst.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/process.h"

#include "base_classes/forte_options.h"
#include "base_classes/rdms.h"
#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "integrals/active_space_integrals.h"
#include "integrals/one_body_integrals.h"
#include "mrdsrg-helper/dsrg_transformed.h"
#include "active_space_method.h"

#include "active_space_solver.h"

namespace forte {

ActiveSpaceSolver::ActiveSpaceSolver(const std::string& method,
                                     const std::map<StateInfo, size_t>& state_nroots_map,
                                     std::shared_ptr<SCFInfo> scf_info,
                                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                                     std::shared_ptr<ForteOptions> options,
                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : method_(method), state_nroots_map_(state_nroots_map), scf_info_(scf_info),
      mo_space_info_(mo_space_info), options_(options), as_ints_(as_ints) {

    print_ = int_to_print_level(options->get_int("PRINT"));
    e_convergence_ = options->get_double("E_CONVERGENCE");
    r_convergence_ = options->get_double("R_CONVERGENCE");
    read_initial_guess_ = options->get_bool("READ_ACTIVE_WFN_GUESS");
    gas_diff_only_ = options->get_bool("PRINT_DIFFERENT_GAS_ONLY");

    if (options->get_str("ACTIVE_SPACE_SOLVER") == "BLOCK2")
        maxiter_ = options_->get_int("BLOCK2_N_TOTAL_SWEEPS");

    auto nactv = mo_space_info_->size("ACTIVE");
    Ua_actv_ = ambit::Tensor::build(ambit::CoreTensor, "Ua", {nactv, nactv});
    Ub_actv_ = ambit::Tensor::build(ambit::CoreTensor, "Ub", {nactv, nactv});
    auto& Ua_data = Ua_actv_.data();
    auto& Ub_data = Ub_actv_.data();
    for (size_t i = 0; i < nactv; ++i) {
        Ua_data[i * nactv + i] = 1.0;
        Ub_data[i * nactv + i] = 1.0;
    }
}

void ActiveSpaceSolver::set_print(PrintLevel level) { print_ = level; }

void ActiveSpaceSolver::set_e_convergence(double e_convergence) { e_convergence_ = e_convergence; }

void ActiveSpaceSolver::set_r_convergence(double r_convergence) { r_convergence_ = r_convergence; }

void ActiveSpaceSolver::set_maxiter(int maxiter) { maxiter_ = maxiter; }

void ActiveSpaceSolver::set_active_space_integrals(std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    as_ints_ = as_ints;
    for (const auto& [state, method] : state_method_map_) {
        method->set_active_space_integrals(as_ints_);
    }
}

void ActiveSpaceSolver::set_active_multipole_integrals(
    std::shared_ptr<ActiveMultipoleIntegrals> as_mp_ints) {
    as_mp_ints_ = as_mp_ints;
}

const std::map<StateInfo, std::vector<double>>& ActiveSpaceSolver::state_energies_map() const {
    return state_energies_map_;
}

const std::map<StateInfo, std::vector<double>>& ActiveSpaceSolver::compute_energy() {
    // check if the integrals are available
    if (not as_ints_) {
        throw std::runtime_error("ActiveSpaceSolver: ActiveSpaceIntegrals are not available.");
    }

    // initialize multipole integrals
    if (as_ints_->ints()->integral_type() != Custom) {
        if (not as_mp_ints_) {
            auto mp_ints = std::make_shared<MultipoleIntegrals>(as_ints_->ints(), mo_space_info_);
            as_mp_ints_ = std::make_shared<ActiveMultipoleIntegrals>(mp_ints);
        }
    }

    state_energies_map_.clear();
    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        size_t nroot = state_nroot.second;

        // so far only FCI and DETCI supports restarting from a previous wavefunction
        if ((method_ == "FCI") or (method_ == "DETCI")) {
            auto [it, inserted] = state_method_map_.try_emplace(state);
            if (inserted) {
                it->second = make_active_space_method(method_, state, nroot, scf_info_,
                                                      mo_space_info_, as_ints_, options_);
            }
            auto method = it->second;
        } else {
            state_method_map_[state] = make_active_space_method(method_, state, nroot, scf_info_,
                                                                mo_space_info_, as_ints_, options_);
        }
        auto method = state_method_map_[state];
        // set the convergence criteria
        method->set_print(print_);
        method->set_e_convergence(e_convergence_);
        method->set_r_convergence(r_convergence_);
        method->set_maxiter(maxiter_);

        state_filename_map_[state] = method->wfn_filename();
        if (read_initial_guess_) {
            method->set_read_wfn_guess(read_initial_guess_);
        }

        // compute the energy of state and save it
        method->compute_energy();
        const auto& energies = method->energies();
        state_energies_map_[state] = energies;
        const auto& spin2 = method->spin2();

        // check that the effective values of S are within a given tolerance
        validate_spin(spin2, state);
        state_spin2_map_[state] = spin2;
    }
    print_energies();

    if (as_ints_->ints()->integral_type() != Custom and
        options_->get_str("ACTIVE_SPACE_SOLVER") != "EXTERNAL") {
        compute_multipole_moment(as_mp_ints_, options_->get_int("MULTIPOLE_MOMENT_LEVEL"));
        py::list trans_list = options_->get_gen_list("TRANSITION_DIPOLES");
        if (trans_list.size())
            compute_fosc_same_orbs(as_mp_ints_);
    }

    return state_energies_map_;
}

void ActiveSpaceSolver::validate_spin(const std::vector<double>& spin2, const StateInfo& state) {
    if (spin2.size() != 0) {
        double S_tolerance = options_->get_double("S_TOLERANCE");
        double target_S = 0.5 * (static_cast<double>(state.multiplicity()) - 1.0);
        for (double root_spin2 : spin2) {
            double root_S = 0.5 * std::sqrt(1.0 + 4.0 * root_spin2) - 0.5;
            if (std::fabs(target_S - root_S) > S_tolerance) {
                std::string msg =
                    "ActiveSpaceSolver: Found a root with S = " + std::to_string(root_S) +
                    " but the target value of S = " + std::to_string(target_S);
                throw std::runtime_error(msg);
            }
        }
    }
}

void ActiveSpaceSolver::print_energies() {
    std::vector<std::string> irrep_symbol = mo_space_info_->irrep_labels();

    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        int irrep = state.irrep();
        int multi = state.multiplicity();
        int nstates = state_nroot.second;
        for (int i = 0; i < nstates; ++i) {
            double energy = state_energies_map_[state][i];
            auto label = "ENERGY ROOT " + std::to_string(i) + " " + std::to_string(multi) +
                         irrep_symbol[irrep];
            push_to_psi4_env_globals(energy, upper_string(label));
        }
    }

    if (print_ < PrintLevel::Brief)
        return;

    print_h2("Energy Summary");
    psi::outfile->Printf("\n    Multi.(2ms)  Irrep.  No.               Energy      <S^2>");
    std::string dash(56, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        int irrep = state.irrep();
        int multi = state.multiplicity();
        int nstates = state_nroot.second;
        int twice_ms = state.twice_ms();

        for (int i = 0; i < nstates; ++i) {
            double energy = state_energies_map_[state][i];
            if (state_spin2_map_[state].size() > 0) {
                double spin2_i = state_spin2_map_[state][i];
                psi::outfile->Printf("\n     %3d  (%3d)   %3s    %2d  %20.12f %10.6f", multi,
                                     twice_ms, irrep_symbol[irrep].c_str(), i, energy, spin2_i);
            } else {
                psi::outfile->Printf("\n     %3d  (%3d)   %3s    %2d  %20.12f       n/a", multi,
                                     twice_ms, irrep_symbol[irrep].c_str(), i, energy);
            }
        }
        psi::outfile->Printf("\n    %s", dash.c_str());
    }
}

void ActiveSpaceSolver::compute_multipole_moment(std::shared_ptr<ActiveMultipoleIntegrals> ampints,
                                                 int level) {
    int max_rdm_level = ampints->dp_many_body_level();
    std::string title = "dipole";
    if (level > 1 and max_rdm_level < ampints->qp_many_body_level()) {
        max_rdm_level = ampints->qp_many_body_level();
        title += "/quadrupole";
    }
    if (level < 1)
        return;

    // compute RDMs
    psi::outfile->Printf("\n  Computing RDMs for %s moments ...", title.c_str());
    std::map<StateInfo, std::vector<std::shared_ptr<RDMs>>> state_rdms_map;
    for (const auto& state_nroots : state_nroots_map_) {
        const auto& [state, nroots] = state_nroots;
        const auto& method = state_method_map_[state];

        std::vector<std::pair<size_t, size_t>> root_list;
        for (size_t i = 0; i < nroots; ++i)
            root_list.emplace_back(i, i);

        method->set_print(PrintLevel::Quiet);
        state_rdms_map[state] = method->rdms(root_list, max_rdm_level, RDMsType::spin_free);
        method->set_print(print_);
    }
    psi::outfile->Printf(" Done");

    // compute dipole moments
    auto dipole_nuc = ampints->nuclear_dipole();
    std::string prefix = ampints->dp_name().empty() ? "" : ampints->dp_name() + " ";
    print_h2("Summary of " + prefix + "Dipole Moments [e a0] (Nuclear + Electronic)");

    psi::outfile->Printf("\n    %8s %14s %14s %14s %14s", "State", "DM_X", "DM_Y", "DM_Z", "|DM|");
    std::string dash(68, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    for (const auto& state_nroot : state_nroots_map_) {
        const auto& [state, nroots] = state_nroot;
        auto irrep_label = state.irrep_label();
        auto multi_label = upper_string(state.multiplicity_label());

        for (size_t i = 0; i < nroots; ++i) {
            std::string name = std::to_string(i) + upper_string(irrep_label);
            auto dipole = ampints->compute_electronic_dipole(state_rdms_map[state][i]);
            dipole->add(*dipole_nuc);

            auto dx = dipole->get(0);
            auto dy = dipole->get(1);
            auto dz = dipole->get(2);
            auto dm = dipole->norm();
            psi::outfile->Printf("\n    %8s%15.8f%15.8f%15.8f%15.8f", name.c_str(), dx, dy, dz, dm);

            push_to_psi4_env_globals(dx, multi_label + " <" + name + "|DM_X|" + name + ">");
            push_to_psi4_env_globals(dy, multi_label + " <" + name + "|DM_Y|" + name + ">");
            push_to_psi4_env_globals(dz, multi_label + " <" + name + "|DM_Z|" + name + ">");
            push_to_psi4_env_globals(dm, multi_label + " |<" + name + "|DM|" + name + ">|");
        }
        psi::outfile->Printf("\n    %s", dash.c_str());
    }
    psi::outfile->Printf("\n    %8s%15.8f%15.8f%15.8f%15.8f", "Nuclear", dipole_nuc->get(0),
                         dipole_nuc->get(1), dipole_nuc->get(2), dipole_nuc->norm());
    psi::outfile->Printf("\n    %s", dash.c_str());

    // compute quadrupole moments
    if (level != 1) {
        auto quadrupole_nuc = ampints->nuclear_quadrupole();
        std::string prefix = ampints->qp_name().empty() ? "" : ampints->qp_name() + " ";
        print_h2("Summary of " + prefix + "Quadrupole Moments [e a0^2] (Nuclear + Electronic)");

        std::vector<std::string> qm_dirs{"QM_XX", "QM_XY", "QM_XZ", "QM_YY", "QM_YZ", "QM_ZZ"};
        psi::outfile->Printf("\n    %8s", "State");
        for (int z = 0; z < 6; ++z)
            psi::outfile->Printf(" %14s", qm_dirs[z].c_str());
        std::string dash(98, '-');
        psi::outfile->Printf("\n    %s", dash.c_str());

        for (const auto& state_nroot : state_nroots_map_) {
            const auto& [state, nroots] = state_nroot;
            auto irrep_label = state.irrep_label();
            auto multi_label = upper_string(state.multiplicity_label());

            for (size_t i = 0; i < nroots; ++i) {
                auto quadrupole = ampints->compute_electronic_quadrupole(state_rdms_map[state][i]);
                quadrupole->add(*quadrupole_nuc);

                std::string name = std::to_string(i) + upper_string(irrep_label);
                psi::outfile->Printf("\n    %8s", name.c_str());

                for (int z = 0; z < 6; ++z) {
                    auto value = quadrupole->get(z);
                    psi::outfile->Printf("%15.8f", value);
                    std::string dir = "|" + qm_dirs[z] + "|";
                    push_to_psi4_env_globals(value, multi_label + " <" + name + dir + name + ">");
                }
            }
            psi::outfile->Printf("\n    %s", dash.c_str());
        }
        psi::outfile->Printf("\n    %8s", "Nuclear");
        for (int z = 0; z < 6; ++z)
            psi::outfile->Printf("%15.8f", quadrupole_nuc->get(z));
        psi::outfile->Printf("\n    %s", dash.c_str());
    }
}

void ActiveSpaceSolver::compute_fosc_same_orbs(std::shared_ptr<ActiveMultipoleIntegrals> ampints) {
    // assume SAME set of orbitals!!!

    // parse the input states
    py::list trans_states = options_->get_gen_list("TRANSITION_DIPOLES");
    std::map<StateInfo, std::vector<size_t>> state_trans_map;
    for (size_t i = 0, size = trans_states.size(); i < size; ++i) {
        py::list integer_triplet = trans_states[i];
        if (integer_triplet.size() != 3) {
            psi::outfile->Printf("\n  Error: invalid input of TRANSITION_DIPOLES.");
            psi::outfile->Printf("\n  Each entry should take an array of three numbers.");
            throw std::runtime_error("Invalid input of TRANSITION_DIPOLES");
        }
        int irrep = py::cast<int>(integer_triplet[0]);
        int multi = py::cast<int>(integer_triplet[1]);
        size_t iroot = py::cast<size_t>(integer_triplet[2]);
        for (const auto& [state, nroot] : state_nroots_map_) {
            if (state.irrep() == irrep and state.multiplicity() == multi and iroot < nroot) {
                state_trans_map[state].push_back(iroot);
                break;
            }
        }
    }

    // generate root list
    std::map<std::pair<StateInfo, StateInfo>, std::vector<std::pair<size_t, size_t>>>
        root_lists_map;
    std::vector<StateInfo> _states;
    for (auto& [state1, roots1] : state_trans_map) {
        std::stable_sort(roots1.begin(), roots1.end());
        std::vector<std::pair<size_t, size_t>> state_ids;
        for (auto i : roots1) {
            for (size_t j = 0, jsize = state_nroots_map_[state1]; j < jsize; ++j) {
                if (i == j)
                    continue;
                state_ids.emplace_back(i, j);
            }
        }
        if (!state_ids.empty())
            root_lists_map[{state1, state1}] = state_ids;

        for (const auto& [state2, nroot2] : state_nroots_map_) {
            if (state1 == state2 or std::find(_states.begin(), _states.end(), state2) != _states.end())
                continue;
            // skip for different multiplicity (no spin-orbit coupling)
            if (state1.multiplicity() != state2.multiplicity())
                continue;
            // skip for the same GAS occupation (used for core-excited states)
            if (gas_diff_only_) {
                if (state1.gas_max() == state2.gas_max() && state1.gas_min() == state2.gas_min())
                    continue;
            }
            std::vector<std::pair<size_t, size_t>> state_ids;
            for (auto i : roots1) {
                for (size_t j = 0; j < nroot2; ++j) {
                    state_ids.emplace_back(i, j);
                }
            }
            if (!state_ids.empty())
                root_lists_map[{state1, state2}] = state_ids;
        }
        _states.push_back(state1);
    }

    // figure out ground state for a given multiplicity
    std::map<int, StateInfo> ground_states;
    std::vector<StateInfo> states;
    for (const auto& [state, _] : state_nroots_map_) {
        states.push_back(state);
        auto multi = state.multiplicity();
        if (ground_states.find(multi) == ground_states.end()) {
            ground_states[multi] = state;
        } else {
            const auto& state2 = ground_states.at(multi);
            ground_states[multi] =
                (state_energies_map_[state][0] < state_energies_map_[state2][0]) ? state : state2;
        }
    }

    // compute transition reduced density matrices
    auto rdm_level = ampints->dp_many_body_level();
    std::map<std::pair<StateInfo, StateInfo>, std::vector<std::shared_ptr<RDMs>>> rdms_map;
    for (const auto& [states, root_list] : root_lists_map) {
        const auto& [bra, ket] = states;
        const auto& method1 = state_method_map_[bra];
        const auto& method2 = state_method_map_[ket];
        rdms_map[{bra, ket}] =
            method1->transition_rdms(root_list, method2, rdm_level, RDMsType::spin_free);

        // analyze transition reduced density matrices
        auto nactv = mo_space_info_->size("ACTIVE");
        auto U = std::make_shared<psi::Matrix>("U", nactv, nactv);
        auto VT = std::make_shared<psi::Matrix>("VT", nactv, nactv);
        auto S = std::make_shared<psi::Vector>("S", nactv);

        std::string title = bra.str_minimum() + " -> " + ket.str_minimum();
        print_h2("Transition Reduced Density Matrix Analysis for " + title);
        for (size_t i = 0, size = root_list.size(); i < size; ++i) {
            auto root1 = root_list[i].first;
            auto root2 = root_list[i].second;
            std::string name = std::to_string(root1) + " -> " + std::to_string(root2);
            auto Dt_matrix = rdms_map[{bra, ket}][i]->SF_G1mat();

            // Dump transition reduced matrices for each transition
            if (options_->get_bool("DUMP_TRANSITION_RDM")) {
                // Change name that contains GAS info so it does not duplicate
                std::stringstream ss1, ss2;
                ss1 << root1 << upper_string(bra.irrep_label()) << "_" << bra.gas_max()[0];
                ss2 << root2 << upper_string(ket.irrep_label()) << "_" << ket.gas_max()[0];
                std::string dt_filename = "trdm_" + ss1.str() + "_" + ss2.str() + ".txt";
                Dt_matrix->save(dt_filename, false, false, true);
            }

            Dt_matrix->svd_a(U, S, VT);

            // Print the major components of transitions and major components to
            // the natural transition orbitals in active space
            std::string name1 = std::to_string(root1) + upper_string(bra.irrep_label());
            std::string name2 = std::to_string(root2) + upper_string(ket.irrep_label());
            psi::outfile->Printf("\n    Transition from State %4s to State %4s :", name1.c_str(),
                                 name2.c_str());
            double maxS = S->get(0);

            // push to Psi4 environment
            std::string prefix = "TRANS " + upper_string(bra.multiplicity_label());
            std::string key = " S_MAX " + name1 + " -> " + name2;
            push_to_psi4_env_globals(maxS, prefix + key);

            for (size_t comp = 0; comp < nactv; comp++) {
                double Svalue = S->get(comp);
                if (Svalue / maxS > 0.1) {
                    // Print the components larger than 10% of the strongest occupation
                    psi::outfile->Printf("\n      Component %2zu with value of W = %6.4f", comp + 1,
                                         Svalue);
                    psi::outfile->Printf("\n        Initial Orbital:");
                    for (size_t j = 0; j < nactv; j++) {
                        double coeff_i = U->get(j, comp);
                        if (coeff_i * coeff_i > 0.10) {
                            // Components with more than 0.1 amplitude from original orbitals
                            psi::outfile->Printf(" %6.4f Orb. %2zu", coeff_i * coeff_i, j);
                        }
                    }
                    psi::outfile->Printf("\n        Final Orbital:");
                    for (size_t j = 0; j < nactv; j++) {
                        double coeff_f = VT->get(comp, j);
                        if (coeff_f * coeff_f > 0.10) {
                            // Components with more than 0.1 amplitude from final orbitals
                            psi::outfile->Printf(" %6.4f Orb. %2zu", coeff_f * coeff_f, j);
                        }
                    }
                }
            }
            psi::outfile->Printf("\n");
        }
    }

    // compute transition dipole moments
    std::map<std::pair<StateInfo, StateInfo>, std::vector<std::shared_ptr<psi::Vector>>> tdp_map;
    std::string prefix = ampints->dp_name().empty() ? "" : ampints->dp_name() + " ";
    print_h2(prefix + "Transition Dipole Moments [e a0]");
    psi::outfile->Printf("\n    %10s %10s %14s %14s %14s %14s", "Bra", "Ket", "DM_X", "DM_Y",
                         "DM_Z", "|DM|");
    std::string dash(81, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());

    for (const auto& [states, root_list] : root_lists_map) {
        const auto& [state1, state2] = states;
        auto multi = state1.multiplicity();
        auto multi_label = state1.multiplicity_label();
        for (size_t i = 0, size = root_list.size(); i < size; ++i) {
            auto td = ampints->compute_electronic_dipole(rdms_map[states][i], true);
            tdp_map[{state1, state2}].push_back(td);

            auto& [root1, root2] = root_list[i];
            std::string name1 = std::to_string(multi) + upper_string(state1.irrep_label());
            std::string name2 = std::to_string(multi) + upper_string(state2.irrep_label());
            psi::outfile->Printf("\n    %5zu %4s %5zu %4s", root1, name1.c_str(), root2,
                                 name2.c_str());

            auto dx = td->get(0);
            auto dy = td->get(1);
            auto dz = td->get(2);
            auto dm = td->norm();
            psi::outfile->Printf("%15.8f%15.8f%15.8f%15.8f", dx, dy, dz, dm);

            std::string prefix = "TRANS " + upper_string(multi_label);
            name1 = std::to_string(root1) + upper_string(state1.irrep_label());
            name2 = std::to_string(root2) + upper_string(state2.irrep_label());
            push_to_psi4_env_globals(dx, prefix + " <" + name1 + "|DM_X|" + name2 + ">");
            push_to_psi4_env_globals(dy, prefix + " <" + name1 + "|DM_Y|" + name2 + ">");
            push_to_psi4_env_globals(dz, prefix + " <" + name1 + "|DM_Z|" + name2 + ">");
            push_to_psi4_env_globals(dm, prefix + " |<" + name1 + "|DM|" + name2 + ">|");
        }
        psi::outfile->Printf("\n    %s", dash.c_str());
    }

    // compute oscillator strengths
    print_h2("Summary for Vertical Transition Energy (in eV)");
    psi::outfile->Printf("\n    %10s %10s %14s %14s", "Init.", "Final", "Energy [eV]", "Osc. Str.");
    dash = std::string(51, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());
    for (const auto& [states, root_list] : root_lists_map) {
        const auto& [state1, state2] = states;
        const auto& energies1 = state_energies_map_[state1];
        const auto& energies2 = state_energies_map_[state2];
        auto multi = state1.multiplicity();
        auto multi_label = upper_string(state1.multiplicity_label());
        for (size_t i = 0, size = root_list.size(); i < size; ++i) {
            auto& [root1, root2] = root_list[i];
            double e_diff = energies2[root2] - energies1[root1];
            double dm2 = tdp_map[states][i]->sum_of_squares();
            auto osc = 2.0 / 3.0 * std::fabs(e_diff) * dm2;

            std::string name1 = std::to_string(multi) + upper_string(state1.irrep_label());
            std::string name2 = std::to_string(multi) + upper_string(state2.irrep_label());
            psi::outfile->Printf("\n    %5zu %4s %5zu %4s", root1, name1.c_str(), root2,
                                 name2.c_str());
            psi::outfile->Printf("%15.8f%15.8f", e_diff * pc_hartree2ev, osc);

            name1 = std::to_string(root1) + upper_string(state1.irrep_label());
            name2 = std::to_string(root2) + upper_string(state2.irrep_label());
            std::string name_env = "OSC. " + multi_label + " " + name1 + " -> " + name2;
            push_to_psi4_env_globals(osc, name_env);
        }
        psi::outfile->Printf("\n    %s", dash.c_str());
    }
}

std::vector<std::shared_ptr<RDMs>> ActiveSpaceSolver::rdms(
    std::map<std::pair<StateInfo, StateInfo>, std::vector<std::pair<size_t, size_t>>>& elements,
    int max_rdm_level, RDMsType rdm_type) {
    std::vector<std::shared_ptr<RDMs>> refs;

    for (const auto& element : elements) {
        const auto& state1 = element.first.first;
        const auto& state2 = element.first.second;

        if (state1 != state2) {
            throw std::runtime_error("ActiveSpaceSolver::reference called with states of different "
                                     "symmetry! This function is not yet supported in Forte.");
        }

        std::vector<std::shared_ptr<RDMs>> state_refs =
            state_method_map_[state1]->rdms(element.second, max_rdm_level, rdm_type);
        for (const auto& state_ref : state_refs) {
            refs.push_back(state_ref);
        }
    }
    return refs;
}

void ActiveSpaceSolver::generalized_rdms(const StateInfo& state, size_t root,
                                         const std::vector<double>& X, ambit::BlockedTensor& result,
                                         bool c_right, int rdm_level,
                                         std::vector<std::string> spin) {
    state_method_map_[state]->generalized_rdms(root, X, result, c_right, rdm_level, spin);
}

void ActiveSpaceSolver::add_sigma_kbody(const StateInfo& state, size_t root,
                                        ambit::BlockedTensor& h,
                                        const std::map<std::string, double>& block_label_to_factor,
                                        std::vector<double>& sigma) {
    state_method_map_[state]->add_sigma_kbody(root, h, block_label_to_factor, sigma);
}

void ActiveSpaceSolver::generalized_sigma(const StateInfo& state, std::shared_ptr<psi::Vector> x,
                                          std::shared_ptr<psi::Vector> sigma) {
    state_method_map_[state]->generalized_sigma(x, sigma);
}

void ActiveSpaceSolver::print_options() {
    print_h2("Summary of Active Space Solver Input");

    std::vector<std::string> irrep_symbol = mo_space_info_->irrep_labels();
    int nstates = 0;
    for (const auto& state_nroot : state_nroots_map_) {
        nstates += state_nroot.second;
    }

    int ltotal = 6 + 2 + 10 + 2 + 6;
    std::string dash(ltotal, '-');
    psi::outfile->Printf("\n    Irrep.  Multi.(2ms)      N");
    psi::outfile->Printf("\n    %s", dash.c_str());
    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        int irrep = state.irrep();
        int multiplicity = state.multiplicity();
        int twice_ms = state.twice_ms();
        int nroots = state_nroot.second;
        psi::outfile->Printf("\n    %5s   %4d  (%3d)    %3d", irrep_symbol[irrep].c_str(),
                             multiplicity, twice_ms, nroots);
    }
    psi::outfile->Printf("\n    %s", dash.c_str());
    psi::outfile->Printf("\n    N: number of roots");
    psi::outfile->Printf("\n    ms: spin z component");
    psi::outfile->Printf("\n    Total number of roots: %3d", nstates);
    psi::outfile->Printf("\n    %s\n", dash.c_str());
}

std::shared_ptr<ActiveSpaceSolver> make_active_space_solver(
    const std::string& method, const std::map<StateInfo, size_t>& state_nroots_map,
    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
    std::shared_ptr<ForteOptions> options, std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    return std::make_shared<ActiveSpaceSolver>(method, state_nroots_map, scf_info, mo_space_info,
                                               options, as_ints);
}

std::map<StateInfo, size_t>
to_state_nroots_map(const std::map<StateInfo, std::vector<double>>& state_weights_map) {
    std::map<StateInfo, size_t> state_nroots_map;
    for (const auto& state_vec : state_weights_map) {
        state_nroots_map[state_vec.first] = state_vec.second.size();
    }
    return state_nroots_map;
}

std::map<StateInfo, std::vector<double>>
make_state_weights_map(std::shared_ptr<ForteOptions> options,
                       std::shared_ptr<MOSpaceInfo> mo_space_info) {
    std::map<StateInfo, std::vector<double>> state_weights_map;

    // make a StateInfo object using the information from psi4
    // TODO: need to optimize for spin-free RDMs
    auto state = make_state_info_from_psi(options); // assumes low-spin

    // check if the user provided a AVG_STATE list
    py::list avg_state = options->get_gen_list("AVG_STATE");

    std::vector<size_t> gas_min(6, 0);
    std::vector<size_t> gas_max(6);
    for (int i = 0; i < 6; ++i) {
        gas_max[i] = 2 * mo_space_info->size("GAS" + std::to_string(i + 1));
    }

    // if AVG_STATE is not defined, do a state-specific computation
    if (avg_state.size() == 0) {
        // assign the weights (0,0,1_root,...) to do a state-specific computation
        int nroot = options->get_int("NROOT");
        int root = options->get_int("ROOT");
        std::vector<double> weights(nroot, 0.0);
        weights[root] = 1.0;
        for (int gasn = 0; gasn < 6; gasn++) {
            auto gas_space_min = options->get_int_list("GAS" + std::to_string(gasn + 1) + "MIN");
            auto gas_space_max = options->get_int_list("GAS" + std::to_string(gasn + 1) + "MAX");
            if (gas_space_min.size() > 0) {
                if (gas_space_min.size() > 1) {
                    std::string msg =
                        "\n  Error: GAS" + std::to_string(gasn + 1) + "MIN has an incorrect size";
                    psi::outfile->Printf(msg.c_str());
                    throw std::runtime_error(msg);
                }
                gas_min[gasn] = gas_space_min[0];
            }
            if (gas_space_max.size() > 0) {
                if (gas_space_max.size() > 1) {
                    std::string msg =
                        "\n  Error: GAS" + std::to_string(gasn + 1) + "MAX has an incorrect size";
                    psi::outfile->Printf(msg.c_str());
                    throw std::runtime_error(msg);
                }
                gas_max[gasn] = gas_space_max[0];
            }
        }
        StateInfo state_this(state.na(), state.nb(), state.multiplicity(), state.twice_ms(),
                             state.irrep(), state.irrep_label(), gas_min, gas_max);
        state_weights_map[state_this] = weights;
    } else {
        double sum_of_weights = 0.0;
        size_t nentry = avg_state.size();
        for (size_t i = 0; i < nentry; ++i) {
            py::list avg_state_list = avg_state[i];
            if (avg_state_list.size() != 3) {
                psi::outfile->Printf("\n  Error: invalid input of AVG_STATE.");
                psi::outfile->Printf("\n  Each entry should take an array of three numbers.");
                throw std::runtime_error("Invalid input of AVG_STATE");
            }

            // read data
            // irreducible representation
            int irrep = py::cast<int>(avg_state_list[0]);
            // irreducible representation label
            std::string irrep_label = mo_space_info->irrep_labels()[irrep];
            // multiplicity (2S + 1)
            int multi = py::cast<int>(avg_state_list[1]);
            // number of states with this irrep and multiplicity
            int nstates_this = py::cast<int>(avg_state_list[2]);

            // check for errors
            int nirrep = mo_space_info->nirrep();
            if (irrep >= nirrep || irrep < 0) {
                psi::outfile->Printf("\n  Error: invalid irrep in AVG_STATE.");
                psi::outfile->Printf(
                    "\n  Please check the input irrep (start from 0) not to exceed %d", nirrep - 1);
                throw std::runtime_error("Invalid irrep in AVG_STATE");
            }
            if (multi < 1) {
                psi::outfile->Printf("\n  Error: invalid multiplicity in AVG_STATE.");
                throw std::runtime_error("Invaid multiplicity in AVG_STATE");
            }
            if (nstates_this < 1) {
                psi::outfile->Printf("\n  Error: invalid \"number of states\" in AVG_STATE.");
                psi::outfile->Printf(
                    "\n  \"Number of states\" of a irrep and multiplicity must > 0.");
                throw std::runtime_error("Invalid nstates in AVG_STATE.");
            }

            std::vector<double> weights;
            py::list avg_weight = options->get_gen_list("AVG_WEIGHT");
            if (avg_weight.empty()) {
                // use equal weights
                weights = std::vector<double>(nstates_this, 1.0);
            } else {
                if (avg_weight.size() != nentry) {
                    psi::outfile->Printf("\n  Error: mismatched number of entries in AVG_STATE "
                                         "(%d) and AVG_WEIGHT (%d).",
                                         nentry, avg_weight.size());
                    throw std::runtime_error(
                        "Mismatched number of entries in AVG_STATE and AVG_WEIGHT.");
                }

                py::list avg_weight_list = avg_weight[i];
                int nweights = avg_weight_list.size();
                if (nweights != nstates_this) {
                    psi::outfile->Printf(
                        "\n  Error: mismatched number of weights in entry %d of AVG_WEIGHT.", i);
                    psi::outfile->Printf("\n  Asked for %d states but only %d weights.",
                                         nstates_this, nweights);
                    throw std::runtime_error("Mismatched number of weights in AVG_WEIGHT.");
                }
                for (int n = 0; n < nstates_this; ++n) {
                    double w = py::cast<double>(avg_weight_list[n]);
                    if (w < 0.0) {
                        psi::outfile->Printf("\n  Error: negative weights in AVG_WEIGHT.");
                        throw std::runtime_error("Negative weights in AVG_WEIGHT.");
                    }
                    weights.push_back(w);
                }
            }
            sum_of_weights += std::accumulate(std::begin(weights), std::end(weights), 0.0);

            for (int gasn = 0; gasn < 6; gasn++) {
                auto gas_space_min =
                    options->get_int_list("GAS" + std::to_string(gasn + 1) + "MIN");
                auto gas_space_max =
                    options->get_int_list("GAS" + std::to_string(gasn + 1) + "MAX");
                if (!gas_space_min.empty()) {
                    if (i >= gas_space_min.size()) {
                        std::string msg = "\n  Error: GAS" + std::to_string(gasn + 1) +
                                          "MIN has an incorrect size";
                        psi::outfile->Printf(msg.c_str());
                        throw std::runtime_error(msg);
                    }
                    gas_min[gasn] = gas_space_min[i];
                }
                if (!gas_space_max.empty()) {
                    if (i >= gas_space_max.size()) {
                        std::string msg = "\n  Error: GAS" + std::to_string(gasn + 1) +
                                          "MAX has an incorrect size";
                        psi::outfile->Printf(msg.c_str());
                        throw std::runtime_error(msg);
                    }
                    gas_max[gasn] = gas_space_max[i];
                }
            }

            StateInfo state_this(state.na(), state.nb(), multi, state.twice_ms(), irrep,
                                 irrep_label, gas_min, gas_max);
            state_weights_map[state_this] = weights;
        }

        // normalize weights
        for (auto& state_weights : state_weights_map) {
            auto& weights = state_weights.second;
            std::transform(weights.begin(), weights.end(), weights.begin(),
                           [sum_of_weights](auto& w) { return w / sum_of_weights; });
        }
    }

    // print function
    auto print_state_weights_map =
        [](const std::map<StateInfo, std::vector<double>>& state_weights_map) {
            for (const auto& state_weights : state_weights_map) {
                const auto& state = state_weights.first;
                const auto& weights = state_weights.second;
                psi::outfile->Printf("\n  State %s weights:", state.str().c_str());
                for (auto x : weights) {
                    psi::outfile->Printf("\n  %18.12f", x);
                }
            }
        };

    if (options->get_int("PRINT") > 1) {
        print_state_weights_map(state_weights_map);
    }
    return state_weights_map;
}

std::shared_ptr<RDMs> ActiveSpaceSolver::compute_average_rdms(
    const std::map<StateInfo, std::vector<double>>& state_weights_map, int max_rdm_level,
    RDMsType rdm_type) {
    auto rdms = RDMs::build(max_rdm_level, mo_space_info_->size("ACTIVE"), rdm_type);

    // Loop through references, add to master ref
    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        size_t nroot = state_nroot.second;
        const auto& weights = state_weights_map.at(state);

        // Get the already-run method
        const auto& method = state_method_map_.at(state);

        // Loop through roots in the method
        for (size_t r = 0; r < nroot; r++) {
            // Don't bother if the weight is zero
            if (weights[r] <= 1e-15)
                continue;

            // Get the RDMs
            std::vector<std::pair<size_t, size_t>> state_ids;
            state_ids.emplace_back(r, r);
            auto method_rdms = method->rdms(state_ids, max_rdm_level, rdm_type)[0];

            // Add contributions
            rdms->axpy(method_rdms, weights[r]);
        }
    }

    return rdms;
}

std::map<StateInfo, std::vector<double>>
ActiveSpaceSolver::compute_complementary_H2caa_overlap(ambit::Tensor Tbra, ambit::Tensor Tket,
                                                       const std::vector<int>& p_syms,
                                                       const std::string& name, bool load) {
    std::map<StateInfo, std::vector<double>> out;
    for (const auto& state_nroots : state_nroots_map_) {
        const auto& state = state_nroots.first;

        std::vector<size_t> roots(state_nroots.second);
        std::iota(roots.begin(), roots.end(), 0);

        const auto method = state_method_map_.at(state);
        out[state] =
            method->compute_complementary_H2caa_overlap(roots, Tbra, Tket, p_syms, name, load);
    }
    return out;
}

void ActiveSpaceSolver::dump_wave_function() {
    const auto& state_filenames = state_filename_map();
    for (const auto& state_filename : state_filenames) {
        const auto& state = state_filename.first;
        state_method_map_[state]->set_dump_wfn(true);
        state_method_map_[state]->dump_wave_function(state_filename.second);
    }
}

std::map<StateInfo, std::shared_ptr<psi::Matrix>> ActiveSpaceSolver::state_ci_wfn_map() const {
    std::map<StateInfo, std::shared_ptr<psi::Matrix>> out;
    for (const auto& pair : state_method_map_) {
        out[pair.first] = pair.second->ci_wave_functions();
    }
    return out;
}

const std::map<StateInfo, std::vector<double>>&
ActiveSpaceSolver::compute_contracted_energy(std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                             int max_rdm_level) {
    if (state_method_map_.size() == 0) {
        throw psi::PSIEXCEPTION("Old CI determinants are not solved. Call compute_energy first.");
    }

    state_energies_map_.clear();
    state_contracted_evecs_map_.clear();

    // prepare integrals
    size_t nactv = mo_space_info_->size("ACTIVE");
    auto init_fill_tensor = [nactv](std::string name, size_t dim, std::vector<double> data) {
        ambit::Tensor out =
            ambit::Tensor::build(ambit::CoreTensor, name, std::vector<size_t>(dim, nactv));
        out.data() = data;
        return out;
    };
    ambit::Tensor oei_a = init_fill_tensor("oei_a", 2, as_ints->oei_a_vector());
    ambit::Tensor oei_b = init_fill_tensor("oei_a", 2, as_ints->oei_b_vector());
    ambit::Tensor tei_aa = init_fill_tensor("tei_aa", 4, as_ints->tei_aa_vector());
    ambit::Tensor tei_ab = init_fill_tensor("tei_ab", 4, as_ints->tei_ab_vector());
    ambit::Tensor tei_bb = init_fill_tensor("tei_bb", 4, as_ints->tei_bb_vector());

    // TODO: check three-body integrals available or not
    //    bool do_three_body = (max_body_ == 3 and max_rdm_level_ == 3) ? true : false;

    // TODO: adapt DressedQuantity for spin-free RDMs
    DressedQuantity ints(0.0, oei_a, oei_b, tei_aa, tei_ab, tei_bb);

    for (const auto& state_nroots : state_nroots_map_) {
        const auto& state = state_nroots.first;
        size_t nroots = state_nroots.second;
        std::string state_name = state.multiplicity_label() + " " + state.irrep_label();
        auto method = state_method_map_.at(state);

        // form the Hermitian effective Hamiltonian
        print_h2("Building Effective Hamiltonian for " + state_name);
        psi::Matrix Heff("Heff " + state_name, nroots, nroots);

        for (size_t A = 0; A < nroots; ++A) {
            for (size_t B = A; B < nroots; ++B) {
                // just compute transition rdms of <A|sqop|B>
                std::vector<std::pair<size_t, size_t>> root_list{std::make_pair(A, B)};
                std::shared_ptr<RDMs> rdms =
                    method->rdms(root_list, max_rdm_level, RDMsType::spin_dependent)[0];

                double H_AB = ints.contract_with_rdms(rdms);
                if (A == B) {
                    H_AB += as_ints->nuclear_repulsion_energy() + as_ints->scalar_energy() +
                            as_ints->frozen_core_energy();
                    Heff.set(A, B, H_AB);
                } else {
                    Heff.set(A, B, H_AB);
                    Heff.set(B, A, H_AB);
                }
            }
        }

        print_h2("Effective Hamiltonian for " + state_name);
        psi::outfile->Printf("\n");
        Heff.print();
        psi::Matrix U("Eigen Vectors of Heff for " + state_name, nroots, nroots);
        psi::Vector E("Eigen Values of Heff for " + state_name, nroots);
        Heff.diagonalize(U, E);
        U.eivprint(E);

        std::vector<double> energies(nroots);
        for (size_t i = 0; i < nroots; ++i) {
            energies[i] = E.get(i);
        }
        state_energies_map_[state] = energies;
        state_contracted_evecs_map_[state] = std::make_shared<psi::Matrix>(U);
    }

    print_energies();
    return state_energies_map_;
}

double
compute_average_state_energy(const std::map<StateInfo, std::vector<double>>& state_energies_map,
                             const std::map<StateInfo, std::vector<double>>& state_weight_map) {
    double average_energy = 0.0;
    // loop over each state and compute the inner product of energies and weights
    for (const auto& state_weights : state_weight_map) {
        const auto& state = state_weights.first;
        const auto& weights = state_weights.second;
        const auto& energies = state_energies_map.at(state);
        average_energy +=
            std::inner_product(energies.begin(), energies.end(), weights.begin(), 0.0);
    }
    return average_energy;
}

std::map<StateInfo, size_t> ActiveSpaceSolver::state_space_size_map() const {
    std::map<StateInfo, size_t> out;
    for (const auto& state_method : state_method_map_) {
        const auto& state = state_method.first;
        const auto& method = state_method.second;
        out[state] = method->space_size();
    }
    return out;
}

std::vector<ambit::Tensor> ActiveSpaceSolver::eigenvectors(const StateInfo& state) const {
    return state_method_map_.at(state)->eigenvectors();
}

} // namespace forte
