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

#include <algorithm>
#include <numeric>
#include <iomanip>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/process.h"

#include "base_classes/forte_options.h"
#include "base_classes/rdms.h"
#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "integrals/active_space_integrals.h"
#include "mrdsrg-helper/dsrg_transformed.h"
#include "active_space_method.h"

#include "active_space_solver.h"

namespace forte {

ActiveSpaceSolver::ActiveSpaceSolver(const std::string& method,
                                     const std::map<StateInfo, size_t>& state_nroots_map,
                                     std::shared_ptr<SCFInfo> scf_info,
                                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                     std::shared_ptr<ForteOptions> options)
    : method_(method), state_nroots_map_(state_nroots_map), scf_info_(scf_info),
      mo_space_info_(mo_space_info), as_ints_(as_ints), options_(options) {

    print_options();

    ms_avg_ = options->get_bool("SPIN_AVG_DENSITY");
}

const std::map<StateInfo, std::vector<double>>& ActiveSpaceSolver::compute_energy() {
    state_energies_map_.clear();
    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        size_t nroot = state_nroot.second;
        // compute the energy of state and save it
        std::shared_ptr<ActiveSpaceMethod> method = make_active_space_method(
            method_, state, nroot, scf_info_, mo_space_info_, as_ints_, options_);
        state_method_map_[state] = method;

        int twice_ms = state.twice_ms();
        if (twice_ms < 0 and ms_avg_) {
            psi::outfile->Printf("\n  Continue to the next symmetry block: No need to find the "
                                 "solution for ms = %d / 2 < 0.",
                                 twice_ms);
            continue;
        }

        method->compute_energy();

        const auto& energies = method->energies();
        state_energies_map_[state] = energies;

        if (twice_ms > 0 and ms_avg_) {
            StateInfo state_spin(state.nb(), state.na(), state.multiplicity(), -twice_ms,
                                 state.irrep(), state.irrep_label());
            state_energies_map_[state_spin] = energies;
        }
    }
    print_energies(state_energies_map_);
    return state_energies_map_;
}

void ActiveSpaceSolver::print_energies(std::map<StateInfo, std::vector<double>>& energies) {
    print_h2("Energy Summary");
    psi::outfile->Printf("\n    Multi.(2ms)  Irrep.  No.               Energy");
    std::string dash(45, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());
    std::vector<std::string> irrep_symbol = psi::Process::environment.molecule()->irrep_labels();

    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        int irrep = state.irrep();
        int multi = state.multiplicity();
        int nstates = state_nroot.second;
        int twice_ms = state.twice_ms();
        if (twice_ms < 0 and ms_avg_) {
            continue;
        }

        for (int i = 0; i < nstates; ++i) {
            auto label = "ENERGY ROOT " + std::to_string(i) + " " + std::to_string(multi) +
                         irrep_symbol[irrep];
            double energy = energies[state][i];
            psi::outfile->Printf("\n     %3d  (%3d)   %3s    %2d  %20.12f", multi, twice_ms,
                                 irrep_symbol[irrep].c_str(), i, energy);

            // make label case insensitive as required by Psi4 Python side
            std::transform(label.begin(), label.end(), label.begin(), ::toupper);
            psi::Process::environment.globals[label] = energy;
        }

        psi::outfile->Printf("\n    %s", dash.c_str());
    }
}

std::vector<RDMs> ActiveSpaceSolver::rdms(
    std::map<std::pair<StateInfo, StateInfo>, std::vector<std::pair<size_t, size_t>>>& elements,
    int max_rdm_level) {
    std::vector<RDMs> refs;

    for (const auto& element : elements) {
        const auto& state1 = element.first.first;
        const auto& state2 = element.first.second;

        if (state1 != state2) {
            throw std::runtime_error("ActiveSpaceSolver::reference called with states of different "
                                     "symmetry! This function is not yet suported in Forte.");
        }

        std::vector<RDMs> state_refs =
            state_method_map_[state1]->rdms(element.second, max_rdm_level);
        for (const auto& state_ref : state_refs) {
            refs.push_back(state_ref);
        }
    }
    return refs;
}

void ActiveSpaceSolver::print_options() {
    print_h2("Summary of Active Space Solver Input");

    std::vector<std::string> irrep_symbol = psi::Process::environment.molecule()->irrep_labels();
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

std::unique_ptr<ActiveSpaceSolver> make_active_space_solver(
    const std::string& method, const std::map<StateInfo, size_t>& state_nroots_map,
    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
    std::shared_ptr<ActiveSpaceIntegrals> as_ints, std::shared_ptr<ForteOptions> options) {
    return std::make_unique<ActiveSpaceSolver>(method, state_nroots_map, scf_info, mo_space_info,
                                               as_ints, options);
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
                       std::shared_ptr<psi::Wavefunction> wfn) {
    std::map<StateInfo, std::vector<double>> state_weights_map;
    auto state = make_state_info_from_psi_wfn(wfn); // assumes low-spin

    py::list avg_state = options->get_gen_list("AVG_STATE");

    if (avg_state.size() == 0) {
        int nroot = options->get_int("NROOT");
        int root = options->get_int("ROOT");

        std::vector<double> weights(nroot, 0.0);
        weights[root] = 1.0;
        state_weights_map[state] = weights;
    } else {
        double sum_of_weights = 0.0;
        size_t nstates = 0;
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
            std::string irrep_label = psi::Process::environment.molecule()->irrep_labels()[irrep];
            // multiplicity (2S + 1)
            int multi = py::cast<int>(avg_state_list[1]);
            // number of states with this irrep and multiplicity
            int nstates_this = py::cast<int>(avg_state_list[2]);

            // check for errors
            int nirrep = wfn->nirrep();
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
            if (avg_weight.size() == 0) {
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

            StateInfo state_this(state.na(), state.nb(), multi, state.twice_ms(), irrep,
                                 irrep_label);
            state_weights_map[state_this] = weights;
            nstates += nstates_this;
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

    // If not average over ms, directly return
    if (not options->get_bool("SPIN_AVG_DENSITY")) {
        if (options->get_int("PRINT") > 1) {
            print_state_weights_map(state_weights_map);
        }
        return state_weights_map;
    }

    // If we average over ms, then each multiplet will be considered as a "state".
    // The weight will be divided by its multiplicity.
    // For example, a triplet state will be treated as [1, 0, -1] each of weight 1/3.

    std::map<StateInfo, std::vector<double>> state_weights_map_ms_avg;

    for (const auto& state_weights : state_weights_map) {
        const auto& state = state_weights.first;
        const auto& weights = state_weights.second;

        auto multiplicity = state.multiplicity();
        auto irrep = state.irrep();
        auto irrep_label = state.irrep_label();
        auto nele = state.na() + state.nb();

        int max_twice_ms = multiplicity - 1;
        for (int i = max_twice_ms; i >= -max_twice_ms; i -= 2) {
            int na = (nele + i) / 2;
            StateInfo state_ms(na, nele - na, multiplicity, i, irrep, irrep_label);

            std::vector<double> weights_ms(weights);
            std::transform(weights_ms.begin(), weights_ms.end(), weights_ms.begin(),
                           [multiplicity](auto& w) { return w / multiplicity; });

            state_weights_map_ms_avg[state_ms] = weights_ms;
        }
    }

    if (options->get_int("PRINT") > 1) {
        print_state_weights_map(state_weights_map_ms_avg);
    }

    return state_weights_map_ms_avg;
}

RDMs ActiveSpaceSolver::compute_average_rdms(
    const std::map<StateInfo, std::vector<double>>& state_weights_map, int max_rdm_level) {

    if (ms_avg_) {
        return compute_avg_rdms_ms_avg(state_weights_map, max_rdm_level);
    }

    return compute_avg_rdms(state_weights_map, max_rdm_level);
}

RDMs ActiveSpaceSolver::compute_avg_rdms(
    const std::map<StateInfo, std::vector<double>>& state_weights_map, int max_rdm_level) {
    if (max_rdm_level <= 0) {
        return RDMs();
    }

    size_t na = mo_space_info_->size("ACTIVE");

    auto g1a = ambit::Tensor::build(ambit::CoreTensor, "g1a", {na, na});
    auto g1b = ambit::Tensor::build(ambit::CoreTensor, "g1b", {na, na});

    ambit::Tensor g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb;

    if (max_rdm_level >= 2) {
        g2aa = ambit::Tensor::build(ambit::CoreTensor, "g2aa", std::vector<size_t>(4, na));
        g2ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab", std::vector<size_t>(4, na));
        g2bb = ambit::Tensor::build(ambit::CoreTensor, "g2bb", std::vector<size_t>(4, na));
    }

    if (max_rdm_level >= 3) {
        g3aaa = ambit::Tensor::build(ambit::CoreTensor, "g3aaa", std::vector<size_t>(6, na));
        g3aab = ambit::Tensor::build(ambit::CoreTensor, "g3aab", std::vector<size_t>(6, na));
        g3abb = ambit::Tensor::build(ambit::CoreTensor, "g3abb", std::vector<size_t>(6, na));
        g3bbb = ambit::Tensor::build(ambit::CoreTensor, "g3bbb", std::vector<size_t>(6, na));
    }

    // Loop through references, add to master ref
    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        size_t nroot = state_nroot.second;
        const auto& weights = state_weights_map.at(state);

        // Get the already-run method
        const auto& method = state_method_map_.at(state);

        // Loop through roots in the method
        for (size_t r = 0; r < nroot; r++) {

            // Get the weight
            double weight = weights[r];

            // Don't bother if the weight is zero
            if (weight <= 1e-15)
                continue;

            // Get the RDMs
            std::vector<std::pair<size_t, size_t>> state_ids;
            state_ids.push_back(std::make_pair(r, r));
            RDMs method_rdms = method->rdms(state_ids, max_rdm_level)[0];

            // Average the RDMs
            g1a("pq") += weight * method_rdms.g1a()("pq");
            g1b("pq") += weight * method_rdms.g1b()("pq");

            if (max_rdm_level >= 2) {
                g2aa("pqrs") += weight * method_rdms.g2aa()("pqrs");
                g2ab("pqrs") += weight * method_rdms.g2ab()("pqrs");
                g2bb("pqrs") += weight * method_rdms.g2bb()("pqrs");
            }

            if (max_rdm_level >= 3) {
                g3aaa("pqrstu") += weight * method_rdms.g3aaa()("pqrstu");
                g3aab("pqrstu") += weight * method_rdms.g3aab()("pqrstu");
                g3abb("pqrstu") += weight * method_rdms.g3abb()("pqrstu");
                g3bbb("pqrstu") += weight * method_rdms.g3bbb()("pqrstu");
            }
        }
    }

    if (max_rdm_level == 1) {
        return RDMs(g1a, g1b);
    }

    if (max_rdm_level == 2) {
        return RDMs(g1a, g1b, g2aa, g2ab, g2bb);
    }

    return RDMs(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb);
}

RDMs ActiveSpaceSolver::compute_avg_rdms_ms_avg(
    const std::map<StateInfo, std::vector<double>>& state_weights_map, int max_rdm_level) {
    if (max_rdm_level <= 0) {
        return RDMs();
    }

    size_t na = mo_space_info_->size("ACTIVE");

    auto g1a = ambit::Tensor::build(ambit::CoreTensor, "g1a", {na, na});

    ambit::Tensor g2ab, g3aab;

    if (max_rdm_level >= 2) {
        g2ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab", std::vector<size_t>(4, na));
    }

    if (max_rdm_level >= 3) {
        g3aab = ambit::Tensor::build(ambit::CoreTensor, "g3aab", std::vector<size_t>(6, na));
    }

    // Loop through references, add to master ref
    for (const auto& state_nroot : state_nroots_map_) {
        const auto& state = state_nroot.first;
        size_t nroot = state_nroot.second;
        const auto& weights = state_weights_map.at(state);

        int twice_ms = state.twice_ms();
        if (twice_ms < 0) {
            continue;
        }

        // Get the already-run method
        const auto& method = state_method_map_.at(state);

        // Loop through roots in the method
        for (size_t r = 0; r < nroot; r++) {

            // Get the weight
            double weight = weights[r];

            // Don't bother if the weight is zero
            if (weight <= 1e-15)
                continue;

            // Get the RDMs
            std::vector<std::pair<size_t, size_t>> state_ids;
            state_ids.push_back(std::make_pair(r, r));
            RDMs method_rdms = method->rdms(state_ids, max_rdm_level)[0];

            // Average the RDMs
            g1a("pq") += weight * method_rdms.g1a()("pq");

            if (max_rdm_level >= 2) {
                g2ab("pqrs") += weight * method_rdms.g2ab()("pqrs");
            }

            if (max_rdm_level >= 3) {
                g3aab("pqrstu") += weight * method_rdms.g3aab()("pqrstu");
            }

            // add ms < 0 components
            if (twice_ms > 0) {
                g1a("pq") += weight * method_rdms.g1b()("pq");

                if (max_rdm_level >= 2) {
                    g2ab("pqrs") += weight * method_rdms.g2ab()("qpsr");
                }

                if (max_rdm_level >= 3) {
                    g3aab("pqrstu") += weight * method_rdms.g3abb()("rpqust");
                }
            }
        }
    }

    if (max_rdm_level == 1) {
        return RDMs(true, g1a);
    }

    if (max_rdm_level == 2) {
        return RDMs(true, g1a, g2ab);
    }

    return RDMs(true, g1a, g2ab, g3aab);
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
                RDMs rdms = method->rdms(root_list, max_rdm_level)[0];

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

    print_energies(state_energies_map_);
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

} // namespace forte
