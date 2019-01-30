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

#include <numeric>
#include <iomanip>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/process.h"

#include "base_classes/forte_options.h"
#include "base_classes/reference.h"
#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"
#include "integrals/active_space_integrals.h"
#include "mrdsrg-helper/dsrg_transformed.h"
#include "active_space_method.h"

#include "active_space_solver.h"

namespace forte {

ActiveSpaceSolver::ActiveSpaceSolver(const std::string& method,
                                     std::map<StateInfo, size_t>& state_map,
                                     std::shared_ptr<SCFInfo> scf_info,
                                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                     std::shared_ptr<ForteOptions> options)
    : method_(method), state_list_(state_map), scf_info_(scf_info), mo_space_info_(mo_space_info),
      as_ints_(as_ints), options_(options) {

    print_options();

    //    // determine the state-specific root number
    //    if (state_list.size() == 1) {
    //        const std::vector<double>& weights = state_list[0].second;
    //        auto it = std::find(weights.begin(), weights.end(), 1.0);
    //        if (it != weights.end()) {
    //            state_specific_root_ = std::distance(weights.begin(), it);
    //        }
    //    }
}

const std::map<StateInfo, std::vector<double>>& ActiveSpaceSolver::compute_energy() {
    double average_energy = 0.0;
    state_energies_map_.clear();
    size_t nstate = 0;
    for (const auto& state_nroot : state_list_) {
        const auto& state = state_nroot.first;
        size_t nroot = state_nroot.second;
        // compute the energy of state and save it
        std::shared_ptr<ActiveSpaceMethod> method = make_active_space_method(
            method_, state, nroot, scf_info_, mo_space_info_, as_ints_, options_);
        method_map_[state] = method;

        method->set_options(options_);
        if (set_rdm_) {
            method->set_max_rdm_level(max_rdm_level_);
        }
        method->compute_energy();
        const auto& energies = method->energies();
        state_energies_map_[state] = energies;
    }
    print_energies(state_energies_map_);
    return state_energies_map_;
}

void ActiveSpaceSolver::print_energies(std::map<StateInfo, std::vector<double>>& energies) {
    print_h2("Energy Summary");
    psi::outfile->Printf("\n    Multi.  Irrep.  No.               Energy");
    std::string dash(41, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());
    std::vector<std::string> irrep_symbol = psi::Process::environment.molecule()->irrep_labels();

    int n = 0;
    for (const auto& state_nroot : state_list_) {
        const auto& state = state_nroot.first;
        int irrep = state.irrep();
        int multi = state.multiplicity();
        int nstates = state_nroot.second;

        for (int i = 0; i < nstates; ++i) {
            auto label = "ENERGY ROOT " + std::to_string(i) + " " + std::to_string(multi) +
                         irrep_symbol[irrep];
            double energy = energies[state][i];
            psi::outfile->Printf("\n     %3d     %3s    %2d   %20.12f", multi,
                                 irrep_symbol[irrep].c_str(), i, energy);
            psi::Process::environment.globals[label] = energy;
            //            psi::outfile->Printf("\n %s = %f", label.c_str(),
            //                                 energy); // TODO remove this line once we are done
            //                                 (Francesco)
        }

        n++;
        psi::outfile->Printf("\n    %s", dash.c_str());
    }
}

std::vector<Reference> ActiveSpaceSolver::reference(
    std::map<std::pair<StateInfo, StateInfo>, std::vector<std::pair<size_t, size_t>>>& elements) {

    std::vector<Reference> refs;

    for (const auto& element : elements) {
        const auto& state1 = element.first.first;
        const auto& state2 = element.first.second;

        if (state1 != state2) {
            throw std::runtime_error("ActiveSpaceSolver::reference called with states of different "
                                     "symmetry! This function is not yet suported in Forte.");
        }

        std::vector<Reference> state_refs = method_map_[state1]->reference(element.second);
        for (const auto& state_ref : state_refs) {
            refs.push_back(state_ref);
        }
    }
}

void ActiveSpaceSolver::set_max_rdm_level(size_t level) {
    max_rdm_level_ = level;
    set_rdm_ = true;
}

void ActiveSpaceSolver::print_options() {
    print_h2("Summary of Active Space Solver Input");

    psi::CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();

    std::vector<std::string> irrep_symbol = psi::Process::environment.molecule()->irrep_labels();
    int nstates = 0;
    for (const auto& state_nroot : state_list_) {
        nstates += state_nroot.second;
    }

    int ltotal = 6 + 2 + 6 + 2 + 7 + 2;
    std::string dash(ltotal, '-');
    psi::outfile->Printf("\n    Irrep.  Multi.  Nstates");
    psi::outfile->Printf("\n    %s", dash.c_str());
    for (const auto& state_nroot : state_list_) {
        const auto& state = state_nroot.first;
        int irrep = state.irrep();
        int multiplicity = state.multiplicity();
        int nroots = state_nroot.second;

        std::stringstream ss;
        ss << std::setw(4) << std::right << irrep_symbol[irrep] << "    " << std::setw(4)
           << std::right << multiplicity << "    " << std::setw(5) << std::right << nroots;
        psi::outfile->Printf("\n    %s", ss.str().c_str());
    }
    psi::outfile->Printf("\n    %s", dash.c_str());
    psi::outfile->Printf("\n    Total number of states: %d", nstates);
    psi::outfile->Printf("\n    %s\n", dash.c_str());
}

std::unique_ptr<ActiveSpaceSolver> make_active_space_solver(
    const std::string& method,
    std::vector<std::pair<StateInfo, std::vector<double>>>& state_weights_list,
    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
    std::shared_ptr<ActiveSpaceIntegrals> as_ints, std::shared_ptr<ForteOptions> options) {
    return std::make_unique<ActiveSpaceSolver>(method, state_weights_list, scf_info, mo_space_info,
                                               as_ints, options);
}

double ActiveSpaceSolver::get_average_state_energy() const {
    return compute_average_state_energy(state_energies_map_, state_list_);
}

std::map<StateInfo, std::vector<double>>
make_state_weights_map(std::shared_ptr<ForteOptions> options,
                        std::shared_ptr<psi::Wavefunction> wfn) {
    std::map<StateInfo, std::vector<double>> state_weights_map;
    auto state = make_state_info_from_psi_wfn(wfn);
    if ((options->psi_options())["AVG_STATE"].size() == 0) {

        int nroot = options->get_int("NROOT");
        int root = options->get_int("ROOT");

        std::vector<double> weights(nroot, 0.0);
        weights[root] = 1.0;
        state_weights_map[state] = weights;
    } else {
        double sum_of_weights = 0.0;
        size_t nstates = 0;
        size_t nentry = (options->psi_options())["AVG_STATE"].size();
        for (size_t i = 0; i < nentry; ++i) {
            if ((options->psi_options())["AVG_STATE"][i].size() != 3) {
                psi::outfile->Printf("\n  Error: invalid input of AVG_STATE. Each "
                                     "entry should take an array of three numbers.");
                throw std::runtime_error("Invalid input of AVG_STATE");
            }

            // read data
            // irreducible representation
            int irrep = (options->psi_options())["AVG_STATE"][i][0].to_integer();
            // multiplicity (2S + 1)
            int multi = (options->psi_options())["AVG_STATE"][i][1].to_integer();
            // number of states with this irrep and multiplicity
            int nstates_this = (options->psi_options())["AVG_STATE"][i][2].to_integer();

            // check for errors
            int nirrep = wfn->nirrep();
            if (irrep >= nirrep || irrep < 0) {
                psi::outfile->Printf("\n  Error: invalid irrep in AVG_STATE. Please "
                                     "check the input irrep (start from 0) not to "
                                     "exceed %d",
                                     nirrep - 1);
                throw std::runtime_error("Invalid irrep in AVG_STATE");
            }
            if (multi < 1) {
                psi::outfile->Printf("\n  Error: invalid multiplicity in AVG_STATE.");
                throw std::runtime_error("Invaid multiplicity in AVG_STATE");
            }

            if (nstates_this < 1) {
                psi::outfile->Printf("\n  Error: invalid nstates in AVG_STATE. "
                                     "nstates of a certain irrep and multiplicity "
                                     "should greater than 0.");
                throw std::runtime_error("Invalid nstates in AVG_STATE.");
            }

            std::vector<double> weights;
            if ((options->psi_options())["AVG_WEIGHT"].has_changed()) {
                if ((options->psi_options())["AVG_WEIGHT"].size() != nentry) {
                    psi::outfile->Printf("\n  Error: mismatched number of entries in "
                                         "AVG_STATE (%d) and AVG_WEIGHT (%d).",
                                         nentry, (options->psi_options())["AVG_WEIGHT"].size());
                    throw std::runtime_error("Mismatched number of entries in AVG_STATE "
                                             "and AVG_WEIGHT.");
                }
                int nweights = (options->psi_options())["AVG_WEIGHT"][i].size();
                if (nweights != nstates_this) {
                    psi::outfile->Printf("\n  Error: mismatched number of weights "
                                         "in entry %d of AVG_WEIGHT. Asked for %d "
                                         "states but only %d weights.",
                                         i, nstates_this, nweights);
                    throw std::runtime_error("Mismatched number of weights in AVG_WEIGHT.");
                }
                for (int n = 0; n < nstates_this; ++n) {
                    double w = (options->psi_options())["AVG_WEIGHT"][i][n].to_double();
                    if (w < 0.0) {
                        psi::outfile->Printf("\n  Error: negative weights in AVG_WEIGHT.");
                        throw std::runtime_error("Negative weights in AVG_WEIGHT.");
                    }
                    weights.push_back(w);
                }
            } else {
                // use equal weights
                weights = std::vector<double>(nstates_this, 1.0);
            }
            sum_of_weights = std::accumulate(std::begin(weights), std::end(weights), 0.0);
            state_weights_map[state] = weights;
            nstates += nstates_this;
        }

        // normalize weights
        for (auto& state_weights : state_weights_map) {
            auto& weights = state_weights.second;
            std::transform(weights.begin(), weights.end(), weights.begin(),
                           [sum_of_weights](auto& w) { return w / sum_of_weights; });
        }
    }
    return state_weights_map;
}

double compute_average_state_energy(
    std::vector<std::pair<StateInfo, std::vector<double>>> state_energies_list,
    std::vector<std::pair<StateInfo, std::vector<double>>> state_weight_list) {
    std::vector<double> weights;
    std::vector<double> energies;
    // flatten state_energies_list and state_weight_list into
    for (const auto& state_weights : state_weight_list) {
        std::copy(state_weights.second.begin(), state_weights.second.end(),
                  std::back_inserter(weights));
    }
    for (const auto& state_energies : state_energies_list) {
        std::copy(state_energies.second.begin(), state_energies.second.end(),
                  std::back_inserter(energies));
    }
    return std::inner_product(energies.begin(), energies.end(), weights.begin(), 0.0);
}

Reference compute_average_reference(std::shared_ptr<ActiveSpaceSolver>,
                                    std::map<StateInfo, std::vector<double>> weights) {
    // For state average
    size_t nactive = mo_space_info_->size(
        "ACTIVE"); // TODO: grab this info from the ActiveSpaceSolver object (Francesco)

    ambit::Tensor g1a = ambit::Tensor::build(ambit::CoreTensor, "g1a", {nactive, nactive});
    ambit::Tensor g1b = ambit::Tensor::build(ambit::CoreTensor, "g1b", {nactive, nactive});

    ambit::Tensor g2aa;
    ambit::Tensor g2ab;
    ambit::Tensor g2bb;

    ambit::Tensor g3aaa;
    ambit::Tensor g3aab;
    ambit::Tensor g3abb;
    ambit::Tensor g3bbb;

    if (max_rdm_level_ >= 2) {
        g2aa =
            ambit::Tensor::build(ambit::CoreTensor, "g2aa", {nactive, nactive, nactive, nactive});
        g2ab =
            ambit::Tensor::build(ambit::CoreTensor, "g2ab", {nactive, nactive, nactive, nactive});
        g2bb =
            ambit::Tensor::build(ambit::CoreTensor, "g2bb", {nactive, nactive, nactive, nactive});
    }

    if (max_rdm_level_ >= 3) {
        g3aaa = ambit::Tensor::build(ambit::CoreTensor, "g3aaa", std::vector<size_t>(6, nactive));
        g3aab = ambit::Tensor::build(ambit::CoreTensor, "g3aab", std::vector<size_t>(6, nactive));
        g3abb = ambit::Tensor::build(ambit::CoreTensor, "g3abb", std::vector<size_t>(6, nactive));
        g3bbb = ambit::Tensor::build(ambit::CoreTensor, "g3bbb", std::vector<size_t>(6, nactive));
    }
    // function that scale pdm by w and add scaled pdm to sa_pdm
    auto scale_add = [](std::vector<double>& sa_pdm, std::vector<double>& pdm, const double& w) {
        std::for_each(pdm.begin(), pdm.end(), [&](double& v) { v *= w; });
        std::transform(sa_pdm.begin(), sa_pdm.end(), pdm.begin(), sa_pdm.begin(),
                       std::plus<double>());
    };

    // Loop through references, add to master ref
    int state_num = 0;
    double energy = 0.0;
    for (const auto& state_nroot : state_list_) {
        const auto& weights = state_weights.second;

        size_t nroot = weights.size();
        // Get the already-run method
        auto& method = method_vec_[state_num];

        std::vector<std::pair<size_t, size_t>> root_list;
        for (size_t r = 0; r < nroot; r++) {
            root_list.push_back(std::make_pair(r, r));
        }
        std::vector<Reference> references = method->reference(root_list);

        // Grab energies to set E in reference
        auto& energies = method->energies();

        for (size_t r = 0; r < nroot; r++) {
            double weight = weights[r];

            // Don't bother if the weight is zero
            if (weight <= 1e-15)
                continue;

            energy += energies[r] * weight;

            // Get the reference of the correct root
            Reference method_ref = references[r];

            // Now the RDMs
            // 1 RDM
            scale_add(g1a.data(), method_ref.g1a().data(), weight);
            scale_add(g1b.data(), method_ref.g1b().data(), weight);

            if (max_rdm_level_ >= 2) {
                // 2 RDM
                scale_add(g2aa.data(), method_ref.g2aa().data(), weight);
                scale_add(g2ab.data(), method_ref.g2ab().data(), weight);
                scale_add(g2bb.data(), method_ref.g2bb().data(), weight);
            }

            if (max_rdm_level_ >= 3) {
                // 3 RDM
                scale_add(g3aaa.data(), method_ref.g3aaa().data(), weight);
                scale_add(g3aab.data(), method_ref.g3aab().data(), weight);
                scale_add(g3abb.data(), method_ref.g3abb().data(), weight);
                scale_add(g3bbb.data(), method_ref.g3bbb().data(), weight);
            }
        }
        state_num++;
    }

    if (max_rdm_level_ >= 3) {
        return Reference(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb);
    }
    // Construct Reference object with RDMs
    Reference ref(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb);
    return ref;
} // namespace forte
} // namespace forte
