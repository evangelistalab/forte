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
#include "active_space_method.h"

#include "active_space_solver.h"

namespace forte {

ActiveSpaceSolver::ActiveSpaceSolver(
    const std::string& method,
    std::vector<std::pair<StateInfo, std::vector<double>>>& state_weights_list,
    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
    std::shared_ptr<ActiveSpaceIntegrals> as_ints, std::shared_ptr<ForteOptions> options)
    : method_(method), state_weights_list_(state_weights_list), scf_info_(scf_info),
      mo_space_info_(mo_space_info), as_ints_(as_ints), options_(options) {
    print_options();
}

double ActiveSpaceSolver::compute_energy() {
    double energy = 0.0;
    std::vector<std::vector<double>> energies(state_weights_list_.size());

    size_t nstate = 0;
    size_t nmethod = 0;
    for (const auto& [state, weights] : state_weights_list_) {
        // compute the energy of state and save it
        size_t nroot = weights.size();
        std::shared_ptr<ActiveSpaceMethod> method = make_active_space_method2(
            method_, state, nroot, scf_info_, mo_space_info_, as_ints_, options_);

        method->set_options(options_);
        if (set_rdm_) {
            method->set_max_rdm_level(max_rdm_level_);
        }
        method->compute_energy();
        energies[nmethod] = method->energies();
        for (size_t r = 0; r < nroot; r++) {
            energy += energies[nmethod][r] * weights[r];
            nstate++;
        }
        method_vec_.push_back(method);
        nmethod++;
    }
    psi::outfile->Printf("\n  Average Energy from %d state(s): %17.15f", nstate, energy);
    print_energies(energies);
    return energies[0][0];
}

void ActiveSpaceSolver::print_energies(std::vector<std::vector<double>>& energies) {
    print_h2("Energy Summary");
    psi::outfile->Printf("\n    Multi.  Irrep.  No.               Energy");
    std::string dash(41, '-');
    psi::outfile->Printf("\n    %s", dash.c_str());
    std::vector<std::string> irrep_symbol = psi::Process::environment.molecule()->irrep_labels();

    int n = 0;
    for (const auto& [state, weights] : state_weights_list_) {
        int irrep = state.irrep();
        int multi = state.multiplicity();
        int nstates = weights.size();

        for (int i = 0; i < nstates; ++i) {
            psi::outfile->Printf("\n     %3d     %3s    %2d   %20.12f", multi,
                                 irrep_symbol[irrep].c_str(), i, energies[n][i]);
            psi::Process::environment.globals["ENERGY ROOT " + std::to_string(n + i)] =
                energies[n][i];
        }
        n++;
        psi::outfile->Printf("\n    %s", dash.c_str());
    }
}

Reference ActiveSpaceSolver::get_reference() {

    // For single state
    if ((state_weights_list_.size() == 1) and (state_weights_list_[0].second.size() == 1)) {
        std::vector<std::pair<size_t, size_t>> root;
        root.push_back(std::make_pair(0, 0));
        Reference ref = method_vec_[0]->get_reference(root)[0];
        return ref;
        // For state average
    } else {
        size_t nactive = mo_space_info_->size("ACTIVE");
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
            g2aa = ambit::Tensor::build(ambit::CoreTensor, "g2aa",
                                       {nactive, nactive, nactive, nactive});
            g2ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab",
                                       {nactive, nactive, nactive, nactive});
            g2bb = ambit::Tensor::build(ambit::CoreTensor, "g2bb",
                                        {nactive, nactive, nactive, nactive});
        }

        if (max_rdm_level_ >= 3) {
            g3aaa =
               ambit::Tensor::build(ambit::CoreTensor, "g3aaa", std::vector<size_t>(6, nactive));
            g3aab =
               ambit::Tensor::build(ambit::CoreTensor, "g3aab", std::vector<size_t>(6, nactive));
            g3abb =
               ambit::Tensor::build(ambit::CoreTensor, "g3abb", std::vector<size_t>(6, nactive));
            g3bbb =
                ambit::Tensor::build(ambit::CoreTensor, "g3bbb", std::vector<size_t>(6, nactive));
        }
        // function that scale pdm by w and add scaled pdm to sa_pdm
        auto scale_add = [](std::vector<double>& sa_pdm, std::vector<double>& pdm,
                            const double& w) {
            std::for_each(pdm.begin(), pdm.end(), [&](double& v) { v *= w; });
            std::transform(sa_pdm.begin(), sa_pdm.end(), pdm.begin(), sa_pdm.begin(),
                           std::plus<double>());
        };

        // Loop through references, add to master ref
        int state_num = 0;
        double energy = 0.0;
        for (const auto& [state, weights] : state_weights_list_) {
            size_t nroot = weights.size();
            // Get the already-run method
            auto& method = method_vec_[state_num];

            std::vector<std::pair<size_t, size_t>> root_list;
            for (size_t r = 0; r < nroot; r++) {
                root_list.push_back(std::make_pair(r, r));
            }
            std::vector<Reference> references = method->get_reference(root_list);

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
                scale_add(g1a.data(), method_ref.L1a().data(), weight);
                scale_add(g1b.data(), method_ref.L1b().data(), weight);

                if (max_rdm_level_ >= 2) {
                    // 2 RDM
                    scale_add(g2aa.data(), method_ref.L2aa().data(), weight);
                    scale_add(g2ab.data(), method_ref.L2ab().data(), weight);
                    scale_add(g2bb.data(), method_ref.L2bb().data(), weight);
                }

                if (max_rdm_level_ >= 3) {
                    // 3 RDM
                    scale_add(g3aaa.data(), method_ref.L3aaa().data(), weight);
                    scale_add(g3aab.data(), method_ref.L3aab().data(), weight);
                    scale_add(g3abb.data(), method_ref.L3abb().data(), weight);
                    scale_add(g3bbb.data(), method_ref.L3bbb().data(), weight);
                }
            }
            state_num++;
        }

        // set energy
  //      ref.set_Eref(energy);

        // compute cumulants
        // move this code to Reference?
 /*       // i.e., ref.update_cumulants()

        // 2-particle
        L2aa("pqrs") -= L1a("pr") * L1a("qs");
        L2aa("pqrs") += L1a("ps") * L1a("qr");

        L2bb("pqrs") -= L1b("pr") * L1b("qs");
        L2bb("pqrs") += L1b("ps") * L1b("qr");

        L2ab("pqrs") -= L1a("pr") * L1b("qs");

        // 3-particle
        L3aaa("pqrstu") -= L1a("ps") * L2aa("qrtu");
        L3aaa("pqrstu") += L1a("pt") * L2aa("qrsu");
        L3aaa("pqrstu") += L1a("pu") * L2aa("qrts");

        L3aaa("pqrstu") -= L1a("qt") * L2aa("prsu");
        L3aaa("pqrstu") += L1a("qs") * L2aa("prtu");
        L3aaa("pqrstu") += L1a("qu") * L2aa("prst");

        L3aaa("pqrstu") -= L1a("ru") * L2aa("pqst");
        L3aaa("pqrstu") += L1a("rs") * L2aa("pqut");
        L3aaa("pqrstu") += L1a("rt") * L2aa("pqsu");

        L3aaa("pqrstu") -= L1a("ps") * L1a("qt") * L1a("ru");
        L3aaa("pqrstu") -= L1a("pt") * L1a("qu") * L1a("rs");
        L3aaa("pqrstu") -= L1a("pu") * L1a("qs") * L1a("rt");

        L3aaa("pqrstu") += L1a("ps") * L1a("qu") * L1a("rt");
        L3aaa("pqrstu") += L1a("pu") * L1a("qt") * L1a("rs");
        L3aaa("pqrstu") += L1a("pt") * L1a("qs") * L1a("ru");

        // aab
        L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
        L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

        L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
        L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

        L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

        L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
        L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");

        // abb
        L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

        L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
        L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

        L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
        L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

        L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
        L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");

        // bbb
        L3bbb("pqrstu") -= L1b("ps") * L2bb("qrtu");
        L3bbb("pqrstu") += L1b("pt") * L2bb("qrsu");
        L3bbb("pqrstu") += L1b("pu") * L2bb("qrts");

        L3bbb("pqrstu") -= L1b("qt") * L2bb("prsu");
        L3bbb("pqrstu") += L1b("qs") * L2bb("prtu");
        L3bbb("pqrstu") += L1b("qu") * L2bb("prst");

        L3bbb("pqrstu") -= L1b("ru") * L2bb("pqst");
        L3bbb("pqrstu") += L1b("rs") * L2bb("pqut");
        L3bbb("pqrstu") += L1b("rt") * L2bb("pqsu");

        L3bbb("pqrstu") -= L1b("ps") * L1b("qt") * L1b("ru");
        L3bbb("pqrstu") -= L1b("pt") * L1b("qu") * L1b("rs");
        L3bbb("pqrstu") -= L1b("pu") * L1b("qs") * L1b("rt");

        L3bbb("pqrstu") += L1b("ps") * L1b("qu") * L1b("rt");
        L3bbb("pqrstu") += L1b("pu") * L1b("qt") * L1b("rs");
        L3bbb("pqrstu") += L1b("pt") * L1b("qs") * L1b("ru");
*/

        // Construct Reference object with RDMs
        Reference ref(g1a,g1b,g2aa,g2ab,g2bb,g3aaa,g3aab,g3abb,g3bbb);
        return ref;
    }
}

void ActiveSpaceSolver::set_max_rdm_level(size_t level) {
    max_rdm_level_ = level;
    set_rdm_ = true;
}

void ActiveSpaceSolver::print_options() {
    print_h2("Summary of Active Space Solver Input");

    //    std::vector<std::pair<std::string, size_t>> info;
    //    info.push_back({"No. a electrons in active", nalfa_ - ncore_ - nfrzc_});
    //    info.push_back({"No. b electrons in active", nbeta_ - ncore_ - nfrzc_});
    //    info.push_back({"multiplicity", multi_});
    //    info.push_back({"spin ms (2 * Sz)", twice_ms_});

    //    for (auto& str_dim : info) {
    //        outfile->Printf("\n    %-30s = %5zu", str_dim.first.c_str(), str_dim.second);
    //    }

    //    print_h2("Orbital Spaces");
    //    auto print_irrep = [&](const string& str, const psi::Dimension& array) {
    //        outfile->Printf("\n    %-30s", str.c_str());
    //        outfile->Printf("[");
    //        for (int h = 0; h < nirrep_; ++h) {
    //            outfile->Printf(" %4d ", array[h]);
    //        }
    //        outfile->Printf("]");
    //    };
    //    print_irrep("TOTAL MO", nmopi_);
    //    print_irrep("FROZEN CORE", frzc_dim_);
    //    print_irrep("FROZEN VIRTUAL", frzv_dim_);
    //    print_irrep("CORRELATED MO", ncmopi_);
    //    print_irrep("CORE", core_dim_);
    //    print_irrep("ACTIVE", actv_dim_);
    //    print_irrep("VIRTUAL", virt_dim_);

    print_h2("State Averaging Summary");

    psi::CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();

    std::vector<std::string> irrep_symbol = psi::Process::environment.molecule()->irrep_labels();
    int nroots_max = 0;
    int nstates = 0;
    for (const auto& [state, weights] : state_weights_list_) {
        int nroots = weights.size();
        nstates += nroots;
        nroots_max = std::max(nroots_max, nroots);
    }

    if (nroots_max == 1) {
        nroots_max = 7;
    } else {
        nroots_max *= 6;
        nroots_max -= 1;
    }

    int ltotal = 6 + 2 + 6 + 2 + 7 + 2 + nroots_max;
    std::string blank(nroots_max - 7, ' ');
    std::string dash(ltotal, '-');
    psi::outfile->Printf("\n    Irrep.  Multi.  Nstates  %sWeights", blank.c_str());
    psi::outfile->Printf("\n    %s", dash.c_str());
    for (const auto& [state, weights] : state_weights_list_) {
        //            int irrep, multi, nroots;
        //            std::vector<double> weights;
        //            std::tie(irrep, multi, nroots, weights) = sa_info_[i];
        int irrep = state.irrep();
        int multiplicity = state.multiplicity();
        int nroots = weights.size();
        std::string w_str;
        for (const double& w : weights) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << w;
            w_str += ss.str() + " ";
        }
        w_str.pop_back(); // delete the last space character

        std::stringstream ss;
        ss << std::setw(4) << std::right << irrep_symbol[irrep] << "    " << std::setw(4)
           << std::right << multiplicity << "    " << std::setw(5) << std::right << nroots << "    "
           << std::setw(nroots_max) << w_str;
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

std::vector<std::pair<StateInfo, std::vector<double>>>
make_state_weights_list(std::shared_ptr<ForteOptions> options,
                        std::shared_ptr<psi::Wavefunction> wfn) {
    std::vector<std::pair<StateInfo, std::vector<double>>> state_weight_list;
    auto state = make_state_info_from_psi_wfn(wfn);
    if ((options->psi_options())["AVG_STATE"].size() == 0) {

        int nroot = options->get_int("NROOT");
        int root = options->get_int("ROOT");

        std::vector<double> weights(nroot, 0.0);
        weights[root] = 1.0;
        state_weight_list.push_back(std::make_pair(state, weights));
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
            state_weight_list.push_back(std::make_pair(state, weights));
            nstates += nstates_this;
        }

        // normalize weights
        for (auto& [state, weights] : state_weight_list) {
            for (auto& w : weights) {
                w /= sum_of_weights;
            }
        }
    }
    return state_weight_list;
}

} // namespace forte
