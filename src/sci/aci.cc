/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/physconst.h"

#include "base_classes/forte_options.h"
#include "base_classes/scf_info.h"
#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "ci_rdm/ci_rdms.h"
#include "sparse_ci/ci_reference.h"

#include "mrpt2.h"
#include "aci.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

using namespace psi;

namespace forte {

bool pairComp(const std::pair<double, Determinant> E1, const std::pair<double, Determinant> E2) {
    return E1.first < E2.first;
}

AdaptiveCI::AdaptiveCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                       std::shared_ptr<ForteOptions> options,
                       std::shared_ptr<MOSpaceInfo> mo_space_info,
                       std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : SelectedCIMethod(state, nroot, scf_info, mo_space_info, as_ints), options_(options) {
    // select the sigma vector type
    sigma_vector_type_ = string_to_sigma_vector_type(options_->get_str("DIAG_ALGORITHM"));
    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");
    sigma_ = options_->get_double("SIGMA");
    nuclear_repulsion_energy_ = as_ints->ints()->nuclear_repulsion_energy();
    nroot_ = nroot;
    startup();
}

void AdaptiveCI::startup() {
    quiet_mode_ = options_->get_bool("ACI_QUIET_MODE");

    wavefunction_symmetry_ = state_.irrep();
    multiplicity_ = state_.multiplicity();

    nact_ = mo_space_info_->size("ACTIVE");
    nactpi_ = mo_space_info_->dimension("ACTIVE");

    // Include frozen_docc and restricted_docc
    frzcpi_ = mo_space_info_->dimension("INACTIVE_DOCC");
    nfrzc_ = mo_space_info_->size("INACTIVE_DOCC");

    nalpha_ = state_.na() - nfrzc_;
    nbeta_ = state_.nb() - nfrzc_;

    // "Correlated" includes restricted_docc
    ncmo_ = mo_space_info_->size("CORRELATED");

    nirrep_ = mo_space_info_->nirrep();

    twice_ms_ = state_.twice_ms();

    // Read options
    gamma_ = options_->get_double("GAMMA");
    screen_thresh_ = options_->get_double("ACI_PRESCREEN_THRESHOLD");
    add_aimed_degenerate_ = options_->get_bool("ACI_ADD_AIMED_DEGENERATE");
    project_out_spin_contaminants_ = options_->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS");
    spin_complete_ = options_->get_bool("ACI_ENFORCE_SPIN_COMPLETE");
    spin_complete_P_ = options_->get_bool("ACI_ENFORCE_SPIN_COMPLETE_P");

    max_cycle_ = options_->get_int("SCI_MAX_CYCLE");

    pre_iter_ = options_->get_int("ACI_PREITERATIONS");

    spin_tol_ = options_->get_double("ACI_SPIN_TOL");
    // set the initial S^@ guess as input multiplicity
    int S = (multiplicity_ - 1.0) / 2.0;
    int S2 = multiplicity_ - 1.0;
    for (int n = 0; n < nroot_; ++n) {
        root_spin_vec_.push_back(std::make_pair(S, S2));
    }

    // get options for algorithm
    pq_function_ = options_->get_str("ACI_PQ_FUNCTION");
    print_weights_ = options_->get_bool("ACI_PRINT_WEIGHTS");

    hole_ = 0;

    max_memory_ = options_->get_int("SIGMA_VECTOR_MAX_MEMORY");

    // Decide when to compute coupling lists
    build_lists_ = true;
    // The Dynamic algorithm does not need lists
    if (sigma_vector_type_ == SigmaVectorType::Dynamic) {
        build_lists_ = false;
    }
}

void AdaptiveCI::print_info() {

    print_method_banner({"Adaptive Configuration Interaction",
                         "written by Jeffrey B. Schriber and Francesco A. Evangelista"});
    outfile->Printf("\n  ==> Reference Information <==\n");
    outfile->Printf("\n  There are %d frozen orbitals.", nfrzc_);
    outfile->Printf("\n  There are %zu active orbitals.\n", nact_);

    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{{"Multiplicity", multiplicity_},
                                                              {"Symmetry", wavefunction_symmetry_},
                                                              {"Number of roots", nroot_},
                                                              {"Root used for properties", root_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Sigma (Eh)", sigma_},
        {"Gamma (Eh^(-1))", gamma_},
        {"Convergence threshold", options_->get_double("ACI_CONVERGENCE")}};
    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Ms", get_ms_string(twice_ms_)},
        {"Diagonalization algorithm", options_->get_str("DIAG_ALGORITHM")},
        {"Excited Algorithm", ex_alg_},
        //        {"Q Type", q_rel_ ? "Relative Energy" : "Absolute Energy"},
        //        {"PT2 Parameters", options_->get_bool("PERTURB_SELECT") ?
        //        "True" : "False"},
        {"Project out spin contaminants", project_out_spin_contaminants_ ? "True" : "False"},
        {"Enforce spin completeness of basis", spin_complete_ ? "True" : "False"},
        {"Enforce complete aimed selection", add_aimed_degenerate_ ? "True" : "False"}};

    // Print some information
    outfile->Printf("\n  ==> Calculation Information <==\n");
    outfile->Printf("\n  %s", std::string(65, '-').c_str());
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s %-5d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %8.2e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n  %s", std::string(65, '-').c_str());

    if (options_->get_bool("PRINT_1BODY_EVALS")) {
        outfile->Printf("\n  Reference orbital energies:");
        std::shared_ptr<Vector> epsilon_a = scf_info_->epsilon_a();

        auto actmo = mo_space_info_->absolute_mo("ACTIVE");

        for (int n = 0, maxn = actmo.size(); n < maxn; ++n) {
            outfile->Printf("\n   %da: %1.6f ", n, epsilon_a->get(actmo[n]));
        }
    }
}

void AdaptiveCI::set_method_variables(
    std::string ex_alg, size_t nroot_method, size_t root,
    const std::vector<std::vector<std::pair<Determinant, double>>>& old_roots) {
    ex_alg_ = ex_alg;
    nroot_ = nroot_method;
    root_ = root;
    ref_root_ = root;
    old_roots_ = old_roots;
}

void AdaptiveCI::find_q_space() {
    print_h2("Finding the Q space");

    local_timer build_space;

    // First get the F space, using one of many algorithms
    std::vector<std::pair<double, Determinant>> F_space;
    std::string screen_alg = options_->get_str("ACI_SCREEN_ALG");

    if ((nroot_ == 1) and (screen_alg == "AVERAGE")) {
        screen_alg = "SR";
    }

    if ((screen_alg == "CORE") and (root_ == 0)) {
        screen_alg = "SR";
    }

    if ((ex_alg_ == "AVERAGE") and (screen_alg != "CORE")) {
        screen_alg = "AVERAGE";
    }

    outfile->Printf("\n  Using %s screening algorithm", screen_alg.c_str());

    // Get the excited determiants
    double remainder = 0.0;
    if (screen_alg == "AVERAGE") {
        // multiroot
        get_excited_determinants_avg(nroot_, P_evecs_, P_evals_, P_space_, F_space);
    } else if (screen_alg == "SR") {
        // single-root optimized
        get_excited_determinants_sr(P_evecs_, P_evals_, P_space_, F_space);
        //    } else if ( (screen_alg == "RESTRICTED")){
        //        // restricted
        //        get_excited_determinants_restrict(nroot_, P_evecs_, P_evals_, P_space_, F_space);
    } else if (screen_alg == "CORE") {
        get_excited_determinants_core(P_evecs_, P_evals_, P_space_, F_space);
    } else if (screen_alg == "BATCH_HASH" or screen_alg == "BATCH_CORE") {
        // hash batch
        remainder = get_excited_determinants_batch(P_evecs_, P_evals_, P_space_, F_space);
    } else if (screen_alg == "BATCH_VEC") {
        // vec batch
        remainder = get_excited_determinants_batch_vecsort(P_evecs_, P_evals_, P_space_, F_space);
    } else {
        std::string except = screen_alg + " is not a valid screening algorithm";
        throw psi::PSIEXCEPTION(except);
    }

    // Add P_space determinants
    PQ_space_.swap(P_space_);

    std::sort(F_space.begin(), F_space.end(), pairComp);

    local_timer screen;
    double ept2 = 0.0 - remainder;
    double sum = remainder;
    size_t last_excluded = 0;
    for (size_t I = 0, max_I = F_space.size(); I < max_I; ++I) {
        double& energy = F_space[I].first;
        Determinant& det = F_space[I].second;
        if (sum + energy < sigma_) {
            sum += energy;
            ept2 -= energy;
            last_excluded = I;

        } else {
            PQ_space_.add(det);
        }
    }
    // Add missing determinants

    if (add_aimed_degenerate_) {
        size_t num_extra = 0;
        for (size_t I = 0, max_I = last_excluded; I < max_I; ++I) {
            size_t J = last_excluded - I;
            if (std::fabs(F_space[last_excluded + 1].first - F_space[J].first) < 1.0e-9) {
                PQ_space_.add(F_space[J].second);
                num_extra++;
            } else {
                break;
            }
        }
        if (num_extra > 0 and (!quiet_mode_)) {
            outfile->Printf("\n  Added %zu missing determinants in aimed selection.", num_extra);
        }
    }

    multistate_pt2_energy_correction_.resize(nroot_);
    multistate_pt2_energy_correction_[ref_root_] = ept2;

    if (screen_alg == "AVERAGE") {
        for (int n = 0; n < nroot_; ++n) {
            multistate_pt2_energy_correction_[n] = ept2;
        }
    }

    outfile->Printf("\n  Time spent building the model space: %1.6f", build_space.get());
    // Check if P+Q space is spin complete
    if (spin_complete_) {
        PQ_space_.make_spin_complete(nact_); // <- xsize
        if (!quiet_mode_)
            outfile->Printf("\n  Spin-complete dimension of the PQ space: %zu", PQ_space_.size());
    }

    if ((ex_alg_ == "ROOT_ORTHOGONALIZE") and (root_ > 0) and cycle_ >= pre_iter_) {
        sparse_solver_->set_root_project(true);
        add_bad_roots(PQ_space_);
        sparse_solver_->add_bad_states(bad_roots_);
    }
}

double AdaptiveCI::average_q_values(std::vector<double>& E2) {
    // f_E2 and f_C1 will store the selected function of the chosen q criteria
    // This functions should only be called when nroot_ > 1

    size_t nroot = E2.size();

    int nav = options_->get_int("ACI_N_AVERAGE");
    int off = options_->get_int("ACI_AVERAGE_OFFSET");
    if (nav == 0)
        nav = nroot;
    if ((off + nav) > nroot)
        off = nroot - nav; // throw psi::PSIEXCEPTION("\n  Your desired number of
                           // roots and the offset exceeds the maximum number of
                           // roots!");

    double f_E2 = 0.0;

    // Choose the function of the couplings for each root
    // If nroot = 1, choose the max

    if (pq_function_ == "MAX" or nroot == 1) {
        f_E2 = *std::max_element(E2.begin(), E2.end());
    } else if (pq_function_ == "AVERAGE") {
        double E2_average = 0.0;
        double dim_inv = 1.0 / nav;
        for (int n = 0; n < nav; ++n) {
            E2_average += E2[n + off] * dim_inv;
        }

        f_E2 = E2_average;
    }
    return f_E2;
}

bool AdaptiveCI::check_convergence(std::vector<std::vector<double>>& energy_history,
                                   psi::SharedVector evals) {
    int nroot = evals->dim();
    int ref = 0;

    if (ex_alg_ == "ROOT_ORTHOGONALIZE") {
        ref = ref_root_;
        nroot = 1;
    }

    if (energy_history.size() == 0) {
        std::vector<double> new_energies;
        for (int n = 0; n < nroot; ++n) {
            double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
            new_energies.push_back(state_n_energy);
        }
        energy_history.push_back(new_energies);
        return false;
    }

    double old_avg_energy = 0.0;
    double new_avg_energy = 0.0;

    std::vector<double> new_energies;
    std::vector<double> old_energies = energy_history[energy_history.size() - 1];
    for (int n = 0; n < nroot; ++n) {
        n += ref;
        double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
        new_energies.push_back(state_n_energy);
        new_avg_energy += state_n_energy;
        old_avg_energy += old_energies[n];
    }
    old_avg_energy /= static_cast<double>(nroot);
    new_avg_energy /= static_cast<double>(nroot);

    energy_history.push_back(new_energies);

    // Check for convergence
    return (std::fabs(new_avg_energy - old_avg_energy) < options_->get_double("ACI_CONVERGENCE"));
}

void AdaptiveCI::prune_q_space(DeterminantHashVec& PQ_space, DeterminantHashVec& P_space,
                               psi::SharedMatrix evecs, int nroot) {
    // Select the new reference space using the sorted CI coefficients
    P_space.clear();

    double tau_p = sigma_ * gamma_;

    int nav = options_->get_int("ACI_N_AVERAGE");
    int off = options_->get_int("ACI_AVERAGE_OFFSET");
    if (nav == 0)
        nav = nroot;

    //  if( options_->get_str("EXCITED_ALGORITHM") == "ROOT_COMBINE" and (nav ==
    //  1) and (nroot > 1)){
    //      off = ref_root_;
    //  }

    if ((off + nav) > nroot)
        off = nroot - nav; // throw psi::PSIEXCEPTION("\n  Your desired number of
                           // roots and the offset exceeds the maximum number of
                           // roots!");

    // Create a vector that stores the absolute value of the CI coefficients
    std::vector<std::pair<double, Determinant>> dm_det_list;
    // for (size_t I = 0, max = PQ_space.size(); I < max; ++I){
    const det_hashvec& detmap = PQ_space.wfn_hash();
    for (size_t i = 0, max_i = detmap.size(); i < max_i; ++i) {
        double criteria = 0.0;
        if ((nroot_ > 1) and (ex_alg_ == "AVERAGE" or cycle_ < pre_iter_)) {
            for (int n = 0; n < nav; ++n) {
                if (pq_function_ == "MAX") {
                    criteria = std::max(criteria, std::fabs(evecs->get(i, n)));
                } else if (pq_function_ == "AVERAGE") {
                    criteria += std::fabs(evecs->get(i, n + off));
                }
            }
            criteria /= static_cast<double>(nav);
        } else {
            criteria = std::fabs(evecs->get(i, ref_root_));
        }
        dm_det_list.push_back(std::make_pair(criteria, detmap[i]));
    }

    // Decide which determinants will go in pruned_space
    // Include all determinants such that
    // sum_I |C_I|^2 < tau_p, where the sum runs over all the excluded
    // determinants
    // Sort the CI coefficients in ascending order
    std::sort(dm_det_list.begin(), dm_det_list.end());

    double sum = 0.0;
    size_t last_excluded = 0;
    for (size_t I = 0, max_I = PQ_space.size(); I < max_I; ++I) {
        double dsum = std::pow(dm_det_list[I].first, 2.0);
        if (sum + dsum < tau_p) { // exclude small contributions that sum to
                                  // less than tau_p
            sum += dsum;
            last_excluded = I;
        } else {
            P_space.add(dm_det_list[I].second);
        }
    }

    // add missing determinants that have the same weight as the last one
    // included
    if (add_aimed_degenerate_) {
        size_t num_extra = 0;
        for (size_t I = 0, max_I = last_excluded; I < max_I; ++I) {
            size_t J = last_excluded - I;
            if (std::fabs(dm_det_list[last_excluded + 1].first - dm_det_list[J].first) < 1.0e-9) {
                P_space.add(dm_det_list[J].second);
                num_extra += 1;
            } else {
                break;
            }
        }
        if (num_extra > 0 and !quiet_mode_) {
            outfile->Printf("\n  Added %zu missing determinants in aimed selection.", num_extra);
        }
    }
}

bool AdaptiveCI::check_stuck(const std::vector<std::vector<double>>& energy_history,
                             psi::SharedVector evals) {
    bool stuck = false;
    int nroot = evals->dim();
    if (cycle_ < 4) {
        stuck = false;
    } else {
        std::vector<double> av_energies;
        for (size_t i = 0; i < cycle_; ++i) {
            double energy = 0.0;
            for (int n = 0; n < nroot; ++n) {
                energy += energy_history[i][n];
            }
            energy /= static_cast<double>(nroot);
            av_energies.push_back(energy);
        }

        if (std::fabs(av_energies[cycle_ - 1] - av_energies[cycle_ - 3]) <
            options_->get_double("ACI_CONVERGENCE")) { // and
            //			std::fabs( av_energies[cycle_-2] -
            // av_energies[cycle_ - 4]
            //)
            //< options_->get_double("ACI_CONVERGENCE") ){
            stuck = true;
        }
    }
    return stuck;
}

int AdaptiveCI::root_follow(DeterminantHashVec& P_ref, std::vector<double>& P_ref_evecs,
                            DeterminantHashVec& P_space, psi::SharedMatrix P_evecs,
                            int num_ref_roots) {
    int ndets = P_space.size();
    int max_dim = std::min(ndets, 1000);
    //    int max_dim = ndets;
    int new_root;
    double old_overlap = 0.0;
    DeterminantHashVec P_int;
    std::vector<double> P_int_evecs;

    // std::vector<std::pair<double, size_t>> det_weights;
    // detmap map = P_ref.wfn_hash();
    // for (detmap::iterator it = map.begin(), endit = map.end(); it != endit;
    //      ++it) {
    //     det_weights.push_back(
    //         std::make_pair(std::abs(P_ref_evecs[it->second]), it->second));
    // }
    // std::sort(det_weights.begin(), det_weights.end());
    // std::reverse(det_weights.begin(), det_weights.end() );

    // for (size_t I = 0; I < 10; ++I) {
    //     outfile->Printf("\n %1.8f   %s", det_weights[I].first,
    //     P_ref.get_det(det_weights[I].second).str().c_str());
    // }

    int max_overlap = std::min(int(P_space.size()), num_ref_roots);

    for (int n = 0; n < max_overlap; ++n) {
        if (!quiet_mode_)
            outfile->Printf("\n\n  Computing overlap for root %d", n);
        double new_overlap = P_ref.overlap(P_ref_evecs, P_space, P_evecs, n);

        new_overlap = std::fabs(new_overlap);
        if (!quiet_mode_) {
            outfile->Printf("\n  Root %d has overlap %f", n, new_overlap);
        }
        // If the overlap is larger, set it as the new root and reference, for
        // now
        if (new_overlap > old_overlap) {

            if (!quiet_mode_) {
                outfile->Printf("\n  Saving reference for root %d", n);
            }
            // Save most important subspace
            new_root = n;
            P_int.subspace(P_space, P_evecs, P_int_evecs, max_dim, n);
            old_overlap = new_overlap;
        }
    }

    // Update the reference P_ref

    P_ref.clear();
    P_ref = P_int;

    P_ref_evecs = P_int_evecs;

    outfile->Printf("\n  Setting reference root to: %d", new_root);

    return new_root;
}

void AdaptiveCI::pre_iter_preparation() {
    // Build the reference determinant and compute its energy
    CI_Reference ref(scf_info_, options_, mo_space_info_, as_ints_, multiplicity_, twice_ms_,
                     wavefunction_symmetry_);
    ref.build_reference(initial_reference_);
    P_space_ = initial_reference_;

    if ((options_->get_bool("SCI_CORE_EX")) and (root_ > 0)) {

        ref_root_ = root_ - 1;

        int ncstate = options_->get_int("ACI_ROOTS_PER_CORE");

        if (((root_) > ncstate) and (root_ > 1)) {
            hole_++;
        }
        int particle = (root_ - 1) - (hole_ * ncstate);

        P_space_.clear();
        Determinant det = initial_reference_[0];
        Determinant detb(det);
        std::vector<int> avir = det.get_alfa_vir(nact_); // TODO check this
        outfile->Printf("\n  %s", str(det, nact_).c_str());
        outfile->Printf("\n  Freezing alpha orbital %d", hole_);
        outfile->Printf("\n  Exciting electron from %d to %d", hole_, avir[particle]);
        det.set_alfa_bit(hole_, false);
        detb.set_beta_bit(hole_, false);

        for (int n = 0, max_n = avir.size(); n < max_n; ++n) {
            if ((mo_symmetry_[hole_] ^ mo_symmetry_[avir[n]]) == 0) {
                det.set_alfa_bit(avir[particle], true);
                detb.set_beta_bit(avir[particle], true);
                break;
            }
        }
        outfile->Printf("\n  %s", str(det, nact_).c_str());
        outfile->Printf("\n  %s", str(detb, nact_).c_str());
        P_space_.add(det);
        P_space_.add(detb);
    }

    if (quiet_mode_) {
        sparse_solver_->set_print_details(false);
    }
    sparse_solver_->set_parallel(true);
    sparse_solver_->set_force_diag(options_->get_bool("FORCE_DIAG_METHOD"));
    sparse_solver_->set_e_convergence(options_->get_double("E_CONVERGENCE"));
    sparse_solver_->set_r_convergence(options_->get_double("R_CONVERGENCE"));
    sparse_solver_->set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver_->set_spin_project(project_out_spin_contaminants_);
    sparse_solver_->set_guess_dimension(options_->get_int("DL_GUESS_SIZE"));
    sparse_solver_->set_num_vecs(options_->get_int("N_GUESS_VEC"));
    sparse_solver_->set_spin_project_full(false);
}

void AdaptiveCI::diagonalize_P_space() {
    cycle_time_.reset();
    // Step 1. Diagonalize the Hamiltonian in the P space
    num_ref_roots_ = std::min(nroot_, int(P_space_.size()));
    std::string cycle_h = "Cycle " + std::to_string(cycle_);

    follow_ = false;
    if (ex_alg_ == "ROOT_COMBINE" or ex_alg_ == "MULTISTATE" or ex_alg_ == "ROOT_ORTHOGONALIZE") {
        follow_ = true;
    }

    if (!quiet_mode_) {
        print_h1(cycle_h);
        print_h2("Diagonalizing the Hamiltonian in the P space");
        outfile->Printf("\n  Initial P space dimension: %zu", P_space_.size());
    }

    // Check that the initial space is spin-complete
    if (spin_complete_ or spin_complete_P_) {
        // assumes P_space handles determinants with only active space orbitals
        P_space_.make_spin_complete(nact_);
        if (!quiet_mode_)
            outfile->Printf("\n  %s: %zu determinants", "Spin-complete dimension of the P space",
                            P_space_.size());
    } else if (!quiet_mode_) {
        outfile->Printf("\n  Not checking for spin-completeness.");
    }
    // Diagonalize H in the P space
    if (ex_alg_ == "ROOT_ORTHOGONALIZE" and root_ > 0 and cycle_ >= pre_iter_) {
        sparse_solver_->set_root_project(true);
        add_bad_roots(P_space_);
        sparse_solver_->add_bad_states(bad_roots_);
    }

    sparse_solver_->manual_guess(false);
    local_timer diag;

    auto sigma_vector = make_sigma_vector(P_space_, as_ints_, max_memory_, sigma_vector_type_);
    std::tie(P_evals_, P_evecs_) = sparse_solver_->diagonalize_hamiltonian(
        P_space_, sigma_vector, num_ref_roots_, multiplicity_);
    auto spin = sparse_solver_->spin();

    if (!quiet_mode_)
        outfile->Printf("\n  Time spent diagonalizing H:   %1.6f s", diag.get());

    // Save ground state energy
    P_energies_.push_back(P_evals_->get(0));

    if ((cycle_ > 1) and options_->get_bool("ACI_APPROXIMATE_RDM")) {
        double diff = std::abs(P_energies_[cycle_] - P_energies_[cycle_ - 1]);
        if (diff <= 1e-5) {
            approx_rdm_ = true;
        }
    }

    // Update the reference root if root following
    if (follow_ and num_ref_roots_ > 1 and (cycle_ >= pre_iter_) and cycle_ > 0) {
        ref_root_ = root_follow(P_ref_, P_ref_evecs_, P_space_, P_evecs_, num_ref_roots_);
    }

    // Print the energy
    if (!quiet_mode_) {
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots_; ++i) {
            double abs_energy =
                P_evals_->get(i) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
            double exc_energy = pc_hartree2ev * (P_evals_->get(i) - P_evals_->get(0));
            outfile->Printf("\n    P-space  CI Energy Root %3d        = "
                            "%.12f Eh = %8.4f eV, S^2 = %5.6f",
                            i, abs_energy, exc_energy, spin[i]);
        }
        outfile->Printf("\n");
    }

    if (!quiet_mode_ and options_->get_bool("ACI_PRINT_REFS"))
        print_wfn(P_space_, P_evecs_, num_ref_roots_);
}

void AdaptiveCI::diagonalize_PQ_space() {
    print_h2("Diagonalizing the Hamiltonian in the P + Q space");

    num_ref_roots_ = std::min(nroot_, int(PQ_space_.size()));

    // Step 3. Diagonalize the Hamiltonian in the P + Q space
    local_timer diag_pq;

    outfile->Printf("\n  Number of reference roots: %d", num_ref_roots_);

    auto sigma_vector = make_sigma_vector(PQ_space_, as_ints_, max_memory_, sigma_vector_type_);
    std::tie(PQ_evals_, PQ_evecs_) = sparse_solver_->diagonalize_hamiltonian(
        PQ_space_, sigma_vector, num_ref_roots_, multiplicity_);

    if (!quiet_mode_)
        outfile->Printf("\n  Total time spent diagonalizing H:   %1.6f s", diag_pq.get());

    // Save the solutions for the next iteration
    //        old_dets.clear();
    //        old_dets = PQ_space_;
    //        old_evecs = PQ_evecs->clone();

    auto spin = sparse_solver_->spin();

    if (!quiet_mode_) {
        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots_; ++i) {
            double abs_energy =
                PQ_evals_->get(i) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
            double exc_energy = pc_hartree2ev * (PQ_evals_->get(i) - PQ_evals_->get(0));
            outfile->Printf("\n    PQ-space CI Energy Root %3d        = "
                            "%.12f Eh = %8.4f eV, S^2 = %8.6f",
                            i, abs_energy, exc_energy, spin[i]);
            outfile->Printf("\n    PQ-space CI Energy + EPT2 Root %3d = %.12f Eh = "
                            "%8.4f eV",
                            i, abs_energy + multistate_pt2_energy_correction_[i],
                            exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                                          multistate_pt2_energy_correction_[0]));
        }
        outfile->Printf("\n");
    }

    num_ref_roots_ = std::min(nroot_, int(PQ_space_.size()));

    // If doing root-following, grab the initial root
    if (follow_ and ((pre_iter_ == 0 and cycle_ == 0) or cycle_ == (pre_iter_ - 1))) {
        size_t dim = std::min(static_cast<int>(PQ_space_.size()), 1000);
        P_ref_.subspace(PQ_space_, PQ_evecs_, P_ref_evecs_, dim, ref_root_);
    }

    // if( follow and num_ref_roots > 0 and (cycle >= (pre_iter_ - 1)) ){
    if (follow_ and (num_ref_roots_ > 1) and (cycle_ >= pre_iter_)) {
        ref_root_ = root_follow(P_ref_, P_ref_evecs_, PQ_space_, PQ_evecs_, num_ref_roots_);
    }
}

bool AdaptiveCI::check_convergence() {
    bool stuck = check_stuck(energy_history_, PQ_evals_);
    if (stuck) {
        outfile->Printf("\n  Procedure is stuck! Quitting...");
        return true;
    }

    // Step 4. Check convergence and break if needed
    bool converged = check_convergence(energy_history_, PQ_evals_);
    if (converged) {
        // if(quiet_mode_) outfile->Printf(
        // "\n----------------------------------------------------------" );
        if (!quiet_mode_)
            outfile->Printf("\n  ***** Calculation Converged *****");
        return true;
    }

    return false;
}

void AdaptiveCI::prune_PQ_to_P() {
    // Step 5. Prune the P + Q space to get an updated P space
    print_h2("Pruning the Q space");

    prune_q_space(PQ_space_, P_space_, PQ_evecs_, num_ref_roots_);

    // Print information about the wave function
    if (!quiet_mode_) {
        print_wfn(PQ_space_, PQ_evecs_, num_ref_roots_);
        outfile->Printf("\n  Cycle %d took: %1.6f s", cycle_, cycle_time_.get());
    }
}

void AdaptiveCI::add_bad_roots(DeterminantHashVec& dets) {
    bad_roots_.clear();

    // Look through each state, save common determinants/coeffs
    int nroot = old_roots_.size();
    for (int i = 0; i < nroot; ++i) {

        std::vector<std::pair<size_t, double>> bad_root;
        size_t nadd = 0;
        std::vector<std::pair<Determinant, double>>& state = old_roots_[i];

        for (size_t I = 0, max_I = state.size(); I < max_I; ++I) {
            if (dets.has_det(state[I].first)) {
                //                outfile->Printf("\n %zu, %f ", I,
                //                detmapper[state[I].first] , state[I].second );
                bad_root.push_back(std::make_pair(dets.get_idx(state[I].first), state[I].second));
                nadd++;
            }
        }
        bad_roots_.push_back(bad_root);

        if (!quiet_mode_) {
            outfile->Printf("\n  Added %zu determinants from root %zu", nadd, i);
        }
    }
}

void AdaptiveCI::print_nos() {

    print_h2("ACI Natural Orbitals");

    // Compute a 1-rdm
    CI_RDMS ci_rdm(PQ_space_, as_ints_, PQ_evecs_, 0, 0);
    ci_rdm.set_max_rdm(1);
    std::vector<double> ordm_a_v;
    std::vector<double> ordm_b_v;

    ci_rdm.compute_1rdm_op(ordm_a_v, ordm_b_v);

    psi::Dimension nmopi = mo_space_info_->dimension("ALL");
    psi::Dimension ncmopi = mo_space_info_->dimension("CORRELATED");
    psi::Dimension fdocc = mo_space_info_->dimension("FROZEN_DOCC");

    std::shared_ptr<psi::Matrix> opdm_a(new psi::Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
    std::shared_ptr<psi::Matrix> opdm_b(new psi::Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));

    int offset = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v, ordm_a_v[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v, ordm_b_v[(u + offset) * nact_ + v + offset]);
            }
        }
        offset += nactpi_[h];
    }

    psi::SharedVector OCC_A(new Vector("ALPHA OCCUPATION", nirrep_, nactpi_));
    psi::SharedVector OCC_B(new Vector("BETA OCCUPATION", nirrep_, nactpi_));
    psi::SharedMatrix NO_A(new psi::Matrix(nirrep_, nactpi_, nactpi_));
    psi::SharedMatrix NO_B(new psi::Matrix(nirrep_, nactpi_, nactpi_));

    opdm_a->diagonalize(NO_A, OCC_A, descending);
    opdm_b->diagonalize(NO_B, OCC_B, descending);

    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            auto irrep_occ =
                std::make_pair(OCC_A->get(h, u) + OCC_B->get(h, u), std::make_pair(h, u + 1));
            vec_irrep_occupation.push_back(irrep_occ);
            //          file << OCC_A->get(h, u) + OCC_B->get(h, u) << "  ";
        }
    }

    CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
              std::greater<std::pair<double, std::pair<int, int>>>());

    size_t count = 0;
    outfile->Printf("\n    ");
    for (auto vec : vec_irrep_occupation) {
        outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second, ct.gamma(vec.second.first).symbol(),
                        vec.first);
        if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
            outfile->Printf("\n    ");
    }
}

void AdaptiveCI::update_sigma() { sigma_ = options_->get_double("ACI_RELAX_SIGMA"); }

void AdaptiveCI::full_mrpt2() {
    if (options_->get_bool("FULL_MRPT2")) {
        MRPT2 pt(options_, as_ints_, mo_space_info_, PQ_space_, PQ_evecs_, PQ_evals_, nroot_);
        std::vector<double> pt2 = pt.compute_energy();
        multistate_pt2_energy_correction_ = pt2;
    }
}

DeterminantHashVec AdaptiveCI::get_PQ_space() { return PQ_space_; }

psi::SharedMatrix AdaptiveCI::get_PQ_evecs() { return PQ_evecs_; }

psi::SharedVector AdaptiveCI::get_PQ_evals() { return PQ_evals_; }

size_t AdaptiveCI::get_ref_root() { return ref_root_; }

std::vector<double> AdaptiveCI::get_multistate_pt2_energy_correction() {
    return multistate_pt2_energy_correction_;
}
void AdaptiveCI::post_iter_process() {
    print_nos();
    full_mrpt2();
}

} // namespace forte
