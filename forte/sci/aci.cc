/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER,
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

#include <algorithm>
#include <numeric>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/molecule.h"
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
    : SelectedCIMethod(state, nroot, scf_info, options, mo_space_info, as_ints) {
    startup();
}

void AdaptiveCI::startup() {
    // Read ACI-specific options
    sigma_ = options_->get_double("SIGMA");
    gamma_ = options_->get_double("GAMMA");
    screen_thresh_ = options_->get_double("ACI_PRESCREEN_THRESHOLD");
    add_aimed_degenerate_ = options_->get_bool("ACI_ADD_AIMED_DEGENERATE");
    project_out_spin_contaminants_ = options_->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS");

    gas_iteration_ = false;
    if (options_->get_str("ACTIVE_REF_TYPE") == "GAS_SINGLE" or
        options_->get_str("ACTIVE_REF_TYPE") == "GAS") {
        gas_iteration_ = true;
    }

    // Whether to do the occupation analysis after calculation
    occ_analysis_ = options_->get_bool("OCC_ANALYSIS");

    // Run only one calculation with initial_reference
    // Useful for CIS/CISD/CAS/GAS-CI for test
    one_cycle_ = options_->get_bool("ONE_CYCLE");
    max_cycle_ = one_cycle_ ? 0 : options_->get_int("SCI_MAX_CYCLE");

    spin_tol_ = options_->get_double("ACI_SPIN_TOL");

    // get options for algorithm
    if (options_->get_str("ACI_PQ_FUNCTION") == "MAX") {
        average_function_ = AverageFunction::MaxF;
    } else {
        average_function_ = AverageFunction::AvgF;
    }

    print_weights_ = options_->get_bool("ACI_PRINT_WEIGHTS");

    naverage_ = options_->get_int("ACI_N_AVERAGE");
    average_offset_ = options_->get_int("ACI_AVERAGE_OFFSET");

    // simple checks
    if (naverage_ == 0)
        naverage_ = nroot_;
    if ((average_offset_ + naverage_) > nroot_) {
        std::string except = "The sum of ACI_N_AVERAGE and ACI_AVERAGE_OFFSET is larger than the "
                             "number of roots requested!";
        throw std::runtime_error(except);
    }

    hole_ = 0;

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
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Multiplicity", multiplicity_},
        {"Symmetry", wavefunction_symmetry_},
        {"Number of roots", nroot_},
        {"Root used for properties", root_},
        {"Roots used for averaging", naverage_},
        {"Root averaging offset", average_offset_}};

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
        {"Enforce complete aimed selection", add_aimed_degenerate_ ? "True" : "False"},
        {"Multiroot averaging ", average_function_ == AverageFunction::MaxF ? "Max" : "Average"}};

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
        if (gas_iteration_) {
            get_gas_excited_determinants_avg(nroot_, P_evecs_, P_evals_, P_space_, F_space);
        }
        // multiroot
        else {
            get_excited_determinants_avg(nroot_, P_evecs_, P_evals_, P_space_, F_space);
        }
    } else if (screen_alg == "SR") {
        if (gas_iteration_) {
            get_gas_excited_determinants_sr(P_evecs_, P_evals_, P_space_, F_space);
        } else {
            // single-root optimized
            get_excited_determinants_sr(P_evecs_, P_evals_, P_space_, F_space);
            //    } else if ( (screen_alg == "RESTRICTED")){
            //        // restricted
            //        get_excited_determinants_restrict(nroot_, P_evecs_, P_evals_, P_space_,
            //        F_space);
        }

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
        throw std::runtime_error(except);
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
            outfile->Printf("\n  Added %zu missing determinants in aimed selection (find_q_space).",
                            num_extra);
        }
    }

    multistate_pt2_energy_correction_.resize(nroot_);
    multistate_pt2_energy_correction_[ref_root_] = ept2;

    if (screen_alg == "AVERAGE") {
        for (int n = 0; n < nroot_; ++n) {
            multistate_pt2_energy_correction_[n] = ept2;
        }
    }

    outfile->Printf("\n  Dimension of the PQ space:                  %zu", PQ_space_.size());
    // Check if P+Q space is spin complete
    if (spin_complete_) {
        PQ_space_.make_spin_complete(nact_); // <- xsize
        if (!quiet_mode_)
            outfile->Printf("\n  Dimension of the PQ space (spin-complete) : %zu",
                            PQ_space_.size());
    }

    if ((ex_alg_ == "ROOT_ORTHOGONALIZE") and (root_ > 0) and cycle_ >= pre_iter_) {
        sparse_solver_->set_root_project(true);
        add_bad_roots(PQ_space_);
        sparse_solver_->add_bad_states(bad_roots_);
    }
    outfile->Printf("\n  Time spent building the model space: %1.6f", build_space.get());
}

double AdaptiveCI::average_q_values(const std::vector<double>& E2) {
    // Choose the function of the couplings for each root
    // If nroot = 1, choose the max
    if ((average_function_ == AverageFunction::MaxF) or (nroot_ == 1)) {
        return *std::max_element(E2.begin(), E2.end());
    } else {
        const auto begin = E2.begin() + average_offset_;
        return std::accumulate(begin, begin + naverage_, 0.0) / static_cast<double>(naverage_);
    }
    return 0.0;
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
                               psi::SharedMatrix evecs) {
    // Select the new reference space using the sorted CI coefficients
    P_space.clear();

    double tau_p = sigma_ * gamma_;

    // Create a vector that stores the absolute value of the CI coefficients
    std::vector<std::pair<double, Determinant>> dm_det_list;
    // for (size_t I = 0, max = PQ_space.size(); I < max; ++I){
    const det_hashvec& detmap = PQ_space.wfn_hash();
    for (size_t i = 0, max_i = detmap.size(); i < max_i; ++i) {
        double criterion = 0.0;
        if ((nroot_ > 1) and (ex_alg_ == "AVERAGE" or cycle_ < pre_iter_)) {
            for (int n = 0; n < naverage_; ++n) {
                if (average_function_ == AverageFunction::MaxF) {
                    criterion = std::max(criterion, std::fabs(evecs->get(i, n + average_offset_)));
                } else {
                    criterion += std::fabs(evecs->get(i, n + average_offset_));
                }
            }
            // divide by naverage_ only if we are averaging
            if (average_function_ == AverageFunction::AvgF) {
                criterion /= static_cast<double>(naverage_);
            }
        } else {
            criterion = std::fabs(evecs->get(i, ref_root_));
        }
        dm_det_list.push_back(std::make_pair(criterion, detmap[i]));
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
            outfile->Printf(
                "\n  Added %zu missing determinants in aimed selection (prune_q_space).",
                num_extra);
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
            outfile->Printf("\n Root %d has overlap %f", n, new_overlap);
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
                     wavefunction_symmetry_, state_);

    ref.build_reference(initial_reference_);

    if (one_cycle_) {
        PQ_space_ = initial_reference_;
        PQ_space_.make_spin_complete(nact_);
    } else {
        P_space_ = initial_reference_;
    }

    // P_space_ = initial_reference_;

    // If the ACI iteration is within the gas space, calculate
    // gas_info and the criterion for single and double excitations
    gas_num_ = 0;
    if (gas_iteration_) {
        //        const auto gas_info = mo_space_info_->gas_info();
        std::vector<size_t> act_mo = mo_space_info_->absolute_mo("ACTIVE");
        std::map<int, int> re_ab_mo;
        for (size_t i = 0; i < act_mo.size(); i++) {
            re_ab_mo[act_mo[i]] = i;
        }

        gas_single_criterion_ = ref.gas_single_criterion();
        gas_double_criterion_ = ref.gas_double_criterion();
        gas_electrons_ = ref.gas_electrons();
        for (size_t gas_count = 0; gas_count < 6; gas_count++) {
            std::string space = "GAS" + std::to_string(gas_count + 1);
            std::vector<size_t> relative_mo;
            auto gas_mo = mo_space_info_->absolute_mo(space);
            for (size_t i = 0, imax = gas_mo.size(); i < imax; ++i) {
                relative_mo.push_back(re_ab_mo[gas_mo[i]]);
            }
            if (!relative_mo.empty()) {
                gas_num_ = gas_num_ + 1;
            }

            relative_gas_mo_.push_back(relative_mo);
        }
    }
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
    sparse_solver_->set_ncollapse_per_root(options_->get_int("DL_COLLAPSE_PER_ROOT"));
    sparse_solver_->set_nsubspace_per_root(options_->get_int("DL_SUBSPACE_PER_ROOT"));
    sparse_solver_->set_spin_project_full(
        (gas_iteration_ and sigma_ == 0.0) ? true : options_->get_bool("SPIN_PROJECT_FULL"));
}

void AdaptiveCI::diagonalize_P_space() {
    cycle_time_.reset();
    // Step 1. Diagonalize the Hamiltonian in the P space
    num_ref_roots_ = std::min(nroot_, P_space_.size());
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
                            "%.12f Eh = %8.4f eV, S^2 = %8.6f",
                            i, abs_energy, exc_energy, spin[i]);
        }
        outfile->Printf("\n");
    }

    if (!quiet_mode_ and options_->get_bool("ACI_PRINT_REFS"))
        print_wfn(P_space_, P_evecs_, num_ref_roots_);
}

void AdaptiveCI::diagonalize_PQ_space() {
    print_h2("Diagonalizing the Hamiltonian in the P + Q space");

    num_ref_roots_ = std::min(nroot_, PQ_space_.size());

    if (one_cycle_) {
        zero_multistate_pt2_energy_correction();
    }

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
    PQ_spin2_ = spin;

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

    num_ref_roots_ = std::min(nroot_, PQ_space_.size());

    // If doing root-following, grab the initial root
    if (follow_ and ((pre_iter_ == 0 and cycle_ == 0) or cycle_ == (pre_iter_ - 1))) {
        size_t dim = std::min(static_cast<int>(PQ_space_.size()), 1000);
        P_ref_.subspace(PQ_space_, PQ_evecs_, P_ref_evecs_, dim, ref_root_);
    }

    // if( follow and num_ref_roots > 0 and (cycle >= (pre_iter_ - 1)) ){
    if (follow_ and (num_ref_roots_ > 1) and (cycle_ >= pre_iter_)) {
        ref_root_ = root_follow(P_ref_, P_ref_evecs_, PQ_space_, PQ_evecs_, num_ref_roots_);
    }

    if (gas_iteration_) {
        print_gas_wfn(PQ_space_, PQ_evecs_);
    }
    if (occ_analysis_) {
        print_occ_number(PQ_space_, PQ_evecs_);
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

    prune_q_space(PQ_space_, P_space_, PQ_evecs_);

    // Print information about the wave function
    if (!quiet_mode_) {
        //        (PQ_space_, PQ_evecs_, num_ref_roots_);
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

    std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
              std::greater<std::pair<double, std::pair<int, int>>>());

    size_t count = 0;
    outfile->Printf("\n    ");
    for (auto vec : vec_irrep_occupation) {
        outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second,
                        mo_space_info_->irrep_label(vec.second.first).c_str(), vec.first);
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

std::vector<double> AdaptiveCI::get_PQ_spin2() { return PQ_spin2_; }

size_t AdaptiveCI::get_ref_root() { return ref_root_; }

std::vector<double> AdaptiveCI::get_multistate_pt2_energy_correction() {
    return multistate_pt2_energy_correction_;
}

void AdaptiveCI::zero_multistate_pt2_energy_correction() {
    multistate_pt2_energy_correction_.assign(nroot_, 0.0);
}

void AdaptiveCI::print_gas_wfn(DeterminantHashVec& space, psi::SharedMatrix evecs) {
    std::vector<std::string> gas_electron_name = {"GAS1_A", "GAS1_B", "GAS2_A", "GAS2_B",
                                                  "GAS3_A", "GAS3_B", "GAS4_A", "GAS4_B",
                                                  "GAS5_A", "GAS5_B", "GAS6_A", "GAS6_B"};
    print_h2("GAS Contribution Analysis");

    for (size_t n = 0; n < nroot_; ++n) {
        DeterminantHashVec tmp;
        std::vector<double> tmp_evecs;

        psi::outfile->Printf("\n  Root %zu:", n);

        size_t max_dets = static_cast<size_t>(evecs->nrow());
        tmp.subspace(space, evecs, tmp_evecs, max_dets, n);

        const size_t gas_config_num = gas_electrons_.size();
        std::vector<double> gas_amp(gas_config_num, 0.0);
        for (size_t I = 0; I < max_dets; ++I) {
            const Determinant& det = tmp.get_det(I);
            std::vector<std::vector<int>> gas_occ_a;
            std::vector<std::vector<int>> gas_vir_a;
            std::vector<std::vector<int>> gas_occ_b;
            std::vector<std::vector<int>> gas_vir_b;
            for (size_t gas_count = 0; gas_count < gas_num_; gas_count++) {
                std::vector<int> occ_a;
                std::vector<int> occ_b;
                std::vector<int> vir_a;
                std::vector<int> vir_b;
                for (const auto& p : relative_gas_mo_[gas_count]) {
                    if (det.get_alfa_bit(p)) {
                        occ_a.push_back(p);
                    } else {
                        vir_a.push_back(p);
                    }
                    if (det.get_beta_bit(p)) {
                        occ_b.push_back(p);
                    } else {
                        vir_b.push_back(p);
                    }
                }
                gas_occ_a.push_back(occ_a);
                gas_vir_a.push_back(vir_a);
                gas_occ_b.push_back(occ_b);
                gas_vir_b.push_back(vir_b);
            }

            // Generate the number of electrons in each GAS
            std::vector<int> gas_configuration;
            for (size_t gas_count = 0; gas_count < 6; gas_count++) {
                if (gas_count < gas_num_) {
                    gas_configuration.push_back(gas_occ_a[gas_count].size());
                    gas_configuration.push_back(gas_occ_b[gas_count].size());
                } else {
                    gas_configuration.push_back(0);
                    gas_configuration.push_back(0);
                }
            }

            auto gas_found =
                std::find(gas_electrons_.begin(), gas_electrons_.end(), gas_configuration);
            if (gas_found != gas_electrons_.end()) {
                size_t gas_config = std::distance(gas_electrons_.begin(), gas_found);
                gas_amp[gas_config] += tmp_evecs[I] * tmp_evecs[I];
            } else {
                outfile->Printf("\n  Not found! There is a problem with the wavefunction!");
                exit(1);
            }
        }

        outfile->Printf("\n    Config.");
        int ndash = 7;
        for (size_t j = 0; j < 2 * gas_num_; j++) {
            std::string name = gas_electron_name[j].substr(3, 3);
            outfile->Printf("  %s", name.c_str());
            ndash += 5;
        }
        outfile->Printf("  Contribution");
        ndash += 14;
        std::string dash(ndash, '-');
        outfile->Printf("\n    %s", dash.c_str());

        std::map<std::vector<size_t>, double> gas_total_amp;
        for (size_t i = 0; i < gas_config_num; i++) {
            outfile->Printf("\n    %6d ", i + 1);
            for (size_t j = 0; j < 2 * gas_num_; j++) {
                outfile->Printf(" %4d", gas_electrons_[i][j]);
            }

            std::vector<size_t> sum_gas;
            for (size_t j = 0; j < 2 * gas_num_; j += 2) {
                sum_gas.push_back(gas_electrons_[i][j] + gas_electrons_[i][j + 1]);
            }
            auto map_found = gas_total_amp.find(sum_gas);
            if (map_found == gas_total_amp.end()) {
                gas_total_amp[sum_gas] = gas_amp[i];
            } else {
                gas_total_amp[sum_gas] += gas_amp[i];
            }

            outfile->Printf(" %12.8f%%", gas_amp[i] * 100.0);
        }
        outfile->Printf("\n    %s", dash.c_str());

        outfile->Printf("\n    %7c", ' ');
        for (size_t j = 0; j < gas_num_; j++) {
            std::string name = "GAS" + std::to_string(j + 1);
            outfile->Printf("  %6s  ", name.c_str());
        }
        outfile->Printf("  Contribution");
        outfile->Printf("\n    %s", dash.c_str());

        for (auto element : gas_total_amp) {
            outfile->Printf("\n    %7c", ' ');
            for (size_t j = 0; j < gas_num_; j++) {
                outfile->Printf("  %6d  ", element.first[j]);
            }
            outfile->Printf(" %12.8f%%", element.second * 100);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}

void AdaptiveCI::print_occ_number(DeterminantHashVec& space, psi::SharedMatrix evecs) {
    std::vector<size_t> act_orb;
    for (size_t i = 0; i < nact_; i++) {
        act_orb.push_back(i);
    }
    double occ_limit = options_->get_double("OCC_LIMIT");
    double corr_limit = options_->get_double("CORR_LIMIT");

    if (corr_limit > 0) {
        corr_limit = -1.0 * corr_limit;
    }

    psi::outfile->Printf("\n  ");
    psi::outfile->Printf("\n  Occupation Number Analysis ");
    for (size_t n = 0; n < nroot_; ++n) {
        DeterminantHashVec tmp;
        std::vector<double> tmp_evecs;

        psi::outfile->Printf("\n  Occupation number of orbitals root %d: ", n);

        size_t max_dets = static_cast<size_t>(evecs->nrow());
        tmp.subspace(space, evecs, tmp_evecs, max_dets, n);

        std::vector<double> alpha_occ(nact_, 0.0);
        std::vector<double> beta_occ(nact_, 0.0);
        std::vector<std::vector<double>> alpha_occ_square(nact_, std::vector<double>(nact_, 0.0));
        std::vector<std::vector<double>> beta_occ_square(nact_, std::vector<double>(nact_, 0.0));
        std::vector<std::vector<double>> alpha_corr(nact_, std::vector<double>(nact_, 0.0));
        std::vector<std::vector<double>> beta_corr(nact_, std::vector<double>(nact_, 0.0));

        for (size_t I = 0; I < max_dets; ++I) {
            const Determinant& det = tmp.get_det(I);
            for (size_t j = 0; j < nact_; ++j) {
                if (det.get_alfa_bit(j)) {
                    alpha_occ[j] += tmp_evecs[I] * tmp_evecs[I];
                    for (size_t k = j + 1; k < nact_; k++) {
                        if (tmp.get_det(I).get_alfa_bit(k)) {
                            alpha_occ_square[j][k] += tmp_evecs[I] * tmp_evecs[I];
                        }
                    }
                }
                if (det.get_beta_bit(j)) {
                    beta_occ[j] += tmp_evecs[I] * tmp_evecs[I];
                    for (size_t k = j + 1; k < nact_; k++) {
                        if (tmp.get_det(I).get_beta_bit(k)) {
                            beta_occ_square[j][k] += tmp_evecs[I] * tmp_evecs[I];
                        }
                    }
                }
            }
        }

        std::vector<size_t> fixed_orb;

        outfile->Printf("\n  Orb    alpha    beta     ");
        for (size_t j = 0; j < nact_; ++j) {
            outfile->Printf("\n  %3zu  %.5f ", j, alpha_occ[j]);
            outfile->Printf(" %.5f ", j, beta_occ[j]);
        }

        outfile->Printf("\n ");
        outfile->Printf("\n  These orbitals can be treated as uocc:");
        for (size_t j = 0; j < nact_; ++j) {
            if (alpha_occ[j] < occ_limit && beta_occ[j] < occ_limit) {
                outfile->Printf(" %d", j);
                fixed_orb.push_back(j);
            }
        }
        outfile->Printf("\n  These orbitals can be treated as docc:");
        for (size_t j = 0; j < nact_; ++j) {
            if (alpha_occ[j] > 1 - occ_limit && beta_occ[j] > 1 - occ_limit) {
                outfile->Printf(" %d", j);
                fixed_orb.push_back(j);
            }
        }

        std::sort(fixed_orb.begin(), fixed_orb.end());
        std::vector<size_t> corr_orb;
        std::set_difference(act_orb.begin(), act_orb.end(), fixed_orb.begin(), fixed_orb.end(),
                            std::inserter(corr_orb, corr_orb.begin()));
        for (size_t j : corr_orb) {
            for (size_t k : corr_orb) {
                if (k > j) {
                    alpha_corr[j][k] = (alpha_occ_square[j][k] - alpha_occ[j] * alpha_occ[k]) /
                                       sqrt(alpha_occ[j] - alpha_occ[j] * alpha_occ[j]) /
                                       sqrt(alpha_occ[k] - alpha_occ[k] * alpha_occ[k]);
                    alpha_corr[k][j] = alpha_corr[j][k];
                    beta_corr[j][k] = (beta_occ_square[j][k] - beta_occ[j] * beta_occ[k]) /
                                      sqrt(beta_occ[j] - beta_occ[j] * beta_occ[j]) /
                                      sqrt(beta_occ[k] - beta_occ[k] * beta_occ[k]);
                    beta_corr[k][j] = beta_corr[j][k];
                }
            }
        }

        std::vector<std::vector<size_t>> corr_orb_list;
        std::vector<size_t> residue_orb = corr_orb;
        while (!residue_orb.empty()) {
            std::vector<size_t> tmp_gas;
            std::vector<size_t> tmp_residue;
            tmp_gas.push_back(residue_orb[0]);
            bool well_sep = false;
            for (size_t i = 1, i_max = residue_orb.size(); i < i_max; i++) {
                tmp_residue.push_back(residue_orb[i]);
            }
            while (!well_sep && !tmp_residue.empty()) {
                std::vector<size_t> tmp_change;
                for (size_t j : tmp_gas) {
                    for (size_t k : tmp_residue) {
                        if (alpha_corr[j][k] < corr_limit && beta_corr[j][k] < corr_limit &&
                            std::find(tmp_change.begin(), tmp_change.end(), k) ==
                                tmp_change.end()) {
                            tmp_change.push_back(k);
                        }
                    }
                }
                if (tmp_change.empty()) {
                    well_sep = true;
                } else {
                    std::vector<size_t> tmp_residue_p;
                    std::sort(tmp_change.begin(), tmp_change.end());
                    tmp_gas.insert(std::end(tmp_gas), std::begin(tmp_change), std::end(tmp_change));
                    std::set_difference(tmp_residue.begin(), tmp_residue.end(), tmp_change.begin(),
                                        tmp_change.end(),
                                        std::inserter(tmp_residue_p, tmp_residue_p.begin()));
                    tmp_residue = tmp_residue_p;
                }
            }
            residue_orb.clear();
            residue_orb = tmp_residue;
            std::sort(tmp_gas.begin(), tmp_gas.end());
            corr_orb_list.push_back(tmp_gas);
            tmp_gas.clear();
        }

        outfile->Printf("\n  ");
        outfile->Printf("\n  Possible GAS occupation:");

        for (auto const& corr_orb : corr_orb_list) {
            size_t max_occ = corr_orb.size();
            std::vector<double> gas_occ_alpha(max_occ + 1, 0.0);
            std::vector<double> gas_occ_beta(max_occ + 1, 0.0);
            std::vector<double> gas_occ_total(max_occ * 2 + 1, 0.0);
            for (size_t I = 0; I < max_dets; ++I) {
                const Determinant& det = tmp.get_det(I);
                size_t tmp_alpha_occ = 0;
                size_t tmp_beta_occ = 0;
                for (size_t j : corr_orb) {
                    if (det.get_alfa_bit(j)) {
                        tmp_alpha_occ += 1;
                    }
                    if (det.get_beta_bit(j)) {
                        tmp_beta_occ += 1;
                    }
                }
                gas_occ_alpha[tmp_alpha_occ] += tmp_evecs[I] * tmp_evecs[I];
                gas_occ_beta[tmp_beta_occ] += tmp_evecs[I] * tmp_evecs[I];
                gas_occ_total[tmp_alpha_occ + tmp_beta_occ] += tmp_evecs[I] * tmp_evecs[I];
            }
            outfile->Printf("\n  Orbitals:");
            for (size_t j : corr_orb) {
                outfile->Printf(" %d", j);
            }
            //            outfile->Printf("\n  Number of Alpha electrons:");
            //            for (size_t k = 0; k <= max_occ; k++) {
            //                outfile->Printf("\n  %d  %.5f", k, gas_occ_alpha[k]);
            //            }
            //            outfile->Printf("\n  Number of Beta electrons:");
            //            for (size_t k = 0; k <= max_occ; k++) {
            //                outfile->Printf("\n  %d  %.5f", k, gas_occ_beta[k]);
            //            }
            outfile->Printf("\n  Number of Total electrons:");
            for (size_t k = 0; k <= max_occ * 2; k++) {
                outfile->Printf("\n  %d  %.5f", k, gas_occ_total[k]);
            }
            outfile->Printf("\n  ");
        }
    }
} // namespace forte

void AdaptiveCI::post_iter_process() {
    //    if (gas_iteration_) {
    //        print_gas_wfn(PQ_space_, PQ_evecs_);
    //    }
    //    if (occ_analysis_) {
    //        print_occ_number(PQ_space_, PQ_evecs_);
    //    }
    print_nos();
    full_mrpt2();
}

} // namespace forte
