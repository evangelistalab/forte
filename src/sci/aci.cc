/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/forte_options.h"
#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "sci/aci.h"

using namespace psi;

namespace forte {

bool pairComp(const std::pair<double, Determinant> E1, const std::pair<double, Determinant> E2){
    return E1.first < E2.first;
}


AdaptiveCI::AdaptiveCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                               std::shared_ptr<ForteOptions> options,
                               std::shared_ptr<MOSpaceInfo> mo_space_info,
                               std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : SelectedCIMethod(state, nroot, scf_info, mo_space_info, as_ints), sparse_solver_(as_ints_),
      options_(options) {
    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");
    sigma_ = options_->get_double("SIGMA");
    nuclear_repulsion_energy_ = as_ints->ints()->nuclear_repulsion_energy();
    nroot_ = nroot;
    startup();
}

void AdaptiveCI::set_fci_ints(std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    as_ints_ = fci_ints;
    nuclear_repulsion_energy_ = as_ints_->ints()->nuclear_repulsion_energy();
    set_ints_ = true;
}

void AdaptiveCI::startup() {
    quiet_mode_ = false;
    if (options_->has_changed("ACI_QUIET_MODE")) {
        quiet_mode_ = options_->get_bool("ACI_QUIET_MODE");
    }

    //    if (!set_ints_) {
    //        set_aci_ints(ints_); // TODO: maybe a BUG?
    //    }

    op_.initialize(mo_symmetry_, as_ints_);
    op_.set_quiet_mode(quiet_mode_);

    wavefunction_symmetry_ = state_.irrep();
    multiplicity_ = state_.multiplicity();

    nact_ = mo_space_info_->size("ACTIVE");
    nactpi_ = mo_space_info_->get_dimension("ACTIVE");

    // Include frozen_docc and restricted_docc
    frzcpi_ = mo_space_info_->get_dimension("INACTIVE_DOCC");
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

    max_cycle_ = 20;
    if (options_->has_changed("SCI_MAX_CYCLE")) {
        max_cycle_ = options_->get_int("SCI_MAX_CYCLE");
    }
    pre_iter_ = 0;
    if (options_->has_changed("ACI_PREITERATIONS")) {
        pre_iter_ = options_->get_int("ACI_PREITERATIONS");
    }

    spin_tol_ = options_->get_double("ACI_SPIN_TOL");
    // set the initial S^@ guess as input multiplicity
    int S = (multiplicity_ - 1.0) / 2.0;
    int S2 = multiplicity_ - 1.0;
    for (int n = 0; n < nroot_; ++n) {
        root_spin_vec_.push_back(std::make_pair(S, S2));
    }

    // get options for algorithm
    perturb_select_ = options_->get_bool("ACI_PERTURB_SELECT");
    pq_function_ = options_->get_str("ACI_PQ_FUNCTION");
    approx_rdm_ = options_->get_bool("ACI_APPROXIMATE_RDM");
    print_weights_ = options_->get_bool("ACI_PRINT_WEIGHTS");

    hole_ = 0;

    diag_method_ = DLSolver;
    if (options_->has_changed("DIAG_ALGORITHM")) {
        if (options_->get_str("DIAG_ALGORITHM") == "FULL") {
            diag_method_ = Full;
        } else if (options_->get_str("DIAG_ALGORITHM") == "DLSTRING") {
            diag_method_ = DLString;
        } else if (options_->get_str("DIAG_ALGORITHM") == "SPARSE") {
            diag_method_ = Sparse;
        } else if (options_->get_str("DIAG_ALGORITHM") == "SOLVER") {
            diag_method_ = DLSolver;
        } else if (options_->get_str("DIAG_ALGORITHM") == "DYNAMIC") {
            diag_method_ = Dynamic;
        }
    }
    aimed_selection_ = false;
    energy_selection_ = false;
    if (options_->get_str("ACI_SELECT_TYPE") == "AIMED_AMP") {
        aimed_selection_ = true;
        energy_selection_ = false;
    } else if (options_->get_str("ACI_SELECT_TYPE") == "AIMED_ENERGY") {
        aimed_selection_ = true;
        energy_selection_ = true;
    } else if (options_->get_str("ACI_SELECT_TYPE") == "ENERGY") {
        aimed_selection_ = false;
        energy_selection_ = true;
    } else if (options_->get_str("ACI_SELECT_TYPE") == "AMP") {
        aimed_selection_ = false;
        energy_selection_ = false;
    }

    if (options_->get_bool("ACI_STREAMLINE_Q") == true) {
        streamline_qspace_ = true;
    } else {
        streamline_qspace_ = false;
    }

    // Set streamline mode to true if possible
    if ((nroot_ == 1) and (aimed_selection_ == true) and (energy_selection_ == true) and
        (perturb_select_ == false)) {

        streamline_qspace_ = true;
    }

    // Decide when to compute coupling lists
    build_lists_ = true;
    if (diag_method_ == Dynamic) {
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
        {"Determinant selection criterion",
         energy_selection_ ? "Second-order Energy" : "First-order Coefficients"},
        {"Selection criterion", aimed_selection_ ? "Aimed selection" : "Threshold"},
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

        auto actmo = mo_space_info_->get_absolute_mo("ACTIVE");

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

void AdaptiveCI::unpaired_density(psi::SharedMatrix Ua, psi::SharedMatrix Ub) {
    //    UPDensity density(as_ints_->ints(), mo_space_info_, options_, Ua, Ub);
    //    density.compute_unpaired_density(ordm_a_, ordm_b_);
}
void AdaptiveCI::unpaired_density(ambit::Tensor Ua, ambit::Tensor Ub) {
    //
    //    Matrix am = tensor_to_matrix(Ua, nactpi_);
    //    Matrix bm = tensor_to_matrix(Ub, nactpi_);
    //
    //    psi::SharedMatrix Uam(new psi::Matrix(nactpi_, nactpi_));
    //    psi::SharedMatrix Ubm(new psi::Matrix(nactpi_, nactpi_));
    //
    //    Uam->copy(am);
    //    Ubm->copy(bm);
    //
    //    UPDensity density(as_ints_->ints(), mo_space_info_, options_, Uam, Ubm);
    //    density.compute_unpaired_density(ordm_a_, ordm_b_);
}

void AdaptiveCI::find_q_space_batched(DeterminantHashVec& P_space, DeterminantHashVec& PQ_space,
                                          psi::SharedVector evals, psi::SharedMatrix evecs) {

    timer find_q("ACI:Build Model Space");
    local_timer build;
    outfile->Printf("\n  Using batched Q_space algorithm");

    std::vector<std::pair<double, Determinant>> F_space;
    double remainder = 0.0;
    if (options_->get_str("ACI_BATCH_ALG") == "HASH") {
        remainder = get_excited_determinants_batch(evecs, evals, P_space, F_space);
    } else if (options_->get_str("ACI_BATCH_ALG") == "VECSORT") {
        remainder = get_excited_determinants_batch_vecsort(evecs, evals, P_space, F_space);
    }

    PQ_space.clear();
    external_wfn_.clear();
    PQ_space.swap(P_space);

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the truncated SD space",
                        F_space.size());
        outfile->Printf("\n  %s: %f s\n", "Time spent building the external space (default)",
                        build.get());
    }

    local_timer sorter;
    std::sort(F_space.begin(), F_space.end(), pairComp);
    outfile->Printf("\n  Time spent sorting: %1.6f", sorter.get());

    local_timer screen;
    double ept2 = 0.0 - remainder;
    local_timer select;
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
            PQ_space.add(det);
        }
    }
    // Add missing determinants
    if (add_aimed_degenerate_) {
        size_t num_extra = 0;
        for (size_t I = 0, max_I = last_excluded; I < max_I; ++I) {
            size_t J = last_excluded - I;
            if (std::fabs(F_space[last_excluded + 1].first - F_space[J].first) < 1.0e-9) {
                PQ_space.add(F_space[J].second);
                num_extra++;
            } else {
                break;
            }
        }
        if (num_extra > 0 and (!quiet_mode_)) {
            outfile->Printf("\n  Added %zu missing determinants in aimed selection.", num_extra);
        }
    }
    outfile->Printf("\n  Time spent selecting: %1.6f", select.get());
    multistate_pt2_energy_correction_.resize(nroot_);
    multistate_pt2_energy_correction_[0] = ept2;

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the P + Q space",
                        PQ_space.size());
        outfile->Printf("\n  %s: %f s", "Time spent screening the model space", screen.get());
    }
}

void AdaptiveCI::default_find_q_space(DeterminantHashVec& P_space, DeterminantHashVec& PQ_space,
                                          psi::SharedVector evals, psi::SharedMatrix evecs) {
    timer find_q("ACI:Build Model Space");
    local_timer build;

    det_hash<double> V_hash;
    get_excited_determinants_sr(evecs, P_space, V_hash);

    // This will contain all the determinants
    PQ_space.clear();
    external_wfn_.clear();
    // Add the P-space determinants and zero the hash
    local_timer erase;
    const det_hashvec& detmap = P_space.wfn_hash();
    for (det_hashvec::iterator it = detmap.begin(), endit = detmap.end(); it != endit; ++it) {
        V_hash.erase(*it);
    }
    PQ_space.swap(P_space);
    outfile->Printf("\n  Time spent preparing PQ_space: %1.6f", erase.get());

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the SD space", V_hash.size());
        outfile->Printf("\n  %s: %f s\n", "Time spent building the external space (default)",
                        build.get());
    }

    local_timer screen;

    // Compute criteria for all dets, store them all
    Determinant zero_det; // <- xsize (nact_);
    std::vector<std::pair<double, Determinant>> F_space(V_hash.size(),
                                                        std::make_pair(0.0, zero_det));
    local_timer build_sort;
#pragma omp parallel
    {
        int num_thread = omp_get_max_threads();
        int tid = omp_get_thread_num();
        size_t N = 0;
        // sorted_dets.reserve(max);
        for (const auto& I : V_hash) {
            if ((N % num_thread) == tid) {
                double delta = as_ints_->energy(I.first) - evals->get(0);
                double V = I.second;

                double criteria = 0.5 * (delta - sqrt(delta * delta + V * V * 4.0));
                F_space[N] = std::make_pair(std::fabs(criteria), I.first);
            }
            N++;
        }
    }
    outfile->Printf("\n  Time spent building sorting list: %1.6f", build_sort.get());

    local_timer sorter;
    std::sort(F_space.begin(), F_space.end(), pairComp);
    outfile->Printf("\n  Time spent sorting: %1.6f", sorter.get());

    double ept2 = 0.0;
    local_timer select;
    double sum = 0.0;
    size_t last_excluded = 0;
    for (size_t I = 0, max_I = F_space.size(); I < max_I; ++I) {
        double& energy = F_space[I].first;
        Determinant& det = F_space[I].second;
        if (sum + energy < sigma_) {
            sum += energy;
            ept2 -= energy;
            last_excluded = I;

        } else {
            PQ_space.add(det);
        }
    }
    // Add missing determinants
    if (add_aimed_degenerate_) {
        size_t num_extra = 0;
        for (size_t I = 0, max_I = last_excluded; I < max_I; ++I) {
            size_t J = last_excluded - I;
            if (std::fabs(F_space[last_excluded + 1].first - F_space[J].first) < 1.0e-9) {
                PQ_space.add(F_space[J].second);
                num_extra++;
            } else {
                break;
            }
        }
        if (num_extra > 0 and (!quiet_mode_)) {
            outfile->Printf("\n  Added %zu missing determinants in aimed selection.", num_extra);
        }
    }
    outfile->Printf("\n  Time spent selecting: %1.6f", select.get());
    multistate_pt2_energy_correction_.resize(nroot_);
    multistate_pt2_energy_correction_[0] = ept2;

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the P + Q space",
                        PQ_space.size());
        outfile->Printf("\n  %s: %f s", "Time spent screening the model space", screen.get());
    }
}

void AdaptiveCI::find_q_space_multiroot(DeterminantHashVec& P_space,
                                            DeterminantHashVec& PQ_space, int nroot,
                                            psi::SharedVector evals, psi::SharedMatrix evecs) {
    timer find_q("ACI:Build Model Space");
    local_timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    det_hash<std::vector<double>> V_hash;
    if (options_->get_bool("ACI_LOW_MEM_SCREENING")) {
        get_excited_determinants_seq(nroot_, evecs, P_space, V_hash);
    } else if (options_->get_bool("SCI_CORE_EX") and (root_ > 0)) {
        get_core_excited_determinants(evecs, P_space, V_hash);
    } else {
        get_excited_determinants(nroot_, evecs, P_space, V_hash);
    }

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the SD space", V_hash.size());
        outfile->Printf("\n  %s: %f s\n", "Time spent building the external space",
                        t_ms_build.get());
    }

    // This will contain all the determinants
    PQ_space.clear();

    // Add the P-space determinants and zero the hash
    PQ_space.copy(P_space);

    local_timer t_ms_screen;

    // Define coupling out of loop, assume perturb_select_ = false
    std::function<double(double A, double B, double C)> C1_eq = [](double A, double B,
                                                                   double C) -> double {
        return 0.5 * ((B - C) - sqrt((B - C) * (B - C) + 4.0 * A * A)) / A;
    };

    std::function<double(double A, double B, double C)> E2_eq = [](double A, double B,
                                                                   double C) -> double {
        return 0.5 * ((B - C) - sqrt((B - C) * (B - C) + 4.0 * A * A));
    };

    if (perturb_select_) {
        C1_eq = [](double A, double B, double C) -> double { return -A / (B - C); };
        E2_eq = [](double A, double B, double C) -> double { return -A * A / (B - C); };
    }

    // Check the coupling between the reference and the SD space

    std::vector<std::pair<double, Determinant>> sorted_dets;
    std::vector<double> ept2(nroot_, 0.0);

    if (aimed_selection_) {
        Determinant zero_det; // <- xsize (nact_);
        sorted_dets.resize(V_hash.size(), std::make_pair(0.0, zero_det));
    }

#pragma omp parallel
    {
        int ithread = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        double criteria;

        std::vector<double> C1(nroot_, 0.0);
        std::vector<double> E2(nroot_, 0.0);
        std::vector<double> e2(nroot_, 0.0);

        size_t count = 0;
        for (const auto& it : V_hash) {
            if ((count % nthreads) == ithread) {
                double EI = as_ints_->energy(it.first);
                for (int n = 0; n < nroot; ++n) {
                    double V = it.second[n];
                    double C1_I = C1_eq(V, EI, evals->get(n));
                    double E2_I = E2_eq(V, EI, evals->get(n));

                    C1[n] = std::fabs(C1_I);
                    E2[n] = std::fabs(E2_I);

                    e2[n] = E2_I;
                }
                if ((ex_alg_ == "AVERAGE" or cycle_ < pre_iter_) and nroot > 1) {
                    criteria = average_q_values(nroot, C1, E2);
                } else {
                    criteria = root_select(nroot, C1, E2);
                }

                if (aimed_selection_) {
                    sorted_dets[count] = std::make_pair(criteria, it.first);
                } else {
                    if (std::fabs(criteria) > sigma_) {
#pragma omp critical
                        { PQ_space.add(it.first); }
                    } else {
#pragma omp critical
                        {
                            for (int n = 0; n < nroot; ++n) {
                                ept2[n] += e2[n];
                            }
                        }
                    }
                }
            }
            count++;
        }
    } // end loop over determinants

    if (aimed_selection_) {
        // Sort the CI coefficients in ascending order
        std::sort(sorted_dets.begin(), sorted_dets.end(), pairComp);

        double sum = 0.0;
        size_t last_excluded = 0;
        for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I) {
            const Determinant& det = sorted_dets[I].second;
            if (sum + sorted_dets[I].first < sigma_) {
                sum += sorted_dets[I].first;
                double EI = as_ints_->energy(det);
                const std::vector<double>& V_vec = V_hash[det];
                for (int n = 0; n < nroot; ++n) {
                    double V = V_vec[n];
                    double E2_I = E2_eq(V, EI, evals->get(n));

                    ept2[n] += E2_I;
                }
                last_excluded = I;

            } else {
                PQ_space.add(sorted_dets[I].second);
            }
        }
        // outfile->Printf("\n sum : %1.12f", sum );
        // add missing determinants that have the same weight as the last one
        // included
        if (add_aimed_degenerate_) {
            size_t num_extra = 0;
            for (size_t I = 0, max_I = last_excluded; I < max_I; ++I) {
                size_t J = last_excluded - I;
                if (std::fabs(sorted_dets[last_excluded + 1].first - sorted_dets[J].first) <
                    1.0e-9) {
                    PQ_space.add(sorted_dets[J].second);
                    num_extra++;
                } else {
                    break;
                }
            }
            if (num_extra > 0 and (!quiet_mode_)) {
                outfile->Printf("\n  Added %zu missing determinants in aimed selection.",
                                num_extra);
            }
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the P + Q space",
                        PQ_space.size());
        outfile->Printf("\n  %s: %f s", "Time spent screening the model space", t_ms_screen.get());
    }
}

double AdaptiveCI::average_q_values(int nroot, std::vector<double>& C1,
                                        std::vector<double>& E2) {
    // f_E2 and f_C1 will store the selected function of the chosen q criteria
    // This functions should only be called when nroot_ > 1

    int nav = options_->get_int("ACI_N_AVERAGE");
    int off = options_->get_int("ACI_AVERAGE_OFFSET");
    if (nav == 0)
        nav = nroot;
    if ((off + nav) > nroot)
        off = nroot - nav; // throw psi::PSIEXCEPTION("\n  Your desired number of
                           // roots and the offset exceeds the maximum number of
                           // roots!");

    double f_C1 = 0.0;
    double f_E2 = 0.0;

    // Choose the function of the couplings for each root
    // If nroot = 1, choose the max

    if (pq_function_ == "MAX" or nroot == 1) {
        f_C1 = *std::max_element(C1.begin(), C1.end());
        f_E2 = *std::max_element(E2.begin(), E2.end());
    } else if (pq_function_ == "AVERAGE") {
        double C1_average = 0.0;
        double E2_average = 0.0;
        double dim_inv = 1.0 / nav;
        for (int n = 0; n < nav; ++n) {
            C1_average += C1[n + off] * dim_inv;
            E2_average += E2[n + off] * dim_inv;
        }

        f_C1 = C1_average;
        f_E2 = E2_average;
    }

    double select_value = 0.0;
    if (aimed_selection_) {
        select_value = energy_selection_ ? f_E2 : (f_C1 * f_C1);
    } else {
        select_value = energy_selection_ ? f_E2 : f_C1;
    }

    return select_value;
}

double AdaptiveCI::root_select(int nroot, std::vector<double>& C1, std::vector<double>& E2) {
    double select_value;

    if (ref_root_ + 1 > nroot_) {
        outfile->Printf("\n  nroot: %d, ref_roof: %d", nroot_, ref_root_);
        throw psi::PSIEXCEPTION("\n  Your selection is not valid. Check ROOT in options.");
    }
    int root = ref_root_;
    if (nroot == 1) {
        ref_root_ = 0;
    }

    if (aimed_selection_) {
        select_value = energy_selection_ ? E2[root] : (C1[root] * C1[root]);
    } else {
        select_value = energy_selection_ ? E2[root] : C1[root];
    }

    return select_value;
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
    if (aimed_selection_) {
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
                if (std::fabs(dm_det_list[last_excluded + 1].first - dm_det_list[J].first) <
                    1.0e-9) {
                    P_space.add(dm_det_list[J].second);
                    num_extra += 1;
                } else {
                    break;
                }
            }
            if (num_extra > 0 and !quiet_mode_) {
                outfile->Printf("\n  Added %zu missing determinants in aimed selection.",
                                num_extra);
            }
        }
    }
    // Include all determinants such that |C_I| > tau_p
    else {
        for (size_t I = 0, max_I = PQ_space.size(); I < max_I; ++I) {
            if (dm_det_list[I].first > tau_p) {
                P_space.add(dm_det_list[I].second);
            }
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

std::vector<std::pair<double, double>> AdaptiveCI::compute_spin(DeterminantHashVec& space,
                                                                    WFNOperator& op,
                                                                    psi::SharedMatrix evecs,
                                                                    int nroot) {
    // WFNOperator op(mo_symmetry_);

    // op.build_strings(space);
    // op.op_lists(space);
    // op.tp_lists(space);

    std::vector<std::pair<double, double>> spin_vec(nroot);
    if (options_->get_str("SIGMA_BUILD_TYPE") == "HZ") {
        op.clear_op_s_lists();
        op.clear_tp_s_lists();
        op.build_strings(space);
        op.op_lists(space);
        op.tp_lists(space);
    }

    if (!build_lists_) {
        for (int n = 0; n < nroot_; ++n) {
            double S2 = op.s2_direct(space, evecs, n);
            double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
            spin_vec[n] = std::make_pair(S, S2);
        }
    } else {
        for (int n = 0; n < nroot_; ++n) {
            double S2 = op.s2(space, evecs, n);
            double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
            spin_vec[n] = std::make_pair(S, S2);
        }
    }
    return spin_vec;
}

void AdaptiveCI::print_wfn(DeterminantHashVec& space, WFNOperator& op, psi::SharedMatrix evecs,
                               int nroot) {
    std::string state_label;
    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet", "decatet"});

    std::vector<std::pair<double, double>> spins = compute_spin(space, op, evecs, nroot);

    for (int n = 0; n < nroot; ++n) {
        DeterminantHashVec tmp;
        std::vector<double> tmp_evecs;

        outfile->Printf("\n\n  Most important contributions to root %3d:", n);

        size_t max_dets = std::min(10, evecs->nrow());
        tmp.subspace(space, evecs, tmp_evecs, max_dets, n);

        for (size_t I = 0; I < max_dets; ++I) {
            outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s", I, tmp_evecs[I],
                            tmp_evecs[I] * tmp_evecs[I], space.get_idx(tmp.get_det(I)),
                            tmp.get_det(I).str(nact_).c_str());
        }
        state_label = s2_labels[std::round(spins[n].first * 2.0)];
        root_spin_vec_[n] = std::make_pair(spins[n].first, spins[n].second);
        outfile->Printf("\n\n  Spin state for root %zu: S^2 = %5.6f, S = %5.3f, %s", n,
                        root_spin_vec_[n].first, root_spin_vec_[n].second, state_label.c_str());
    }
}

std::vector<double> AdaptiveCI::davidson_correction(std::vector<Determinant>& P_dets,
                                                        psi::SharedVector P_evals,
                                                        psi::SharedMatrix PQ_evecs,
                                                        std::vector<Determinant>& PQ_dets,
                                                        psi::SharedVector PQ_evals) {
    outfile->Printf("\n  There are %zu PQ dets.", PQ_dets.size());
    outfile->Printf("\n  There are %zu P dets.", P_dets.size());

    // The energy correction per root
    std::vector<double> dc(nroot_, 0.0);

    std::unordered_map<Determinant, double, Determinant::Hash> PQ_map;
    for (int n = 0; n < nroot_; ++n) {

        // Build the map for each root
        for (size_t I = 0, max = PQ_dets.size(); I < max; ++I) {
            PQ_map[PQ_dets[I]] = PQ_evecs->get(I, n);
        }

        // Compute the sum of c^2 of all P space dets
        double c_sum = 0.0;
        for (auto& P : P_dets) {
            c_sum += PQ_map[P] * PQ_map[P];
        }
        c_sum = 1 - c_sum;
        outfile->Printf("\n c_sum : %1.12f", c_sum);
        dc[n] = c_sum * (PQ_evals->get(n) - P_evals->get(n));
    }
    return dc;
}

void AdaptiveCI::print_nos() {
    print_h2("NATURAL ORBITALS");

    std::shared_ptr<psi::Matrix> opdm_a(new psi::Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
    std::shared_ptr<psi::Matrix> opdm_b(new psi::Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));

    int offset = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v, ordm_a_.data()[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v, ordm_b_.data()[(u + offset) * nact_ + v + offset]);
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

    // std::ofstream file;
    // file.open("nos.txt",std::ios_base::app);
    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            auto irrep_occ =
                std::make_pair(OCC_A->get(h, u) + OCC_B->get(h, u), std::make_pair(h, u + 1));
            vec_irrep_occupation.push_back(irrep_occ);
            //          file << OCC_A->get(h, u) + OCC_B->get(h, u) << "  ";
        }
    }
    // file << endl;
    // file.close();

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
    outfile->Printf("\n\n");

    // Compute active space weights
    if (print_weights_) {
        double no_thresh = options_->get_double("ACI_NO_THRESHOLD");

        std::vector<int> active(nirrep_, 0);
        std::vector<std::vector<int>> active_idx(nirrep_);
        std::vector<int> docc(nirrep_, 0);

        print_h2("Active Space Weights");
        for (size_t h = 0; h < nirrep_; ++h) {
            std::vector<double> weights(nactpi_[h], 0.0);
            std::vector<double> oshell(nactpi_[h], 0.0);
            for (int p = 0; p < nactpi_[h]; ++p) {
                for (int q = 0; q < nactpi_[h]; ++q) {
                    double occ = OCC_A->get(h, q) + OCC_B->get(h, q);
                    if ((occ >= no_thresh) and (occ <= (2.0 - no_thresh))) {
                        weights[p] += (NO_A->get(h, p, q)) * (NO_A->get(h, p, q));
                        oshell[p] += (NO_A->get(h, p, q)) * (NO_A->get(h, p, q)) * (2 - occ) * occ;
                    }
                }
            }

            outfile->Printf("\n  Irrep %d:", h);
            outfile->Printf("\n  Active idx     MO idx        Weight         OS-Weight");
            outfile->Printf("\n ------------   --------   -------------    -------------");
            for (int w = 0; w < nactpi_[h]; ++w) {
                outfile->Printf("\n      %0.2d           %d       %1.9f      %1.9f", w + 1,
                                w + frzcpi_[h] + 1, weights[w], oshell[w]);
                if (weights[w] >= 0.9) {
                    active[h]++;
                    active_idx[h].push_back(w + frzcpi_[h] + 1);
                }
            }
        }
    }
}

/*
void AdaptiveCI::convert_to_string(const std::vector<Determinant>& space) {
    size_t space_size = space.size();
    size_t nalfa_str = 0;
    size_t nbeta_str = 0;

    alfa_list_.clear();
    beta_list_.clear();

    a_to_b_.clear();
    b_to_a_.clear();

    string_hash<size_t> alfa_map;
    string_hash<size_t> beta_map;

    for (size_t I = 0; I < space_size; ++I) {

        Determinant det = space[I];
        STLBitsetString alfa;
        STLBitsetString beta;

        alfa.set_nmo(ncmo_);
        beta.set_nmo(ncmo_);

        for (int i = 0; i < ncmo_; ++i) {
            alfa.set_bit(i, det.get_alfa_bit(i));
            beta.set_bit(i, det.get_alfa_bit(i));
        }

        size_t a_id;
        size_t b_id;

        // Once we find a new alfa string, add it to the list
        string_hash<size_t>::iterator a_it = alfa_map.find(alfa);
        if (a_it == alfa_map.end()) {
            a_id = nalfa_str;
            alfa_map[alfa] = a_id;
            nalfa_str++;
        } else {
            a_id = a_it->second;
        }

        string_hash<size_t>::iterator b_it = beta_map.find(beta);
        if (b_it == beta_map.end()) {
            b_id = nbeta_str;
            beta_map[beta] = b_id;
            nbeta_str++;
        } else {
            b_id = b_it->second;
        }

        a_to_b_.resize(nalfa_str);
        b_to_a_.resize(nbeta_str);

        alfa_list_.resize(nalfa_str);
        beta_list_.resize(nbeta_str);

        alfa_list_[a_id] = alfa;
        beta_list_[b_id] = beta;

        a_to_b_[a_id].push_back(b_id);
        b_to_a_[b_id].push_back(a_id);
    }
}
*/

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
        outfile->Printf("\n  %s", det.str(nact_).c_str());
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
        outfile->Printf("\n  %s", det.str(nact_).c_str());
        outfile->Printf("\n  %s", detb.str(nact_).c_str());
        P_space_.add(det);
        P_space_.add(detb);
    }

    if (quiet_mode_) {
        sparse_solver_.set_print_details(false);
    }
    sparse_solver_.set_parallel(true);
    sparse_solver_.set_force_diag(options_->get_bool("FORCE_DIAG_METHOD"));
    sparse_solver_.set_e_convergence(options_->get_double("E_CONVERGENCE"));
    sparse_solver_.set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver_.set_spin_project(project_out_spin_contaminants_);
    //    sparse_solver.set_spin_project_full(project_out_spin_contaminants_);
    sparse_solver_.set_guess_dimension(options_->get_int("DL_GUESS_SIZE"));
    sparse_solver_.set_num_vecs(options_->get_int("N_GUESS_VEC"));
    sparse_solver_.set_sigma_method(options_->get_str("SIGMA_BUILD_TYPE"));
    sparse_solver_.set_spin_project_full(false);
    sparse_solver_.set_max_memory(options_->get_int("SIGMA_VECTOR_MAX_MEMORY"));

    // if (det_save_)
    //     det_list_.open("det_list.txt");

    if (streamline_qspace_ and !quiet_mode_)
        outfile->Printf("\n  Using streamlined Q-space builder.");

    // approx_rdm_ = false;
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
        print_h2(cycle_h);
        outfile->Printf("\n  Initial P space dimension: %zu", P_space_.size());
    }

    // Check that the initial space is spin-complete
    if (spin_complete_) {
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
        sparse_solver_.set_root_project(true);
        add_bad_roots(P_space_);
        sparse_solver_.add_bad_states(bad_roots_);
    }

    if (sparse_solver_.sigma_method_ == "HZ") {
        op_.clear_op_lists();
        op_.clear_tp_lists();
        op_.build_strings(P_space_);
        op_.op_lists(P_space_);
        op_.tp_lists(P_space_);
    } else if (diag_method_ != Dynamic) {
        op_.clear_op_s_lists();
        op_.clear_tp_s_lists();
        op_.build_strings(P_space_);
        op_.op_s_lists(P_space_);
        op_.tp_s_lists(P_space_);
    }

    sparse_solver_.manual_guess(false);
    local_timer diag;
    sparse_solver_.diagonalize_hamiltonian_map(P_space_, op_, P_evals_, P_evecs_, num_ref_roots_,
                                               multiplicity_, diag_method_);
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
                            "%.12f Eh = %8.4f eV",
                            i, abs_energy, exc_energy);
        }
        outfile->Printf("\n");
    }

    if (!quiet_mode_ and options_->get_bool("ACI_PRINT_REFS"))
        print_wfn(P_space_, op_, P_evecs_, num_ref_roots_);
}

void AdaptiveCI::find_q_space() {
    // Step 2. Find determinants in the Q space
    local_timer build_space;
    if (options_->get_bool("ACI_BATCHED_SCREENING")) {
        find_q_space_batched(P_space_, PQ_space_, P_evals_, P_evecs_);
    } else if (streamline_qspace_) {
        default_find_q_space(P_space_, PQ_space_, P_evals_, P_evecs_);
    } else {
        find_q_space_multiroot(P_space_, PQ_space_, num_ref_roots_, P_evals_, P_evecs_);
    }
    outfile->Printf("\n  Time spent building the model space: %1.6f", build_space.get());
    // Check if P+Q space is spin complete
    if (spin_complete_) {
        PQ_space_.make_spin_complete(nact_); // <- xsize
        if (!quiet_mode_)
            outfile->Printf("\n  Spin-complete dimension of the PQ space: %zu", PQ_space_.size());
    }

    if ((ex_alg_ == "ROOT_ORTHOGONALIZE") and (root_ > 0) and cycle_ >= pre_iter_) {
        sparse_solver_.set_root_project(true);
        add_bad_roots(PQ_space_);
        sparse_solver_.add_bad_states(bad_roots_);
    }
}

void AdaptiveCI::diagonalize_PQ_space() {
    // Step 3. Diagonalize the Hamiltonian in the P + Q space
    if (sparse_solver_.sigma_method_ == "HZ") {
        op_.clear_op_lists();
        op_.clear_tp_lists();
        local_timer str;
        op_.build_strings(PQ_space_);
        outfile->Printf("\n  Time spent building strings      %1.6f s", str.get());
        op_.op_lists(PQ_space_);
        op_.tp_lists(PQ_space_);
    } else if (diag_method_ != Dynamic) {
        op_.clear_op_s_lists();
        op_.clear_tp_s_lists();
        op_.build_strings(PQ_space_);
        op_.op_s_lists(PQ_space_);
        op_.tp_s_lists(PQ_space_);
    }
    local_timer diag_pq;

    sparse_solver_.diagonalize_hamiltonian_map(PQ_space_, op_, PQ_evals_, PQ_evecs_, num_ref_roots_,
                                               multiplicity_, diag_method_);

    if (!quiet_mode_)
        outfile->Printf("\n  Total time spent diagonalizing H:   %1.6f s", diag_pq.get());

    // Save the solutions for the next iteration
    //        old_dets.clear();
    //        old_dets = PQ_space_;
    //        old_evecs = PQ_evecs->clone();

    if (!quiet_mode_) {
        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots_; ++i) {
            double abs_energy =
                PQ_evals_->get(i) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
            double exc_energy = pc_hartree2ev * (PQ_evals_->get(i) - PQ_evals_->get(0));
            outfile->Printf("\n    PQ-space CI Energy Root %3d        = "
                            "%.12f Eh = %8.4f eV",
                            i, abs_energy, exc_energy);
            outfile->Printf("\n    PQ-space CI Energy + EPT2 Root %3d = %.12f Eh = "
                            "%8.4f eV",
                            i, abs_energy + multistate_pt2_energy_correction_[i],
                            exc_energy +
                                pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
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
    prune_q_space(PQ_space_, P_space_, PQ_evecs_, num_ref_roots_);

    // Print information about the wave function
    if (!quiet_mode_) {
        print_wfn(PQ_space_, op_, PQ_evecs_, num_ref_roots_);
        outfile->Printf("\n  Cycle %d took: %1.6f s", cycle_, cycle_time_.get());
    }
}

std::vector<std::pair<size_t, double>>
AdaptiveCI::dl_initial_guess(std::vector<Determinant>& old_dets, std::vector<Determinant>& dets,
                                 psi::SharedMatrix& evecs, int root) {
    std::vector<std::pair<size_t, double>> guess;

    // Build a hash of new dets
    det_hash<size_t> detmap;
    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        detmap[dets[I]] = I;
    }

    // Loop through old dets, store index of old det
    for (size_t I = 0, max_I = old_dets.size(); I < max_I; ++I) {
        Determinant& det = old_dets[I];
        if (detmap.count(det) != 0) {
            guess.push_back(std::make_pair(detmap[det], evecs->get(I, root)));
        }
    }
    return guess;
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

DeterminantHashVec AdaptiveCI::approximate_wfn(DeterminantHashVec& PQ_space,
                                                   psi::SharedMatrix& evecs,
                                                   psi::SharedVector& evals,
                                                   psi::SharedMatrix& new_evecs) {
    DeterminantHashVec new_wfn;
    new_wfn.copy(PQ_space);

    det_hash<std::vector<double>> external_space;
    get_excited_determinants(1, evecs, PQ_space, external_space);

    size_t n_ref = PQ_space.size();
    size_t n_external = external_space.size();
    size_t total_size = n_ref + n_external;

    outfile->Printf("\n  Size of external space: %zu", n_external);
    new_evecs.reset(new psi::Matrix("U", total_size, 1));
    double sum = 0.0;

    for (size_t I = 0; I < n_ref; ++I) {
        double val = evecs->get(I, 0);
        new_evecs->set(I, 0, val);
        sum += val * val;
    }

    double E0 = evals->get(0);
    for (auto& I : external_space) {
        new_wfn.add(I.first);
        double val = I.second[0] / (E0 - as_ints_->energy(I.first));
        new_evecs->set(new_wfn.get_idx(I.first), 0, val);
        sum += val * val;
    }

    outfile->Printf("\n  Norm of approximate wfn: %1.12f", std::sqrt(sum));
    // Normalize new evecs
    sum = 1.0 / std::sqrt(sum);
    new_evecs->scale_column(0, 0, sum);

    return new_wfn;
}

void AdaptiveCI::compute_nos() {

    print_h2("ACI NO Transformation");

    psi::Dimension nmopi = mo_space_info_->get_dimension("ALL");
    psi::Dimension ncmopi = mo_space_info_->get_dimension("CORRELATED");
    psi::Dimension fdocc = mo_space_info_->get_dimension("FROZEN_DOCC");
    psi::Dimension rdocc = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    psi::Dimension ruocc = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    std::shared_ptr<psi::Matrix> opdm_a(new psi::Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
    std::shared_ptr<psi::Matrix> opdm_b(new psi::Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));

    int offset = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v, ordm_a_.data()[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v, ordm_b_.data()[(u + offset) * nact_ + v + offset]);
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

    // Build full transformation matrices from e-vecs
    psi::SharedMatrix Ua = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    psi::SharedMatrix Ub = std::make_shared<psi::Matrix>("Ub", nmopi, nmopi);

    Ua->identity();
    Ub->identity();

    for (size_t h = 0; h < nirrep_; ++h) {
        size_t irrep_offset = 0;

        // Frozen core and Restricted docc are unchanged
        irrep_offset += fdocc[h] + rdocc[h];
        // Only change the active block
        for (int p = 0; p < nactpi_[h]; ++p) {
            for (int q = 0; q < nactpi_[h]; ++q) {
                Ua->set(h, p + irrep_offset, q + irrep_offset, NO_A->get(h, p, q));
                Ub->set(h, p + irrep_offset, q + irrep_offset, NO_B->get(h, p, q));
            }
        }
    }
    // Retransform the integrals in the new basis
    as_ints_->ints()->rotate_orbitals(Ua, Ub);
}

void AdaptiveCI::upcast_reference(DeterminantHashVec& ref) {
    psi::Dimension act_dim = mo_space_info_->get_dimension("ACTIVE");
    psi::Dimension corr_dim = mo_space_info_->get_dimension("CORRELATED");
    psi::Dimension core_dim = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    psi::Dimension vir_dim = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    size_t nact = mo_space_info_->size("ACTIVE");
    size_t ncmo = mo_space_info_->size("CORRELATED");
    outfile->Printf("\n  Upcasting reference from %d orbitals to %d orbitals", nact, ncmo);

    det_hashvec ref_dets;
    ref_dets.swap(ref.wfn_hash());
    ref.clear();

    size_t ndet = ref_dets.size();

    for (size_t I = 0; I < ndet; ++I) {
        int offset = 0;
        int act_offset = 0;
        const Determinant& old_det = ref_dets[I];
        Determinant new_det(old_det);
        for (size_t h = 0; h < nirrep_; ++h) {

            // fill the rdocc orbitals with electrons
            for (int i = 0; i < core_dim[h]; ++i) {
                new_det.set_alfa_bit(i + offset, true);
                new_det.set_beta_bit(i + offset, true);
            }
            offset += core_dim[h];

            // Copy active occupation
            for (int p = 0; p < act_dim[h]; ++p) {
                new_det.set_alfa_bit(p + offset, old_det.get_alfa_bit(p + act_offset));
                new_det.set_beta_bit(p + offset, old_det.get_beta_bit(p + act_offset));
            }
            offset += act_dim[h] + vir_dim[h];
            act_offset += act_dim[h];
        }
        ref.add(new_det);
    }
}

void AdaptiveCI::spin_analysis() {
    size_t nact = static_cast<unsigned long>(nact_);
    size_t nact2 = nact * nact;
    size_t nact3 = nact * nact2;

    // First build rdms as ambit tensors
    // ambit::Tensor L1a = ambit::Tensor::build(ambit::CoreTensor, "L1a", {nact, nact});
    // ambit::Tensor L1b = ambit::Tensor::build(ambit::CoreTensor, "L1b", {nact, nact});
    // ambit::Tensor L2aa = ambit::Tensor::build(ambit::CoreTensor, "L2aa", {nact, nact, nact,
    // nact});
    // ambit::Tensor L2ab = ambit::Tensor::build(ambit::CoreTensor, "L2ab", {nact, nact, nact,
    // nact});
    // ambit::Tensor L2bb = ambit::Tensor::build(ambit::CoreTensor, "L2bb", {nact, nact, nact,
    // nact});

    psi::SharedMatrix UA(new psi::Matrix(nact, nact));
    psi::SharedMatrix UB(new psi::Matrix(nact, nact));

    if (options_->get_str("SPIN_BASIS") == "IAO") {

        // outfile->Printf("\n  Computing spin correlation in IAO basis \n");
        // psi::SharedMatrix Ca = ints_->Ca();
        // std::shared_ptr<IAOBuilder> IAO =
        //     IAOBuilder::build(reference_wavefunction_->basisset(),
        //                       reference_wavefunction_->get_basisset("MINAO_BASIS"), Ca,
        //                       options_->;
        // outfile->Printf("\n  Computing IAOs\n");
        // std::map<std::string, psi::SharedMatrix> iao_info = IAO->build_iaos();
        // psi::SharedMatrix iao_orbs(iao_info["A"]->clone());

        // psi::SharedMatrix Cainv(Ca->clone());
        // Cainv->invert();
        // psi::SharedMatrix iao_coeffs = psi::Matrix::doublet(Cainv, iao_orbs, false, false);

        // size_t new_dim = iao_orbs->colspi()[0];

        // auto labels = IAO->print_IAO(iao_orbs, new_dim, nmo_, reference_wavefunction_);
        // std::vector<int> IAO_inds;
        // if (options_->get_bool("PI_ACTIVE_SPACE")) {
        //     for (size_t i = 0, maxi = labels.size(); i < maxi; ++i) {
        //         std::string label = labels[i];
        //         if (label.find("z") != std::string::npos) {
        //             IAO_inds.push_back(i);
        //         }
        //     }
        // } else {
        //     nact = new_dim;
        //     for (size_t i = 0; i < new_dim; ++i) {
        //         IAO_inds.push_back(i);
        //     }
        // }

        // std::vector<size_t> active_mo = mo_space_info_->get_absolute_mo("ACTIVE");
        // for (size_t i = 0; i < nact; ++i) {
        //     int idx = IAO_inds[i];
        //     outfile->Printf("\n Using IAO %d", idx);
        //     for (size_t j = 0; j < nact; ++j) {
        //         int mo = active_mo[j];
        //         UA->set(j, i, iao_coeffs->get(mo, idx));
        //     }
        // }
        // UB->copy(UA);
        // outfile->Printf("\n");

    } else if (options_->get_str("SPIN_BASIS") == "NO") {

        outfile->Printf("\n  Computing spin correlation in NO basis \n");
        psi::SharedMatrix RDMa(new psi::Matrix(nact, nact));
        psi::SharedMatrix RDMb(new psi::Matrix(nact, nact));

        for (size_t i = 0; i < nact; ++i) {
            for (size_t j = 0; j < nact; ++j) {
                RDMa->set(i, j, ordm_a_.data()[i * nact + j]);
                RDMb->set(i, j, ordm_b_.data()[i * nact + j]);
            }
        }

        // psi::SharedMatrix NOa;
        // psi::SharedMatrix NOb;
        psi::SharedVector occa(new Vector(nact));
        psi::SharedVector occb(new Vector(nact));

        RDMa->diagonalize(UA, occa);
        RDMb->diagonalize(UB, occb);

        int nmo = mo_space_info_->size("ALL");
        psi::SharedMatrix Ua_full(new psi::Matrix(nmo, nmo));
        psi::SharedMatrix Ub_full(new psi::Matrix(nmo, nmo));

        Ua_full->identity();
        Ub_full->identity();

        auto actpi = mo_space_info_->get_absolute_mo("ACTIVE");
        auto nactpi = mo_space_info_->get_dimension("ACTIVE");
        for (size_t h = 0; h < nirrep_; ++h) {
            // skip frozen/restricted docc
            int nact = nactpi[h];
            for (int i = 0; i < nact; ++i) {
                for (int j = 0; j < nact; ++j) {
                    Ua_full->set(actpi[i], actpi[j], UA->get(i, j));
                    Ub_full->set(actpi[i], actpi[j], UB->get(i, j));
                }
            }
        }

        psi::SharedMatrix CA = as_ints_->ints()->Ca();
        psi::SharedMatrix CB = as_ints_->ints()->Cb();

        psi::SharedMatrix Ca_new = psi::Matrix::doublet(CA, Ua_full, false, false);
        psi::SharedMatrix Cb_new = psi::Matrix::doublet(CB, Ub_full, false, false);

        CA->copy(Ca_new);
        CB->copy(Cb_new);

    } else if (options_->get_str("SPIN_BASIS") == "LOCAL") {
        outfile->Printf("\n  Computing spin correlation in local basis \n");

        auto loc = std::make_shared<LOCALIZE>(options_, as_ints_->ints(), mo_space_info_);

        std::vector<size_t> actmo = mo_space_info_->get_absolute_mo("ACTIVE");
        std::vector<int> loc_mo(2);
        loc_mo[0] = static_cast<int>(actmo[0]);
        loc_mo[1] = static_cast<int>(actmo.back());
        loc->set_orbital_space(loc_mo);
        loc->compute_transformation();
        UA = loc->get_Ua()->clone();
        UB = loc->get_Ub()->clone();

    } else if (options_->get_str("SPIN_BASIS") == "CANONICAL") {
        outfile->Printf("\n  Computing spin correlation in reference basis \n");
        UA->identity();
        UB->identity();
    }

    ambit::Tensor Ua = ambit::Tensor::build(ambit::CoreTensor, "U", {nact, nact});
    ambit::Tensor Ub = ambit::Tensor::build(ambit::CoreTensor, "U", {nact, nact});
    Ua.iterate([&](const std::vector<size_t>& i, double& value) { value = UA->get(i[0], i[1]); });
    Ub.iterate([&](const std::vector<size_t>& i, double& value) { value = UB->get(i[0], i[1]); });

    //    new_dim = nact;
    // 1 rdms first
    ambit::Tensor L1aT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L1a", {nact, nact});
    ambit::Tensor L1bT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L1b", {nact, nact});

    L1aT("pq") = Ua("ap") * ordm_a_("ab") * Ua("bq");
    L1bT("pq") = Ub("ap") * ordm_b_("ab") * Ub("bq");
    // 2 rdms
    ambit::Tensor L2aaT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L2aa", {nact, nact, nact, nact});
    ambit::Tensor L2abT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L2ab", {nact, nact, nact, nact});
    ambit::Tensor L2bbT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L2bb", {nact, nact, nact, nact});

    L2aaT("pqrs") = Ua("ap") * Ua("bq") * trdm_aa_("abcd") * Ua("cr") * Ua("ds");
    L2abT("pqrs") = Ua("ap") * Ub("bq") * trdm_ab_("abcd") * Ua("cr") * Ub("ds");
    L2bbT("pqrs") = Ub("ap") * Ub("bq") * trdm_bb_("abcd") * Ub("cr") * Ub("ds");

    // Now form the spin correlation
    psi::SharedMatrix spin_corr(new psi::Matrix("Spin Correlation", nact, nact));
    psi::SharedMatrix spin_fluct(new psi::Matrix("Spin Fluctuation", nact, nact));
    psi::SharedMatrix spin_z(new psi::Matrix("Spin-z Correlation", nact, nact));

    std::vector<double> l1a(L1aT.data());
    std::vector<double> l1b(L1bT.data());
    std::vector<double> l2aa(L2aaT.data());
    std::vector<double> l2ab(L2abT.data());
    std::vector<double> l2bb(L2bbT.data());
    for (size_t i = 0; i < nact; ++i) {
        for (size_t j = 0; j < nact; ++j) {
            double value = (l2aa[i * nact3 + j * nact2 + i * nact + j] +
                            l2bb[i * nact3 + j * nact2 + i * nact + j] -
                            l2ab[i * nact3 + j * nact2 + i * nact + j] -
                            l2ab[j * nact3 + i * nact2 + j * nact + i]);
            if (i == j) {
                value += (l1a[nact * i + j] + l1b[nact * i + j]);
            }

            value +=
                (l1a[nact * i + i] - l1b[nact * i + i]) * (l1a[nact * j + j] - l1b[nact * j + j]);

            spin_z->set(i, j, value);
        }
    }

    for (size_t i = 0; i < nact; ++i) {
        for (size_t j = 0; j < nact; ++j) {
            double value = 0.0;
            if (i == j) {
                value += 0.75 * (l1a[nact * i + j] + l1b[nact * i + j]);
            }
            value -= 0.5 * (l2ab[i * nact3 + j * nact2 + j * nact + i] +
                            l2ab[j * nact3 + i * nact2 + i * nact + j]);

            value += 0.25 * (l2aa[i * nact3 + j * nact2 + i * nact + j] +
                             l2bb[i * nact3 + j * nact2 + i * nact + j] -
                             l2ab[i * nact3 + j * nact2 + i * nact + j] -
                             l2ab[j * nact3 + i * nact2 + j * nact + i]);

            spin_corr->set(i, j, value);
            value -=
                0.25 *
                (l1a[i * nact + i] * l1a[j * nact + j] + l1b[i * nact + i] * l1b[j * nact + j] -
                 l1a[i * nact + i] * l1b[j * nact + j] - l1b[i * nact + i] * l1a[j * nact + j]);
            spin_fluct->set(i, j, value);
        }
    }
    outfile->Printf("\n");
    // spin_corr->print();
    spin_fluct->print();
    spin_z->print();
    psi::SharedMatrix spin_evecs(new psi::Matrix(nact, nact));
    psi::SharedVector spin_evals(new Vector(nact));
    psi::SharedMatrix spin_evecs2(new psi::Matrix(nact, nact));
    psi::SharedVector spin_evals2(new Vector(nact));

    //    spin_corr->diagonalize(spin_evecs, spin_evals);
    //    spin_evals->print();

    if (options_->get_bool("SPIN_MAT_TO_FILE")) {
        std::ofstream file;
        file.open("spin_mat.txt", std::ofstream::out | std::ofstream::trunc);
        for (size_t i = 0; i < nact; ++i) {
            for (size_t j = 0; j < nact; ++j) {
                file << std::setw(12) << std::setprecision(6) << spin_corr->get(i, j) << " ";
            }
            file << "\n";
        }
        file.close();
        std::ofstream file2;
        file.open("spin_fluct.txt", std::ofstream::out | std::ofstream::trunc);
        for (size_t i = 0; i < nact; ++i) {
            for (size_t j = 0; j < nact; ++j) {
                file << std::setw(12) << std::setprecision(6) << spin_fluct->get(i, j) << " ";
            }
            file2 << "\n";
        }
        file2.close();
    }
    /*
        // Build spin-correlation densities
        psi::SharedMatrix Ca = reference_wavefunction_->Ca();
        psi::Dimension nactpi = mo_space_info_->get_dimension("ACTIVE");
        std::vector<size_t> actpi = mo_space_info_->get_absolute_mo("ACTIVE");
        psi::SharedMatrix Ca_copy = Ca->clone();
        for( int i = 0; i < nact; ++i ){
            psi::SharedVector vec = std::make_shared<Vector>(nmo_);
            vec->zero();
            for( int j = 0; j < nact; ++j ){
                auto col = Ca_copy->get_column(0,actpi[j]);
                double spin = spin_corr->get(j,i);
                for( int k = 0; k < nmo_; ++k ){
                    double val = col->get(k) * col->get(k);
                    col->set(k, val);
                }
                col->scale(spin);
                vec->add(col);
           }
            Ca->set_column(0,actpi[i], vec);
        }
    */
}

void AdaptiveCI::update_sigma() { sigma_ = options_->get_double("ACI_RELAX_SIGMA"); }

void AdaptiveCI::full_mrpt2() {
    if( options_->get_bool("FULL_MRPT2") ){
        MRPT2 pt(options_, as_ints_, mo_space_info_, PQ_space_, PQ_evecs_,
                 PQ_evals_, nroot_);
        std::vector<double> pt2 = pt.compute_energy();
        multistate_pt2_energy_correction_ = pt2;
    }
}

DeterminantHashVec AdaptiveCI::get_PQ_space() { return PQ_space_; }
psi::SharedMatrix AdaptiveCI::get_PQ_evecs() { return PQ_evecs_; }
psi::SharedVector AdaptiveCI::get_PQ_evals() { return PQ_evals_; }
WFNOperator AdaptiveCI::get_op() { return op_; }
size_t AdaptiveCI::get_ref_root() { return ref_root_; }
std::vector<double> AdaptiveCI::get_multistate_pt2_energy_correction() {
    return multistate_pt2_energy_correction_;
}

} // namespace forte
