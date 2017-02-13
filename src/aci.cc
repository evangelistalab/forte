/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

//#include <cmath>
//#include <functional>
//#include <algorithm>
//#include <unordered_map>
//#include <numeric>

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libpsio/psio.hpp"

#include "aci.h"
#include "ci_rdms.h"
#include "fci/fci_integrals.h"
#include "sparse_ci_solver.h"
#include "stl_bitset_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

bool pairComp(const std::pair<double, STLBitsetDeterminant> E1,
              const std::pair<double, STLBitsetDeterminant> E2) {
    return E1.first < E2.first;
}

AdaptiveCI::AdaptiveCI(SharedWavefunction ref_wfn, Options& options,
                       std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    op_.initialize(mo_space_info_);
    startup();
}

AdaptiveCI::~AdaptiveCI() {}

void AdaptiveCI::startup() {
    quiet_mode_ = false;
    if (options_["QUIET_MODE"].has_changed()) {
        quiet_mode_ = options_.get_bool("QUIET_MODE");
    }

    fci_ints_ = std::make_shared<FCIIntegrals>(
        ints_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
        mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));

    auto active_mo = mo_space_info_->get_corr_abs_mo("ACTIVE");
    ambit::Tensor tei_active_aa =
        ints_->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab =
        ints_->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb =
        ints_->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab,
                                    tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

    STLBitsetDeterminant::set_ints(fci_ints_);

    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();
    // Get wfn info
    wavefunction_symmetry_ = 0;
    if (options_["ROOT_SYM"].has_changed()) {
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }
    wavefunction_multiplicity_ = 1;
    if (options_["MULTIPLICITY"].has_changed()) {
        wavefunction_multiplicity_ = options_.get_int("MULTIPLICITY");
    }

    nact_ = mo_space_info_->size("ACTIVE");
    nactpi_ = mo_space_info_->get_dimension("ACTIVE");

    // Include frozen_docc and restricted_docc
    frzcpi_ = mo_space_info_->get_dimension("INACTIVE_DOCC");
    nfrzc_ = mo_space_info_->size("INACTIVE_DOCC");

    // "Correlated" includes restricted_docc
    ncmo_ = mo_space_info_->size("CORRELATED");

    // Number of correlated electrons
    nactel_ = 0;
    nalpha_ = 0;
    nbeta_ = 0;
    int nel = 0;
    for (int h = 0; h < nirrep_; ++h) {
        nel += 2 * doccpi_[h] + soccpi_[h];
    }

    int ms = wavefunction_multiplicity_ - 1;
    nactel_ = nel - 2 * nfrzc_;
    nalpha_ = (nactel_ + ms) / 2;
    nbeta_ = nactel_ - nalpha_;

    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");

    // Build the reference determinant and compute its energy
    reference_determinant_ = STLBitsetDeterminant(get_occupation());

    // Read options
    nroot_ = options_.get_int("NROOT");
    sigma_ = options_.get_double("SIGMA");
    gamma_ = options_.get_double("GAMMA");
    screen_thresh_ = options_.get_double("PRESCREEN_THRESHOLD");
    add_aimed_degenerate_ = options_.get_bool("ACI_ADD_AIMED_DEGENERATE");
    project_out_spin_contaminants_ =
        options_.get_bool("PROJECT_OUT_SPIN_CONTAMINANTS");
    spin_complete_ = options_.get_bool("ENFORCE_SPIN_COMPLETE");
    rdm_level_ = options_.get_int("ACI_MAX_RDM");

    max_cycle_ = 20;
    if (options_["MAX_ACI_CYCLE"].has_changed()) {
        max_cycle_ = options_.get_int("MAX_ACI_CYCLE");
    }
    pre_iter_ = 0;
    if (options_["ACI_PREITERATIONS"].has_changed()) {
        pre_iter_ = options_.get_int("ACI_PREITERATIONS");
    }

    spin_tol_ = options_.get_double("SPIN_TOL");
    // set the initial S^@ guess as input multiplicity
    int S = (wavefunction_multiplicity_ - 1.0) / 2.0;
    int S2 = wavefunction_multiplicity_ - 1.0;
    for (int n = 0; n < nroot_; ++n) {
        root_spin_vec_.push_back(make_pair(S, S2));
    }

    // get options for algorithm
    perturb_select_ = options_.get_bool("PERTURB_SELECT");
    pq_function_ = options_.get_str("PQ_FUNCTION");
    q_rel_ = options_.get_bool("Q_REL");
    q_reference_ = options_.get_str("Q_REFERENCE");
    ex_alg_ = options_.get_str("EXCITED_ALGORITHM");
    post_root_ = max(nroot_, options_.get_int("POST_ROOT"));
    post_diagonalize_ = options_.get_bool("POST_DIAGONALIZE");
    det_save_ = options_.get_bool("SAVE_DET_FILE");
    ref_root_ = options_.get_int("ROOT");
    root_ = options_.get_int("ROOT");

    reference_type_ = "SR";
    if (options_["ACI_INITIAL_SPACE"].has_changed()) {
        reference_type_ = options_.get_str("ACI_INITIAL_SPACE");
    }

    diag_method_ = DLString;
    if (options_["DIAG_ALGORITHM"].has_changed()) {
        if (options_.get_str("DIAG_ALGORITHM") == "FULL") {
            diag_method_ = Full;
        } else if (options_.get_str("DIAG_ALGORITHM") == "DLSTRING") {
            diag_method_ = DLString;
        } else if (options_.get_str("DIAG_ALGORITHM") == "SOLVER") {
            diag_method_ = DLSolver;
        }
    }
    aimed_selection_ = false;
    energy_selection_ = false;
    if (options_.get_str("SELECT_TYPE") == "AIMED_AMP") {
        aimed_selection_ = true;
        energy_selection_ = false;
    } else if (options_.get_str("SELECT_TYPE") == "AIMED_ENERGY") {
        aimed_selection_ = true;
        energy_selection_ = true;
    } else if (options_.get_str("SELECT_TYPE") == "ENERGY") {
        aimed_selection_ = false;
        energy_selection_ = true;
    } else if (options_.get_str("SELECT_TYPE") == "AMP") {
        aimed_selection_ = false;
        energy_selection_ = false;
    }

    if (options_.get_bool("STREAMLINE_Q") == true) {
        streamline_qspace_ = true;
    } else {
        streamline_qspace_ = false;
    }

    // Set streamline mode to true if possible
    if ((nroot_ == 1) and (aimed_selection_ == true) and
        (energy_selection_ == true) and (perturb_select_ == false)) {

        streamline_qspace_ = true;
    }
}

void AdaptiveCI::print_info() {

    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Multiplicity", wavefunction_multiplicity_},
        {"Symmetry", wavefunction_symmetry_},
        {"Number of roots", nroot_},
        {"Root used for properties", options_.get_int("ROOT")}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Sigma", sigma_},
        {"Gamma", gamma_},
        {"Convergence threshold", options_.get_double("E_CONVERGENCE")}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Determinant selection criterion", energy_selection_
                                                ? "Second-order Energy"
                                                : "First-order Coefficients"},
        {"Selection criterion",
         aimed_selection_ ? "Aimed selection" : "Threshold"},
        {"Excited Algorithm", options_.get_str("EXCITED_ALGORITHM")},
        //        {"Q Type", q_rel_ ? "Relative Energy" : "Absolute Energy"},
        //        {"PT2 Parameters", options_.get_bool("PERTURB_SELECT") ?
        //        "True" : "False"},
        {"Project out spin contaminants",
         project_out_spin_contaminants_ ? "True" : "False"},
        {"Enforce spin completeness of basis",
         spin_complete_ ? "True" : "False"},
        {"Enforce complete aimed selection",
         add_aimed_degenerate_ ? "True" : "False"}};

    // Print some information
    outfile->Printf("\n  ==> Calculation Information <==\n");
    outfile->Printf("\n  %s", string(65, '-').c_str());
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s %-5d", str_dim.first.c_str(),
                        str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %8.2e", str_dim.first.c_str(),
                        str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %s", str_dim.first.c_str(),
                        str_dim.second.c_str());
    }
    outfile->Printf("\n  %s", string(65, '-').c_str());
    outfile->Flush();
}

std::vector<int> AdaptiveCI::get_occupation() {

    std::vector<int> occupation(2 * nact_, 0);

    // Get reference type
    std::string ref_type = options_.get_str("REFERENCE");
    // if(!quiet_mode_) outfile->Printf("\n  Using %s reference.\n",
    // ref_type.c_str());

    // nyms denotes the number of electrons needed to assign symmetry and
    // multiplicity
    int nsym = wavefunction_multiplicity_ - 1;
    int orb_sym = wavefunction_symmetry_;

    if (wavefunction_multiplicity_ == 1) {
        nsym = 2;
    }

    // Grab an ordered list of orbital energies, sym labels, and idxs
    std::vector<std::tuple<double, int, int>> labeled_orb_en;
    std::vector<std::tuple<double, int, int>> labeled_orb_en_alfa;
    std::vector<std::tuple<double, int, int>> labeled_orb_en_beta;

    // For a restricted reference
    if (ref_type == "RHF" or ref_type == "RKS" or ref_type == "ROHF") {
        labeled_orb_en = sym_labeled_orbitals("RHF");

        // Build initial reference determinant from restricted reference
        for (int i = 0; i < nalpha_; ++i) {
            occupation[std::get<2>(labeled_orb_en[i])] = 1;
        }
        for (int i = 0; i < nbeta_; ++i) {
            occupation[nact_ + std::get<2>(labeled_orb_en[i])] = 1;
        }

        // Loop over as many outer-shell electrons as needed to get correct sym
        for (int k = 1; k <= nsym;) {

            bool add = false;
            // Remove electron from highest energy docc
            occupation[std::get<2>(labeled_orb_en[nalpha_ - k])] = 0;

            // Determine proper symmetry for new occupation
            orb_sym = wavefunction_symmetry_;

            if (wavefunction_multiplicity_ == 1) {
                orb_sym = std::get<1>(labeled_orb_en[nalpha_ - 1]) ^ orb_sym;
            } else {
                for (int i = 1; i <= nsym; ++i) {
                    orb_sym =
                        std::get<1>(labeled_orb_en[nalpha_ - i]) ^ orb_sym;
                }
                orb_sym = std::get<1>(labeled_orb_en[nalpha_ - k]) ^ orb_sym;
            }

            // Add electron to lowest-energy orbital of proper symmetry
            // Loop from current occupation to max MO until correct orbital is
            // reached
            for (int i = nalpha_ - k, maxi = nact_; i < maxi; ++i) {
                if (orb_sym == std::get<1>(labeled_orb_en[i]) and
                    occupation[std::get<2>(labeled_orb_en[i])] != 1) {
                    occupation[std::get<2>(labeled_orb_en[i])] = 1;
                    add = true;
                    break;
                } else {
                    continue;
                }
            }
            // If a new occupation could not be created, put electron back and
            // remove a different one
            if (!add) {
                occupation[std::get<2>(labeled_orb_en[nalpha_ - k])] = 1;
                ++k;
            } else {
                break;
            }

        } // End loop over k

    } else {
        labeled_orb_en_alfa = sym_labeled_orbitals("ALFA");
        labeled_orb_en_beta = sym_labeled_orbitals("BETA");

        // For an unrestricted reference
        // Make the reference
        // For singlets, this will be closed-shell

        for (int i = 0; i < nalpha_; ++i) {
            occupation[std::get<2>(labeled_orb_en_alfa[i])] = 1;
        }
        for (int i = 0; i < nbeta_; ++i) {
            occupation[std::get<2>(labeled_orb_en_beta[i]) + nact_] = 1;
        }

        if (nalpha_ >= nbeta_) {

            // Loop over k
            for (int k = 1; k < nsym;) {

                bool add = false;
                // Remove highest energy alpha electron
                occupation[std::get<2>(labeled_orb_en_alfa[nalpha_ - k])] = 0;

                // Determine proper symmetry for new electron

                orb_sym = wavefunction_symmetry_;

                if (wavefunction_multiplicity_ == 1) {
                    orb_sym =
                        std::get<1>(labeled_orb_en_alfa[nalpha_ - 1]) ^ orb_sym;
                } else {
                    for (int i = 1; i <= nsym; ++i) {
                        orb_sym =
                            std::get<1>(labeled_orb_en_alfa[nalpha_ - i]) ^
                            orb_sym;
                    }
                    orb_sym =
                        std::get<1>(labeled_orb_en_alfa[nalpha_ - k]) ^ orb_sym;
                }

                // Add electron to lowest-energy orbital of proper symmetry
                for (int i = nalpha_ - k; i < nactel_; ++i) {
                    if (orb_sym == std::get<1>(labeled_orb_en_alfa[i]) and
                        occupation[std::get<2>(labeled_orb_en_alfa[i])] != 1) {
                        occupation[std::get<2>(labeled_orb_en_alfa[i])] = 1;
                        add = true;
                        break;
                    } else {
                        continue;
                    }
                }

                // If a new occupation could not be made,
                // add electron back and try a different one

                if (!add) {
                    occupation[std::get<2>(labeled_orb_en_alfa[nalpha_ - k])] =
                        1;
                    ++k;
                } else {
                    break;
                }

            }    //	End loop over k
        } else { // End if(nalpha_ >= nbeta_ )

            for (int k = 1; k < nsym;) {

                bool add = false;

                // Remove highest-energy beta electron
                occupation[std::get<2>(labeled_orb_en_beta[nbeta_ - k])] = 0;

                // Determine proper symetry for new occupation
                orb_sym = wavefunction_symmetry_;

                if (wavefunction_multiplicity_ == 1) {
                    orb_sym =
                        std::get<1>(labeled_orb_en_beta[nbeta_ - 1]) ^ orb_sym;
                } else {
                    for (int i = 1; i <= nsym; ++i) {
                        orb_sym = std::get<1>(labeled_orb_en_beta[nbeta_ - i]) ^
                                  orb_sym;
                    }
                    orb_sym =
                        std::get<1>(labeled_orb_en_beta[nbeta_ - k]) ^ orb_sym;
                }

                // Add electron to lowest-energy beta orbital

                for (int i = nbeta_ - k; i < nactel_; ++i) {
                    if (orb_sym == std::get<1>(labeled_orb_en_beta[i]) and
                        occupation[std::get<2>(labeled_orb_en_beta[i])] != 1) {
                        occupation[std::get<2>(labeled_orb_en_beta[i])] = 1;
                        add = true;
                        break;
                    }
                }

                // If a new occupation could not be made,
                // replace the electron and try again

                if (!add) {
                    occupation[std::get<2>(labeled_orb_en_beta[nbeta_ - k])] =
                        1;
                    ++k;
                } else {
                    break;
                }

            } // End loop over k
        }     // End if nalpha_ < nbeta_
    }
    return occupation;
}

double AdaptiveCI::compute_energy() {
    if (!quiet_mode_) {
        print_method_banner({"Adaptive Configuration Interaction",
                             "written by Francesco A. Evangelista"});
        outfile->Printf("\n  ==> Reference Information <==\n");
        outfile->Printf("\n  There are %d frozen orbitals.", nfrzc_);
        outfile->Printf("\n  There are %zu active orbitals.\n", nact_);
        reference_determinant_.print();
        outfile->Printf("\n  REFERENCE ENERGY:         %1.12f",
                        reference_determinant_.energy() +
                            nuclear_repulsion_energy_ +
                            fci_ints_->scalar_energy());
        print_info();
    }

    if (ex_alg_ == "COMPOSITE") {
        ex_alg_ = "AVERAGE";
    }

    Timer aci_elapse;

    // The eigenvalues and eigenvectors
    SharedMatrix PQ_evecs;
    SharedVector PQ_evals;

    // Compute wavefunction and energy
    size_t dim;
    int nrun = 1;
    bool multi_state = false;

    if (options_.get_str("EXCITED_ALGORITHM") == "ROOT_COMBINE" or
        options_.get_str("EXCITED_ALGORITHM") == "MULTISTATE" or
        options_.get_str("EXCITED_ALGORITHM") == "ROOT_ORTHOGONALIZE") {
        nrun = nroot_;
        multi_state = true;
    }

    DeterminantMap full_space;
    std::vector<size_t> sizes(nroot_);
    SharedVector energies(new Vector(nroot_));
    std::vector<double> pt2_energies(nroot_);

    DeterminantMap PQ_space;

    for (int i = 0; i < nrun; ++i) {
        nroot_ = options_.get_int("NROOT");
        if (!quiet_mode_)
            outfile->Printf("\n  Computing wavefunction for root %d", i);

        if (multi_state) {
            ref_root_ = i;
            root_ = i;
        }

        compute_aci(PQ_space, PQ_evecs, PQ_evals);

        if (ex_alg_ == "ROOT_COMBINE") {
            sizes[i] = PQ_space.size();
            if (!quiet_mode_)
                outfile->Printf("\n  Combining determinant spaces");
            // Combine selected determinants into total space
            full_space.merge(PQ_space);
            PQ_space.clear();
        } else if ((ex_alg_ == "ROOT_ORTHOGONALIZE")) { // and i != (nrun - 1))
            // orthogonalize
            save_old_root(PQ_space, PQ_evecs, i);
            energies->set(i, PQ_evals->get(0));
            pt2_energies[i] = multistate_pt2_energy_correction_[0];
        } else if ((ex_alg_ == "MULTISTATE")) {
            // orthogonalize
            save_old_root(PQ_space, PQ_evecs, i);
            // compute_rdms( PQ_space_, PQ_evecs, i,i);
        }
        if (ex_alg_ == "ROOT_ORTHOGONALIZE") {
            root_ = i;
            wfn_analyzer(PQ_space, PQ_evecs, nroot_);
        }
    }
    dim = PQ_space.size();
    final_wfn_.copy(PQ_space);
    PQ_space.clear();

    int froot = options_.get_int("ROOT");
    if (ex_alg_ == "ROOT_ORTHOGONALIZE") {
        froot = nroot_ - 1;
        multistate_pt2_energy_correction_ = pt2_energies;
        PQ_evals = energies;
    }

    WFNOperator op_c(mo_space_info_);
    if (ex_alg_ == "ROOT_COMBINE") {
        outfile->Printf("\n\n  ==> Diagonalizing Final Space <==");
        dim = full_space.size();

        for (int n = 0; n < nroot_; ++n) {
            outfile->Printf("\n  Determinants for root %d: %zu", n, sizes[n]);
        }

        outfile->Printf("\n  Size of combined space: %zu", dim);

        op_c.op_lists(full_space);
        op_c.tp_lists(full_space);

        SparseCISolver sparse_solver;
        sparse_solver.set_parallel(true);
        sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
        sparse_solver.set_maxiter_davidson(
            options_.get_int("MAXITER_DAVIDSON"));
        sparse_solver.set_spin_project(project_out_spin_contaminants_);
        sparse_solver.set_force_diag(options_.get_bool("FORCE_DIAG_METHOD"));
        sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
        sparse_solver.diagonalize_hamiltonian_map(
            full_space, op_c, PQ_evals, PQ_evecs, nroot_,
            wavefunction_multiplicity_, diag_method_);
    }

    if (ex_alg_ == "MULTISTATE") {
        Timer multi;
        compute_multistate(PQ_evals);
        outfile->Printf("\n  Time spent computing multistate solution: %1.5f",
                        multi.get());
        //    PQ_evals->print();
    }

    // Compute the RDMs
    if (options_.get_int("ACI_MAX_RDM") >= 3 or (rdm_level_ >= 3)) {
        op_.three_lists(final_wfn_);
    }

    if (ex_alg_ == "ROOT_COMBINE") {
        compute_rdms(full_space, op_c, PQ_evecs, 0, 0);
    } else {
        compute_rdms(final_wfn_, op_, PQ_evecs, 0, 0);
    }

    if (!quiet_mode_) {
        if (ex_alg_ == "ROOT_COMBINE") {
            print_final(full_space, PQ_evecs, PQ_evals);
        } else if (ex_alg_ == "ROOT_ORTHOGONALIZE") {
            print_final(final_wfn_, PQ_evecs, energies);
        } else {
            print_final(final_wfn_, PQ_evecs, PQ_evals);
        }
    }

    outfile->Flush();
    //	std::vector<double> davidson;
    //	if(options_.get_str("SIZE_CORRECTION") == "DAVIDSON" ){
    //		davidson = davidson_correction( P_space_ , P_evals, PQ_evecs,
    // PQ_space_, PQ_evals );
    //	for( auto& i : davidson ){
    //		outfile->Printf("\n Davidson corr: %1.9f", i);
    //	}}

    double root_energy = PQ_evals->get(froot) + nuclear_repulsion_energy_ +
                         fci_ints_->scalar_energy();
    double root_energy_pt2 =
        root_energy + multistate_pt2_energy_correction_[froot];
    Process::environment.globals["CURRENT ENERGY"] = root_energy;
    Process::environment.globals["ACI ENERGY"] = root_energy;
    Process::environment.globals["ACI+PT2 ENERGY"] = root_energy_pt2;
    outfile->Printf("\n\n  %s: %f s", "Adaptive-CI (bitset) ran in ",
                    aci_elapse.get());
    outfile->Printf("\n\n  %s: %d", "Saving information for root",
                    options_.get_int("ROOT"));

    // printf( "\n%1.5f\n", aci_elapse.get());
    return PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_ +
           fci_ints_->scalar_energy();
}

void AdaptiveCI::print_final(DeterminantMap& dets, SharedMatrix& PQ_evecs,
                             SharedVector& PQ_evals) {
    size_t dim = dets.size();
    // Print a summary
    outfile->Printf("\n\n  ==> ACI Summary <==\n");

    outfile->Printf("\n  Iterations required:                         %zu",
                    cycle_);
    outfile->Printf("\n  Dimension of optimized determinant space:    %zu\n",
                    dim);
    if (nroot_ == 1) {
        outfile->Printf("\n  ACI(%.3f) Correlation energy: %.12f Eh", sigma_,
                        reference_determinant_.energy() -
                            PQ_evals->get(ref_root_));
    }

    for (int i = 0; i < nroot_; ++i) {
        double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_ +
                            fci_ints_->scalar_energy();
        double exc_energy =
            pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
        outfile->Printf(
            "\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV", i,
            abs_energy, exc_energy);
        outfile->Printf(
            "\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV", i,
            abs_energy + multistate_pt2_energy_correction_[i],
            exc_energy +
                pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                 multistate_pt2_energy_correction_[0]));
        //    	if(options_.get_str("SIZE_CORRECTION") == "DAVIDSON" ){
        //        outfile->Printf("\n  * Adaptive-CI Energy Root %3d + D1   =
        //        %.12f Eh = %8.4f eV",i,abs_energy + davidson[i],
        //                exc_energy + pc_hartree2ev * (davidson[i] -
        //                davidson[0]));
        //    	}
    }

    if (ex_alg_ == "ROOT_SELECT") {
        outfile->Printf("\n\n  Energy optimized for Root %d: %.12f", ref_root_,
                        PQ_evals->get(ref_root_) + nuclear_repulsion_energy_ +
                            fci_ints_->scalar_energy());
        outfile->Printf("\n\n  Root %d Energy + PT2:         %.12f", ref_root_,
                        PQ_evals->get(ref_root_) + nuclear_repulsion_energy_ +
                            fci_ints_->scalar_energy() +
                            multistate_pt2_energy_correction_[ref_root_]);
    }

    if (ex_alg_ != "ROOT_ORTHOGONALIZE") {
        outfile->Printf("\n\n  ==> Wavefunction Information <==");

        print_wfn(dets, PQ_evecs, nroot_);

        //    outfile->Printf("\n\n     Order		 # of Dets        Total
        //    |c^2|
        //    ");
        //    outfile->Printf(  "\n  __________ 	____________
        //    ________________ ");
        //    wfn_analyzer(dets, PQ_evecs, nroot_);
    }

    //   if(options_.get_bool("DETERMINANT_HISTORY")){
    //   	outfile->Printf("\n Det history (number,cycle,origin)");
    //   	size_t counter = 0;
    //   	for( auto &I : PQ_space_ ){
    //   		outfile->Printf("\n Det number : %zu", counter);
    //   		for( auto &n : det_history_[I]){
    //   			outfile->Printf("\n %zu	   %s", n.first,
    //   n.second.c_str());
    //   		}
    //   		++counter;
    //   	}
    //   }
}

void AdaptiveCI::default_find_q_space(DeterminantMap& P_space,
                                      DeterminantMap& PQ_space,
                                      SharedVector evals, SharedMatrix evecs) {
    Timer build;

    // This hash saves the determinant coupling to the model space eigenfunction
    det_hash<std::vector<double>> V_hash;

    // Get the excited Determinants
    get_excited_determinants(nroot_, evecs, P_space, V_hash);

    // This will contain all the determinants
    PQ_space.clear();

    // Add the P-space determinants and zero the hash
    det_hash<size_t> detmap = P_space.wfn_hash();
    for (det_hash<size_t>::iterator it = detmap.begin(), endit = detmap.end();
         it != endit; ++it) {
        PQ_space.add(it->first);
        V_hash.erase(it->first);
    }

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the SD space",
                        V_hash.size());
        outfile->Printf("\n  %s: %f s\n", "Time spent building the model space",
                        build.get());
    }
    outfile->Flush();

    Timer screen;

    // Compute criteria for all dets, store them all
    std::vector<std::pair<double, STLBitsetDeterminant>> sorted_dets;
    //    int ithread = omp_get_thread_num();
    //    int nthreads = omp_get_num_threads();

    for (const auto& I : V_hash) {
        // outfile->Printf("\n  %s     %1.8f", I.first.str().c_str(),
        // I.second[0]);

        //    if( (count % nthreads) != ithread ){
        //        count++;
        //        continue;
        //    }

        double delta = I.first.energy() - evals->get(0);
        double V = I.second[0];

        double criteria = 0.5 * (delta - sqrt(delta * delta + V * V * 4.0));
        sorted_dets.push_back(std::make_pair(std::fabs(criteria), I.first));
    }
    std::sort(sorted_dets.begin(), sorted_dets.end(), pairComp);
    std::vector<double> ept2(nroot_, 0.0);

    double sum = 0.0;
    size_t last_excluded = 0;
    for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I) {
        double energy = sorted_dets[I].first;
        if (sum + energy < sigma_) {
            sum += energy;
            ept2[0] -= energy;
            last_excluded = I;
        } else {
            PQ_space.add(sorted_dets[I].second);
        }
    }
    // printf( "\n last excluded : %zu \n", last_excluded );
    // printf( "\n sum : %1.12f \n", sum );
    // Add missing determinants
    if (add_aimed_degenerate_) {
        size_t num_extra = 0;
        for (size_t I = 0, max_I = last_excluded; I < max_I; ++I) {
            size_t J = last_excluded - I;
            if (std::fabs(sorted_dets[last_excluded + 1].first -
                          sorted_dets[J].first) < 1.0e-9) {
                PQ_space.add(sorted_dets[J].second);
                num_extra++;
            } else {
                break;
            }
        }
        if (num_extra > 0 and (!quiet_mode_)) {
            outfile->Printf(
                "\n  Added %zu missing determinants in aimed selection.",
                num_extra);
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants",
                        "Dimension of the P + Q space", PQ_space.size());
        outfile->Printf("\n  %s: %f s", "Time spent screening the model space",
                        screen.get());
    }
    outfile->Flush();
}

void AdaptiveCI::find_q_space(DeterminantMap& P_space, DeterminantMap& PQ_space,
                              int nroot, SharedVector evals,
                              SharedMatrix evecs) {
    Timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    det_hash<std::vector<double>> V_hash;
    get_excited_determinants(nroot_, evecs, P_space, V_hash);

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the SD space",
                        V_hash.size());
        outfile->Printf("\n  %s: %f s\n", "Time spent building the model space",
                        t_ms_build.get());
    }
    outfile->Flush();

    // This will contain all the determinants
    PQ_space.clear();

    // Add the P-space determinants and zero the hash
    det_hash<size_t> detmap = P_space.wfn_hash();
    for (det_hash<size_t>::iterator it = detmap.begin(), endit = detmap.end();
         it != endit; ++it) {
        PQ_space.add(it->first);
        V_hash.erase(it->first);
    }

    Timer t_ms_screen;

    std::vector<double> C1(nroot_, 0.0);
    std::vector<double> E2(nroot_, 0.0);
    std::vector<double> e2(nroot_, 0.0);
    std::vector<double> ept2(nroot_, 0.0);
    double criteria;
    std::vector<std::pair<double, STLBitsetDeterminant>> sorted_dets;

    // Define coupling out of loop, assume perturb_select_ = false
    std::function<double(double A, double B, double C)> C1_eq =
        [](double A, double B, double C) -> double {
        return 0.5 * ((B - C) - sqrt((B - C) * (B - C) + 4.0 * A * A)) / A;
    };

    std::function<double(double A, double B, double C)> E2_eq =
        [](double A, double B, double C) -> double {
        return 0.5 * ((B - C) - sqrt((B - C) * (B - C) + 4.0 * A * A));
    };

    if (perturb_select_) {
        C1_eq = [](double A, double B, double C) -> double {
            return -A / (B - C);
        };
        E2_eq = [](double A, double B, double C) -> double {
            return -A * A / (B - C);
        };
    }

    // Check the coupling between the reference and the SD space
    //    int ithread = omp_get_thread_num();
    //    int nthreads = omp_get_num_threads();
    for (const auto& it : V_hash) {
        //        if( (count % nthreads) != ithread ){
        //            count++;
        //            continue;
        //        }
        double EI = it.first.energy();
        for (int n = 0; n < nroot; ++n) {
            double V = it.second[n];
            double C1_I = C1_eq(V, EI, evals->get(n));
            double E2_I = E2_eq(V, EI, evals->get(n));

            C1[n] = std::fabs(C1_I);
            E2[n] = std::fabs(E2_I);

            e2[n] = E2_I;
        }

        if (ex_alg_ == "AVERAGE" and nroot > 1) {
            criteria = average_q_values(nroot, C1, E2);
        } else {
            criteria = root_select(nroot, C1, E2);
        }

        if (aimed_selection_) {
            sorted_dets.push_back(std::make_pair(criteria, it.first));
        } else {
            if (std::fabs(criteria) > sigma_) {
                PQ_space.add(it.first);
            } else {
                for (int n = 0; n < nroot; ++n) {
                    ept2[n] += e2[n];
                }
            }
        }
    } // end loop over determinants

    if (aimed_selection_) {
        // Sort the CI coefficients in ascending order
        std::sort(sorted_dets.begin(), sorted_dets.end(), pairComp);

        double sum = 0.0;
        size_t last_excluded = 0;
        for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I) {
            const STLBitsetDeterminant& det = sorted_dets[I].second;
            if (sum + sorted_dets[I].first < sigma_) {
                sum += sorted_dets[I].first;
                double EI = det.energy();
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
                if (std::fabs(sorted_dets[last_excluded + 1].first -
                              sorted_dets[J].first) < 1.0e-9) {
                    PQ_space.add(sorted_dets[J].second);
                    num_extra++;
                } else {
                    break;
                }
            }
            if (num_extra > 0 and (!quiet_mode_)) {
                outfile->Printf(
                    "\n  Added %zu missing determinants in aimed selection.",
                    num_extra);
            }
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants",
                        "Dimension of the P + Q space", PQ_space.size());
        outfile->Printf("\n  %s: %f s", "Time spent screening the model space",
                        t_ms_screen.get());
    }
    outfile->Flush();
}

double AdaptiveCI::average_q_values(int nroot, std::vector<double>& C1,
                                    std::vector<double>& E2) {
    // f_E2 and f_C1 will store the selected function of the chosen q criteria
    // This functions should only be called when nroot_ > 1

    int nav = options_.get_int("N_AVERAGE");
    int off = options_.get_int("AVERAGE_OFFSET");
    if (nav == 0)
        nav = nroot;
    if ((off + nav) > nroot)
        off = nroot - nav; // throw PSIEXCEPTION("\n  Your desired number of
                           // roots and the offset exceeds the maximum number of
                           // roots!");

    double f_C1 = 0.0;
    double f_E2 = 0.0;

    std::vector<double> dE2(nroot, 0.0);

    q_rel_ = options_.get_bool("Q_REL");

    if (q_rel_ == true and nroot > 1) {
        if (q_reference_ == "ADJACENT") {
            for (int n = 1; n < nroot; ++n) {
                dE2[n] = std::fabs(E2[n - 1] - E2[n]);
            }
        } else { // Default to "GS"
            for (int n = 1; n < nroot; ++n) {
                dE2[n] = std::fabs(E2[n] - E2[0]);
            }
        }
    } else if (q_rel_ == true and nroot == 1) {
        q_rel_ = false;
    }

    // Choose the function of the couplings for each root
    // If nroot = 1, choose the max

    if (pq_function_ == "MAX" or nroot == 1) {
        f_C1 = *std::max_element(C1.begin(), C1.end());
        f_E2 = (q_rel_ and (nroot != 1))
                   ? *std::max_element(dE2.begin(), dE2.end())
                   : *std::max_element(E2.begin(), E2.end());
    } else if (pq_function_ == "AVERAGE") {
        double C1_average = 0.0;
        double E2_average = 0.0;
        double dE2_average = 0.0;
        double dim_inv = 1.0 / nav;
        for (int n = 0; n < nav; ++n) {
            C1_average += C1[n + off] * dim_inv;
            E2_average += E2[n + off] * dim_inv;
        }
        if (q_rel_) {
            double inv = 1.0 / (nroot - 1.0);
            for (int n = 1; n < nroot; ++n) {
                dE2_average += dE2[n] * inv;
            }
        }
        f_C1 = C1_average;
        f_E2 = q_rel_ ? dE2_average : E2_average;
    }

    double select_value = 0.0;
    if (aimed_selection_) {
        select_value = energy_selection_ ? f_E2 : (f_C1 * f_C1);
    } else {
        select_value = energy_selection_ ? f_E2 : f_C1;
    }

    return select_value;
}

double AdaptiveCI::root_select(int nroot, std::vector<double>& C1,
                               std::vector<double>& E2) {
    double select_value;

    if (ref_root_ + 1 > nroot_) {
        throw PSIEXCEPTION(
            "\n  Your selection is not valid. Check ROOT in options.");
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

void AdaptiveCI::get_excited_determinants(
    int nroot, SharedMatrix evecs, DeterminantMap& P_space,
    det_hash<std::vector<double>>& V_hash) {
    size_t max_P = P_space.size();
    std::vector<STLBitsetDeterminant> P_dets = P_space.determinants();

// Loop over reference determinants
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0 and !quiet_mode_) {
            outfile->Printf("\n  Using %d threads.", omp_get_max_threads());
        }
        // This will store the excited determinant info for each thread
        std::vector<std::pair<STLBitsetDeterminant, std::vector<double>>>
            thread_ex_dets; //( noalpha * nvalpha  );

#pragma omp for schedule(guided)
        for (size_t P = 0; P < max_P; ++P) {
            STLBitsetDeterminant& det(P_dets[P]);

            std::vector<int> aocc = det.get_alfa_occ();
            std::vector<int> bocc = det.get_beta_occ();
            std::vector<int> avir = det.get_alfa_vir();
            std::vector<int> bvir = det.get_beta_vir();

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            STLBitsetDeterminant new_det(det);

            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = det.slater_rules_single_alpha(ii, aa);
                        if ((std::fabs(HIJ) * evecs->get_row(0, P)->norm() >=
                             screen_thresh_)) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * noalpha + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(
                                    std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }

            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = det.slater_rules_single_beta(ii, aa);
                        if ((std::fabs(HIJ) * evecs->get_row(0, P)->norm() >=
                             screen_thresh_)) {
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * nobeta + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(
                                    std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < noalpha; ++j) {
                    int jj = aocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = a + 1; b < nvalpha; ++b) {
                            int bb = avir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
                                 mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) *
                                         evecs->get_row(0, P)->norm() >=
                                     screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii,jj,aa,bb);

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot,
                                                                     0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] +=
                                                HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i *
                                        // noalpha*noalpha*nvalpha +
                                        // j*nvalpha*noalpha +  a*nvalpha + b ]
                                        // = std::make_pair(new_det,coupling);
                                        thread_ex_dets.push_back(
                                            std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate ab excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = 0; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
                                 mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) *
                                         evecs->get_row(0, P)->norm() >=
                                     screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii,jj,aa,bb);

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot,
                                                                     0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] +=
                                                HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i * nobeta * nvalpha
                                        // *nvbeta + j * bvalpha * nvbeta + a *
                                        // nvalpha]
                                        thread_ex_dets.push_back(
                                            std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate bb excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        for (int b = a + 1; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^
                                 (mo_symmetry_[jj] ^
                                  (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) ==
                                0) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) *
                                         evecs->get_row(0, P)->norm() >=
                                     screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii,jj,aa,bb);

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot,
                                                                     0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] +=
                                                HIJ * evecs->get(P, n);
                                        }
                                        thread_ex_dets.push_back(
                                            std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            for (size_t I = 0, maxI = thread_ex_dets.size(); I < maxI; ++I) {
                std::vector<double>& coupling = thread_ex_dets[I].second;
                STLBitsetDeterminant& det = thread_ex_dets[I].first;
                if (V_hash.count(det) != 0) {
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[det][n] += coupling[n];
                    }
                } else {
                    V_hash[det] = coupling;
                }
            }
        }
    } // Close threads
}

bool AdaptiveCI::check_convergence(
    std::vector<std::vector<double>>& energy_history, SharedVector evals) {
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
    std::vector<double> old_energies =
        energy_history[energy_history.size() - 1];
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
    return (std::fabs(new_avg_energy - old_avg_energy) <
            options_.get_double("ACI_CONVERGENCE"));
}

void AdaptiveCI::prune_q_space(DeterminantMap& PQ_space,
                               DeterminantMap& P_space, SharedMatrix evecs,
                               int nroot) {
    // Select the new reference space using the sorted CI coefficients
    P_space.clear();

    double tau_p = sigma_ * gamma_;

    int nav = options_.get_int("N_AVERAGE");
    int off = options_.get_int("AVERAGE_OFFSET");
    if (nav == 0)
        nav = nroot;

    //  if( options_.get_str("EXCITED_ALGORITHM") == "ROOT_COMBINE" and (nav ==
    //  1) and (nroot > 1)){
    //      off = ref_root_;
    //  }

    if ((off + nav) > nroot)
        off = nroot - nav; // throw PSIEXCEPTION("\n  Your desired number of
                           // roots and the offset exceeds the maximum number of
                           // roots!");

    // Create a vector that stores the absolute value of the CI coefficients
    std::vector<std::pair<double, STLBitsetDeterminant>> dm_det_list;
    // for (size_t I = 0, max = PQ_space.size(); I < max; ++I){
    det_hash<size_t> detmap = PQ_space.wfn_hash();
    for (det_hash<size_t>::iterator it = detmap.begin(), endit = detmap.end();
         it != endit; ++it) {
        double criteria = 0.0;
        if (ex_alg_ == "AVERAGE") {
            for (int n = 0; n < nav; ++n) {
                if (pq_function_ == "MAX") {
                    criteria = std::max(criteria,
                                        std::fabs(evecs->get(it->second, n)));
                } else if (pq_function_ == "AVERAGE") {
                    criteria += std::fabs(evecs->get(it->second, n + off));
                }
            }
            criteria /= static_cast<double>(nav);
        } else {
            criteria = std::fabs(evecs->get(it->second, ref_root_));
        }
        dm_det_list.push_back(std::make_pair(criteria, it->first));
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
                if (std::fabs(dm_det_list[last_excluded + 1].first -
                              dm_det_list[J].first) < 1.0e-9) {
                    P_space.add(dm_det_list[J].second);
                    num_extra += 1;
                } else {
                    break;
                }
            }
            if (num_extra > 0) {
                outfile->Printf(
                    "\n  Added %zu missing determinants in aimed selection.",
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

bool AdaptiveCI::check_stuck(std::vector<std::vector<double>>& energy_history,
                             SharedVector evals) {
    bool stuck = false;
    int nroot = evals->dim();
    if (cycle_ < 4) {
        stuck = false;
    } else {
        std::vector<double> av_energies;
        for (int i = 0; i < cycle_; ++i) {
            double energy = 0.0;
            for (int n = 0; n < nroot; ++n) {
                energy += energy_history[i][n];
            }
            energy /= static_cast<double>(nroot);
            av_energies.push_back(energy);
        }

        if (std::fabs(av_energies[cycle_ - 1] - av_energies[cycle_ - 3]) <
            options_.get_double("ACI_CONVERGENCE")) { // and
            //			std::fabs( av_energies[cycle_-2] - av_energies[cycle_ - 4]
            //)
            //< options_.get_double("ACI_CONVERGENCE") ){
            stuck = true;
        }
    }
    return stuck;
}

std::vector<std::pair<double, double>>
AdaptiveCI::compute_spin(DeterminantMap& space, SharedMatrix evecs, int nroot) {
    WFNOperator op(mo_space_info_);

    op.op_lists(space);
    op.tp_lists(space);

    std::vector<std::pair<double, double>> spin_vec(nroot);
    for (int n = 0; n < nroot_; ++n) {
        double S2 = op.s2(space, evecs, n);
        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        spin_vec[n] = make_pair(S, S2);
    }

    return spin_vec;
}

void AdaptiveCI::wfn_analyzer(DeterminantMap& det_space, SharedMatrix evecs,
                              int nroot) {

    std::vector<bool> occ(2 * nact_, 0);
    std::vector<std::tuple<double, int, int>> labeled_orb_en =
        sym_labeled_orbitals("RHF");
    for (int i = 0; i < nalpha_; ++i) {
        occ[std::get<2>(labeled_orb_en[i])] = 1;
    }
    for (int i = 0; i < nbeta_; ++i) {
        occ[nact_ + std::get<2>(labeled_orb_en[i])] = 1;
    }

    // bool print_final_wfn = options_.get_bool("SAVE_FINAL_WFN");

    // std::ofstream final_wfn;
    // if( print_final_wfn ){
    //     final_wfn.open("final_wfn_"+ std::to_string(root_) +  ".txt");
    //     final_wfn << det_space.size() << "  " << nact_ << "  " << nalpha_ <<
    //     "  " << nbeta_ << endl;
    // }

    STLBitsetDeterminant rdet(occ);
    auto ref_bits = rdet.bits();
    for (int n = 0; n < nroot; ++n) {
        std::vector<std::pair<size_t, double>> excitation_counter(
            1 + (1 + cycle_) * 2, std::make_pair(0, 0.0));

        det_hash<size_t> detmap = det_space.wfn_hash();
        for (det_hash<size_t>::iterator it = detmap.begin(),
                                        endit = detmap.end();
             it != endit; ++it) {
            int ndiff = 0;
            auto ex_bits = it->first.bits();

            double coeff =
                evecs->get(it->second, n) * evecs->get(it->second, n);

            // Compute number of differences in both alpha and beta strings wrt
            // ref
            for (size_t a = 0; a < nact_ * 2; ++a) {
                if (ref_bits[a] != ex_bits[a]) {
                    ++ndiff;
                }
            }
            ndiff /= 2;
            excitation_counter[ndiff] =
                std::make_pair(excitation_counter[ndiff].first + 1,
                               excitation_counter[ndiff].second + coeff);

            //         if( print_final_wfn and (n == ref_root_) ){

            //             auto abits =
            //             det_space[I].get_alfa_bits_vector_bool();
            //             auto bbits =
            //             det_space[I].get_beta_bits_vector_bool();

            //             final_wfn << std::setw(18) << std::setprecision(12)
            //             <<  evecs->get(I,n) << "  ";// <<  abits << "  " <<
            //             bbits << det_space[I].str().c_str() << endl;
            //             for( size_t i = 0; i < nact_; ++i ){
            //                 final_wfn << abits[i];
            //             }
            //             final_wfn << "   ";
            //             for( size_t i = 0; i < nact_; ++i ){
            //                 final_wfn << bbits[i];
            //             }
            //             final_wfn << endl;
            //         }
        }
        int order = 0;
        size_t det = 0;
        for (auto& i : excitation_counter) {
            outfile->Printf("\n        %d           %8zu           %.11f",
                            order, i.first, i.second);
            det += i.first;
            if (det == det_space.size())
                break;
            ++order;
        }
        outfile->Printf("\n\n  Highest-order excitation searched:     %zu  \n",
                        excitation_counter.size() - 1);
    }
    // if( print_final_wfn ) final_wfn.close();
    outfile->Flush();
}

std::vector<std::tuple<double, int, int>>
AdaptiveCI::sym_labeled_orbitals(std::string type) {
    std::vector<std::tuple<double, int, int>> labeled_orb;

    if (type == "RHF" or type == "ROHF" or type == "ALFA") {

        // Create a vector of orbital energy and index pairs
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (int a = 0; a < nactpi_[h]; ++a) {
                orb_e.push_back(
                    make_pair(epsilon_a_->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, symmetry, and idx
        for (size_t a = 0; a < nact_; ++a) {
            labeled_orb.push_back(
                make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    if (type == "BETA") {
        // Create a vector of orbital energies and index pairs
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (size_t a = 0, max = nactpi_[h]; a < max; ++a) {
                orb_e.push_back(
                    make_pair(epsilon_b_->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, sym, and idx
        for (size_t a = 0; a < nact_; ++a) {
            labeled_orb.push_back(
                make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    return labeled_orb;
}

void AdaptiveCI::print_wfn(DeterminantMap& space, SharedMatrix evecs,
                           int nroot) {
    std::string state_label;
    std::vector<string> s2_labels({"singlet", "doublet", "triplet", "quartet",
                                   "quintet", "sextet", "septet", "octet",
                                   "nonet", "decatet"});

    std::vector<std::pair<double, double>> spins =
        compute_spin(space, evecs, nroot);

    for (int n = 0; n < nroot; ++n) {
        DeterminantMap tmp;
        std::vector<double> tmp_evecs;

        outfile->Printf("\n\n  Most important contributions to root %3d:", n);

        size_t max_dets = std::min(10, evecs->nrow());
        tmp.subspace(space, evecs, tmp_evecs, max_dets, n);

        for (size_t I = 0; I < max_dets; ++I) {
            outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s", I, tmp_evecs[I],
                            tmp_evecs[I] * tmp_evecs[I],
                            space.get_idx(tmp.get_det(I)),
                            tmp.get_det(I).str().c_str());
        }

        state_label = s2_labels[std::round(spins[n].first * 2.0)];
        root_spin_vec_.clear();
        root_spin_vec_[n] = make_pair(spins[n].first, spins[n].second);
        outfile->Printf(
            "\n\n  Spin state for root %zu: S^2 = %5.6f, S = %5.3f, %s", n,
            root_spin_vec_[n].first, root_spin_vec_[n].second,
            state_label.c_str());
    }
    outfile->Flush();
}

void AdaptiveCI::full_spin_transform(DeterminantMap& det_space, SharedMatrix cI,
                                     int nroot) {
    //	Timer timer;
    //	outfile->Printf("\n  Performing spin projection...");
    //
    //	// Build the S^2 Matrix
    //	size_t det_size = det_space.size();
    //	SharedMatrix S2(new Matrix("S^2", det_size, det_size));
    //
    //	for(size_t I = 0; I < det_size; ++I ){
    //		for(size_t J = 0; J <= I; ++J){
    //			S2->set(I,J, det_space[I].spin2(det_space[J]) );
    //			S2->set(J,I, S2->get(I,J) );
    //		}
    //	}
    //
    //	//Diagonalize S^2, evals will be in ascending order
    //	SharedMatrix T(new Matrix("T", det_size, det_size));
    //	SharedVector evals(new Vector("evals", det_size));
    //	S2->diagonalize(T, evals);
    //
    //	//evals->print();
    //
    //	// Count the number of CSFs with correct spin
    //	// and get their indices wrt columns in T
    //	size_t csf_num = 0;
    //	size_t csf_idx = 0;
    //	double criteria = (0.25 * (wavefunction_multiplicity_ *
    // wavefunction_multiplicity_ - 1.0));
    //	//double criteria = static_cast<double>(wavefunction_multiplicity_) -
    // 1.0;
    //	for(size_t l = 0; l < det_size; ++l){
    //		if( std::fabs(evals->get(l) - criteria) <= 0.01 ){
    //			csf_num++;
    //		}else if( csf_num == 0 ){
    //			csf_idx++;
    //		}else{
    //			continue;
    //		}
    //	}
    //	outfile->Printf("\n  Number of CSFs: %zu", csf_num);
    //
    //	// Perform the transformation wrt csf eigenvectors
    //	// CHECK FOR TRIPLET (SHOULD INCLUDE CSF_IDX
    //	SharedMatrix C_trans(new Matrix("C_trans", det_size, nroot));
    //	SharedMatrix C(new Matrix("C", det_size, nroot));
    //	C->gemm('t','n',csf_num,nroot,det_size,1.0,T,det_size,cI,nroot,0.0,nroot);
    //	C_trans->gemm('n','n',det_size,nroot, csf_num,
    // 1.0,T,det_size,C,nroot,0.0,nroot);
    //
    //	//Normalize transformed vectors
    //	for( int n = 0; n < nroot; ++n ){
    //		double denom = 0.0;
    //		for( size_t I = 0; I < det_size; ++I){
    //			denom += C_trans->get(I,n) * C_trans->get(I,n);
    //		}
    //		denom = std::sqrt( 1.0/denom );
    //		C_trans->scale_column( 0, n, denom );
    //	}
    //	PQ_spin_evecs_.reset(new Matrix("PQ SPIN EVECS", det_size, nroot));
    //	PQ_spin_evecs_ = C_trans->clone();
    //
    //	outfile->Printf("\n  Time spent performing spin transformation: %6.6f",
    // timer.get());
    //	outfile->Flush();
}

double AdaptiveCI::compute_spin_contamination(DeterminantMap& space,
                                              SharedMatrix evecs, int nroot) {
    auto spins = compute_spin(space, evecs, nroot);
    double spin_contam = 0.0;
    for (int n = 0; n < nroot; ++n) {
        spin_contam += spins[n].second;
    }
    spin_contam /= static_cast<double>(nroot);
    spin_contam -=
        (0.25 *
         (wavefunction_multiplicity_ * wavefunction_multiplicity_ - 1.0));

    return spin_contam;
}

void AdaptiveCI::save_dets_to_file(DeterminantMap& space, SharedMatrix evecs) {
    // Use for single-root calculations only
    det_hash<size_t> detmap = space.wfn_hash();
    for (det_hash<size_t>::iterator it = detmap.begin(), endit = detmap.end();
         it != endit; ++it) {
        det_list_ << it->first.str().c_str() << " "
                  << fabs(evecs->get(it->second, 0)) << " ";
        //	for(size_t J = 0, maxJ = space.size(); J < maxJ; ++J){
        //		det_list_ << space[I].slater_rules(space[J]) << " ";
        //	}
        //	det_list_ << "\n";
    }
    det_list_ << "\n";
}

std::vector<double>
AdaptiveCI::davidson_correction(std::vector<STLBitsetDeterminant>& P_dets,
                                SharedVector P_evals, SharedMatrix PQ_evecs,
                                std::vector<STLBitsetDeterminant>& PQ_dets,
                                SharedVector PQ_evals) {
    outfile->Printf("\n  There are %zu PQ dets.", PQ_dets.size());
    outfile->Printf("\n  There are %zu P dets.", P_dets.size());

    // The energy correction per root
    std::vector<double> dc(nroot_, 0.0);

    std::unordered_map<STLBitsetDeterminant, double, STLBitsetDeterminant::Hash>
        PQ_map;
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

void AdaptiveCI::set_max_rdm(int rdm) { rdm_level_ = rdm; }

Reference AdaptiveCI::reference() {
    // const std::vector<STLBitsetDeterminant>& final_wfn =
    //     final_wfn_.determinants();
    CI_RDMS ci_rdms(options_, final_wfn_, fci_ints_, evecs_, 0, 0);
    ci_rdms.set_max_rdm(rdm_level_);
    Reference aci_ref =
        ci_rdms.reference(ordm_a_, ordm_b_, trdm_aa_, trdm_ab_, trdm_bb_,
                          trdm_aaa_, trdm_aab_, trdm_abb_, trdm_bbb_);
    return aci_ref;
}

void AdaptiveCI::print_nos() {
    print_h2("NATURAL ORBITALS");

    std::shared_ptr<Matrix> opdm_a(
        new Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
    std::shared_ptr<Matrix> opdm_b(
        new Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v,
                            ordm_a_[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v,
                            ordm_b_[(u + offset) * nact_ + v + offset]);
            }
        }
        offset += nactpi_[h];
    }
    SharedVector OCC_A(new Vector("ALPHA OCCUPATION", nirrep_, nactpi_));
    SharedVector OCC_B(new Vector("BETA OCCUPATION", nirrep_, nactpi_));
    SharedMatrix NO_A(new Matrix(nirrep_, nactpi_, nactpi_));
    SharedMatrix NO_B(new Matrix(nirrep_, nactpi_, nactpi_));

    opdm_a->diagonalize(NO_A, OCC_A, descending);
    opdm_b->diagonalize(NO_B, OCC_B, descending);

    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            auto irrep_occ = std::make_pair(OCC_A->get(h, u) + OCC_B->get(h, u),
                                            std::make_pair(h, u + 1));
            vec_irrep_occupation.push_back(irrep_occ);
        }
    }
    CharacterTable ct =
        Process::environment.molecule()->point_group()->char_table();
    std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
              std::greater<std::pair<double, std::pair<int, int>>>());

    size_t count = 0;
    outfile->Printf("\n    ");
    for (auto vec : vec_irrep_occupation) {
        outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second,
                        ct.gamma(vec.second.first).symbol(), vec.first);
        if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
            outfile->Printf("\n    ");
    }
    outfile->Printf("\n\n");

    // Compute active space weights
    double no_thresh = options_.get_double("NO_THRESHOLD");

    std::vector<int> active(nirrep_, 0);
    std::vector<std::vector<int>> active_idx(nirrep_);
    std::vector<int> docc(nirrep_, 0);

    print_h2("Active Space Weights");
    for (int h = 0; h < nirrep_; ++h) {
        std::vector<double> weights(nactpi_[h], 0.0);
        std::vector<double> oshell(nactpi_[h], 0.0);
        for (int p = 0; p < nactpi_[h]; ++p) {
            for (int q = 0; q < nactpi_[h]; ++q) {
                double occ = OCC_A->get(h, q) + OCC_B->get(h, q);
                if ((occ >= no_thresh) and (occ <= (2.0 - no_thresh))) {
                    weights[p] += (NO_A->get(h, p, q)) * (NO_A->get(h, p, q));
                    oshell[p] += (NO_A->get(h, p, q)) * (NO_A->get(h, p, q)) *
                                 (2 - occ) * occ;
                }
            }
        }

        outfile->Printf("\n  Irrep %d:", h);
        outfile->Printf(
            "\n  Active idx     MO idx        Weight         OS-Weight");
        outfile->Printf(
            "\n ------------   --------   -------------    -------------");
        for (int w = 0; w < nactpi_[h]; ++w) {
            outfile->Printf("\n      %0.2d           %d       %1.9f      %1.9f",
                            w + 1, w + frzcpi_[h] + 1, weights[w], oshell[w]);
            if (weights[w] >= 0.9) {
                active[h]++;
                active_idx[h].push_back(w + frzcpi_[h] + 1);
            }
        }
    }
}
// TODO: move to operator.cc
// void AdaptiveCI::compute_H_expectation_val( const
// std::vector<STLBitsetDeterminant>& space, SharedVector& evals, const
// SharedMatrix evecs, int nroot, DiagonalizationMethod diag_method)
//{
//    size_t space_size = space.size();
//    SparseCISolver ssolver;
//
//    evals->zero();
//
//    if( (space_size <= 200) or (diag_method == Full) ){
//        outfile->Printf("\n  Using full algorithm.");
//        SharedMatrix Hd = ssolver.build_full_hamiltonian( space );
//        for( int n = 0; n < nroot; ++n){
//            for( size_t I = 0; I < space_size; ++I){
//                for( size_t J = 0; J < space_size; ++J){
//                    evals->add(n, evecs->get(I,n) * Hd->get(I,J) *
//                    evecs->get(J,n) );
//                }
//            }
//        }
//    }else{
//        outfile->Printf("\n  Using sparse algorithm.");
//        auto Hs = ssolver.build_sparse_hamiltonian( space );
//        for( int n = 0; n < nroot; ++n){
//            for( size_t I = 0; I < space_size; ++I){
//                std::vector<double> H_val = Hs[I].second;
//                std::vector<int> Hidx = Hs[I].first;
//                for( size_t J = 0, max_J = H_val.size(); J < max_J; ++J){
//                    evals->add(n, evecs->get(I,n) * H_val[J] *
//                    evecs->get(Hidx[J],n) );
//                }
//            }
//        }
//    }
//}

void AdaptiveCI::convert_to_string(
    const std::vector<STLBitsetDeterminant>& space) {
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

        STLBitsetDeterminant det = space[I];
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

void AdaptiveCI::build_initial_reference(DeterminantMap& space) {
    STLBitsetDeterminant det = space.get_det(0);

    std::vector<int> aocc = det.get_alfa_occ();
    std::vector<int> bocc = det.get_beta_occ();
    std::vector<int> avir = det.get_alfa_vir();
    std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    if (reference_type_ == "CIS" or reference_type_ == "CISD") {
        // Generate alpha excitations
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    STLBitsetDeterminant ndet(det);
                    ndet.set_alfa_bit(ii, false);
                    ndet.set_alfa_bit(aa, true);
                    space.add(ndet);
                }
            }
        }
        // Generate beta excitations
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    STLBitsetDeterminant ndet(det);
                    ndet.set_beta_bit(ii, false);
                    ndet.set_beta_bit(aa, true);
                    space.add(ndet);
                }
            }
        }
    }

    if (reference_type_ == "CID" or reference_type_ == "CISD") {
        // Generate alpha-alpha excitations
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int j = i + 1; j < noalpha; ++j) {
                int jj = aocc[j];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    for (int b = a + 1; b < nvalpha; ++b) {
                        int bb = avir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
                             mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                            STLBitsetDeterminant new_det(det);
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(jj, false);
                            new_det.set_alfa_bit(aa, true);
                            new_det.set_alfa_bit(bb, true);
                            space.add(new_det);
                        }
                    }
                }
            }
        }
        // Then the alpha-beta
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int j = 0; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    for (int b = 0; b < nvbeta; ++b) {
                        int bb = bvir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
                             mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                            STLBitsetDeterminant new_det(det);
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_beta_bit(jj, false);
                            new_det.set_alfa_bit(aa, true);
                            new_det.set_beta_bit(bb, true);
                            space.add(new_det);
                        }
                    }
                }
            }
        }
        // Lastly the beta-beta
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int j = i + 1; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    for (int b = a + 1; b < nvbeta; ++b) {
                        int bb = bvir[b];
                        if ((mo_symmetry_[ii] ^
                             (mo_symmetry_[jj] ^
                              (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                            STLBitsetDeterminant new_det(det);
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(jj, false);
                            new_det.set_beta_bit(aa, true);
                            new_det.set_beta_bit(bb, true);
                            space.add(new_det);
                        }
                    }
                }
            }
        }
    }
}

int AdaptiveCI::root_follow(DeterminantMap& P_ref,
                            std::vector<double>& P_ref_evecs,
                            DeterminantMap& P_space, SharedMatrix P_evecs,
                            int num_ref_roots) {
    int ndets = P_space.size();
    int max_dim = std::min(ndets, 1000);
    //    int max_dim = ndets;
    int new_root;
    double old_overlap = 0.0;
    DeterminantMap P_int;
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
   //     outfile->Printf("\n %1.8f   %s", det_weights[I].first, P_ref.get_det(det_weights[I].second).str().c_str());
   // }

    for (int n = 0; n < num_ref_roots; ++n) {
        if (!quiet_mode_)
            outfile->Printf("\n\n  Computing overlap for root %d", n);
        double new_overlap = P_ref.overlap(P_ref_evecs, P_space, P_evecs, n);

        new_overlap = std::fabs(new_overlap);
        outfile->Printf("\n  Root %d has overlap %f", n, new_overlap);
        // If the overlap is larger, set it as the new root and reference, for
        // now
        if (new_overlap > old_overlap) {

            outfile->Printf("\n  Saving reference for root %d", n);
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

void AdaptiveCI::project_determinant_space(DeterminantMap& space,
                                           SharedMatrix evecs,
                                           SharedVector evals, int nroot) {
    //	double spin_contamination = compute_spin_contamination(space, evecs,
    // nroot);
    //	if(spin_contamination >= spin_tol_){
    //		if( !quiet_mode_ ) outfile->Printf("\n  Average spin contamination
    //per
    // root is %1.5f", spin_contamination);
    //		full_spin_transform(space, evecs, nroot);
    //		evecs->zero();
    //		evecs = PQ_spin_evecs_->clone();
    //        compute_H_expectation_val(space,evals,evecs,nroot,diag_method_);
    //	}else if (!quiet_mode_){
    //		outfile->Printf("\n  Average spin contamination (%1.5f) is less
    //than
    // tolerance (%1.5f)", spin_contamination, spin_tol_);
    //		outfile->Printf("\n  No need to perform spin projection.");
    //	}
}

void AdaptiveCI::compute_aci(DeterminantMap& PQ_space, SharedMatrix& PQ_evecs,
                             SharedVector& PQ_evals) {

    bool print_refs = false;
    bool multi_root = false;

    if (options_["FIRST_ITER_ROOTS"].has_changed()) {
        multi_root = options_.get_bool("FIRST_ITER_ROOTS");
    }

    if (options_["PRINT_REFS"].has_changed()) {
        print_refs = options_.get_bool("PRINT_REFS");
    }

    if ((options_.get_str("EXCITED_ALGORITHM") == "ROOT_ORTHOGONALIZE" or
         options_.get_str("EXCITED_ALGORITHM") == "MULTISTATE" or
         options_.get_str("EXCITED_ALGORITHM") == "ROOT_COMBINE") and
        root_ == 0 and !multi_root) {
        nroot_ = 1;
    }

    SharedMatrix P_evecs;
    SharedVector P_evals;

    // This will store part of the wavefunction for computing an overlap
    //    std::vector<std::pair<STLBitsetDeterminant,double>> P_ref;

    DeterminantMap P_ref;
    std::vector<double> P_ref_evecs;
    DeterminantMap P_space(reference_determinant_);

    // Use the reference determinant as a starting point
    //    P_space_map_[reference_determinant_] = 1;
    //    P_space.add(reference_determinant_);

    // det_history_[reference_determinant_].push_back(std::make_pair(0, "I"));

    if (reference_type_ != "SR") {
        build_initial_reference(P_space);
    }

    outfile->Flush();

    size_t nvec = options_.get_int("N_GUESS_VEC");

    std::vector<std::vector<double>> energy_history;
    SparseCISolver sparse_solver;
    if (quiet_mode_)
        sparse_solver.set_print_details(false);
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("MAXITER_DAVIDSON"));
    sparse_solver.set_spin_project(project_out_spin_contaminants_);
    sparse_solver.set_force_diag(options_.get_bool("FORCE_DIAG_METHOD"));
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_num_vecs(nvec);
    int spin_projection = options_.get_int("SPIN_PROJECTION");

    if (det_save_)
        det_list_.open("det_list.txt");

    if (streamline_qspace_ and !quiet_mode_)
        outfile->Printf("\n  Using streamlined Q-space builder.");

    ex_alg_ = options_.get_str("EXCITED_ALGORITHM");

    std::vector<STLBitsetDeterminant> old_dets;
    SharedMatrix old_evecs;

    if (options_.get_str("EXCITED_ALGORITHM") == "ROOT_SELECT") {
        ref_root_ = options_.get_int("ROOT");
    }

    int cycle;
    for (cycle = 0; cycle < max_cycle_; ++cycle) {
        Timer cycle_time;
        // Step 1. Diagonalize the Hamiltonian in the P space
        int num_ref_roots = std::min(nroot_, int(P_space.size()));
        cycle_ = cycle;
        std::string cycle_h = "Cycle " + std::to_string(cycle_);

        bool follow = false;
        if (options_.get_str("EXCITED_ALGORITHM") == "ROOT_SELECT" or
            options_.get_str("EXCITED_ALGORITHM") == "ROOT_COMBINE" or
            options_.get_str("EXCITED_ALGORITHM") == "MULTISTATE" or
            options_.get_str("EXCITED_ALGORITHM") == "ROOT_ORTHOGONALIZE") {

            follow = true;
        }

        if (!quiet_mode_) {
            print_h2(cycle_h);
            outfile->Printf("\n  Initial P space dimension: %zu",
                            P_space.size());
        }

        // Check that the initial space is spin-complete
        if (spin_complete_) {
            P_space.make_spin_complete();
            if (!quiet_mode_)
                outfile->Printf("\n  %s: %zu determinants",
                                "Spin-complete dimension of the P space",
                                P_space.size());
        } else if (!quiet_mode_) {
            outfile->Printf("\n  Not checking for spin-completeness.");
        }
        // Diagonalize H in the P space
        if (ex_alg_ == "ROOT_ORTHOGONALIZE" and root_ > 0 and
            cycle >= pre_iter_) {
            sparse_solver.set_root_project(true);
            add_bad_roots(P_space);
            sparse_solver.add_bad_states(bad_roots_);
        }

        // Grab and set the guess
        //    if( cycle > 2 and nroot_ == 1){
        //       for( int n = 0; n < num_ref_roots; ++n ){
        //           auto guess = dl_initial_guess( old_dets, P_space_,
        //           old_evecs, ref_root_ );
        //            outfile->Printf("\n  Setting guess");
        //           sparse_solver.set_initial_guess( guess );
        //        }
        //    }

        op_.clear_op_lists();
        op_.clear_tp_lists();
        op_.op_lists(P_space);
        op_.tp_lists(P_space);
        sparse_solver.manual_guess(false);
        Timer diag;
        sparse_solver.diagonalize_hamiltonian_map(
            P_space, op_, P_evals, P_evecs, num_ref_roots,
            wavefunction_multiplicity_, diag_method_);
        if (!quiet_mode_)
            outfile->Printf("\n  Time spent diagonalizing H:   %1.6f s",
                            diag.get());

        if (cycle < pre_iter_) {
            ex_alg_ = "AVERAGE";
        } else if (cycle == pre_iter_ and follow) {
            ex_alg_ = options_.get_str("EXCITED_ALGORITHM");
        }
        // If doing root-following, grab the initial root
        //   if( follow and pre_iter_ == 0 ){
        //       P_ref_evecs.resize( P_space.size() );
        //       det_hash<size_t> detmap = P_space.wfn_hash();
        //       for( auto& det : detmap ){
        //           P_ref.add( det.first );
        //           P_ref_evecs[det.second] = P_evecs->get(det.second,
        //           ref_root_) ;
        //       }
        //   }

        if (follow and num_ref_roots > 1 and (cycle >= pre_iter_) and
            cycle > 0) {
            ref_root_ = root_follow(P_ref, P_ref_evecs, P_space, P_evecs,
                                    num_ref_roots);
        }

        // Save the dimension of the previous PQ space
        // size_t PQ_space_prev = PQ_space_.size();

        // Use spin projection to ensure the P space is spin pure
        if ((spin_projection == 1 or spin_projection == 3) and
            P_space.size() <= 200) {
            project_determinant_space(P_space, P_evecs, P_evals, num_ref_roots);
        }

        // Print the energy
        if (!quiet_mode_) {
            outfile->Printf("\n");
            for (int i = 0; i < num_ref_roots; ++i) {
                double abs_energy = P_evals->get(i) +
                                    nuclear_repulsion_energy_ +
                                    fci_ints_->scalar_energy();
                double exc_energy =
                    pc_hartree2ev * (P_evals->get(i) - P_evals->get(0));
                outfile->Printf("\n    P-space  CI Energy Root %3d        = "
                                "%.12f Eh = %8.4f eV",
                                i, abs_energy, exc_energy);
            }
            outfile->Printf("\n");
            outfile->Flush();
        }

        if (!quiet_mode_ and print_refs)
            print_wfn(P_space, P_evecs, num_ref_roots);

        // Step 2. Find determinants in the Q space

        if (streamline_qspace_) {
            default_find_q_space(P_space, PQ_space, P_evals, P_evecs);
            //            find_q_space(num_ref_roots, P_evals, P_evecs);
        } else {
            find_q_space(P_space, PQ_space, num_ref_roots, P_evals, P_evecs);
        }

        // Check if P+Q space is spin complete
        if (spin_complete_) {
            PQ_space.make_spin_complete();
            if (!quiet_mode_)
                outfile->Printf(
                    "\n  Spin-complete dimension of the PQ space: %zu",
                    PQ_space.size());
        }

        if ((options_.get_str("EXCITED_ALGORITHM") == "ROOT_ORTHOGONALIZE") and
            (root_ > 0) and cycle >= pre_iter_) {
            sparse_solver.set_root_project(true);
            add_bad_roots(PQ_space);
            sparse_solver.add_bad_states(bad_roots_);
        }

        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        op_.clear_op_lists();
        op_.clear_tp_lists();
        op_.op_lists(PQ_space);
        op_.tp_lists(PQ_space);
        Timer diag_pq;

        sparse_solver.diagonalize_hamiltonian_map(
            PQ_space, op_, PQ_evals, PQ_evecs, num_ref_roots,
            wavefunction_multiplicity_, diag_method_);

        if (!quiet_mode_)
            outfile->Printf("\n  Time spent diagonalizing H:   %1.6f s",
                            diag_pq.get());
        if (det_save_)
            save_dets_to_file(PQ_space, PQ_evecs);

        // Save the solutions for the next iteration
        //        old_dets.clear();
        //        old_dets = PQ_space_;
        //        old_evecs = PQ_evecs->clone();

        // Ensure the solutions are spin-pure
        if ((spin_projection == 1 or spin_projection == 3) and
            PQ_space.size() <= 200) {
            project_determinant_space(PQ_space, PQ_evecs, PQ_evals,
                                      num_ref_roots);
        }

        if (!quiet_mode_) {
            // Print the energy
            outfile->Printf("\n");
            for (int i = 0; i < num_ref_roots; ++i) {
                double abs_energy = PQ_evals->get(i) +
                                    nuclear_repulsion_energy_ +
                                    fci_ints_->scalar_energy();
                double exc_energy =
                    pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
                outfile->Printf("\n    PQ-space CI Energy Root %3d        = "
                                "%.12f Eh = %8.4f eV",
                                i, abs_energy, exc_energy);
                outfile->Printf(
                    "\n    PQ-space CI Energy + EPT2 Root %3d = %.12f Eh = "
                    "%8.4f eV",
                    i, abs_energy + multistate_pt2_energy_correction_[i],
                    exc_energy +
                        pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                         multistate_pt2_energy_correction_[0]));
            }
            outfile->Printf("\n");
            outfile->Flush();
        }
        // if(quiet_mode_){
        // 	double abs_energy = PQ_evals->get(0) + nuclear_repulsion_energy_
        // + fci_ints_->scalar_energy();
        //     outfile->Printf("\n    %2d               %zu
        //     %1.12f", cycle_, PQ_space_.size(), abs_energy );
        // }
        num_ref_roots = std::min(nroot_, int(PQ_space.size()));

        // If doing root-following, grab the initial root
        if (follow and
            (cycle == (pre_iter_ - 1) or (pre_iter_ == 0 and cycle == 0))) {

            if (options_.get_str("EXCITED_ALGORITHM") == "ROOT_SELECT") {
                ref_root_ = options_.get_int("ROOT");
            }
            size_t dim = std::min( static_cast<int>(PQ_space.size()) , 1000 );
            P_ref.subspace( PQ_space, PQ_evecs, P_ref_evecs, dim, ref_root_ );
        }

        // if( follow and num_ref_roots > 0 and (cycle >= (pre_iter_ - 1)) ){
        if (follow and num_ref_roots > 0 and (cycle >= pre_iter_)) {
            ref_root_ = root_follow(P_ref, P_ref_evecs, PQ_space, PQ_evecs,
                                    num_ref_roots);
        }

        bool stuck = check_stuck(energy_history, PQ_evals);
        if (stuck and (options_.get_str("EXCITED_ALGORITHM") != "COMPOSITE")) {
            outfile->Printf("\n  Procedure is stuck! Quitting...");
            break;
        } else if (stuck and
                   (options_.get_str("EXCITED_ALGORITHM") == "COMPOSITE") and
                   ex_alg_ == "AVERAGE") {
            outfile->Printf("\n  Root averaging algorithm converged.");
            outfile->Printf("\n  Now optimizing PQ Space for root %d",
                            options_.get_int("ROOT"));
            ex_alg_ = options_.get_str("EXCITED_ALGORITHM");
            pre_iter_ = cycle + 1;
        }

        // Step 4. Check convergence and break if needed
        bool converged = check_convergence(energy_history, PQ_evals);
        if (converged and (ex_alg_ == "AVERAGE") and
            options_.get_str("EXCITED_ALGORITHM") == "COMPOSITE") {
            outfile->Printf("\n  Root averaging algorithm converged.");
            outfile->Printf("\n  Now optimizing PQ Space for root %d",
                            options_.get_int("ROOT"));
            ex_alg_ = options_.get_str("EXCITED_ALGORITHM");
            pre_iter_ = cycle + 1;
        } else if (converged) {
            // if(quiet_mode_) outfile->Printf(
            // "\n----------------------------------------------------------" );
            if (!quiet_mode_)
                outfile->Printf("\n  ***** Calculation Converged *****");
            break;
        }

        // Step 5. Prune the P + Q space to get an updated P space
        prune_q_space(PQ_space, P_space, PQ_evecs, num_ref_roots);

        // Print information about the wave function
        if (!quiet_mode_) {
            print_wfn(PQ_space, PQ_evecs, num_ref_roots);
            outfile->Printf("\n  Cycle %d took: %1.6f s", cycle,
                            cycle_time.get());
        }

        ex_alg_ = options_.get_str("EXCITED_ALGORITHM");
    } // end iterations

    if (det_save_)
        det_list_.close();

    // Ensure the solutions are spin-pure
    if ((spin_projection == 2 or spin_projection == 3) and
        PQ_space.size() <= 200) {
        project_determinant_space(PQ_space, PQ_evecs, PQ_evals, nroot_);
    } else if (!quiet_mode_) {
        outfile->Printf("\n  Not performing spin projection.");
    }
}

std::vector<std::pair<size_t, double>>
AdaptiveCI::dl_initial_guess(std::vector<STLBitsetDeterminant>& old_dets,
                             std::vector<STLBitsetDeterminant>& dets,
                             SharedMatrix& evecs, int root) {
    std::vector<std::pair<size_t, double>> guess;

    // Build a hash of new dets
    det_hash<size_t> detmap;
    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        detmap[dets[I]] = I;
    }

    // Loop through old dets, store index of old det
    for (size_t I = 0, max_I = old_dets.size(); I < max_I; ++I) {
        STLBitsetDeterminant& det = old_dets[I];
        if (detmap.count(det) != 0) {
            guess.push_back(std::make_pair(detmap[det], evecs->get(I, root)));
        }
    }
    return guess;
}

void AdaptiveCI::compute_rdms(DeterminantMap& dets, WFNOperator& op,
                              SharedMatrix& PQ_evecs, int root1, int root2) {

    // std::vector<STLBitsetDeterminant> det_vec = dets.determinants();

    CI_RDMS ci_rdms_(options_, dets, fci_ints_, PQ_evecs, root1, root2);
    ci_rdms_.set_max_rdm(rdm_level_);
    if (rdm_level_ >= 1) {
        Timer one_r;
        ci_rdms_.compute_1rdm(ordm_a_, ordm_b_, op);
        if (!quiet_mode_)
            outfile->Printf("\n  1-RDM  took %2.6f s (determinant)",
                            one_r.get());

        if (options_.get_bool("PRINT_NO")) {
            print_nos();
        }
    }
    if (rdm_level_ >= 2) {
        Timer two_r;
        ci_rdms_.compute_2rdm(trdm_aa_, trdm_ab_, trdm_bb_, op);
        if (!quiet_mode_)
            outfile->Printf("\n  2-RDMS took %2.6f s (determinant)",
                            two_r.get());
    }
    if (rdm_level_ >= 3) {
        Timer tr;
        ci_rdms_.compute_3rdm(trdm_aaa_, trdm_aab_, trdm_abb_, trdm_bbb_, op);
        if (!quiet_mode_)
            outfile->Printf("\n  3-RDMs took %2.6f s (determinant)", tr.get());

        if (options_.get_bool("TEST_RDMS")) {
            ci_rdms_.rdm_test(ordm_a_, ordm_b_, trdm_aa_, trdm_bb_, trdm_ab_,
                              trdm_aaa_, trdm_aab_, trdm_abb_, trdm_bbb_);
        }
    }
}

void AdaptiveCI::add_bad_roots(DeterminantMap& dets) {
    bad_roots_.clear();

    // Look through each state, save common determinants/coeffs
    int nroot = old_roots_.size();
    size_t idx = dets.size();
    for (int i = 0; i < nroot; ++i) {

        std::vector<std::pair<size_t, double>> bad_root;
        size_t nadd = 0;
        std::vector<std::pair<STLBitsetDeterminant, double>>& state =
            old_roots_[i];

        for (size_t I = 0, max_I = state.size(); I < max_I; ++I) {
            if (dets.has_det(state[I].first)) {
                //                outfile->Printf("\n %zu, %f ", I,
                //                detmapper[state[I].first] , state[I].second );
                bad_root.push_back(std::make_pair(dets.get_idx(state[I].first),
                                                  state[I].second));
                nadd++;
            }
        }
        bad_roots_.push_back(bad_root);
        outfile->Printf("\n  Added %zu determinants from root %zu", nadd, i);
    }
}

void AdaptiveCI::save_old_root(DeterminantMap& dets, SharedMatrix& PQ_evecs,
                               int root) {
    std::vector<std::pair<STLBitsetDeterminant, double>> vec;
    outfile->Printf("\n  Saving root %d, ref_root is %d", root, ref_root_);
    det_hash<size_t> detmap = dets.wfn_hash();
    for (auto it = detmap.begin(), endit = detmap.end(); it != endit; ++it) {
        vec.push_back(
            std::make_pair(it->first, PQ_evecs->get(it->second, ref_root_)));
    }
    old_roots_.push_back(vec);
    outfile->Printf("\n  Number of old roots: %zu", old_roots_.size());
}

void AdaptiveCI::compute_multistate(SharedVector& PQ_evals) {
    outfile->Printf("\n  Computing multistate solution");
    int nroot = old_roots_.size();

    // Form the overlap matrix

    SharedMatrix S(new Matrix(nroot, nroot));
    S->identity();
    for (int A = 0; A < nroot; ++A) {
        std::vector<std::pair<STLBitsetDeterminant, double>>& stateA =
            old_roots_[A];
        size_t ndetA = stateA.size();
        for (int B = 0; B < nroot; ++B) {
            if (A == B)
                continue;
            std::vector<std::pair<STLBitsetDeterminant, double>>& stateB =
                old_roots_[B];
            size_t ndetB = stateB.size();
            double overlap = 0.0;

            for (size_t I = 0; I < ndetA; ++I) {
                STLBitsetDeterminant& detA = stateA[I].first;
                for (size_t J = 0; J < ndetB; ++J) {
                    STLBitsetDeterminant& detB = stateB[J].first;
                    if (detA == detB) {
                        overlap += stateA[I].second * stateB[J].second;
                    }
                }
            }
            S->set(A, B, overlap);
        }
    }
    // Diagonalize the overlap
    SharedMatrix Sevecs(new Matrix(nroot, nroot));
    SharedVector Sevals(new Vector(nroot));
    S->diagonalize(Sevecs, Sevals);

    // Form symmetric orthogonalization matrix

    SharedMatrix Strans(new Matrix(nroot, nroot));
    SharedMatrix Sint(new Matrix(nroot, nroot));
    SharedMatrix Diag(new Matrix(nroot, nroot));
    Diag->identity();
    for (int n = 0; n < nroot; ++n) {
        Diag->set(n, n, 1.0 / sqrt(Sevals->get(n)));
    }

    Sint->gemm(false, true, 1.0, Diag, Sevecs, 1.0);
    Strans->gemm(false, false, 1.0, Sevecs, Sint, 1.0);

    // Form the Hamiltonian

    SharedMatrix H(new Matrix(nroot, nroot));

#pragma omp parallel for
    for (int A = 0; A < nroot; ++A) {
        std::vector<std::pair<STLBitsetDeterminant, double>>& stateA =
            old_roots_[A];
        size_t ndetA = stateA.size();
        for (int B = A; B < nroot; ++B) {
            std::vector<std::pair<STLBitsetDeterminant, double>>& stateB =
                old_roots_[B];
            size_t ndetB = stateB.size();
            double HIJ = 0.0;
            for (size_t I = 0; I < ndetA; ++I) {
                STLBitsetDeterminant& detA = stateA[I].first;
                for (size_t J = 0; J < ndetB; ++J) {
                    STLBitsetDeterminant& detB = stateB[J].first;
                    HIJ += detA.slater_rules(detB) * stateA[I].second *
                           stateB[J].second;
                }
            }
            H->set(A, B, HIJ);
            H->set(B, A, HIJ);
        }
    }
    //    H->print();
    H->transform(Strans);

    SharedMatrix Hevecs(new Matrix(nroot, nroot));
    SharedVector Hevals(new Vector(nroot));

    H->diagonalize(Hevecs, Hevals);

    for (int n = 0; n < nroot; ++n) {
        PQ_evals->set(n, Hevals->get(n)); // + nuclear_repulsion_energy_ +
                                          // fci_ints_->scalar_energy());
    }

    //    PQ_evals->print();
}
}
} // EndNamespaces
