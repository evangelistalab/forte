/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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

#include "../helpers/printing.h"
#include "asci.h"

using namespace psi;

namespace psi {
namespace forte {

//#ifdef _OPENMP
//#include <omp.h>
//#else
//#define omp_get_max_threads() 1
//#define omp_get_thread_num() 0
//#define omp_get_num_threads() 1
//#endif

void set_ASCI_options(ForteOptions& foptions) {
    /* Convergence Threshold -*/
    foptions.add_double("ASCI_CONVERGENCE", 1e-5, "ASCI Convergence threshold");
    foptions.add_int("ASCI_MAX_CYCLE", 20, "ASCI MAX Cycle");
    foptions.add_int("ASCI_TDET", 2000, "ASCI Max det");
    foptions.add_int("ASCI_CDET", 200, "ASCI Max reference det");
    foptions.add_double("ASCI_PRESCREEN_THRESHOLD", 1e-12, "ASCI PRESCREEN THRESH");
}

bool pairCompDescend(const std::pair<double, Determinant> E1, const std::pair<double, Determinant> E2) {
    return E1.first > E2.first;
}

ASCI::ASCI(SharedWavefunction ref_wfn, Options& options,
                       std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");
}

ASCI::~ASCI() {}

void ASCI::set_fci_ints(std::shared_ptr<FCIIntegrals> fci_ints) {
    fci_ints_ = fci_ints;
    nuclear_repulsion_energy_ =
        molecule_->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength());
    set_ints_ = true;
}

void ASCI::set_asci_ints(SharedWavefunction ref_wfn, std::shared_ptr<ForteIntegrals> ints) {
    timer int_timer("ACI:Form Integrals");
    ints_ = ints;
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    fci_ints_ = std::make_shared<FCIIntegrals>(ints, mo_space_info_->get_corr_abs_mo("ACTIVE"),
                                               mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));

    auto active_mo = mo_space_info_->get_corr_abs_mo("ACTIVE");
    ambit::Tensor tei_active_aa = ints->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab = ints->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb = ints->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

    nuclear_repulsion_energy_ =
        molecule_->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength());
}

void ASCI::startup() {

    if (!set_ints_) {
        set_asci_ints(reference_wavefunction_, ints_);
    }

    op_.initialize(mo_symmetry_, fci_ints_);

    wavefunction_symmetry_ = 0;
    if (options_["ROOT_SYM"].has_changed()) {
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }
    multiplicity_ = 1;
    if (options_["MULTIPLICITY"].has_changed()) {
        multiplicity_ = options_.get_int("MULTIPLICITY");
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

    twice_ms_ = multiplicity_ - 1;
    if (options_["MS"].has_changed()) {
        twice_ms_ = std::round(2.0 * options_.get_double("MS"));
    }

    nactel_ = nel - 2 * nfrzc_;
    nalpha_ = (nactel_ + twice_ms_) / 2;
    nbeta_ = nactel_ - nalpha_;

    // Build the reference determinant and compute its energy
    CI_Reference ref(reference_wavefunction_, options_, mo_space_info_, fci_ints_, multiplicity_,
                     twice_ms_, wavefunction_symmetry_);
    ref.build_reference(initial_reference_);

    // Read options
    nroot_ = options_.get_int("NROOT");

    max_cycle_ = 20;
    if (options_["ASCI_MAX_CYCLE"].has_changed()) {
        max_cycle_ = options_.get_int("ASCI_MAX_CYCLE");
    }

    diag_method_ = DLSolver;
    if (options_["DIAG_ALGORITHM"].has_changed()) {
        if (options_.get_str("DIAG_ALGORITHM") == "FULL") {
            diag_method_ = Full;
        } else if (options_.get_str("DIAG_ALGORITHM") == "DLSTRING") {
            diag_method_ = DLString;
        } else if (options_.get_str("DIAG_ALGORITHM") == "SPARSE") {
            diag_method_ = Sparse;
        } else if (options_.get_str("DIAG_ALGORITHM") == "SOLVER") {
            diag_method_ = DLSolver;
        } else if (options_.get_str("DIAG_ALGORITHM") == "DYNAMIC") {
            diag_method_ = Dynamic;
        }
    }
    // Decide when to compute coupling lists
    build_lists_ = true;
    if( diag_method_ == Dynamic ){
        build_lists_ = false;
    }

    t_det_ = options_.get_int("ASCI_TDET");
    c_det_ = options_.get_int("ASCI_CDET");
    root_spin_vec_.resize(nroot_);

}

void ASCI::print_info() {

    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Multiplicity", multiplicity_},
        {"Symmetry", wavefunction_symmetry_},
        {"Number of roots", nroot_},
        {"CDet", c_det_},
        {"TDet", t_det_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Convergence threshold", options_.get_double("ASCI_CONVERGENCE")}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Ms", get_ms_string(twice_ms_)},
        {"Diagonalization algorithm", options_.get_str("DIAG_ALGORITHM")}};

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
}

double ASCI::compute_energy() {
    timer energy_timer("ASCI:Energy");

    startup();
    print_method_banner({"ASCI",
                         "written by Jeffrey B. Schriber and Francesco A. Evangelista"});
    outfile->Printf("\n  ==> Reference Information <==\n");
    outfile->Printf("\n  There are %d frozen orbitals.", nfrzc_);
    outfile->Printf("\n  There are %zu active orbitals.\n", nact_);
    print_info();
    outfile->Printf("\n  Using %d threads", omp_get_max_threads());

    Timer asci_elapse;

    // The eigenvalues and eigenvectors
    SharedMatrix PQ_evecs;
    SharedVector PQ_evals;

    // Compute wavefunction and energy
    DeterminantHashVec full_space;
    std::vector<size_t> sizes(nroot_);
    SharedVector energies(new Vector(nroot_));

    DeterminantHashVec PQ_space;

    SharedMatrix P_evecs;
    SharedVector P_evals;

    // Set the P space dets
    DeterminantHashVec P_ref;
    std::vector<double> P_ref_evecs;
    DeterminantHashVec P_space(initial_reference_);

    size_t nvec = options_.get_int("N_GUESS_VEC");
    std::string sigma_method = options_.get_str("SIGMA_BUILD_TYPE");
    std::vector<std::vector<double>> energy_history;
    SparseCISolver sparse_solver(fci_ints_);
    sparse_solver.set_parallel(true);
    sparse_solver.set_force_diag(options_.get_bool("FORCE_DIAG_METHOD"));
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(true);
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_num_vecs(nvec);
    sparse_solver.set_sigma_method(sigma_method);
    sparse_solver.set_spin_project_full(false);
    sparse_solver.set_max_memory(options_.get_int("SIGMA_VECTOR_MAX_MEMORY"));

    // Save the P_space energies to predict convergence
    std::vector<double> P_energies;
    // approx_rdm_ = false;

    int cycle;
    for (cycle = 0; cycle < max_cycle_; ++cycle) {
        Timer cycle_time;

        // Step 1. Diagonalize the Hamiltonian in the P space
        std::string cycle_h = "Cycle " + std::to_string(cycle);
        print_h2(cycle_h);
        outfile->Printf("\n  Initial P space dimension: %zu", P_space.size());

        if (sigma_method == "HZ") {
            op_.clear_op_lists();
            op_.clear_tp_lists();
            op_.build_strings(P_space);
            op_.op_lists(P_space);
            op_.tp_lists(P_space);
        } else if (diag_method_ != Dynamic) {
            op_.clear_op_s_lists();
            op_.clear_tp_s_lists();
            op_.build_strings(P_space);
            op_.op_s_lists(P_space);
            op_.tp_s_lists(P_space);
        }

        sparse_solver.manual_guess(false);
        Timer diag;
        sparse_solver.diagonalize_hamiltonian_map(P_space, op_, P_evals, P_evecs, nroot_,
                                                  multiplicity_, diag_method_);
        outfile->Printf("\n  Time spent diagonalizing H:   %1.6f s", diag.get());

        P_energies.push_back(P_evals->get(0));

        // Print the energy
        outfile->Printf("\n");
        double P_abs_energy =
            P_evals->get(0) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
        outfile->Printf("\n    P-space  CI Energy Root 0       = "
                        "%.12f ", P_abs_energy);
        outfile->Printf("\n");
        
        // Step 2. Find determinants in the Q space
        Timer build_space;
        find_q_space(P_space, PQ_space, P_evals, P_evecs);
        outfile->Printf("\n  Time spent building the model space: %1.6f", build_space.get());

        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        if (sigma_method == "HZ") {
            op_.clear_op_lists();
            op_.clear_tp_lists();
            Timer str;
            op_.build_strings(PQ_space);
            outfile->Printf("\n  Time spent building strings      %1.6f s", str.get());
            op_.op_lists(PQ_space);
            op_.tp_lists(PQ_space);
        } else if (diag_method_ != Dynamic) {
            op_.clear_op_s_lists();
            op_.clear_tp_s_lists();
            op_.build_strings(PQ_space);
            op_.op_s_lists(PQ_space);
            op_.tp_s_lists(PQ_space);
        }
        Timer diag_pq;

        sparse_solver.diagonalize_hamiltonian_map(PQ_space, op_, PQ_evals, PQ_evecs, nroot_,
                                                  multiplicity_, diag_method_);

        outfile->Printf("\n  Total time spent diagonalizing H:   %1.6f s", diag_pq.get());

        // Print the energy
        outfile->Printf("\n");
        double abs_energy =
            PQ_evals->get(0) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
        outfile->Printf("\n    PQ-space CI Energy Root 0        = "
                        "%.12f Eh", abs_energy);
        outfile->Printf("\n");
        


        // Step 4. Check convergence and break if needed
        bool converged = check_convergence(energy_history, PQ_evals);
        if (converged) {
            outfile->Printf("\n  ***** Calculation Converged *****");
            break;
        }

        // Step 5. Prune the P + Q space to get an updated P space
        prune_q_space(PQ_space, P_space, PQ_evecs);

        // Print information about the wave function
        print_wfn(PQ_space, op_, PQ_evecs, nroot_);
        outfile->Printf("\n  Cycle %d took: %1.6f s", cycle, cycle_time.get());
    } // end iterations

    double root_energy =
        PQ_evals->get(0) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();

    Process::environment.globals["CURRENT ENERGY"] = root_energy;
    Process::environment.globals["ACI ENERGY"] = root_energy;

    outfile->Printf("\n\n  %s: %f s", "ASCI ran in ", asci_elapse.get());

    double pt2 = 0.0;
    if (options_.get_bool("MRPT2")) {
        MRPT2 pt(reference_wavefunction_, options_, ints_, mo_space_info_, PQ_space, PQ_evecs,PQ_evals);
        pt2 = pt.compute_energy();
    }


    size_t dim = PQ_space.size();
    // Print a summary
    outfile->Printf("\n\n  ==> ASCI Summary <==\n");

    outfile->Printf("\n  Iterations required:                         %zu", cycle);
    outfile->Printf("\n  Dimension of optimized determinant space:    %zu\n", dim);
    outfile->Printf(    "\n  * AS-CI Energy Root 0        = %.12f Eh",root_energy);
    if (options_.get_bool("MRPT2")) {
        outfile->Printf("\n  * AS-CI+PT2 Energy Root 0    = %.12f Eh",root_energy+pt2);
    }

    outfile->Printf("\n\n  ==> Wavefunction Information <==");

    print_wfn(PQ_space, op_, PQ_evecs, nroot_);

    

    return root_energy;
}

void ASCI::find_q_space(DeterminantHashVec& P_space, DeterminantHashVec& PQ_space,
                                      SharedVector evals, SharedMatrix evecs) {
    timer find_q("ACI:Build Model Space");
    Timer build;

    det_hash<double> V_hash;
    get_excited_determinants_sr(evecs, P_space, V_hash);

    // This will contain all the determinants
    PQ_space.clear();
    external_wfn_.clear();
    // Add the P-space determinants and zero the hash
    const det_hashvec& detmap = P_space.wfn_hash();
    for (det_hashvec::iterator it = detmap.begin(), endit = detmap.end(); it != endit; ++it) {
        V_hash.erase(*it);
    }
  //  PQ_space.swap(P_space);

    outfile->Printf("\n  %s: %zu determinants", "Dimension of the Ref + SD space", V_hash.size());
    outfile->Printf("\n  %s: %f s\n", "Time spent building the external space (default)",
                    build.get());

    Timer screen;
    // Compute criteria for all dets, store them all
    Determinant zero_det; // <- xsize (nact_);
    std::vector<std::pair<double, Determinant>> F_space(V_hash.size(),
                                                        std::make_pair(0.0, zero_det));

    Timer build_sort;
    size_t max = V_hash.size();
    size_t N = 0;
    // sorted_dets.reserve(max);
    for (const auto& I : V_hash) {
        double delta = fci_ints_->energy(I.first) - evals->get(0);
        double V = I.second;

        double criteria = V / delta;
        F_space[N] = std::make_pair(std::fabs(criteria), I.first);
        
        N++;
    }
    for( const auto& I : detmap ){
        F_space.push_back( std::make_pair( std::fabs(evecs->get(P_space.get_idx(I), 0)), I )); 
    }
    outfile->Printf("\n  Time spent building sorting list: %1.6f", build_sort.get());

    Timer sorter;
    std::sort(F_space.begin(), F_space.end(), pairCompDescend);
    outfile->Printf("\n  Time spent sorting: %1.6f", sorter.get());
    Timer select;

    size_t maxI = std::min( t_det_, int(F_space.size()) );
    for (size_t I = 0; I < maxI; ++I) {
        auto& pair = F_space[I];
        PQ_space.add(pair.second);
    }
    outfile->Printf("\n  Time spent selecting: %1.6f", select.get());
    outfile->Printf("\n  %s: %zu determinants", "Dimension of the P + Q space",
                    PQ_space.size());
    outfile->Printf("\n  %s: %f s", "Time spent screening the model space", screen.get());
    
}

bool ASCI::check_convergence(std::vector<std::vector<double>>& energy_history,
                                   SharedVector evals) {
    int nroot = evals->dim();
    int ref = 0;

    if (energy_history.size() == 0) {
        std::vector<double> new_energies;
        double state_n_energy = evals->get(0) + nuclear_repulsion_energy_;
        new_energies.push_back(state_n_energy);
        energy_history.push_back(new_energies);
        return false;
    }

    double old_avg_energy = 0.0;
    double new_avg_energy = 0.0;

    std::vector<double> new_energies;
    std::vector<double> old_energies = energy_history[energy_history.size() - 1];
    double state_n_energy = evals->get(0) + nuclear_repulsion_energy_;
    new_energies.push_back(state_n_energy);
    new_avg_energy += state_n_energy;
    old_avg_energy += old_energies[0];
    
    old_avg_energy /= static_cast<double>(nroot);
    new_avg_energy /= static_cast<double>(nroot);

    energy_history.push_back(new_energies);

    // Check for convergence
    return (std::fabs(new_avg_energy - old_avg_energy) < options_.get_double("ACI_CONVERGENCE"));
}

void ASCI::prune_q_space(DeterminantHashVec& PQ_space, DeterminantHashVec& P_space,
                               SharedMatrix evecs) {
    // Select the new reference space using the sorted CI coefficients
    P_space.clear();

    // Create a vector that stores the absolute value of the CI coefficients
    std::vector<std::pair<double, Determinant>> dm_det_list;
    // for (size_t I = 0, max = PQ_space.size(); I < max; ++I){
    const det_hashvec& detmap = PQ_space.wfn_hash();
    for (size_t i = 0, max_i = detmap.size(); i < max_i; ++i) {
        double criteria = std::fabs(evecs->get(i, 0));
        dm_det_list.push_back(std::make_pair(criteria, detmap[i]));
    }

    // Decide which determinants will go in pruned_space
    // Include all determinants such that
    // sum_I |C_I|^2 < tau_p, where the sum runs over all the excluded
    // determinants
    // Sort the CI coefficients in ascending order
    std::sort(dm_det_list.begin(), dm_det_list.end(), pairCompDescend);
    size_t Imax = std::min( c_det_,int(dm_det_list.size()) );

    for (size_t I = 0; I < Imax; ++I) {
        P_space.add(dm_det_list[I].second);
    }
}

//bool ASCI::check_stuck(std::vector<std::vector<double>>& energy_history, SharedVector evals) {
//    bool stuck = false;
//    int nroot = evals->dim();
//    if (cycle_ < 4) {
//        stuck = false;
//    } else {
//        std::vector<double> av_energies;
//        for (int i = 0; i < cycle_; ++i) {
//            double energy = 0.0;
//            for (int n = 0; n < nroot; ++n) {
//                energy += energy_history[i][n];
//            }
//            energy /= static_cast<double>(nroot);
//            av_energies.push_back(energy);
//        }
//
//        if (std::fabs(av_energies[cycle_ - 1] - av_energies[cycle_ - 3]) <
//            options_.get_double("ACI_CONVERGENCE")) { // and
//            //			std::fabs( av_energies[cycle_-2] -
//            // av_energies[cycle_ - 4]
//            //)
//            //< options_.get_double("ACI_CONVERGENCE") ){
//            stuck = true;
//        }
//    }
//    return stuck;
//}

std::vector<std::pair<double, double>> ASCI::compute_spin(DeterminantHashVec& space,
                                                                WFNOperator& op, SharedMatrix evecs,
                                                                int nroot) {
    // WFNOperator op(mo_symmetry_);

    // op.build_strings(space);
    // op.op_lists(space);
    // op.tp_lists(space);

    std::vector<std::pair<double, double>> spin_vec(nroot);
    if (options_.get_str("SIGMA_BUILD_TYPE") == "HZ") {
        op.clear_op_s_lists();
        op.clear_tp_s_lists();
        op.build_strings(space);
        op.op_lists(space);
        op.tp_lists(space);
    }

    if ( !build_lists_ ) {
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

std::vector<std::tuple<double, int, int>> ASCI::sym_labeled_orbitals(std::string type) {
    std::vector<std::tuple<double, int, int>> labeled_orb;

    if (type == "RHF" or type == "ROHF" or type == "ALFA") {

        // Create a vector of orbital energy and index pairs
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (int a = 0; a < nactpi_[h]; ++a) {
                orb_e.push_back(std::make_pair(epsilon_a_->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, symmetry, and idx
        for (size_t a = 0; a < nact_; ++a) {
            labeled_orb.push_back(
                std::make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
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
                orb_e.push_back(std::make_pair(epsilon_b_->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, sym, and idx
        for (size_t a = 0; a < nact_; ++a) {
            labeled_orb.push_back(
                std::make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    return labeled_orb;
}

void ASCI::print_wfn(DeterminantHashVec& space, WFNOperator& op, SharedMatrix evecs,
                           int nroot) {
    std::string state_label;
    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet", "decatet"});

    std::vector<std::pair<double, double>> spins = compute_spin(space, op, evecs, nroot);

    DeterminantHashVec tmp;
    std::vector<double> tmp_evecs;

    outfile->Printf("\n\n  Most important contributions to root 0:");

    size_t max_dets = std::min(10, evecs->nrow());
    tmp.subspace(space, evecs, tmp_evecs, max_dets, 0);

    for (size_t I = 0; I < max_dets; ++I) {
        outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s", I, tmp_evecs[I],
                        tmp_evecs[I] * tmp_evecs[I], space.get_idx(tmp.get_det(I)),
                        tmp.get_det(I).str(nact_).c_str());
    }
    state_label = s2_labels[std::round(spins[0].first * 2.0)];
    root_spin_vec_.clear();
    root_spin_vec_[0] = std::make_pair(spins[0].first, spins[0].second);
    outfile->Printf("\n\n  Spin state for root 0: S^2 = %5.6f, S = %5.3f, %s", 
                    root_spin_vec_[0].first, root_spin_vec_[0].second, state_label.c_str());
}


double ASCI::compute_spin_contamination(DeterminantHashVec& space, WFNOperator& op,
                                              SharedMatrix evecs, int nroot) {
    auto spins = compute_spin(space, op, evecs, nroot);
    double spin_contam = 0.0;
    for (int n = 0; n < nroot; ++n) {
        spin_contam += spins[n].second;
    }
    spin_contam /= static_cast<double>(nroot);
    spin_contam -= (0.25 * (multiplicity_ * multiplicity_ - 1.0));

    return spin_contam;
}

//void ASCI::save_dets_to_file(DeterminantHashVec& space, SharedMatrix evecs) {
//    // Use for single-root calculations only
//    const det_hashvec& detmap = space.wfn_hash();
//    for (size_t i = 0, max_i = detmap.size(); i < max_i; ++i) {
//        det_list_ << detmap[i].str(nact_).c_str() << " " << std::fabs(evecs->get(i, 0)) << " ";
//        //	for(size_t J = 0, maxJ = space.size(); J < maxJ; ++J){
//        //		det_list_ << space[I].slater_rules(space[J]) << " ";
//        //	}
//        //	det_list_ << "\n";
//    }
//    det_list_ << "\n";
//}


//Reference ASCI::reference() {
//    // const std::vector<Determinant>& final_wfn =
//    //     final_wfn_.determinants();
//    CI_RDMS ci_rdms(final_wfn_, fci_ints_, evecs_, 0, 0);
//    ci_rdms.set_max_rdm(rdm_level_);
//    Reference aci_ref = ci_rdms.reference(ordm_a_, ordm_b_, trdm_aa_, trdm_ab_, trdm_bb_, trdm_aaa_,
//                                          trdm_aab_, trdm_abb_, trdm_bbb_);
//    return aci_ref;
//}

//void ASCI::print_nos() {
//    print_h2("NATURAL ORBITALS");
//
//    std::shared_ptr<Matrix> opdm_a(new Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
//    std::shared_ptr<Matrix> opdm_b(new Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));
//
//    int offset = 0;
//    for (int h = 0; h < nirrep_; h++) {
//        for (int u = 0; u < nactpi_[h]; u++) {
//            for (int v = 0; v < nactpi_[h]; v++) {
//                opdm_a->set(h, u, v, ordm_a_[(u + offset) * nact_ + v + offset]);
//                opdm_b->set(h, u, v, ordm_b_[(u + offset) * nact_ + v + offset]);
//            }
//        }
//        offset += nactpi_[h];
//    }
//    SharedVector OCC_A(new Vector("ALPHA OCCUPATION", nirrep_, nactpi_));
//    SharedVector OCC_B(new Vector("BETA OCCUPATION", nirrep_, nactpi_));
//    SharedMatrix NO_A(new Matrix(nirrep_, nactpi_, nactpi_));
//    SharedMatrix NO_B(new Matrix(nirrep_, nactpi_, nactpi_));
//
//    opdm_a->diagonalize(NO_A, OCC_A, descending);
//    opdm_b->diagonalize(NO_B, OCC_B, descending);
//
//    // std::ofstream file;
//    // file.open("nos.txt",std::ios_base::app);
//    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
//    for (int h = 0; h < nirrep_; h++) {
//        for (int u = 0; u < nactpi_[h]; u++) {
//            auto irrep_occ =
//                std::make_pair(OCC_A->get(h, u) + OCC_B->get(h, u), std::make_pair(h, u + 1));
//            vec_irrep_occupation.push_back(irrep_occ);
//            //          file << OCC_A->get(h, u) + OCC_B->get(h, u) << "  ";
//        }
//    }
//    // file << endl;
//    // file.close();
//
//    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
//    std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
//              std::greater<std::pair<double, std::pair<int, int>>>());
//
//    size_t count = 0;
//    outfile->Printf("\n    ");
//    for (auto vec : vec_irrep_occupation) {
//        outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second, ct.gamma(vec.second.first).symbol(),
//                        vec.first);
//        if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
//            outfile->Printf("\n    ");
//    }
//    outfile->Printf("\n\n");
//
//    // Compute active space weights
//    if (print_weights_) {
//        double no_thresh = options_.get_double("ACI_NO_THRESHOLD");
//
//        std::vector<int> active(nirrep_, 0);
//        std::vector<std::vector<int>> active_idx(nirrep_);
//        std::vector<int> docc(nirrep_, 0);
//
//        print_h2("Active Space Weights");
//        for (int h = 0; h < nirrep_; ++h) {
//            std::vector<double> weights(nactpi_[h], 0.0);
//            std::vector<double> oshell(nactpi_[h], 0.0);
//            for (int p = 0; p < nactpi_[h]; ++p) {
//                for (int q = 0; q < nactpi_[h]; ++q) {
//                    double occ = OCC_A->get(h, q) + OCC_B->get(h, q);
//                    if ((occ >= no_thresh) and (occ <= (2.0 - no_thresh))) {
//                        weights[p] += (NO_A->get(h, p, q)) * (NO_A->get(h, p, q));
//                        oshell[p] += (NO_A->get(h, p, q)) * (NO_A->get(h, p, q)) * (2 - occ) * occ;
//                    }
//                }
//            }
//
//            outfile->Printf("\n  Irrep %d:", h);
//            outfile->Printf("\n  Active idx     MO idx        Weight         OS-Weight");
//            outfile->Printf("\n ------------   --------   -------------    -------------");
//            for (int w = 0; w < nactpi_[h]; ++w) {
//                outfile->Printf("\n      %0.2d           %d       %1.9f      %1.9f", w + 1,
//                                w + frzcpi_[h] + 1, weights[w], oshell[w]);
//                if (weights[w] >= 0.9) {
//                    active[h]++;
//                    active_idx[h].push_back(w + frzcpi_[h] + 1);
//                }
//            }
//        }
//    }
//}


std::vector<std::pair<size_t, double>>
ASCI::dl_initial_guess(std::vector<Determinant>& old_dets, std::vector<Determinant>& dets,
                             SharedMatrix& evecs, int root) {
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

//void ASCI::compute_rdms(std::shared_ptr<FCIIntegrals> fci_ints, DeterminantHashVec& dets,
//                              WFNOperator& op, SharedMatrix& PQ_evecs, int root1, int root2) {
//
//    ordm_a_.clear();
//    ordm_b_.clear();
//
//    trdm_aa_.clear();
//    trdm_ab_.clear();
//    trdm_bb_.clear();
//
//    trdm_aaa_.clear();
//    trdm_aab_.clear();
//    trdm_abb_.clear();
//    trdm_bbb_.clear();
//
//    CI_RDMS ci_rdms_(dets, fci_ints, PQ_evecs, root1, root2);
//
////    double total_time = 0.0;
//    ci_rdms_.set_max_rdm(rdm_level_);
//
//    
//    if(options_.get_bool("ACI_DIRECT_RDMS") ){
//       // Timer dyn;
//     //   CI_RDMS ci_rdms_(final_wfn_, fci_ints_, PQ_evecs, 0, 0);
//        ci_rdms_.compute_rdms_dynamic(ordm_a_, ordm_b_, trdm_aa_, trdm_ab_, trdm_bb_,
//                                        trdm_aaa_,trdm_aab_,trdm_abb_,trdm_bbb_);
//                print_nos();
//       // double dt = dyn.get();
//       // outfile->Printf("\n  RDMS (bits) took           %1.6f", dt);
//    } else {
//        if (rdm_level_ >= 1) {
//            Timer one_r;
//            ci_rdms_.compute_1rdm(ordm_a_, ordm_b_, op);
//            outfile->Printf("\n  1-RDM  took %2.6f s (determinant)", one_r.get());
//
//            if (options_.get_bool("ACI_PRINT_NO")) {
//                print_nos();
//            }
//        }
//        if (rdm_level_ >= 2) {
//            Timer two_r;
//            ci_rdms_.compute_2rdm(trdm_aa_, trdm_ab_, trdm_bb_, op);
//            outfile->Printf("\n  2-RDMS took %2.6f s (determinant)", two_r.get());
//        }
//        if (rdm_level_ >= 3) {
//            Timer tr;
//            ci_rdms_.compute_3rdm(trdm_aaa_, trdm_aab_, trdm_abb_, trdm_bbb_, op);
//            outfile->Printf("\n  3-RDMs took %2.6f s (determinant)", tr.get());
//        }
//    }
//    if (options_.get_bool("ACI_TEST_RDMS")) {
//        ci_rdms_.rdm_test(ordm_a_, ordm_b_, trdm_aa_, trdm_bb_, trdm_ab_, trdm_aaa_, trdm_aab_,
//                          trdm_abb_, trdm_bbb_);
//    }
//
//    if (approx_rdm_ and (rdm_level_ >= 2)) {
//        outfile->Printf("\n  Computing energy with new RDMs");
//
//        double en = ci_rdms_.get_energy(ordm_a_, ordm_b_, trdm_aa_, trdm_bb_, trdm_ab_);
//        outfile->Printf("\n  Energy from approximate RDM:  %1.12f", en);
//    }
//}

void ASCI::get_excited_determinants_sr(SharedMatrix evecs, DeterminantHashVec& P_space,
                                             det_hash<double>& V_hash) {
    Timer build;
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();
    int nroot = 1;
    double screen_thresh_ = options_.get_double("ASCI_PRESCREEN_THRESHOLD");

// Loop over reference determinants
#pragma omp parallel
    {
        int num_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t bin_size = max_P / num_thread;
        bin_size += (tid < (max_P % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (max_P % num_thread))
                ? tid * bin_size
                : (max_P % num_thread) * (bin_size + 1) + (tid - (max_P % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        det_hash<double> V_hash_t;
        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double Cp = evecs->get(P, 0);

            std::vector<int> aocc = det.get_alfa_occ(nact_); // TODO check size
            std::vector<int> bocc = det.get_beta_occ(nact_); // TODO check size
            std::vector<int> avir = det.get_alfa_vir(nact_); // TODO check size
            std::vector<int> bvir = det.get_beta_vir(nact_); // TODO check size

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            Determinant new_det(det);
            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa) * Cp;
                        if (std::abs(HIJ) >= screen_thresh_) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            V_hash_t[new_det] += HIJ;
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
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa) * Cp;
                        if (std::abs(HIJ) >= screen_thresh_) {
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            V_hash_t[new_det] += HIJ;
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb) * Cp;
                                if (std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                    V_hash_t[new_det] += HIJ;
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb) * Cp;
                                if (std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                    V_hash_t[new_det] += HIJ;
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb) * Cp;
                                if (std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);
                                    V_hash_t[new_det] += HIJ;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (tid == 0)
            outfile->Printf("\n  Time spent forming F space: %20.6f", build.get());
        Timer merge_t;
#pragma omp critical
        {
            for (auto& pair : V_hash_t) {
                const Determinant& det = pair.first;
                V_hash[det] += pair.second;
            }
        }
        if (tid == 0)
            outfile->Printf("\n  Time spent merging thread F spaces: %20.6f", merge_t.get());
    } // Close threads
}

}}
