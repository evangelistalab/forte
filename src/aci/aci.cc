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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libpsio/psio.hpp"

#include "../ci_rdms.h"
#include "../ci_reference.h"
#include "../fci/fci_integrals.h"
#include "../forte_options.h"
#include "../mrpt2.h"
#include "../orbital-helper/unpaired_density.h"
#include "../reference.h"
#include "../sparse_ci_solver.h"
#include "../stl_bitset_determinant.h"
#include "aci.h"

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

void set_ACI_options(ForteOptions& foptions) {
    /* Convergence Threshold -*/
    foptions.add_double("ACI_CONVERGENCE", 1e-9, "ACI Convergence threshold");

    /*- The selection type for the Q-space-*/
    foptions.add_str("ACI_SELECT_TYPE", "AIMED_ENERGY", "The energy selection criteria");
    /*-Threshold for the selection of the P space -*/
    foptions.add_double("SIGMA", 0.01, "The energy selection threshold");
    /*- The threshold for the selection of the Q space -*/
    foptions.add_double("GAMMA", 1.0, "The reference space selection threshold");
    /*- The SD-space prescreening threshold -*/
    foptions.add_double("ACI_PRESCREEN_THRESHOLD", 1e-12, "The SD space prescreening threshold");
    /*- The type of selection parameters to use*/
    foptions.add_bool("ACI_PERTURB_SELECT", false, "Type of energy selection");
    /*Function of q-space criteria, per root*/
    foptions.add_str("ACI_PQ_FUNCTION", "AVERAGE", "Function for SA-ACI");
    /* Method to calculate excited state */
    foptions.add_str("ACI_EXCITED_ALGORITHM", "ROOT_ORTHOGONALIZE", "The excited state algorithm");
    /*Number of roots to compute*/
    foptions.add_int("ACI_NROOT", 1, "Number of roots for ACI computation");
    /*Roots to compute*/
    foptions.add_int("ACI_ROOT", 0, "Root for single-state computations");
    /*- Compute 1-RDM? -*/
    foptions.add_int("ACI_MAX_RDM", 1, "Order of RDM to compute");
    /*- Type of spin projection
     * 0 - None
     * 1 - Project initial P spaces at each iteration
     * 2 - Project only after converged PQ space
     * 3 - Do 1 and 2 -*/
  //  foptions.add_int("ACI_SPIN_PROJECTION", 0, "Type of spin projection");
    /*- Add determinants to enforce spin-complete set? -*/
    foptions.add_bool("ACI_ENFORCE_SPIN_COMPLETE", true,
                      "Enforce determinant spaces to be spin-complete");
    /*- Project out spin contaminants in Davidson-Liu's algorithm? -*/
    foptions.add_bool("ACI_PROJECT_OUT_SPIN_CONTAMINANTS", true,
                      "Project out spin contaminants in Davidson-Liu's algorithm");
    /*- Project solution in full diagonalization algorithm -*/
    foptions.add_bool("SPIN_PROJECT_FULL", false,
                      "Project solution in full diagonalization algorithm");
    /*- Add "degenerate" determinants not included in the aimed selection?
     * -*/
    foptions.add_bool("ACI_ADD_AIMED_DEGENERATE", true,
                      "Add degenerate determinants not included in the aimed selection");
    /*- Perform size extensivity correction -*/
    foptions.add_str("ACI_SIZE_CORRECTION", "", "Perform size extensivity correction");
    /*- Sets the maximum cycle -*/
    foptions.add_int("ACI_MAX_CYCLE", 20, "Maximum number of cycles");
    /*- Control print level -*/
    foptions.add_bool("ACI_QUIET_MODE", false, "Print during ACI procedure");
    /*- Control streamlining -*/
    foptions.add_bool("ACI_STREAMLINE_Q", false, "Do streamlined algorithm");
    /*- Initial reference wavefunction -*/
    foptions.add_str("ACI_INITIAL_SPACE", "CAS", "The initial reference space");
    /*- Number of iterations to run SA-ACI before SS-ACI -*/
    foptions.add_int("ACI_PREITERATIONS", 0, "Number of iterations to run SA-ACI before SS-ACI");
    /*- Number of roots to average -*/
    foptions.add_int("ACI_N_AVERAGE", 1, "Number of roots to averag");
    /*- Offset for state averaging -*/
    foptions.add_int("ACI_AVERAGE_OFFSET", 0, "Offset for state averaging");
    /*- Print final wavefunction to file? -*/
    foptions.add_bool("ACI_SAVE_FINAL_WFN", false, "Print final wavefunction to file");
    /*- Print the P space? -*/
    foptions.add_bool("ACI_PRINT_REFS", false, "Print the P space");
    /*- Set the initial guess space size for DL solver -*/
    foptions.add_int("DL_GUESS_SIZE", 100, "Set the initial guess space size for DL solver");
    /*- Number of guess vectors for Sparse CI solver -*/
    foptions.add_int("N_GUESS_VEC", 10, "Number of guess vectors for Sparse CI solver");
    foptions.add_double("ACI_NO_THRESHOLD", 0.02, "Threshold for active space prediction");
    foptions.add_double("ACI_SPIN_TOL", 0.02, "Tolerance for S^2 value");

    /*- Approximate 1RDM? -*/
    foptions.add_bool("ACI_APPROXIMATE_RDM", false, "Approximate the RDMs");
    /*- Test RDMs -*/
    foptions.add_bool("ACI_TEST_RDMS", false, "Run test for the RDMs");

    /*- Do compute nroots on first cycle? -*/
    foptions.add_bool("ACI_FIRST_ITER_ROOTS", false, "Compute all roots on first iteration?");
    foptions.add_bool("ACI_PRINT_WEIGHTS", false, "Print weights for active space prediction");

    /*- Print Natural orbitals -*/
    foptions.add_bool("ACI_PRINT_NO", true, "Print the natural orbitals");

    /*- Save the final wavefunction -*/
    foptions.add_bool("SAVE_FINAL_WFN", false, "Save the final wavefunction to a file");

    /*- Compute ACI-NOs -*/
    foptions.add_bool("ACI_NO", false, "Computes ACI natural orbitals");

    /*- Compute full PT2 energy -*/
    foptions.add_bool("MRPT2", false, "Compute full PT2 energy");

    /*- Compute unpaired electron density -*/
    foptions.add_bool("UNPAIRED_DENSITY", false, "Compute unpaired electron density");

    /*- Add all active singles -*/
    foptions.add_bool("ACI_ADD_SINGLES", false,
                      "Adds all active single excitations to the final wave function");
}

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

    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");
    op_.initialize(mo_symmetry_);
    startup();
}

AdaptiveCI::~AdaptiveCI() {}

void AdaptiveCI::set_aci_ints(SharedWavefunction ref_wfn, std::shared_ptr<ForteIntegrals> ints) {
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

    STLBitsetDeterminant::set_ints(fci_ints_);

    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();
}

void AdaptiveCI::startup() {
    quiet_mode_ = false;
    if (options_["ACI_QUIET_MODE"].has_changed()) {
        quiet_mode_ = options_.get_bool("ACI_QUIET_MODE");
    }

    set_aci_ints(reference_wavefunction_, ints_);

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

    STLBitsetDeterminant det;

    // Build the reference determinant and compute its energy

    CI_Reference ref(reference_wavefunction_, options_, mo_space_info_, det, multiplicity_,
                     twice_ms_, wavefunction_symmetry_);
    ref.build_reference(initial_reference_);

    // Read options
    nroot_ = options_.get_int("ACI_NROOT");
    sigma_ = options_.get_double("SIGMA");
    gamma_ = options_.get_double("GAMMA");
    screen_thresh_ = options_.get_double("ACI_PRESCREEN_THRESHOLD");
    add_aimed_degenerate_ = options_.get_bool("ACI_ADD_AIMED_DEGENERATE");
    project_out_spin_contaminants_ = options_.get_bool("ACI_PROJECT_OUT_SPIN_CONTAMINANTS");
    spin_complete_ = options_.get_bool("ACI_ENFORCE_SPIN_COMPLETE");
    rdm_level_ = options_.get_int("ACI_MAX_RDM");

    max_cycle_ = 20;
    if (options_["ACI_MAX_CYCLE"].has_changed()) {
        max_cycle_ = options_.get_int("ACI_MAX_CYCLE");
    }
    pre_iter_ = 0;
    if (options_["ACI_PREITERATIONS"].has_changed()) {
        pre_iter_ = options_.get_int("ACI_PREITERATIONS");
    }

    spin_tol_ = options_.get_double("ACI_SPIN_TOL");
    // set the initial S^@ guess as input multiplicity
    int S = (multiplicity_ - 1.0) / 2.0;
    int S2 = multiplicity_ - 1.0;
    for (int n = 0; n < nroot_; ++n) {
        root_spin_vec_.push_back(make_pair(S, S2));
    }

    // get options for algorithm
    perturb_select_ = options_.get_bool("ACI_PERTURB_SELECT");
    pq_function_ = options_.get_str("ACI_PQ_FUNCTION");
    ex_alg_ = options_.get_str("ACI_EXCITED_ALGORITHM");
    ref_root_ = options_.get_int("ACI_ROOT");
    root_ = options_.get_int("ACI_ROOT");
    approx_rdm_ = options_.get_bool("ACI_APPROXIMATE_RDM");
    print_weights_ = options_.get_bool("ACI_PRINT_WEIGHTS");

    diag_method_ = DLString;
    if (options_["DIAG_ALGORITHM"].has_changed()) {
        if (options_.get_str("DIAG_ALGORITHM") == "FULL") {
            diag_method_ = Full;
        } else if (options_.get_str("DIAG_ALGORITHM") == "DLSTRING") {
            diag_method_ = DLString;
        } else if (options_.get_str("DIAG_ALGORITHM") == "SPARSE") {
            diag_method_ = Sparse;
        } else if (options_.get_str("DIAG_ALGORITHM") == "SOLVER") {
            diag_method_ = DLSolver;
        }
    }
    aimed_selection_ = false;
    energy_selection_ = false;
    if (options_.get_str("ACI_SELECT_TYPE") == "AIMED_AMP") {
        aimed_selection_ = true;
        energy_selection_ = false;
    } else if (options_.get_str("ACI_SELECT_TYPE") == "AIMED_ENERGY") {
        aimed_selection_ = true;
        energy_selection_ = true;
    } else if (options_.get_str("ACI_SELECT_TYPE") == "ENERGY") {
        aimed_selection_ = false;
        energy_selection_ = true;
    } else if (options_.get_str("ACI_SELECT_TYPE") == "AMP") {
        aimed_selection_ = false;
        energy_selection_ = false;
    }

    if (options_.get_bool("ACI_STREAMLINE_Q") == true) {
        streamline_qspace_ = true;
    } else {
        streamline_qspace_ = false;
    }

    // Set streamline mode to true if possible
    if ((nroot_ == 1) and (aimed_selection_ == true) and (energy_selection_ == true) and
        (perturb_select_ == false)) {

        streamline_qspace_ = true;
    }
}

void AdaptiveCI::print_info() {

    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Multiplicity", multiplicity_},
        {"Symmetry", wavefunction_symmetry_},
        {"Number of roots", nroot_},
        {"Root used for properties", options_.get_int("ACI_ROOT")}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Sigma (Eh)", sigma_},
        {"Gamma (Eh^(-1))", gamma_},
        {"Convergence threshold", options_.get_double("ACI_CONVERGENCE")}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Ms", get_ms_string(twice_ms_)},
        {"Determinant selection criterion",
         energy_selection_ ? "Second-order Energy" : "First-order Coefficients"},
        {"Selection criterion", aimed_selection_ ? "Aimed selection" : "Threshold"},
        {"Excited Algorithm", options_.get_str("ACI_EXCITED_ALGORITHM")},
        //        {"Q Type", q_rel_ ? "Relative Energy" : "Absolute Energy"},
        //        {"PT2 Parameters", options_.get_bool("PERTURB_SELECT") ?
        //        "True" : "False"},
        {"Project out spin contaminants", project_out_spin_contaminants_ ? "True" : "False"},
        {"Enforce spin completeness of basis", spin_complete_ ? "True" : "False"},
        {"Enforce complete aimed selection", add_aimed_degenerate_ ? "True" : "False"}};

    // Print some information
    outfile->Printf("\n  ==> Calculation Information <==\n");
    outfile->Printf("\n  %s", string(65, '-').c_str());
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s %-5d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %8.2e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n  %s", string(65, '-').c_str());
    outfile->Flush();
}

double AdaptiveCI::compute_energy() {
    if (options_["ACI_QUIET_MODE"].has_changed()) {
        quiet_mode_ = options_.get_bool("ACI_QUIET_MODE");
    }
    print_method_banner({"Adaptive Configuration Interaction",
                         "written by Jeffrey B. Schriber and Francesco A. Evangelista"});
    outfile->Printf("\n  ==> Reference Information <==\n");
    outfile->Printf("\n  There are %d frozen orbitals.", nfrzc_);
    outfile->Printf("\n  There are %zu active orbitals.\n", nact_);
    //       reference_determinant_.print();
    //        outfile->Printf("\n  REFERENCE ENERGY:         %1.12f",
    //                        reference_determinant_.energy() +
    //                            nuclear_repulsion_energy_ +
    //                            fci_ints_->scalar_energy());
    print_info();
    if (!quiet_mode_) {
        outfile->Printf("\n  Using %d threads", omp_get_max_threads());
    }

    if (ex_alg_ == "COMPOSITE") {
        ex_alg_ = "AVERAGE";
    }

    op_.set_quiet_mode(quiet_mode_);
    Timer aci_elapse;

    // The eigenvalues and eigenvectors
    SharedMatrix PQ_evecs;
    SharedVector PQ_evals;

    // Compute wavefunction and energy
    size_t dim;
    int nrun = 1;
    bool multi_state = false;

    if (options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_COMBINE" or
        options_.get_str("ACI_EXCITED_ALGORITHM") == "MULTISTATE" or
        options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_ORTHOGONALIZE") {
        nrun = nroot_;
        multi_state = true;
    }

    DeterminantMap full_space;
    std::vector<size_t> sizes(nroot_);
    SharedVector energies(new Vector(nroot_));
    std::vector<double> pt2_energies(nroot_);

    DeterminantMap PQ_space;

    for (int i = 0; i < nrun; ++i) {
        nroot_ = options_.get_int("ACI_NROOT");
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
        if (ex_alg_ == "ROOT_ORTHOGONALIZE" and (nroot_ > 1)) {
            root_ = i;
            // wfn_analyzer(PQ_space, PQ_evecs, nroot_);
        }
    }
    dim = PQ_space.size();
    final_wfn_.copy(PQ_space);
    PQ_space.clear();

    int froot = options_.get_int("ACI_ROOT");
    if (ex_alg_ == "ROOT_ORTHOGONALIZE") {
        froot = nroot_ - 1;
        multistate_pt2_energy_correction_ = pt2_energies;
        PQ_evals = energies;
    }

    WFNOperator op_c(mo_symmetry_);
    if (ex_alg_ == "ROOT_COMBINE") {
        outfile->Printf("\n\n  ==> Diagonalizing Final Space <==");
        dim = full_space.size();

        for (int n = 0; n < nroot_; ++n) {
            outfile->Printf("\n  Determinants for root %d: %zu", n, sizes[n]);
        }

        outfile->Printf("\n  Size of combined space: %zu", dim);

        op_c.build_strings(full_space);
        op_c.op_lists(full_space);
        op_c.tp_lists(full_space);

        SparseCISolver sparse_solver;
        sparse_solver.set_parallel(true);
        sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
        sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
        sparse_solver.set_spin_project_full(project_out_spin_contaminants_);
        sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
        // sparse_solver.set_spin_project_full(false);
        sparse_solver.diagonalize_hamiltonian_map(full_space, op_c, PQ_evals, PQ_evecs, nroot_,
                                                  multiplicity_, diag_method_);
    }

    if (ex_alg_ == "MULTISTATE") {
        Timer multi;
        compute_multistate(PQ_evals);
        outfile->Printf("\n  Time spent computing multistate solution: %1.5f s", multi.get());
        //    PQ_evals->print();
    }

    std::string sigma_method = options_.get_str("SIGMA_BUILD_TYPE");
    if (options_.get_bool("ACI_ADD_SINGLES")) {

        outfile->Printf("\n  Adding singles");

        op_.add_singles(final_wfn_);
        if (sigma_method == "HZ") {
            op_.clear_op_lists();
            op_.clear_tp_lists();
            Timer str;
            op_.build_strings(final_wfn_);
            outfile->Printf("\n  Time spent building strings      %1.6f s", str.get());
            op_.op_lists(final_wfn_);
            op_.tp_lists(final_wfn_);
        } else {
            op_.clear_op_s_lists();
            op_.clear_tp_s_lists();
            op_.build_strings(final_wfn_);
            op_.op_s_lists(final_wfn_);
            op_.tp_s_lists(final_wfn_);
        }

        SparseCISolver sparse_solver;
        sparse_solver.set_parallel(true);
        sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
        sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
        sparse_solver.set_spin_project_full(project_out_spin_contaminants_);
        sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
        sparse_solver.diagonalize_hamiltonian_map(final_wfn_, op_, PQ_evals, PQ_evecs, nroot_,
                                                  multiplicity_, diag_method_);
    }

    //** Optionally compute full PT2 energy **//
    if (options_.get_bool("MRPT2")) {
        MRPT2 pt(reference_wavefunction_, options_, ints_, mo_space_info_, final_wfn_, PQ_evecs,
                 PQ_evals);

        multistate_pt2_energy_correction_[0] = pt.compute_energy();
    }

    // if (!quiet_mode_) {
    if (ex_alg_ == "ROOT_COMBINE") {
        print_final(full_space, PQ_evecs, PQ_evals);
    } else if (ex_alg_ == "ROOT_ORTHOGONALIZE" and nroot_ > 1) {
        print_final(final_wfn_, PQ_evecs, energies);
    } else {
        print_final(final_wfn_, PQ_evecs, PQ_evals);
    }
    //  }

    //** Compute the RDMs **//

    if (options_.get_int("ACI_MAX_RDM") >= 3 or (rdm_level_ >= 3)) {
        op_.three_s_lists(final_wfn_);
    }
    SharedMatrix new_evecs;
    if (ex_alg_ == "ROOT_COMBINE") {
        compute_rdms(full_space, op_c, PQ_evecs, 0, 0);
    } else if (approx_rdm_) {
        DeterminantMap approx = approximate_wfn(final_wfn_, PQ_evecs, external_wfn_, new_evecs);
        //    WFNOperator op1(mo_space_info_);
        //    op1.op_lists(approx);
        op_.clear_op_lists();
        op_.clear_tp_lists();
        op_.build_strings(approx);
        op_.op_lists(approx);
        outfile->Printf("\n  Size of approx: %zu  size of var: %zu", approx.size(),
                        final_wfn_.size());
        compute_rdms(approx, op_, new_evecs, 0, 0);
    } else {

        op_.clear_op_s_lists();
        op_.clear_tp_s_lists();
        op_.op_s_lists(final_wfn_);
        op_.tp_s_lists(final_wfn_);
        compute_rdms(final_wfn_, op_, PQ_evecs, 0, 0);
    }

    outfile->Flush();
    //	std::vector<double> davidson;
    //	if(options_.get_str("SIZE_CORRECTION") == "DAVIDSON" ){
    //		davidson = davidson_correction( P_space_ , P_evals, PQ_evecs,
    // PQ_space_, PQ_evals );
    //	for( auto& i : davidson ){
    //		outfile->Printf("\n Davidson corr: %1.9f", i);
    //	}}

    double root_energy =
        PQ_evals->get(froot) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
    double root_energy_pt2 = root_energy + multistate_pt2_energy_correction_[froot];

    Process::environment.globals["CURRENT ENERGY"] = root_energy;
    Process::environment.globals["ACI ENERGY"] = root_energy;
    Process::environment.globals["ACI+PT2 ENERGY"] = root_energy_pt2;
    //  if(!quiet_mode_){
    outfile->Printf("\n\n  %s: %f s", "Adaptive-CI (bitset) ran in ", aci_elapse.get());
    outfile->Printf("\n\n  %s: %d", "Saving information for root", options_.get_int("ACI_ROOT"));
    //  }

    // printf( "\n%1.5f\n", aci_elapse.get());

    if (options_.get_bool("UNPAIRED_DENSITY")) {
        UPDensity density(reference_wavefunction_, mo_space_info_);
        density.compute_unpaired_density(ordm_a_, ordm_b_);
    }

    return PQ_evals->get(options_.get_int("ACI_ROOT")) + nuclear_repulsion_energy_ +
           fci_ints_->scalar_energy();
}

void AdaptiveCI::diagonalize_final_and_compute_rdms() {
    print_h2("Diagonalizing ACI Hamiltonian");
    //    reference_determinant_.print();
    //    outfile->Printf("\n  REFERENCE ENERGY:         %1.12f",
    //                    reference_determinant_.energy() +
    //                        nuclear_repulsion_energy_ +
    //                        fci_ints_->scalar_energy());

    SharedMatrix final_evecs;
    SharedVector final_evals;

    op_.clear_op_s_lists();
    op_.clear_tp_s_lists();
    op_.build_strings(final_wfn_);
    op_.op_s_lists(final_wfn_);
    op_.tp_s_lists(final_wfn_);

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(project_out_spin_contaminants_);
    //   sparse_solver.set_spin_project_full(project_out_spin_contaminants_);
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_spin_project_full(false);
    sparse_solver.diagonalize_hamiltonian_map(final_wfn_, op_, final_evals, final_evecs, nroot_,
                                              multiplicity_, diag_method_);

    print_final(final_wfn_, final_evecs, final_evals);

    op_.clear_op_s_lists();
    op_.clear_tp_s_lists();
    op_.op_lists(final_wfn_);
    op_.tp_lists(final_wfn_);
    op_.three_s_lists(final_wfn_);

    compute_rdms(final_wfn_, op_, final_evecs, 0, 0);
}

DeterminantMap AdaptiveCI::get_wavefunction() { return final_wfn_; }

void AdaptiveCI::print_final(DeterminantMap& dets, SharedMatrix& PQ_evecs, SharedVector& PQ_evals) {
    size_t dim = dets.size();
    // Print a summary
    outfile->Printf("\n\n  ==> ACI Summary <==\n");

    outfile->Printf("\n  Iterations required:                         %zu", cycle_);
    outfile->Printf("\n  Dimension of optimized determinant space:    %zu\n", dim);

    for (int i = 0; i < nroot_; ++i) {
        double abs_energy =
            PQ_evals->get(i) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
        double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
        outfile->Printf("\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV", i,
                        abs_energy, exc_energy);
        outfile->Printf("\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV", i,
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
        outfile->Printf("\n\n  Energy optimized for Root %d: %.12f Eh", ref_root_,
                        PQ_evals->get(ref_root_) + nuclear_repulsion_energy_ +
                            fci_ints_->scalar_energy());
        outfile->Printf("\n\n  Root %d Energy + PT2:         %.12f Eh", ref_root_,
                        PQ_evals->get(ref_root_) + nuclear_repulsion_energy_ +
                            fci_ints_->scalar_energy() +
                            multistate_pt2_energy_correction_[ref_root_]);
    }

    if ((ex_alg_ != "ROOT_ORTHOGONALIZE") or (nroot_ == 1)) {
        outfile->Printf("\n\n  ==> Wavefunction Information <==");

        print_wfn(dets, PQ_evecs, nroot_);

        //         outfile->Printf("\n\n     Order		 # of Dets        Total
        //         |c^2|");
        //         outfile->Printf(  "\n  __________ 	____________
        //         ________________ ");
        //         wfn_analyzer(dets, PQ_evecs, nroot_);
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

void AdaptiveCI::default_find_q_space(DeterminantMap& P_space, DeterminantMap& PQ_space,
                                      SharedVector evals, SharedMatrix evecs) {
    Timer build;

    // This hash saves the determinant coupling to the model space eigenfunction
    det_hash<std::vector<double>> V_hash;

    // Get the excited Determinants
    get_excited_determinants(nroot_, evecs, P_space, V_hash);

    // This will contain all the determinants
    PQ_space.clear();
    external_wfn_.clear();
    // Add the P-space determinants and zero the hash
    det_hash<size_t> detmap = P_space.wfn_hash();
    for (det_hash<size_t>::iterator it = detmap.begin(), endit = detmap.end(); it != endit; ++it) {
        PQ_space.add(it->first);
        V_hash.erase(it->first);
    }

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the SD space", V_hash.size());
        outfile->Printf("\n  %s: %f s\n", "Time spent building the model space (default)",
                        build.get());
    }
    outfile->Flush();

    Timer screen;

    // Compute criteria for all dets, store them all
    std::vector<std::pair<double, STLBitsetDeterminant>> sorted_dets(V_hash.size());
    //    int ithread = omp_get_thread_num();
    //    int nthreads = omp_get_num_threads();

    size_t max = V_hash.size();
#pragma omp parallel
    {
        int num_thread = omp_get_max_threads();
        int tid = omp_get_thread_num();

        size_t N = 0;
        for (const auto& I : V_hash) {

            if ((N % num_thread) == tid) {
                double delta = I.first.energy() - evals->get(0);
                double V = I.second[0];

                double criteria = 0.5 * (delta - sqrt(delta * delta + V * V * 4.0));
                sorted_dets[N] = std::make_pair(std::fabs(criteria), I.first);
            }
            N++;
        }
    }
    std::sort(sorted_dets.begin(), sorted_dets.end(), pairComp);
    std::vector<double> ept2(nroot_, 0.0);

    double sum = 0.0;
    size_t last_excluded = 0;
    for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I) {
        double& energy = sorted_dets[I].first;
        STLBitsetDeterminant& det = sorted_dets[I].second;
        if (sum + energy < sigma_) {
            sum += energy;
            ept2[0] -= energy;
            last_excluded = I;

            // Optionally save an approximate external wfn
            if (approx_rdm_) {
                external_wfn_[det] = V_hash[det][0] / (evals->get(0) - det.energy());
            }
        } else {
            PQ_space.add(det);
        }
    }
    // Add missing determinants
    if (add_aimed_degenerate_) {
        size_t num_extra = 0;
        for (size_t I = 0, max_I = last_excluded; I < max_I; ++I) {
            size_t J = last_excluded - I;
            if (std::fabs(sorted_dets[last_excluded + 1].first - sorted_dets[J].first) < 1.0e-9) {
                PQ_space.add(sorted_dets[J].second);
                num_extra++;
            } else {
                break;
            }
        }
        if (num_extra > 0 and (!quiet_mode_)) {
            outfile->Printf("\n  Added %zu missing determinants in aimed selection.", num_extra);
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the P + Q space",
                        PQ_space.size());
        outfile->Printf("\n  %s: %f s", "Time spent screening the model space", screen.get());
    }
    outfile->Flush();
}

void AdaptiveCI::find_q_space(DeterminantMap& P_space, DeterminantMap& PQ_space, int nroot,
                              SharedVector evals, SharedMatrix evecs) {
    Timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    det_hash<std::vector<double>> V_hash;
    get_excited_determinants(nroot_, evecs, P_space, V_hash);

    if (!quiet_mode_) {
        outfile->Printf("\n  %s: %zu determinants", "Dimension of the SD space", V_hash.size());
        outfile->Printf("\n  %s: %f s\n", "Time spent building the model space", t_ms_build.get());
    }
    outfile->Flush();

    // This will contain all the determinants
    PQ_space.clear();

    // Add the P-space determinants and zero the hash
    PQ_space.copy(P_space);

    Timer t_ms_screen;

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

    std::vector<std::pair<double, STLBitsetDeterminant>> sorted_dets;
    std::vector<double> ept2(nroot_, 0.0);

    if (aimed_selection_) {
        sorted_dets.resize(V_hash.size());
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
                // Optionally save an approximate external wfn
                if (approx_rdm_) {
                    external_wfn_[det] = V_hash[det][0] / (evals->get(0) - det.energy());
                }
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
    outfile->Flush();
}

double AdaptiveCI::average_q_values(int nroot, std::vector<double>& C1, std::vector<double>& E2) {
    // f_E2 and f_C1 will store the selected function of the chosen q criteria
    // This functions should only be called when nroot_ > 1

    int nav = options_.get_int("ACI_N_AVERAGE");
    int off = options_.get_int("ACI_AVERAGE_OFFSET");
    if (nav == 0)
        nav = nroot;
    if ((off + nav) > nroot)
        off = nroot - nav; // throw PSIEXCEPTION("\n  Your desired number of
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
        throw PSIEXCEPTION("\n  Your selection is not valid. Check ROOT in options.");
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
void AdaptiveCI::get_excited_determinants2(int nroot, SharedMatrix evecs, DeterminantMap& P_space,
                                           det_hash<std::vector<double>>& V_hash) {
    const size_t n_dets = P_space.size();

    int nmo = STLBitsetDeterminant::nmo_;
    double max_mem = options_.get_double("PT2_MAX_MEM");

    size_t guess_size = n_dets * nmo * nmo;
    double nbyte = (1073741824 * max_mem) / (sizeof(double));

    int nbin = static_cast<int>(std::ceil(guess_size / (nbyte)));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int ntds = omp_get_num_threads();

        if ((ntds > nbin)) {
            nbin = ntds;
        }

        if (tid == 0) {
            outfile->Printf("\n  Number of bins for exitation space:  %d", nbin);
            outfile->Printf("\n  Number of threads: %d", ntds);
        }
        size_t bin_size = n_dets / ntds;
        bin_size += (tid < (n_dets % ntds)) ? 1 : 0;
        size_t start_idx = (tid < (n_dets % ntds)) ? tid * bin_size
                                                   : (n_dets % ntds) * (bin_size + 1) +
                                                         (tid - (n_dets % ntds)) * bin_size;
        size_t end_idx = start_idx + bin_size;
        for (int bin = 0; bin < nbin; ++bin) {

            det_hash<std::vector<double>> A_I;
            // std::vector<std::pair<STLBitsetDeterminant, std::vector<double>>> A_I;

            const std::vector<STLBitsetDeterminant>& dets = P_space.determinants();
            for (size_t I = start_idx; I < end_idx; ++I) {
                double c_norm = evecs->get_row(0, I)->norm();
                const STLBitsetDeterminant& det = dets[I];
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
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if (P_space.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = std::hash<bit_t>()(new_det.bits_);
                            if ((hash_val % nbin) == bin) {
                                double HIJ = new_det.slater_rules_single_alpha(ii, aa);
                                if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                    std::vector<double> coupling(nroot_);
                                    for (int n = 0; n < nroot_; ++n) {
                                        coupling[n] = HIJ * evecs->get(I, n);
                                        if (A_I.find(new_det) != A_I.end()) {
                                            coupling[n] += A_I[new_det][n];
                                        }
                                        A_I[new_det] = coupling;
                                    }
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
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if (P_space.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = std::hash<bit_t>()(new_det.bits_);
                            if ((hash_val % nbin) == bin) {
                                double HIJ = new_det.slater_rules_single_beta(ii, aa);
                                if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                    std::vector<double> coupling(nroot_);
                                    for (int n = 0; n < nroot_; ++n) {
                                        coupling[n] = HIJ * evecs->get(I, n);
                                        if (A_I.find(new_det) != A_I.end()) {
                                            coupling[n] += A_I[new_det][n];
                                        }
                                        A_I[new_det] = coupling;
                                    }
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
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_aa(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = std::hash<bit_t>()(new_det.bits_);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
                                            }
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
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_bb(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = std::hash<bit_t>()(new_det.bits_);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
                                            }
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
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_ab(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = std::hash<bit_t>()(new_det.bits_);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
                                            }
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
                for (auto& dpair : A_I) {
                    const std::vector<double>& coupling = dpair.second;
                    const STLBitsetDeterminant& det = dpair.first;
                    if (V_hash.count(det) != 0) {
                        for (int n = 0; n < nroot; ++n) {
                            V_hash[det][n] += coupling[n];
                        }
                    } else {
                        V_hash[det] = coupling;
                    }
                }
            }
            outfile->Printf("\n TD, bin, size of hash: %d %d %zu", tid, bin, A_I.size());
        }
    }
}

void AdaptiveCI::get_excited_determinants(int nroot, SharedMatrix evecs, DeterminantMap& P_space,
                                          det_hash<std::vector<double>>& V_hash) {
    size_t max_P = P_space.size();
    std::vector<STLBitsetDeterminant> P_dets = P_space.determinants();

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

        if (omp_get_thread_num() == 0 and !quiet_mode_) {
            outfile->Printf("\n  Using %d threads.", num_thread);
        }
        // This will store the excited determinant info for each thread
        std::vector<std::pair<STLBitsetDeterminant, std::vector<double>>>
            thread_ex_dets; //( noalpha * nvalpha  );

        for (size_t P = start_idx; P < end_idx; ++P) {
            STLBitsetDeterminant& det(P_dets[P]);
            double evecs_P_row_norm = evecs->get_row(0, P)->norm();

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
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            //      if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
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
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
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
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
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
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i *
                                        // noalpha*noalpha*nvalpha +
                                        // j*nvalpha*noalpha +  a*nvalpha + b ]
                                        // = std::make_pair(new_det,coupling);
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i * nobeta * nvalpha
                                        // *nvbeta + j * bvalpha * nvbeta + a *
                                        // nvalpha]
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
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
                                 (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    // if( std::abs(HIJ * evecs->get(0, P)) >= screen_thresh_ ){
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
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

bool AdaptiveCI::check_convergence(std::vector<std::vector<double>>& energy_history,
                                   SharedVector evals) {
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
    return (std::fabs(new_avg_energy - old_avg_energy) < options_.get_double("ACI_CONVERGENCE"));
}

void AdaptiveCI::prune_q_space(DeterminantMap& PQ_space, DeterminantMap& P_space,
                               SharedMatrix evecs, int nroot) {
    // Select the new reference space using the sorted CI coefficients
    P_space.clear();

    double tau_p = sigma_ * gamma_;

    int nav = options_.get_int("ACI_N_AVERAGE");
    int off = options_.get_int("ACI_AVERAGE_OFFSET");
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
    for (det_hash<size_t>::iterator it = detmap.begin(), endit = detmap.end(); it != endit; ++it) {
        double criteria = 0.0;
        if (ex_alg_ == "AVERAGE") {
            for (int n = 0; n < nav; ++n) {
                if (pq_function_ == "MAX") {
                    criteria = std::max(criteria, std::fabs(evecs->get(it->second, n)));
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

bool AdaptiveCI::check_stuck(std::vector<std::vector<double>>& energy_history, SharedVector evals) {
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
            //			std::fabs( av_energies[cycle_-2] -
            // av_energies[cycle_ - 4]
            //)
            //< options_.get_double("ACI_CONVERGENCE") ){
            stuck = true;
        }
    }
    return stuck;
}

std::vector<std::pair<double, double>> AdaptiveCI::compute_spin(DeterminantMap& space,
                                                                SharedMatrix evecs, int nroot) {
    // WFNOperator op(mo_symmetry_);

    // op.build_strings(space);
    // op.op_lists(space);
    // op.tp_lists(space);
    if (options_.get_str("SIGMA_BUILD_TYPE") == "HZ") {
        op_.clear_op_s_lists();
        op_.clear_tp_s_lists();
        op_.build_strings(space);
        op_.op_lists(space);
        op_.tp_lists(space);
    }

    std::vector<std::pair<double, double>> spin_vec(nroot);
    for (int n = 0; n < nroot_; ++n) {
        double S2 = op_.s2(space, evecs, n);
        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        spin_vec[n] = make_pair(S, S2);
    }

    return spin_vec;
}

/*
void AdaptiveCI::wfn_analyzer(DeterminantMap& det_space, SharedMatrix evecs, int nroot) {

    std::vector<bool> occ(2 * nact_, 0);
    std::vector<std::tuple<double, int, int>> labeled_orb_en = sym_labeled_orbitals("RHF");
    for (int i = 0; i < nalpha_; ++i) {
        occ[std::get<2>(labeled_orb_en[i])] = 1;
    }
    for (int i = 0; i < nbeta_; ++i) {
        occ[nact_ + std::get<2>(labeled_orb_en[i])] = 1;
    }

    //     bool print_final_wfn = options_.get_bool("SAVE_FINAL_WFN");

    //  std::ofstream final_wfn;
    //  if( print_final_wfn ){
    //      final_wfn.open("final_wfn_"+ std::to_string(root_) +  ".txt");
    //      final_wfn << det_space.size() << "  " << nact_ << "  " << nalpha_ <<
    //      "  " << nbeta_ << endl;
    //  }
    outfile->Printf("\n  ndet: %zu", det_space.size());

    STLBitsetDeterminant rdet(occ);
    auto ref_bits = rdet.bits();
    int max_ex = 1 + (cycle_ + 1) * 2;
    for (int n = 0; n < nroot; ++n) {
        std::vector<double> c2_vals(max_ex, 0.0);
        std::vector<size_t> ndet(max_ex, 0);

        //        det_hash<size_t> detmap = det_space.wfn_hash();
        //        for (det_hash<size_t>::iterator it = detmap.begin(),
        //                                        endit = detmap.end();
        //             it != endit; ++it) {

        const std::vector<STLBitsetDeterminant>& dets = det_space.determinants();
        for (size_t I = 0, maxI = det_space.size(); I < maxI; ++I) {
            int ndiff = 0;
            auto ex_bits = dets[I].bits();

            double coeff = evecs->get(I, n) * evecs->get(I, n);

            // Compute number of differences in both alpha and beta strings wrt
            // ref
            for (size_t a = 0; a < nact_ * 2; ++a) {
                if (ref_bits[a] != ex_bits[a]) {
                    ++ndiff;
                }
            }
            ndiff = static_cast<int>(ndiff / 2.0);
            c2_vals[ndiff] += coeff;
            ndet[ndiff]++;
            //            std::make_pair(excitation_counter[ndiff].first + 1,
            //                           excitation_counter[ndiff].second + coeff);

            //         if( print_final_wfn and (n == root_) ){

            //             auto abits =
            //             it->first.get_alfa_bits_vector_bool();
            //             auto bbits =
            //             it->first.get_beta_bits_vector_bool();

            //             final_wfn << std::setw(18) << std::setprecision(12)
            //             <<  evecs->get(it->second,n) << "  ";// <<  abits << "  "
            //             << bbits <<
            //             it->first.str().c_str() << endl;
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
        for (int i = 0, maxi = c2_vals.size(); i < maxi; ++i) {
            outfile->Printf("\n       %d        %8zu           %.11f", i, ndet[i], c2_vals[i]);
        }
    }
    //  if( print_final_wfn ) final_wfn.close();
    //  outfile->Flush();
}
*/

void AdaptiveCI::print_wfn(DeterminantMap& space, SharedMatrix evecs, int nroot) {
    std::string state_label;
    std::vector<string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet", "sextet",
                                   "septet", "octet", "nonet", "decatet"});

    std::vector<std::pair<double, double>> spins = compute_spin(space, evecs, nroot);

    for (int n = 0; n < nroot; ++n) {
        DeterminantMap tmp;
        std::vector<double> tmp_evecs;

        outfile->Printf("\n\n  Most important contributions to root %3d:", n);

        size_t max_dets = std::min(10, evecs->nrow());
        tmp.subspace(space, evecs, tmp_evecs, max_dets, n);

        for (size_t I = 0; I < max_dets; ++I) {
            outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s", I, tmp_evecs[I],
                            tmp_evecs[I] * tmp_evecs[I], space.get_idx(tmp.get_det(I)),
                            tmp.get_det(I).str().c_str());
        }

        state_label = s2_labels[std::round(spins[n].first * 2.0)];
        root_spin_vec_.clear();
        root_spin_vec_[n] = make_pair(spins[n].first, spins[n].second);
        outfile->Printf("\n\n  Spin state for root %zu: S^2 = %5.6f, S = %5.3f, %s", n,
                        root_spin_vec_[n].first, root_spin_vec_[n].second, state_label.c_str());
    }
    outfile->Flush();
}

void AdaptiveCI::full_spin_transform(DeterminantMap& det_space, SharedMatrix cI, int nroot) {
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
    //	//double criteria = static_cast<double>(vefunction_multiplicity_) -
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

double AdaptiveCI::compute_spin_contamination(DeterminantMap& space, SharedMatrix evecs,
                                              int nroot) {
    auto spins = compute_spin(space, evecs, nroot);
    double spin_contam = 0.0;
    for (int n = 0; n < nroot; ++n) {
        spin_contam += spins[n].second;
    }
    spin_contam /= static_cast<double>(nroot);
    spin_contam -= (0.25 * (multiplicity_ * multiplicity_ - 1.0));

    return spin_contam;
}

void AdaptiveCI::save_dets_to_file(DeterminantMap& space, SharedMatrix evecs) {
    // Use for single-root calculations only
    det_hash<size_t> detmap = space.wfn_hash();
    for (det_hash<size_t>::iterator it = detmap.begin(), endit = detmap.end(); it != endit; ++it) {
        det_list_ << it->first.str().c_str() << " " << fabs(evecs->get(it->second, 0)) << " ";
        //	for(size_t J = 0, maxJ = space.size(); J < maxJ; ++J){
        //		det_list_ << space[I].slater_rules(space[J]) << " ";
        //	}
        //	det_list_ << "\n";
    }
    det_list_ << "\n";
}

std::vector<double> AdaptiveCI::davidson_correction(std::vector<STLBitsetDeterminant>& P_dets,
                                                    SharedVector P_evals, SharedMatrix PQ_evecs,
                                                    std::vector<STLBitsetDeterminant>& PQ_dets,
                                                    SharedVector PQ_evals) {
    outfile->Printf("\n  There are %zu PQ dets.", PQ_dets.size());
    outfile->Printf("\n  There are %zu P dets.", P_dets.size());

    // The energy correction per root
    std::vector<double> dc(nroot_, 0.0);

    std::unordered_map<STLBitsetDeterminant, double, STLBitsetDeterminant::Hash> PQ_map;
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
    Reference aci_ref = ci_rdms.reference(ordm_a_, ordm_b_, trdm_aa_, trdm_ab_, trdm_bb_, trdm_aaa_,
                                          trdm_aab_, trdm_abb_, trdm_bbb_);
    return aci_ref;
}

void AdaptiveCI::print_nos() {
    print_h2("NATURAL ORBITALS");

    std::shared_ptr<Matrix> opdm_a(new Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
    std::shared_ptr<Matrix> opdm_b(new Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v, ordm_a_[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v, ordm_b_[(u + offset) * nact_ + v + offset]);
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

    // ofstream file;
    // file.open("nos.txt",std::ios_base::app);
    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            auto irrep_occ =
                std::make_pair(OCC_A->get(h, u) + OCC_B->get(h, u), std::make_pair(h, u + 1));
            vec_irrep_occupation.push_back(irrep_occ);
            //          file << OCC_A->get(h, u) + OCC_B->get(h, u) << "  ";
        }
    }
    // file << endl;
    // file.close();

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
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
        double no_thresh = options_.get_double("ACI_NO_THRESHOLD");

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

/*
void AdaptiveCI::convert_to_string(const std::vector<STLBitsetDeterminant>& space) {
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
*/

int AdaptiveCI::root_follow(DeterminantMap& P_ref, std::vector<double>& P_ref_evecs,
                            DeterminantMap& P_space, SharedMatrix P_evecs, int num_ref_roots) {
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
    //     outfile->Printf("\n %1.8f   %s", det_weights[I].first,
    //     P_ref.get_det(det_weights[I].second).str().c_str());
    // }

    for (int n = 0; n < num_ref_roots; ++n) {
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

void AdaptiveCI::compute_aci(DeterminantMap& PQ_space, SharedMatrix& PQ_evecs,
                             SharedVector& PQ_evals) {

    bool print_refs = false;
    bool multi_root = false;

    if (options_["ACI_FIRST_ITER_ROOTS"].has_changed()) {
        multi_root = options_.get_bool("ACI_FIRST_ITER_ROOTS");
    }

    if (options_["ACI_PRINT_REFS"].has_changed()) {
        print_refs = options_.get_bool("ACI_PRINT_REFS");
    }

    if ((options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_ORTHOGONALIZE" or
         options_.get_str("ACI_EXCITED_ALGORITHM") == "MULTISTATE" or
         options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_COMBINE") and
        root_ == 0 and !multi_root) {
        nroot_ = 1;
    }

    SharedMatrix P_evecs;
    SharedVector P_evals;

    DeterminantMap P_ref;
    std::vector<double> P_ref_evecs;
    DeterminantMap P_space(initial_reference_);

    outfile->Flush();

    size_t nvec = options_.get_int("N_GUESS_VEC");
    std::string sigma_method = options_.get_str("SIGMA_BUILD_TYPE");

    std::vector<std::vector<double>> energy_history;
    SparseCISolver sparse_solver;
    if (quiet_mode_) {
        sparse_solver.set_print_details(false);
    }
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(project_out_spin_contaminants_);
    //    sparse_solver.set_spin_project_full(project_out_spin_contaminants_);
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_num_vecs(nvec);
    sparse_solver.set_sigma_method(sigma_method);
    sparse_solver.set_spin_project_full(false);
    int spin_projection = options_.get_int("ACI_SPIN_PROJECTION");

    // if (det_save_)
    //     det_list_.open("det_list.txt");

    if (streamline_qspace_ and !quiet_mode_)
        outfile->Printf("\n  Using streamlined Q-space builder.");

    ex_alg_ = options_.get_str("ACI_EXCITED_ALGORITHM");

    std::vector<STLBitsetDeterminant> old_dets;
    SharedMatrix old_evecs;

    if (options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_SELECT") {
        ref_root_ = options_.get_int("ACI_ROOT");
    }

    // Save the P_space energies to predict convergence
    std::vector<double> P_energies;
    approx_rdm_ = false;

    int cycle;
    for (cycle = 0; cycle < max_cycle_; ++cycle) {
        Timer cycle_time;
        // Step 1. Diagonalize the Hamiltonian in the P space
        int num_ref_roots = std::min(nroot_, int(P_space.size()));
        cycle_ = cycle;
        std::string cycle_h = "Cycle " + std::to_string(cycle_);

        bool follow = false;
        if (options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_SELECT" or
            options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_COMBINE" or
            options_.get_str("ACI_EXCITED_ALGORITHM") == "MULTISTATE" or
            options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_ORTHOGONALIZE") {

            follow = true;
        }

        if (!quiet_mode_) {
            print_h2(cycle_h);
            outfile->Printf("\n  Initial P space dimension: %zu", P_space.size());
        }

        // Check that the initial space is spin-complete
        if (spin_complete_) {
            P_space.make_spin_complete();
            if (!quiet_mode_)
                outfile->Printf("\n  %s: %zu determinants",
                                "Spin-complete dimension of the P space", P_space.size());
        } else if (!quiet_mode_) {
            outfile->Printf("\n  Not checking for spin-completeness.");
        }
        // Diagonalize H in the P space
        if (ex_alg_ == "ROOT_ORTHOGONALIZE" and root_ > 0 and cycle >= pre_iter_) {
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

        if (sigma_method == "HZ") {
            op_.clear_op_lists();
            op_.clear_tp_lists();
            op_.build_strings(P_space);
            op_.op_lists(P_space);
            op_.tp_lists(P_space);
        } else {
            op_.clear_op_s_lists();
            op_.clear_tp_s_lists();
            op_.build_strings(P_space);
            op_.op_s_lists(P_space);
            op_.tp_s_lists(P_space);
        }

        sparse_solver.manual_guess(false);
        Timer diag;
        sparse_solver.diagonalize_hamiltonian_map(P_space, op_, P_evals, P_evecs, num_ref_roots,
                                                  multiplicity_, diag_method_);
        if (!quiet_mode_)
            outfile->Printf("\n  Time spent diagonalizing H:   %1.6f s", diag.get());

        // Save ground state energy
        P_energies.push_back(P_evals->get(0));

        if ((cycle > 1) and options_.get_bool("ACI_APPROXIMATE_RDM")) {
            double diff = std::abs(P_energies[cycle] - P_energies[cycle - 1]);
            if (diff <= 1e-5) {
                approx_rdm_ = true;
            }
        }

        if (cycle < pre_iter_) {
            ex_alg_ = "AVERAGE";
        } else if (cycle == pre_iter_ and follow) {
            ex_alg_ = options_.get_str("ACI_EXCITED_ALGORITHM");
        }

        // Update the reference root if root following
        if (follow and num_ref_roots > 1 and (cycle >= pre_iter_) and cycle > 0) {
            ref_root_ = root_follow(P_ref, P_ref_evecs, P_space, P_evecs, num_ref_roots);
        }

        // Use spin projection to ensure the P space is spin pure
        if ((spin_projection == 1 or spin_projection == 3) and P_space.size() <= 200) {
            project_determinant_space(P_space, P_evecs, P_evals, num_ref_roots);
        }

        // Print the energy
        if (!quiet_mode_) {
            outfile->Printf("\n");
            for (int i = 0; i < num_ref_roots; ++i) {
                double abs_energy =
                    P_evals->get(i) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
                double exc_energy = pc_hartree2ev * (P_evals->get(i) - P_evals->get(0));
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
                outfile->Printf("\n  Spin-complete dimension of the PQ space: %zu",
                                PQ_space.size());
        }

        if ((options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_ORTHOGONALIZE") and (root_ > 0) and
            cycle >= pre_iter_) {
            sparse_solver.set_root_project(true);
            add_bad_roots(PQ_space);
            sparse_solver.add_bad_states(bad_roots_);
        }

        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        if (sigma_method == "HZ") {
            op_.clear_op_lists();
            op_.clear_tp_lists();
            Timer str;
            op_.build_strings(PQ_space);
            outfile->Printf("\n  Time spent building strings      %1.6f s", str.get());
            op_.op_lists(PQ_space);
            op_.tp_lists(PQ_space);
        } else {
            op_.clear_op_s_lists();
            op_.clear_tp_s_lists();
            op_.build_strings(PQ_space);
            op_.op_s_lists(PQ_space);
            op_.tp_s_lists(PQ_space);
        }
        Timer diag_pq;

        sparse_solver.diagonalize_hamiltonian_map(PQ_space, op_, PQ_evals, PQ_evecs, num_ref_roots,
                                                  multiplicity_, diag_method_);

        if (!quiet_mode_)
            outfile->Printf("\n  Total time spent diagonalizing H:   %1.6f s", diag_pq.get());
        //  if (det_save_)
        //      save_dets_to_file(PQ_space, PQ_evecs);

        // Save the solutions for the next iteration
        //        old_dets.clear();
        //        old_dets = PQ_space_;
        //        old_evecs = PQ_evecs->clone();

        // Ensure the solutions are spin-pure
        if ((spin_projection == 1 or spin_projection == 3) and PQ_space.size() <= 200) {
            project_determinant_space(PQ_space, PQ_evecs, PQ_evals, num_ref_roots);
        }

        if (!quiet_mode_) {
            // Print the energy
            outfile->Printf("\n");
            for (int i = 0; i < num_ref_roots; ++i) {
                double abs_energy =
                    PQ_evals->get(i) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
                double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
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
            outfile->Flush();
        }

        num_ref_roots = std::min(nroot_, int(PQ_space.size()));

        // If doing root-following, grab the initial root
        if (follow and (cycle == (pre_iter_ - 1) or (pre_iter_ == 0 and cycle == 0))) {

            if (options_.get_str("ACI_EXCITED_ALGORITHM") == "ROOT_SELECT") {
                ref_root_ = options_.get_int("ACI_ROOT");
            }
            size_t dim = std::min(static_cast<int>(PQ_space.size()), 1000);
            P_ref.subspace(PQ_space, PQ_evecs, P_ref_evecs, dim, ref_root_);
        }

        // if( follow and num_ref_roots > 0 and (cycle >= (pre_iter_ - 1)) ){
        if (follow and (num_ref_roots > 1) and (cycle >= pre_iter_)) {
            ref_root_ = root_follow(P_ref, P_ref_evecs, PQ_space, PQ_evecs, num_ref_roots);
        }

        bool stuck = check_stuck(energy_history, PQ_evals);
        if (stuck and (options_.get_str("ACI_EXCITED_ALGORITHM") != "COMPOSITE")) {
            outfile->Printf("\n  Procedure is stuck! Quitting...");
            break;
        } else if (stuck and (options_.get_str("ACI_EXCITED_ALGORITHM") == "COMPOSITE") and
                   ex_alg_ == "AVERAGE") {
            outfile->Printf("\n  Root averaging algorithm converged.");
            outfile->Printf("\n  Now optimizing PQ Space for root %d",
                            options_.get_int("ACI_ROOT"));
            ex_alg_ = options_.get_str("ACI_EXCITED_ALGORITHM");
            pre_iter_ = cycle + 1;
        }

        // Step 4. Check convergence and break if needed
        bool converged = check_convergence(energy_history, PQ_evals);
        if (converged and (ex_alg_ == "AVERAGE") and
            options_.get_str("ACI_EXCITED_ALGORITHM") == "COMPOSITE") {
            outfile->Printf("\n  Root averaging algorithm converged.");
            outfile->Printf("\n  Now optimizing PQ Space for root %d",
                            options_.get_int("ACI_ROOT"));
            ex_alg_ = options_.get_str("ACI_EXCITED_ALGORITHM");
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
            outfile->Printf("\n  Cycle %d took: %1.6f s", cycle, cycle_time.get());
        }

        ex_alg_ = options_.get_str("ACI_EXCITED_ALGORITHM");
    } // end iterations

    // if (det_save_)
    //     det_list_.close();

    // Ensure the solutions are spin-pure
    if ((spin_projection == 2 or spin_projection == 3) and PQ_space.size() <= 200) {
        project_determinant_space(PQ_space, PQ_evecs, PQ_evals, nroot_);
    } else if (!quiet_mode_) {
        outfile->Printf("\n  Not performing spin projection.");
    }
}

std::vector<std::pair<size_t, double>>
AdaptiveCI::dl_initial_guess(std::vector<STLBitsetDeterminant>& old_dets,
                             std::vector<STLBitsetDeterminant>& dets, SharedMatrix& evecs,
                             int root) {
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

void AdaptiveCI::compute_rdms(DeterminantMap& dets, WFNOperator& op, SharedMatrix& PQ_evecs,
                              int root1, int root2) {

    ordm_a_.clear();
    ordm_b_.clear();

    trdm_aa_.clear();
    trdm_ab_.clear();
    trdm_bb_.clear();

    trdm_aaa_.clear();
    trdm_aab_.clear();
    trdm_abb_.clear();
    trdm_bbb_.clear();

    CI_RDMS ci_rdms_(options_, dets, fci_ints_, PQ_evecs, root1, root2);
    ci_rdms_.set_max_rdm(rdm_level_);
    if (rdm_level_ >= 1) {
        Timer one_r;
        ci_rdms_.compute_1rdm(ordm_a_, ordm_b_, op);
        outfile->Printf("\n  1-RDM  took %2.6f s (determinant)", one_r.get());

        if (options_.get_bool("ACI_PRINT_NO")) {
            print_nos();
        }
    }
    if (rdm_level_ >= 2) {
        Timer two_r;
        ci_rdms_.compute_2rdm(trdm_aa_, trdm_ab_, trdm_bb_, op);
        outfile->Printf("\n  2-RDMS took %2.6f s (determinant)", two_r.get());
    }
    if (rdm_level_ >= 3) {
        Timer tr;
        ci_rdms_.compute_3rdm(trdm_aaa_, trdm_aab_, trdm_abb_, trdm_bbb_, op);
        outfile->Printf("\n  3-RDMs took %2.6f s (determinant)", tr.get());

        if (options_.get_bool("ACI_TEST_RDMS")) {
            ci_rdms_.rdm_test(ordm_a_, ordm_b_, trdm_aa_, trdm_bb_, trdm_ab_, trdm_aaa_, trdm_aab_,
                              trdm_abb_, trdm_bbb_);
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
        std::vector<std::pair<STLBitsetDeterminant, double>>& state = old_roots_[i];

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

void AdaptiveCI::save_old_root(DeterminantMap& dets, SharedMatrix& PQ_evecs, int root) {
    std::vector<std::pair<STLBitsetDeterminant, double>> vec;

    if (!quiet_mode_ and nroot_ > 0) {
        outfile->Printf("\n  Saving root %d, ref_root is %d", root, ref_root_);
    }
    det_hash<size_t> detmap = dets.wfn_hash();
    for (auto it = detmap.begin(), endit = detmap.end(); it != endit; ++it) {
        vec.push_back(std::make_pair(it->first, PQ_evecs->get(it->second, ref_root_)));
    }
    old_roots_.push_back(vec);
    if (!quiet_mode_ and nroot_ > 0) {
        outfile->Printf("\n  Number of old roots: %zu", old_roots_.size());
    }
}

void AdaptiveCI::compute_multistate(SharedVector& PQ_evals) {
    outfile->Printf("\n  Computing multistate solution");
    int nroot = old_roots_.size();

    // Form the overlap matrix

    SharedMatrix S(new Matrix(nroot, nroot));
    S->identity();
    for (int A = 0; A < nroot; ++A) {
        std::vector<std::pair<STLBitsetDeterminant, double>>& stateA = old_roots_[A];
        size_t ndetA = stateA.size();
        for (int B = 0; B < nroot; ++B) {
            if (A == B)
                continue;
            std::vector<std::pair<STLBitsetDeterminant, double>>& stateB = old_roots_[B];
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
        std::vector<std::pair<STLBitsetDeterminant, double>>& stateA = old_roots_[A];
        size_t ndetA = stateA.size();
        for (int B = A; B < nroot; ++B) {
            std::vector<std::pair<STLBitsetDeterminant, double>>& stateB = old_roots_[B];
            size_t ndetB = stateB.size();
            double HIJ = 0.0;
            for (size_t I = 0; I < ndetA; ++I) {
                STLBitsetDeterminant& detA = stateA[I].first;
                for (size_t J = 0; J < ndetB; ++J) {
                    STLBitsetDeterminant& detB = stateB[J].first;
                    HIJ += detA.slater_rules(detB) * stateA[I].second * stateB[J].second;
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

DeterminantMap AdaptiveCI::approximate_wfn(DeterminantMap& PQ_space, SharedMatrix& evecs,
                                           det_hash<double>& external_space,
                                           SharedMatrix& new_evecs) {
    DeterminantMap new_wfn;
    new_wfn.copy(PQ_space);

    size_t n_ref = PQ_space.size();
    size_t n_external = external_space.size();
    size_t total_size = n_ref + n_external;

    outfile->Printf("\n  Size of external space: %zu", n_external);
    new_evecs.reset(new Matrix("U", total_size, 1));
    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
    for (size_t I = 0; I < n_ref; ++I) {
        double val = evecs->get(I, 0);
        new_evecs->set(I, 0, val);
        sum += val * val;
    }

    for (auto& I : external_space) {
        new_wfn.add(I.first);
        new_evecs->set(new_wfn.get_idx(I.first), 0, I.second);
        sum += I.second * I.second;
    }

    outfile->Printf("\n  Norm of approximate wfn: %1.12f", std::sqrt(sum));
    // Normalize new evecs
    sum = 1.0 / std::sqrt(sum);
    new_evecs->scale_column(0, 0, sum);

    return new_wfn;
}

void AdaptiveCI::compute_nos() {

    print_h2("ACI NO Transformation");

    Dimension nmopi = reference_wavefunction_->nmopi();
    Dimension ncmopi = mo_space_info_->get_dimension("CORRELATED");
    Dimension fdocc = mo_space_info_->get_dimension("FROZEN_DOCC");
    Dimension rdocc = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    Dimension ruocc = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    std::shared_ptr<Matrix> opdm_a(new Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
    std::shared_ptr<Matrix> opdm_b(new Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v, ordm_a_[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v, ordm_b_[(u + offset) * nact_ + v + offset]);
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

    // Build full transformation matrices from e-vecs
    Matrix Ua("Ua", nmopi, nmopi);
    Matrix Ub("Ub", nmopi, nmopi);

    Ua.identity();
    Ub.identity();

    for (int h = 0; h < nirrep_; ++h) {
        size_t irrep_offset = 0;

        // Frozen core and Restricted docc are unchanged
        irrep_offset += fdocc[h] + rdocc[h];
        ;
        // Only change the active block
        for (int p = 0; p < nactpi_[h]; ++p) {
            for (int q = 0; q < nactpi_[h]; ++q) {
                Ua.set(h, p + irrep_offset, q + irrep_offset, NO_A->get(h, p, q));
                Ub.set(h, p + irrep_offset, q + irrep_offset, NO_B->get(h, p, q));
            }
        }
    }

    // Transform the orbital coefficients
    SharedMatrix Ca = reference_wavefunction_->Ca();
    SharedMatrix Cb = reference_wavefunction_->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());

    Ca_new->gemm(false, false, 1.0, Ca, Ua, 0.0);
    Cb_new->gemm(false, false, 1.0, Cb, Ub, 0.0);

    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // Retransform the integarms in the new basis
    ints_->retransform_integrals();
}
}
} // EndNamespaces
