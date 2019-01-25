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

#include "sci.h"
#include "helpers/timer.h"
namespace forte {
SelectedCIMethod::SelectedCIMethod(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                       std::shared_ptr<MOSpaceInfo> mo_space_info,
                       std::shared_ptr<ActiveSpaceIntegrals> as_ints){}

double SelectedCIMethod::compute_energy() {
    timer energy_timer("SelectedCI:Energy");
/*
    startup();

    print_info();

    local_timer asci_elapse;

    // The eigenvalues and eigenvectors
    psi::SharedMatrix PQ_evecs;
    psi::SharedVector PQ_evals;

    // Compute wavefunction and energy
    DeterminantHashVec full_space;
    std::vector<size_t> sizes(nroot_);
    psi::SharedVector energies(new Vector(nroot_));

    DeterminantHashVec PQ_space;

    psi::SharedMatrix P_evecs;
    psi::SharedVector P_evals;

    // Set the P space dets
    DeterminantHashVec P_ref;
    std::vector<double> P_ref_evecs;
    DeterminantHashVec P_space(initial_reference_);

    size_t nvec = options_->get_int("N_GUESS_VEC");
    std::string sigma_method = options_->get_str("SIGMA_BUILD_TYPE");
    std::vector<std::vector<double>> energy_history;
    SparseCISolver sparse_solver(as_ints_);
    sparse_solver.set_parallel(true);
    sparse_solver.set_force_diag(options_->get_bool("FORCE_DIAG_METHOD"));
    sparse_solver.set_e_convergence(options_->get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(true);
    sparse_solver.set_guess_dimension(options_->get_int("DL_GUESS_SIZE"));
    sparse_solver.set_num_vecs(nvec);
    sparse_solver.set_sigma_method(sigma_method);
    sparse_solver.set_spin_project_full(false);
    sparse_solver.set_max_memory(options_->get_int("SIGMA_VECTOR_MAX_MEMORY"));

    // Save the P_space energies to predict convergence
    std::vector<double> P_energies;

    int cycle;
    for (cycle = 0; cycle < max_cycle_; ++cycle) {
        local_timer cycle_time;

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
        local_timer diag;
        sparse_solver.diagonalize_hamiltonian_map(P_space, op_, P_evals, P_evecs, nroot_,
                                                  multiplicity_, diag_method_);
        outfile->Printf("\n  Time spent diagonalizing H:   %1.6f s", diag.get());

        P_energies.push_back(P_evals->get(0));

        // Print the energy
        outfile->Printf("\n");
        double P_abs_energy =
            P_evals->get(0) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
        outfile->Printf("\n    P-space  CI Energy Root 0       = "
                        "%.12f ",
                        P_abs_energy);
        outfile->Printf("\n");

        // Step 2. Find determinants in the Q space
        local_timer build_space;
        find_q_space(P_space, PQ_space, P_evals, P_evecs);
        outfile->Printf("\n  Time spent building the model space: %1.6f", build_space.get());

        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        if (sigma_method == "HZ") {
            op_.clear_op_lists();
            op_.clear_tp_lists();
            local_timer str;
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
        local_timer diag_pq;

        sparse_solver.diagonalize_hamiltonian_map(PQ_space, op_, PQ_evals, PQ_evecs, nroot_,
                                                  multiplicity_, diag_method_);

        outfile->Printf("\n  Total time spent diagonalizing H:   %1.6f s", diag_pq.get());

        // Print the energy
        outfile->Printf("\n");
        double abs_energy =
            PQ_evals->get(0) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
        outfile->Printf("\n    PQ-space CI Energy Root 0        = "
                        "%.12f Eh",
                        abs_energy);
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

    double root_energy = PQ_evals->get(0) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();

    energies_.resize(nroot_,0.0);
    for( int n = 0; n < nroot_; ++n ){
        energies_[n] = PQ_evals->get(n) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
    }


    psi::Process::environment.globals["CURRENT ENERGY"] = root_energy;
    psi::Process::environment.globals["ASCI ENERGY"] = root_energy;

    outfile->Printf("\n\n  %s: %f s", "ASCI ran in ", asci_elapse.get());

    double pt2 = 0.0;
    //  if (options_->get_bool("MRPT2")) {
    //      MRPT2 pt(reference_wavefunction_, options_, ints_, mo_space_info_, PQ_space, PQ_evecs,
    //               PQ_evals);
    //      pt2 = pt.compute_energy();
    //  }

    size_t dim = PQ_space.size();
    // Print a summary
    outfile->Printf("\n\n  ==> ASCI Summary <==\n");

    outfile->Printf("\n  Iterations required:                         %zu", cycle);
    outfile->Printf("\n  psi::Dimension of optimized determinant space:    %zu\n", dim);
    outfile->Printf("\n  * AS-CI Energy Root 0        = %.12f Eh", root_energy);
    if (options_->get_bool("MRPT2")) {
        outfile->Printf("\n  * AS-CI+PT2 Energy Root 0    = %.12f Eh", root_energy + pt2);
    }

    outfile->Printf("\n\n  ==> Wavefunction Information <==");

    print_wfn(PQ_space, op_, PQ_evecs, nroot_);

//    compute_rdms(as_ints_, PQ_space, op_, PQ_evecs, 0, 0);

    return root_energy;
    */
    return 0.0;
}

} // namespace forte
