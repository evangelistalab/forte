/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#pragma once

#include "sci/sci.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "helpers/timer.h"

namespace forte {

class RDMs;

/**
 * @brief The ASCI class
 * This class implements the Adaptively Selected CI algorithm
 */
class ASCI : public SelectedCIMethod {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */

    ASCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
         std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
         std::shared_ptr<ActiveSpaceIntegrals> as_ints);
    /// Destructor
    ~ASCI();

    // ==> Class Interface <==

    void set_options(std::shared_ptr<ForteOptions>) override{}; // TODO : define

    /// Get the wavefunction
    DeterminantHashVec get_wavefunction();

    /// Compute the ACI-NOs
    void compute_nos();

    void set_fci_ints(std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

    void pre_iter_preparation() override;
    void diagonalize_P_space() override;
    void diagonalize_PQ_space() override;
    void post_iter_process() override;

    void set_method_variables(
        std::string ex_alg, size_t nroot_method, size_t root,
        const std::vector<std::vector<std::pair<Determinant, double>>>& old_roots) override;

    DeterminantHashVec get_PQ_space() override;
    std::shared_ptr<psi::Matrix> get_PQ_evecs() override;
    std::shared_ptr<psi::Vector> get_PQ_evals() override;

    //    std::shared_ptr<WFNOperator> get_op() override;

    size_t get_ref_root() override;

    /// Check if the procedure has converged
    bool check_convergence() override;

    /// Find all the relevant excitations out of the P space
    void find_q_space() override;

    std::vector<double> get_multistate_pt2_energy_correction() override;

  private:
    // ==> Class data <==

    DeterminantHashVec final_wfn_;
    // Temporarily added
    std::shared_ptr<psi::Matrix> P_evecs_;
    std::shared_ptr<psi::Vector> P_evals_;
    DeterminantHashVec P_space_;
    DeterminantHashVec P_ref_;
    std::vector<double> P_ref_evecs_;
    std::vector<double> P_energies_;
    std::vector<std::vector<double>> energy_history_;
    size_t ref_root_;
    size_t root_;
    std::string ex_alg_;
    int num_ref_roots_;
    bool follow_;
    local_timer cycle_time_;

    // Temporarily added interface to ExcitedStateSolver
    std::shared_ptr<psi::Matrix> PQ_evecs_;
    std::shared_ptr<psi::Vector> PQ_evals_;
    DeterminantHashVec PQ_space_;
    /// Roots to project out
    std::vector<std::vector<std::pair<size_t, double>>> bad_roots_;
    /// Storage of past roots
    std::vector<std::vector<std::pair<Determinant, double>>> old_roots_;

    /// The reference determinant
    std::vector<Determinant> initial_reference_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    /// The last iteration
    bool set_ints_ = false;

    // ==> ASCI Options <==
    /// The threshold applied to the primary space
    int c_det_;
    /// The threshold applied to the secondary space
    int t_det_;

    /// Compute 1-RDM?
    bool compute_rdms_;
    /// The CI coeffiecients
    std::shared_ptr<psi::Matrix> evecs_;

    bool build_lists_;
    bool print_weights_ = false;

    /// Order of RDM to compute
    int rdm_level_ = 1;

    /// Timing variables
    double build_H_;
    double diag_H_;
    double build_space_;
    double screen_space_;
    double spin_trans_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Print information about this calculation
    void print_info() override;

    // Optimized for a single root
    void get_excited_determinants_sr(std::shared_ptr<psi::Matrix> evecs,
                                     DeterminantHashVec& P_space, det_hash<double>& V_hash);

    /// Prune the space of determinants
    void prune_PQ_to_P() override;

    /// Print natural orbitals
    void print_nos();

    /// Compute the RDMs
    void compute_rdms(std::shared_ptr<ActiveSpaceIntegrals> fci_ints, DeterminantHashVec& dets,
                      DeterminantSubstitutionLists& op, std::shared_ptr<psi::Matrix>& PQ_evecs,
                      int root1, int root2, int max_level);

    void add_bad_roots(DeterminantHashVec& dets);

    int root_follow(DeterminantHashVec& P_ref, std::vector<double>& P_ref_evecs,
                    DeterminantHashVec& P_space, std::shared_ptr<psi::Matrix> P_evecs,
                    int num_ref_roots);
};

} // namespace forte
