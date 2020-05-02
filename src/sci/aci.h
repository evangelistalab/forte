/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _aci_h_
#define _aci_h_

#include <fstream>
#include <iomanip>

#include "sci/sci.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "helpers/timer.h"

using d1 = std::vector<double>;
using d2 = std::vector<d1>;

namespace forte {

class Reference;

/**
 * @brief The AdaptiveCI class
 * This class implements an adaptive CI algorithm
 */
class AdaptiveCI : public SelectedCIMethod {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    AdaptiveCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    // ==> Class Interface <==

    /// Set options from an option object
    /// @param options the options passed in
    void set_options(std::shared_ptr<ForteOptions>) override {}

    // Interfaces of SCI algorithm
    /// Print the banner and starting information.
    void print_info() override;
    /// Pre-iter preparation, usually includes preparing an initial reference
    void pre_iter_preparation() override;
    /// Step 1. Diagonalize the Hamiltonian in the P space
    void diagonalize_P_space() override;
    /// Step 2. Find determinants in the Q space
    void find_q_space() override;
    /// Step 3. Diagonalize the Hamiltonian in the P + Q space
    void diagonalize_PQ_space() override;
    /// Step 4. Check convergence
    bool check_convergence() override;
    /// Step 5. Prune the P + Q space to get an updated P space
    void prune_PQ_to_P() override;
    /// Post-iter process
    void post_iter_process() override;
    /// Full PT2 correction
    void full_mrpt2();

    // Temporarily added interface to ExcitedStateSolver
    /// Set the class variable
    void set_method_variables(
        std::string ex_alg, size_t nroot_method, size_t root,
        const std::vector<std::vector<std::pair<Determinant, double>>>& old_roots) override;
    /// Getters
    DeterminantHashVec get_PQ_space() override;
    psi::SharedMatrix get_PQ_evecs() override;
    psi::SharedVector get_PQ_evals() override;
    //    std::shared_ptr<WFNOperator> get_op() override;
    size_t get_ref_root() override;
    std::vector<double> get_multistate_pt2_energy_correction() override;

    /// Set the printing level
    void set_quiet(bool quiet) { quiet_mode_ = quiet; }

    /// Compute the ACI-NOs
    void print_nos();

    void semi_canonicalize();
    void set_fci_ints(std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

    void upcast_reference(DeterminantHashVec& ref);

    // Update sigma
    void update_sigma();

  private:
    // Temporarily added
    psi::SharedMatrix P_evecs_;
    psi::SharedVector P_evals_;
    DeterminantHashVec P_space_;
    DeterminantHashVec P_ref_;
    std::vector<double> P_ref_evecs_;
    std::vector<double> P_energies_;
    std::vector<std::vector<double>> energy_history_;
    int num_ref_roots_;
    bool follow_;
    local_timer cycle_time_;

    // Temporarily added interface to ExcitedStateSolver
    psi::SharedMatrix PQ_evecs_;
    psi::SharedVector PQ_evals_;
    DeterminantHashVec PQ_space_;

    // ==> Class data <==
    /// Forte options
    std::shared_ptr<ForteOptions> options_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The number of correlated molecular orbitals
    int ncmo_;
    /// The multiplicity of the reference
    int multiplicity_;
    /// M_s of the reference
    int twice_ms_;
    /// The number of active electrons
    int nactel_;
    /// The number of correlated alpha electrons
    int nalpha_;
    /// The number of correlated beta electrons
    int nbeta_;
    /// The number of frozen core orbitals
    int nfrzc_;
    psi::Dimension frzcpi_;
    /// The number of correlated molecular orbitals per irrep
    psi::Dimension ncmopi_;
    /// The number of restricted docc orbitals per irrep
    psi::Dimension rdoccpi_;
    /// The number of active orbitals per irrep
    psi::Dimension nactpi_;
    /// The number of restricted docc
    size_t rdocc_;
    /// The number of restricted virtual
    size_t rvir_;
    /// The number of frozen virtual
    size_t fvir_;
    /// The number of irreps
    size_t nirrep_;

    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The reference determinant
    std::vector<Determinant> initial_reference_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    size_t pre_iter_;
    bool set_ints_ = false;

    // ==> ACI Options <==
    /// The threshold applied to the primary space
    double sigma_;
    /// The threshold applied to the secondary space
    double gamma_;
    /// The prescreening threshold
    double screen_thresh_;
    /// The number of roots computed
    int nroot_;
    /// Use threshold from perturbation theory?
    bool perturb_select_;

    /// Add missing degenerate determinants excluded from the aimed selection?
    bool add_aimed_degenerate_;
    /// Add missing degenerate determinants excluded from the aimed selection?
    bool project_out_spin_contaminants_;

    /// The function of the q-space criteria per root
    std::string pq_function_;
    /// the q reference
    std::string q_reference_;
    /// Algorithm for computing excited states
    std::string ex_alg_;
    /// The reference root
    int ref_root_;
    /// The reference root
    int root_;
    /// Enable aimed selection
    bool aimed_selection_;
    /// If true select by energy, if false use first-order coefficient
    bool energy_selection_;
    /// Number of roots to calculate for final excited state
    int post_root_;
    /// Rediagonalize H?
    bool post_diagonalize_;
    /// Print warning?
    bool print_warning_;
    /// Spin tolerance
    double spin_tol_;
    /// Compute 1-RDM?
    bool compute_rdms_;
    /// Enforce spin completeness of the P and P + Q spaces?
    bool spin_complete_;
    /// Enforce spin completeness of the P space?
    bool spin_complete_P_ = false;
    /// Print a determinant analysis?
    bool det_hist_;
    /// Save dets to file?
    bool det_save_;
    /// Order of RDM to compute
    //    int rdm_level_;
    /// Control amount of printing
    bool quiet_mode_;
    /// Control streamlining
    bool streamline_qspace_;
    /// The CI coeffiecients
    psi::SharedMatrix evecs_;

    bool build_lists_;

    /// A map of determinants in the P space
    std::unordered_map<Determinant, int, Determinant::Hash> P_space_map_;
    /// A History of Determinants
    std::unordered_map<Determinant, std::vector<std::pair<size_t, std::string>>, Determinant::Hash>
        det_history_;
    /// Stream for printing determinant coefficients
    std::ofstream det_list_;
    /// Roots to project out
    std::vector<std::vector<std::pair<size_t, double>>> bad_roots_;
    /// Storage of past roots
    std::vector<std::vector<std::pair<Determinant, double>>> old_roots_;

    /// A Vector to store spin of each root
    std::vector<std::pair<double, double>> root_spin_vec_;
    /// Form initial guess space with correct spin? ****OBSOLETE?*****
    bool do_guess_;
    /// Spin-symmetrized evecs
    psi::SharedMatrix PQ_spin_evecs_;
    /// The unselected part of the SD space
    det_hash<double> external_wfn_;
    /// Do approximate RDM?
    bool approx_rdm_ = false;

    bool print_weights_;

    /// The alpha MO always unoccupied
    size_t hole_;

    /// Timing variables
    double build_H_;
    double diag_H_;
    double build_space_;
    double screen_space_;
    double spin_trans_;

    // The RDMS
    //    std::vector<double> ordm_a_;
    //    std::vector<double> ordm_b_;
    //    std::vector<double> trdm_aa_;
    //    std::vector<double> trdm_ab_;
    //    std::vector<double> trdm_bb_;
    //    std::vector<double> trdm_aaa_;
    //    std::vector<double> trdm_aab_;
    //    std::vector<double> trdm_abb_;
    //    std::vector<double> trdm_bbb_;

    ambit::Tensor ordm_a_;
    ambit::Tensor ordm_b_;
    ambit::Tensor trdm_aa_;
    ambit::Tensor trdm_ab_;
    ambit::Tensor trdm_bb_;
    ambit::Tensor trdm_aaa_;
    ambit::Tensor trdm_aab_;
    ambit::Tensor trdm_abb_;
    ambit::Tensor trdm_bbb_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Generate set of state-averaged q-criteria and determinants
    double average_q_values(std::vector<double>& E2);

    /// Get criteria for a specific root
    double root_select(int nroot, std::vector<double>& C1, std::vector<double>& E2);

    /// Basic determinant generator (threaded, no batching, all determinants stored)
    void get_excited_determinants_avg(int nroot, psi::SharedMatrix evecs, psi::SharedVector evals,
                                      DeterminantHashVec& P_space,
                                      std::vector<std::pair<double, Determinant>>& F_space);

    /// Get excited determinants with a specified hole
    //  void get_excited_determinants_restrict(int nroot, psi::SharedMatrix evecs, psi::SharedVector
    //  evals,  DeterminantHashVec& P_space,
    //                                         std::vector<std::pair<double, Determinant>>&
    //                                         F_space);
    /// Get excited determinants with a specified hole
    void get_excited_determinants_core(psi::SharedMatrix evecs, psi::SharedVector evals,
                                       DeterminantHashVec& P_space,
                                       std::vector<std::pair<double, Determinant>>& F_space);

    // Optimized for a single root
    void get_excited_determinants_sr(psi::SharedMatrix evecs, psi::SharedVector evals,
                                     DeterminantHashVec& P_space,
                                     std::vector<std::pair<double, Determinant>>& F_space);

    // (DEFAULT in batching) Optimized batching algorithm, prescreens the batches to significantly
    // reduce storage, based on hashes
    double get_excited_determinants_batch(psi::SharedMatrix evecs, psi::SharedVector evals,
                                          DeterminantHashVec& P_space,
                                          std::vector<std::pair<double, Determinant>>& F_space);

    // Gets excited determinants using sorting of vectors
    double
    get_excited_determinants_batch_vecsort(psi::SharedMatrix evecs, psi::SharedVector evals,
                                           DeterminantHashVec& P_space,
                                           std::vector<std::pair<double, Determinant>>& F_space);

    /// (DEFAULT)  Builds excited determinants for a bin, uses all threads, hash-based
    det_hash<double> get_bin_F_space(int bin, int nbin, double E0, psi::SharedMatrix evecs,
                                     DeterminantHashVec& P_space);

    /// Builds core excited determinants for a bin, uses all threads, hash-based
    det_hash<double> get_bin_F_space_core(int bin, int nbin, double E0, psi::SharedMatrix evecs,
                                          DeterminantHashVec& P_space);
    /// Builds excited determinants in batch using sorting of vectors
    std::pair<std::vector<std::vector<std::pair<Determinant, double>>>, std::vector<size_t>>
    get_bin_F_space_vecsort(int bin, int nbin, psi::SharedMatrix evecs,
                            DeterminantHashVec& P_space);

    /// Prune the space of determinants
    void prune_q_space(DeterminantHashVec& PQ_space, DeterminantHashVec& P_space,
                       psi::SharedMatrix evecs, int nroot);

    /// Check if the procedure has converged
    bool check_convergence(std::vector<std::vector<double>>& energy_history,
                           psi::SharedVector new_energies);

    /// Check if the procedure is stuck
    bool check_stuck(const std::vector<std::vector<double>>& energy_history,
                     psi::SharedVector evals);

    /// Compute overlap for root following
    int root_follow(DeterminantHashVec& P_ref, std::vector<double>& P_ref_evecs,
                    DeterminantHashVec& P_space, psi::SharedMatrix P_evecs, int num_ref_roots);

    /// Add roots to be projected out in DL
    void add_bad_roots(DeterminantHashVec& dets);
};

} // namespace forte

#endif // _aci_h_
