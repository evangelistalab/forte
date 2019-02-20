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

#ifndef _as_ci_h_
#define _as_ci_h_

#include "base_classes/forte_options.h"
#include "ci_rdm/ci_rdms.h"
#include "sparse_ci/ci_reference.h"
#include "integrals/active_space_integrals.h"
#include "mrpt2.h"
#include "orbital-helpers/unpaired_density.h"
#include "sparse_ci/determinant_hashvector.h"
#include "base_classes/rdms.h"
#include "base_classes/active_space_method.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "orbital-helpers/localize.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

namespace forte {

class RDMs;

/**
 * @brief The AdaptiveCI class
 * This class implements an adaptive CI algorithm
 */
class ASCI : public ActiveSpaceMethod {
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

    /// Compute the energy
    double compute_energy() override;

    /// Compute the reduced density matrices up to a given particle rank (max_rdm_level)
    std::vector<RDMs> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                           int max_rdm_level) override;

    /// Returns the transition reduced density matrices between roots of different symmetry up to a
    /// given level (max_rdm_level)
    std::vector<RDMs> transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                      std::shared_ptr<ActiveSpaceMethod> method2,
                                      int max_rdm_level) override;

    void set_options(std::shared_ptr<ForteOptions>) override{}; // TODO : define

    /// Get the wavefunction
    DeterminantHashVec get_wavefunction();

    /// Compute the ACI-NOs
    void compute_nos();

    void set_fci_ints(std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

  private:
    // ==> Class data <==

    DeterminantHashVec final_wfn_;

    WFNOperator op_;

    /// HF info
    std::shared_ptr<SCFInfo> scf_info_;
    /// Options
    std::shared_ptr<ForteOptions> options_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The multiplicity of the reference
    int multiplicity_;
    /// M_s of the reference
    int twice_ms_;
    /// Number of irreps
    size_t nirrep_;
    /// The number of frozen core orbitals
    int nfrzc_;
    /// The number of frozen core orbital per irrets
    psi::Dimension frzcpi_;
    /// The number of active orbitals per irrep
    psi::Dimension nactpi_;
    /// The number of active orbitals
    size_t nact_;
    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The reference determinant
    std::vector<Determinant> initial_reference_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    /// The last iteration
    int max_cycle_;
    bool set_ints_ = false;

    // ==> ACI Options <==
    /// The threshold applied to the primary space
    int c_det_;
    /// The threshold applied to the secondary space
    int t_det_;

    /// The eigensolver type
    DiagonalizationMethod diag_method_ = DLString;
    /// Compute 1-RDM?
    bool compute_rdms_;
    /// The CI coeffiecients
    psi::SharedMatrix evecs_;

    bool build_lists_;
    bool print_weights_ = false;

    /// Order of RDM to compute
    int rdm_level_ = 1;
    /// A Vector to store spin of each root
    std::vector<std::pair<double, double>> root_spin_vec_;

    /// Timing variables
    double build_H_;
    double diag_H_;
    double build_space_;
    double screen_space_;
    double spin_trans_;

    // The RDMS
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

    /// Compute an aci wavefunction
    void compute_aci(DeterminantHashVec& PQ_space, psi::SharedMatrix& PQ_evecs,
                     psi::SharedVector& PQ_evals);

    /// Print information about this calculation
    void print_info();

    /// Print a wave function
    void print_wfn(DeterminantHashVec& space, WFNOperator& op, psi::SharedMatrix evecs, int nroot);

    /// Find all the relevant excitations out of the P space
    void find_q_space(DeterminantHashVec& P_space, DeterminantHashVec& PQ_space,
                      psi::SharedVector evals, psi::SharedMatrix evecs);

    // Optimized for a single root
    void get_excited_determinants_sr(psi::SharedMatrix evecs, DeterminantHashVec& P_space,
                                     det_hash<double>& V_hash);

    /// Prune the space of determinants
    void prune_q_space(DeterminantHashVec& PQ_space, DeterminantHashVec& P_space,
                       psi::SharedMatrix evecs);

    /// Check if the procedure has converged
    bool check_convergence(std::vector<std::vector<double>>& energy_history,
                           psi::SharedVector new_energies);

    /// Computes spin
    std::vector<std::pair<double, double>> compute_spin(DeterminantHashVec& space, WFNOperator& op,
                                                        psi::SharedMatrix evecs, int nroot);

    /// Check for spin contamination
    double compute_spin_contamination(DeterminantHashVec& space, WFNOperator& op,
                                      psi::SharedMatrix evecs, int nroot);

    /// Print natural orbitals
    void print_nos();

    /// Compute the RDMs
    void compute_rdms(std::shared_ptr<ActiveSpaceIntegrals> fci_ints, DeterminantHashVec& dets,
                      WFNOperator& op, psi::SharedMatrix& PQ_evecs, int root1, int root2,
                      int max_level);
};

} // namespace forte

#endif // _as_ci_h_
