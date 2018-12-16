/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "forte_options.h"
#include "ci_rdm/ci_rdms.h"
#include "sparse_ci/ci_reference.h"
#include "fci/fci_integrals.h"
#include "mrpt2.h"
#include "orbital-helpers/unpaired_density.h"
#include "sparse_ci/determinant_hashvector.h"
#include "base_classes/reference.h"
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

class Reference;

/// Set the ACI options
void set_ASCI_options(ForteOptions& foptions);

/**
 * @brief The AdaptiveCI class
 * This class implements an adaptive CI algorithm
 */
class ASCI : public psi::Wavefunction {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    ASCI(psi::SharedWavefunction ref_wfn, psi::Options& options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~ASCI();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

    /// Update the reference file
    Reference reference();

    /// Get the wavefunction
    DeterminantHashVec get_wavefunction();

    /// Compute the ACI-NOs
    void compute_nos();


    void set_asci_ints(psi::SharedWavefunction ref_Wfn, std::shared_ptr<ForteIntegrals> ints);

    void set_fci_ints(std::shared_ptr<FCIIntegrals> fci_ints);

  private:
    // ==> Class data <==

    DeterminantHashVec final_wfn_;

    WFNOperator op_;

    /// The molecular integrals required by Explorer
    std::shared_ptr<ForteIntegrals> ints_;
    /// Pointer to FCI integrals
    std::shared_ptr<FCIIntegrals> fci_ints_;
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The multiplicity of the reference
    int multiplicity_;
    /// M_s of the reference
    int twice_ms_;
    /// The number of frozen core orbitals
    int nfrzc_;
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
    /// The number of roots computed
    int nroot_;

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
    std::vector<double> ordm_a_;
    std::vector<double> ordm_b_;
    std::vector<double> trdm_aa_;
    std::vector<double> trdm_ab_;
    std::vector<double> trdm_bb_;
    std::vector<double> trdm_aaa_;
    std::vector<double> trdm_aab_;
    std::vector<double> trdm_abb_;
    std::vector<double> trdm_bbb_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Compute an aci wavefunction
    void compute_aci(DeterminantHashVec& PQ_space, psi::SharedMatrix& PQ_evecs, psi::SharedVector& PQ_evals);

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
    void compute_rdms(std::shared_ptr<FCIIntegrals> fci_ints, DeterminantHashVec& dets,
                      WFNOperator& op, psi::SharedMatrix& PQ_evecs, int root1, int root2);

};

} // namespace forte


#endif // _as_ci_h_
