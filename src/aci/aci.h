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

#ifndef _adaptive_ci_h_
#define _adaptive_ci_h_

#include <fstream>
#include <iomanip>

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/physconst.h"

#include "../determinant_map.h"
#include "../sparse_ci_solver.h"

using d1 = std::vector<double>;
using d2 = std::vector<d1>;

namespace psi {
namespace forte {

class Reference;

/// Set the ACI options
void set_ACI_options(ForteOptions& foptions);

/**
 * @brief The AdaptiveCI class
 * This class implements an adaptive CI algorithm
 */
class AdaptiveCI : public Wavefunction {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    AdaptiveCI(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~AdaptiveCI();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

    /// Update the reference file
    Reference reference();

    /// Set the RDM
    void set_max_rdm(int rdm);
    /// Set the printing level
    void set_quiet(bool quiet) { quiet_mode_ = quiet; }

    /// Get the wavefunction
    DeterminantMap get_wavefunction();

    /// Compute the ACI-NOs
    void compute_nos();

    void diagonalize_final_and_compute_rdms();

    void set_aci_ints(SharedWavefunction ref_Wfn, std::shared_ptr<ForteIntegrals> ints);

    void semi_canonicalize();

  private:
    // ==> Class data <==

    DeterminantMap final_wfn_;

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
    /// The number of correlated molecular orbitals per irrep
    Dimension ncmopi_;
    /// The number of restricted docc orbitals per irrep
    Dimension rdoccpi_;
    /// The number of active orbitals per irrep
    Dimension nactpi_;
    /// The number of active orbitals
    size_t nact_;
    /// The number of restricted docc
    size_t rdocc_;
    /// The number of restricted virtual
    size_t rvir_;
    /// The number of frozen virtual
    size_t fvir_;

    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The reference determinant
    std::vector<STLBitsetDeterminant> initial_reference_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    /// The current iteration
    int cycle_;
    /// The last iteration
    int max_cycle_;
    int pre_iter_;

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
    /// The eigensolver type
    DiagonalizationMethod diag_method_ = DLString;
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
    /// Enforce spin completeness?
    bool spin_complete_;
    /// Print a determinant analysis?
    bool det_hist_;
    /// Save dets to file?
    bool det_save_;
    /// Order of RDM to compute
    int rdm_level_;
    /// Control amount of printing
    bool quiet_mode_;
    /// Control streamlining
    bool streamline_qspace_;
    /// The CI coeffiecients
    SharedMatrix evecs_;

    /// A map of determinants in the P space
    std::unordered_map<STLBitsetDeterminant, int, STLBitsetDeterminant::Hash> P_space_map_;
    /// A History of Determinants
    std::unordered_map<STLBitsetDeterminant, std::vector<std::pair<size_t, std::string>>,
                       STLBitsetDeterminant::Hash>
        det_history_;
    /// Stream for printing determinant coefficients
    std::ofstream det_list_;
    /// Roots to project out
    std::vector<std::vector<std::pair<size_t, double>>> bad_roots_;
    /// Storage of past roots
    std::vector<std::vector<std::pair<STLBitsetDeterminant, double>>> old_roots_;

    /// A Vector to store spin of each root
    std::vector<std::pair<double, double>> root_spin_vec_;
    /// Form initial guess space with correct spin? ****OBSOLETE?*****
    bool do_guess_;
    /// Spin-symmetrized evecs
    SharedMatrix PQ_spin_evecs_;
    /// The unselected part of the SD space
    det_hash<double> external_wfn_;
    /// Do approximate RDM?
    bool approx_rdm_ = false;

    bool print_weights_;

    /// Timing variables
    double build_H_;
    double diag_H_;
    double build_space_;
    double screen_space_;
    double spin_trans_;

    // The RMDS
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
    void compute_aci(DeterminantMap& PQ_space, SharedMatrix& PQ_evecs, SharedVector& PQ_evals);

    /// Print information about this calculation
    void print_info();

    /// Print a wave function
    void print_wfn(DeterminantMap& space, SharedMatrix evecs, int nroot);

    /// Streamlined version of find q space
    void default_find_q_space(DeterminantMap& P_space, DeterminantMap& PQ_space, SharedVector evals,
                              SharedMatrix evecs);

    /// Find all the relevant excitations out of the P space
    void find_q_space(DeterminantMap& P_space, DeterminantMap& PQ_space, int nroot,
                      SharedVector evals, SharedMatrix evecs);

    /// Generate set of state-averaged q-criteria and determinants
    double average_q_values(int nroot, std::vector<double>& C1, std::vector<double>& E2);

    /// Get criteria for a specific root
    double root_select(int nroot, std::vector<double>& C1, std::vector<double>& E2);

    /// Find all the relevant excitations out of the P space - single root
    /// version
    void find_q_space_single_root(int nroot, SharedVector evals, SharedMatrix evecs);

    /// Alternate/experimental determinant generator
    void get_excited_determinants(int nroot, SharedMatrix evecs, DeterminantMap& P_space,
                                  det_hash<std::vector<double>>& V_hash);
    /// Alternate/experimental determinant generator
    void get_excited_determinants2(int nroot, SharedMatrix evecs, DeterminantMap& P_space,
                                   det_hash<std::vector<double>>& V_hash);

    /// Prune the space of determinants
    void prune_q_space(DeterminantMap& PQ_space, DeterminantMap& P_space, SharedMatrix evecs,
                       int nroot);

    /// Check if the procedure has converged
    bool check_convergence(std::vector<std::vector<double>>& energy_history,
                           SharedVector new_energies);

    /// Check if the procedure is stuck
    bool check_stuck(std::vector<std::vector<double>>& energy_history, SharedVector evals);

    /// Computes spin
    std::vector<std::pair<double, double>> compute_spin(DeterminantMap& space, SharedMatrix evecs,
                                                        int nroot);

    /// Compute 1-RDM
    void compute_1rdm(SharedMatrix A, SharedMatrix B, std::vector<STLBitsetDeterminant>& det_space,
                      SharedMatrix evecs, int nroot);

    /// Compute full S^2 matrix and diagonalize it
    void full_spin_transform(DeterminantMap& det_space, SharedMatrix cI, int nroot);

    /// Check for spin contamination
    double compute_spin_contamination(DeterminantMap& space, SharedMatrix evecs, int nroot);

    /// Save coefficients of lowest-root determinant
    void save_dets_to_file(DeterminantMap& space, SharedMatrix evecs);
    /// Compute the Davidson correction
    std::vector<double> davidson_correction(std::vector<STLBitsetDeterminant>& P_dets,
                                            SharedVector P_evals, SharedMatrix PQ_evecs,
                                            std::vector<STLBitsetDeterminant>& PQ_dets,
                                            SharedVector PQ_evals);

    //    void compute_H_expectation_val(const
    //    std::vector<STLBitsetDeterminant>& space,
    //                                    SharedVector& evals,
    //                                    const SharedMatrix evecs,
    //                                    int nroot,
    //                                    DiagonalizationMethod diag_method);
    //

    /// Print natural orbitals
    void print_nos();

    /// Convert from determinant to string representation
    void convert_to_string(const std::vector<STLBitsetDeterminant>& space);

    /// Compute overlap for root following
    int root_follow(DeterminantMap& P_ref, std::vector<double>& P_ref_evecs,
                    DeterminantMap& P_space, SharedMatrix P_evecs, int num_ref_roots);

    /// Project ACI wavefunction
    void project_determinant_space(DeterminantMap& space, SharedMatrix evecs, SharedVector evals,
                                   int nroot);

    /// Compute the RDMs
    void compute_rdms(DeterminantMap& dets, WFNOperator& op, SharedMatrix& PQ_evecs, int root1,
                      int root2);

    /// Save older roots
    void save_old_root(DeterminantMap& dets, SharedMatrix& PQ_evecs, int root);

    /// Add roots to be projected out in DL
    void add_bad_roots(DeterminantMap& dets);

    /// Print Summary
    void print_final(DeterminantMap& dets, SharedMatrix& PQ_evecs, SharedVector& PQ_evals);

    void compute_multistate(SharedVector& PQ_evals);

    void block_diagonalize_fock(const d2& Fa, const d2& Fb, SharedMatrix& Ua, SharedMatrix& Ub,
                                const std::string& name);

    DeterminantMap approximate_wfn(DeterminantMap& PQ_space, SharedMatrix& evecs,
                                   det_hash<double>& external_space, SharedMatrix& new_evecs);

    std::vector<std::pair<size_t, double>>
    dl_initial_guess(std::vector<STLBitsetDeterminant>& old_dets,
                     std::vector<STLBitsetDeterminant>& dets, SharedMatrix& evecs, int nroot);

    //    int david2(double **A, int N, int M, double *eps, double **v,double
    //    cutoff, int print);
    //    /// Perform a Davidson-Liu diagonalization
    //    void davidson_liu(SharedMatrix H,SharedVector Eigenvalues,SharedMatrix
    //    Eigenvectors,int nroots);

    //    /// Perform a Davidson-Liu diagonalization on a sparse matrix
    //    bool davidson_liu_sparse(std::vector<std::vector<std::pair<int,double>
    //    > > H_sparse,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int
    //    nroots);
};
}
} // End Namespaces

#endif // _adaptive_ci_h_
