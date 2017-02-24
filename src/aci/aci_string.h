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

#ifndef _adaptive_ci_h_
#define _adaptive_ci_h_

#include <fstream>

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/physconst.h"

#include "../integrals/integrals.h"
#include "../helpers.h"
#include "../stl_bitset_determinant.h"
#include "../stl_bitset_string.h"
#include "../sparse_ci_solver.h"
#include "../fci/fci_integrals.h"
#include "../ci_rdms.h"

namespace psi {
namespace forte {

/**
 * @brief The ACIString class
 * This class implements an adaptive CI algorithm
 */
class ACIString : public Wavefunction {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    ACIString(SharedWavefunction ref_wfn, Options& options,
              std::shared_ptr<ForteIntegrals> ints,
              std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~ACIString();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

    /// Update the reference file
    Reference reference();

    /// Set the RDM
    void set_max_rdm(int rdm);
    /// Set the printing level
    void set_quiet(bool quiet) { quiet_mode_ = quiet; }

  private:
    // ==> Class data <==

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
    int wavefunction_multiplicity_;
    /// The number of active electrons
    int nactel_;
    /// The number of correlated alpha electrons
    int noalpha_;
    /// The number of correlated beta electrons
    int nobeta_;
    /// The number of alpha virtuals
    int nvalpha_;
    /// The number of beta virtuals
    int nvbeta_;
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
    STLBitsetDeterminant reference_determinant_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    /// The current iteration
    int cycle_;
    /// The last iteration
    int max_cycle_;

    // ==> ACI Options <==
    /// The threshold applied to the primary space
    double tau_p_;
    /// The threshold applied to the secondary space
    double tau_q_;
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
    /// The type of q criteria
    bool q_rel_;
    /// the q reference
    std::string q_reference_;
    /// Algorithm for computing excited states
    std::string ex_alg_;
    /// The eigensolver type
    DiagonalizationMethod diag_method_ = DLString;
    /// The reference root
    int ref_root_;
    /// Enable aimed selection
    bool aimed_selection_;
    /// If true select by energy, if false use first-order coefficient
    bool energy_selection_;
    /// Smooth the Hamiltonian in the P space?
    bool do_smooth_;
    /// The threshold for smoothing elements of the Hamiltonian
    double smooth_threshold_;
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

    /// Alfa strings
    std::vector<std::vector<STLBitsetString>> alfa_str_;
    /// Beta strings
    std::vector<std::vector<STLBitsetString>> beta_str_;
    /// Coupling vector
    std::vector<std::tuple<int, size_t, int, size_t>> str_to_det_;

    /// A vector of determinants in the P space
    std::vector<STLBitsetDeterminant> P_space_;
    /// A vector of determinants in the P + Q space
    std::vector<STLBitsetDeterminant> PQ_space_;
    /// The CI coeffiecients
    SharedMatrix evecs_;

    /// A map of determinants in the P space
    std::unordered_map<STLBitsetDeterminant, int, STLBitsetDeterminant::Hash>
        P_space_map_;
    /// A History of Determinants
    std::unordered_map<STLBitsetDeterminant,
                       std::vector<std::pair<size_t, std::string>>,
                       STLBitsetDeterminant::Hash> det_history_;
    /// Stream for printing determinant coefficients
    std::ofstream det_list_;

    /// A Vector to store spin of each root
    std::vector<std::pair<double, double>> root_spin_vec_;
    /// Form initial guess space with correct spin? ****OBSOLETE?*****
    bool do_guess_;
    /// Spin-symmetrized evecs
    SharedMatrix PQ_spin_evecs_;

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

    /// Get the reference occupation
    std::vector<int> get_occupation();

    /// Print information about this calculation
    void print_info();

    /// Print a wave function
    void print_wfn(std::vector<STLBitsetDeterminant> space, SharedMatrix evecs,
                   int nroot);

    /// Streamlined version of find q space
    void default_find_q_space(SharedVector evals, SharedMatrix evecs);

    /// Find all the relevant excitations out of the P space
    void find_q_space(int nroot, SharedVector evals, SharedMatrix evecs);

    /// Generate set of state-averaged q-criteria and determinants
    double average_q_values(int nroot, std::vector<double> C1,
                            std::vector<double> E2);

    /// Get criteria for a specific root
    double root_select(int nroot, std::vector<double> C1,
                       std::vector<double> E2);

    /// Find all the relevant excitations out of the P space - single root
    /// version
    void find_q_space_single_root(int nroot, SharedVector evals,
                                  SharedMatrix evecs);

    /// Generate alpha excitations
    void add_single_sub_alfa(std::vector<std::vector<STLBitsetString>> str);

    /// Generate beta excitations
    void add_single_sub_beta(std::vector<std::vector<STLBitsetString>> str);

    /// Generate alpha-alpha excitations
    void add_double_sub_alfa(std::vector<std::vector<STLBitsetString>> str);

    /// Generate beta-beta excitations
    void add_double_sub_beta(std::vector<std::vector<STLBitsetString>> str);

//    /// Generate excited determinants
//    void generate_excited_determinants(
//        int nroot, int I, SharedMatrix evecs, STLBitsetDeterminant& det,
//        std::unordered_map<STLBitsetDeterminant, std::vector<double>,
//                           STLBitsetDeterminant::Hash>& V_hash);

    /// Prune the space of determinants
    void prune_q_space(
        std::vector<STLBitsetDeterminant>& large_space,
        std::vector<STLBitsetDeterminant>& pruned_space,
        std::unordered_map<STLBitsetDeterminant, int,
                           STLBitsetDeterminant::Hash>& pruned_space_map,
        SharedMatrix evecs, int nroot);

    void smooth_hamiltonian(std::vector<STLBitsetDeterminant>& space,
                            SharedVector evals, SharedMatrix evecs, int nroot);

    /// Check if the procedure has converged
    bool check_convergence(std::vector<std::vector<double>>& energy_history,
                           SharedVector new_energies);

    /// Check if the procedure is stuck
    bool check_stuck(std::vector<std::vector<double>>& energy_history,
                     SharedVector evals);

    /// Analyze the wavefunction
    void wfn_analyzer(std::vector<STLBitsetDeterminant> det_space,
                      SharedMatrix evecs, int nroot);

    /// Returns a vector of orbital energy, sym label pairs
    std::vector<std::pair<double, std::pair<int, int>>>
    sym_labeled_orbitals(std::string type);

    /// Computes spin
    std::vector<std::pair<std::pair<double, double>, std::pair<size_t, double>>>
    compute_spin(std::vector<STLBitsetDeterminant> space, SharedMatrix evecs,
                 int nroot);

    /// Compute full S^2 matrix and diagonalize it
    void full_spin_transform(std::vector<STLBitsetDeterminant> det_space,
                             SharedMatrix cI, int nroot);
    /// Check for spin contamination
    double compute_spin_contamination(std::vector<STLBitsetDeterminant> space,
                                      SharedMatrix evecs, int nroot);

    /// Compute the Davidson correction
    std::vector<double>
    davidson_correction(std::vector<STLBitsetDeterminant> P_dets,
                        SharedVector P_evals, SharedMatrix PQ_evecs,
                        std::vector<STLBitsetDeterminant> PQ_dets,
                        SharedVector PQ_evals);

    void
    compute_H_expectation_val(const std::vector<STLBitsetDeterminant> space,
                              SharedVector& evals, const SharedMatrix evecs,
                              int nroot, DiagonalizationMethod diag_method);

    /// Print natural orbitals
    void print_nos();

    /// Get the symmetry of a string
    int get_sym(STLBitsetString str);

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
