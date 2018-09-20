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

#include <fstream>
#include <iomanip>

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/physconst.h"

#include "forte_options.h"
#include "ci_rdm/ci_rdms.h"
#include "ci_reference.h"
#include "fci/fci_integrals.h"
#include "mrpt2.h"
#include "orbital-helper/unpaired_density.h"
#include "determinant_hashvector.h"
#include "reference.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "sparse_ci/determinant.h"
#include "orbital-helper/iao_builder.h"
#include "orbital-helper/localize.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

using d1 = std::vector<double>;
using d2 = std::vector<d1>;

namespace psi {
namespace forte {

class Reference;

/// Set the ACI options
void set_ASCI_options(ForteOptions& foptions);

/**
 * @brief The AdaptiveCI class
 * This class implements an adaptive CI algorithm
 */
class ASCI : public Wavefunction {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    ASCI(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
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


    void set_asci_ints(SharedWavefunction ref_Wfn, std::shared_ptr<ForteIntegrals> ints);

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
    /// The reference root
    bool print_warning_;
    /// Compute 1-RDM?
    bool compute_rdms_;
    /// The CI coeffiecients
    SharedMatrix evecs_;

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
    SharedMatrix PQ_spin_evecs_;
    /// The unselected part of the SD space
    det_hash<double> external_wfn_;
    /// Do approximate RDM?
    bool approx_rdm_ = false;

    bool print_weights_;

    bool set_rdm_ = false;

    /// The alpha MO always unoccupied
    int hole_;

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
    void compute_aci(DeterminantHashVec& PQ_space, SharedMatrix& PQ_evecs, SharedVector& PQ_evals);

    /// Print information about this calculation
    void print_info();

    /// Print a wave function
    void print_wfn(DeterminantHashVec& space, WFNOperator& op, SharedMatrix evecs, int nroot);

    /// Batched version of find q space
    void find_q_space_batched(DeterminantHashVec& P_space, DeterminantHashVec& PQ_space,
                              SharedVector evals, SharedMatrix evecs);

    /// Streamlined version of find q space
    void default_find_q_space(DeterminantHashVec& P_space, DeterminantHashVec& PQ_space,
                              SharedVector evals, SharedMatrix evecs);

    /// Find all the relevant excitations out of the P space
    void find_q_space(DeterminantHashVec& P_space, DeterminantHashVec& PQ_space,
                      SharedVector evals, SharedMatrix evecs);

    /// Basic determinant generator (threaded, no batching, all determinants stored)
    void get_excited_determinants(int nroot, SharedMatrix evecs, DeterminantHashVec& P_space,
                                  det_hash<std::vector<double>>& V_hash);

    /// Alternate/experimental determinant generator (threaded, each thread builds part of F)
    void get_excited_determinants_seq(int nroot, SharedMatrix evecs, DeterminantHashVec& P_space,
                                   det_hash<std::vector<double>>& V_hash);
    /// Get excited determinants with a specified hole
    void get_core_excited_determinants(SharedMatrix evecs, DeterminantHashVec& P_space,
                                       det_hash<std::vector<double>>& V_hash);

    // Optimized for a single root
    void get_excited_determinants_sr(SharedMatrix evecs, DeterminantHashVec& P_space,
                                     det_hash<double>& V_hash);

    // Primitive batching algorithm, each thread does one bin, to be removed
    double get_excited_determinants_batch_old(SharedMatrix evecs, SharedVector evals,
                                          DeterminantHashVec& P_space,
                                          std::vector<std::pair<double, Determinant>>& F_space);

    // (DEFAULT in batching) Optimized batching algorithm, prescreens the batches to significantly reduce storage, based on hashes
    double get_excited_determinants_batch(SharedMatrix evecs, SharedVector evals,
                                           DeterminantHashVec& P_space,
                                           std::vector<std::pair<double, Determinant>>& F_space);

    // Gets excited determinants using sorting of vectors
    double get_excited_determinants_batch_vecsort(SharedMatrix evecs, SharedVector evals,
                                           DeterminantHashVec& P_space,
                                           std::vector<std::pair<double, Determinant>>& F_space);

    /// Builds excited determinants for a bin, no threading, hash-based, to be removed
    det_hash<double> get_bin_F_space_old(int bin, int nbin, SharedMatrix evecs,
                                     DeterminantHashVec& P_space);

    /// (DEFAULT)  Builds excited determinants for a bin, uses all threads, hash-based
    det_hash<double> get_bin_F_space(int bin, int nbin,double E0, SharedMatrix evecs,
                                      DeterminantHashVec& P_space);

    /// Builds excited determinants in batch using sorting of vectors
    std::pair<std::vector<std::vector<std::pair<Determinant, double>>>, std::vector<size_t>>
    get_bin_F_space_vecsort(int bin, int nbin, SharedMatrix evecs, DeterminantHashVec& P_space);

    /// Prescreening algorithm, aware of sigma, very experimental
    // double prescreen_F(int bin, int nbin, double E0, SharedMatrix evecs,DeterminantHashVec& P_space);

    /// Prune the space of determinants
    void prune_q_space(DeterminantHashVec& PQ_space, DeterminantHashVec& P_space,
                       SharedMatrix evecs);

    /// Check if the procedure has converged
    bool check_convergence(std::vector<std::vector<double>>& energy_history,
                           SharedVector new_energies);

    /// Check if the procedure is stuck
    bool check_stuck(std::vector<std::vector<double>>& energy_history, SharedVector evals);

    /// Computes spin
    std::vector<std::pair<double, double>> compute_spin(DeterminantHashVec& space, WFNOperator& op,
                                                        SharedMatrix evecs, int nroot);

    /// Compute 1-RDM
    void compute_1rdm(SharedMatrix A, SharedMatrix B, std::vector<Determinant>& det_space,
                      SharedMatrix evecs, int nroot);

    /// Compute full S^2 matrix and diagonalize it
    void full_spin_transform(DeterminantHashVec& det_space, SharedMatrix cI, int nroot);

    /// Check for spin contamination
    double compute_spin_contamination(DeterminantHashVec& space, WFNOperator& op,
                                      SharedMatrix evecs, int nroot);

    /// Save coefficients of lowest-root determinant
    void save_dets_to_file(DeterminantHashVec& space, SharedMatrix evecs);
    /// Compute the Davidson correction
    std::vector<double> davidson_correction(std::vector<Determinant>& P_dets, SharedVector P_evals,
                                            SharedMatrix PQ_evecs,
                                            std::vector<Determinant>& PQ_dets,
                                            SharedVector PQ_evals);

    //    void compute_H_expectation_val(const
    //    std::vector<Determinant>& space,
    //                                    SharedVector& evals,
    //                                    const SharedMatrix evecs,
    //                                    int nroot,
    //                                    DiagonalizationMethod diag_method);
    //

    /// Print natural orbitals
    void print_nos();

    /// Convert from determinant to string representation
    void convert_to_string(const std::vector<Determinant>& space);

    /// Compute overlap for root following
    int root_follow(DeterminantHashVec& P_ref, std::vector<double>& P_ref_evecs,
                    DeterminantHashVec& P_space, SharedMatrix P_evecs, int num_ref_roots);

    /// Project ACI wavefunction
    void project_determinant_space(DeterminantHashVec& space, SharedMatrix evecs,
                                   SharedVector evals, int nroot);

    /// Compute the RDMs
    void compute_rdms(std::shared_ptr<FCIIntegrals> fci_ints, DeterminantHashVec& dets,
                      WFNOperator& op, SharedMatrix& PQ_evecs, int root1, int root2);

    /// Save older roots
    void save_old_root(DeterminantHashVec& dets, SharedMatrix& PQ_evecs, int root);

    /// Add roots to be projected out in DL
    void add_bad_roots(DeterminantHashVec& dets);

    /// Print Summary
    void print_final(DeterminantHashVec& dets, SharedMatrix& PQ_evecs, SharedVector& PQ_evals);

    void compute_multistate(SharedVector& PQ_evals);

    void block_diagonalize_fock(const d2& Fa, const d2& Fb, SharedMatrix& Ua, SharedMatrix& Ub,
                                const std::string& name);

    DeterminantHashVec approximate_wfn(DeterminantHashVec& PQ_space, SharedMatrix& evecs,
                                       SharedVector& PQ_evals, SharedMatrix& new_evecs);

    std::vector<std::pair<size_t, double>> dl_initial_guess(std::vector<Determinant>& old_dets,
                                                            std::vector<Determinant>& dets,
                                                            SharedMatrix& evecs, int nroot);

    std::vector<std::tuple<double, int, int>> sym_labeled_orbitals(std::string type);

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

} // namespace forte
} // namespace psi

#endif // _as_ci_h_
