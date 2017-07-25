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

#ifndef _sparse_ci_h_
#define _sparse_ci_h_

#include "fci/fci_integrals.h"
#include "stl_bitset_determinant.h"
#include "determinant_map.h"
#include "operator.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define BIGNUM 1E100
#define MAXIT 100

namespace psi {
namespace forte {

enum DiagonalizationMethod { Full, DLSolver, DLString, DLDisk, MPI, Sparse };

/**
 * @brief The SigmaVector class
 * Base class for a sigma vector object.
 */
class SigmaVector {
  public:
    SigmaVector(size_t size) : size_(size){};

    size_t size() { return size_; }

    virtual void compute_sigma(SharedVector sigma, SharedVector b) = 0;
    //    virtual void compute_sigma(Matrix& sigma, Matrix& b, int nroot) = 0;
    virtual void get_diagonal(Vector& diag) = 0;
    virtual void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states) = 0;

  protected:
    size_t size_;
};

/**
 * @brief The SigmaVectorSparse class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorSparse : public SigmaVector {
  public:
    SigmaVectorSparse(std::vector<std::pair<std::vector<size_t>, std::vector<double>>>& H, std::shared_ptr<FCIIntegrals> fci_ints)
        : SigmaVector(H.size()), H_(H), fci_ints_(fci_ints){};

    void compute_sigma(SharedVector sigma, SharedVector b);
    //   void compute_sigma(Matrix& sigma, Matrix& b, int nroot) {}
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    std::vector<std::pair<std::vector<size_t>, std::vector<double>>>& H_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
};

/**
 * @brief The SigmaVectorList class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorList : public SigmaVector {
  public:
    SigmaVectorList(const std::vector<STLBitsetDeterminant>& space, bool print_detail, std::shared_ptr<FCIIntegrals> fci_ints);

    void compute_sigma(SharedVector sigma, SharedVector b);
    //  void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void get_hamiltonian(Matrix& H);
    std::vector<std::pair<std::vector<int>, std::vector<double>>> get_sparse_hamiltonian();
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    const std::vector<STLBitsetDeterminant>& space_;
    std::shared_ptr<FCIIntegrals> fci_ints_;

    // Create the list of a_p|N>
    std::vector<std::vector<std::pair<size_t, short>>> a_ann_list;
    std::vector<std::vector<std::pair<size_t, short>>> b_ann_list;
    // Create the list of a+_q |N-1>
    std::vector<std::vector<std::pair<size_t, short>>> a_cre_list;
    std::vector<std::vector<std::pair<size_t, short>>> b_cre_list;

    // Create the list of a_q a_p|N>
    std::vector<std::vector<std::tuple<size_t, short, short>>> aa_ann_list;
    std::vector<std::vector<std::tuple<size_t, short, short>>> ab_ann_list;
    std::vector<std::vector<std::tuple<size_t, short, short>>> bb_ann_list;
    // Create the list of a+_s a+_r |N-2>
    std::vector<std::vector<std::tuple<size_t, short, short>>> aa_cre_list;
    std::vector<std::vector<std::tuple<size_t, short, short>>> ab_cre_list;
    std::vector<std::vector<std::tuple<size_t, short, short>>> bb_cre_list;
    std::vector<double> diag_;

    bool print_details_ = true;
};

/* Uses ann/cre lists in sigma builds (Harrison and Zarrabian method) */
class SigmaVectorWfn1 : public SigmaVector {
  public:
    SigmaVectorWfn1(const DeterminantMap& space, WFNOperator& op, std::shared_ptr<FCIIntegrals> fci_ints);

    void compute_sigma(SharedVector sigma, SharedVector b);
    //   void compute_sigma(Matrix& sigma, Matrix& b, int nroot) {}
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);
    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    // Create the list of a_p|N>
    std::vector<std::vector<std::pair<size_t, short>>>& a_ann_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_ann_list_;
    // Create the list of a+_q |N-1>
    std::vector<std::vector<std::pair<size_t, short>>>& a_cre_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_cre_list_;

    // Create the list of a_q a_p|N>
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_ann_list_;
    // Create the list of a+_s a+_r |N-2>
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_cre_list_;
    std::vector<double> diag_;
    const DeterminantMap& space_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
};

/* Uses only cre lists, sparse sigma build */
class SigmaVectorWfn2 : public SigmaVector {
  public:
    SigmaVectorWfn2(const DeterminantMap& space, WFNOperator& op, std::shared_ptr<FCIIntegrals> fci_ints);
    std::vector<std::vector<std::pair<size_t, short>>>& a_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_list_;

    void compute_sigma(SharedVector sigma, SharedVector b);
    // void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    bool print_;
    bool use_disk_ = false;

    const DeterminantMap& space_;
    // size_t noalfa_;
    // size_t nobeta_;

    std::vector<double> diag_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
};
/* Uses only cre lists, DGEMM sigma build */
class SigmaVectorWfn3 : public SigmaVector {
  public:
    SigmaVectorWfn3(const DeterminantMap& space, WFNOperator& op, std::shared_ptr<FCIIntegrals> fci_ints);
    std::vector<std::vector<std::pair<size_t, short>>>& a_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_list_;

    void compute_sigma(SharedVector sigma, SharedVector b);
    // void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    bool print_;
    bool use_disk_ = false;

    std::shared_ptr<FCIIntegrals> fci_ints_;
    const DeterminantMap& space_;
    // size_t noalfa_;
    // size_t nobeta_;

    std::vector<double> diag_;

    SharedMatrix aa_tei_;
    SharedMatrix ab_tei_;
    SharedMatrix bb_tei_;
};

#ifdef HAVE_MPI
class SigmaVectorMPI : public SigmaVector {
  public:
    SigmaVectorMPI(const DeterminantMap& space, WFNOperator& op, std::shared_ptr<FCIIntegrals> fci_ints);

    void compute_sigma(SharedVector sigma, SharedVector b);
    void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;
  protected:
    std::shared_ptr<FCIIntegrals> fci_ints_;
};
#endif

/**
 * @brief The SparseCISolver class
 * This class diagonalizes the Hamiltonian in a basis
 * of determinants.
 */
class SparseCISolver {
  public:
    // ==> Class Interface <==

    /**
     * Diagonalize the Hamiltonian in a basis of determinants
     * @param space The basis for the CI given as a vector of
     * STLBitsetDeterminant objects
     * @param nroot The number of solutions to find
     * @param diag_method The diagonalization algorithm
     * @param multiplicity The spin multiplicity of the solution (2S + 1).  1 =
     * singlet, 2 = doublet, ...
     */

    SparseCISolver(std::shared_ptr<FCIIntegrals> fci_ints) { fci_ints_ = fci_ints;}

    void diagonalize_hamiltonian(const std::vector<STLBitsetDeterminant>& space,
                                 SharedVector& evals, SharedMatrix& evecs, int nroot,
                                 int multiplicity, DiagonalizationMethod diag_method);

    void diagonalize_hamiltonian_map(const DeterminantMap& space, WFNOperator& op,
                                     SharedVector& evals, SharedMatrix& evecs, int nroot,
                                     int multiplicity, DiagonalizationMethod diag_method);

    /// Enable/disable the parallel algorithms
    void set_parallel(bool parallel) { parallel_ = parallel; }

    /// Enable/disable printing of details
    void set_print_details(bool print_details) { print_details_ = print_details; }

    /// Enable/disable spin projection
    void set_spin_project(bool value);

    /// Enable/disable spin projection in full algorithm
    void set_spin_project_full(bool value);

    /// Enable/disable root projection
    void set_root_project(bool value);

    /// Set convergence threshold
    void set_e_convergence(double value);

    /// The maximum number of iterations for the Davidson algorithm
    void set_maxiter_davidson(int value);
    SharedMatrix build_full_hamiltonian(const std::vector<STLBitsetDeterminant>& space);
    std::vector<std::pair<std::vector<int>, std::vector<double>>>
    build_sparse_hamiltonian(const std::vector<STLBitsetDeterminant>& space);

    /// Add roots to project out during Davidson-Liu procedure
    void add_bad_states(std::vector<std::vector<std::pair<size_t, double>>>& roots);

    /// Set option to force diagonalization type
    void set_force_diag(int value);

    /// Set the size of the guess space
    void set_guess_dimension(size_t value) { dl_guess_ = value; };

    /// Set the initial guess
    void set_initial_guess(std::vector<std::pair<size_t, double>>& guess);
    void manual_guess(bool value);
    void set_num_vecs(size_t value);
    void set_sigma_method(std::string value);
    std::string sigma_method_ = "SPARSE";

    /// Set a customized SigmaVector for Davidson-Liu algorithm
    void set_sigma_vector(SigmaVector* sigma_vec) { sigma_vec_ = sigma_vec; }

  private:
    /// Form the full Hamiltonian and diagonalize it (for debugging)
    void diagonalize_full(const std::vector<STLBitsetDeterminant>& space, SharedVector& evals,
                          SharedMatrix& evecs, int nroot, int multiplicity);

    void diagonalize_mpi(const DeterminantMap& space, WFNOperator& op, SharedVector& evals,
                         SharedMatrix& evecs, int nroot, int multiplicity);

    void diagonalize_dl(const DeterminantMap& space, WFNOperator& op, SharedVector& evals,
                        SharedMatrix& evecs, int nroot, int multiplicity);

    void diagonalize_dl_sparse(const DeterminantMap& space, WFNOperator& op, SharedVector& evals,
                               SharedMatrix& evecs, int nroot, int multiplicity);

    void diagonalize_davidson_liu_solver(const std::vector<STLBitsetDeterminant>& space,
                                         SharedVector& evals, SharedMatrix& evecs, int nroot,
                                         int multiplicity);

    //   void diagonalize_davidson_liu_string(
    //       const std::vector<STLBitsetDeterminant>& space, SharedVector& evals,
    //       SharedMatrix& evecs, int nroot, int multiplicity, bool disk);
    /// Build the full Hamiltonian matrix

    std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>>
    initial_guess(const std::vector<STLBitsetDeterminant>& space, int nroot, int multiplicity);

    std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>>
    initial_guess_map(const DeterminantMap& space, int nroot, int multiplicity);

    /// The Davidson-Liu algorithm
    bool davidson_liu_solver(const std::vector<STLBitsetDeterminant>& space,
                             SigmaVector* sigma_vector, SharedVector Eigenvalues,
                             SharedMatrix Eigenvectors, int nroot, int multiplicity);

    bool davidson_liu_solver_map(const DeterminantMap& space, SigmaVector* sigma_vector,
                                 SharedVector Eigenvalues, SharedMatrix Eigenvectors, int nroot,
                                 int multiplicity);
    /// Use a OMP parallel algorithm?
    bool parallel_ = false;
    /// Print details?
    bool print_details_ = true;
    /// Project solutions onto given multiplicity?
    bool spin_project_ = false;
    /// Project solutions onto given multiplicity in full algorithm?
    bool spin_project_full_ = true;
    /// Project solutions onto given root?
    bool root_project_ = false;
    /// The energy convergence threshold
    double e_convergence_ = 1.0e-12;
    /// Number of collapse vectors per roots
    int ncollapse_per_root_ = 2;
    /// Number of max subspace vectors per roots
    int nsubspace_per_root_ = 4;
    /// Maximum number of iterations in the Davidson-Liu algorithm
    int maxiter_davidson_ = 100;
    /// Initial guess size per root
    size_t dl_guess_ = 200;

    /// Additional roots to project out
    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

    /// Set the initial guess?
    bool set_guess_ = false;
    std::vector<std::pair<size_t, double>> guess_;
    // Number of guess vectors
    size_t nvec_ = 10;
    std::shared_ptr<FCIIntegrals> fci_ints_;

    /// The SigmaVector object for Davidson-Liu algorithm
    SigmaVector* sigma_vec_ = nullptr;
};
}
}

#endif // _sparse_ci_h_
