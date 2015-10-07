/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _sparse_ci_h_
#define _sparse_ci_h_

#include "dynamic_bitset_determinant.h"
#include "stl_bitset_determinant.h"

#define BIGNUM 1E100
#define MAXIT 100

namespace psi{ namespace forte{

enum DiagonalizationMethod {Full,DavidsonLiuDense,DavidsonLiuSparse,DavidsonLiuList};

/**
 * @brief The SigmaVector class
 * Base class for a sigma vector object.
 */
class SigmaVector
{
public:
    SigmaVector(size_t size) : size_(size) {};

    size_t size() {return size_;}

    virtual void compute_sigma(SharedVector sigma, SharedVector b) = 0;
    virtual void compute_sigma(Matrix& sigma, Matrix& b, int nroot) = 0;
    virtual void get_diagonal(Vector& diag) = 0;

protected:
    size_t size_;
};

/**
 * @brief The SigmaVectorFull class
 * Computes the sigma vector from a full Hamiltonian.
 */
class SigmaVectorFull : public SigmaVector
{
public:
    SigmaVectorFull(SharedMatrix H) : SigmaVector(H->ncol()), H_(H) {};

    void compute_sigma(SharedVector sigma, SharedVector b) {}
    void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);

protected:
    SharedMatrix H_;
};

/**
 * @brief The SigmaVectorSparse class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorSparse : public SigmaVector
{
public:
    SigmaVectorSparse(std::vector<std::pair<std::vector<int>,std::vector<double>>>& H) : SigmaVector(H.size()), H_(H) {};

    void compute_sigma(SharedVector sigma, SharedVector b) {}
    void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);

protected:
    std::vector<std::pair<std::vector<int>,std::vector<double>>>& H_;
};


/**
 * @brief The SigmaVectorSparse class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorSparse2 : public SigmaVector
{
public:
    SigmaVectorSparse2(std::vector<std::pair<std::vector<int>,SharedVector>>& H) : SigmaVector(H.size()), H_(H) {};

    void compute_sigma(SharedVector sigma, SharedVector b) {}
    void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);

protected:
    std::vector<std::pair<std::vector<int>,SharedVector>>& H_;
};


/**
 * @brief The SigmaVectorSparse class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorList : public SigmaVector
{
public:
    SigmaVectorList(const std::vector<STLBitsetDeterminant>& space);

    void compute_sigma(SharedVector sigma, SharedVector b);
    void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void get_hamiltonian(Matrix& H);
    std::vector<std::pair<std::vector<int>,std::vector<double>>> get_sparse_hamiltonian();

protected:
    const std::vector<STLBitsetDeterminant>& space_;
    // Create the list of a_p|N>
    std::vector<std::vector<std::pair<size_t,short>>> a_ann_list;
    std::vector<std::vector<std::pair<size_t,short>>> b_ann_list;
    // Create the list of a+_q |N-1>
    std::vector<std::vector<std::pair<size_t,short>>> a_cre_list;
    std::vector<std::vector<std::pair<size_t,short>>> b_cre_list;

    // Create the list of a_q a_p|N>
    std::vector<std::vector<std::tuple<size_t,short,short>>> aa_ann_list;
    std::vector<std::vector<std::tuple<size_t,short,short>>> ab_ann_list;
    std::vector<std::vector<std::tuple<size_t,short,short>>> bb_ann_list;
    // Create the list of a+_s a+_r |N-2>
    std::vector<std::vector<std::tuple<size_t,short,short>>> aa_cre_list;
    std::vector<std::vector<std::tuple<size_t,short,short>>> ab_cre_list;
    std::vector<std::vector<std::tuple<size_t,short,short>>> bb_cre_list;
    std::vector<double> diag_;
};


/**
 * @brief The SparseCISolver class
 * This class diagonalizes the Hamiltonian in a basis
 * of determinants.
 */
class SparseCISolver
{
public:    
    // ==> Class Interface <==

    /**
     * Diagonalize the Hamiltonian in a basis of determinants
     * @param space The basis for the CI given as a vector of STLBitsetDeterminant objects
     * @param nroot The number of solutions to find
     * @param diag_method The diagonalization algorithm
     * @param multiplicity The spin multiplicity of the solution (2S + 1).  1 = singlet, 2 = doublet, ...
     */
    void diagonalize_hamiltonian(const std::vector<STLBitsetDeterminant>& space,
                                   SharedVector& evals,
                                   SharedMatrix& evecs,
                                   int nroot,
                                   int multiplicity,
                                   DiagonalizationMethod diag_method);

	/**
	 * Compute the energy when CI vector is already known
	 * @param space The basis for the CI given as a vector of STLBitsetDeterminant objects
	 * @param evecs The known eigenvectors
	 * @param nroot The number of solutions to compute
	 */
	void compute_H_expectation_val(const std::vector<STLBitsetDeterminant> space,
									SharedVector& evals,
									const SharedMatrix evecs,
									int nroot,
									DiagonalizationMethod diag_method);

    /// Enable/disable the parallel algorithms
    void set_parallel(bool parallel) {parallel_ = parallel;}

    /// Enable/disable printing of details
    void set_print_details(bool print_details) {print_details_ = print_details;}

    /// Enable/disable spin projection
    void set_spin_project(bool value);

    /// Set convergence threshold
    void set_e_convergence(double value);

    /// The maximum number of iterations for the Davidson algorithm
    void set_maxiter_davidson(int value);

private:
    /// Form the full Hamiltonian and diagonalize it (for debugging)
    void diagonalize_full(const std::vector<STLBitsetDeterminant>& space,
                          SharedVector& evals,
                          SharedMatrix& evecs,
                          int nroot,
                          int multiplicity);

    /// Form the full Hamiltonian and use the Davidson-Liu method to compute the first nroot eigenvalues
    void diagonalize_davidson_liu_dense(const std::vector<STLBitsetDeterminant>& space,
                                        SharedVector& evals,
                                        SharedMatrix& evecs,
                                        int nroot,
                                        int multiplicity);

    /// Form a sparse Hamiltonian and use the Davidson-Liu method to compute the first nroot eigenvalues
    void diagonalize_davidson_liu_sparse(const std::vector<STLBitsetDeterminant>& space,
                                         SharedVector& evals,
                                         SharedMatrix& evecs,
                                         int nroot,
                                         int multiplicity);

    /// Form a sparse Hamiltonian using strings and use the Davidson-Liu method to compute the first nroot eigenvalues
    void diagonalize_davidson_liu_list(const std::vector<STLBitsetDeterminant> &space,
                                       SharedVector& evals,
                                       SharedMatrix& evecs,
                                       int nroot,
                                       int multiplicity);

    void diagonalize_davidson_liu_solver(const std::vector<STLBitsetDeterminant>& space, SharedVector& evals, SharedMatrix& evecs, int nroot, int multiplicity);

    /// Build the full Hamiltonian matrix
    SharedMatrix build_full_hamiltonian(const std::vector<STLBitsetDeterminant>& space);

    /// Build a sparse Hamiltonian matrix
    std::vector<std::pair<std::vector<int>,std::vector<double>>> build_sparse_hamiltonian(const std::vector<STLBitsetDeterminant> &space);
    std::vector<std::pair<std::vector<int>,std::vector<double>>> build_sparse_hamiltonian_parallel(const std::vector<STLBitsetDeterminant> &space);

    /// Computed initial guess for the Davidson-Liu algorithm
    std::vector<std::pair<double, std::vector<std::pair<size_t, double> > > > initial_guess(const std::vector<STLBitsetDeterminant>& space, int nroot, int multiplicity);

    /// The Davidson-Liu algorithm
    bool davidson_liu(SigmaVector* sigma_vector,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroot_s);
    bool davidson_liu_guess(std::vector<std::pair<double,std::vector<std::pair<size_t,double>>>> guess, SigmaVector* sigma_vector, SharedVector Eigenvalues, SharedMatrix Eigenvectors, int nroot, int multiplicity);
    bool davidson_liu_solver(const std::vector<STLBitsetDeterminant>& space,
                                             SigmaVector* sigma_vector,
                                             SharedVector Eigenvalues,
                                             SharedMatrix Eigenvectors,
                                             int nroot,
                                             int multiplicity);
    /// Use a OMP parallel algorithm?
    bool parallel_ = false;
    /// Print details?
    bool print_details_ = false;
    /// Project solutions onto given multiplicity?
    bool spin_project_ = false;
    /// The energy convergence threshold
    double e_convergence_ = 1.0e-12;
    /// Number of collapse vectors per roots
    int ncollapse_per_root_ = 2;
    /// Number of max subspace vectors per roots
    int nsubspace_per_root_ = 4;
    /// Maximum number of iterations in the Davidson-Liu algorithm
    int maxiter_davidson_ = 100;
};

}}

#endif // _sparse_ci_h_
