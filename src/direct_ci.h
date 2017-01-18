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

#ifndef _sparse_ci_h_
#define _sparse_ci_h_

#include "stl_bitset_determinant.h"
#include "operator.h"
#include "determinant_map.h"

#define BIGNUM 1E100
#define MAXIT 100

namespace psi{ namespace forte{

enum DiagonalizationMethod {Full,DLSolver};

/**
 * @brief The SigmaBuilder class
 * Base class for a sigma vector object.
 */
class SigmaBuilder
{
public:
    SigmaBuilder( DeterminantMap& wfn, WFNOperator& op );

    size_t size() {return size_;}

    void compute_sigma(SharedVector sigma, SharedVector b);
    void get_diagonal(Vector& diag);

protected:

    DeterminantMap& wfn_;

    WFNOperator& op_;

    size_t size_;

//    // Create the list of a_p|N>
//    std::vector<std::vector<std::pair<size_t,short>>>& a_ann_list;
//    std::vector<std::vector<std::pair<size_t,short>>>& b_ann_list;
//    // Create the list of a+_q |N-1>
//    std::vector<std::vector<std::pair<size_t,short>>>& a_cre_list;
//    std::vector<std::vector<std::pair<size_t,short>>>& b_cre_list;
//
//    // Create the list of a_q a_p|N>
//    std::vector<std::vector<std::tuple<size_t,short,short>>>& aa_ann_list;
//    std::vector<std::vector<std::tuple<size_t,short,short>>>& ab_ann_list;
//    std::vector<std::vector<std::tuple<size_t,short,short>>>& bb_ann_list;
//    // Create the list of a+_s a+_r |N-2>
//    std::vector<std::vector<std::tuple<size_t,short,short>>>& aa_cre_list;
//    std::vector<std::vector<std::tuple<size_t,short,short>>>& ab_cre_list;
//    std::vector<std::vector<std::tuple<size_t,short,short>>>& bb_cre_list;
    std::vector<double> diag_;

	bool print_details_ = true;
};


/**
 * @brief The SparseCISolver class
 * This class diagonalizes the Hamiltonian in a basis
 * of determinants.
 */
class DirectCI
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
    void diagonalize_hamiltonian( DeterminantMap& wfn, WFNOperator& op, SharedMatrix& evecs, SharedVector& evals, int nroot, int multiplicity, DiagonalizationMethod diag_method );

    /// Enable/disable the parallel algorithms
    void set_parallel(bool parallel) {parallel_ = parallel;}

    /// Enable/disable printing of details
    void set_print_details(bool print_details) {print_details_ = print_details;}

    /// Enable/disable spin projection
    void set_spin_project(bool value);

    /// Set convergence threshold
    void set_e_convergence(double value);

    /// Set true to ignore the size test of the space in diagonalize_hamiltonian
    void set_force_diag_method(bool force_diag_method) {force_diag_method_ = force_diag_method;}

    /// The maximum number of iterations for the Davidson algorithm
    void set_maxiter_davidson(int value);

private:
    /// Form the full Hamiltonian and diagonalize it (for debugging)
    void diagonalize_full( DeterminantMap& wfn, WFNOperator& op, SharedMatrix& evecs,  SharedVector& evals );

    void diagonalize_davidson_liu( DeterminantMap& wfn, WFNOperator& op, SharedMatrix& evecs, SharedVector& evals, int nroot, int multiplicity);

    SharedMatrix build_full_hamiltonian( DeterminantMap& wfn, WFNOperator& op );

    std::vector<std::pair<double, std::vector<std::pair<size_t, double> > > > initial_guess( DeterminantMap& wfn, int nroot, int multiplicity);

    /// The Davidson-Liu algorithm
    bool davidson_liu_solver( DeterminantMap& wfn, SigmaBuilder& svl, SharedMatrix Eigenvectors, SharedVector Eigenvalues, int nroot, int multiplicity);

    /// Use a OMP parallel algorithm?
    bool parallel_ = false;
    /// Print details?
    bool print_details_ = true;
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
    /// Force to use diag_method no matter how small the space is
    bool force_diag_method_ = false;
};

}}

#endif // _sparse_ci_h_

