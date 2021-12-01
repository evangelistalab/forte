/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "sparse_ci/determinant_hashvector.h"

#define BIGNUM 1E100
#define MAXIT 100

namespace psi {
class Matrix;
class Vector;
} // namespace psi

namespace forte {

class SigmaVector;
class ActiveSpaceIntegrals;

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
     * Determinant objects
     * @param nroot The number of solutions to find
     * @param diag_method The diagonalization algorithm
     * @param multiplicity The spin multiplicity of the solution (2S + 1).  1 =
     * singlet, 2 = doublet, ...
     */

    SparseCISolver();

    /// Diagonalize the Hamiltonian
    std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
    diagonalize_hamiltonian(const DeterminantHashVec& space, std::shared_ptr<SigmaVector> sigma_vec,
                            int nroot, int multiplicity);

    std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
    diagonalize_hamiltonian(const std::vector<Determinant>& space,
                            std::shared_ptr<SigmaVector> sigma_vec, int nroot, int multiplicity);

    /// Diagonalize the full Hamiltonian
    std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
    diagonalize_hamiltonian_full(const std::vector<Determinant>& space,
                                 std::shared_ptr<ActiveSpaceIntegrals> as_ints, int nroot,
                                 int multiplicity);

    std::vector<double> spin() { return spin_; }

    std::vector<double> energy() { return energies_; }

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

    /// Set the energy convergence threshold
    void set_e_convergence(double value);

    /// Set the residual 2-norm convergence threshold
    void set_r_convergence(double value);

    /// The maximum number of iterations for the Davidson algorithm
    void set_maxiter_davidson(int value);

    void set_ncollapse_per_root(int value);
    void set_nsubspace_per_root(int value);

    /// Build the full Hamiltonian matrix
    std::shared_ptr<psi::Matrix>
    build_full_hamiltonian(const std::vector<Determinant>& space,
                           std::shared_ptr<forte::ActiveSpaceIntegrals> as_ints);

    /// Add roots to project out during Davidson-Liu procedure
    void add_bad_states(std::vector<std::vector<std::pair<size_t, double>>>& roots);

    /// Set option to force diagonalization type
    void set_force_diag(bool value);

    /// Set the size of the guess space
    void set_guess_dimension(size_t value) { dl_guess_ = value; }

    /// Set the initial guess
    void set_initial_guess(const std::vector<std::vector<std::pair<size_t, double>>>& guess);
    void manual_guess(bool value);
    void set_num_vecs(size_t value);

  private:
    std::vector<std::tuple<int, double, std::vector<std::pair<size_t, double>>>>
    initial_guess(const DeterminantHashVec& space, std::shared_ptr<SigmaVector> sigma_vector,
                  int nroot, int multiplicity);

    bool davidson_liu_solver(const DeterminantHashVec& space,
                             std::shared_ptr<SigmaVector> sigma_vector,
                             std::shared_ptr<psi::Vector> Eigenvalues,
                             std::shared_ptr<psi::Matrix> Eigenvectors, int nroot,
                             int multiplicity);

    /// The energy of each state
    std::vector<double> energies_;
    /// The expectation value of S^2 for each state
    std::vector<double> spin_;
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
    /// The residual 2-norm convergence threshold
    double r_convergence_ = 1.0e-6;
    /// Number of collapse vectors per roots
    int ncollapse_per_root_ = 2;
    /// Number of max subspace vectors per roots
    int nsubspace_per_root_ = 4;
    /// Maximum number of iterations in the Davidson-Liu algorithm
    int maxiter_davidson_ = 100;
    /// Number of determinants used to form guess vector per root
    size_t dl_guess_ = 50;
    /// Options for forcing diagonalization method
    bool force_diag_ = false;
    /// Additional roots to project out
    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

    /// Set the initial guess?
    bool set_guess_ = false;
    std::vector<std::vector<std::pair<size_t, double>>> guess_; // nroot of guess size of (id, coefficent)
    // Number of guess vectors
    size_t nvec_ = 10;
};
} // namespace forte

#endif // _sparse_ci_h_
