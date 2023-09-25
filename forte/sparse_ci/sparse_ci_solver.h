/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "psi4/libmints/dimension.h"

#define BIGNUM 1E100
#define MAXIT 100

namespace psi {
class Matrix;
class Vector;
} // namespace psi

namespace forte {

class SigmaVector;
class ActiveSpaceIntegrals;
class SpinAdapter;
class DavidsonLiuSolver;

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

    /// Spin adapt the wave function
    void set_spin_adapt(bool value);

    /// Spin adapt the wave function using a full preconditioner?
    void set_spin_adapt_full_preconditioner(bool value);

    /// Enable/disable root projection
    void set_root_project(bool value);

    /// Set the energy convergence threshold
    void set_e_convergence(double value);

    /// Set the residual 2-norm convergence threshold
    void set_r_convergence(double value);

    /// The maximum number of iterations for the Davidson algorithm
    void set_maxiter_davidson(int value);

    /// Set the number of guess vectors for each root
    void set_guess_per_root(int value);

    /// Set the number of collapse vectors for each root
    void set_collapse_per_root(int value);

    /// Set the maximum subspace size for each root
    void set_subspace_per_root(int value);

    /// Build the full Hamiltonian matrix
    std::shared_ptr<psi::Matrix>
    build_full_hamiltonian(const std::vector<Determinant>& space,
                           std::shared_ptr<forte::ActiveSpaceIntegrals> as_ints);

    /// Add roots to project out during Davidson-Liu procedure
    void add_bad_states(std::vector<std::vector<std::pair<size_t, double>>>& roots);

    /// Set option to force diagonalization type
    void set_force_diag(bool value);

    /// Set the number of determinants per root to use to form the initial guess
    void set_ndets_per_guess_state(size_t value) { ndets_per_guess_ = value; }

    /// Set the initial guess for the Davidson-Liu solver in the form of a vector of vectors of
    /// pairs of the form (determinant index, coefficient). If this vector is not empty, then we
    /// will use the user guess instead of the standard guess.
    void set_initial_guess(const std::vector<std::vector<std::pair<size_t, double>>>& guess);

    /// Reset the initial guess
    void reset_initial_guess();

  private:
    /// std::vector<std::tuple<int, double, std::vector<std::pair<size_t, double>>>>
    void initial_guess_det(const DeterminantHashVec& space,
                           std::shared_ptr<SigmaVector> sigma_vector, size_t guess_size,
                           DavidsonLiuSolver& dls, int multiplicity, bool do_spin_project);

    std::vector<Determinant>
    initial_guess_generate_dets(const DeterminantHashVec& space,
                                const std::shared_ptr<SigmaVector> sigma_vector,
                                const size_t num_guess_states) const;

    bool davidson_liu_solver(const DeterminantHashVec& space,
                             std::shared_ptr<SigmaVector> sigma_vector,
                             std::shared_ptr<psi::Vector> Eigenvalues,
                             std::shared_ptr<psi::Matrix> Eigenvectors, int nroot,
                             int multiplicity);

    /// @brief Compute initial guess vectors in the CSF basis
    /// @param diag The diagonal of the Hamiltonian in the CSF basis
    /// @param n The number of guess vectors to generate
    /// @param dls The Davidson-Liu-Solver object
    /// @param temp A temporary vector of dimension ncfs to store the guess vectors
    /// @param multiplicity The multiplicity
    void initial_guess_csf(std::shared_ptr<psi::Vector> diag, size_t n, DavidsonLiuSolver& dls,
                           std::shared_ptr<psi::Vector> temp, int multiplicity);

    /// @brief Compute the diagonal of the Hamiltonian in the CSF basis
    /// @param ci_ints The integrals object
    /// @param spin_adapter The spin adapter object
    std::shared_ptr<psi::Vector> form_Hdiag_csf(std::shared_ptr<ActiveSpaceIntegrals> ci_ints,
                                                std::shared_ptr<SpinAdapter> spin_adapter);

    /// The energy of each state
    std::vector<double> energies_;
    /// The FCI determinant list
    std::vector<Determinant> dets_;
    /// The number of correlated molecular orbitals per irrep
    psi::Dimension cmopi_;
    /// The expectation value of S^2 for each state
    std::vector<double> spin_;
    /// A object that handles spin adaptation
    std::shared_ptr<SpinAdapter> spin_adapter_;
    /// Use a OMP parallel algorithm?
    bool parallel_ = false;
    /// Print details?
    bool print_details_ = true;
    /// Project solutions onto given multiplicity?
    bool spin_project_ = false;
    /// Project solutions onto given multiplicity in full algorithm?
    bool spin_project_full_ = true;
    /// Spin adapt the wave function?
    bool spin_adapt_ = false;
    /// Use the full preconditioner for spin adaptation?
    /// When set to false, it uses an approximate diagonal preconditioner
    bool spin_adapt_full_preconditioner_ = false;
    /// Project solutions onto given root?
    bool root_project_ = false;
    /// The energy convergence threshold
    double e_convergence_ = 1.0e-12;
    /// The residual 2-norm convergence threshold
    double r_convergence_ = 1.0e-6;
    /// The number of guess vectors for each root
    size_t guess_per_root_ = 2;
    /// Number of collapse vectors per roots
    size_t collapse_per_root_ = 2;
    /// Number of max subspace vectors per roots
    size_t subspace_per_root_ = 4;
    /// Maximum number of iterations in the Davidson-Liu algorithm
    int maxiter_davidson_ = 100;
    /// Number of determinants used to form guess vector per root
    size_t ndets_per_guess_ = 10;
    /// Options for forcing diagonalization method
    bool force_diag_ = false;
    /// Additional roots to project out
    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;
    /// A variable to control printing information
    int print_ = 0;
    // nroot of guess size of (id, coefficent)
    std::vector<std::vector<std::pair<size_t, double>>> user_guess_;
};
} // namespace forte

#endif // _sparse_ci_h_
