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

#ifndef _fci_solver_h_
#define _fci_solver_h_

#include "base_classes/active_space_method.h"
#include "psi4/libmints/dimension.h"

namespace forte {
class FCIVector;
class StringLists;
class SpinAdapter;
class DavidsonLiuSolver2;

/// @brief The FCISolver class
/// This class performs Full CI calculations in the active space.
/// The active space is defined by a set of orbitals and a number of electrons.
/// The class uses the Davidson-Liu algorithm to compute the FCI energy and the RDMs.
/// The class also provides the possibility to compute the transition RDMs between roots of
/// different symmetry.
///
/// This class uses a determinant-based approach to compute the sigma vectors.
/// It can also run spin-adapted computations using a basis of configuration state functions (CSFs)
/// instead of determinants.
class FCISolver : public ActiveSpaceMethod {
    using sparse_vec = std::vector<std::pair<size_t, double>>;
    using sparse_mat = std::vector<std::vector<std::pair<size_t, double>>>;

  public:
    // ==> Class Constructor and Destructor <==

    /// @brief Construct a FCISolver object
    /// @param state the electronic state to compute
    /// @param nroot the number of roots
    /// @param mo_space_info a MOSpaceInfo object that defines the orbital spaces
    /// @param as_ints molecular integrals defined only for the active space orbitals
    FCISolver(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
              std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    ~FCISolver() = default;

    // ==> Class Interface <==

    /// Compute the FCI energy
    double compute_energy() override;

    /// Returns the reduced density matrices up to a given rank (max_rdm_level)
    std::vector<std::shared_ptr<RDMs>> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                            int max_rdm_level, RDMsType type) override;

    /// Returns the transition reduced density matrices between roots of different symmetry up to a
    /// given level (max_rdm_level)
    std::vector<std::shared_ptr<RDMs>>
    transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                    RDMsType type) override;

    /// Set the options
    void set_options(std::shared_ptr<ForteOptions> options) override;

    /// Compute RDMs on a given root
    void compute_rdms_root(size_t root1, size_t root2, int max_rdm_level);

    /// Set the number of determinants per root to use to form the initial guess
    void set_ndets_per_guess_state(size_t value);

    /// Set the maximum number of DL iterations
    void set_maxiter_davidson(int value);

    /// Set the number of guess vectors to use
    void set_guess_per_root(int value);

    /// Set the number of collapse vectors for each root
    void set_collapse_per_root(int value);

    /// Set the maximum subspace size for each root
    void set_subspace_per_root(int value);

    /// Spin adapt the FCI wave function
    void set_spin_adapt(bool value);

    /// Spin adapt the FCI wave function using a full preconditioner?
    void set_spin_adapt_full_preconditioner(bool value);

    /// When set to true before calling compute_energy(), it will test the
    /// reduce density matrices.  Watch out, this function is very slow!
    void set_test_rdms(bool value);

    /// Print the Natural Orbitals
    void set_print_no(bool value);

    /// Return a FCIVector
    std::shared_ptr<FCIVector> get_FCIWFN();

    /// Return eigen vectors (n_DL_guesses x ndets)
    std::shared_ptr<psi::Matrix> evecs();

    /// Return the CI wave functions for current state symmetry (ndets x nroots)
    std::shared_ptr<psi::Matrix> ci_wave_functions() override;

    /// Return string lists
    std::shared_ptr<StringLists> lists();

    /// Return symmetry
    int symmetry();

  private:
    // ==> Class Data <==

    /// The psi::Dimension object for the active space
    psi::Dimension active_dim_;

    /// A object that stores string information
    std::shared_ptr<StringLists> lists_;

    /// A object that handles spin adaptation
    std::shared_ptr<SpinAdapter> spin_adapter_;

    /// The FCI energy
    double energy_;

    /// The FCI wave function
    std::shared_ptr<FCIVector> C_;

    /// The FCI determinant list
    std::vector<Determinant> dets_;

    /// The Davidson-Liu-Solver object
    std::shared_ptr<DavidsonLiuSolver2> dl_solver_;

    /// Eigenvectors
    std::shared_ptr<psi::Matrix> eigen_vecs_;

    /// The number of irreps
    int nirrep_;
    /// The symmetry of the wave function
    int symmetry_;
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The number of guess vectors for each root
    size_t guess_per_root_ = 2;
    /// The number of collapse vectors for each root.
    size_t collapse_per_root_ = 2;
    /// The maximum subspace size for each root
    size_t subspace_per_root_ = 4;
    /// The number of determinants selected for each guess vector
    size_t ndets_per_guess_ = 10;
    /// Iterations for FCI
    int maxiter_davidson_ = 30;
    /// Test the RDMs?
    bool test_rdms_ = false;
    /// Print the NO from the 1-RDM
    bool print_no_ = false;
    /// Spin adapt the FCI wave function?
    bool spin_adapt_ = false;
    /// Use the full preconditioner for spin adaptation?
    /// When set to false, it uses an approximate diagonal preconditioner
    bool spin_adapt_full_preconditioner_ = false;

    // ==> Private class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// @brief Compute initial guess vectors in the determinant basis
    /// @note  This function is used when spin-adaptation is not used. Here we generate
    /// num_guess_states
    /// @param diag The diagonal of the Hamiltonian in the determinant basis
    /// @param num_guess_states The number of guess vectors to generate
    /// @param fci_ints The integrals object
    /// @param temp A temporary vector of dimension ndets to store the guess vectors
    std::pair<sparse_mat, sparse_mat>
    initial_guess_det(FCIVector& diag, size_t num_guess_states,
                      std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

    /// @brief Generate at least num_guess_states of the lowest energy determinants
    /// @param diag
    /// @param num_guess_states
    /// @return
    std::vector<Determinant> initial_guess_generate_dets(FCIVector& diag, size_t num_guess_states);

    /// @brief Compute initial guess vectors in the CSF basis
    /// @note  This function is only used for spin-adapted calculations and works differnt than
    /// the one used to generate the guess vectors in the determinant basis. Here we generate
    /// num_guess_states CSFs and use those as initial guess.
    /// @param diag The diagonal of the Hamiltonian in the CSF basis
    /// @param num_guess_states The number of guess vectors to generate
    /// @param temp A temporary vector of dimension ncfs to store the guess vectors
    sparse_mat initial_guess_csf(std::shared_ptr<psi::Vector> diag, size_t num_guess_states);

    /// @brief Compute the diagonal of the Hamiltonian in the CSF basis
    /// @param fci_ints The integrals object
    /// @param spin_adapter The spin adapter object
    std::shared_ptr<psi::Vector> form_Hdiag_csf(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                                std::shared_ptr<SpinAdapter> spin_adapter);

    /// @brief Print a summary of the FCI calculation
    void print_solutions(size_t guess_size, std::shared_ptr<psi::Vector> b,
                         std::shared_ptr<psi::Vector> b_basis,
                         std::shared_ptr<DavidsonLiuSolver2> dls);

    /// @brief Test the RDMs
    void test_rdms(std::shared_ptr<psi::Vector> b, std::shared_ptr<psi::Vector> b_basis,
                   std::shared_ptr<DavidsonLiuSolver2> dls);
};
} // namespace forte

#endif // _fci_solver_h_
