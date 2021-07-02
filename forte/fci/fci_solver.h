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

#ifndef _fci_solver_h_
#define _fci_solver_h_

#include "base_classes/active_space_method.h"
#include "psi4/libmints/dimension.h"

namespace forte {

class FCIVector;
class StringLists;

/**
 * @brief The FCISolver class
 * This class performs Full CI calculations.
 */
class FCISolver : public ActiveSpaceMethod {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * @brief FCISolver A class that performs a FCI computation in an active space
     * @param state the electronic state to compute
     * @param nroot the number of roots
     * @param mo_space_info a MOSpaceInfo object that defines the orbital spaces
     * @param as_ints molecular integrals defined only for the active space orbitals
     */
    FCISolver(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
              std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    ~FCISolver() = default;

    // ==> Class Interface <==

    /// Compute the FCI energy
    double compute_energy() override;

    /// Returns the reduced density matrices up to a given rank (max_rdm_level)
    std::vector<RDMs> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                           int max_rdm_level) override;

    /// Returns the transition reduced density matrices between roots of different symmetry up to a
    /// given level (max_rdm_level)
    std::vector<RDMs> transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                      std::shared_ptr<ActiveSpaceMethod> method2,
                                      int max_rdm_level) override;

    /// Set the options
    void set_options(std::shared_ptr<ForteOptions> options) override;

    /// Compute RDMs on a given root
    void compute_rdms_root(size_t root1, size_t root2, int max_rdm_level);

    /// Set the number of trial vectors per root
    void set_ntrial_per_root(int value);
    /// Set the convergence for FCI
    void set_fci_iterations(int value);
    /// Set the number of collapse vectors for each root
    void set_collapse_per_root(int value);
    /// Set the maximum subspace size for each root
    void set_subspace_per_root(int value);
    /// When set to true before calling compute_energy(), it will test the
    /// reduce density matrices.  Watch out, this function is very slow!
    void set_test_rdms(bool value) { test_rdms_ = value; }
    /// Print the Natural Orbitals
    void set_print_no(bool value) { print_no_ = value; }
    /// Return a FCIVector
    std::shared_ptr<FCIVector> get_FCIWFN() { return C_; }
    /// Return eigen vectors
    psi::SharedMatrix evecs() { return eigen_vecs_; }
    /// Return string lists
    std::shared_ptr<StringLists> lists() { return lists_; }
    /// Return symmetry
    int symmetry() { return symmetry_; }

  private:
    // ==> Class Data <==

    /// The psi::Dimension object for the active space
    psi::Dimension active_dim_;

    /// A object that stores string information
    std::shared_ptr<StringLists> lists_;

    /// The FCI energy
    double energy_;

    /// The FCI wave function
    std::shared_ptr<FCIVector> C_;

    /// Eigen vectors
    psi::SharedMatrix eigen_vecs_;

    /// The number of irreps
    int nirrep_;
    /// The symmetry of the wave function
    int symmetry_;
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The multiplicity (2S + 1) of the state to target.
    /// (1 = singlet, 2 = doublet, 3 = triplet, ...)
    int multiplicity_;
    /// The number of trial guess vectors to generate per root
    size_t ntrial_per_root_ = 1;
    /// The number of collapse vectors for each root
    size_t collapse_per_root_ = 2;
    /// The maximum subspace size for each root
    size_t subspace_per_root_ = 4;
    /// Iterations for FCI
    int fci_iterations_ = 30;
    /// Test the RDMs?
    bool test_rdms_ = false;
    /// Print the NO from the 1-RDM
    bool print_no_ = false;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Initial CI wave function guess
    std::vector<std::pair<int, std::vector<std::tuple<size_t, size_t, size_t, double>>>>
    initial_guess(FCIVector& diag, size_t n, std::shared_ptr<ActiveSpaceIntegrals> fci_ints);
};
} // namespace forte

#endif // _fci_solver_h_
