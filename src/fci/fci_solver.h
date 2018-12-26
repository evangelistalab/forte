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

#ifndef _fci_solver_h_
#define _fci_solver_h_

#include "psi4/libmints/wavefunction.h"
#include "psi4/physconst.h"

#include "fci_vector.h"

#include "helpers/mo_space_info.h"
#include "helpers/timer.h"
#include "integrals/integrals.h"
#include "string_lists.h"
#include "base_classes/reference.h"
#include "base_classes/active_space_solver.h"
#include "forte_options.h"

namespace forte {

/**
 * @brief The FCISolver class
 * This class performs Full CI calculations.
 */
class FCISolver : public ActiveSpaceSolver {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * @brief FCISolver
     * @param active_dim The dimension of the active orbital space
     * @param core_mo A vector of doubly occupied orbitals
     * @param active_mo A vector of active orbitals
     * @param state_ state information including na, nb, multiplicity and symmetry
     * @param ints An integral object
     * @param mo_space_info Information about molecular orbital spaces
     * @param initial_guess_per_root get from options object
     * @param print Control printing of FCISolver
     */
    //    FCISolver(psi::Dimension active_dim, std::vector<size_t> core_mo, std::vector<size_t>
    //    active_mo,
    //              StateInfo state, std::shared_ptr<ForteIntegrals> ints,
    //              std::shared_ptr<MOSpaceInfo> mo_space_info, size_t initial_guess_per_root, int
    //              print, ForteOptions options);

    FCISolver(StateInfo state, std::shared_ptr<MOSpaceInfo> mo_space_info,
              std::shared_ptr<ForteIntegrals> ints);

    ~FCISolver() {}

    /// Compute the FCI energy
    double compute_energy() override;

    /// Return a reference object
    Reference get_reference() override;

    /// Set the options
    void set_options(std::shared_ptr<ForteOptions> options) override;

    /// Compute RDMs on a given root
    void compute_rdms_root(int root);

    /// Set the number of trial vectors per root
    void set_ntrial_per_root(int value);
    /// Set the energy convergence threshold
    void set_e_convergence(double value);
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
    /// Set the print level
    void set_print(int level) { print_ = level; }
    /// Return a FCIWfn
    std::shared_ptr<FCIWfn> get_FCIWFN() { return C_; }

    /// Return eigen vectors
    psi::SharedMatrix eigen_vecs() { return eigen_vecs_; }
    /// Return eigen values
    psi::SharedVector eigen_vals() { return eigen_vals_; }
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
    std::shared_ptr<FCIWfn> C_;

    /// Eigen vectors
    psi::SharedMatrix eigen_vecs_;
    /// Eigen values
    psi::SharedVector eigen_vals_;

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
    /// The energy convergence criterion
    double e_convergence_ = 1.0e-12;
    /// Iterations for FCI
    int fci_iterations_ = 30;
    /// Test the RDMs?
    bool test_rdms_ = false;
    /// Print the NO from the 1-RDM
    bool print_no_ = false;
    /// A variable to control printing information
    int print_ = 0;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Initial CI wave function guess
    std::vector<std::pair<int, std::vector<std::tuple<size_t, size_t, size_t, double>>>>
    initial_guess(FCIWfn& diag, size_t n, std::shared_ptr<ActiveSpaceIntegrals> fci_ints);
};
} // namespace forte

#endif // _fci_solver_h_
