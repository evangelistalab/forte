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
#include "psi4/liboptions/liboptions.h"
#include "psi4/physconst.h"

#include "fci_vector.h"

#include "../helpers.h"
#include "../integrals/integrals.h"
#include "string_lists.h"
#include "../reference.h"

namespace psi {
namespace forte {

/**
 * @brief The FCISolver class
 * This class performs Full CI calculations.
 */
class FCISolver {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * @brief FCISolver
     * @param active_dim The dimension of the active orbital space
     * @param core_mo A vector of doubly occupied orbitals
     * @param active_mo A vector of active orbitals
     * @param na Number of alpha electrons
     * @param nb Number of beta electrons
     * @param multiplicity The spin multiplicity (2S + 1).  1 = singlet, 2 =
     * doublet, ...
     * @param symmetry The irrep of the FCI wave function
     * @param ints An integral object
     * @param mo_space_info -> MOSpaceInfo
     * @param initial_guess_per_root get from options object
     * @param print Control printing of FCISolver
     */
    FCISolver(Dimension active_dim, std::vector<size_t> core_mo, std::vector<size_t> active_mo,
              size_t na, size_t nb, size_t multiplicity, size_t symmetry,
              std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
              size_t initial_guess_per_root, int print, Options& options);
    /**
     * @brief FCISolver
     * @param active_dim The dimension of the active orbital space
     * @param core_mo A Vector of doubly occupied orbitals
     * @param active_mo A vector of active orbitals
     * @param na Number of alpha electrons
     * @param nb Number of beta electrons
     * @param multiplicity The spin multiplicity (2S + 1)
     * @param symmetry The Irrep of the FCI wave function
     * @param ints An integral object
     * @param mo_space_info -> mo_space_info object
     * @param options object
     */
    FCISolver(Dimension active_dim, std::vector<size_t> core_mo, std::vector<size_t> active_mo,
              size_t na, size_t nb, size_t multiplicity, size_t symmetry,
              std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
              Options& options);

    ~FCISolver() {}

    /// Compute the FCI energy
    double compute_energy();

    //double compute_energy_in_subspace();

    /// Compute RDMs on a given root
    void compute_rdms_root(int root);

    /// Return a reference object
    Reference reference();

    /// Set the number of desired roots
    void set_nroot(int value);
    /// Set the root that will be used to compute the properties
    void set_root(int value);
    /// Set the maximum RDM computed (0 - 3)
    void set_max_rdm_level(int value);
    /// Set the convergence for FCI
    void set_fci_iterations(int value);
    /// Set the number of collapse vectors for each root
    void set_collapse_per_root(int value);
    /// Set the maximum subspace size for each root
    void set_subspace_per_root(int value);

    /// If you actually change the integrals in your code, you should set this
    /// to false.
    void use_user_integrals_and_restricted_docc(bool user_provide_integrals) {
        provide_integrals_and_restricted_docc_ = user_provide_integrals;
    }
    /// If you want to use your own integrals need to set FCIIntegrals (This is
    /// normally not set)
    void set_integral_pointer(std::shared_ptr<FCIIntegrals> fci_ints) { fci_ints_ = fci_ints; }

    /// When set to true before calling compute_energy(), it will test the
    /// reduce density matrices.  Watch out, this function is very slow!
    void set_test_rdms(bool value) { test_rdms_ = value; }
    /// Print the Natural Orbitals
    void set_print_no(bool value) { print_no_ = value; }
    /// Return a FCIWfn
    std::shared_ptr<FCIVector> get_FCIWFN() { return C_; }

    /// Return eigen vectors
    SharedMatrix eigen_vecs() { return eigen_vecs_; }
    /// Return eigen values
    SharedVector eigen_vals() { return eigen_vals_; }
    /// Return string lists
    std::shared_ptr<StringLists> lists() { return lists_; }
    /// Return symmetry
    int symmetry() { return symmetry_; }

  private:
    // ==> Class Data <==

    /// The Dimension object for the active space
    Dimension active_dim_;

    /// The orbitals frozen at the CI level
    std::vector<size_t> core_mo_;

    /// The orbitals treated at the CI level
    std::vector<size_t> active_mo_;

    /// A object that stores string information
    std::shared_ptr<StringLists> lists_;

    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;

    std::shared_ptr<FCIIntegrals> fci_ints_;

    /// The FCI energy
    double energy_;

    /// The FCI wave function
    std::shared_ptr<FCIVector> C_;

    /// Eigen vectors
    SharedMatrix eigen_vecs_;
    /// Eigen values
    SharedVector eigen_vals_;

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
    /// The number of roots (default = 1)
    int nroot_ = 1;
    /// The root used to compute properties (zero based, default = 0)
    int root_ = 0;
    /// The number of trial guess vectors to generate per root
    size_t ntrial_per_root_;
    /// The number of collapse vectors for each root
    size_t collapse_per_root_ = 2;
    /// The maximum subspace size for each root
    size_t subspace_per_root_ = 4;
    /// The maximum RDM computed (0 - 3)
    int max_rdm_level_;
    /// Iterations for FCI
    int fci_iterations_ = 30;
    /// Test the RDMs?
    bool test_rdms_ = false;
    /// Print the NO from the 1-RDM
    bool print_no_ = false;
    /// A variable to control printing information
    int print_ = 0;
    /// Use the user specified integrals and restricted_docc_operator
    bool provide_integrals_and_restricted_docc_ = false;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();
    /// The mo_space_info object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Initial CI wave function guess
    std::vector<std::pair<int, std::vector<std::tuple<size_t, size_t, size_t, double>>>>
    initial_guess(FCIVector& diag, size_t n, size_t multiplicity,
                  std::shared_ptr<FCIIntegrals> fci_ints);
    /// The options object
    Options& options_;

    double subspace_energy();

    /// Decompose and reconstruct the FCI wave function
    void fci_svd(FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy, double TAU);

    void fci_svd_tiles(FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy, int tile_dim, double OMEGA);

    void string_stats(std::vector<SharedMatrix> C);

    void string_trimmer(std::vector<SharedMatrix>& C, double DELTA, FCIVector& HC,
                        std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy);

    void tile_chopper(std::vector<SharedMatrix>& C, double ETA,
                      FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints,
                      double fci_energy, int dim);

    void zero_tile(std::vector<SharedMatrix>& C,
                   std::vector<int> b_r,
                   std::vector<int> e_r,
                   std::vector<int> b_c,
                   std::vector<int> e_c,
                   double tile_norm_cut,
                   int dim, int n, int d,
                   int h, int i, int j, int& Npar );

    void py_mat_print(SharedMatrix C_h, const std::string& input);

    void add_to_sig_vect(std::vector<std::tuple<double, int, int, int> >& sorted_sigma,
                         std::vector<SharedMatrix> C,
                         std::vector<int> b_r,
                         std::vector<int> e_r,
                         std::vector<int> b_c,
                         std::vector<int> e_c,
                         int dim, int n, int d,
                         int h, int i, int j);

    void add_to_tle_vect(std::vector<SharedMatrix>& C,
                         std::vector<int> b_r,
                         std::vector<int> e_r,
                         std::vector<int> b_c,
                         std::vector<int> e_c,
                         int dim, int n, int d,
                         int h, int i, int j,
                         std::vector<std::tuple<double, int, int, int> >& sorted_tiles);

    void patch_Cmat(std::vector<std::tuple<double, int, int, int> >& sorted_sigma,
                         std::vector<SharedMatrix>& C,
                         std::vector<std::vector<std::vector<int> > > rank_tile_inirrep,
                         std::vector<int> b_r,
                         std::vector<int> e_r,
                         std::vector<int> b_c,
                         std::vector<int> e_c,
                         int dim, int n, int d,
                         int h, int i, int j,
                         int& N_par);

    void basis_cluster(std::vector<SharedMatrix>& C, std::vector<std::pair<double, int> >& st_vec);

    void rev_basis_cluster(std::vector<SharedMatrix>& C, std::vector<std::pair<double, int> > st_vec);

    static bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);

};
}
}

#endif // _fci_solver_h_
