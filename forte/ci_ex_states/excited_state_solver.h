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

#ifndef _excited_state_solver_h_
#define _excited_state_solver_h_

#include "base_classes/active_space_method.h"
#include "base_classes/rdms.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/sparse_ci_solver.h"

namespace psi {
class Matrix;
}

namespace forte {
class SelectedCIMethod;
class DeterminantSubstitutionLists;

class ExcitedStateSolver : public ActiveSpaceMethod {
  public:
    ExcitedStateSolver(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
                       std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                       std::unique_ptr<SelectedCIMethod> sci);

    /// Define a destructor to remove wfn
    ~ExcitedStateSolver();

    // ==> Class Interface <==

    /// Compute the energy and return it
    virtual double compute_energy() override;

    /// Returns the reduced density matrices up to a given level (max_rdm_level)
    std::vector<std::shared_ptr<RDMs>> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                            int max_rdm_level, RDMsType rdm_type) override;

    /// Returns the transition reduced density matrices between roots of different symmetry up to a
    /// given level (max_rdm_level)
    std::vector<std::shared_ptr<RDMs>>
    transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                    RDMsType rdm_type) override;

    /// Set options from an option object
    /// @param options the options passed in
    virtual void set_options(std::shared_ptr<ForteOptions> options) override;

    //    void add_external_excitations(DeterminantHashVec& ref);

    /// Set excitation algorithm
    void set_excitation_algorithm(std::string ex_alg);

    /// Set core excitation
    void set_core_excitation(bool core_ex);

    /// Dump temporary wave function to disk
    void dump_wave_function(const std::string& filename) override;

    /// Read temporary ave function from disk
    /// Return the number of active orbitals, set of determinants, CI coefficients
    std::tuple<size_t, std::vector<Determinant>, psi::SharedMatrix>
    read_wave_function(const std::string& filename) override;

  protected:
    /// The number of active orbitals
    size_t nact_;
    std::unique_ptr<SelectedCIMethod> sci_;
    std::shared_ptr<SparseCISolver> sparse_solver_;
    /// Algorithm for computing excited states
    std::string ex_alg_;
    /// Type of excited state to compute
    bool core_ex_;
    /// Storage of past roots
    std::vector<std::vector<std::pair<Determinant, double>>> old_roots_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    /// Computes RDMs without coupling lists
    bool direct_rdms_ = false;
    /// Run test for the RDMs
    bool test_rdms_ = false;
    /// Print final wavefunction to file
    bool save_final_wfn_ = false;
    /// Compute all roots on first iteration?
    bool first_iter_roots_ = false;
    /// Do full EN-MRPT2 correction?
    bool full_pt2_ = false;
    /// Calculate transition dipole?
    bool transition_dipole_ = false;

  private:
    /// Print information about this calculation
    void print_info();
    /// Save older roots
    void save_old_root(DeterminantHashVec& dets, std::shared_ptr<psi::Matrix>& PQ_evecs, int root,
                       int ref_root);

    void compute_multistate(psi::SharedVector& PQ_evals);

    /// Print Summary
    void print_final(DeterminantHashVec& dets, std::shared_ptr<psi::Matrix>& PQ_evecs,
                     psi::SharedVector& PQ_evals, size_t cycle);
    /// Save a wave function
    void wfn_to_file(DeterminantHashVec& det_space, std::shared_ptr<psi::Matrix> evecs, int root);
    /// Print a wave function
    void print_wfn(DeterminantHashVec& space, std::shared_ptr<psi::Matrix> evecs, int nroot);

    /// Compute the RDMs
    std::shared_ptr<RDMs> compute_rdms(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                       DeterminantHashVec& dets,
                                       std::shared_ptr<psi::Matrix>& PQ_evecs, int root1, int root2,
                                       int max_rdm_level, RDMsType rdm_type);
};
} // namespace forte
#endif // _excited_state_solver_h_
