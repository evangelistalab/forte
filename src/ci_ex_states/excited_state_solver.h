/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "sparse_ci/determinant_hashvector.h"
#include "base_classes/active_space_method.h"
#include "sparse_ci/sparse_ci_solver.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

namespace forte {
class SelectedCIMethod;

class ExcitedStateSolver : public ActiveSpaceMethod {
  public:
    ExcitedStateSolver(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
                       std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                       std::shared_ptr<SelectedCIMethod> sci);

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~ExcitedStateSolver() = default;

    // ==> Class Interface <==

    /// Compute the energy and return it
    virtual double compute_energy() override;

    /// Returns the reference
    virtual std::vector<Reference>
    reference(std::vector<std::pair<size_t, size_t>>& roots) override = 0;

    /// Set options from an option object
    /// @param options the options passed in
    virtual void set_options(std::shared_ptr<ForteOptions> options) override;

    /// Set excitation algorithm
    void set_excitation_algorithm(std::string ex_alg);

    /// Set excitation type
    void set_excitation_type(std::string ex_type);

    /// Set the printing level
    void set_quiet(bool quiet);

  protected:
    DeterminantHashVec final_wfn_;
    WFNOperator op_;

    std::shared_ptr<SelectedCIMethod> sci_;
    std::shared_ptr<SparseCISolver> sparse_solver_;
    /// Algorithm for computing excited states
    std::string ex_alg_;
    /// Type of excited state to compute
    std::string ex_type_;
    /// Control amount of printing
    bool quiet_mode_;
    /// Storage of past roots
    std::vector<std::vector<std::pair<Determinant, double>>> old_roots_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    /// The eigensolver type
    DiagonalizationMethod diag_method_ = DLString;
    /// The CI coeffiecients
    psi::SharedMatrix evecs_;
    /// Adds all active single excitations to the final wave function
    bool add_singles_ = false;

  private:
    /// Print information about this calculation
    void print_info();
    /// Save older roots
    void save_old_root(DeterminantHashVec& dets, psi::SharedMatrix& PQ_evecs, int root,
                       int ref_root);

    void compute_multistate(psi::SharedVector& PQ_evals);
    /// Computes spin
    std::vector<std::pair<double, double>> compute_spin(DeterminantHashVec& space, WFNOperator& op,
                                                        psi::SharedMatrix evecs, int nroot);
    /// Print Summary
    void print_final(DeterminantHashVec& dets, psi::SharedMatrix& PQ_evecs,
                     psi::SharedVector& PQ_evals);
    /// Print a wave function
    void print_wfn(DeterminantHashVec& space, WFNOperator& op, psi::SharedMatrix evecs, int nroot);
};
}
#endif // _excited_state_solver_h_
