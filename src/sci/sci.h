/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _sci_h_
#define _sci_h_

#include <memory>
#include <vector>

#include "base_classes/active_space_method.h"
#include "base_classes/state_info.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/determinant_substitution_lists.h"
#include "sparse_ci/sigma_vector.h"

namespace forte {

class ActiveSpaceIntegrals;
class ForteIntegrals;
class ForteOptions;
class MOSpaceInfo;
class Reference;
class SCFInfo;
class SparseCISolver;

class SelectedCIMethod {
  public:
    SelectedCIMethod(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~SelectedCIMethod() = default;

    // ==> Class Interface <==

    /// Set options from an option object
    /// @param options the options passed in
    virtual void set_options(std::shared_ptr<ForteOptions> options) = 0;

    /// Compute the energy and return it
    virtual double compute_energy();

    // Interfaces of SCI algorithm
    /// Print the banner and starting information.
    virtual void print_info() = 0;
    /// Pre-iter preparation, usually includes preparing an initial reference
    virtual void pre_iter_preparation() = 0;
    /// Step 1. Diagonalize the Hamiltonian in the P space
    virtual void diagonalize_P_space() = 0;
    /// Step 2. Find determinants in the Q space
    virtual void find_q_space() = 0;
    /// Step 3. Diagonalize the Hamiltonian in the P + Q space
    virtual void diagonalize_PQ_space() = 0;
    /// Step 4. Check convergence
    virtual bool check_convergence() = 0;
    /// Step 5. Prune the P + Q space to get an updated P space
    virtual void prune_PQ_to_P() = 0;
    /// Post-iter process
    virtual void post_iter_process() = 0;

    // Temporarily added interface to ExcitedStateSolver
    /// Set the class variable
    virtual void set_method_variables(
        std::string ex_alg, size_t nroot_method, size_t root,
        const std::vector<std::vector<std::pair<Determinant, double>>>& old_roots) = 0;
    /// Getters
    virtual DeterminantHashVec get_PQ_space() = 0;
    virtual psi::SharedMatrix get_PQ_evecs() = 0;
    virtual psi::SharedVector get_PQ_evals() = 0;
    //    virtual std::shared_ptr<WFNOperator> get_op() = 0;
    virtual size_t get_ref_root() = 0;
    virtual std::vector<double> get_multistate_pt2_energy_correction() = 0;
    virtual size_t get_cycle();

    void print_wfn(DeterminantHashVec& space, psi::SharedMatrix evecs, int nroot,
                   size_t max_dets_to_print = 10);

    SigmaVectorType sigma_vector_type() const;
    /// Return the maximum amount of memory allowed
    size_t max_memory() const;

  protected:
    /// The state to calculate
    StateInfo state_;

    /// The number of roots (default = 1)
    size_t nroot_ = 1;

    /// The sigma vector type
    SigmaVectorType sigma_vector_type_ = SigmaVectorType::Dynamic;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The molecular integrals for the active space
    /// This object holds only the integrals for the orbital contained in the active_mo_ vector.
    /// The one-electron integrals and scalar energy contains contributions from the
    /// doubly occupied orbitals specified by the core_mo_ vector.
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    /// Some HF info
    std::shared_ptr<SCFInfo> scf_info_;

    /// The sparse CI solver (allocated at creation by the base class)
    std::shared_ptr<SparseCISolver> sparse_solver_;

    /// The current iteration
    size_t cycle_;

    /// Maximum number of SCI iterations
    size_t max_cycle_;

    /// Maximum memory size
    size_t max_memory_ = 0;

    /// The number of active orbitals
    size_t nact_;
};
} // namespace forte

#endif // _sci_h_
