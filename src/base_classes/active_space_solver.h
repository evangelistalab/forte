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

#ifndef _active_space_solver_h_
#define _active_space_solver_h_

#include <vector>

#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"
#include "base_classes/reference.h"

namespace forte {

class ForteIntegrals;
class ActiveSpaceIntegrals;
class MOSpaceInfo;

class ActiveSpaceSolver {
  public:
    // Constructor for a single state computation
    ActiveSpaceSolver(StateInfo state, std::shared_ptr<MOSpaceInfo> mo_space_info,
                      std::shared_ptr<ForteIntegrals> ints);
    // Constructor for a multi-state computation
    ActiveSpaceSolver(const std::vector<std::pair<StateInfo, double>>& states_weights,
                      std::shared_ptr<MOSpaceInfo> mo_space_info,
                      std::shared_ptr<ForteIntegrals> ints);

    // Default constructor
    ActiveSpaceSolver() = default;

    // enable deletion of a Derived* through a Base*
    virtual ~ActiveSpaceSolver() = default;

    /// Compute the energy and return it
    virtual double compute_energy() = 0;

    /// Returns the reference
    virtual Reference get_reference() = 0;

    virtual void set_options(std::shared_ptr<ForteOptions> options) = 0;

    /// Pass a set of ActiveSpaceIntegrals to the solver (e.g. an effective Hamiltonian from
    /// MR-DSRG)
    void set_active_space_integrals(std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
        as_ints_ = as_ints;
    }

  protected:
    /// Make the ActiveSpaceIntegrals object
    void make_active_space_ints();

    /// The list of active orbitals (in absolute ordering)
    std::vector<size_t> active_mo_;
    /// The list of doubly occupied orbitals (in absolute ordering)
    std::vector<size_t> core_mo_;

    /// A list of electronic states and their weights
    std::vector<std::pair<StateInfo, double>> states_weights_;
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// The molecular integrals object
    std::shared_ptr<ForteIntegrals> ints_;
    /// The molecular integrals for the active space
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;
};

std::shared_ptr<ActiveSpaceSolver>
make_active_space_solver(const std::string& type, StateInfo state,
                         std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info);

} // namespace forte

#endif // _active_space_solver_h_
