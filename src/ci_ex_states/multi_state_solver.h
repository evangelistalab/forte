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

#ifndef _multi_state_solver_h_
#define _multi_state_solver_h_

#include "ci_ex_states/excited_state_solver.h"
namespace forte {
class MultiStateSolver : public ExcitedStateSolver
{
  public:
    MultiStateSolver(StateInfo state, size_t nroot,
                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                     std::shared_ptr<SelectedCIMethod> sci);

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~MultiStateSolver() = default;

    // ==> Class Interface <==

    /// Compute the energy and return it
    double compute_energy() override;

    /// Returns the reference
    std::vector<Reference> reference(std::vector<std::pair<size_t, size_t>>& roots) override;

    /// Set options from an option object
    /// @param options the options passed in
    void set_options(std::shared_ptr<ForteOptions> options) override;
};
}
#endif // _multi_state_solver_h_
