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

#ifndef _sci_h_
#define _sci_h_

#include "base_classes/active_space_method.h"
namespace forte {
class SelectedCI : public ActiveSpaceMethod {
  public:
    SelectedCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Default constructor
    SelectedCI() = default;

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~SelectedCI() = default;

    // ==> Class Interface <==

    /// Compute the energy and return it
    double compute_energy() override;

    /// Returns the reference
    virtual Reference get_reference(int root = 0) = 0;
};
} // namespace forte
#endif // _sci_h_
