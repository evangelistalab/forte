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

#include "fci_vector.h"
#include "fci_solver.h"

namespace forte {

std::vector<std::shared_ptr<RDMs>>
FCISolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>& /*root_list*/,
                           std::shared_ptr<ActiveSpaceMethod> /*method2*/, int max_rdm_level,
                           RDMsType /*rdm_type*/) {

    throw std::runtime_error("FCISolver::transition_rdms is not implemented!");

    if (max_rdm_level > 3 || max_rdm_level < 1) {
        throw std::runtime_error("Invalid max_rdm_level, required 1 <= max_rdm_level <= 3.");
    }

    std::vector<std::shared_ptr<RDMs>> refs;
    return refs;
}

} // namespace forte
