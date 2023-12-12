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

#pragma once

#include <vector>
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

namespace forte {

class ActiveSpaceSolver;
class ActiveSpaceIntegrals;

/**
 * @class ContractedCISolver
 * @brief Contracted Configuration Interaction Solver for multi-state DSRG
 */
class ContractedCISolver {
  public:
    /**
     * @brief ContractedCISolver constructor
     *
     * @param as_solver active space solver
     * @param as_ints active space integrals
     */
    ContractedCISolver(std::shared_ptr<ActiveSpaceSolver> as_solver,
                       std::shared_ptr<ActiveSpaceIntegrals> as_ints, int max_rdm_level,
                       int max_body);

    /// TODO: the as_ints should handle 3-body integrals, as_ints may not be Hermitian

    /// build the effective Hamiltonian <A|H|B> and diagonalize it
    void compute_Heff();

    /// compute the new densities and return a new RDMs??

    /// get the eigen values
    std::vector<std::vector<double>> get_energies() { return evals_; }

    /// get the eigen vectors
    std::vector<psi::Matrix> get_evecs() { return evecs_; }

  private:
    /// active space solver
    std::shared_ptr<ActiveSpaceSolver> as_solver_;

    /// active space integrals
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    /// max many-body level available for as_ints
    int max_body_;

    /// max rdm level
    int max_rdm_level_;

    /// eigen values
    std::vector<std::vector<double>> evals_;

    /// eigen vectors
    std::vector<psi::Matrix> evecs_;
};
} // namespace forte
