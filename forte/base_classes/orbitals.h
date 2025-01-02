/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/libmints/matrix.h"

namespace psi {
class Wavefunction;
} // namespace psi

namespace forte {

/// @brief The Orbitals class stores information about the molecular orbitals

class Orbitals {
  public:
    // ==> Class Constructor <==
    Orbitals(const std::shared_ptr<const psi::Wavefunction>& wfn, bool restricted);

    // ==> Class Methods <==
    /// @return The alpha orbital coefficient matrix
    const std::shared_ptr<const psi::Matrix> Ca() const { return Ca_; }
    /// @return The beta orbital coefficient matrix
    const std::shared_ptr<const psi::Matrix> Cb() const { return Cb_; }

  private:
    // ==> Class Data <==
    /// @brief The alpha orbitals coefficient matrix
    std::shared_ptr<psi::Matrix> Ca_;
    /// @brief The beta orbitals coefficient matrix
    std::shared_ptr<psi::Matrix> Cb_;
};

std::unique_ptr<Orbitals> make_orbitals(const std::shared_ptr<const psi::Wavefunction>& wfn,
                                        bool restricted);

} // namespace forte
