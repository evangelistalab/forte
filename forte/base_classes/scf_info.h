/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libmints/dimension.h"

#include "forte/helpers/observer.h"

namespace psi {
class Vector;
class Matrix;
class Wavefunction;
} // namespace psi

namespace forte {
class Observer;

class SCFInfo : public Subject {
  public:
    /// Constructor
    SCFInfo(const psi::Dimension& nmopi, const psi::Dimension& doccpi, const psi::Dimension& soccpi,
            double reference_energy, std::shared_ptr<psi::Vector> epsilon_a,
            std::shared_ptr<psi::Vector> epsilon_b, std::shared_ptr<psi::Matrix> Ca,
            std::shared_ptr<psi::Matrix> Cb);

    /// Constructor based on Psi4 Wavefunction
    SCFInfo(std::shared_ptr<psi::Wavefunction> wfn);

    /// return the number of orbitals per irrep
    psi::Dimension nmopi();

    /// return the number of doubly occupied orbitals per irrep
    psi::Dimension doccpi();

    /// return the number of singly occupied orbitals per irrep
    psi::Dimension soccpi();

    /// return the reference energy
    double reference_energy();

    /// alpha orbital energy
    std::shared_ptr<psi::Vector> epsilon_a();

    /// beta orbital energy
    std::shared_ptr<psi::Vector> epsilon_b();

    /// @return the alpha orbital coefficients
    std::shared_ptr<psi::Matrix> _Ca();

    /// @return the beta orbital coefficients
    std::shared_ptr<psi::Matrix> _Cb();

    /// @return the alpha orbital coefficients (const version)
    std::shared_ptr<const psi::Matrix> Ca() const;

    /// @return the beta orbital coefficients (const version)
    std::shared_ptr<const psi::Matrix> Cb() const;

    void
    reorder_orbitals(const std::vector<std::vector<size_t>>& new_order,
                     std::shared_ptr<psi::Wavefunction> wfn = std::shared_ptr<psi::Wavefunction>());

  private:
    // Orbitals per irrep
    psi::Dimension nmopi_;

    // Doubly occupied orbitals per irrep
    psi::Dimension doccpi_;

    // Singly occupied orbitals per irrep
    psi::Dimension soccpi_;

    // SCF energy
    double energy_;

    /// alpha orbital energy
    std::shared_ptr<psi::Vector> epsilon_a_;

    /// beta orbital energy
    std::shared_ptr<psi::Vector> epsilon_b_;

    // Alpha orbital coefficients (nso x nmo)
    std::shared_ptr<psi::Matrix> Ca_;

    // Beta orbital coefficients (nso x nmo)
    std::shared_ptr<psi::Matrix> Cb_;

    // List of observers
    std::vector<std::pair<std::string, std::weak_ptr<Observer>>> observers_;
};

} // namespace forte
