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
    SCFInfo(std::shared_ptr<psi::Wavefunction> psi4_wfn);

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

    /// @return the string representation of the SCFInfo object
    std::string to_string() const;

    /// Update the MO coefficients, update psi::Wavefunction, and re-transform integrals
    /// @param Ca the alpha MO coefficients
    /// @param Cb the beta MO coefficients
    /// @param re_transform re-transform integrals if true
    void update_orbitals(std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb,
                         bool transform_ints = true);

    /// Reorder the orbitals according to the new order
    /// @param new_order the new order of the orbitals. This is a vector of vectors, where the first
    /// index corresponds to the irrep and the second index corresponds to the orbital index (zero
    /// based) within the irrep.
    /// @param wfn the wavefunction to update
    void reorder_orbitals(const std::vector<std::vector<size_t>>& new_order);

    /// Rotate the MO coefficients, update psi::Wavefunction, and re-transform integrals
    /// @param Ua the alpha unitary transformation matrix
    /// @param Ub the beta unitary transformation matrix
    /// @param re_transform re-transform integrals if true
    void rotate_orbitals(std::shared_ptr<psi::Matrix> Ua, std::shared_ptr<psi::Matrix> Ub,
                         bool transform_ints = true);

  private:
    // A psi4 Wavefunction object. Stored only if the constructor based on Wavefunction is used.
    std::shared_ptr<psi::Wavefunction> psi4_wfn_;

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

    /// Alpha orbital coefficients (nso x nmo)
    std::shared_ptr<psi::Matrix> Ca_;

    /// Beta orbital coefficients (nso x nmo)
    std::shared_ptr<psi::Matrix> Cb_;

    /// List of observers
    std::vector<std::pair<std::string, std::weak_ptr<Observer>>> observers_;

    /// Common initialization function
    void initialize(const psi::Dimension& nmopi, const psi::Dimension& doccpi,
                    const psi::Dimension& soccpi, double reference_energy,
                    std::shared_ptr<psi::Vector> epsilon_a, std::shared_ptr<psi::Vector> epsilon_b,
                    std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb);

    void update_psi4_wavefunction();
};

} // namespace forte
