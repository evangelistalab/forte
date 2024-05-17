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

#include "psi4/libmints/wavefunction.h"
#include "helpers/helpers.h"

#include "orbitals.h"

namespace forte {

Orbitals::Orbitals(const std::shared_ptr<psi::Matrix>& Ca, const std::shared_ptr<psi::Matrix>& Cb) {
    Ca_ = Ca->clone();
    Cb_ = Cb->clone();
}

const std::shared_ptr<psi::Matrix> Orbitals::Ca() const { return Ca_; }

const std::shared_ptr<psi::Matrix> Orbitals::Cb() const { return Cb_; }

void Orbitals::set(const std::shared_ptr<psi::Matrix>& Ca, const std::shared_ptr<psi::Matrix>& Cb) {
    if (not(elementwise_compatible_matrices(Ca_, Ca) and
            elementwise_compatible_matrices(Cb_, Cb))) {
        throw std::runtime_error(
            "Orbital::set: orbital coefficient matrices have different dimensions!");
    }
    Ca_ = Ca->clone();
    Cb_ = Cb->clone();
}

void Orbitals::rotate(std::shared_ptr<psi::Matrix> Ua, std::shared_ptr<psi::Matrix> Ub) {
    // 1. Rotate the orbital coefficients and store them in the ForteIntegral object
    auto Ca_rotated = psi::linalg::doublet(Ca_, Ua);
    auto Cb_rotated = psi::linalg::doublet(Cb_, Ub);
    Ca_->copy(Ca_rotated);
    Cb_->copy(Cb_rotated);
}

void Orbitals::copy(const Orbitals& other) {
    Ca_->copy(*other.Ca_);
    Cb_->copy(*other.Cb_);
}

bool Orbitals::are_spin_restricted(double threshold) const {
    return matrix_distance(Ca_, Cb_) < threshold;
}

std::unique_ptr<Orbitals>
make_orbitals_from_psi(const std::shared_ptr<const psi::Wavefunction>& wfn, bool restricted) {
    return std::make_unique<Orbitals>(wfn->Ca(), restricted ? wfn->Ca() : wfn->Cb());
}

} // namespace forte