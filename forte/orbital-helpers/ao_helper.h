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

#ifndef _atomic_orbital_h_
#define _atomic_orbital_h_

#include "psi4/psi4-dec.h"
#include "psi4/lib3index/denominator.h"

namespace forte {

class AtomicOrbitalHelper {
  protected:
    psi::SharedMatrix AO_Screen_;
    psi::SharedMatrix TransAO_Screen_;

    psi::SharedMatrix CMO_;
    psi::SharedVector eps_rdocc_;
    psi::SharedVector eps_virtual_;

    psi::SharedMatrix POcc_;
    psi::SharedMatrix PVir_;
    void Compute_Psuedo_Density();

    // LaplaceDenominator Laplace_;
    psi::SharedMatrix Occupied_Laplace_;
    psi::SharedMatrix Virtual_Laplace_;
    double laplace_tolerance_ = 1e-10;

    int weights_;
    int nbf_;
    int nrdocc_;
    int nvir_;
    /// How many orbitals does it take to go from occupied to virtual (ie should
    /// be active)
    int shift_;

  public:
    psi::SharedMatrix AO_Screen() { return AO_Screen_; }
    psi::SharedMatrix TransAO_Screen() { return TransAO_Screen_; }
    psi::SharedMatrix Occupied_Laplace() { return Occupied_Laplace_; }
    psi::SharedMatrix Virtual_Laplace() { return Virtual_Laplace_; }
    psi::SharedMatrix POcc() { return POcc_; }
    psi::SharedMatrix PVir() { return PVir_; }
    int Weights() { return weights_; }

    AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ, psi::SharedVector eps_vir,
                        double laplace_tolerance);
    AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ, psi::SharedVector eps_vir,
                        double laplace_tolerance, int shift);
    /// Compute (mu nu | mu nu)^{(1/2)}
    void Compute_AO_Screen(std::shared_ptr<psi::BasisSet>& primary);
    void Estimate_TransAO_Screen(std::shared_ptr<psi::BasisSet>& primary,
                                 std::shared_ptr<psi::BasisSet>& auxiliary);

    ~AtomicOrbitalHelper();
};
} // namespace forte

#endif
