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

#ifndef _scf_info_h_
#define _scf_info_h_

#include "psi4/libmints/dimension.h"
#include "psi4/libmints/wavefunction.h"

// class Vector;

namespace forte {

class SCFInfo {
  public:
    /// Constructor based on Psi4 Wavefunction
    SCFInfo(psi::SharedWavefunction wfn);

    /// return doccpi
    psi::Dimension doccpi();

    /// return soccpi
    psi::Dimension soccpi();

    /// return nsopi
    psi::Dimension nsopi();

    /// return nso
    int nso();

    /// return energy
    double reference_energy();

    /// alpha orbital energy
    std::shared_ptr<psi::Vector> epsilon_a();

    /// beta orbital energy
    std::shared_ptr<psi::Vector> epsilon_b();

  private:
    // Doubly occupied in RHF
    psi::Dimension doccpi_;

    // Singly occupied in RHF
    psi::Dimension soccpi_;

    /// The number of SO (AO for C matrices)
    psi::Dimension nsopi_;

    // SCF energy
    double energy_;

    /// alpha orbital energy
    std::shared_ptr<psi::Vector> epsilon_a_;

    /// beta orbital energy
    std::shared_ptr<psi::Vector> epsilon_b_;
};

} // namespace forte

#endif // _scf_info_h_
