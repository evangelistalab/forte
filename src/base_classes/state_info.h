/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _state_info_h_
#define _state_info_h_

namespace forte {

class StateInfo {
  public:
    /// Constructor
    StateInfo(int na, int nb, int multiplicity, int twice_ms, int irrep);
    /// Constructor based on Psi4 Wavefunction
    StateInfo(psi::SharedWavefunction wfn);

    /// return the number of alpha electrons
    int na() const;
    /// return the number of beta electrons
    int nb() const;
    /// return the multiplicity
    int multiplicity() const;
    /// return twice Ms
    int twice_ms() const;
    /// return the irrep
    int irrep() const;

    /// Return the name of the molecule (needed for DMRG)
    std::string name() const;

  private:
    // number of alpha electrons (including core, excludes ecp)
    int na_;
    // numebr of beta electrons (including core, excludes ecp)
    int nb_;
    // 2S + 1
    int multiplicity_;
    // 2Ms
    int twice_ms_;
    // Irrep
    int irrep_;
};

} // namespace forte

#endif // _state_info_h_
