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

#ifndef _state_info_h_
#define _state_info_h_

#include <string>
#include <vector>

namespace psi {
class Wavefunction;
}

namespace forte {

class StateInfo {
  public:
    /// Constructor
    StateInfo(int na, int nb, int multiplicity, int twice_ms, int irrep, const std::string& irrep_label = "");

    StateInfo() = default;

    /// multiplicity labels
    static const std::vector<std::string> multiplicity_labels;

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
    /// return the multiplicity symbol
    const std::string& multiplicity_label() const;
    /// return the irrep symbol
    const std::string& irrep_label() const;
    /// Comparison operator for StateInfo objects
    bool operator<(const StateInfo& rhs) const;
    /// Comparison operator for StateInfo objects
    bool operator!=(const StateInfo& rhs) const;
    /// Comparison operator for StateInfo objects
    bool operator==(const StateInfo& rhs) const;

    /// string representation of this object
    std::string str() const;

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
    // Irrep label
    std::string irrep_label_;
};

/**
 * @brief make_state_info_from_psi_wfn Make a StateInfo object by reading variables set in a psi4
 *        Wavefunction object
 * @param wfn the psi wave function object
 * @return a StateInfo object
 */
StateInfo make_state_info_from_psi_wfn(std::shared_ptr<psi::Wavefunction> wfn);

} // namespace forte

#endif // _state_info_h_
