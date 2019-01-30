/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/molecule.h"

#include "state_info.h"

namespace forte {

StateInfo::StateInfo(int na, int nb, int multiplicity, int twice_ms, int irrep)
    : na_(na), nb_(nb), multiplicity_(multiplicity), twice_ms_(twice_ms), irrep_(irrep) {}

int StateInfo::na() const { return na_; }

int StateInfo::nb() const { return nb_; }

int StateInfo::multiplicity() const { return multiplicity_; }

int StateInfo::twice_ms() const { return twice_ms_; }

int StateInfo::irrep() const { return irrep_; }

bool StateInfo::operator<(const StateInfo& rhs) const {
    return std::tie(na_, nb_, multiplicity_, twice_ms_, irrep_) <
           std::tie(rhs.na_, rhs.nb_, rhs.multiplicity_, rhs.twice_ms_, rhs.irrep_);
}

StateInfo make_state_info_from_psi_wfn(std::shared_ptr<psi::Wavefunction> wfn) {
    int charge = psi::Process::environment.molecule()->molecular_charge();
    if (wfn->options()["CHARGE"].has_changed()) {
        charge = wfn->options().get_int("CHARGE");
    }

    int nel = 0;
    int natom = psi::Process::environment.molecule()->natom();
    for (int i = 0; i < natom; i++) {
        nel += static_cast<int>(psi::Process::environment.molecule()->Z(i));
    }
    // If the charge has changed, recompute the number of electrons
    // Or if you cannot find the number of electrons
    nel -= charge;

    size_t multiplicity = psi::Process::environment.molecule()->multiplicity();
    if (wfn->options()["MULTIPLICITY"].has_changed()) {
        multiplicity = wfn->options().get_int("MULTIPLICITY");
    }

    // If the user did not specify ms determine the value from the input or
    // take the lowest value consistent with the value of "MULTIPLICITY"
    // For example:
    //    singlet: multiplicity = 1 -> twice_ms = 0 (ms = 0)
    //    doublet: multiplicity = 2 -> twice_ms = 1 (ms = 1/2)
    //    triplet: multiplicity = 3 -> twice_ms = 0 (ms = 0)
    size_t twice_ms = (multiplicity + 1) % 2;
    if (wfn->options()["MS"].has_changed()) {
        twice_ms = std::round(2.0 * wfn->options().get_double("MS"));
    }

    if (((nel - twice_ms) % 2) != 0)
        throw psi::PSIEXCEPTION("\n\n  make_state_info_from_psi_wfn: Wrong value of M_s.\n\n");

    size_t na = (nel + twice_ms) / 2;
    size_t nb = nel - na;

    size_t irrep = 0;
    if (wfn->options()["ROOT_SYM"].has_changed()) {
        irrep = wfn->options().get_int("ROOT_SYM");
    }
    return StateInfo(na, nb, multiplicity, twice_ms, irrep);
}

} // namespace forte
