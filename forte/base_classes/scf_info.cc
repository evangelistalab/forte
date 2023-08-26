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

#include "psi4/libmints/wavefunction.h"
#include "scf_info.h"

namespace forte {

SCFInfo::SCFInfo(psi::SharedWavefunction wfn)
    : nmopi_(wfn->nmopi()), doccpi_(wfn->doccpi()), soccpi_(wfn->soccpi()), energy_(wfn->energy()),
      epsilon_a_(wfn->epsilon_a()), epsilon_b_(wfn->epsilon_b()) {}

SCFInfo::SCFInfo(const psi::Dimension& nmopi, const psi::Dimension& doccpi,
                 const psi::Dimension& soccpi, double reference_energy,
                 std::shared_ptr<psi::Vector> epsilon_a, std::shared_ptr<psi::Vector> epsilon_b)
    : nmopi_(nmopi), doccpi_(doccpi), soccpi_(soccpi), energy_(reference_energy),
      epsilon_a_(epsilon_a), epsilon_b_(epsilon_b) {}

psi::Dimension SCFInfo::nmopi() { return nmopi_; }

psi::Dimension SCFInfo::doccpi() { return doccpi_; }

psi::Dimension SCFInfo::soccpi() { return soccpi_; }

double SCFInfo::reference_energy() { return energy_; }

std::shared_ptr<psi::Vector> SCFInfo::epsilon_a() { return epsilon_a_; }

std::shared_ptr<psi::Vector> SCFInfo::epsilon_b() { return epsilon_b_; }

} // namespace forte
