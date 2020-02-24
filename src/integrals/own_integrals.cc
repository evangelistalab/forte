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

#include <cmath>
#include <numeric>

#include "psi4/libmints/basisset.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"

#include "helpers/blockedtensorfactory.h"

#include "own_integrals.h"

using namespace psi;

namespace forte {

OwnIntegrals::OwnIntegrals(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> ref_wfn,
                           std::shared_ptr<MOSpaceInfo> mo_space_info,
                           IntegralSpinRestriction restricted)
    : ForteIntegrals(options, ref_wfn, mo_space_info, restricted) {
    integral_type_ = Own;
    // If code calls constructor print things
    // But if someone calls retransform integrals do not print it
    print_info();

    outfile->Printf("\n Avoiding Generation of Integrals");
    //    freeze_core_orbitals();
}
} // namespace forte
