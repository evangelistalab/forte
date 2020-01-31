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
#include <map>
#include <numeric>
#include <regex>
#include <vector>

#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/integral.h"
#include "psi4/libpsio/psio.hpp"

#include "pao_builder.h"

using namespace psi;

namespace forte {

	PAObuilder::PAObuilder(psi::SharedMatrix C, psi::Dimension noccpi,
                                     std::shared_ptr<BasisSet> basis)
    : C_(C), noccpi_(noccpi), basis_(basis) {
    startup();
}

void PAObuilder::startup() {
    // build D and S from inputs

}

SharedMatrix PAObuilder::build_A_virtual(int nbf_A, double pao_threshold) {
	// Build fragment virtual
	SharedMatrix C_virtual_A(new Matrix("C_vir A", nirrep_, nmopi_, nmopi_));

    return C_virtual_A;
}

SharedMatrix PAObuilder::build_B_virtual() {
	// Build environment virtual
	SharedMatrix C_virtual_B(new Matrix("C_vir B", nirrep_, nmopi_, nmopi_));
	throw PSIEXCEPTION("Environment PAO generations not available now!");
	return C_virtual_B;
}

} // namespace forte
