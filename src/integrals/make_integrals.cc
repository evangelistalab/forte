/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/forte_options.h"
#include "helpers/timer.h"

#include "integrals/integrals.h"
#include "integrals/cholesky_integrals.h"
#include "integrals/custom_integrals.h"
#include "integrals/df_integrals.h"
#include "integrals/diskdf_integrals.h"
#include "integrals/conventional_integrals.h"
#include "integrals/own_integrals.h"

#include "make_integrals.h"

namespace forte {

std::shared_ptr<ForteIntegrals> make_forte_integrals(std::shared_ptr<psi::Wavefunction> ref_wfn,
                                                     std::shared_ptr<ForteOptions> options,
                                                     std::shared_ptr<MOSpaceInfo> mo_space_info) {
    timer int_timer("Integrals");
    std::shared_ptr<ForteIntegrals> ints;
    if (options->get_str("INT_TYPE") == "CHOLESKY") {
        ints = std::make_shared<CholeskyIntegrals>(options, ref_wfn, mo_space_info,
                                                   IntegralSpinRestriction::Restricted);
    } else if (options->get_str("INT_TYPE") == "DF") {
        ints = std::make_shared<DFIntegrals>(options, ref_wfn, mo_space_info,
                                             IntegralSpinRestriction::Restricted);
    } else if (options->get_str("INT_TYPE") == "DISKDF") {
        ints = std::make_shared<DISKDFIntegrals>(options, ref_wfn, mo_space_info,
                                                 IntegralSpinRestriction::Restricted);
    } else if (options->get_str("INT_TYPE") == "CONVENTIONAL") {
        ints = std::make_shared<ConventionalIntegrals>(options, ref_wfn, mo_space_info,
                                                       IntegralSpinRestriction::Restricted);
    } else if (options->get_str("INT_TYPE") == "DISTDF") {
#ifdef HAVE_GA
        ints = std::make_shared<DistDFIntegrals>(options, ref_wfn, mo_space_info,
                                                 IntegralSpinRestriction::Restricted);
#endif
    } else if (options->get_str("INT_TYPE") == "CUSTOM") {
        ints = std::make_shared<CustomIntegrals>(options, ref_wfn, mo_space_info,
                                                 IntegralSpinRestriction::Restricted);
    } else if (options->get_str("INT_TYPE") == "OWNINTEGRALS") {
        ints = std::make_shared<OwnIntegrals>(options, ref_wfn, mo_space_info,
                                              IntegralSpinRestriction::Restricted);
    } else {
        psi::outfile->Printf("\n Please check your int_type. Choices are CHOLESKY, DF, DISKDF , "
                             "DISTRIBUTEDDF Effective, CONVENTIONAL or OwnIntegrals");
        throw std::runtime_error("INT_TYPE is not correct.  Check options");
    }

    if (options->get_bool("PRINT_INTS")) {
        ints->print_ints();
    }

    return ints;
}

} // namespace forte
