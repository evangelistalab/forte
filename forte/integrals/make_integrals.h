/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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

#ifndef _make_integrals_h_
#define _make_integrals_h_

namespace forte {
/**
 *  @brief Make a ForteIntegrals object with the help of psi4
 *
 *  This function reads the integral type from the option INT_TYPE,
 *  but if the variable int_type is provided, its value will override
 *  the value read from the options object.
 */
std::shared_ptr<ForteIntegrals> make_forte_integrals_from_psi4(
    std::shared_ptr<psi::Wavefunction> ref_wfn, std::shared_ptr<ForteOptions> options,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::string int_type = "");

/**
 *  @brief Make a ForteIntegrals object by passing integrals stored in vectors
 */
std::shared_ptr<ForteIntegrals>
make_custom_forte_integrals(std::shared_ptr<ForteOptions> options,
                            std::shared_ptr<MOSpaceInfo> mo_space_info, double scalar,
                            const std::vector<double>& oei_a, const std::vector<double>& oei_b,
                            const std::vector<double>& tei_aa, const std::vector<double>& tei_ab,
                            const std::vector<double>& tei_bb);

} // namespace forte

#endif // _make_integrals_h_
