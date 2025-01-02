/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER,
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

#pragma once

namespace forte {
/// @brief Make a ForteIntegrals object with the help of psi4
///     This function reads the integral type from the option INT_TYPE,
///     but if the variable int_type is provided, its value will override
///     the value read from the options object.
/// @param ref_wfn A psi4 Wavefunction object
/// @param options A ForteOptions object
/// @param scf_info A SCFInfo object
/// @param mo_space_info A MOSpaceInfo object
/// @param int_type The type of integrals to be used
/// @return A shared pointer to a ForteIntegrals object
std::shared_ptr<ForteIntegrals> make_forte_integrals_from_psi4(
    std::shared_ptr<psi::Wavefunction> ref_wfn, std::shared_ptr<ForteOptions> options,
    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
    std::string int_type = "");

/// @brief Make a ForteIntegrals object by passing integrals stored in vectors
/// @param options A ForteOptions object
/// @param scf_info A SCFInfo object
/// @param mo_space_info A MOSpaceInfo object
/// @param scalar The scalar term in the Hamiltonian
/// @param oei_a A vector of alpha one-electron integrals
/// @param oei_b A vector of beta one-electron integrals
/// @param tei_aa A vector of alpha-alpha antisymmetrized two-electron integrals (<pq||rs>)
/// @param tei_ab A vector of alpha-beta antisymmetrized two-electron integrals (<pq|rs>)
/// @param tei_bb A vector of beta-beta antisymmetrized two-electron integrals (<pq||rs>)
std::shared_ptr<ForteIntegrals> make_custom_forte_integrals(
    std::shared_ptr<ForteOptions> options, std::shared_ptr<SCFInfo> scf_info,

    std::shared_ptr<MOSpaceInfo> mo_space_info, double scalar, const std::vector<double>& oei_a,
    const std::vector<double>& oei_b, const std::vector<double>& tei_aa,
    const std::vector<double>& tei_ab, const std::vector<double>& tei_bb);

} // namespace forte
