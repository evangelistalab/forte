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

#pragma once

#include <memory>

#include "base_classes/forte_options.h"
#include "mrdsrg-helper/dsrg_transformed.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "mrdsrg-spin-integrated/master_mrdsrg.h"
#include "mrdsrg-spin-integrated/three_dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/mrdsrg.h"
#include "mrdsrg-so/mrdsrg_so.h"
#include "mrdsrg-so/so-mrdsrg.h"
#include "mrdsrg-spin-adapted/dsrg_mrpt.h"
#include "mrdsrg-spin-adapted/sadsrg.h"

namespace forte {

/// Reference relaxation, relaxed dipoles, transition dipoles,
/// general sequence of running dsrg should be implemented in this class

// class RUN_DSRG {
//  public:
//    /**
//     * Constructor
//     * @param ref_wfn The reference wavefunction object
//     * @param options The main options object
//     * @param ints A pointer to an allocated integral object
//     * @param mo_space_info The MOSpaceInfo object
//     */
//    RUN_DSRG(RDMs rdms, psi::SharedWavefunction ref_wfn, std::shared_ptr<ForteOptions> options,
//             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

//    /// Compute DSRG energy
//    double compute_dsrg_energy();

//    /// Compute DSRG density
//    void compute_dsrg_density();

// protected:
//    /// RDMs type (FCI for FCI_MO)
//    std::string ref_type_;

//};

std::unique_ptr<MASTER_DSRG> make_dsrg_method(std::shared_ptr<RDMs> rdms,
                                              std::shared_ptr<SCFInfo> scf_info,
                                              std::shared_ptr<ForteOptions> options,
                                              std::shared_ptr<ForteIntegrals> ints,
                                              std::shared_ptr<MOSpaceInfo> mo_space_info);

std::unique_ptr<MRDSRG_SO> make_dsrg_so_y(std::shared_ptr<RDMs> rdms,
                                          std::shared_ptr<SCFInfo> scf_info,
                                          std::shared_ptr<ForteOptions> options,
                                          std::shared_ptr<ForteIntegrals> ints,
                                          std::shared_ptr<MOSpaceInfo> mo_space_info);

std::unique_ptr<SOMRDSRG> make_dsrg_so_f(std::shared_ptr<RDMs> rdms,
                                         std::shared_ptr<SCFInfo> scf_info,
                                         std::shared_ptr<ForteOptions> options,
                                         std::shared_ptr<ForteIntegrals> ints,
                                         std::shared_ptr<MOSpaceInfo> mo_space_info);

std::unique_ptr<DSRG_MRPT> make_dsrg_spin_adapted(std::shared_ptr<RDMs> rdms,
                                                  std::shared_ptr<SCFInfo> scf_info,
                                                  std::shared_ptr<ForteOptions> options,
                                                  std::shared_ptr<ForteIntegrals> ints,
                                                  std::shared_ptr<MOSpaceInfo> mo_space_info);

std::unique_ptr<SADSRG> make_sadsrg_method(std::shared_ptr<RDMs> rdms,
                                           std::shared_ptr<SCFInfo> scf_info,
                                           std::shared_ptr<ForteOptions> options,
                                           std::shared_ptr<ForteIntegrals> ints,
                                           std::shared_ptr<MOSpaceInfo> mo_space_info);

} // namespace forte
