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

#include "mrdsrg-so/mrdsrg_so.h"
#include "mrdsrg-so/so-mrdsrg.h"
#include "mrdsrg-spin-adapted/dsrg_mrpt.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "mrdsrg-spin-integrated/mrdsrg.h"
#include "mrdsrg-spin-integrated/three_dsrg_mrpt2.h"
#include "mrdsrg-spin-adapted/sa_mrdsrg.h"
#include "mrdsrg-spin-adapted/sa_mrpt2.h"
#include "mrdsrg-spin-adapted/sa_mrpt3.h"

#include "run_dsrg.h"

namespace forte {

std::unique_ptr<MASTER_DSRG> make_dsrg_method(const std::string& method, RDMs rdms,
                                              std::shared_ptr<SCFInfo> scf_info,
                                              std::shared_ptr<ForteOptions> options,
                                              std::shared_ptr<ForteIntegrals> ints,
                                              std::shared_ptr<MOSpaceInfo> mo_space_info) {
    std::unique_ptr<MASTER_DSRG> dsrg_method;
    if (method == "DSRG-MRPT2") {
        dsrg_method = std::make_unique<DSRG_MRPT2>(rdms, scf_info, options, ints, mo_space_info);
    } else if (method == "DSRG-MRPT3") {
        dsrg_method = std::make_unique<DSRG_MRPT3>(rdms, scf_info, options, ints, mo_space_info);
    } else if (method == "THREE-DSRG-MRPT2") {
        dsrg_method =
            std::make_unique<THREE_DSRG_MRPT2>(rdms, scf_info, options, ints, mo_space_info);
    } else if (method == "MRDSRG") {
        dsrg_method = std::make_unique<MRDSRG>(rdms, scf_info, options, ints, mo_space_info);
    } else {
        throw psi::PSIEXCEPTION("Method name " + method + " not recognized.");
    }
    return dsrg_method;
}

std::unique_ptr<SADSRG> make_sadsrg_method(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                           std::shared_ptr<ForteOptions> options,
                                           std::shared_ptr<ForteIntegrals> ints,
                                           std::shared_ptr<MOSpaceInfo> mo_space_info) {
    std::unique_ptr<SADSRG> dsrg_method;
    auto corr_level = options->get_str("CORR_LEVEL");
    if (corr_level == "PT2") {
        dsrg_method = std::make_unique<SA_MRPT2>(rdms, scf_info, options, ints, mo_space_info);
    } else if (corr_level == "PT3") {
        dsrg_method = std::make_unique<SA_MRPT3>(rdms, scf_info, options, ints, mo_space_info);
    } else {
        dsrg_method = std::make_unique<SA_MRDSRG>(rdms, scf_info, options, ints, mo_space_info);
    }
    return dsrg_method;
}

std::unique_ptr<MRDSRG_SO> make_dsrg_so_y(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                          std::shared_ptr<ForteOptions> options,
                                          std::shared_ptr<ForteIntegrals> ints,
                                          std::shared_ptr<MOSpaceInfo> mo_space_info) {
    return std::make_unique<MRDSRG_SO>(rdms, scf_info, options, ints, mo_space_info);
}

std::unique_ptr<SOMRDSRG> make_dsrg_so_f(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                         std::shared_ptr<ForteOptions> options,
                                         std::shared_ptr<ForteIntegrals> ints,
                                         std::shared_ptr<MOSpaceInfo> mo_space_info) {
    return std::make_unique<SOMRDSRG>(rdms, scf_info, options, ints, mo_space_info);
}

std::unique_ptr<DSRG_MRPT> make_dsrg_spin_adapted(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                                  std::shared_ptr<ForteOptions> options,
                                                  std::shared_ptr<ForteIntegrals> ints,
                                                  std::shared_ptr<MOSpaceInfo> mo_space_info) {
    return std::make_unique<DSRG_MRPT>(rdms, scf_info, options, ints, mo_space_info);
}

} // namespace forte
