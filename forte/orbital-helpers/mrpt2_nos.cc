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
#include <memory>

#include "ambit/blocked_tensor.h"

#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/helpers.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "mrdsrg-spin-adapted/sa_mrpt2.h"
#include "mrpt2_nos.h"

using namespace ambit;
using namespace psi;

namespace forte {

MRPT2_NOS::MRPT2_NOS(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalTransform(ints, mo_space_info), options_(options) {
    mrpt2_ = std::make_shared<SA_MRPT2>(rdms, scf_info, options, ints, mo_space_info);
}

void MRPT2_NOS::compute_transformation() {
    // compute unrelaxed 1-RDMs for core and virtual blocks
    mrpt2_->build_1rdm_unrelaxed(D1c_, D1v_);

    psi::Process::environment.arrays["MRPT2 1RDM CC"] = D1c_;
    psi::Process::environment.arrays["MRPT2 1RDM VV"] = D1v_;

    // diagonalize unrelaxed 1-RDM
    auto core_mospi = D1c_->rowspi();
    auto virt_mospi = D1v_->rowspi();

    psi::Vector D1c_evals("D1c_evals", core_mospi);
    psi::Matrix D1c_evecs("D1c_evecs", core_mospi, core_mospi);
    D1c_->diagonalize(D1c_evecs, D1c_evals, descending);

    psi::Vector D1v_evals("D1v_evals", virt_mospi);
    psi::Matrix D1v_evecs("D1v_evecs", virt_mospi, virt_mospi);
    D1v_->diagonalize(D1v_evecs, D1v_evals, descending);

    // print natural orbitals
    if (options_->get_bool("NAT_ORBS_PRINT")) {
        D1c_evals.print();
        D1v_evals.print();
    }

    // suggest active space
    if (options_->get_bool("NAT_ACT")) {
        suggest_active_space();
    }

    // build transformation matrix
    auto nmopi = mo_space_info_->dimension("ALL");
    auto ncmopi = mo_space_info_->dimension("CORRELATED");
    auto frzcpi = mo_space_info_->dimension("FROZEN_DOCC");

    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    Ua_->identity();

    Slice slice_core(frzcpi, mo_space_info_->dimension("INACTIVE_DOCC"));
    Ua_->set_block(slice_core, D1c_evecs);

    Slice slice_virt(frzcpi + mo_space_info_->dimension("GENERALIZED HOLE"), frzcpi + ncmopi);
    Ua_->set_block(slice_virt, D1v_evecs);

    Ub_ = Ua_->clone();
}

void MRPT2_NOS::suggest_active_space() {
    // print original active space

    // suggest new active space
}

} // namespace forte