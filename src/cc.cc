/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "psi4/libmints/molecule.h"

#include "cc.h"
#include "helpers.h"

namespace psi {
namespace forte {

CC::CC(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info),
      BTF_(new BlockedTensorFactory(options)), tensor_type_(CoreTensor) {
    startup();
}

/// Destructor
CC::~CC() {}

/// Compute the corr_level energy with fixed reference
double CC::compute_energy() { return 0.0; }

void CC::startup() {
    // frozen-core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // orbital spaces
    BlockedTensor::reset_mo_spaces();
    aocc_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bocc_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    avir_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvir_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    aocc_label_ = "o";
    avir_label_ = "v";
    bocc_label_ = "o";
    bvir_label_ = "v";

    BTF_->add_mo_space(aocc_label_, "ijklmn", aocc_mos_, AlphaSpin);
    BTF_->add_mo_space(bocc_label_, "IJKLMN", bocc_mos_, BetaSpin);
    BTF_->add_mo_space(avir_label_, "abcdef", avir_mos_, AlphaSpin);
    BTF_->add_mo_space(bvir_label_, "ABCDEF", bvir_mos_, BetaSpin);

    // define composite spaces
    BTF_->add_composite_mo_space("g", "pqrsto", {aocc_label_, avir_label_});
    BTF_->add_composite_mo_space("G", "PQRSTO", {bocc_label_, bvir_label_});

    // prepare integrals
    H_ = BTF_->build(tensor_type_, "H", spin_cases({"gg"}));
    V_ = BTF_->build(tensor_type_, "V", spin_cases({"gggg"}));
    //    build_ints();

    // build Fock matrix
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
    //    build_fock(H_, V_);
}

}
}
