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
#include "psi4/libpsi4util/PsiOutStream.h"

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
double CC::compute_energy() {
    initial_mp2_t();
    double Ecc = cc_energy();
    outfile->Printf("\n  CCSD energy = %.12f", Ecc);
    return 0.0;
}

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
    bocc_label_ = "O";
    bvir_label_ = "V";

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
    build_ints();

    // build Fock matrix
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
    if (eri_df_) {
        build_fock_df(H_, V_);
    } else {
        build_fock(H_, V_);
    }
}

void CC::build_ints() {
    // prepare one-electron integrals
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0], i[1]);
        else
            value = ints_->oei_b(i[0], i[1]);
    });

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        fill_three_index_ints(B_);
    } else {
        V_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                    value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
                if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                    value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
                if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                    value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
            });
    }
}

void CC::fill_three_index_ints(ambit::BlockedTensor T) {
    const auto& block_labels = T.block_labels();
    for (const std::string& string_block : block_labels) {
        auto mo_to_index = BTF_->get_mo_to_index();
        std::vector<size_t> first_index = mo_to_index[string_block.substr(0, 1)];
        std::vector<size_t> second_index = mo_to_index[string_block.substr(1, 1)];
        std::vector<size_t> third_index = mo_to_index[string_block.substr(2, 1)];
        ambit::Tensor block = ints_->three_integral_block(first_index, second_index, third_index);
        T.block(string_block).copy(block);
    }
}

void CC::build_fock(BlockedTensor& H, BlockedTensor& V) {
    // the core-core density is an identity matrix
    BlockedTensor D1c = BTF_->build(tensor_type_, "Gamma1 core", spin_cases({"oo"}));
    for (size_t m = 0, nc = mo_space_info_->size("RESTRICTED_DOCC"); m < nc; ++m) {
        D1c.block("oo").data()[m * nc + m] = 1.0;
        D1c.block("OO").data()[m * nc + m] = 1.0;
    }

    // build Fock matrix
    F_["pq"] = H["pq"];
    F_["pq"] += V["pnqm"] * D1c["mn"];
    F_["pq"] += V["pNqM"] * D1c["MN"];
    F_["PQ"] = H["PQ"];
    F_["PQ"] += V["nPmQ"] * D1c["mn"];
    F_["PQ"] += V["PNQM"] * D1c["MN"];

    // obtain diagonal elements of Fock matrix
    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    Fa_.resize(ncmo_);
    Fb_.resize(ncmo_);
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            Fb_[i[0]] = value;
        }
    });
}

void CC::build_fock_df(BlockedTensor& H, BlockedTensor& B) {
    // the core-core density is an identity matrix
    BlockedTensor D1c = BTF_->build(tensor_type_, "Gamma1 core", spin_cases({"oo"}));
    for (size_t m = 0, nc = mo_space_info_->size("RESTRICTED_DOCC"); m < nc; ++m) {
        D1c.block("oo").data()[m * nc + m] = 1.0;
        D1c.block("OO").data()[m * nc + m] = 1.0;
    }

    // build Fock matrix
    F_["pq"] = H["pq"];
    F_["PQ"] = H["PQ"];

    BlockedTensor temp = BTF_->build(tensor_type_, "B temp", {"L"});
    temp["g"] = B["gmn"] * D1c["mn"];
    F_["pq"] += temp["g"] * B["gpq"];
    F_["PQ"] += temp["g"] * B["gPQ"];

    temp["g"] = B["gMN"] * D1c["MN"];
    F_["pq"] += temp["g"] * B["gpq"];
    F_["PQ"] += temp["g"] * B["gPQ"];

    // exchange
    F_["pq"] -= B["gpn"] * B["gmq"] * D1c["mn"];

    F_["PQ"] -= B["gPN"] * B["gMQ"] * D1c["MN"];

    // obtain diagonal elements of Fock matrix
    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo_);
    Fb_ = std::vector<double>(ncmo_);
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            Fb_[i[0]] = value;
        }
    });
}

void CC::initial_mp2_t() {
    T1_ = BTF_->build(tensor_type_, "T1", spin_cases({"ov"}));
    T2_ = BTF_->build(tensor_type_, "T1", spin_cases({"oovv"}));

    T2_["ijab"] = V_["ijab"];
    T2_["iJaB"] = V_["iJaB"];
    T2_["IJAB"] = V_["IJAB"];

    T2_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                value /= Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]];
            if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                value /= Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]];
            if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                value /= Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]];
        });
}

double CC::cc_energy() {
    double Ecc = 0.0;

    Ecc += F_["ia"] * T1_["ia"];
    Ecc += F_["IA"] * T1_["IA"];

    Ecc += 0.5 * V_["ijab"] * T1_["ia"] * T1_["jb"];
    Ecc += V_["iJaB"] * T1_["ia"] * T1_["JB"];
    Ecc += 0.5 * V_["IJAB"] * T1_["IA"] * T1_["JB"];

    Ecc += 0.25 * V_["ijab"] * T2_["ijab"];
    Ecc += V_["iJaB"] * T2_["iJaB"];
    Ecc += 0.25 * V_["IJAB"] * T2_["IJAB"];

    return Ecc;
}
}
}
