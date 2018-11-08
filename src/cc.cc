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
#include "psi4/libpsi4util/process.h"

#include "cc.h"
#include "helpers.h"
#include "helpers/printing.h"

namespace psi {
namespace forte {

CC::CC(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info),
      BTF_(new BlockedTensorFactory(options)), tensor_type_(CoreTensor) {
    reference_wavefunction_ = ref_wfn;
    startup();
}

/// Destructor
CC::~CC() {}

/// Compute the corr_level energy with fixed reference
double CC::compute_energy() {
    timer t("compute_energy");
    outfile->Printf("\n  * Reference energy        = %18.12f", E_ref_);
    compute_denominators();
    initial_mp2_t();
    double Ecc = cc_energy();
    double pre_Ecc = Ecc;
    outfile->Printf("\n  MP2 correlation energy    = %18.12f", Ecc);
    outfile->Printf("\n  * MP2 total energy        = %18.12f\n", Ecc + E_ref_);

    outfile->Printf("\n  ------------------------------------------------");
    outfile->Printf("\n  Iter         E(CCSD)         dE          dT");
    outfile->Printf("\n  ----- ------------------ ----------- -----------");
    for (size_t i = 1; i <= maxiter_; ++i) {
        compute_effective_tau();
        compute_intermediates();
        update_t();
        Ecc = cc_energy();
        double DT_norm = DT1_.norm() + DT2_.norm();
        outfile->Printf("\n  %4zu   %16.12f   %.3e   %.3e", i, Ecc, fabs(Ecc - pre_Ecc), DT_norm);
        if (fabs(Ecc - pre_Ecc) <= e_convergence_ and DT_norm <= r_convergence_)
            break;
        pre_Ecc = Ecc;
    }
    outfile->Printf("\n  ------------------------------------------------\n");

    outfile->Printf("\n  CCSD correlation energy   = %18.12f", Ecc);
    outfile->Printf("\n  * CCSD total energy       = %18.12f\n", Ecc + E_ref_);
    Process::environment.globals["CURRENT ENERGY"] = Ecc + E_ref_;
    return Ecc + E_ref_;
}

void CC::startup() {
    timer t("startup");
    print_method_banner({"Coupled Cluster Singles and Doubles (CCSD)", "Tianyuan Zhang"});

    e_convergence_ = options_.get_double("E_CONVERGENCE");
    r_convergence_ = options_.get_double("R_CONVERGENCE");
    maxiter_ = options_.get_int("MAXITER");

    outfile->Printf("\n  --------------------------");
    outfile->Printf("\n  Parameters");
    outfile->Printf("\n  --------------------------");
    outfile->Printf("\n  E_convergence  =   %.1e", e_convergence_);
    outfile->Printf("\n  R_convergence  =   %.1e", r_convergence_);
    outfile->Printf("\n  Maxiter        =   %d", maxiter_);
    outfile->Printf("\n  --------------------------\n");

    // frozen-core energy
    ambit::BlockedTensor::reset_mo_spaces();
    frozen_core_energy_ = ints_->frozen_core_energy();

    eri_df_ = false;
    ints_type_ = options_.get_str("INT_TYPE");
    if (ints_type_ == "CHOLESKY" || ints_type_ == "DF" || ints_type_ == "DISKDF") {
        eri_df_ = true;
        aux_mos_ = std::vector<size_t>(ints_->nthree());
        std::iota(aux_mos_.begin(), aux_mos_.end(), 0);
        aux_label_ = "L";
        BTF_->add_mo_space(aux_label_, "g", aux_mos_, NoSpin);
        for (auto s : aux_mos_) {
            outfile->Printf("\naux_mos_: %zu", s);
        }
    }

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

    build_ints();

    // build Fock matrix
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
    if (eri_df_) {
        build_fock_df(H_, B_);
    } else {
        build_fock(H_, V_);
    }

    T1_ = BTF_->build(tensor_type_, "T1", spin_cases({"ov"}));
    T2_ = BTF_->build(tensor_type_, "T2", spin_cases({"oovv"}));

    tilde_tau_ = BTF_->build(tensor_type_, "tilde_tau", spin_cases({"oovv"}));
    tau_ = BTF_->build(tensor_type_, "tau", spin_cases({"oovv"}));

    D1_ = BTF_->build(tensor_type_, "D1", spin_cases({"ov"}));
    D2_ = BTF_->build(tensor_type_, "D2", spin_cases({"oovv"}));

    W1_ = BTF_->build(tensor_type_, "W1", spin_cases({"oo", "ov", "vv"}));
    //    W2_ = BTF_->build(tensor_type_, "W2", spin_cases({"ovvo","oooo","vvvv"}));
    W2_ = BTF_->build(tensor_type_, "W2", {"oooo", "oOoO", "OOOO", "vvvv", "vVvV", "VVVV", "ovvo",
                                           "OVVO", "oVvO", "OvVo", "OvvO", "oVVo"});

    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"ov"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"oovv"}));
}

void CC::build_ints() {
    // prepare integrals
    timer t("build_ints");

    H_ = BTF_->build(tensor_type_, "H", spin_cases({"gg"}));
    // prepare one-electron integrals
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0], i[1]);
        else
            value = ints_->oei_b(i[0], i[1]);
    });

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        B_ = BTF_->build(tensor_type_, "B 3-idx", {"Lgg", "LGG"});
        outfile->Printf("\nBlocks: %s\n", B_.block_labels()[0].c_str());
        fill_three_index_ints(B_);
    } else {
        V_ = BTF_->build(tensor_type_, "V", spin_cases({"gggg"}));
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

        for (auto s : first_index) {
            outfile->Printf("\nfirst_index: %zu", s);
        }
        std::vector<size_t> second_index = mo_to_index[string_block.substr(1, 1)];
        std::vector<size_t> third_index = mo_to_index[string_block.substr(2, 1)];

        for (auto s : third_index) {
            outfile->Printf("\nthird_index: %zu", s);
        }
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

    E_ref_ = Process::environment.molecule()->nuclear_repulsion_energy(
                 reference_wavefunction_->get_dipole_field_strength()) +
             ints_->frozen_core_energy();
    for (const std::string block : {"oo", "OO"}) {
        for (size_t m = 0, nc = mo_space_info_->size("RESTRICTED_DOCC"); m < nc; ++m) {
            E_ref_ += 0.5 * H.block(block).data()[m * nc + m];
            E_ref_ += 0.5 * F_.block(block).data()[m * nc + m];
        }
    }

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

void CC::compute_denominators() {
    timer t("denominators");
    D1_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin)
                value = 1.0 / (Fa_[i[0]] - Fa_[i[1]]);
            if (spin[0] == BetaSpin)
                value = 1.0 / (Fb_[i[0]] - Fb_[i[1]]);
        });

    D2_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                value = 1.0 / (Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
            if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                value = 1.0 / (Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
            if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                value = 1.0 / (Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
        });
}

void CC::compute_effective_tau() {
    timer t("effective_tau");
    tau_["ijab"] = T1_["ia"] * T1_["jb"];
    tau_["ijab"] -= T1_["ib"] * T1_["ja"];
    tau_["iJaB"] = T1_["ia"] * T1_["JB"];
    tau_["IJAB"] = T1_["IA"] * T1_["JB"];
    tau_["IJAB"] -= T1_["IB"] * T1_["JA"];

    tilde_tau_["ijab"] = T2_["ijab"];
    tilde_tau_["ijab"] += 0.5 * tau_["ijab"];
    tilde_tau_["iJaB"] = T2_["iJaB"];
    tilde_tau_["iJaB"] += 0.5 * tau_["iJaB"];
    tilde_tau_["IJAB"] = T2_["IJAB"];
    tilde_tau_["IJAB"] += 0.5 * tau_["IJAB"];

    tau_["ijab"] += T2_["ijab"];
    tau_["iJaB"] += T2_["iJaB"];
    tau_["IJAB"] += T2_["IJAB"];
}

void CC::compute_intermediates() {
    timer t("intermediates");

    timer Fae("Fae");

    W1_["ae"] = F_["ae"];
    W1_["AE"] = F_["AE"];
    for (size_t m = 0, nc = mo_space_info_->size("RESTRICTED_UOCC"); m < nc; ++m) {
        W1_.block("vv").data()[m * nc + m] = 0.0;
        W1_.block("VV").data()[m * nc + m] = 0.0;
    }

    W1_["ae"] -= 0.5 * T1_["ma"] * F_["me"];
    W1_["AE"] -= 0.5 * T1_["MA"] * F_["ME"];

    W1_["ae"] += T1_["mf"] * V_["mafe"];
    W1_["ae"] += T1_["MF"] * V_["aMeF"];
    W1_["AE"] += T1_["mf"] * V_["mAfE"];
    W1_["AE"] += T1_["MF"] * V_["MAFE"];

    W1_["ae"] -= 0.5 * tilde_tau_["mnaf"] * V_["mnef"];
    W1_["ae"] -= tilde_tau_["mNaF"] * V_["mNeF"];
    W1_["AE"] -= tilde_tau_["nMfA"] * V_["nMfE"];
    W1_["AE"] -= 0.5 * tilde_tau_["MNAF"] * V_["MNEF"];

    Fae.stop();

    timer Fmi("Fmi");

    W1_["mi"] = F_["mi"];
    W1_["MI"] = F_["MI"];
    for (size_t m = 0, nc = mo_space_info_->size("RESTRICTED_DOCC"); m < nc; ++m) {
        W1_.block("oo").data()[m * nc + m] = 0.0;
        W1_.block("OO").data()[m * nc + m] = 0.0;
    }

    W1_["mi"] += 0.5 * T1_["ie"] * F_["me"];
    W1_["MI"] += 0.5 * T1_["IE"] * F_["ME"];

    W1_["mi"] += T1_["ne"] * V_["mnie"];
    W1_["mi"] += T1_["NE"] * V_["mNiE"];
    W1_["MI"] += T1_["ne"] * V_["nMeI"];
    W1_["MI"] += T1_["NE"] * V_["MNIE"];

    W1_["mi"] += 0.5 * tilde_tau_["inef"] * V_["mnef"];
    W1_["mi"] += tilde_tau_["iNeF"] * V_["mNeF"];
    W1_["MI"] += tilde_tau_["nIfE"] * V_["nMfE"];
    W1_["MI"] += 0.5 * tilde_tau_["INEF"] * V_["MNEF"];

    Fmi.stop();

    timer Fme("Fme");

    W1_["me"] = F_["me"];
    W1_["ME"] = F_["ME"];

    W1_["me"] += T1_["nf"] * V_["mnef"];
    W1_["me"] += T1_["NF"] * V_["mNeF"];
    W1_["ME"] += T1_["nf"] * V_["nMfE"];
    W1_["ME"] += T1_["NF"] * V_["MNEF"];

    Fme.stop();

    timer Wmnij("Wmnij");

    W2_["mnij"] = V_["mnij"];
    W2_["mNiJ"] = V_["mNiJ"];
    W2_["MNIJ"] = V_["MNIJ"];

    W2_["mnij"] += T1_["je"] * V_["mnie"];
    W2_["mNiJ"] += T1_["JE"] * V_["mNiE"];
    W2_["MNIJ"] += T1_["JE"] * V_["MNIE"];

    W2_["mnij"] -= T1_["ie"] * V_["mnje"];
    W2_["mNiJ"] += T1_["ie"] * V_["mNeJ"];
    W2_["MNIJ"] -= T1_["IE"] * V_["MNJE"];

    W2_["mnij"] += 0.5 * tau_["ijef"] * V_["mnef"];
    W2_["mNiJ"] += tau_["iJeF"] * V_["mNeF"];
    W2_["MNIJ"] += 0.5 * tau_["IJEF"] * V_["MNEF"];

    Wmnij.stop();

    timer Wmbej("Wmbej");

    W2_["mbej"] = V_["mbej"];
    W2_["mBeJ"] = V_["mBeJ"];
    W2_["MBEJ"] = V_["MBEJ"];
    W2_["mBEj"] = -V_["mBjE"];
    W2_["MbeJ"] = -V_["bMeJ"];
    W2_["MbEj"] = V_["bMjE"];

    W2_["mbej"] += T1_["jf"] * V_["mbef"];
    W2_["mBeJ"] += T1_["JF"] * V_["mBeF"];
    W2_["MBEJ"] += T1_["JF"] * V_["MBEF"];
    W2_["mBEj"] -= T1_["jf"] * V_["mBfE"];
    W2_["MbeJ"] -= T1_["JF"] * V_["bMeF"];
    W2_["MbEj"] += T1_["jf"] * V_["bMfE"];

    W2_["mbej"] -= T1_["nb"] * V_["mnej"];
    W2_["mBeJ"] -= T1_["NB"] * V_["mNeJ"];
    W2_["MBEJ"] -= T1_["NB"] * V_["MNEJ"];
    W2_["mBEj"] += T1_["NB"] * V_["mNjE"];
    W2_["MbeJ"] += T1_["nb"] * V_["nMeJ"];
    W2_["MbEj"] -= T1_["nb"] * V_["nMjE"];

    W2_["mbej"] -= 0.5 * T2_["jnfb"] * V_["mnef"];
    W2_["mbej"] += 0.5 * T2_["jNbF"] * V_["mNeF"];
    W2_["mBeJ"] += 0.5 * T2_["nJfB"] * V_["mnef"];
    W2_["mBeJ"] -= 0.5 * T2_["JNFB"] * V_["mNeF"];
    W2_["MBEJ"] += 0.5 * T2_["nJfB"] * V_["nMfE"];
    W2_["MBEJ"] -= 0.5 * T2_["JNFB"] * V_["MNEF"];
    W2_["mBEj"] += 0.5 * T2_["jNfB"] * V_["mNfE"];
    W2_["MbeJ"] += 0.5 * T2_["nJbF"] * V_["nMeF"];
    W2_["MbEj"] -= 0.5 * T2_["jnfb"] * V_["nMfE"];
    W2_["MbEj"] += 0.5 * T2_["jNbF"] * V_["MNEF"];

    W2_["mbej"] -= T1_["jf"] * T1_["nb"] * V_["mnef"];
    W2_["mBeJ"] -= T1_["JF"] * T1_["NB"] * V_["mNeF"];
    W2_["MBEJ"] -= T1_["JF"] * T1_["NB"] * V_["MNEF"];
    W2_["mBEj"] += T1_["jf"] * T1_["NB"] * V_["mNfE"];
    W2_["MbeJ"] += T1_["JF"] * T1_["nb"] * V_["nMeF"];
    W2_["MbEj"] -= T1_["jf"] * T1_["nb"] * V_["nMfE"];

    Wmbej.stop();
}

void CC::update_t() {
    timer t("update_t");
    ambit::BlockedTensor NT1, NT2;

    timer t1("T1");

    NT1 = BTF_->build(tensor_type_, "NT1", spin_cases({"ov"}));

    NT1["ia"] = F_["ia"];
    NT1["IA"] = F_["IA"];

    NT1["ia"] += T1_["ie"] * W1_["ae"];
    NT1["IA"] += T1_["IE"] * W1_["AE"];

    NT1["ia"] -= T1_["ma"] * W1_["mi"];
    NT1["IA"] -= T1_["MA"] * W1_["MI"];

    NT1["ia"] += T2_["imae"] * W1_["me"];
    NT1["ia"] += T2_["iMaE"] * W1_["ME"];
    NT1["IA"] += T2_["mIeA"] * W1_["me"];
    NT1["IA"] += T2_["IMAE"] * W1_["ME"];

    NT1["ia"] -= T1_["nf"] * V_["naif"];
    NT1["ia"] += T1_["NF"] * V_["aNiF"];
    NT1["IA"] += T1_["nf"] * V_["nAfI"];
    NT1["IA"] -= T1_["NF"] * V_["NAIF"];

    NT1["ia"] -= 0.5 * T2_["imef"] * V_["maef"];
    NT1["ia"] += T2_["iMeF"] * V_["aMeF"];
    NT1["IA"] += T2_["mIeF"] * V_["mAeF"];
    NT1["IA"] -= 0.5 * T2_["IMEF"] * V_["MAEF"];

    NT1["ia"] -= 0.5 * T2_["mnae"] * V_["nmei"];
    NT1["ia"] -= T2_["mNaE"] * V_["mNiE"];
    NT1["IA"] -= T2_["nMeA"] * V_["nMeI"];
    NT1["IA"] -= 0.5 * T2_["MNAE"] * V_["NMEI"];

    t1.stop();
    timer t2("T2");

    NT2 = BTF_->build(tensor_type_, "NT2", spin_cases({"oovv"}));

    NT2["ijab"] = V_["ijab"];
    NT2["iJaB"] = V_["iJaB"];
    NT2["IJAB"] = V_["IJAB"];

    NT2["ijab"] += T2_["ijae"] * W1_["be"];
    NT2["iJaB"] += T2_["iJaE"] * W1_["BE"];
    NT2["IJAB"] += T2_["IJAE"] * W1_["BE"];

    NT2["ijab"] -= T2_["ijbe"] * W1_["ae"];
    NT2["iJaB"] += T2_["iJeB"] * W1_["ae"];
    NT2["IJAB"] -= T2_["IJBE"] * W1_["AE"];

    NT2["ijab"] -= 0.5 * T2_["ijae"] * T1_["mb"] * W1_["me"];
    NT2["iJaB"] -= 0.5 * T2_["iJaE"] * T1_["MB"] * W1_["ME"];
    NT2["IJAB"] -= 0.5 * T2_["IJAE"] * T1_["MB"] * W1_["ME"];

    NT2["ijab"] += 0.5 * T2_["ijbe"] * T1_["ma"] * W1_["me"];
    NT2["iJaB"] -= 0.5 * T2_["iJeB"] * T1_["ma"] * W1_["me"];
    NT2["IJAB"] += 0.5 * T2_["IJBE"] * T1_["MA"] * W1_["ME"];

    NT2["ijab"] -= T2_["imab"] * W1_["mj"];
    NT2["iJaB"] -= T2_["iMaB"] * W1_["MJ"];
    NT2["IJAB"] -= T2_["IMAB"] * W1_["MJ"];

    NT2["ijab"] += T2_["jmab"] * W1_["mi"];
    NT2["iJaB"] -= T2_["mJaB"] * W1_["mi"];
    NT2["IJAB"] += T2_["JMAB"] * W1_["MI"];

    NT2["ijab"] -= 0.5 * T2_["imab"] * T1_["je"] * W1_["me"];
    NT2["iJaB"] -= 0.5 * T2_["iMaB"] * T1_["JE"] * W1_["ME"];
    NT2["IJAB"] -= 0.5 * T2_["IMAB"] * T1_["JE"] * W1_["ME"];

    NT2["ijab"] += 0.5 * T2_["jmab"] * T1_["ie"] * W1_["me"];
    NT2["iJaB"] -= 0.5 * T2_["mJaB"] * T1_["ie"] * W1_["me"];
    NT2["IJAB"] += 0.5 * T2_["JMAB"] * T1_["IE"] * W1_["ME"];

    NT2["ijab"] += 0.5 * tau_["mnab"] * W2_["mnij"];
    NT2["iJaB"] += tau_["mNaB"] * W2_["mNiJ"];
    NT2["IJAB"] += 0.5 * tau_["MNAB"] * W2_["MNIJ"];

    NT2["ijab"] += 0.5 * tau_["ijef"] * V_["abef"];
    NT2["iJaB"] += tau_["iJeF"] * V_["aBeF"];
    NT2["IJAB"] += 0.5 * tau_["IJEF"] * V_["ABEF"];

    NT2["ijab"] += 0.5 * tau_["ijef"] * T1_["ma"] * V_["bmef"];
    NT2["iJaB"] -= tau_["iJeF"] * T1_["ma"] * V_["mBeF"];
    NT2["IJAB"] += 0.5 * tau_["IJEF"] * T1_["MA"] * V_["BMEF"];

    NT2["ijab"] -= 0.5 * tau_["ijef"] * T1_["mb"] * V_["amef"];
    NT2["iJaB"] -= tau_["iJeF"] * T1_["MB"] * V_["aMeF"];
    NT2["IJAB"] -= 0.5 * tau_["IJEF"] * T1_["MB"] * V_["AMEF"];

    NT2["ijab"] += T2_["imae"] * W2_["mbej"];
    NT2["ijab"] += T2_["iMaE"] * W2_["MbEj"];
    NT2["iJaB"] += T2_["imae"] * W2_["mBeJ"];
    NT2["iJaB"] += T2_["iMaE"] * W2_["MBEJ"];
    NT2["IJAB"] += T2_["mIeA"] * W2_["mBeJ"];
    NT2["IJAB"] += T2_["IMAE"] * W2_["MBEJ"];

    NT2["ijab"] -= T2_["imbe"] * W2_["maej"];
    NT2["ijab"] -= T2_["iMbE"] * W2_["MaEj"];
    NT2["iJaB"] += T2_["iMeB"] * W2_["MaeJ"];
    NT2["IJAB"] -= T2_["mIeB"] * W2_["mAeJ"];
    NT2["IJAB"] -= T2_["IMBE"] * W2_["MAEJ"];

    NT2["ijab"] -= T2_["jmae"] * W2_["mbei"];
    NT2["ijab"] -= T2_["jMaE"] * W2_["MbEi"];
    NT2["iJaB"] += T2_["mJaE"] * W2_["mBEi"];
    NT2["IJAB"] -= T2_["mJeA"] * W2_["mBeI"];
    NT2["IJAB"] -= T2_["JMAE"] * W2_["MBEI"];

    NT2["ijab"] += T2_["jmbe"] * W2_["maei"];
    NT2["ijab"] += T2_["jMbE"] * W2_["MaEi"];
    NT2["iJaB"] += T2_["mJeB"] * W2_["maei"];
    NT2["iJaB"] += T2_["JMBE"] * W2_["MaEi"];
    NT2["IJAB"] += T2_["mJeB"] * W2_["mAeI"];
    NT2["IJAB"] += T2_["JMBE"] * W2_["MAEI"];

    NT2["ijab"] -= T1_["ie"] * T1_["ma"] * V_["mbej"];
    NT2["iJaB"] -= T1_["ie"] * T1_["ma"] * V_["mBeJ"];
    NT2["IJAB"] -= T1_["IE"] * T1_["MA"] * V_["MBEJ"];

    NT2["ijab"] += T1_["ie"] * T1_["mb"] * V_["maej"];
    NT2["iJaB"] -= T1_["ie"] * T1_["MB"] * V_["aMeJ"];
    NT2["IJAB"] += T1_["IE"] * T1_["MB"] * V_["MAEJ"];

    NT2["ijab"] += T1_["je"] * T1_["ma"] * V_["mbei"];
    NT2["iJaB"] -= T1_["JE"] * T1_["ma"] * V_["mBiE"];
    NT2["IJAB"] += T1_["JE"] * T1_["MA"] * V_["MBEI"];

    NT2["ijab"] -= T1_["je"] * T1_["mb"] * V_["maei"];
    NT2["iJaB"] -= T1_["JE"] * T1_["MB"] * V_["aMiE"];
    NT2["IJAB"] -= T1_["JE"] * T1_["MB"] * V_["MAEI"];

    NT2["ijab"] += T1_["ie"] * V_["abej"];
    NT2["iJaB"] += T1_["ie"] * V_["aBeJ"];
    NT2["IJAB"] += T1_["IE"] * V_["ABEJ"];

    NT2["ijab"] -= T1_["je"] * V_["abei"];
    NT2["iJaB"] += T1_["JE"] * V_["aBiE"];
    NT2["IJAB"] -= T1_["JE"] * V_["ABEI"];

    NT2["ijab"] -= T1_["ma"] * V_["mbij"];
    NT2["iJaB"] -= T1_["ma"] * V_["mBiJ"];
    NT2["IJAB"] -= T1_["MA"] * V_["MBIJ"];

    NT2["ijab"] += T1_["mb"] * V_["maij"];
    NT2["iJaB"] -= T1_["MB"] * V_["aMiJ"];
    NT2["IJAB"] += T1_["MB"] * V_["MAIJ"];

    t2.stop();

    DT1_["ia"] = T1_["ia"];
    DT1_["IA"] = T1_["IA"];
    DT2_["ijab"] = T2_["ijab"];
    DT2_["iJaB"] = T2_["iJaB"];
    DT2_["IJAB"] = T2_["IJAB"];

    T1_["ia"] = NT1["ia"] * D1_["ia"];
    T1_["IA"] = NT1["IA"] * D1_["IA"];

    T2_["ijab"] = NT2["ijab"] * D2_["ijab"];
    T2_["iJaB"] = NT2["iJaB"] * D2_["iJaB"];
    T2_["IJAB"] = NT2["IJAB"] * D2_["IJAB"];

    NT1["ia"] = DT1_["ia"];
    NT1["IA"] = DT1_["IA"];

    NT2["ijab"] = DT2_["ijab"];
    NT2["iJaB"] = DT2_["iJaB"];
    NT2["IJAB"] = DT2_["IJAB"];

    DT1_["ia"] = T1_["ia"] - NT1["ia"];
    DT1_["IA"] = T1_["IA"] - NT1["IA"];

    DT2_["ijab"] = T2_["ijab"] - NT2["ijab"];
    DT2_["iJaB"] = T2_["iJaB"] - NT2["iJaB"];
    DT2_["IJAB"] = T2_["IJAB"] - NT2["IJAB"];
}

void CC::initial_mp2_t() {
    timer t("initial_mp2_t");
    T2_["ijab"] = V_["ijab"] * D2_["ijab"];
    T2_["iJaB"] = V_["iJaB"] * D2_["iJaB"];
    T2_["IJAB"] = V_["IJAB"] * D2_["IJAB"];
}

double CC::cc_energy() {
    timer t("E_corr");
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
