/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/timer.h"
#include "sadsrg.h"

using namespace psi;

namespace forte {

double SADSRG::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    E += 2.0 * H1["am"] * T1["ma"];

    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp110", {"aa"});
    temp["uv"] += H1["ev"] * T1["ue"];
    temp["uv"] -= H1["um"] * T1["mv"];

    E += L1_["vu"] * temp["uv"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("110", timer.get());
    return E;
}

double SADSRG::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H1["ex"] * T2["uvey"];
    temp["uvxy"] -= H1["vm"] * T2["muyx"];

    E += L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("120", timer.get());
    return E;
}

double SADSRG::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;

    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H2["evxy"] * T1["ue"];
    temp["uvxy"] -= H2["uvmy"] * T1["mx"];

    E += L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("210", timer.get());
    return E;
}

std::vector<double> SADSRG::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2,
                                     const double& alpha, double& C0) {
    local_timer timer;

    std::vector<double> Eout{0.0, 0.0, 0.0};
    double E = 0.0;

    // [H2, T2] (C_2)^4 from ccvv
    E += H2["efmn"] * S2["mnef"];

    // [H2, T2] (C_2)^4 L1 from cavv
    E += H2["efmu"] * S2["mvef"] * L1_["uv"];

    // [H2, T2] (C_2)^4 L1 from ccav
    E += H2["vemn"] * S2["mnue"] * Eta1_["uv"];

    Eout[0] += E;

    // other terms involving T2 with at least two active indices
    auto Esmall = H2_T2_C0_T2small(H2, T2, S2);

    for (int i = 0; i < 3; ++i) {
        E += Esmall[i];
        Eout[i] += Esmall[i];
        Eout[i] *= alpha;
    }

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
    return Eout;
}

std::vector<double> SADSRG::H2_T2_C0_T2small(BlockedTensor& H2, BlockedTensor& T2,
                                             BlockedTensor& S2) {
    /**
     * Note the following blocks should be available in memory.
     * H2: vvaa, aacc, avca, avac, vaaa, aaca
     * T2: aavv, ccaa, caav, acav, aava, caaa
     * S2: aavv, ccaa, caav, acav, aava, caaa
     */

    double E1 = 0.0, E2 = 0.0, E3 = 0.0;

    // [H2, T2] L1 from aavv
    E1 += 0.25 * H2["efxu"] * S2["yvef"] * L1_["uv"] * L1_["xy"];

    // [H2, T2] L1 from ccaa
    E1 += 0.25 * H2["vymn"] * S2["mnux"] * Eta1_["uv"] * Eta1_["xy"];

    // [H2, T2] L1 from caav
    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_caav", {"aaaa"});
    temp["uxyv"] += 0.5 * H2["vemx"] * S2["myue"];
    temp["uxyv"] += 0.5 * H2["vexm"] * S2["ymue"];
    E1 += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    // [H2, T2] L1 from caaa and aaav
    temp.zero();
    temp.set_name("temp_aaav_caaa");
    temp["uxyv"] += 0.25 * H2["evwx"] * S2["zyeu"] * L1_["wz"];
    temp["uxyv"] += 0.25 * H2["vzmx"] * S2["myuw"] * Eta1_["wz"];
    E1 += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    // <[Hbar2, T2]> C_4 (C_2)^2
    temp.zero();
    temp.set_name("temp_H2T2C0_L2");

    // HH
    temp["uvxy"] += 0.5 * H2["uvmn"] * T2["mnxy"];
    temp["uvxy"] += 0.5 * H2["uvmw"] * T2["mzxy"] * L1_["wz"];

    // PP
    temp["uvxy"] += 0.5 * H2["efxy"] * T2["uvef"];
    temp["uvxy"] += 0.5 * H2["ezxy"] * T2["uvew"] * Eta1_["wz"];

    // HP
    temp["uvxy"] += H2["uexm"] * S2["vmye"];
    temp["uvxy"] -= H2["uemx"] * T2["vmye"];
    temp["uvxy"] -= H2["vemx"] * T2["muye"];

    // HP with Gamma1
    temp["uvxy"] += 0.5 * H2["euwx"] * S2["zvey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["euxw"] * T2["zvey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["evxw"] * T2["uzey"] * L1_["wz"];

    // HP with Eta1
    temp["uvxy"] += 0.5 * H2["wumx"] * S2["mvzy"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["uwmx"] * T2["mvzy"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["vwmx"] * T2["muyz"] * Eta1_["wz"];

    E2 += temp["uvxy"] * L2_["uvxy"];

    // <[Hbar2, T2]> C_6 C_2
    if (do_cu3_) {
        E3 += H2.block("vaaa")("ewxy") * T2.block("aava")("uvez") * rdms_.SF_L3()("xyzuwv");
        E3 -= H2.block("aaca")("uvmz") * T2.block("caaa")("mwxy") * rdms_.SF_L3()("xyzuwv");
    }

    return {E1, E2, E3};
}

void SADSRG::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["qa"] -= alpha * H1["qi"] * T1["ia"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("111", timer.get());
}

void SADSRG::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    C1["ia"] += 2.0 * alpha * H1["bm"] * T2["imab"];
    C1["ia"] -= alpha * H1["bm"] * T2["miab"];

    C1["ia"] += alpha * H1["bu"] * T2["ivab"] * L1_["uv"];
    C1["ia"] -= 0.5 * alpha * H1["bu"] * T2["viab"] * L1_["uv"];

    C1["ia"] -= alpha * H1["vj"] * T2["ijau"] * L1_["uv"];
    C1["ia"] += 0.5 * alpha * H1["vj"] * T2["jiau"] * L1_["uv"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("121", timer.get());
}

void SADSRG::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    C1["qp"] += 2.0 * alpha * T1["ma"] * H2["qapm"];
    C1["qp"] -= alpha * T1["ma"] * H2["aqpm"];

    C1["qp"] += alpha * T1["xe"] * L1_["yx"] * H2["qepy"];
    C1["qp"] -= 0.5 * alpha * T1["xe"] * L1_["yx"] * H2["eqpy"];

    C1["qp"] -= alpha * T1["mu"] * L1_["uv"] * H2["qvpm"];
    C1["qp"] += 0.5 * alpha * T1["mu"] * L1_["uv"] * H2["vqpm"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("211", timer.get());
}

void SADSRG::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    C1["ir"] += alpha * H2["abrm"] * S2["imab"];

    C1["ir"] += 0.5 * alpha * L1_["uv"] * S2["ivab"] * H2["abru"];

    C1["ir"] += 0.25 * alpha * S2["ijux"] * L1_["xy"] * L1_["uv"] * H2["vyrj"];

    C1["ir"] -= 0.5 * alpha * L1_["uv"] * S2["imub"] * H2["vbrm"];
    C1["ir"] -= 0.5 * alpha * L1_["uv"] * S2["miub"] * H2["bvrm"];

    C1["ir"] -= 0.25 * alpha * S2["iyub"] * L1_["uv"] * L1_["xy"] * H2["vbrx"];
    C1["ir"] -= 0.25 * alpha * S2["iybu"] * L1_["uv"] * L1_["xy"] * H2["bvrx"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1 ir
    C1["ir"] += 0.5 * alpha * T2["ijxy"] * L2_["xyuv"] * H2["uvrj"];

    C1["ir"] += 0.5 * alpha * H2["aurx"] * S2["ivay"] * L2_["xyuv"];
    C1["ir"] -= 0.5 * alpha * H2["uarx"] * T2["ivay"] * L2_["xyuv"];
    C1["ir"] -= 0.5 * alpha * H2["uarx"] * T2["ivya"] * L2_["xyvu"];

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    C1["pa"] -= alpha * H2["peij"] * S2["ijae"];

    C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * S2["ijau"] * H2["pvij"];

    C1["pa"] -= 0.25 * alpha * S2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * H2["pbux"];

    C1["pa"] += 0.5 * alpha * Eta1_["uv"] * S2["vjae"] * H2["peuj"];
    C1["pa"] += 0.5 * alpha * Eta1_["uv"] * S2["jvae"] * H2["peju"];

    C1["pa"] += 0.25 * alpha * S2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * H2["pyuj"];
    C1["pa"] += 0.25 * alpha * S2["jvax"] * Eta1_["xy"] * Eta1_["uv"] * H2["pyju"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1 pa
    C1["pa"] -= 0.5 * alpha * L2_["xyuv"] * T2["uvab"] * H2["pbxy"];

    C1["pa"] -= 0.5 * alpha * H2["puix"] * S2["ivay"] * L2_["xyuv"];
    C1["pa"] += 0.5 * alpha * H2["puxi"] * T2["ivay"] * L2_["xyuv"];
    C1["pa"] += 0.5 * alpha * H2["puxi"] * T2["viay"] * L2_["xyvu"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    C1["jb"] += 0.5 * alpha * H2["avxy"] * S2["ujab"] * L2_["xyuv"];

    C1["jb"] -= 0.5 * alpha * H2["uviy"] * S2["ijxb"] * L2_["xyuv"];

    C1["qs"] += alpha * H2["eqxs"] * T2["uvey"] * L2_["xyuv"];
    C1["qs"] -= 0.5 * alpha * H2["eqsx"] * T2["uvey"] * L2_["xyuv"];

    C1["qs"] -= alpha * H2["uqms"] * T2["mvxy"] * L2_["xyuv"];
    C1["qs"] += 0.5 * alpha * H2["uqsm"] * T2["mvxy"] * L2_["xyuv"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void SADSRG::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C2) {
    local_timer timer;

    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
    C2["jibp"] += alpha * T2["ijab"] * H1["ap"];

    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
    C2["jqba"] -= alpha * T2["ijab"] * H1["qi"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("122", timer.get());
}

void SADSRG::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C2) {
    local_timer timer;

    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
    C2["riqp"] += alpha * T1["ia"] * H2["arpq"];

    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
    C2["srqa"] -= alpha * T1["ia"] * H2["rsiq"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("212", timer.get());
}

void SADSRG::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                      BlockedTensor& C2) {
    local_timer timer;

    // particle-particle contractions
    C2["ijrs"] += alpha * H2["abrs"] * T2["ijab"];

    C2["ijrs"] -= 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["ybrs"];
    C2["jisr"] -= 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["ybrs"];

    // hole-hole contractions
    C2["pqab"] += alpha * H2["pqij"] * T2["ijab"];

    C2["pqab"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
    C2["qpba"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];

    // hole-particle contractions
    std::vector<std::string> blocks;
    for (const std::string& block : C2.block_labels()) {
        if (block.substr(1, 1) == virt_label_ or block.substr(3, 1) == core_label_)
            continue;
        else
            blocks.push_back(block);
    }

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp", blocks);
    temp["qjsb"] += alpha * H2["aqms"] * S2["mjab"];
    temp["qjsb"] -= alpha * H2["aqsm"] * T2["mjab"];
    temp["qjsb"] += 0.5 * alpha * L1_["xy"] * S2["yjab"] * H2["aqxs"];
    temp["qjsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * H2["aqsx"];
    temp["qjsb"] -= 0.5 * alpha * L1_["xy"] * S2["ijxb"] * H2["yqis"];
    temp["qjsb"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["yqsi"];

    C2["qjsb"] += temp["qjsb"];
    C2["jqbs"] += temp["qjsb"];

    blocks.clear();
    for (const std::string& block : C2.block_labels()) {
        if (block.substr(0, 1) == virt_label_ or block.substr(3, 1) == core_label_)
            continue;
        else
            blocks.push_back(block);
    }

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", blocks);
    temp["jqsb"] -= alpha * H2["aqsm"] * T2["mjba"];
    temp["jqsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjba"] * H2["aqsx"];
    temp["jqsb"] += 0.5 * alpha * L1_["xy"] * T2["ijbx"] * H2["yqsi"];

    C2["jqsb"] += temp["jqsb"];
    C2["qjbs"] += temp["jqsb"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void SADSRG::V_T1_C0_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;

    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp120", {"Laa"});
    temp["gux"] += B["gex"] * T1["ue"];
    temp["gux"] -= B["gum"] * T1["mx"];

    E += L2_["xyuv"] * temp["gux"] * B["gvy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("210", timer.get());
}

std::vector<double> SADSRG::V_T2_C0_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2,
                                       const double& alpha, double& C0) {
    local_timer timer;

    std::vector<double> Eout{0.0, 0.0, 0.0};
    double E = 0.0;

    // [H2, T2] (C_2)^4 from ccvv, cavv, and ccav
    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_220", {"Lvc"});
    temp["gem"] += B["gfn"] * S2["mnef"];
    temp["gem"] += B["gfu"] * S2["mvef"] * L1_["uv"];
    temp["gem"] += B["gvn"] * S2["nmue"] * Eta1_["uv"];
    E += temp["gem"] * B["gem"];
    Eout[0] += E;

    // form H2 for other blocks that fits memory
    std::vector<std::string> blocks{"aacc", "aaca", "vvaa", "vaaa", "avac", "avca"};
    auto H2 = ambit::BlockedTensor::build(tensor_type_, "temp_H2", blocks);
    H2["abij"] = B["gai"] * B["gbj"];

    auto Esmall = H2_T2_C0_T2small(H2, T2, S2);

    for (int i = 0; i < 3; ++i) {
        Eout[i] += Esmall[i];
    }
    E += Esmall[0] + Esmall[1] + Esmall[2];

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
    return Eout;
}

void SADSRG::V_T1_C1_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha,
                        BlockedTensor& C1) {
    local_timer timer;

    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp211", {"L"});
    temp["g"] += 2.0 * alpha * T1["ma"] * B["gam"];
    temp["g"] += alpha * T1["xe"] * L1_["yx"] * B["gey"];
    temp["g"] -= alpha * T1["mu"] * L1_["uv"] * B["gvm"];
    C1["qp"] += temp["g"] * B["gqp"];

    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp211", {"Lgc"});
    temp["gpm"] -= alpha * T1["ma"] * B["gap"];
    temp["gpm"] += 0.5 * alpha * T1["mu"] * L1_["uv"] * B["gvp"];
    C1["qp"] += temp["gpm"] * B["gqm"];

    C1["qp"] -= 0.5 * alpha * T1["xe"] * L1_["yx"] * B["gep"] * B["gqy"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("211", timer.get());
}

void SADSRG::V_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                        BlockedTensor& C1) {
    local_timer timer;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp221", {"Lhp"});

    temp["gia"] += alpha * B["gbm"] * S2["imab"];

    temp["gia"] += 0.5 * alpha * L1_["uv"] * S2["ivab"] * B["gbu"];

    temp["giv"] += 0.25 * alpha * S2["ijux"] * L1_["xy"] * L1_["uv"] * B["gyj"];

    temp["giv"] -= 0.5 * alpha * L1_["uv"] * S2["imub"] * B["gbm"];
    temp["gia"] -= 0.5 * alpha * L1_["uv"] * S2["miua"] * B["gvm"];

    temp["giv"] -= 0.25 * alpha * S2["iyub"] * L1_["uv"] * L1_["xy"] * B["gbx"];
    temp["gia"] -= 0.25 * alpha * S2["iyau"] * L1_["uv"] * L1_["xy"] * B["gvx"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1 ir
    temp["giu"] += 0.5 * alpha * T2["ijxy"] * L2_["xyuv"] * B["gvj"];

    temp["gia"] += 0.5 * alpha * B["gux"] * S2["ivay"] * L2_["xyuv"];
    temp["giu"] -= 0.5 * alpha * B["gax"] * T2["ivay"] * L2_["xyuv"];
    temp["giu"] -= 0.5 * alpha * B["gax"] * T2["ivya"] * L2_["xyvu"];

    C1["ir"] += temp["gia"] * B["gar"];

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    temp.zero();

    temp["gia"] -= alpha * B["gej"] * S2["ijae"];

    temp["gia"] -= 0.5 * alpha * Eta1_["uv"] * S2["ijau"] * B["gvj"];

    temp["gua"] -= 0.25 * alpha * S2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * B["gbx"];

    temp["gua"] += 0.5 * alpha * Eta1_["uv"] * S2["vjae"] * B["gej"];
    temp["gia"] += 0.5 * alpha * Eta1_["uv"] * S2["ivae"] * B["geu"];

    temp["gua"] += 0.25 * alpha * S2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * B["gyj"];
    temp["gia"] += 0.25 * alpha * S2["ivax"] * Eta1_["xy"] * Eta1_["uv"] * B["gyu"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1 pa
    temp["gxa"] -= 0.5 * alpha * L2_["xyuv"] * T2["uvab"] * B["gby"];

    temp["gia"] -= 0.5 * alpha * B["gux"] * S2["ivay"] * L2_["xyuv"];
    temp["gxa"] += 0.5 * alpha * B["gui"] * T2["ivay"] * L2_["xyuv"];
    temp["gxa"] += 0.5 * alpha * B["gui"] * T2["viay"] * L2_["xyvu"];

    C1["pa"] += temp["gia"] * B["gpi"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp221", {"Laa"});
    temp["gxu"] = B["gvy"] * L2_["xyuv"];
    C1["jb"] += 0.5 * alpha * B["gax"] * S2["ujab"] * temp["gxu"];
    C1["jb"] -= 0.5 * alpha * B["gui"] * S2["ijxb"] * temp["gxu"];

    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp221", {"L"});
    temp["g"] += alpha * B["gex"] * T2["uvey"] * L2_["xyuv"];
    temp["g"] -= alpha * B["gum"] * T2["mvxy"] * L2_["xyuv"];
    C1["qs"] += temp["g"] * B["gqs"];

    C1["qs"] -= 0.5 * alpha * B["ges"] * B["gqx"] * T2["uvey"] * L2_["xyuv"];

    C1["qs"] += 0.5 * alpha * B["gus"] * B["gqm"] * T2["mvxy"] * L2_["xyuv"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void SADSRG::V_T1_C2_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha,
                        BlockedTensor& C2) {
    local_timer timer;

    C2["irpq"] += alpha * T1["ia"] * B["gap"] * B["grq"];
    C2["riqp"] += alpha * T1["ia"] * B["gap"] * B["grq"];
    C2["rsaq"] -= alpha * T1["ia"] * B["gri"] * B["gsq"];
    C2["srqa"] -= alpha * T1["ia"] * B["gri"] * B["gsq"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("212", timer.get());
}

void SADSRG::V_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                        BlockedTensor& C2) {
    local_timer timer;

    // particle-particle contractions
    // TODO: need to investigate why using "r" fails when C2 does not contain all "rs"
    C2["ijes"] += batched("e", alpha * B["gae"] * B["gbs"] * T2["ijab"]);
    C2["ijus"] += batched("u", alpha * B["gau"] * B["gbs"] * T2["ijab"]);
    C2["ijms"] += batched("m", alpha * B["gam"] * B["gbs"] * T2["ijab"]);

    std::vector<std::string> C2blocks;
    for (const std::string& block : C2.block_labels()) {
        if (block[0] == virt_label_[0] or block[1] == virt_label_[0])
            continue;
        C2blocks.push_back(block);
    }

    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222", C2blocks);
    temp["ijes"] += batched("e", L1_["xy"] * T2["ijxb"] * B["gye"] * B["gbs"]);
    temp["ijks"] += L1_["xy"] * T2["ijxb"] * B["gyk"] * B["gbs"];

    C2["ijrs"] -= 0.5 * alpha * temp["ijrs"];
    C2["jisr"] -= 0.5 * alpha * temp["ijrs"];

    // hole-hole contractions
    std::vector<std::string> Vblocks;
    for (const std::string& block : C2.block_labels()) {
        if (block[2] == core_label_[0] or block[3] == core_label_[0])
            continue;
        std::string s = block.substr(0, 2) + "hh";
        if (std::find(Vblocks.begin(), Vblocks.end(), s) == Vblocks.end())
            Vblocks.push_back(s);
    }

    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222", Vblocks);
    temp["pqij"] = B["gpi"] * B["gqj"];

    C2["pqab"] += alpha * temp["pqij"] * T2["ijab"];

    C2["pqab"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * temp["pqxj"];
    C2["qpba"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * temp["pqxj"];

    // hole-particle contractions
    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222", {"Lhp"});
    temp["gjb"] += alpha * B["gam"] * S2["mjab"];
    temp["gjb"] += 0.5 * alpha * L1_["xy"] * S2["yjab"] * B["gax"];
    temp["gjb"] -= 0.5 * alpha * L1_["xy"] * S2["ijxb"] * B["gyi"];

    C2["qjsb"] += temp["gjb"] * B["gqs"];
    C2["jqbs"] += temp["gjb"] * B["gqs"];

    // exchange like terms
    V_T2_C2_DF_PH_X(B, T2, alpha, C2);

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void SADSRG::V_T2_C2_DF_PH_X(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                             BlockedTensor& C2) {

    std::vector<std::string> qjsb_small, qjsb_large, jqsb_small, jqsb_large;

    for (const std::string& block : C2.block_labels()) {
        auto j = block.substr(1, 1);
        auto b = block.substr(3, 1);

        if (j != virt_label_ and b != core_label_) {
            if (std::count(block.begin(), block.end(), 'v') > 2) {
                qjsb_large.push_back(block);
            } else {
                qjsb_small.push_back(block);
            }
        }

        j = block.substr(0, 1);
        if (j != virt_label_ and b != core_label_) {
            if (std::count(block.begin(), block.end(), 'v') > 2) {
                jqsb_large.push_back(block);
            } else {
                jqsb_small.push_back(block);
            }
        }
    }

    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", qjsb_small);
    temp["qjsb"] -= alpha * B["gas"] * B["gqm"] * T2["mjab"];
    temp["qjsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * B["gas"] * B["gqx"];
    temp["qjsb"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * B["gys"] * B["gqi"];

    C2["qjsb"] += temp["qjsb"];
    C2["jqbs"] += temp["qjsb"];

    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", jqsb_small);
    temp["jqsb"] -= alpha * B["gas"] * B["gqm"] * T2["mjba"];
    temp["jqsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjba"] * B["gas"] * B["gqx"];
    temp["jqsb"] += 0.5 * alpha * L1_["xy"] * T2["ijbx"] * B["gys"] * B["gqi"];

    C2["jqsb"] += temp["jqsb"];
    C2["qjbs"] += temp["jqsb"];

    if (qjsb_large.size() != 0) {
        C2["e,j,f,v0"] -= batched("e", alpha * B["g,a,f"] * B["g,e,m"] * T2["m,j,a,v0"]);
        C2["j,e,v0,f"] -= batched("e", alpha * B["g,a,f"] * B["g,e,m"] * T2["m,j,a,v0"]);

        temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", {"ahpv"});
        temp["xjae"] = L1_["xy"] * T2["yjae"];
        C2["e,j,f,v0"] -= batched("e", 0.5 * alpha * temp["x,j,a,v0"] * B["g,a,f"] * B["g,e,x"]);
        C2["j,e,v0,f"] -= batched("e", 0.5 * alpha * temp["x,j,a,v0"] * B["g,a,f"] * B["g,e,x"]);

        temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", {"hhav"});
        temp["ijye"] = L1_["xy"] * T2["ijxe"];
        C2["e,j,f,v0"] += batched("e", 0.5 * alpha * temp["i,j,y,v0"] * B["g,y,f"] * B["g,e,i"]);
        C2["j,e,v0,f"] += batched("e", 0.5 * alpha * temp["i,j,y,v0"] * B["g,y,f"] * B["g,e,i"]);
    }

    if (jqsb_large.size() != 0) {
        C2["j,e,f,v0"] -= batched("e", alpha * B["g,a,f"] * B["g,e,m"] * T2["m,j,v0,a"]);
        C2["e,j,v0,f"] -= batched("e", alpha * B["g,a,f"] * B["g,e,m"] * T2["m,j,v0,a"]);

        temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", {"ahvp"});
        temp["xjea"] = L1_["xy"] * T2["yjea"];
        C2["j,e,f,v0"] -= batched("e", 0.5 * alpha * temp["x,j,v0,a"] * B["g,a,f"] * B["g,e,x"]);
        C2["e,j,v0,f"] -= batched("e", 0.5 * alpha * temp["x,j,v0,a"] * B["g,a,f"] * B["g,e,x"]);

        temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", {"hhva"});
        temp["ijey"] = L1_["xy"] * T2["ijex"];
        C2["j,e,f,v0"] += batched("e", 0.5 * alpha * temp["i,j,v0,y"] * B["g,y,f"] * B["g,e,i"]);
        C2["e,j,v0,f"] += batched("e", 0.5 * alpha * temp["i,j,v0,y"] * B["g,y,f"] * B["g,e,i"]);
    }
}

void SADSRG::H_A_Ca(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                    BlockedTensor& S2, const double& alpha, BlockedTensor& C1, BlockedTensor& C2) {
    // set up G2["pqrs"] = 2 * H2["pqrs"] - H2["pqsr"]
    auto G2 = ambit::BlockedTensor::build(tensor_type_, "G2H", {"avac", "aaac", "avaa"});
    G2["pqrs"] = 2.0 * H2["pqrs"] - H2["pqsr"];

    H_A_Ca_small(H1, H2, G2, T1, T2, S2, alpha, C1, C2);

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "tempHACa", {"aa"});
    temp["wz"] += H2["efzm"] * S2["wmef"];
    temp["wz"] -= H2["wemn"] * S2["mnze"];

    C1["uv"] += alpha * temp["uv"];
    C1["vu"] += alpha * temp["uv"];
}

void SADSRG::H_A_Ca_small(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& G2,
                          BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& S2,
                          const double& alpha, BlockedTensor& C1, BlockedTensor& C2) {
    /**
     * The following blocks should be available in memory:
     * G2: avac, aaac, avaa
     * H2: vvaa, aacc, avca, avac, vaaa, aaca, aaaa
     * T2: aavv, ccaa, caav, acav, aava, caaa, aaaa
     * S2: the same as T2
     */

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "tempHACa", {"aa"});

    temp["uv"] += H1["ev"] * T1["ue"];
    temp["uv"] -= H1["um"] * T1["mv"];

    H_T_C1a_smallG(G2, T1, T2, temp);

    H_T_C1a_smallS(H1, H2, T2, S2, temp);

    C1["uv"] += alpha * temp["uv"];
    C1["vu"] += alpha * temp["uv"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});

    H_T_C2a_smallS(H1, H2, T1, T2, S2, temp);

    C2["uvxy"] += alpha * temp["uvxy"];
    C2["xyuv"] += alpha * temp["uvxy"];
}

void SADSRG::H_T_C1a_smallG(BlockedTensor& G2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C1) {
    /**
     * The following blocks should be available in memory:
     * G2: avac, aaac, avaa
     * T2: aava, caaa
     */

    C1["uv"] += T1["ma"] * G2["uavm"];
    C1["uv"] += 0.5 * T1["xe"] * L1_["yx"] * G2["uevy"];
    C1["uv"] -= 0.5 * T1["mx"] * L1_["xy"] * G2["uyvm"];

    C1["wz"] += 0.5 * G2["wezx"] * T2["uvey"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * G2["wuzm"] * T2["mvxy"] * L2_["xyuv"];
}

void SADSRG::H_T_C1a_smallS(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T2,
                            BlockedTensor& S2, BlockedTensor& C1) {
    /**
     * The following blocks should be available in memory:
     * H2: vvaa, aacc, avca, avac, vaaa, aaca, aaaa
     * T2: aavv, ccaa, caav, acav, aava, caaa, aaaa
     * S2: the same as T2
     */

    // C1["uv"] += H1["bm"] * S2["umvb"];
    C1["uv"] += H1["em"] * S2["umve"];
    C1["uv"] += H1["xm"] * S2["muxv"];

    // C1["uv"] += 0.5 * H1["bx"] * S2["uyvb"] * L1_["xy"];
    C1["uv"] += 0.5 * H1["ex"] * S2["yuev"] * L1_["xy"];
    C1["uv"] += 0.5 * H1["wx"] * S2["uyvw"] * L1_["xy"];

    // C1["uv"] -= 0.5 * H1["yj"] * S2["ujvx"] * L1_["xy"];
    C1["uv"] -= 0.5 * H1["ym"] * S2["muxv"] * L1_["xy"];
    C1["uv"] -= 0.5 * H1["yw"] * S2["uwvx"] * L1_["xy"];

    C1["wz"] += H2["uemz"] * S2["mwue"];
    C1["wz"] += H2["uezm"] * S2["wmue"];
    C1["wz"] += H2["vumz"] * S2["mwvu"];

    C1["wz"] -= H2["wemu"] * S2["muze"];
    C1["wz"] -= H2["weum"] * S2["umze"];
    C1["wz"] -= H2["ewvu"] * S2["vuez"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});

    // temp["wzuv"] += 0.5 * S2["wvab"] * H2["abzu"];
    temp["wzuv"] += 0.5 * S2["wvef"] * H2["efzu"];
    temp["wzuv"] += 0.5 * S2["wvex"] * H2["exzu"];
    temp["wzuv"] += 0.5 * S2["vwex"] * H2["exuz"];
    temp["wzuv"] += 0.5 * S2["wvxy"] * H2["xyzu"];

    // temp["wzuv"] -= 0.5 * S2["wmub"] * H2["vbzm"];
    temp["wzuv"] -= 0.5 * S2["wmue"] * H2["vezm"];
    temp["wzuv"] -= 0.5 * S2["mwxu"] * H2["xvmz"];

    // temp["wzuv"] -= 0.5 * S2["mwub"] * H2["bvzm"];
    temp["wzuv"] -= 0.5 * S2["mwue"] * H2["vemz"];
    temp["wzuv"] -= 0.5 * S2["mwux"] * H2["vxmz"];

    temp["wzuv"] += 0.25 * S2["jwxu"] * L1_["xy"] * H2["yvjz"];
    temp["wzuv"] -= 0.25 * S2["ywbu"] * L1_["xy"] * H2["bvxz"];
    temp["wzuv"] -= 0.25 * S2["wybu"] * L1_["xy"] * H2["bvzx"];
    C1["wz"] += temp["wzuv"] * L1_["uv"];

    temp.zero();

    // temp["wzuv"] -= 0.5 * S2["ijzu"] * H2["wvij"];
    temp["wzuv"] -= 0.5 * S2["mnzu"] * H2["wvmn"];
    temp["wzuv"] -= 0.5 * S2["mxzu"] * H2["wvmx"];
    temp["wzuv"] -= 0.5 * S2["mxuz"] * H2["vwmx"];
    temp["wzuv"] -= 0.5 * S2["xyzu"] * H2["wvxy"];

    // temp["wzuv"] += 0.5 * S2["vjze"] * H2["weuj"];
    temp["wzuv"] += 0.5 * S2["vmze"] * H2["weum"];
    temp["wzuv"] += 0.5 * S2["xvez"] * H2["ewxu"];

    // temp["wzuv"] += 0.5 * S2["jvze"] * H2["weju"];
    temp["wzuv"] += 0.5 * S2["mvze"] * H2["wemu"];
    temp["wzuv"] += 0.5 * S2["vxez"] * H2["ewux"];

    temp["wzuv"] -= 0.25 * S2["yvbz"] * Eta1_["xy"] * H2["bwxu"];
    temp["wzuv"] += 0.25 * S2["jvxz"] * Eta1_["xy"] * H2["ywju"];
    temp["wzuv"] += 0.25 * S2["jvzx"] * Eta1_["xy"] * H2["wyju"];
    C1["wz"] += temp["wzuv"] * Eta1_["uv"];

    C1["wz"] += 0.5 * H2["vujz"] * T2["jwyx"] * L2_["xyuv"];
    C1["wz"] += 0.5 * H2["auzx"] * S2["wvay"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * H2["auxz"] * T2["wvay"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * H2["auxz"] * T2["vway"] * L2_["xyvu"];

    C1["wz"] -= 0.5 * H2["bwyx"] * T2["vubz"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * H2["wuix"] * S2["ivzy"] * L2_["xyuv"];
    C1["wz"] += 0.5 * H2["uwix"] * T2["ivzy"] * L2_["xyuv"];
    C1["wz"] += 0.5 * H2["uwix"] * T2["ivyz"] * L2_["xyvu"];

    C1["wz"] += 0.5 * H2["avxy"] * S2["uwaz"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * H2["uviy"] * S2["iwxz"] * L2_["xyuv"];
}

void SADSRG::H_T_C2a_smallS(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                            BlockedTensor& T2, BlockedTensor& S2, BlockedTensor& C2) {
    /**
     * The following blocks should be available in memory:
     * H2: vvaa, aacc, avca, avac, vaaa, aaca, aaaa
     * T2: aavv, ccaa, caav, acav, aava, caaa, aaaa
     * S2: the same as T2
     */

    // C2["uvxy"] += H2["abxy"] * T2["uvab"];
    C2["uvxy"] += H2["efxy"] * T2["uvef"];
    C2["uvxy"] += H2["wzxy"] * T2["uvwz"];
    C2["uvxy"] += H2["ewxy"] * T2["uvew"];
    C2["uvxy"] += H2["ewyx"] * T2["vuew"];

    // C2["uvxy"] += H2["uvij"] * T2["ijxy"];
    C2["uvxy"] += H2["uvmn"] * T2["mnxy"];
    C2["uvxy"] += H2["uvwz"] * T2["wzxy"];
    C2["uvxy"] += H2["vumw"] * T2["mwyx"];
    C2["uvxy"] += H2["uvmw"] * T2["mwxy"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});
    temp["uvxy"] += H1["ax"] * T2["uvay"];
    temp["uvxy"] -= H1["ui"] * T2["ivxy"];
    temp["uvxy"] += T1["ua"] * H2["avxy"];
    temp["uvxy"] -= T1["ix"] * H2["uviy"];

    temp["uvxy"] -= 0.5 * L1_["wz"] * T2["vuaw"] * H2["azyx"];
    temp["uvxy"] -= 0.5 * Eta1_["wz"] * T2["izyx"] * H2["vuiw"];

    // temp["uvxy"] += H2["aumx"] * S2["mvay"];
    temp["uvxy"] += H2["uexm"] * S2["vmye"];
    temp["uvxy"] += H2["wumx"] * S2["mvwy"];

    temp["uvxy"] += 0.5 * L1_["wz"] * S2["zvay"] * H2["auwx"];
    temp["uvxy"] -= 0.5 * L1_["wz"] * S2["ivwy"] * H2["zuix"];

    // temp["uvxy"] -= H2["auxm"] * T2["mvay"];
    temp["uvxy"] -= H2["uemx"] * T2["vmye"];
    temp["uvxy"] -= H2["uwmx"] * T2["mvwy"];

    temp["uvxy"] -= 0.5 * L1_["wz"] * T2["zvay"] * H2["auxw"];
    temp["uvxy"] += 0.5 * L1_["wz"] * T2["ivwy"] * H2["uzix"];

    // temp["uvxy"] -= H2["avxm"] * T2["muya"];
    temp["uvxy"] -= H2["vemx"] * T2["muye"];
    temp["uvxy"] -= H2["vwmx"] * T2["muyw"];

    temp["uvxy"] -= 0.5 * L1_["wz"] * T2["uzay"] * H2["avxw"];
    temp["uvxy"] += 0.5 * L1_["wz"] * T2["iuyw"] * H2["vzix"];

    C2["uvxy"] += temp["uvxy"];
    C2["vuyx"] += temp["uvxy"];
}
} // namespace forte
