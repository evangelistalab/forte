#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/dipole.h"

#include "helpers/printing.h"
#include "helpers/timer.h"

#include "sadsrg.h"

using namespace psi;

namespace forte {

void SADSRG::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
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
}

void SADSRG::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H1["ex"] * T2["uvey"];
    temp["uvxy"] -= H1["vm"] * T2["umxy"];

    E += L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("120", timer.get());
}

void SADSRG::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
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

void SADSRG::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;

    // <[Hbar2, T2]> (C_2)^4

    // form a temporary tensor K2 = 2 * H2["ijab"] - H2["ijba"]
    BlockedTensor K2;
    auto build_K2_blocks = [&](const std::vector<std::string>& blocks) {
        K2 = ambit::BlockedTensor::build(tensor_type_, "temp_K2", blocks);
        K2["abij"] = 2.0 * H2["abij"];
        K2["abij"] -= H2["abji"];
    };

    // [H2, T2] from ccvv
    build_K2_blocks({"vvcc"});
    E += K2["efmn"] * T2["mnef"];

    // [H2, T2] L1 from cavv
    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_cavv", {"aa"});
    build_K2_blocks({"vvca"});
    temp["vu"] += K2["efmu"] * T2["mvef"];
    E += temp["vu"] * L1_["uv"];

    // [H2, T2] L1 from ccav
    temp.zero();
    temp.set_name("temp_ccav");
    build_K2_blocks({"avcc"});
    temp["vu"] += K2["vemn"] * T2["mnue"];
    E += temp["vu"] * Eta1_["uv"];

    // [H2, T2] L1 from aavv
    temp = ambit::BlockedTensor::build(tensor_type_, "temp_aavv", {"aaaa"});
    build_K2_blocks({"vvaa"});
    temp["yvxu"] += K2["efxu"] * T2["yvef"];
    E += 0.25 * temp["yvxu"] * L1_["uv"] * L1_["xy"];

    // [H2, T2] L1 from ccaa
    temp.zero();
    temp.set_name("temp_ccaa");
    build_K2_blocks({"aacc"});
    temp["vyux"] += K2["vymn"] * T2["mnux"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];

    // [H2, T2] L1 from caav
    temp.zero();
    temp.set_name("temp_caav");
    build_K2_blocks({"avca", "avac"});
    temp["uxyv"] += 0.5 * K2["vemx"] * T2["myue"];
    temp["uxyv"] += 0.5 * K2["vexm"] * T2["ymue"];
    E += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    // [H2, T2] L1 from caaa and aaav
    temp.zero();
    temp.set_name("temp_aaav_caaa");
    build_K2_blocks({"avaa", "aaca"});
    temp["uxyv"] += 0.25 * K2["vexw"] * T2["yzue"] * L1_["wz"];
    temp["uxyv"] += 0.25 * K2["vzmx"] * T2["myuw"] * Eta1_["wz"];
    E += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

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
    build_K2_blocks({"avac"});
    temp["uvxy"] += K2["uexm"] * T2["vmye"];
    temp["uvxy"] -= H2["uexm"] * T2["mvye"];
    temp["uvxy"] -= H2["vemx"] * T2["umey"];

    // HP with Gamma1
    build_K2_blocks({"vaaa"});
    temp["uvxy"] += 0.5 * K2["euwx"] * T2["zvey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["euwx"] * T2["vzey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["evxw"] * T2["uzey"] * L1_["wz"];

    // HP with Eta1
    build_K2_blocks({"aaca"});
    temp["uvxy"] += 0.5 * K2["wumx"] * T2["mvzy"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["wumx"] * T2["mvyz"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["vwmx"] * T2["muyz"] * Eta1_["wz"];

    E += temp["uvxy"] * L2_["uvxy"];

    // <[Hbar2, T2]> C_6 C_2
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        E += H2.block("vaaa")("ewxy") * T2.block("aava")("uvez") * rdms_.SF_L3()("xyzuwv");
        E -= H2.block("aaca")("uvmz") * T2.block("caaa")("mwxy") * rdms_.SF_L3()("xyzuwv");
    }

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
}

void SADSRG::V_T2_C0_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;

    // <[Hbar2, T2]> (C_2)^4

    // form a temporary tensor K2 = 2 * H2["ijab"] - H2["ijba"]
    BlockedTensor K2;
    auto build_K2_blocks = [&](const std::vector<std::string>& blocks) {
        K2 = ambit::BlockedTensor::build(tensor_type_, "temp_K2", blocks);
        K2["abij"] = 2.0 * B["gai"] * B["gbj"];
        K2["abij"] -= B["gaj"] * B["gbi"];
    };

    // [H2, T2] from ccvv
    build_K2_blocks({"vvcc"});
    E += K2["efmn"] * T2["mnef"];

    // [H2, T2] L1 from cavv
    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_cavv", {"aa"});
    build_K2_blocks({"vvca"});
    temp["vu"] += K2["efmu"] * T2["mvef"];
    E += temp["vu"] * L1_["uv"];

    // [H2, T2] L1 from ccav
    temp.zero();
    temp.set_name("temp_ccav");
    build_K2_blocks({"avcc"});
    temp["vu"] += K2["vemn"] * T2["mnue"];
    E += temp["vu"] * Eta1_["uv"];

    // [H2, T2] L1 from aavv
    temp = ambit::BlockedTensor::build(tensor_type_, "temp_aavv", {"aaaa"});
    build_K2_blocks({"vvaa"});
    temp["yvxu"] += K2["efxu"] * T2["yvef"];
    E += 0.25 * temp["yvxu"] * L1_["uv"] * L1_["xy"];

    // [H2, T2] L1 from ccaa
    temp.zero();
    temp.set_name("temp_ccaa");
    build_K2_blocks({"aacc"});
    temp["vyux"] += K2["vymn"] * T2["mnux"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];

    // [H2, T2] L1 from caav
    temp.zero();
    temp.set_name("temp_caav");
    build_K2_blocks({"avca", "avac"});
    temp["uxyv"] += 0.5 * K2["vemx"] * T2["myue"];
    temp["uxyv"] += 0.5 * K2["vexm"] * T2["ymue"];
    E += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    // [H2, T2] L1 from caaa and aaav
    temp.zero();
    temp.set_name("temp_aaav_caaa");
    build_K2_blocks({"avaa", "aaca"});
    temp["uxyv"] += 0.25 * K2["vexw"] * T2["yzue"] * L1_["wz"];
    temp["uxyv"] += 0.25 * K2["vzmx"] * T2["myuw"] * Eta1_["wz"];
    E += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    // <[Hbar2, T2]> C_4 (C_2)^2
    temp.zero();
    temp.set_name("temp_H2T2C0_L2");

    std::vector<std::string> blocks{"aacc", "aaca", "vvaa", "vaaa", "avac", "avca"};
    auto H2 = ambit::BlockedTensor::build(tensor_type_, "temp_H2", blocks);
    H2["abij"] = B["gai"] * B["gbj"];

    // HH
    temp["uvxy"] += 0.5 * H2["uvmn"] * T2["mnxy"];
    temp["uvxy"] += 0.5 * H2["uvmw"] * T2["mzxy"] * L1_["wz"];

    // PP
    temp["uvxy"] += 0.5 * H2["efxy"] * T2["uvef"];
    temp["uvxy"] += 0.5 * H2["ezxy"] * T2["uvew"] * Eta1_["wz"];

    // HP
    build_K2_blocks({"avac"});
    temp["uvxy"] += K2["uexm"] * T2["vmye"];
    temp["uvxy"] -= H2["uexm"] * T2["mvye"];
    temp["uvxy"] -= H2["vemx"] * T2["umey"];

    // HP with Gamma1
    build_K2_blocks({"vaaa"});
    temp["uvxy"] += 0.5 * K2["euwx"] * T2["zvey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["euwx"] * T2["vzey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["evxw"] * T2["uzey"] * L1_["wz"];

    // HP with Eta1
    build_K2_blocks({"aaca"});
    temp["uvxy"] += 0.5 * K2["wumx"] * T2["mvzy"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["wumx"] * T2["mvyz"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["vwmx"] * T2["muyz"] * Eta1_["wz"];

    E += temp["uvxy"] * L2_["uvxy"];

    // <[Hbar2, T2]> C_6 C_2
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        E += H2.block("vaaa")("ewxy") * T2.block("aava")("uvez") * rdms_.SF_L3()("xyzuwv");
        E -= H2.block("aaca")("uvmz") * T2.block("caaa")("mwxy") * rdms_.SF_L3()("xyzuwv");
    }

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
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

void SADSRG::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    auto S2 = ambit::BlockedTensor::build(tensor_type_, "S2", T2.block_labels());
    S2["ijab"] = 2.0 * T2["ijab"];
    S2["ijab"] -= T2["ijba"];

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

void SADSRG::V_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                        BlockedTensor& C1) {
    local_timer timer;

    auto S2 = ambit::BlockedTensor::build(tensor_type_, "S2", T2.block_labels());
    S2["ijab"] = 2.0 * T2["ijab"];
    S2["ijab"] -= T2["ijba"];

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
    C1["jb"] += 0.5 * alpha * B["gax"] * B["gvy"] * S2["ujab"] * L2_["xyuv"];

    C1["jb"] -= 0.5 * alpha * B["gui"] * B["gvy"] * S2["ijxb"] * L2_["xyuv"];

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

void SADSRG::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C2) {
    local_timer timer;

    std::vector<std::string> hhpp;
    for (const std::string& block : T2.block_labels()) {
        if (C2.is_block(block)) {
            hhpp.push_back(block);
        }
    }

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp122", hhpp);
    temp["ijab"] += alpha * T2["ijcb"] * H1["ca"];
    temp["ijab"] -= alpha * T2["kjab"] * H1["ik"];
    C2["ijab"] += temp["ijab"];
    C2["jiba"] += temp["ijab"];

    C2["ijmb"] += alpha * T2["ijab"] * H1["am"];
    C2["jibm"] += alpha * T2["ijab"] * H1["am"];

    C2["ejab"] -= alpha * T2["ijab"] * H1["ei"];
    C2["jeba"] -= alpha * T2["ijab"] * H1["ei"];

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

void SADSRG::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C2) {
    local_timer timer;

    auto S2 = ambit::BlockedTensor::build(tensor_type_, "S2", T2.block_labels());
    S2["ijab"] = 2.0 * T2["ijab"];
    S2["ijab"] -= T2["ijba"];

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
    for (const std::string& block: C2.block_labels()) {
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
    for (const std::string& block: C2.block_labels()) {
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

void SADSRG::V_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                        BlockedTensor& C2) {
    local_timer timer;

    auto S2 = ambit::BlockedTensor::build(tensor_type_, "S2", T2.block_labels());
    S2["ijab"] = 2.0 * T2["ijab"];
    S2["ijab"] -= T2["ijba"];

    // particle-particle contractions
    C2["ijrs"] += alpha * B["gar"] * B["gbs"] * T2["ijab"];

    C2["ijrs"] -= 0.5 * alpha * L1_["xy"] * T2["ijxb"] * B["gyr"] * B["gbs"];
    C2["ijrs"] -= 0.5 * alpha * L1_["xy"] * T2["ijbx"] * B["gbr"] * B["gys"];

    // hole-hole contractions
    C2["pqab"] += alpha * B["gpi"] * B["gqj"] * T2["ijab"];

    C2["pqab"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * B["gpx"] * B["gqj"];
    C2["pqab"] -= 0.5 * alpha * Eta1_["xy"] * T2["jyab"] * B["gpj"] * B["gqx"];

    // hole-particle contractions
    C2["qjsb"] += alpha * B["gam"] * B["gqs"] * T2["mjab"];
    C2["qjsb"] -= alpha * B["gas"] * B["gqm"] * T2["mjab"];

    C2["qjsb"] += 0.5 * alpha * L1_["xy"] * S2["yjab"] * B["gax"] * B["gqs"];
    C2["qjsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * B["gas"] * B["gqx"];

    C2["qjsb"] -= 0.5 * alpha * L1_["xy"] * S2["ijxb"] * B["gyi"] * B["gqs"];
    C2["qjsb"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * B["gys"] * B["gqi"];

    C2["iqsb"] -= alpha * T2["imab"] * B["gas"] * B["gqm"];
    C2["iqsb"] -= 0.5 * alpha * L1_["xy"] * T2["iyab"] * B["gas"] * B["gqx"];
    C2["iqsb"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * B["gys"] * B["gqj"];

    C2["qjas"] -= alpha * T2["mjab"] * B["gqm"] * B["gbs"];
    C2["qjas"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * B["gqx"] * B["gbs"];
    C2["qjas"] += 0.5 * alpha * L1_["xy"] * T2["ijax"] * B["gqi"] * B["gys"];

    C2["iqas"] += alpha * S2["imab"] * B["gbm"] * B["gqs"];
    C2["iqas"] -= alpha * T2["imab"] * B["gbs"] * B["gqm"];

    C2["iqas"] += 0.5 * alpha * L1_["xy"] * S2["iyab"] * B["gbx"] * B["gqs"];
    C2["iqas"] -= 0.5 * alpha * L1_["xy"] * T2["iyab"] * B["gbs"] * B["gqx"];

    C2["iqas"] -= 0.5 * alpha * L1_["xy"] * S2["ijax"] * B["gyj"] * B["gqs"];
    C2["iqas"] += 0.5 * alpha * L1_["xy"] * T2["ijax"] * B["gys"] * B["gqj"];

    //    C2["qjsb"] += alpha * H2["aqms"] * S2["mjab"];
    //    C2["qjsb"] -= alpha * H2["aqsm"] * T2["mjab"];
    //    C2["qjsb"] += 0.5 * alpha * L1_["xy"] * S2["yjab"] * H2["aqxs"];
    //    C2["qjsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * H2["aqsx"];
    //    C2["qjsb"] -= 0.5 * alpha * L1_["xy"] * S2["ijxb"] * H2["yqis"];
    //    C2["qjsb"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["yqsi"];

    //    C2["jqbs"] += alpha * H2["aqms"] * S2["mjab"];
    //    C2["jqbs"] -= alpha * H2["aqsm"] * T2["mjab"];
    //    C2["jqbs"] += 0.5 * alpha * L1_["xy"] * S2["yjab"] * H2["aqxs"];
    //    C2["jqbs"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * H2["aqsx"];
    //    C2["jqbs"] -= 0.5 * alpha * L1_["xy"] * S2["ijxb"] * H2["yqis"];
    //    C2["jqbs"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["yqsi"];

    //    C2["jqsb"] -= alpha * H2["aqsm"] * T2["mjba"];
    //    C2["jqsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjba"] * H2["aqsx"];
    //    C2["jqsb"] += 0.5 * alpha * L1_["xy"] * T2["ijbx"] * H2["yqsi"];

    //    C2["qjbs"] -= alpha * H2["aqsm"] * T2["mjba"];
    //    C2["qjbs"] -= 0.5 * alpha * L1_["xy"] * T2["yjba"] * H2["aqsx"];
    //    C2["qjbs"] += 0.5 * alpha * L1_["xy"] * T2["ijbx"] * H2["yqsi"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}
} // namespace forte
