/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <map>
#include <vector>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "mrdsrg_so.h"

using namespace psi;

namespace forte {

void MRDSRG_SO::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T1] -> C0 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hp"});
    temp["jb"] = T1["ia"] * Eta1["ab"] * L1["ji"];
    E += temp["jb"] * H1["bj"];
    E *= alpha;
    C0 += E;
    //    outfile->Printf("  Done. Timing %10.3f s; Energy = %14.10f Eh",
    //    timer.get(), E);
}

void MRDSRG_SO::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T1] -> C0 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] += H2["evxy"] * T1["ue"];
    temp["uvxy"] -= H2["uvmy"] * T1["mx"];
    E += 0.5 * temp["uvxy"] * L2["xyuv"];
    E *= alpha;
    C0 += E;
    //    outfile->Printf("  Done. Timing %10.3f s; Energy = %14.10f Eh",
    //    timer.get(), E);
}

void MRDSRG_SO::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T2] -> C0 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] += H1["ex"] * T2["uvey"];
    temp["uvxy"] -= H1["vm"] * T2["umxy"];
    E += 0.5 * temp["uvxy"] * L2["xyuv"];
    E *= alpha;
    C0 += E;
    //    outfile->Printf("  Done. Timing %10.3f s; Energy = %14.10f Eh",
    //    timer.get(), E);
}

void MRDSRG_SO::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T2] -> C0 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    // <[Hbar2, T2]> (C_2)^4
    double E = 0.25 * H2["efmn"] * T2["mnef"];

    BlockedTensor temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhaa"});
    BlockedTensor temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"hhaa"});
    temp1["klux"] = T2["ijux"] * L1["ki"] * L1["lj"];
    temp2["klvy"] = temp1["klux"] * Eta1["uv"] * Eta1["xy"];
    E += 0.25 * H2["vykl"] * temp2["klvy"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhav"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"hhav"});
    temp1["klue"] = T2["ijue"] * L1["ki"] * L1["lj"];
    temp2["klve"] = temp1["klue"] * Eta1["uv"];
    E += 0.5 * H2["vekl"] * temp2["klve"];

    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"aaaa"});
    temp2["yvxu"] -= H2["fexu"] * T2["yvef"];
    E += 0.25 * temp2["yvxu"] * L1["xy"] * L1["uv"];

    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"aa"});
    temp2["vu"] -= H2["femu"] * T2["mvef"];
    E += 0.5 * temp2["vu"] * L1["uv"];

    // <[Hbar2, T2]> C_4 (C_2)^2 HH
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"aahh"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"aaaa"});
    temp1["uvij"] = H2["uvkl"] * L1["ki"] * L1["lj"];
    temp2["uvxy"] += 0.125 * temp1["uvij"] * T2["ijxy"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PP
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"aapp"});
    temp1["uvcd"] = T2["uvab"] * Eta1["ac"] * Eta1["bd"];
    temp2["uvxy"] += 0.125 * temp1["uvcd"] * H2["cdxy"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PH
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hapa"});
    temp1["juby"] = T2["iuay"] * L1["ji"] * Eta1["ab"];
    temp2["uvxy"] += H2["vbjx"] * temp1["juby"];
    E += temp2["uvxy"] * L2["xyuv"];

    // <[Hbar2, T2]> C_6 C_2
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        temp1 = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
        temp1["uvwxyz"] += H2["uviz"] * T2["iwxy"];
        temp1["uvwxyz"] += H2["waxy"] * T2["uvaz"];
        E += 0.25 * temp1["uvwxyz"] * L3["xyzuvw"];
    }

    E *= alpha;
    C0 += E;
    //    outfile->Printf("  Done. Timing %10.3f s; Energy = %14.10f Eh",
    //    timer.get(), E);
}

void MRDSRG_SO::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C1) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T1] -> C1 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["pa"] -= alpha * H1["pi"] * T1["ia"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C1) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T1] -> C1 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C1["qp"] += alpha * T1["ia"] * H2["qapj"] * L1["ji"];
    C1["qp"] -= alpha * T1["mu"] * H2["qvpm"] * L1["uv"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T2] -> C1 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C1["ia"] += alpha * T2["ijab"] * H1["bk"] * L1["kj"];
    C1["ia"] -= alpha * T2["ijau"] * H1["vj"] * L1["uv"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T2] -> C1 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    // [Hbar2, T2] (C_2)^3 -> C1
    BlockedTensor temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhgh"});
    temp1["ijrk"] = H2["abrk"] * T2["ijab"];
    C1["ir"] += 0.5 * alpha * temp1["ijrk"] * L1["kj"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhaa"});
    temp1["ijvy"] = T2["ijux"] * L1["uv"] * L1["xy"];
    C1["ir"] += 0.5 * alpha * temp1["ijvy"] * H2["vyrj"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhap"});
    temp1["ikvb"] = T2["ijub"] * L1["kj"] * L1["uv"];
    C1["ir"] -= alpha * temp1["ikvb"] * H2["vbrk"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhpa"});
    temp1["ijav"] = T2["ijau"] * L1["uv"];
    C1["pa"] -= 0.5 * alpha * temp1["ijav"] * H2["pvij"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhpp"});
    temp1["klab"] = T2["ijab"] * L1["ki"] * L1["lj"];
    C1["pa"] -= 0.5 * alpha * temp1["klab"] * H2["pbkl"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhpa"});
    temp1["ikav"] = T2["ijau"] * L1["uv"] * L1["kj"];
    C1["pa"] += alpha * temp1["ikav"] * H2["pvik"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhaa"});
    temp1["ijuv"] = T2["ijxy"] * L2["xyuv"];
    C1["ir"] += 0.25 * alpha * temp1["ijuv"] * H2["uvrj"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"aapp"});
    temp1["xyab"] = T2["uvab"] * L2["xyuv"];
    C1["pa"] -= 0.25 * alpha * temp1["xyab"] * H2["pbxy"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hapa"});
    temp1["iuax"] = T2["iyav"] * L2["uvxy"];
    C1["ir"] += alpha * temp1["iuax"] * H2["axru"];
    C1["pa"] -= alpha * temp1["iuax"] * H2["pxiu"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"pa"});
    temp1["au"] = H2["avxy"] * L2["xyuv"];
    C1["jb"] += 0.5 * alpha * temp1["au"] * T2["ujab"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"ah"});
    temp1["ui"] = H2["xyiv"] * L2["uvxy"];
    C1["jb"] -= 0.5 * alpha * temp1["ui"] * T2["ijub"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"av"});
    temp1["xe"] = T2["uvey"] * L2["xyuv"];
    C1["qs"] += 0.5 * alpha * temp1["xe"] * H2["eqxs"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"ca"});
    temp1["mx"] = T2["myuv"] * L2["uvxy"];
    C1["qs"] -= 0.5 * alpha * temp1["mx"] * H2["xqms"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T2] -> C2 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
    C2["ijap"] += alpha * T2["ijab"] * H1["bp"];
    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
    C2["iqab"] -= alpha * T2["ijab"] * H1["qj"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C2) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T1] -> C2 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
    C2["ripq"] += alpha * T1["ia"] * H2["rapq"];
    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
    C2["rspa"] -= alpha * T1["ia"] * H2["rspi"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T2] -> C2 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    // particle-particle contractions
    C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"];

    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpa"});
    temp["ijby"] = T2["ijbx"] * L1["xy"];
    C2["ijrs"] -= alpha * temp["ijby"] * H2["byrs"];

    // hole-hole contractions
    C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ahpp"});
    temp["xjab"] = T2["yjab"] * Eta1["xy"];
    C2["pqab"] -= alpha * temp["xjab"] * H2["pqxj"];

    // particle-hole contractions
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpp"});
    temp["kjab"] = T2["ijab"] * L1["ki"];
    C2["qjsb"] += alpha * temp["kjab"] * H2["aqks"];
    C2["qjas"] += alpha * temp["kjab"] * H2["bqks"];

    C2["iqsb"] -= alpha * temp["kiab"] * H2["aqks"];
    C2["iqas"] -= alpha * temp["kiab"] * H2["bqks"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhap"});
    temp["ijvb"] = T2["ijub"] * L1["uv"];
    C2["qjsb"] -= alpha * temp["ijvb"] * H2["vqis"];
    C2["iqsb"] -= alpha * temp["ijvb"] * H2["vqjs"];

    C2["qjas"] += alpha * temp["ijva"] * H2["vqis"];
    C2["iqas"] += alpha * temp["ijva"] * H2["vqjs"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T2_C3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C3) {
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gggggg"});
    temp["rsjabq"] -= H2["rsqi"] * T2["ijab"];
    C3["rsjabq"] += alpha * temp["rsjabq"];
    C3["rjsabq"] -= alpha * temp["rsjabq"];
    C3["jrsabq"] += alpha * temp["rsjabq"];
    C3["rsjaqb"] -= alpha * temp["rsjabq"];
    C3["rjsaqb"] += alpha * temp["rsjabq"];
    C3["jrsaqb"] -= alpha * temp["rsjabq"];
    C3["rsjqab"] += alpha * temp["rsjabq"];
    C3["rjsqab"] -= alpha * temp["rsjabq"];
    C3["jrsqab"] += alpha * temp["rsjabq"];

    temp.zero();
    temp["ijspqb"] += H2["aspq"] * T2["ijba"];
    C3["ijspqb"] += alpha * temp["ijspqb"];
    C3["isjpqb"] -= alpha * temp["ijspqb"];
    C3["sijpqb"] += alpha * temp["ijspqb"];
    C3["ijspbq"] -= alpha * temp["ijspqb"];
    C3["isjpbq"] += alpha * temp["ijspqb"];
    C3["sijpbq"] -= alpha * temp["ijspqb"];
    C3["ijsbpq"] += alpha * temp["ijspqb"];
    C3["isjbpq"] -= alpha * temp["ijspqb"];
    C3["sijbpq"] += alpha * temp["ijspqb"];
}

void MRDSRG_SO::H3_T1_C1(BlockedTensor& H3, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C1) {
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gaagaa"});
    temp["qxypuv"] += 0.5 * H3["qeypuv"] * T1["xe"];
    temp["qxypuv"] -= 0.5 * H3["qxypmv"] * T1["mu"];

    C1["qp"] += alpha * temp["qxypuv"] * L2["uvxy"];
}

void MRDSRG_SO::H3_T1_C2(BlockedTensor& H3, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C2) {
    C2["toqr"] += alpha * H3["atojqr"] * T1["ia"] * L1["ji"];
    C2["toqr"] -= alpha * H3["ytomqr"] * T1["mx"] * L1["xy"];
}

void MRDSRG_SO::H3_T2_C1(BlockedTensor& H3, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    // (6:2)
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"});
    temp["ax"] += (1.0 / 12.0) * H3["ayzuvw"] * L3["uvwxyz"];
    C1["jb"] += alpha * temp["ax"] * T2["xjab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"});
    temp["ui"] += (1.0 / 12.0) * H3["xyzivw"] * L3["uvwxyz"];
    C1["jb"] -= alpha * temp["ui"] * T2["ijub"];

    // (5:3)
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ppga"});
    temp["abpx"] += 0.25 * H3["abypuv"] * L2["uvxy"];
    C1["ip"] += alpha * temp["abpx"] * T2["ixab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"paga"});
    temp["azpx"] += 0.5 * H3["azypuv"] * L2["uvxy"];
    C1["ip"] -= alpha * T2["ixaw"] * L1["wz"] * temp["azpx"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gahh"});
    temp["suij"] += 0.25 * H3["sxyijv"] * L2["uvxy"];
    C1["sa"] += alpha * temp["suij"] * T2["ijau"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gaha"});
    temp["suiw"] += 0.5 * H3["sxyiwv"] * L2["uvxy"];
    C1["sa"] -= alpha * temp["suiw"] * Eta1["wz"] * T2["izau"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aagh"});
    temp["uzpj"] += 0.5 * H3["xyzpjv"] * L2["uvxy"];
    C1["ip"] -= alpha * T2["ijuw"] * L1["wz"] * temp["uzpj"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pagc"});
    temp["bupm"] += 0.5 * H3["xybpmv"] * L2["uvxy"];
    C1["ip"] += alpha * T2["imub"] * temp["bupm"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"paga"});
    temp["bupw"] += 0.5 * H3["xybpwv"] * L2["uvxy"];
    C1["ip"] += alpha * T2["izub"] * temp["bupw"] * L1["wz"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gpaa"});
    temp["sbwx"] += 0.5 * H3["sbyuvw"] * L2["uvxy"];
    C1["sa"] -= alpha * temp["sbwx"] * Eta1["wz"] * T2["xzab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gvha"});
    temp["sejx"] += 0.5 * H3["seyjuv"] * L2["uvxy"];
    C1["sa"] += alpha * temp["sejx"] * T2["xjae"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gaha"});
    temp["szjx"] += 0.5 * H3["szyujv"] * L2["uvxy"];
    C1["sa"] += alpha * temp["szjx"] * T2["xjaw"] * Eta1["wz"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aagh"});
    temp["uvpj"] += (1.0 / 12.0) * H3["xyzpjw"] * L3["uvwxyz"];
    C1["ip"] += alpha * T2["ijuv"] * temp["uvpj"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"paga"});
    temp["aupx"] += 0.25 * H3["ayzpvw"] * L3["uvwxyz"];
    C1["ip"] += alpha * T2["ixau"] * temp["aupx"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gpaa"});
    temp["sbxy"] += (1.0 / 12.0) * H3["sbzuvw"] * L3["uvwxyz"];
    C1["sa"] -= alpha * temp["sbxy"] * T2["xyab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gaha"});
    temp["suix"] += 0.25 * H3["syzivw"] * L3["uvwxyz"];
    C1["sa"] -= alpha * temp["suix"] * T2["ixau"];

    // (4:4)
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpp"});
    temp["klab"] += 0.25 * T2["ijab"] * L1["ki"] * L1["lj"];
    temp["klav"] -= 0.5 * T2["ijau"] * L1["ki"] * L1["uv"] * L1["lj"];
    C1["or"] += alpha * H3["aboklr"] * temp["klab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhaa"});
    temp["ijvy"] -= 0.25 * T2["ijux"] * L1["uv"] * L1["xy"];
    temp["ikvy"] += 0.5 * T2["ijux"] * L1["xy"] * L1["kj"] * L1["uv"];
    C1["or"] += alpha * H3["vyoijr"] * temp["ijvy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aapp"});
    temp["uvab"] += 0.125 * T2["xyab"] * L2["uvxy"];
    C1["or"] += alpha * H3["abouvr"] * temp["uvab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aava"});
    temp["uvez"] += 0.25 * T2["xyew"] * L1["wz"] * L2["uvxy"];
    C1["or"] -= alpha * H3["ezouvr"] * temp["uvez"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhaa"});
    temp["ijxy"] += 0.125 * T2["ijuv"] * L2["uvxy"];
    C1["or"] += alpha * H3["xyoijr"] * temp["ijxy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"caaa"});
    temp["mwxy"] += 0.25 * T2["mzuv"] * L2["uvxy"] * Eta1["wz"];
    C1["or"] -= alpha * H3["xyomwr"] * temp["mwxy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["iuax"] += T2["iyav"] * L2["uvxy"];
    C1["or"] += alpha * H3["axoiur"] * temp["iuax"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aava"});
    temp["wuex"] -= T2["zyev"] * Eta1["wz"] * L2["uvxy"];
    temp["uvex"] += 0.25 * T2["yzew"] * L3["uvwxyz"];
    C1["or"] += alpha * H3["exowur"] * temp["wuex"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"caaa"});
    temp["muzx"] -= T2["mywv"] * L2["uvxy"] * L1["wz"];
    temp["muyz"] += 0.25 * T2["mxvw"] * L3["uvwxyz"];
    C1["or"] += alpha * H3["zxomur"] * temp["muzx"];
}

void MRDSRG_SO::H3_T2_C2(BlockedTensor& H3, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    // (4:2)
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhapaa"});
    BlockedTensor temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"ghpg"});
    temp["ijuaxy"] += 0.5 * T2["ijav"] * L2["uvxy"];
    C2["ijpq"] += alpha * temp["ijuaxy"] * H3["axypqu"];
    temp2["ojar"] += alpha * temp["ijvaxy"] * H3["xyoivr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"haappa"});
    temp["iuvabx"] += 0.5 * T2["iyab"] * L2["uvxy"];
    C2["stab"] -= alpha * temp["iuvabx"] * H3["stxiuv"];
    temp2["ojar"] -= alpha * temp["juvaby"] * H3["byouvr"];

    C2["ojar"] += temp2["ojar"];
    C2["ojra"] -= temp2["ojar"];
    C2["joar"] -= temp2["ojar"];
    C2["jora"] += temp2["ojar"];

    // (3:3)
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpp"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"gggp"});
    temp["ijcb"] += 0.5 * T2["ijab"] * Eta1["ac"];
    temp2["torb"] += alpha * temp["ijcb"] * H3["ctorij"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aapp"});
    temp["uxab"] += 0.5 * T2["vyab"] * Eta1["uv"] * Eta1["xy"];
    temp2["torb"] += alpha * temp["uxab"] * H3["atorux"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"happ"});
    temp["iucb"] += T2["ivab"] * Eta1["ac"] * Eta1["uv"];
    temp2["torb"] -= alpha * temp["iucb"] * H3["ctoriu"];

    C2["torb"] += temp2["torb"];
    C2["tobr"] -= temp2["torb"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpp"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"ghgg"});
    temp["kjab"] += 0.5 * T2["ijab"] * L1["ki"];
    temp2["ojqr"] -= alpha * temp["kjab"] * H3["abokqr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhaa"});
    temp["ijvy"] += 0.5 * T2["ijux"] * L1["uv"] * L1["xy"];
    temp2["ojqr"] -= alpha * temp["ijvy"] * H3["vyoiqr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpa"});
    temp["kjav"] += T2["ijau"] * L1["uv"] * L1["ki"];
    temp2["ojqr"] += alpha * temp["kjav"] * H3["avokqr"];

    C2["ojqr"] += temp2["ojqr"];
    C2["joqr"] -= temp2["ojqr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aapp"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"ggpg"});
    temp["uvab"] += 0.25 * T2["xyab"] * L2["uvxy"];
    temp2["rsaq"] += alpha * temp["uvab"] * H3["brsuvq"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["iuax"] += T2["iyav"] * L2["uvxy"];
    temp2["rsaq"] -= alpha * temp["iuax"] * H3["rsxiqu"];

    C2["rsaq"] += temp2["rsaq"];
    C2["rsqa"] -= temp2["rsaq"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhaa"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"hggg"});
    temp["ijxy"] += 0.25 * T2["ijuv"] * L2["uvxy"];
    temp2["ispq"] -= alpha * temp["ijxy"] * H3["xysjpq"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["iuax"] += T2["iyav"] * L2["uvxy"];
    temp2["ispq"] += alpha * temp["iuax"] * H3["asxpqu"];

    C2["ispq"] += temp2["ispq"];
    C2["sipq"] -= temp2["ispq"];

    // (2:4)
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"});
    temp["ue"] += 0.5 * T2["xyev"] * L2["uvxy"];
    C2["toqr"] += alpha * temp["ue"] * H3["etouqr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"});
    temp["mx"] += 0.5 * T2["myuv"] * L2["uvxy"];
    C2["toqr"] -= alpha * temp["mx"] * H3["xtomqr"];
}

void MRDSRG_SO::comm_H_A_2(int n, BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                           BlockedTensor& T2, double& C0, BlockedTensor& C1, BlockedTensor& C2) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    BlockedTensor temp;
    double factor = 1.0 / n;

    C0 += 1.0 * H1["h0,p0"] * T1["h1,p0"] * L1["h1,h0"];
    C0 += (1.0 / 2.0) * H1["a0,p0"] * T2["a2,a3,p0,a1"] * L2["a2,a3,a0,a1"];
    C0 += -1.0 * H1["a0,h0"] * T1["h0,a1"] * L1["a1,a0"];
    C0 += (-1.0 / 2.0) * H1["a0,h0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C0 += (1.0 / 8.0) * H2["a0,a1,p0,p1"] * T2["a2,a3,p0,p1"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 8.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 2.0) * H2["a1,a2,p0,a0"] * T1["a3,p0"] * L2["a1,a2,a0,a3"];
    C0 += (-1.0 / 4.0) * H2["a1,a2,p0,a0"] * T2["a4,a5,p0,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    C0 += (1.0 / 2.0) * H2["a1,a2,h0,a0"] * T1["h0,a3"] * L2["a1,a2,a0,a3"];
    C0 += (1.0 / 4.0) * H2["a1,a2,h0,a0"] * T2["h0,a3,a4,a5"] * L3["a1,a2,a3,a0,a4,a5"];
    C0 += (1.0 / 4.0) * H2["h0,h1,p0,p1"] * T2["h2,h3,p0,p1"] * L1["h2,h0"] * L1["h3,h1"];
    C0 += 1.0 * H2["h0,a1,p0,a0"] * T2["h1,a3,p0,a2"] * L1["h1,h0"] * L2["a1,a2,a0,a3"];
    C0 += (-1.0 / 4.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C0 += (1.0 / 4.0) * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h2,h1"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 4.0) * H2["a1,a2,p0,a0"] * T2["a4,a5,p0,a3"] * L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C0 += -1.0 * H2["a1,a2,h0,a0"] * T2["h0,a3,a4,a5"] * L1["a4,a1"] * L2["a2,a3,a0,a5"];
    C0 += (-1.0 / 2.0) * H2["h0,h1,p0,a0"] * T2["h2,h3,p0,a1"] * L1["h2,h0"] * L1["h3,h1"] * L1["a1,a0"];
    C0 += (1.0 / 2.0) * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h2,h1"] * L1["a2,a0"] * L1["a3,a1"];

    C1["a0,a1"] += -1.0 * H1["a0,h0"] * T1["h0,a1"];
    C1["a0,a1"] += 1.0 * H1["a1,p0"] * T1["a0,p0"];
    C1["a0,a1"] += 1.0 * H1["h0,p0"] * T2["h1,a0,p0,a1"] * L1["h1,h0"];
    C1["a0,a1"] += 1.0 * H1["a2,h0"] * T2["h0,a0,a1,a3"] * L1["a3,a2"];
    C1["a0,a1"] += (1.0 / 2.0) * H2["h0,a1,p0,p1"] * T2["h1,a0,p0,p1"] * L1["h1,h0"];
    C1["a0,a1"] += 1.0 * H2["h0,a1,p0,a0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["a0,a1"] += (-1.0 / 2.0) * H2["a0,a2,h0,h1"] * T2["h0,h1,a1,a3"] * L1["a3,a2"];
    C1["a0,a1"] += 1.0 * H2["a0,a2,h0,a1"] * T1["h0,a3"] * L1["a3,a2"];
    C1["a0,a1"] += (1.0 / 2.0) * H2["a0,a2,h0,a1"] * T2["h0,a3,a4,a5"] * L2["a4,a5,a2,a3"];
    C1["a0,a1"] += -1.0 * H2["a0,a3,h0,a2"] * T2["h0,a4,a1,a5"] * L2["a3,a4,a2,a5"];
    C1["a0,a1"] += (-1.0 / 2.0) * H2["a1,a2,p0,a0"] * T2["a4,a5,p0,a3"] * L2["a4,a5,a2,a3"];
    C1["a0,a1"] += 1.0 * H2["a1,a3,p0,a2"] * T2["a0,a5,p0,a4"] * L2["a3,a4,a2,a5"];
    C1["a0,a1"] += (-1.0 / 4.0) * H2["a2,a3,p0,a0"] * T2["a4,a5,p0,a1"] * L2["a4,a5,a2,a3"];
    C1["a0,a1"] += (1.0 / 4.0) * H2["a2,a3,h0,a1"] * T2["h0,a0,a4,a5"] * L2["a4,a5,a2,a3"];
    C1["a0,a1"] += (1.0 / 2.0) * H2["a3,a4,p0,a2"] * T2["a0,a5,p0,a1"] * L2["a3,a4,a2,a5"];
    C1["a0,a1"] += (-1.0 / 2.0) * H2["a3,a4,h0,a2"] * T2["h0,a0,a1,a5"] * L2["a3,a4,a2,a5"];
    C1["a0,a1"] += (-1.0 / 2.0) * H2["h0,h1,p0,a0"] * T2["h2,h3,p0,a1"] * L1["h2,h0"] * L1["h3,h1"];
    C1["a0,a1"] += -1.0 * H2["h0,a1,p0,a2"] * T2["h1,a0,p0,a3"] * L1["h1,h0"] * L1["a3,a2"];
    C1["a0,a1"] += 1.0 * H2["a0,a2,h0,h1"] * T2["h0,h2,a1,a3"] * L1["h2,h1"] * L1["a3,a2"];
    C1["a0,a1"] += (1.0 / 2.0) * H2["a2,a3,h0,a1"] * T2["h0,a0,a4,a5"] * L1["a4,a2"] * L1["a5,a3"];

    C1["a0,c0"] += 1.0 * H1["c0,p0"] * T1["a0,p0"];
    C1["a0,c0"] += (1.0 / 2.0) * H2["h0,c0,p0,p1"] * T2["h1,a0,p0,p1"] * L1["h1,h0"];
    C1["a0,c0"] += 1.0 * H2["h0,c0,p0,a0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["a0,c0"] += (-1.0 / 2.0) * H2["c0,a1,p0,a0"] * T2["a3,a4,p0,a2"] * L2["a3,a4,a1,a2"];
    C1["a0,c0"] += 1.0 * H2["c0,a2,p0,a1"] * T2["a0,a4,p0,a3"] * L2["a2,a3,a1,a4"];
    C1["a0,c0"] += 1.0 * H2["a0,a1,h0,c0"] * T1["h0,a2"] * L1["a2,a1"];
    C1["a0,c0"] += (1.0 / 2.0) * H2["a0,a1,h0,c0"] * T2["h0,a2,a3,a4"] * L2["a3,a4,a1,a2"];
    C1["a0,c0"] += (1.0 / 4.0) * H2["a1,a2,h0,c0"] * T2["h0,a0,a3,a4"] * L2["a3,a4,a1,a2"];
    C1["a0,c0"] += -1.0 * H2["h0,c0,p0,a1"] * T2["h1,a0,p0,a2"] * L1["h1,h0"] * L1["a2,a1"];
    C1["a0,c0"] += (1.0 / 2.0) * H2["a1,a2,h0,c0"] * T2["h0,a0,a3,a4"] * L1["a3,a1"] * L1["a4,a2"];

    C1["a0,v0"] += 1.0 * H1["v0,p0"] * T1["a0,p0"];
    C1["a0,v0"] += -1.0 * H1["a0,h0"] * T1["h0,v0"];
    C1["a0,v0"] += 1.0 * H1["h0,p0"] * T2["h1,a0,p0,v0"] * L1["h1,h0"];
    C1["a0,v0"] += 1.0 * H1["a1,h0"] * T2["h0,a0,v0,a2"] * L1["a2,a1"];
    C1["a0,v0"] += (1.0 / 2.0) * H2["h0,v0,p0,p1"] * T2["h1,a0,p0,p1"] * L1["h1,h0"];
    C1["a0,v0"] += 1.0 * H2["h0,v0,p0,a0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["a0,v0"] += (-1.0 / 2.0) * H2["v0,a1,p0,a0"] * T2["a3,a4,p0,a2"] * L2["a3,a4,a1,a2"];
    C1["a0,v0"] += 1.0 * H2["v0,a2,p0,a1"] * T2["a0,a4,p0,a3"] * L2["a2,a3,a1,a4"];
    C1["a0,v0"] += (-1.0 / 2.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,v0,a2"] * L1["a2,a1"];
    C1["a0,v0"] += 1.0 * H2["a0,a1,h0,v0"] * T1["h0,a2"] * L1["a2,a1"];
    C1["a0,v0"] += (1.0 / 2.0) * H2["a0,a1,h0,v0"] * T2["h0,a2,a3,a4"] * L2["a3,a4,a1,a2"];
    C1["a0,v0"] += -1.0 * H2["a0,a2,h0,a1"] * T2["h0,a3,v0,a4"] * L2["a2,a3,a1,a4"];
    C1["a0,v0"] += (-1.0 / 4.0) * H2["a1,a2,p0,a0"] * T2["a3,a4,p0,v0"] * L2["a3,a4,a1,a2"];
    C1["a0,v0"] += (1.0 / 4.0) * H2["a1,a2,h0,v0"] * T2["h0,a0,a3,a4"] * L2["a3,a4,a1,a2"];
    C1["a0,v0"] += (1.0 / 2.0) * H2["a2,a3,p0,a1"] * T2["a0,a4,p0,v0"] * L2["a2,a3,a1,a4"];
    C1["a0,v0"] += (-1.0 / 2.0) * H2["a2,a3,h0,a1"] * T2["h0,a0,v0,a4"] * L2["a2,a3,a1,a4"];
    C1["a0,v0"] += (-1.0 / 2.0) * H2["h0,h1,p0,a0"] * T2["h2,h3,p0,v0"] * L1["h2,h0"] * L1["h3,h1"];
    C1["a0,v0"] += -1.0 * H2["h0,v0,p0,a1"] * T2["h1,a0,p0,a2"] * L1["h1,h0"] * L1["a2,a1"];
    C1["a0,v0"] += 1.0 * H2["a0,a1,h0,h1"] * T2["h0,h2,v0,a2"] * L1["h2,h1"] * L1["a2,a1"];
    C1["a0,v0"] += (1.0 / 2.0) * H2["a1,a2,h0,v0"] * T2["h0,a0,a3,a4"] * L1["a3,a1"] * L1["a4,a2"];

    C1["c0,a0"] += -1.0 * H1["c0,h0"] * T1["h0,a0"];
    C1["c0,a0"] += 1.0 * H1["a0,p0"] * T1["c0,p0"];
    C1["c0,a0"] += 1.0 * H1["h0,p0"] * T2["h1,c0,p0,a0"] * L1["h1,h0"];
    C1["c0,a0"] += 1.0 * H1["a1,h0"] * T2["h0,c0,a0,a2"] * L1["a2,a1"];
    C1["c0,a0"] += (1.0 / 2.0) * H2["h0,a0,p0,p1"] * T2["h1,c0,p0,p1"] * L1["h1,h0"];
    C1["c0,a0"] += 1.0 * H2["h0,a0,p0,c0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["c0,a0"] += (-1.0 / 2.0) * H2["c0,a1,h0,h1"] * T2["h0,h1,a0,a2"] * L1["a2,a1"];
    C1["c0,a0"] += 1.0 * H2["c0,a1,h0,a0"] * T1["h0,a2"] * L1["a2,a1"];
    C1["c0,a0"] += (1.0 / 2.0) * H2["c0,a1,h0,a0"] * T2["h0,a2,a3,a4"] * L2["a3,a4,a1,a2"];
    C1["c0,a0"] += -1.0 * H2["c0,a2,h0,a1"] * T2["h0,a3,a0,a4"] * L2["a2,a3,a1,a4"];
    C1["c0,a0"] += (-1.0 / 2.0) * H2["a0,a1,p0,c0"] * T2["a3,a4,p0,a2"] * L2["a3,a4,a1,a2"];
    C1["c0,a0"] += 1.0 * H2["a0,a2,p0,a1"] * T2["c0,a4,p0,a3"] * L2["a2,a3,a1,a4"];
    C1["c0,a0"] += (-1.0 / 4.0) * H2["a1,a2,p0,c0"] * T2["a3,a4,p0,a0"] * L2["a3,a4,a1,a2"];
    C1["c0,a0"] += (1.0 / 4.0) * H2["a1,a2,h0,a0"] * T2["h0,c0,a3,a4"] * L2["a3,a4,a1,a2"];
    C1["c0,a0"] += (1.0 / 2.0) * H2["a2,a3,p0,a1"] * T2["c0,a4,p0,a0"] * L2["a2,a3,a1,a4"];
    C1["c0,a0"] += (-1.0 / 2.0) * H2["a2,a3,h0,a1"] * T2["h0,c0,a0,a4"] * L2["a2,a3,a1,a4"];
    C1["c0,a0"] += (-1.0 / 2.0) * H2["h0,h1,p0,c0"] * T2["h2,h3,p0,a0"] * L1["h2,h0"] * L1["h3,h1"];
    C1["c0,a0"] += -1.0 * H2["h0,a0,p0,a1"] * T2["h1,c0,p0,a2"] * L1["h1,h0"] * L1["a2,a1"];
    C1["c0,a0"] += 1.0 * H2["c0,a1,h0,h1"] * T2["h0,h2,a0,a2"] * L1["h2,h1"] * L1["a2,a1"];
    C1["c0,a0"] += (1.0 / 2.0) * H2["a1,a2,h0,a0"] * T2["h0,c0,a3,a4"] * L1["a3,a1"] * L1["a4,a2"];

    C1["c0,c1"] += 1.0 * H1["c1,p0"] * T1["c0,p0"];
    C1["c0,c1"] += (1.0 / 2.0) * H2["h0,c1,p0,p1"] * T2["h1,c0,p0,p1"] * L1["h1,h0"];
    C1["c0,c1"] += 1.0 * H2["h0,c1,p0,c0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["c0,c1"] += 1.0 * H2["c0,a0,h0,c1"] * T1["h0,a1"] * L1["a1,a0"];
    C1["c0,c1"] += (1.0 / 2.0) * H2["c0,a0,h0,c1"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["c0,c1"] += (-1.0 / 2.0) * H2["c1,a0,p0,c0"] * T2["a2,a3,p0,a1"] * L2["a2,a3,a0,a1"];
    C1["c0,c1"] += 1.0 * H2["c1,a1,p0,a0"] * T2["c0,a3,p0,a2"] * L2["a1,a2,a0,a3"];
    C1["c0,c1"] += (1.0 / 4.0) * H2["a0,a1,h0,c1"] * T2["h0,c0,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["c0,c1"] += -1.0 * H2["h0,c1,p0,a0"] * T2["h1,c0,p0,a1"] * L1["h1,h0"] * L1["a1,a0"];
    C1["c0,c1"] += (1.0 / 2.0) * H2["a0,a1,h0,c1"] * T2["h0,c0,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];

    C1["c0,v0"] += 1.0 * H1["v0,p0"] * T1["c0,p0"];
    C1["c0,v0"] += -1.0 * H1["c0,h0"] * T1["h0,v0"];
    C1["c0,v0"] += 1.0 * H1["h0,p0"] * T2["h1,c0,p0,v0"] * L1["h1,h0"];
    C1["c0,v0"] += 1.0 * H1["a0,h0"] * T2["h0,c0,v0,a1"] * L1["a1,a0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["h0,v0,p0,p1"] * T2["h1,c0,p0,p1"] * L1["h1,h0"];
    C1["c0,v0"] += 1.0 * H2["h0,v0,p0,c0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v0,a0,p0,c0"] * T2["a2,a3,p0,a1"] * L2["a2,a3,a0,a1"];
    C1["c0,v0"] += 1.0 * H2["v0,a1,p0,a0"] * T2["c0,a3,p0,a2"] * L2["a1,a2,a0,a3"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["c0,a0,h0,h1"] * T2["h0,h1,v0,a1"] * L1["a1,a0"];
    C1["c0,v0"] += 1.0 * H2["c0,a0,h0,v0"] * T1["h0,a1"] * L1["a1,a0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c0,a0,h0,v0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["c0,v0"] += -1.0 * H2["c0,a1,h0,a0"] * T2["h0,a2,v0,a3"] * L2["a1,a2,a0,a3"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["a0,a1,p0,c0"] * T2["a2,a3,p0,v0"] * L2["a2,a3,a0,a1"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["a0,a1,h0,v0"] * T2["h0,c0,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["a1,a2,p0,a0"] * T2["c0,a3,p0,v0"] * L2["a1,a2,a0,a3"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["a1,a2,h0,a0"] * T2["h0,c0,v0,a3"] * L2["a1,a2,a0,a3"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["h0,h1,p0,c0"] * T2["h2,h3,p0,v0"] * L1["h2,h0"] * L1["h3,h1"];
    C1["c0,v0"] += -1.0 * H2["h0,v0,p0,a0"] * T2["h1,c0,p0,a1"] * L1["h1,h0"] * L1["a1,a0"];
    C1["c0,v0"] += 1.0 * H2["c0,a0,h0,h1"] * T2["h0,h2,v0,a1"] * L1["h2,h1"] * L1["a1,a0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["a0,a1,h0,v0"] * T2["h0,c0,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];

    C1["v0,a0"] += -1.0 * H1["v0,h0"] * T1["h0,a0"];
    C1["v0,a0"] += 1.0 * H2["h0,a0,p0,v0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["v0,a0"] += (-1.0 / 2.0) * H2["v0,a1,h0,h1"] * T2["h0,h1,a0,a2"] * L1["a2,a1"];
    C1["v0,a0"] += 1.0 * H2["v0,a1,h0,a0"] * T1["h0,a2"] * L1["a2,a1"];
    C1["v0,a0"] += (1.0 / 2.0) * H2["v0,a1,h0,a0"] * T2["h0,a2,a3,a4"] * L2["a3,a4,a1,a2"];
    C1["v0,a0"] += -1.0 * H2["v0,a2,h0,a1"] * T2["h0,a3,a0,a4"] * L2["a2,a3,a1,a4"];
    C1["v0,a0"] += (-1.0 / 2.0) * H2["a0,a1,p0,v0"] * T2["a3,a4,p0,a2"] * L2["a3,a4,a1,a2"];
    C1["v0,a0"] += (-1.0 / 4.0) * H2["a1,a2,p0,v0"] * T2["a3,a4,p0,a0"] * L2["a3,a4,a1,a2"];
    C1["v0,a0"] += (-1.0 / 2.0) * H2["h0,h1,p0,v0"] * T2["h2,h3,p0,a0"] * L1["h2,h0"] * L1["h3,h1"];
    C1["v0,a0"] += 1.0 * H2["v0,a1,h0,h1"] * T2["h0,h2,a0,a2"] * L1["h2,h1"] * L1["a2,a1"];

    C1["v0,c0"] += 1.0 * H2["h0,c0,p0,v0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["v0,c0"] += 1.0 * H2["v0,a0,h0,c0"] * T1["h0,a1"] * L1["a1,a0"];
    C1["v0,c0"] += (1.0 / 2.0) * H2["v0,a0,h0,c0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["v0,c0"] += (-1.0 / 2.0) * H2["c0,a0,p0,v0"] * T2["a2,a3,p0,a1"] * L2["a2,a3,a0,a1"];

    C1["v0,v1"] += -1.0 * H1["v0,h0"] * T1["h0,v1"];
    C1["v0,v1"] += 1.0 * H2["h0,v1,p0,v0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["v0,v1"] += (-1.0 / 2.0) * H2["v0,a0,h0,h1"] * T2["h0,h1,v1,a1"] * L1["a1,a0"];
    C1["v0,v1"] += 1.0 * H2["v0,a0,h0,v1"] * T1["h0,a1"] * L1["a1,a0"];
    C1["v0,v1"] += (1.0 / 2.0) * H2["v0,a0,h0,v1"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["v0,v1"] += -1.0 * H2["v0,a1,h0,a0"] * T2["h0,a2,v1,a3"] * L2["a1,a2,a0,a3"];
    C1["v0,v1"] += (-1.0 / 2.0) * H2["v1,a0,p0,v0"] * T2["a2,a3,p0,a1"] * L2["a2,a3,a0,a1"];
    C1["v0,v1"] += (-1.0 / 4.0) * H2["a0,a1,p0,v0"] * T2["a2,a3,p0,v1"] * L2["a2,a3,a0,a1"];
    C1["v0,v1"] += (-1.0 / 2.0) * H2["h0,h1,p0,v0"] * T2["h2,h3,p0,v1"] * L1["h2,h0"] * L1["h3,h1"];
    C1["v0,v1"] += 1.0 * H2["v0,a0,h0,h1"] * T2["h0,h2,v1,a1"] * L1["h2,h1"] * L1["a1,a0"];

    C2["a0,a1,a2,a3"] += (-1.0 / 2.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"];
    C2["a0,a1,a2,a3"] += (1.0 / 2.0) * H2["a2,a3,p0,p1"] * T2["a0,a1,p0,p1"];
    C2["a0,a1,a2,a3"] += 1.0 * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h2,h1"];
    C2["a0,a1,a2,a3"] += -1.0 * H2["a2,a3,p0,a4"] * T2["a0,a1,p0,a5"] * L1["a5,a4"];

    C2["a0,a1,c0,c1"] += (1.0 / 2.0) * H2["c0,c1,p0,p1"] * T2["a0,a1,p0,p1"];
    C2["a0,a1,c0,c1"] += -1.0 * H2["c0,c1,p0,a2"] * T2["a0,a1,p0,a3"] * L1["a3,a2"];

    C2["a0,a1,v0,v1"] += (1.0 / 2.0) * H2["v0,v1,p0,p1"] * T2["a0,a1,p0,p1"];
    C2["a0,a1,v0,v1"] += (-1.0 / 2.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,v0,v1"];
    C2["a0,a1,v0,v1"] += -1.0 * H2["v0,v1,p0,a2"] * T2["a0,a1,p0,a3"] * L1["a3,a2"];
    C2["a0,a1,v0,v1"] += 1.0 * H2["a0,a1,h0,h1"] * T2["h0,h2,v0,v1"] * L1["h2,h1"];

    C2["c0,c1,a0,a1"] += (-1.0 / 2.0) * H2["c0,c1,h0,h1"] * T2["h0,h1,a0,a1"];
    C2["c0,c1,a0,a1"] += (1.0 / 2.0) * H2["a0,a1,p0,p1"] * T2["c0,c1,p0,p1"];
    C2["c0,c1,a0,a1"] += 1.0 * H2["c0,c1,h0,h1"] * T2["h0,h2,a0,a1"] * L1["h2,h1"];
    C2["c0,c1,a0,a1"] += -1.0 * H2["a0,a1,p0,a2"] * T2["c0,c1,p0,a3"] * L1["a3,a2"];

    C2["c0,c1,c2,c3"] += (1.0 / 2.0) * H2["c2,c3,p0,p1"] * T2["c0,c1,p0,p1"];
    C2["c0,c1,c2,c3"] += -1.0 * H2["c2,c3,p0,a0"] * T2["c0,c1,p0,a1"] * L1["a1,a0"];

    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v0,v1,p0,p1"] * T2["c0,c1,p0,p1"];
    C2["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c1,h0,h1"] * T2["h0,h1,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v0,v1,p0,a0"] * T2["c0,c1,p0,a1"] * L1["a1,a0"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c0,c1,h0,h1"] * T2["h0,h2,v0,v1"] * L1["h2,h1"];

    C2["v0,v1,a0,a1"] += (-1.0 / 2.0) * H2["v0,v1,h0,h1"] * T2["h0,h1,a0,a1"];
    C2["v0,v1,a0,a1"] += 1.0 * H2["v0,v1,h0,h1"] * T2["h0,h2,a0,a1"] * L1["h2,h1"];

    C2["v0,v1,v2,v3"] += (-1.0 / 2.0) * H2["v0,v1,h0,h1"] * T2["h0,h1,v2,v3"];
    C2["v0,v1,v2,v3"] += 1.0 * H2["v0,v1,h0,h1"] * T2["h0,h2,v2,v3"] * L1["h2,h1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});
    temp["a0,a1,a2,a3"] += -1.0 * H1["a0,h0"] * T2["h0,a1,a2,a3"];
    temp["a0,a1,a2,a3"] += -1.0 * H2["a2,a3,p0,a0"] * T1["a1,p0"];
    C2["a0,a1,a2,a3"] += temp["a0,a1,a2,a3"];
    C2["a1,a0,a2,a3"] -= temp["a0,a1,a2,a3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});
    temp["a0,a1,a2,a3"] += 1.0 * H1["a2,p0"] * T2["a0,a1,p0,a3"];
    temp["a0,a1,a2,a3"] += 1.0 * H2["a0,a1,h0,a2"] * T1["h0,a3"];
    C2["a0,a1,a2,a3"] += temp["a0,a1,a2,a3"];
    C2["a0,a1,a3,a2"] -= temp["a0,a1,a2,a3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaca"});
    temp["a0,a1,c0,a2"] += 1.0 * H1["c0,p0"] * T2["a0,a1,p0,a2"];
    temp["a0,a1,c0,a2"] += (1.0 / 2.0) * H2["c0,a2,p0,p1"] * T2["a0,a1,p0,p1"];
    temp["a0,a1,c0,a2"] += 1.0 * H2["a0,a1,h0,c0"] * T1["h0,a2"];
    temp["a0,a1,c0,a2"] += -1.0 * H2["c0,a2,p0,a3"] * T2["a0,a1,p0,a4"] * L1["a4,a3"];
    C2["a0,a1,c0,a2"] += temp["a0,a1,c0,a2"];
    C2["a0,a1,a2,c0"] -= temp["a0,a1,c0,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aacc"});
    temp["a0,a1,c0,c1"] += -1.0 * H2["c0,c1,p0,a0"] * T1["a1,p0"];
    C2["a0,a1,c0,c1"] += temp["a0,a1,c0,c1"];
    C2["a1,a0,c0,c1"] -= temp["a0,a1,c0,c1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aava"});
    temp["a0,a1,v0,a2"] += 1.0 * H1["v0,p0"] * T2["a0,a1,p0,a2"];
    temp["a0,a1,v0,a2"] += -1.0 * H1["a2,p0"] * T2["a0,a1,p0,v0"];
    temp["a0,a1,v0,a2"] += (1.0 / 2.0) * H2["v0,a2,p0,p1"] * T2["a0,a1,p0,p1"];
    temp["a0,a1,v0,a2"] += (-1.0 / 2.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,v0,a2"];
    temp["a0,a1,v0,a2"] += 1.0 * H2["a0,a1,h0,v0"] * T1["h0,a2"];
    temp["a0,a1,v0,a2"] += -1.0 * H2["a0,a1,h0,a2"] * T1["h0,v0"];
    temp["a0,a1,v0,a2"] += -1.0 * H2["v0,a2,p0,a3"] * T2["a0,a1,p0,a4"] * L1["a4,a3"];
    temp["a0,a1,v0,a2"] += 1.0 * H2["a0,a1,h0,h1"] * T2["h0,h2,v0,a2"] * L1["h2,h1"];
    C2["a0,a1,v0,a2"] += temp["a0,a1,v0,a2"];
    C2["a0,a1,a2,v0"] -= temp["a0,a1,v0,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aavc"});
    temp["a0,a1,v0,c0"] += -1.0 * H1["c0,p0"] * T2["a0,a1,p0,v0"];
    temp["a0,a1,v0,c0"] += (1.0 / 2.0) * H2["v0,c0,p0,p1"] * T2["a0,a1,p0,p1"];
    temp["a0,a1,v0,c0"] += -1.0 * H2["a0,a1,h0,c0"] * T1["h0,v0"];
    temp["a0,a1,v0,c0"] += -1.0 * H2["v0,c0,p0,a2"] * T2["a0,a1,p0,a3"] * L1["a3,a2"];
    C2["a0,a1,v0,c0"] += temp["a0,a1,v0,c0"];
    C2["a0,a1,c0,v0"] -= temp["a0,a1,v0,c0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aavv"});
    temp["a0,a1,v0,v1"] += 1.0 * H1["v0,p0"] * T2["a0,a1,p0,v1"];
    temp["a0,a1,v0,v1"] += 1.0 * H2["a0,a1,h0,v0"] * T1["h0,v1"];
    C2["a0,a1,v0,v1"] += temp["a0,a1,v0,v1"];
    C2["a0,a1,v1,v0"] -= temp["a0,a1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aavv"});
    temp["a0,a1,v0,v1"] += -1.0 * H1["a0,h0"] * T2["h0,a1,v0,v1"];
    temp["a0,a1,v0,v1"] += -1.0 * H2["v0,v1,p0,a0"] * T1["a1,p0"];
    C2["a0,a1,v0,v1"] += temp["a0,a1,v0,v1"];
    C2["a1,a0,v0,v1"] -= temp["a0,a1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"caaa"});
    temp["c0,a0,a1,a2"] += -1.0 * H1["c0,h0"] * T2["h0,a0,a1,a2"];
    temp["c0,a0,a1,a2"] += 1.0 * H1["a0,h0"] * T2["h0,c0,a1,a2"];
    temp["c0,a0,a1,a2"] += (-1.0 / 2.0) * H2["c0,a0,h0,h1"] * T2["h0,h1,a1,a2"];
    temp["c0,a0,a1,a2"] += (1.0 / 2.0) * H2["a1,a2,p0,p1"] * T2["c0,a0,p0,p1"];
    temp["c0,a0,a1,a2"] += -1.0 * H2["a1,a2,p0,c0"] * T1["a0,p0"];
    temp["c0,a0,a1,a2"] += 1.0 * H2["a1,a2,p0,a0"] * T1["c0,p0"];
    temp["c0,a0,a1,a2"] += 1.0 * H2["c0,a0,h0,h1"] * T2["h0,h2,a1,a2"] * L1["h2,h1"];
    temp["c0,a0,a1,a2"] += -1.0 * H2["a1,a2,p0,a3"] * T2["c0,a0,p0,a4"] * L1["a4,a3"];
    C2["c0,a0,a1,a2"] += temp["c0,a0,a1,a2"];
    C2["a0,c0,a1,a2"] -= temp["c0,a0,a1,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cacc"});
    temp["c0,a0,c1,c2"] += (1.0 / 2.0) * H2["c1,c2,p0,p1"] * T2["c0,a0,p0,p1"];
    temp["c0,a0,c1,c2"] += -1.0 * H2["c1,c2,p0,c0"] * T1["a0,p0"];
    temp["c0,a0,c1,c2"] += 1.0 * H2["c1,c2,p0,a0"] * T1["c0,p0"];
    temp["c0,a0,c1,c2"] += -1.0 * H2["c1,c2,p0,a1"] * T2["c0,a0,p0,a2"] * L1["a2,a1"];
    C2["c0,a0,c1,c2"] += temp["c0,a0,c1,c2"];
    C2["a0,c0,c1,c2"] -= temp["c0,a0,c1,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cavv"});
    temp["c0,a0,v0,v1"] += -1.0 * H1["c0,h0"] * T2["h0,a0,v0,v1"];
    temp["c0,a0,v0,v1"] += 1.0 * H1["a0,h0"] * T2["h0,c0,v0,v1"];
    temp["c0,a0,v0,v1"] += (1.0 / 2.0) * H2["v0,v1,p0,p1"] * T2["c0,a0,p0,p1"];
    temp["c0,a0,v0,v1"] += -1.0 * H2["v0,v1,p0,c0"] * T1["a0,p0"];
    temp["c0,a0,v0,v1"] += 1.0 * H2["v0,v1,p0,a0"] * T1["c0,p0"];
    temp["c0,a0,v0,v1"] += (-1.0 / 2.0) * H2["c0,a0,h0,h1"] * T2["h0,h1,v0,v1"];
    temp["c0,a0,v0,v1"] += -1.0 * H2["v0,v1,p0,a1"] * T2["c0,a0,p0,a2"] * L1["a2,a1"];
    temp["c0,a0,v0,v1"] += 1.0 * H2["c0,a0,h0,h1"] * T2["h0,h2,v0,v1"] * L1["h2,h1"];
    C2["c0,a0,v0,v1"] += temp["c0,a0,v0,v1"];
    C2["a0,c0,v0,v1"] -= temp["c0,a0,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccaa"});
    temp["c0,c1,a0,a1"] += -1.0 * H1["c0,h0"] * T2["h0,c1,a0,a1"];
    temp["c0,c1,a0,a1"] += -1.0 * H2["a0,a1,p0,c0"] * T1["c1,p0"];
    C2["c0,c1,a0,a1"] += temp["c0,c1,a0,a1"];
    C2["c1,c0,a0,a1"] -= temp["c0,c1,a0,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccaa"});
    temp["c0,c1,a0,a1"] += 1.0 * H1["a0,p0"] * T2["c0,c1,p0,a1"];
    temp["c0,c1,a0,a1"] += 1.0 * H2["c0,c1,h0,a0"] * T1["h0,a1"];
    C2["c0,c1,a0,a1"] += temp["c0,c1,a0,a1"];
    C2["c0,c1,a1,a0"] -= temp["c0,c1,a0,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccca"});
    temp["c0,c1,c2,a0"] += 1.0 * H1["c2,p0"] * T2["c0,c1,p0,a0"];
    temp["c0,c1,c2,a0"] += 1.0 * H2["c0,c1,h0,c2"] * T1["h0,a0"];
    temp["c0,c1,c2,a0"] += (1.0 / 2.0) * H2["c2,a0,p0,p1"] * T2["c0,c1,p0,p1"];
    temp["c0,c1,c2,a0"] += -1.0 * H2["c2,a0,p0,a1"] * T2["c0,c1,p0,a2"] * L1["a2,a1"];
    C2["c0,c1,c2,a0"] += temp["c0,c1,c2,a0"];
    C2["c0,c1,a0,c2"] -= temp["c0,c1,c2,a0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccc"});
    temp["c0,c1,c2,c3"] += -1.0 * H2["c2,c3,p0,c0"] * T1["c1,p0"];
    C2["c0,c1,c2,c3"] += temp["c0,c1,c2,c3"];
    C2["c1,c0,c2,c3"] -= temp["c0,c1,c2,c3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccva"});
    temp["c0,c1,v0,a0"] += 1.0 * H1["v0,p0"] * T2["c0,c1,p0,a0"];
    temp["c0,c1,v0,a0"] += -1.0 * H1["a0,p0"] * T2["c0,c1,p0,v0"];
    temp["c0,c1,v0,a0"] += (1.0 / 2.0) * H2["v0,a0,p0,p1"] * T2["c0,c1,p0,p1"];
    temp["c0,c1,v0,a0"] += (-1.0 / 2.0) * H2["c0,c1,h0,h1"] * T2["h0,h1,v0,a0"];
    temp["c0,c1,v0,a0"] += 1.0 * H2["c0,c1,h0,v0"] * T1["h0,a0"];
    temp["c0,c1,v0,a0"] += -1.0 * H2["c0,c1,h0,a0"] * T1["h0,v0"];
    temp["c0,c1,v0,a0"] += -1.0 * H2["v0,a0,p0,a1"] * T2["c0,c1,p0,a2"] * L1["a2,a1"];
    temp["c0,c1,v0,a0"] += 1.0 * H2["c0,c1,h0,h1"] * T2["h0,h2,v0,a0"] * L1["h2,h1"];
    C2["c0,c1,v0,a0"] += temp["c0,c1,v0,a0"];
    C2["c0,c1,a0,v0"] -= temp["c0,c1,v0,a0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
    temp["c0,c1,v0,c2"] += -1.0 * H1["c2,p0"] * T2["c0,c1,p0,v0"];
    temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H2["v0,c2,p0,p1"] * T2["c0,c1,p0,p1"];
    temp["c0,c1,v0,c2"] += -1.0 * H2["c0,c1,h0,c2"] * T1["h0,v0"];
    temp["c0,c1,v0,c2"] += -1.0 * H2["v0,c2,p0,a0"] * T2["c0,c1,p0,a1"] * L1["a1,a0"];
    C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
    C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["v0,p0"] * T2["c0,c1,p0,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c0,c1,h0,v0"] * T1["h0,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H1["c0,h0"] * T2["h0,c1,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v0,v1,p0,c0"] * T1["c1,p0"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vaaa"});
    temp["v0,a0,a1,a2"] += -1.0 * H1["v0,h0"] * T2["h0,a0,a1,a2"];
    temp["v0,a0,a1,a2"] += (-1.0 / 2.0) * H2["v0,a0,h0,h1"] * T2["h0,h1,a1,a2"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["a1,a2,p0,v0"] * T1["a0,p0"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a0,h0,h1"] * T2["h0,h2,a1,a2"] * L1["h2,h1"];
    C2["v0,a0,a1,a2"] += temp["v0,a0,a1,a2"];
    C2["a0,v0,a1,a2"] -= temp["v0,a0,a1,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vacc"});
    temp["v0,a0,c0,c1"] += -1.0 * H2["c0,c1,p0,v0"] * T1["a0,p0"];
    C2["v0,a0,c0,c1"] += temp["v0,a0,c0,c1"];
    C2["a0,v0,c0,c1"] -= temp["v0,a0,c0,c1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vavv"});
    temp["v0,a0,v1,v2"] += -1.0 * H1["v0,h0"] * T2["h0,a0,v1,v2"];
    temp["v0,a0,v1,v2"] += (-1.0 / 2.0) * H2["v0,a0,h0,h1"] * T2["h0,h1,v1,v2"];
    temp["v0,a0,v1,v2"] += -1.0 * H2["v1,v2,p0,v0"] * T1["a0,p0"];
    temp["v0,a0,v1,v2"] += 1.0 * H2["v0,a0,h0,h1"] * T2["h0,h2,v1,v2"] * L1["h2,h1"];
    C2["v0,a0,v1,v2"] += temp["v0,a0,v1,v2"];
    C2["a0,v0,v1,v2"] -= temp["v0,a0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcaa"});
    temp["v0,c0,a0,a1"] += -1.0 * H1["v0,h0"] * T2["h0,c0,a0,a1"];
    temp["v0,c0,a0,a1"] += (-1.0 / 2.0) * H2["v0,c0,h0,h1"] * T2["h0,h1,a0,a1"];
    temp["v0,c0,a0,a1"] += -1.0 * H2["a0,a1,p0,v0"] * T1["c0,p0"];
    temp["v0,c0,a0,a1"] += 1.0 * H2["v0,c0,h0,h1"] * T2["h0,h2,a0,a1"] * L1["h2,h1"];
    C2["v0,c0,a0,a1"] += temp["v0,c0,a0,a1"];
    C2["c0,v0,a0,a1"] -= temp["v0,c0,a0,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
    temp["v0,c0,c1,c2"] += -1.0 * H2["c1,c2,p0,v0"] * T1["c0,p0"];
    C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
    C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
    temp["v0,c0,v1,v2"] += -1.0 * H1["v0,h0"] * T2["h0,c0,v1,v2"];
    temp["v0,c0,v1,v2"] += (-1.0 / 2.0) * H2["v0,c0,h0,h1"] * T2["h0,h1,v1,v2"];
    temp["v0,c0,v1,v2"] += -1.0 * H2["v1,v2,p0,v0"] * T1["c0,p0"];
    temp["v0,c0,v1,v2"] += 1.0 * H2["v0,c0,h0,h1"] * T2["h0,h2,v1,v2"] * L1["h2,h1"];
    C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
    C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvaa"});
    temp["v0,v1,a0,a1"] += 1.0 * H2["v0,v1,h0,a0"] * T1["h0,a1"];
    C2["v0,v1,a0,a1"] += temp["v0,v1,a0,a1"];
    C2["v0,v1,a1,a0"] -= temp["v0,v1,a0,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvca"});
    temp["v0,v1,c0,a0"] += 1.0 * H2["v0,v1,h0,c0"] * T1["h0,a0"];
    C2["v0,v1,c0,a0"] += temp["v0,v1,c0,a0"];
    C2["v0,v1,a0,c0"] -= temp["v0,v1,c0,a0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvva"});
    temp["v0,v1,v2,a0"] += (-1.0 / 2.0) * H2["v0,v1,h0,h1"] * T2["h0,h1,v2,a0"];
    temp["v0,v1,v2,a0"] += 1.0 * H2["v0,v1,h0,v2"] * T1["h0,a0"];
    temp["v0,v1,v2,a0"] += -1.0 * H2["v0,v1,h0,a0"] * T1["h0,v2"];
    temp["v0,v1,v2,a0"] += 1.0 * H2["v0,v1,h0,h1"] * T2["h0,h2,v2,a0"] * L1["h2,h1"];
    C2["v0,v1,v2,a0"] += temp["v0,v1,v2,a0"];
    C2["v0,v1,a0,v2"] -= temp["v0,v1,v2,a0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
    temp["v0,v1,v2,c0"] += -1.0 * H2["v0,v1,h0,c0"] * T1["h0,v2"];
    C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
    C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvv"});
    temp["v0,v1,v2,v3"] += 1.0 * H2["v0,v1,h0,v2"] * T1["h0,v3"];
    C2["v0,v1,v2,v3"] += temp["v0,v1,v2,v3"];
    C2["v0,v1,v3,v2"] -= temp["v0,v1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});
    temp["a0,a1,a2,a3"] += 1.0 * H2["h0,a2,p0,a0"] * T2["h1,a1,p0,a3"] * L1["h1,h0"];
    temp["a0,a1,a2,a3"] += -1.0 * H2["a0,a4,h0,a2"] * T2["h0,a1,a3,a5"] * L1["a5,a4"];
    C2["a0,a1,a2,a3"] += temp["a0,a1,a2,a3"];
    C2["a0,a1,a3,a2"] -= temp["a0,a1,a2,a3"];
    C2["a1,a0,a2,a3"] -= temp["a0,a1,a2,a3"];
    C2["a1,a0,a3,a2"] += temp["a0,a1,a2,a3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaca"});
    temp["a0,a1,c0,a2"] += -1.0 * H2["c0,a2,p0,a0"] * T1["a1,p0"];
    temp["a0,a1,c0,a2"] += 1.0 * H2["h0,c0,p0,a0"] * T2["h1,a1,p0,a2"] * L1["h1,h0"];
    temp["a0,a1,c0,a2"] += -1.0 * H2["a0,a3,h0,c0"] * T2["h0,a1,a2,a4"] * L1["a4,a3"];
    C2["a0,a1,c0,a2"] += temp["a0,a1,c0,a2"];
    C2["a0,a1,a2,c0"] -= temp["a0,a1,c0,a2"];
    C2["a1,a0,c0,a2"] -= temp["a0,a1,c0,a2"];
    C2["a1,a0,a2,c0"] += temp["a0,a1,c0,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aava"});
    temp["a0,a1,v0,a2"] += -1.0 * H1["a0,h0"] * T2["h0,a1,v0,a2"];
    temp["a0,a1,v0,a2"] += -1.0 * H2["v0,a2,p0,a0"] * T1["a1,p0"];
    temp["a0,a1,v0,a2"] += 1.0 * H2["h0,v0,p0,a0"] * T2["h1,a1,p0,a2"] * L1["h1,h0"];
    temp["a0,a1,v0,a2"] += -1.0 * H2["h0,a2,p0,a0"] * T2["h1,a1,p0,v0"] * L1["h1,h0"];
    temp["a0,a1,v0,a2"] += -1.0 * H2["a0,a3,h0,v0"] * T2["h0,a1,a2,a4"] * L1["a4,a3"];
    temp["a0,a1,v0,a2"] += 1.0 * H2["a0,a3,h0,a2"] * T2["h0,a1,v0,a4"] * L1["a4,a3"];
    C2["a0,a1,v0,a2"] += temp["a0,a1,v0,a2"];
    C2["a0,a1,a2,v0"] -= temp["a0,a1,v0,a2"];
    C2["a1,a0,v0,a2"] -= temp["a0,a1,v0,a2"];
    C2["a1,a0,a2,v0"] += temp["a0,a1,v0,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aavc"});
    temp["a0,a1,v0,c0"] += -1.0 * H2["v0,c0,p0,a0"] * T1["a1,p0"];
    temp["a0,a1,v0,c0"] += -1.0 * H2["h0,c0,p0,a0"] * T2["h1,a1,p0,v0"] * L1["h1,h0"];
    temp["a0,a1,v0,c0"] += 1.0 * H2["a0,a2,h0,c0"] * T2["h0,a1,v0,a3"] * L1["a3,a2"];
    C2["a0,a1,v0,c0"] += temp["a0,a1,v0,c0"];
    C2["a0,a1,c0,v0"] -= temp["a0,a1,v0,c0"];
    C2["a1,a0,v0,c0"] -= temp["a0,a1,v0,c0"];
    C2["a1,a0,c0,v0"] += temp["a0,a1,v0,c0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aavv"});
    temp["a0,a1,v0,v1"] += 1.0 * H2["h0,v0,p0,a0"] * T2["h1,a1,p0,v1"] * L1["h1,h0"];
    temp["a0,a1,v0,v1"] += -1.0 * H2["a0,a2,h0,v0"] * T2["h0,a1,v1,a3"] * L1["a3,a2"];
    C2["a0,a1,v0,v1"] += temp["a0,a1,v0,v1"];
    C2["a0,a1,v1,v0"] -= temp["a0,a1,v0,v1"];
    C2["a1,a0,v0,v1"] -= temp["a0,a1,v0,v1"];
    C2["a1,a0,v1,v0"] += temp["a0,a1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"caaa"});
    temp["c0,a0,a1,a2"] += 1.0 * H1["a1,p0"] * T2["c0,a0,p0,a2"];
    temp["c0,a0,a1,a2"] += 1.0 * H2["c0,a0,h0,a1"] * T1["h0,a2"];
    temp["c0,a0,a1,a2"] += 1.0 * H2["h0,a1,p0,c0"] * T2["h1,a0,p0,a2"] * L1["h1,h0"];
    temp["c0,a0,a1,a2"] += -1.0 * H2["h0,a1,p0,a0"] * T2["h1,c0,p0,a2"] * L1["h1,h0"];
    temp["c0,a0,a1,a2"] += -1.0 * H2["c0,a3,h0,a1"] * T2["h0,a0,a2,a4"] * L1["a4,a3"];
    temp["c0,a0,a1,a2"] += 1.0 * H2["a0,a3,h0,a1"] * T2["h0,c0,a2,a4"] * L1["a4,a3"];
    C2["c0,a0,a1,a2"] += temp["c0,a0,a1,a2"];
    C2["c0,a0,a2,a1"] -= temp["c0,a0,a1,a2"];
    C2["a0,c0,a1,a2"] -= temp["c0,a0,a1,a2"];
    C2["a0,c0,a2,a1"] += temp["c0,a0,a1,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"caca"});
    temp["c0,a0,c1,a1"] += 1.0 * H1["c1,p0"] * T2["c0,a0,p0,a1"];
    temp["c0,a0,c1,a1"] += 1.0 * H2["c0,a0,h0,c1"] * T1["h0,a1"];
    temp["c0,a0,c1,a1"] += (1.0 / 2.0) * H2["c1,a1,p0,p1"] * T2["c0,a0,p0,p1"];
    temp["c0,a0,c1,a1"] += -1.0 * H2["c1,a1,p0,c0"] * T1["a0,p0"];
    temp["c0,a0,c1,a1"] += 1.0 * H2["c1,a1,p0,a0"] * T1["c0,p0"];
    temp["c0,a0,c1,a1"] += 1.0 * H2["h0,c1,p0,c0"] * T2["h1,a0,p0,a1"] * L1["h1,h0"];
    temp["c0,a0,c1,a1"] += -1.0 * H2["h0,c1,p0,a0"] * T2["h1,c0,p0,a1"] * L1["h1,h0"];
    temp["c0,a0,c1,a1"] += -1.0 * H2["c0,a2,h0,c1"] * T2["h0,a0,a1,a3"] * L1["a3,a2"];
    temp["c0,a0,c1,a1"] += -1.0 * H2["c1,a1,p0,a2"] * T2["c0,a0,p0,a3"] * L1["a3,a2"];
    temp["c0,a0,c1,a1"] += 1.0 * H2["a0,a2,h0,c1"] * T2["h0,c0,a1,a3"] * L1["a3,a2"];
    C2["c0,a0,c1,a1"] += temp["c0,a0,c1,a1"];
    C2["c0,a0,a1,c1"] -= temp["c0,a0,c1,a1"];
    C2["a0,c0,c1,a1"] -= temp["c0,a0,c1,a1"];
    C2["a0,c0,a1,c1"] += temp["c0,a0,c1,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cava"});
    temp["c0,a0,v0,a1"] += 1.0 * H1["v0,p0"] * T2["c0,a0,p0,a1"];
    temp["c0,a0,v0,a1"] += -1.0 * H1["c0,h0"] * T2["h0,a0,v0,a1"];
    temp["c0,a0,v0,a1"] += 1.0 * H1["a0,h0"] * T2["h0,c0,v0,a1"];
    temp["c0,a0,v0,a1"] += -1.0 * H1["a1,p0"] * T2["c0,a0,p0,v0"];
    temp["c0,a0,v0,a1"] += (1.0 / 2.0) * H2["v0,a1,p0,p1"] * T2["c0,a0,p0,p1"];
    temp["c0,a0,v0,a1"] += -1.0 * H2["v0,a1,p0,c0"] * T1["a0,p0"];
    temp["c0,a0,v0,a1"] += 1.0 * H2["v0,a1,p0,a0"] * T1["c0,p0"];
    temp["c0,a0,v0,a1"] += (-1.0 / 2.0) * H2["c0,a0,h0,h1"] * T2["h0,h1,v0,a1"];
    temp["c0,a0,v0,a1"] += 1.0 * H2["c0,a0,h0,v0"] * T1["h0,a1"];
    temp["c0,a0,v0,a1"] += -1.0 * H2["c0,a0,h0,a1"] * T1["h0,v0"];
    temp["c0,a0,v0,a1"] += 1.0 * H2["h0,v0,p0,c0"] * T2["h1,a0,p0,a1"] * L1["h1,h0"];
    temp["c0,a0,v0,a1"] += -1.0 * H2["h0,v0,p0,a0"] * T2["h1,c0,p0,a1"] * L1["h1,h0"];
    temp["c0,a0,v0,a1"] += -1.0 * H2["h0,a1,p0,c0"] * T2["h1,a0,p0,v0"] * L1["h1,h0"];
    temp["c0,a0,v0,a1"] += 1.0 * H2["h0,a1,p0,a0"] * T2["h1,c0,p0,v0"] * L1["h1,h0"];
    temp["c0,a0,v0,a1"] += -1.0 * H2["v0,a1,p0,a2"] * T2["c0,a0,p0,a3"] * L1["a3,a2"];
    temp["c0,a0,v0,a1"] += 1.0 * H2["c0,a0,h0,h1"] * T2["h0,h2,v0,a1"] * L1["h2,h1"];
    temp["c0,a0,v0,a1"] += -1.0 * H2["c0,a2,h0,v0"] * T2["h0,a0,a1,a3"] * L1["a3,a2"];
    temp["c0,a0,v0,a1"] += 1.0 * H2["c0,a2,h0,a1"] * T2["h0,a0,v0,a3"] * L1["a3,a2"];
    temp["c0,a0,v0,a1"] += 1.0 * H2["a0,a2,h0,v0"] * T2["h0,c0,a1,a3"] * L1["a3,a2"];
    temp["c0,a0,v0,a1"] += -1.0 * H2["a0,a2,h0,a1"] * T2["h0,c0,v0,a3"] * L1["a3,a2"];
    C2["c0,a0,v0,a1"] += temp["c0,a0,v0,a1"];
    C2["c0,a0,a1,v0"] -= temp["c0,a0,v0,a1"];
    C2["a0,c0,v0,a1"] -= temp["c0,a0,v0,a1"];
    C2["a0,c0,a1,v0"] += temp["c0,a0,v0,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cavc"});
    temp["c0,a0,v0,c1"] += -1.0 * H1["c1,p0"] * T2["c0,a0,p0,v0"];
    temp["c0,a0,v0,c1"] += (1.0 / 2.0) * H2["v0,c1,p0,p1"] * T2["c0,a0,p0,p1"];
    temp["c0,a0,v0,c1"] += -1.0 * H2["v0,c1,p0,c0"] * T1["a0,p0"];
    temp["c0,a0,v0,c1"] += 1.0 * H2["v0,c1,p0,a0"] * T1["c0,p0"];
    temp["c0,a0,v0,c1"] += -1.0 * H2["c0,a0,h0,c1"] * T1["h0,v0"];
    temp["c0,a0,v0,c1"] += -1.0 * H2["h0,c1,p0,c0"] * T2["h1,a0,p0,v0"] * L1["h1,h0"];
    temp["c0,a0,v0,c1"] += 1.0 * H2["h0,c1,p0,a0"] * T2["h1,c0,p0,v0"] * L1["h1,h0"];
    temp["c0,a0,v0,c1"] += -1.0 * H2["v0,c1,p0,a1"] * T2["c0,a0,p0,a2"] * L1["a2,a1"];
    temp["c0,a0,v0,c1"] += 1.0 * H2["c0,a1,h0,c1"] * T2["h0,a0,v0,a2"] * L1["a2,a1"];
    temp["c0,a0,v0,c1"] += -1.0 * H2["a0,a1,h0,c1"] * T2["h0,c0,v0,a2"] * L1["a2,a1"];
    C2["c0,a0,v0,c1"] += temp["c0,a0,v0,c1"];
    C2["c0,a0,c1,v0"] -= temp["c0,a0,v0,c1"];
    C2["a0,c0,v0,c1"] -= temp["c0,a0,v0,c1"];
    C2["a0,c0,c1,v0"] += temp["c0,a0,v0,c1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cavv"});
    temp["c0,a0,v0,v1"] += 1.0 * H1["v0,p0"] * T2["c0,a0,p0,v1"];
    temp["c0,a0,v0,v1"] += 1.0 * H2["c0,a0,h0,v0"] * T1["h0,v1"];
    temp["c0,a0,v0,v1"] += 1.0 * H2["h0,v0,p0,c0"] * T2["h1,a0,p0,v1"] * L1["h1,h0"];
    temp["c0,a0,v0,v1"] += -1.0 * H2["h0,v0,p0,a0"] * T2["h1,c0,p0,v1"] * L1["h1,h0"];
    temp["c0,a0,v0,v1"] += -1.0 * H2["c0,a1,h0,v0"] * T2["h0,a0,v1,a2"] * L1["a2,a1"];
    temp["c0,a0,v0,v1"] += 1.0 * H2["a0,a1,h0,v0"] * T2["h0,c0,v1,a2"] * L1["a2,a1"];
    C2["c0,a0,v0,v1"] += temp["c0,a0,v0,v1"];
    C2["c0,a0,v1,v0"] -= temp["c0,a0,v0,v1"];
    C2["a0,c0,v0,v1"] -= temp["c0,a0,v0,v1"];
    C2["a0,c0,v1,v0"] += temp["c0,a0,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccaa"});
    temp["c0,c1,a0,a1"] += 1.0 * H2["h0,a0,p0,c0"] * T2["h1,c1,p0,a1"] * L1["h1,h0"];
    temp["c0,c1,a0,a1"] += -1.0 * H2["c0,a2,h0,a0"] * T2["h0,c1,a1,a3"] * L1["a3,a2"];
    C2["c0,c1,a0,a1"] += temp["c0,c1,a0,a1"];
    C2["c0,c1,a1,a0"] -= temp["c0,c1,a0,a1"];
    C2["c1,c0,a0,a1"] -= temp["c0,c1,a0,a1"];
    C2["c1,c0,a1,a0"] += temp["c0,c1,a0,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccca"});
    temp["c0,c1,c2,a0"] += -1.0 * H2["c2,a0,p0,c0"] * T1["c1,p0"];
    temp["c0,c1,c2,a0"] += 1.0 * H2["h0,c2,p0,c0"] * T2["h1,c1,p0,a0"] * L1["h1,h0"];
    temp["c0,c1,c2,a0"] += -1.0 * H2["c0,a1,h0,c2"] * T2["h0,c1,a0,a2"] * L1["a2,a1"];
    C2["c0,c1,c2,a0"] += temp["c0,c1,c2,a0"];
    C2["c0,c1,a0,c2"] -= temp["c0,c1,c2,a0"];
    C2["c1,c0,c2,a0"] -= temp["c0,c1,c2,a0"];
    C2["c1,c0,a0,c2"] += temp["c0,c1,c2,a0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccva"});
    temp["c0,c1,v0,a0"] += -1.0 * H1["c0,h0"] * T2["h0,c1,v0,a0"];
    temp["c0,c1,v0,a0"] += -1.0 * H2["v0,a0,p0,c0"] * T1["c1,p0"];
    temp["c0,c1,v0,a0"] += 1.0 * H2["h0,v0,p0,c0"] * T2["h1,c1,p0,a0"] * L1["h1,h0"];
    temp["c0,c1,v0,a0"] += -1.0 * H2["h0,a0,p0,c0"] * T2["h1,c1,p0,v0"] * L1["h1,h0"];
    temp["c0,c1,v0,a0"] += -1.0 * H2["c0,a1,h0,v0"] * T2["h0,c1,a0,a2"] * L1["a2,a1"];
    temp["c0,c1,v0,a0"] += 1.0 * H2["c0,a1,h0,a0"] * T2["h0,c1,v0,a2"] * L1["a2,a1"];
    C2["c0,c1,v0,a0"] += temp["c0,c1,v0,a0"];
    C2["c0,c1,a0,v0"] -= temp["c0,c1,v0,a0"];
    C2["c1,c0,v0,a0"] -= temp["c0,c1,v0,a0"];
    C2["c1,c0,a0,v0"] += temp["c0,c1,v0,a0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
    temp["c0,c1,v0,c2"] += -1.0 * H2["v0,c2,p0,c0"] * T1["c1,p0"];
    temp["c0,c1,v0,c2"] += -1.0 * H2["h0,c2,p0,c0"] * T2["h1,c1,p0,v0"] * L1["h1,h0"];
    temp["c0,c1,v0,c2"] += 1.0 * H2["c0,a0,h0,c2"] * T2["h0,c1,v0,a1"] * L1["a1,a0"];
    C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
    C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];
    C2["c1,c0,v0,c2"] -= temp["c0,c1,v0,c2"];
    C2["c1,c0,c2,v0"] += temp["c0,c1,v0,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H2["h0,v0,p0,c0"] * T2["h1,c1,p0,v1"] * L1["h1,h0"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,a0,h0,v0"] * T2["h0,c1,v1,a1"] * L1["a1,a0"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vaaa"});
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a0,h0,a1"] * T1["h0,a2"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["h0,a1,p0,v0"] * T2["h1,a0,p0,a2"] * L1["h1,h0"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,a3,h0,a1"] * T2["h0,a0,a2,a4"] * L1["a4,a3"];
    C2["v0,a0,a1,a2"] += temp["v0,a0,a1,a2"];
    C2["v0,a0,a2,a1"] -= temp["v0,a0,a1,a2"];
    C2["a0,v0,a1,a2"] -= temp["v0,a0,a1,a2"];
    C2["a0,v0,a2,a1"] += temp["v0,a0,a1,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vaca"});
    temp["v0,a0,c0,a1"] += 1.0 * H2["v0,a0,h0,c0"] * T1["h0,a1"];
    temp["v0,a0,c0,a1"] += -1.0 * H2["c0,a1,p0,v0"] * T1["a0,p0"];
    temp["v0,a0,c0,a1"] += 1.0 * H2["h0,c0,p0,v0"] * T2["h1,a0,p0,a1"] * L1["h1,h0"];
    temp["v0,a0,c0,a1"] += -1.0 * H2["v0,a2,h0,c0"] * T2["h0,a0,a1,a3"] * L1["a3,a2"];
    C2["v0,a0,c0,a1"] += temp["v0,a0,c0,a1"];
    C2["v0,a0,a1,c0"] -= temp["v0,a0,c0,a1"];
    C2["a0,v0,c0,a1"] -= temp["v0,a0,c0,a1"];
    C2["a0,v0,a1,c0"] += temp["v0,a0,c0,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vava"});
    temp["v0,a0,v1,a1"] += -1.0 * H1["v0,h0"] * T2["h0,a0,v1,a1"];
    temp["v0,a0,v1,a1"] += (-1.0 / 2.0) * H2["v0,a0,h0,h1"] * T2["h0,h1,v1,a1"];
    temp["v0,a0,v1,a1"] += 1.0 * H2["v0,a0,h0,v1"] * T1["h0,a1"];
    temp["v0,a0,v1,a1"] += -1.0 * H2["v0,a0,h0,a1"] * T1["h0,v1"];
    temp["v0,a0,v1,a1"] += -1.0 * H2["v1,a1,p0,v0"] * T1["a0,p0"];
    temp["v0,a0,v1,a1"] += 1.0 * H2["h0,v1,p0,v0"] * T2["h1,a0,p0,a1"] * L1["h1,h0"];
    temp["v0,a0,v1,a1"] += -1.0 * H2["h0,a1,p0,v0"] * T2["h1,a0,p0,v1"] * L1["h1,h0"];
    temp["v0,a0,v1,a1"] += 1.0 * H2["v0,a0,h0,h1"] * T2["h0,h2,v1,a1"] * L1["h2,h1"];
    temp["v0,a0,v1,a1"] += -1.0 * H2["v0,a2,h0,v1"] * T2["h0,a0,a1,a3"] * L1["a3,a2"];
    temp["v0,a0,v1,a1"] += 1.0 * H2["v0,a2,h0,a1"] * T2["h0,a0,v1,a3"] * L1["a3,a2"];
    C2["v0,a0,v1,a1"] += temp["v0,a0,v1,a1"];
    C2["v0,a0,a1,v1"] -= temp["v0,a0,v1,a1"];
    C2["a0,v0,v1,a1"] -= temp["v0,a0,v1,a1"];
    C2["a0,v0,a1,v1"] += temp["v0,a0,v1,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vavc"});
    temp["v0,a0,v1,c0"] += -1.0 * H2["v0,a0,h0,c0"] * T1["h0,v1"];
    temp["v0,a0,v1,c0"] += -1.0 * H2["v1,c0,p0,v0"] * T1["a0,p0"];
    temp["v0,a0,v1,c0"] += -1.0 * H2["h0,c0,p0,v0"] * T2["h1,a0,p0,v1"] * L1["h1,h0"];
    temp["v0,a0,v1,c0"] += 1.0 * H2["v0,a1,h0,c0"] * T2["h0,a0,v1,a2"] * L1["a2,a1"];
    C2["v0,a0,v1,c0"] += temp["v0,a0,v1,c0"];
    C2["v0,a0,c0,v1"] -= temp["v0,a0,v1,c0"];
    C2["a0,v0,v1,c0"] -= temp["v0,a0,v1,c0"];
    C2["a0,v0,c0,v1"] += temp["v0,a0,v1,c0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vavv"});
    temp["v0,a0,v1,v2"] += 1.0 * H2["v0,a0,h0,v1"] * T1["h0,v2"];
    temp["v0,a0,v1,v2"] += 1.0 * H2["h0,v1,p0,v0"] * T2["h1,a0,p0,v2"] * L1["h1,h0"];
    temp["v0,a0,v1,v2"] += -1.0 * H2["v0,a1,h0,v1"] * T2["h0,a0,v2,a2"] * L1["a2,a1"];
    C2["v0,a0,v1,v2"] += temp["v0,a0,v1,v2"];
    C2["v0,a0,v2,v1"] -= temp["v0,a0,v1,v2"];
    C2["a0,v0,v1,v2"] -= temp["v0,a0,v1,v2"];
    C2["a0,v0,v2,v1"] += temp["v0,a0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcaa"});
    temp["v0,c0,a0,a1"] += 1.0 * H2["v0,c0,h0,a0"] * T1["h0,a1"];
    temp["v0,c0,a0,a1"] += 1.0 * H2["h0,a0,p0,v0"] * T2["h1,c0,p0,a1"] * L1["h1,h0"];
    temp["v0,c0,a0,a1"] += -1.0 * H2["v0,a2,h0,a0"] * T2["h0,c0,a1,a3"] * L1["a3,a2"];
    C2["v0,c0,a0,a1"] += temp["v0,c0,a0,a1"];
    C2["v0,c0,a1,a0"] -= temp["v0,c0,a0,a1"];
    C2["c0,v0,a0,a1"] -= temp["v0,c0,a0,a1"];
    C2["c0,v0,a1,a0"] += temp["v0,c0,a0,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcca"});
    temp["v0,c0,c1,a0"] += 1.0 * H2["v0,c0,h0,c1"] * T1["h0,a0"];
    temp["v0,c0,c1,a0"] += -1.0 * H2["c1,a0,p0,v0"] * T1["c0,p0"];
    temp["v0,c0,c1,a0"] += 1.0 * H2["h0,c1,p0,v0"] * T2["h1,c0,p0,a0"] * L1["h1,h0"];
    temp["v0,c0,c1,a0"] += -1.0 * H2["v0,a1,h0,c1"] * T2["h0,c0,a0,a2"] * L1["a2,a1"];
    C2["v0,c0,c1,a0"] += temp["v0,c0,c1,a0"];
    C2["v0,c0,a0,c1"] -= temp["v0,c0,c1,a0"];
    C2["c0,v0,c1,a0"] -= temp["v0,c0,c1,a0"];
    C2["c0,v0,a0,c1"] += temp["v0,c0,c1,a0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcva"});
    temp["v0,c0,v1,a0"] += -1.0 * H1["v0,h0"] * T2["h0,c0,v1,a0"];
    temp["v0,c0,v1,a0"] += (-1.0 / 2.0) * H2["v0,c0,h0,h1"] * T2["h0,h1,v1,a0"];
    temp["v0,c0,v1,a0"] += 1.0 * H2["v0,c0,h0,v1"] * T1["h0,a0"];
    temp["v0,c0,v1,a0"] += -1.0 * H2["v0,c0,h0,a0"] * T1["h0,v1"];
    temp["v0,c0,v1,a0"] += -1.0 * H2["v1,a0,p0,v0"] * T1["c0,p0"];
    temp["v0,c0,v1,a0"] += 1.0 * H2["h0,v1,p0,v0"] * T2["h1,c0,p0,a0"] * L1["h1,h0"];
    temp["v0,c0,v1,a0"] += -1.0 * H2["h0,a0,p0,v0"] * T2["h1,c0,p0,v1"] * L1["h1,h0"];
    temp["v0,c0,v1,a0"] += 1.0 * H2["v0,c0,h0,h1"] * T2["h0,h2,v1,a0"] * L1["h2,h1"];
    temp["v0,c0,v1,a0"] += -1.0 * H2["v0,a1,h0,v1"] * T2["h0,c0,a0,a2"] * L1["a2,a1"];
    temp["v0,c0,v1,a0"] += 1.0 * H2["v0,a1,h0,a0"] * T2["h0,c0,v1,a2"] * L1["a2,a1"];
    C2["v0,c0,v1,a0"] += temp["v0,c0,v1,a0"];
    C2["v0,c0,a0,v1"] -= temp["v0,c0,v1,a0"];
    C2["c0,v0,v1,a0"] -= temp["v0,c0,v1,a0"];
    C2["c0,v0,a0,v1"] += temp["v0,c0,v1,a0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
    temp["v0,c0,v1,c1"] += -1.0 * H2["v0,c0,h0,c1"] * T1["h0,v1"];
    temp["v0,c0,v1,c1"] += -1.0 * H2["v1,c1,p0,v0"] * T1["c0,p0"];
    temp["v0,c0,v1,c1"] += -1.0 * H2["h0,c1,p0,v0"] * T2["h1,c0,p0,v1"] * L1["h1,h0"];
    temp["v0,c0,v1,c1"] += 1.0 * H2["v0,a0,h0,c1"] * T2["h0,c0,v1,a1"] * L1["a1,a0"];
    C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
    C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
    temp["v0,c0,v1,v2"] += 1.0 * H2["v0,c0,h0,v1"] * T1["h0,v2"];
    temp["v0,c0,v1,v2"] += 1.0 * H2["h0,v1,p0,v0"] * T2["h1,c0,p0,v2"] * L1["h1,h0"];
    temp["v0,c0,v1,v2"] += -1.0 * H2["v0,a0,h0,v1"] * T2["h0,c0,v2,a1"] * L1["a1,a0"];
    C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
    C2["v0,c0,v2,v1"] -= temp["v0,c0,v1,v2"];
    C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];
    C2["c0,v0,v2,v1"] += temp["v0,c0,v1,v2"];

    // scale by factor
    C0 *= factor;
    C1.scale(factor);
    C2.scale(factor);

    // LDSRG(2*)
    if (foptions_->get_str("CORR_LEVEL") == "LDSRG2*") {
        if (n == 2) {
            sr_ldsrg2star_comm2(C1, C2);
        } else if (n == 3) {
            sr_ldsrg2star_comm3(C1, C2);
        }
    }

    // LDSRG(2+)
    if (foptions_->get_str("CORR_LEVEL") == "LDSRG2+") {
        C1["pq"] += W1["pq"];
        C2["pqrs"] += W2["pqrs"];

        double f = factor / (n + 1.0);
        sr_ldsrg2plus(f, H2, T1, T2, W1, W2);
    }

    // add T dagger
    C0 *= 2.0;
    H1["pq"] = C1["pq"];
    C1["pq"] += H1["qp"];
    H2["pqrs"] = C2["pqrs"];
    C2["pqrs"] += H2["rspq"];
}

void MRDSRG_SO::comm_H_A_3(double factor, BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& H3,
                           BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& T3, double& C0,
                           BlockedTensor& C1, BlockedTensor& C2, BlockedTensor& C3) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    C3.zero();
    BlockedTensor temp;

    C0 += 1.0 * H1["h0,p0"] * T1["h1,p0"] * L1["h1,h0"];
    C0 += (1.0 / 2.0) * H1["a0,p0"] * T2["a2,a3,p0,a1"] * L2["a2,a3,a0,a1"];
    C0 += (1.0 / 12.0) * H1["a0,p0"] * T3["a3,a4,a5,p0,a1,a2"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += -1.0 * H1["a0,h0"] * T1["h0,a1"] * L1["a1,a0"];
    C0 += (-1.0 / 2.0) * H1["a0,h0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 12.0) * H1["a0,h0"] * T3["h0,a1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (1.0 / 8.0) * H2["a0,a1,p0,p1"] * T2["a2,a3,p0,p1"] * L2["a2,a3,a0,a1"];
    C0 += (1.0 / 24.0) * H2["a0,a1,p0,p1"] * T3["a3,a4,a5,p0,p1,a2"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 8.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 24.0) * H2["a0,a1,h0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 2.0) * H2["a1,a2,p0,a0"] * T1["a3,p0"] * L2["a1,a2,a0,a3"];
    C0 += (-1.0 / 4.0) * H2["a1,a2,p0,a0"] * T2["a4,a5,p0,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    C0 += (1.0 / 2.0) * H2["a1,a2,h0,a0"] * T1["h0,a3"] * L2["a1,a2,a0,a3"];
    C0 += (1.0 / 4.0) * H2["a1,a2,h0,a0"] * T2["h0,a3,a4,a5"] * L3["a1,a2,a3,a0,a4,a5"];
    C0 +=
        (1.0 / 216.0) * H3["a0,a1,a2,p0,p1,p2"] * T3["a3,a4,a5,p0,p1,p2"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 216.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] *
          L3["a3,a4,a5,a0,a1,a2"];
    C0 += (1.0 / 24.0) * H3["a1,a2,a3,p0,p1,a0"] * T2["a4,a5,p0,p1"] * L3["a1,a2,a3,a0,a4,a5"];
    C0 += (-1.0 / 24.0) * H3["a1,a2,a3,h0,h1,a0"] * T2["h0,h1,a4,a5"] * L3["a1,a2,a3,a0,a4,a5"];
    C0 += (1.0 / 12.0) * H3["a2,a3,a4,p0,a0,a1"] * T1["a5,p0"] * L3["a2,a3,a4,a0,a1,a5"];
    C0 += (-1.0 / 12.0) * H3["a2,a3,a4,h0,a0,a1"] * T1["h0,a5"] * L3["a2,a3,a4,a0,a1,a5"];
    C0 += (1.0 / 4.0) * H2["h0,h1,p0,p1"] * T2["h2,h3,p0,p1"] * L1["h2,h0"] * L1["h3,h1"];
    C0 +=
        (1.0 / 4.0) * H2["h0,a0,p0,p1"] * T3["h1,a2,a3,p0,p1,a1"] * L1["h1,h0"] * L2["a2,a3,a0,a1"];
    C0 += 1.0 * H2["h0,a1,p0,a0"] * T2["h1,a3,p0,a2"] * L1["h1,h0"] * L2["a1,a2,a0,a3"];
    C0 += (1.0 / 4.0) * H2["h0,a1,p0,a0"] * T3["h1,a4,a5,p0,a2,a3"] * L1["h1,h0"] *
          L3["a1,a2,a3,a0,a4,a5"];
    C0 += (-1.0 / 4.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C0 += (1.0 / 4.0) * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h2,h1"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 4.0) * H2["a0,a1,h0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L1["a3,a0"] *
          L2["a4,a5,a1,a2"];
    C0 += (1.0 / 12.0) * H2["a0,a1,h0,h1"] * T3["h0,h2,a2,a3,a4,a5"] * L1["h2,h1"] *
          L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 4.0) * H2["a1,a2,p0,a0"] * T2["a4,a5,p0,a3"] * L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C0 += (-1.0 / 12.0) * H2["a1,a2,p0,a0"] * T3["a5,a6,a7,p0,a3,a4"] * L1["a3,a0"] *
          L3["a5,a6,a7,a1,a2,a4"];
    C0 += (1.0 / 2.0) * H2["a1,a2,p0,a0"] * T3["a5,a6,a7,p0,a3,a4"] * L2["a1,a3,a0,a5"] *
          L2["a6,a7,a2,a4"];
    C0 += (-1.0 / 8.0) * H2["a1,a2,p0,a0"] * T3["a5,a6,a7,p0,a3,a4"] * L2["a3,a4,a0,a5"] *
          L2["a6,a7,a1,a2"];
    C0 += -1.0 * H2["a1,a2,h0,a0"] * T2["h0,a3,a4,a5"] * L1["a4,a1"] * L2["a2,a3,a0,a5"];
    C0 += (-1.0 / 4.0) * H2["a1,a2,h0,a0"] * T3["h0,a3,a4,a5,a6,a7"] * L1["a5,a1"] *
          L3["a2,a3,a4,a0,a6,a7"];
    C0 += (-1.0 / 2.0) * H2["a1,a2,h0,a0"] * T3["h0,a3,a4,a5,a6,a7"] * L2["a1,a3,a0,a5"] *
          L2["a6,a7,a2,a4"];
    C0 += (1.0 / 8.0) * H2["a1,a2,h0,a0"] * T3["h0,a3,a4,a5,a6,a7"] * L2["a3,a4,a0,a5"] *
          L2["a6,a7,a1,a2"];
    C0 += (1.0 / 24.0) * H3["h0,a0,a1,p0,p1,p2"] * T3["h1,a2,a3,p0,p1,p2"] * L1["h1,h0"] *
          L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 4.0) * H3["h0,a1,a2,p0,p1,a0"] * T2["h1,a3,p0,p1"] * L1["h1,h0"] *
          L2["a1,a2,a0,a3"];
    C0 += (-1.0 / 8.0) * H3["h0,a1,a2,p0,p1,a0"] * T3["h1,a4,a5,p0,p1,a3"] * L1["h1,h0"] *
          L3["a1,a2,a3,a0,a4,a5"];
    C0 += (1.0 / 4.0) * H3["h0,a2,a3,p0,a0,a1"] * T2["h1,a5,p0,a4"] * L1["h1,h0"] *
          L3["a2,a3,a4,a0,a1,a5"];
    C0 += (-1.0 / 24.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L1["a3,a0"] *
          L2["a4,a5,a1,a2"];
    C0 += (1.0 / 72.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h3,h2"] *
          L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 72.0) * H3["a1,a2,a3,p0,p1,a0"] * T3["a5,a6,a7,p0,p1,a4"] * L1["a4,a0"] *
          L3["a5,a6,a7,a1,a2,a3"];
    C0 += (1.0 / 8.0) * H3["a1,a2,a3,p0,p1,a0"] * T3["a5,a6,a7,p0,p1,a4"] * L2["a1,a2,a0,a5"] *
          L2["a6,a7,a3,a4"];
    C0 += (1.0 / 8.0) * H3["a1,a2,a3,p0,p1,a0"] * T3["a5,a6,a7,p0,p1,a4"] * L2["a1,a4,a0,a5"] *
          L2["a6,a7,a2,a3"];
    C0 +=
        (1.0 / 4.0) * H3["a1,a2,a3,h0,h1,a0"] * T2["h0,h1,a4,a5"] * L1["a4,a1"] * L2["a2,a3,a0,a5"];
    C0 += (1.0 / 12.0) * H3["a1,a2,a3,h0,h1,a0"] * T2["h0,h2,a4,a5"] * L1["h2,h1"] *
          L3["a1,a2,a3,a0,a4,a5"];
    C0 += (1.0 / 8.0) * H3["a1,a2,a3,h0,h1,a0"] * T3["h0,h1,a4,a5,a6,a7"] * L1["a5,a1"] *
          L3["a2,a3,a4,a0,a6,a7"];
    C0 += (-1.0 / 8.0) * H3["a1,a2,a3,h0,h1,a0"] * T3["h0,h1,a4,a5,a6,a7"] * L2["a1,a2,a0,a5"] *
          L2["a6,a7,a3,a4"];
    C0 += (-1.0 / 8.0) * H3["a1,a2,a3,h0,h1,a0"] * T3["h0,h1,a4,a5,a6,a7"] * L2["a1,a4,a0,a5"] *
          L2["a6,a7,a2,a3"];
    C0 += (-1.0 / 12.0) * H3["a2,a3,a4,p0,a0,a1"] * T2["a6,a7,p0,a5"] * L1["a5,a0"] *
          L3["a2,a3,a4,a1,a6,a7"];
    C0 += (-1.0 / 2.0) * H3["a2,a3,a4,p0,a0,a1"] * T2["a6,a7,p0,a5"] * L2["a2,a3,a0,a6"] *
          L2["a4,a5,a1,a7"];
    C0 += (1.0 / 8.0) * H3["a2,a3,a4,p0,a0,a1"] * T2["a6,a7,p0,a5"] * L2["a2,a5,a0,a1"] *
          L2["a6,a7,a3,a4"];
    C0 += (-1.0 / 8.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L2["a2,a3,a0,a7"] *
          L3["a4,a5,a6,a1,a8,a9"];
    C0 += (1.0 / 24.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L2["a2,a5,a0,a1"] *
          L3["a7,a8,a9,a3,a4,a6"];
    C0 += (-1.0 / 4.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L2["a2,a5,a0,a7"] *
          L3["a3,a4,a6,a1,a8,a9"];
    C0 += (1.0 / 144.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L2["a5,a6,a0,a1"] *
          L3["a7,a8,a9,a2,a3,a4"];
    C0 += (-1.0 / 24.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L2["a5,a6,a0,a7"] *
          L3["a2,a3,a4,a1,a8,a9"];
    C0 += (1.0 / 16.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L2["a7,a8,a2,a3"] *
          L3["a4,a5,a6,a0,a1,a9"];
    C0 += (1.0 / 8.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L2["a7,a8,a2,a5"] *
          L3["a3,a4,a6,a0,a1,a9"];
    C0 += (-1.0 / 4.0) * H3["a2,a3,a4,h0,a0,a1"] * T2["h0,a5,a6,a7"] * L1["a6,a2"] *
          L3["a3,a4,a5,a0,a1,a7"];
    C0 += (1.0 / 2.0) * H3["a2,a3,a4,h0,a0,a1"] * T2["h0,a5,a6,a7"] * L2["a2,a3,a0,a6"] *
          L2["a4,a5,a1,a7"];
    C0 += (-1.0 / 8.0) * H3["a2,a3,a4,h0,a0,a1"] * T2["h0,a5,a6,a7"] * L2["a2,a5,a0,a1"] *
          L2["a6,a7,a3,a4"];
    C0 += (1.0 / 8.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a2,a3,a0,a7"] *
          L3["a4,a5,a6,a1,a8,a9"];
    C0 += (-1.0 / 24.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a2,a5,a0,a1"] *
          L3["a7,a8,a9,a3,a4,a6"];
    C0 += (1.0 / 4.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a2,a5,a0,a7"] *
          L3["a3,a4,a6,a1,a8,a9"];
    C0 += (-1.0 / 144.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a5,a6,a0,a1"] *
          L3["a7,a8,a9,a2,a3,a4"];
    C0 += (1.0 / 24.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a5,a6,a0,a7"] *
          L3["a2,a3,a4,a1,a8,a9"];
    C0 += (-1.0 / 16.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a7,a8,a2,a3"] *
          L3["a4,a5,a6,a0,a1,a9"];
    C0 += (-1.0 / 8.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a7,a8,a2,a5"] *
          L3["a3,a4,a6,a0,a1,a9"];
    C0 += (-1.0 / 2.0) * H2["h0,h1,p0,a0"] * T2["h2,h3,p0,a1"] * L1["h2,h0"] * L1["h3,h1"] *
          L1["a1,a0"];
    C0 += (-1.0 / 4.0) * H2["h0,h1,p0,a0"] * T3["h2,h3,a3,p0,a1,a2"] * L1["h2,h0"] * L1["h3,h1"] *
          L2["a1,a2,a0,a3"];
    C0 += (-1.0 / 2.0) * H2["h0,a1,p0,a0"] * T3["h1,a4,a5,p0,a2,a3"] * L1["h1,h0"] * L1["a2,a0"] *
          L2["a4,a5,a1,a3"];
    C0 += (1.0 / 2.0) * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h2,h1"] * L1["a2,a0"] *
          L1["a3,a1"];
    C0 += (1.0 / 2.0) * H2["a0,a1,h0,h1"] * T3["h0,h2,a2,a3,a4,a5"] * L1["h2,h1"] * L1["a3,a0"] *
          L2["a4,a5,a1,a2"];
    C0 += (1.0 / 4.0) * H2["a1,a2,h0,a0"] * T3["h0,a3,a4,a5,a6,a7"] * L1["a5,a1"] * L1["a6,a2"] *
          L2["a3,a4,a0,a7"];
    C0 += (1.0 / 36.0) * H3["h0,h1,h2,p0,p1,p2"] * T3["h3,h4,h5,p0,p1,p2"] * L1["h3,h0"] *
          L1["h4,h1"] * L1["h5,h2"];
    C0 += (1.0 / 4.0) * H3["h0,h1,a1,p0,p1,a0"] * T3["h2,h3,a3,p0,p1,a2"] * L1["h2,h0"] *
          L1["h3,h1"] * L2["a1,a2,a0,a3"];
    C0 += (1.0 / 4.0) * H3["h0,h1,a2,p0,a0,a1"] * T2["h2,h3,p0,a3"] * L1["h2,h0"] * L1["h3,h1"] *
          L2["a2,a3,a0,a1"];
    C0 += (1.0 / 8.0) * H3["h0,h1,a2,p0,a0,a1"] * T3["h2,h3,a5,p0,a3,a4"] * L1["h2,h0"] *
          L1["h3,h1"] * L3["a2,a3,a4,a0,a1,a5"];
    C0 += (-1.0 / 8.0) * H3["h0,a1,a2,p0,p1,a0"] * T3["h1,a4,a5,p0,p1,a3"] * L1["h1,h0"] *
          L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C0 += (1.0 / 2.0) * H3["h0,a2,a3,p0,a0,a1"] * T2["h1,a5,p0,a4"] * L1["h1,h0"] * L1["a4,a0"] *
          L2["a2,a3,a1,a5"];
    C0 += (1.0 / 4.0) * H3["h0,a2,a3,p0,a0,a1"] * T3["h1,a6,a7,p0,a4,a5"] * L1["h1,h0"] *
          L1["a4,a0"] * L3["a2,a3,a5,a1,a6,a7"];
    C0 += (-1.0 / 4.0) * H3["h0,a2,a3,p0,a0,a1"] * T3["h1,a6,a7,p0,a4,a5"] * L1["h1,h0"] *
          L2["a2,a3,a0,a6"] * L2["a4,a5,a1,a7"];
    C0 += (-1.0 / 4.0) * H3["h0,a2,a3,p0,a0,a1"] * T3["h1,a6,a7,p0,a4,a5"] * L1["h1,h0"] *
          L2["a2,a4,a0,a1"] * L2["a6,a7,a3,a5"];
    C0 += (1.0 / 2.0) * H3["h0,a2,a3,p0,a0,a1"] * T3["h1,a6,a7,p0,a4,a5"] * L1["h1,h0"] *
          L2["a2,a4,a0,a6"] * L2["a3,a5,a1,a7"];
    C0 += (1.0 / 16.0) * H3["h0,a2,a3,p0,a0,a1"] * T3["h1,a6,a7,p0,a4,a5"] * L1["h1,h0"] *
          L2["a4,a5,a0,a1"] * L2["a6,a7,a2,a3"];
    C0 += (-1.0 / 36.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L1["a3,a0"] *
          L1["a4,a1"] * L1["a5,a2"];
    C0 += (1.0 / 8.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h3,h2"] *
          L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C0 += (-1.0 / 72.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h3,h4,a3,a4,a5"] * L1["h3,h1"] *
          L1["h4,h2"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 2.0) * H3["a1,a2,a3,h0,h1,a0"] * T2["h0,h2,a4,a5"] * L1["h2,h1"] * L1["a4,a1"] *
          L2["a2,a3,a0,a5"];
    C0 += (-1.0 / 4.0) * H3["a1,a2,a3,h0,h1,a0"] * T3["h0,h1,a4,a5,a6,a7"] * L1["a5,a1"] *
          L1["a6,a2"] * L2["a3,a4,a0,a7"];
    C0 += (-1.0 / 4.0) * H3["a1,a2,a3,h0,h1,a0"] * T3["h0,h2,a4,a5,a6,a7"] * L1["h2,h1"] *
          L1["a5,a1"] * L3["a2,a3,a4,a0,a6,a7"];
    C0 += (1.0 / 4.0) * H3["a1,a2,a3,h0,h1,a0"] * T3["h0,h2,a4,a5,a6,a7"] * L1["h2,h1"] *
          L2["a1,a2,a0,a5"] * L2["a6,a7,a3,a4"];
    C0 += (1.0 / 4.0) * H3["a1,a2,a3,h0,h1,a0"] * T3["h0,h2,a4,a5,a6,a7"] * L1["h2,h1"] *
          L2["a1,a4,a0,a5"] * L2["a6,a7,a2,a3"];
    C0 += (1.0 / 72.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L1["a5,a0"] *
          L1["a6,a1"] * L3["a7,a8,a9,a2,a3,a4"];
    C0 += (-1.0 / 4.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L1["a5,a0"] *
          L2["a2,a3,a1,a7"] * L2["a8,a9,a4,a6"];
    C0 += (-1.0 / 4.0) * H3["a2,a3,a4,p0,a0,a1"] * T3["a7,a8,a9,p0,a5,a6"] * L1["a5,a0"] *
          L2["a2,a6,a1,a7"] * L2["a8,a9,a3,a4"];
    C0 += (-1.0 / 4.0) * H3["a2,a3,a4,h0,a0,a1"] * T2["h0,a5,a6,a7"] * L1["a6,a2"] * L1["a7,a3"] *
          L2["a4,a5,a0,a1"];
    C0 += (-1.0 / 8.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a2"] *
          L1["a8,a3"] * L3["a4,a5,a6,a0,a1,a9"];
    C0 += (1.0 / 4.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a2"] *
          L2["a3,a4,a0,a8"] * L2["a5,a6,a1,a9"];
    C0 += (1.0 / 4.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a2"] *
          L2["a3,a5,a0,a1"] * L2["a8,a9,a4,a6"];
    C0 += (-1.0 / 2.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a2"] *
          L2["a3,a5,a0,a8"] * L2["a4,a6,a1,a9"];
    C0 += (-1.0 / 16.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a2"] *
          L2["a5,a6,a0,a1"] * L2["a8,a9,a3,a4"];
    C0 += (-1.0 / 12.0) * H3["h0,h1,h2,p0,p1,a0"] * T3["h3,h4,h5,p0,p1,a1"] * L1["h3,h0"] *
          L1["h4,h1"] * L1["h5,h2"] * L1["a1,a0"];
    C0 += (1.0 / 24.0) * H3["h0,h1,h2,p0,a0,a1"] * T3["h3,h4,h5,p0,a2,a3"] * L1["h3,h0"] *
          L1["h4,h1"] * L1["h5,h2"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 2.0) * H3["h0,h1,a2,p0,a0,a1"] * T3["h2,h3,a5,p0,a3,a4"] * L1["h2,h0"] *
          L1["h3,h1"] * L1["a3,a0"] * L2["a2,a4,a1,a5"];
    C0 += (1.0 / 8.0) * H3["h0,a2,a3,p0,a0,a1"] * T3["h1,a6,a7,p0,a4,a5"] * L1["h1,h0"] *
          L1["a4,a0"] * L1["a5,a1"] * L2["a6,a7,a2,a3"];
    C0 += (1.0 / 12.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h3,h2"] *
          L1["a3,a0"] * L1["a4,a1"] * L1["a5,a2"];
    C0 += (-1.0 / 8.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h3,h4,a3,a4,a5"] * L1["h3,h1"] *
          L1["h4,h2"] * L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C0 += (1.0 / 2.0) * H3["a1,a2,a3,h0,h1,a0"] * T3["h0,h2,a4,a5,a6,a7"] * L1["h2,h1"] *
          L1["a5,a1"] * L1["a6,a2"] * L2["a3,a4,a0,a7"];
    C0 += (-1.0 / 24.0) * H3["a2,a3,a4,h0,a0,a1"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a2"] *
          L1["a8,a3"] * L1["a9,a4"] * L2["a5,a6,a0,a1"];
    C0 += (1.0 / 12.0) * H3["h0,h1,h2,p0,a0,a1"] * T3["h3,h4,h5,p0,a2,a3"] * L1["h3,h0"] *
          L1["h4,h1"] * L1["h5,h2"] * L1["a2,a0"] * L1["a3,a1"];
    C0 += (-1.0 / 12.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h3,h4,a3,a4,a5"] * L1["h3,h1"] *
          L1["h4,h2"] * L1["a3,a0"] * L1["a4,a1"] * L1["a5,a2"];

    C1["g0,g1"] += 1.0 * H2["g1,h0,g0,p0"] * T1["h1,p0"] * L1["h1,h0"];
    C1["g0,g1"] += -1.0 * H2["g1,h0,g0,a0"] * T1["h0,a1"] * L1["a1,a0"];
    C1["g0,g1"] += (-1.0 / 2.0) * H2["g1,h0,g0,a0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["g0,g1"] +=
        (-1.0 / 12.0) * H2["g1,h0,g0,a0"] * T3["h0,a1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["g0,g1"] += (1.0 / 2.0) * H2["g1,a0,g0,p0"] * T2["a2,a3,p0,a1"] * L2["a2,a3,a0,a1"];
    C1["g0,g1"] +=
        (1.0 / 12.0) * H2["g1,a0,g0,p0"] * T3["a3,a4,a5,p0,a1,a2"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["g0,g1"] += (-1.0 / 8.0) * H3["g1,h0,h1,g0,a0,a1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["g0,g1"] +=
        (-1.0 / 24.0) * H3["g1,h0,h1,g0,a0,a1"] * T3["h0,h1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["g0,g1"] += (1.0 / 2.0) * H3["g1,h0,a2,g0,a0,a1"] * T1["h0,a3"] * L2["a2,a3,a0,a1"];
    C1["g0,g1"] +=
        (1.0 / 4.0) * H3["g1,h0,a2,g0,a0,a1"] * T2["h0,a3,a4,a5"] * L3["a2,a4,a5,a0,a1,a3"];
    C1["g0,g1"] += (1.0 / 8.0) * H3["g1,a0,a1,g0,p0,p1"] * T2["a2,a3,p0,p1"] * L2["a2,a3,a0,a1"];
    C1["g0,g1"] +=
        (1.0 / 24.0) * H3["g1,a0,a1,g0,p0,p1"] * T3["a3,a4,a5,p0,p1,a2"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["g0,g1"] += (-1.0 / 2.0) * H3["g1,a1,a2,g0,p0,a0"] * T1["a3,p0"] * L2["a1,a2,a0,a3"];
    C1["g0,g1"] +=
        (-1.0 / 4.0) * H3["g1,a1,a2,g0,p0,a0"] * T2["a4,a5,p0,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["g0,g1"] +=
        (1.0 / 4.0) * H3["g1,h0,h1,g0,p0,p1"] * T2["h2,h3,p0,p1"] * L1["h2,h0"] * L1["h3,h1"];
    C1["g0,g1"] +=
        (-1.0 / 4.0) * H3["g1,h0,h1,g0,a0,a1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C1["g0,g1"] +=
        (1.0 / 4.0) * H3["g1,h0,h1,g0,a0,a1"] * T2["h0,h2,a2,a3"] * L1["h2,h1"] * L2["a2,a3,a0,a1"];
    C1["g0,g1"] += (-1.0 / 4.0) * H3["g1,h0,h1,g0,a0,a1"] * T3["h0,h1,a2,a3,a4,a5"] * L1["a3,a0"] *
                   L2["a4,a5,a1,a2"];
    C1["g0,g1"] += (1.0 / 12.0) * H3["g1,h0,h1,g0,a0,a1"] * T3["h0,h2,a2,a3,a4,a5"] * L1["h2,h1"] *
                   L3["a3,a4,a5,a0,a1,a2"];
    C1["g0,g1"] += (1.0 / 4.0) * H3["g1,h0,a0,g0,p0,p1"] * T3["h1,a2,a3,p0,p1,a1"] * L1["h1,h0"] *
                   L2["a2,a3,a0,a1"];
    C1["g0,g1"] +=
        1.0 * H3["g1,h0,a1,g0,p0,a0"] * T2["h1,a3,p0,a2"] * L1["h1,h0"] * L2["a1,a2,a0,a3"];
    C1["g0,g1"] += (1.0 / 4.0) * H3["g1,h0,a1,g0,p0,a0"] * T3["h1,a4,a5,p0,a2,a3"] * L1["h1,h0"] *
                   L3["a1,a2,a3,a0,a4,a5"];
    C1["g0,g1"] +=
        -1.0 * H3["g1,h0,a2,g0,a0,a1"] * T2["h0,a3,a4,a5"] * L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C1["g0,g1"] += (-1.0 / 4.0) * H3["g1,h0,a2,g0,a0,a1"] * T3["h0,a3,a4,a5,a6,a7"] * L1["a5,a0"] *
                   L3["a2,a6,a7,a1,a3,a4"];
    C1["g0,g1"] += (-1.0 / 2.0) * H3["g1,h0,a2,g0,a0,a1"] * T3["h0,a3,a4,a5,a6,a7"] *
                   L2["a2,a5,a0,a3"] * L2["a6,a7,a1,a4"];
    C1["g0,g1"] += (1.0 / 8.0) * H3["g1,h0,a2,g0,a0,a1"] * T3["h0,a3,a4,a5,a6,a7"] *
                   L2["a5,a6,a0,a1"] * L2["a3,a4,a2,a7"];
    C1["g0,g1"] += (-1.0 / 4.0) * H3["g1,a1,a2,g0,p0,a0"] * T2["a4,a5,p0,a3"] * L1["a3,a0"] *
                   L2["a4,a5,a1,a2"];
    C1["g0,g1"] += (-1.0 / 12.0) * H3["g1,a1,a2,g0,p0,a0"] * T3["a5,a6,a7,p0,a3,a4"] * L1["a3,a0"] *
                   L3["a5,a6,a7,a1,a2,a4"];
    C1["g0,g1"] += (1.0 / 2.0) * H3["g1,a1,a2,g0,p0,a0"] * T3["a5,a6,a7,p0,a3,a4"] *
                   L2["a1,a3,a0,a5"] * L2["a6,a7,a2,a4"];
    C1["g0,g1"] += (-1.0 / 8.0) * H3["g1,a1,a2,g0,p0,a0"] * T3["a5,a6,a7,p0,a3,a4"] *
                   L2["a3,a4,a0,a5"] * L2["a6,a7,a1,a2"];
    C1["g0,g1"] += (-1.0 / 2.0) * H3["g1,h0,h1,g0,p0,a0"] * T2["h2,h3,p0,a1"] * L1["h2,h0"] *
                   L1["h3,h1"] * L1["a1,a0"];
    C1["g0,g1"] += (-1.0 / 4.0) * H3["g1,h0,h1,g0,p0,a0"] * T3["h2,h3,a3,p0,a1,a2"] * L1["h2,h0"] *
                   L1["h3,h1"] * L2["a1,a2,a0,a3"];
    C1["g0,g1"] += (1.0 / 2.0) * H3["g1,h0,h1,g0,a0,a1"] * T2["h0,h2,a2,a3"] * L1["h2,h1"] *
                   L1["a2,a0"] * L1["a3,a1"];
    C1["g0,g1"] += (1.0 / 2.0) * H3["g1,h0,h1,g0,a0,a1"] * T3["h0,h2,a2,a3,a4,a5"] * L1["h2,h1"] *
                   L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C1["g0,g1"] += (-1.0 / 2.0) * H3["g1,h0,a1,g0,p0,a0"] * T3["h1,a4,a5,p0,a2,a3"] * L1["h1,h0"] *
                   L1["a2,a0"] * L2["a4,a5,a1,a3"];
    C1["g0,g1"] += (1.0 / 4.0) * H3["g1,h0,a2,g0,a0,a1"] * T3["h0,a3,a4,a5,a6,a7"] * L1["a5,a0"] *
                   L1["a6,a1"] * L2["a3,a4,a2,a7"];

    C1["g0,p0"] += -1.0 * H1["h0,g0"] * T1["h0,p0"];
    C1["g0,p0"] += (-1.0 / 2.0) * H2["h0,h1,g0,a0"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C1["g0,p0"] += (-1.0 / 4.0) * H2["h0,h1,g0,a0"] * T3["h0,h1,a3,p0,a1,a2"] * L2["a1,a2,a0,a3"];
    C1["g0,p0"] += -1.0 * H2["h0,a1,g0,a0"] * T2["h0,a3,p0,a2"] * L2["a1,a2,a0,a3"];
    C1["g0,p0"] +=
        (-1.0 / 4.0) * H2["h0,a1,g0,a0"] * T3["h0,a4,a5,p0,a2,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["g0,p0"] += (-1.0 / 4.0) * H2["a0,a1,g0,p1"] * T2["a2,a3,p0,p1"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] +=
        (-1.0 / 12.0) * H2["a0,a1,g0,p1"] * T3["a3,a4,a5,p0,p1,a2"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["g0,p0"] +=
        (-1.0 / 24.0) * H3["h0,h1,h2,g0,a0,a1"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["h0,h1,a2,g0,a0,a1"] * T2["h0,h1,p0,a3"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] +=
        (1.0 / 8.0) * H3["h0,h1,a2,g0,a0,a1"] * T3["h0,h1,a5,p0,a3,a4"] * L3["a2,a3,a4,a0,a1,a5"];
    C1["g0,p0"] +=
        (-1.0 / 4.0) * H3["h0,a2,a3,g0,a0,a1"] * T2["h0,a5,p0,a4"] * L3["a2,a3,a4,a0,a1,a5"];
    C1["g0,p0"] +=
        (-1.0 / 72.0) * H3["a0,a1,a2,g0,p1,p2"] * T3["a3,a4,a5,p0,p1,p2"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["g0,p0"] +=
        (-1.0 / 12.0) * H3["a1,a2,a3,g0,p1,a0"] * T2["a4,a5,p0,p1"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["g0,p0"] += (-1.0 / 2.0) * H2["h0,h1,g0,p1"] * T2["h2,h3,p0,p1"] * L1["h2,h0"] * L1["h3,h1"];
    C1["g0,p0"] += 1.0 * H2["h0,h1,g0,a0"] * T2["h0,h2,p0,a1"] * L1["h2,h1"] * L1["a1,a0"];
    C1["g0,p0"] +=
        (1.0 / 2.0) * H2["h0,h1,g0,a0"] * T3["h0,h2,a3,p0,a1,a2"] * L1["h2,h1"] * L2["a1,a2,a0,a3"];
    C1["g0,p0"] += (-1.0 / 2.0) * H2["h0,a0,g0,p1"] * T3["h1,a2,a3,p0,p1,a1"] * L1["h1,h0"] *
                   L2["a2,a3,a0,a1"];
    C1["g0,p0"] +=
        (1.0 / 2.0) * H2["h0,a1,g0,a0"] * T3["h0,a4,a5,p0,a2,a3"] * L1["a2,a0"] * L2["a4,a5,a1,a3"];
    C1["g0,p0"] += (-1.0 / 12.0) * H3["h0,h1,h2,g0,a0,a1"] * T3["h0,h1,h2,p0,a2,a3"] * L1["a2,a0"] *
                   L1["a3,a1"];
    C1["g0,p0"] += (1.0 / 8.0) * H3["h0,h1,h2,g0,a0,a1"] * T3["h0,h1,h3,p0,a2,a3"] * L1["h3,h2"] *
                   L2["a2,a3,a0,a1"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["h0,h1,a2,g0,a0,a1"] * T2["h0,h2,p0,a3"] * L1["h2,h1"] *
                   L2["a2,a3,a0,a1"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["h0,h1,a2,g0,a0,a1"] * T3["h0,h1,a5,p0,a3,a4"] * L1["a3,a0"] *
                   L2["a2,a4,a1,a5"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["h0,h1,a2,g0,a0,a1"] * T3["h0,h2,a5,p0,a3,a4"] * L1["h2,h1"] *
                   L3["a2,a3,a4,a0,a1,a5"];
    C1["g0,p0"] += (-1.0 / 8.0) * H3["h0,a0,a1,g0,p1,p2"] * T3["h1,a2,a3,p0,p1,p2"] * L1["h1,h0"] *
                   L2["a2,a3,a0,a1"];
    C1["g0,p0"] +=
        (1.0 / 2.0) * H3["h0,a1,a2,g0,p1,a0"] * T2["h1,a3,p0,p1"] * L1["h1,h0"] * L2["a1,a2,a0,a3"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["h0,a1,a2,g0,p1,a0"] * T3["h1,a4,a5,p0,p1,a3"] * L1["h1,h0"] *
                   L3["a1,a2,a3,a0,a4,a5"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["h0,a2,a3,g0,a0,a1"] * T2["h0,a5,p0,a4"] * L1["a4,a0"] *
                   L2["a2,a3,a1,a5"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["h0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] * L1["a4,a0"] *
                   L3["a2,a3,a5,a1,a6,a7"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["h0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] *
                   L2["a2,a3,a0,a6"] * L2["a4,a5,a1,a7"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["h0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] *
                   L2["a2,a4,a0,a1"] * L2["a6,a7,a3,a5"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["h0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] *
                   L2["a2,a4,a0,a6"] * L2["a3,a5,a1,a7"];
    C1["g0,p0"] += (-1.0 / 16.0) * H3["h0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] *
                   L2["a4,a5,a0,a1"] * L2["a6,a7,a2,a3"];
    C1["g0,p0"] += (1.0 / 36.0) * H3["a1,a2,a3,g0,p1,a0"] * T3["a5,a6,a7,p0,p1,a4"] * L1["a4,a0"] *
                   L3["a5,a6,a7,a1,a2,a3"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["a1,a2,a3,g0,p1,a0"] * T3["a5,a6,a7,p0,p1,a4"] *
                   L2["a1,a2,a0,a5"] * L2["a6,a7,a3,a4"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["a1,a2,a3,g0,p1,a0"] * T3["a5,a6,a7,p0,p1,a4"] *
                   L2["a1,a4,a0,a5"] * L2["a6,a7,a2,a3"];
    C1["g0,p0"] += (-1.0 / 12.0) * H3["h0,h1,h2,g0,p1,p2"] * T3["h3,h4,h5,p0,p1,p2"] * L1["h3,h0"] *
                   L1["h4,h1"] * L1["h5,h2"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["h0,h1,h2,g0,a0,a1"] * T3["h0,h1,h3,p0,a2,a3"] * L1["h3,h2"] *
                   L1["a2,a0"] * L1["a3,a1"];
    C1["g0,p0"] += (-1.0 / 8.0) * H3["h0,h1,h2,g0,a0,a1"] * T3["h0,h3,h4,p0,a2,a3"] * L1["h3,h1"] *
                   L1["h4,h2"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["h0,h1,a1,g0,p1,a0"] * T3["h2,h3,a3,p0,p1,a2"] * L1["h2,h0"] *
                   L1["h3,h1"] * L2["a1,a2,a0,a3"];
    C1["g0,p0"] += 1.0 * H3["h0,h1,a2,g0,a0,a1"] * T3["h0,h2,a5,p0,a3,a4"] * L1["h2,h1"] *
                   L1["a3,a0"] * L2["a2,a4,a1,a5"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["h0,a1,a2,g0,p1,a0"] * T3["h1,a4,a5,p0,p1,a3"] * L1["h1,h0"] *
                   L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C1["g0,p0"] += (-1.0 / 8.0) * H3["h0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] * L1["a4,a0"] *
                   L1["a5,a1"] * L2["a6,a7,a2,a3"];
    C1["g0,p0"] += (1.0 / 6.0) * H3["h0,h1,h2,g0,p1,a0"] * T3["h3,h4,h5,p0,p1,a1"] * L1["h3,h0"] *
                   L1["h4,h1"] * L1["h5,h2"] * L1["a1,a0"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["h0,h1,h2,g0,a0,a1"] * T3["h0,h3,h4,p0,a2,a3"] * L1["h3,h1"] *
                   L1["h4,h2"] * L1["a2,a0"] * L1["a3,a1"];

    C1["h0,g0"] += 1.0 * H1["p0,g0"] * T1["h0,p0"];
    C1["h0,g0"] += (1.0 / 2.0) * H2["p0,p1,g0,h1"] * T2["h0,h2,p0,p1"] * L1["h2,h1"];
    C1["h0,g0"] += (1.0 / 4.0) * H2["p0,p1,g0,a0"] * T3["h0,a2,a3,p0,p1,a1"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] += 1.0 * H2["p0,a1,g0,a0"] * T2["h0,a3,p0,a2"] * L2["a1,a3,a0,a2"];
    C1["h0,g0"] +=
        (1.0 / 4.0) * H2["p0,a1,g0,a0"] * T3["h0,a4,a5,p0,a2,a3"] * L3["a1,a4,a5,a0,a2,a3"];
    C1["h0,g0"] += (1.0 / 4.0) * H2["a0,a1,g0,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] +=
        (1.0 / 12.0) * H2["a0,a1,g0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,g0"] +=
        (1.0 / 24.0) * H3["p0,p1,p2,g0,a0,a1"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["p0,p1,a2,g0,a0,a1"] * T2["h0,a3,p0,p1"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] +=
        (-1.0 / 8.0) * H3["p0,p1,a2,g0,a0,a1"] * T3["h0,a4,a5,p0,p1,a3"] * L3["a2,a4,a5,a0,a1,a3"];
    C1["h0,g0"] +=
        (1.0 / 4.0) * H3["p0,a2,a3,g0,a0,a1"] * T2["h0,a5,p0,a4"] * L3["a2,a3,a5,a0,a1,a4"];
    C1["h0,g0"] +=
        (1.0 / 72.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,g0"] +=
        (1.0 / 12.0) * H3["a1,a2,a3,g0,h1,a0"] * T2["h0,h1,a4,a5"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["h0,g0"] += -1.0 * H2["p0,a0,g0,h1"] * T2["h0,h2,p0,a1"] * L1["h2,h1"] * L1["a1,a0"];
    C1["h0,g0"] += (-1.0 / 2.0) * H2["p0,a0,g0,h1"] * T3["h0,h2,a3,p0,a1,a2"] * L1["h2,h1"] *
                   L2["a1,a2,a0,a3"];
    C1["h0,g0"] += (-1.0 / 2.0) * H2["p0,a1,g0,a0"] * T3["h0,a4,a5,p0,a2,a3"] * L1["a2,a1"] *
                   L2["a4,a5,a0,a3"];
    C1["h0,g0"] += (1.0 / 2.0) * H2["a0,a1,g0,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C1["h0,g0"] +=
        (1.0 / 2.0) * H2["a0,a1,g0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C1["h0,g0"] += (1.0 / 12.0) * H3["p0,p1,p2,g0,h1,h2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h3,h1"] *
                   L1["h4,h2"];
    C1["h0,g0"] += (1.0 / 2.0) * H3["p0,p1,a1,g0,h1,a0"] * T3["h0,h2,a3,p0,p1,a2"] * L1["h2,h1"] *
                   L2["a1,a3,a0,a2"];
    C1["h0,g0"] += (-1.0 / 8.0) * H3["p0,p1,a2,g0,a0,a1"] * T3["h0,a4,a5,p0,p1,a3"] * L1["a3,a2"] *
                   L2["a4,a5,a0,a1"];
    C1["h0,g0"] +=
        (1.0 / 2.0) * H3["p0,a1,a2,g0,h1,a0"] * T2["h0,h2,p0,a3"] * L1["h2,h1"] * L2["a1,a2,a0,a3"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["p0,a1,a2,g0,h1,a0"] * T3["h0,h2,a5,p0,a3,a4"] * L1["h2,h1"] *
                   L3["a1,a2,a5,a0,a3,a4"];
    C1["h0,g0"] +=
        (1.0 / 2.0) * H3["p0,a2,a3,g0,a0,a1"] * T2["h0,a5,p0,a4"] * L1["a4,a2"] * L2["a3,a5,a0,a1"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["p0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] * L1["a4,a2"] *
                   L3["a3,a6,a7,a0,a1,a5"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["p0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] *
                   L2["a2,a3,a0,a4"] * L2["a6,a7,a1,a5"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["p0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] *
                   L2["a2,a6,a0,a1"] * L2["a4,a5,a3,a7"];
    C1["h0,g0"] += (1.0 / 2.0) * H3["p0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] *
                   L2["a2,a6,a0,a4"] * L2["a3,a7,a1,a5"];
    C1["h0,g0"] += (1.0 / 16.0) * H3["p0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] *
                   L2["a6,a7,a0,a1"] * L2["a4,a5,a2,a3"];
    C1["h0,g0"] += (1.0 / 8.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L1["a3,a0"] *
                   L2["a4,a5,a1,a2"];
    C1["h0,g0"] += (-1.0 / 36.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h3,h2"] *
                   L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,g0"] += (-1.0 / 2.0) * H3["a1,a2,a3,g0,h1,a0"] * T2["h0,h1,a4,a5"] * L1["a4,a1"] *
                   L2["a2,a3,a0,a5"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["a1,a2,a3,g0,h1,a0"] * T3["h0,h1,a4,a5,a6,a7"] * L1["a5,a1"] *
                   L3["a2,a3,a4,a0,a6,a7"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["a1,a2,a3,g0,h1,a0"] * T3["h0,h1,a4,a5,a6,a7"] *
                   L2["a1,a2,a0,a5"] * L2["a6,a7,a3,a4"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["a1,a2,a3,g0,h1,a0"] * T3["h0,h1,a4,a5,a6,a7"] *
                   L2["a1,a4,a0,a5"] * L2["a6,a7,a2,a3"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["p0,p1,a0,g0,h1,h2"] * T3["h0,h3,h4,p0,p1,a1"] * L1["h3,h1"] *
                   L1["h4,h2"] * L1["a1,a0"];
    C1["h0,g0"] += (1.0 / 8.0) * H3["p0,a0,a1,g0,h1,h2"] * T3["h0,h3,h4,p0,a2,a3"] * L1["h3,h1"] *
                   L1["h4,h2"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] += -1.0 * H3["p0,a1,a2,g0,h1,a0"] * T3["h0,h2,a5,p0,a3,a4"] * L1["h2,h1"] *
                   L1["a3,a1"] * L2["a2,a5,a0,a4"];
    C1["h0,g0"] += (1.0 / 8.0) * H3["p0,a2,a3,g0,a0,a1"] * T3["h0,a6,a7,p0,a4,a5"] * L1["a4,a2"] *
                   L1["a5,a3"] * L2["a6,a7,a0,a1"];
    C1["h0,g0"] += (1.0 / 12.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L1["a3,a0"] *
                   L1["a4,a1"] * L1["a5,a2"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h3,h2"] *
                   L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C1["h0,g0"] += (1.0 / 2.0) * H3["a1,a2,a3,g0,h1,a0"] * T3["h0,h1,a4,a5,a6,a7"] * L1["a5,a1"] *
                   L1["a6,a2"] * L2["a3,a4,a0,a7"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["p0,a0,a1,g0,h1,h2"] * T3["h0,h3,h4,p0,a2,a3"] * L1["h3,h1"] *
                   L1["h4,h2"] * L1["a2,a0"] * L1["a3,a1"];
    C1["h0,g0"] += (-1.0 / 6.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h3,h2"] *
                   L1["a3,a0"] * L1["a4,a1"] * L1["a5,a2"];

    C1["h0,p0"] += 1.0 * H1["h1,p1"] * T2["h0,h2,p0,p1"] * L1["h2,h1"];
    C1["h0,p0"] += (1.0 / 2.0) * H1["a0,p1"] * T3["h0,a2,a3,p0,p1,a1"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] += -1.0 * H1["a0,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C1["h0,p0"] += (-1.0 / 2.0) * H1["a0,h1"] * T3["h0,h1,a3,p0,a1,a2"] * L2["a1,a2,a0,a3"];
    C1["h0,p0"] += (1.0 / 8.0) * H2["a0,a1,p1,p2"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] += (-1.0 / 8.0) * H2["a0,a1,h1,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] += (-1.0 / 2.0) * H2["a1,a2,p1,a0"] * T2["h0,a3,p0,p1"] * L2["a1,a2,a0,a3"];
    C1["h0,p0"] +=
        (-1.0 / 4.0) * H2["a1,a2,p1,a0"] * T3["h0,a4,a5,p0,p1,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["h0,p0"] += (1.0 / 2.0) * H2["a1,a2,h1,a0"] * T2["h0,h1,p0,a3"] * L2["a1,a2,a0,a3"];
    C1["h0,p0"] +=
        (1.0 / 4.0) * H2["a1,a2,h1,a0"] * T3["h0,h1,a5,p0,a3,a4"] * L3["a1,a2,a5,a0,a3,a4"];
    C1["h0,p0"] +=
        (1.0 / 24.0) * H3["a1,a2,a3,p1,p2,a0"] * T3["h0,a4,a5,p0,p1,p2"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["h0,p0"] +=
        (-1.0 / 24.0) * H3["a1,a2,a3,h1,h2,a0"] * T3["h0,h1,h2,p0,a4,a5"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["h0,p0"] +=
        (1.0 / 12.0) * H3["a2,a3,a4,p1,a0,a1"] * T2["h0,a5,p0,p1"] * L3["a2,a3,a4,a0,a1,a5"];
    C1["h0,p0"] +=
        (-1.0 / 12.0) * H3["a2,a3,a4,h1,a0,a1"] * T2["h0,h1,p0,a5"] * L3["a2,a3,a4,a0,a1,a5"];
    C1["h0,p0"] +=
        (1.0 / 4.0) * H2["h1,h2,p1,p2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h3,h1"] * L1["h4,h2"];
    C1["h0,p0"] +=
        1.0 * H2["h1,a1,p1,a0"] * T3["h0,h2,a3,p0,p1,a2"] * L1["h2,h1"] * L2["a1,a2,a0,a3"];
    C1["h0,p0"] +=
        (-1.0 / 4.0) * H2["a0,a1,h1,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C1["h0,p0"] +=
        (1.0 / 4.0) * H2["a0,a1,h1,h2"] * T3["h0,h1,h3,p0,a2,a3"] * L1["h3,h2"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] += (-1.0 / 4.0) * H2["a1,a2,p1,a0"] * T3["h0,a4,a5,p0,p1,a3"] * L1["a3,a0"] *
                   L2["a4,a5,a1,a2"];
    C1["h0,p0"] +=
        -1.0 * H2["a1,a2,h1,a0"] * T3["h0,h1,a5,p0,a3,a4"] * L1["a3,a1"] * L2["a2,a5,a0,a4"];
    C1["h0,p0"] += (-1.0 / 4.0) * H3["h1,a1,a2,p1,p2,a0"] * T3["h0,h2,a3,p0,p1,p2"] * L1["h2,h1"] *
                   L2["a1,a2,a0,a3"];
    C1["h0,p0"] += (1.0 / 4.0) * H3["h1,a2,a3,p1,a0,a1"] * T3["h0,h2,a5,p0,p1,a4"] * L1["h2,h1"] *
                   L3["a2,a3,a4,a0,a1,a5"];
    C1["h0,p0"] += (1.0 / 4.0) * H3["a1,a2,a3,h1,h2,a0"] * T3["h0,h1,h2,p0,a4,a5"] * L1["a4,a1"] *
                   L2["a2,a3,a0,a5"];
    C1["h0,p0"] += (1.0 / 12.0) * H3["a1,a2,a3,h1,h2,a0"] * T3["h0,h1,h3,p0,a4,a5"] * L1["h3,h2"] *
                   L3["a1,a2,a3,a0,a4,a5"];
    C1["h0,p0"] += (-1.0 / 12.0) * H3["a2,a3,a4,p1,a0,a1"] * T3["h0,a6,a7,p0,p1,a5"] * L1["a5,a0"] *
                   L3["a2,a3,a4,a1,a6,a7"];
    C1["h0,p0"] += (-1.0 / 2.0) * H3["a2,a3,a4,p1,a0,a1"] * T3["h0,a6,a7,p0,p1,a5"] *
                   L2["a2,a3,a0,a6"] * L2["a4,a5,a1,a7"];
    C1["h0,p0"] += (1.0 / 8.0) * H3["a2,a3,a4,p1,a0,a1"] * T3["h0,a6,a7,p0,p1,a5"] *
                   L2["a2,a5,a0,a1"] * L2["a6,a7,a3,a4"];
    C1["h0,p0"] += (-1.0 / 4.0) * H3["a2,a3,a4,h1,a0,a1"] * T3["h0,h1,a7,p0,a5,a6"] * L1["a5,a2"] *
                   L3["a3,a4,a7,a0,a1,a6"];
    C1["h0,p0"] += (1.0 / 2.0) * H3["a2,a3,a4,h1,a0,a1"] * T3["h0,h1,a7,p0,a5,a6"] *
                   L2["a2,a3,a0,a5"] * L2["a4,a7,a1,a6"];
    C1["h0,p0"] += (-1.0 / 8.0) * H3["a2,a3,a4,h1,a0,a1"] * T3["h0,h1,a7,p0,a5,a6"] *
                   L2["a2,a7,a0,a1"] * L2["a5,a6,a3,a4"];
    C1["h0,p0"] += (-1.0 / 2.0) * H2["h1,h2,p1,a0"] * T3["h0,h3,h4,p0,p1,a1"] * L1["h3,h1"] *
                   L1["h4,h2"] * L1["a1,a0"];
    C1["h0,p0"] += (1.0 / 2.0) * H2["a0,a1,h1,h2"] * T3["h0,h1,h3,p0,a2,a3"] * L1["h3,h2"] *
                   L1["a2,a0"] * L1["a3,a1"];
    C1["h0,p0"] += (1.0 / 4.0) * H3["h1,h2,a2,p1,a0,a1"] * T3["h0,h3,h4,p0,p1,a3"] * L1["h3,h1"] *
                   L1["h4,h2"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] += (1.0 / 2.0) * H3["h1,a2,a3,p1,a0,a1"] * T3["h0,h2,a5,p0,p1,a4"] * L1["h2,h1"] *
                   L1["a4,a0"] * L2["a2,a3,a1,a5"];
    C1["h0,p0"] += (-1.0 / 2.0) * H3["a1,a2,a3,h1,h2,a0"] * T3["h0,h1,h3,p0,a4,a5"] * L1["h3,h2"] *
                   L1["a4,a1"] * L2["a2,a3,a0,a5"];
    C1["h0,p0"] += (-1.0 / 4.0) * H3["a2,a3,a4,h1,a0,a1"] * T3["h0,h1,a7,p0,a5,a6"] * L1["a5,a2"] *
                   L1["a6,a3"] * L2["a4,a7,a0,a1"];

    C2["g0,g1,g2,g3"] += 1.0 * H3["g2,g3,h0,g0,g1,p0"] * T1["h1,p0"] * L1["h1,h0"];
    C2["g0,g1,g2,g3"] += -1.0 * H3["g2,g3,h0,g0,g1,a0"] * T1["h0,a1"] * L1["a1,a0"];
    C2["g0,g1,g2,g3"] +=
        (-1.0 / 2.0) * H3["g2,g3,h0,g0,g1,a0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C2["g0,g1,g2,g3"] +=
        (-1.0 / 12.0) * H3["g2,g3,h0,g0,g1,a0"] * T3["h0,a1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C2["g0,g1,g2,g3"] +=
        (1.0 / 2.0) * H3["g2,g3,a0,g0,g1,p0"] * T2["a2,a3,p0,a1"] * L2["a2,a3,a0,a1"];
    C2["g0,g1,g2,g3"] +=
        (1.0 / 12.0) * H3["g2,g3,a0,g0,g1,p0"] * T3["a3,a4,a5,p0,a1,a2"] * L3["a3,a4,a5,a0,a1,a2"];

    C2["g0,g1,p0,p1"] += (-1.0 / 2.0) * H2["h0,h1,g0,g1"] * T2["h0,h1,p0,p1"];
    C2["g0,g1,p0,p1"] += 1.0 * H2["h0,h1,g0,g1"] * T2["h0,h2,p0,p1"] * L1["h2,h1"];
    C2["g0,g1,p0,p1"] +=
        (1.0 / 2.0) * H2["h0,a0,g0,g1"] * T3["h0,a2,a3,p0,p1,a1"] * L2["a2,a3,a0,a1"];
    C2["g0,g1,p0,p1"] +=
        (-1.0 / 6.0) * H3["h0,h1,h2,g0,g1,a0"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    C2["g0,g1,p0,p1"] +=
        (-1.0 / 2.0) * H3["h0,h1,a1,g0,g1,a0"] * T3["h0,h1,a3,p0,p1,a2"] * L2["a1,a2,a0,a3"];
    C2["g0,g1,p0,p1"] +=
        (-1.0 / 2.0) * H3["h0,a1,a2,g0,g1,a0"] * T2["h0,a3,p0,p1"] * L2["a1,a2,a0,a3"];
    C2["g0,g1,p0,p1"] +=
        (-1.0 / 4.0) * H3["h0,a1,a2,g0,g1,a0"] * T3["h0,a4,a5,p0,p1,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    C2["g0,g1,p0,p1"] +=
        (1.0 / 36.0) * H3["a0,a1,a2,g0,g1,p2"] * T3["a3,a4,a5,p0,p1,p2"] * L3["a3,a4,a5,a0,a1,a2"];
    C2["g0,g1,p0,p1"] +=
        (1.0 / 2.0) * H3["h0,h1,h2,g0,g1,a0"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h3,h2"] * L1["a1,a0"];
    C2["g0,g1,p0,p1"] +=
        1.0 * H3["h0,h1,a1,g0,g1,a0"] * T3["h0,h2,a3,p0,p1,a2"] * L1["h2,h1"] * L2["a1,a2,a0,a3"];
    C2["g0,g1,p0,p1"] += (1.0 / 4.0) * H3["h0,a0,a1,g0,g1,p2"] * T3["h1,a2,a3,p0,p1,p2"] *
                         L1["h1,h0"] * L2["a2,a3,a0,a1"];
    C2["g0,g1,p0,p1"] += (-1.0 / 4.0) * H3["h0,a1,a2,g0,g1,a0"] * T3["h0,a4,a5,p0,p1,a3"] *
                         L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C2["g0,g1,p0,p1"] += (1.0 / 6.0) * H3["h0,h1,h2,g0,g1,p2"] * T3["h3,h4,h5,p0,p1,p2"] *
                         L1["h3,h0"] * L1["h4,h1"] * L1["h5,h2"];
    C2["g0,g1,p0,p1"] += (-1.0 / 2.0) * H3["h0,h1,h2,g0,g1,a0"] * T3["h0,h3,h4,p0,p1,a1"] *
                         L1["h3,h1"] * L1["h4,h2"] * L1["a1,a0"];

    C2["h0,h1,g0,g1"] += (1.0 / 2.0) * H2["p0,p1,g0,g1"] * T2["h0,h1,p0,p1"];
    C2["h0,h1,g0,g1"] += -1.0 * H2["p0,a0,g0,g1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C2["h0,h1,g0,g1"] +=
        (-1.0 / 2.0) * H2["p0,a0,g0,g1"] * T3["h0,h1,a3,p0,a1,a2"] * L2["a1,a2,a0,a3"];
    C2["h0,h1,g0,g1"] +=
        (1.0 / 6.0) * H3["p0,p1,p2,g0,g1,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h3,h2"];
    C2["h0,h1,g0,g1"] +=
        (1.0 / 2.0) * H3["p0,p1,a1,g0,g1,a0"] * T3["h0,h1,a3,p0,p1,a2"] * L2["a1,a3,a0,a2"];
    C2["h0,h1,g0,g1"] +=
        (1.0 / 2.0) * H3["p0,a1,a2,g0,g1,a0"] * T2["h0,h1,p0,a3"] * L2["a1,a2,a0,a3"];
    C2["h0,h1,g0,g1"] +=
        (1.0 / 4.0) * H3["p0,a1,a2,g0,g1,a0"] * T3["h0,h1,a5,p0,a3,a4"] * L3["a1,a2,a5,a0,a3,a4"];
    C2["h0,h1,g0,g1"] +=
        (-1.0 / 36.0) * H3["a0,a1,a2,g0,g1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C2["h0,h1,g0,g1"] += (-1.0 / 2.0) * H3["p0,p1,a0,g0,g1,h2"] * T3["h0,h1,h3,p0,p1,a1"] *
                         L1["h3,h2"] * L1["a1,a0"];
    C2["h0,h1,g0,g1"] += (1.0 / 4.0) * H3["p0,a0,a1,g0,g1,h2"] * T3["h0,h1,h3,p0,a2,a3"] *
                         L1["h3,h2"] * L2["a2,a3,a0,a1"];
    C2["h0,h1,g0,g1"] +=
        -1.0 * H3["p0,a1,a2,g0,g1,a0"] * T3["h0,h1,a5,p0,a3,a4"] * L1["a3,a1"] * L2["a2,a5,a0,a4"];
    C2["h0,h1,g0,g1"] += (-1.0 / 4.0) * H3["a0,a1,a2,g0,g1,h2"] * T3["h0,h1,h2,a3,a4,a5"] *
                         L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C2["h0,h1,g0,g1"] += (1.0 / 2.0) * H3["p0,a0,a1,g0,g1,h2"] * T3["h0,h1,h3,p0,a2,a3"] *
                         L1["h3,h2"] * L1["a2,a0"] * L1["a3,a1"];
    C2["h0,h1,g0,g1"] += (-1.0 / 6.0) * H3["a0,a1,a2,g0,g1,h2"] * T3["h0,h1,h2,a3,a4,a5"] *
                         L1["a3,a0"] * L1["a4,a1"] * L1["a5,a2"];

    C2["h0,h1,p0,p1"] += 1.0 * H1["h2,p2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h3,h2"];
    C2["h0,h1,p0,p1"] += -1.0 * H1["a0,h2"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    C2["h0,h1,p0,p1"] +=
        (-1.0 / 2.0) * H2["a1,a2,p2,a0"] * T3["h0,h1,a3,p0,p1,p2"] * L2["a1,a2,a0,a3"];
    C2["h0,h1,p0,p1"] +=
        (1.0 / 2.0) * H2["a1,a2,h2,a0"] * T3["h0,h1,h2,p0,p1,a3"] * L2["a1,a2,a0,a3"];
    C2["h0,h1,p0,p1"] +=
        (1.0 / 12.0) * H3["a2,a3,a4,p2,a0,a1"] * T3["h0,h1,a5,p0,p1,p2"] * L3["a2,a3,a4,a0,a1,a5"];
    C2["h0,h1,p0,p1"] +=
        (-1.0 / 12.0) * H3["a2,a3,a4,h2,a0,a1"] * T3["h0,h1,h2,p0,p1,a5"] * L3["a2,a3,a4,a0,a1,a5"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggp"});
    temp["g0,g1,g2,p0"] += -1.0 * H2["g2,h0,g0,g1"] * T1["h0,p0"];
    temp["g0,g1,g2,p0"] += (-1.0 / 2.0) * H3["g2,h0,h1,g0,g1,a0"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp["g0,g1,g2,p0"] +=
        (-1.0 / 4.0) * H3["g2,h0,h1,g0,g1,a0"] * T3["h0,h1,a3,p0,a1,a2"] * L2["a1,a2,a0,a3"];
    temp["g0,g1,g2,p0"] += -1.0 * H3["g2,h0,a1,g0,g1,a0"] * T2["h0,a3,p0,a2"] * L2["a1,a2,a0,a3"];
    temp["g0,g1,g2,p0"] +=
        (-1.0 / 4.0) * H3["g2,h0,a1,g0,g1,a0"] * T3["h0,a4,a5,p0,a2,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    temp["g0,g1,g2,p0"] +=
        (-1.0 / 4.0) * H3["g2,a0,a1,g0,g1,p1"] * T2["a2,a3,p0,p1"] * L2["a2,a3,a0,a1"];
    temp["g0,g1,g2,p0"] +=
        (-1.0 / 12.0) * H3["g2,a0,a1,g0,g1,p1"] * T3["a3,a4,a5,p0,p1,a2"] * L3["a3,a4,a5,a0,a1,a2"];
    temp["g0,g1,g2,p0"] +=
        (-1.0 / 2.0) * H3["g2,h0,h1,g0,g1,p1"] * T2["h2,h3,p0,p1"] * L1["h2,h0"] * L1["h3,h1"];
    temp["g0,g1,g2,p0"] +=
        1.0 * H3["g2,h0,h1,g0,g1,a0"] * T2["h0,h2,p0,a1"] * L1["h2,h1"] * L1["a1,a0"];
    temp["g0,g1,g2,p0"] += (1.0 / 2.0) * H3["g2,h0,h1,g0,g1,a0"] * T3["h0,h2,a3,p0,a1,a2"] *
                           L1["h2,h1"] * L2["a1,a2,a0,a3"];
    temp["g0,g1,g2,p0"] += (-1.0 / 2.0) * H3["g2,h0,a0,g0,g1,p1"] * T3["h1,a2,a3,p0,p1,a1"] *
                           L1["h1,h0"] * L2["a2,a3,a0,a1"];
    temp["g0,g1,g2,p0"] += (1.0 / 2.0) * H3["g2,h0,a1,g0,g1,a0"] * T3["h0,a4,a5,p0,a2,a3"] *
                           L1["a2,a0"] * L2["a4,a5,a1,a3"];
    C2["g0,g1,g2,p0"] += temp["g0,g1,g2,p0"];
    C2["g0,g1,p0,g2"] -= temp["g0,g1,g2,p0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghgg"});
    temp["g0,h0,g1,g2"] += 1.0 * H2["g1,g2,g0,p0"] * T1["h0,p0"];
    temp["g0,h0,g1,g2"] += (1.0 / 2.0) * H3["g1,g2,h1,g0,p0,p1"] * T2["h0,h2,p0,p1"] * L1["h2,h1"];
    temp["g0,h0,g1,g2"] +=
        (1.0 / 4.0) * H3["g1,g2,h1,g0,a0,a1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,g1,g2"] +=
        (1.0 / 12.0) * H3["g1,g2,h1,g0,a0,a1"] * T3["h0,h1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    temp["g0,h0,g1,g2"] +=
        (1.0 / 4.0) * H3["g1,g2,a0,g0,p0,p1"] * T3["h0,a2,a3,p0,p1,a1"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,g1,g2"] += 1.0 * H3["g1,g2,a1,g0,p0,a0"] * T2["h0,a3,p0,a2"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,g1,g2"] +=
        (1.0 / 4.0) * H3["g1,g2,a1,g0,p0,a0"] * T3["h0,a4,a5,p0,a2,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    temp["g0,h0,g1,g2"] +=
        -1.0 * H3["g1,g2,h1,g0,p0,a0"] * T2["h0,h2,p0,a1"] * L1["h2,h1"] * L1["a1,a0"];
    temp["g0,h0,g1,g2"] += (-1.0 / 2.0) * H3["g1,g2,h1,g0,p0,a0"] * T3["h0,h2,a3,p0,a1,a2"] *
                           L1["h2,h1"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,g1,g2"] +=
        (1.0 / 2.0) * H3["g1,g2,h1,g0,a0,a1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    temp["g0,h0,g1,g2"] += (1.0 / 2.0) * H3["g1,g2,h1,g0,a0,a1"] * T3["h0,h1,a2,a3,a4,a5"] *
                           L1["a3,a0"] * L2["a4,a5,a1,a2"];
    temp["g0,h0,g1,g2"] += (-1.0 / 2.0) * H3["g1,g2,a1,g0,p0,a0"] * T3["h0,a4,a5,p0,a2,a3"] *
                           L1["a2,a0"] * L2["a4,a5,a1,a3"];
    C2["g0,h0,g1,g2"] += temp["g0,h0,g1,g2"];
    C2["h0,g0,g1,g2"] -= temp["g0,h0,g1,g2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghpp"});
    temp["g0,h0,p0,p1"] += 1.0 * H1["h1,g0"] * T2["h0,h1,p0,p1"];
    temp["g0,h0,p0,p1"] += (1.0 / 2.0) * H2["h1,h2,g0,a0"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    temp["g0,h0,p0,p1"] += 1.0 * H2["h1,a1,g0,a0"] * T3["h0,h1,a3,p0,p1,a2"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,p0,p1"] +=
        (1.0 / 4.0) * H2["a0,a1,g0,p2"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,p0,p1"] +=
        (-1.0 / 4.0) * H3["h1,h2,a2,g0,a0,a1"] * T3["h0,h1,h2,p0,p1,a3"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,p0,p1"] +=
        (1.0 / 4.0) * H3["h1,a2,a3,g0,a0,a1"] * T3["h0,h1,a5,p0,p1,a4"] * L3["a2,a3,a4,a0,a1,a5"];
    temp["g0,h0,p0,p1"] +=
        (1.0 / 12.0) * H3["a1,a2,a3,g0,p2,a0"] * T3["h0,a4,a5,p0,p1,p2"] * L3["a1,a2,a3,a0,a4,a5"];
    temp["g0,h0,p0,p1"] +=
        (1.0 / 2.0) * H2["h1,h2,g0,p2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h3,h1"] * L1["h4,h2"];
    temp["g0,h0,p0,p1"] +=
        -1.0 * H2["h1,h2,g0,a0"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h3,h2"] * L1["a1,a0"];
    temp["g0,h0,p0,p1"] += (1.0 / 2.0) * H3["h1,h2,a2,g0,a0,a1"] * T3["h0,h1,h3,p0,p1,a3"] *
                           L1["h3,h2"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,p0,p1"] += (-1.0 / 2.0) * H3["h1,a1,a2,g0,p2,a0"] * T3["h0,h2,a3,p0,p1,p2"] *
                           L1["h2,h1"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,p0,p1"] += (1.0 / 2.0) * H3["h1,a2,a3,g0,a0,a1"] * T3["h0,h1,a5,p0,p1,a4"] *
                           L1["a4,a0"] * L2["a2,a3,a1,a5"];
    C2["g0,h0,p0,p1"] += temp["g0,h0,p0,p1"];
    C2["h0,g0,p0,p1"] -= temp["g0,h0,p0,p1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"hhgp"});
    temp["h0,h1,g0,p0"] += -1.0 * H1["p1,g0"] * T2["h0,h1,p0,p1"];
    temp["h0,h1,g0,p0"] += (-1.0 / 2.0) * H2["p1,p2,g0,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h3,h2"];
    temp["h0,h1,g0,p0"] += -1.0 * H2["p1,a1,g0,a0"] * T3["h0,h1,a3,p0,p1,a2"] * L2["a1,a3,a0,a2"];
    temp["h0,h1,g0,p0"] +=
        (-1.0 / 4.0) * H2["a0,a1,g0,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["h0,h1,g0,p0"] +=
        (1.0 / 4.0) * H3["p1,p2,a2,g0,a0,a1"] * T3["h0,h1,a3,p0,p1,p2"] * L2["a2,a3,a0,a1"];
    temp["h0,h1,g0,p0"] +=
        (-1.0 / 4.0) * H3["p1,a2,a3,g0,a0,a1"] * T3["h0,h1,a5,p0,p1,a4"] * L3["a2,a3,a5,a0,a1,a4"];
    temp["h0,h1,g0,p0"] +=
        (-1.0 / 12.0) * H3["a1,a2,a3,g0,h2,a0"] * T3["h0,h1,h2,p0,a4,a5"] * L3["a1,a2,a3,a0,a4,a5"];
    temp["h0,h1,g0,p0"] +=
        1.0 * H2["p1,a0,g0,h2"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h3,h2"] * L1["a1,a0"];
    temp["h0,h1,g0,p0"] +=
        (-1.0 / 2.0) * H2["a0,a1,g0,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    temp["h0,h1,g0,p0"] += (-1.0 / 2.0) * H3["p1,a1,a2,g0,h2,a0"] * T3["h0,h1,h3,p0,p1,a3"] *
                           L1["h3,h2"] * L2["a1,a2,a0,a3"];
    temp["h0,h1,g0,p0"] += (-1.0 / 2.0) * H3["p1,a2,a3,g0,a0,a1"] * T3["h0,h1,a5,p0,p1,a4"] *
                           L1["a4,a2"] * L2["a3,a5,a0,a1"];
    temp["h0,h1,g0,p0"] += (1.0 / 2.0) * H3["a1,a2,a3,g0,h2,a0"] * T3["h0,h1,h2,p0,a4,a5"] *
                           L1["a4,a1"] * L2["a2,a3,a0,a5"];
    C2["h0,h1,g0,p0"] += temp["h0,h1,g0,p0"];
    C2["h0,h1,p0,g0"] -= temp["h0,h1,g0,p0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghgp"});
    temp["g0,h0,g1,p0"] += 1.0 * H2["g1,h1,g0,p1"] * T2["h0,h2,p0,p1"] * L1["h2,h1"];
    temp["g0,h0,g1,p0"] += -1.0 * H2["g1,h1,g0,a0"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp["g0,h0,g1,p0"] +=
        (-1.0 / 2.0) * H2["g1,h1,g0,a0"] * T3["h0,h1,a3,p0,a1,a2"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,g1,p0"] +=
        (1.0 / 2.0) * H2["g1,a0,g0,p1"] * T3["h0,a2,a3,p0,p1,a1"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,g1,p0"] +=
        (-1.0 / 8.0) * H3["g1,h1,h2,g0,a0,a1"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,g1,p0"] +=
        (1.0 / 2.0) * H3["g1,h1,a2,g0,a0,a1"] * T2["h0,h1,p0,a3"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,g1,p0"] +=
        (1.0 / 4.0) * H3["g1,h1,a2,g0,a0,a1"] * T3["h0,h1,a5,p0,a3,a4"] * L3["a2,a3,a4,a0,a1,a5"];
    temp["g0,h0,g1,p0"] +=
        (1.0 / 8.0) * H3["g1,a0,a1,g0,p1,p2"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,g1,p0"] +=
        (-1.0 / 2.0) * H3["g1,a1,a2,g0,p1,a0"] * T2["h0,a3,p0,p1"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,g1,p0"] +=
        (-1.0 / 4.0) * H3["g1,a1,a2,g0,p1,a0"] * T3["h0,a4,a5,p0,p1,a3"] * L3["a1,a2,a3,a0,a4,a5"];
    temp["g0,h0,g1,p0"] +=
        (1.0 / 4.0) * H3["g1,h1,h2,g0,p1,p2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h3,h1"] * L1["h4,h2"];
    temp["g0,h0,g1,p0"] += (-1.0 / 4.0) * H3["g1,h1,h2,g0,a0,a1"] * T3["h0,h1,h2,p0,a2,a3"] *
                           L1["a2,a0"] * L1["a3,a1"];
    temp["g0,h0,g1,p0"] += (1.0 / 4.0) * H3["g1,h1,h2,g0,a0,a1"] * T3["h0,h1,h3,p0,a2,a3"] *
                           L1["h3,h2"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,g1,p0"] +=
        1.0 * H3["g1,h1,a1,g0,p1,a0"] * T3["h0,h2,a3,p0,p1,a2"] * L1["h2,h1"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,g1,p0"] +=
        -1.0 * H3["g1,h1,a2,g0,a0,a1"] * T3["h0,h1,a5,p0,a3,a4"] * L1["a3,a0"] * L2["a2,a4,a1,a5"];
    temp["g0,h0,g1,p0"] += (-1.0 / 4.0) * H3["g1,a1,a2,g0,p1,a0"] * T3["h0,a4,a5,p0,p1,a3"] *
                           L1["a3,a0"] * L2["a4,a5,a1,a2"];
    temp["g0,h0,g1,p0"] += (-1.0 / 2.0) * H3["g1,h1,h2,g0,p1,a0"] * T3["h0,h3,h4,p0,p1,a1"] *
                           L1["h3,h1"] * L1["h4,h2"] * L1["a1,a0"];
    temp["g0,h0,g1,p0"] += (1.0 / 2.0) * H3["g1,h1,h2,g0,a0,a1"] * T3["h0,h1,h3,p0,a2,a3"] *
                           L1["h3,h2"] * L1["a2,a0"] * L1["a3,a1"];
    C2["g0,h0,g1,p0"] += temp["g0,h0,g1,p0"];
    C2["g0,h0,p0,g1"] -= temp["g0,h0,g1,p0"];
    C2["h0,g0,g1,p0"] -= temp["g0,h0,g1,p0"];
    C2["h0,g0,p0,g1"] += temp["g0,h0,g1,p0"];

    C3["g0,g1,g2,p0,p1,p2"] += (-1.0 / 6.0) * H3["h0,h1,h2,g0,g1,g2"] * T3["h0,h1,h2,p0,p1,p2"];
    C3["g0,g1,g2,p0,p1,p2"] +=
        (1.0 / 2.0) * H3["h0,h1,h2,g0,g1,g2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h3,h2"];
    C3["g0,g1,g2,p0,p1,p2"] +=
        (-1.0 / 4.0) * H3["h0,a0,a1,g0,g1,g2"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a2,a3,a0,a1"];
    C3["g0,g1,g2,p0,p1,p2"] += (-1.0 / 2.0) * H3["h0,h1,h2,g0,g1,g2"] * T3["h0,h3,h4,p0,p1,p2"] *
                               L1["h3,h1"] * L1["h4,h2"];

    C3["h0,h1,h2,g0,g1,g2"] += (1.0 / 6.0) * H3["p0,p1,p2,g0,g1,g2"] * T3["h0,h1,h2,p0,p1,p2"];
    C3["h0,h1,h2,g0,g1,g2"] +=
        (-1.0 / 2.0) * H3["p0,p1,a0,g0,g1,g2"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    C3["h0,h1,h2,g0,g1,g2"] +=
        (1.0 / 4.0) * H3["p0,a0,a1,g0,g1,g2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C3["h0,h1,h2,g0,g1,g2"] +=
        (1.0 / 2.0) * H3["p0,a0,a1,g0,g1,g2"] * T3["h0,h1,h2,p0,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggggp"});
    temp["g0,g1,g2,g3,g4,p0"] += -1.0 * H3["g3,g4,h0,g0,g1,g2"] * T1["h0,p0"];
    C3["g0,g1,g2,g3,g4,p0"] += temp["g0,g1,g2,g3,g4,p0"];
    C3["g0,g1,g2,g3,p0,g4"] -= temp["g0,g1,g2,g3,g4,p0"];
    C3["g0,g1,g2,p0,g3,g4"] += temp["g0,g1,g2,g3,g4,p0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggggpp"});
    temp["g0,g1,g2,g3,p0,p1"] += (-1.0 / 2.0) * H3["g3,h0,h1,g0,g1,g2"] * T2["h0,h1,p0,p1"];
    temp["g0,g1,g2,g3,p0,p1"] += 1.0 * H3["g3,h0,h1,g0,g1,g2"] * T2["h0,h2,p0,p1"] * L1["h2,h1"];
    temp["g0,g1,g2,g3,p0,p1"] +=
        (1.0 / 2.0) * H3["g3,h0,a0,g0,g1,g2"] * T3["h0,a2,a3,p0,p1,a1"] * L2["a2,a3,a0,a1"];
    C3["g0,g1,g2,g3,p0,p1"] += temp["g0,g1,g2,g3,p0,p1"];
    C3["g0,g1,g2,p0,g3,p1"] -= temp["g0,g1,g2,g3,p0,p1"];
    C3["g0,g1,g2,p0,p1,g3"] += temp["g0,g1,g2,g3,p0,p1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gghggg"});
    temp["g0,g1,h0,g2,g3,g4"] += 1.0 * H3["g2,g3,g4,g0,g1,p0"] * T1["h0,p0"];
    C3["g0,g1,h0,g2,g3,g4"] += temp["g0,g1,h0,g2,g3,g4"];
    C3["g0,h0,g1,g2,g3,g4"] -= temp["g0,g1,h0,g2,g3,g4"];
    C3["h0,g0,g1,g2,g3,g4"] += temp["g0,g1,h0,g2,g3,g4"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gghppp"});
    temp["g0,g1,h0,p0,p1,p2"] += (-1.0 / 2.0) * H2["h1,h2,g0,g1"] * T3["h0,h1,h2,p0,p1,p2"];
    temp["g0,g1,h0,p0,p1,p2"] += 1.0 * H2["h1,h2,g0,g1"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h3,h2"];
    temp["g0,g1,h0,p0,p1,p2"] +=
        (-1.0 / 2.0) * H3["h1,a1,a2,g0,g1,a0"] * T3["h0,h1,a3,p0,p1,p2"] * L2["a1,a2,a0,a3"];
    C3["g0,g1,h0,p0,p1,p2"] += temp["g0,g1,h0,p0,p1,p2"];
    C3["g0,h0,g1,p0,p1,p2"] -= temp["g0,g1,h0,p0,p1,p2"];
    C3["h0,g0,g1,p0,p1,p2"] += temp["g0,g1,h0,p0,p1,p2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghhggg"});
    temp["g0,h0,h1,g1,g2,g3"] += (1.0 / 2.0) * H3["g1,g2,g3,g0,p0,p1"] * T2["h0,h1,p0,p1"];
    temp["g0,h0,h1,g1,g2,g3"] += -1.0 * H3["g1,g2,g3,g0,p0,a0"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp["g0,h0,h1,g1,g2,g3"] +=
        (-1.0 / 2.0) * H3["g1,g2,g3,g0,p0,a0"] * T3["h0,h1,a3,p0,a1,a2"] * L2["a1,a2,a0,a3"];
    C3["g0,h0,h1,g1,g2,g3"] += temp["g0,h0,h1,g1,g2,g3"];
    C3["h0,g0,h1,g1,g2,g3"] -= temp["g0,h0,h1,g1,g2,g3"];
    C3["h0,h1,g0,g1,g2,g3"] += temp["g0,h0,h1,g1,g2,g3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghhppp"});
    temp["g0,h0,h1,p0,p1,p2"] += -1.0 * H1["h2,g0"] * T3["h0,h1,h2,p0,p1,p2"];
    C3["g0,h0,h1,p0,p1,p2"] += temp["g0,h0,h1,p0,p1,p2"];
    C3["h0,g0,h1,p0,p1,p2"] -= temp["g0,h0,h1,p0,p1,p2"];
    C3["h0,h1,g0,p0,p1,p2"] += temp["g0,h0,h1,p0,p1,p2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"hhhggp"});
    temp["h0,h1,h2,g0,g1,p0"] += (1.0 / 2.0) * H2["p1,p2,g0,g1"] * T3["h0,h1,h2,p0,p1,p2"];
    temp["h0,h1,h2,g0,g1,p0"] += -1.0 * H2["p1,a0,g0,g1"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    temp["h0,h1,h2,g0,g1,p0"] +=
        (1.0 / 2.0) * H3["p1,a1,a2,g0,g1,a0"] * T3["h0,h1,h2,p0,p1,a3"] * L2["a1,a2,a0,a3"];
    C3["h0,h1,h2,g0,g1,p0"] += temp["h0,h1,h2,g0,g1,p0"];
    C3["h0,h1,h2,g0,p0,g1"] -= temp["h0,h1,h2,g0,g1,p0"];
    C3["h0,h1,h2,p0,g0,g1"] += temp["h0,h1,h2,g0,g1,p0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"hhhgpp"});
    temp["h0,h1,h2,g0,p0,p1"] += 1.0 * H1["p2,g0"] * T3["h0,h1,h2,p0,p1,p2"];
    C3["h0,h1,h2,g0,p0,p1"] += temp["h0,h1,h2,g0,p0,p1"];
    C3["h0,h1,h2,p0,g0,p1"] -= temp["h0,h1,h2,g0,p0,p1"];
    C3["h0,h1,h2,p0,p1,g0"] += temp["h0,h1,h2,g0,p0,p1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gghggp"});
    temp["g0,g1,h0,g2,g3,p0"] += 1.0 * H3["g2,g3,h1,g0,g1,p1"] * T2["h0,h2,p0,p1"] * L1["h2,h1"];
    temp["g0,g1,h0,g2,g3,p0"] += -1.0 * H3["g2,g3,h1,g0,g1,a0"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp["g0,g1,h0,g2,g3,p0"] +=
        (-1.0 / 2.0) * H3["g2,g3,h1,g0,g1,a0"] * T3["h0,h1,a3,p0,a1,a2"] * L2["a1,a2,a0,a3"];
    temp["g0,g1,h0,g2,g3,p0"] +=
        (1.0 / 2.0) * H3["g2,g3,a0,g0,g1,p1"] * T3["h0,a2,a3,p0,p1,a1"] * L2["a2,a3,a0,a1"];
    C3["g0,g1,h0,g2,g3,p0"] += temp["g0,g1,h0,g2,g3,p0"];
    C3["g0,g1,h0,g2,p0,g3"] -= temp["g0,g1,h0,g2,g3,p0"];
    C3["g0,g1,h0,p0,g2,g3"] += temp["g0,g1,h0,g2,g3,p0"];
    C3["g0,h0,g1,g2,g3,p0"] -= temp["g0,g1,h0,g2,g3,p0"];
    C3["g0,h0,g1,g2,p0,g3"] += temp["g0,g1,h0,g2,g3,p0"];
    C3["g0,h0,g1,p0,g2,g3"] -= temp["g0,g1,h0,g2,g3,p0"];
    C3["h0,g0,g1,g2,g3,p0"] += temp["g0,g1,h0,g2,g3,p0"];
    C3["h0,g0,g1,g2,p0,g3"] -= temp["g0,g1,h0,g2,g3,p0"];
    C3["h0,g0,g1,p0,g2,g3"] += temp["g0,g1,h0,g2,g3,p0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gghgpp"});
    temp["g0,g1,h0,g2,p0,p1"] += 1.0 * H2["g2,h1,g0,g1"] * T2["h0,h1,p0,p1"];
    temp["g0,g1,h0,g2,p0,p1"] +=
        (1.0 / 2.0) * H3["g2,h1,h2,g0,g1,a0"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    temp["g0,g1,h0,g2,p0,p1"] +=
        1.0 * H3["g2,h1,a1,g0,g1,a0"] * T3["h0,h1,a3,p0,p1,a2"] * L2["a1,a2,a0,a3"];
    temp["g0,g1,h0,g2,p0,p1"] +=
        (1.0 / 4.0) * H3["g2,a0,a1,g0,g1,p2"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a2,a3,a0,a1"];
    temp["g0,g1,h0,g2,p0,p1"] +=
        (1.0 / 2.0) * H3["g2,h1,h2,g0,g1,p2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h3,h1"] * L1["h4,h2"];
    temp["g0,g1,h0,g2,p0,p1"] +=
        -1.0 * H3["g2,h1,h2,g0,g1,a0"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h3,h2"] * L1["a1,a0"];
    C3["g0,g1,h0,g2,p0,p1"] += temp["g0,g1,h0,g2,p0,p1"];
    C3["g0,g1,h0,p0,g2,p1"] -= temp["g0,g1,h0,g2,p0,p1"];
    C3["g0,g1,h0,p0,p1,g2"] += temp["g0,g1,h0,g2,p0,p1"];
    C3["g0,h0,g1,g2,p0,p1"] -= temp["g0,g1,h0,g2,p0,p1"];
    C3["g0,h0,g1,p0,g2,p1"] += temp["g0,g1,h0,g2,p0,p1"];
    C3["g0,h0,g1,p0,p1,g2"] -= temp["g0,g1,h0,g2,p0,p1"];
    C3["h0,g0,g1,g2,p0,p1"] += temp["g0,g1,h0,g2,p0,p1"];
    C3["h0,g0,g1,p0,g2,p1"] -= temp["g0,g1,h0,g2,p0,p1"];
    C3["h0,g0,g1,p0,p1,g2"] += temp["g0,g1,h0,g2,p0,p1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghhggp"});
    temp["g0,h0,h1,g1,g2,p0"] += -1.0 * H2["g1,g2,g0,p1"] * T2["h0,h1,p0,p1"];
    temp["g0,h0,h1,g1,g2,p0"] +=
        (-1.0 / 2.0) * H3["g1,g2,h2,g0,p1,p2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h3,h2"];
    temp["g0,h0,h1,g1,g2,p0"] +=
        (-1.0 / 4.0) * H3["g1,g2,h2,g0,a0,a1"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,h1,g1,g2,p0"] +=
        -1.0 * H3["g1,g2,a1,g0,p1,a0"] * T3["h0,h1,a3,p0,p1,a2"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,h1,g1,g2,p0"] +=
        1.0 * H3["g1,g2,h2,g0,p1,a0"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h3,h2"] * L1["a1,a0"];
    temp["g0,h0,h1,g1,g2,p0"] += (-1.0 / 2.0) * H3["g1,g2,h2,g0,a0,a1"] * T3["h0,h1,h2,p0,a2,a3"] *
                                 L1["a2,a0"] * L1["a3,a1"];
    C3["g0,h0,h1,g1,g2,p0"] += temp["g0,h0,h1,g1,g2,p0"];
    C3["g0,h0,h1,g1,p0,g2"] -= temp["g0,h0,h1,g1,g2,p0"];
    C3["g0,h0,h1,p0,g1,g2"] += temp["g0,h0,h1,g1,g2,p0"];
    C3["h0,g0,h1,g1,g2,p0"] -= temp["g0,h0,h1,g1,g2,p0"];
    C3["h0,g0,h1,g1,p0,g2"] += temp["g0,h0,h1,g1,g2,p0"];
    C3["h0,g0,h1,p0,g1,g2"] -= temp["g0,h0,h1,g1,g2,p0"];
    C3["h0,h1,g0,g1,g2,p0"] += temp["g0,h0,h1,g1,g2,p0"];
    C3["h0,h1,g0,g1,p0,g2"] -= temp["g0,h0,h1,g1,g2,p0"];
    C3["h0,h1,g0,p0,g1,g2"] += temp["g0,h0,h1,g1,g2,p0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghhgpp"});
    temp["g0,h0,h1,g1,p0,p1"] += 1.0 * H2["g1,h2,g0,p2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h3,h2"];
    temp["g0,h0,h1,g1,p0,p1"] += -1.0 * H2["g1,h2,g0,a0"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    temp["g0,h0,h1,g1,p0,p1"] +=
        (1.0 / 2.0) * H3["g1,h2,a2,g0,a0,a1"] * T3["h0,h1,h2,p0,p1,a3"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,h1,g1,p0,p1"] +=
        (-1.0 / 2.0) * H3["g1,a1,a2,g0,p2,a0"] * T3["h0,h1,a3,p0,p1,p2"] * L2["a1,a2,a0,a3"];
    C3["g0,h0,h1,g1,p0,p1"] += temp["g0,h0,h1,g1,p0,p1"];
    C3["g0,h0,h1,p0,g1,p1"] -= temp["g0,h0,h1,g1,p0,p1"];
    C3["g0,h0,h1,p0,p1,g1"] += temp["g0,h0,h1,g1,p0,p1"];
    C3["h0,g0,h1,g1,p0,p1"] -= temp["g0,h0,h1,g1,p0,p1"];
    C3["h0,g0,h1,p0,g1,p1"] += temp["g0,h0,h1,g1,p0,p1"];
    C3["h0,g0,h1,p0,p1,g1"] -= temp["g0,h0,h1,g1,p0,p1"];
    C3["h0,h1,g0,g1,p0,p1"] += temp["g0,h0,h1,g1,p0,p1"];
    C3["h0,h1,g0,p0,g1,p1"] -= temp["g0,h0,h1,g1,p0,p1"];
    C3["h0,h1,g0,p0,p1,g1"] += temp["g0,h0,h1,g1,p0,p1"];

    // scale by factor
    C0 *= factor;
    C1.scale(factor);
    C2.scale(factor);
    C3.scale(factor);

    // add T dagger
    C0 *= 2.0;
    H1["pq"] = C1["pq"];
    C1["pq"] += H1["qp"];
    H2["pqrs"] = C2["pqrs"];
    C2["pqrs"] += H2["rspq"];
    H3["g0,g1,g2,g3,g4,g5"] = C3["g0,g1,g2,g3,g4,g5"];
    C3["g0,g1,g2,g3,g4,g5"] += H3["g3,g4,g5,g0,g1,g2"];
}

} // namespace forte
