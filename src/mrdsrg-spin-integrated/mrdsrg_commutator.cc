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
#include <map>
#include <vector>

#include "../helpers.h"
#include "../mini-boost/boost/format.hpp"
#include "mrdsrg.h"

#define TIME_LINE(x)                                                                               \
    timer_on(#x);                                                                                  \
    x;                                                                                             \
    timer_off(#x)

namespace psi {
namespace forte {

void MRDSRG::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    Timer timer;

    double E = 0.0;
    E += H1["em"] * T1["me"];
    E += H1["ex"] * T1["ye"] * Gamma1_["xy"];
    E += H1["xm"] * T1["my"] * Eta1_["yx"];

    E += H1["EM"] * T1["ME"];
    E += H1["EX"] * T1["YE"] * Gamma1_["XY"];
    E += H1["XM"] * T1["MY"] * Eta1_["YX"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("110", timer.get());
}

void MRDSRG::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    Timer timer;
    BlockedTensor temp;
    double E = 0.0;

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"}));
    TIME_LINE(temp["uvxy"] += H1["ex"] * T2["uvey"]);
    TIME_LINE(temp["uvxy"] -= H1["vm"] * T2["umxy"]);
    TIME_LINE(E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"}));
    TIME_LINE(temp["UVXY"] += H1["EX"] * T2["UVEY"]);
    TIME_LINE(temp["UVXY"] -= H1["VM"] * T2["UMXY"]);
    TIME_LINE(E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaA"}));
    TIME_LINE(temp["uVxY"] += H1["ex"] * T2["uVeY"]);
    TIME_LINE(temp["uVxY"] += H1["EY"] * T2["uVxE"]);
    TIME_LINE(temp["uVxY"] -= H1["VM"] * T2["uMxY"]);
    TIME_LINE(temp["uVxY"] -= H1["um"] * T2["mVxY"]);
    TIME_LINE(E += temp["uVxY"] * Lambda2_["xYuV"]);

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("120", timer.get());
}

void MRDSRG::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    Timer timer;
    BlockedTensor temp;
    double E = 0.0;

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] += H2["evxy"] * T1["ue"];
    temp["uvxy"] -= H2["uvmy"] * T1["mx"];
    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"});
    temp["UVXY"] += H2["EVXY"] * T1["UE"];
    temp["UVXY"] -= H2["UVMY"] * T1["MX"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaA"});
    temp["uVxY"] += H2["eVxY"] * T1["ue"];
    temp["uVxY"] += H2["uExY"] * T1["VE"];
    temp["uVxY"] -= H2["uVmY"] * T1["mx"];
    temp["uVxY"] -= H2["uVxM"] * T1["MY"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("210", timer.get());
}

void MRDSRG::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
    Timer timer;

    // <[Hbar2, T2]> (C_2)^4
    TIME_LINE(double E = H2["eFmN"] * T2["mNeF"]);
    TIME_LINE(E += 0.25 * H2["efmn"] * T2["mnef"]);
    TIME_LINE(E += 0.25 * H2["EFMN"] * T2["MNEF"]);

    TIME_LINE(BlockedTensor temp =
                  ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aa"})));
    TIME_LINE(temp["vu"] += 0.5 * H2["efmu"] * T2["mvef"]);
    TIME_LINE(temp["vu"] += H2["fEuM"] * T2["vMfE"]);
    TIME_LINE(temp["VU"] += 0.5 * H2["EFMU"] * T2["MVEF"]);
    TIME_LINE(temp["VU"] += H2["eFmU"] * T2["mVeF"]);
    TIME_LINE(E += temp["vu"] * Gamma1_["uv"]);
    TIME_LINE(E += temp["VU"] * Gamma1_["UV"]);

    TIME_LINE(temp.zero());
    TIME_LINE(temp["vu"] += 0.5 * H2["vemn"] * T2["mnue"]);
    TIME_LINE(temp["vu"] += H2["vEmN"] * T2["mNuE"]);
    TIME_LINE(temp["VU"] += 0.5 * H2["VEMN"] * T2["MNUE"]);
    TIME_LINE(temp["VU"] += H2["eVnM"] * T2["nMeU"]);
    TIME_LINE(E += temp["vu"] * Eta1_["uv"]);
    TIME_LINE(E += temp["VU"] * Eta1_["UV"]);

    TIME_LINE(temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"})));
    TIME_LINE(temp["yvxu"] += H2["efxu"] * T2["yvef"]);
    TIME_LINE(temp["yVxU"] += H2["eFxU"] * T2["yVeF"]);
    TIME_LINE(temp["YVXU"] += H2["EFXU"] * T2["YVEF"]);
    TIME_LINE(E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"]);
    TIME_LINE(E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"]);
    TIME_LINE(E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"]);

    TIME_LINE(temp.zero());
    TIME_LINE(temp["vyux"] += H2["vymn"] * T2["mnux"]);
    TIME_LINE(temp["vYuX"] += H2["vYmN"] * T2["mNuX"]);
    TIME_LINE(temp["VYUX"] += H2["VYMN"] * T2["MNUX"]);
    TIME_LINE(E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"]);
    TIME_LINE(E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"]);
    TIME_LINE(E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"]);

    TIME_LINE(temp.zero());
    TIME_LINE(temp["vyux"] += H2["vemx"] * T2["myue"]);
    TIME_LINE(temp["vyux"] += H2["vExM"] * T2["yMuE"]);
    TIME_LINE(temp["VYUX"] += H2["eVmX"] * T2["mYeU"]);
    TIME_LINE(temp["VYUX"] += H2["VEXM"] * T2["YMUE"]);
    TIME_LINE(E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"]);
    TIME_LINE(E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"]);
    TIME_LINE(temp["yVxU"] = H2["eVxM"] * T2["yMeU"]);
    TIME_LINE(E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"]);
    TIME_LINE(temp["vYuX"] = H2["vEmX"] * T2["mYuE"]);
    TIME_LINE(E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"]);

    TIME_LINE(temp.zero());
    TIME_LINE(temp["yvxu"] += 0.5 * Gamma1_["wz"] * H2["vexw"] * T2["yzue"]);
    TIME_LINE(temp["yvxu"] += Gamma1_["WZ"] * H2["vExW"] * T2["yZuE"]);
    TIME_LINE(temp["yvxu"] += 0.5 * Eta1_["wz"] * T2["myuw"] * H2["vzmx"]);
    TIME_LINE(temp["yvxu"] += Eta1_["WZ"] * T2["yMuW"] * H2["vZxM"]);
    TIME_LINE(E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"]);

    TIME_LINE(temp["YVXU"] += 0.5 * Gamma1_["WZ"] * H2["VEXW"] * T2["YZUE"]);
    TIME_LINE(temp["YVXU"] += Gamma1_["wz"] * H2["eVwX"] * T2["zYeU"]);
    TIME_LINE(temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2["MYUW"] * H2["VZMX"]);
    TIME_LINE(temp["YVXU"] += Eta1_["wz"] * H2["zVmX"] * T2["mYwU"]);
    TIME_LINE(E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"]);

    // <[Hbar2, T2]> C_4 (C_2)^2 HH -- combined with PH
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"})));
    TIME_LINE(temp["uvxy"] += 0.125 * H2["uvmn"] * T2["mnxy"]);
    TIME_LINE(temp["uvxy"] += 0.25 * Gamma1_["wz"] * H2["uvmw"] * T2["mzxy"]);
    TIME_LINE(temp["uVxY"] += H2["uVmN"] * T2["mNxY"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["wz"] * T2["zMxY"] * H2["uVwM"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["WZ"] * H2["uVmW"] * T2["mZxY"]);
    TIME_LINE(temp["UVXY"] += 0.125 * H2["UVMN"] * T2["MNXY"]);
    TIME_LINE(temp["UVXY"] += 0.25 * Gamma1_["WZ"] * H2["UVMW"] * T2["MZXY"]);

    // <[Hbar2, T2]> C_4 (C_2)^2 PP -- combined with PH
    TIME_LINE(temp["uvxy"] += 0.125 * H2["efxy"] * T2["uvef"]);
    TIME_LINE(temp["uvxy"] += 0.25 * Eta1_["wz"] * T2["uvew"] * H2["ezxy"]);
    TIME_LINE(temp["uVxY"] += H2["eFxY"] * T2["uVeF"]);
    TIME_LINE(temp["uVxY"] += Eta1_["wz"] * H2["zExY"] * T2["uVwE"]);
    TIME_LINE(temp["uVxY"] += Eta1_["WZ"] * T2["uVeW"] * H2["eZxY"]);
    TIME_LINE(temp["UVXY"] += 0.125 * H2["EFXY"] * T2["UVEF"]);
    TIME_LINE(temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2["UVEW"] * H2["EZXY"]);

    // <[Hbar2, T2]> C_4 (C_2)^2 PH
    TIME_LINE(temp["uvxy"] += H2["eumx"] * T2["mvey"]);
    TIME_LINE(temp["uvxy"] += H2["uExM"] * T2["vMyE"]);
    TIME_LINE(temp["uvxy"] += Gamma1_["wz"] * T2["zvey"] * H2["euwx"]);
    TIME_LINE(temp["uvxy"] += Gamma1_["WZ"] * H2["uExW"] * T2["vZyE"]);
    TIME_LINE(temp["uvxy"] += Eta1_["zw"] * H2["wumx"] * T2["mvzy"]);
    TIME_LINE(temp["uvxy"] += Eta1_["ZW"] * T2["vMyZ"] * H2["uWxM"]);
    TIME_LINE(E += temp["uvxy"] * Lambda2_["xyuv"]);

    TIME_LINE(temp["UVXY"] += H2["eUmX"] * T2["mVeY"]);
    TIME_LINE(temp["UVXY"] += H2["EUMX"] * T2["MVEY"]);
    TIME_LINE(temp["UVXY"] += Gamma1_["wz"] * T2["zVeY"] * H2["eUwX"]);
    TIME_LINE(temp["UVXY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["EUWX"]);
    TIME_LINE(temp["UVXY"] += Eta1_["zw"] * H2["wUmX"] * T2["mVzY"]);
    TIME_LINE(temp["UVXY"] += Eta1_["ZW"] * H2["WUMX"] * T2["MVZY"]);
    TIME_LINE(E += temp["UVXY"] * Lambda2_["XYUV"]);

    TIME_LINE(temp["uVxY"] += H2["uexm"] * T2["mVeY"]);
    TIME_LINE(temp["uVxY"] += H2["uExM"] * T2["MVEY"]);
    TIME_LINE(temp["uVxY"] -= H2["eVxM"] * T2["uMeY"]);
    TIME_LINE(temp["uVxY"] -= H2["uEmY"] * T2["mVxE"]);
    TIME_LINE(temp["uVxY"] += H2["eVmY"] * T2["umxe"]);
    TIME_LINE(temp["uVxY"] += H2["EVMY"] * T2["uMxE"]);

    TIME_LINE(temp["uVxY"] += Gamma1_["wz"] * T2["zVeY"] * H2["uexw"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["uExW"]);
    TIME_LINE(temp["uVxY"] -= Gamma1_["WZ"] * H2["eVxW"] * T2["uZeY"]);
    TIME_LINE(temp["uVxY"] -= Gamma1_["wz"] * T2["zVxE"] * H2["uEwY"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["wz"] * T2["zuex"] * H2["eVwY"]);
    TIME_LINE(temp["uVxY"] -= Gamma1_["WZ"] * H2["EVYW"] * T2["uZxE"]);

    TIME_LINE(temp["uVxY"] += Eta1_["zw"] * H2["wumx"] * T2["mVzY"]);
    TIME_LINE(temp["uVxY"] += Eta1_["ZW"] * T2["VMYZ"] * H2["uWxM"]);
    TIME_LINE(temp["uVxY"] -= Eta1_["zw"] * H2["wVxM"] * T2["uMzY"]);
    TIME_LINE(temp["uVxY"] -= Eta1_["ZW"] * T2["mVxZ"] * H2["uWmY"]);
    TIME_LINE(temp["uVxY"] += Eta1_["zw"] * T2["umxz"] * H2["wVmY"]);
    TIME_LINE(temp["uVxY"] += Eta1_["ZW"] * H2["WVMY"] * T2["uMxZ"]);
    TIME_LINE(E += temp["uVxY"] * Lambda2_["xYuV"]);

    // <[Hbar2, T2]> C_6 C_2
    if (options_.get_str("THREEPDC") != "ZERO") {
        TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"}));
        TIME_LINE(temp["uvwxyz"] += H2["uviz"] * T2["iwxy"]); //  aaaaaa from hole
        TIME_LINE(temp["uvwxyz"] += H2["waxy"] * T2["uvaz"]); //  aaaaaa from particle
        TIME_LINE(E += 0.25 * temp["uvwxyz"] * Lambda3_["xyzuvw"]);

        TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"}));
        TIME_LINE(temp["UVWXYZ"] += H2["UVIZ"] * T2["IWXY"]); //  AAAAAA from hole
        TIME_LINE(temp["UVWXYZ"] += H2["WAXY"] * T2["UVAZ"]); //  AAAAAA from particle
        TIME_LINE(E += 0.25 * temp["UVWXYZ"] * Lambda3_["XYZUVW"]);

        TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"}));
        TIME_LINE(temp["uvWxyZ"] -= H2["uviy"] * T2["iWxZ"]);       //  aaAaaA from hole
        TIME_LINE(temp["uvWxyZ"] -= H2["uWiZ"] * T2["ivxy"]);       //  aaAaaA from hole
        TIME_LINE(temp["uvWxyZ"] += 2.0 * H2["uWyI"] * T2["vIxZ"]); //  aaAaaA from hole

        TIME_LINE(temp["uvWxyZ"] += H2["aWxZ"] * T2["uvay"]);       //  aaAaaA from particle
        TIME_LINE(temp["uvWxyZ"] -= H2["vaxy"] * T2["uWaZ"]);       //  aaAaaA from particle
        TIME_LINE(temp["uvWxyZ"] -= 2.0 * H2["vAxZ"] * T2["uWyA"]); //  aaAaaA from particle
        TIME_LINE(E += 0.5 * temp["uvWxyZ"] * Lambda3_["xyZuvW"]);

        TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"}));
        TIME_LINE(temp["uVWxYZ"] -= H2["VWIZ"] * T2["uIxY"]);       //  aAAaAA from hole
        TIME_LINE(temp["uVWxYZ"] -= H2["uVxI"] * T2["IWYZ"]);       //  aAAaAA from hole
        TIME_LINE(temp["uVWxYZ"] += 2.0 * H2["uViZ"] * T2["iWxY"]); //  aAAaAA from hole

        TIME_LINE(temp["uVWxYZ"] += H2["uAxY"] * T2["VWAZ"]);       //  aAAaAA from particle
        TIME_LINE(temp["uVWxYZ"] -= H2["WAYZ"] * T2["uVxA"]);       //  aAAaAA from particle
        TIME_LINE(temp["uVWxYZ"] -= 2.0 * H2["aWxY"] * T2["uVaZ"]); //  aAAaAA from particle
        TIME_LINE(E += 0.5 * temp["uVWxYZ"] * Lambda3_["xYZuVW"]);
    }

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
}

void MRDSRG::H2_T2_C0_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, double& C0) {
    Timer timer;

    // <[Hbar2, T2]> (C_2)^4
    TIME_LINE(double E = B["gem"] * B["gFN"] * T2["mNeF"]);
    TIME_LINE(E += 0.25 * B["gem"] * B["gfn"] * T2["mnef"]);
    TIME_LINE(E -= 0.25 * B["gen"] * B["gfm"] * T2["mnef"]);
    TIME_LINE(E += 0.25 * B["gEM"] * B["gFN"] * T2["MNEF"]);
    TIME_LINE(E -= 0.25 * B["gEN"] * B["gFM"] * T2["MNEF"]);

    TIME_LINE(BlockedTensor temp =
                  ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aa"})));
    TIME_LINE(temp["vu"] += 0.5 * B["gem"] * B["gfu"] * T2["mvef"]);
    TIME_LINE(temp["vu"] -= 0.5 * B["geu"] * B["gfm"] * T2["mvef"]);
    TIME_LINE(temp["vu"] += B["gfu"] * B["gEM"] * T2["vMfE"]);
    TIME_LINE(temp["VU"] += 0.5 * B["gEM"] * B["gFU"] * T2["MVEF"]);
    TIME_LINE(temp["VU"] -= 0.5 * B["gEU"] * B["gFM"] * T2["MVEF"]);
    TIME_LINE(temp["VU"] += B["gem"] * B["gFU"] * T2["mVeF"]);
    TIME_LINE(E += temp["vu"] * Gamma1_["uv"]);
    TIME_LINE(E += temp["VU"] * Gamma1_["UV"]);

    TIME_LINE(temp.zero());
    TIME_LINE(temp["vu"] += 0.5 * B["gvm"] * B["gen"] * T2["mnue"]);
    TIME_LINE(temp["vu"] -= 0.5 * B["gvn"] * B["gem"] * T2["mnue"]);
    TIME_LINE(temp["vu"] += B["gvm"] * B["gEN"] * T2["mNuE"]);
    TIME_LINE(temp["VU"] += 0.5 * B["gVM"] * B["gEN"] * T2["MNUE"]);
    TIME_LINE(temp["VU"] -= 0.5 * B["gVN"] * B["gEM"] * T2["MNUE"]);
    TIME_LINE(temp["VU"] += B["gen"] * B["gVM"] * T2["nMeU"]);
    TIME_LINE(E += temp["vu"] * Eta1_["uv"]);
    TIME_LINE(E += temp["VU"] * Eta1_["UV"]);

    TIME_LINE(temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"})));
    TIME_LINE(temp["yvxu"] += B["gex"] * B["gfu"] * T2["yvef"]);
    TIME_LINE(temp["yvxu"] -= B["geu"] * B["gfx"] * T2["yvef"]);
    TIME_LINE(temp["yVxU"] += B["gex"] * B["gFU"] * T2["yVeF"]);
    TIME_LINE(temp["YVXU"] += B["gEX"] * B["gFU"] * T2["YVEF"]);
    TIME_LINE(temp["YVXU"] -= B["gEU"] * B["gFX"] * T2["YVEF"]);
    TIME_LINE(E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"]);
    TIME_LINE(E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"]);
    TIME_LINE(E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"]);

    TIME_LINE(temp.zero());
    TIME_LINE(temp["vyux"] += B["gvm"] * B["gyn"] * T2["mnux"]);
    TIME_LINE(temp["vyux"] -= B["gvn"] * B["gym"] * T2["mnux"]);
    TIME_LINE(temp["vYuX"] += B["gvm"] * B["gYN"] * T2["mNuX"]);
    TIME_LINE(temp["VYUX"] += B["gVM"] * B["gYN"] * T2["MNUX"]);
    TIME_LINE(temp["VYUX"] -= B["gVN"] * B["gYM"] * T2["MNUX"]);
    TIME_LINE(E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"]);
    TIME_LINE(E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"]);
    TIME_LINE(E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"]);

    TIME_LINE(temp.zero());
    TIME_LINE(temp["vyux"] += B["gvm"] * B["gex"] * T2["myue"]);
    TIME_LINE(temp["vyux"] -= B["gvx"] * B["gem"] * T2["myue"]);
    TIME_LINE(temp["vyux"] += B["gvx"] * B["gEM"] * T2["yMuE"]);
    TIME_LINE(temp["VYUX"] += B["gem"] * B["gVX"] * T2["mYeU"]);
    TIME_LINE(temp["VYUX"] += B["gVX"] * B["gEM"] * T2["YMUE"]);
    TIME_LINE(temp["VYUX"] -= B["gVM"] * B["gEX"] * T2["YMUE"]);
    TIME_LINE(E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"]);
    TIME_LINE(E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"]);
    TIME_LINE(temp["yVxU"] = B["gex"] * B["gVM"] * T2["yMeU"]);
    TIME_LINE(E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"]);
    TIME_LINE(temp["vYuX"] = B["gvm"] * B["gEX"] * T2["mYuE"]);
    TIME_LINE(E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"]);

    TIME_LINE(temp.zero());
    TIME_LINE(temp["yvxu"] += 0.5 * Gamma1_["wz"] * B["gvx"] * B["gew"] * T2["yzue"]);
    TIME_LINE(temp["yvxu"] -= 0.5 * Gamma1_["wz"] * B["gvw"] * B["gex"] * T2["yzue"]);
    TIME_LINE(temp["yvxu"] += Gamma1_["WZ"] * B["gvx"] * B["gEW"] * T2["yZuE"]);
    TIME_LINE(temp["yvxu"] += 0.5 * Eta1_["wz"] * T2["myuw"] * B["gvm"] * B["gzx"]);
    TIME_LINE(temp["yvxu"] -= 0.5 * Eta1_["wz"] * T2["myuw"] * B["gvx"] * B["gzm"]);
    TIME_LINE(temp["yvxu"] += Eta1_["WZ"] * T2["yMuW"] * B["gvx"] * B["gZM"]);
    TIME_LINE(E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"]);

    TIME_LINE(temp["YVXU"] += 0.5 * Gamma1_["WZ"] * B["gVX"] * B["gEW"] * T2["YZUE"]);
    TIME_LINE(temp["YVXU"] -= 0.5 * Gamma1_["WZ"] * B["gVW"] * B["gEX"] * T2["YZUE"]);
    TIME_LINE(temp["YVXU"] += Gamma1_["wz"] * B["gew"] * B["gVX"] * T2["zYeU"]);
    TIME_LINE(temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2["MYUW"] * B["gVM"] * B["gZX"]);
    TIME_LINE(temp["YVXU"] -= 0.5 * Eta1_["WZ"] * T2["MYUW"] * B["gVX"] * B["gZM"]);
    TIME_LINE(temp["YVXU"] += Eta1_["wz"] * B["gzm"] * B["gVX"] * T2["mYwU"]);
    TIME_LINE(E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"]);

    // <[Hbar2, T2]> C_4 (C_2)^2 HH -- combined with PH
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"})));
    TIME_LINE(temp["uvxy"] += 0.125 * B["gum"] * B["gvn"] * T2["mnxy"]);
    TIME_LINE(temp["uvxy"] -= 0.125 * B["gun"] * B["gvm"] * T2["mnxy"]);
    TIME_LINE(temp["uvxy"] += 0.25 * Gamma1_["wz"] * B["gum"] * B["gvw"] * T2["mzxy"]);
    TIME_LINE(temp["uvxy"] -= 0.25 * Gamma1_["wz"] * B["guw"] * B["gvm"] * T2["mzxy"]);
    TIME_LINE(temp["uVxY"] += B["gum"] * B["gVN"] * T2["mNxY"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["wz"] * T2["zMxY"] * B["guw"] * B["gVM"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["WZ"] * B["gum"] * B["gVW"] * T2["mZxY"]);
    TIME_LINE(temp["UVXY"] += 0.125 * B["gUM"] * B["gVN"] * T2["MNXY"]);
    TIME_LINE(temp["UVXY"] -= 0.125 * B["gUN"] * B["gVM"] * T2["MNXY"]);
    TIME_LINE(temp["UVXY"] += 0.25 * Gamma1_["WZ"] * B["gUM"] * B["gVW"] * T2["MZXY"]);
    TIME_LINE(temp["UVXY"] -= 0.25 * Gamma1_["WZ"] * B["gUW"] * B["gVM"] * T2["MZXY"]);

    // <[Hbar2, T2]> C_4 (C_2)^2 PP -- combined with PH
    TIME_LINE(temp["uvxy"] += 0.125 * B["gex"] * B["gfy"] * T2["uvef"]);
    TIME_LINE(temp["uvxy"] -= 0.125 * B["gey"] * B["gfx"] * T2["uvef"]);
    TIME_LINE(temp["uvxy"] += 0.25 * Eta1_["wz"] * T2["uvew"] * B["gex"] * B["gzy"]);
    TIME_LINE(temp["uvxy"] -= 0.25 * Eta1_["wz"] * T2["uvew"] * B["gey"] * B["gzx"]);
    TIME_LINE(temp["uVxY"] += B["gex"] * B["gFY"] * T2["uVeF"]);
    TIME_LINE(temp["uVxY"] += Eta1_["wz"] * B["gzx"] * B["gEY"] * T2["uVwE"]);
    TIME_LINE(temp["uVxY"] += Eta1_["WZ"] * T2["uVeW"] * B["gex"] * B["gZY"]);
    TIME_LINE(temp["UVXY"] += 0.125 * B["gEX"] * B["gFY"] * T2["UVEF"]);
    TIME_LINE(temp["UVXY"] -= 0.125 * B["gEY"] * B["gFX"] * T2["UVEF"]);
    TIME_LINE(temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2["UVEW"] * B["gEX"] * B["gZY"]);
    TIME_LINE(temp["UVXY"] -= 0.25 * Eta1_["WZ"] * T2["UVEW"] * B["gEY"] * B["gZX"]);

    // <[Hbar2, T2]> C_4 (C_2)^2 PH
    TIME_LINE(temp["uvxy"] += B["gem"] * B["gux"] * T2["mvey"]);
    TIME_LINE(temp["uvxy"] -= B["gex"] * B["gum"] * T2["mvey"]);
    TIME_LINE(temp["uvxy"] += B["gux"] * B["gEM"] * T2["vMyE"]);
    TIME_LINE(temp["uvxy"] += Gamma1_["wz"] * T2["zvey"] * B["gew"] * B["gux"]);
    TIME_LINE(temp["uvxy"] -= Gamma1_["wz"] * T2["zvey"] * B["gex"] * B["guw"]);
    TIME_LINE(temp["uvxy"] += Gamma1_["WZ"] * B["gux"] * B["gEW"] * T2["vZyE"]);
    TIME_LINE(temp["uvxy"] += Eta1_["zw"] * B["gwm"] * B["gux"] * T2["mvzy"]);
    TIME_LINE(temp["uvxy"] -= Eta1_["zw"] * B["gwx"] * B["gum"] * T2["mvzy"]);
    TIME_LINE(temp["uvxy"] += Eta1_["ZW"] * T2["vMyZ"] * B["gux"] * B["gWM"]);
    TIME_LINE(E += temp["uvxy"] * Lambda2_["xyuv"]);

    TIME_LINE(temp["UVXY"] += B["gem"] * B["gUX"] * T2["mVeY"]);
    TIME_LINE(temp["UVXY"] += B["gEM"] * B["gUX"] * T2["MVEY"]);
    TIME_LINE(temp["UVXY"] -= B["gEX"] * B["gUM"] * T2["MVEY"]);
    TIME_LINE(temp["UVXY"] += Gamma1_["wz"] * T2["zVeY"] * B["gew"] * B["gUX"]);
    TIME_LINE(temp["UVXY"] += Gamma1_["WZ"] * T2["ZVEY"] * B["gEW"] * B["gUX"]);
    TIME_LINE(temp["UVXY"] -= Gamma1_["WZ"] * T2["ZVEY"] * B["gEX"] * B["gUW"]);
    TIME_LINE(temp["UVXY"] += Eta1_["zw"] * B["gwm"] * B["gUX"] * T2["mVzY"]);
    TIME_LINE(temp["UVXY"] += Eta1_["ZW"] * B["gWM"] * B["gUX"] * T2["MVZY"]);
    TIME_LINE(temp["UVXY"] -= Eta1_["ZW"] * B["gWX"] * B["gUM"] * T2["MVZY"]);
    TIME_LINE(E += temp["UVXY"] * Lambda2_["XYUV"]);

    TIME_LINE(temp["uVxY"] += B["gux"] * B["gem"] * T2["mVeY"]);
    TIME_LINE(temp["uVxY"] -= B["gum"] * B["gex"] * T2["mVeY"]);
    TIME_LINE(temp["uVxY"] += B["gux"] * B["gEM"] * T2["MVEY"]);
    TIME_LINE(temp["uVxY"] -= B["gex"] * B["gVM"] * T2["uMeY"]);
    TIME_LINE(temp["uVxY"] -= B["gum"] * B["gEY"] * T2["mVxE"]);
    TIME_LINE(temp["uVxY"] += B["gem"] * B["gVY"] * T2["umxe"]);
    TIME_LINE(temp["uVxY"] += B["gEM"] * B["gVY"] * T2["uMxE"]);
    TIME_LINE(temp["uVxY"] -= B["gEY"] * B["gVM"] * T2["uMxE"]);

    TIME_LINE(temp["uVxY"] += Gamma1_["wz"] * T2["zVeY"] * B["gux"] * B["gew"]);
    TIME_LINE(temp["uVxY"] -= Gamma1_["wz"] * T2["zVeY"] * B["guw"] * B["gex"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["WZ"] * T2["ZVEY"] * B["gux"] * B["gEW"]);
    TIME_LINE(temp["uVxY"] -= Gamma1_["WZ"] * B["gex"] * B["gVW"] * T2["uZeY"]);
    TIME_LINE(temp["uVxY"] -= Gamma1_["wz"] * T2["zVxE"] * B["guw"] * B["gEY"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["wz"] * T2["zuex"] * B["gew"] * B["gVY"]);
    TIME_LINE(temp["uVxY"] -= Gamma1_["WZ"] * B["gEY"] * B["gVW"] * T2["uZxE"]);
    TIME_LINE(temp["uVxY"] += Gamma1_["WZ"] * B["gEW"] * B["gVY"] * T2["uZxE"]);

    TIME_LINE(temp["uVxY"] += Eta1_["zw"] * B["gwm"] * B["gux"] * T2["mVzY"]);
    TIME_LINE(temp["uVxY"] -= Eta1_["zw"] * B["gwx"] * B["gum"] * T2["mVzY"]);
    TIME_LINE(temp["uVxY"] += Eta1_["ZW"] * T2["VMYZ"] * B["gux"] * B["gWM"]);
    TIME_LINE(temp["uVxY"] -= Eta1_["zw"] * B["gwx"] * B["gVM"] * T2["uMzY"]);
    TIME_LINE(temp["uVxY"] -= Eta1_["ZW"] * T2["mVxZ"] * B["gum"] * B["gWY"]);
    TIME_LINE(temp["uVxY"] += Eta1_["zw"] * T2["umxz"] * B["gwm"] * B["gVY"]);
    TIME_LINE(temp["uVxY"] += Eta1_["ZW"] * B["gWM"] * B["gVY"] * T2["uMxZ"]);
    TIME_LINE(temp["uVxY"] -= Eta1_["ZW"] * B["gWY"] * B["gVM"] * T2["uMxZ"]);
    TIME_LINE(E += temp["uVxY"] * Lambda2_["xYuV"]);

    // <[Hbar2, T2]> C_6 C_2
    if (options_.get_str("THREEPDC") != "ZERO") {
        TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"}));
        TIME_LINE(temp["uvwxyz"] += B["gui"] * B["gvz"] * T2["iwxy"]); //  aaaaaa from hole
        TIME_LINE(temp["uvwxyz"] -= B["guz"] * B["gvi"] * T2["iwxy"]); //  aaaaaa from hole
        TIME_LINE(temp["uvwxyz"] += B["gwx"] * B["gay"] * T2["uvaz"]); //  aaaaaa from particle
        TIME_LINE(temp["uvwxyz"] -= B["gwy"] * B["gax"] * T2["uvaz"]); //  aaaaaa from particle
        TIME_LINE(E += 0.25 * temp["uvwxyz"] * Lambda3_["xyzuvw"]);

        TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"}));
        TIME_LINE(temp["UVWXYZ"] += B["gUI"] * B["gVZ"] * T2["IWXY"]); //  AAAAAA from hole
        TIME_LINE(temp["UVWXYZ"] -= B["gUZ"] * B["gVI"] * T2["IWXY"]); //  AAAAAA from hole
        TIME_LINE(temp["UVWXYZ"] += B["gWX"] * B["gAY"] * T2["UVAZ"]); //  AAAAAA from particle
        TIME_LINE(temp["UVWXYZ"] -= B["gWY"] * B["gAX"] * T2["UVAZ"]); //  AAAAAA from particle
        TIME_LINE(E += 0.25 * temp["UVWXYZ"] * Lambda3_["XYZUVW"]);

        TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"}));
        TIME_LINE(temp["uvWxyZ"] -= B["gui"] * B["gvy"] * T2["iWxZ"]);       //  aaAaaA from hole
        TIME_LINE(temp["uvWxyZ"] += B["guy"] * B["gvi"] * T2["iWxZ"]);       //  aaAaaA from hole
        TIME_LINE(temp["uvWxyZ"] -= B["gui"] * B["gWZ"] * T2["ivxy"]);       //  aaAaaA from hole
        TIME_LINE(temp["uvWxyZ"] += 2.0 * B["guy"] * B["gWI"] * T2["vIxZ"]); //  aaAaaA from hole

        TIME_LINE(temp["uvWxyZ"] += B["gax"] * B["gWZ"] * T2["uvay"]); //  aaAaaA from particle
        TIME_LINE(temp["uvWxyZ"] -= B["gvx"] * B["gay"] * T2["uWaZ"]); //  aaAaaA from particle
        TIME_LINE(temp["uvWxyZ"] += B["gvy"] * B["gax"] * T2["uWaZ"]); //  aaAaaA from particle
        TIME_LINE(temp["uvWxyZ"] -=
                  2.0 * B["gvx"] * B["gAZ"] * T2["uWyA"]); //  aaAaaA from particle
        TIME_LINE(E += 0.5 * temp["uvWxyZ"] * Lambda3_["xyZuvW"]);

        TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"}));
        TIME_LINE(temp["uVWxYZ"] -= B["gVI"] * B["gWZ"] * T2["uIxY"]);       //  aAAaAA from hole
        TIME_LINE(temp["uVWxYZ"] += B["gVZ"] * B["gWI"] * T2["uIxY"]);       //  aAAaAA from hole
        TIME_LINE(temp["uVWxYZ"] -= B["gux"] * B["gVI"] * T2["IWYZ"]);       //  aAAaAA from hole
        TIME_LINE(temp["uVWxYZ"] += 2.0 * B["gui"] * B["gVZ"] * T2["iWxY"]); //  aAAaAA from hole

        TIME_LINE(temp["uVWxYZ"] += B["gux"] * B["gAY"] * T2["VWAZ"]); //  aAAaAA from particle
        TIME_LINE(temp["uVWxYZ"] -= B["gWY"] * B["gAZ"] * T2["uVxA"]); //  aAAaAA from particle
        TIME_LINE(temp["uVWxYZ"] += B["gWZ"] * B["gAY"] * T2["uVxA"]); //  aAAaAA from particle
        TIME_LINE(temp["uVWxYZ"] -=
                  2.0 * B["gax"] * B["gWY"] * T2["uVaZ"]); //  aAAaAA from particle
        TIME_LINE(E += 0.5 * temp["uVWxYZ"] * Lambda3_["xYZuVW"]);
    }

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
}

void MRDSRG::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C1) {
    Timer timer;

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["qa"] -= alpha * T1["ia"] * H1["qi"];

    C1["IP"] += alpha * H1["AP"] * T1["IA"];
    C1["QA"] -= alpha * T1["IA"] * H1["QI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("111", timer.get());
}

void MRDSRG::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C1) {
    Timer timer;

    TIME_LINE(C1["ia"] += alpha * H1["bm"] * T2["imab"]);
    TIME_LINE(C1["ia"] += alpha * H1["bu"] * T2["ivab"] * Gamma1_["uv"]);
    TIME_LINE(C1["ia"] -= alpha * H1["vj"] * T2["ijau"] * Gamma1_["uv"]);
    TIME_LINE(C1["ia"] += alpha * H1["BM"] * T2["iMaB"]);
    TIME_LINE(C1["ia"] += alpha * H1["BU"] * T2["iVaB"] * Gamma1_["UV"]);
    TIME_LINE(C1["ia"] -= alpha * H1["VJ"] * T2["iJaU"] * Gamma1_["UV"]);

    TIME_LINE(C1["IA"] += alpha * H1["bm"] * T2["mIbA"]);
    TIME_LINE(C1["IA"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vIbA"]);
    TIME_LINE(C1["IA"] -= alpha * H1["vj"] * T2["jIuA"] * Gamma1_["uv"]);
    TIME_LINE(C1["IA"] += alpha * H1["BM"] * T2["IMAB"]);
    TIME_LINE(C1["IA"] += alpha * H1["BU"] * T2["IVAB"] * Gamma1_["UV"]);
    TIME_LINE(C1["IA"] -= alpha * H1["VJ"] * T2["IJAU"] * Gamma1_["UV"]);

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("121", timer.get());
}

void MRDSRG::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C1) {
    Timer timer;

    C1["qp"] += alpha * T1["ma"] * H2["qapm"];
    C1["qp"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["qepy"];
    C1["qp"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["qvpm"];
    C1["qp"] += alpha * T1["MA"] * H2["qApM"];
    C1["qp"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["qEpY"];
    C1["qp"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["qVpM"];

    C1["QP"] += alpha * T1["ma"] * H2["aQmP"];
    C1["QP"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eQyP"];
    C1["QP"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vQmP"];
    C1["QP"] += alpha * T1["MA"] * H2["QAPM"];
    C1["QP"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["QEPY"];
    C1["QP"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["QVPM"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("211", timer.get());
}

void MRDSRG::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C1) {
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    TIME_LINE(C1["ir"] += 0.5 * alpha * H2["abrm"] * T2["imab"]);
    TIME_LINE(C1["ir"] += alpha * H2["aBrM"] * T2["iMaB"]);
    TIME_LINE(C1["IR"] += 0.5 * alpha * H2["ABRM"] * T2["IMAB"]);
    TIME_LINE(C1["IR"] += alpha * H2["aBmR"] * T2["mIaB"]);

    TIME_LINE(C1["ir"] += 0.5 * alpha * Gamma1_["uv"] * H2["abru"] * T2["ivab"]);
    TIME_LINE(C1["ir"] += alpha * Gamma1_["UV"] * H2["aBrU"] * T2["iVaB"]);
    TIME_LINE(C1["IR"] += 0.5 * alpha * Gamma1_["UV"] * H2["ABRU"] * T2["IVAB"]);
    TIME_LINE(C1["IR"] += alpha * Gamma1_["uv"] * H2["aBuR"] * T2["vIaB"]);

    TIME_LINE(C1["ir"] += 0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vyrj"]);
    TIME_LINE(C1["IR"] += 0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYRJ"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"}));
    TIME_LINE(temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"]);
    TIME_LINE(C1["ir"] += alpha * temp["iJvY"] * H2["vYrJ"]);
    TIME_LINE(C1["IR"] += alpha * temp["jIvY"] * H2["vYjR"]);

    TIME_LINE(C1["ir"] -= alpha * Gamma1_["uv"] * H2["vbrm"] * T2["imub"]);
    TIME_LINE(C1["ir"] -= alpha * Gamma1_["uv"] * H2["vBrM"] * T2["iMuB"]);
    TIME_LINE(C1["ir"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * H2["bVrM"]);
    TIME_LINE(C1["IR"] -= alpha * Gamma1_["UV"] * H2["VBRM"] * T2["IMUB"]);
    TIME_LINE(C1["IR"] -= alpha * Gamma1_["UV"] * H2["bVmR"] * T2["mIbU"]);
    TIME_LINE(C1["IR"] -= alpha * Gamma1_["uv"] * H2["vBmR"] * T2["mIuB"]);

    TIME_LINE(C1["ir"] -= alpha * H2["vbrx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["iyub"]);
    TIME_LINE(C1["ir"] -= alpha * H2["vBrX"] * Gamma1_["uv"] * Gamma1_["XY"] * T2["iYuB"]);
    TIME_LINE(C1["ir"] -= alpha * H2["bVrX"] * Gamma1_["XY"] * Gamma1_["UV"] * T2["iYbU"]);
    TIME_LINE(C1["IR"] -= alpha * H2["VBRX"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["IYUB"]);
    TIME_LINE(C1["IR"] -= alpha * H2["vBxR"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yIuB"]);
    TIME_LINE(C1["IR"] -= alpha * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"] * H2["bVxR"]);

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    TIME_LINE(C1["pa"] -= 0.5 * alpha * H2["peij"] * T2["ijae"]);
    TIME_LINE(C1["pa"] -= alpha * H2["pEiJ"] * T2["iJaE"]);
    TIME_LINE(C1["PA"] -= 0.5 * alpha * H2["PEIJ"] * T2["IJAE"]);
    TIME_LINE(C1["PA"] -= alpha * H2["ePiJ"] * T2["iJeA"]);

    TIME_LINE(C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijau"] * H2["pvij"]);
    TIME_LINE(C1["pa"] -= alpha * Eta1_["UV"] * T2["iJaU"] * H2["pViJ"]);
    TIME_LINE(C1["PA"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJAU"] * H2["PVIJ"]);
    TIME_LINE(C1["PA"] -= alpha * Eta1_["uv"] * T2["iJuA"] * H2["vPiJ"]);

    TIME_LINE(C1["pa"] -= 0.5 * alpha * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * H2["pbux"]);
    TIME_LINE(C1["PA"] -= 0.5 * alpha * T2["VYAB"] * Eta1_["UV"] * Eta1_["XY"] * H2["PBUX"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"}));
    TIME_LINE(temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"]);
    TIME_LINE(C1["pa"] -= alpha * H2["pBuX"] * temp["uXaB"]);
    TIME_LINE(C1["PA"] -= alpha * H2["bPuX"] * temp["uXbA"]);

    TIME_LINE(C1["pa"] += alpha * Eta1_["uv"] * T2["vjae"] * H2["peuj"]);
    TIME_LINE(C1["pa"] += alpha * Eta1_["uv"] * T2["vJaE"] * H2["pEuJ"]);
    TIME_LINE(C1["pa"] += alpha * Eta1_["UV"] * H2["pEjU"] * T2["jVaE"]);
    TIME_LINE(C1["PA"] += alpha * Eta1_["UV"] * T2["VJAE"] * H2["PEUJ"]);
    TIME_LINE(C1["PA"] += alpha * Eta1_["uv"] * T2["vJeA"] * H2["ePuJ"]);
    TIME_LINE(C1["PA"] += alpha * Eta1_["UV"] * H2["ePjU"] * T2["jVeA"]);

    TIME_LINE(C1["pa"] += alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * H2["pyuj"]);
    TIME_LINE(C1["pa"] += alpha * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"] * H2["pYuJ"]);
    TIME_LINE(C1["pa"] += alpha * T2["jVaX"] * Eta1_["XY"] * Eta1_["UV"] * H2["pYjU"]);
    TIME_LINE(C1["PA"] += alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * H2["PYUJ"]);
    TIME_LINE(C1["PA"] += alpha * T2["vJxA"] * Eta1_["uv"] * Eta1_["xy"] * H2["yPuJ"]);
    TIME_LINE(C1["PA"] += alpha * H2["yPjU"] * Eta1_["UV"] * Eta1_["xy"] * T2["jVxA"]);

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    TIME_LINE(C1["ir"] += 0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * H2["uvrj"]);
    TIME_LINE(C1["IR"] += 0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * H2["UVRJ"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"}));
    TIME_LINE(temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"]);
    TIME_LINE(C1["ir"] += alpha * H2["uVrJ"] * temp["iJuV"]);
    TIME_LINE(C1["IR"] += alpha * H2["uVjR"] * temp["jIuV"]);

    TIME_LINE(C1["pa"] -= 0.25 * alpha * Lambda2_["xyuv"] * T2["uvab"] * H2["pbxy"]);
    TIME_LINE(C1["PA"] -= 0.25 * alpha * Lambda2_["XYUV"] * T2["UVAB"] * H2["PBXY"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"}));
    TIME_LINE(temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"]);
    TIME_LINE(C1["pa"] -= alpha * H2["pBxY"] * temp["xYaB"]);
    TIME_LINE(C1["PA"] -= alpha * H2["bPxY"] * temp["xYbA"]);

    TIME_LINE(C1["ir"] -= alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uArX"]);
    TIME_LINE(C1["IR"] -= alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["aUxR"]);
    TIME_LINE(C1["pa"] += alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["pUxI"]);
    TIME_LINE(C1["PA"] += alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uPiX"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"}));
    TIME_LINE(temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"]);
    TIME_LINE(temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"]);
    TIME_LINE(C1["ir"] += alpha * temp["ixau"] * H2["aurx"]);
    TIME_LINE(C1["pa"] -= alpha * H2["puix"] * temp["ixau"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hApA"}));
    TIME_LINE(temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"]);
    TIME_LINE(temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"]);
    TIME_LINE(C1["ir"] += alpha * temp["iXaU"] * H2["aUrX"]);
    TIME_LINE(C1["pa"] -= alpha * H2["pUiX"] * temp["iXaU"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aHaP"}));
    TIME_LINE(temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"]);
    TIME_LINE(temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"]);
    TIME_LINE(C1["IR"] += alpha * temp["xIuA"] * H2["uAxR"]);
    TIME_LINE(C1["PA"] -= alpha * H2["uPxI"] * temp["xIuA"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HAPA"}));
    TIME_LINE(temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"]);
    TIME_LINE(temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"]);
    TIME_LINE(C1["IR"] += alpha * temp["IXAU"] * H2["AURX"]);
    TIME_LINE(C1["PA"] -= alpha * H2["PUIX"] * temp["IXAU"]);

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"}));
    TIME_LINE(temp["au"] += 0.5 * Lambda2_["xyuv"] * H2["avxy"]);
    TIME_LINE(temp["au"] += Lambda2_["xYuV"] * H2["aVxY"]);
    TIME_LINE(C1["jb"] += alpha * temp["au"] * T2["ujab"]);
    TIME_LINE(C1["JB"] += alpha * temp["au"] * T2["uJaB"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"PA"}));
    TIME_LINE(temp["AU"] += 0.5 * Lambda2_["XYUV"] * H2["AVXY"]);
    TIME_LINE(temp["AU"] += Lambda2_["xYvU"] * H2["vAxY"]);
    TIME_LINE(C1["jb"] += alpha * temp["AU"] * T2["jUbA"]);
    TIME_LINE(C1["JB"] += alpha * temp["AU"] * T2["UJAB"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"}));
    TIME_LINE(temp["xi"] += 0.5 * Lambda2_["xyuv"] * H2["uviy"]);
    TIME_LINE(temp["xi"] += Lambda2_["xYuV"] * H2["uViY"]);
    TIME_LINE(C1["jb"] -= alpha * temp["xi"] * T2["ijxb"]);
    TIME_LINE(C1["JB"] -= alpha * temp["xi"] * T2["iJxB"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AH"}));
    TIME_LINE(temp["XI"] += 0.5 * Lambda2_["XYUV"] * H2["UVIY"]);
    TIME_LINE(temp["XI"] += Lambda2_["yXvU"] * H2["vUyI"]);
    TIME_LINE(C1["jb"] -= alpha * temp["XI"] * T2["jIbX"]);
    TIME_LINE(C1["JB"] -= alpha * temp["XI"] * T2["IJXB"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"}));
    TIME_LINE(temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"]);
    TIME_LINE(temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"]);
    TIME_LINE(C1["qs"] += alpha * temp["xe"] * H2["eqxs"]);
    TIME_LINE(C1["QS"] += alpha * temp["xe"] * H2["eQxS"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AV"}));
    TIME_LINE(temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"]);
    TIME_LINE(temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"]);
    TIME_LINE(C1["qs"] += alpha * temp["XE"] * H2["qEsX"]);
    TIME_LINE(C1["QS"] += alpha * temp["XE"] * H2["EQXS"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"}));
    TIME_LINE(temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"]);
    TIME_LINE(temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"]);
    TIME_LINE(C1["qs"] -= alpha * temp["mu"] * H2["uqms"]);
    TIME_LINE(C1["QS"] -= alpha * temp["mu"] * H2["uQmS"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"CA"}));
    TIME_LINE(temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"]);
    TIME_LINE(temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"]);
    TIME_LINE(C1["qs"] -= alpha * temp["MU"] * H2["qUsM"]);
    TIME_LINE(C1["QS"] -= alpha * temp["MU"] * H2["UQMS"]);

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void MRDSRG::H2_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    TIME_LINE(C1["ir"] += 0.5 * alpha * B["gar"] * B["gbm"] * T2["imab"]);
    TIME_LINE(C1["ir"] -= 0.5 * alpha * B["gam"] * B["gbr"] * T2["imab"]);
    TIME_LINE(C1["ir"] += alpha * B["gar"] * B["gBM"] * T2["iMaB"]);
    TIME_LINE(C1["IR"] += 0.5 * alpha * B["gAR"] * B["gBM"] * T2["IMAB"]);
    TIME_LINE(C1["IR"] -= 0.5 * alpha * B["gAM"] * B["gBR"] * T2["IMAB"]);
    TIME_LINE(C1["IR"] += alpha * B["gam"] * B["gBR"] * T2["mIaB"]);

    TIME_LINE(C1["ir"] += 0.5 * alpha * Gamma1_["uv"] * B["gar"] * B["gbu"] * T2["ivab"]);
    TIME_LINE(C1["ir"] -= 0.5 * alpha * Gamma1_["uv"] * B["gau"] * B["gbr"] * T2["ivab"]);
    TIME_LINE(C1["ir"] += alpha * Gamma1_["UV"] * B["gar"] * B["gBU"] * T2["iVaB"]);
    TIME_LINE(C1["IR"] += 0.5 * alpha * Gamma1_["UV"] * B["gAR"] * B["gBU"] * T2["IVAB"]);
    TIME_LINE(C1["IR"] -= 0.5 * alpha * Gamma1_["UV"] * B["gAU"] * B["gBR"] * T2["IVAB"]);
    TIME_LINE(C1["IR"] += alpha * Gamma1_["uv"] * B["gau"] * B["gBR"] * T2["vIaB"]);

    TIME_LINE(C1["ir"] +=
              0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * B["gvr"] * B["gyj"]);
    TIME_LINE(C1["ir"] -=
              0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * B["gvj"] * B["gyr"]);
    TIME_LINE(C1["IR"] +=
              0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * B["gVR"] * B["gYJ"]);
    TIME_LINE(C1["IR"] -=
              0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * B["gVJ"] * B["gYR"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"}));
    TIME_LINE(temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"]);
    TIME_LINE(C1["ir"] += alpha * temp["iJvY"] * B["gvr"] * B["gYJ"]);
    TIME_LINE(C1["IR"] += alpha * temp["jIvY"] * B["gvj"] * B["gYR"]);

    TIME_LINE(C1["ir"] -= alpha * Gamma1_["uv"] * B["gvr"] * B["gbm"] * T2["imub"]);
    TIME_LINE(C1["ir"] += alpha * Gamma1_["uv"] * B["gvm"] * B["gbr"] * T2["imub"]);
    TIME_LINE(C1["ir"] -= alpha * Gamma1_["uv"] * B["gvr"] * B["gBM"] * T2["iMuB"]);
    TIME_LINE(C1["ir"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * B["gbr"] * B["gVM"]);
    TIME_LINE(C1["IR"] -= alpha * Gamma1_["UV"] * B["gVR"] * B["gBM"] * T2["IMUB"]);
    TIME_LINE(C1["IR"] += alpha * Gamma1_["UV"] * B["gVM"] * B["gBR"] * T2["IMUB"]);
    TIME_LINE(C1["IR"] -= alpha * Gamma1_["UV"] * B["gbm"] * B["gVR"] * T2["mIbU"]);
    TIME_LINE(C1["IR"] -= alpha * Gamma1_["uv"] * B["gvm"] * B["gBR"] * T2["mIuB"]);

    TIME_LINE(C1["ir"] -= alpha * B["gvr"] * B["gbx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["iyub"]);
    TIME_LINE(C1["ir"] += alpha * B["gvx"] * B["gbr"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["iyub"]);
    TIME_LINE(C1["ir"] -= alpha * B["gvr"] * B["gBX"] * Gamma1_["uv"] * Gamma1_["XY"] * T2["iYuB"]);
    TIME_LINE(C1["ir"] -= alpha * B["gbr"] * B["gVX"] * Gamma1_["XY"] * Gamma1_["UV"] * T2["iYbU"]);
    TIME_LINE(C1["IR"] -= alpha * B["gVR"] * B["gBX"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["IYUB"]);
    TIME_LINE(C1["IR"] += alpha * B["gVX"] * B["gBR"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["IYUB"]);
    TIME_LINE(C1["IR"] -= alpha * B["gvx"] * B["gBR"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yIuB"]);
    TIME_LINE(C1["IR"] -= alpha * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"] * B["gbx"] * B["gVR"]);

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    TIME_LINE(C1["pa"] -= 0.5 * alpha * B["gpi"] * B["gej"] * T2["ijae"]);
    TIME_LINE(C1["pa"] += 0.5 * alpha * B["gpj"] * B["gei"] * T2["ijae"]);
    TIME_LINE(C1["pa"] -= alpha * B["gpi"] * B["gEJ"] * T2["iJaE"]);
    TIME_LINE(C1["PA"] -= 0.5 * alpha * B["gPI"] * B["gEJ"] * T2["IJAE"]);
    TIME_LINE(C1["PA"] += 0.5 * alpha * B["gPJ"] * B["gEI"] * T2["IJAE"]);
    TIME_LINE(C1["PA"] -= alpha * B["gei"] * B["gPJ"] * T2["iJeA"]);

    TIME_LINE(C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijau"] * B["gpi"] * B["gvj"]);
    TIME_LINE(C1["pa"] += 0.5 * alpha * Eta1_["uv"] * T2["ijau"] * B["gpj"] * B["gvi"]);
    TIME_LINE(C1["pa"] -= alpha * Eta1_["UV"] * T2["iJaU"] * B["gpi"] * B["gVJ"]);
    TIME_LINE(C1["PA"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJAU"] * B["gPI"] * B["gVJ"]);
    TIME_LINE(C1["PA"] += 0.5 * alpha * Eta1_["UV"] * T2["IJAU"] * B["gPJ"] * B["gVI"]);
    TIME_LINE(C1["PA"] -= alpha * Eta1_["uv"] * T2["iJuA"] * B["gvi"] * B["gPJ"]);

    TIME_LINE(C1["pa"] -=
              0.5 * alpha * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * B["gpu"] * B["gbx"]);
    TIME_LINE(C1["pa"] +=
              0.5 * alpha * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * B["gpx"] * B["gbu"]);
    TIME_LINE(C1["PA"] -=
              0.5 * alpha * T2["VYAB"] * Eta1_["UV"] * Eta1_["XY"] * B["gPU"] * B["gBX"]);
    TIME_LINE(C1["PA"] +=
              0.5 * alpha * T2["VYAB"] * Eta1_["UV"] * Eta1_["XY"] * B["gPX"] * B["gBU"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"}));
    TIME_LINE(temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"]);
    TIME_LINE(C1["pa"] -= alpha * B["gpu"] * B["gBX"] * temp["uXaB"]);
    TIME_LINE(C1["PA"] -= alpha * B["gbu"] * B["gPX"] * temp["uXbA"]);

    TIME_LINE(C1["pa"] += alpha * Eta1_["uv"] * T2["vjae"] * B["gpu"] * B["gej"]);
    TIME_LINE(C1["pa"] -= alpha * Eta1_["uv"] * T2["vjae"] * B["gpj"] * B["geu"]);
    TIME_LINE(C1["pa"] += alpha * Eta1_["uv"] * T2["vJaE"] * B["gpu"] * B["gEJ"]);
    TIME_LINE(C1["pa"] += alpha * Eta1_["UV"] * B["gpj"] * B["gEU"] * T2["jVaE"]);
    TIME_LINE(C1["PA"] += alpha * Eta1_["UV"] * T2["VJAE"] * B["gPU"] * B["gEJ"]);
    TIME_LINE(C1["PA"] -= alpha * Eta1_["UV"] * T2["VJAE"] * B["gPJ"] * B["gEU"]);
    TIME_LINE(C1["PA"] += alpha * Eta1_["uv"] * T2["vJeA"] * B["geu"] * B["gPJ"]);
    TIME_LINE(C1["PA"] += alpha * Eta1_["UV"] * B["gej"] * B["gPU"] * T2["jVeA"]);

    TIME_LINE(C1["pa"] += alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * B["gpu"] * B["gyj"]);
    TIME_LINE(C1["pa"] -= alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * B["gpj"] * B["gyu"]);
    TIME_LINE(C1["pa"] += alpha * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"] * B["gpu"] * B["gYJ"]);
    TIME_LINE(C1["pa"] += alpha * T2["jVaX"] * Eta1_["XY"] * Eta1_["UV"] * B["gpj"] * B["gYU"]);
    TIME_LINE(C1["PA"] += alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * B["gPU"] * B["gYJ"]);
    TIME_LINE(C1["PA"] -= alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * B["gPJ"] * B["gYU"]);
    TIME_LINE(C1["PA"] += alpha * T2["vJxA"] * Eta1_["uv"] * Eta1_["xy"] * B["gyu"] * B["gPJ"]);
    TIME_LINE(C1["PA"] += alpha * B["gyj"] * B["gPU"] * Eta1_["UV"] * Eta1_["xy"] * T2["jVxA"]);

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    TIME_LINE(C1["ir"] += 0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * B["gur"] * B["gvj"]);
    TIME_LINE(C1["ir"] -= 0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * B["guj"] * B["gvr"]);
    TIME_LINE(C1["IR"] += 0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * B["gUR"] * B["gVJ"]);
    TIME_LINE(C1["IR"] -= 0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * B["gUJ"] * B["gVR"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"}));
    TIME_LINE(temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"]);
    TIME_LINE(C1["ir"] += alpha * B["gur"] * B["gVJ"] * temp["iJuV"]);
    TIME_LINE(C1["IR"] += alpha * B["guj"] * B["gVR"] * temp["jIuV"]);

    TIME_LINE(C1["pa"] -= 0.25 * alpha * Lambda2_["xyuv"] * T2["uvab"] * B["gpx"] * B["gby"]);
    TIME_LINE(C1["pa"] += 0.25 * alpha * Lambda2_["xyuv"] * T2["uvab"] * B["gpy"] * B["gbx"]);
    TIME_LINE(C1["PA"] -= 0.25 * alpha * Lambda2_["XYUV"] * T2["UVAB"] * B["gPX"] * B["gBY"]);
    TIME_LINE(C1["PA"] += 0.25 * alpha * Lambda2_["XYUV"] * T2["UVAB"] * B["gPY"] * B["gBX"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"}));
    TIME_LINE(temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"]);
    TIME_LINE(C1["pa"] -= alpha * B["gpx"] * B["gBY"] * temp["xYaB"]);
    TIME_LINE(C1["PA"] -= alpha * B["gbx"] * B["gPY"] * temp["xYbA"]);

    TIME_LINE(C1["ir"] -= alpha * Lambda2_["yXuV"] * T2["iVyA"] * B["gur"] * B["gAX"]);
    TIME_LINE(C1["IR"] -= alpha * Lambda2_["xYvU"] * T2["vIaY"] * B["gax"] * B["gUR"]);
    TIME_LINE(C1["pa"] += alpha * Lambda2_["xYvU"] * T2["vIaY"] * B["gpx"] * B["gUI"]);
    TIME_LINE(C1["PA"] += alpha * Lambda2_["yXuV"] * T2["iVyA"] * B["gui"] * B["gPX"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"}));
    TIME_LINE(temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"]);
    TIME_LINE(temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"]);
    TIME_LINE(C1["ir"] += alpha * temp["ixau"] * B["gar"] * B["gux"]);
    TIME_LINE(C1["ir"] -= alpha * temp["ixau"] * B["gax"] * B["gur"]);
    TIME_LINE(C1["pa"] -= alpha * B["gpi"] * B["gux"] * temp["ixau"]);
    TIME_LINE(C1["pa"] += alpha * B["gpx"] * B["gui"] * temp["ixau"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hApA"}));
    TIME_LINE(temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"]);
    TIME_LINE(temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"]);
    TIME_LINE(C1["ir"] += alpha * temp["iXaU"] * B["gar"] * B["gUX"]);
    TIME_LINE(C1["pa"] -= alpha * B["gpi"] * B["gUX"] * temp["iXaU"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aHaP"}));
    TIME_LINE(temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"]);
    TIME_LINE(temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"]);
    TIME_LINE(C1["IR"] += alpha * temp["xIuA"] * B["gux"] * B["gAR"]);
    TIME_LINE(C1["PA"] -= alpha * B["gux"] * B["gPI"] * temp["xIuA"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HAPA"}));
    TIME_LINE(temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"]);
    TIME_LINE(temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"]);
    TIME_LINE(C1["IR"] += alpha * temp["IXAU"] * B["gAR"] * B["gUX"]);
    TIME_LINE(C1["IR"] -= alpha * temp["IXAU"] * B["gAX"] * B["gUR"]);
    TIME_LINE(C1["PA"] -= alpha * B["gPI"] * B["gUX"] * temp["IXAU"]);
    TIME_LINE(C1["PA"] += alpha * B["gPX"] * B["gUI"] * temp["IXAU"]);

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"}));
    TIME_LINE(temp["au"] += 0.5 * Lambda2_["xyuv"] * B["gax"] * B["gvy"]);
    TIME_LINE(temp["au"] -= 0.5 * Lambda2_["xyuv"] * B["gay"] * B["gvx"]);
    TIME_LINE(temp["au"] += Lambda2_["xYuV"] * B["gax"] * B["gVY"]);
    TIME_LINE(C1["jb"] += alpha * temp["au"] * T2["ujab"]);
    TIME_LINE(C1["JB"] += alpha * temp["au"] * T2["uJaB"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"PA"}));
    TIME_LINE(temp["AU"] += 0.5 * Lambda2_["XYUV"] * B["gAX"] * B["gVY"]);
    TIME_LINE(temp["AU"] -= 0.5 * Lambda2_["XYUV"] * B["gAY"] * B["gVX"]);
    TIME_LINE(temp["AU"] += Lambda2_["xYvU"] * B["gvx"] * B["gAY"]);
    TIME_LINE(C1["jb"] += alpha * temp["AU"] * T2["jUbA"]);
    TIME_LINE(C1["JB"] += alpha * temp["AU"] * T2["UJAB"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"}));
    TIME_LINE(temp["xi"] += 0.5 * Lambda2_["xyuv"] * B["gui"] * B["gvy"]);
    TIME_LINE(temp["xi"] -= 0.5 * Lambda2_["xyuv"] * B["guy"] * B["gvi"]);
    TIME_LINE(temp["xi"] += Lambda2_["xYuV"] * B["gui"] * B["gVY"]);
    TIME_LINE(C1["jb"] -= alpha * temp["xi"] * T2["ijxb"]);
    TIME_LINE(C1["JB"] -= alpha * temp["xi"] * T2["iJxB"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AH"}));
    TIME_LINE(temp["XI"] += 0.5 * Lambda2_["XYUV"] * B["gUI"] * B["gVY"]);
    TIME_LINE(temp["XI"] -= 0.5 * Lambda2_["XYUV"] * B["gUY"] * B["gVI"]);
    TIME_LINE(temp["XI"] += Lambda2_["yXvU"] * B["gvy"] * B["gUI"]);
    TIME_LINE(C1["jb"] -= alpha * temp["XI"] * T2["jIbX"]);
    TIME_LINE(C1["JB"] -= alpha * temp["XI"] * T2["IJXB"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"}));
    TIME_LINE(temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"]);
    TIME_LINE(temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"]);
    TIME_LINE(C1["qs"] += alpha * temp["xe"] * B["gex"] * B["gqs"]);
    TIME_LINE(C1["qs"] -= alpha * temp["xe"] * B["ges"] * B["gqx"]);
    TIME_LINE(C1["QS"] += alpha * temp["xe"] * B["gex"] * B["gQS"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AV"}));
    TIME_LINE(temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"]);
    TIME_LINE(temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"]);
    TIME_LINE(C1["qs"] += alpha * temp["XE"] * B["gqs"] * B["gEX"]);
    TIME_LINE(C1["QS"] += alpha * temp["XE"] * B["gEX"] * B["gQS"]);
    TIME_LINE(C1["QS"] -= alpha * temp["XE"] * B["gES"] * B["gQX"]);

    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"}));
    TIME_LINE(temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"]);
    TIME_LINE(temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"]);
    TIME_LINE(C1["qs"] -= alpha * temp["mu"] * B["gum"] * B["gqs"]);
    TIME_LINE(C1["qs"] += alpha * temp["mu"] * B["gus"] * B["gqm"]);
    TIME_LINE(C1["QS"] -= alpha * temp["mu"] * B["gum"] * B["gQS"]);
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"CA"}));
    TIME_LINE(temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"]);
    TIME_LINE(temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"]);
    TIME_LINE(C1["qs"] -= alpha * temp["MU"] * B["gqs"] * B["gUM"]);
    TIME_LINE(C1["QS"] -= alpha * temp["MU"] * B["gUM"] * B["gQS"]);
    TIME_LINE(C1["QS"] += alpha * temp["MU"] * B["gUS"] * B["gQM"]);

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void MRDSRG::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C2) {
    Timer timer;

    TIME_LINE(C2["ijpb"] += alpha * T2["ijab"] * H1["ap"]);
    TIME_LINE(C2["ijap"] += alpha * T2["ijab"] * H1["bp"]);
    TIME_LINE(C2["qjab"] -= alpha * T2["ijab"] * H1["qi"]);
    TIME_LINE(C2["iqab"] -= alpha * T2["ijab"] * H1["qj"]);

    TIME_LINE(C2["iJpB"] += alpha * T2["iJaB"] * H1["ap"]);
    TIME_LINE(C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"]);
    TIME_LINE(C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"]);
    TIME_LINE(C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"]);

    TIME_LINE(C2["IJPB"] += alpha * T2["IJAB"] * H1["AP"]);
    TIME_LINE(C2["IJAP"] += alpha * T2["IJAB"] * H1["BP"]);
    TIME_LINE(C2["QJAB"] -= alpha * T2["IJAB"] * H1["QI"]);
    TIME_LINE(C2["IQAB"] -= alpha * T2["IJAB"] * H1["QJ"]);

    //    // probably not worth doing the following because contracting one
    //    index should be fast
    //    BlockedTensor temp =
    //    ambit::BlockedTensor::build(tensor_type_,"temp",{"hhpg","HHPG"});
    //    temp["ijap"] = alpha * T2["ijab"] * H1["bp"];
    //    temp["IJAP"] = alpha * T2["IJAB"] * H1["BP"];

    //    C2["ijpb"] -= temp["ijbp"]; // use permutation of temp
    //    C2["ijap"] += temp["ijap"]; // explicitly evaluate by temp
    //    C2["iJpA"] += alpha * T2["iJbA"] * H1["bp"];
    //    C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"];
    //    C2["IJPB"] -= temp["IJBP"]; // use permutation of temp
    //    C2["IJAP"] += temp["IJAP"]; // explicitly evaluate by temp

    //    temp =
    //    ambit::BlockedTensor::build(tensor_type_,"temp",{"ghpp","GHPP"});
    //    temp["qjab"] = alpha * T2["ijab"] * H1["qi"];
    //    temp["QJAB"] = alpha * T2["IJAB"] * H1["QI"];

    //    C2["qjab"] -= temp["qjab"]; // explicitly evaluate by temp
    //    C2["iqab"] += temp["qiab"]; // use permutation of temp
    //    C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"];
    //    C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"];
    //    C2["QJAB"] -= temp["QJAB"]; // explicitly evaluate by temp
    //    C2["IQAB"] += temp["QIAB"]; // use permutation of temp

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("122", timer.get());
}

void MRDSRG::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C2) {
    Timer timer;

    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
    C2["ripq"] += alpha * T1["ia"] * H2["rapq"];
    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
    C2["rspa"] -= alpha * T1["ia"] * H2["rspi"];

    C2["iRpQ"] += alpha * T1["ia"] * H2["aRpQ"];
    C2["rIpQ"] += alpha * T1["IA"] * H2["rApQ"];
    C2["rSaQ"] -= alpha * T1["ia"] * H2["rSiQ"];
    C2["rSpA"] -= alpha * T1["IA"] * H2["rSpI"];

    C2["IRPQ"] += alpha * T1["IA"] * H2["ARPQ"];
    C2["RIPQ"] += alpha * T1["IA"] * H2["RAPQ"];
    C2["RSAQ"] -= alpha * T1["IA"] * H2["RSIQ"];
    C2["RSPA"] -= alpha * T1["IA"] * H2["RSPI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("212", timer.get());
}

void MRDSRG::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C2) {
    Timer timer;

    // particle-particle contractions
    forte::timer pp("H2_T2_C2 pp");
    TIME_LINE(C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"]);
    TIME_LINE(C2["iJrS"] += alpha * H2["aBrS"] * T2["iJaB"]);
    TIME_LINE(C2["IJRS"] += 0.5 * alpha * H2["ABRS"] * T2["IJAB"]);

    TIME_LINE(C2["ijrs"] -= alpha * Gamma1_["xy"] * H2["ybrs"] * T2["ijxb"]);
    TIME_LINE(C2["iJrS"] -= alpha * Gamma1_["xy"] * H2["yBrS"] * T2["iJxB"]);
    TIME_LINE(C2["iJrS"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * H2["bYrS"]);
    TIME_LINE(C2["IJRS"] -= alpha * Gamma1_["XY"] * H2["YBRS"] * T2["IJXB"]);
    pp.stop();

    // hole-hole contractions
    forte::timer hh("H2_T2_C2 hh");
    TIME_LINE(C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"]);
    TIME_LINE(C2["pQaB"] += alpha * H2["pQiJ"] * T2["iJaB"]);
    TIME_LINE(C2["PQAB"] += 0.5 * alpha * H2["PQIJ"] * T2["IJAB"]);

    TIME_LINE(C2["pqab"] -= alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"]);
    TIME_LINE(C2["pQaB"] -= alpha * Eta1_["xy"] * T2["yJaB"] * H2["pQxJ"]);
    TIME_LINE(C2["pQaB"] -= alpha * Eta1_["XY"] * H2["pQjX"] * T2["jYaB"]);
    TIME_LINE(C2["PQAB"] -= alpha * Eta1_["XY"] * T2["YJAB"] * H2["PQXJ"]);
    hh.stop();

    // hole-particle contractions
    forte::timer hp("H2_T2_C2 hp");
    forte::timer tempBuild("temp build");
    TIME_LINE(BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ghgp"}));
    tempBuild.stop();
    TIME_LINE(temp["qjsb"] += alpha * H2["aqms"] * T2["mjab"]);
    TIME_LINE(temp["qjsb"] += alpha * H2["qAsM"] * T2["jMbA"]);
    TIME_LINE(temp["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"]);
    TIME_LINE(temp["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"]);
    TIME_LINE(temp["qjsb"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["ijxb"]);
    TIME_LINE(temp["qjsb"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["jIbX"]);
    forte::timer resorting("Resorting");
    TIME_LINE(C2["qjsb"] += temp["qjsb"]);
    TIME_LINE(C2["jqsb"] -= temp["qjsb"]);
    TIME_LINE(C2["qjbs"] -= temp["qjsb"]);
    TIME_LINE(C2["jqbs"] += temp["qjsb"]);
    resorting.stop();

    forte::timer tempBuild2("temp build");
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"GHGP"}));
    tempBuild2.stop();
    TIME_LINE(temp["QJSB"] += alpha * H2["AQMS"] * T2["MJAB"]);
    TIME_LINE(temp["QJSB"] += alpha * H2["aQmS"] * T2["mJaB"]);
    TIME_LINE(temp["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"]);
    TIME_LINE(temp["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"]);
    TIME_LINE(temp["QJSB"] -= alpha * Gamma1_["XY"] * H2["YQIS"] * T2["IJXB"]);
    TIME_LINE(temp["QJSB"] -= alpha * Gamma1_["xy"] * H2["yQiS"] * T2["iJxB"]);
    forte::timer resorting2("Resorting");
    TIME_LINE(C2["QJSB"] += temp["QJSB"]);
    TIME_LINE(C2["JQSB"] -= temp["QJSB"]);
    TIME_LINE(C2["QJBS"] -= temp["QJSB"]);
    TIME_LINE(C2["JQBS"] += temp["QJSB"]);
    resorting2.stop();

    TIME_LINE(C2["qJsB"] += alpha * H2["aqms"] * T2["mJaB"]);
    TIME_LINE(C2["qJsB"] += alpha * H2["qAsM"] * T2["MJAB"]);
    TIME_LINE(C2["qJsB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aqxs"]);
    TIME_LINE(C2["qJsB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["qAsX"]);
    TIME_LINE(C2["qJsB"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["iJxB"]);
    TIME_LINE(C2["qJsB"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["IJXB"]);

    TIME_LINE(C2["iQsB"] -= alpha * T2["iMaB"] * H2["aQsM"]);
    TIME_LINE(C2["iQsB"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * H2["aQsX"]);
    TIME_LINE(C2["iQsB"] += alpha * Gamma1_["xy"] * H2["yQsJ"] * T2["iJxB"]);

    TIME_LINE(C2["qJaS"] -= alpha * T2["mJaB"] * H2["qBmS"]);
    TIME_LINE(C2["qJaS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["qBxS"]);
    TIME_LINE(C2["qJaS"] += alpha * Gamma1_["XY"] * H2["qYiS"] * T2["iJaX"]);

    TIME_LINE(C2["iQaS"] += alpha * T2["imab"] * H2["bQmS"]);
    TIME_LINE(C2["iQaS"] += alpha * T2["iMaB"] * H2["BQMS"]);
    TIME_LINE(C2["iQaS"] += alpha * Gamma1_["xy"] * T2["iyab"] * H2["bQxS"]);
    TIME_LINE(C2["iQaS"] += alpha * Gamma1_["XY"] * T2["iYaB"] * H2["BQXS"]);
    TIME_LINE(C2["iQaS"] -= alpha * Gamma1_["xy"] * H2["yQjS"] * T2["ijax"]);
    TIME_LINE(C2["iQaS"] -= alpha * Gamma1_["XY"] * H2["YQJS"] * T2["iJaX"]);
    hp.stop();

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void MRDSRG::H2_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    Timer timer;

    // particle-particle contractions
    forte::timer pp("H2_T2_C2 pp");
    TIME_LINE(C2["ijrs"] += batched("r", 0.5 * alpha * B["gar"] * B["gbs"] * T2["ijab"]));
    TIME_LINE(C2["ijrs"] -= batched("s", 0.5 * alpha * B["gas"] * B["gbr"] * T2["ijab"]));
    TIME_LINE(C2["iJrS"] += batched("r", alpha * B["gar"] * B["gBS"] * T2["iJaB"]));
    TIME_LINE(C2["IJRS"] += batched("R", 0.5 * alpha * B["gAR"] * B["gBS"] * T2["IJAB"]));
    TIME_LINE(C2["IJRS"] -= batched("S", 0.5 * alpha * B["gAS"] * B["gBR"] * T2["IJAB"]));

    TIME_LINE(C2["ijrs"] -= alpha * Gamma1_["xy"] * B["gyr"] * B["gbs"] * T2["ijxb"]);
    TIME_LINE(C2["ijrs"] += alpha * Gamma1_["xy"] * B["gys"] * B["gbr"] * T2["ijxb"]);
    TIME_LINE(C2["iJrS"] -= alpha * Gamma1_["xy"] * B["gyr"] * B["gBS"] * T2["iJxB"]);
    TIME_LINE(C2["iJrS"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * B["gbr"] * B["gYS"]);
    TIME_LINE(C2["IJRS"] -= alpha * Gamma1_["XY"] * B["gYR"] * B["gBS"] * T2["IJXB"]);
    TIME_LINE(C2["IJRS"] += alpha * Gamma1_["XY"] * B["gYS"] * B["gBR"] * T2["IJXB"]);
    pp.stop();

    // hole-hole contractions
    forte::timer hh("H2_T2_C2 hh");
    TIME_LINE(C2["pqab"] += 0.5 * alpha * B["gpi"] * B["gqj"] * T2["ijab"]);
    TIME_LINE(C2["pqab"] -= 0.5 * alpha * B["gpj"] * B["gqi"] * T2["ijab"]);
    TIME_LINE(C2["pQaB"] += alpha * B["gpi"] * B["gQJ"] * T2["iJaB"]);
    TIME_LINE(C2["PQAB"] += 0.5 * alpha * B["gPI"] * B["gQJ"] * T2["IJAB"]);
    TIME_LINE(C2["PQAB"] -= 0.5 * alpha * B["gPJ"] * B["gQI"] * T2["IJAB"]);

    TIME_LINE(C2["pqab"] -= alpha * Eta1_["xy"] * T2["yjab"] * B["gpx"] * B["gqj"]);
    TIME_LINE(C2["pqab"] += alpha * Eta1_["xy"] * T2["yjab"] * B["gpj"] * B["gqx"]);
    TIME_LINE(C2["pQaB"] -= alpha * Eta1_["xy"] * T2["yJaB"] * B["gpx"] * B["gQJ"]);
    TIME_LINE(C2["pQaB"] -= alpha * Eta1_["XY"] * B["gpj"] * B["gQX"] * T2["jYaB"]);
    TIME_LINE(C2["PQAB"] -= alpha * Eta1_["XY"] * T2["YJAB"] * B["gPX"] * B["gQJ"]);
    TIME_LINE(C2["PQAB"] += alpha * Eta1_["XY"] * T2["YJAB"] * B["gPJ"] * B["gQX"]);
    hh.stop();

    // hole-particle contractions
    forte::timer hp("H2_T2_C2 hp");
    forte::timer tempBuild("temp build");
    TIME_LINE(BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ghgp"}));
    tempBuild.stop();
    TIME_LINE(temp["qjsb"] += alpha * B["gam"] * B["gqs"] * T2["mjab"]);
    TIME_LINE(temp["qjsb"] -= alpha * B["gas"] * B["gqm"] * T2["mjab"]);
    TIME_LINE(temp["qjsb"] += alpha * B["gqs"] * B["gAM"] * T2["jMbA"]);
    TIME_LINE(temp["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * B["gax"] * B["gqs"]);
    TIME_LINE(temp["qjsb"] -= alpha * Gamma1_["xy"] * T2["yjab"] * B["gas"] * B["gqx"]);
    TIME_LINE(temp["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * B["gqs"] * B["gAX"]);
    TIME_LINE(temp["qjsb"] -= alpha * Gamma1_["xy"] * B["gyi"] * B["gqs"] * T2["ijxb"]);
    TIME_LINE(temp["qjsb"] += alpha * Gamma1_["xy"] * B["gys"] * B["gqi"] * T2["ijxb"]);
    TIME_LINE(temp["qjsb"] -= alpha * Gamma1_["XY"] * B["gqs"] * B["gYI"] * T2["jIbX"]);
    forte::timer resorting("Resorting");
    TIME_LINE(C2["qjsb"] += temp["qjsb"]);
    TIME_LINE(C2["jqsb"] -= temp["qjsb"]);
    TIME_LINE(C2["qjbs"] -= temp["qjsb"]);
    TIME_LINE(C2["jqbs"] += temp["qjsb"]);
    resorting.stop();

    forte::timer tempBuild2("temp build");
    TIME_LINE(temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"GHGP"}));
    tempBuild2.stop();
    TIME_LINE(temp["QJSB"] += alpha * B["gAM"] * B["gQS"] * T2["MJAB"]);
    TIME_LINE(temp["QJSB"] -= alpha * B["gAS"] * B["gQM"] * T2["MJAB"]);
    TIME_LINE(temp["QJSB"] += alpha * B["gam"] * B["gQS"] * T2["mJaB"]);
    TIME_LINE(temp["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * B["gAX"] * B["gQS"]);
    TIME_LINE(temp["QJSB"] -= alpha * Gamma1_["XY"] * T2["YJAB"] * B["gAS"] * B["gQX"]);
    TIME_LINE(temp["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * B["gax"] * B["gQS"]);
    TIME_LINE(temp["QJSB"] -= alpha * Gamma1_["XY"] * B["gYI"] * B["gQS"] * T2["IJXB"]);
    TIME_LINE(temp["QJSB"] += alpha * Gamma1_["XY"] * B["gYS"] * B["gQI"] * T2["IJXB"]);
    TIME_LINE(temp["QJSB"] -= alpha * Gamma1_["xy"] * B["gyi"] * B["gQS"] * T2["iJxB"]);
    forte::timer resorting2("Resorting");
    TIME_LINE(C2["QJSB"] += temp["QJSB"]);
    TIME_LINE(C2["JQSB"] -= temp["QJSB"]);
    TIME_LINE(C2["QJBS"] -= temp["QJSB"]);
    TIME_LINE(C2["JQBS"] += temp["QJSB"]);
    resorting2.stop();

    TIME_LINE(C2["qJsB"] += alpha * B["gam"] * B["gqs"] * T2["mJaB"]);
    TIME_LINE(C2["qJsB"] -= alpha * B["gas"] * B["gqm"] * T2["mJaB"]);
    TIME_LINE(C2["qJsB"] += alpha * B["gqs"] * B["gAM"] * T2["MJAB"]);
    TIME_LINE(C2["qJsB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * B["gax"] * B["gqs"]);
    TIME_LINE(C2["qJsB"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * B["gas"] * B["gqx"]);
    TIME_LINE(C2["qJsB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * B["gqs"] * B["gAX"]);
    TIME_LINE(C2["qJsB"] -= alpha * Gamma1_["xy"] * B["gyi"] * B["gqs"] * T2["iJxB"]);
    TIME_LINE(C2["qJsB"] += alpha * Gamma1_["xy"] * B["gys"] * B["gqi"] * T2["iJxB"]);
    TIME_LINE(C2["qJsB"] -= alpha * Gamma1_["XY"] * B["gqs"] * B["gYI"] * T2["IJXB"]);

    TIME_LINE(C2["iQsB"] -= alpha * T2["iMaB"] * B["gas"] * B["gQM"]);
    TIME_LINE(C2["iQsB"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * B["gas"] * B["gQX"]);
    TIME_LINE(C2["iQsB"] += alpha * Gamma1_["xy"] * B["gys"] * B["gQJ"] * T2["iJxB"]);

    TIME_LINE(C2["qJaS"] -= alpha * T2["mJaB"] * B["gqm"] * B["gBS"]);
    TIME_LINE(C2["qJaS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * B["gqx"] * B["gBS"]);
    TIME_LINE(C2["qJaS"] += alpha * Gamma1_["XY"] * B["gqi"] * B["gYS"] * T2["iJaX"]);

    TIME_LINE(C2["iQaS"] += alpha * T2["imab"] * B["gbm"] * B["gQS"]);
    TIME_LINE(C2["iQaS"] += alpha * T2["iMaB"] * B["gBM"] * B["gQS"]);
    TIME_LINE(C2["iQaS"] -= alpha * T2["iMaB"] * B["gBS"] * B["gQM"]);
    TIME_LINE(C2["iQaS"] += alpha * Gamma1_["xy"] * T2["iyab"] * B["gbx"] * B["gQS"]);
    TIME_LINE(C2["iQaS"] += alpha * Gamma1_["XY"] * T2["iYaB"] * B["gBX"] * B["gQS"]);
    TIME_LINE(C2["iQaS"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * B["gBS"] * B["gQX"]);
    TIME_LINE(C2["iQaS"] -= alpha * Gamma1_["xy"] * B["gyj"] * B["gQS"] * T2["ijax"]);
    TIME_LINE(C2["iQaS"] -= alpha * Gamma1_["XY"] * B["gYJ"] * B["gQS"] * T2["iJaX"]);
    TIME_LINE(C2["iQaS"] += alpha * Gamma1_["XY"] * B["gYS"] * B["gQJ"] * T2["iJaX"]);
    hp.stop();

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void MRDSRG::H1_G1_C0(BlockedTensor& H1, BlockedTensor& G1, const double& alpha, double& C0) {
    Timer timer;

    double E = 0.0;
    E += H1["qm"] * G1["mq"];
    E += H1["qu"] * G1["vq"] * Gamma1_["uv"];
    E -= H1["mp"] * G1["pm"];
    E -= H1["vp"] * G1["pu"] * Gamma1_["uv"];

    E += H1["QM"] * G1["MQ"];
    E += H1["QU"] * G1["VQ"] * Gamma1_["UV"];
    E -= H1["MP"] * G1["PM"];
    E -= H1["VP"] * G1["PU"] * Gamma1_["UV"];

    E *= alpha;
    C0 += E;

    dsrg_time_.add("110", timer.get());
}

void MRDSRG::H1_G2_C0(BlockedTensor& H1, BlockedTensor& G2, const double& alpha, double& C0) {
    Timer timer;

    BlockedTensor temp;
    double E = 0.0;

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["xyuv"] += H1["qu"] * G2["xyqv"];
    temp["xyuv"] -= H1["xp"] * G2["pyuv"];
    E += 0.5 * temp["xyuv"] * Lambda2_["uvxy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"});
    temp["XYUV"] += H1["QU"] * G2["XYQV"];
    temp["XYUV"] -= H1["XP"] * G2["PYUV"];
    E += 0.5 * temp["XYUV"] * Lambda2_["UVXY"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaA"});
    temp["xYuV"] += H1["qu"] * G2["xYqV"];
    temp["xYuV"] += H1["QV"] * G2["xYuQ"];
    temp["xYuV"] -= H1["xp"] * G2["pYuV"];
    temp["xYuV"] -= H1["YP"] * G2["xPuV"];
    E += temp["xYuV"] * Lambda2_["uVxY"];

    E *= alpha;
    C0 += E;
    dsrg_time_.add("120", timer.get());
}

void MRDSRG::H2_G2_C0(BlockedTensor& H2, BlockedTensor& G2, const double& alpha, double& C0) {
    Timer timer;
    double E = 0.0;

    // <[Hbar2, T2]> (C_2)^4
    // No gamma nor eta
    E += H2["eFmN"] * G2["mNeF"];
    E += 0.25 * H2["efmn"] * G2["mnef"];
    E += 0.25 * H2["EFMN"] * G2["MNEF"];
    E -= G2["eFmN"] * H2["mNeF"];
    E -= 0.25 * G2["efmn"] * H2["mnef"];
    E -= 0.25 * G2["EFMN"] * H2["MNEF"];

    // one gamma
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aa"}));
    temp["vu"] += 0.5 * H2["efmu"] * G2["mvef"];
    temp["vu"] += H2["fEuM"] * G2["vMfE"];
    temp["vu"] -= 0.5 * G2["efmu"] * H2["mvef"];
    temp["vu"] -= G2["fEuM"] * H2["vMfE"];
    E += temp["vu"] * Gamma1_["uv"];

    temp["VU"] += 0.5 * H2["EFMU"] * G2["MVEF"];
    temp["VU"] += H2["eFmU"] * G2["mVeF"];
    temp["VU"] -= 0.5 * G2["EFMU"] * H2["MVEF"];
    temp["VU"] -= G2["eFmU"] * H2["mVeF"];
    E += temp["VU"] * Gamma1_["UV"];

    // one eta
    temp.zero();
    temp["vu"] += 0.5 * H2["vemn"] * G2["mnue"];
    temp["vu"] += H2["vEmN"] * G2["mNuE"];
    temp["vu"] -= 0.5 * G2["vemn"] * H2["mnue"];
    temp["vu"] -= G2["vEmN"] * H2["mNuE"];
    E += temp["vu"] * Eta1_["uv"];

    temp["VU"] += 0.5 * H2["VEMN"] * G2["MNUE"];
    temp["VU"] += H2["eVnM"] * G2["nMeU"];
    temp["VU"] -= 0.5 * G2["VEMN"] * H2["MNUE"];
    temp["VU"] -= G2["eVnM"] * H2["nMeU"];
    E += temp["VU"] * Eta1_["UV"];

    // two gamma
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"}));
    temp["yvxu"] += H2["efxu"] * G2["yvef"];
    temp["yVxU"] += H2["eFxU"] * G2["yVeF"];
    temp["YVXU"] += H2["EFXU"] * G2["YVEF"];
    temp["yvxu"] -= G2["efxu"] * H2["yvef"];
    temp["yVxU"] -= G2["eFxU"] * H2["yVeF"];
    temp["YVXU"] -= G2["EFXU"] * H2["YVEF"];
    E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
    E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
    E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];

    // two eta
    temp.zero();
    temp["vyux"] += H2["vymn"] * G2["mnux"];
    temp["vYuX"] += H2["vYmN"] * G2["mNuX"];
    temp["VYUX"] += H2["VYMN"] * G2["MNUX"];
    temp["vyux"] -= G2["vymn"] * H2["mnux"];
    temp["vYuX"] -= G2["vYmN"] * H2["mNuX"];
    temp["VYUX"] -= G2["VYMN"] * H2["MNUX"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
    E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
    E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];

    // one gamma, one eta
    temp.zero();
    temp["vyux"] += H2["vemx"] * G2["myue"];
    temp["vyux"] += H2["vExM"] * G2["yMuE"];
    temp["vyux"] -= G2["vemx"] * H2["myue"];
    temp["vyux"] -= G2["vExM"] * H2["yMuE"];
    E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];

    temp["VYUX"] += H2["eVmX"] * G2["mYeU"];
    temp["VYUX"] += H2["VEXM"] * G2["YMUE"];
    temp["VYUX"] -= G2["eVmX"] * H2["mYeU"];
    temp["VYUX"] -= G2["VEXM"] * H2["YMUE"];
    E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];

    temp["yVxU"] = H2["eVxM"] * G2["yMeU"];
    temp["yVxU"] -= G2["eVxM"] * H2["yMeU"];
    E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];

    temp["vYuX"] = H2["vEmX"] * G2["mYuE"];
    temp["vYuX"] -= G2["vEmX"] * H2["mYuE"];
    E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];

    // one gamma, two eta; two gamma, one eta
    temp.zero();
    temp["yvxu"] += 0.5 * Gamma1_["wz"] * H2["vexw"] * G2["yzue"];
    temp["yvxu"] += Gamma1_["WZ"] * H2["vExW"] * G2["yZuE"];
    temp["yvxu"] -= 0.5 * Gamma1_["wz"] * G2["vexw"] * H2["yzue"];
    temp["yvxu"] -= Gamma1_["WZ"] * G2["vExW"] * H2["yZuE"];

    temp["yvxu"] += 0.5 * Eta1_["wz"] * G2["myuw"] * H2["vzmx"];
    temp["yvxu"] += Eta1_["WZ"] * G2["yMuW"] * H2["vZxM"];
    temp["yvxu"] -= 0.5 * Eta1_["wz"] * H2["myuw"] * G2["vzmx"];
    temp["yvxu"] -= Eta1_["WZ"] * H2["yMuW"] * G2["vZxM"];
    E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];

    temp["YVXU"] += 0.5 * Gamma1_["WZ"] * H2["VEXW"] * G2["YZUE"];
    temp["YVXU"] += Gamma1_["wz"] * H2["eVwX"] * G2["zYeU"];
    temp["YVXU"] -= 0.5 * Gamma1_["WZ"] * G2["VEXW"] * H2["YZUE"];
    temp["YVXU"] -= Gamma1_["wz"] * G2["eVwX"] * H2["zYeU"];

    temp["YVXU"] += 0.5 * Eta1_["WZ"] * G2["MYUW"] * H2["VZMX"];
    temp["YVXU"] += Eta1_["wz"] * H2["zVmX"] * G2["mYwU"];
    temp["YVXU"] -= 0.5 * Eta1_["WZ"] * H2["MYUW"] * G2["VZMX"];
    temp["YVXU"] -= Eta1_["wz"] * G2["zVmX"] * H2["mYwU"];
    E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];

    // <[Hbar2, T2]> C_4 (C_2)^2 ++,--
    temp.zero();
    temp["xyuv"] += 0.125 * H2["xypq"] * G2["pquv"];
    temp["xyuv"] -= 0.25 * H2["xype"] * G2["peuv"];
    temp["xyuv"] -= 0.25 * H2["xypw"] * G2["pzuv"] * Eta1_["wz"];

    temp["xYuV"] += H2["xYpQ"] * G2["pQuV"];
    temp["xYuV"] -= H2["xYpE"] * G2["pEuV"];
    temp["xYuV"] -= H2["xYeP"] * G2["ePuV"];
    temp["xYuV"] -= H2["xYpW"] * G2["pZuV"] * Eta1_["WZ"];
    temp["xYuV"] -= H2["xYwP"] * G2["zPuV"] * Eta1_["wz"];

    temp["XYUV"] += 0.125 * H2["XYPQ"] * G2["PQUV"];
    temp["XYUV"] -= 0.25 * H2["XYPE"] * G2["PEUV"];
    temp["XYUV"] -= 0.25 * H2["XYPW"] * G2["PZUV"] * Eta1_["WZ"];

    // <[Hbar2, T2]> C_4 (C_2)^2 --,++
    temp["xyuv"] += 0.125 * H2["pquv"] * G2["xypq"];
    temp["xyuv"] -= 0.25 * H2["pmuv"] * G2["xypm"];
    temp["xyuv"] -= 0.25 * H2["pzuv"] * G2["xypw"] * Gamma1_["wz"];

    temp["xYuV"] += H2["pQuV"] * G2["xYpQ"];
    temp["xYuV"] -= H2["pMuV"] * G2["xYpM"];
    temp["xYuV"] -= H2["mPuV"] * G2["xYmP"];
    temp["xYuV"] -= H2["pZuV"] * G2["xYpW"] * Gamma1_["WZ"];
    temp["xYuV"] -= H2["zPuV"] * G2["xYwP"] * Gamma1_["wz"];

    temp["XYUV"] += 0.125 * H2["PQUV"] * G2["XYPQ"];
    temp["XYUV"] -= 0.25 * H2["PMUV"] * G2["XYPM"];
    temp["XYUV"] -= 0.25 * H2["PZUV"] * G2["XYPW"] * Gamma1_["WZ"];

    // <[Hbar2, T2]> C_4 (C_2)^2 +-,+-
    temp["xyuv"] += H2["pxum"] * G2["myvp"];
    temp["xyuv"] += H2["xPuM"] * G2["yMvP"];
    temp["xyuv"] += H2["pxuw"] * G2["zyvp"] * Gamma1_["wz"];
    temp["xyuv"] += H2["xPuW"] * G2["yZvP"] * Gamma1_["WZ"];
    temp["xyuv"] -= H2["mxup"] * G2["pyvm"];
    temp["xyuv"] -= H2["xMuP"] * G2["yPvM"];
    temp["xyuv"] -= H2["zxup"] * G2["pyvw"] * Gamma1_["wz"];
    temp["xyuv"] -= H2["xZuP"] * G2["yPvW"] * Gamma1_["WZ"];
    E += temp["xyuv"] * Lambda2_["xyuv"];

    temp["XYUV"] += H2["pXmU"] * G2["mYpV"];
    temp["XYUV"] += H2["PXUM"] * G2["MYVP"];
    temp["XYUV"] += H2["pXwU"] * G2["zYpV"] * Gamma1_["wz"];
    temp["XYUV"] += H2["XPUW"] * G2["YZVP"] * Gamma1_["WZ"];
    temp["XYUV"] -= H2["mXpU"] * G2["pYmV"];
    temp["XYUV"] -= H2["MXUP"] * G2["PYVM"];
    temp["XYUV"] -= H2["zXpU"] * G2["pYwV"] * Gamma1_["wz"];
    temp["XYUV"] -= H2["XZUP"] * G2["YPVW"] * Gamma1_["WZ"];
    E += temp["XYUV"] * Lambda2_["XYUV"];

    temp["xYuV"] += H2["xpum"] * G2["mYpV"];
    temp["xYuV"] += H2["xPuM"] * G2["MYPV"];
    temp["xYuV"] -= H2["pYuM"] * G2["xMpV"];
    temp["xYuV"] -= H2["xPmV"] * G2["mYuP"];
    temp["xYuV"] += H2["pYmV"] * G2["xmup"];
    temp["xYuV"] += H2["PYMV"] * G2["xMuP"];

    temp["xYuV"] += H2["xpuw"] * G2["zYpV"] * Gamma1_["wz"];
    temp["xYuV"] += H2["xPuW"] * G2["ZYPV"] * Gamma1_["WZ"];
    temp["xYuV"] -= H2["pYuW"] * G2["xZpV"] * Gamma1_["WZ"];
    temp["xYuV"] -= H2["xPwV"] * G2["zYuP"] * Gamma1_["wz"];
    temp["xYuV"] += H2["pYwV"] * G2["xzup"] * Gamma1_["wz"];
    temp["xYuV"] += H2["PYWV"] * G2["xZuP"] * Gamma1_["WZ"];

    temp["xYuV"] -= H2["xmup"] * G2["pYmV"];
    temp["xYuV"] -= H2["xMuP"] * G2["PYMV"];
    temp["xYuV"] += H2["mYuP"] * G2["xPmV"];
    temp["xYuV"] += H2["xMpV"] * G2["pYuM"];
    temp["xYuV"] -= H2["mYpV"] * G2["xpum"];
    temp["xYuV"] -= H2["MYPV"] * G2["xPuM"];

    temp["xYuV"] -= H2["xzup"] * G2["pYwV"] * Gamma1_["wz"];
    temp["xYuV"] -= H2["xZuP"] * G2["PYWV"] * Gamma1_["WZ"];
    temp["xYuV"] += H2["zYuP"] * G2["xPwV"] * Gamma1_["wz"];
    temp["xYuV"] += H2["xZpV"] * G2["pYuW"] * Gamma1_["WZ"];
    temp["xYuV"] -= H2["zYpV"] * G2["xpuw"] * Gamma1_["wz"];
    temp["xYuV"] -= H2["ZYPV"] * G2["xPuW"] * Gamma1_["WZ"];
    E += temp["xYuV"] * Lambda2_["xYuV"];

    // <[Hbar2, T2]> C_6 C_2
    if (options_.get_str("THREEPDC") != "ZERO") {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
        temp["xyzuvw"] += H2["yzpu"] * G2["pxvw"];
        temp["xyzuvw"] -= H2["zpuv"] * G2["xywp"];
        E += 0.25 * temp["xyzuvw"] * Lambda3_["xyzuvw"];

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
        temp["XYZUVW"] += H2["YZPU"] * G2["PXVW"];
        temp["XYZUVW"] -= H2["ZPUV"] * G2["XYWP"];
        E += 0.25 * temp["XYZUVW"] * Lambda3_["XYZUVW"];

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"});
        temp["xyZuvW"] += 2.0 * H2["yZuP"] * G2["xPvW"];
        temp["xyZuvW"] += H2["xypu"] * G2["pZvW"];
        temp["xyZuvW"] += H2["yZpW"] * G2["pxuv"];

        temp["xyZuvW"] -= 2.0 * H2["yPuW"] * G2["xZvP"];
        temp["xyZuvW"] -= H2["pZuW"] * G2["xyvp"];
        temp["xyZuvW"] -= H2["ypuv"] * G2["xZpW"];
        E += 0.5 * temp["xyZuvW"] * Lambda3_["xyZuvW"];

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
        temp["xYZuVW"] += 2.0 * H2["xZpV"] * G2["pYuW"];
        temp["xYZuVW"] += H2["YZPV"] * G2["xPuW"];
        temp["xYZuVW"] += H2["xZuP"] * G2["PYVW"];

        temp["xYZuVW"] -= 2.0 * H2["pZuV"] * G2["xYpW"];
        temp["xYZuVW"] -= H2["xPuV"] * G2["YZWP"];
        temp["xYZuVW"] -= H2["ZPVW"] * G2["xYuP"];
        E += 0.5 * temp["xYZuVW"] * Lambda3_["xYZuVW"];
    }

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    dsrg_time_.add("220", timer.get());
}

void MRDSRG::H1_G1_C1(BlockedTensor& H1, BlockedTensor& G1, const double& alpha,
                      BlockedTensor& C1) {
    Timer timer;

    C1["sp"] += alpha * H1["qp"] * G1["sq"];
    C1["qr"] -= alpha * H1["qp"] * G1["pr"];

    C1["SP"] += alpha * H1["QP"] * G1["SQ"];
    C1["QR"] -= alpha * H1["QP"] * G1["PR"];

    dsrg_time_.add("111", timer.get());
}

void MRDSRG::H1_G2_C1(BlockedTensor& H1, BlockedTensor& G2, const double& alpha,
                      BlockedTensor& C1) {
    Timer timer;

    C1["os"] += alpha * H1["qm"] * G2["moqs"];
    C1["os"] += alpha * H1["qu"] * G2["voqs"] * Gamma1_["uv"];
    C1["os"] -= alpha * H1["mp"] * G2["poms"];
    C1["os"] -= alpha * H1["vp"] * G2["pous"] * Gamma1_["uv"];
    C1["os"] += alpha * H1["QM"] * G2["oMsQ"];
    C1["os"] += alpha * H1["QU"] * G2["oVsQ"] * Gamma1_["UV"];
    C1["os"] -= alpha * H1["MP"] * G2["oPsM"];
    C1["os"] -= alpha * H1["VP"] * G2["oPsU"] * Gamma1_["UV"];

    C1["OS"] += alpha * H1["qm"] * G2["mOqS"];
    C1["OS"] += alpha * H1["qu"] * G2["vOqS"] * Gamma1_["uv"];
    C1["OS"] -= alpha * H1["mp"] * G2["pOmS"];
    C1["OS"] -= alpha * H1["vp"] * G2["pOuS"] * Gamma1_["uv"];
    C1["OS"] += alpha * H1["QM"] * G2["MOQS"];
    C1["OS"] += alpha * H1["QU"] * G2["VOQS"] * Gamma1_["UV"];
    C1["OS"] -= alpha * H1["MP"] * G2["POMS"];
    C1["OS"] -= alpha * H1["VP"] * G2["POUS"] * Gamma1_["UV"];

    dsrg_time_.add("121", timer.get());
}

void MRDSRG::H2_G2_C1(BlockedTensor& H2, BlockedTensor& G2, const double& alpha,
                      BlockedTensor& C1) {
    Timer timer;

    // [Hbar2, T2] (C_2)^3 -> C1 +--,++-
    C1["pq"] += 0.5 * alpha * H2["rsqm"] * G2["pmrs"];
    C1["pq"] += alpha * H2["rSqM"] * G2["pMrS"];
    C1["PQ"] += 0.5 * alpha * H2["RSQM"] * G2["PMRS"];
    C1["PQ"] += alpha * H2["rSmQ"] * G2["mPrS"];

    C1["pq"] += 0.5 * alpha * H2["rsqu"] * G2["pvrs"] * Gamma1_["uv"];
    C1["pq"] += alpha * H2["rSqU"] * G2["pVrS"] * Gamma1_["UV"];
    C1["PQ"] += 0.5 * alpha * H2["RSQU"] * G2["PVRS"] * Gamma1_["UV"];
    C1["PQ"] += alpha * H2["rSuQ"] * G2["vPrS"] * Gamma1_["uv"];

    C1["pq"] += 0.5 * alpha * H2["mnqr"] * G2["prmn"];
    C1["pq"] += alpha * H2["mNqR"] * G2["pRmN"];
    C1["PQ"] += 0.5 * alpha * H2["MNQR"] * G2["PRMN"];
    C1["PQ"] += alpha * H2["mNrQ"] * G2["rPmN"];

    C1["pq"] += alpha * H2["mvqr"] * G2["prmu"] * Gamma1_["uv"];
    C1["pq"] += alpha * H2["mVqR"] * G2["pRmU"] * Gamma1_["UV"];
    C1["pq"] += alpha * H2["vMqR"] * G2["pRuM"] * Gamma1_["uv"];
    C1["PQ"] += alpha * H2["mVrQ"] * G2["rPmU"] * Gamma1_["UV"];
    C1["PQ"] += alpha * H2["vMrQ"] * G2["rPuM"] * Gamma1_["uv"];
    C1["PQ"] += alpha * H2["MVQR"] * G2["PRMU"] * Gamma1_["UV"];

    C1["pq"] += 0.5 * alpha * H2["vyqr"] * G2["prux"] * Gamma1_["xy"] * Gamma1_["uv"];
    C1["pq"] += alpha * H2["vYqR"] * G2["pRuX"] * Gamma1_["XY"] * Gamma1_["uv"];
    C1["PQ"] += 0.5 * alpha * H2["VYQR"] * G2["PRUX"] * Gamma1_["XY"] * Gamma1_["UV"];
    C1["PQ"] += alpha * H2["vYrQ"] * G2["rPuX"] * Gamma1_["XY"] * Gamma1_["uv"];

    C1["pq"] -= alpha * H2["rnqm"] * G2["pmrn"];
    C1["pq"] -= alpha * H2["rNqM"] * G2["pMrN"];
    C1["pq"] -= alpha * H2["nRqM"] * G2["pMnR"];
    C1["PQ"] -= alpha * H2["rNmQ"] * G2["mPrN"];
    C1["PQ"] -= alpha * H2["nRmQ"] * G2["mPnR"];
    C1["PQ"] -= alpha * H2["RNQM"] * G2["MPNR"];

    C1["pq"] -= alpha * H2["rvqm"] * G2["pmru"] * Gamma1_["uv"];
    C1["pq"] -= alpha * H2["rVqM"] * G2["pMrU"] * Gamma1_["UV"];
    C1["pq"] -= alpha * H2["vRqM"] * G2["pMuR"] * Gamma1_["uv"];
    C1["PQ"] -= alpha * H2["rVmQ"] * G2["mPrU"] * Gamma1_["UV"];
    C1["PQ"] -= alpha * H2["vRmQ"] * G2["mPuR"] * Gamma1_["uv"];
    C1["PQ"] -= alpha * H2["RVQM"] * G2["MPUR"] * Gamma1_["UV"];

    C1["pq"] -= alpha * H2["rnqu"] * G2["pvrn"] * Gamma1_["uv"];
    C1["pq"] -= alpha * H2["rNqU"] * G2["pVrN"] * Gamma1_["UV"];
    C1["pq"] -= alpha * H2["nRqU"] * G2["pVnR"] * Gamma1_["UV"];
    C1["PQ"] -= alpha * H2["rNuQ"] * G2["vPrN"] * Gamma1_["uv"];
    C1["PQ"] -= alpha * H2["nRuQ"] * G2["vPnR"] * Gamma1_["uv"];
    C1["PQ"] -= alpha * H2["RNQU"] * G2["VPNR"] * Gamma1_["UV"];

    C1["pq"] -= alpha * H2["ryqu"] * G2["pvrx"] * Gamma1_["uv"] * Gamma1_["xy"];
    C1["pq"] -= alpha * H2["rYqU"] * G2["pVrX"] * Gamma1_["UV"] * Gamma1_["XY"];
    C1["pq"] -= alpha * H2["yRqU"] * G2["pVxR"] * Gamma1_["UV"] * Gamma1_["xy"];
    C1["PQ"] -= alpha * H2["yRuQ"] * G2["vPxR"] * Gamma1_["uv"] * Gamma1_["xy"];
    C1["PQ"] -= alpha * H2["rYuQ"] * G2["vPrX"] * Gamma1_["uv"] * Gamma1_["XY"];
    C1["PQ"] -= alpha * H2["RYUQ"] * G2["VPRX"] * Gamma1_["UV"] * Gamma1_["XY"];

    // [Hbar2, T2] (C_2)^3 -> C1 ++-,+--
    C1["pq"] -= 0.5 * alpha * H2["epsr"] * G2["rsqe"];
    C1["pq"] -= alpha * H2["pErS"] * G2["rSqE"];
    C1["PQ"] -= 0.5 * alpha * H2["EPSR"] * G2["RSQE"];
    C1["PQ"] -= alpha * H2["ePrS"] * G2["rSeQ"];

    C1["pq"] -= 0.5 * alpha * H2["vpsr"] * G2["rsqu"] * Eta1_["uv"];
    C1["pq"] -= alpha * H2["pVrS"] * G2["rSqU"] * Eta1_["UV"];
    C1["PQ"] -= 0.5 * alpha * H2["VPSR"] * G2["RSQU"] * Eta1_["UV"];
    C1["PQ"] -= alpha * H2["vPrS"] * G2["rSuQ"] * Eta1_["uv"];

    C1["pq"] -= 0.5 * alpha * H2["rpfe"] * G2["efqr"];
    C1["pq"] -= alpha * H2["pReF"] * G2["eFqR"];
    C1["PQ"] -= 0.5 * alpha * H2["RPFE"] * G2["EFQR"];
    C1["PQ"] -= alpha * H2["rPeF"] * G2["eFrQ"];

    C1["pq"] -= alpha * H2["rpue"] * G2["evqr"] * Eta1_["uv"];
    C1["pq"] -= alpha * H2["pRuE"] * G2["vEqR"] * Eta1_["uv"];
    C1["pq"] -= alpha * H2["pReU"] * G2["eVqR"] * Eta1_["UV"];
    C1["PQ"] -= alpha * H2["rPuE"] * G2["vErQ"] * Eta1_["uv"];
    C1["PQ"] -= alpha * H2["rPeU"] * G2["eVrQ"] * Eta1_["UV"];
    C1["PQ"] -= alpha * H2["RPUE"] * G2["EVQR"] * Eta1_["UV"];

    C1["pq"] -= 0.5 * alpha * H2["rpxu"] * G2["vyqr"] * Eta1_["uv"] * Eta1_["xy"];
    C1["pq"] -= alpha * H2["pRxU"] * G2["yVqR"] * Eta1_["UV"] * Eta1_["xy"];
    C1["PQ"] -= 0.5 * alpha * H2["RPXU"] * G2["VYQR"] * Eta1_["UV"] * Eta1_["XY"];
    C1["PQ"] -= alpha * H2["rPxU"] * G2["yVrQ"] * Eta1_["UV"] * Eta1_["xy"];

    C1["pq"] += alpha * H2["fper"] * G2["reqf"];
    C1["pq"] += alpha * H2["pFrE"] * G2["rEqF"];
    C1["pq"] += alpha * H2["pFeR"] * G2["eRqF"];
    C1["PQ"] += alpha * H2["fPeR"] * G2["eRfQ"];
    C1["PQ"] += alpha * H2["fPrE"] * G2["rEfQ"];
    C1["PQ"] += alpha * H2["FPER"] * G2["REQF"];

    C1["pq"] += alpha * H2["vper"] * G2["requ"] * Eta1_["uv"];
    C1["pq"] += alpha * H2["pVrE"] * G2["rEqU"] * Eta1_["UV"];
    C1["pq"] += alpha * H2["pVeR"] * G2["eRqU"] * Eta1_["UV"];
    C1["PQ"] += alpha * H2["vPeR"] * G2["eRuQ"] * Eta1_["uv"];
    C1["PQ"] += alpha * H2["vPrE"] * G2["rEuQ"] * Eta1_["uv"];
    C1["PQ"] += alpha * H2["VPER"] * G2["REQU"] * Eta1_["UV"];

    C1["pq"] += alpha * H2["fpur"] * G2["rvqf"] * Eta1_["uv"];
    C1["pq"] += alpha * H2["pFrU"] * G2["rVqF"] * Eta1_["UV"];
    C1["pq"] += alpha * H2["pFuR"] * G2["vRqF"] * Eta1_["uv"];
    C1["PQ"] += alpha * H2["fPrU"] * G2["rVfQ"] * Eta1_["UV"];
    C1["PQ"] += alpha * H2["fPuR"] * G2["vRfQ"] * Eta1_["uv"];
    C1["PQ"] += alpha * H2["FPUR"] * G2["RVQF"] * Eta1_["UV"];

    C1["pq"] += alpha * H2["ypur"] * G2["rvqx"] * Eta1_["uv"] * Eta1_["xy"];
    C1["pq"] += alpha * H2["pYrU"] * G2["rVqX"] * Eta1_["UV"] * Eta1_["XY"];
    C1["pq"] += alpha * H2["pYuR"] * G2["vRqX"] * Eta1_["uv"] * Eta1_["XY"];
    C1["PQ"] += alpha * H2["yPuR"] * G2["vRxQ"] * Eta1_["uv"] * Eta1_["xy"];
    C1["PQ"] += alpha * H2["yPrU"] * G2["rVxQ"] * Eta1_["UV"] * Eta1_["xy"];
    C1["PQ"] += alpha * H2["YPUR"] * G2["RVQX"] * Eta1_["UV"] * Eta1_["XY"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    C1["pq"] += 0.25 * alpha * H2["xyqr"] * G2["pruv"] * Lambda2_["vuyx"];
    C1["pq"] += alpha * H2["xYqR"] * G2["pRuV"] * Lambda2_["uVxY"];
    C1["PQ"] += 0.25 * alpha * H2["XYQR"] * G2["PRUV"] * Lambda2_["VUYX"];
    C1["PQ"] += alpha * H2["xYrQ"] * G2["rPuV"] * Lambda2_["uVxY"];

    C1["pq"] -= 0.25 * alpha * G2["xyqr"] * H2["pruv"] * Lambda2_["vuyx"];
    C1["pq"] -= alpha * G2["xYqR"] * H2["pRuV"] * Lambda2_["uVxY"];
    C1["PQ"] -= 0.25 * alpha * G2["XYQR"] * H2["PRUV"] * Lambda2_["VUYX"];
    C1["PQ"] -= alpha * G2["xYrQ"] * H2["rPuV"] * Lambda2_["uVxY"];

    C1["pq"] += alpha * H2["rxqu"] * G2["pyrv"] * Lambda2_["uvxy"];
    C1["pq"] += alpha * H2["rxqu"] * G2["pYrV"] * Lambda2_["uVxY"];
    C1["pq"] += alpha * H2["rXqU"] * G2["pyrv"] * Lambda2_["vUyX"];
    C1["pq"] += alpha * H2["rXqU"] * G2["pYrV"] * Lambda2_["UVXY"];
    C1["pq"] -= alpha * H2["xRqU"] * G2["pYvR"] * Lambda2_["vUxY"];
    C1["PQ"] -= alpha * H2["rXuQ"] * G2["yPrV"] * Lambda2_["uVyX"];
    C1["PQ"] += alpha * H2["xRuQ"] * G2["yPvR"] * Lambda2_["uvxy"];
    C1["PQ"] += alpha * H2["xRuQ"] * G2["PYRV"] * Lambda2_["uVxY"];
    C1["PQ"] += alpha * H2["RXQU"] * G2["yPvR"] * Lambda2_["vUyX"];
    C1["PQ"] += alpha * H2["RXQU"] * G2["PYRV"] * Lambda2_["UVXY"];

    C1["pq"] -= alpha * G2["rxqu"] * H2["pyrv"] * Lambda2_["uvxy"];
    C1["pq"] -= alpha * G2["rxqu"] * H2["pYrV"] * Lambda2_["uVxY"];
    C1["pq"] -= alpha * G2["rXqU"] * H2["pyrv"] * Lambda2_["vUyX"];
    C1["pq"] -= alpha * G2["rXqU"] * H2["pYrV"] * Lambda2_["UVXY"];
    C1["pq"] += alpha * G2["xRqU"] * H2["pYvR"] * Lambda2_["vUxY"];
    C1["PQ"] += alpha * G2["rXuQ"] * H2["yPrV"] * Lambda2_["uVyX"];
    C1["PQ"] -= alpha * G2["xRuQ"] * H2["yPvR"] * Lambda2_["uvxy"];
    C1["PQ"] -= alpha * G2["xRuQ"] * H2["PYRV"] * Lambda2_["uVxY"];
    C1["PQ"] -= alpha * G2["RXQU"] * H2["yPvR"] * Lambda2_["vUyX"];
    C1["PQ"] -= alpha * G2["RXQU"] * H2["PYRV"] * Lambda2_["UVXY"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    C1["pq"] += 0.5 * alpha * H2["rpuq"] * G2["xyrv"] * Lambda2_["vuyx"];
    C1["pq"] += 0.5 * alpha * H2["pRqU"] * G2["XYRV"] * Lambda2_["VUYX"];
    C1["PQ"] += 0.5 * alpha * H2["RPUQ"] * G2["XYRV"] * Lambda2_["VUYX"];
    C1["PQ"] += 0.5 * alpha * H2["rPuQ"] * G2["xyrv"] * Lambda2_["vuyx"];
    C1["pq"] += alpha * H2["rpuq"] * G2["xYrV"] * Lambda2_["uVxY"];
    C1["pq"] += alpha * H2["pRqU"] * G2["xYvR"] * Lambda2_["vUxY"];
    C1["PQ"] += alpha * H2["RPUQ"] * G2["yXvR"] * Lambda2_["vUyX"];
    C1["PQ"] += alpha * H2["rPuQ"] * G2["xYrV"] * Lambda2_["uVxY"];

    C1["pq"] -= 0.5 * alpha * G2["rpuq"] * H2["xyrv"] * Lambda2_["vuyx"];
    C1["pq"] -= 0.5 * alpha * G2["pRqU"] * H2["XYRV"] * Lambda2_["VUYX"];
    C1["PQ"] -= 0.5 * alpha * G2["RPUQ"] * H2["XYRV"] * Lambda2_["VUYX"];
    C1["PQ"] -= 0.5 * alpha * G2["rPuQ"] * H2["xyrv"] * Lambda2_["vuyx"];
    C1["pq"] -= alpha * G2["rpuq"] * H2["xYrV"] * Lambda2_["uVxY"];
    C1["pq"] -= alpha * G2["pRqU"] * H2["xYvR"] * Lambda2_["vUxY"];
    C1["PQ"] -= alpha * G2["RPUQ"] * H2["yXvR"] * Lambda2_["vUyX"];
    C1["PQ"] -= alpha * G2["rPuQ"] * H2["xYrV"] * Lambda2_["uVxY"];

    C1["pq"] -= 0.5 * alpha * H2["xprq"] * G2["ryuv"] * Lambda2_["vuyx"];
    C1["pq"] -= 0.5 * alpha * H2["pXqR"] * G2["RYUV"] * Lambda2_["VUYX"];
    C1["PQ"] -= 0.5 * alpha * H2["XPRQ"] * G2["RYUV"] * Lambda2_["VUYX"];
    C1["PQ"] -= 0.5 * alpha * H2["xPrQ"] * G2["ryuv"] * Lambda2_["vuyx"];
    C1["pq"] -= alpha * H2["xprq"] * G2["rYuV"] * Lambda2_["uVxY"];
    C1["pq"] -= alpha * H2["pXqR"] * G2["yRuV"] * Lambda2_["uVyX"];
    C1["PQ"] -= alpha * H2["XPRQ"] * G2["yRvU"] * Lambda2_["vUyX"];
    C1["PQ"] -= alpha * H2["xPrQ"] * G2["rYuV"] * Lambda2_["uVxY"];

    C1["pq"] += 0.5 * alpha * G2["xprq"] * H2["ryuv"] * Lambda2_["vuyx"];
    C1["pq"] += 0.5 * alpha * G2["pXqR"] * H2["RYUV"] * Lambda2_["VUYX"];
    C1["PQ"] += 0.5 * alpha * G2["XPRQ"] * H2["RYUV"] * Lambda2_["VUYX"];
    C1["PQ"] += 0.5 * alpha * G2["xPrQ"] * H2["ryuv"] * Lambda2_["vuyx"];
    C1["pq"] += alpha * G2["xprq"] * H2["rYuV"] * Lambda2_["uVxY"];
    C1["pq"] += alpha * G2["pXqR"] * H2["yRuV"] * Lambda2_["uVyX"];
    C1["PQ"] += alpha * G2["XPRQ"] * H2["yRvU"] * Lambda2_["vUyX"];
    C1["PQ"] += alpha * G2["xPrQ"] * H2["rYuV"] * Lambda2_["uVxY"];

    dsrg_time_.add("221", timer.get());
}

void MRDSRG::H1_G2_C2(BlockedTensor& H1, BlockedTensor& G2, const double& alpha,
                      BlockedTensor& C2) {
    Timer timer;

    C2["tops"] += alpha * H1["rp"] * G2["tors"];
    C2["torp"] += alpha * H1["sp"] * G2["tors"];
    C2["qors"] -= alpha * H1["qt"] * G2["tors"];
    C2["tqrs"] -= alpha * H1["qo"] * G2["tors"];

    C2["tOpS"] += alpha * H1["rp"] * G2["tOrS"];
    C2["tOrP"] += alpha * H1["SP"] * G2["tOrS"];
    C2["qOrS"] -= alpha * H1["qt"] * G2["tOrS"];
    C2["tQrS"] -= alpha * H1["QO"] * G2["tOrS"];

    C2["TOPS"] += alpha * H1["RP"] * G2["TORS"];
    C2["TORP"] += alpha * H1["SP"] * G2["TORS"];
    C2["QORS"] -= alpha * H1["QT"] * G2["TORS"];
    C2["TQRS"] -= alpha * H1["QO"] * G2["TORS"];

    dsrg_time_.add("122", timer.get());
}

void MRDSRG::H2_G2_C2(BlockedTensor& H2, BlockedTensor& G2, const double& alpha,
                      BlockedTensor& C2) {
    Timer timer;

    // --,++
    C2["pqrs"] += 0.5 * alpha * H2["tors"] * G2["pqto"];
    C2["pQrS"] += alpha * H2["tOrS"] * G2["pQtO"];
    C2["PQRS"] += 0.5 * alpha * H2["TORS"] * G2["PQTO"];

    C2["pqrs"] -= alpha * H2["mtrs"] * G2["pqmt"];
    C2["pQrS"] -= alpha * H2["mTrS"] * G2["pQmT"];
    C2["pQrS"] -= alpha * H2["tMrS"] * G2["pQtM"];
    C2["PQRS"] -= alpha * H2["MTRS"] * G2["PQMT"];

    C2["pqrs"] -= alpha * H2["vtrs"] * G2["pqut"] * Gamma1_["uv"];
    C2["pQrS"] -= alpha * H2["vTrS"] * G2["pQuT"] * Gamma1_["uv"];
    C2["pQrS"] -= alpha * H2["tVrS"] * G2["pQtU"] * Gamma1_["UV"];
    C2["PQRS"] -= alpha * H2["VTRS"] * G2["PQUT"] * Gamma1_["UV"];

    // ++,--
    C2["pqrs"] += 0.5 * alpha * H2["pqto"] * G2["tors"];
    C2["pQrS"] += alpha * H2["pQtO"] * G2["tOrS"];
    C2["PQRS"] += 0.5 * alpha * H2["PQTO"] * G2["TORS"];

    C2["pqrs"] -= alpha * H2["pqet"] * G2["etrs"];
    C2["pQrS"] -= alpha * H2["pQeT"] * G2["eTrS"];
    C2["pQrS"] -= alpha * H2["pQtE"] * G2["tErS"];
    C2["PQRS"] -= alpha * H2["PQET"] * G2["ETRS"];

    C2["pqrs"] -= alpha * H2["pqut"] * G2["vtrs"] * Eta1_["uv"];
    C2["pQrS"] -= alpha * H2["pQuT"] * G2["vTrS"] * Eta1_["uv"];
    C2["pQrS"] -= alpha * H2["pQtU"] * G2["tVrS"] * Eta1_["UV"];
    C2["PQRS"] -= alpha * H2["PQUT"] * G2["VTRS"] * Eta1_["UV"];

    // +-,+-
    C2["pqrs"] += alpha * H2["tprm"] * G2["mqst"];
    C2["pqrs"] += alpha * H2["pTrM"] * G2["qMsT"];
    C2["pqrs"] -= alpha * H2["tqrm"] * G2["mpst"];
    C2["pqrs"] -= alpha * H2["qTrM"] * G2["pMsT"];
    C2["pqrs"] -= alpha * H2["tpsm"] * G2["mqrt"];
    C2["pqrs"] -= alpha * H2["pTsM"] * G2["qMrT"];
    C2["pqrs"] += alpha * H2["tqsm"] * G2["mprt"];
    C2["pqrs"] += alpha * H2["qTsM"] * G2["pMrT"];

    C2["pQrS"] += alpha * H2["tpmr"] * G2["mQtS"];
    C2["pQrS"] += alpha * H2["pTrM"] * G2["MQTS"];
    C2["pQrS"] -= alpha * H2["tQrM"] * G2["pMtS"];
    C2["pQrS"] -= alpha * H2["pTmS"] * G2["mQrT"];
    C2["pQrS"] += alpha * H2["tQmS"] * G2["pmrt"];
    C2["pQrS"] += alpha * H2["TQMS"] * G2["pMrT"];

    C2["PQRS"] += alpha * H2["tPmR"] * G2["mQtS"];
    C2["PQRS"] += alpha * H2["TPRM"] * G2["MQST"];
    C2["PQRS"] -= alpha * H2["tQmR"] * G2["mPtS"];
    C2["PQRS"] -= alpha * H2["TQRM"] * G2["MPST"];
    C2["PQRS"] -= alpha * H2["tPmS"] * G2["mQtR"];
    C2["PQRS"] -= alpha * H2["TPSM"] * G2["MQRT"];
    C2["PQRS"] += alpha * H2["tQmS"] * G2["mPtR"];
    C2["PQRS"] += alpha * H2["TQSM"] * G2["MPRT"];

    C2["pqrs"] += alpha * H2["tpru"] * G2["vqst"] * Gamma1_["uv"];
    C2["pqrs"] += alpha * H2["pTrU"] * G2["qVsT"] * Gamma1_["UV"];
    C2["pqrs"] -= alpha * H2["tqru"] * G2["vpst"] * Gamma1_["uv"];
    C2["pqrs"] -= alpha * H2["qTrU"] * G2["pVsT"] * Gamma1_["UV"];
    C2["pqrs"] -= alpha * H2["tpsu"] * G2["vqrt"] * Gamma1_["uv"];
    C2["pqrs"] -= alpha * H2["pTsU"] * G2["qVrT"] * Gamma1_["UV"];
    C2["pqrs"] += alpha * H2["tqsu"] * G2["vprt"] * Gamma1_["uv"];
    C2["pqrs"] += alpha * H2["qTsU"] * G2["pVrT"] * Gamma1_["UV"];

    C2["pQrS"] += alpha * H2["tpur"] * G2["vQtS"] * Gamma1_["uv"];
    C2["pQrS"] += alpha * H2["pTrU"] * G2["VQTS"] * Gamma1_["UV"];
    C2["pQrS"] -= alpha * H2["tQrU"] * G2["pVtS"] * Gamma1_["UV"];
    C2["pQrS"] -= alpha * H2["pTuS"] * G2["vQrT"] * Gamma1_["uv"];
    C2["pQrS"] += alpha * H2["tQuS"] * G2["pvrt"] * Gamma1_["uv"];
    C2["pQrS"] += alpha * H2["TQUS"] * G2["pVrT"] * Gamma1_["UV"];

    C2["PQRS"] += alpha * H2["tPuR"] * G2["vQtS"] * Gamma1_["uv"];
    C2["PQRS"] += alpha * H2["TPRU"] * G2["VQST"] * Gamma1_["UV"];
    C2["PQRS"] -= alpha * H2["tQuR"] * G2["vPtS"] * Gamma1_["uv"];
    C2["PQRS"] -= alpha * H2["TQRU"] * G2["VPST"] * Gamma1_["UV"];
    C2["PQRS"] -= alpha * H2["tPuS"] * G2["vQtR"] * Gamma1_["uv"];
    C2["PQRS"] -= alpha * H2["TPSU"] * G2["VQRT"] * Gamma1_["UV"];
    C2["PQRS"] += alpha * H2["tQuS"] * G2["vPtR"] * Gamma1_["uv"];
    C2["PQRS"] += alpha * H2["TQSU"] * G2["VPRT"] * Gamma1_["UV"];

    C2["pqrs"] -= alpha * G2["tprm"] * H2["mqst"];
    C2["pqrs"] -= alpha * G2["pTrM"] * H2["qMsT"];
    C2["pqrs"] += alpha * G2["tqrm"] * H2["mpst"];
    C2["pqrs"] += alpha * G2["qTrM"] * H2["pMsT"];
    C2["pqrs"] += alpha * G2["tpsm"] * H2["mqrt"];
    C2["pqrs"] += alpha * G2["pTsM"] * H2["qMrT"];
    C2["pqrs"] -= alpha * G2["tqsm"] * H2["mprt"];
    C2["pqrs"] -= alpha * G2["qTsM"] * H2["pMrT"];

    C2["pQrS"] -= alpha * G2["tpmr"] * H2["mQtS"];
    C2["pQrS"] -= alpha * G2["pTrM"] * H2["MQTS"];
    C2["pQrS"] += alpha * G2["tQrM"] * H2["pMtS"];
    C2["pQrS"] += alpha * G2["pTmS"] * H2["mQrT"];
    C2["pQrS"] -= alpha * G2["tQmS"] * H2["pmrt"];
    C2["pQrS"] -= alpha * G2["TQMS"] * H2["pMrT"];

    C2["PQRS"] -= alpha * G2["tPmR"] * H2["mQtS"];
    C2["PQRS"] -= alpha * G2["TPRM"] * H2["MQST"];
    C2["PQRS"] += alpha * G2["tQmR"] * H2["mPtS"];
    C2["PQRS"] += alpha * G2["TQRM"] * H2["MPST"];
    C2["PQRS"] += alpha * G2["tPmS"] * H2["mQtR"];
    C2["PQRS"] += alpha * G2["TPSM"] * H2["MQRT"];
    C2["PQRS"] -= alpha * G2["tQmS"] * H2["mPtR"];
    C2["PQRS"] -= alpha * G2["TQSM"] * H2["MPRT"];

    C2["pqrs"] -= alpha * G2["tpru"] * H2["vqst"] * Gamma1_["uv"];
    C2["pqrs"] -= alpha * G2["pTrU"] * H2["qVsT"] * Gamma1_["UV"];
    C2["pqrs"] += alpha * G2["tqru"] * H2["vpst"] * Gamma1_["uv"];
    C2["pqrs"] += alpha * G2["qTrU"] * H2["pVsT"] * Gamma1_["UV"];
    C2["pqrs"] += alpha * G2["tpsu"] * H2["vqrt"] * Gamma1_["uv"];
    C2["pqrs"] += alpha * G2["pTsU"] * H2["qVrT"] * Gamma1_["UV"];
    C2["pqrs"] -= alpha * G2["tqsu"] * H2["vprt"] * Gamma1_["uv"];
    C2["pqrs"] -= alpha * G2["qTsU"] * H2["pVrT"] * Gamma1_["UV"];

    C2["pQrS"] -= alpha * G2["tpur"] * H2["vQtS"] * Gamma1_["uv"];
    C2["pQrS"] -= alpha * G2["pTrU"] * H2["VQTS"] * Gamma1_["UV"];
    C2["pQrS"] += alpha * G2["tQrU"] * H2["pVtS"] * Gamma1_["UV"];
    C2["pQrS"] += alpha * G2["pTuS"] * H2["vQrT"] * Gamma1_["uv"];
    C2["pQrS"] -= alpha * G2["tQuS"] * H2["pvrt"] * Gamma1_["uv"];
    C2["pQrS"] -= alpha * G2["TQUS"] * H2["pVrT"] * Gamma1_["UV"];

    C2["PQRS"] -= alpha * G2["tPuR"] * H2["vQtS"] * Gamma1_["uv"];
    C2["PQRS"] -= alpha * G2["TPRU"] * H2["VQST"] * Gamma1_["UV"];
    C2["PQRS"] += alpha * G2["tQuR"] * H2["vPtS"] * Gamma1_["uv"];
    C2["PQRS"] += alpha * G2["TQRU"] * H2["VPST"] * Gamma1_["UV"];
    C2["PQRS"] += alpha * G2["tPuS"] * H2["vQtR"] * Gamma1_["uv"];
    C2["PQRS"] += alpha * G2["TPSU"] * H2["VQRT"] * Gamma1_["UV"];
    C2["PQRS"] -= alpha * G2["tQuS"] * H2["vPtR"] * Gamma1_["uv"];
    C2["PQRS"] -= alpha * G2["TQSU"] * H2["VPRT"] * Gamma1_["UV"];

    dsrg_time_.add("222", timer.get());
}
}
}
