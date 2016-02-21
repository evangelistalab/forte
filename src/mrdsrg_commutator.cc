#include <algorithm>
#include <vector>
#include <map>
#include <boost/format.hpp>

#include "helpers.h"
#include "mrdsrg.h"

namespace psi{ namespace forte{

void MRDSRG::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0){
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

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T1] -> C0 : %12.3f",timer.get());
    }
    dsrg_time_.add("110",timer.get());
}

void MRDSRG::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0){
    Timer timer;
    BlockedTensor temp;
    double E = 0.0;

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aaaa"});
    temp["uvxy"] += H1["ex"] * T2["uvey"];
    temp["uvxy"] -= H1["vm"] * T2["umxy"];
    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AAAA"});
    temp["UVXY"] += H1["EX"] * T2["UVEY"];
    temp["UVXY"] -= H1["VM"] * T2["UMXY"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aAaA"});
    temp["uVxY"] += H1["ex"] * T2["uVeY"];
    temp["uVxY"] += H1["EY"] * T2["uVxE"];
    temp["uVxY"] -= H1["VM"] * T2["uMxY"];
    temp["uVxY"] -= H1["um"] * T2["mVxY"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    E  *= alpha;
    C0 += E;

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f",timer.get());
    }
    dsrg_time_.add("120",timer.get());
}

void MRDSRG::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0){
    Timer timer;
    BlockedTensor temp;
    double E = 0.0;

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aaaa"});
    temp["uvxy"] += H2["evxy"] * T1["ue"];
    temp["uvxy"] -= H2["uvmy"] * T1["mx"];
    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AAAA"});
    temp["UVXY"] += H2["EVXY"] * T1["UE"];
    temp["UVXY"] -= H2["UVMY"] * T1["MX"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aAaA"});
    temp["uVxY"] += H2["eVxY"] * T1["ue"];
    temp["uVxY"] += H2["uExY"] * T1["VE"];
    temp["uVxY"] -= H2["uVmY"] * T1["mx"];
    temp["uVxY"] -= H2["uVxM"] * T1["MY"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    E  *= alpha;
    C0 += E;

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f",timer.get());
    }
    dsrg_time_.add("210",timer.get());
}

void MRDSRG::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0){
    Timer timer;

    // <[Hbar2, T2]> (C_2)^4
    double E = H2["eFmN"] * T2["mNeF"];
    E += 0.25 * H2["efmn"] * T2["mnef"];
    E += 0.25 * H2["EFMN"] * T2["MNEF"];

    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["vu"] += 0.5 * H2["efmu"] * T2["mvef"];
    temp["vu"] += H2["fEuM"] * T2["vMfE"];
    temp["VU"] += 0.5 * H2["EFMU"] * T2["MVEF"];
    temp["VU"] += H2["eFmU"] * T2["mVeF"];
    E += temp["vu"] * Gamma1_["uv"];
    E += temp["VU"] * Gamma1_["UV"];

    temp.zero();
    temp["vu"] += 0.5 * H2["vemn"] * T2["mnue"];
    temp["vu"] += H2["vEmN"] * T2["mNuE"];
    temp["VU"] += 0.5 * H2["VEMN"] * T2["MNUE"];
    temp["VU"] += H2["eVnM"] * T2["nMeU"];
    E += temp["vu"] * Eta1_["uv"];
    E += temp["VU"] * Eta1_["UV"];

    temp = BTF_->build(tensor_type_,"temp",spin_cases({"aaaa"}));
    temp["yvxu"] += H2["efxu"] * T2["yvef"];
    temp["yVxU"] += H2["eFxU"] * T2["yVeF"];
    temp["YVXU"] += H2["EFXU"] * T2["YVEF"];
    E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
    E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
    E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];

    temp.zero();
    temp["vyux"] += H2["vymn"] * T2["mnux"];
    temp["vYuX"] += H2["vYmN"] * T2["mNuX"];
    temp["VYUX"] += H2["VYMN"] * T2["MNUX"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
    E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
    E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];

    temp.zero();
    temp["vyux"] += H2["vemx"] * T2["myue"];
    temp["vyux"] += H2["vExM"] * T2["yMuE"];
    temp["VYUX"] += H2["eVmX"] * T2["mYeU"];
    temp["VYUX"] += H2["VEXM"] * T2["YMUE"];
    E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
    E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
    temp["yVxU"] = H2["eVxM"] * T2["yMeU"];
    E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
    temp["vYuX"] = H2["vEmX"] * T2["mYuE"];
    E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];

    temp.zero();
    temp["yvxu"] += 0.5 * Gamma1_["wz"] * H2["vexw"] * T2["yzue"];
    temp["yvxu"] += Gamma1_["WZ"] * H2["vExW"] * T2["yZuE"];
    temp["yvxu"] += 0.5 * Eta1_["wz"] * T2["myuw"] * H2["vzmx"];
    temp["yvxu"] += Eta1_["WZ"] * T2["yMuW"] * H2["vZxM"];
    E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];

    temp["YVXU"] += 0.5 * Gamma1_["WZ"] * H2["VEXW"] * T2["YZUE"];
    temp["YVXU"] += Gamma1_["wz"] * H2["eVwX"] * T2["zYeU"];
    temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2["MYUW"] * H2["VZMX"];
    temp["YVXU"] += Eta1_["wz"] * H2["zVmX"] * T2["mYwU"];
    E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];

    // <[Hbar2, T2]> C_4 (C_2)^2 HH -- combined with PH
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",spin_cases({"aaaa"}));
    temp["uvxy"] += 0.125 * H2["uvmn"] * T2["mnxy"];
    temp["uvxy"] += 0.25 * Gamma1_["wz"] * H2["uvmw"] * T2["mzxy"];
    temp["uVxY"] += H2["uVmN"] * T2["mNxY"];
    temp["uVxY"] += Gamma1_["wz"] * T2["zMxY"] * H2["uVwM"];
    temp["uVxY"] += Gamma1_["WZ"] * H2["uVmW"] * T2["mZxY"];
    temp["UVXY"] += 0.125 * H2["UVMN"] * T2["MNXY"];
    temp["UVXY"] += 0.25 * Gamma1_["WZ"] * H2["UVMW"] * T2["MZXY"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PP -- combined with PH
    temp["uvxy"] += 0.125 * H2["efxy"] * T2["uvef"];
    temp["uvxy"] += 0.25 * Eta1_["wz"] * T2["uvew"] * H2["ezxy"];
    temp["uVxY"] += H2["eFxY"] * T2["uVeF"];
    temp["uVxY"] += Eta1_["wz"] * H2["zExY"] * T2["uVwE"];
    temp["uVxY"] += Eta1_["WZ"] * T2["uVeW"] * H2["eZxY"];
    temp["UVXY"] += 0.125 * H2["EFXY"] * T2["UVEF"];
    temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2["UVEW"] * H2["EZXY"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PH
    temp["uvxy"] += H2["eumx"] * T2["mvey"];
    temp["uvxy"] += H2["uExM"] * T2["vMyE"];
    temp["uvxy"] += Gamma1_["wz"] * T2["zvey"] * H2["euwx"];
    temp["uvxy"] += Gamma1_["WZ"] * H2["uExW"] * T2["vZyE"];
    temp["uvxy"] += Eta1_["zw"] * H2["wumx"] * T2["mvzy"];
    temp["uvxy"] += Eta1_["ZW"] * T2["vMyZ"] * H2["uWxM"];
    E += temp["uvxy"] * Lambda2_["xyuv"];

    temp["UVXY"] += H2["eUmX"] * T2["mVeY"];
    temp["UVXY"] += H2["EUMX"] * T2["MVEY"];
    temp["UVXY"] += Gamma1_["wz"] * T2["zVeY"] * H2["eUwX"];
    temp["UVXY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["EUWX"];
    temp["UVXY"] += Eta1_["zw"] * H2["wUmX"] * T2["mVzY"];
    temp["UVXY"] += Eta1_["ZW"] * H2["WUMX"] * T2["MVZY"];
    E += temp["UVXY"] * Lambda2_["XYUV"];

    temp["uVxY"] += H2["uexm"] * T2["mVeY"];
    temp["uVxY"] += H2["uExM"] * T2["MVEY"];
    temp["uVxY"] -= H2["eVxM"] * T2["uMeY"];
    temp["uVxY"] -= H2["uEmY"] * T2["mVxE"];
    temp["uVxY"] += H2["eVmY"] * T2["umxe"];
    temp["uVxY"] += H2["EVMY"] * T2["uMxE"];

    temp["uVxY"] += Gamma1_["wz"] * T2["zVeY"] * H2["uexw"];
    temp["uVxY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["uExW"];
    temp["uVxY"] -= Gamma1_["WZ"] * H2["eVxW"] * T2["uZeY"];
    temp["uVxY"] -= Gamma1_["wz"] * T2["zVxE"] * H2["uEwY"];
    temp["uVxY"] += Gamma1_["wz"] * T2["zuex"] * H2["eVwY"];
    temp["uVxY"] -= Gamma1_["WZ"] * H2["EVYW"] * T2["uZxE"];

    temp["uVxY"] += Eta1_["zw"] * H2["wumx"] * T2["mVzY"];
    temp["uVxY"] += Eta1_["ZW"] * T2["VMYZ"] * H2["uWxM"];
    temp["uVxY"] -= Eta1_["zw"] * H2["wVxM"] * T2["uMzY"];
    temp["uVxY"] -= Eta1_["ZW"] * T2["mVxZ"] * H2["uWmY"];
    temp["uVxY"] += Eta1_["zw"] * T2["umxz"] * H2["wVmY"];
    temp["uVxY"] += Eta1_["ZW"] * H2["WVMY"] * T2["uMxZ"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    // <[Hbar2, T2]> C_6 C_2
    if(options_.get_str("THREEPDC") != "ZERO"){
        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aaaaaa"});
        temp["uvwxyz"] += H2["uviz"] * T2["iwxy"];      //  aaaaaa from hole
        temp["uvwxyz"] += H2["waxy"] * T2["uvaz"];      //  aaaaaa from particle
        E += 0.25 * temp["uvwxyz"] * Lambda3_["xyzuvw"];

        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AAAAAA"});
        temp["UVWXYZ"] += H2["UVIZ"] * T2["IWXY"];      //  AAAAAA from hole
        temp["UVWXYZ"] += H2["WAXY"] * T2["UVAZ"];      //  AAAAAA from particle
        E += 0.25 * temp["UVWXYZ"] * Lambda3_["XYZUVW"];

        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aaAaaA"});
        temp["uvWxyZ"] -= H2["uviy"] * T2["iWxZ"];      //  aaAaaA from hole
        temp["uvWxyZ"] -= H2["uWiZ"] * T2["ivxy"];      //  aaAaaA from hole
        temp["uvWxyZ"] += 2.0 * H2["uWyI"] * T2["vIxZ"];//  aaAaaA from hole

        temp["uvWxyZ"] += H2["aWxZ"] * T2["uvay"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= H2["vaxy"] * T2["uWaZ"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= 2.0 * H2["vAxZ"] * T2["uWyA"];//  aaAaaA from particle
        E += 0.5 * temp["uvWxyZ"] * Lambda3_["xyZuvW"];

        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aAAaAA"});
        temp["uVWxYZ"] -= H2["VWIZ"] * T2["uIxY"];      //  aAAaAA from hole
        temp["uVWxYZ"] -= H2["uVxI"] * T2["IWYZ"];      //  aAAaAA from hole
        temp["uVWxYZ"] += 2.0 * H2["uViZ"] * T2["iWxY"];//  aAAaAA from hole

        temp["uVWxYZ"] += H2["uAxY"] * T2["VWAZ"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= H2["WAYZ"] * T2["uVxA"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= 2.0 * H2["aWxY"] * T2["uVaZ"];//  aAAaAA from particle
        E += 0.5 * temp["uVWxYZ"] * Lambda3_["xYZuVW"];
    }

    // multiply prefactor and copy to C0
    E  *= alpha;
    C0 += E;

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f",timer.get());
    }
    dsrg_time_.add("220",timer.get());
}

void MRDSRG::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["qa"] -= alpha * T1["ia"] * H1["qi"];

    C1["IP"] += alpha * H1["AP"] * T1["IA"];
    C1["QA"] -= alpha * T1["IA"] * H1["QI"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("111",timer.get());
}

void MRDSRG::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["ia"] += alpha * H1["bm"] * T2["imab"];
    C1["ia"] += alpha * H1["bu"] * T2["ivab"] * Gamma1_["uv"];
    C1["ia"] -= alpha * H1["vj"] * T2["ijau"] * Gamma1_["uv"];
    C1["ia"] += alpha * H1["BM"] * T2["iMaB"];
    C1["ia"] += alpha * H1["BU"] * T2["iVaB"] * Gamma1_["UV"];
    C1["ia"] -= alpha * H1["VJ"] * T2["iJaU"] * Gamma1_["UV"];

    C1["IA"] += alpha * H1["bm"] * T2["mIbA"];
    C1["IA"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vIbA"];
    C1["IA"] -= alpha * H1["vj"] * T2["jIuA"] * Gamma1_["uv"];
    C1["IA"] += alpha * H1["BM"] * T2["IMAB"];
    C1["IA"] += alpha * H1["BU"] * T2["IVAB"] * Gamma1_["UV"];
    C1["IA"] -= alpha * H1["VJ"] * T2["IJAU"] * Gamma1_["UV"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("121",timer.get());
}

void MRDSRG::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
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

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("211",timer.get());
}

void MRDSRG::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    C1["ir"] += 0.5 * alpha * H2["abrm"] * T2["imab"];
    C1["ir"] += alpha * H2["aBrM"] * T2["iMaB"];
    C1["IR"] += 0.5 * alpha * H2["ABRM"] * T2["IMAB"];
    C1["IR"] += alpha * H2["aBmR"] * T2["mIaB"];

    C1["ir"] += 0.5 * alpha * Gamma1_["uv"] * H2["abru"] * T2["ivab"];
    C1["ir"] += alpha * Gamma1_["UV"] * H2["aBrU"] * T2["iVaB"];
    C1["IR"] += 0.5 * alpha * Gamma1_["UV"] * H2["ABRU"] * T2["IVAB"];
    C1["IR"] += alpha * Gamma1_["uv"] * H2["aBuR"] * T2["vIaB"];

    C1["ir"] += 0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vyrj"];
    C1["IR"] += 0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYRJ"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hHaA"});
    temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"];
    C1["ir"] += alpha * temp["iJvY"] * H2["vYrJ"];
    C1["IR"] += alpha * temp["jIvY"] * H2["vYjR"];

    C1["ir"] -= alpha * Gamma1_["uv"] * H2["vbrm"] * T2["imub"];
    C1["ir"] -= alpha * Gamma1_["uv"] * H2["vBrM"] * T2["iMuB"];
    C1["ir"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * H2["bVrM"];
    C1["IR"] -= alpha * Gamma1_["UV"] * H2["VBRM"] * T2["IMUB"];
    C1["IR"] -= alpha * Gamma1_["UV"] * H2["bVmR"] * T2["mIbU"];
    C1["IR"] -= alpha * Gamma1_["uv"] * H2["vBmR"] * T2["mIuB"];

    C1["ir"] -= alpha * H2["vbrx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["iyub"];
    C1["ir"] -= alpha * H2["vBrX"] * Gamma1_["uv"] * Gamma1_["XY"] * T2["iYuB"];
    C1["ir"] -= alpha * H2["bVrX"] * Gamma1_["XY"] * Gamma1_["UV"] * T2["iYbU"];
    C1["IR"] -= alpha * H2["VBRX"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["IYUB"];
    C1["IR"] -= alpha * H2["vBxR"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yIuB"];
    C1["IR"] -= alpha * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"] * H2["bVxR"];

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    C1["pa"] -= 0.5 * alpha * H2["peij"] * T2["ijae"];
    C1["pa"] -= alpha * H2["pEiJ"] * T2["iJaE"];
    C1["PA"] -= 0.5 * alpha * H2["PEIJ"] * T2["IJAE"];
    C1["PA"] -= alpha * H2["ePiJ"] * T2["iJeA"];

    C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijau"] * H2["pvij"];
    C1["pa"] -= alpha * Eta1_["UV"] * T2["iJaU"] * H2["pViJ"];
    C1["PA"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJAU"] * H2["PVIJ"];
    C1["PA"] -= alpha * Eta1_["uv"] * T2["iJuA"] * H2["vPiJ"];

    C1["pa"] -= 0.5 * alpha * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * H2["pbux"];
    C1["PA"] -= 0.5 * alpha * T2["VYAB"] * Eta1_["UV"] * Eta1_["XY"] * H2["PBUX"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aApP"});
    temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"];
    C1["pa"] -= alpha * H2["pBuX"] * temp["uXaB"];
    C1["PA"] -= alpha * H2["bPuX"] * temp["uXbA"];

    C1["pa"] += alpha * Eta1_["uv"] * T2["vjae"] * H2["peuj"];
    C1["pa"] += alpha * Eta1_["uv"] * T2["vJaE"] * H2["pEuJ"];
    C1["pa"] += alpha * Eta1_["UV"] * H2["pEjU"] * T2["jVaE"];
    C1["PA"] += alpha * Eta1_["UV"] * T2["VJAE"] * H2["PEUJ"];
    C1["PA"] += alpha * Eta1_["uv"] * T2["vJeA"] * H2["ePuJ"];
    C1["PA"] += alpha * Eta1_["UV"] * H2["ePjU"] * T2["jVeA"];

    C1["pa"] += alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * H2["pyuj"];
    C1["pa"] += alpha * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"] * H2["pYuJ"];
    C1["pa"] += alpha * T2["jVaX"] * Eta1_["XY"] * Eta1_["UV"] * H2["pYjU"];
    C1["PA"] += alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * H2["PYUJ"];
    C1["PA"] += alpha * T2["vJxA"] * Eta1_["uv"] * Eta1_["xy"] * H2["yPuJ"];
    C1["PA"] += alpha * H2["yPjU"] * Eta1_["UV"] * Eta1_["xy"] * T2["jVxA"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    C1["ir"] +=  0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * H2["uvrj"];
    C1["IR"] +=  0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * H2["UVRJ"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hHaA"});
    temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"];
    C1["ir"] += alpha * H2["uVrJ"] * temp["iJuV"];
    C1["IR"] += alpha * H2["uVjR"] * temp["jIuV"];

    C1["pa"] -=  0.25 * alpha * Lambda2_["xyuv"] * T2["uvab"] * H2["pbxy"];
    C1["PA"] -=  0.25 * alpha * Lambda2_["XYUV"] * T2["UVAB"] * H2["PBXY"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aApP"});
    temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"];
    C1["pa"] -= alpha * H2["pBxY"] * temp["xYaB"];
    C1["PA"] -= alpha * H2["bPxY"] * temp["xYbA"];

    C1["ir"] -= alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uArX"];
    C1["IR"] -= alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["aUxR"];
    C1["pa"] += alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["pUxI"];
    C1["PA"] += alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uPiX"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hapa"});
    temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"];
    temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"];
    C1["ir"] += alpha * temp["ixau"] * H2["aurx"];
    C1["pa"] -= alpha * H2["puix"] * temp["ixau"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hApA"});
    temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"];
    temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"];
    C1["ir"] += alpha * temp["iXaU"] * H2["aUrX"];
    C1["pa"] -= alpha * H2["pUiX"] * temp["iXaU"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aHaP"});
    temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"];
    temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"];
    C1["IR"] += alpha * temp["xIuA"] * H2["uAxR"];
    C1["PA"] -= alpha * H2["uPxI"] * temp["xIuA"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"HAPA"});
    temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"];
    temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"];
    C1["IR"] += alpha * temp["IXAU"] * H2["AURX"];
    C1["PA"] -= alpha * H2["PUIX"] * temp["IXAU"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"pa"});
    temp["au"] += 0.5 * Lambda2_["xyuv"] * H2["avxy"];
    temp["au"] += Lambda2_["xYuV"] * H2["aVxY"];
    C1["jb"] += alpha * temp["au"] * T2["ujab"];
    C1["JB"] += alpha * temp["au"] * T2["uJaB"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"PA"});
    temp["AU"] += 0.5 * Lambda2_["XYUV"] * H2["AVXY"];
    temp["AU"] += Lambda2_["xYvU"] * H2["vAxY"];
    C1["jb"] += alpha * temp["AU"] * T2["jUbA"];
    C1["JB"] += alpha * temp["AU"] * T2["UJAB"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ah"});
    temp["xi"] += 0.5 * Lambda2_["xyuv"] * H2["uviy"];
    temp["xi"] += Lambda2_["xYuV"] * H2["uViY"];
    C1["jb"] -= alpha * temp["xi"] * T2["ijxb"];
    C1["JB"] -= alpha * temp["xi"] * T2["iJxB"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AH"});
    temp["XI"] += 0.5 * Lambda2_["XYUV"] * H2["UVIY"];
    temp["XI"] += Lambda2_["yXvU"] * H2["vUyI"];
    C1["jb"] -= alpha * temp["XI"] * T2["jIbX"];
    C1["JB"] -= alpha * temp["XI"] * T2["IJXB"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"av"});
    temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"];
    temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"];
    C1["qs"] += alpha * temp["xe"] * H2["eqxs"];
    C1["QS"] += alpha * temp["xe"] * H2["eQxS"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AV"});
    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"];
    temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"];
    C1["qs"] += alpha * temp["XE"] * H2["qEsX"];
    C1["QS"] += alpha * temp["XE"] * H2["EQXS"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ca"});
    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"];
    temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"];
    C1["qs"] -= alpha * temp["mu"] * H2["uqms"];
    C1["QS"] -= alpha * temp["mu"] * H2["uQmS"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"CA"});
    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"];
    temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"];
    C1["qs"] -= alpha * temp["MU"] * H2["qUsM"];
    C1["QS"] -= alpha * temp["MU"] * H2["UQMS"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("221",timer.get());
}

void MRDSRG::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
    Timer timer;

    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
    C2["ijap"] += alpha * T2["ijab"] * H1["bp"];
    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
    C2["iqab"] -= alpha * T2["ijab"] * H1["qj"];

    C2["iJpB"] += alpha * T2["iJaB"] * H1["ap"];
    C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"];
    C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"];
    C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"];

    C2["IJPB"] += alpha * T2["IJAB"] * H1["AP"];
    C2["IJAP"] += alpha * T2["IJAB"] * H1["BP"];
    C2["QJAB"] -= alpha * T2["IJAB"] * H1["QI"];
    C2["IQAB"] -= alpha * T2["IJAB"] * H1["QJ"];

//    // probably not worth doing the following because contracting one index should be fast
//    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hhpg","HHPG"});
//    temp["ijap"] = alpha * T2["ijab"] * H1["bp"];
//    temp["IJAP"] = alpha * T2["IJAB"] * H1["BP"];

//    C2["ijpb"] -= temp["ijbp"]; // use permutation of temp
//    C2["ijap"] += temp["ijap"]; // explicitly evaluate by temp
//    C2["iJpA"] += alpha * T2["iJbA"] * H1["bp"];
//    C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"];
//    C2["IJPB"] -= temp["IJBP"]; // use permutation of temp
//    C2["IJAP"] += temp["IJAP"]; // explicitly evaluate by temp

//    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ghpp","GHPP"});
//    temp["qjab"] = alpha * T2["ijab"] * H1["qi"];
//    temp["QJAB"] = alpha * T2["IJAB"] * H1["QI"];

//    C2["qjab"] -= temp["qjab"]; // explicitly evaluate by temp
//    C2["iqab"] += temp["qiab"]; // use permutation of temp
//    C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"];
//    C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"];
//    C2["QJAB"] -= temp["QJAB"]; // explicitly evaluate by temp
//    C2["IQAB"] += temp["QIAB"]; // use permutation of temp

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("122",timer.get());
}

void MRDSRG::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2){
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

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("212",timer.get());
}

void MRDSRG::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
    Timer timer;

    // particle-particle contractions
    C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"];
    C2["iJrS"] += alpha * H2["aBrS"] * T2["iJaB"];
    C2["IJRS"] += 0.5 * alpha * H2["ABRS"] * T2["IJAB"];

    C2["ijrs"] -= alpha * Gamma1_["xy"] * H2["ybrs"] * T2["ijxb"];
    C2["iJrS"] -= alpha * Gamma1_["xy"] * H2["yBrS"] * T2["iJxB"];
    C2["iJrS"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * H2["bYrS"];
    C2["IJRS"] -= alpha * Gamma1_["XY"] * H2["YBRS"] * T2["IJXB"];

    // hole-hole contractions
    C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"];
    C2["pQaB"] += alpha * H2["pQiJ"] * T2["iJaB"];
    C2["PQAB"] += 0.5 * alpha * H2["PQIJ"] * T2["IJAB"];

    C2["pqab"] -= alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
    C2["pQaB"] -= alpha * Eta1_["xy"] * T2["yJaB"] * H2["pQxJ"];
    C2["pQaB"] -= alpha * Eta1_["XY"] * H2["pQjX"] * T2["jYaB"];
    C2["PQAB"] -= alpha * Eta1_["XY"] * T2["YJAB"] * H2["PQXJ"];

    // hole-particle contractions
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ghgp"});
    temp["qjsb"] += alpha * H2["aqms"] * T2["mjab"];
    temp["qjsb"] += alpha * H2["qAsM"] * T2["jMbA"];
    temp["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
    temp["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
    temp["qjsb"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["ijxb"];
    temp["qjsb"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["jIbX"];
    C2["qjsb"] += temp["qjsb"];
    C2["jqsb"] -= temp["qjsb"];
    C2["qjbs"] -= temp["qjsb"];
    C2["jqbs"] += temp["qjsb"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"GHGP"});
    temp["QJSB"] += alpha * H2["AQMS"] * T2["MJAB"];
    temp["QJSB"] += alpha * H2["aQmS"] * T2["mJaB"];
    temp["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
    temp["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
    temp["QJSB"] -= alpha * Gamma1_["XY"] * H2["YQIS"] * T2["IJXB"];
    temp["QJSB"] -= alpha * Gamma1_["xy"] * H2["yQiS"] * T2["iJxB"];
    C2["QJSB"] += temp["QJSB"];
    C2["JQSB"] -= temp["QJSB"];
    C2["QJBS"] -= temp["QJSB"];
    C2["JQBS"] += temp["QJSB"];

    C2["qJsB"] += alpha * H2["aqms"] * T2["mJaB"];
    C2["qJsB"] += alpha * H2["qAsM"] * T2["MJAB"];
    C2["qJsB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aqxs"];
    C2["qJsB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["qAsX"];
    C2["qJsB"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["iJxB"];
    C2["qJsB"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["IJXB"];

    C2["iQsB"] -= alpha * T2["iMaB"] * H2["aQsM"];
    C2["iQsB"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * H2["aQsX"];
    C2["iQsB"] += alpha * Gamma1_["xy"] * H2["yQsJ"] * T2["iJxB"];

    C2["qJaS"] -= alpha * T2["mJaB"] * H2["qBmS"];
    C2["qJaS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["qBxS"];
    C2["qJaS"] += alpha * Gamma1_["XY"] * H2["qYiS"] * T2["iJaX"];

    C2["iQaS"] += alpha * T2["imab"] * H2["bQmS"];
    C2["iQaS"] += alpha * T2["iMaB"] * H2["BQMS"];
    C2["iQaS"] += alpha * Gamma1_["xy"] * T2["iyab"] * H2["bQxS"];
    C2["iQaS"] += alpha * Gamma1_["XY"] * T2["iYaB"] * H2["BQXS"];
    C2["iQaS"] -= alpha * Gamma1_["xy"] * H2["yQjS"] * T2["ijax"];
    C2["iQaS"] -= alpha * Gamma1_["XY"] * H2["YQJS"] * T2["iJaX"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("222",timer.get());
}

}}
