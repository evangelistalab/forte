/**
 * Solve several multipliers directly.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "../dsrg_mrpt2.h"
#include "helpers/timer.h"

using namespace ambit;
using namespace psi;

namespace forte {

void DSRG_MRPT2::set_tau() {
    outfile->Printf("\n    Initializing multipliers for two-body amplitude.. ");
    Tau1 = BTF_->build(CoreTensor, "Tau1", spin_cases({"hhpp"}));
    Tau2 = BTF_->build(CoreTensor, "Tau2", spin_cases({"hhpp"}));
    // Tau * Delta
    // <[V, T2]> (C_2)^4
    if (PT2_TERM) {
        Tau2["mnef"] += 0.25 * V_["efmn"];
        Tau2["mneu"] += 0.25 * V_["evmn"] * Eta1_["uv"];
        Tau2["mnue"] += 0.25 * V_["vemn"] * Eta1_["uv"];
        Tau2["mnux"] += 0.25 * V_["vymn"] * Eta1_["uv"] * Eta1_["xy"];
        Tau2["mvef"] += 0.25 * V_["efmu"] * Gamma1_["uv"];
        Tau2["mvez"] += 0.25 * V_["ewmu"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["mvze"] += 0.25 * V_["wemu"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["mvzx"] += 0.25 * V_["wymu"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        Tau2["vmef"] += 0.25 * V_["efum"] * Gamma1_["uv"];
        Tau2["vmez"] += 0.25 * V_["ewum"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["vmze"] += 0.25 * V_["weum"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["vmzx"] += 0.25 * V_["wyum"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        Tau2["vyef"] += 0.25 * V_["efux"] * Gamma1_["uv"] * Gamma1_["xy"];
        Tau2["vyez"] += 0.25 * V_["ewux"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];
        Tau2["vyze"] += 0.25 * V_["weux"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];
        Tau2["v,y,z,u1"] +=
            0.25 * V_["w,a1,u,x"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"] * Eta1_["u1,a1"];

        Tau2["MNEF"] += 0.25 * V_["EFMN"];
        Tau2["MNEU"] += 0.25 * V_["EVMN"] * Eta1_["UV"];
        Tau2["MNUE"] += 0.25 * V_["VEMN"] * Eta1_["UV"];
        Tau2["MNUX"] += 0.25 * V_["VYMN"] * Eta1_["UV"] * Eta1_["XY"];
        Tau2["MVEF"] += 0.25 * V_["EFMU"] * Gamma1_["UV"];
        Tau2["MVEZ"] += 0.25 * V_["EWMU"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["MVZE"] += 0.25 * V_["WEMU"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["MVZX"] += 0.25 * V_["WYMU"] * Gamma1_["UV"] * Eta1_["ZW"] * Eta1_["XY"];
        Tau2["VMEF"] += 0.25 * V_["EFUM"] * Gamma1_["UV"];
        Tau2["VMEZ"] += 0.25 * V_["EWUM"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["VMZE"] += 0.25 * V_["WEUM"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["VMZX"] += 0.25 * V_["WYUM"] * Gamma1_["UV"] * Eta1_["ZW"] * Eta1_["XY"];
        Tau2["VYEF"] += 0.25 * V_["EFUX"] * Gamma1_["UV"] * Gamma1_["XY"];
        Tau2["VYEZ"] += 0.25 * V_["EWUX"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"];
        Tau2["VYZE"] += 0.25 * V_["WEUX"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"];
        Tau2["V,Y,Z,U1"] +=
            0.25 * V_["W,A1,U,X"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"] * Eta1_["U1,A1"];

        Tau2["mNeF"] += 0.25 * V_["eFmN"];
        Tau2["mNeZ"] += 0.25 * V_["eWmN"] * Eta1_["ZW"];
        Tau2["mNzE"] += 0.25 * V_["wEmN"] * Eta1_["zw"];
        Tau2["mNzX"] += 0.25 * V_["wYmN"] * Eta1_["zw"] * Eta1_["XY"];
        Tau2["mVeF"] += 0.25 * V_["eFmU"] * Gamma1_["UV"];
        Tau2["mVeZ"] += 0.25 * V_["eWmU"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["mVzE"] += 0.25 * V_["wEmU"] * Gamma1_["UV"] * Eta1_["zw"];
        Tau2["mVzX"] += 0.25 * V_["wYmU"] * Gamma1_["UV"] * Eta1_["zw"] * Eta1_["XY"];
        Tau2["vMeF"] += 0.25 * V_["eFuM"] * Gamma1_["uv"];
        Tau2["vMeZ"] += 0.25 * V_["eWuM"] * Gamma1_["uv"] * Eta1_["ZW"];
        Tau2["vMzE"] += 0.25 * V_["wEuM"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["vMzX"] += 0.25 * V_["wYuM"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["XY"];
        Tau2["vYeF"] += 0.25 * V_["eFuX"] * Gamma1_["uv"] * Gamma1_["XY"];
        Tau2["vYeZ"] += 0.25 * V_["eWuX"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["ZW"];
        Tau2["vYzE"] += 0.25 * V_["wEuX"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["zw"];
        Tau2["v,Y,z,U1"] +=
            0.25 * V_["w,A1,u,X"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["zw"] * Eta1_["U1,A1"];
    }
    // <[V, T2]> C_4 (C_2)^2 PP
    if (X1_TERM) {
        Tau2["uvef"] += 0.125 * V_["efxy"] * Lambda2_["xyuv"];
        Tau2["uvez"] += 0.125 * V_["ewxy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Tau2["uvze"] += 0.125 * V_["wexy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Tau2["u,v,z,u1"] +=
            0.125 * V_["w,a1,x,y"] * Eta1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xyuv"];
        Tau2["UVEF"] += 0.125 * V_["EFXY"] * Lambda2_["XYUV"];
        Tau2["UVEZ"] += 0.125 * V_["EWXY"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Tau2["UVZE"] += 0.125 * V_["WEXY"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Tau2["U,V,Z,U1"] +=
            0.125 * V_["W,A1,X,Y"] * Eta1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["XYUV"];
        Tau2["uVeF"] += 0.250 * V_["eFxY"] * Lambda2_["xYuV"];
        Tau2["uVeZ"] += 0.250 * V_["eWxY"] * Eta1_["ZW"] * Lambda2_["xYuV"];
        Tau2["uVzE"] += 0.250 * V_["wExY"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Tau2["u,V,z,U1"] +=
            0.250 * V_["w,A1,x,Y"] * Eta1_["zw"] * Eta1_["U1,A1"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        Tau2["mnxy"] += 0.125 * V_["uvmn"] * Lambda2_["xyuv"];
        Tau2["mwxy"] += 0.125 * V_["uvmz"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Tau2["a1,m,x,y"] += 0.125 * V_["u,v,u1,m"] * Gamma1_["u1,a1"] * Lambda2_["xyuv"];
        Tau2["a1,w,x,y"] +=
            0.125 * V_["u,v,u1,z"] * Gamma1_["u1,a1"] * Gamma1_["zw"] * Lambda2_["xyuv"];

        Tau2["MNXY"] += 0.125 * V_["UVMN"] * Lambda2_["XYUV"];
        Tau2["MWXY"] += 0.125 * V_["UVMZ"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Tau2["WMXY"] += 0.125 * V_["UVZM"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Tau2["W,A1,X,Y"] +=
            0.125 * V_["U,V,Z,U1"] * Gamma1_["ZW"] * Gamma1_["U1,A1"] * Lambda2_["XYUV"];

        Tau2["mNxY"] += 0.250 * V_["uVmN"] * Lambda2_["xYuV"];
        Tau2["mWxY"] += 0.250 * V_["uVmZ"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Tau2["wMxY"] += 0.250 * V_["uVzM"] * Gamma1_["zw"] * Lambda2_["xYuV"];
        Tau2["w,A1,x,Y"] +=
            0.250 * V_["u,V,z,U1"] * Gamma1_["zw"] * Gamma1_["U1,A1"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
        temp["muye"] -= 0.25 * V_["vemx"] * Lambda2_["xyuv"];
        temp["muyz"] -= 0.25 * V_["vwmx"] * Eta1_["zw"] * Lambda2_["xyuv"];
        temp["zuye"] -= 0.25 * V_["vewx"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        temp["z,u,y,u1"] -=
            0.25 * V_["v,a1,w,x"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xyuv"];
        temp["muye"] -= 0.25 * V_["eVmX"] * Lambda2_["yXuV"];
        temp["muyz"] -= 0.25 * V_["wVmX"] * Eta1_["zw"] * Lambda2_["yXuV"];
        temp["zuye"] -= 0.25 * V_["eVwX"] * Gamma1_["zw"] * Lambda2_["yXuV"];
        temp["z,u,y,u1"] -=
            0.25 * V_["a1,V,w,X"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["yXuV"];
        temp["MUYE"] -= 0.25 * V_["VEMX"] * Lambda2_["XYUV"];
        temp["MUYZ"] -= 0.25 * V_["VWMX"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        temp["ZUYE"] -= 0.25 * V_["VEWX"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        temp["Z,U,Y,U1"] -=
            0.25 * V_["V,A1,W,X"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["XYUV"];
        temp["MUYE"] -= 0.25 * V_["vExM"] * Lambda2_["xYvU"];
        temp["MUYZ"] -= 0.25 * V_["vWxM"] * Eta1_["ZW"] * Lambda2_["xYvU"];
        temp["ZUYE"] -= 0.25 * V_["vExW"] * Gamma1_["ZW"] * Lambda2_["xYvU"];
        temp["Z,U,Y,U1"] -=
            0.25 * V_["v,A1,x,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["xYvU"];

        /************************************ Symmetrization ************************************/
        Tau2["mUyE"] -= 0.25 * V_["vEmX"] * Lambda2_["yXvU"];
        Tau2["mUyZ"] -= 0.25 * V_["vWmX"] * Eta1_["ZW"] * Lambda2_["yXvU"];
        Tau2["zUyE"] -= 0.25 * V_["vEwX"] * Gamma1_["zw"] * Lambda2_["yXvU"];
        Tau2["z,U,y,U1"] -=
            0.25 * V_["v,A1,w,X"] * Gamma1_["zw"] * Eta1_["U1,A1"] * Lambda2_["yXvU"];
        Tau2["muye"]     += temp["muye"];
        Tau2["muyz"]     += temp["muyz"];
        Tau2["zuye"]     += temp["zuye"];
        Tau2["z,u,y,u1"] += temp["z,u,y,u1"];
        Tau2["MUYE"]     += temp["MUYE"]; 
        Tau2["MUYZ"]     += temp["MUYZ"]; 
        Tau2["ZUYE"]     += temp["ZUYE"]; 
        Tau2["Z,U,Y,U1"] += temp["Z,U,Y,U1"];
        
        Tau2["uMyE"] -= 0.25 * V_["vExM"] * Lambda2_["xyuv"];
        Tau2["uMyZ"] -= 0.25 * V_["vWxM"] * Eta1_["ZW"] * Lambda2_["xyuv"];
        Tau2["uZyE"] -= 0.25 * V_["vExW"] * Gamma1_["ZW"] * Lambda2_["xyuv"];
        Tau2["u,Z,y,U1"] -=
            0.25 * V_["v,A1,x,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["xyuv"];
        Tau2["uMyE"] += 0.25 * V_["VEXM"] * Lambda2_["yXuV"];
        Tau2["uMyZ"] += 0.25 * V_["VWXM"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Tau2["uZyE"] += 0.25 * V_["VEXW"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Tau2["u,Z,y,U1"] +=
            0.25 * V_["V,A1,X,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["yXuV"];
        Tau2["umye"]     -= temp["muye"];
        Tau2["umyz"]     -= temp["muyz"];
        Tau2["uzye"]     -= temp["zuye"];
        Tau2["u,z,y,u1"] -= temp["z,u,y,u1"];
        Tau2["UMYE"]     -= temp["MUYE"]; 
        Tau2["UMYZ"]     -= temp["MUYZ"]; 
        Tau2["UZYE"]     -= temp["ZUYE"]; 
        Tau2["U,Z,Y,U1"] -= temp["Z,U,Y,U1"];

        Tau2["uMeY"] -= 0.25 * V_["eVxM"] * Lambda2_["xYuV"];
        Tau2["uMzY"] -= 0.25 * V_["wVxM"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Tau2["uZeY"] -= 0.25 * V_["eVxW"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Tau2["u,Z,u1,Y"] -=
            0.25 * V_["a1,V,x,W"] * Gamma1_["ZW"] * Eta1_["u1,a1"] * Lambda2_["xYuV"];
        Tau2["umey"]     += temp["muye"];
        Tau2["umzy"]     += temp["muyz"];
        Tau2["uzey"]     += temp["zuye"];
        Tau2["u,z,u1,y"] += temp["z,u,y,u1"];
        Tau2["UMEY"]     += temp["MUYE"]; 
        Tau2["UMZY"]     += temp["MUYZ"]; 
        Tau2["UZEY"]     += temp["ZUYE"]; 
        Tau2["U,Z,U1,Y"] += temp["Z,U,Y,U1"];

        Tau2["mUeY"] -= 0.25 * V_["eVmX"] * Lambda2_["XYUV"];
        Tau2["mUzY"] -= 0.25 * V_["wVmX"] * Eta1_["zw"] * Lambda2_["XYUV"];
        Tau2["zUeY"] -= 0.25 * V_["eVwX"] * Gamma1_["zw"] * Lambda2_["XYUV"];
        Tau2["z,U,u1,Y"] -=
            0.25 * V_["a1,V,w,X"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["XYUV"];
        Tau2["mUeY"] += 0.25 * V_["evmx"] * Lambda2_["xYvU"];
        Tau2["mUzY"] += 0.25 * V_["wvmx"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Tau2["zUeY"] += 0.25 * V_["evwx"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Tau2["z,U,u1,Y"] +=
            0.25 * V_["a1,v,w,x"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xYvU"];
        Tau2["muey"]     -= temp["muye"];
        Tau2["muzy"]     -= temp["muyz"];
        Tau2["zuey"]     -= temp["zuye"];
        Tau2["z,u,u1,y"] -= temp["z,u,y,u1"];
        Tau2["MUEY"]     -= temp["MUYE"]; 
        Tau2["MUZY"]     -= temp["MUYZ"]; 
        Tau2["ZUEY"]     -= temp["ZUYE"]; 
        Tau2["Z,U,U1,Y"] -= temp["Z,U,Y,U1"];
    }
    // <[V, T2]> C_6 C_2
    if (X4_TERM) {
        BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
        temp.block("caaa")("mwxy") += 0.125 * V_.block("aaca")("uvmz") * rdms_.L3aaa()("xyzuvw");
        temp.block("caaa")("mwxy") -= 0.250 * V_.block("aAcA")("uVmZ") * rdms_.L3aab()("xyZuwV");
        temp.block("CAAA")("MWXY") += 0.125 * V_.block("AACA")("UVMZ") * rdms_.L3bbb()("XYZUVW");
        temp.block("CAAA")("MWXY") -= 0.250 * V_.block("aAaC")("uVzM") * rdms_.L3abb()("zXYuVW");
        temp.block("aava")("xyew") -= 0.125 * V_.block("vaaa")("ezuv") * rdms_.L3aaa()("xyzuvw");
        temp.block("aava")("xyew") += 0.250 * V_.block("vAaA")("eZuV") * rdms_.L3aab()("xyZuwV");
        temp.block("AAVA")("XYEW") -= 0.125 * V_.block("VAAA")("EZUV") * rdms_.L3bbb()("XYZUVW");
        temp.block("AAVA")("XYEW") += 0.250 * V_.block("aVaA")("zEuV") * rdms_.L3abb()("zXYuVW");

        /************************************ Symmetrization ************************************/
        Tau2.block("cAaA")("mWxY") -= 0.125 * V_.block("aaca")("uvmz") * rdms_.L3aab()("xzYuvW");
        Tau2.block("cAaA")("mWxY") += 0.250 * V_.block("aAcA")("uVmZ") * rdms_.L3abb()("xYZuVW");
        Tau2["mwxy"] += temp["mwxy"];
        Tau2["MWXY"] += temp["MWXY"];

        Tau2.block("aCaA")("wMxY") += 0.125 * V_.block("AAAC")("UVZM") * rdms_.L3abb()("xYZwUV");
        Tau2.block("aCaA")("wMxY") += 0.250 * V_.block("aAaC")("uVzM") * rdms_.L3aab()("xzYuwV");
        Tau2["wmxy"] -= temp["mwxy"];
        Tau2["WMXY"] -= temp["MWXY"];

        /************************************ Symmetrization ************************************/
        Tau2.block("aAvA")("xYeW") += 0.125 * V_.block("vaaa")("ezuv") * rdms_.L3aab()("xzYuvW");
        Tau2.block("aAvA")("xYeW") -= 0.250 * V_.block("vAaA")("eZuV") * rdms_.L3abb()("xYZuVW");
        Tau2["xyew"] += temp["xyew"];
        Tau2["XYEW"] += temp["XYEW"];

        Tau2.block("aAaV")("xYwE") -= 0.125 * V_.block("AVAA")("ZEUV") * rdms_.L3abb()("xYZwUV");
        Tau2.block("aAaV")("xYwE") -= 0.250 * V_.block("aVaA")("zEuV") * rdms_.L3aab()("xzYuwV");
        Tau2["xywe"] -= temp["xyew"];
        Tau2["XYWE"] -= temp["XYEW"];
    }
    if (CORRELATION_TERM) {
        BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
        temp["iuax"] += 0.25 * DelGam1["xu"] * Sigma2["ia"];
        temp["IUAX"] += 0.25 * DelGam1["XU"] * Sigma2["IA"];

        /************************************ Symmetrization ************************************/
        Tau2["iUaX"] += 0.25 * DelGam1["XU"] * Sigma2["ia"];
        Tau2["iuax"] += temp["iuax"];
        Tau2["IUAX"] += temp["IUAX"];

        Tau2["iuxa"] -= temp["iuax"];
        Tau2["IUXA"] -= temp["IUAX"];

        Tau2["uIxA"] += 0.25 * DelGam1["xu"] * Sigma2["IA"];
        Tau2["uixa"] += temp["iuax"];
        Tau2["UIXA"] += temp["IUAX"];

        Tau2["uiax"] -= temp["iuax"];
        Tau2["UIAX"] -= temp["IUAX"];
    }
    // <[F, T2]>
    if (X5_TERM) {
        BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
        temp["uvey"] += 0.125 * F_["ex"] * Lambda2_["xyuv"];
        temp["UVEY"] += 0.125 * F_["EX"] * Lambda2_["XYUV"];
        temp["umxy"] -= 0.125 * F_["vm"] * Lambda2_["xyuv"];
        temp["UMXY"] -= 0.125 * F_["VM"] * Lambda2_["XYUV"];

        /************************************ Symmetrization ************************************/
        Tau2["uVeY"] += 0.125 * F_["ex"] * Lambda2_["xYuV"];
        Tau2["uvey"] += temp["uvey"];
        Tau2["UVEY"] += temp["UVEY"];

        Tau2["vUyE"] += 0.125 * F_["EX"] * Lambda2_["yXvU"];
        Tau2["vuye"] += temp["uvey"];
        Tau2["VUYE"] += temp["UVEY"];

        Tau2["uVyE"] += 0.125 * F_["EX"] * Lambda2_["yXuV"];
        Tau2["uvye"] -= temp["uvey"];
        Tau2["UVYE"] -= temp["UVEY"];

        Tau2["vUeY"] += 0.125 * F_["ex"] * Lambda2_["xYvU"];
        Tau2["vuey"] -= temp["uvey"];
        Tau2["VUEY"] -= temp["UVEY"];

        /************************************ Symmetrization ************************************/
        Tau2["uMxY"] -= 0.125 * F_["VM"] * Lambda2_["xYuV"];
        Tau2["umxy"] += temp["umxy"];
        Tau2["UMXY"] += temp["UMXY"];

        Tau2["mUyX"] -= 0.125 * F_["vm"] * Lambda2_["yXvU"];
        Tau2["muyx"] += temp["umxy"];
        Tau2["MUYX"] += temp["UMXY"];

        Tau2["uMyX"] -= 0.125 * F_["VM"] * Lambda2_["yXuV"];
        Tau2["umyx"] -= temp["umxy"];
        Tau2["UMYX"] -= temp["UMXY"];

        Tau2["mUxY"] -= 0.125 * F_["vm"] * Lambda2_["xYvU"];
        Tau2["muxy"] -= temp["umxy"];
        Tau2["MUXY"] -= temp["UMXY"];
    }

    if (CORRELATION_TERM) {
        BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
        temp["iuax"] += 0.25 * DelGam1["xu"] * Xi3["ia"];
        temp["IUAX"] += 0.25 * DelGam1["XU"] * Xi3["IA"];

        /************************************ Symmetrization ************************************/
        Tau2["iUaX"] += 0.25 * DelGam1["XU"] * Xi3["ia"];
        Tau2["iuax"] += temp["iuax"];
        Tau2["IUAX"] += temp["IUAX"];

        Tau2["iuxa"] -= temp["iuax"];
        Tau2["IUXA"] -= temp["IUAX"];

        Tau2["uIxA"] += 0.25 * DelGam1["xu"] * Xi3["IA"];
        Tau2["uixa"] += temp["iuax"];
        Tau2["UIXA"] += temp["IUAX"];

        Tau2["uiax"] -= temp["iuax"];
        Tau2["UIAX"] -= temp["IUAX"];
    }

    // NOTICE: remove the internal parts based on the DSRG theories
    Tau2.block("aaaa").zero();
    Tau2.block("aAaA").zero();
    Tau2.block("AAAA").zero();

    // Tau * [1 - e^(-s * Delta^2)]
    Tau1["ijab"] = Tau2["ijab"] * Eeps2_m1["ijab"];
    Tau1["IJAB"] = Tau2["IJAB"] * Eeps2_m1["IJAB"];
    Tau1["iJaB"] = Tau2["iJaB"] * Eeps2_m1["iJaB"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_sigma() {
    outfile->Printf("\n    Initializing multipliers for one-body amplitude.. ");
    Sigma = BTF_->build(CoreTensor, "Sigma", spin_cases({"hp"}));
    Sigma1 = BTF_->build(CoreTensor, "Sigma * DelEeps1", spin_cases({"hp"}));
    Sigma2 = BTF_->build(CoreTensor, "Sigma * Eeps1", spin_cases({"hp"}));
    Sigma3 = BTF_->build(CoreTensor, "Sigma * (1 + Eeps1)", spin_cases({"hp"}));

    if (X5_TERM) {
        Sigma["xe"] += 0.5 * T2_["uvey"] * Lambda2_["xyuv"];
        Sigma["xe"] += T2_["uVeY"] * Lambda2_["xYuV"];
        Sigma["XE"] += 0.5 * T2_["UVEY"] * Lambda2_["XYUV"];
        Sigma["XE"] += T2_["uVyE"] * Lambda2_["yXuV"];

        Sigma["mv"] -= 0.5 * T2_["umxy"] * Lambda2_["xyuv"];
        Sigma["mv"] -= T2_["mUxY"] * Lambda2_["xYvU"];
        Sigma["MV"] -= 0.5 * T2_["UMXY"] * Lambda2_["XYUV"];
        Sigma["MV"] -= T2_["uMxY"] * Lambda2_["xYuV"];
    }

    if (X7_TERM) {
        Sigma["me"] += T1_["me"];
        Sigma["mv"] += T1_["mu"] * Eta1_["uv"];
        Sigma["ve"] += T1_["ue"] * Gamma1_["uv"];

        Sigma["ME"] += T1_["ME"];
        Sigma["MV"] += T1_["MU"] * Eta1_["UV"];
        Sigma["VE"] += T1_["UE"] * Gamma1_["UV"];
    }

    Sigma1["ia"] = Sigma["ia"] * DelEeps1["ia"];
    Sigma1["IA"] = Sigma["IA"] * DelEeps1["IA"];

    Sigma2["ia"] = Sigma["ia"] * Eeps1["ia"];
    Sigma2["IA"] = Sigma["IA"] * Eeps1["IA"];

    Sigma3["ia"] = Sigma["ia"];
    Sigma3["ia"] += Sigma2["ia"];
    Sigma3["IA"] = Sigma["IA"];
    Sigma3["IA"] += Sigma2["IA"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_xi() {
    outfile->Printf("\n    Initializing multipliers for renormalize Fock ... ");
    Xi = BTF_->build(CoreTensor, "Xi", spin_cases({"hp"}));
    Xi1 = BTF_->build(CoreTensor, "Xi * Eeps1_m2", spin_cases({"hp"}));
    Xi2 = BTF_->build(CoreTensor, "Xi * Eeps1", spin_cases({"hp"}));
    Xi3 = BTF_->build(CoreTensor, "Xi * Eeps1_m1", spin_cases({"hp"}));

    if (X6_TERM) {
        Xi["ue"] += 0.5 * V_["evxy"] * Lambda2_["xyuv"];
        Xi["ue"] += V_["eVxY"] * Lambda2_["xYuV"];
        Xi["UE"] += 0.5 * V_["EVXY"] * Lambda2_["XYUV"];
        Xi["UE"] += V_["vExY"] * Lambda2_["xYvU"];

        Xi["mx"] -= 0.5 * V_["uvmy"] * Lambda2_["xyuv"];
        Xi["mx"] -= V_["uVmY"] * Lambda2_["xYuV"];
        Xi["MX"] -= 0.5 * V_["UVMY"] * Lambda2_["XYUV"];
        Xi["MX"] -= V_["uVyM"] * Lambda2_["yXuV"];
    }

    if (X7_TERM) {
        Xi["me"] += F_["em"];
        Xi["mu"] += F_["vm"] * Eta1_["uv"];
        Xi["ue"] += F_["ev"] * Gamma1_["uv"];

        Xi["ME"] += F_["EM"];
        Xi["MU"] += F_["VM"] * Eta1_["UV"];
        Xi["UE"] += F_["EV"] * Gamma1_["UV"];
    }

    Xi1["ia"] = Xi["ia"] * Eeps1_m2["ia"];
    Xi1["IA"] = Xi["IA"] * Eeps1_m2["IA"];

    Xi2["ia"] = Xi["ia"] * Eeps1["ia"];
    Xi2["IA"] = Xi["IA"] * Eeps1["IA"];

    Xi3["ia"] = Xi["ia"] * Eeps1_m1["ia"];
    Xi3["IA"] = Xi["IA"] * Eeps1_m1["IA"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_kappa() {
    outfile->Printf("\n    Initializing multipliers for renormalize ERIs ... ");
    Kappa = BTF_->build(CoreTensor, "Kappa", spin_cases({"hhpp"}));
    // <[V, T2]> (C_2)^4
    if (PT2_TERM) {
        Kappa["mnef"] += 0.25 * T2_["mnef"];
        Kappa["mnev"] += 0.25 * T2_["mneu"] * Eta1_["uv"];
        Kappa["mnve"] += 0.25 * T2_["mnue"] * Eta1_["uv"];
        Kappa["mnvy"] += 0.25 * T2_["mnux"] * Eta1_["uv"] * Eta1_["xy"];
        Kappa["muef"] += 0.25 * T2_["mvef"] * Gamma1_["uv"];
        Kappa["muew"] += 0.25 * T2_["mvez"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["muwe"] += 0.25 * T2_["mvze"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["muwy"] += 0.25 * T2_["mvzx"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        Kappa["umef"] += 0.25 * T2_["vmef"] * Gamma1_["uv"];
        Kappa["umew"] += 0.25 * T2_["vmez"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["umwe"] += 0.25 * T2_["vmze"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["umwy"] += 0.25 * T2_["vmzx"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        Kappa["uxef"] += 0.25 * T2_["vyef"] * Gamma1_["uv"] * Gamma1_["xy"];
        Kappa["uxew"] += 0.25 * T2_["vyez"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];
        Kappa["uxwe"] += 0.25 * T2_["vyze"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];

        Kappa["MNEF"] += 0.25 * T2_["MNEF"];
        Kappa["MNEV"] += 0.25 * T2_["MNEU"] * Eta1_["UV"];
        Kappa["MNVE"] += 0.25 * T2_["MNUE"] * Eta1_["UV"];
        Kappa["MNVY"] += 0.25 * T2_["MNUX"] * Eta1_["UV"] * Eta1_["XY"];
        Kappa["MUEF"] += 0.25 * T2_["MVEF"] * Gamma1_["UV"];
        Kappa["MUEW"] += 0.25 * T2_["MVEZ"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["MUWE"] += 0.25 * T2_["MVZE"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["MUWY"] += 0.25 * T2_["MVZX"] * Gamma1_["UV"] * Eta1_["ZW"] * Eta1_["XY"];
        Kappa["UMEF"] += 0.25 * T2_["VMEF"] * Gamma1_["UV"];
        Kappa["UMEW"] += 0.25 * T2_["VMEZ"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["UMWE"] += 0.25 * T2_["VMZE"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["UMWY"] += 0.25 * T2_["VMZX"] * Gamma1_["UV"] * Eta1_["ZW"] * Eta1_["XY"];
        Kappa["UXEF"] += 0.25 * T2_["VYEF"] * Gamma1_["UV"] * Gamma1_["XY"];
        Kappa["UXEW"] += 0.25 * T2_["VYEZ"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"];
        Kappa["UXWE"] += 0.25 * T2_["VYZE"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"];

        Kappa["mNeF"] += 0.25 * T2_["mNeF"];
        Kappa["mNeV"] += 0.25 * T2_["mNeU"] * Eta1_["UV"];
        Kappa["mNvE"] += 0.25 * T2_["mNuE"] * Eta1_["uv"];
        Kappa["mNvY"] += 0.25 * T2_["mNuX"] * Eta1_["uv"] * Eta1_["XY"];
        Kappa["mUeF"] += 0.25 * T2_["mVeF"] * Gamma1_["UV"];
        Kappa["mUeW"] += 0.25 * T2_["mVeZ"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["mUwE"] += 0.25 * T2_["mVzE"] * Gamma1_["UV"] * Eta1_["zw"];
        Kappa["mUwY"] += 0.25 * T2_["mVzX"] * Gamma1_["UV"] * Eta1_["zw"] * Eta1_["XY"];
        Kappa["uMeF"] += 0.25 * T2_["vMeF"] * Gamma1_["uv"];
        Kappa["uMeW"] += 0.25 * T2_["vMeZ"] * Gamma1_["uv"] * Eta1_["ZW"];
        Kappa["uMwE"] += 0.25 * T2_["vMzE"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["uMwY"] += 0.25 * T2_["vMzX"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["XY"];
        Kappa["uXeF"] += 0.25 * T2_["vYeF"] * Gamma1_["uv"] * Gamma1_["XY"];
        Kappa["uXeW"] += 0.25 * T2_["vYeZ"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["ZW"];
        Kappa["uXwE"] += 0.25 * T2_["vYzE"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["zw"];
    }
    // <[V, T2]> C_4 (C_2)^2 PP
    if (X1_TERM) {
        Kappa["xyef"] += 0.125 * T2_["uvef"] * Lambda2_["xyuv"];
        Kappa["xyew"] += 0.125 * T2_["uvez"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["xywe"] += 0.125 * T2_["uvze"] * Eta1_["zw"] * Lambda2_["xyuv"];

        Kappa["XYEF"] += 0.125 * T2_["UVEF"] * Lambda2_["XYUV"];
        Kappa["XYEW"] += 0.125 * T2_["UVEZ"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XYWE"] += 0.125 * T2_["UVZE"] * Eta1_["ZW"] * Lambda2_["XYUV"];

        Kappa["xYeF"] += 0.250 * T2_["uVeF"] * Lambda2_["xYuV"];
        Kappa["xYeW"] += 0.250 * T2_["uVeZ"] * Eta1_["ZW"] * Lambda2_["xYuV"];
        Kappa["xYwE"] += 0.250 * T2_["uVzE"] * Eta1_["zw"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        Kappa["mnuv"] += 0.125 * T2_["mnxy"] * Lambda2_["xyuv"];
        Kappa["mzuv"] += 0.125 * T2_["mwxy"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["zmuv"] += 0.125 * T2_["wmxy"] * Gamma1_["zw"] * Lambda2_["xyuv"];

        Kappa["MNUV"] += 0.125 * T2_["MNXY"] * Lambda2_["XYUV"];
        Kappa["MZUV"] += 0.125 * T2_["MWXY"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["ZMUV"] += 0.125 * T2_["WMXY"] * Gamma1_["ZW"] * Lambda2_["XYUV"];

        Kappa["mNuV"] += 0.250 * T2_["mNxY"] * Lambda2_["xYuV"];
        Kappa["mZuV"] += 0.250 * T2_["mWxY"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Kappa["zMuV"] += 0.250 * T2_["wMxY"] * Gamma1_["zw"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        Kappa["mxve"] -= 0.25 * T2_["muye"] * Lambda2_["xyuv"];
        Kappa["mxvw"] -= 0.25 * T2_["muyz"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["wxve"] -= 0.25 * T2_["zuye"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["mxve"] -= 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        Kappa["mxvw"] -= 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Kappa["wxve"] -= 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Kappa["MXVE"] -= 0.25 * T2_["MUYE"] * Lambda2_["XYUV"];
        Kappa["MXVW"] -= 0.25 * T2_["MUYZ"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["WXVE"] -= 0.25 * T2_["ZUYE"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["MXVE"] -= 0.25 * T2_["uMyE"] * Lambda2_["yXuV"];
        Kappa["MXVW"] -= 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Kappa["WXVE"] -= 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Kappa["mXvE"] -= 0.25 * T2_["mUyE"] * Lambda2_["yXvU"];
        Kappa["mXvW"] -= 0.25 * T2_["mUyZ"] * Eta1_["ZW"] * Lambda2_["yXvU"];
        Kappa["wXvE"] -= 0.25 * T2_["zUyE"] * Gamma1_["zw"] * Lambda2_["yXvU"];

        Kappa["xmve"] -= 0.25 * T2_["umye"] * Lambda2_["xyuv"];
        Kappa["xmvw"] -= 0.25 * T2_["umyz"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["xwve"] -= 0.25 * T2_["uzye"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["xmve"] += 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        Kappa["xmvw"] += 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Kappa["xwve"] += 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Kappa["XMVE"] -= 0.25 * T2_["UMYE"] * Lambda2_["XYUV"];
        Kappa["XMVW"] -= 0.25 * T2_["UMYZ"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XWVE"] -= 0.25 * T2_["UZYE"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XMVE"] += 0.25 * T2_["uMyE"] * Lambda2_["yXuV"];
        Kappa["XMVW"] += 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Kappa["XWVE"] += 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Kappa["xMvE"] -= 0.25 * T2_["uMyE"] * Lambda2_["xyuv"];
        Kappa["xMvW"] -= 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["xyuv"];
        Kappa["xWvE"] -= 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["xyuv"];
        Kappa["xMvE"] += 0.25 * T2_["UMYE"] * Lambda2_["xYvU"];
        Kappa["xMvW"] += 0.25 * T2_["UMYZ"] * Eta1_["ZW"] * Lambda2_["xYvU"];
        Kappa["xWvE"] += 0.25 * T2_["UZYE"] * Gamma1_["ZW"] * Lambda2_["xYvU"];

        Kappa["xmev"] -= 0.25 * T2_["umey"] * Lambda2_["xyuv"];
        Kappa["xmwv"] -= 0.25 * T2_["umzy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["xwev"] -= 0.25 * T2_["uzey"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["xmev"] -= 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        Kappa["xmwv"] -= 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Kappa["xwev"] -= 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Kappa["XMEV"] -= 0.25 * T2_["UMEY"] * Lambda2_["XYUV"];
        Kappa["XMWV"] -= 0.25 * T2_["UMZY"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XWEV"] -= 0.25 * T2_["UZEY"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XMEV"] -= 0.25 * T2_["uMyE"] * Lambda2_["yXuV"];
        Kappa["XMWV"] -= 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Kappa["XWEV"] -= 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Kappa["xMeV"] -= 0.25 * T2_["uMeY"] * Lambda2_["xYuV"];
        Kappa["xMwV"] -= 0.25 * T2_["uMzY"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Kappa["xWeV"] -= 0.25 * T2_["uZeY"] * Gamma1_["ZW"] * Lambda2_["xYuV"];

        Kappa["mxev"] -= 0.25 * T2_["muey"] * Lambda2_["xyuv"];
        Kappa["mxwv"] -= 0.25 * T2_["muzy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["wxev"] -= 0.25 * T2_["zuey"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["mxev"] += 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        Kappa["mxwv"] += 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Kappa["wxev"] += 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Kappa["MXEV"] -= 0.25 * T2_["MUEY"] * Lambda2_["XYUV"];
        Kappa["MXWV"] -= 0.25 * T2_["MUZY"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["WXEV"] -= 0.25 * T2_["ZUEY"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["MXEV"] += 0.25 * T2_["uMyE"] * Lambda2_["yXuV"];
        Kappa["MXWV"] += 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Kappa["WXEV"] += 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Kappa["mXeV"] -= 0.25 * T2_["mUeY"] * Lambda2_["XYUV"];
        Kappa["mXwV"] -= 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["XYUV"];
        Kappa["wXeV"] -= 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["XYUV"];
        Kappa["mXeV"] += 0.25 * T2_["muey"] * Lambda2_["yXuV"];
        Kappa["mXwV"] += 0.25 * T2_["muzy"] * Eta1_["zw"] * Lambda2_["yXuV"];
        Kappa["wXeV"] += 0.25 * T2_["zuey"] * Gamma1_["zw"] * Lambda2_["yXuV"];
    }
    // <[V, T2]> C_6 C_2
    if (X4_TERM) {
        Kappa.block("caaa")("mzuv") += 0.125 * T2_.block("caaa")("mwxy") * rdms_.L3aaa()("xyzuvw");
        Kappa.block("caaa")("mzuv") -= 0.250 * T2_.block("cAaA")("mWxY") * rdms_.L3aab()("xzYuvW");
        Kappa.block("CAAA")("MZUV") += 0.125 * T2_.block("CAAA")("MWXY") * rdms_.L3bbb()("XYZUVW");
        Kappa.block("CAAA")("MZUV") -= 0.250 * T2_.block("aCaA")("wMxY") * rdms_.L3abb()("xYZwUV");
        Kappa.block("cAaA")("mZuV") -= 0.125 * T2_.block("caaa")("mwxy") * rdms_.L3aab()("xyZuwV");
        Kappa.block("cAaA")("mZuV") += 0.250 * T2_.block("cAaA")("mWxY") * rdms_.L3abb()("xYZuVW");

        Kappa.block("acaa")("zmuv") += 0.125 * T2_.block("acaa")("wmxy") * rdms_.L3aaa()("xyzuvw");
        Kappa.block("acaa")("zmuv") += 0.250 * T2_.block("cAaA")("mWxY") * rdms_.L3aab()("xzYuvW");
        Kappa.block("ACAA")("ZMUV") += 0.125 * T2_.block("ACAA")("WMXY") * rdms_.L3bbb()("XYZUVW");
        Kappa.block("ACAA")("ZMUV") += 0.250 * T2_.block("aCaA")("wMxY") * rdms_.L3abb()("xYZwUV");
        Kappa.block("aCaA")("zMuV") += 0.125 * T2_.block("ACAA")("WMXY") * rdms_.L3abb()("zXYuVW");
        Kappa.block("aCaA")("zMuV") += 0.250 * T2_.block("aCaA")("wMxY") * rdms_.L3aab()("xzYuwV");

        Kappa.block("aava")("uvez") -= 0.125 * T2_.block("aava")("xyew") * rdms_.L3aaa()("xyzuvw");
        Kappa.block("aava")("uvez") += 0.250 * T2_.block("aAvA")("xYeW") * rdms_.L3aab()("xzYuvW");
        Kappa.block("AAVA")("UVEZ") -= 0.125 * T2_.block("AAVA")("XYEW") * rdms_.L3bbb()("XYZUVW");
        Kappa.block("AAVA")("UVEZ") += 0.250 * T2_.block("aAaV")("xYwE") * rdms_.L3abb()("xYZwUV");
        Kappa.block("aAvA")("uVeZ") += 0.125 * T2_.block("aava")("xyew") * rdms_.L3aab()("xyZuwV");
        Kappa.block("aAvA")("uVeZ") -= 0.250 * T2_.block("aAvA")("xYeW") * rdms_.L3abb()("xYZuVW");

        Kappa.block("aaav")("uvze") -= 0.125 * T2_.block("aaav")("xywe") * rdms_.L3aaa()("xyzuvw");
        Kappa.block("aaav")("uvze") -= 0.250 * T2_.block("aAvA")("xYeW") * rdms_.L3aab()("xzYuvW");
        Kappa.block("AAAV")("UVZE") -= 0.125 * T2_.block("AAAV")("XYWE") * rdms_.L3bbb()("XYZUVW");
        Kappa.block("AAAV")("UVZE") -= 0.250 * T2_.block("aAaV")("xYwE") * rdms_.L3abb()("xYZwUV");
        Kappa.block("aAaV")("uVzE") -= 0.125 * T2_.block("AAAV")("XYWE") * rdms_.L3abb()("zXYuVW");
        Kappa.block("aAaV")("uVzE") -= 0.250 * T2_.block("aAaV")("xYwE") * rdms_.L3aab()("xzYuwV");
    }
    // <[V, T1]>
    if (X6_TERM) {
        Kappa["xyev"] += 0.25 * T1_["ue"] * Lambda2_["xyuv"];
        Kappa["XYEV"] += 0.25 * T1_["UE"] * Lambda2_["XYUV"];
        Kappa["xYeV"] += 0.25 * T1_["ue"] * Lambda2_["xYuV"];

        Kappa["yxve"] += 0.25 * T1_["ue"] * Lambda2_["yxvu"];
        Kappa["YXVE"] += 0.25 * T1_["UE"] * Lambda2_["YXVU"];
        Kappa["yXvE"] += 0.25 * T1_["UE"] * Lambda2_["yXvU"];

        Kappa["myuv"] -= 0.25 * T1_["mx"] * Lambda2_["xyuv"];
        Kappa["MYUV"] -= 0.25 * T1_["MX"] * Lambda2_["XYUV"];
        Kappa["mYuV"] -= 0.25 * T1_["mx"] * Lambda2_["xYuV"];

        Kappa["ymvu"] -= 0.25 * T1_["mx"] * Lambda2_["yxvu"];
        Kappa["YMVU"] -= 0.25 * T1_["MX"] * Lambda2_["YXVU"];
        Kappa["yMvU"] -= 0.25 * T1_["MX"] * Lambda2_["yXvU"];
    }
    outfile->Printf("Done");
}

} // namespace forte