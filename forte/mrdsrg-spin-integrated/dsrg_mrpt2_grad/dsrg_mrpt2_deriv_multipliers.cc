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
    Tau2 = BTF_->build(CoreTensor, "Tau2", {"hhpp", "hHpP"});
    // Tau * Delta
    // <[V, T2]> (C_2)^4
    if (PT2_TERM) {
        /**************************************** α β α β ****************************************/
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
        /********************************** α β α β **********************************/
        Tau2["uVeF"] += 0.25 * V_["eFxY"] * Lambda2_["xYuV"];
        Tau2["uVeZ"] += 0.25 * V_["eWxY"] * Eta1_["ZW"] * Lambda2_["xYuV"];
        Tau2["uVzE"] += 0.25 * V_["wExY"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Tau2["u,V,z,U1"] += 0.25 * V_["w,A1,x,Y"] * Eta1_["zw"] * Eta1_["U1,A1"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        /************************************ α β α β ************************************/
        Tau2["mNxY"] += 0.25 * V_["uVmN"] * Lambda2_["xYuV"];
        Tau2["mWxY"] += 0.25 * V_["uVmZ"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Tau2["wMxY"] += 0.25 * V_["uVzM"] * Gamma1_["zw"] * Lambda2_["xYuV"];
        Tau2["w,A1,x,Y"] +=
            0.25 * V_["u,V,z,U1"] * Gamma1_["zw"] * Gamma1_["U1,A1"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        /********************************** α β α β **********************************/
        Tau2["mUyE"] -= 0.25 * V_["vEmX"] * Lambda2_["yXvU"];
        Tau2["mUyZ"] -= 0.25 * V_["vWmX"] * Eta1_["ZW"] * Lambda2_["yXvU"];
        Tau2["zUyE"] -= 0.25 * V_["vEwX"] * Gamma1_["zw"] * Lambda2_["yXvU"];
        Tau2["z,U,y,U1"] -=
            0.25 * V_["v,A1,w,X"] * Gamma1_["zw"] * Eta1_["U1,A1"] * Lambda2_["yXvU"];
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
        Tau2["uMeY"] -= 0.25 * V_["eVxM"] * Lambda2_["xYuV"];
        Tau2["uMzY"] -= 0.25 * V_["wVxM"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Tau2["uZeY"] -= 0.25 * V_["eVxW"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Tau2["u,Z,u1,Y"] -=
            0.25 * V_["a1,V,x,W"] * Gamma1_["ZW"] * Eta1_["u1,a1"] * Lambda2_["xYuV"];
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
    }
    // <[V, T2]> C_6 C_2
    if (X4_TERM) {
        /**************************************** α β α β ****************************************/
        Tau2.block("cAaA")("mWxY") -= 0.125 * V_.block("aaca")("uvmz") * rdms_->L3aab()("xzYuvW");
        Tau2.block("cAaA")("mWxY") += 0.250 * V_.block("aAcA")("uVmZ") * rdms_->L3abb()("xYZuVW");
        Tau2.block("aCaA")("wMxY") += 0.125 * V_.block("AAAC")("UVZM") * rdms_->L3abb()("xYZwUV");
        Tau2.block("aCaA")("wMxY") += 0.250 * V_.block("aAaC")("uVzM") * rdms_->L3aab()("xzYuwV");
        Tau2.block("aAvA")("xYeW") += 0.125 * V_.block("vaaa")("ezuv") * rdms_->L3aab()("xzYuvW");
        Tau2.block("aAvA")("xYeW") -= 0.250 * V_.block("vAaA")("eZuV") * rdms_->L3abb()("xYZuVW");
        Tau2.block("aAaV")("xYwE") -= 0.125 * V_.block("AVAA")("ZEUV") * rdms_->L3abb()("xYZwUV");
        Tau2.block("aAaV")("xYwE") -= 0.250 * V_.block("aVaA")("zEuV") * rdms_->L3aab()("xzYuwV");
    }
    if (CORRELATION_TERM) {
        /******************** α β α β ********************/
        Tau2["iUaX"] += 0.25 * DelGam1["XU"] * sigma2_xi3["ia"];
        Tau2["uIxA"] += 0.25 * DelGam1["xu"] * sigma2_xi3["IA"];
    }
    // <[F, T2]>
    if (X5_TERM) {
        /******************** α β α β ********************/
        Tau2["uVeY"] += 0.125 * F_["ex"] * Lambda2_["xYuV"];
        Tau2["vUyE"] += 0.125 * F_["EX"] * Lambda2_["yXvU"];
        Tau2["uVyE"] += 0.125 * F_["EX"] * Lambda2_["yXuV"];
        Tau2["vUeY"] += 0.125 * F_["ex"] * Lambda2_["xYvU"];
        Tau2["uMxY"] -= 0.125 * F_["VM"] * Lambda2_["xYuV"];
        Tau2["mUyX"] -= 0.125 * F_["vm"] * Lambda2_["yXvU"];
        Tau2["uMyX"] -= 0.125 * F_["VM"] * Lambda2_["yXuV"];
        Tau2["mUxY"] -= 0.125 * F_["vm"] * Lambda2_["xYvU"];
    }

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"hhpp"}, true);
    // Tau * Delta
    // <[V, T2]> (C_2)^4
    if (PT2_TERM) {
        /**************************************** α α α α ****************************************/
        temp["mnef"] += 0.0625 * V_["efmn"];
        temp["mnux"] += 0.0625 * V_["vymn"] * Eta1_["uv"] * Eta1_["xy"];
        temp["vyef"] += 0.0625 * V_["efux"] * Gamma1_["uv"] * Gamma1_["xy"];
        temp["v,y,z,u1"] +=
            0.0625 * V_["w,a1,u,x"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"] * Eta1_["u1,a1"];
        temp["mneu"] += 0.125 * V_["evmn"] * Eta1_["uv"];
        temp["mvez"] += 0.250 * V_["ewmu"] * Gamma1_["uv"] * Eta1_["zw"];
        temp["mvef"] += 0.125 * V_["efmu"] * Gamma1_["uv"];
        temp["mvzx"] += 0.125 * V_["wymu"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        temp["vyez"] += 0.125 * V_["ewux"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];
    }
    // <[V, T2]> C_4 (C_2)^2 PP
    if (X1_TERM) {
        /********************************** α α α α **********************************/
        temp["uvez"] += 0.0625 * V_["ewxy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        temp["uvef"] += 0.03125 * V_["efxy"] * Lambda2_["xyuv"];
        temp["u,v,z,u1"] +=
            0.03125 * V_["w,a1,x,y"] * Eta1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xyuv"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        /************************************ α α α α ************************************/
        temp["mwxy"] += 0.0625 * V_["uvmz"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        temp["mnxy"] += 0.03125 * V_["uvmn"] * Lambda2_["xyuv"];
        temp["a1,w,x,y"] +=
            0.03125 * V_["u,v,u1,z"] * Gamma1_["u1,a1"] * Gamma1_["zw"] * Lambda2_["xyuv"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        /********************************** α α α α **********************************/
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
    }
    // <[V, T2]> C_6 C_2
    if (X4_TERM) {
        /**************************************** α α α α ****************************************/
        temp.block("caaa")("mwxy") += 0.0625 * V_.block("aaca")("uvmz") * rdms_->L3aaa()("xyzuvw");
        temp.block("caaa")("mwxy") -= 0.125 * V_.block("aAcA")("uVmZ") * rdms_->L3aab()("xyZuwV");
        temp.block("aava")("xyew") -= 0.0625 * V_.block("vaaa")("ezuv") * rdms_->L3aaa()("xyzuvw");
        temp.block("aava")("xyew") += 0.125 * V_.block("vAaA")("eZuV") * rdms_->L3aab()("xyZuwV");
    }
    if (CORRELATION_TERM) {
        /******************** α α α α ********************/
        temp["iuax"] += 0.25 * DelGam1["xu"] * sigma2_xi3["ia"];
    }
    // <[F, T2]>
    if (X5_TERM) {
        /******************** α α α α ********************/
        temp["uvey"] += 0.125 * F_["ex"] * Lambda2_["xyuv"];
        temp["umxy"] -= 0.125 * F_["vm"] * Lambda2_["xyuv"];
    }
    /****** Symmetrization *****/
    Tau2["ijab"] += temp["ijab"];
    Tau2["ijba"] -= temp["ijab"];
    Tau2["jiab"] -= temp["ijab"];
    Tau2["jiba"] += temp["ijab"];
    // Remove the internal terms based on the DSRG formalism
    Tau2.block("aaaa").zero();
    Tau2.block("aAaA").zero();
    outfile->Printf("Done");
}

void DSRG_MRPT2::set_sigma_xi() {
    outfile->Printf("\n    Initializing multipliers for one-body amplitude.. ");
    sigma3_xi3 = BTF_->build(CoreTensor, "Sigma3 + Xi3", spin_cases({"hp"}));
    sigma2_xi3 = BTF_->build(CoreTensor, "Sigma2 + Xi3", spin_cases({"hp"}));
    sigma1_xi1_xi2 = BTF_->build(CoreTensor, "2s * Sigma1 + Xi1 - 2s * Xi2", spin_cases({"hp"}));
    {
        BlockedTensor Sigma = BTF_->build(CoreTensor, "Sigma", {"hp"}, true);
        BlockedTensor Sigma1 = BTF_->build(CoreTensor, "Sigma * DelEeps1", {"hp"}, true);
        BlockedTensor Sigma2 = BTF_->build(CoreTensor, "Sigma * Eeps1", {"hp"}, true);
        BlockedTensor Sigma3 = BTF_->build(CoreTensor, "Sigma * (1 + Eeps1)", {"hp"}, true);
        if (X5_TERM) {
            Sigma["xe"] += 0.5 * T2_["uvey"] * Lambda2_["xyuv"];
            Sigma["xe"] += T2_["uVeY"] * Lambda2_["xYuV"];
            Sigma["mv"] -= 0.5 * T2_["umxy"] * Lambda2_["xyuv"];
            Sigma["mv"] -= T2_["mUxY"] * Lambda2_["xYvU"];
        }
        if (X7_TERM) {
            Sigma["me"] += T1_["me"];
            Sigma["mv"] += T1_["mu"] * Eta1_["uv"];
            Sigma["ve"] += T1_["ue"] * Gamma1_["uv"];
        }
        Sigma1["ia"] = Sigma["ia"] * DelEeps1["ia"];
        Sigma2["ia"] = Sigma["ia"] * Eeps1["ia"];
        Sigma3["ia"] = Sigma["ia"];
        Sigma3["ia"] += Sigma2["ia"];

        outfile->Printf("Done");
        outfile->Printf("\n    Initializing multipliers for renormalize Fock ... ");
        BlockedTensor Xi = BTF_->build(CoreTensor, "Xi", {"hp"});
        BlockedTensor Xi1 = BTF_->build(CoreTensor, "Xi * Eeps1_m2", {"hp"});
        BlockedTensor Xi2 = BTF_->build(CoreTensor, "Xi * Eeps1", {"hp"});
        BlockedTensor Xi3 = BTF_->build(CoreTensor, "Xi * Eeps1_m1", {"hp"});
        if (X6_TERM) {
            Xi["ue"] += 0.5 * V_["evxy"] * Lambda2_["xyuv"];
            Xi["ue"] += V_["eVxY"] * Lambda2_["xYuV"];
            Xi["mx"] -= 0.5 * V_["uvmy"] * Lambda2_["xyuv"];
            Xi["mx"] -= V_["uVmY"] * Lambda2_["xYuV"];
        }
        if (X7_TERM) {
            Xi["me"] += F_["em"];
            Xi["mu"] += F_["vm"] * Eta1_["uv"];
            Xi["ue"] += F_["ev"] * Gamma1_["uv"];
        }
        Xi1["ia"] = Xi["ia"] * Eeps1_m2["ia"];
        Xi2["ia"] = Xi["ia"] * Eeps1["ia"];
        Xi3["ia"] = Xi["ia"] * Eeps1_m1["ia"];

        sigma3_xi3["ia"] = Sigma3["ia"];
        sigma3_xi3["ia"] += Xi3["ia"];
        sigma2_xi3["ia"] = Sigma2["ia"];
        sigma2_xi3["ia"] += Xi3["ia"];
        sigma1_xi1_xi2["ia"] = 2 * s_ * Sigma1["ia"];
        sigma1_xi1_xi2["ia"] += Xi1["ia"];
        sigma1_xi1_xi2["ia"] -= 2 * s_ * Xi2["ia"];
    }
    std::map<string, string> capital_blocks;
    capital_blocks = {{"ca", "CA"}, {"cv", "CV"}, {"aa", "AA"}, {"av", "AV"}};
    auto blocklabels = {"ca", "cv", "aa", "av"};
    for (const std::string& block : blocklabels) {
        sigma3_xi3.block(capital_blocks[block])("pq") = sigma3_xi3.block(block)("pq");
        sigma2_xi3.block(capital_blocks[block])("pq") = sigma2_xi3.block(block)("pq");
        sigma1_xi1_xi2.block(capital_blocks[block])("pq") = sigma1_xi1_xi2.block(block)("pq");
    }
    outfile->Printf("Done");
}

void DSRG_MRPT2::set_kappa() {
    outfile->Printf("\n    Initializing multipliers for renormalize ERIs ... ");
    Kappa = BTF_->build(CoreTensor, "Kappa", {"hhpp", "hHpP"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"hhpp"}, true);
    // <[V, T2]> (C_2)^4
    if (PT2_TERM) {
        /************************************ α α α α ************************************/
        temp["mnev"] += 0.125 * T2_["mneu"] * Eta1_["uv"];
        temp["muew"] += 0.250 * T2_["mvez"] * Gamma1_["uv"] * Eta1_["zw"];
        temp["uxew"] += 0.125 * T2_["vyez"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];
        temp["muef"] += 0.125 * T2_["mvef"] * Gamma1_["uv"];
        temp["muwy"] += 0.125 * T2_["mvzx"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        temp["mnef"] += 0.0625 * T2_["mnef"];
        temp["mnvy"] += 0.0625 * T2_["mnux"] * Eta1_["uv"] * Eta1_["xy"];
        temp["uxef"] += 0.0625 * T2_["vyef"] * Gamma1_["uv"] * Gamma1_["xy"];

        /************************************ α β α β ************************************/
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
        /****************************** α α α α ******************************/
        temp["xyew"] += 0.0625 * T2_["uvez"] * Eta1_["zw"] * Lambda2_["xyuv"];
        temp["xyef"] += 0.03125 * T2_["uvef"] * Lambda2_["xyuv"];

        /****************************** α β α β ******************************/
        Kappa["xYeF"] += 0.25 * T2_["uVeF"] * Lambda2_["xYuV"];
        Kappa["xYeW"] += 0.25 * T2_["uVeZ"] * Eta1_["ZW"] * Lambda2_["xYuV"];
        Kappa["xYwE"] += 0.25 * T2_["uVzE"] * Eta1_["zw"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        /******************************* α α α α *******************************/
        temp["mzuv"] += 0.0625 * T2_["mwxy"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        temp["mnuv"] += 0.03125 * T2_["mnxy"] * Lambda2_["xyuv"];

        /******************************* α β α β *******************************/
        Kappa["mNuV"] += 0.250 * T2_["mNxY"] * Lambda2_["xYuV"];
        Kappa["mZuV"] += 0.250 * T2_["mWxY"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Kappa["zMuV"] += 0.250 * T2_["wMxY"] * Gamma1_["zw"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        /******************************* α α α α *******************************/
        temp["mxve"] -= 0.25 * T2_["muye"] * Lambda2_["xyuv"];
        temp["mxvw"] -= 0.25 * T2_["muyz"] * Eta1_["zw"] * Lambda2_["xyuv"];
        temp["wxve"] -= 0.25 * T2_["zuye"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        temp["mxve"] -= 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        temp["mxvw"] -= 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        temp["wxve"] -= 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];

        /******************************* α β α β *******************************/
        Kappa["mXvE"] -= 0.25 * T2_["mUyE"] * Lambda2_["yXvU"];
        Kappa["mXvW"] -= 0.25 * T2_["mUyZ"] * Eta1_["ZW"] * Lambda2_["yXvU"];
        Kappa["wXvE"] -= 0.25 * T2_["zUyE"] * Gamma1_["zw"] * Lambda2_["yXvU"];
        Kappa["xMvE"] -= 0.25 * T2_["uMyE"] * Lambda2_["xyuv"];
        Kappa["xMvW"] -= 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["xyuv"];
        Kappa["xWvE"] -= 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["xyuv"];
        Kappa["xMvE"] += 0.25 * T2_["UMYE"] * Lambda2_["xYvU"];
        Kappa["xMvW"] += 0.25 * T2_["UMYZ"] * Eta1_["ZW"] * Lambda2_["xYvU"];
        Kappa["xWvE"] += 0.25 * T2_["UZYE"] * Gamma1_["ZW"] * Lambda2_["xYvU"];
        Kappa["xMeV"] -= 0.25 * T2_["uMeY"] * Lambda2_["xYuV"];
        Kappa["xMwV"] -= 0.25 * T2_["uMzY"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Kappa["xWeV"] -= 0.25 * T2_["uZeY"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Kappa["mXeV"] -= 0.25 * T2_["mUeY"] * Lambda2_["XYUV"];
        Kappa["mXwV"] -= 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["XYUV"];
        Kappa["wXeV"] -= 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["XYUV"];
        Kappa["mXeV"] += 0.25 * T2_["muey"] * Lambda2_["yXuV"];
        Kappa["mXwV"] += 0.25 * T2_["muzy"] * Eta1_["zw"] * Lambda2_["yXuV"];
        Kappa["wXeV"] += 0.25 * T2_["zuey"] * Gamma1_["zw"] * Lambda2_["yXuV"];
    }
    // <[V, T2]> C_6 C_2
    if (X4_TERM) {
        /**************************************** α α α α ****************************************/
        temp.block("caaa")("mzuv") += 0.0625 * T2_.block("caaa")("mwxy") * rdms_->L3aaa()("xyzuvw");
        temp.block("caaa")("mzuv") -= 0.125 * T2_.block("cAaA")("mWxY") * rdms_->L3aab()("xzYuvW");
        temp.block("aava")("uvez") -= 0.0625 * T2_.block("aava")("xyew") * rdms_->L3aaa()("xyzuvw");
        temp.block("aava")("uvez") += 0.125 * T2_.block("aAvA")("xYeW") * rdms_->L3aab()("xzYuvW");

        /**************************************** α β α β ****************************************/
        Kappa.block("cAaA")("mZuV") -= 0.125 * T2_.block("caaa")("mwxy") * rdms_->L3aab()("xyZuwV");
        Kappa.block("cAaA")("mZuV") += 0.250 * T2_.block("cAaA")("mWxY") * rdms_->L3abb()("xYZuVW");
        Kappa.block("aCaA")("zMuV") += 0.125 * T2_.block("ACAA")("WMXY") * rdms_->L3abb()("zXYuVW");
        Kappa.block("aCaA")("zMuV") += 0.250 * T2_.block("aCaA")("wMxY") * rdms_->L3aab()("xzYuwV");
        Kappa.block("aAvA")("uVeZ") += 0.125 * T2_.block("aava")("xyew") * rdms_->L3aab()("xyZuwV");
        Kappa.block("aAvA")("uVeZ") -= 0.250 * T2_.block("aAvA")("xYeW") * rdms_->L3abb()("xYZuVW");
        Kappa.block("aAaV")("uVzE") -= 0.125 * T2_.block("AAAV")("XYWE") * rdms_->L3abb()("zXYuVW");
        Kappa.block("aAaV")("uVzE") -= 0.250 * T2_.block("aAaV")("xYwE") * rdms_->L3aab()("xzYuwV");
    }
    // <[V, T1]>
    if (X6_TERM) {
        /********************* α α α α *********************/
        temp["xyev"] += 0.125 * T1_["ue"] * Lambda2_["xyuv"];
        temp["myuv"] -= 0.125 * T1_["mx"] * Lambda2_["xyuv"];

        /********************* α β α β *********************/
        Kappa["xYeV"] += 0.25 * T1_["ue"] * Lambda2_["xYuV"];
        Kappa["yXvE"] += 0.25 * T1_["UE"] * Lambda2_["yXvU"];
        Kappa["mYuV"] -= 0.25 * T1_["mx"] * Lambda2_["xYuV"];
        Kappa["yMvU"] -= 0.25 * T1_["MX"] * Lambda2_["yXvU"];
    }
    /******* Symmetrization ******/
    Kappa["ijab"] += temp["ijab"];
    Kappa["ijba"] -= temp["ijab"];
    Kappa["jiab"] -= temp["ijab"];
    Kappa["jiba"] += temp["ijab"];
    outfile->Printf("Done");
}

} // namespace forte