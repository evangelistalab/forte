/**
 * Set the constant vector b of the Linear System Ax=b.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "../dsrg_mrpt2.h"
#include "helpers/timer.h"

using namespace ambit;
using namespace psi;

namespace forte {

void DSRG_MRPT2::set_b(int dim, std::map<string, int> preidx, std::map<string, int> block_dim) {
    outfile->Printf("\n    Initializing b of the Linear System ............. ");
    b.resize(dim);

    /*-----------------------------------------------------------------------*
     |                                                                       |
     |  Adding the orbital contribution to the b of the Linear System Ax=b.  |
     |                                                                       |
     *-----------------------------------------------------------------------*/

    Z_b = BTF_->build(CoreTensor, "b(AX=b)", spin_cases({"gg"}));
    // NOTICE: constant b for z{core-virtual}
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor sigma_xi = BTF_->build(CoreTensor, "Sigma3 + Xi3", spin_cases({"hp"}));
    sigma_xi["ia"] =  Sigma3["ia"];
    sigma_xi["ia"] += Xi3["ia"];
    sigma_xi["IA"] =  Sigma3["IA"];
    sigma_xi["IA"] += Xi3["IA"];

    if (CORRELATION_TERM) {
        Z_b["em"] += 0.5 * sigma_xi["ma"] * F["ea"];
        Z_b["em"] += 0.5 * sigma_xi["ia"] * V["ieam"];
        Z_b["em"] += 0.5 * sigma_xi["IA"] * V["eImA"];
        Z_b["em"] += 0.5 * sigma_xi["ia"] * V["aeim"];
        Z_b["em"] += 0.5 * sigma_xi["IA"] * V["eAmI"];
        Z_b["em"] -= 0.5 * sigma_xi["ie"] * F["im"];
        Z_b["em"] +=       Tau1["mjab"] * V["abej"];
        Z_b["em"] += 2.0 * Tau1["mJaB"] * V["aBeJ"];
        Z_b["em"] -=       Tau1["ijeb"] * V["mbij"];
        Z_b["em"] -= 2.0 * Tau1["iJeB"] * V["mBiJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["em"] += temp["mlcd"] * V["cdel"];
        Z_b["em"] += 2.0 * temp["mLcD"] * V["cDeL"];
        temp.zero();

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        Z_b["em"] -= temp["kled"] * V["mdkl"];
        Z_b["em"] -= 2.0 * temp["kLeD"] * V["mDkL"];
        temp.zero();
    }
    Z_b["em"] += Z["m1,n1"] * V["n1,e,m1,m"];
    Z_b["em"] += Z["M1,N1"] * V["e,N1,m,M1"];

    Z_b["em"] += Z["e1,f"] * V["f,e,e1,m"];
    Z_b["em"] += Z["E1,F"] * V["e,F,m,E1"];

    // NOTICE: constant b for z{active-active}
    {
        BlockedTensor temp_z = BTF_->build(CoreTensor, "temporal matrix Z{aa} for symmetrization", spin_cases({"aa"}));
        if (CORRELATION_TERM) {
            temp_z["wz"] += 0.5 * sigma_xi["za"] * F["wa"];
            temp_z["wz"] += 0.5 * sigma_xi["iz"] * F["iw"];
            temp_z["wz"] += 0.5 * sigma_xi["ia"] * V["awiv"] * Gamma1_["zv"];
            temp_z["wz"] += 0.5 * sigma_xi["IA"] * V["wAvI"] * Gamma1_["zv"];
            temp_z["wz"] += 0.5 * sigma_xi["ia"] * V["auiw"] * Gamma1_["uz"];
            temp_z["wz"] += 0.5 * sigma_xi["IA"] * V["uAwI"] * Gamma1_["uz"];
            temp_z["wz"] +=       Tau1["ijzb"] * V["wbij"];
            temp_z["wz"] += 2.0 * Tau1["iJzB"] * V["wBiJ"];
            temp_z["wz"] +=       Tau1["zjab"] * V["abwj"];
            temp_z["wz"] += 2.0 * Tau1["zJaB"] * V["aBwJ"];

            temp["klzd"] += Kappa["klzd"] * Eeps2_p["klzd"];
            temp["kLzD"] += Kappa["kLzD"] * Eeps2_p["kLzD"];
            temp_z["wz"] += temp["klzd"] * V["wdkl"];
            temp_z["wz"] += 2.0 * temp["kLzD"] * V["wDkL"];
            temp.zero();

            temp["zlcd"] += Kappa["zlcd"] * Eeps2_p["zlcd"];
            temp["zLcD"] += Kappa["zLcD"] * Eeps2_p["zLcD"];
            temp_z["wz"] += temp["zlcd"] * V["cdwl"];
            temp_z["wz"] += 2.0 * temp["zLcD"] * V["cDwL"];
            temp.zero();
        }
        temp_z["wz"] += Z["m1,n1"] * V["n1,v,m1,w"] * Gamma1_["zv"];
        temp_z["wz"] += Z["M1,N1"] * V["v,N1,w,M1"] * Gamma1_["zv"];
        temp_z["wz"] += Z["e1,f1"] * V["f1,v,e1,w"] * Gamma1_["zv"];
        temp_z["wz"] += Z["E1,F1"] * V["v,F1,w,E1"] * Gamma1_["zv"];
        Z_b["wz"] += temp_z["wz"];
        Z_b["zw"] -= temp_z["wz"];
    }

    // NOTICE: constant b for z{virtual-active}
    if (CORRELATION_TERM) {
        Z_b["ew"] += 0.5 * sigma_xi["wa"] * F["ea"];
        Z_b["ew"] += 0.5 * sigma_xi["iw"] * F["ie"];
        Z_b["ew"] += 0.5 * sigma_xi["ia"] * V["aeiv"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * sigma_xi["IA"] * V["eAvI"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * sigma_xi["ia"] * V["auie"] * Gamma1_["uw"];
        Z_b["ew"] += 0.5 * sigma_xi["IA"] * V["uAeI"] * Gamma1_["uw"];
        Z_b["ew"] -= 0.5 * sigma_xi["ie"] * F["iw"];
        Z_b["ew"] +=       Tau1["ijwb"] * V["ebij"];
        Z_b["ew"] += 2.0 * Tau1["iJwB"] * V["eBiJ"];
        Z_b["ew"] +=       Tau1["wjab"] * V["abej"];
        Z_b["ew"] += 2.0 * Tau1["wJaB"] * V["aBeJ"];
        Z_b["ew"] -=       Tau1["ijeb"] * V["wbij"];
        Z_b["ew"] -= 2.0 * Tau1["iJeB"] * V["wBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["ew"] += temp["klwd"] * V["edkl"];
        Z_b["ew"] += 2.0 * temp["kLwD"] * V["eDkL"];
        temp.zero();

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["ew"] += temp["wlcd"] * V["cdel"];
        Z_b["ew"] += 2.0 * temp["wLcD"] * V["cDeL"];
        temp.zero();

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        Z_b["ew"] -= temp["kled"] * V["wdkl"];
        Z_b["ew"] -= 2.0 * temp["kLeD"] * V["wDkL"];
        temp.zero();
    }
    Z_b["ew"] -= Z["e,f1"] * F["f1,w"];
    Z_b["ew"] += Z["m1,n1"] * V["m1,e,n1,v"] * Gamma1_["wv"];
    Z_b["ew"] += Z["M1,N1"] * V["e,M1,v,N1"] * Gamma1_["wv"];
    Z_b["ew"] += Z["e1,f1"] * V["e1,e,f1,v"] * Gamma1_["wv"];
    Z_b["ew"] += Z["E1,F1"] * V["e,E1,v,F1"] * Gamma1_["wv"];

    // NOTICE: constant b for z{core-active}
    if (CORRELATION_TERM) {
        Z_b["mw"] += 0.5 * sigma_xi["wa"] * F["ma"];
        Z_b["mw"] += 0.5 * sigma_xi["iw"] * F["im"];
        Z_b["mw"] += 0.5 * sigma_xi["ia"] * V["amiv"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * sigma_xi["IA"] * V["mAvI"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * sigma_xi["ia"] * V["auim"] * Gamma1_["uw"];
        Z_b["mw"] += 0.5 * sigma_xi["IA"] * V["uAmI"] * Gamma1_["uw"];

        Z_b["mw"] -= 0.5 * sigma_xi["ma"] * F["wa"];
        Z_b["mw"] -= 0.5 * sigma_xi["ia"] * V["amiw"];
        Z_b["mw"] -= 0.5 * sigma_xi["IA"] * V["mAwI"];
        Z_b["mw"] -= 0.5 * sigma_xi["ia"] * V["awim"];
        Z_b["mw"] -= 0.5 * sigma_xi["IA"] * V["wAmI"];

        Z_b["mw"] += Tau1["ijwb"] * V["mbij"];
        Z_b["mw"] += 2.0 * Tau1["iJwB"] * V["mBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["mw"] += temp["klwd"] * V["mdkl"];
        Z_b["mw"] += 2.0 * temp["kLwD"] * V["mDkL"];
        temp.zero();

        Z_b["mw"] += Tau1["wjab"] * V["abmj"];
        Z_b["mw"] += 2.0 * Tau1["wJaB"] * V["aBmJ"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["mw"] += temp["wlcd"] * V["cdml"];
        Z_b["mw"] += 2.0 * temp["wLcD"] * V["cDmL"];
        temp.zero();

        Z_b["mw"] -= Tau1["mjab"] * V["abwj"];
        Z_b["mw"] -= 2.0 * Tau1["mJaB"] * V["aBwJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["mw"] -= temp["mlcd"] * V["cdwl"];
        Z_b["mw"] -= 2.0 * temp["mLcD"] * V["cDwL"];
        temp.zero();
    }
    Z_b["mw"] += Z["m1,n1"] * V["n1,v,m1,m"] * Gamma1_["wv"];
    Z_b["mw"] += Z["M1,N1"] * V["v,N1,m,M1"] * Gamma1_["wv"];
    Z_b["mw"] += Z["e1,f1"] * V["f1,v,e1,m"] * Gamma1_["wv"];
    Z_b["mw"] += Z["E1,F1"] * V["v,F1,m,E1"] * Gamma1_["wv"];
    Z_b["mw"] -= Z["m,n1"] * F["n1,w"];
    Z_b["mw"] -= Z["m1,n1"] * V["n1,w,m1,m"];
    Z_b["mw"] -= Z["M1,N1"] * V["w,N1,m,M1"];
    Z_b["mw"] -= Z["e1,f"] * V["f,w,e1,m"];
    Z_b["mw"] -= Z["E1,F"] * V["w,F,m,E1"];

    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"gg"});
    BlockedTensor temp3 = BTF_->build(CoreTensor, "Z{active-active} diagonal components", {"aa", "AA"});

    // ACTIVE-ACTIVE
    temp3["uv"] += Z["uv"] * I["uv"];
    temp3["UV"] += Z["UV"] * I["UV"];

    // VIRTUAL-CORE
    temp2["em"] += Z_b["em"];
    temp2["em"] += temp3["uv"] * V["veum"];
    temp2["em"] += temp3["UV"] * V["eVmU"];

    // CORE-ACTIVE
    temp2["mw"] += Z_b["mw"];
    temp2["mw"] += temp3["u1,a1"] * V["a1,v,u1,m"] * Gamma1_["wv"];
    temp2["mw"] += temp3["U1,A1"] * V["v,A1,m,U1"] * Gamma1_["wv"];
    temp2["mw"] += temp3["wv"] * F["vm"];
    temp2["mw"] -= temp3["uv"] * V["vwum"];
    temp2["mw"] -= temp3["UV"] * V["wVmU"];

    // VIRTUAL-ACTIVE
    temp2["ew"] += Z_b["ew"];
    temp2["ew"] += temp3["u1,a1"] * V["u1,e,a1,v"] * Gamma1_["wv"];
    temp2["ew"] += temp3["U1,A1"] * V["e,U1,v,A1"] * Gamma1_["wv"];
    temp2["ew"] += temp3["wv"] * F["ve"];

    // ACTIVE-ACTIVE
    temp2["wz"] += Z_b["wz"];
    temp2["wz"] += temp3["a1,u1"] * V["u1,v,a1,w"] * Gamma1_["zv"];
    temp2["wz"] += temp3["A1,U1"] * V["v,U1,w,A1"] * Gamma1_["zv"];
    temp2["wz"] -= temp3["a1,u1"] * V["u1,v,a1,z"] * Gamma1_["wv"];
    temp2["wz"] -= temp3["A1,U1"] * V["v,U1,z,A1"] * Gamma1_["wv"];

    for (const std::string& block : {"vc", "ca", "va", "aa"}) {
        (temp2.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block != "aa") {
                int index = preidx[block] + i[0] * block_dim[block] + i[1];
                b.at(index) = value;
            } else if (block == "aa" && i[0] > i[1]) {
                int index = preidx[block] + i[0] * (i[0] - 1) / 2 + i[1];
                b.at(index) = value;
            }
        });
    }

    /*------------------------------------------------------------------*
     |                                                                  |
     |  Adding the CI contribution to the b of the Linear System Ax=b.  |
     |                                                                  |
     *------------------------------------------------------------------*/

    auto b_ck = ambit::Tensor::build(ambit::CoreTensor, "ci equations b part", {ndets});

    // Solving the multiplier Alpha (the CI normalization condition)
    Alpha = 0.0;
    Alpha += H["vu"] * Gamma1_["uv"];
    Alpha += H["VU"] * Gamma1_["UV"];
    Alpha += V_sumA_Alpha["v,u"] * Gamma1_["uv"];
    Alpha += V_sumB_Alpha["v,u"] * Gamma1_["uv"];
    Alpha += V_sumB_Beta["V,U"] * Gamma1_["UV"];
    Alpha += V_sumA_Beta["V,U"] * Gamma1_["UV"];
    Alpha += 0.25 * V["xyuv"] * Gamma2_["uvxy"];
    Alpha += 0.25 * V["XYUV"] * Gamma2_["UVXY"];
    Alpha += V["xYuV"] * Gamma2_["uVxY"];

    temp = BTF_->build(CoreTensor, "temporal tensor", {"aa", "AA"});
    if (PT2_TERM) {
        temp["uv"] += 0.25 * T2_["vmef"] * V_["efum"];
        temp["uv"] += 0.25 * T2_["vmez"] * V_["ewum"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["vmze"] * V_["weum"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["vmzx"] * V_["wyum"] * Eta1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["vwef"] * V_["efuz"] * Gamma1_["zw"];
        temp["uv"] += 0.25 * T2_["vwex"] * V_["eyuz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["vwxe"] * V_["yeuz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.50 * T2_["vMeF"] * V_["eFuM"];
        temp["uv"] += 0.50 * T2_["vMeZ"] * V_["eWuM"] * Eta1_["ZW"];
        temp["uv"] += 0.50 * T2_["vMzE"] * V_["wEuM"] * Eta1_["zw"];
        temp["uv"] += 0.50 * T2_["vMzX"] * V_["wYuM"] * Eta1_["zw"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["vWeF"] * V_["eFuZ"] * Gamma1_["ZW"];
        temp["uv"] += 0.50 * T2_["vWeX"] * V_["eYuZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["vWxE"] * V_["yEuZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] += 0.25 * T2_["VMEF"] * V_["EFUM"];
        temp["UV"] += 0.25 * T2_["VMEZ"] * V_["EWUM"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["VMZE"] * V_["WEUM"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["VMZX"] * V_["WYUM"] * Eta1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["VWEF"] * V_["EFUZ"] * Gamma1_["ZW"];
        temp["UV"] += 0.25 * T2_["VWEX"] * V_["EYUZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["VWXE"] * V_["YEUZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["mVeF"] * V_["eFmU"];
        temp["UV"] += 0.50 * T2_["mVeZ"] * V_["eWmU"] * Eta1_["ZW"];
        temp["UV"] += 0.50 * T2_["mVzE"] * V_["wEmU"] * Eta1_["zw"];
        temp["UV"] += 0.50 * T2_["mVzX"] * V_["wYmU"] * Eta1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wVeF"] * V_["eFzU"] * Gamma1_["zw"];
        temp["UV"] += 0.50 * T2_["wVeX"] * V_["eYzU"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wVxE"] * V_["yEzU"] * Gamma1_["zw"] * Eta1_["xy"];

        temp["uv"] += 0.25 * T2_["umef"] * V_["efvm"];
        temp["uv"] += 0.25 * T2_["umez"] * V_["ewvm"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["umze"] * V_["wevm"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["umzx"] * V_["wyvm"] * Eta1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["uwef"] * V_["efvz"] * Gamma1_["zw"];
        temp["uv"] += 0.25 * T2_["uwex"] * V_["eyvz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["uwxe"] * V_["yevz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.50 * T2_["uMeF"] * V_["eFvM"];
        temp["uv"] += 0.50 * T2_["uMeZ"] * V_["eWvM"] * Eta1_["ZW"];
        temp["uv"] += 0.50 * T2_["uMzE"] * V_["wEvM"] * Eta1_["zw"];
        temp["uv"] += 0.50 * T2_["uMzX"] * V_["wYvM"] * Eta1_["zw"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["uWeF"] * V_["eFvZ"] * Gamma1_["ZW"];
        temp["uv"] += 0.50 * T2_["uWeX"] * V_["eYvZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["uWxE"] * V_["yEvZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] += 0.25 * T2_["UMEF"] * V_["EFVM"];
        temp["UV"] += 0.25 * T2_["UMEZ"] * V_["EWVM"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["UMZE"] * V_["WEVM"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["UMZX"] * V_["WYVM"] * Eta1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["UWEF"] * V_["EFVZ"] * Gamma1_["ZW"];
        temp["UV"] += 0.25 * T2_["UWEX"] * V_["EYVZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["UWXE"] * V_["YEVZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["mUeF"] * V_["eFmV"];
        temp["UV"] += 0.50 * T2_["mUeZ"] * V_["eWmV"] * Eta1_["ZW"];
        temp["UV"] += 0.50 * T2_["mUzE"] * V_["wEmV"] * Eta1_["zw"];
        temp["UV"] += 0.50 * T2_["mUzX"] * V_["wYmV"] * Eta1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wUeF"] * V_["eFzV"] * Gamma1_["zw"];
        temp["UV"] += 0.50 * T2_["wUeX"] * V_["eYzV"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wUxE"] * V_["yEzV"] * Gamma1_["zw"] * Eta1_["xy"];

        temp["uv"] += 0.25 * T2_["mvef"] * V_["efmu"];
        temp["uv"] += 0.25 * T2_["mvez"] * V_["ewmu"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["mvze"] * V_["wemu"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["mvzx"] * V_["wymu"] * Eta1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["wvef"] * V_["efzu"] * Gamma1_["zw"];
        temp["uv"] += 0.25 * T2_["wvex"] * V_["eyzu"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["wvxe"] * V_["yezu"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.50 * T2_["vMeF"] * V_["eFuM"];
        temp["uv"] += 0.50 * T2_["vMeZ"] * V_["eWuM"] * Eta1_["ZW"];
        temp["uv"] += 0.50 * T2_["vMzE"] * V_["wEuM"] * Eta1_["zw"];
        temp["uv"] += 0.50 * T2_["vMzX"] * V_["wYuM"] * Eta1_["zw"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["vWeF"] * V_["eFuZ"] * Gamma1_["ZW"];
        temp["uv"] += 0.50 * T2_["vWeX"] * V_["eYuZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["vWxE"] * V_["yEuZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] += 0.25 * T2_["MVEF"] * V_["EFMU"];
        temp["UV"] += 0.25 * T2_["MVEZ"] * V_["EWMU"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["MVZE"] * V_["WEMU"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["MVZX"] * V_["WYMU"] * Eta1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["WVEF"] * V_["EFZU"] * Gamma1_["ZW"];
        temp["UV"] += 0.25 * T2_["WVEX"] * V_["EYZU"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["WVXE"] * V_["YEZU"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["mVeF"] * V_["eFmU"];
        temp["UV"] += 0.50 * T2_["mVeZ"] * V_["eWmU"] * Eta1_["ZW"];
        temp["UV"] += 0.50 * T2_["mVzE"] * V_["wEmU"] * Eta1_["zw"];
        temp["UV"] += 0.50 * T2_["mVzX"] * V_["wYmU"] * Eta1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wVeF"] * V_["eFzU"] * Gamma1_["zw"];
        temp["UV"] += 0.50 * T2_["wVeX"] * V_["eYzU"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wVxE"] * V_["yEzU"] * Gamma1_["zw"] * Eta1_["xy"];

        temp["uv"] += 0.25 * T2_["muef"] * V_["efmv"];
        temp["uv"] += 0.25 * T2_["muez"] * V_["ewmv"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["muze"] * V_["wemv"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["muzx"] * V_["wymv"] * Eta1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["wuef"] * V_["efzv"] * Gamma1_["zw"];
        temp["uv"] += 0.25 * T2_["wuex"] * V_["eyzv"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["wuxe"] * V_["yezv"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.50 * T2_["uMeF"] * V_["eFvM"];
        temp["uv"] += 0.50 * T2_["uMeZ"] * V_["eWvM"] * Eta1_["ZW"];
        temp["uv"] += 0.50 * T2_["uMzE"] * V_["wEvM"] * Eta1_["zw"];
        temp["uv"] += 0.50 * T2_["uMzX"] * V_["wYvM"] * Eta1_["zw"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["uWeF"] * V_["eFvZ"] * Gamma1_["ZW"];
        temp["uv"] += 0.50 * T2_["uWeX"] * V_["eYvZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["uWxE"] * V_["yEvZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] += 0.25 * T2_["MUEF"] * V_["EFMV"];
        temp["UV"] += 0.25 * T2_["MUEZ"] * V_["EWMV"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["MUZE"] * V_["WEMV"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["MUZX"] * V_["WYMV"] * Eta1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["WUEF"] * V_["EFZV"] * Gamma1_["ZW"];
        temp["UV"] += 0.25 * T2_["WUEX"] * V_["EYZV"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["WUXE"] * V_["YEZV"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["mUeF"] * V_["eFmV"];
        temp["UV"] += 0.50 * T2_["mUeZ"] * V_["eWmV"] * Eta1_["ZW"];
        temp["UV"] += 0.50 * T2_["mUzE"] * V_["wEmV"] * Eta1_["zw"];
        temp["UV"] += 0.50 * T2_["mUzX"] * V_["wYmV"] * Eta1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wUeF"] * V_["eFzV"] * Gamma1_["zw"];
        temp["UV"] += 0.50 * T2_["wUeX"] * V_["eYzV"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wUxE"] * V_["yEzV"] * Gamma1_["zw"] * Eta1_["xy"];

        temp["uv"] -= 0.25 * T2_["mnue"] * V_["vemn"];
        temp["uv"] -= 0.25 * T2_["mnuz"] * V_["vwmn"] * Eta1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwue"] * V_["vemz"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwux"] * V_["vymz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wmue"] * V_["vezm"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["wmux"] * V_["vyzm"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wyue"] * V_["vezx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp["uv"] -= 0.50 * T2_["mNuE"] * V_["vEmN"];
        temp["uv"] -= 0.50 * T2_["mNuZ"] * V_["vWmN"] * Eta1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWuE"] * V_["vEmZ"] * Gamma1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWuX"] * V_["vYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wMuE"] * V_["vEzM"] * Gamma1_["zw"];
        temp["uv"] -= 0.50 * T2_["wMuX"] * V_["vYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wYuE"] * V_["vEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
        temp["UV"] -= 0.25 * T2_["MNUE"] * V_["VEMN"];
        temp["UV"] -= 0.25 * T2_["MNUZ"] * V_["VWMN"] * Eta1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWUE"] * V_["VEMZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWUX"] * V_["VYMZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WMUE"] * V_["VEZM"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["WMUX"] * V_["VYZM"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WYUE"] * V_["VEZX"] * Gamma1_["ZW"] * Gamma1_["XY"];
        temp["UV"] -= 0.50 * T2_["mNeU"] * V_["eVmN"];
        temp["UV"] -= 0.50 * T2_["mNzU"] * V_["wVmN"] * Eta1_["zw"];
        temp["UV"] -= 0.50 * T2_["mWeU"] * V_["eVmZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.50 * T2_["mWxU"] * V_["yVmZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wMeU"] * V_["eVzM"] * Gamma1_["zw"];
        temp["UV"] -= 0.50 * T2_["wMxU"] * V_["yVzM"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wYeU"] * V_["eVzX"] * Gamma1_["zw"] * Gamma1_["XY"];

        temp["uv"] -= 0.25 * T2_["mnve"] * V_["uemn"];
        temp["uv"] -= 0.25 * T2_["mnvz"] * V_["uwmn"] * Eta1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwve"] * V_["uemz"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwvx"] * V_["uymz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wmve"] * V_["uezm"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["wmvx"] * V_["uyzm"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wyve"] * V_["uezx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp["uv"] -= 0.50 * T2_["mNvE"] * V_["uEmN"];
        temp["uv"] -= 0.50 * T2_["mNvZ"] * V_["uWmN"] * Eta1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWvE"] * V_["uEmZ"] * Gamma1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWvX"] * V_["uYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wMvE"] * V_["uEzM"] * Gamma1_["zw"];
        temp["uv"] -= 0.50 * T2_["wMvX"] * V_["uYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wYvE"] * V_["uEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
        temp["UV"] -= 0.25 * T2_["MNVE"] * V_["UEMN"];
        temp["UV"] -= 0.25 * T2_["MNVZ"] * V_["UWMN"] * Eta1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWVE"] * V_["UEMZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWVX"] * V_["UYMZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WMVE"] * V_["UEZM"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["WMVX"] * V_["UYZM"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WYVE"] * V_["UEZX"] * Gamma1_["ZW"] * Gamma1_["XY"];
        temp["UV"] -= 0.50 * T2_["mNeV"] * V_["eUmN"];
        temp["UV"] -= 0.50 * T2_["mNzV"] * V_["wUmN"] * Eta1_["zw"];
        temp["UV"] -= 0.50 * T2_["mWeV"] * V_["eUmZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.50 * T2_["mWxV"] * V_["yUmZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wMeV"] * V_["eUzM"] * Gamma1_["zw"];
        temp["UV"] -= 0.50 * T2_["wMxV"] * V_["yUzM"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wYeV"] * V_["eUzX"] * Gamma1_["zw"] * Gamma1_["XY"];

        temp["uv"] -= 0.25 * T2_["mneu"] * V_["evmn"];
        temp["uv"] -= 0.25 * T2_["mnzu"] * V_["wvmn"] * Eta1_["zw"];
        temp["uv"] -= 0.25 * T2_["mweu"] * V_["evmz"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwxu"] * V_["yvmz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wmeu"] * V_["evzm"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["wmxu"] * V_["yvzm"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wyeu"] * V_["evzx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp["uv"] -= 0.50 * T2_["mNuE"] * V_["vEmN"];
        temp["uv"] -= 0.50 * T2_["mNuZ"] * V_["vWmN"] * Eta1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWuE"] * V_["vEmZ"] * Gamma1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWuX"] * V_["vYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wMuE"] * V_["vEzM"] * Gamma1_["zw"];
        temp["uv"] -= 0.50 * T2_["wMuX"] * V_["vYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wYuE"] * V_["vEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
        temp["UV"] -= 0.25 * T2_["MNEU"] * V_["EVMN"];
        temp["UV"] -= 0.25 * T2_["MNZU"] * V_["WVMN"] * Eta1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWEU"] * V_["EVMZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWXU"] * V_["YVMZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WMEU"] * V_["EVZM"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["WMXU"] * V_["YVZM"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WYEU"] * V_["EVZX"] * Gamma1_["ZW"] * Gamma1_["XY"];
        temp["UV"] -= 0.50 * T2_["mNeU"] * V_["eVmN"];
        temp["UV"] -= 0.50 * T2_["mNzU"] * V_["wVmN"] * Eta1_["zw"];
        temp["UV"] -= 0.50 * T2_["mWeU"] * V_["eVmZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.50 * T2_["mWxU"] * V_["yVmZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wMeU"] * V_["eVzM"] * Gamma1_["zw"];
        temp["UV"] -= 0.50 * T2_["wMxU"] * V_["yVzM"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wYeU"] * V_["eVzX"] * Gamma1_["zw"] * Gamma1_["XY"];

        temp["uv"] -= 0.25 * T2_["mnev"] * V_["eumn"];
        temp["uv"] -= 0.25 * T2_["mnzv"] * V_["wumn"] * Eta1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwev"] * V_["eumz"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwxv"] * V_["yumz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wmev"] * V_["euzm"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["wmxv"] * V_["yuzm"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wyev"] * V_["euzx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp["uv"] -= 0.50 * T2_["mNvE"] * V_["uEmN"];
        temp["uv"] -= 0.50 * T2_["mNvZ"] * V_["uWmN"] * Eta1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWvE"] * V_["uEmZ"] * Gamma1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWvX"] * V_["uYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wMvE"] * V_["uEzM"] * Gamma1_["zw"];
        temp["uv"] -= 0.50 * T2_["wMvX"] * V_["uYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wYvE"] * V_["uEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
        temp["UV"] -= 0.25 * T2_["MNEV"] * V_["EUMN"];
        temp["UV"] -= 0.25 * T2_["MNZV"] * V_["WUMN"] * Eta1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWEV"] * V_["EUMZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWXV"] * V_["YUMZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WMEV"] * V_["EUZM"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["WMXV"] * V_["YUZM"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WYEV"] * V_["EUZX"] * Gamma1_["ZW"] * Gamma1_["XY"];
        temp["UV"] -= 0.50 * T2_["mNeV"] * V_["eUmN"];
        temp["UV"] -= 0.50 * T2_["mNzV"] * V_["wUmN"] * Eta1_["zw"];
        temp["UV"] -= 0.50 * T2_["mWeV"] * V_["eUmZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.50 * T2_["mWxV"] * V_["yUmZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wMeV"] * V_["eUzM"] * Gamma1_["zw"];
        temp["UV"] -= 0.50 * T2_["wMxV"] * V_["yUzM"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wYeV"] * V_["eUzX"] * Gamma1_["zw"] * Gamma1_["XY"];
    }

    if (X1_TERM) {
        temp["zw"] -= 0.125 * T2_["uvwe"] * V_["zexy"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVwE"] * V_["zExY"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVWE"] * V_["ZEXY"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVeW"] * V_["eZxY"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvze"] * V_["wexy"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVzE"] * V_["wExY"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVZE"] * V_["WEXY"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVeZ"] * V_["eWxY"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvew"] * V_["ezxy"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVwE"] * V_["zExY"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVEW"] * V_["EZXY"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVeW"] * V_["eZxY"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvez"] * V_["ewxy"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVzE"] * V_["wExY"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVEZ"] * V_["EWXY"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVeZ"] * V_["eWxY"] * Lambda2_["xYuV"];
    }

    if (X2_TERM) {
        temp["zw"] += 0.125 * T2_["wmxy"] * V_["uvzm"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["wMxY"] * V_["uVzM"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["WMXY"] * V_["UVZM"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["mWxY"] * V_["uVmZ"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["zmxy"] * V_["uvwm"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["zMxY"] * V_["uVwM"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["ZMXY"] * V_["UVWM"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["mZxY"] * V_["uVmW"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["mwxy"] * V_["uvmz"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["wMxY"] * V_["uVzM"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["MWXY"] * V_["UVMZ"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["mWxY"] * V_["uVmZ"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["mzxy"] * V_["uvmw"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["zMxY"] * V_["uVwM"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["MZXY"] * V_["UVMW"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["mZxY"] * V_["uVmW"] * Lambda2_["xYuV"];
    }

    if (X3_TERM) {
        temp["zw"] -= V_["vezx"] * T2_["wuye"] * Lambda2_["xyuv"];
        temp["zw"] -= V_["vezx"] * T2_["wUeY"] * Lambda2_["xYvU"];
        temp["zw"] -= V_["vEzX"] * T2_["wUyE"] * Lambda2_["yXvU"];
        temp["zw"] -= V_["eVzX"] * T2_["wUeY"] * Lambda2_["XYUV"];
        temp["zw"] -= V_["eVzX"] * T2_["wuye"] * Lambda2_["yXuV"];
        temp["ZW"] -= V_["eVxZ"] * T2_["uWeY"] * Lambda2_["xYuV"];
        temp["ZW"] -= V_["vExZ"] * T2_["uWyE"] * Lambda2_["xyuv"];
        temp["ZW"] -= V_["vExZ"] * T2_["WUYE"] * Lambda2_["xYvU"];
        temp["ZW"] -= V_["VEZX"] * T2_["WUYE"] * Lambda2_["XYUV"];
        temp["ZW"] -= V_["VEZX"] * T2_["uWyE"] * Lambda2_["yXuV"];

        temp["wz"] -= V_["vewx"] * T2_["zuye"] * Lambda2_["xyuv"];
        temp["wz"] -= V_["vewx"] * T2_["zUeY"] * Lambda2_["xYvU"];
        temp["wz"] -= V_["vEwX"] * T2_["zUyE"] * Lambda2_["yXvU"];
        temp["wz"] -= V_["eVwX"] * T2_["zUeY"] * Lambda2_["XYUV"];
        temp["wz"] -= V_["eVwX"] * T2_["zuye"] * Lambda2_["yXuV"];
        temp["WZ"] -= V_["eVxW"] * T2_["uZeY"] * Lambda2_["xYuV"];
        temp["WZ"] -= V_["vExW"] * T2_["uZyE"] * Lambda2_["xyuv"];
        temp["WZ"] -= V_["vExW"] * T2_["ZUYE"] * Lambda2_["xYvU"];
        temp["WZ"] -= V_["VEWX"] * T2_["ZUYE"] * Lambda2_["XYUV"];
        temp["WZ"] -= V_["VEWX"] * T2_["uZyE"] * Lambda2_["yXuV"];

        temp["zw"] += V_["vwmx"] * T2_["muyz"] * Lambda2_["xyuv"];
        temp["zw"] += V_["vwmx"] * T2_["mUzY"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["vWmX"] * T2_["mUyZ"] * Lambda2_["yXvU"];
        temp["zw"] += V_["wVmX"] * T2_["mUzY"] * Lambda2_["XYUV"];
        temp["zw"] += V_["wVmX"] * T2_["muyz"] * Lambda2_["yXuV"];
        temp["zw"] += V_["wVxM"] * T2_["uMzY"] * Lambda2_["xYuV"];
        temp["ZW"] += V_["vWxM"] * T2_["uMyZ"] * Lambda2_["xyuv"];
        temp["ZW"] += V_["vWxM"] * T2_["MUYZ"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["VWMX"] * T2_["MUYZ"] * Lambda2_["XYUV"];
        temp["ZW"] += V_["VWMX"] * T2_["uMyZ"] * Lambda2_["yXuV"];

        temp["zw"] += V_["vzmx"] * T2_["muyw"] * Lambda2_["xyuv"];
        temp["zw"] += V_["vzmx"] * T2_["mUwY"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["vZmX"] * T2_["mUyW"] * Lambda2_["yXvU"];
        temp["zw"] += V_["zVmX"] * T2_["mUwY"] * Lambda2_["XYUV"];
        temp["zw"] += V_["zVmX"] * T2_["muyw"] * Lambda2_["yXuV"];
        temp["zw"] += V_["zVxM"] * T2_["uMwY"] * Lambda2_["xYuV"];
        temp["ZW"] += V_["vZxM"] * T2_["uMyW"] * Lambda2_["xyuv"];
        temp["ZW"] += V_["vZxM"] * T2_["MUYW"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["VZMX"] * T2_["MUYW"] * Lambda2_["XYUV"];
        temp["ZW"] += V_["VZMX"] * T2_["uMyW"] * Lambda2_["yXuV"];
    }

    if (CORRELATION_TERM) {
        temp["vu"] += Sigma3["ia"] * V["auiv"];
        temp["vu"] += Sigma3["IA"] * V["uAvI"];
        temp["VU"] += Sigma3["IA"] * V["AUIV"];
        temp["VU"] += Sigma3["ia"] * V["aUiV"];

        temp["vu"] += Sigma3["ia"] * V["aviu"];
        temp["vu"] += Sigma3["IA"] * V["vAuI"];
        temp["VU"] += Sigma3["IA"] * V["AVIU"];
        temp["VU"] += Sigma3["ia"] * V["aViU"];

        temp["xu"] += Sigma2["ia"] * Delta1["xu"] * T2_["iuax"];
        temp["xu"] += Sigma2["IA"] * Delta1["xu"] * T2_["uIxA"];
        temp["XU"] += Sigma2["IA"] * Delta1["XU"] * T2_["IUAX"];
        temp["XU"] += Sigma2["ia"] * Delta1["XU"] * T2_["iUaX"];

        temp["xu"] += Sigma2["ia"] * Delta1["ux"] * T2_["ixau"];
        temp["xu"] += Sigma2["IA"] * Delta1["ux"] * T2_["xIuA"];
        temp["XU"] += Sigma2["IA"] * Delta1["UX"] * T2_["IXAU"];
        temp["XU"] += Sigma2["ia"] * Delta1["UX"] * T2_["iXaU"];
    }

    if (CORRELATION_TERM) {
        temp["vu"] += Xi3["ia"] * V["auiv"];
        temp["vu"] += Xi3["IA"] * V["uAvI"];
        temp["VU"] += Xi3["IA"] * V["AUIV"];
        temp["VU"] += Xi3["ia"] * V["aUiV"];

        temp["vu"] += Xi3["ia"] * V["aviu"];
        temp["vu"] += Xi3["IA"] * V["vAuI"];
        temp["VU"] += Xi3["IA"] * V["AVIU"];
        temp["VU"] += Xi3["ia"] * V["aViU"];

        temp["xu"] += Xi3["ia"] * Delta1["xu"] * T2_["iuax"];
        temp["xu"] += Xi3["IA"] * Delta1["xu"] * T2_["uIxA"];
        temp["XU"] += Xi3["IA"] * Delta1["XU"] * T2_["IUAX"];
        temp["XU"] += Xi3["ia"] * Delta1["XU"] * T2_["iUaX"];

        temp["xu"] += Xi3["ia"] * Delta1["ux"] * T2_["ixau"];
        temp["xu"] += Xi3["IA"] * Delta1["ux"] * T2_["xIuA"];
        temp["XU"] += Xi3["IA"] * Delta1["UX"] * T2_["IXAU"];
        temp["XU"] += Xi3["ia"] * Delta1["UX"] * T2_["iXaU"];
    }

    if (X7_TERM) {
        temp["uv"] += F_["ev"] * T1_["ue"];
        temp["UV"] += F_["EV"] * T1_["UE"];

        temp["uv"] += F_["eu"] * T1_["ve"];
        temp["UV"] += F_["EU"] * T1_["VE"];

        temp["uv"] -= F_["vm"] * T1_["mu"];
        temp["UV"] -= F_["VM"] * T1_["MU"];

        temp["uv"] -= F_["um"] * T1_["mv"];
        temp["UV"] -= F_["UM"] * T1_["MV"];
    }

    Alpha += 0.5 * temp["uv"] * Gamma1_["uv"];
    Alpha += 0.5 * temp["UV"] * Gamma1_["UV"];

    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", {"aaaa", "AAAA", "aAaA"});

    if (X1_TERM) {
        temp4["uvxy"] += 0.125 * V_["efxy"] * T2_["uvef"];
        temp4["uvxy"] += 0.125 * V_["ewxy"] * T2_["uvez"] * Eta1_["zw"];
        temp4["uvxy"] += 0.125 * V_["wexy"] * T2_["uvze"] * Eta1_["zw"];

        temp4["UVXY"] += 0.125 * V_["EFXY"] * T2_["UVEF"];
        temp4["UVXY"] += 0.125 * V_["EWXY"] * T2_["UVEZ"] * Eta1_["ZW"];
        temp4["UVXY"] += 0.125 * V_["WEXY"] * T2_["UVZE"] * Eta1_["ZW"];

        temp4["uVxY"] += V_["eFxY"] * T2_["uVeF"];
        temp4["uVxY"] += V_["eWxY"] * T2_["uVeZ"] * Eta1_["ZW"];
        temp4["uVxY"] += V_["wExY"] * T2_["uVzE"] * Eta1_["zw"];
    }

    if (X2_TERM) {
        temp4["uvxy"] += 0.125 * V_["uvmn"] * T2_["mnxy"];
        temp4["uvxy"] += 0.125 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["zw"];
        temp4["uvxy"] += 0.125 * V_["uvzm"] * T2_["wmxy"] * Gamma1_["zw"];

        temp4["UVXY"] += 0.125 * V_["UVMN"] * T2_["MNXY"];
        temp4["UVXY"] += 0.125 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["ZW"];
        temp4["UVXY"] += 0.125 * V_["UVZM"] * T2_["WMXY"] * Gamma1_["ZW"];

        temp4["uVxY"] += V_["uVmN"] * T2_["mNxY"];
        temp4["uVxY"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZW"];
        temp4["uVxY"] += V_["uVzM"] * T2_["wMxY"] * Gamma1_["zw"];
    }

    if (X3_TERM) {
        temp4["xyuv"] -= V_["vemx"] * T2_["muye"];
        temp4["xyuv"] -= V_["vwmx"] * T2_["muyz"] * Eta1_["zw"];
        temp4["xyuv"] -= V_["vezx"] * T2_["wuye"] * Gamma1_["zw"];
        temp4["xyuv"] -= V_["vExM"] * T2_["uMyE"];
        temp4["xyuv"] -= V_["vWxM"] * T2_["uMyZ"] * Eta1_["ZW"];
        temp4["xyuv"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"];
        temp4["XYUV"] -= V_["eVmX"] * T2_["mUeY"];
        temp4["XYUV"] -= V_["wVmX"] * T2_["mUzY"] * Eta1_["zw"];
        temp4["XYUV"] -= V_["eVzX"] * T2_["wUeY"] * Gamma1_["zw"];
        temp4["XYUV"] -= V_["VEMX"] * T2_["MUYE"];
        temp4["XYUV"] -= V_["VWMX"] * T2_["MUYZ"] * Eta1_["ZW"];
        temp4["XYUV"] -= V_["VEZX"] * T2_["WUYE"] * Gamma1_["ZW"];
        temp4["xYvU"] -= V_["vemx"] * T2_["mUeY"];
        temp4["xYvU"] -= V_["vwmx"] * T2_["mUzY"] * Eta1_["zw"];
        temp4["xYvU"] -= V_["vezx"] * T2_["wUeY"] * Gamma1_["zw"];
        temp4["xYvU"] -= V_["vExM"] * T2_["MUYE"];
        temp4["xYvU"] -= V_["vWxM"] * T2_["MUYZ"] * Eta1_["ZW"];
        temp4["xYvU"] -= V_["vExZ"] * T2_["WUYE"] * Gamma1_["ZW"];
        temp4["yXuV"] -= V_["eVmX"] * T2_["muye"];
        temp4["yXuV"] -= V_["wVmX"] * T2_["muyz"] * Eta1_["zw"];
        temp4["yXuV"] -= V_["eVzX"] * T2_["wuye"] * Gamma1_["zw"];
        temp4["yXuV"] -= V_["VEMX"] * T2_["uMyE"];
        temp4["yXuV"] -= V_["VWMX"] * T2_["uMyZ"] * Eta1_["ZW"];
        temp4["yXuV"] -= V_["VEZX"] * T2_["uWyE"] * Gamma1_["ZW"];
        temp4["yXvU"] -= V_["vEmX"] * T2_["mUyE"];
        temp4["yXvU"] -= V_["vWmX"] * T2_["mUyZ"] * Eta1_["ZW"];
        temp4["yXvU"] -= V_["vEzX"] * T2_["wUyE"] * Gamma1_["zw"];
        temp4["xYuV"] -= V_["eVxM"] * T2_["uMeY"];
        temp4["xYuV"] -= V_["wVxM"] * T2_["uMzY"] * Eta1_["zw"];
        temp4["xYuV"] -= V_["eVxZ"] * T2_["uWeY"] * Gamma1_["ZW"];
    }

    if (X5_TERM) {
        temp4["uvxy"] += 0.5 * F_["ex"] * T2_["uvey"];
        temp4["UVXY"] += 0.5 * F_["EX"] * T2_["UVEY"];
        temp4["xYuV"] += F_["ex"] * T2_["uVeY"];
        temp4["yXuV"] += F_["EX"] * T2_["uVyE"];
        temp4["uvxy"] -= 0.5 * F_["vm"] * T2_["umxy"];
        temp4["UVXY"] -= 0.5 * F_["VM"] * T2_["UMXY"];
        temp4["xYvU"] -= F_["vm"] * T2_["mUxY"];
        temp4["xYuV"] -= F_["VM"] * T2_["uMxY"];
    }

    if (X6_TERM) {
        temp4["uvxy"] += 0.5 * T1_["ue"] * V_["evxy"];
        temp4["UVXY"] += 0.5 * T1_["UE"] * V_["EVXY"];
        temp4["xYuV"] += T1_["ue"] * V_["eVxY"];
        temp4["xYvU"] += T1_["UE"] * V_["vExY"];
        temp4["uvxy"] -= 0.5 * T1_["mx"] * V_["uvmy"];
        temp4["UVXY"] -= 0.5 * T1_["MX"] * V_["UVMY"];
        temp4["xYuV"] -= T1_["mx"] * V_["uVmY"];
        temp4["yXuV"] -= T1_["MX"] * V_["uVyM"];
    }

    Alpha += 2 * temp4["uvxy"] * Lambda2_["xyuv"];
    Alpha += 2 * temp4["UVXY"] * Lambda2_["XYUV"];
    Alpha += 2 * temp4["uVxY"] * Lambda2_["xYuV"];
    Alpha -= temp4["uvxy"] * Gamma2_["xyuv"];
    Alpha -= temp4["UVXY"] * Gamma2_["XYUV"];
    Alpha -= temp4["uVxY"] * Gamma2_["xYuV"];

    if (X4_TERM) {
        Alpha +=
            0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") * rdms_.g3aaa()("xyzuvw");
        Alpha +=
            0.25 * V_.block("AACA")("UVMZ") * T2_.block("CAAA")("MWXY") * rdms_.g3bbb()("XYZUVW");
        Alpha -=
            0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") * rdms_.g3aab()("xyZuvW");
        Alpha -=
            0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") * rdms_.g3aab()("xyZuvW");
        Alpha +=
            1.00 * V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * rdms_.g3aab()("xyZuvW");
        Alpha -=
            0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") * rdms_.g3abb()("xYZuVW");
        Alpha -=
            0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") * rdms_.g3abb()("xYZuVW");
        Alpha +=
            1.00 * V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * rdms_.g3abb()("xYZuVW");

        Alpha -= 1.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"] * Gamma2_["xyvw"];
        Alpha -= 2.0 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["uz"] * Gamma2_["xYvW"];
        Alpha += 1.0 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["uz"] * Gamma2_["XYVW"];
        Alpha += 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["uz"] * Gamma2_["xYwV"];
        Alpha -= 1.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["UZ"] * Gamma2_["XYVW"];
        Alpha -= 2.0 * V_["UVMZ"] * T2_["wMxY"] * Gamma1_["UZ"] * Gamma2_["xYwV"];
        Alpha += 1.0 * V_["vUmZ"] * T2_["mwxy"] * Gamma1_["UZ"] * Gamma2_["xyvw"];
        Alpha += 2.0 * V_["vUmZ"] * T2_["mWxY"] * Gamma1_["UZ"] * Gamma2_["xYvW"];

        Alpha -= 0.5 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"] * Gamma2_["xyuv"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["wz"] * Gamma2_["xYuV"];
        Alpha -= 0.5 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["WZ"] * Gamma2_["XYUV"];
        Alpha -= 2.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["WZ"] * Gamma2_["xYuV"];

        Alpha += 2.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xu"] * Gamma2_["vwzy"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mwxy"] * Gamma1_["xu"] * Gamma2_["wVyZ"];
        Alpha += 2.0 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma2_["vWzY"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma2_["VWZY"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["xu"] * Gamma2_["wVzY"];

        Alpha += 2.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["XU"] * Gamma2_["VWZY"];
        Alpha += 2.0 * V_["vUzM"] * T2_["MWXY"] * Gamma1_["XU"] * Gamma2_["vWzY"];
        Alpha += 2.0 * V_["UVMZ"] * T2_["wMyX"] * Gamma1_["XU"] * Gamma2_["wVyZ"];
        Alpha += 2.0 * V_["vUzM"] * T2_["wMyX"] * Gamma1_["XU"] * Gamma2_["vwzy"];
        Alpha -= 2.0 * V_["vUmZ"] * T2_["mWyX"] * Gamma1_["XU"] * Gamma2_["vWyZ"];

        Alpha += 1.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xw"] * Gamma2_["uvzy"];
        Alpha -= 2.0 * V_["uVmZ"] * T2_["mwxy"] * Gamma1_["xw"] * Gamma2_["uVyZ"];
        Alpha -= 1.0 * V_["UVMZ"] * T2_["wMxY"] * Gamma1_["xw"] * Gamma2_["UVZY"];
        Alpha += 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["xw"] * Gamma2_["uVzY"];

        Alpha -= 1.0 * V_["uvmz"] * T2_["mWyX"] * Gamma1_["XW"] * Gamma2_["uvzy"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mWyX"] * Gamma1_["XW"] * Gamma2_["uVyZ"];
        Alpha += 1.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["XW"] * Gamma2_["UVZY"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["XW"] * Gamma2_["uVzY"];

        Alpha += 6 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha += 6 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha -= 6 * V_["vUmZ"] * T2_["mwxy"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha -= 6 * V_["vUmZ"] * T2_["mWxY"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["YW"];

        Alpha += 6 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha += 6 * V_["UVMZ"] * T2_["wMyX"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["yw"];
        Alpha -= 6 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha -= 6 * V_["uVzM"] * T2_["wMyX"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["yw"];

        Alpha += 3.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["yv"];
        Alpha += 6.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["WZ"] * Gamma1_["xu"] * Gamma1_["YV"];
        Alpha += 3.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["WZ"] * Gamma1_["XU"] * Gamma1_["YV"];
        Alpha += 6.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["YV"];

        Alpha -=
            0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * rdms_.g3aaa()("xyzuvw");
        Alpha -=
            0.25 * V_.block("VAAA")("EWXY") * T2_.block("AAVA")("UVEZ") * rdms_.g3bbb()("XYZUVW");
        Alpha +=
            0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * rdms_.g3aab()("xyZuvW");
        Alpha -=
            0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * rdms_.g3aab()("xyZuvW");
        Alpha -=
            1.00 * V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * rdms_.g3aab()("xyZuvW");
        Alpha +=
            0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * rdms_.g3abb()("xYZuVW");
        Alpha -=
            0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * rdms_.g3abb()("xYZuVW");
        Alpha -=
            1.00 * V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * rdms_.g3abb()("xYZuVW");

        Alpha += 1.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["uz"] * Gamma2_["xyvw"];
        Alpha += 2.0 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["uz"] * Gamma2_["xYvW"];
        Alpha -= 1.0 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["uz"] * Gamma2_["XYVW"];
        Alpha -= 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["uz"] * Gamma2_["xYwV"];
        Alpha += 1.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["UZ"] * Gamma2_["XYVW"];
        Alpha += 2.0 * V_["EZUV"] * T2_["xYwE"] * Gamma1_["UZ"] * Gamma2_["xYwV"];
        Alpha -= 1.0 * V_["eZvU"] * T2_["xyew"] * Gamma1_["UZ"] * Gamma2_["xyvw"];
        Alpha -= 2.0 * V_["eZvU"] * T2_["xYeW"] * Gamma1_["UZ"] * Gamma2_["xYvW"];

        Alpha += 0.5 * V_["ezuv"] * T2_["xyew"] * Gamma1_["wz"] * Gamma2_["xyuv"];
        Alpha += 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["wz"] * Gamma2_["xYuV"];
        Alpha += 0.5 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["WZ"] * Gamma2_["XYUV"];
        Alpha += 2.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["WZ"] * Gamma2_["xYuV"];

        Alpha -= 2.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["xu"] * Gamma2_["vwzy"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["xyew"] * Gamma1_["xu"] * Gamma2_["wVyZ"];
        Alpha -= 2.0 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["xu"] * Gamma2_["vWzY"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["xu"] * Gamma2_["VWZY"];
        Alpha += 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["xu"] * Gamma2_["wVzY"];
        Alpha -= 2.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["XU"] * Gamma2_["VWZY"];
        Alpha -= 2.0 * V_["zEvU"] * T2_["XYEW"] * Gamma1_["XU"] * Gamma2_["vWzY"];
        Alpha -= 2.0 * V_["EZUV"] * T2_["yXwE"] * Gamma1_["XU"] * Gamma2_["wVyZ"];
        Alpha -= 2.0 * V_["zEvU"] * T2_["yXwE"] * Gamma1_["XU"] * Gamma2_["vwzy"];
        Alpha += 2.0 * V_["eZvU"] * T2_["yXeW"] * Gamma1_["XU"] * Gamma2_["vWyZ"];

        Alpha -= 1.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["xw"] * Gamma2_["uvzy"];
        Alpha += 2.0 * V_["eZuV"] * T2_["xyew"] * Gamma1_["xw"] * Gamma2_["uVyZ"];
        Alpha += 1.0 * V_["EZUV"] * T2_["xYwE"] * Gamma1_["xw"] * Gamma2_["UVZY"];
        Alpha -= 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["xw"] * Gamma2_["uVzY"];
        Alpha += 1.0 * V_["ezuv"] * T2_["yXeW"] * Gamma1_["XW"] * Gamma2_["uvzy"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["yXeW"] * Gamma1_["XW"] * Gamma2_["uVyZ"];
        Alpha -= 1.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["XW"] * Gamma2_["UVZY"];
        Alpha += 2.0 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["XW"] * Gamma2_["uVzY"];

        Alpha -= 6 * V_["ezuv"] * T2_["xyew"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha -= 6 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha += 6 * V_["eZvU"] * T2_["xyew"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha += 6 * V_["eZvU"] * T2_["xYeW"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha -= 6 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha -= 6 * V_["EZUV"] * T2_["yXwE"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["yw"];
        Alpha += 6 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha += 6 * V_["zEuV"] * T2_["yXwE"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["yw"];

        Alpha -= 3.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["yv"];
        Alpha -= 6.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["WZ"] * Gamma1_["xu"] * Gamma1_["YV"];
        Alpha -= 3.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["WZ"] * Gamma1_["XU"] * Gamma1_["YV"];
        Alpha -= 6.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["YV"];
    }

    Alpha += Z["mn"] * V["m,a1,n,u1"] * Gamma1_["u1,a1"];
    Alpha += Z["mn"] * V["m,A1,n,U1"] * Gamma1_["U1,A1"];
    Alpha += Z["MN"] * V["M,A1,N,U1"] * Gamma1_["U1,A1"];
    Alpha += Z["MN"] * V["a1,M,u1,N"] * Gamma1_["u1,a1"];

    Alpha += Z["ef"] * V["e,a1,f,u1"] * Gamma1_["u1,a1"];
    Alpha += Z["ef"] * V["e,A1,f,U1"] * Gamma1_["U1,A1"];
    Alpha += Z["EF"] * V["E,A1,F,U1"] * Gamma1_["U1,A1"];
    Alpha += Z["EF"] * V["a1,E,u1,F"] * Gamma1_["u1,a1"];

    Alpha += temp3["uv"] * V["u,a1,v,u1"] * Gamma1_["u1,a1"];
    Alpha += temp3["uv"] * V["u,A1,v,U1"] * Gamma1_["U1,A1"];
    Alpha += temp3["UV"] * V["U,A1,V,U1"] * Gamma1_["U1,A1"];
    Alpha += temp3["UV"] * V["a1,U,u1,V"] * Gamma1_["u1,a1"];

    b_ck("K") += 2 * Alpha * ci("K");

    BlockedTensor temp5 = BTF_->build(CoreTensor, "temporal tensor", {"aa", "AA"});
    temp5["uv"] += Z["mn"] * V["mvnu"];
    temp5["uv"] += Z["MN"] * V["vMuN"];
    temp5["uv"] += Z["ef"] * V["evfu"];
    temp5["uv"] += Z["EF"] * V["vEuF"];
    temp5["UV"] += Z["MN"] * V["MVNU"];
    temp5["UV"] += Z["mn"] * V["mVnU"];
    temp5["UV"] += Z["EF"] * V["EVFU"];
    temp5["UV"] += Z["ef"] * V["eVfU"];

    /// Call the generalized sigma function to complete contractions
    for (const auto& pair: as_solver_->state_space_size_map()) {
        const auto& state = pair.first;
        std::map<std::string, double> block_factor1;
        std::map<std::string, double> block_factor2;
        std::map<std::string, double> block_factor3;
        block_factor1["aa"] = 1.0;
        block_factor1["AA"] = 1.0;
        block_factor2["aaaa"] = 1.0;
        block_factor2["aAaA"] = 1.0;
        block_factor2["AAAA"] = 1.0;
        block_factor3["aaaaaa"] = 1.0;
        block_factor3["aaAaaA"] = 1.0;
        block_factor3["aAAaAA"] = 1.0;
        block_factor3["AAAAAA"] = 1.0;

        auto sym_1 = BTF_->build(CoreTensor, "symmetrized 1-body tensor", spin_cases({"aa"}));
        auto sym_2 = BTF_->build(CoreTensor, "symmetrized 2-body tensor", spin_cases({"aaaa"}));
        auto sym_3 = BTF_->build(CoreTensor, "symmetrized 3-body tensor", spin_cases({"aaaaaa"}));
        {
            auto temp_1 = BTF_->build(CoreTensor, "1-body intermediate tensor", spin_cases({"aa"}));

            if (X4_TERM) {
                // -0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") * dlamb3_aaa("Kxyzuvw")
                temp_1["uz"] += 0.50 * V_["uvmz"] * T2_["mwxy"] * Gamma2_["xyvw"];
                temp_1["wz"] += 0.25 * V_["uvmz"] * T2_["mwxy"] * Gamma2_["xyuv"];
                temp_1["xu"] -= 1.00 * V_["uvmz"] * T2_["mwxy"] * Gamma2_["vwzy"];
                temp_1["xw"] -= 0.50 * V_["uvmz"] * T2_["mwxy"] * Gamma2_["uvzy"];
                temp_1["uz"] -= 2.00 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xv"] * Gamma1_["yw"];
                temp_1["xv"] -= 2.00 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"] * Gamma1_["yw"];
                temp_1["yw"] -= 2.00 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"] * Gamma1_["xv"];
                temp_1["wz"] -= 1.00 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xu"] * Gamma1_["yv"];
                temp_1["xu"] -= 1.00 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"] * Gamma1_["yv"];
                temp_1["yv"] -= 1.00 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"] * Gamma1_["xu"];

                // -0.25 * V_.block("AACA")("UVMZ") * T2_.block("CAAA")("MWXY") * dlamb3_bbb("KXYZUVW")
                temp_1["UZ"] += 0.50 * V_["UVMZ"] * T2_["MWXY"] * Gamma2_["XYVW"];
                temp_1["WZ"] += 0.25 * V_["UVMZ"] * T2_["MWXY"] * Gamma2_["XYUV"];
                temp_1["XU"] -= 1.00 * V_["UVMZ"] * T2_["MWXY"] * Gamma2_["VWZY"];
                temp_1["XW"] -= 0.50 * V_["UVMZ"] * T2_["MWXY"] * Gamma2_["UVZY"];
                temp_1["UZ"] -= 2.00 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["XV"] * Gamma1_["YW"];
                temp_1["XV"] -= 2.00 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["UZ"] * Gamma1_["YW"];
                temp_1["YW"] -= 2.00 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["UZ"] * Gamma1_["XV"];
                temp_1["WZ"] -= 1.00 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["XU"] * Gamma1_["YV"];
                temp_1["XU"] -= 1.00 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["WZ"] * Gamma1_["YV"];
                temp_1["YV"] -= 1.00 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["WZ"] * Gamma1_["XU"];

                // 0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") * dlamb3_aab("KxyZuvW")
                temp_1["xu"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["yZvW"];
                temp_1["vy"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xu"] * Gamma1_["yv"];
                temp_1["xv"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["yZuW"];
                temp_1["uy"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xv"] * Gamma1_["yu"];
                temp_1["yu"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["xZvW"];
                temp_1["vx"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yu"] * Gamma1_["xv"];
                temp_1["yv"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["ZW"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["xyuv"];
                temp_1["ux"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["xv"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yu"] * Gamma1_["xv"];

                // 0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") * dlamb3_aab("KxyZuvW")
                temp_1["xu"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["yZvW"];
                temp_1["vy"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xu"] * Gamma1_["yv"];
                temp_1["xv"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["yZuW"];
                temp_1["uy"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xv"] * Gamma1_["yu"];
                temp_1["yu"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["xZvW"];
                temp_1["vx"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yu"] * Gamma1_["xv"];
                temp_1["yv"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["ZW"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["xyuv"];
                temp_1["ux"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["xv"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yu"] * Gamma1_["xv"];

                // - V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * dlamb3_aab("KxyZuvW")
                temp_1["xu"] += V_["uWyM"] * T2_["vMxZ"] * Lambda2_["yZvW"];
                temp_1["vy"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xu"] * Gamma1_["yv"];
                temp_1["xv"] -= V_["uWyM"] * T2_["vMxZ"] * Lambda2_["yZuW"];
                temp_1["uy"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xv"] * Gamma1_["yu"];
                temp_1["yu"] -= V_["uWyM"] * T2_["vMxZ"] * Lambda2_["xZvW"];
                temp_1["vx"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yu"] * Gamma1_["xv"];
                temp_1["yv"] += V_["uWyM"] * T2_["vMxZ"] * Lambda2_["xZuW"];
                temp_1["ux"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["ZW"] += V_["uWyM"] * T2_["vMxZ"] * Lambda2_["xyuv"];
                temp_1["ux"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["xv"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yu"] * Gamma1_["xv"];

                // 0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") * dlamb3_abb("KxYZuVW")
                temp_1["xu"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Lambda2_["YZVW"];
                temp_1["YV"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["xu"] * Gamma1_["YV"];
                temp_1["YW"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["xu"] * Gamma1_["ZV"];
                temp_1["ZV"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YV"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["YW"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Lambda2_["xZuV"];
                temp_1["ux"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ZV"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YW"] * Gamma1_["xu"];
                temp_1["ZV"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Lambda2_["xYuW"];
                temp_1["ux"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["YW"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZV"] * Gamma1_["xu"];
                temp_1["ZW"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Lambda2_["xYuV"];
                temp_1["ux"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["YV"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["xu"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["YV"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["xu"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ZV"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YW"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZV"] * Gamma1_["xu"];

                // 0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") * dlamb3_abb("KxYZuVW")
                temp_1["xu"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Lambda2_["YZVW"];
                temp_1["YV"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["xu"] * Gamma1_["YV"];
                temp_1["YW"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["xu"] * Gamma1_["ZV"];
                temp_1["ZV"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YV"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["YW"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Lambda2_["xZuV"];
                temp_1["ux"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ZV"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YW"] * Gamma1_["xu"];
                temp_1["ZV"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Lambda2_["xYuW"];
                temp_1["ux"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["YW"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZV"] * Gamma1_["xu"];
                temp_1["ZW"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Lambda2_["xYuV"];
                temp_1["ux"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["YV"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["xu"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["YV"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["xu"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ZV"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YW"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZV"] * Gamma1_["xu"];

                // - V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * dlamb3_abb("KxYZuVW")
                temp_1["xu"] += V_["uVmZ"] * T2_["mWxY"] * Lambda2_["YZVW"];
                temp_1["YV"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma1_["YV"];
                temp_1["YW"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma1_["ZV"];
                temp_1["ZV"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YV"] += V_["uVmZ"] * T2_["mWxY"] * Lambda2_["xZuW"];
                temp_1["ux"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["YW"] -= V_["uVmZ"] * T2_["mWxY"] * Lambda2_["xZuV"];
                temp_1["ux"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ZV"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YW"] * Gamma1_["xu"];
                temp_1["ZV"] -= V_["uVmZ"] * T2_["mWxY"] * Lambda2_["xYuW"];
                temp_1["ux"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["YW"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZV"] * Gamma1_["xu"];
                temp_1["ZW"] += V_["uVmZ"] * T2_["mWxY"] * Lambda2_["xYuV"];
                temp_1["ux"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["YV"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["xu"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["YV"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["xu"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ZV"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YW"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZV"] * Gamma1_["xu"];

                // 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * dlamb3_aaa("Kxyzuvw")
                temp_1["uz"] -= 0.50 * V_["ewxy"] * T2_["uvez"] * Gamma2_["xyvw"];
                temp_1["wz"] -= 0.25 * V_["ewxy"] * T2_["uvez"] * Gamma2_["xyuv"];
                temp_1["xu"] += 1.00 * V_["ewxy"] * T2_["uvez"] * Gamma2_["vwzy"];
                temp_1["xw"] += 0.50 * V_["ewxy"] * T2_["uvez"] * Gamma2_["uvzy"];
                temp_1["uz"] += 2.00 * V_["ewxy"] * T2_["uvez"] * Gamma1_["xv"] * Gamma1_["yw"];
                temp_1["xv"] += 2.00 * V_["ewxy"] * T2_["uvez"] * Gamma1_["uz"] * Gamma1_["yw"];
                temp_1["yw"] += 2.00 * V_["ewxy"] * T2_["uvez"] * Gamma1_["uz"] * Gamma1_["xv"];
                temp_1["wz"] += 1.00 * V_["ewxy"] * T2_["uvez"] * Gamma1_["xu"] * Gamma1_["yv"];
                temp_1["xu"] += 1.00 * V_["ewxy"] * T2_["uvez"] * Gamma1_["wz"] * Gamma1_["yv"];
                temp_1["yv"] += 1.00 * V_["ewxy"] * T2_["uvez"] * Gamma1_["wz"] * Gamma1_["xu"];

                // 0.25 * V_.block("VAAA")("EWXY") * T2_.block("AAVA")("UVEZ") * dlamb3_bbb("KXYZUVW")
                temp_1["UZ"] -= 0.50 * V_["EWXY"] * T2_["UVEZ"] * Gamma2_["XYVW"];
                temp_1["WZ"] -= 0.25 * V_["EWXY"] * T2_["UVEZ"] * Gamma2_["XYUV"];
                temp_1["XU"] += 1.00 * V_["EWXY"] * T2_["UVEZ"] * Gamma2_["VWZY"];
                temp_1["XW"] += 0.50 * V_["EWXY"] * T2_["UVEZ"] * Gamma2_["UVZY"];
                temp_1["UZ"] += 2.00 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["XV"] * Gamma1_["YW"];
                temp_1["XV"] += 2.00 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["UZ"] * Gamma1_["YW"];
                temp_1["YW"] += 2.00 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["UZ"] * Gamma1_["XV"];
                temp_1["WZ"] += 1.00 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["XU"] * Gamma1_["YV"];
                temp_1["XU"] += 1.00 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["WZ"] * Gamma1_["YV"];
                temp_1["YV"] += 1.00 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["WZ"] * Gamma1_["XU"];

                // -0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * dlamb3_aab("KxyZuvW")
                temp_1["xu"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["yZvW"];
                temp_1["vy"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xu"] * Gamma1_["yv"];
                temp_1["xv"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["yZuW"];
                temp_1["uy"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xv"] * Gamma1_["yu"];
                temp_1["yu"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["xZvW"];
                temp_1["vx"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yu"] * Gamma1_["xv"];
                temp_1["yv"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["xZuW"];
                temp_1["ux"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["ZW"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["xyuv"];
                temp_1["ux"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["xv"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yu"] * Gamma1_["xv"];

                // 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * dlamb3_aab("KxyZuvW")
                temp_1["xu"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["yZvW"];
                temp_1["vy"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xu"] * Gamma1_["yv"];
                temp_1["xv"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["yZuW"];
                temp_1["uy"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xv"] * Gamma1_["yu"];
                temp_1["yu"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["xZvW"];
                temp_1["vx"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yu"] * Gamma1_["xv"];
                temp_1["yv"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["ZW"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["xyuv"];
                temp_1["ux"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["xv"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yu"] * Gamma1_["xv"];

                // V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * dlamb3_aab("KxyZuvW")
                temp_1["xu"] -= V_["vExZ"] * T2_["uWyE"] * Lambda2_["yZvW"];
                temp_1["vy"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["xu"] * Gamma1_["yv"];
                temp_1["xv"] += V_["vExZ"] * T2_["uWyE"] * Lambda2_["yZuW"];
                temp_1["uy"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["xv"] * Gamma1_["yu"];
                temp_1["yu"] += V_["vExZ"] * T2_["uWyE"] * Lambda2_["xZvW"];
                temp_1["vx"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["yu"] * Gamma1_["xv"];
                temp_1["yv"] -= V_["vExZ"] * T2_["uWyE"] * Lambda2_["xZuW"];
                temp_1["ux"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["ZW"] -= V_["vExZ"] * T2_["uWyE"] * Lambda2_["xyuv"];
                temp_1["ux"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["yv"] * Gamma1_["xu"];
                temp_1["xv"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["yu"] * Gamma1_["xv"];

                // -0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * dlamb3_abb("KxYZuVW")
                temp_1["xu"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Lambda2_["YZVW"];
                temp_1["YV"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["xu"] * Gamma1_["YV"];
                temp_1["YW"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["xu"] * Gamma1_["ZV"];
                temp_1["ZV"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YV"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Lambda2_["xZuW"];
                temp_1["ux"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["YW"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Lambda2_["xZuV"];
                temp_1["ux"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ZV"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YW"] * Gamma1_["xu"];
                temp_1["ZV"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Lambda2_["xYuW"];
                temp_1["ux"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["YW"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZV"] * Gamma1_["xu"];
                temp_1["ZW"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Lambda2_["xYuV"];
                temp_1["ux"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["YV"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["xu"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["YV"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["xu"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ZV"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YW"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZV"] * Gamma1_["xu"];

                // 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * dlamb3_abb("KxYZuVW")
                temp_1["xu"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Lambda2_["YZVW"];
                temp_1["YV"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["xu"] * Gamma1_["YV"];
                temp_1["YW"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["xu"] * Gamma1_["ZV"];
                temp_1["ZV"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YV"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ZW"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["YW"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Lambda2_["xZuV"];
                temp_1["ux"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ZV"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YW"] * Gamma1_["xu"];
                temp_1["ZV"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Lambda2_["xYuW"];
                temp_1["ux"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["YW"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZV"] * Gamma1_["xu"];
                temp_1["ZW"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Lambda2_["xYuV"];
                temp_1["ux"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["YV"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["xu"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["YV"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["xu"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ZV"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YW"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZV"] * Gamma1_["xu"];

                // V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * dlamb3_abb("KxYZuVW")
                temp_1["xu"] -= V_["eWxY"] * T2_["uVeZ"] * Lambda2_["YZVW"];
                temp_1["YV"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["xu"] * Gamma1_["YV"];
                temp_1["YW"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["xu"] * Gamma1_["ZV"];
                temp_1["ZV"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YV"] -= V_["eWxY"] * T2_["uVeZ"] * Lambda2_["xZuW"];
                temp_1["ux"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ZW"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["YW"] += V_["eWxY"] * T2_["uVeZ"] * Lambda2_["xZuV"];
                temp_1["ux"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ZV"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YW"] * Gamma1_["xu"];
                temp_1["ZV"] += V_["eWxY"] * T2_["uVeZ"] * Lambda2_["xYuW"];
                temp_1["ux"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["YW"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZV"] * Gamma1_["xu"];
                temp_1["ZW"] -= V_["eWxY"] * T2_["uVeZ"] * Lambda2_["xYuV"];
                temp_1["ux"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["YV"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["xu"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["YV"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["ZW"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YV"] * Gamma1_["xu"];
                temp_1["xu"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ZV"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["xu"] * Gamma1_["YW"];
                temp_1["YW"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZV"] * Gamma1_["xu"];
            }

            // - H.block("aa")("vu") * cc1a("Kuv")
            // - H.block("AA")("VU") * cc1b("KUV")
            temp_1["uv"] -= H["vu"];
            temp_1["UV"] -= H["VU"];

            // - V_sumA_Alpha.block("aa")("vu") * cc1a("Kuv")
            // - V_sumB_Alpha.block("aa")("vu") * cc1a("Kuv")
            // - V_sumB_Beta.block("AA")("VU") * cc1b("KUV")
            // - V_sumA_Beta.block("AA")("VU") * cc1b("KUV")
            temp_1["uv"] -= V_sumA_Alpha["vu"];
            temp_1["uv"] -= V_sumB_Alpha["vu"];
            temp_1["UV"] -= V_sumB_Beta["VU"];
            temp_1["UV"] -= V_sumA_Beta["VU"];

            // - 0.5 * temp.block("aa")("uv") * cc1a("Kuv")
            // - 0.5 * temp.block("AA")("UV") * cc1b("KUV")
            temp_1["uv"] -= 0.5 * temp["vu"];
            temp_1["UV"] -= 0.5 * temp["VU"];

            /// temp4 * dlambda2
            //      temp4.block("aaaa")("uvxy") * dlambda2_aaaa("Kxyuv")
            //      temp4.block("AAAA")("UVXY") * dlambda2_bbbb("KXYUV")
            //      temp4.block("aAaA")("uVxY") * dlambda2_abab("KxYuV")
            temp_1["ux"] += temp4["uvxy"] * Gamma1_["yv"];
            temp_1["vy"] += temp4["uvxy"] * Gamma1_["xu"];
            temp_1["vx"] -= temp4["uvxy"] * Gamma1_["yu"];
            temp_1["uy"] -= temp4["uvxy"] * Gamma1_["xv"];
            temp_1["XU"] += temp4["UVXY"] * Gamma1_["YV"];
            temp_1["YV"] += temp4["UVXY"] * Gamma1_["XU"];
            temp_1["XV"] -= temp4["UVXY"] * Gamma1_["YU"];
            temp_1["YU"] -= temp4["UVXY"] * Gamma1_["XV"];
            temp_1["ux"] += temp4["uVxY"] * Gamma1_["YV"];
            temp_1["YV"] += temp4["uVxY"] * Gamma1_["xu"];

            // - temp3.block("aa")("xy") * V.block("aaaa")("xvyu") * cc1a("Kuv")
            // - temp3.block("aa")("xy") * V.block("aAaA")("xVyU") * cc1b("KUV")
            // - temp3.block("AA")("XY") * V.block("AAAA")("XVYU") * cc1b("KUV")
            // - temp3.block("AA")("XY") * V.block("aAaA")("vXuY") * cc1a("Kuv")
            temp_1["uv"] -= temp3["xy"] * V["xvyu"];
            temp_1["UV"] -= temp3["xy"] * V["xVyU"];
            temp_1["UV"] -= temp3["XY"] * V["XVYU"];
            temp_1["uv"] -= temp3["XY"] * V["vXuY"];

            // - temp5.block("aa")("uv") * cc1a("Kuv")
            // - temp5.block("AA")("UV") * cc1b("KUV")
            temp_1["uv"] -= temp5["uv"];
            temp_1["UV"] -= temp5["UV"];

            /// Symmetrization
            //  α α
            sym_1["uv"] += temp_1["uv"];
            sym_1["uv"] += temp_1["vu"];
            //  β β
            sym_1["UV"] += temp_1["UV"];
            sym_1["UV"] += temp_1["VU"];
        }
        {
            auto temp_2 = BTF_->build(CoreTensor, "2-body intermediate tensor", spin_cases({"aaaa"}));

            if (X4_TERM) {
                // -0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") * dlamb3_aaa("Kxyzuvw")
                temp_2["xyvw"] += 0.50 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"];
                temp_2["xyuv"] += 0.25 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"];
                temp_2["vwzy"] -= 1.00 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xu"];
                temp_2["uvzy"] -= 0.50 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xw"];

                // -0.25 * V_.block("AACA")("UVMZ") * T2_.block("CAAA")("MWXY") * dlamb3_bbb("KXYZUVW")
                temp_2["XYVW"] += 0.50 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["UZ"];
                temp_2["XYUV"] += 0.25 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["WZ"];
                temp_2["VWZY"] -= 1.00 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["XU"];
                temp_2["UVZY"] -= 0.50 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["XW"];

                // 0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") * dlamb3_aab("KxyZuvW")
                temp_2["yZvW"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xu"];
                temp_2["yZuW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xv"];
                temp_2["xZvW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yu"];
                temp_2["xZuW"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yv"];
                temp_2["xyuv"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"];

                // 0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") * dlamb3_aab("KxyZuvW")
                temp_2["yZvW"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xu"];
                temp_2["yZuW"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xv"];
                temp_2["xZvW"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yu"];
                temp_2["xZuW"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yv"];
                temp_2["xyuv"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"];

                // - V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * dlamb3_aab("KxyZuvW")
                temp_2["yZvW"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xu"];
                temp_2["yZuW"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xv"];
                temp_2["xZvW"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yu"];
                temp_2["xZuW"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yv"];
                temp_2["xyuv"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"];

                // 0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") * dlamb3_abb("KxYZuVW")
                temp_2["YZVW"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["xu"];
                temp_2["xZuW"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YV"];
                temp_2["xZuV"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YW"];
                temp_2["xYuW"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZV"];
                temp_2["xYuV"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZW"];

                // 0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") * dlamb3_abb("KxYZuVW")
                temp_2["YZVW"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["xu"];
                temp_2["xZuW"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YV"];
                temp_2["xZuV"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YW"];
                temp_2["xYuW"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZV"];
                temp_2["xYuV"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZW"];

                // - V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * dlamb3_abb("KxYZuVW")
                temp_2["YZVW"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"];
                temp_2["xZuW"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YV"];
                temp_2["xZuV"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YW"];
                temp_2["xYuW"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZV"];
                temp_2["xYuV"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZW"];

                // 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * dlamb3_aaa("Kxyzuvw")
                temp_2["xyvw"] -= 0.50 * V_["ewxy"] * T2_["uvez"] * Gamma1_["uz"];
                temp_2["xyuv"] -= 0.25 * V_["ewxy"] * T2_["uvez"] * Gamma1_["wz"];
                temp_2["vwzy"] += 1.00 * V_["ewxy"] * T2_["uvez"] * Gamma1_["xu"];
                temp_2["uvzy"] += 0.50 * V_["ewxy"] * T2_["uvez"] * Gamma1_["xw"];

                // 0.25 * V_.block("VAAA")("EWXY") * T2_.block("AAVA")("UVEZ") * dlamb3_bbb("KXYZUVW")
                temp_2["XYVW"] -= 0.50 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["UZ"];
                temp_2["XYUV"] -= 0.25 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["WZ"];
                temp_2["VWZY"] += 1.00 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["XU"];
                temp_2["UVZY"] += 0.50 * V_["EWXY"] * T2_["UVEZ"] * Gamma1_["XW"];

                // -0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * dlamb3_aab("KxyZuvW")
                temp_2["yZvW"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xu"];
                temp_2["yZuW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xv"];
                temp_2["xZvW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yu"];
                temp_2["xZuW"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yv"];
                temp_2["xyuv"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"];

                // 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * dlamb3_aab("KxyZuvW")
                temp_2["yZvW"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xu"];
                temp_2["yZuW"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xv"];
                temp_2["xZvW"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yu"];
                temp_2["xZuW"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yv"];
                temp_2["xyuv"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"];

                // V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * dlamb3_aab("KxyZuvW")
                temp_2["yZvW"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["xu"];
                temp_2["yZuW"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["xv"];
                temp_2["xZvW"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["yu"];
                temp_2["xZuW"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["yv"];
                temp_2["xyuv"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"];

                // -0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * dlamb3_abb("KxYZuVW")
                temp_2["YZVW"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["xu"];
                temp_2["xZuW"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YV"];
                temp_2["xZuV"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YW"];
                temp_2["xYuW"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZV"];
                temp_2["xYuV"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZW"];

                // 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * dlamb3_abb("KxYZuVW")
                temp_2["YZVW"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["xu"];
                temp_2["xZuW"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YV"];
                temp_2["xZuV"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YW"];
                temp_2["xYuW"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZV"];
                temp_2["xYuV"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZW"];

                // V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * dlamb3_abb("KxYZuVW")
                temp_2["YZVW"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["xu"];
                temp_2["xZuW"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YV"];
                temp_2["xZuV"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YW"];
                temp_2["xYuW"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZV"];
                temp_2["xYuV"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZW"];
            }

            /// CI contribution
            //  -0.25 * V.block("aaaa")("xyuv") * cc2aa("Kuvxy")
            //  -0.25 * V.block("AAAA")("XYUV") * cc2bb("KUVXY")
            //  - V.block("aAaA")("xYuV") * cc2ab("KuVxY")
            temp_2["uvxy"] -= 0.25 * V["xyuv"];
            temp_2["UVXY"] -= 0.25 * V["XYUV"];
            temp_2["uVxY"] -=        V["xYuV"];
            
            /// temp4 * dlambda2
            //      temp4.block("aaaa")("uvxy") * dlambda2_aaaa("Kxyuv")
            //      temp4.block("AAAA")("UVXY") * dlambda2_bbbb("KXYUV")
            //      temp4.block("aAaA")("uVxY") * dlambda2_abab("KxYuV")
            temp_2["uvxy"] -= temp4["uvxy"];
            temp_2["UVXY"] -= temp4["UVXY"];
            temp_2["uVxY"] -= temp4["uVxY"];

            /// Symmetrization
            //  α α α α
            //  Antisymmetrization
            sym_2["xyuv"] += temp_2["uvxy"];
            sym_2["xyuv"] -= temp_2["uvyx"];
            sym_2["xyuv"] -= temp_2["vuxy"];
            sym_2["xyuv"] += temp_2["vuyx"];
            //  Antisymmetrization
            sym_2["uvxy"] += temp_2["uvxy"];
            sym_2["uvxy"] -= temp_2["uvyx"];
            sym_2["uvxy"] -= temp_2["vuxy"];
            sym_2["uvxy"] += temp_2["vuyx"];
            /// Symmetrization
            //  β β β β
            //  Antisymmetrization
            sym_2["UVXY"] += temp_2["UVXY"];
            sym_2["UVXY"] -= temp_2["UVYX"];
            sym_2["UVXY"] -= temp_2["VUXY"];
            sym_2["UVXY"] += temp_2["VUYX"];
            //  Antisymmetrization
            sym_2["XYUV"] += temp_2["UVXY"];
            sym_2["XYUV"] -= temp_2["UVYX"];
            sym_2["XYUV"] -= temp_2["VUXY"];
            sym_2["XYUV"] += temp_2["VUYX"];
            /// Symmetrization
            //  α β α β
            sym_2["uVxY"] += temp_2["uVxY"];
            sym_2["uVxY"] += temp_2["xYuV"];
        }
        {
            auto temp_3 = BTF_->build(CoreTensor, "3-body intermediate tensor", spin_cases({"aaaaaa"}));
            if (X4_TERM) {

                // -0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") * dlamb3_aaa("Kxyzuvw")
                temp_3["xyzuvw"] -= 0.25 * V_["uvmz"] * T2_["mwxy"];
            
                // -0.25 * V_.block("AACA")("UVMZ") * T2_.block("CAAA")("MWXY") * dlamb3_bbb("KXYZUVW")
                temp_3["XYZUVW"] -= 0.25 * V_["UVMZ"] * T2_["MWXY"];

                // 0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") * dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"];

                // 0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") * dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] += 0.50 * V_["uWmZ"] * T2_["mvxy"];

                // - V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] -= V_["uWyM"] * T2_["vMxZ"];

                // 0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") * dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] += 0.50 * V_["VWMZ"] * T2_["uMxY"];

                // 0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") * dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] += 0.50 * V_["uVxM"] * T2_["MWYZ"];

                // - V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] -= V_["uVmZ"] * T2_["mWxY"];

                // 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * dlamb3_aaa("Kxyzuvw")
                temp_3["xyzuvw"] += 0.25 * V_["ewxy"] * T2_["uvez"];

                // 0.25 * V_.block("VAAA")("EWXY") * T2_.block("AAVA")("UVEZ") * dlamb3_bbb("KXYZUVW")
                temp_3["XYZUVW"] += 0.25 * V_["EWXY"] * T2_["UVEZ"];

                // -0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"];

                // 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] += 0.50 * V_["vexy"] * T2_["uWeZ"];

                // V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] += V_["vExZ"] * T2_["uWyE"];

                // -0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] -= 0.50 * V_["uExY"] * T2_["VWEZ"];

                // 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] += 0.50 * V_["WEYZ"] * T2_["uVxE"];

                // V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] += V_["eWxY"] * T2_["uVeZ"];
            }
            /// Symmetrization
            //  α α α α α α
            //  Antisymmetrization
            sym_3["xyzuvw"] += temp_3["uvwxyz"];
            sym_3["xyzuvw"] -= temp_3["uwvxyz"];
            sym_3["xyzuvw"] -= temp_3["vuwxyz"];
            sym_3["xyzuvw"] += temp_3["vwuxyz"];
            sym_3["xyzuvw"] += temp_3["wuvxyz"];
            sym_3["xyzuvw"] -= temp_3["wvuxyz"];
            sym_3["xyzuvw"] -= temp_3["uvwxzy"];
            sym_3["xyzuvw"] += temp_3["uwvxzy"];
            sym_3["xyzuvw"] += temp_3["vuwxzy"];
            sym_3["xyzuvw"] -= temp_3["vwuxzy"];
            sym_3["xyzuvw"] -= temp_3["wuvxzy"];
            sym_3["xyzuvw"] += temp_3["wvuxzy"];
            sym_3["xyzuvw"] -= temp_3["uvwyxz"];
            sym_3["xyzuvw"] += temp_3["uwvyxz"];
            sym_3["xyzuvw"] += temp_3["vuwyxz"];
            sym_3["xyzuvw"] -= temp_3["vwuyxz"];
            sym_3["xyzuvw"] -= temp_3["wuvyxz"];
            sym_3["xyzuvw"] += temp_3["wvuyxz"];
            sym_3["xyzuvw"] += temp_3["uvwyzx"];
            sym_3["xyzuvw"] -= temp_3["uwvyzx"];
            sym_3["xyzuvw"] -= temp_3["vuwyzx"];
            sym_3["xyzuvw"] += temp_3["vwuyzx"];
            sym_3["xyzuvw"] += temp_3["wuvyzx"];
            sym_3["xyzuvw"] -= temp_3["wvuyzx"];
            sym_3["xyzuvw"] += temp_3["uvwzxy"];
            sym_3["xyzuvw"] -= temp_3["uwvzxy"];
            sym_3["xyzuvw"] -= temp_3["vuwzxy"];
            sym_3["xyzuvw"] += temp_3["vwuzxy"];
            sym_3["xyzuvw"] += temp_3["wuvzxy"];
            sym_3["xyzuvw"] -= temp_3["wvuzxy"];
            sym_3["xyzuvw"] -= temp_3["uvwzyx"];
            sym_3["xyzuvw"] += temp_3["uwvzyx"];
            sym_3["xyzuvw"] += temp_3["vuwzyx"];
            sym_3["xyzuvw"] -= temp_3["vwuzyx"];
            sym_3["xyzuvw"] -= temp_3["wuvzyx"];
            sym_3["xyzuvw"] += temp_3["wvuzyx"];
            //  Antisymmetrization
            sym_3["uvwxyz"] += temp_3["uvwxyz"];
            sym_3["uvwxyz"] -= temp_3["uwvxyz"];
            sym_3["uvwxyz"] -= temp_3["vuwxyz"];
            sym_3["uvwxyz"] += temp_3["vwuxyz"];
            sym_3["uvwxyz"] += temp_3["wuvxyz"];
            sym_3["uvwxyz"] -= temp_3["wvuxyz"];
            sym_3["uvwxyz"] -= temp_3["uvwxzy"];
            sym_3["uvwxyz"] += temp_3["uwvxzy"];
            sym_3["uvwxyz"] += temp_3["vuwxzy"];
            sym_3["uvwxyz"] -= temp_3["vwuxzy"];
            sym_3["uvwxyz"] -= temp_3["wuvxzy"];
            sym_3["uvwxyz"] += temp_3["wvuxzy"];
            sym_3["uvwxyz"] -= temp_3["uvwyxz"];
            sym_3["uvwxyz"] += temp_3["uwvyxz"];
            sym_3["uvwxyz"] += temp_3["vuwyxz"];
            sym_3["uvwxyz"] -= temp_3["vwuyxz"];
            sym_3["uvwxyz"] -= temp_3["wuvyxz"];
            sym_3["uvwxyz"] += temp_3["wvuyxz"];
            sym_3["uvwxyz"] += temp_3["uvwyzx"];
            sym_3["uvwxyz"] -= temp_3["uwvyzx"];
            sym_3["uvwxyz"] -= temp_3["vuwyzx"];
            sym_3["uvwxyz"] += temp_3["vwuyzx"];
            sym_3["uvwxyz"] += temp_3["wuvyzx"];
            sym_3["uvwxyz"] -= temp_3["wvuyzx"];
            sym_3["uvwxyz"] += temp_3["uvwzxy"];
            sym_3["uvwxyz"] -= temp_3["uwvzxy"];
            sym_3["uvwxyz"] -= temp_3["vuwzxy"];
            sym_3["uvwxyz"] += temp_3["vwuzxy"];
            sym_3["uvwxyz"] += temp_3["wuvzxy"];
            sym_3["uvwxyz"] -= temp_3["wvuzxy"];
            sym_3["uvwxyz"] -= temp_3["uvwzyx"];
            sym_3["uvwxyz"] += temp_3["uwvzyx"];
            sym_3["uvwxyz"] += temp_3["vuwzyx"];
            sym_3["uvwxyz"] -= temp_3["vwuzyx"];
            sym_3["uvwxyz"] -= temp_3["wuvzyx"];
            sym_3["uvwxyz"] += temp_3["wvuzyx"];
            /// Symmetrization
            //  β β β β β β
            //  Antisymmetrization
            sym_3["XYZUVW"] += temp_3["UVWXYZ"];
            sym_3["XYZUVW"] -= temp_3["UWVXYZ"];
            sym_3["XYZUVW"] -= temp_3["VUWXYZ"];
            sym_3["XYZUVW"] += temp_3["VWUXYZ"];
            sym_3["XYZUVW"] += temp_3["WUVXYZ"];
            sym_3["XYZUVW"] -= temp_3["WVUXYZ"];
            sym_3["XYZUVW"] -= temp_3["UVWXZY"];
            sym_3["XYZUVW"] += temp_3["UWVXZY"];
            sym_3["XYZUVW"] += temp_3["VUWXZY"];
            sym_3["XYZUVW"] -= temp_3["VWUXZY"];
            sym_3["XYZUVW"] -= temp_3["WUVXZY"];
            sym_3["XYZUVW"] += temp_3["WVUXZY"];
            sym_3["XYZUVW"] -= temp_3["UVWYXZ"];
            sym_3["XYZUVW"] += temp_3["UWVYXZ"];
            sym_3["XYZUVW"] += temp_3["VUWYXZ"];
            sym_3["XYZUVW"] -= temp_3["VWUYXZ"];
            sym_3["XYZUVW"] -= temp_3["WUVYXZ"];
            sym_3["XYZUVW"] += temp_3["WVUYXZ"];
            sym_3["XYZUVW"] += temp_3["UVWYZX"];
            sym_3["XYZUVW"] -= temp_3["UWVYZX"];
            sym_3["XYZUVW"] -= temp_3["VUWYZX"];
            sym_3["XYZUVW"] += temp_3["VWUYZX"];
            sym_3["XYZUVW"] += temp_3["WUVYZX"];
            sym_3["XYZUVW"] -= temp_3["WVUYZX"];
            sym_3["XYZUVW"] += temp_3["UVWZXY"];
            sym_3["XYZUVW"] -= temp_3["UWVZXY"];
            sym_3["XYZUVW"] -= temp_3["VUWZXY"];
            sym_3["XYZUVW"] += temp_3["VWUZXY"];
            sym_3["XYZUVW"] += temp_3["WUVZXY"];
            sym_3["XYZUVW"] -= temp_3["WVUZXY"];
            sym_3["XYZUVW"] -= temp_3["UVWZYX"];
            sym_3["XYZUVW"] += temp_3["UWVZYX"];
            sym_3["XYZUVW"] += temp_3["VUWZYX"];
            sym_3["XYZUVW"] -= temp_3["VWUZYX"];
            sym_3["XYZUVW"] -= temp_3["WUVZYX"];
            sym_3["XYZUVW"] += temp_3["WVUZYX"];
            //  Antisymmetrization
            sym_3["UVWXYZ"] += temp_3["UVWXYZ"];
            sym_3["UVWXYZ"] -= temp_3["UWVXYZ"];
            sym_3["UVWXYZ"] -= temp_3["VUWXYZ"];
            sym_3["UVWXYZ"] += temp_3["VWUXYZ"];
            sym_3["UVWXYZ"] += temp_3["WUVXYZ"];
            sym_3["UVWXYZ"] -= temp_3["WVUXYZ"];
            sym_3["UVWXYZ"] -= temp_3["UVWXZY"];
            sym_3["UVWXYZ"] += temp_3["UWVXZY"];
            sym_3["UVWXYZ"] += temp_3["VUWXZY"];
            sym_3["UVWXYZ"] -= temp_3["VWUXZY"];
            sym_3["UVWXYZ"] -= temp_3["WUVXZY"];
            sym_3["UVWXYZ"] += temp_3["WVUXZY"];
            sym_3["UVWXYZ"] -= temp_3["UVWYXZ"];
            sym_3["UVWXYZ"] += temp_3["UWVYXZ"];
            sym_3["UVWXYZ"] += temp_3["VUWYXZ"];
            sym_3["UVWXYZ"] -= temp_3["VWUYXZ"];
            sym_3["UVWXYZ"] -= temp_3["WUVYXZ"];
            sym_3["UVWXYZ"] += temp_3["WVUYXZ"];
            sym_3["UVWXYZ"] += temp_3["UVWYZX"];
            sym_3["UVWXYZ"] -= temp_3["UWVYZX"];
            sym_3["UVWXYZ"] -= temp_3["VUWYZX"];
            sym_3["UVWXYZ"] += temp_3["VWUYZX"];
            sym_3["UVWXYZ"] += temp_3["WUVYZX"];
            sym_3["UVWXYZ"] -= temp_3["WVUYZX"];
            sym_3["UVWXYZ"] += temp_3["UVWZXY"];
            sym_3["UVWXYZ"] -= temp_3["UWVZXY"];
            sym_3["UVWXYZ"] -= temp_3["VUWZXY"];
            sym_3["UVWXYZ"] += temp_3["VWUZXY"];
            sym_3["UVWXYZ"] += temp_3["WUVZXY"];
            sym_3["UVWXYZ"] -= temp_3["WVUZXY"];
            sym_3["UVWXYZ"] -= temp_3["UVWZYX"];
            sym_3["UVWXYZ"] += temp_3["UWVZYX"];
            sym_3["UVWXYZ"] += temp_3["VUWZYX"];
            sym_3["UVWXYZ"] -= temp_3["VWUZYX"];
            sym_3["UVWXYZ"] -= temp_3["WUVZYX"];
            sym_3["UVWXYZ"] += temp_3["WVUZYX"];
            /// Symmetrization
            //  α α β α α β
            sym_3["xyZuvW"] += temp_3["uvWxyZ"];
            sym_3["xyZuvW"] -= temp_3["vuWxyZ"];
            sym_3["xyZuvW"] -= temp_3["uvWyxZ"];
            sym_3["xyZuvW"] += temp_3["vuWyxZ"];
            //  Antisymmetrization
            sym_3["uvWxyZ"] += temp_3["uvWxyZ"];
            sym_3["uvWxyZ"] -= temp_3["vuWxyZ"];
            sym_3["uvWxyZ"] -= temp_3["uvWyxZ"];
            sym_3["uvWxyZ"] += temp_3["vuWyxZ"];
            /// Symmetrization
            //  α β β α β β
            sym_3["xYZuVW"] += temp_3["uVWxYZ"];
            sym_3["xYZuVW"] -= temp_3["uWVxYZ"];
            sym_3["xYZuVW"] -= temp_3["uVWxZY"];
            sym_3["xYZuVW"] += temp_3["uWVxZY"];
            //  Antisymmetrization
            sym_3["uVWxYZ"] += temp_3["uVWxYZ"];
            sym_3["uVWxYZ"] -= temp_3["uWVxYZ"];
            sym_3["uVWxYZ"] -= temp_3["uVWxZY"];
            sym_3["uVWxYZ"] += temp_3["uWVxZY"];
        }
        as_solver_->add_sigma_kbody(state, 0, sym_1, block_factor1, b_ck.data());
        as_solver_->add_sigma_kbody(state, 0, sym_2, block_factor2, b_ck.data());
        as_solver_->add_sigma_kbody(state, 0, sym_3, block_factor3, b_ck.data());
    }

    for (const std::string& block : {"ci"}) {
        (b_ck).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = preidx[block] + i[0];
            b.at(index) = value;
        });
    }

    outfile->Printf("Done");
}

} // namespace forte