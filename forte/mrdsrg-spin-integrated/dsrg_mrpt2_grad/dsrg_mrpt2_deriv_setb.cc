/**
 * Set the constant vector b of the Linear System Ax=b.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "helpers/timer.h"

using namespace ambit;
using namespace psi;

namespace forte {

void DSRG_MRPT2::set_b(int dim, const std::map<string, int>& block_dim) {
    outfile->Printf("\n    Initializing b of the Linear System ............. ");
    b.resize(dim);
    /*-----------------------------------------------------------------------*
     |                                                                       |
     |  Adding the orbital contribution to the b of the Linear System Ax=b.  |
     |                                                                       |
     *-----------------------------------------------------------------------*/
    if (CORRELATION_TERM) {
        Z_b["ew"] -= Z["e,f1"] * F["f1,w"];
        Z_b["mw"] -= Z["mn"] * F["nw"];
        Z_b["em"] += 0.5 * sigma3_xi3["ma"] * F["ea"];
        Z_b["em"] -= 0.5 * sigma3_xi3["ie"] * F["im"];
        Z_b["ew"] += 0.5 * sigma3_xi3["wa"] * F["ea"];
        Z_b["ew"] += 0.5 * sigma3_xi3["iw"] * F["ie"];
        Z_b["ew"] -= 0.5 * sigma3_xi3["ie"] * F["iw"];
        Z_b["mw"] += 0.5 * sigma3_xi3["wa"] * F["ma"];
        Z_b["mw"] += 0.5 * sigma3_xi3["iw"] * F["im"];
        Z_b["mw"] -= 0.5 * sigma3_xi3["ma"] * F["wa"];
        if (eri_df_) {
            Z_b["em"] += 2.0 * sigma3_xi3["ia"] * B["gia"] * B["gem"];
            Z_b["em"] -= 0.5 * sigma3_xi3["ia"] * B["gim"] * B["gea"];
            Z_b["em"] -= 0.5 * sigma3_xi3["ia"] * B["gam"] * B["gei"];

            temp_z["wz"] += sigma3_xi3["ia"] * Gamma1_["zv"] * B["gai"] * B["gwv"];
            temp_z["wz"] -= 0.5 * sigma3_xi3["ia"] * Gamma1_["zv"] * B["gav"] * B["gwi"];
            temp_z["wz"] += sigma3_xi3["ia"] * Gamma1_["uz"] * B["gai"] * B["guw"];
            temp_z["wz"] -= 0.5 * sigma3_xi3["ia"] * Gamma1_["uz"] * B["gaw"] * B["gui"];

            Z_b["ew"] += sigma3_xi3["ia"] * Gamma1_["wv"] * B["gai"] * B["gev"];
            Z_b["ew"] -= 0.5 * sigma3_xi3["ia"] * Gamma1_["wv"] * B["gav"] * B["gei"];
            Z_b["ew"] += sigma3_xi3["ia"] * Gamma1_["uw"] * B["gai"] * B["gue"];
            Z_b["ew"] -= 0.5 * sigma3_xi3["ia"] * Gamma1_["uw"] * B["gae"] * B["gui"];

            Z_b["mw"] += sigma3_xi3["ia"] * Gamma1_["wv"] * B["gai"] * B["gmv"];
            Z_b["mw"] -= 0.5 * sigma3_xi3["ia"] * Gamma1_["wv"] * B["gav"] * B["gmi"];
            Z_b["mw"] += sigma3_xi3["ia"] * Gamma1_["uw"] * B["gai"] * B["gum"];
            Z_b["mw"] -= 0.5 * sigma3_xi3["ia"] * Gamma1_["uw"] * B["gam"] * B["gui"];
            Z_b["mw"] -= sigma3_xi3["ia"] * B["gai"] * B["gmw"];
            Z_b["mw"] += 0.5 * sigma3_xi3["ia"] * B["gaw"] * B["gmi"];
            Z_b["mw"] -= sigma3_xi3["ia"] * B["gai"] * B["gwm"];
            Z_b["mw"] += 0.5 * sigma3_xi3["ia"] * B["gam"] * B["gwi"];
        } else {
            Z_b["em"] += 0.5 * sigma3_xi3["ia"] * V["ieam"];
            Z_b["em"] += 0.5 * sigma3_xi3["IA"] * V["eImA"];
            Z_b["em"] += 0.5 * sigma3_xi3["ia"] * V["aeim"];
            Z_b["em"] += 0.5 * sigma3_xi3["IA"] * V["eAmI"];

            temp_z["wz"] += 0.5 * sigma3_xi3["ia"] * V["awiv"] * Gamma1_["zv"];
            temp_z["wz"] += 0.5 * sigma3_xi3["IA"] * V["wAvI"] * Gamma1_["zv"];
            temp_z["wz"] += 0.5 * sigma3_xi3["ia"] * V["auiw"] * Gamma1_["uz"];
            temp_z["wz"] += 0.5 * sigma3_xi3["IA"] * V["uAwI"] * Gamma1_["uz"];

            Z_b["ew"] += 0.5 * sigma3_xi3["ia"] * V["aeiv"] * Gamma1_["wv"];
            Z_b["ew"] += 0.5 * sigma3_xi3["IA"] * V["eAvI"] * Gamma1_["wv"];
            Z_b["ew"] += 0.5 * sigma3_xi3["ia"] * V["auie"] * Gamma1_["uw"];
            Z_b["ew"] += 0.5 * sigma3_xi3["IA"] * V["uAeI"] * Gamma1_["uw"];

            Z_b["mw"] += 0.5 * sigma3_xi3["ia"] * V["amiv"] * Gamma1_["wv"];
            Z_b["mw"] += 0.5 * sigma3_xi3["IA"] * V["mAvI"] * Gamma1_["wv"];
            Z_b["mw"] += 0.5 * sigma3_xi3["ia"] * V["auim"] * Gamma1_["uw"];
            Z_b["mw"] += 0.5 * sigma3_xi3["IA"] * V["uAmI"] * Gamma1_["uw"];
            Z_b["mw"] -= 0.5 * sigma3_xi3["ia"] * V["amiw"];
            Z_b["mw"] -= 0.5 * sigma3_xi3["IA"] * V["mAwI"];
            Z_b["mw"] -= 0.5 * sigma3_xi3["ia"] * V["awim"];
            Z_b["mw"] -= 0.5 * sigma3_xi3["IA"] * V["wAmI"];
        }
    }
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"gg"}, true);
    BlockedTensor temp3 =
        BTF_->build(CoreTensor, "Z{active-active} diagonal components", {"aa", "AA"}, true);
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal matrix{aa} for temp2 symmetrization",
                                      spin_cases({"aa"}), true);
    if (eri_df_) {
        Z_b["em"] += 2.0 * Z["m1,n1"] * B["g,n1,m1"] * B["gem"];
        Z_b["em"] -= Z["m1,n1"] * B["g,n1,m"] * B["g,e,m1"];
        Z_b["em"] += 2.0 * Z["e1,f"] * B["g,f,e1"] * B["gem"];
        Z_b["em"] -= Z["e1,f"] * B["gfm"] * B["g,e,e1"];

        temp_z["wz"] += 2.0 * Z["m1,n1"] * Gamma1_["zv"] * B["g,n1,m1"] * B["gvw"];
        temp_z["wz"] -= Z["m1,n1"] * Gamma1_["zv"] * B["g,n1,w"] * B["g,v,m1"];
        temp_z["wz"] += 2.0 * Z["e1,f1"] * Gamma1_["zv"] * B["g,f1,e1"] * B["gvw"];
        temp_z["wz"] -= Z["e1,f1"] * Gamma1_["zv"] * B["g,f1,w"] * B["g,v,e1"];
        Z_b["wz"] += temp_z["wz"];
        Z_b["zw"] -= temp_z["wz"];

        Z_b["ew"] += 2.0 * Z["m1,n1"] * Gamma1_["wv"] * B["g,m1,n1"] * B["gev"];
        Z_b["ew"] -= Z["m1,n1"] * Gamma1_["wv"] * B["g,m1,v"] * B["g,e,n1"];
        Z_b["ew"] += 2.0 * Z["e1,f1"] * Gamma1_["wv"] * B["g,e1,f1"] * B["gev"];
        Z_b["ew"] -= Z["e1,f1"] * Gamma1_["wv"] * B["g,e1,v"] * B["g,e,f1"];

        Z_b["mw"] += 2.0 * Z["m1,n1"] * Gamma1_["wv"] * B["g,n1,m1"] * B["gvm"];
        Z_b["mw"] -= Z["m1,n1"] * Gamma1_["wv"] * B["g,n1,m"] * B["g,v,m1"];
        Z_b["mw"] += 2.0 * Z["e1,f1"] * Gamma1_["wv"] * B["g,f1,e1"] * B["gvm"];
        Z_b["mw"] -= Z["e1,f1"] * Gamma1_["wv"] * B["g,f1,m"] * B["g,v,e1"];
        Z_b["mw"] -= 2.0 * Z["m1,n1"] * B["g,n1,m1"] * B["gwm"];
        Z_b["mw"] += Z["m1,n1"] * B["g,n1,m"] * B["g,w,m1"];
        Z_b["mw"] -= 2.0 * Z["e1,f"] * B["g,f,e1"] * B["gwm"];
        Z_b["mw"] += Z["e1,f"] * B["gfm"] * B["g,w,e1"];

        temp3["uv"] += Z["uv"] * I["uv"];
        temp3["UV"] += Z["UV"] * I["UV"];

        temp4["wz"] += 0.5 * Z_b["wz"];
        temp4["wz"] += 2.0 * temp3["a1,u1"] * Gamma1_["zv"] * B["g,u1,a1"] * B["gvw"];
        temp4["wz"] -= temp3["a1,u1"] * Gamma1_["zv"] * B["g,u1,w"] * B["g,v,a1"];

        temp2["em"] += Z_b["em"];
        temp2["em"] += 2.0 * temp3["uv"] * B["gvu"] * B["gem"];
        temp2["em"] -= temp3["uv"] * B["gvm"] * B["geu"];
        temp2["mw"] += Z_b["mw"];
        temp2["mw"] += temp3["wv"] * F["vm"];
        temp2["mw"] += 2.0 * temp3["u1,a1"] * Gamma1_["wv"] * B["g,a1,u1"] * B["gvm"];
        temp2["mw"] -= temp3["u1,a1"] * Gamma1_["wv"] * B["g,a1,m"] * B["g,v,u1"];
        temp2["mw"] -= 2.0 * temp3["uv"] * B["gvu"] * B["gwm"];
        temp2["mw"] += temp3["uv"] * B["gvm"] * B["gwu"];
        temp2["ew"] += Z_b["ew"];
        temp2["ew"] += temp3["wv"] * F["ve"];
        temp2["ew"] += 2.0 * temp3["u1,a1"] * Gamma1_["wv"] * B["g,u1,a1"] * B["gev"];
        temp2["ew"] -= temp3["u1,a1"] * Gamma1_["wv"] * B["g,u1,v"] * B["g,e,a1"];
        temp2["wz"] += temp4["wz"];
        temp2["zw"] -= temp4["wz"];
    } else {
        Z_b["em"] += Z["m1,n1"] * V["n1,e,m1,m"];
        Z_b["em"] += Z["M1,N1"] * V["e,N1,m,M1"];
        Z_b["em"] += Z["e1,f"] * V["f,e,e1,m"];
        Z_b["em"] += Z["E1,F"] * V["e,F,m,E1"];

        temp_z["wz"] += Z["m1,n1"] * V["n1,v,m1,w"] * Gamma1_["zv"];
        temp_z["wz"] += Z["M1,N1"] * V["v,N1,w,M1"] * Gamma1_["zv"];
        temp_z["wz"] += Z["e1,f1"] * V["f1,v,e1,w"] * Gamma1_["zv"];
        temp_z["wz"] += Z["E1,F1"] * V["v,F1,w,E1"] * Gamma1_["zv"];
        Z_b["wz"] += temp_z["wz"];
        Z_b["zw"] -= temp_z["wz"];

        Z_b["ew"] += Z["m1,n1"] * V["m1,e,n1,v"] * Gamma1_["wv"];
        Z_b["ew"] += Z["M1,N1"] * V["e,M1,v,N1"] * Gamma1_["wv"];
        Z_b["ew"] += Z["e1,f1"] * V["e1,e,f1,v"] * Gamma1_["wv"];
        Z_b["ew"] += Z["E1,F1"] * V["e,E1,v,F1"] * Gamma1_["wv"];

        Z_b["mw"] += Z["m1,n1"] * V["n1,v,m1,m"] * Gamma1_["wv"];
        Z_b["mw"] += Z["M1,N1"] * V["v,N1,m,M1"] * Gamma1_["wv"];
        Z_b["mw"] += Z["e1,f1"] * V["f1,v,e1,m"] * Gamma1_["wv"];
        Z_b["mw"] += Z["E1,F1"] * V["v,F1,m,E1"] * Gamma1_["wv"];
        Z_b["mw"] -= Z["m1,n1"] * V["n1,w,m1,m"];
        Z_b["mw"] -= Z["M1,N1"] * V["w,N1,m,M1"];
        Z_b["mw"] -= Z["e1,f"] * V["f,w,e1,m"];
        Z_b["mw"] -= Z["E1,F"] * V["w,F,m,E1"];

        temp3["uv"] += Z["uv"] * I["uv"];
        temp3["UV"] += Z["UV"] * I["UV"];

        temp4["wz"] += 0.5 * Z_b["wz"];
        temp4["wz"] += temp3["a1,u1"] * V["u1,v,a1,w"] * Gamma1_["zv"];
        temp4["wz"] += temp3["A1,U1"] * V["v,U1,w,A1"] * Gamma1_["zv"];

        temp2["em"] += Z_b["em"];
        temp2["em"] += temp3["uv"] * V["veum"];
        temp2["em"] += temp3["UV"] * V["eVmU"];
        temp2["mw"] += Z_b["mw"];
        temp2["mw"] += temp3["wv"] * F["vm"];
        temp2["mw"] += temp3["u1,a1"] * V["a1,v,u1,m"] * Gamma1_["wv"];
        temp2["mw"] += temp3["U1,A1"] * V["v,A1,m,U1"] * Gamma1_["wv"];
        temp2["mw"] -= temp3["uv"] * V["vwum"];
        temp2["mw"] -= temp3["UV"] * V["wVmU"];
        temp2["ew"] += Z_b["ew"];
        temp2["ew"] += temp3["wv"] * F["ve"];
        temp2["ew"] += temp3["u1,a1"] * V["u1,e,a1,v"] * Gamma1_["wv"];
        temp2["ew"] += temp3["U1,A1"] * V["e,U1,v,A1"] * Gamma1_["wv"];
        temp2["wz"] += temp4["wz"];
        temp2["zw"] -= temp4["wz"];
    }

    for (const std::string block : {"vc", "ca", "va", "aa"}) {
        (temp2.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block != "aa") {
                int index = preidx[block] + i[0] * block_dim.at(block) + i[1];
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
    Alpha += 2.0 * H["vu"] * Gamma1_["uv"];
    Alpha += 2.0 * V_pmqm["vu"] * Gamma1_["uv"];
    if (eri_df_) {
        Alpha += Gamma2_["uvxy"] * B["gxu"] * B["gyv"];
        Alpha += Gamma2_["uVxY"] * B["gxu"] * B["gYV"];
    } else {
        Alpha += 0.5 * Gamma2_["uvxy"] * V["xyuv"];
        Alpha += Gamma2_["uVxY"] * V["xYuV"];
    }

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"aa"}, true);
    BlockedTensor temp_alpha =
        BTF_->build(CoreTensor, "Unsymmetrized contractions in Alpha", {"aa"}, true);
    if (PT2_TERM) {
        temp_alpha["uv"] += 0.50 * T2_["vmef"] * V_["efum"];
        temp_alpha["uv"] += T2_["vmez"] * V_["ewum"] * Eta1_["zw"];
        temp_alpha["uv"] += 0.50 * T2_["vmzx"] * V_["wyum"] * Eta1_["zw"] * Eta1_["xy"];
        temp_alpha["uv"] += 0.50 * T2_["vwef"] * V_["efuz"] * Gamma1_["zw"];
        temp_alpha["uv"] += T2_["vwex"] * V_["eyuz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp_alpha["uv"] += T2_["vMeF"] * V_["eFuM"];
        temp_alpha["uv"] += T2_["vMeZ"] * V_["eWuM"] * Eta1_["ZW"];
        temp_alpha["uv"] += T2_["vMzE"] * V_["wEuM"] * Eta1_["zw"];
        temp_alpha["uv"] += T2_["vMzX"] * V_["wYuM"] * Eta1_["zw"] * Eta1_["XY"];
        temp_alpha["uv"] += T2_["vWeF"] * V_["eFuZ"] * Gamma1_["ZW"];
        temp_alpha["uv"] += T2_["vWeX"] * V_["eYuZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp_alpha["uv"] += T2_["vWxE"] * V_["yEuZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp_alpha["uv"] -= 0.50 * T2_["mnue"] * V_["vemn"];
        temp_alpha["uv"] -= 0.50 * T2_["mnuz"] * V_["vwmn"] * Eta1_["zw"];
        temp_alpha["uv"] -= T2_["mwue"] * V_["vemz"] * Gamma1_["zw"];
        temp_alpha["uv"] -= T2_["mwux"] * V_["vymz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp_alpha["uv"] -= 0.50 * T2_["wyue"] * V_["vezx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp_alpha["uv"] -= T2_["mNuE"] * V_["vEmN"];
        temp_alpha["uv"] -= T2_["mNuZ"] * V_["vWmN"] * Eta1_["ZW"];
        temp_alpha["uv"] -= T2_["mWuE"] * V_["vEmZ"] * Gamma1_["ZW"];
        temp_alpha["uv"] -= T2_["mWuX"] * V_["vYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp_alpha["uv"] -= T2_["wMuE"] * V_["vEzM"] * Gamma1_["zw"];
        temp_alpha["uv"] -= T2_["wMuX"] * V_["vYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp_alpha["uv"] -= T2_["wYuE"] * V_["vEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
    }
    if (X1_TERM) {
        temp_alpha["zw"] -= 0.25 * T2_["uvwe"] * V_["zexy"] * Lambda2_["xyuv"];
        temp_alpha["zw"] -= T2_["uVwE"] * V_["zExY"] * Lambda2_["xYuV"];
    }
    if (X2_TERM) {
        temp_alpha["zw"] += 0.25 * T2_["wmxy"] * V_["uvzm"] * Lambda2_["xyuv"];
        temp_alpha["zw"] += T2_["wMxY"] * V_["uVzM"] * Lambda2_["xYuV"];
    }
    if (X3_TERM) {
        temp_alpha["zw"] -= V_["vezx"] * T2_["wuye"] * Lambda2_["xyuv"];
        temp_alpha["zw"] -= V_["vezx"] * T2_["wUeY"] * Lambda2_["xYvU"];
        temp_alpha["zw"] -= V_["vEzX"] * T2_["wUyE"] * Lambda2_["yXvU"];
        temp_alpha["zw"] -= V_["eVzX"] * T2_["wUeY"] * Lambda2_["XYUV"];
        temp_alpha["zw"] -= V_["eVzX"] * T2_["wuye"] * Lambda2_["yXuV"];
        temp_alpha["zw"] += V_["vwmx"] * T2_["muyz"] * Lambda2_["xyuv"];
        temp_alpha["zw"] += V_["vwmx"] * T2_["mUzY"] * Lambda2_["xYvU"];
        temp_alpha["zw"] += V_["wVmX"] * T2_["mUzY"] * Lambda2_["XYUV"];
        temp_alpha["zw"] += V_["wVmX"] * T2_["muyz"] * Lambda2_["yXuV"];
        temp_alpha["zw"] += V_["wVxM"] * T2_["uMzY"] * Lambda2_["xYuV"];
    }
    if (CORRELATION_TERM) {
        if (eri_df_) {
            temp_alpha["vu"] += 2.0 * sigma3_xi3["ia"] * B["gai"] * B["guv"];
            temp_alpha["vu"] -= sigma3_xi3["ia"] * B["gav"] * B["gui"];
        } else {
            temp_alpha["vu"] += sigma3_xi3["ia"] * V["auiv"];
            temp_alpha["vu"] += sigma3_xi3["IA"] * V["uAvI"];
        }
        temp_alpha["xu"] += sigma2_xi3["ia"] * Delta1["xu"] * T2_["iuax"];
        temp_alpha["xu"] += sigma2_xi3["IA"] * Delta1["xu"] * T2_["uIxA"];
    }
    if (X7_TERM) {
        temp_alpha["uv"] += F_["ev"] * T1_["ue"];
        temp_alpha["uv"] -= F_["vm"] * T1_["mu"];
    }
    temp["uv"] += temp_alpha["uv"];
    temp["vu"] += temp_alpha["uv"];
    Alpha += temp["uv"] * Gamma1_["uv"];

    temp4 = BTF_->build(CoreTensor, "temporal tensor 4", {"aaaa", "aAaA"}, true);
    if (X1_TERM) {
        temp4["uvxy"] += 0.125 * V_["efxy"] * T2_["uvef"];
        temp4["uvxy"] += 0.25 * V_["ewxy"] * T2_["uvez"] * Eta1_["zw"];

        temp4["uVxY"] += V_["eFxY"] * T2_["uVeF"];
        temp4["uVxY"] += V_["eWxY"] * T2_["uVeZ"] * Eta1_["ZW"];
        temp4["uVxY"] += V_["wExY"] * T2_["uVzE"] * Eta1_["zw"];
    }
    if (X2_TERM) {
        temp4["uvxy"] += 0.125 * V_["uvmn"] * T2_["mnxy"];
        temp4["uvxy"] += 0.25 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["zw"];

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
        temp4["xYuV"] += F_["ex"] * T2_["uVeY"];
        temp4["yXuV"] += F_["EX"] * T2_["uVyE"];
        temp4["uvxy"] -= 0.5 * F_["vm"] * T2_["umxy"];
        temp4["xYvU"] -= F_["vm"] * T2_["mUxY"];
        temp4["xYuV"] -= F_["VM"] * T2_["uMxY"];
    }
    if (X6_TERM) {
        temp4["uvxy"] += 0.5 * T1_["ue"] * V_["evxy"];
        temp4["xYuV"] += T1_["ue"] * V_["eVxY"];
        temp4["xYvU"] += T1_["UE"] * V_["vExY"];
        temp4["uvxy"] -= 0.5 * T1_["mx"] * V_["uvmy"];
        temp4["xYuV"] -= T1_["mx"] * V_["uVmY"];
        temp4["yXuV"] -= T1_["MX"] * V_["uVyM"];
    }
    Alpha += 4 * temp4["uvxy"] * Lambda2_["xyuv"];
    Alpha += 2 * temp4["uVxY"] * Lambda2_["xYuV"];
    Alpha -= 2 * temp4["uvxy"] * Gamma2_["xyuv"];
    Alpha -= temp4["uVxY"] * Gamma2_["xYuV"];
    if (X4_TERM) {
        Alpha +=
            0.5 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") * rdms_->g3aaa()("xyzuvw");
        Alpha -=
            0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") * rdms_->g3aab()("xyZuvW");
        Alpha -=
            0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") * rdms_->g3aab()("xyZuvW");
        Alpha +=
            1.00 * V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * rdms_->g3aab()("xyZuvW");
        Alpha -=
            0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") * rdms_->g3abb()("xYZuVW");
        Alpha -=
            0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") * rdms_->g3abb()("xYZuVW");
        Alpha +=
            1.00 * V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * rdms_->g3abb()("xYZuVW");

        Alpha -= 2.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"] * Gamma2_["xyvw"];
        Alpha -= 2.0 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["uz"] * Gamma2_["xYvW"];
        Alpha += 1.0 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["uz"] * Gamma2_["XYVW"];
        Alpha += 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["uz"] * Gamma2_["xYwV"];
        Alpha -= 2.0 * V_["UVMZ"] * T2_["wMxY"] * Gamma1_["UZ"] * Gamma2_["xYwV"];
        Alpha += 1.0 * V_["vUmZ"] * T2_["mwxy"] * Gamma1_["UZ"] * Gamma2_["xyvw"];
        Alpha += 2.0 * V_["vUmZ"] * T2_["mWxY"] * Gamma1_["UZ"] * Gamma2_["xYvW"];

        Alpha -= V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"] * Gamma2_["xyuv"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["wz"] * Gamma2_["xYuV"];
        Alpha -= 2.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["WZ"] * Gamma2_["xYuV"];

        Alpha += 4.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xu"] * Gamma2_["vwzy"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mwxy"] * Gamma1_["xu"] * Gamma2_["wVyZ"];
        Alpha += 2.0 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma2_["vWzY"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma2_["VWZY"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["xu"] * Gamma2_["wVzY"];

        Alpha += 2.0 * V_["vUzM"] * T2_["MWXY"] * Gamma1_["XU"] * Gamma2_["vWzY"];
        Alpha += 2.0 * V_["UVMZ"] * T2_["wMyX"] * Gamma1_["XU"] * Gamma2_["wVyZ"];
        Alpha += 2.0 * V_["vUzM"] * T2_["wMyX"] * Gamma1_["XU"] * Gamma2_["vwzy"];
        Alpha -= 2.0 * V_["vUmZ"] * T2_["mWyX"] * Gamma1_["XU"] * Gamma2_["vWyZ"];

        Alpha += 2.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xw"] * Gamma2_["uvzy"];
        Alpha -= 2.0 * V_["uVmZ"] * T2_["mwxy"] * Gamma1_["xw"] * Gamma2_["uVyZ"];
        Alpha -= 1.0 * V_["UVMZ"] * T2_["wMxY"] * Gamma1_["xw"] * Gamma2_["UVZY"];
        Alpha += 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["xw"] * Gamma2_["uVzY"];

        Alpha -= 1.0 * V_["uvmz"] * T2_["mWyX"] * Gamma1_["XW"] * Gamma2_["uvzy"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mWyX"] * Gamma1_["XW"] * Gamma2_["uVyZ"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["XW"] * Gamma2_["uVzY"];

        Alpha += 12 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha += 6 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha -= 6 * V_["vUmZ"] * T2_["mwxy"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha -= 6 * V_["vUmZ"] * T2_["mWxY"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["YW"];

        Alpha += 6 * V_["UVMZ"] * T2_["wMyX"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["yw"];
        Alpha -= 6 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha -= 6 * V_["uVzM"] * T2_["wMyX"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["yw"];

        Alpha += 6.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["yv"];
        Alpha += 6.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["WZ"] * Gamma1_["xu"] * Gamma1_["YV"];
        Alpha += 6.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["YV"];

        Alpha -=
            0.5 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * rdms_->g3aaa()("xyzuvw");
        Alpha +=
            0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * rdms_->g3aab()("xyZuvW");
        Alpha -=
            0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * rdms_->g3aab()("xyZuvW");
        Alpha -=
            1.00 * V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * rdms_->g3aab()("xyZuvW");
        Alpha +=
            0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * rdms_->g3abb()("xYZuVW");
        Alpha -=
            0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * rdms_->g3abb()("xYZuVW");
        Alpha -=
            1.00 * V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * rdms_->g3abb()("xYZuVW");

        Alpha += 2.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["uz"] * Gamma2_["xyvw"];
        Alpha += 2.0 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["uz"] * Gamma2_["xYvW"];
        Alpha -= 1.0 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["uz"] * Gamma2_["XYVW"];
        Alpha -= 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["uz"] * Gamma2_["xYwV"];
        Alpha += 2.0 * V_["EZUV"] * T2_["xYwE"] * Gamma1_["UZ"] * Gamma2_["xYwV"];
        Alpha -= 1.0 * V_["eZvU"] * T2_["xyew"] * Gamma1_["UZ"] * Gamma2_["xyvw"];
        Alpha -= 2.0 * V_["eZvU"] * T2_["xYeW"] * Gamma1_["UZ"] * Gamma2_["xYvW"];

        Alpha += V_["ezuv"] * T2_["xyew"] * Gamma1_["wz"] * Gamma2_["xyuv"];
        Alpha += 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["wz"] * Gamma2_["xYuV"];
        Alpha += 2.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["WZ"] * Gamma2_["xYuV"];

        Alpha -= 4.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["xu"] * Gamma2_["vwzy"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["xyew"] * Gamma1_["xu"] * Gamma2_["wVyZ"];
        Alpha -= 2.0 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["xu"] * Gamma2_["vWzY"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["xu"] * Gamma2_["VWZY"];
        Alpha += 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["xu"] * Gamma2_["wVzY"];
        Alpha -= 2.0 * V_["zEvU"] * T2_["XYEW"] * Gamma1_["XU"] * Gamma2_["vWzY"];
        Alpha -= 2.0 * V_["EZUV"] * T2_["yXwE"] * Gamma1_["XU"] * Gamma2_["wVyZ"];
        Alpha -= 2.0 * V_["zEvU"] * T2_["yXwE"] * Gamma1_["XU"] * Gamma2_["vwzy"];
        Alpha += 2.0 * V_["eZvU"] * T2_["yXeW"] * Gamma1_["XU"] * Gamma2_["vWyZ"];

        Alpha -= 2.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["xw"] * Gamma2_["uvzy"];
        Alpha += 2.0 * V_["eZuV"] * T2_["xyew"] * Gamma1_["xw"] * Gamma2_["uVyZ"];
        Alpha += 1.0 * V_["EZUV"] * T2_["xYwE"] * Gamma1_["xw"] * Gamma2_["UVZY"];
        Alpha -= 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["xw"] * Gamma2_["uVzY"];
        Alpha += 1.0 * V_["ezuv"] * T2_["yXeW"] * Gamma1_["XW"] * Gamma2_["uvzy"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["yXeW"] * Gamma1_["XW"] * Gamma2_["uVyZ"];
        Alpha += 2.0 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["XW"] * Gamma2_["uVzY"];

        Alpha -= 12 * V_["ezuv"] * T2_["xyew"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha -= 6 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha += 6 * V_["eZvU"] * T2_["xyew"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha += 6 * V_["eZvU"] * T2_["xYeW"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha -= 6 * V_["EZUV"] * T2_["yXwE"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["yw"];
        Alpha += 6 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha += 6 * V_["zEuV"] * T2_["yXwE"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["yw"];

        Alpha -= 6.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["yv"];
        Alpha -= 6.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["WZ"] * Gamma1_["xu"] * Gamma1_["YV"];
        Alpha -= 6.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["YV"];
    }
    if (eri_df_) {
        Alpha += 4.0 * Z["mn"] * Gamma1_["u1,a1"] * B["gmn"] * B["g,a1,u1"];
        Alpha -= 2.0 * Z["mn"] * Gamma1_["u1,a1"] * B["g,m,u1"] * B["g,a1,n"];
        Alpha += 4.0 * Z["ef"] * Gamma1_["u1,a1"] * B["gef"] * B["g,a1,u1"];
        Alpha -= 2.0 * Z["ef"] * Gamma1_["u1,a1"] * B["g,e,u1"] * B["g,a1,f"];
        Alpha += 4.0 * temp3["uv"] * Gamma1_["u1,a1"] * B["guv"] * B["g,a1,u1"];
        Alpha -= 2.0 * temp3["uv"] * Gamma1_["u1,a1"] * B["g,u,u1"] * B["g,a1,v"];
    } else {
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
    }

    b_ck("K") += 2 * Alpha * ci("K");

    BlockedTensor temp5 = BTF_->build(CoreTensor, "temporal tensor", {"aa", "AA"}, true);
    if (eri_df_) {
        temp5["uv"] += 2.0 * Z["mn"] * B["gmn"] * B["gvu"];
        temp5["uv"] -= Z["mn"] * B["gmu"] * B["gvn"];
        temp5["uv"] += 2.0 * Z["ef"] * B["gef"] * B["gvu"];
        temp5["uv"] -= Z["ef"] * B["geu"] * B["gvf"];
    } else {
        temp5["uv"] += Z["mn"] * V["mvnu"];
        temp5["uv"] += Z["MN"] * V["vMuN"];
        temp5["uv"] += Z["ef"] * V["evfu"];
        temp5["uv"] += Z["EF"] * V["vEuF"];
    }

    /// Call the generalized sigma function to complete contractions
    for (const auto& pair : as_solver_->state_space_size_map()) {
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

        auto sym_1 = BTF_->build(CoreTensor, "symmetrized 1-body tensor", spin_cases({"aa"}), true);
        auto sym_2 =
            BTF_->build(CoreTensor, "symmetrized 2-body tensor", spin_cases({"aaaa"}), true);
        auto sym_3 =
            BTF_->build(CoreTensor, "symmetrized 3-body tensor", spin_cases({"aaaaaa"}), true);
        {
            auto temp_1 = BTF_->build(CoreTensor, "1-body intermediate tensor", {"aa"}, true);

            if (X4_TERM) {
                // -0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") *
                // dlamb3_aaa("Kxyzuvw")
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

                // 0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") *
                // dlamb3_aab("KxyZuvW")
                temp_1["xu"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["yZvW"];
                temp_1["vy"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["yZuW"];
                temp_1["uy"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["xZvW"];
                temp_1["vx"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ux"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xv"] * Gamma1_["ZW"];

                // 0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") *
                // dlamb3_aab("KxyZuvW")
                temp_1["xu"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["yZvW"];
                temp_1["vy"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["yZuW"];
                temp_1["uy"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["xZvW"];
                temp_1["vx"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ux"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["uWmZ"] * T2_["mvxy"] * Gamma1_["xv"] * Gamma1_["ZW"];

                // - V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * dlamb3_aab("KxyZuvW")
                temp_1["xu"] += V_["uWyM"] * T2_["vMxZ"] * Lambda2_["yZvW"];
                temp_1["vy"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] -= V_["uWyM"] * T2_["vMxZ"] * Lambda2_["yZuW"];
                temp_1["uy"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["yu"] -= V_["uWyM"] * T2_["vMxZ"] * Lambda2_["xZvW"];
                temp_1["vx"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yv"] += V_["uWyM"] * T2_["vMxZ"] * Lambda2_["xZuW"];
                temp_1["ux"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ux"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] += V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] -= V_["uWyM"] * T2_["vMxZ"] * Gamma1_["xv"] * Gamma1_["ZW"];

                // 0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") *
                // dlamb3_abb("KxYZuVW")
                temp_1["xu"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Lambda2_["YZVW"];
                temp_1["ux"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ux"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ux"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ux"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["xu"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["xu"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZV"] * Gamma1_["YW"];

                // 0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") *
                // dlamb3_abb("KxYZuVW")
                temp_1["xu"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Lambda2_["YZVW"];
                temp_1["ux"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ux"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ux"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ux"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["xu"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["xu"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZV"] * Gamma1_["YW"];

                // - V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * dlamb3_abb("KxYZuVW")
                temp_1["xu"] += V_["uVmZ"] * T2_["mWxY"] * Lambda2_["YZVW"];
                temp_1["ux"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ux"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ux"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ux"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["xu"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["xu"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZV"] * Gamma1_["YW"];

                // 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") *
                // dlamb3_aaa("Kxyzuvw")
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

                // -0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") *
                // dlamb3_aab("KxyZuvW")
                temp_1["xu"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["yZvW"];
                temp_1["vy"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["yZuW"];
                temp_1["uy"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["yu"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["xZvW"];
                temp_1["vx"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yv"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Lambda2_["xZuW"];
                temp_1["ux"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ux"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xv"] * Gamma1_["ZW"];

                // 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") *
                // dlamb3_aab("KxyZuvW")
                temp_1["xu"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["yZvW"];
                temp_1["vy"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["yZuW"];
                temp_1["uy"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["xZvW"];
                temp_1["vx"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Lambda2_["xZuW"];
                temp_1["ux"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ux"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] -= 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] += 0.50 * V_["vexy"] * T2_["uWeZ"] * Gamma1_["xv"] * Gamma1_["ZW"];

                // V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * dlamb3_aab("KxyZuvW")
                temp_1["xu"] -= V_["vExZ"] * T2_["uWyE"] * Lambda2_["yZvW"];
                temp_1["vy"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] += V_["vExZ"] * T2_["uWyE"] * Lambda2_["yZuW"];
                temp_1["uy"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["xv"] * Gamma1_["ZW"];
                temp_1["yu"] += V_["vExZ"] * T2_["uWyE"] * Lambda2_["xZvW"];
                temp_1["vx"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yv"] -= V_["vExZ"] * T2_["uWyE"] * Lambda2_["xZuW"];
                temp_1["ux"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["ux"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"] * Gamma1_["yv"];
                temp_1["vy"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"] * Gamma1_["xu"];
                temp_1["vx"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"] * Gamma1_["yu"];
                temp_1["uy"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"] * Gamma1_["xv"];
                temp_1["xu"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["yv"] * Gamma1_["ZW"];
                temp_1["yv"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["xu"] * Gamma1_["ZW"];
                temp_1["xv"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["yu"] * Gamma1_["ZW"];
                temp_1["yu"] += V_["vExZ"] * T2_["uWyE"] * Gamma1_["xv"] * Gamma1_["ZW"];

                // -0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") *
                // dlamb3_abb("KxYZuVW")
                temp_1["xu"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Lambda2_["YZVW"];
                temp_1["ux"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ux"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ux"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ux"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["xu"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["xu"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZV"] * Gamma1_["YW"];

                // 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") *
                // dlamb3_abb("KxYZuVW")
                temp_1["xu"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Lambda2_["YZVW"];
                temp_1["ux"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ux"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ux"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ux"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["xu"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["xu"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZV"] * Gamma1_["YW"];

                // V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * dlamb3_abb("KxYZuVW")
                temp_1["xu"] -= V_["eWxY"] * T2_["uVeZ"] * Lambda2_["YZVW"];
                temp_1["ux"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["ux"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YW"] * Gamma1_["ZV"];
                temp_1["ux"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
                temp_1["ux"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZW"] * Gamma1_["YV"];
                temp_1["xu"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YV"] * Gamma1_["ZW"];
                temp_1["xu"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZV"] * Gamma1_["YW"];
            }

            // - H.block("aa")("vu") * cc1a("Kuv")
            // - H.block("AA")("VU") * cc1b("KUV")
            temp_1["uv"] -= H["vu"];

            // - V_sumA_Alpha.block("aa")("vu") * cc1a("Kuv")
            // - V_sumB_Alpha.block("aa")("vu") * cc1a("Kuv")
            // - V_sumB_Beta.block("AA")("VU") * cc1b("KUV")
            // - V_sumA_Beta.block("AA")("VU") * cc1b("KUV")
            temp_1["uv"] -= V_pmqm["vu"];

            // - 0.5 * temp.block("aa")("uv") * cc1a("Kuv")
            // - 0.5 * temp.block("AA")("UV") * cc1b("KUV")
            temp_1["uv"] -= 0.5 * temp["vu"];

            /// temp4 * dlambda2
            //      temp4.block("aaaa")("uvxy") * dlambda2_aaaa("Kxyuv")
            //      temp4.block("AAAA")("UVXY") * dlambda2_bbbb("KXYUV")
            //      temp4.block("aAaA")("uVxY") * dlambda2_abab("KxYuV")
            temp_1["ux"] += temp4["uvxy"] * Gamma1_["yv"];
            temp_1["vy"] += temp4["uvxy"] * Gamma1_["xu"];
            temp_1["vx"] -= temp4["uvxy"] * Gamma1_["yu"];
            temp_1["uy"] -= temp4["uvxy"] * Gamma1_["xv"];
            temp_1["ux"] += temp4["uVxY"] * Gamma1_["YV"];

            // - temp3.block("aa")("xy") * V.block("aaaa")("xvyu") * cc1a("Kuv")
            // - temp3.block("aa")("xy") * V.block("aAaA")("xVyU") * cc1b("KUV")
            // - temp3.block("AA")("XY") * V.block("AAAA")("XVYU") * cc1b("KUV")
            // - temp3.block("AA")("XY") * V.block("aAaA")("vXuY") * cc1a("Kuv")
            if (eri_df_) {
                temp_1["uv"] -= 2.0 * temp3["xy"] * B["gxy"] * B["gvu"];
                temp_1["uv"] += temp3["xy"] * B["gxu"] * B["gvy"];
            } else {
                temp_1["uv"] -= temp3["xy"] * V["xvyu"];
                temp_1["uv"] -= temp3["XY"] * V["vXuY"];
            }

            // - temp5.block("aa")("uv") * cc1a("Kuv")
            // - temp5.block("AA")("UV") * cc1b("KUV")
            temp_1["uv"] -= temp5["uv"];

            /// Symmetrization
            //  α α
            sym_1["uv"] += temp_1["uv"];
            sym_1["uv"] += temp_1["vu"];
            //  β β
            sym_1.block("AA")("pq") = sym_1.block("aa")("pq");
        }
        {
            auto temp_2 =
                BTF_->build(CoreTensor, "2-body intermediate tensor", {"aaaa", "aAaA"}, true);

            if (X4_TERM) {
                // -0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") *
                // dlamb3_aaa("Kxyzuvw")
                temp_2["xyvw"] += 0.50 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"];
                temp_2["xyuv"] += 0.25 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"];
                temp_2["vwzy"] -= 1.00 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xu"];
                temp_2["uvzy"] -= 0.50 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xw"];

                // 0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") *
                // dlamb3_aab("KxyZuvW")
                temp_2["yZvW"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xu"];
                temp_2["yZuW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["xv"];
                temp_2["xZvW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yu"];
                temp_2["xZuW"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["yv"];
                temp_2["xyuv"] -= 0.50 * V_["uvmy"] * T2_["mWxZ"] * Gamma1_["ZW"];

                // 0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") *
                // dlamb3_aab("KxyZuvW")
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

                // 0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") *
                // dlamb3_abb("KxYZuVW")
                temp_2["xZuW"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YV"];
                temp_2["xZuV"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["YW"];
                temp_2["xYuW"] += 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZV"];
                temp_2["xYuV"] -= 0.50 * V_["VWMZ"] * T2_["uMxY"] * Gamma1_["ZW"];

                // 0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") *
                // dlamb3_abb("KxYZuVW")
                temp_2["xZuW"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YV"];
                temp_2["xZuV"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["YW"];
                temp_2["xYuW"] += 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZV"];
                temp_2["xYuV"] -= 0.50 * V_["uVxM"] * T2_["MWYZ"] * Gamma1_["ZW"];

                // - V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * dlamb3_abb("KxYZuVW")
                temp_2["xZuW"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YV"];
                temp_2["xZuV"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["YW"];
                temp_2["xYuW"] -= V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZV"];
                temp_2["xYuV"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZW"];

                // 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") *
                // dlamb3_aaa("Kxyzuvw")
                temp_2["xyvw"] -= 0.50 * V_["ewxy"] * T2_["uvez"] * Gamma1_["uz"];
                temp_2["xyuv"] -= 0.25 * V_["ewxy"] * T2_["uvez"] * Gamma1_["wz"];
                temp_2["vwzy"] += 1.00 * V_["ewxy"] * T2_["uvez"] * Gamma1_["xu"];
                temp_2["uvzy"] += 0.50 * V_["ewxy"] * T2_["uvez"] * Gamma1_["xw"];

                // -0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") *
                // dlamb3_aab("KxyZuvW")
                temp_2["yZvW"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xu"];
                temp_2["yZuW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["xv"];
                temp_2["xZvW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yu"];
                temp_2["xZuW"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["yv"];
                temp_2["xyuv"] += 0.50 * V_["eWxZ"] * T2_["uvey"] * Gamma1_["ZW"];

                // 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") *
                // dlamb3_aab("KxyZuvW")
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

                // -0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") *
                // dlamb3_abb("KxYZuVW")
                temp_2["xZuW"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YV"];
                temp_2["xZuV"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["YW"];
                temp_2["xYuW"] -= 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZV"];
                temp_2["xYuV"] += 0.50 * V_["uExY"] * T2_["VWEZ"] * Gamma1_["ZW"];

                // 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") *
                // dlamb3_abb("KxYZuVW")
                temp_2["xZuW"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YV"];
                temp_2["xZuV"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["YW"];
                temp_2["xYuW"] += 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZV"];
                temp_2["xYuV"] -= 0.50 * V_["WEYZ"] * T2_["uVxE"] * Gamma1_["ZW"];

                // V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * dlamb3_abb("KxYZuVW")
                temp_2["xZuW"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YV"];
                temp_2["xZuV"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["YW"];
                temp_2["xYuW"] += V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZV"];
                temp_2["xYuV"] -= V_["eWxY"] * T2_["uVeZ"] * Gamma1_["ZW"];
            }

            /// CI contribution
            //  -0.25 * V.block("aaaa")("xyuv") * cc2aa("Kuvxy")
            //  -0.25 * V.block("AAAA")("XYUV") * cc2bb("KUVXY")
            //  - V.block("aAaA")("xYuV") * cc2ab("KuVxY")
            if (eri_df_) {
                temp_2["uvxy"] -= 0.5 * B["gxu"] * B["gyv"];
                temp_2["uVxY"] -= B["gxu"] * B["gYV"];
            } else {
                temp_2["uvxy"] -= 0.25 * V["xyuv"];
                temp_2["uVxY"] -= V["xYuV"];
            }

            /// temp4 * dlambda2
            //      temp4.block("aaaa")("uvxy") * dlambda2_aaaa("Kxyuv")
            //      temp4.block("AAAA")("UVXY") * dlambda2_bbbb("KXYUV")
            //      temp4.block("aAaA")("uVxY") * dlambda2_abab("KxYuV")
            temp_2["uvxy"] -= temp4["uvxy"];
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
            //  β β β β
            sym_2.block("AAAA")("pqrs") = sym_2.block("aaaa")("pqrs");
            /// Symmetrization
            //  α β α β
            sym_2["uVxY"] += temp_2["uVxY"];
            sym_2["uVxY"] += temp_2["xYuV"];
        }
        {
            auto temp_3 =
                BTF_->build(CoreTensor, "3-body intermediate tensor", spin_cases({"aaaaaa"}), true);
            if (X4_TERM) {

                // -0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") *
                // dlamb3_aaa("Kxyzuvw")
                temp_3["xyzuvw"] -= 0.25 * V_["uvmz"] * T2_["mwxy"];

                // 0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") *
                // dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] += 0.50 * V_["uvmy"] * T2_["mWxZ"];

                // 0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") *
                // dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] += 0.50 * V_["uWmZ"] * T2_["mvxy"];

                // - V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] -= V_["uWyM"] * T2_["vMxZ"];

                // 0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") *
                // dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] += 0.50 * V_["VWMZ"] * T2_["uMxY"];

                // 0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") *
                // dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] += 0.50 * V_["uVxM"] * T2_["MWYZ"];

                // - V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] -= V_["uVmZ"] * T2_["mWxY"];

                // 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") *
                // dlamb3_aaa("Kxyzuvw")
                temp_3["xyzuvw"] += 0.25 * V_["ewxy"] * T2_["uvez"];

                // -0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") *
                // dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] -= 0.50 * V_["eWxZ"] * T2_["uvey"];

                // 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") *
                // dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] += 0.50 * V_["vexy"] * T2_["uWeZ"];

                // V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * dlamb3_aab("KxyZuvW")
                temp_3["xyZuvW"] += V_["vExZ"] * T2_["uWyE"];

                // -0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") *
                // dlamb3_abb("KxYZuVW")
                temp_3["xYZuVW"] -= 0.50 * V_["uExY"] * T2_["VWEZ"];

                // 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") *
                // dlamb3_abb("KxYZuVW")
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
            //  β β β β β β
            sym_3.block("AAAAAA")("pqrsto") = sym_3.block("aaaaaa")("pqrsto");
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

    for (const auto block : {"ci"}) {
        (b_ck).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = preidx[block] + i[0];
            b.at(index) = value;
        });
    }

    outfile->Printf("Done");
}

} // namespace forte
