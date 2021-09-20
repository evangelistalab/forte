/**
 * Solve the z-vector equations.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "../dsrg_mrpt2.h"
#include "helpers/timer.h"
#include "psi4/libpsi4util/process.h"

using namespace ambit;
using namespace psi;

namespace forte {

void DSRG_MRPT2::set_z() {
    Z = BTF_->build(CoreTensor, "Z Matrix", spin_cases({"gg"}));
    outfile->Printf("\n    Initializing Diagonal Entries of the OPDM Z ..... ");
    set_z_cc();
    set_z_vv();
    set_z_aa_diag();
    outfile->Printf("Done");
    // NOTICE: LAPACK solver (Deprecated in the future)
    solve_z();
    // iterative solver
    // solve_linear_iter();
}

void DSRG_MRPT2::set_w() {
    outfile->Printf("\n    Solving Entries of the EWDM W.................... ");
    W = BTF_->build(CoreTensor, "Energy weighted density matrix(Lagrangian)", spin_cases({"gg"}));
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));

    // NOTICE: w for {virtual-general}
    if (CORRELATION_TERM) {
        W["pe"] += 0.5 * Sigma3["ie"] * F["ip"];
    }

    if (CORRELATION_TERM) {
        W["pe"] += 0.5 * Xi3["ie"] * F["ip"];
    }

    if (CORRELATION_TERM) {
        W["pe"] += Tau1["ijeb"] * V["pbij"];
        W["pe"] += 2.0 * Tau1["iJeB"] * V["pBiJ"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        W["pe"] += temp["kled"] * V["pdkl"];
        W["pe"] += 2.0 * temp["kLeD"] * V["pDkL"];
        temp.zero();
    }
    W["pe"] += Z["e,m1"] * F["m1,p"];
    W["pe"] += Z["eu"] * H["vp"] * Gamma1_["uv"];
    W["pe"] += Z["eu"] * V_sumA_Alpha["v,p"] * Gamma1_["uv"];
    W["pe"] += Z["eu"] * V_sumB_Alpha["v,p"] * Gamma1_["uv"];
    W["pe"] += 0.5 * Z["eu"] * V["xypv"] * Gamma2_["uvxy"];
    W["pe"] += Z["eu"] * V["xYpV"] * Gamma2_["uVxY"];
    W["pe"] += Z["e,f1"] * F["f1,p"];
    W["ei"] = W["ie"];

    // NOTICE: w for {core-hole}
    if (CORRELATION_TERM) {
        W["jm"] += 0.5 * Sigma3["ma"] * F["ja"];
        W["jm"] += 0.5 * Sigma3["ia"] * V["amij"];
        W["jm"] += 0.5 * Sigma3["IA"] * V["mAjI"];
        W["jm"] += 0.5 * Sigma3["ia"] * V["ajim"];
        W["jm"] += 0.5 * Sigma3["IA"] * V["jAmI"];
    }

    if (CORRELATION_TERM) {
        W["jm"] += 0.5 * Xi3["ma"] * F["ja"];
        W["jm"] += 0.5 * Xi3["ia"] * V["amij"];
        W["jm"] += 0.5 * Xi3["IA"] * V["mAjI"];
        W["jm"] += 0.5 * Xi3["ia"] * V["ajim"];
        W["jm"] += 0.5 * Xi3["IA"] * V["jAmI"];
    }

    if (CORRELATION_TERM) {
        W["im"] += Tau1["mjab"] * V["abij"];
        W["im"] += 2.0 * Tau1["mJaB"] * V["aBiJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        W["im"] += temp["mlcd"] * V["cdil"];
        W["im"] += 2.0 * temp["mLcD"] * V["cDiL"];
        temp.zero();
    }
    W["im"] += Z["e1,m"] * F["i,e1"];
    W["im"] += Z["e1,m1"] * V["e1,i,m1,m"];
    W["im"] += Z["E1,M1"] * V["i,E1,m,M1"];
    W["im"] += Z["e1,m1"] * V["e1,m,m1,i"];
    W["im"] += Z["E1,M1"] * V["m,E1,i,M1"];
    W["im"] += Z["mu"] * F["ui"];
    W["im"] -= Z["mu"] * H["vi"] * Gamma1_["uv"];
    W["im"] -= Z["mu"] * V_sumA_Alpha["vi"] * Gamma1_["uv"];
    W["im"] -= Z["mu"] * V_sumB_Alpha["vi"] * Gamma1_["uv"];
    W["im"] -= 0.5 * Z["mu"] * V["xyiv"] * Gamma2_["uvxy"];
    W["im"] -= Z["mu"] * V["xYiV"] * Gamma2_["uVxY"];
    W["im"] += Z["n1,u"] * V["u,i,n1,m"];
    W["im"] += Z["N1,U"] * V["i,U,m,N1"];
    W["im"] += Z["n1,u"] * V["u,m,n1,i"];
    W["im"] += Z["N1,U"] * V["m,U,i,N1"];
    W["im"] -= Z["n1,u"] * Gamma1_["uv"] * V["v,i,n1,m"];
    W["im"] -= Z["N1,U"] * Gamma1_["UV"] * V["i,V,m,N1"];
    W["im"] -= Z["n1,u"] * Gamma1_["uv"] * V["v,m,n1,i"];
    W["im"] -= Z["N1,U"] * Gamma1_["UV"] * V["m,V,i,N1"];
    W["im"] += Z["e1,u"] * Gamma1_["uv"] * V["e1,i,v,m"];
    W["im"] += Z["E1,U"] * Gamma1_["UV"] * V["i,E1,m,V"];
    W["im"] += Z["e1,u"] * Gamma1_["uv"] * V["e1,m,v,i"];
    W["im"] += Z["E1,U"] * Gamma1_["UV"] * V["m,E1,i,V"];
    W["im"] += Z["m,n1"] * F["n1,i"];
    W["im"] += Z["m1,n1"] * V["n1,i,m1,m"];
    W["im"] += Z["M1,N1"] * V["i,N1,m,M1"];
    W["im"] += Z["uv"] * V["vium"];
    W["im"] += Z["UV"] * V["iVmU"];
    W["im"] += Z["e1,f"] * V["f,i,e1,m"];
    W["im"] += Z["E1,F"] * V["i,F,m,E1"];

    // CI contribution
    W.block("cc")("nm") += 0.5 * V.block("acac")("umvn") * x_ci("I") * cc1a("Iuv");
    W.block("cc")("nm") += 0.5 * V.block("cAcA")("mUnV") * x_ci("I") * cc1b("IUV");

    W.block("ac")("xm") += 0.5 * V.block("acaa")("umvx") * x_ci("I") * cc1a("Iuv");
    W.block("ac")("xm") += 0.5 * V.block("cAaA")("mUxV") * x_ci("I") * cc1b("IUV");
    W["mu"] = W["um"];

    // NOTICE: w for {active-active}
    if (CORRELATION_TERM) {
        W["zw"] += 0.5 * Sigma3["wa"] * F["za"];
        W["zw"] += 0.5 * Sigma3["iw"] * F["iz"];
        W["zw"] += 0.5 * Sigma3["ia"] * V["aziv"] * Gamma1_["wv"];
        W["zw"] += 0.5 * Sigma3["IA"] * V["zAvI"] * Gamma1_["wv"];
        W["zw"] += 0.5 * Sigma3["ia"] * V["auiz"] * Gamma1_["uw"];
        W["zw"] += 0.5 * Sigma3["IA"] * V["uAzI"] * Gamma1_["uw"];
    }

    if (CORRELATION_TERM) {
        W["zw"] += 0.5 * Xi3["wa"] * F["za"];
        W["zw"] += 0.5 * Xi3["iw"] * F["iz"];
        W["zw"] += 0.5 * Xi3["ia"] * V["aziv"] * Gamma1_["wv"];
        W["zw"] += 0.5 * Xi3["IA"] * V["zAvI"] * Gamma1_["wv"];
        W["zw"] += 0.5 * Xi3["ia"] * V["auiz"] * Gamma1_["uw"];
        W["zw"] += 0.5 * Xi3["IA"] * V["uAzI"] * Gamma1_["uw"];
    }

    if (CORRELATION_TERM) {
        W["zw"] += Tau1["ijwb"] * V["zbij"];
        W["zw"] += 2.0 * Tau1["iJwB"] * V["zBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        W["zw"] += temp["klwd"] * V["zdkl"];
        W["zw"] += 2.0 * temp["kLwD"] * V["zDkL"];
        temp.zero();

        W["zw"] += Tau1["wjab"] * V["abzj"];
        W["zw"] += 2.0 * Tau1["wJaB"] * V["aBzJ"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        W["zw"] += temp["wlcd"] * V["cdzl"];
        W["zw"] += 2.0 * temp["wLcD"] * V["cDzL"];
        temp.zero();
    }
    W["zw"] += Z["e1,m1"] * V["e1,u,m1,z"] * Gamma1_["uw"];
    W["zw"] += Z["E1,M1"] * V["u,E1,z,M1"] * Gamma1_["uw"];
    W["zw"] += Z["e1,m1"] * V["e1,z,m1,u"] * Gamma1_["uw"];
    W["zw"] += Z["E1,M1"] * V["z,E1,u,M1"] * Gamma1_["uw"];
    W["zw"] += Z["n1,w"] * F["z,n1"];
    W["zw"] += Z["n1,u"] * V["u,v,n1,z"] * Gamma1_["wv"];
    W["zw"] += Z["N1,U"] * V["v,U,z,N1"] * Gamma1_["wv"];
    W["zw"] += Z["n1,u"] * V["u,z,n1,v"] * Gamma1_["wv"];
    W["zw"] += Z["N1,U"] * V["z,U,v,N1"] * Gamma1_["wv"];
    W["zw"] -= Z["n1,u"] * H["z,n1"] * Gamma1_["uw"];
    W["zw"] -= Z["n1,u"] * V_sumA_Alpha["z,n1"] * Gamma1_["uw"];
    W["zw"] -= Z["n1,u"] * V_sumB_Alpha["z,n1"] * Gamma1_["uw"];
    W["zw"] -= 0.5 * Z["n1,u"] * V["x,y,n1,z"] * Gamma2_["u,w,x,y"];
    W["zw"] -= Z["N1,U"] * V["y,X,z,N1"] * Gamma2_["w,U,y,X"];
    W["zw"] -= Z["n1,u"] * V["z,y,n1,v"] * Gamma2_["u,v,w,y"];
    W["zw"] -= Z["n1,u"] * V["z,Y,n1,V"] * Gamma2_["u,V,w,Y"];
    W["zw"] -= Z["N1,U"] * V["z,Y,v,N1"] * Gamma2_["v,U,w,Y"];
    W["zw"] += Z["e1,u"] * H["z,e1"] * Gamma1_["uw"];
    W["zw"] += Z["e1,u"] * V_sumA_Alpha["z,e1"] * Gamma1_["uw"];
    W["zw"] += Z["e1,u"] * V_sumB_Alpha["z,e1"] * Gamma1_["uw"];
    W["zw"] += 0.5 * Z["e1,u"] * V["e1,z,x,y"] * Gamma2_["u,w,x,y"];
    W["zw"] += Z["E1,U"] * V["z,E1,y,X"] * Gamma2_["w,U,y,X"];
    W["zw"] += Z["e1,u"] * V["e1,v,z,y"] * Gamma2_["u,v,w,y"];
    W["zw"] += Z["e1,u"] * V["e1,V,z,Y"] * Gamma2_["u,V,w,Y"];
    W["zw"] += Z["E1,U"] * V["v,E1,z,Y"] * Gamma2_["v,U,w,Y"];
    W["zw"] += Z["m1,n1"] * V["n1,v,m1,z"] * Gamma1_["wv"];
    W["zw"] += Z["M1,N1"] * V["v,N1,z,M1"] * Gamma1_["wv"];
    W["zw"] += Z["e1,f1"] * V["f1,v,e1,z"] * Gamma1_["wv"];
    W["zw"] += Z["E1,F1"] * V["v,F1,z,E1"] * Gamma1_["wv"];
    W["zw"] += Z["u1,a1"] * V["a1,v,u1,z"] * Gamma1_["wv"];
    W["zw"] += Z["U1,A1"] * V["v,A1,z,U1"] * Gamma1_["wv"];
    W["zw"] += Z["wv"] * F["vz"];

    W.block("aa")("zw") += 0.50 * x_ci("I") * H.block("aa")("vz") * cc1a("Iwv");

    W.block("aa")("zw") += 0.25 * V_sumA_Alpha.block("aa")("uz") * cc1a("Iuw") * x_ci("I");
    W.block("aa")("zw") += 0.25 * V_sumB_Alpha.block("aa")("uz") * cc1a("Iuw") * x_ci("I");
    W.block("aa")("zw") += 0.25 * V_sumA_Alpha.block("aa")("zv") * cc1a("Iwv") * x_ci("I");
    W.block("aa")("zw") += 0.25 * V_sumB_Alpha.block("aa")("zv") * cc1a("Iwv") * x_ci("I");

    W.block("aa")("zw") += 0.125 * x_ci("I") * V.block("aaaa")("zvxy") * cc2aa("Iwvxy");
    W.block("aa")("zw") += 0.250 * x_ci("I") * V.block("aAaA")("zVxY") * cc2ab("IwVxY");
    W.block("aa")("zw") += 0.125 * x_ci("I") * V.block("aaaa")("uzxy") * cc2aa("Iuwxy");
    W.block("aa")("zw") += 0.250 * x_ci("I") * V.block("aAaA")("zUxY") * cc2ab("IwUxY");

    // CASSCF reference
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"gg"}));

    W["mp"] += F["mp"];
    temp1["vp"] = H["vp"];
    temp1["vp"] += V_sumA_Alpha["vp"];
    temp1["vp"] += V_sumB_Alpha["vp"];
    W["up"] += temp1["vp"] * Gamma1_["uv"];
    W["up"] += 0.5 * V["xypv"] * Gamma2_["uvxy"];
    W["up"] += V["xYpV"] * Gamma2_["uVxY"];

    // Copy alpha-alpha to beta-beta
    (W.block("CC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("cc").data()[i[0] * ncore + i[1]];
    });
    (W.block("AA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("aa").data()[i[0] * na + i[1]];
    });
    (W.block("VV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("vv").data()[i[0] * nvirt + i[1]];
    });
    (W.block("CV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("cv").data()[i[0] * nvirt + i[1]];
    });
    (W.block("VC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("vc").data()[i[0] * ncore + i[1]];
    });
    (W.block("CA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("ca").data()[i[0] * na + i[1]];
    });
    (W.block("AC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("ac").data()[i[0] * ncore + i[1]];
    });
    (W.block("AV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("av").data()[i[0] * nvirt + i[1]];
    });
    (W.block("VA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("va").data()[i[0] * na + i[1]];
    });
    outfile->Printf("Done");
}

void DSRG_MRPT2::set_z_cc() {
    BlockedTensor val1 = BTF_->build(CoreTensor, "val1", {"c"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // core-core diagonal entries
    if (CORRELATION_TERM) {
        val1["m"] += -2 * s_ * Sigma1["ma"] * F["ma"];
        val1["m"] += -2 * s_ * DelGam1["xu"] * T2_["muax"] * Sigma1["ma"];
        val1["m"] += -2 * s_ * DelGam1["XU"] * T2_["mUaX"] * Sigma1["ma"];
    }
    if (CORRELATION_TERM) {
        val1["m"] -= Xi1["ma"] * F["ma"];
        val1["m"] += 2 * s_ * Xi2["ma"] * F["ma"];
        val1["m"] -= Xi1["ma"] * T2_["muax"] * DelGam1["xu"];
        val1["m"] -= Xi1["ma"] * T2_["mUaX"] * DelGam1["XU"];
        val1["m"] += 2 * s_ * Xi2["ma"] * T2_["muax"] * DelGam1["xu"];
        val1["m"] += 2 * s_ * Xi2["ma"] * T2_["mUaX"] * DelGam1["XU"];
    }
    if (CORRELATION_TERM) {
        temp["mjab"] += V["abmj"] * Eeps2["mjab"];
        temp["mJaB"] += V["aBmJ"] * Eeps2["mJaB"];
        val1["m"] += 4.0 * s_ * Tau2["mjab"] * temp["mjab"];
        val1["m"] += 8.0 * s_ * Tau2["mJaB"] * temp["mJaB"];
        temp.zero();

        val1["m"] -= 2.0 * T2OverDelta["mjab"] * Tau2["mjab"];
        val1["m"] -= 4.0 * T2OverDelta["mJaB"] * Tau2["mJaB"];

        temp["mlcd"] += V["cdml"] * Eeps2["mlcd"];
        temp["mLcD"] += V["cDmL"] * Eeps2["mLcD"];
        temp_1["mlcd"] += Kappa["mlcd"] * Delta2["mlcd"];
        temp_1["mLcD"] += Kappa["mLcD"] * Delta2["mLcD"];
        val1["m"] -= 4.0 * s_ * temp["mlcd"] * temp_1["mlcd"];
        val1["m"] -= 8.0 * s_ * temp["mLcD"] * temp_1["mLcD"];
        temp.zero();
        temp_1.zero();
    }
    BlockedTensor zmn = BTF_->build(CoreTensor, "z{mn} normal", {"cc"});
    // core-core block entries within normal conditions
    if (CORRELATION_TERM) {
        zmn["mn"] += 0.5 * Sigma3["na"] * F["ma"];
        zmn["mn"] -= 0.5 * Sigma3["ma"] * F["na"];
    }
    if (CORRELATION_TERM) {
        zmn["mn"] += 0.5 * Xi3["na"] * F["ma"];
        zmn["mn"] -= 0.5 * Xi3["ma"] * F["na"];
    }
    if (CORRELATION_TERM) {
        zmn["mn"] += Tau1["njab"] * V["abmj"];
        zmn["mn"] += 2.0 * Tau1["nJaB"] * V["aBmJ"];

        temp["nlcd"] += Kappa["nlcd"] * Eeps2_p["nlcd"];
        temp["nLcD"] += Kappa["nLcD"] * Eeps2_p["nLcD"];
        zmn["mn"] += temp["nlcd"] * V["cdml"];
        zmn["mn"] += 2.0 * temp["nLcD"] * V["cDmL"];
        temp.zero();

        zmn["mn"] -= Tau1["mjab"] * V["abnj"];
        zmn["mn"] -= 2.0 * Tau1["mJaB"] * V["aBnJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        zmn["mn"] -= temp["mlcd"] * V["cdnl"];
        zmn["mn"] -= 2.0 * temp["mLcD"] * V["cDnL"];
        temp.zero();
    }

    for (const std::string& block : {"cc", "CC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                value = val1.block("c").data()[i[0]];
            } else {
                auto dmt = Delta1.block("cc").data()[i[1] * ncore + i[0]];
                if (std::fabs(dmt) > 1e-12) {
                    value = zmn.block("cc").data()[i[0] * ncore + i[1]] / dmt;
                }
            }
        });
    }
}

void DSRG_MRPT2::set_z_vv() {
    BlockedTensor val2 = BTF_->build(CoreTensor, "val2", {"v"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // virtual-virtual diagonal entries
    if (CORRELATION_TERM) {
        val2["e"] += 2 * s_ * Sigma1["ie"] * F["ie"];
        val2["e"] += 2 * s_ * DelGam1["xu"] * T2_["iuex"] * Sigma1["ie"];
        val2["e"] += 2 * s_ * DelGam1["XU"] * T2_["iUeX"] * Sigma1["ie"];
    }

    if (CORRELATION_TERM) {
        val2["e"] += Xi1["ie"] * F["ie"];
        val2["e"] -= 2 * s_ * Xi2["ie"] * F["ie"];
        val2["e"] += Xi1["ie"] * T2_["iuex"] * DelGam1["xu"];
        val2["e"] += Xi1["ie"] * T2_["iUeX"] * DelGam1["XU"];
        val2["e"] -= 2 * s_ * Xi2["ie"] * T2_["iuex"] * DelGam1["xu"];
        val2["e"] -= 2 * s_ * Xi2["ie"] * T2_["iUeX"] * DelGam1["XU"];
    }

    if (CORRELATION_TERM) {
        temp["ijeb"] += V["ebij"] * Eeps2["ijeb"];
        temp["iJeB"] += V["eBiJ"] * Eeps2["iJeB"];
        val2["e"] -= 4.0 * s_ * Tau2["ijeb"] * temp["ijeb"];
        val2["e"] -= 8.0 * s_ * Tau2["iJeB"] * temp["iJeB"];
        temp.zero();

        val2["e"] += 2.0 * T2OverDelta["ijeb"] * Tau2["ijeb"];
        val2["e"] += 4.0 * T2OverDelta["iJeB"] * Tau2["iJeB"];

        temp["kled"] += V["edkl"] * Eeps2["kled"];
        temp["kLeD"] += V["eDkL"] * Eeps2["kLeD"];
        temp_1["kled"] += Kappa["kled"] * Delta2["kled"];
        temp_1["kLeD"] += Kappa["kLeD"] * Delta2["kLeD"];
        val2["e"] += 4.0 * s_ * temp["kled"] * temp_1["kled"];
        val2["e"] += 8.0 * s_ * temp["kLeD"] * temp_1["kLeD"];
        temp.zero();
        temp_1.zero();
    }

    BlockedTensor zef = BTF_->build(CoreTensor, "z{ef} normal", {"vv"});
    // virtual-virtual block entries within normal conditions
    if (CORRELATION_TERM) {
        zef["ef"] += 0.5 * Sigma3["if"] * F["ie"];
        zef["ef"] -= 0.5 * Sigma3["ie"] * F["if"];
    }

    if (CORRELATION_TERM) {
        zef["ef"] += 0.5 * Xi3["if"] * F["ie"];
        zef["ef"] -= 0.5 * Xi3["ie"] * F["if"];
    }

    if (CORRELATION_TERM) {
        zef["ef"] += Tau1["ijfb"] * V["ebij"];
        zef["ef"] += 2.0 * Tau1["iJfB"] * V["eBiJ"];

        temp["klfd"] += Kappa["klfd"] * Eeps2_p["klfd"];
        temp["kLfD"] += Kappa["kLfD"] * Eeps2_p["kLfD"];
        zef["ef"] += temp["klfd"] * V["edkl"];
        zef["ef"] += 2.0 * temp["kLfD"] * V["eDkL"];
        temp.zero();

        zef["ef"] -= Tau1["ijeb"] * V["fbij"];
        zef["ef"] -= 2.0 * Tau1["iJeB"] * V["fBiJ"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        zef["ef"] -= temp["kled"] * V["fdkl"];
        zef["ef"] -= 2.0 * temp["kLeD"] * V["fDkL"];
        temp.zero();
    }

    for (const std::string& block : {"vv", "VV"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                value = val2.block("v").data()[i[0]];
            } else {
                auto dmt = Delta1.block("vv").data()[i[1] * nvirt + i[0]];
                if (std::fabs(dmt) > 1e-12) {
                    value = zef.block("vv").data()[i[0] * nvirt + i[1]] / dmt;
                }
            }
        });
    }
}

void DSRG_MRPT2::set_z_aa_diag() {
    BlockedTensor val3 = BTF_->build(CoreTensor, "val3", {"a"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // active-active diagonal entries
    if (CORRELATION_TERM) {
        val3["w"] += -2 * s_ * Sigma1["wa"] * F["wa"];
        val3["w"] += -2 * s_ * DelGam1["xu"] * T2_["wuax"] * Sigma1["wa"];
        val3["w"] += -2 * s_ * DelGam1["XU"] * T2_["wUaX"] * Sigma1["wa"];
        val3["w"] += 2 * s_ * Sigma1["iw"] * F["iw"];
        val3["w"] += 2 * s_ * DelGam1["xu"] * T2_["iuwx"] * Sigma1["iw"];
        val3["w"] += 2 * s_ * DelGam1["XU"] * T2_["iUwX"] * Sigma1["iw"];

        val3["w"] += Sigma2["ia"] * T2_["iuaw"] * Gamma1_["wu"];
        val3["w"] += Sigma2["IA"] * T2_["uIwA"] * Gamma1_["wu"];
        val3["w"] -= Sigma2["ia"] * T2_["iwax"] * Gamma1_["xw"];
        val3["w"] -= Sigma2["IA"] * T2_["wIxA"] * Gamma1_["xw"];
    }

    if (CORRELATION_TERM) {
        val3["w"] -= Xi1["wa"] * F["wa"];
        val3["w"] += 2 * s_ * Xi2["wa"] * F["wa"];
        val3["w"] -= Xi1["wa"] * T2_["wuax"] * DelGam1["xu"];
        val3["w"] -= Xi1["wa"] * T2_["wUaX"] * DelGam1["XU"];
        val3["w"] += 2 * s_ * Xi2["wa"] * T2_["wuax"] * DelGam1["xu"];
        val3["w"] += 2 * s_ * Xi2["wa"] * T2_["wUaX"] * DelGam1["XU"];

        val3["w"] += Xi1["iw"] * F["iw"];
        val3["w"] -= 2 * s_ * Xi2["iw"] * F["iw"];
        val3["w"] += Xi1["iw"] * T2_["iuwx"] * DelGam1["xu"];
        val3["w"] += Xi1["iw"] * T2_["iUwX"] * DelGam1["XU"];
        val3["w"] -= 2 * s_ * Xi2["iw"] * T2_["iuwx"] * DelGam1["xu"];
        val3["w"] -= 2 * s_ * Xi2["iw"] * T2_["iUwX"] * DelGam1["XU"];

        val3["w"] += Xi3["ia"] * T2_["iuaw"] * Gamma1_["wu"];
        val3["w"] += Xi3["IA"] * T2_["uIwA"] * Gamma1_["wu"];
        val3["w"] -= Xi3["ia"] * T2_["iwax"] * Gamma1_["xw"];
        val3["w"] -= Xi3["IA"] * T2_["wIxA"] * Gamma1_["xw"];
    }

    if (CORRELATION_TERM) {
        temp["ujab"] += V["abuj"] * Eeps2["ujab"];
        temp["uJaB"] += V["aBuJ"] * Eeps2["uJaB"];
        val3["u"] += 4.0 * s_ * Tau2["ujab"] * temp["ujab"];
        val3["u"] += 8.0 * s_ * Tau2["uJaB"] * temp["uJaB"];
        temp.zero();

        val3["u"] -= 2.0 * T2OverDelta["ujab"] * Tau2["ujab"];
        val3["u"] -= 4.0 * T2OverDelta["uJaB"] * Tau2["uJaB"];

        temp["ulcd"] += V["cdul"] * Eeps2["ulcd"];
        temp["uLcD"] += V["cDuL"] * Eeps2["uLcD"];
        temp_1["ulcd"] += Kappa["ulcd"] * Delta2["ulcd"];
        temp_1["uLcD"] += Kappa["uLcD"] * Delta2["uLcD"];
        val3["u"] -= 4.0 * s_ * temp["ulcd"] * temp_1["ulcd"];
        val3["u"] -= 8.0 * s_ * temp["uLcD"] * temp_1["uLcD"];
        temp.zero();
        temp_1.zero();

        temp["ijub"] += V["ubij"] * Eeps2["ijub"];
        temp["iJuB"] += V["uBiJ"] * Eeps2["iJuB"];
        val3["u"] -= 4.0 * s_ * Tau2["ijub"] * temp["ijub"];
        val3["u"] -= 8.0 * s_ * Tau2["iJuB"] * temp["iJuB"];
        temp.zero();

        val3["u"] += 2.0 * T2OverDelta["ijub"] * Tau2["ijub"];
        val3["u"] += 4.0 * T2OverDelta["iJuB"] * Tau2["iJuB"];

        temp["klud"] += V["udkl"] * Eeps2["klud"];
        temp["kLuD"] += V["uDkL"] * Eeps2["kLuD"];
        temp_1["klud"] += Kappa["klud"] * Delta2["klud"];
        temp_1["kLuD"] += Kappa["kLuD"] * Delta2["kLuD"];
        val3["u"] += 4.0 * s_ * temp["klud"] * temp_1["klud"];
        val3["u"] += 8.0 * s_ * temp["kLuD"] * temp_1["kLuD"];
        temp.zero();
        temp_1.zero();
    }

    for (const std::string& block : {"aa", "AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                value = val3.block("a").data()[i[0]];
            }
        });
    }
}

// TODO
void DSRG_MRPT2::compute_density_vc() {

}
// TODO
void DSRG_MRPT2::compute_density_ca() {

}
// TODO
void DSRG_MRPT2::compute_density_va() {

}
// TODO
void DSRG_MRPT2::compute_density_aa() {

}
// TODO
void DSRG_MRPT2::compute_x_ci() {

}
// TODO
void DSRG_MRPT2::solve_linear_iter() {
    int dim_vc = nvirt * ncore, dim_ca = ncore * na, dim_va = nvirt * na,
        dim_aa = na * (na - 1) / 2, dim_ci = ndets;
    // dim_ci = 0;
    int dim = dim_vc + dim_ca + dim_va + dim_aa + dim_ci;
    int N = dim;
    int NRHS = 1, LDA = N, LDB = N;
    int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    std::vector<int> ipiv(N);
    std::map<string, int> preidx = {{"vc", 0},
                                    {"VC", 0},
                                    {"ca", dim_vc},
                                    {"CA", dim_vc},
                                    {"va", dim_vc + dim_ca},
                                    {"VA", dim_vc + dim_ca},
                                    {"aa", dim_vc + dim_ca + dim_va},
                                    {"AA", dim_vc + dim_ca + dim_va},
                                    {"ci", dim_vc + dim_ca + dim_va + dim_aa}};

    std::map<string, int> block_dim = {{"vc", ncore}, {"VC", ncore}, {"ca", na}, {"CA", na},
                                       {"va", na},    {"VA", na},    {"aa", 0},  {"AA", 0}};
    set_b(dim, preidx, block_dim);

    /*------------------------------------------------------------------*
     |                                                                  |
     |  //NOTICE:Iterative solver from here (avoiding directly form A)  |
     |                                                                  |
     *------------------------------------------------------------------*/

    outfile->Printf("\n    Solving the linear system ....................... ");

    bool converged = false;
    int iter = 1;
    // int maxiter = options_.get_int("CPHF_MAXITER");
    int maxiter = 200;
    BlockedTensor Z_old = BTF_->build(CoreTensor, "reference Z Matrix before iteration", spin_cases({"gg"}));
    ambit::Tensor x_ci_old = ambit::Tensor::build(ambit::CoreTensor, "reference multiplier x before iteration", {ndets});

    while (iter <= maxiter) {
        Z_old["pq"] = Z["pq"];
        x_ci_old("I") = x_ci("I");

        compute_density_vc();
        compute_density_ca();
        compute_density_va();
        compute_density_aa();
        compute_x_ci();

        Z_old["pq"] -= Z["pq"];
        x_ci_old("I") -= x_ci("I");

        double Z_norm = Z_old.norm();
        double x_ci_norm = x_ci_old.norm();
        outfile->Printf("\n    * %4d    %12.7e *", iter, Z_norm);
        outfile->Printf("\n    * %4d    %12.7e *", iter, x_ci_norm);
        // if (Z_norm < d_convergence_ && x_ci_norm < d_convergence_) {NOTICE where can I get the d_convergence_?
        if (Z_norm < 1e-6 && x_ci_norm < 1e-6) {
            converged = true;
            break;
        }
        iter++;
    }

    if (!converged) {
        throw PSIEXCEPTION("The DSRG Z vector equations did not converge.");       
    }
}

void DSRG_MRPT2::solve_z() {

    int dim_vc = nvirt * ncore, dim_ca = ncore * na, dim_va = nvirt * na,
        dim_aa = na * (na - 1) / 2, dim_ci = ndets;
    // dim_ci = 0;
    int dim = dim_vc + dim_ca + dim_va + dim_aa + dim_ci;
    int N = dim;
    int NRHS = 1, LDA = N, LDB = N;
    int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    std::vector<int> ipiv(N);
    std::map<string, int> preidx = {{"vc", 0},
                                    {"VC", 0},
                                    {"ca", dim_vc},
                                    {"CA", dim_vc},
                                    {"va", dim_vc + dim_ca},
                                    {"VA", dim_vc + dim_ca},
                                    {"aa", dim_vc + dim_ca + dim_va},
                                    {"AA", dim_vc + dim_ca + dim_va},
                                    {"ci", dim_vc + dim_ca + dim_va + dim_aa}};

    std::map<string, int> block_dim = {{"vc", ncore}, {"VC", ncore}, {"ca", na}, {"CA", na},
                                       {"va", na},    {"VA", na},    {"aa", 0},  {"AA", 0}};
    set_b(dim, preidx, block_dim);

    outfile->Printf("\n    Initializing A of the Linear System ............. ");
    std::vector<double> A(dim * dim);
    // NOTICE: Linear system A
    BlockedTensor temp1 = BTF_->build(
        CoreTensor, "temporal tensor 1",
        {
            "vcvc", "vcca", "vcva", "vcaa", "cavc", "caca", "cava", "caaa", "vavc", "vaca", "vava",
            "vaaa", "aavc", "aaca", "aava", "aaaa", "vcVC", "vcCA", "vcVA", "vcAA", "caVC", "caCA",
            "caVA", "caAA", "vaVC", "vaCA", "vaVA", "vaAA", "aaVC", "aaCA", "aaVA", "aaAA",
        });

    // VIRTUAL-CORE
    temp1["e,m,e1,m1"] += Delta1["m1,e1"] * I["e1,e"] * I["m1,m"];

    temp1["e,m,e1,m1"] -= V["e1,e,m1,m"];
    temp1["e,m,E1,M1"] -= V["e,E1,m,M1"];
    temp1["e,m,e1,m1"] -= V["m1,e,e1,m"];
    temp1["e,m,E1,M1"] -= V["e,M1,m,E1"];

    temp1["e,m,m1,u"] -= F["ue"] * I["m,m1"];

    temp1["e,m,n1,u"] -= V["u,e,n1,m"];
    temp1["e,m,N1,U"] -= V["e,U,m,N1"];
    temp1["e,m,n1,u"] -= V["n1,e,u,m"];
    temp1["e,m,N1,U"] -= V["e,N1,m,U"];

    temp1["e,m,n1,u"] += Gamma1_["uv"] * V["v,e,n1,m"];
    temp1["e,m,N1,U"] += Gamma1_["UV"] * V["e,V,m,N1"];
    temp1["e,m,n1,u"] += Gamma1_["uv"] * V["n1,e,v,m"];
    temp1["e,m,N1,U"] += Gamma1_["UV"] * V["e,N1,m,V"];

    temp1["e,m,e1,u"] -= Gamma1_["uv"] * V["e1,e,v,m"];
    temp1["e,m,E1,U"] -= Gamma1_["UV"] * V["e,E1,m,V"];
    temp1["e,m,e1,u"] -= Gamma1_["uv"] * V["v,e,e1,m"];
    temp1["e,m,E1,U"] -= Gamma1_["UV"] * V["e,V,m,E1"];

    temp1["e,m,e1,u"] += F["um"] * I["e,e1"];

    temp1["e,m,u,v"] -= V["veum"];
    temp1["e,m,U,V"] -= V["eVmU"];

    // CORE-ACTIVE
    temp1["m,w,e1,m1"] += F["w,e1"] * I["m,m1"];
    temp1["m,w,e1,m1"] += V["e1,w,m1,m"];
    temp1["m,w,E1,M1"] += V["w,E1,m,M1"];
    temp1["m,w,e1,m1"] += V["e1,m,m1,w"];
    temp1["m,w,E1,M1"] += V["m,E1,w,M1"];

    temp1["m,w,m1,u"] += F["uw"] * I["m1,m"];
    temp1["m,w,m1,u"] -= H["vw"] * Gamma1_["uv"] * I["m1,m"];
    temp1["m,w,m1,u"] -= V["v,n1,w,n"] * Gamma1_["uv"] * I["n,n1"] * I["m,m1"];
    temp1["m,w,m1,u"] -= V["v,N1,w,N"] * Gamma1_["uv"] * I["N,N1"] * I["m,m1"];
    temp1["m,w,m1,u"] -= 0.5 * V["xywv"] * Gamma2_["uvxy"] * I["m,m1"];
    temp1["m,w,m1,u"] -= V["xYwV"] * Gamma2_["uVxY"] * I["m,m1"];

    temp1["m,w,n1,u"] += V["u,w,n1,m"];
    temp1["m,w,N1,U"] += V["w,U,m,N1"];
    temp1["m,w,n1,u"] += V["u,m,n1,w"];
    temp1["m,w,N1,U"] += V["m,U,w,N1"];

    temp1["m,w,n1,u"] -= Gamma1_["uv"] * V["v,w,n1,m"];
    temp1["m,w,N1,U"] -= Gamma1_["UV"] * V["w,V,m,N1"];
    temp1["m,w,n1,u"] -= Gamma1_["uv"] * V["v,m,n1,w"];
    temp1["m,w,N1,U"] -= Gamma1_["UV"] * V["m,V,w,N1"];

    temp1["m,w,e1,u"] += Gamma1_["uv"] * V["e1,w,v,m"];
    temp1["m,w,E1,U"] += Gamma1_["UV"] * V["w,E1,m,V"];
    temp1["m,w,e1,u"] += Gamma1_["uv"] * V["e1,m,v,w"];
    temp1["m,w,E1,U"] += Gamma1_["UV"] * V["m,E1,w,V"];

    temp1["m,w,u,v"] += V["vwum"];
    temp1["m,w,U,V"] += V["wVmU"];

    temp1["m,w,e1,m1"] -= V["e1,u,m1,m"] * Gamma1_["uw"];
    temp1["m,w,E1,M1"] -= V["u,E1,m,M1"] * Gamma1_["uw"];
    temp1["m,w,e1,m1"] -= V["e1,m,m1,u"] * Gamma1_["uw"];
    temp1["m,w,E1,M1"] -= V["m,E1,u,M1"] * Gamma1_["uw"];

    temp1["m,w,n1,w1"] -= F["m,n1"] * I["w,w1"];

    temp1["m,w,n1,u"] -= V["u,v,n1,m"] * Gamma1_["wv"];
    temp1["m,w,N1,U"] -= V["v,U,m,N1"] * Gamma1_["wv"];
    temp1["m,w,n1,u"] -= V["u,m,n1,v"] * Gamma1_["wv"];
    temp1["m,w,N1,U"] -= V["m,U,v,N1"] * Gamma1_["wv"];

    temp1["m,w,n1,u"] += H["m,n1"] * Gamma1_["uw"];
    temp1["m,w,n1,u"] += V["m,m2,n1,n2"] * Gamma1_["uw"] * I["m2,n2"];
    temp1["m,w,n1,u"] += V["m,M2,n1,N2"] * Gamma1_["uw"] * I["M2,N2"];
    temp1["m,w,n1,u"] += 0.5 * V["x,y,n1,m"] * Gamma2_["u,w,x,y"];
    temp1["m,w,N1,U"] += V["y,X,m,N1"] * Gamma2_["w,U,y,X"];
    temp1["m,w,n1,u"] += V["m,y,n1,v"] * Gamma2_["u,v,w,y"];
    temp1["m,w,n1,u"] += V["m,Y,n1,V"] * Gamma2_["u,V,w,Y"];
    temp1["m,w,N1,U"] += V["m,Y,v,N1"] * Gamma2_["v,U,w,Y"];

    temp1["m,w,e1,u"] -= H["m,e1"] * Gamma1_["uw"];
    temp1["m,w,e1,u"] -= V["e1,n2,m,m2"] * Gamma1_["uw"] * I["m2,n2"];
    temp1["m,w,e1,u"] -= V["e1,N2,m,M2"] * Gamma1_["uw"] * I["M2,N2"];
    temp1["m,w,e1,u"] -= 0.5 * V["e1,m,x,y"] * Gamma2_["u,w,x,y"];
    temp1["m,w,E1,U"] -= V["m,E1,y,X"] * Gamma2_["w,U,y,X"];
    temp1["m,w,e1,u"] -= V["e1,v,m,y"] * Gamma2_["u,v,w,y"];
    temp1["m,w,e1,u"] -= V["e1,V,m,Y"] * Gamma2_["u,V,w,Y"];
    temp1["m,w,E1,U"] -= V["v,E1,m,Y"] * Gamma2_["v,U,w,Y"];

    temp1["m,w,u1,a1"] -= V["a1,v,u1,m"] * Gamma1_["wv"];
    temp1["m,w,U1,A1"] -= V["v,A1,m,U1"] * Gamma1_["wv"];

    temp1["m,w,w1,v"] -= F["vm"] * I["w,w1"];

    // VIRTUAL-ACTIVE
    temp1["e,w,e1,m1"] += F["m1,w"] * I["e,e1"];

    temp1["e,w,e1,u"] += H["vw"] * Gamma1_["uv"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["v,m,w,m1"] * Gamma1_["uv"] * I["m,m1"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["v,M,w,M1"] * Gamma1_["uv"] * I["M,M1"] * I["e,e1"];
    temp1["e,w,e1,u"] += 0.5 * V["xywv"] * Gamma2_["uvxy"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["xYwV"] * Gamma2_["uVxY"] * I["e,e1"];

    temp1["e,w,e1,m1"] -= V["e1,u,m1,e"] * Gamma1_["uw"];
    temp1["e,w,E1,M1"] -= V["u,E1,e,M1"] * Gamma1_["uw"];
    temp1["e,w,e1,m1"] -= V["e1,e,m1,u"] * Gamma1_["uw"];
    temp1["e,w,E1,M1"] -= V["e,E1,u,M1"] * Gamma1_["uw"];

    temp1["e,w,n1,u"] -= V["u,v,n1,e"] * Gamma1_["wv"];
    temp1["e,w,N1,U"] -= V["v,U,e,N1"] * Gamma1_["wv"];
    temp1["e,w,n1,u"] -= V["u,e,n1,v"] * Gamma1_["wv"];
    temp1["e,w,N1,U"] -= V["e,U,v,N1"] * Gamma1_["wv"];

    temp1["e,w,n1,u"] += H["e,n1"] * Gamma1_["uw"];
    temp1["e,w,n1,u"] += V["e,m,n1,m1"] * Gamma1_["uw"] * I["m,m1"];
    temp1["e,w,n1,u"] += V["e,M,n1,M1"] * Gamma1_["uw"] * I["M,M1"];
    temp1["e,w,n1,u"] += 0.5 * V["x,y,n1,e"] * Gamma2_["u,w,x,y"];
    temp1["e,w,N1,U"] += V["y,X,e,N1"] * Gamma2_["w,U,y,X"];
    temp1["e,w,n1,u"] += V["e,y,n1,v"] * Gamma2_["u,v,w,y"];
    temp1["e,w,n1,u"] += V["e,Y,n1,V"] * Gamma2_["u,V,w,Y"];
    temp1["e,w,N1,U"] += V["e,Y,v,N1"] * Gamma2_["v,U,w,Y"];

    temp1["e,w,u1,a1"] -= V["u1,e,a1,v"] * Gamma1_["wv"];
    temp1["e,w,U1,A1"] -= V["e,U1,v,A1"] * Gamma1_["wv"];

    temp1["e,w,w1,v"] -= F["ve"] * I["w,w1"];

    temp1["e,w,e1,u"] -= H["e,e1"] * Gamma1_["uw"];
    temp1["e,w,e1,u"] -= V["e,m,e1,m1"] * Gamma1_["uw"] * I["m,m1"];
    temp1["e,w,e1,u"] -= V["e,M,e1,M1"] * Gamma1_["uw"] * I["M,M1"];
    temp1["e,w,e1,u"] -= 0.5 * V["e1,e,x,y"] * Gamma2_["u,w,x,y"];
    temp1["e,w,E1,U"] -= V["e,E1,y,X"] * Gamma2_["w,U,y,X"];
    temp1["e,w,e1,u"] -= V["e,y,e1,v"] * Gamma2_["u,v,w,y"];
    temp1["e,w,e1,u"] -= V["e,Y,e1,V"] * Gamma2_["u,V,w,Y"];
    temp1["e,w,E1,U"] -= V["e,Y,v,E1"] * Gamma2_["v,U,w,Y"];

    // ACTIVE-ACTIVE
    temp1["w,z,w1,z1"] += Delta1["zw"] * I["w,w1"] * I["z,z1"];

    temp1["w,z,e1,m1"] -= V["e1,u,m1,w"] * Gamma1_["uz"];
    temp1["w,z,E1,M1"] -= V["u,E1,w,M1"] * Gamma1_["uz"];
    temp1["w,z,e1,m1"] -= V["e1,w,m1,u"] * Gamma1_["uz"];
    temp1["w,z,E1,M1"] -= V["w,E1,u,M1"] * Gamma1_["uz"];

    temp1["w,z,e1,m1"] += V["e1,u,m1,z"] * Gamma1_["uw"];
    temp1["w,z,E1,M1"] += V["u,E1,z,M1"] * Gamma1_["uw"];
    temp1["w,z,e1,m1"] += V["e1,z,m1,u"] * Gamma1_["uw"];
    temp1["w,z,E1,M1"] += V["z,E1,u,M1"] * Gamma1_["uw"];

    temp1["w,z,n1,z1"] -= F["w,n1"] * I["z,z1"];
    temp1["w,z,n1,w1"] += F["z,n1"] * I["w,w1"];

    temp1["w,z,n1,u"] -= V["u,v,n1,w"] * Gamma1_["zv"];
    temp1["w,z,N1,U"] -= V["v,U,w,N1"] * Gamma1_["zv"];
    temp1["w,z,n1,u"] -= V["u,w,n1,v"] * Gamma1_["zv"];
    temp1["w,z,N1,U"] -= V["w,U,v,N1"] * Gamma1_["zv"];

    temp1["w,z,n1,u"] += V["u,v,n1,z"] * Gamma1_["wv"];
    temp1["w,z,N1,U"] += V["v,U,z,N1"] * Gamma1_["wv"];
    temp1["w,z,n1,u"] += V["u,z,n1,v"] * Gamma1_["wv"];
    temp1["w,z,N1,U"] += V["z,U,v,N1"] * Gamma1_["wv"];

    temp1["w,z,n1,u"] += H["w,n1"] * Gamma1_["uz"];
    temp1["w,z,n1,u"] += V["w,m1,n1,m"] * Gamma1_["uz"] * I["m1,m"];
    temp1["w,z,n1,u"] += V["w,M1,n1,M"] * Gamma1_["uz"] * I["M1,M"];
    temp1["w,z,n1,u"] += 0.5 * V["x,y,n1,w"] * Gamma2_["u,z,x,y"];
    temp1["w,z,N1,U"] += V["y,X,w,N1"] * Gamma2_["z,U,y,X"];
    temp1["w,z,n1,u"] += V["w,y,n1,v"] * Gamma2_["u,v,z,y"];
    temp1["w,z,n1,u"] += V["w,Y,n1,V"] * Gamma2_["u,V,z,Y"];
    temp1["w,z,N1,U"] += V["w,Y,v,N1"] * Gamma2_["v,U,z,Y"];

    temp1["w,z,n1,u"] -= H["z,n1"] * Gamma1_["uw"];
    temp1["w,z,n1,u"] -= V["z,m1,n1,m"] * Gamma1_["uw"] * I["m1,m"];
    temp1["w,z,n1,u"] -= V["z,M1,n1,M"] * Gamma1_["uw"] * I["M1,M"];
    temp1["w,z,n1,u"] -= 0.5 * V["x,y,n1,z"] * Gamma2_["u,w,x,y"];
    temp1["w,z,N1,U"] -= V["y,X,z,N1"] * Gamma2_["w,U,y,X"];
    temp1["w,z,n1,u"] -= V["z,y,n1,v"] * Gamma2_["u,v,w,y"];
    temp1["w,z,n1,u"] -= V["z,Y,n1,V"] * Gamma2_["u,V,w,Y"];
    temp1["w,z,N1,U"] -= V["z,Y,v,N1"] * Gamma2_["v,U,w,Y"];

    temp1["w,z,e1,u"] -= H["w,e1"] * Gamma1_["uz"];
    temp1["w,z,e1,u"] -= V["e1,m,w,m1"] * Gamma1_["uz"] * I["m1,m"];
    temp1["w,z,e1,u"] -= V["e1,M,w,M1"] * Gamma1_["uz"] * I["M1,M"];
    temp1["w,z,e1,u"] -= 0.5 * V["e1,w,x,y"] * Gamma2_["u,z,x,y"];
    temp1["w,z,E1,U"] -= V["w,E1,y,X"] * Gamma2_["z,U,y,X"];
    temp1["w,z,e1,u"] -= V["e1,v,w,y"] * Gamma2_["u,v,z,y"];
    temp1["w,z,e1,u"] -= V["e1,V,w,Y"] * Gamma2_["u,V,z,Y"];
    temp1["w,z,E1,U"] -= V["v,E1,w,Y"] * Gamma2_["v,U,z,Y"];

    temp1["w,z,e1,u"] += H["z,e1"] * Gamma1_["uw"];
    temp1["w,z,e1,u"] += V["e1,m,z,m1"] * Gamma1_["uw"] * I["m1,m"];
    temp1["w,z,e1,u"] += V["e1,M,z,M1"] * Gamma1_["uw"] * I["M1,M"];
    temp1["w,z,e1,u"] += 0.5 * V["e1,z,x,y"] * Gamma2_["u,w,x,y"];
    temp1["w,z,E1,U"] += V["z,E1,y,X"] * Gamma2_["w,U,y,X"];
    temp1["w,z,e1,u"] += V["e1,v,z,y"] * Gamma2_["u,v,w,y"];
    temp1["w,z,e1,u"] += V["e1,V,z,Y"] * Gamma2_["u,V,w,Y"];
    temp1["w,z,E1,U"] += V["v,E1,z,Y"] * Gamma2_["v,U,w,Y"];

    temp1["w,z,u1,a1"] -= V["a1,v,u1,w"] * Gamma1_["zv"];
    temp1["w,z,U1,A1"] -= V["v,A1,w,U1"] * Gamma1_["zv"];

    temp1["w,z,u1,a1"] += V["a1,v,u1,z"] * Gamma1_["wv"];
    temp1["w,z,U1,A1"] += V["v,A1,z,U1"] * Gamma1_["wv"];

    for (const std::string& row : {"vc", "ca", "va", "aa"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];

        for (const std::string& col : {"vc", "VC", "ca", "CA", "va", "VA", "aa", "AA"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];
            if (row != "aa" && col != "aa" && col != "AA") {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0] * idx1 + i[1]) + dim * (pre2 + i[2] * idx2 + i[3]);
                    A.at(index) += value;
                });
            } else if (row == "aa" && col != "aa" && col != "AA") {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    if (i[0] > i[1]) {
                        int index = (pre1 + i[0] * (i[0] - 1) / 2 + i[1]) +
                                    dim * (pre2 + i[2] * idx2 + i[3]);
                        A.at(index) += value;
                    }
                });
            } else if (row != "aa" && (col == "aa" || col == "AA")) {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i2 = i[2] > i[3] ? i[2] : i[3], i3 = i[2] > i[3] ? i[3] : i[2];
                    if (i2 != i3) {
                        int index =
                            (pre1 + i[0] * idx1 + i[1]) + dim * (pre2 + i2 * (i2 - 1) / 2 + i3);
                        A.at(index) += value;
                    }
                });
            } else {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i2 = i[2] > i[3] ? i[2] : i[3], i3 = i[2] > i[3] ? i[3] : i[2];
                    if (i[0] > i[1] && i2 != i3) {
                        int index = (pre1 + i[0] * (i[0] - 1) / 2 + i[1]) +
                                    dim * (pre2 + i2 * (i2 - 1) / 2 + i3);
                        A.at(index) += value;
                    }
                });
            }
        }
    }

    // CI contribution
    auto ci_vc =
        ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{vc}", {ndets, nvirt, ncore});
    auto ci_ca =
        ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{ca}", {ndets, ncore, na});
    auto ci_va =
        ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{va}", {ndets, nvirt, na});
    auto ci_aa =
        ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{aa}", {ndets, na, na});

    // CI contribution to Z{VC}
    ci_vc("Iem") -= H.block("vc")("em") * ci("I");
    ci_vc("Iem") -= V_sumA_Alpha.block("cv")("me") * ci("I");
    ci_vc("Iem") -= V_sumB_Alpha.block("cv")("me") * ci("I");
    ci_vc("Iem") -= 0.5 * V.block("avac")("veum") * cc1a("Iuv");
    ci_vc("Iem") -= 0.5 * V.block("vAcA")("eVmU") * cc1b("IUV");

    // CI contribution to Z{CA}
    ci_ca("Imw") -= 0.50 * H.block("ac")("vm") * cc1a("Iwv");
    ci_ca("Imw") -= 0.25 * V_sumA_Alpha.block("ac")("um") * cc1a("Iuw");
    ci_ca("Imw") -= 0.25 * V_sumB_Alpha.block("ac")("um") * cc1a("Iuw");
    ci_ca("Imw") -= 0.25 * V_sumA_Alpha.block("ca")("mv") * cc1a("Iwv");
    ci_ca("Imw") -= 0.25 * V_sumB_Alpha.block("ca")("mv") * cc1a("Iwv");
    ci_ca("Imw") -= 0.125 * V.block("aaca")("xymv") * cc2aa("Iwvxy");
    ci_ca("Imw") -= 0.250 * V.block("aAcA")("xYmV") * cc2ab("IwVxY");
    ci_ca("Imw") -= 0.125 * V.block("aaac")("xyum") * cc2aa("Iuwxy");
    ci_ca("Imw") -= 0.250 * V.block("aAcA")("xYmU") * cc2ab("IwUxY");

    ci_ca("Imw") += H.block("ac")("wm") * ci("I");
    ci_ca("Imw") += V_sumA_Alpha.block("ca")("mw") * ci("I");
    ci_ca("Imw") += V_sumB_Alpha.block("ca")("mw") * ci("I");
    ci_ca("Imw") += 0.5 * V.block("aaac")("vwum") * cc1a("Iuv");
    ci_ca("Imw") += 0.5 * V.block("aAcA")("wVmU") * cc1b("IUV");

    // CI contribution to Z{VA}
    ci_va("Iew") -= 0.50 * H.block("av")("ve") * cc1a("Iwv");
    ci_va("Iew") -= 0.25 * V_sumA_Alpha.block("av")("ue") * cc1a("Iuw");
    ci_va("Iew") -= 0.25 * V_sumB_Alpha.block("av")("ue") * cc1a("Iuw");
    ci_va("Iew") -= 0.25 * V_sumA_Alpha.block("va")("ev") * cc1a("Iwv");
    ci_va("Iew") -= 0.25 * V_sumB_Alpha.block("va")("ev") * cc1a("Iwv");
    ci_va("Iew") -= 0.125 * V.block("vaaa")("evxy") * cc2aa("Iwvxy");
    ci_va("Iew") -= 0.250 * V.block("vAaA")("eVxY") * cc2ab("IwVxY");
    ci_va("Iew") -= 0.125 * V.block("avaa")("uexy") * cc2aa("Iuwxy");
    ci_va("Iew") -= 0.250 * V.block("vAaA")("eUxY") * cc2ab("IwUxY");

    // CI contribution to Z{AA}
    ci_aa("Iwz") -= 0.50 * H.block("aa")("vw") * cc1a("Izv");
    ci_aa("Iwz") += 0.50 * H.block("aa")("vz") * cc1a("Iwv");
    ci_aa("Iwz") -= 0.25 * V_sumA_Alpha.block("aa")("uw") * cc1a("Iuz");
    ci_aa("Iwz") -= 0.25 * V_sumB_Alpha.block("aa")("uw") * cc1a("Iuz");
    ci_aa("Iwz") -= 0.25 * V_sumA_Alpha.block("aa")("wv") * cc1a("Izv");
    ci_aa("Iwz") -= 0.25 * V_sumB_Alpha.block("aa")("wv") * cc1a("Izv");
    ci_aa("Iwz") += 0.25 * V_sumA_Alpha.block("aa")("uz") * cc1a("Iuw");
    ci_aa("Iwz") += 0.25 * V_sumB_Alpha.block("aa")("uz") * cc1a("Iuw");
    ci_aa("Iwz") += 0.25 * V_sumA_Alpha.block("aa")("zv") * cc1a("Iwv");
    ci_aa("Iwz") += 0.25 * V_sumB_Alpha.block("aa")("zv") * cc1a("Iwv");

    ci_aa("Iwz") -= 0.125 * V.block("aaaa")("wvxy") * cc2aa("Izvxy");
    ci_aa("Iwz") -= 0.250 * V.block("aAaA")("wVxY") * cc2ab("IzVxY");
    ci_aa("Iwz") -= 0.125 * V.block("aaaa")("uwxy") * cc2aa("Iuzxy");
    ci_aa("Iwz") -= 0.250 * V.block("aAaA")("wUxY") * cc2ab("IzUxY");
    ci_aa("Iwz") += 0.125 * V.block("aaaa")("zvxy") * cc2aa("Iwvxy");
    ci_aa("Iwz") += 0.250 * V.block("aAaA")("zVxY") * cc2ab("IwVxY");
    ci_aa("Iwz") += 0.125 * V.block("aaaa")("uzxy") * cc2aa("Iuwxy");
    ci_aa("Iwz") += 0.250 * V.block("aAaA")("zUxY") * cc2ab("IwUxY");

    for (const std::string& row : {"vc", "ca", "va", "aa"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];
        auto temp_ci = ci_vc;
        if (row == "ca")
            temp_ci = ci_ca;
        else if (row == "va")
            temp_ci = ci_va;
        else if (row == "aa")
            temp_ci = ci_aa;

        for (const std::string& col : {"ci"}) {
            int pre2 = preidx[col];
            if (row != "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[1] * idx1 + i[2]) + dim * (pre2 + i[0]);
                    A.at(index) += value;
                });
            } else if (row == "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    if (i[1] > i[2]) {
                        int index = (pre1 + i[1] * (i[1] - 1) / 2 + i[2]) + dim * (pre2 + i[0]);
                        A.at(index) += value;
                    }
                });
            }
        }
    }

    auto ck_vc_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{vc} alpha part",
                                        {ndets, nvirt, ncore});
    auto ck_ca_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{ca} alpha part",
                                        {ndets, ncore, na});
    auto ck_va_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{va} alpha part",
                                        {ndets, nvirt, na});
    auto ck_aa_a =
        ambit::Tensor::build(ambit::CoreTensor, "ci equations z{aa} alpha part", {ndets, na, na});

    auto ck_vc_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{vc} beta part",
                                        {ndets, nvirt, ncore});
    auto ck_ca_b =
        ambit::Tensor::build(ambit::CoreTensor, "ci equations z{ca} beta part", {ndets, ncore, na});
    auto ck_va_b =
        ambit::Tensor::build(ambit::CoreTensor, "ci equations z{va} beta part", {ndets, nvirt, na});
    auto ck_aa_b =
        ambit::Tensor::build(ambit::CoreTensor, "ci equations z{aa} beta part", {ndets, na, na});

    // virtual-core
    ck_vc_a("Kem") += 2 * V.block("vaca")("eumv") * cc1a("Kuv");
    ck_vc_a("Kem") += 2 * V.block("vAcA")("eUmV") * cc1b("KUV");

    ck_vc_b("KEM") += 2 * V.block("VACA")("EUMV") * cc1b("KUV");
    ck_vc_b("KEM") += 2 * V.block("aVaC")("uEvM") * cc1a("Kuv");

    // contribution from Alpha
    ck_vc_a("Kem") += -4 * ci("K") * V.block("vaca")("exmy") * Gamma1_.block("aa")("xy");
    ck_vc_a("Kem") += -4 * ci("K") * V.block("vAcA")("eXmY") * Gamma1_.block("AA")("XY");
    ck_vc_b("KEM") += -4 * ci("K") * V.block("VACA")("EXMY") * Gamma1_.block("AA")("XY");
    ck_vc_b("KEM") += -4 * ci("K") * V.block("aVaC")("xEyM") * Gamma1_.block("aa")("xy");

    // core-active
    ck_ca_a("Knu") += 2 * V.block("aaca")("uynx") * cc1a("Kxy");
    ck_ca_a("Knu") += 2 * V.block("aAcA")("uYnX") * cc1b("KXY");
    ck_ca_a("Knu") -= 2 * H.block("ac")("vn") * cc1a("Kuv");
    ck_ca_a("Knu") -= 2 * V_sumA_Alpha.block("ac")("vn") * cc1a("Kuv");
    ck_ca_a("Knu") -= 2 * V_sumB_Alpha.block("ac")("vn") * cc1a("Kuv");
    ck_ca_a("Knu") -= V.block("aaca")("xynv") * cc2aa("Kuvxy");
    ck_ca_a("Knu") -= 2 * V.block("aAcA")("xYnV") * cc2ab("KuVxY");

    // NOTICE beta
    ck_ca_b("KNU") += 2 * V.block("AACA")("UYNX") * cc1b("KXY");
    ck_ca_b("KNU") += 2 * V.block("aAaC")("yUxN") * cc1a("Kxy");
    ck_ca_b("KNU") -= 2 * H.block("AC")("VN") * cc1b("KUV");
    ck_ca_b("KNU") -= 2 * V_sumB_Beta.block("AC")("VN") * cc1b("KUV");
    ck_ca_b("KNU") -= 2 * V_sumA_Beta.block("AC")("VN") * cc1b("KUV");
    ck_ca_b("KNU") -= V.block("AACA")("XYNV") * cc2bb("KUVXY");
    ck_ca_b("KNU") -= 2 * V.block("aAaC")("xYvN") * cc2ab("KvUxY");

    // contribution from Alpha
    ck_ca_a("Knu") += -4 * ci("K") * V.block("aaca")("uynx") * Gamma1_.block("aa")("xy");
    ck_ca_a("Knu") += -4 * ci("K") * V.block("aAcA")("uYnX") * Gamma1_.block("AA")("XY");
    ck_ca_a("Knu") += 4 * ci("K") * H.block("ac")("vn") * Gamma1_.block("aa")("uv");
    ck_ca_a("Knu") += 4 * ci("K") * V_sumA_Alpha.block("ac")("vn") * Gamma1_.block("aa")("uv");
    ck_ca_a("Knu") += 4 * ci("K") * V_sumB_Alpha.block("ac")("vn") * Gamma1_.block("aa")("uv");
    ck_ca_a("Knu") += 2 * ci("K") * V.block("aaca")("xynv") * Gamma2_.block("aaaa")("uvxy");
    ck_ca_a("Knu") += 4 * ci("K") * V.block("aAcA")("xYnV") * Gamma2_.block("aAaA")("uVxY");

    // NOTICE beta
    ck_ca_b("KNU") += -4 * ci("K") * V.block("AACA")("UYNX") * Gamma1_.block("AA")("XY");
    ck_ca_b("KNU") += -4 * ci("K") * V.block("aAaC")("yUxN") * Gamma1_.block("aa")("xy");
    ck_ca_b("KNU") += 4 * ci("K") * H.block("AC")("VN") * Gamma1_.block("AA")("UV");
    ck_ca_b("KNU") += 4 * ci("K") * V_sumB_Beta.block("AC")("VN") * Gamma1_.block("AA")("UV");
    ck_ca_b("KNU") += 4 * ci("K") * V_sumA_Beta.block("AC")("VN") * Gamma1_.block("AA")("UV");
    ck_ca_b("KNU") += 2 * ci("K") * V.block("AACA")("XYNV") * Gamma2_.block("AAAA")("UVXY");
    ck_ca_b("KNU") += 4 * ci("K") * V.block("aAaC")("xYvN") * Gamma2_.block("aAaA")("vUxY");

    // virtual-active
    ck_va_a("Keu") += 2 * H.block("av")("ve") * cc1a("Kuv");
    ck_va_a("Keu") += 2 * V_sumA_Alpha.block("av")("ve") * cc1a("Kuv");
    ck_va_a("Keu") += 2 * V_sumB_Alpha.block("av")("ve") * cc1a("Kuv");
    ck_va_a("Keu") += V.block("vaaa")("evxy") * cc2aa("Kuvxy");
    ck_va_a("Keu") += 2 * V.block("vAaA")("eVxY") * cc2ab("KuVxY");

    // NOTICE beta
    ck_va_b("KEU") += 2 * H.block("AV")("VE") * cc1b("KUV");
    ck_va_b("KEU") += 2 * V_sumB_Beta.block("AV")("VE") * cc1b("KUV");
    ck_va_b("KEU") += 2 * V_sumA_Beta.block("AV")("VE") * cc1b("KUV");
    ck_va_b("KEU") += V.block("VAAA")("EVXY") * cc2bb("KUVXY");
    ck_va_b("KEU") += 2 * V.block("aVaA")("vExY") * cc2ab("KvUxY");

    /// contribution from Alpha
    ck_va_a("Keu") += -4 * ci("K") * H.block("av")("ve") * Gamma1_.block("aa")("uv");
    ck_va_a("Keu") += -4 * ci("K") * V_sumA_Alpha.block("av")("ve") * Gamma1_.block("aa")("uv");
    ck_va_a("Keu") += -4 * ci("K") * V_sumB_Alpha.block("av")("ve") * Gamma1_.block("aa")("uv");
    ck_va_a("Keu") += -2 * ci("K") * V.block("vaaa")("evxy") * Gamma2_.block("aaaa")("uvxy");
    ck_va_a("Keu") += -4 * ci("K") * V.block("vAaA")("eVxY") * Gamma2_.block("aAaA")("uVxY");

    // NOTICE beta
    ck_va_b("KEU") += -4 * ci("K") * H.block("AV")("VE") * Gamma1_.block("AA")("UV");
    ck_va_b("KEU") += -4 * ci("K") * V_sumB_Beta.block("AV")("VE") * Gamma1_.block("AA")("UV");
    ck_va_b("KEU") += -4 * ci("K") * V_sumA_Beta.block("AV")("VE") * Gamma1_.block("AA")("UV");
    ck_va_b("KEU") += -2 * ci("K") * V.block("VAAA")("EVXY") * Gamma2_.block("AAAA")("UVXY");
    ck_va_b("KEU") += -4 * ci("K") * V.block("aVaA")("vExY") * Gamma2_.block("aAaA")("vUxY");

    // active-active
    ck_aa_a("Kuv") += V.block("aaaa")("uyvx") * cc1a("Kxy");
    ck_aa_a("Kuv") += V.block("aAaA")("uYvX") * cc1b("KXY");

    // NOTICE beta
    ck_aa_b("KUV") += V.block("AAAA")("UYVX") * cc1b("KXY");
    ck_aa_b("KUV") += V.block("aAaA")("yUxV") * cc1a("Kxy");

    /// contribution from Alpha
    ck_aa_a("Kuv") += -2 * ci("K") * V.block("aaaa")("uyvx") * Gamma1_.block("aa")("xy");
    ck_aa_a("Kuv") += -2 * ci("K") * V.block("aAaA")("uYvX") * Gamma1_.block("AA")("XY");

    // NOTICE beta
    ck_aa_b("KUV") += -2 * ci("K") * V.block("AAAA")("UYVX") * Gamma1_.block("AA")("XY");
    ck_aa_b("KUV") += -2 * ci("K") * V.block("aAaA")("yUxV") * Gamma1_.block("aa")("xy");

    // CI equations' contribution to A
    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"vc", "ca", "va", "aa"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];
            auto temp_ci = ck_vc_a;

            if (col == "ca")
                temp_ci = ck_ca_a;
            else if (col == "va")
                temp_ci = ck_va_a;
            else if (col == "aa")
                temp_ci = ck_aa_a;

            if (col != "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0]) + dim * (pre2 + i[1] * idx2 + i[2]);
                    A.at(index) += value;
                });
            }

            else if (col == "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i1 = i[1] > i[2] ? i[1] : i[2], i2 = i[1] > i[2] ? i[2] : i[1];
                    if (i1 != i2) {
                        int index = (pre1 + i[0]) + dim * (pre2 + i1 * (i1 - 1) / 2 + i2);
                        A.at(index) += value;
                    }
                });
            }
        }
    }

    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"VC", "CA", "VA", "AA"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];
            auto temp_ci = ck_vc_b;

            if (col == "CA")
                temp_ci = ck_ca_b;
            else if (col == "VA")
                temp_ci = ck_va_b;
            else if (col == "AA")
                temp_ci = ck_aa_b;

            if (col != "AA") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0]) + dim * (pre2 + i[1] * idx2 + i[2]);
                    A.at(index) += value;
                });
            }

            else if (col == "AA") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i1 = i[1] > i[2] ? i[1] : i[2], i2 = i[1] > i[2] ? i[2] : i[1];
                    if (i1 != i2) {
                        int index = (pre1 + i[0]) + dim * (pre2 + i1 * (i1 - 1) / 2 + i2);
                        A.at(index) += value;
                    }
                });
            }
        }
    }

    auto ck_ci =
        ambit::Tensor::build(ambit::CoreTensor, "ci equations ci multiplier part", {ndets, ndets});
    auto I_ci = ambit::Tensor::build(ambit::CoreTensor, "identity", {ndets, ndets});

    I_ci.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = (i[0] == i[1]) ? 1.0 : 0.0; });

    ck_ci("KI") += H.block("cc")("mn") * I.block("cc")("mn") * I_ci("KI");
    ck_ci("KI") += H.block("CC")("MN") * I.block("CC")("MN") * I_ci("KI");
    ck_ci("KI") += cc.cc1a()("KIuv") * H.block("aa")("uv");
    ck_ci("KI") += cc.cc1b()("KIUV") * H.block("AA")("UV");

    ck_ci("KI") += 0.5 * V_sumA_Alpha["m,m1"] * I["m,m1"] * I_ci("KI");
    ck_ci("KI") += 0.5 * V_sumB_Beta["M,M1"] * I["M,M1"] * I_ci("KI");
    ck_ci("KI") += V_sumB_Alpha["m,m1"] * I["m,m1"] * I_ci("KI");

    ck_ci("KI") += cc.cc1a()("KIuv") * V_sumA_Alpha.block("aa")("uv");
    ck_ci("KI") += cc.cc1b()("KIUV") * V_sumB_Beta.block("AA")("UV");

    ck_ci("KI") += cc.cc1a()("KIuv") * V_sumB_Alpha.block("aa")("uv");
    ck_ci("KI") += cc.cc1b()("KIUV") * V_sumA_Beta.block("AA")("UV");

    ck_ci("KI") += 0.25 * cc.cc2aa()("KIuvxy") * V.block("aaaa")("uvxy");
    ck_ci("KI") += 0.25 * cc.cc2bb()("KIUVXY") * V.block("AAAA")("UVXY");
    ck_ci("KI") += 0.50 * cc.cc2ab()("KIuVxY") * V.block("aAaA")("uVxY");
    ck_ci("KI") += 0.50 * cc.cc2ab()("IKuVxY") * V.block("aAaA")("uVxY");

    ck_ci("KI") -= (Eref_ - Enuc_ - Efrzc_) * I_ci("KI");

    // NOTICE: QR decomposition with column pivoting
    int dim2 = ndets;
    int n2 = dim2, lda2 = dim2;
    int lwork = 3 * n2 + 3;
    std::vector<int> jpvt(n2);
    std::vector<double> tau(dim2);
    std::vector<double> work(lwork);
    std::vector<double> A2(dim2 * dim2);

    for (const std::string& row : {"ci"}) {
        for (const std::string& col : {"ci"}) {
            (ck_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = i[1] + dim2 * i[0];
                A2.at(index) = value;
            });
        }
    }

    C_DGEQP3(n2, n2, &A2[0], lda2, &jpvt[0], &tau[0], &work[0], lwork);

    const int ROW2DEL = jpvt[ndets - 1] - 1;

    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"ci"}) {
            int pre2 = preidx[col];

            (ck_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                if (i[0] != ROW2DEL) {
                    int index = (pre1 + i[0]) + dim * (pre2 + i[1]);
                    A.at(index) += value;
                }
            });

            (ci).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = (pre1 + ROW2DEL) + dim * (pre2 + i[0]);
                A.at(index) += value;
            });

            for (int j = 0; j < pre2; j++)
                A.at((pre1 + ROW2DEL) + dim * j) = 0.0;
            b.at(pre1 + ROW2DEL) = 0.0;
        }
    }

    outfile->Printf("Done");
    outfile->Printf("\n    Solving Off-diagonal Entries of Z ............... ");
    int info;

    C_DGESV(n, nrhs, &A[0], lda, &ipiv[0], &b[0], ldb);

    for (const std::string& block : {"vc", "ca", "va", "aa"}) {
        int pre = preidx[block], idx = block_dim[block];
        if (block != "aa") {
            (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = pre + i[0] * idx + i[1];
                value = b.at(index);
            });
        } else {
            (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                int i0 = i[0] > i[1] ? i[0] : i[1], i1 = i[0] > i[1] ? i[1] : i[0];
                if (i0 != i1) {
                    int index = pre + i0 * (i0 - 1) / 2 + i1;
                    value = b.at(index);
                }
            });
        }
    }
    for (const std::string& block : {"ci"}) {
        int pre = preidx[block];
        (x_ci).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = pre + i[0];
            value = b.at(index);
        });
    }
    Z["me"] = Z["em"];
    Z["wm"] = Z["mw"];
    Z["we"] = Z["ew"];

    // Beta part
    for (const std::string& block : {"VC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("vc").data()[i[0] * ncore + i[1]];
        });
    }
    for (const std::string& block : {"CA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("ca").data()[i[0] * na + i[1]];
        });
    }
    for (const std::string& block : {"VA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("va").data()[i[0] * na + i[1]];
        });
    }
    for (const std::string& block : {"AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("aa").data()[i[0] * na + i[1]];
        });
    }
    Z["ME"] = Z["EM"];
    Z["WM"] = Z["MW"];
    Z["WE"] = Z["EW"];

    outfile->Printf("Done");
}

}// namespace forte