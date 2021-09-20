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

    if (CORRELATION_TERM) {
        Z_b["em"] += 0.5 * Sigma3["ma"] * F["ea"];
        Z_b["em"] += 0.5 * Sigma3["ia"] * V["ieam"];
        Z_b["em"] += 0.5 * Sigma3["IA"] * V["eImA"];
        Z_b["em"] += 0.5 * Sigma3["ia"] * V["aeim"];
        Z_b["em"] += 0.5 * Sigma3["IA"] * V["eAmI"];
        Z_b["em"] -= 0.5 * Sigma3["ie"] * F["im"];
    }

    if (CORRELATION_TERM) {
        Z_b["em"] += 0.5 * Xi3["ma"] * F["ea"];
        Z_b["em"] += 0.5 * Xi3["ia"] * V["ieam"];
        Z_b["em"] += 0.5 * Xi3["IA"] * V["eImA"];
        Z_b["em"] += 0.5 * Xi3["ia"] * V["aeim"];
        Z_b["em"] += 0.5 * Xi3["IA"] * V["eAmI"];
        Z_b["em"] -= 0.5 * Xi3["ie"] * F["im"];
    }

    if (CORRELATION_TERM) {
        Z_b["em"] += Tau1["mjab"] * V["abej"];
        Z_b["em"] += 2.0 * Tau1["mJaB"] * V["aBeJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["em"] += temp["mlcd"] * V["cdel"];
        Z_b["em"] += 2.0 * temp["mLcD"] * V["cDeL"];
        temp.zero();

        Z_b["em"] -= Tau1["ijeb"] * V["mbij"];
        Z_b["em"] -= 2.0 * Tau1["iJeB"] * V["mBiJ"];

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
    if (CORRELATION_TERM) {
        Z_b["wz"] += 0.5 * Sigma3["za"] * F["wa"];
        Z_b["wz"] += 0.5 * Sigma3["iz"] * F["iw"];
        Z_b["wz"] += 0.5 * Sigma3["ia"] * V["awiv"] * Gamma1_["zv"];
        Z_b["wz"] += 0.5 * Sigma3["IA"] * V["wAvI"] * Gamma1_["zv"];
        Z_b["wz"] += 0.5 * Sigma3["ia"] * V["auiw"] * Gamma1_["uz"];
        Z_b["wz"] += 0.5 * Sigma3["IA"] * V["uAwI"] * Gamma1_["uz"];

        Z_b["wz"] -= 0.5 * Sigma3["wa"] * F["za"];
        Z_b["wz"] -= 0.5 * Sigma3["iw"] * F["iz"];
        Z_b["wz"] -= 0.5 * Sigma3["ia"] * V["aziv"] * Gamma1_["wv"];
        Z_b["wz"] -= 0.5 * Sigma3["IA"] * V["zAvI"] * Gamma1_["wv"];
        Z_b["wz"] -= 0.5 * Sigma3["ia"] * V["auiz"] * Gamma1_["uw"];
        Z_b["wz"] -= 0.5 * Sigma3["IA"] * V["uAzI"] * Gamma1_["uw"];
    }

    if (CORRELATION_TERM) {
        Z_b["wz"] += 0.5 * Xi3["za"] * F["wa"];
        Z_b["wz"] += 0.5 * Xi3["iz"] * F["iw"];
        Z_b["wz"] += 0.5 * Xi3["ia"] * V["awiv"] * Gamma1_["zv"];
        Z_b["wz"] += 0.5 * Xi3["IA"] * V["wAvI"] * Gamma1_["zv"];
        Z_b["wz"] += 0.5 * Xi3["ia"] * V["auiw"] * Gamma1_["uz"];
        Z_b["wz"] += 0.5 * Xi3["IA"] * V["uAwI"] * Gamma1_["uz"];

        Z_b["wz"] -= 0.5 * Xi3["wa"] * F["za"];
        Z_b["wz"] -= 0.5 * Xi3["iw"] * F["iz"];
        Z_b["wz"] -= 0.5 * Xi3["ia"] * V["aziv"] * Gamma1_["wv"];
        Z_b["wz"] -= 0.5 * Xi3["IA"] * V["zAvI"] * Gamma1_["wv"];
        Z_b["wz"] -= 0.5 * Xi3["ia"] * V["auiz"] * Gamma1_["uw"];
        Z_b["wz"] -= 0.5 * Xi3["IA"] * V["uAzI"] * Gamma1_["uw"];
    }

    if (CORRELATION_TERM) {
        Z_b["wz"] += Tau1["ijzb"] * V["wbij"];
        Z_b["wz"] += 2.0 * Tau1["iJzB"] * V["wBiJ"];

        temp["klzd"] += Kappa["klzd"] * Eeps2_p["klzd"];
        temp["kLzD"] += Kappa["kLzD"] * Eeps2_p["kLzD"];
        Z_b["wz"] += temp["klzd"] * V["wdkl"];
        Z_b["wz"] += 2.0 * temp["kLzD"] * V["wDkL"];
        temp.zero();

        Z_b["wz"] += Tau1["zjab"] * V["abwj"];
        Z_b["wz"] += 2.0 * Tau1["zJaB"] * V["aBwJ"];

        temp["zlcd"] += Kappa["zlcd"] * Eeps2_p["zlcd"];
        temp["zLcD"] += Kappa["zLcD"] * Eeps2_p["zLcD"];
        Z_b["wz"] += temp["zlcd"] * V["cdwl"];
        Z_b["wz"] += 2.0 * temp["zLcD"] * V["cDwL"];
        temp.zero();

        Z_b["wz"] -= Tau1["ijwb"] * V["zbij"];
        Z_b["wz"] -= 2.0 * Tau1["iJwB"] * V["zBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["wz"] -= temp["klwd"] * V["zdkl"];
        Z_b["wz"] -= 2.0 * temp["kLwD"] * V["zDkL"];
        temp.zero();

        Z_b["wz"] -= Tau1["wjab"] * V["abzj"];
        Z_b["wz"] -= 2.0 * Tau1["wJaB"] * V["aBzJ"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["wz"] -= temp["wlcd"] * V["cdzl"];
        Z_b["wz"] -= 2.0 * temp["wLcD"] * V["cDzL"];
        temp.zero();
    }
    Z_b["wz"] += Z["m1,n1"] * V["n1,v,m1,w"] * Gamma1_["zv"];
    Z_b["wz"] += Z["M1,N1"] * V["v,N1,w,M1"] * Gamma1_["zv"];

    Z_b["wz"] += Z["e1,f1"] * V["f1,v,e1,w"] * Gamma1_["zv"];
    Z_b["wz"] += Z["E1,F1"] * V["v,F1,w,E1"] * Gamma1_["zv"];

    Z_b["wz"] -= Z["m1,n1"] * V["n1,v,m1,z"] * Gamma1_["wv"];
    Z_b["wz"] -= Z["M1,N1"] * V["v,N1,z,M1"] * Gamma1_["wv"];

    Z_b["wz"] -= Z["e1,f1"] * V["f1,v,e1,z"] * Gamma1_["wv"];
    Z_b["wz"] -= Z["E1,F1"] * V["v,F1,z,E1"] * Gamma1_["wv"];

    // NOTICE: constant b for z{virtual-active}
    if (CORRELATION_TERM) {
        Z_b["ew"] += 0.5 * Sigma3["wa"] * F["ea"];
        Z_b["ew"] += 0.5 * Sigma3["iw"] * F["ie"];
        Z_b["ew"] += 0.5 * Sigma3["ia"] * V["aeiv"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * Sigma3["IA"] * V["eAvI"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * Sigma3["ia"] * V["auie"] * Gamma1_["uw"];
        Z_b["ew"] += 0.5 * Sigma3["IA"] * V["uAeI"] * Gamma1_["uw"];
        Z_b["ew"] -= 0.5 * Sigma3["ie"] * F["iw"];
    }

    if (CORRELATION_TERM) {
        Z_b["ew"] += 0.5 * Xi3["wa"] * F["ea"];
        Z_b["ew"] += 0.5 * Xi3["iw"] * F["ie"];
        Z_b["ew"] += 0.5 * Xi3["ia"] * V["aeiv"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * Xi3["IA"] * V["eAvI"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * Xi3["ia"] * V["auie"] * Gamma1_["uw"];
        Z_b["ew"] += 0.5 * Xi3["IA"] * V["uAeI"] * Gamma1_["uw"];
        Z_b["ew"] -= 0.5 * Xi3["ie"] * F["iw"];
    }

    if (CORRELATION_TERM) {
        Z_b["ew"] += Tau1["ijwb"] * V["ebij"];
        Z_b["ew"] += 2.0 * Tau1["iJwB"] * V["eBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["ew"] += temp["klwd"] * V["edkl"];
        Z_b["ew"] += 2.0 * temp["kLwD"] * V["eDkL"];
        temp.zero();

        Z_b["ew"] += Tau1["wjab"] * V["abej"];
        Z_b["ew"] += 2.0 * Tau1["wJaB"] * V["aBeJ"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["ew"] += temp["wlcd"] * V["cdel"];
        Z_b["ew"] += 2.0 * temp["wLcD"] * V["cDeL"];
        temp.zero();

        Z_b["ew"] -= Tau1["ijeb"] * V["wbij"];
        Z_b["ew"] -= 2.0 * Tau1["iJeB"] * V["wBiJ"];

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
        Z_b["mw"] += 0.5 * Sigma3["wa"] * F["ma"];
        Z_b["mw"] += 0.5 * Sigma3["iw"] * F["im"];
        Z_b["mw"] += 0.5 * Sigma3["ia"] * V["amiv"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * Sigma3["IA"] * V["mAvI"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * Sigma3["ia"] * V["auim"] * Gamma1_["uw"];
        Z_b["mw"] += 0.5 * Sigma3["IA"] * V["uAmI"] * Gamma1_["uw"];

        Z_b["mw"] -= 0.5 * Sigma3["ma"] * F["wa"];
        Z_b["mw"] -= 0.5 * Sigma3["ia"] * V["amiw"];
        Z_b["mw"] -= 0.5 * Sigma3["IA"] * V["mAwI"];
        Z_b["mw"] -= 0.5 * Sigma3["ia"] * V["awim"];
        Z_b["mw"] -= 0.5 * Sigma3["IA"] * V["wAmI"];
    }

    if (CORRELATION_TERM) {
        Z_b["mw"] += 0.5 * Xi3["wa"] * F["ma"];
        Z_b["mw"] += 0.5 * Xi3["iw"] * F["im"];
        Z_b["mw"] += 0.5 * Xi3["ia"] * V["amiv"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * Xi3["IA"] * V["mAvI"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * Xi3["ia"] * V["auim"] * Gamma1_["uw"];
        Z_b["mw"] += 0.5 * Xi3["IA"] * V["uAmI"] * Gamma1_["uw"];

        Z_b["mw"] -= 0.5 * Xi3["ma"] * F["wa"];
        Z_b["mw"] -= 0.5 * Xi3["ia"] * V["amiw"];
        Z_b["mw"] -= 0.5 * Xi3["IA"] * V["mAwI"];
        Z_b["mw"] -= 0.5 * Xi3["ia"] * V["awim"];
        Z_b["mw"] -= 0.5 * Xi3["IA"] * V["wAmI"];
    }

    if (CORRELATION_TERM) {
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
    b_ck("K") -= H.block("aa")("vu") * cc1a("Kuv");
    b_ck("K") -= H.block("AA")("VU") * cc1b("KUV");

    b_ck("K") -= V_sumA_Alpha.block("aa")("vu") * cc1a("Kuv");
    b_ck("K") -= V_sumB_Alpha.block("aa")("vu") * cc1a("Kuv");

    b_ck("K") -= V_sumB_Beta.block("AA")("VU") * cc1b("KUV");
    b_ck("K") -= V_sumA_Beta.block("AA")("VU") * cc1b("KUV");

    b_ck("K") -= 0.25 * V.block("aaaa")("xyuv") * cc2aa("Kuvxy");
    b_ck("K") -= 0.25 * V.block("AAAA")("XYUV") * cc2bb("KUVXY");
    b_ck("K") -= V.block("aAaA")("xYuV") * cc2ab("KuVxY");

    // Solving the multiplier Alpha (the CI normalization condition)
    Alpha = 0.0;
    Alpha += H["vu"] * Gamma1_["uv"];
    Alpha += H["VU"] * Gamma1_["UV"];
    Alpha += V_sumA_Alpha["v,u"] * Gamma1_["uv"];
    Alpha += V_sumB_Alpha["v,u"] * Gamma1_["uv"];
    Alpha += V["V,M,U,M1"] * Gamma1_["UV"] * I["M,M1"];
    Alpha += V["m,V,m1,U"] * Gamma1_["UV"] * I["m,m1"];
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

    b_ck("K") -= 0.5 * temp.block("aa")("uv") * cc1a("Kuv");
    b_ck("K") -= 0.5 * temp.block("AA")("UV") * cc1b("KUV");
    Alpha += 0.5 * temp["uv"] * Gamma1_["uv"];
    Alpha += 0.5 * temp["UV"] * Gamma1_["UV"];
    temp.zero();

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

    b_ck("K") -= temp4.block("aaaa")("uvxy") * dlamb_aa("Kxyuv");
    b_ck("K") -= temp4.block("AAAA")("UVXY") * dlamb_bb("KXYUV");
    b_ck("K") -= temp4.block("aAaA")("uVxY") * dlamb_ab("KxYuV");
    Alpha += 2 * temp4["uvxy"] * Lambda2_["xyuv"];
    Alpha += 2 * temp4["UVXY"] * Lambda2_["XYUV"];
    Alpha += 2 * temp4["uVxY"] * Lambda2_["xYuV"];
    Alpha -= temp4["uvxy"] * Gamma2_["xyuv"];
    Alpha -= temp4["UVXY"] * Gamma2_["XYUV"];
    Alpha -= temp4["uVxY"] * Gamma2_["xYuV"];
    temp4.zero();

    if (X4_TERM) {
        b_ck("K") -=
            0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") * dlamb3_aaa("Kxyzuvw");
        b_ck("K") -=
            0.25 * V_.block("AACA")("UVMZ") * T2_.block("CAAA")("MWXY") * dlamb3_bbb("KXYZUVW");
        b_ck("K") +=
            0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") * dlamb3_aab("KxyZuvW");
        b_ck("K") +=
            0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") * dlamb3_aab("KxyZuvW");
        b_ck("K") -=
            1.00 * V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * dlamb3_aab("KxyZuvW");
        b_ck("K") +=
            0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") * dlamb3_abb("KxYZuVW");
        b_ck("K") +=
            0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") * dlamb3_abb("KxYZuVW");
        b_ck("K") -=
            1.00 * V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * dlamb3_abb("KxYZuVW");
        b_ck("K") +=
            0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * dlamb3_aaa("Kxyzuvw");
        b_ck("K") +=
            0.25 * V_.block("VAAA")("EWXY") * T2_.block("AAVA")("UVEZ") * dlamb3_bbb("KXYZUVW");
        b_ck("K") -=
            0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * dlamb3_aab("KxyZuvW");
        b_ck("K") +=
            0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * dlamb3_aab("KxyZuvW");
        b_ck("K") +=
            1.00 * V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * dlamb3_aab("KxyZuvW");
        b_ck("K") -=
            0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * dlamb3_abb("KxYZuVW");
        b_ck("K") +=
            0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * dlamb3_abb("KxYZuVW");
        b_ck("K") +=
            1.00 * V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * dlamb3_abb("KxYZuVW");
    }

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

    b_ck("K") -= temp3.block("aa")("xy") * V.block("aaaa")("xvyu") * cc1a("Kuv");
    b_ck("K") -= temp3.block("aa")("xy") * V.block("aAaA")("xVyU") * cc1b("KUV");
    b_ck("K") -= temp3.block("AA")("XY") * V.block("AAAA")("XVYU") * cc1b("KUV");
    b_ck("K") -= temp3.block("AA")("XY") * V.block("aAaA")("vXuY") * cc1a("Kuv");

    temp["uv"] += Z["mn"] * V["mvnu"];
    temp["uv"] += Z["MN"] * V["vMuN"];
    temp["uv"] += Z["ef"] * V["evfu"];
    temp["uv"] += Z["EF"] * V["vEuF"];
    temp["UV"] += Z["MN"] * V["MVNU"];
    temp["UV"] += Z["mn"] * V["mVnU"];
    temp["UV"] += Z["EF"] * V["EVFU"];
    temp["UV"] += Z["ef"] * V["eVfU"];

    b_ck("K") -= temp.block("aa")("uv") * cc1a("Kuv");
    b_ck("K") -= temp.block("AA")("UV") * cc1b("KUV");

    for (const std::string& block : {"ci"}) {
        (b_ck).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = preidx[block] + i[0];
            b.at(index) = value;
        });
    }

    outfile->Printf("Done");
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