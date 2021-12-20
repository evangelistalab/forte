/**
 * Solve the z-vector equations.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "../dsrg_mrpt2.h"
#include "helpers/timer.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/vector.h"

#include <numeric>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace ambit;
using namespace psi;

namespace forte {

int max_iter = 200;
double err = 1e-9;

void DSRG_MRPT2::set_zvec_moinfo() {
    int dim_vc = nvirt * ncore, dim_ca = ncore * na, dim_va = nvirt * na,
        dim_aa = na * (na - 1) / 2, dim_ci = ndets;
    dim = dim_vc + dim_ca + dim_va + dim_aa + dim_ci;
    preidx = {{"vc", 0},
              {"VC", 0},
              {"ca", dim_vc},
              {"CA", dim_vc},
              {"va", dim_vc + dim_ca},
              {"VA", dim_vc + dim_ca},
              {"aa", dim_vc + dim_ca + dim_va},
              {"AA", dim_vc + dim_ca + dim_va},
              {"ci", dim_vc + dim_ca + dim_va + dim_aa}};

    block_dim = {{"vc", ncore}, {"VC", ncore}, {"ca", na}, {"CA", na},
                 {"va", na},    {"VA", na},    {"aa", 0},  {"AA", 0}};
}

void DSRG_MRPT2::set_z() {
    Z = BTF_->build(CoreTensor, "Z Matrix", spin_cases({"gg"}));
    outfile->Printf("\n    Initializing Diagonal Entries of the OPDM Z ..... ");
    set_z_cc();
    set_z_vv();
    set_z_aa_diag();
    outfile->Printf("Done");
    solve_linear_iter();
}

void DSRG_MRPT2::set_w() {
    outfile->Printf("\n    Solving Entries of the EWDM W.................... ");
    W = BTF_->build(CoreTensor, "Energy weighted density matrix(Lagrangian)", spin_cases({"gg"}));
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));

    // Form Gamma_tilde
    for (const auto& pair : as_solver_->state_energies_map()) {
        const auto& state = pair.first;
        auto g1r = BTF_->build(tensor_type_, "1GRDM_ket", spin_cases({"aa"}));
        auto g2r = BTF_->build(tensor_type_, "2GRDM_ket", spin_cases({"aaaa"}));
        auto vec_ptr = x_ci.data();

        as_solver_->generalized_rdms(state, 0, vec_ptr, Gamma1_tilde, false, 1);
        as_solver_->generalized_rdms(state, 0, vec_ptr, g1r, true, 1);
        as_solver_->generalized_rdms(state, 0, vec_ptr, Gamma2_tilde, false, 2);
        as_solver_->generalized_rdms(state, 0, vec_ptr, g2r, true, 2);

        Gamma1_tilde["uv"] += g1r["uv"];
        Gamma1_tilde["UV"] += g1r["UV"];
        Gamma2_tilde["uvxy"] += g2r["uvxy"];
        Gamma2_tilde["UVXY"] += g2r["UVXY"];
        Gamma2_tilde["uVxY"] += g2r["uVxY"];
    }

    // NOTICE: w for {virtual-general}
    if (CORRELATION_TERM) {
        W["pe"] += 0.5 * sigma3_xi3["ie"] * F["ip"];
   
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
        W["jm"] += 0.5 * sigma3_xi3["ma"] * F["ja"];
        W["jm"] += 0.5 * sigma3_xi3["ia"] * V["amij"];
        W["jm"] += 0.5 * sigma3_xi3["IA"] * V["mAjI"];
        W["jm"] += 0.5 * sigma3_xi3["ia"] * V["ajim"];
        W["jm"] += 0.5 * sigma3_xi3["IA"] * V["jAmI"];

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

    W.block("cc")("nm") += 0.5 * V.block("acac")("umvn") * Gamma1_tilde.block("aa")("uv");
    W.block("cc")("nm") += 0.5 * V.block("cAcA")("mUnV") * Gamma1_tilde.block("AA")("UV");
    W.block("ac")("xm") += 0.5 * V.block("acaa")("umvx") * Gamma1_tilde.block("aa")("uv");
    W.block("ac")("xm") += 0.5 * V.block("cAaA")("mUxV") * Gamma1_tilde.block("AA")("UV");
    W["mu"] = W["um"];

    // NOTICE: w for {active-active}
    if (CORRELATION_TERM) {
        W["zw"] += 0.5 * sigma3_xi3["wa"] * F["za"];
        W["zw"] += 0.5 * sigma3_xi3["iw"] * F["iz"];
        W["zw"] += 0.5 * sigma3_xi3["ia"] * V["aziv"] * Gamma1_["wv"];
        W["zw"] += 0.5 * sigma3_xi3["IA"] * V["zAvI"] * Gamma1_["wv"];
        W["zw"] += 0.5 * sigma3_xi3["ia"] * V["auiz"] * Gamma1_["uw"];
        W["zw"] += 0.5 * sigma3_xi3["IA"] * V["uAzI"] * Gamma1_["uw"];

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

    W.block("aa")("zw") += 0.50 * H.block("aa")("vz") * Gamma1_tilde.block("aa")("wv");

    W.block("aa")("zw") += 0.25 * V_sumA_Alpha.block("aa")("uz") * Gamma1_tilde.block("aa")("uw");
    W.block("aa")("zw") += 0.25 * V_sumB_Alpha.block("aa")("uz") * Gamma1_tilde.block("aa")("uw");
    W.block("aa")("zw") += 0.25 * V_sumA_Alpha.block("aa")("zv") * Gamma1_tilde.block("aa")("wv");
    W.block("aa")("zw") += 0.25 * V_sumB_Alpha.block("aa")("zv") * Gamma1_tilde.block("aa")("wv");

    W.block("aa")("zw") += 0.125 * V.block("aaaa")("zvxy") * Gamma2_tilde.block("aaaa")("wvxy");
    W.block("aa")("zw") += 0.250 * V.block("aAaA")("zVxY") * Gamma2_tilde.block("aAaA")("wVxY");
    W.block("aa")("zw") += 0.125 * V.block("aaaa")("uzxy") * Gamma2_tilde.block("aaaa")("uwxy");
    W.block("aa")("zw") += 0.250 * V.block("aAaA")("zUxY") * Gamma2_tilde.block("aAaA")("wUxY");

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
        val1["m"] -= sigma1_xi1_xi2["ma"] * F["ma"];
        val1["m"] -= DelGam1["xu"] * T2_["muax"] * sigma1_xi1_xi2["ma"];
        val1["m"] -= DelGam1["XU"] * T2_["mUaX"] * sigma1_xi1_xi2["ma"];

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
        zmn["mn"] += 0.5 * sigma3_xi3["na"] * F["ma"];
        zmn["mn"] -= 0.5 * sigma3_xi3["ma"] * F["na"];

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
        val2["e"] += sigma1_xi1_xi2["ie"] * F["ie"];
        val2["e"] += DelGam1["xu"] * T2_["iuex"] * sigma1_xi1_xi2["ie"];
        val2["e"] += DelGam1["XU"] * T2_["iUeX"] * sigma1_xi1_xi2["ie"];

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
        zef["ef"] += 0.5 * sigma3_xi3["if"] * F["ie"];
        zef["ef"] -= 0.5 * sigma3_xi3["ie"] * F["if"];

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
        val3["w"] -=  sigma1_xi1_xi2["wa"] * F["wa"];
        val3["w"] -=  DelGam1["xu"] * T2_["wuax"] * sigma1_xi1_xi2["wa"];
        val3["w"] -=  DelGam1["XU"] * T2_["wUaX"] * sigma1_xi1_xi2["wa"];
        val3["w"] +=  sigma1_xi1_xi2["iw"] * F["iw"];
        val3["w"] +=  DelGam1["xu"] * T2_["iuwx"] * sigma1_xi1_xi2["iw"];
        val3["w"] +=  DelGam1["XU"] * T2_["iUwX"] * sigma1_xi1_xi2["iw"];

        val3["w"] += sigma2_xi3["ia"] * T2_["iuaw"] * Gamma1_["wu"];
        val3["w"] += sigma2_xi3["IA"] * T2_["uIwA"] * Gamma1_["wu"];
        val3["w"] -= sigma2_xi3["ia"] * T2_["iwax"] * Gamma1_["xw"];
        val3["w"] -= sigma2_xi3["IA"] * T2_["wIxA"] * Gamma1_["xw"];

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

double f_norm(std::vector<double> const& vec) {
    double accum = 0.0;
    for (int i = 0; i < vec.size(); ++i) {
        accum += vec[i] * vec[i];
    }
    return std::sqrt(accum);
}

double diff_f_norm(std::vector<double> const& vec1, std::vector<double> const& vec2) {
    double accum = 0.0;
    if (vec1.size() != vec2.size()) {
        exit(1);
    }
    for (int i = 0; i < vec1.size(); ++i) {
        double diff = vec1[i] - vec2[i];
        accum += diff * diff;
    }
    return std::sqrt(accum);
}

void DSRG_MRPT2::z_vector_contraction(std::vector<double> & qk_vec, std::vector<double> & y_vec) {
    BlockedTensor qk = BTF_->build(CoreTensor, "vector qk (orbital rotation) in GMRES", {"vc", "VC", "ca", "CA", "va", "VA", "aa", "AA"});
    auto qk_ci = ambit::Tensor::build(ambit::CoreTensor, "qk (ci) in GMRES", {ndets});

    for (const std::string& row : {"vc", "VC", "ca", "CA", "va", "VA", "aa", "AA"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];
        if (row != "aa" && row != "AA") {
            qk.block(row).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = pre1 + i[0] * idx1 + i[1];
                value = qk_vec.at(index);
            });
        } else {
            qk.block(row).iterate([&](const std::vector<size_t>& i, double& value) {
                int i0 = i[0] > i[1] ? i[0] : i[1], i1 = i[0] > i[1] ? i[1] : i[0];
                if (i[0] != i[1]) {
                    int index = pre1 + i0 * (i0 - 1) / 2 + i1;
                    value = qk_vec.at(index);
                }
            });
        }
    }

    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];
        (qk_ci).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = pre1 + i[0];
            value = qk_vec.at(index);
        });
    }

    BlockedTensor y = BTF_->build(CoreTensor, "y (orbital rotation) in GMRES", {"vc", "ca", "va", "aa"});
    auto y_ci = ambit::Tensor::build(ambit::CoreTensor, "y (ci) in GMRES", {ndets});

    /// MO RESPONSE -- MO RESPONSE
    // VIRTUAL-CORE
    y["em"] += Delta1["me"] * qk["em"];
    y["em"] -= V["e1,e,m1,m"] * qk["e1,m1"];
    y["em"] -= V["e,E1,m,M1"] * qk["E1,M1"];
    y["em"] -= V["m1,e,e1,m"] * qk["e1,m1"];
    y["em"] -= V["e,M1,m,E1"] * qk["E1,M1"];
    y["em"] -= F["ue"] * qk["mu"];
    y["em"] -= V["u,e,n1,m"] * qk["n1,u"];
    y["em"] -= V["e,U,m,N1"] * qk["N1,U"];
    y["em"] -= V["n1,e,u,m"] * qk["n1,u"];
    y["em"] -= V["e,N1,m,U"] * qk["N1,U"];
    y["em"] += Gamma1_["uv"] * V["v,e,n1,m"] * qk["n1,u"];
    y["em"] += Gamma1_["UV"] * V["e,V,m,N1"] * qk["N1,U"];
    y["em"] += Gamma1_["uv"] * V["n1,e,v,m"] * qk["n1,u"];
    y["em"] += Gamma1_["UV"] * V["e,N1,m,V"] * qk["N1,U"];
    y["em"] -= Gamma1_["uv"] * V["e1,e,v,m"] * qk["e1,u"];
    y["em"] -= Gamma1_["UV"] * V["e,E1,m,V"] * qk["E1,U"];
    y["em"] -= Gamma1_["uv"] * V["v,e,e1,m"] * qk["e1,u"];
    y["em"] -= Gamma1_["UV"] * V["e,V,m,E1"] * qk["E1,U"];
    y["em"] += F["um"] * qk["eu"];
    y["em"] -= V["veum"] * qk["uv"];
    y["em"] -= V["eVmU"] * qk["UV"];

    // CORE-ACTIVE
    y["mw"] += F["we"] * qk["em"];
    y["mw"] += V["e1,w,m1,m"] * qk["e1,m1"];
    y["mw"] += V["w,E1,m,M1"] * qk["E1,M1"];
    y["mw"] += V["e1,m,m1,w"] * qk["e1,m1"];
    y["mw"] += V["m,E1,w,M1"] * qk["E1,M1"];
    y["mw"] += F["uw"] * qk["mu"];
    y["mw"] -= H["vw"] * Gamma1_["uv"] * qk["mu"];
    y["mw"] -= V_sumA_Alpha["v,w"] * Gamma1_["uv"] * qk["mu"];
    y["mw"] -= V_sumB_Alpha["v,w"] * Gamma1_["uv"] * qk["mu"];
    y["mw"] -= 0.5 * V["xywv"] * Gamma2_["uvxy"] * qk["mu"];
    y["mw"] -= V["xYwV"] * Gamma2_["uVxY"] * qk["mu"];
    y["mw"] += V["u,w,n1,m"] * qk["n1,u"];
    y["mw"] += V["w,U,m,N1"] * qk["N1,U"];
    y["mw"] += V["u,m,n1,w"] * qk["n1,u"];
    y["mw"] += V["m,U,w,N1"] * qk["N1,U"];
    y["mw"] -= Gamma1_["uv"] * V["v,w,n1,m"] * qk["n1,u"];
    y["mw"] -= Gamma1_["UV"] * V["w,V,m,N1"] * qk["N1,U"];
    y["mw"] -= Gamma1_["uv"] * V["v,m,n1,w"] * qk["n1,u"];
    y["mw"] -= Gamma1_["UV"] * V["m,V,w,N1"] * qk["N1,U"];
    y["mw"] += Gamma1_["uv"] * V["e1,w,v,m"] * qk["e1,u"];
    y["mw"] += Gamma1_["UV"] * V["w,E1,m,V"] * qk["E1,U"];
    y["mw"] += Gamma1_["uv"] * V["e1,m,v,w"] * qk["e1,u"];
    y["mw"] += Gamma1_["UV"] * V["m,E1,w,V"] * qk["E1,U"];
    y["mw"] += V["vwum"] * qk["uv"];
    y["mw"] += V["wVmU"] * qk["UV"];
    y["mw"] -= V["e1,u,m1,m"] * Gamma1_["uw"] * qk["e1,m1"];
    y["mw"] -= V["u,E1,m,M1"] * Gamma1_["uw"] * qk["E1,M1"];
    y["mw"] -= V["e1,m,m1,u"] * Gamma1_["uw"] * qk["e1,m1"];
    y["mw"] -= V["m,E1,u,M1"] * Gamma1_["uw"] * qk["E1,M1"];
    y["mw"] -= F["m,n1"] * qk["n1,w"];
    y["mw"] -= V["u,v,n1,m"] * Gamma1_["wv"] * qk["n1,u"];
    y["mw"] -= V["v,U,m,N1"] * Gamma1_["wv"] * qk["N1,U"];
    y["mw"] -= V["u,m,n1,v"] * Gamma1_["wv"] * qk["n1,u"];
    y["mw"] -= V["m,U,v,N1"] * Gamma1_["wv"] * qk["N1,U"];
    y["mw"] += H["m,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["mw"] += V_sumA_Alpha["m,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["mw"] += V_sumB_Alpha["m,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["mw"] += 0.5 * V["x,y,n1,m"] * Gamma2_["u,w,x,y"] * qk["n1,u"];
    y["mw"] += V["y,X,m,N1"] * Gamma2_["w,U,y,X"] * qk["N1,U"];
    y["mw"] += V["m,y,n1,v"] * Gamma2_["u,v,w,y"] * qk["n1,u"];
    y["mw"] += V["m,Y,n1,V"] * Gamma2_["u,V,w,Y"] * qk["n1,u"];
    y["mw"] += V["m,Y,v,N1"] * Gamma2_["v,U,w,Y"] * qk["N1,U"];
    y["mw"] -= H["m,e1"] * Gamma1_["uw"] * qk["e1,u"];
    y["mw"] -= V_sumA_Alpha["e1,m"] * Gamma1_["uw"] * qk["e1,u"];
    y["mw"] -= V_sumB_Alpha["e1,m"] * Gamma1_["uw"] * qk["e1,u"];
    y["mw"] -= 0.5 * V["e1,m,x,y"] * Gamma2_["u,w,x,y"] * qk["e1,u"];
    y["mw"] -= V["m,E1,y,X"] * Gamma2_["w,U,y,X"] * qk["E1,U"];
    y["mw"] -= V["e1,v,m,y"] * Gamma2_["u,v,w,y"] * qk["e1,u"];
    y["mw"] -= V["e1,V,m,Y"] * Gamma2_["u,V,w,Y"] * qk["e1,u"];
    y["mw"] -= V["v,E1,m,Y"] * Gamma2_["v,U,w,Y"] * qk["E1,U"];
    y["mw"] -= V["a1,v,u1,m"] * Gamma1_["wv"] * qk["u1,a1"];
    y["mw"] -= V["v,A1,m,U1"] * Gamma1_["wv"] * qk["U1,A1"];
    y["mw"] -= F["vm"] * qk["wv"];

    // VIRTUAL-ACTIVE
    y["ew"] += F["m1,w"] * qk["e,m1"];
    y["ew"] += H["vw"] * Gamma1_["uv"] * qk["eu"];
    y["ew"] += V_sumA_Alpha["v,w"] * Gamma1_["uv"] * qk["eu"];
    y["ew"] += V_sumB_Alpha["v,w"] * Gamma1_["uv"] * qk["eu"];
    y["ew"] += 0.5 * V["xywv"] * Gamma2_["uvxy"] * qk["eu"];
    y["ew"] += V["xYwV"] * Gamma2_["uVxY"] * qk["eu"];
    y["ew"] -= V["e1,u,m1,e"] * Gamma1_["uw"] * qk["e1,m1"];
    y["ew"] -= V["u,E1,e,M1"] * Gamma1_["uw"] * qk["E1,M1"];
    y["ew"] -= V["e1,e,m1,u"] * Gamma1_["uw"] * qk["e1,m1"];
    y["ew"] -= V["e,E1,u,M1"] * Gamma1_["uw"] * qk["E1,M1"];
    y["ew"] -= V["u,v,n1,e"] * Gamma1_["wv"] * qk["n1,u"];
    y["ew"] -= V["v,U,e,N1"] * Gamma1_["wv"] * qk["N1,U"];
    y["ew"] -= V["u,e,n1,v"] * Gamma1_["wv"] * qk["n1,u"];
    y["ew"] -= V["e,U,v,N1"] * Gamma1_["wv"] * qk["N1,U"];
    y["ew"] += H["e,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["ew"] += V_sumA_Alpha["e,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["ew"] += V_sumB_Alpha["e,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["ew"] += 0.5 * V["x,y,n1,e"] * Gamma2_["u,w,x,y"] * qk["n1,u"];
    y["ew"] += V["y,X,e,N1"] * Gamma2_["w,U,y,X"] * qk["N1,U"];
    y["ew"] += V["e,y,n1,v"] * Gamma2_["u,v,w,y"] * qk["n1,u"];
    y["ew"] += V["e,Y,n1,V"] * Gamma2_["u,V,w,Y"] * qk["n1,u"];
    y["ew"] += V["e,Y,v,N1"] * Gamma2_["v,U,w,Y"] * qk["N1,U"];
    y["ew"] -= V["u1,e,a1,v"] * Gamma1_["wv"] * qk["u1,a1"];
    y["ew"] -= V["e,U1,v,A1"] * Gamma1_["wv"] * qk["U1,A1"];
    y["ew"] -= F["ve"] * qk["wv"];
    y["ew"] -= H["e,e1"] * Gamma1_["uw"] * qk["e1,u"];
    y["ew"] -= V_sumA_Alpha["e,e1"] * Gamma1_["uw"] * qk["e1,u"];
    y["ew"] -= V_sumB_Alpha["e,e1"] * Gamma1_["uw"] * qk["e1,u"];
    y["ew"] -= 0.5 * V["e1,e,x,y"] * Gamma2_["u,w,x,y"] * qk["e1,u"];
    y["ew"] -= V["e,E1,y,X"] * Gamma2_["w,U,y,X"] * qk["E1,U"];
    y["ew"] -= V["e,y,e1,v"] * Gamma2_["u,v,w,y"] * qk["e1,u"];
    y["ew"] -= V["e,Y,e1,V"] * Gamma2_["u,V,w,Y"] * qk["e1,u"];
    y["ew"] -= V["e,Y,v,E1"] * Gamma2_["v,U,w,Y"] * qk["E1,U"];

    // ACTIVE-ACTIVE
    {
        BlockedTensor temp_y = BTF_->build(CoreTensor, "temporal matrix for y{aa} symmetrization", spin_cases({"aa"}));
        temp_y["wz"] -= V["e1,u,m1,w"] * Gamma1_["uz"] * qk["e1,m1"];
        temp_y["wz"] -= V["u,E1,w,M1"] * Gamma1_["uz"] * qk["E1,M1"];
        temp_y["wz"] -= V["e1,w,m1,u"] * Gamma1_["uz"] * qk["e1,m1"];
        temp_y["wz"] -= V["w,E1,u,M1"] * Gamma1_["uz"] * qk["E1,M1"];
        temp_y["wz"] -= F["w,n1"] * qk["n1,z"];
        temp_y["wz"] -= V["u,v,n1,w"] * Gamma1_["zv"] * qk["n1,u"];
        temp_y["wz"] -= V["v,U,w,N1"] * Gamma1_["zv"] * qk["N1,U"];
        temp_y["wz"] -= V["u,w,n1,v"] * Gamma1_["zv"] * qk["n1,u"];
        temp_y["wz"] -= V["w,U,v,N1"] * Gamma1_["zv"] * qk["N1,U"];
        temp_y["wz"] += H["w,n1"] * Gamma1_["uz"] * qk["n1,u"];
        temp_y["wz"] += V_sumA_Alpha["w,n1"] * Gamma1_["uz"] * qk["n1,u"];
        temp_y["wz"] += V_sumB_Alpha["w,n1"] * Gamma1_["uz"] * qk["n1,u"];
        temp_y["wz"] += 0.5 * V["x,y,n1,w"] * Gamma2_["u,z,x,y"] * qk["n1,u"];
        temp_y["wz"] += V["y,X,w,N1"] * Gamma2_["z,U,y,X"] * qk["N1,U"];
        temp_y["wz"] += V["w,y,n1,v"] * Gamma2_["u,v,z,y"] * qk["n1,u"];
        temp_y["wz"] += V["w,Y,n1,V"] * Gamma2_["u,V,z,Y"] * qk["n1,u"];
        temp_y["wz"] += V["w,Y,v,N1"] * Gamma2_["v,U,z,Y"] * qk["N1,U"];
        temp_y["wz"] -= H["w,e1"] * Gamma1_["uz"] * qk["e1,u"];
        temp_y["wz"] -= V_sumA_Alpha["e1,w"] * Gamma1_["uz"] * qk["e1,u"];
        temp_y["wz"] -= V_sumB_Alpha["e1,w"] * Gamma1_["uz"] * qk["e1,u"];
        temp_y["wz"] -= 0.5 * V["e1,w,x,y"] * Gamma2_["u,z,x,y"] * qk["e1,u"];
        temp_y["wz"] -= V["w,E1,y,X"] * Gamma2_["z,U,y,X"] * qk["E1,U"];
        temp_y["wz"] -= V["e1,v,w,y"] * Gamma2_["u,v,z,y"] * qk["e1,u"];
        temp_y["wz"] -= V["e1,V,w,Y"] * Gamma2_["u,V,z,Y"] * qk["e1,u"];
        temp_y["wz"] -= V["v,E1,w,Y"] * Gamma2_["v,U,z,Y"] * qk["E1,U"];
        temp_y["wz"] -= V["a1,v,u1,w"] * Gamma1_["zv"] * qk["u1,a1"];
        temp_y["wz"] -= V["v,A1,w,U1"] * Gamma1_["zv"] * qk["U1,A1"];

        y["wz"] += temp_y["wz"];
        y["zw"] -= temp_y["wz"];  
        y["wz"] += Delta1["zw"] * qk["wz"];   
    }

    /// MO RESPONSE -- CI EQUATION
    // Form contraction between qk_ci and ci, cc1, cc2
    auto cc1_qkci = BTF_->build(CoreTensor, "cc1 * qk_ci", spin_cases({"aa"}));
    auto cc2_qkci = BTF_->build(CoreTensor, "cc2 * qk_ci", spin_cases({"aaaa"}));
    for (const auto& pair : as_solver_->state_energies_map()) {
        const auto& state = pair.first;
        auto g1r = BTF_->build(tensor_type_, "1GRDM_ket", spin_cases({"aa"}));
        auto g2r = BTF_->build(tensor_type_, "2GRDM_ket", spin_cases({"aaaa"}));
        auto vec_ptr = qk_ci.data();
        as_solver_->generalized_rdms(state, 0, vec_ptr, cc1_qkci, false, 1);
        as_solver_->generalized_rdms(state, 0, vec_ptr, g1r, true, 1);
        as_solver_->generalized_rdms(state, 0, vec_ptr, cc2_qkci, false, 2);
        as_solver_->generalized_rdms(state, 0, vec_ptr, g2r, true, 2);

        cc1_qkci["uv"] += g1r["uv"];
        cc1_qkci["UV"] += g1r["UV"];
        cc2_qkci["uvxy"] += g2r["uvxy"];
        cc2_qkci["UVXY"] += g2r["UVXY"];
        cc2_qkci["uVxY"] += g2r["uVxY"];
    }

    double ci_qk_dot;
    ci_qk_dot = ci("I") * qk_ci("I");

    // VIRTUAL-CORE
    y.block("vc")("em") -= ci_qk_dot * H.block("vc")("em");
    y.block("vc")("em") -= ci_qk_dot * V_sumA_Alpha.block("cv")("me");
    y.block("vc")("em") -= ci_qk_dot * V_sumB_Alpha.block("cv")("me");
    y.block("vc")("em") -= 0.5 * V.block("avac")("veum") * cc1_qkci.block("aa")("uv");
    y.block("vc")("em") -= 0.5 * V.block("vAcA")("eVmU") * cc1_qkci.block("AA")("UV");

    // CORE-ACTIVE
    y.block("ca")("mw") -= 0.50 * H.block("ac")("vm") * cc1_qkci.block("aa")("wv");
    y.block("ca")("mw") -= 0.50 * V_sumA_Alpha.block("ac")("um") * cc1_qkci.block("aa")("uw");
    y.block("ca")("mw") -= 0.50 * V_sumB_Alpha.block("ac")("um") * cc1_qkci.block("aa")("uw");
    y.block("ca")("mw") -= 0.25 * V.block("aaca")("xymv") * cc2_qkci.block("aaaa")("wvxy");
    y.block("ca")("mw") -= 0.50 * V.block("aAcA")("xYmV") * cc2_qkci.block("aAaA")("wVxY");

    y.block("ca")("mw") += ci_qk_dot * H.block("ac")("wm");
    y.block("ca")("mw") += ci_qk_dot * V_sumA_Alpha.block("ca")("mw");
    y.block("ca")("mw") += ci_qk_dot * V_sumB_Alpha.block("ca")("mw");
    y.block("ca")("mw") += 0.5 * V.block("aaac")("vwum") * cc1_qkci.block("aa")("uv");
    y.block("ca")("mw") += 0.5 * V.block("aAcA")("wVmU") * cc1_qkci.block("AA")("UV");

    // VIRTUAL-ACTIVE
    y.block("va")("ew") -= 0.50 * H.block("av")("ve") * cc1_qkci.block("aa")("wv");
    y.block("va")("ew") -= 0.50 * V_sumA_Alpha.block("av")("ue") * cc1_qkci.block("aa")("uw");
    y.block("va")("ew") -= 0.50 * V_sumB_Alpha.block("av")("ue") * cc1_qkci.block("aa")("uw");
    y.block("va")("ew") -= 0.25 * V.block("vaaa")("evxy") * cc2_qkci.block("aaaa")("wvxy");
    y.block("va")("ew") -= 0.50 * V.block("vAaA")("eVxY") * cc2_qkci.block("aAaA")("wVxY");

    // ACTIVE-ACTIVE
    {
        BlockedTensor temp_y = BTF_->build(CoreTensor, "temporal matrix for y{aa} symmetrization", spin_cases({"aa"}));
        temp_y.block("aa")("wz") -= 0.50 * H.block("aa")("vw") * cc1_qkci.block("aa")("zv");
        temp_y.block("aa")("wz") -= 0.50 * V_sumA_Alpha.block("aa")("uw") * cc1_qkci.block("aa")("uz");
        temp_y.block("aa")("wz") -= 0.50 * V_sumB_Alpha.block("aa")("uw") * cc1_qkci.block("aa")("uz");
        temp_y.block("aa")("wz") -= 0.25 * V.block("aaaa")("wvxy") * cc2_qkci.block("aaaa")("zvxy");
        temp_y.block("aa")("wz") -= 0.50 * V.block("aAaA")("wVxY") * cc2_qkci.block("aAaA")("zVxY");

        y["wz"] += temp_y["wz"];
        y["zw"] -= temp_y["wz"];
    }

    /// CI EQUATION -- MO RESPONSE
    //  virtual-core
    //  contribution from the multiplier Alpha
    y_ci("K") += -4 * ci("K") * V.block("vaca")("exmy") * Gamma1_.block("aa")("xy") * qk.block("vc")("em");
    y_ci("K") += -4 * ci("K") * V.block("vAcA")("eXmY") * Gamma1_.block("AA")("XY") * qk.block("vc")("em");
    y_ci("K") += -4 * ci("K") * V.block("VACA")("EXMY") * Gamma1_.block("AA")("XY") * qk.block("VC")("EM");
    y_ci("K") += -4 * ci("K") * V.block("aVaC")("xEyM") * Gamma1_.block("aa")("xy") * qk.block("VC")("EM");

    //  core-active
    //  contribution from the multiplier Alpha
    y_ci("K") += -4 * ci("K") * V.block("aaca")("uynx") * Gamma1_.block("aa")("xy") * qk.block("ca")("nu");
    y_ci("K") += -4 * ci("K") * V.block("aAcA")("uYnX") * Gamma1_.block("AA")("XY") * qk.block("ca")("nu");
    y_ci("K") += 4 * ci("K") * H.block("ac")("vn") * Gamma1_.block("aa")("uv") * qk.block("ca")("nu");
    y_ci("K") += 4 * ci("K") * V_sumA_Alpha.block("ac")("vn") * Gamma1_.block("aa")("uv") * qk.block("ca")("nu");
    y_ci("K") += 4 * ci("K") * V_sumB_Alpha.block("ac")("vn") * Gamma1_.block("aa")("uv") * qk.block("ca")("nu");
    y_ci("K") += 2 * ci("K") * V.block("aaca")("xynv") * Gamma2_.block("aaaa")("uvxy") * qk.block("ca")("nu");
    y_ci("K") += 4 * ci("K") * V.block("aAcA")("xYnV") * Gamma2_.block("aAaA")("uVxY") * qk.block("ca")("nu");

    //  beta
    y_ci("K") += -4 * ci("K") * V.block("AACA")("UYNX") * Gamma1_.block("AA")("XY") * qk.block("CA")("NU");
    y_ci("K") += -4 * ci("K") * V.block("aAaC")("yUxN") * Gamma1_.block("aa")("xy") * qk.block("CA")("NU");
    y_ci("K") += 4 * ci("K") * H.block("AC")("VN") * Gamma1_.block("AA")("UV") * qk.block("CA")("NU");
    y_ci("K") += 4 * ci("K") * V_sumB_Beta.block("AC")("VN") * Gamma1_.block("AA")("UV") * qk.block("CA")("NU");
    y_ci("K") += 4 * ci("K") * V_sumA_Beta.block("AC")("VN") * Gamma1_.block("AA")("UV") * qk.block("CA")("NU");
    y_ci("K") += 2 * ci("K") * V.block("AACA")("XYNV") * Gamma2_.block("AAAA")("UVXY") * qk.block("CA")("NU");
    y_ci("K") += 4 * ci("K") * V.block("aAaC")("xYvN") * Gamma2_.block("aAaA")("vUxY") * qk.block("CA")("NU");

    //  virtual-active
    //  contribution from the multiplier Alpha
    y_ci("K") += -4 * ci("K") * H.block("av")("ve") * Gamma1_.block("aa")("uv") * qk.block("va")("eu");
    y_ci("K") += -4 * ci("K") * V_sumA_Alpha.block("av")("ve") * Gamma1_.block("aa")("uv") * qk.block("va")("eu");
    y_ci("K") += -4 * ci("K") * V_sumB_Alpha.block("av")("ve") * Gamma1_.block("aa")("uv") * qk.block("va")("eu");
    y_ci("K") += -2 * ci("K") * V.block("vaaa")("evxy") * Gamma2_.block("aaaa")("uvxy") * qk.block("va")("eu");
    y_ci("K") += -4 * ci("K") * V.block("vAaA")("eVxY") * Gamma2_.block("aAaA")("uVxY") * qk.block("va")("eu");

    // beta
    y_ci("K") += -4 * ci("K") * H.block("AV")("VE") * Gamma1_.block("AA")("UV") * qk.block("VA")("EU");
    y_ci("K") += -4 * ci("K") * V_sumB_Beta.block("AV")("VE") * Gamma1_.block("AA")("UV") * qk.block("VA")("EU");
    y_ci("K") += -4 * ci("K") * V_sumA_Beta.block("AV")("VE") * Gamma1_.block("AA")("UV") * qk.block("VA")("EU");
    y_ci("K") += -2 * ci("K") * V.block("VAAA")("EVXY") * Gamma2_.block("AAAA")("UVXY") * qk.block("VA")("EU");
    y_ci("K") += -4 * ci("K") * V.block("aVaA")("vExY") * Gamma2_.block("aAaA")("vUxY") * qk.block("VA")("EU");

    //  active-active
    //  contribution from the multiplier Alpha
    y_ci("K") += -2 * ci("K") * V.block("aaaa")("uyvx") * Gamma1_.block("aa")("xy") * qk.block("aa")("uv");
    y_ci("K") += -2 * ci("K") * V.block("aAaA")("uYvX") * Gamma1_.block("AA")("XY") * qk.block("aa")("uv");

    // beta
    y_ci("K") += -2 * ci("K") * V.block("AAAA")("UYVX") * Gamma1_.block("AA")("XY") * qk.block("AA")("UV");
    y_ci("K") += -2 * ci("K") * V.block("aAaA")("yUxV") * Gamma1_.block("aa")("xy") * qk.block("AA")("UV");

    /// CI EQUATION -- MO RESPONSE
    //  Call the generalized sigma function to complete the contraction
    for (const auto& pair: as_solver_->state_space_size_map()) {
        const auto& state = pair.first;
        std::map<std::string, double> block_factor1;
        std::map<std::string, double> block_factor2;
        block_factor1["aa"] = 1.0;
        block_factor1["AA"] = 1.0;
        block_factor2["aaaa"] = 1.0;
        block_factor2["aAaA"] = 1.0;
        block_factor2["AAAA"] = 1.0;

        auto sym_1 = BTF_->build(CoreTensor, "symmetrized 1-body tensor", spin_cases({"aa"}));
        auto sym_2 = BTF_->build(CoreTensor, "symmetrized 2-body tensor", spin_cases({"aaaa"}));
        {
            auto temp_1 = BTF_->build(CoreTensor, "1-body intermediate tensor", spin_cases({"aa"}));
            // virtual-core
            temp_1["uv"] += 2 * V["eumv"] * qk["em"];
            temp_1["UV"] += 2 * V["eUmV"] * qk["em"];
            // beta
            temp_1["UV"] += 2 * V["EUMV"] * qk["EM"];
            temp_1["uv"] += 2 * V["uEvM"] * qk["EM"];
            // core-active
            temp_1["xy"] += 2 * V["uynx"] * qk["nu"];
            temp_1["XY"] += 2 * V["uYnX"] * qk["nu"];
            temp_1["uv"] -= 2 * H["vn"] * qk["nu"];
            temp_1["uv"] -= 2 * V_sumA_Alpha["vn"] * qk["nu"];
            temp_1["uv"] -= 2 * V_sumB_Alpha["vn"] * qk["nu"];
            // beta
            temp_1["XY"] += 2 * V["UYNX"] * qk["NU"];
            temp_1["xy"] += 2 * V["yUxN"] * qk["NU"];
            temp_1["UV"] -= 2 * H["VN"] * qk["NU"];
            temp_1["UV"] -= 2 * V_sumB_Beta["VN"] * qk["NU"];
            temp_1["UV"] -= 2 * V_sumA_Beta["VN"] * qk["NU"];
            // virtual-active
            temp_1["uv"] += 2 * H["ve"] * qk["eu"];
            temp_1["uv"] += 2 * V_sumA_Alpha["ve"] * qk["eu"];
            temp_1["uv"] += 2 * V_sumB_Alpha["ve"] * qk["eu"];
            // beta
            temp_1["UV"] += 2 * H["VE"] * qk["EU"];
            temp_1["UV"] += 2 * V_sumB_Beta["VE"] * qk["EU"];
            temp_1["UV"] += 2 * V_sumA_Beta["VE"] * qk["EU"];
            // active-active
            temp_1["xy"] += V["uyvx"] * qk["uv"];
            temp_1["XY"] += V["uYvX"] * qk["uv"];
            // beta
            temp_1["XY"] += V["UYVX"] * qk["UV"];
            temp_1["xy"] += V["yUxV"] * qk["UV"];

            /// Symmetrization
            //  α α
            sym_1["uv"] += temp_1["uv"];
            sym_1["uv"] += temp_1["vu"];
            /// Symmetrization
            //  β β
            sym_1["UV"] += temp_1["UV"];
            sym_1["UV"] += temp_1["VU"];
        }
        {
            auto temp_2 = BTF_->build(CoreTensor, "2-body intermediate tensor", spin_cases({"aaaa"}));
            // core-active
            temp_2["uvxy"] -= V["xynv"] * qk["nu"];
            temp_2["uVxY"] -= 2 * V["xYnV"] * qk["nu"];
            // beta
            temp_2["UVXY"] -= V["XYNV"] * qk["NU"];
            temp_2["vUxY"] -= 2 * V["xYvN"] * qk["NU"];
            // virtual-active
            temp_2["uvxy"] += V["evxy"] * qk["eu"];
            temp_2["uVxY"] += 2 * V["eVxY"] * qk["eu"];
            // beta
            temp_2["UVXY"] += V["EVXY"] * qk["EU"];
            temp_2["vUxY"] += 2 * V["vExY"] * qk["EU"];

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
        as_solver_->add_sigma_kbody(state, 0, sym_1, block_factor1, y_ci.data());
        as_solver_->add_sigma_kbody(state, 0, sym_2, block_factor2, y_ci.data());
    }

    /// CI EQUATION -- CI EQUATION
    y_ci("K") += H.block("cc")("mn") * I.block("cc")("mn") * qk_ci("K");
    y_ci("K") += H.block("CC")("MN") * I.block("CC")("MN") * qk_ci("K");
    y_ci("K") += 0.5 * V_sumA_Alpha["m,m1"] * I["m,m1"] * qk_ci("K");
    y_ci("K") += 0.5 * V_sumB_Beta["M,M1"] * I["M,M1"] * qk_ci("K");
    y_ci("K") += V_sumB_Alpha["m,m1"] * I["m,m1"] * qk_ci("K");
    y_ci("K") -= (Eref_ - Enuc_ - Efrzc_) * qk_ci("K");
    // NOTICE : Efrzc_ is subtracted since it has been counted once in the generalized_sigma.
    y_ci("K") -= Efrzc_ * qk_ci("K");

    //  Call the generalized sigma function to complete the contraction
    //  sum_{J} <I| H |J> x_J where H is the active space Hamiltonian, which includes
    //      cc1a("IJuv") * fa("uv") * x("J")
    //      cc1b("IJuv") * fb("uv") * x("J")
    //      0.25 * cc2aa("IJuvxy") * vaa("uvxy") * x("J")
    //      cc2ab("IJuvxy") * vab("uvxy") * x("J")
    //      0.25 * cc2bb("IJuvxy") * vbb("uvxy") * x("J")
    //  where fa and fb are core Fock matrices
    for (const auto& pair: as_solver_->state_space_size_map()) {
        const auto& state = pair.first;
        psi::SharedVector svq(new psi::Vector(ndets));
        psi::SharedVector svy(new psi::Vector(ndets));

        for (int i = 0; i < ndets; ++i) {
            svq->set(i, qk_ci.data()[i]);
            svy->set(i, y_ci.data()[i]);
        }
        as_solver_->generalized_sigma(state, svq, svy);
        for (int i = 0; i < ndets; ++i) {
            y_ci.data()[i] += svy->get(i);
        }
    }

    /// Fill the y (y = A * qk) and pass it to the GMRES solver
    for (const std::string& row : {"vc", "ca", "va", "aa"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];
        if (row != "aa") {
            y.block(row).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = pre1 + i[0] * idx1 + i[1];
                y_vec.at(index) = value;
            });
        } else {
            y.block(row).iterate([&](const std::vector<size_t>& i, double& value) {
                if (i[0] > i[1]) {
                    int index = pre1 + i[0] * (i[0] - 1) / 2 + i[1];
                    y_vec.at(index)  = value;
                }
            });
        }
    }

    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];
        (y_ci).iterate([&](const std::vector<size_t>& i, double& value) {   
            y_vec.at(pre1 + i[0]) = value;
        });
    }
}

void DSRG_MRPT2::set_preconditioner(std::vector<double> & D) {
    BlockedTensor D_mo = BTF_->build(CoreTensor, "Preconditioner (orbital rotation) in GMRES", {"vc", "ca", "va", "aa"});
    D_mo["e,m"] += Delta1["m,e"];

    // VIRTUAL-CORE
    D_mo["e,m"] -= V["e1,e,m1,m"] * I["e1,e"] * I["m1,m"];
    D_mo["e,m"] -= V["m1,e,e1,m"] * I["e1,e"] * I["m1,m"];

    // CORE-ACTIVE
    D_mo["m,w"] += F["uw"] * one_vec["m"] * I["uw"];
    D_mo["m,w"] -= H["vw"] * Gamma1_["wv"] * one_vec["m"];
    D_mo["m,w"] -= V_sumA_Alpha["v,w"] * Gamma1_["wv"] * one_vec["m"];
    D_mo["m,w"] -= V_sumB_Alpha["v,w"] * Gamma1_["wv"] * one_vec["m"];
    D_mo["m,w"] -= 0.5 * V["xywv"] * Gamma2_["wvxy"] * one_vec["m"];
    D_mo["m,w"] -= V["xYwV"] * Gamma2_["wVxY"] * one_vec["m"];
    D_mo["m,w"] += V["u,w,n1,m"] * I["m,n1"] * I["wu"];
    D_mo["m,w"] += V["u,m,n1,w"] * I["m,n1"] * I["wu"];
    D_mo["m,w"] -= Gamma1_["wv"] * V["v,w,n1,m"] * I["m,n1"];
    D_mo["m,w"] -= Gamma1_["wv"] * V["v,m,n1,w"] * I["m,n1"];
    D_mo["m,w"] -= F["m,n1"] * one_vec["w"] * I["m,n1"];
    D_mo["m,w"] -= V["w,v,n1,m"] * Gamma1_["wv"] * I["m,n1"];
    D_mo["m,w"] -= V["w,m,n1,v"] * Gamma1_["wv"] * I["m,n1"];
    D_mo["m,w"] += H["m,n1"] * Gamma1_["uw"] * I["wu"] * I["m,n1"];
    D_mo["m,w"] += V_sumA_Alpha["m,n1"] * Gamma1_["uw"] * I["m,n1"] * I["wu"];
    D_mo["m,w"] += V_sumB_Alpha["m,n1"] * Gamma1_["uw"] * I["m,n1"] * I["wu"];
    D_mo["m,w"] += 0.5 * V["x,y,n1,m"] * Gamma2_["u,w,x,y"] * I["m,n1"] * I["wu"];
    D_mo["m,w"] += V["m,y,n1,v"] * Gamma2_["u,v,w,y"] * I["m,n1"] * I["wu"];
    D_mo["m,w"] += V["m,Y,n1,V"] * Gamma2_["u,V,w,Y"] * I["m,n1"] * I["wu"];

    // VIRTUAL-ACTIVE
    D_mo["e,w"] += H["vw"] * Gamma1_["wv"] * one_vec["e"];
    D_mo["e,w"] += V_sumA_Alpha["v,w"] * Gamma1_["wv"] * one_vec["e"];
    D_mo["e,w"] += V_sumB_Alpha["v,w"] * Gamma1_["wv"] * one_vec["e"];
    D_mo["e,w"] += 0.5 * V["xywv"] * Gamma2_["wvxy"] * one_vec["e"];
    D_mo["e,w"] += V["xYwV"] * Gamma2_["wVxY"] * one_vec["e"];
    D_mo["e,w"] -= H["e,e1"] * Gamma1_["uw"] * I["e,e1"] * I["uw"];
    D_mo["e,w"] -= V_sumA_Alpha["e,e1"] * Gamma1_["uw"] * I["e,e1"] * I["uw"];
    D_mo["e,w"] -= V_sumB_Alpha["e,e1"] * Gamma1_["uw"] * I["e,e1"] * I["uw"];
    D_mo["e,w"] -= 0.5 * V["e1,e,x,y"] * Gamma2_["u,w,x,y"] * I["e,e1"] * I["uw"];
    D_mo["e,w"] -= V["e,y,e1,v"] * Gamma2_["u,v,w,y"] * I["e,e1"] * I["uw"];
    D_mo["e,w"] -= V["e,Y,e1,V"] * Gamma2_["u,V,w,Y"] * I["e,e1"] * I["uw"];

    // ACTIVE-ACTIVE
    D_mo["w,z"] += Delta1["zw"] * one_vec["w"];
    D_mo["w,z"] -= V["z,v,u1,w"] * Gamma1_["zv"] * I["u1,w"];
    D_mo["w,z"] += V["a1,v,w,z"] * Gamma1_["wv"] * I["z,a1"];

    for (const std::string& row : {"vc", "ca", "va", "aa"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];
        if (row != "aa") {
            D_mo.block(row).iterate([&](const std::vector<size_t>& i, double& value) {
                if (std::fabs(value) > err) {
                    int index = pre1 + i[0] * idx1 + i[1];
                    D.at(index) = 1.0 / value;
                }
            });
        } else {
            D_mo.block(row).iterate([&](const std::vector<size_t>& i, double& value) {
                if (std::fabs(value) > err) { 
                    if (i[0] > i[1]) {
                        int index = pre1 + i[0] * (i[0] - 1) / 2 + i[1];
                        D.at(index)  = 1.0 / value;
                    }
                }
            });
        }
    }

    // Attention :  d_ci are the approximate preconditioner components for the CI part
    double d_ci = 0.0;
    d_ci += H.block("cc")("mn") * I.block("cc")("mn");
    d_ci += H.block("CC")("MN") * I.block("CC")("MN");
    d_ci += 0.5 * V_sumA_Alpha["m,m1"] * I["m,m1"];
    d_ci += 0.5 * V_sumB_Beta["M,M1"] * I["M,M1"];
    d_ci += V_sumB_Alpha["m,m1"] * I["m,m1"];
    d_ci -= Eref_ - Enuc_ - Efrzc_;

    if (std::fabs(d_ci) > err) {
        double value = 1.0 / d_ci;
        int idx = preidx["ci"];
        for (int i = 0; i < ndets; ++i) {    
            D.at(idx + i) = value;
        }
    }
}

void DSRG_MRPT2::gmres_solver(std::vector<double> & x_new) {
    outfile->Printf("\n    Solving the linear system ....................... ");
    int iters;
    std::vector<double> x_old(dim);
    x_old = x_new;
    std::vector<double> r(dim);
    std::vector<double> q(max_iter * dim, 0.0);
    std::vector<double> h((max_iter + 1) * max_iter, 0.0);
    std::vector<double> bh(max_iter + 1, 0.0);
    // D is a Jacobi preconditioner
    std::vector<double> D(dim, 1.0);

    set_preconditioner(D);

    for (int i = 0; i < b.size(); ++i) {
        b[i] *= D[i];
    }

    z_vector_contraction(x_old, r);

    for (int i = 0; i < r.size(); ++i) {
        r[i] = b[i] - D[i] * r[i];
    }

    bh[0] = f_norm(r);

    for (int j = 0; j < dim; ++j) {
        // index here : i * dim + j, where i = 0
        q[j] = r[j] / bh[0];
    }

    std::vector<double> y_vec(dim, 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        if (diff_f_norm(x_old, x_new) < err && iter > 2) {
            break;
        }
        iters = iter + 1;
        x_old = x_new;

        std::vector<double> qk_vec(dim);
        for (int i = 0; i < dim; ++i) {
            qk_vec.at(i) = q[iter * dim + i];
        }

        z_vector_contraction(qk_vec, y_vec);

        for (int i = 0; i < y_vec.size(); ++i) {
            y_vec[i] *= D[i];
        }

        for (int i = 0; i < iter+1; ++i) {
            h[i + iter * (max_iter + 1)] = C_DDOT(dim, &q[i * dim], 1, &y_vec[0], 1);
            for (int j = 0; j < dim; ++j) {
                y_vec[j] -= h[i + iter * (max_iter + 1)] * q[i * dim + j];
            }
        }

        h[(iter + 1) + iter * (max_iter + 1)] = f_norm(y_vec);
        bool condition = (std::fabs(h[(iter + 1) + iter * (max_iter + 1)]) < 1e-10) || (iter == max_iter - 1);

        std::vector<double> ck(max_iter + 1, 0.0);
        ck = bh;

        int lwork = 2 * max_iter;
        std::vector<double> work(lwork);

        std::vector<double> h_sub((iter + 2) * (iter + 1), 0.0);
        for (int i = 0; i < iter + 2; ++i) {
            for (int j = 0; j < iter + 1; ++j) {
                h_sub[i + j * (iter + 2)] = h[i + j * (max_iter + 1)];
            }
        }

        C_DGELS('n', iter+2, iter+1, 1, &h_sub[0], iter+2, &ck[0], iter+2, &work[0], lwork);

        if (!condition) {
            for (int j = 0; j < dim; ++j) {
                q[(iter + 1) * dim + j] = y_vec[j] / h[(iter + 1) + iter * (max_iter + 1)];
            }
            C_DGEMV('t', iter+2, dim, 1.0, &(q[0]), dim, &(ck[0]), 1, 0, &(x_new[0]), 1);
        }
        else if (iter == max_iter - 1){
            throw PSIEXCEPTION("GMRES solution is not converged, please change max iteration or error threshold in GMRES.");
        } else {
            C_DGEMV('t', max_iter, dim, 1.0, &(q[0]), dim, &(ck[0]), 1, 0, &(x_new[0]), 1);
            break;
        }
    }
    outfile->Printf("Done");
    outfile->Printf("\n        Z vector equation was solved in %d iterations", iters);
}

void DSRG_MRPT2::solve_linear_iter() {
    set_zvec_moinfo();
    set_b(dim, preidx, block_dim);
    std::vector<double> solution(dim, 0.0);
    gmres_solver(solution);

    // Conduct projection to get the correct solution
    double ci_xci_dot = C_DDOT(ndets, &solution[preidx["ci"]], 1, &ci.data()[0], 1);
    {
        int idx = preidx["ci"];
        auto ci_vec = ci.data();
        for (int i = idx; i < dim; ++i) {
            solution[i] -= ci_xci_dot * ci_vec[i - idx];
        }
    }

    // Write the solution of z-vector equations (stored in solution) into the Z matrix
    for (const std::string& block : {"vc", "ca", "va", "aa"}) {
        int pre = preidx[block], idx = block_dim[block];
        if (block != "aa") {
            (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = pre + i[0] * idx + i[1];
                value = solution.at(index);
            });
        } else {
            (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                int i0 = i[0] > i[1] ? i[0] : i[1], i1 = i[0] > i[1] ? i[1] : i[0];
                if (i0 != i1) {
                    int index = pre + i0 * (i0 - 1) / 2 + i1;
                    value = solution.at(index);
                }
            });
        }
    }
    for (const std::string& block : {"ci"}) {
        int pre = preidx[block];
        (x_ci).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = pre + i[0];
            value = solution.at(index);
        });
    }
    Z["me"] = Z["em"];
    Z["wm"] = Z["mw"];
    Z["we"] = Z["ew"];

    // Beta part
    // Caution: This is only valid when restricted orbitals are assumed 
    //          i.e. MO coefficients (alpha) == MO coefficients (beta)
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
}

}// namespace forte