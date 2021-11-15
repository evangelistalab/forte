/**
 * Solve the z-vector equations.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "../dsrg_mrpt2.h"
#include "helpers/timer.h"
#include "psi4/libpsi4util/process.h"

#include <numeric>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace ambit;
using namespace psi;

namespace forte {

int ROW2DEL;
int max_iter = 200;
double err = 1e-9;
std::vector<double> A_ci;

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
    // NOTICE: LAPACK direct solver (only use it when memory is not the bottleneck)
    // solve_z();
    // iterative solver
    solve_linear_iter();
}

void DSRG_MRPT2::set_w() {
    outfile->Printf("\n    Solving Entries of the EWDM W.................... ");
    W = BTF_->build(CoreTensor, "Energy weighted density matrix(Lagrangian)", spin_cases({"gg"}));
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));

    // Form Gamma_tilde
    for (const auto& pair : as_solver_->state_energies_map()) {
        const auto& state = pair.first;
        auto evecs = as_solver_->eigenvectors(state);
        auto g1r = BTF_->build(tensor_type_, "1GRDM_ket", spin_cases({"aa"}));
        auto g2r = BTF_->build(tensor_type_, "2GRDM_ket", spin_cases({"aaaa"}));
        for (int i = 0, nroots = evecs.size(); i < nroots; ++i) {
            as_solver_->generalized_rdms(state, 0, x_ci.data(), Gamma1_tilde, false, 1);
            as_solver_->generalized_rdms(state, 0, x_ci.data(), g1r, true, 1);
            as_solver_->generalized_rdms(state, 0, x_ci.data(), Gamma2_tilde, false, 2);
            as_solver_->generalized_rdms(state, 0, x_ci.data(), g2r, true, 2);
        }    
        Gamma1_tilde["uv"] += g1r["uv"];
        Gamma1_tilde["UV"] += g1r["UV"];

        Gamma2_tilde["uvxy"] += g2r["uvxy"];
        Gamma2_tilde["UVXY"] += g2r["UVXY"];
        Gamma2_tilde["uVxY"] += g2r["uVxY"];
    }

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

    W.block("cc")("nm") += 0.5 * V.block("acac")("umvn") * Gamma1_tilde.block("aa")("uv");
    W.block("cc")("nm") += 0.5 * V.block("cAcA")("mUnV") * Gamma1_tilde.block("AA")("UV");
    W.block("ac")("xm") += 0.5 * V.block("acaa")("umvx") * Gamma1_tilde.block("aa")("uv");
    W.block("ac")("xm") += 0.5 * V.block("cAaA")("mUxV") * Gamma1_tilde.block("AA")("UV");
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
    auto ck_ci =
        ambit::Tensor::build(ambit::CoreTensor, "ci equations ci multiplier part", {ndets, ndets});

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
    y["wz"] += Delta1["zw"] * qk["w,z"];
    y["wz"] -= V["e1,u,m1,w"] * Gamma1_["uz"] * qk["e1,m1"];
    y["wz"] -= V["u,E1,w,M1"] * Gamma1_["uz"] * qk["E1,M1"];
    y["wz"] -= V["e1,w,m1,u"] * Gamma1_["uz"] * qk["e1,m1"];
    y["wz"] -= V["w,E1,u,M1"] * Gamma1_["uz"] * qk["E1,M1"];
    y["wz"] += V["e1,u,m1,z"] * Gamma1_["uw"] * qk["e1,m1"];
    y["wz"] += V["u,E1,z,M1"] * Gamma1_["uw"] * qk["E1,M1"];
    y["wz"] += V["e1,z,m1,u"] * Gamma1_["uw"] * qk["e1,m1"];
    y["wz"] += V["z,E1,u,M1"] * Gamma1_["uw"] * qk["E1,M1"];
    y["wz"] -= F["w,n1"] * qk["n1,z"];
    y["wz"] += F["z,n1"] * qk["n1,w"];
    y["wz"] -= V["u,v,n1,w"] * Gamma1_["zv"] * qk["n1,u"];
    y["wz"] -= V["v,U,w,N1"] * Gamma1_["zv"] * qk["N1,U"];
    y["wz"] -= V["u,w,n1,v"] * Gamma1_["zv"] * qk["n1,u"];
    y["wz"] -= V["w,U,v,N1"] * Gamma1_["zv"] * qk["N1,U"];
    y["wz"] += V["u,v,n1,z"] * Gamma1_["wv"] * qk["n1,u"];
    y["wz"] += V["v,U,z,N1"] * Gamma1_["wv"] * qk["N1,U"];
    y["wz"] += V["u,z,n1,v"] * Gamma1_["wv"] * qk["n1,u"];
    y["wz"] += V["z,U,v,N1"] * Gamma1_["wv"] * qk["N1,U"];
    y["wz"] += H["w,n1"] * Gamma1_["uz"] * qk["n1,u"];
    y["wz"] += V_sumA_Alpha["w,n1"] * Gamma1_["uz"] * qk["n1,u"];
    y["wz"] += V_sumB_Alpha["w,n1"] * Gamma1_["uz"] * qk["n1,u"];
    y["wz"] += 0.5 * V["x,y,n1,w"] * Gamma2_["u,z,x,y"] * qk["n1,u"];
    y["wz"] += V["y,X,w,N1"] * Gamma2_["z,U,y,X"] * qk["N1,U"];
    y["wz"] += V["w,y,n1,v"] * Gamma2_["u,v,z,y"] * qk["n1,u"];
    y["wz"] += V["w,Y,n1,V"] * Gamma2_["u,V,z,Y"] * qk["n1,u"];
    y["wz"] += V["w,Y,v,N1"] * Gamma2_["v,U,z,Y"] * qk["N1,U"];
    y["wz"] -= H["z,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["wz"] -= V_sumA_Alpha["z,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["wz"] -= V_sumB_Alpha["z,n1"] * Gamma1_["uw"] * qk["n1,u"];
    y["wz"] -= 0.5 * V["x,y,n1,z"] * Gamma2_["u,w,x,y"] * qk["n1,u"];
    y["wz"] -= V["y,X,z,N1"] * Gamma2_["w,U,y,X"] * qk["N1,U"];
    y["wz"] -= V["z,y,n1,v"] * Gamma2_["u,v,w,y"] * qk["n1,u"];
    y["wz"] -= V["z,Y,n1,V"] * Gamma2_["u,V,w,Y"] * qk["n1,u"];
    y["wz"] -= V["z,Y,v,N1"] * Gamma2_["v,U,w,Y"] * qk["N1,U"];
    y["wz"] -= H["w,e1"] * Gamma1_["uz"] * qk["e1,u"];
    y["wz"] -= V_sumA_Alpha["e1,w"] * Gamma1_["uz"] * qk["e1,u"];
    y["wz"] -= V_sumB_Alpha["e1,w"] * Gamma1_["uz"] * qk["e1,u"];
    y["wz"] -= 0.5 * V["e1,w,x,y"] * Gamma2_["u,z,x,y"] * qk["e1,u"];
    y["wz"] -= V["w,E1,y,X"] * Gamma2_["z,U,y,X"] * qk["E1,U"];
    y["wz"] -= V["e1,v,w,y"] * Gamma2_["u,v,z,y"] * qk["e1,u"];
    y["wz"] -= V["e1,V,w,Y"] * Gamma2_["u,V,z,Y"] * qk["e1,u"];
    y["wz"] -= V["v,E1,w,Y"] * Gamma2_["v,U,z,Y"] * qk["E1,U"];
    y["wz"] += H["z,e1"] * Gamma1_["uw"] * qk["e1,u"];
    y["wz"] += V_sumA_Alpha["e1,z"] * Gamma1_["uw"] * qk["e1,u"];
    y["wz"] += V_sumB_Alpha["e1,z"] * Gamma1_["uw"] * qk["e1,u"];
    y["wz"] += 0.5 * V["e1,z,x,y"] * Gamma2_["u,w,x,y"] * qk["e1,u"];
    y["wz"] += V["z,E1,y,X"] * Gamma2_["w,U,y,X"] * qk["E1,U"];
    y["wz"] += V["e1,v,z,y"] * Gamma2_["u,v,w,y"] * qk["e1,u"];
    y["wz"] += V["e1,V,z,Y"] * Gamma2_["u,V,w,Y"] * qk["e1,u"];
    y["wz"] += V["v,E1,z,Y"] * Gamma2_["v,U,w,Y"] * qk["E1,U"];
    y["wz"] -= V["a1,v,u1,w"] * Gamma1_["zv"] * qk["u1,a1"];
    y["wz"] -= V["v,A1,w,U1"] * Gamma1_["zv"] * qk["U1,A1"];
    y["wz"] += V["a1,v,u1,z"] * Gamma1_["wv"] * qk["u1,a1"];
    y["wz"] += V["v,A1,z,U1"] * Gamma1_["wv"] * qk["U1,A1"];

    /// MO RESPONSE -- CI EQUATION
    // !!!!!!! NOTICE we may need find a routine to contract qk_ci with ci, cc1, cc2 and cc3 beforehand !!!!!!!
    // VIRTUAL-CORE
    y.block("vc")("em") -= H.block("vc")("em") * ci("I") * qk_ci("I");
    y.block("vc")("em") -= V_sumA_Alpha.block("cv")("me") * ci("I") * qk_ci("I");
    y.block("vc")("em") -= V_sumB_Alpha.block("cv")("me") * ci("I") * qk_ci("I");
    y.block("vc")("em") -= 0.5 * V.block("avac")("veum") * cc1a("Iuv") * qk_ci("I");
    y.block("vc")("em") -= 0.5 * V.block("vAcA")("eVmU") * cc1b("IUV") * qk_ci("I");

    // CORE-ACTIVE
    y.block("ca")("mw") -= 0.50 * H.block("ac")("vm") * cc1a("Iwv") * qk_ci("I");
    y.block("ca")("mw") -= 0.25 * V_sumA_Alpha.block("ac")("um") * cc1a("Iuw") * qk_ci("I");
    y.block("ca")("mw") -= 0.25 * V_sumB_Alpha.block("ac")("um") * cc1a("Iuw") * qk_ci("I");
    y.block("ca")("mw") -= 0.25 * V_sumA_Alpha.block("ca")("mv") * cc1a("Iwv") * qk_ci("I");
    y.block("ca")("mw") -= 0.25 * V_sumB_Alpha.block("ca")("mv") * cc1a("Iwv") * qk_ci("I");
    y.block("ca")("mw") -= 0.125 * V.block("aaca")("xymv") * cc2aa("Iwvxy") * qk_ci("I");
    y.block("ca")("mw") -= 0.250 * V.block("aAcA")("xYmV") * cc2ab("IwVxY") * qk_ci("I");
    y.block("ca")("mw") -= 0.125 * V.block("aaac")("xyum") * cc2aa("Iuwxy") * qk_ci("I");
    y.block("ca")("mw") -= 0.250 * V.block("aAcA")("xYmU") * cc2ab("IwUxY") * qk_ci("I");

    y.block("ca")("mw") += H.block("ac")("wm") * ci("I") * qk_ci("I");
    y.block("ca")("mw") += V_sumA_Alpha.block("ca")("mw") * ci("I") * qk_ci("I");
    y.block("ca")("mw") += V_sumB_Alpha.block("ca")("mw") * ci("I") * qk_ci("I");
    y.block("ca")("mw") += 0.5 * V.block("aaac")("vwum") * cc1a("Iuv") * qk_ci("I");
    y.block("ca")("mw") += 0.5 * V.block("aAcA")("wVmU") * cc1b("IUV") * qk_ci("I");

    // VIRTUAL-ACTIVE
    y.block("va")("ew") -= 0.50 * H.block("av")("ve") * cc1a("Iwv") * qk_ci("I");
    y.block("va")("ew") -= 0.25 * V_sumA_Alpha.block("av")("ue") * cc1a("Iuw") * qk_ci("I");
    y.block("va")("ew") -= 0.25 * V_sumB_Alpha.block("av")("ue") * cc1a("Iuw") * qk_ci("I");
    y.block("va")("ew") -= 0.25 * V_sumA_Alpha.block("va")("ev") * cc1a("Iwv") * qk_ci("I");
    y.block("va")("ew") -= 0.25 * V_sumB_Alpha.block("va")("ev") * cc1a("Iwv") * qk_ci("I");
    y.block("va")("ew") -= 0.125 * V.block("vaaa")("evxy") * cc2aa("Iwvxy") * qk_ci("I");
    y.block("va")("ew") -= 0.250 * V.block("vAaA")("eVxY") * cc2ab("IwVxY") * qk_ci("I");
    y.block("va")("ew") -= 0.125 * V.block("avaa")("uexy") * cc2aa("Iuwxy") * qk_ci("I");
    y.block("va")("ew") -= 0.250 * V.block("vAaA")("eUxY") * cc2ab("IwUxY") * qk_ci("I");

    // ACTIVE-ACTIVE
    y.block("aa")("wz") -= 0.50 * H.block("aa")("vw") * cc1a("Izv") * qk_ci("I");
    y.block("aa")("wz") += 0.50 * H.block("aa")("vz") * cc1a("Iwv") * qk_ci("I");
    y.block("aa")("wz") -= 0.25 * V_sumA_Alpha.block("aa")("uw") * cc1a("Iuz") * qk_ci("I");
    y.block("aa")("wz") -= 0.25 * V_sumB_Alpha.block("aa")("uw") * cc1a("Iuz") * qk_ci("I");
    y.block("aa")("wz") -= 0.25 * V_sumA_Alpha.block("aa")("wv") * cc1a("Izv") * qk_ci("I");
    y.block("aa")("wz") -= 0.25 * V_sumB_Alpha.block("aa")("wv") * cc1a("Izv") * qk_ci("I");
    y.block("aa")("wz") += 0.25 * V_sumA_Alpha.block("aa")("uz") * cc1a("Iuw") * qk_ci("I");
    y.block("aa")("wz") += 0.25 * V_sumB_Alpha.block("aa")("uz") * cc1a("Iuw") * qk_ci("I");
    y.block("aa")("wz") += 0.25 * V_sumA_Alpha.block("aa")("zv") * cc1a("Iwv") * qk_ci("I");
    y.block("aa")("wz") += 0.25 * V_sumB_Alpha.block("aa")("zv") * cc1a("Iwv") * qk_ci("I");

    y.block("aa")("wz") -= 0.125 * V.block("aaaa")("wvxy") * cc2aa("Izvxy") * qk_ci("I");
    y.block("aa")("wz") -= 0.250 * V.block("aAaA")("wVxY") * cc2ab("IzVxY") * qk_ci("I");
    y.block("aa")("wz") -= 0.125 * V.block("aaaa")("uwxy") * cc2aa("Iuzxy") * qk_ci("I");
    y.block("aa")("wz") -= 0.250 * V.block("aAaA")("wUxY") * cc2ab("IzUxY") * qk_ci("I");
    y.block("aa")("wz") += 0.125 * V.block("aaaa")("zvxy") * cc2aa("Iwvxy") * qk_ci("I");
    y.block("aa")("wz") += 0.250 * V.block("aAaA")("zVxY") * cc2ab("IwVxY") * qk_ci("I");
    y.block("aa")("wz") += 0.125 * V.block("aaaa")("uzxy") * cc2aa("Iuwxy") * qk_ci("I");
    y.block("aa")("wz") += 0.250 * V.block("aAaA")("zUxY") * cc2ab("IwUxY") * qk_ci("I");

    /// CI EQUATION -- MO RESPONSE
    // virtual-core
    y_ci("K") += 2 * V.block("vaca")("eumv") * cc1a("Kuv") * qk.block("vc")("em");
    y_ci("K") += 2 * V.block("vAcA")("eUmV") * cc1b("KUV") * qk.block("vc")("em");

    // beta
    y_ci("K") += 2 * V.block("VACA")("EUMV") * cc1b("KUV") * qk.block("VC")("EM");
    y_ci("K") += 2 * V.block("aVaC")("uEvM") * cc1a("Kuv") * qk.block("VC")("EM");

    // contribution from the multiplier Alpha
    y_ci("K") += -4 * ci("K") * V.block("vaca")("exmy") * Gamma1_.block("aa")("xy") * qk.block("vc")("em");
    y_ci("K") += -4 * ci("K") * V.block("vAcA")("eXmY") * Gamma1_.block("AA")("XY") * qk.block("vc")("em");
    y_ci("K") += -4 * ci("K") * V.block("VACA")("EXMY") * Gamma1_.block("AA")("XY") * qk.block("VC")("EM");
    y_ci("K") += -4 * ci("K") * V.block("aVaC")("xEyM") * Gamma1_.block("aa")("xy") * qk.block("VC")("EM");

    // core-active
    y_ci("K") += 2 * V.block("aaca")("uynx") * cc1a("Kxy") * qk.block("ca")("nu");
    y_ci("K") += 2 * V.block("aAcA")("uYnX") * cc1b("KXY") * qk.block("ca")("nu");
    y_ci("K") -= 2 * H.block("ac")("vn") * cc1a("Kuv") * qk.block("ca")("nu");
    y_ci("K") -= 2 * V_sumA_Alpha.block("ac")("vn") * cc1a("Kuv") * qk.block("ca")("nu");
    y_ci("K") -= 2 * V_sumB_Alpha.block("ac")("vn") * cc1a("Kuv") * qk.block("ca")("nu");
    y_ci("K") -= V.block("aaca")("xynv") * cc2aa("Kuvxy") * qk.block("ca")("nu");
    y_ci("K") -= 2 * V.block("aAcA")("xYnV") * cc2ab("KuVxY") * qk.block("ca")("nu");

    // beta
    y_ci("K") += 2 * V.block("AACA")("UYNX") * cc1b("KXY") * qk.block("CA")("NU");
    y_ci("K") += 2 * V.block("aAaC")("yUxN") * cc1a("Kxy") * qk.block("CA")("NU");
    y_ci("K") -= 2 * H.block("AC")("VN") * cc1b("KUV") * qk.block("CA")("NU");
    y_ci("K") -= 2 * V_sumB_Beta.block("AC")("VN") * cc1b("KUV") * qk.block("CA")("NU");
    y_ci("K") -= 2 * V_sumA_Beta.block("AC")("VN") * cc1b("KUV") * qk.block("CA")("NU");
    y_ci("K") -= V.block("AACA")("XYNV") * cc2bb("KUVXY") * qk.block("CA")("NU");
    y_ci("K") -= 2 * V.block("aAaC")("xYvN") * cc2ab("KvUxY") * qk.block("CA")("NU");

    // contribution from the multiplier Alpha
    y_ci("K") += -4 * ci("K") * V.block("aaca")("uynx") * Gamma1_.block("aa")("xy") * qk.block("ca")("nu");
    y_ci("K") += -4 * ci("K") * V.block("aAcA")("uYnX") * Gamma1_.block("AA")("XY") * qk.block("ca")("nu");
    y_ci("K") += 4 * ci("K") * H.block("ac")("vn") * Gamma1_.block("aa")("uv") * qk.block("ca")("nu");
    y_ci("K") += 4 * ci("K") * V_sumA_Alpha.block("ac")("vn") * Gamma1_.block("aa")("uv") * qk.block("ca")("nu");
    y_ci("K") += 4 * ci("K") * V_sumB_Alpha.block("ac")("vn") * Gamma1_.block("aa")("uv") * qk.block("ca")("nu");
    y_ci("K") += 2 * ci("K") * V.block("aaca")("xynv") * Gamma2_.block("aaaa")("uvxy") * qk.block("ca")("nu");
    y_ci("K") += 4 * ci("K") * V.block("aAcA")("xYnV") * Gamma2_.block("aAaA")("uVxY") * qk.block("ca")("nu");

    // beta
    y_ci("K") += -4 * ci("K") * V.block("AACA")("UYNX") * Gamma1_.block("AA")("XY") * qk.block("CA")("NU");
    y_ci("K") += -4 * ci("K") * V.block("aAaC")("yUxN") * Gamma1_.block("aa")("xy") * qk.block("CA")("NU");
    y_ci("K") += 4 * ci("K") * H.block("AC")("VN") * Gamma1_.block("AA")("UV") * qk.block("CA")("NU");
    y_ci("K") += 4 * ci("K") * V_sumB_Beta.block("AC")("VN") * Gamma1_.block("AA")("UV") * qk.block("CA")("NU");
    y_ci("K") += 4 * ci("K") * V_sumA_Beta.block("AC")("VN") * Gamma1_.block("AA")("UV") * qk.block("CA")("NU");
    y_ci("K") += 2 * ci("K") * V.block("AACA")("XYNV") * Gamma2_.block("AAAA")("UVXY") * qk.block("CA")("NU");
    y_ci("K") += 4 * ci("K") * V.block("aAaC")("xYvN") * Gamma2_.block("aAaA")("vUxY") * qk.block("CA")("NU");

    // virtual-active
    y_ci("K") += 2 * H.block("av")("ve") * cc1a("Kuv") * qk.block("va")("eu");
    y_ci("K") += 2 * V_sumA_Alpha.block("av")("ve") * cc1a("Kuv") * qk.block("va")("eu");
    y_ci("K") += 2 * V_sumB_Alpha.block("av")("ve") * cc1a("Kuv") * qk.block("va")("eu");
    y_ci("K") += V.block("vaaa")("evxy") * cc2aa("Kuvxy") * qk.block("va")("eu");
    y_ci("K") += 2 * V.block("vAaA")("eVxY") * cc2ab("KuVxY") * qk.block("va")("eu");

    // beta
    y_ci("K") += 2 * H.block("AV")("VE") * cc1b("KUV") * qk.block("VA")("EU");
    y_ci("K") += 2 * V_sumB_Beta.block("AV")("VE") * cc1b("KUV") * qk.block("VA")("EU");
    y_ci("K") += 2 * V_sumA_Beta.block("AV")("VE") * cc1b("KUV") * qk.block("VA")("EU");
    y_ci("K") += V.block("VAAA")("EVXY") * cc2bb("KUVXY") * qk.block("VA")("EU");
    y_ci("K") += 2 * V.block("aVaA")("vExY") * cc2ab("KvUxY") * qk.block("VA")("EU");

    /// contribution from the multiplier Alpha
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

    // active-active
    y_ci("K") += V.block("aaaa")("uyvx") * cc1a("Kxy") * qk.block("aa")("uv");
    y_ci("K") += V.block("aAaA")("uYvX") * cc1b("KXY") * qk.block("aa")("uv");

    // beta
    y_ci("K") += V.block("AAAA")("UYVX") * cc1b("KXY") * qk.block("AA")("UV");
    y_ci("K") += V.block("aAaA")("yUxV") * cc1a("Kxy") * qk.block("AA")("UV");

    /// contribution from the multiplier Alpha
    y_ci("K") += -2 * ci("K") * V.block("aaaa")("uyvx") * Gamma1_.block("aa")("xy") * qk.block("aa")("uv");
    y_ci("K") += -2 * ci("K") * V.block("aAaA")("uYvX") * Gamma1_.block("AA")("XY") * qk.block("aa")("uv");

    // beta
    y_ci("K") += -2 * ci("K") * V.block("AAAA")("UYVX") * Gamma1_.block("AA")("XY") * qk.block("AA")("UV");
    y_ci("K") += -2 * ci("K") * V.block("aAaA")("yUxV") * Gamma1_.block("aa")("xy") * qk.block("AA")("UV");


    /// CI EQUATION -- CI 
    y_ci("K") += H.block("cc")("mn") * I.block("cc")("mn") * qk_ci("K");
    y_ci("K") += H.block("CC")("MN") * I.block("CC")("MN") * qk_ci("K");
    y_ci("K") += cc.cc1a()("KIuv") * H.block("aa")("uv") * qk_ci("I");
    y_ci("K") += cc.cc1b()("KIUV") * H.block("AA")("UV") * qk_ci("I");

    y_ci("K") += 0.5 * V_sumA_Alpha["m,m1"] * I["m,m1"] * qk_ci("K");
    y_ci("K") += 0.5 * V_sumB_Beta["M,M1"] * I["M,M1"] * qk_ci("K");
    y_ci("K") += V_sumB_Alpha["m,m1"] * I["m,m1"] * qk_ci("K");

    y_ci("K") += cc.cc1a()("KIuv") * V_sumA_Alpha.block("aa")("uv") * qk_ci("I");
    y_ci("K") += cc.cc1b()("KIUV") * V_sumB_Beta.block("AA")("UV") * qk_ci("I");

    y_ci("K") += cc.cc1a()("KIuv") * V_sumB_Alpha.block("aa")("uv") * qk_ci("I");
    y_ci("K") += cc.cc1b()("KIUV") * V_sumA_Beta.block("AA")("UV") * qk_ci("I");

    y_ci("K") += 0.25 * cc.cc2aa()("KIuvxy") * V.block("aaaa")("uvxy") * qk_ci("I");
    y_ci("K") += 0.25 * cc.cc2bb()("KIUVXY") * V.block("AAAA")("UVXY") * qk_ci("I");
    y_ci("K") += 0.50 * cc.cc2ab()("KIuVxY") * V.block("aAaA")("uVxY") * qk_ci("I");
    y_ci("K") += 0.50 * cc.cc2ab()("IKuVxY") * V.block("aAaA")("uVxY") * qk_ci("I");

    y_ci("K") -= (Eref_ - Enuc_ - Efrzc_) * qk_ci("K");

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
            int index = pre1 + i[0];
            if (i[0] != ROW2DEL) {      
                y_vec.at(index) = value;
            } else {
                double product_ciqk = ci("I") * qk_ci("I");
                y_vec.at(index) = product_ciqk;
            }
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

    for (int i = 0; i < ndets; ++i) {
        double value = A_ci.at(i * ndets + i);
        if (std::fabs(value) > err) {
            D.at(preidx["ci"] + i) = 1.0 / value;
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
            std::cout << std::setprecision(9)<< std::fixed << "x = [ " << x_new[0] << " , " << x_new[1] << " ]"<< std::endl;
        }
        else if (iter == max_iter - 1){
            throw PSIEXCEPTION("GMRES solution is not converged, please change max iteration or error threshold in GMRES.");
        } else {
            C_DGEMV('t', max_iter, dim, 1.0, &(q[0]), dim, &(ck[0]), 1, 0, &(x_new[0]), 1);
            std::cout << std::setprecision(9)<< std::fixed << "x = [ " << x_new[0] << " , " << x_new[1] << " ]"<< std::endl;
            break;
        }
    }
    outfile->Printf("Done");
    outfile->Printf("\n        Z vector equation was solved in %d iterations", iters);
}

void DSRG_MRPT2::remove_rankdeficiency() {

    auto ck_ci =
        ambit::Tensor::build(ambit::CoreTensor, "ci equations ci multiplier part", {ndets, ndets});

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
    // Only need be computed once before the iteration, to obtain ROW2DEL

    int lwork = 3 * ndets + 3;
    std::vector<int> jpvt(ndets);
    std::vector<double> tau(ndets);
    std::vector<double> work(lwork);
    A_ci.resize(ndets * ndets);

    (ck_ci).iterate([&](const std::vector<size_t>& i, double& value) {
        int index = i[0] * ndets + i[1];
        A_ci.at(index) = value;
    });

    C_DGEQP3(ndets, ndets, &A_ci[0], ndets, &jpvt[0], &tau[0], &work[0], lwork);

    ROW2DEL = jpvt[ndets - 1] - 1;

    // Substitute the linearly dependent row with the equation : \sum_I x_I c_I = 0
    (ci).iterate([&](const std::vector<size_t>& i, double& value) {
        int index = ROW2DEL * ndets + i[0];
        A_ci.at(index) = value;
    });

    // b has been changed here !!
    b.at(preidx["ci"] + ROW2DEL) = 0.0;
}

void DSRG_MRPT2::solve_linear_iter() {
    set_zvec_moinfo();
    set_b(dim, preidx, block_dim);
    remove_rankdeficiency();
    std::vector<double> solution(dim, 0.0);
    gmres_solver(solution);

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

/// This is a direct solver, thus shall only be used when memory is sufficient.
void DSRG_MRPT2::solve_z() { 
    int N = dim;
    int NRHS = 1, LDA = N, LDB = N;
    int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    std::vector<int> ipiv(N);

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

    (ck_ci).iterate([&](const std::vector<size_t>& i, double& value) {
        int index = i[1] + dim2 * i[0];
        A2.at(index) = value;
    });

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