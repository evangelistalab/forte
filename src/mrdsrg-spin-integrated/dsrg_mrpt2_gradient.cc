/**
 * DSRG-MRPT2 gradient code by Shuhe Wang
 *
 * The computation procedure is listed as belows:
 * (1), Set MOs spaces;
 * (2), Set Tensors (F, H, V etc.);
 * (3), Compute and write the Lagrangian;
 * (4), Write 1RDMs and 2RDMs coefficients;
 * (5), Back-transform the TPDM.
 */
#include <algorithm>
#include <map>
#include <vector>
#include <math.h>
#include <numeric>
#include <ctype.h>
#include <string>

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/dipole.h"

#include "helpers/timer.h"
#include "ci_rdm/ci_rdms.h"
#include "boost/format.hpp"
#include "sci/fci_mo.h"
#include "fci/fci_solver.h"
#include "helpers/printing.h"
#include "dsrg_mrpt2.h"

#include "psi4/libmints/factory.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

#include "gradient_tpdm/backtransform_tpdm.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/psifiles.h"

#include "master_mrdsrg.h"

using namespace ambit;
using namespace psi;

namespace forte {


const bool PT2_TERM = true;

void DSRG_MRPT2::set_all_variables() {
    // TODO: set global variables for future use.
    // NOTICE: This function may better be merged into "dsrg_mrpt2.cc" in the future!!
    outfile->Printf("\n    Set Relevant Variables and Tensors .............. ");

    nmo_ = mo_space_info_->size("CORRELATED");
    s = dsrg_source_->get_s();
    scale_ci = 1.0;
    core_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    virt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    core_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_DOCC");
    actv_all_ = mo_space_info_->get_absolute_mo("ACTIVE");
    virt_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_UOCC");
    core_mos_relative = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    actv_mos_relative = mo_space_info_->get_relative_mo("ACTIVE");
    virt_mos_relative = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");
    irrep_vec = mo_space_info_->get_dimension("ALL");
    na_ = mo_space_info_->size("ACTIVE");
    ncore_ = mo_space_info_->size("RESTRICTED_DOCC");
    nvirt_ = mo_space_info_->size("RESTRICTED_UOCC");
    nirrep_ = mo_space_info_->nirrep();
    ndets = ci_vectors_[0].dims()[0];
    // Initialize tensors.
    // NOTICE: Space of Gamma and Eta is extended at this time.
    Gamma1 = BTF_->build(CoreTensor, "Gamma1", spin_cases({"gg"}));
    Gamma2 = BTF_->build(CoreTensor, "Gamma2", spin_cases({"aaaa"}));
    Eta1 = BTF_->build(CoreTensor, "Eta1", spin_cases({"gg"}));
    H = BTF_->build(CoreTensor, "One-Electron Integral", spin_cases({"gg"}));
    V = BTF_->build(CoreTensor, "Electron Repulsion Integral", spin_cases({"gggg"}));
    F = BTF_->build(CoreTensor, "Fock Matrix", spin_cases({"gg"}));
    W_ = BTF_->build(CoreTensor, "Lagrangian", spin_cases({"gg"}));
    Eeps1 = BTF_->build(CoreTensor, "e^[-s*(Delta1)^2]", spin_cases({"hp"}));
    Eeps2 = BTF_->build(CoreTensor, "e^[-s*(Delta2)^2]", spin_cases({"hhpp"}));
    Eeps2_p = BTF_->build(CoreTensor, "1+e^[-s*(Delta2)^2]", spin_cases({"hhpp"}));
    Eeps2_m1 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)", spin_cases({"hhpp"}));
    Eeps2_m2 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)^2", spin_cases({"hhpp"}));
    Delta1 = BTF_->build(CoreTensor, "Delta1", spin_cases({"gg"}));
    Delta2 = BTF_->build(CoreTensor, "Delta2", spin_cases({"hhpp"}));
    CI_1 = BTF_->build(CoreTensor, "CI-based one-body coefficients for Z-vector equations", spin_cases({"aa"}));
    CI_2 = BTF_->build(CoreTensor, "CI-based two-body coefficients for Z-vector equations", spin_cases({"aaaa"}));
    //NOTICE: The dimension may be further reduced.
    Z = BTF_->build(CoreTensor, "Z Matrix", spin_cases({"gg"}));
    Z_b = BTF_->build(CoreTensor, "b(AX=b)", spin_cases({"gg"}));
    Tau1 = BTF_->build(CoreTensor, "Tau1", spin_cases({"hhpp"}));
    Tau2 = BTF_->build(CoreTensor, "Tau2", spin_cases({"hhpp"}));
    Kappa = BTF_->build(CoreTensor, "Kappa", spin_cases({"hhpp"}));    
    Alpha = 0.0;
    I = BTF_->build(CoreTensor, "identity matrix", spin_cases({"gg"}));
    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });

    V_N_Alpha = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral alpha", {"gg"});
    V_N_Beta = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral beta", {"gg"});
    V_R_Alpha = BTF_->build(CoreTensor, "index-reversed Dimention-reduced Electron Repulsion Integral alpha", {"gg"});
    V_R_Beta = BTF_->build(CoreTensor, "index-reversed Dimention-reduced Electron Repulsion Integral beta", {"GG"});
    V_all_Beta = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral all beta", {"GG"});


    x_ci = ambit::Tensor::build(ambit::CoreTensor, "solution for ci multipliers", {ndets});

    set_tensor();

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_tensor() {
    set_density();
    set_h();
    set_v();
    set_fock();
    set_dsrg_tensor();
}

void DSRG_MRPT2::set_density() {
    Gamma1.block("aa")("pq") = rdms_.g1a()("pq");
    Gamma1.block("AA")("pq") = rdms_.g1b()("pq");

    for (const std::string& block : {"aa", "AA", "vv", "VV"}) {
        (Eta1.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = (i[0] == i[1]) ? 1.0 : 0.0;
        });
    }

    for (const std::string& block : {"cc", "CC"}) {
        (Gamma1.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = (i[0] == i[1]) ? 1.0 : 0.0;
        });
    }

    Eta1["uv"] -= Gamma1["uv"];
    Eta1["UV"] -= Gamma1["UV"];

    // 2-body density
    Gamma2.block("aaaa")("pqrs") = rdms_.g2aa()("pqrs");
    Gamma2.block("aAaA")("pqrs") = rdms_.g2ab()("pqrs");
    Gamma2.block("AAAA")("pqrs") = rdms_.g2bb()("pqrs");
}

void DSRG_MRPT2::set_h() {
    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->oei_a(i[0], i[1]);
        } else {
            value = ints_->oei_b(i[0], i[1]);
        }
    });
}

void DSRG_MRPT2::set_v() {
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            if (spin[1] == AlphaSpin) {
                value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
            } else {
                value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
            }
        } else if (spin[1] == BetaSpin) {
            value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
        }
    });

    // Summation of V["pmqm"] over index "m"
    V_N_Alpha["pq"] = V["pmqn"] * I["mn"];
    // Summation of V["pMqM"] over index "M"
    V_N_Beta["pq"] = V["pMqN"] * I["MN"];
    // Summation of V["mpmq"] over index "m"
    V_R_Alpha["pq"] = V["mpnq"] * I["mn"];
    // Summation of V["mPmQ"] over index "m"
    V_R_Beta["PQ"] = V["mPnQ"] * I["mn"];
    // Summation of V["PMQM"] over index "M"
    V_all_Beta["PQ"] = V["PMQN"] * I["MN"];

}

void DSRG_MRPT2::set_fock() {
    psi::SharedMatrix D1a(new psi::Matrix("D1a", nmo_, nmo_));
    psi::SharedMatrix D1b(new psi::Matrix("D1b", nmo_, nmo_));

    // Fill core-core blocks
    for (size_t m = 0, ncore = core_mos_.size(); m < ncore; m++) {
        D1a->set(core_mos_[m], core_mos_[m], 1.0);
        D1b->set(core_mos_[m], core_mos_[m], 1.0);
    }

    // Fill active-active blocks
    Gamma1.block("aa").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1a->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });
    Gamma1.block("AA").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1b->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });

    // Make Fock matrices
    ints_->make_fock_matrix(D1a, D1b);
    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->get_fock_a(i[0], i[1]);
        } else {
            value = ints_->get_fock_b(i[0], i[1]);
        }
    });
}

void DSRG_MRPT2::set_dsrg_tensor() {
    Eeps1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = dsrg_source_->compute_renormalized(Fa_[i[0]] - Fa_[i[1]]);}
            else { value = dsrg_source_->compute_renormalized(Fb_[i[0]] - Fb_[i[1]]);}
        }
    );
    Delta1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = Fa_[i[0]] - Fa_[i[1]];}
            else { value = Fb_[i[0]] - Fb_[i[1]];}
        }
    );
    Delta2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]];}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]];}
            else { value = Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]];}
        }
    );
    Eeps2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);}
            else { value = dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);}
        }
    );
    Eeps2_p.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = 1.0 + dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);}
            else { value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);}
        }
    );
    Eeps2_m1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = dsrg_source_->compute_denominator(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]], 1);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = dsrg_source_->compute_denominator(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]], 1);}
            else { value = dsrg_source_->compute_denominator(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]], 1);}
        }
    );
    Eeps2_m2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = dsrg_source_->compute_denominator(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]], 2);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = dsrg_source_->compute_denominator(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]], 2);}
            else { value = dsrg_source_->compute_denominator(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]], 2);}
        }
    );   
}

void DSRG_MRPT2::set_multiplier() {
    set_tau();
    set_kappa();
    set_z();
    Z.print();
    set_w();   
}

void DSRG_MRPT2::set_tau() {
    outfile->Printf("\n    Initializing Diagonal Entries of Tau ............ ");  
    Tau1["ijab"] += 0.25 * Eeps2_m1["ijab"] * V_["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Tau1["IJAB"] += 0.25 * Eeps2_m1["IJAB"] * V_["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    Tau1["iJaB"] += 0.25 * Eeps2_m1["iJaB"] * V_["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    Tau2["ijab"] += 0.25 * V_["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Tau2["IJAB"] += 0.25 * V_["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    Tau2["iJaB"] += 0.25 * V_["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    // NOTICE: remove the internal parts based on the DSRG theories
    Tau1["xyuv"] -= 0.25 * Eeps2_m1["xyuv"] * V_["u1,v1,x1,y1"] * Gamma1["x,x1"] * Gamma1["y,y1"] * Eta1["u,u1"] * Eta1["v,v1"];
    Tau1["XYUV"] -= 0.25 * Eeps2_m1["XYUV"] * V_["U1,V1,X1,Y1"] * Gamma1["X,X1"] * Gamma1["Y,Y1"] * Eta1["U,U1"] * Eta1["V,V1"];
    Tau1["xYuV"] -= 0.25 * Eeps2_m1["xYuV"] * V_["u1,V1,x1,Y1"] * Gamma1["x,x1"] * Gamma1["Y,Y1"] * Eta1["u,u1"] * Eta1["V,V1"];

    Tau2["xyuv"] -= 0.25 * V_["u1,v1,x1,y1"] * Gamma1["x,x1"] * Gamma1["y,y1"] * Eta1["u,u1"] * Eta1["v,v1"];
    Tau2["XYUV"] -= 0.25 * V_["U1,V1,X1,Y1"] * Gamma1["X,X1"] * Gamma1["Y,Y1"] * Eta1["U,U1"] * Eta1["V,V1"];
    Tau2["xYuV"] -= 0.25 * V_["u1,V1,x1,Y1"] * Gamma1["x,x1"] * Gamma1["Y,Y1"] * Eta1["u,u1"] * Eta1["V,V1"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_kappa() {
    outfile->Printf("\n    Initializing Diagonal Entries of Kappa .......... ");
    Kappa["klcd"] += 0.25 * T2_["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Kappa["KLCD"] += 0.25 * T2_["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    Kappa["kLcD"] += 0.25 * T2_["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    outfile->Printf("Done");
}

void DSRG_MRPT2::set_z() {
    outfile->Printf("\n    Initializing Diagonal Entries of Z .............. ");
    set_z_cc();  
    set_z_vv();
    set_z_aa_diag();
    outfile->Printf("Done");
    // LAPACK solver
    solve_z();
}

void DSRG_MRPT2::set_w() {
    outfile->Printf("\n    Solving Entries of W ............................ ");
    auto cc = coupling_coefficients_;
    auto ci = ci_vectors_[0];
    auto cc1a_ = cc.cc1a();
    auto cc1b_ = cc.cc1b();
    auto cc2aa_ = cc.cc2aa();
    auto cc2bb_ = cc.cc2bb();
    auto cc2ab_ = cc.cc2ab();
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));

    //NOTICE: w for {virtual-general}
    if (PT2_TERM) {
        W_["pe"] += Tau1["ijeb"] * V["ijpb"];
        W_["pe"] += 2.0 * Tau1["iJeB"] * V["iJpB"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        W_["pe"] += temp["kled"] * V["klpd"];
        W_["pe"] += 2.0 * temp["kLeD"] * V["kLpD"];
        temp.zero();
    }
    W_["pe"] += Z["e,m1"] * F["m1,p"];
    W_["pe"] += Z["eu"] * H["vp"] * Gamma1["uv"];
    W_["pe"] += Z["eu"] * V_N_Alpha["v,p"] * Gamma1["uv"];
    W_["pe"] += Z["eu"] * V_N_Beta["v,p"] * Gamma1["uv"];
    W_["pe"] += 0.5 * Z["eu"] * V["xypv"] * Gamma2["uvxy"];
    W_["pe"] += Z["eu"] * V["xYpV"] * Gamma2["uVxY"];
    W_["pe"] += Z["e,f1"] * F["f1,p"];

    W_["ei"] = W_["ie"];

    //NOTICE: w for {core-hole}
    if (PT2_TERM) {
        W_["im"] += Tau1["mjab"] * V["ijab"];
        W_["im"] += 2.0 * Tau1["mJaB"] * V["iJaB"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        W_["im"] += temp["mlcd"] * V["ilcd"];
        W_["im"] += 2.0 * temp["mLcD"] * V["iLcD"];
        temp.zero();
    }
    W_["im"] += Z["e1,m"] * F["i,e1"];
    W_["im"] += Z["e1,m1"] * V["m1,m,e1,i"];
    W_["im"] += Z["E1,M1"] * V["m,M1,i,E1"];
    W_["im"] += Z["e1,m1"] * V["m1,i,e1,m"];
    W_["im"] += Z["E1,M1"] * V["i,M1,m,E1"];
    W_["im"] += Z["mu"] * F["ui"];
    W_["im"] -= Z["mu"] * H["vi"] * Gamma1["uv"];
    W_["im"] -= Z["mu"] * V_N_Alpha["vi"] * Gamma1["uv"];
    W_["im"] -= Z["mu"] * V_N_Beta["vi"] * Gamma1["uv"];
    W_["im"] -= 0.5 * Z["mu"] * V["xyiv"] * Gamma2["uvxy"];
    W_["im"] -= Z["mu"] * V["xYiV"] * Gamma2["uVxY"];
    W_["im"] += Z["n1,u"] * V["u,i,n1,m"];
    W_["im"] += Z["N1,U"] * V["i,U,m,N1"];
    W_["im"] += Z["n1,u"] * V["u,m,n1,i"];
    W_["im"] += Z["N1,U"] * V["m,U,i,N1"];
    W_["im"] -= Z["n1,u"] * Gamma1["uv"] * V["v,i,n1,m"];
    W_["im"] -= Z["N1,U"] * Gamma1["UV"] * V["i,V,m,N1"];
    W_["im"] -= Z["n1,u"] * Gamma1["uv"] * V["v,m,n1,i"];
    W_["im"] -= Z["N1,U"] * Gamma1["UV"] * V["m,V,i,N1"];
    W_["im"] += Z["e1,u"] * Gamma1["uv"] * V["v,m,e1,i"];
    W_["im"] += Z["E1,U"] * Gamma1["UV"] * V["m,V,i,E1"];
    W_["im"] += Z["e1,u"] * Gamma1["uv"] * V["v,i,e1,m"];
    W_["im"] += Z["E1,U"] * Gamma1["UV"] * V["i,V,m,E1"];
    W_["im"] += Z["m,n1"] * F["n1,i"];
    W_["im"] += Z["m1,n1"] * V["n1,i,m1,m"];
    W_["im"] += Z["M1,N1"] * V["i,N1,m,M1"];
    W_["im"] += Z["uv"] * V["vium"];
    W_["im"] += Z["UV"] * V["iVmU"];
    W_["im"] += Z["e1,f"] * V["f,i,e1,m"];
    W_["im"] += Z["E1,F"] * V["i,F,m,E1"];

    // CI contribution
    W_.block("cc")("nm") += (1.0 - scale_ci) * H.block("cc")("nm");
    W_.block("cc")("nm") += (1.0 - scale_ci) * V_N_Alpha.block("cc")("mn");
    W_.block("cc")("nm") += (1.0 - scale_ci) * V_N_Beta.block("cc")("mn");
    W_.block("cc")("nm") += V.block("acac")("umvn") * x_ci("I") * ci("J") * cc1a_("IJuv");
    W_.block("cc")("nm") += V.block("cAcA")("mUnV") * x_ci("I") * ci("J") * cc1b_("IJUV");

    W_.block("ac")("xm") += (1.0 - scale_ci) * H.block("ac")("xm");
    W_.block("ac")("xm") += (1.0 - scale_ci) * V_N_Alpha.block("ca")("mx");
    W_.block("ac")("xm") += (1.0 - scale_ci) * V_N_Beta.block("ca")("mx");
    W_.block("ac")("xm") += V.block("acaa")("umvx") * x_ci("I") * ci("J") * cc1a_("IJuv");
    W_.block("ac")("xm") += V.block("cAaA")("mUxV") * x_ci("I") * ci("J") * cc1b_("IJUV");

    W_["mu"] = W_["um"];

    //NOTICE: w for {active-active}
    if (PT2_TERM) {
        W_["zw"] += Tau1["ijwb"] * V["ijzb"];
        W_["zw"] += 2.0 * Tau1["iJwB"] * V["iJzB"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        W_["zw"] += temp["klwd"] * V["klzd"];
        W_["zw"] += 2.0 * temp["kLwD"] * V["kLzD"];
        temp.zero();

        W_["zw"] += Tau1["wjab"] * V["zjab"];
        W_["zw"] += 2.0 * Tau1["wJaB"] * V["zJaB"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        W_["zw"] += temp["wlcd"] * V["zlcd"];
        W_["zw"] += 2.0 * temp["wLcD"] * V["zLcD"];
        temp.zero();
    }
    W_["zw"] += Z["e1,m1"] * V["m1,z,e1,u"] * Gamma1["uw"]; 
    W_["zw"] += Z["E1,M1"] * V["z,M1,u,E1"] * Gamma1["uw"]; 
    W_["zw"] += Z["e1,m1"] * V["m1,u,e1,z"] * Gamma1["uw"]; 
    W_["zw"] += Z["E1,M1"] * V["u,M1,z,E1"] * Gamma1["uw"];
    W_["zw"] += Z["n1,w"] * F["z,n1"];
    W_["zw"] += Z["n1,u"] * V["u,v,n1,z"] * Gamma1["wv"];
    W_["zw"] += Z["N1,U"] * V["v,U,z,N1"] * Gamma1["wv"];
    W_["zw"] += Z["n1,u"] * V["u,z,n1,v"] * Gamma1["wv"];
    W_["zw"] += Z["N1,U"] * V["z,U,v,N1"] * Gamma1["wv"];   
    W_["zw"] -= Z["n1,u"] * H["z,n1"] * Gamma1["uw"];
    W_["zw"] -= Z["n1,u"] * V_N_Alpha["z,n1"] * Gamma1["uw"];
    W_["zw"] -= Z["n1,u"] * V_N_Beta["z,n1"] * Gamma1["uw"];
    W_["zw"] -= 0.5 * Z["n1,u"] * V["x,y,n1,z"] * Gamma2["u,w,x,y"];
    W_["zw"] -= Z["N1,U"] * V["y,X,z,N1"] * Gamma2["w,U,y,X"];
    W_["zw"] -= Z["n1,u"] * V["z,y,n1,v"] * Gamma2["u,v,w,y"];
    W_["zw"] -= Z["n1,u"] * V["z,Y,n1,V"] * Gamma2["u,V,w,Y"];
    W_["zw"] -= Z["N1,U"] * V["z,Y,v,N1"] * Gamma2["v,U,w,Y"];
    W_["zw"] += Z["e1,u"] * H["z,e1"] * Gamma1["uw"];
    W_["zw"] += Z["e1,u"] * V_N_Alpha["z,e1"] * Gamma1["uw"];
    W_["zw"] += Z["e1,u"] * V_N_Beta["z,e1"] * Gamma1["uw"];
    W_["zw"] += 0.5 * Z["e1,u"] * V["x,y,e1,z"] * Gamma2["u,w,x,y"];
    W_["zw"] += Z["E1,U"] * V["y,X,z,E1"] * Gamma2["w,U,y,X"];
    W_["zw"] += Z["e1,u"] * V["z,y,e1,v"] * Gamma2["u,v,w,y"];
    W_["zw"] += Z["e1,u"] * V["z,Y,e1,V"] * Gamma2["u,V,w,Y"];
    W_["zw"] += Z["E1,U"] * V["z,Y,v,E1"] * Gamma2["v,U,w,Y"];
    W_["zw"] += Z["m1,n1"] * V["n1,v,m1,z"] * Gamma1["wv"];
    W_["zw"] += Z["M1,N1"] * V["v,N1,z,M1"] * Gamma1["wv"];
    W_["zw"] += Z["e1,f1"] * V["f1,v,e1,z"] * Gamma1["wv"];
    W_["zw"] += Z["E1,F1"] * V["v,F1,z,E1"] * Gamma1["wv"];
    W_["zw"] += Z["u1,v1"] * V["v1,v,u1,z"] * Gamma1["wv"];
    W_["zw"] += Z["U1,V1"] * V["v,V1,z,U1"] * Gamma1["wv"];
    W_["zw"] += Z["wv"] * F["vz"];
  
    W_.block("aa")("zw") += 0.5 * x_ci("I") * H.block("aa")("vz") * cc1a_("IJwv") * ci("J");
    W_.block("aa")("zw") += 0.5 * x_ci("I") * H.block("aa")("zu") * cc1a_("IJuw") * ci("J");
    W_.block("aa")("zw") += 0.5 * V_N_Alpha.block("aa")("uz") * cc1a_("IJuw") * x_ci("I") * ci("J"); 
    W_.block("aa")("zw") += 0.5 * V_N_Beta.block("aa")("uz") * cc1a_("IJuw") * x_ci("I") * ci("J"); 
    W_.block("aa")("zw") += 0.5 * V_N_Alpha.block("aa")("zv") * cc1a_("IJwv") * x_ci("I") * ci("J"); 
    W_.block("aa")("zw") += 0.5 * V_N_Beta.block("aa")("zv") * cc1a_("IJwv") * x_ci("I") * ci("J"); 
    W_.block("aa")("zw") += 0.125 * x_ci("I") * V.block("aaaa")("zvxy") * cc2aa_("IJwvxy") * ci("J");
    W_.block("aa")("zw") += 0.250 * x_ci("I") * V.block("aAaA")("zVxY") * cc2ab_("IJwVxY") * ci("J");
    W_.block("aa")("zw") += 0.125 * x_ci("I") * V.block("aaaa")("uzxy") * cc2aa_("IJuwxy") * ci("J");
    W_.block("aa")("zw") += 0.250 * x_ci("I") * V.block("aAaA")("zUxY") * cc2ab_("IJwUxY") * ci("J");
    W_.block("aa")("zw") += 0.125 * x_ci("I") * V.block("aaaa")("uvzy") * cc2aa_("IJuvwy") * ci("J");
    W_.block("aa")("zw") += 0.250 * x_ci("I") * V.block("aAaA")("uVzY") * cc2ab_("IJuVwY") * ci("J");
    W_.block("aa")("zw") += 0.125 * x_ci("I") * V.block("aaaa")("uvxz") * cc2aa_("IJuvxw") * ci("J");
    W_.block("aa")("zw") += 0.250 * x_ci("I") * V.block("aAaA")("uVzX") * cc2ab_("IJuVwX") * ci("J");

    // CASSCF reference
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"gg"}));

    W_["mp"] += scale_ci * F["mp"];
    temp1["vp"] = H["vp"];
    temp1["vp"] += V_N_Alpha["vp"];
    temp1["vp"] += V_N_Beta["vp"];
    W_["up"] += scale_ci * temp1["vp"] * Gamma1["uv"];
    W_["up"] += 0.5 * scale_ci * V["xypv"] * Gamma2["uvxy"];
    W_["up"] += scale_ci * V["xYpV"] * Gamma2["uVxY"];

    // Copy alpha-alpha to beta-beta 
    (W_.block("CC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("cc").data()[i[0] * ncore_ + i[1]];
    });
    (W_.block("AA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("aa").data()[i[0] * na_ + i[1]];
    });
    (W_.block("VV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("vv").data()[i[0] * nvirt_ + i[1]];
    });
    (W_.block("CV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("cv").data()[i[0] * nvirt_ + i[1]];
    });
    (W_.block("VC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("vc").data()[i[0] * ncore_ + i[1]];
    });
    (W_.block("CA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("ca").data()[i[0] * na_ + i[1]];
    });
    (W_.block("AC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("ac").data()[i[0] * ncore_ + i[1]];
    });
    (W_.block("AV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("av").data()[i[0] * nvirt_ + i[1]];
    });
    (W_.block("VA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("va").data()[i[0] * na_ + i[1]];
    });
    outfile->Printf("Done");
}

void DSRG_MRPT2::change_zmn_degenerate(BlockedTensor& temp1, 
        BlockedTensor& temp2, BlockedTensor& zmn_d, double coeff) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["abml"] = temp1["abmj"] * Gamma1["lj"];
    temp4["adml"] = temp3["abml"] * Eta1["bd"];
    temp3.zero();
    temp3["cdml"] = temp4["adml"] * Eta1["ac"];
    zmn_d["mn"] += coeff * temp3["cdml"] * temp2["cdmnl"];
    temp3.zero();
    temp4.zero();

    temp3["aBmL"] = temp1["aBmJ"] * Gamma1["LJ"];
    temp4["aDmL"] = temp3["aBmL"] * Eta1["BD"];
    temp3.zero();
    temp3["cDmL"] = temp4["aDmL"] * Eta1["ac"];
    zmn_d["mn"] += 2.0 * coeff * temp3["cDmL"] * temp2["cDmnL"];

    temp1.zero();
    temp2.zero();
}

void DSRG_MRPT2::set_z_cc() {   
    BlockedTensor val1 = BTF_->build(CoreTensor, "val1", {"c"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // core-core diagonal entries
    if (PT2_TERM) {
        temp["mjab"] += V["mjab"] * Eeps2["mjab"];
        temp["mJaB"] += V["mJaB"] * Eeps2["mJaB"];
        val1["m"] += 4.0 * s * Tau2["mjab"] * temp["mjab"]; 
        val1["m"] += 8.0 * s * Tau2["mJaB"] * temp["mJaB"]; 
        temp.zero();

        temp["mjab"] += V["mjab"] * Eeps2_m2["mjab"];
        temp["mJaB"] += V["mJaB"] * Eeps2_m2["mJaB"];
        val1["m"] -= 0.5 * temp["mjab"] * V_["cdkl"] * Gamma1["km"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
        val1["m"] -= 1.0 * temp["mJaB"] * V_["cDkL"] * Gamma1["km"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
        temp.zero();

        temp["mlcd"] += V["mlcd"] * Eeps2["mlcd"];
        temp["mLcD"] += V["mLcD"] * Eeps2["mLcD"];
        temp_1["mlcd"] += Kappa["mlcd"] * Delta2["mlcd"];
        temp_1["mLcD"] += Kappa["mLcD"] * Delta2["mLcD"];
        val1["m"] -= 4.0 * s * temp["mlcd"] * temp_1["mlcd"];
        val1["m"] -= 8.0 * s * temp["mLcD"] * temp_1["mLcD"];
        temp.zero();
        temp_1.zero();
    }

    BlockedTensor zmn = BTF_->build(CoreTensor, "z{mn} normal", {"cc"});
    // core-core block entries within normal conditions
    if (PT2_TERM) {
        zmn["mn"] += Tau1["njab"] * V["mjab"];
        zmn["mn"] += 2.0 * Tau1["nJaB"] * V["mJaB"];

        temp["nlcd"] += Kappa["nlcd"] * Eeps2_p["nlcd"];
        temp["nLcD"] += Kappa["nLcD"] * Eeps2_p["nLcD"];
        zmn["mn"] += temp["nlcd"] * V["mlcd"] ; 
        zmn["mn"] += 2.0 * temp["nLcD"] * V["mLcD"];
        temp.zero(); 

        zmn["mn"] -= Tau1["mjab"] * V["njab"];
        zmn["mn"] -= 2.0 * Tau1["mJaB"] * V["nJaB"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        zmn["mn"] -= temp["mlcd"] * V["nlcd"]; 
        zmn["mn"] -= 2.0 * temp["mLcD"] * V["nLcD"]; 
        temp.zero();
    }

    BlockedTensor zmn_d = BTF_->build(CoreTensor, "z{mn} degenerate orbital case", {"cc"});
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppcch", "pPccH"});
    // core-core block entries within degenerate conditions
    if (PT2_TERM) {
        // Alpha
        temp1["abmj"] = V["abmj"] * Delta2["mjab"] * Eeps2["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Delta2["mJaB"] * Eeps2["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2_m1["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2_m1["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, -0.5 * s);

        temp1["abmj"] = V["abmj"] * Eeps2_m1["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2_m1["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Delta2["mlcd"] * Eeps2["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Delta2["mLcD"] * Eeps2["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, -0.5 * s);

        temp1["abmj"] = V["abmj"] * Eeps2_p["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2_p["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, 0.5 * s);

        temp1["abmj"] = V["abmj"] * Eeps2["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2_p["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2_p["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, 0.5 * s);

        temp1["abmj"] = V["abmj"] * Eeps2_p["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2_p["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2_m2["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2_m2["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, -0.25);

        temp1["abmj"] = V["abmj"] * Eeps2_m2["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2_m2["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2_p["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2_p["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, -0.25);
    }

    for (const std::string& block : {"cc", "CC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = val1.block("c").data()[i[0]];}
            else {
                auto dmt = Delta1.block("cc").data()[i[1] * ncore_ + i[0]];
                if (std::fabs(dmt) > 1e-12) { value = zmn.block("cc").data()[i[0] * ncore_ + i[1]] / dmt;}
                else { value = zmn_d.block("cc").data()[i[0] * ncore_ + i[1]];}
            }       
        });
    }  
}

void DSRG_MRPT2::change_zef_degenerate(BlockedTensor& temp1, 
        BlockedTensor& temp2, BlockedTensor& zef_d, double coeff) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["fdkj"] = temp1["fdkl"] * Gamma1["lj"];
    temp4["fdij"] = temp3["fdkj"] * Gamma1["ki"];
    temp3.zero();
    temp3["fbij"] = temp4["fdij"] * Eta1["bd"];
    zef_d["ef"] += coeff * temp3["fbij"] * temp2["efbij"];
    temp3.zero();
    temp4.zero();

    temp3["fDkJ"] = temp1["fDkL"] * Gamma1["LJ"];
    temp4["fDiJ"] = temp3["fDkJ"] * Gamma1["ki"];
    temp3.zero();
    temp3["fBiJ"] = temp4["fDiJ"] * Eta1["BD"];
    zef_d["ef"] += 2.0 * coeff * temp3["fBiJ"] * temp2["efBiJ"];

    temp1.zero();
    temp2.zero();
}

void DSRG_MRPT2::set_z_vv() {
    BlockedTensor val2 = BTF_->build(CoreTensor, "val2", {"v"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // virtual-virtual diagonal entries
    if (PT2_TERM) {
        temp["ijeb"] += V["ijeb"] * Eeps2["ijeb"];
        temp["iJeB"] += V["iJeB"] * Eeps2["iJeB"];
        val2["e"] -= 4.0 * s * Tau2["ijeb"] * temp["ijeb"]; 
        val2["e"] -= 8.0 * s * Tau2["iJeB"] * temp["iJeB"]; 
        temp.zero();

        temp["ijeb"] += V["ijeb"] * Eeps2_m2["ijeb"];
        temp["iJeB"] += V["iJeB"] * Eeps2_m2["iJeB"];
        val2["e"] += 0.5 * temp["ijeb"] * V_["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ec"] * Eta1["bd"];
        val2["e"] += 1.0 * temp["iJeB"] * V_["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ec"] * Eta1["BD"];
        temp.zero();

        temp["kled"] += V["kled"] * Eeps2["kled"];
        temp["kLeD"] += V["kLeD"] * Eeps2["kLeD"];
        temp_1["kled"] += Kappa["kled"] * Delta2["kled"];
        temp_1["kLeD"] += Kappa["kLeD"] * Delta2["kLeD"];
        val2["e"] += 4.0 * s * temp["kled"] * temp_1["kled"];
        val2["e"] += 8.0 * s * temp["kLeD"] * temp_1["kLeD"];
        temp.zero();
        temp_1.zero();
    }
   
    BlockedTensor zef = BTF_->build(CoreTensor, "z{ef} normal", {"vv"});
    // virtual-virtual block entries within normal conditions
    if (PT2_TERM) {
        zef["ef"] += Tau1["ijfb"] * V["ijeb"];
        zef["ef"] += 2.0 * Tau1["iJfB"] * V["iJeB"];

        temp["klfd"] += Kappa["klfd"] * Eeps2_p["klfd"];
        temp["kLfD"] += Kappa["kLfD"] * Eeps2_p["kLfD"];
        zef["ef"] += temp["klfd"] * V["kled"] ; 
        zef["ef"] += 2.0 * temp["kLfD"] * V["kLeD"];
        temp.zero(); 

        zef["ef"] -= Tau1["ijeb"] * V["ijfb"];
        zef["ef"] -= 2.0 * Tau1["iJeB"] * V["iJfB"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        zef["ef"] -= temp["kled"] * V["klfd"]; 
        zef["ef"] -= 2.0 * temp["kLeD"] * V["kLfD"]; 
        temp.zero();
    }

    BlockedTensor zef_d = BTF_->build(CoreTensor, "z{ef} degenerate orbital case", {"vv"});
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"vvphh", "vvPhH"});
    // core-core block entries within degenerate conditions
    if (PT2_TERM) {
        temp1["fdkl"] = V["fdkl"] * Eeps2_m1["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_m1["kLfD"];
        temp2["efbij"] = V["ebij"] * Delta2["ijfb"] * Eeps2["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Delta2["iJfB"] * Eeps2["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, 0.5 * s);

        temp1["fdkl"] = V["fdkl"] * Delta2["klfd"] * Eeps2["klfd"];
        temp1["fDkL"] = V["fDkL"] * Delta2["kLfD"] * Eeps2["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_m1["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_m1["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, 0.5 * s);

        temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, -0.5 * s);

        temp1["fdkl"] = V["fdkl"] * Eeps2["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_p["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_p["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, -0.5 * s);

        temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_m2["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_m2["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, 0.25);

        temp1["fdkl"] = V["fdkl"] * Eeps2_m2["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_m2["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_p["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_p["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, 0.25);
    }

    for (const std::string& block : {"vv", "VV"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = val2.block("v").data()[i[0]];}
            else {
                auto dmt = Delta1.block("vv").data()[i[1] * nvirt_ + i[0]];
                if (std::fabs(dmt) > 1e-12) { value = zef.block("vv").data()[i[0] * nvirt_ + i[1]] / dmt;}
                else { value = zef_d.block("vv").data()[i[0] * nvirt_ + i[1]];}
            }       
        });
    }  
}

void DSRG_MRPT2::set_z_aa_diag() {
    BlockedTensor val3 = BTF_->build(CoreTensor, "val3", {"a"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // virtual-virtual diagonal entries
    if (PT2_TERM) {
        temp["ujab"] += V["ujab"] * Eeps2["ujab"];
        temp["uJaB"] += V["uJaB"] * Eeps2["uJaB"];
        val3["u"] += 4.0 * s * Tau2["ujab"] * temp["ujab"]; 
        val3["u"] += 8.0 * s * Tau2["uJaB"] * temp["uJaB"]; 
        temp.zero();

        temp["ujab"] += V["ujab"] * Eeps2_m2["ujab"];
        temp["uJaB"] += V["uJaB"] * Eeps2_m2["uJaB"];
        val3["u"] -= 0.5 * temp["ujab"] * V_["cdkl"] * Gamma1["ku"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
        val3["u"] -= 1.0 * temp["uJaB"] * V_["cDkL"] * Gamma1["ku"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
        temp.zero();

        // remove internal terms
        temp["uvxy"] += V["uvxy"] * Eeps2_m2["uvxy"];
        temp["uVxY"] += V["uVxY"] * Eeps2_m2["uVxY"];
        val3["u"] += 0.5 * temp["uvxy"] * V_["cdkl"] * Gamma1["ku"] * Gamma1["lv"] * Eta1["xc"] * Eta1["yd"];
        val3["u"] += 1.0 * temp["uVxY"] * V_["cDkL"] * Gamma1["ku"] * Gamma1["LV"] * Eta1["xc"] * Eta1["YD"];
        temp.zero();

        temp["ulcd"] += V["ulcd"] * Eeps2["ulcd"];
        temp["uLcD"] += V["uLcD"] * Eeps2["uLcD"];
        temp_1["ulcd"] += Kappa["ulcd"] * Delta2["ulcd"];
        temp_1["uLcD"] += Kappa["uLcD"] * Delta2["uLcD"];
        val3["u"] -= 4.0 * s * temp["ulcd"] * temp_1["ulcd"];
        val3["u"] -= 8.0 * s * temp["uLcD"] * temp_1["uLcD"];
        temp.zero();
        temp_1.zero();

        temp["ijub"] += V["ijub"] * Eeps2["ijub"];
        temp["iJuB"] += V["iJuB"] * Eeps2["iJuB"];
        val3["u"] -= 4.0 * s * Tau2["ijub"] * temp["ijub"]; 
        val3["u"] -= 8.0 * s * Tau2["iJuB"] * temp["iJuB"]; 
        temp.zero();

        temp["ijub"] += V["ijub"] * Eeps2_m2["ijub"];
        temp["iJuB"] += V["iJuB"] * Eeps2_m2["iJuB"];
        val3["u"] += 0.5 * temp["ijub"] * V_["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["uc"] * Eta1["bd"];
        val3["u"] += 1.0 * temp["iJuB"] * V_["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["uc"] * Eta1["BD"];
        temp.zero();

        // remove internal terms
        temp["xyuv"] += V["xyuv"] * Eeps2_m2["xyuv"];
        temp["xYuV"] += V["xYuV"] * Eeps2_m2["xYuV"];
        val3["u"] -= 0.5 * temp["xyuv"] * V_["cdkl"] * Gamma1["kx"] * Gamma1["ly"] * Eta1["uc"] * Eta1["vd"];
        val3["u"] -= 1.0 * temp["xYuV"] * V_["cDkL"] * Gamma1["kx"] * Gamma1["LY"] * Eta1["uc"] * Eta1["VD"];
        temp.zero();

        temp["klud"] += V["klud"] * Eeps2["klud"];
        temp["kLuD"] += V["kLuD"] * Eeps2["kLuD"];
        temp_1["klud"] += Kappa["klud"] * Delta2["klud"];
        temp_1["kLuD"] += Kappa["kLuD"] * Delta2["kLuD"];
        val3["u"] += 4.0 * s * temp["klud"] * temp_1["klud"];
        val3["u"] += 8.0 * s * temp["kLuD"] * temp_1["kLuD"];
        temp.zero();
        temp_1.zero();
    }
  
    for (const std::string& block : {"aa", "AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = val3.block("a").data()[i[0]];}
        });
    } 
}

void DSRG_MRPT2::set_b() {
    outfile->Printf("\n    Initializing b of the Linear System ............. ");
    //NOTICE: constant b for z{core-virtual}
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));

    if (PT2_TERM) {
        Z_b["em"] += Tau1["mjab"] * V["ejab"];
        Z_b["em"] += 2.0 * Tau1["mJaB"] * V["eJaB"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["em"] += temp["mlcd"] * V["elcd"];
        Z_b["em"] += 2.0 * temp["mLcD"] * V["eLcD"];
        temp.zero();

        Z_b["em"] -= Tau1["ijeb"] * V["ijmb"];
        Z_b["em"] -= 2.0 * Tau1["iJeB"] * V["iJmB"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        Z_b["em"] -= temp["kled"] * V["klmd"];
        Z_b["em"] -= 2.0 * temp["kLeD"] * V["kLmD"];
        temp.zero();
    }
    Z_b["em"] += Z["m1,n1"] * V["n1,e,m1,m"];
    Z_b["em"] += Z["M1,N1"] * V["e,N1,m,M1"];

    Z_b["em"] += Z["e1,f"] * V["f,e,e1,m"];
    Z_b["em"] += Z["E1,F"] * V["e,F,m,E1"];

    //NOTICE: constant b for z{active-active}
    if (PT2_TERM) {
        Z_b["wz"] += Tau1["ijzb"] * V["ijwb"];
        Z_b["wz"] += 2.0 * Tau1["iJzB"] * V["iJwB"];

        temp["klzd"] += Kappa["klzd"] * Eeps2_p["klzd"];
        temp["kLzD"] += Kappa["kLzD"] * Eeps2_p["kLzD"];
        Z_b["wz"] += temp["klzd"] * V["klwd"];
        Z_b["wz"] += 2.0 * temp["kLzD"] * V["kLwD"];
        temp.zero();

        Z_b["wz"] += Tau1["zjab"] * V["wjab"];
        Z_b["wz"] += 2.0 * Tau1["zJaB"] * V["wJaB"];

        temp["zlcd"] += Kappa["zlcd"] * Eeps2_p["zlcd"];
        temp["zLcD"] += Kappa["zLcD"] * Eeps2_p["zLcD"];
        Z_b["wz"] += temp["zlcd"] * V["wlcd"];
        Z_b["wz"] += 2.0 * temp["zLcD"] * V["wLcD"];
        temp.zero();

        Z_b["wz"] -= Tau1["ijwb"] * V["ijzb"];
        Z_b["wz"] -= 2.0 * Tau1["iJwB"] * V["iJzB"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["wz"] -= temp["klwd"] * V["klzd"];
        Z_b["wz"] -= 2.0 * temp["kLwD"] * V["kLzD"];
        temp.zero();

        Z_b["wz"] -= Tau1["wjab"] * V["zjab"];
        Z_b["wz"] -= 2.0 * Tau1["wJaB"] * V["zJaB"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["wz"] -= temp["wlcd"] * V["zlcd"];
        Z_b["wz"] -= 2.0 * temp["wLcD"] * V["zLcD"];
        temp.zero();
    }
    Z_b["wz"] += Z["m1,n1"] * V["n1,v,m1,w"] * Gamma1["zv"];
    Z_b["wz"] += Z["M1,N1"] * V["v,N1,w,M1"] * Gamma1["zv"];

    Z_b["wz"] += Z["e1,f1"] * V["f1,v,e1,w"] * Gamma1["zv"];
    Z_b["wz"] += Z["E1,F1"] * V["v,F1,w,E1"] * Gamma1["zv"];

    Z_b["wz"] -= Z["m1,n1"] * V["n1,v,m1,z"] * Gamma1["wv"];
    Z_b["wz"] -= Z["M1,N1"] * V["v,N1,z,M1"] * Gamma1["wv"];

    Z_b["wz"] -= Z["e1,f1"] * V["f1,v,e1,z"] * Gamma1["wv"];
    Z_b["wz"] -= Z["E1,F1"] * V["v,F1,z,E1"] * Gamma1["wv"];

    //NOTICE: constant b for z{virtual-active}
    if (PT2_TERM) {
        Z_b["ew"] += Tau1["ijwb"] * V["ijeb"];
        Z_b["ew"] += 2.0 * Tau1["iJwB"] * V["iJeB"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["ew"] += temp["klwd"] * V["kled"];
        Z_b["ew"] += 2.0 * temp["kLwD"] * V["kLeD"];
        temp.zero();

        Z_b["ew"] += Tau1["wjab"] * V["ejab"];
        Z_b["ew"] += 2.0 * Tau1["wJaB"] * V["eJaB"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["ew"] += temp["wlcd"] * V["elcd"];
        Z_b["ew"] += 2.0 * temp["wLcD"] * V["eLcD"];
        temp.zero();

        Z_b["ew"] -= Tau1["ijeb"] * V["ijwb"];
        Z_b["ew"] -= 2.0 * Tau1["iJeB"] * V["iJwB"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        Z_b["ew"] -= temp["kled"] * V["klwd"];
        Z_b["ew"] -= 2.0 * temp["kLeD"] * V["kLwD"];
        temp.zero();
    }
    Z_b["ew"] -= Z["e,f1"] * F["f1,w"];
    Z_b["ew"] += Z["m1,n1"] * V["n1,v,m1,e"] * Gamma1["wv"];
    Z_b["ew"] += Z["M1,N1"] * V["v,N1,e,M1"] * Gamma1["wv"];
    Z_b["ew"] += Z["e1,f1"] * V["f1,v,e1,e"] * Gamma1["wv"];
    Z_b["ew"] += Z["E1,F1"] * V["v,F1,e,E1"] * Gamma1["wv"];

    //NOTICE: constant b for z{core-active}
    if (PT2_TERM) {
        Z_b["mw"] += Tau1["ijwb"] * V["ijmb"];
        Z_b["mw"] += 2.0 * Tau1["iJwB"] * V["iJmB"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["mw"] += temp["klwd"] * V["klmd"];
        Z_b["mw"] += 2.0 * temp["kLwD"] * V["kLmD"];
        temp.zero();

        Z_b["mw"] += Tau1["wjab"] * V["mjab"];
        Z_b["mw"] += 2.0 * Tau1["wJaB"] * V["mJaB"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["mw"] += temp["wlcd"] * V["mlcd"];
        Z_b["mw"] += 2.0 * temp["wLcD"] * V["mLcD"];
        temp.zero();

        Z_b["mw"] -= Tau1["mjab"] * V["wjab"];
        Z_b["mw"] -= 2.0 * Tau1["mJaB"] * V["wJaB"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["mw"] -= temp["mlcd"] * V["wlcd"];
        Z_b["mw"] -= 2.0 * temp["mLcD"] * V["wLcD"];
        temp.zero();
    }
    Z_b["mw"] += Z["m1,n1"] * V["n1,v,m1,m"] * Gamma1["wv"];
    Z_b["mw"] += Z["M1,N1"] * V["v,N1,m,M1"] * Gamma1["wv"];

    Z_b["mw"] += Z["e1,f1"] * V["f1,v,e1,m"] * Gamma1["wv"];
    Z_b["mw"] += Z["E1,F1"] * V["v,F1,m,E1"] * Gamma1["wv"];

    Z_b["mw"] -= Z["m,n1"] * F["n1,w"];

    Z_b["mw"] -= Z["m1,n1"] * V["n1,w,m1,m"];
    Z_b["mw"] -= Z["M1,N1"] * V["w,N1,m,M1"];

    Z_b["mw"] -= Z["e1,f"] * V["f,w,e1,m"];
    Z_b["mw"] -= Z["E1,F"] * V["w,F,m,E1"];
}


void DSRG_MRPT2::solve_z() {
    set_b();

    int dim_vc = nvirt_ * ncore_,
        dim_ca = ncore_ * na_,
        dim_va = nvirt_ * na_,
        dim_aa = na_ * (na_ - 1) / 2,
        dim_ci = ndets;
        // dim_ci = 0;
    int dim = dim_vc + dim_ca + dim_va + dim_aa + dim_ci;
    int N=dim;
    int NRHS=1, LDA=N,LDB=N;
    int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    std::vector<int> ipiv(N);

    std::vector<double> A(dim * dim);
    std::vector<double> b(dim);

    std::map<string,int> preidx = {
        {"vc", 0}, {"VC", 0}, {"ca", dim_vc}, {"CA", dim_vc},
        {"va", dim_vc + dim_ca}, {"VA", dim_vc + dim_ca},
        {"aa", dim_vc + dim_ca + dim_va}, {"AA", dim_vc + dim_ca + dim_va},
        {"ci", dim_vc + dim_ca + dim_va + dim_aa} 
    };

    std::map<string,int> block_dim = {
        {"vc", ncore_}, {"VC", ncore_}, {"ca", na_}, {"CA", na_},
        {"va", na_}, {"VA", na_}, {"aa", 0}, {"AA", 0}
    };

    //NOTICE:b

    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"gg"});
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", {"aa","AA"});

    temp3["uv"] += Z["uv"] * I["uv"];
    temp3["UV"] += Z["UV"] * I["UV"];

    // VIRTUAL-CORE

    temp2["em"] += Z_b["em"];
    temp2["em"] += temp3["uv"] * V["veum"];
    temp2["em"] += temp3["UV"] * V["eVmU"];

    // CORE-ACTIVE

    temp2["mw"] += Z_b["mw"];
    temp2["mw"] += temp3["u1,v1"] * V["v1,v,u1,m"] * Gamma1["w,v"];
    temp2["mw"] += temp3["U1,V1"] * V["v,V1,m,U1"] * Gamma1["w,v"];
    temp2["mw"] += temp3["wv"] * F["vm"];
    temp2["mw"] -= temp3["uv"] * V["vwum"];
    temp2["mw"] -= temp3["UV"] * V["wVmU"];

    // VIRTUAL-ACTIVE

    temp2["ew"] += Z_b["ew"];
    temp2["ew"] += temp3["u1,v1"] * V["v1,v,u1,e"] * Gamma1["wv"];
    temp2["ew"] += temp3["U1,V1"] * V["v,V1,e,U1"] * Gamma1["wv"];
    temp2["ew"] += temp3["wv"] * F["ve"];

    // ACTIVE-ACTIVE

    temp2["wz"] += Z_b["wz"];
    temp2["wz"] += temp3["v1,u1"] * V["u1,v,v1,w"] * Gamma1["zv"];
    temp2["wz"] += temp3["V1,U1"] * V["v,U1,w,V1"] * Gamma1["zv"];
    temp2["wz"] -= temp3["v1,u1"] * V["u1,v,v1,z"] * Gamma1["wv"];
    temp2["wz"] -= temp3["V1,U1"] * V["v,U1,z,V1"] * Gamma1["wv"];

    for (const std::string& block : {"vc", "ca", "va", "aa"}) {
        (temp2.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block != "aa") {
                int index = preidx[block] + i[0] * block_dim[block] + i[1];
                b.at(index) = value;
            }
            else if (block == "aa" && i[0] > i[1]) {
                int index = preidx[block] + i[0] * (i[0] - 1) / 2 + i[1];
                b.at(index) = value;
            }    
        });
    } 

    outfile->Printf("Done");
    outfile->Printf("\n    Initializing A of the Linear System ............. ");

    //NOTICE:A
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"gggg","ggGG"});

    // VIRTUAL-CORE
    temp1["e,m,e1,m1"] += Delta1["m1,e1"] * I["e1,e"] * I["m1,m"];

    temp1["e,m,e1,m1"] -= V["m1,m,e1,e"];
    temp1["e,m,E1,M1"] -= V["m,M1,e,E1"];
    temp1["e,m,e1,m1"] -= V["m1,e,e1,m"];
    temp1["e,m,E1,M1"] -= V["e,M1,m,E1"];

    temp1["e,m,m1,u"] -= F["ue"] * I["m,m1"];

    temp1["e,m,n1,u"] -= V["u,e,n1,m"];
    temp1["e,m,N1,U"] -= V["e,U,m,N1"];
    temp1["e,m,n1,u"] -= V["u,m,n1,e"];
    temp1["e,m,N1,U"] -= V["m,U,e,N1"];

    temp1["e,m,n1,u"] += Gamma1["uv"] * V["v,e,n1,m"];
    temp1["e,m,N1,U"] += Gamma1["UV"] * V["e,V,m,N1"];
    temp1["e,m,n1,u"] += Gamma1["uv"] * V["v,m,n1,e"];
    temp1["e,m,N1,U"] += Gamma1["UV"] * V["m,V,e,N1"];

    temp1["e,m,e1,u"] -= Gamma1["uv"] * V["v,m,e1,e"];
    temp1["e,m,E1,U"] -= Gamma1["UV"] * V["m,V,e,E1"];
    temp1["e,m,e1,u"] -= Gamma1["uv"] * V["v,e,e1,m"];
    temp1["e,m,E1,U"] -= Gamma1["UV"] * V["e,V,m,E1"];

    temp1["e,m,e1,u"] += F["um"] * I["e,e1"];

    temp1["e,m,u,v"] -= V["veum"];
    temp1["e,m,U,V"] -= V["eVmU"];

    // CORE-ACTIVE
    temp1["m,w,e1,m1"] += F["w,e1"] * I["m,m1"];
    temp1["m,w,e1,m1"] += V["m1,m,e1,w"];
    temp1["m,w,E1,M1"] += V["m,M1,w,E1"];
    temp1["m,w,e1,m1"] += V["m1,w,e1,m"];
    temp1["m,w,E1,M1"] += V["w,M1,m,E1"];

    temp1["m,w,m1,u"] += F["uw"] * I["m1,m"];
    temp1["m,w,m1,u"] -= H["vw"] * Gamma1["uv"] * I["m1,m"];
    temp1["m,w,m1,u"] -= V["v,n1,w,n"] * Gamma1["uv"] * I["n,n1"] * I["m,m1"];
    temp1["m,w,m1,u"] -= V["v,N1,w,N"] * Gamma1["uv"] * I["N,N1"] * I["m,m1"];
    temp1["m,w,m1,u"] -= 0.5 * V["xywv"] * Gamma2["uvxy"] * I["m,m1"];
    temp1["m,w,m1,u"] -= V["xYwV"] * Gamma2["uVxY"] * I["m,m1"];

    temp1["m,w,n1,u"] += V["u,w,n1,m"];
    temp1["m,w,N1,U"] += V["w,U,m,N1"];
    temp1["m,w,n1,u"] += V["u,m,n1,w"];
    temp1["m,w,N1,U"] += V["m,U,w,N1"];

    temp1["m,w,n1,u"] -= Gamma1["uv"] * V["v,w,n1,m"];
    temp1["m,w,N1,U"] -= Gamma1["UV"] * V["w,V,m,N1"];
    temp1["m,w,n1,u"] -= Gamma1["uv"] * V["v,m,n1,w"];
    temp1["m,w,N1,U"] -= Gamma1["UV"] * V["m,V,w,N1"];

    temp1["m,w,e1,u"] += Gamma1["uv"] * V["v,m,e1,w"];
    temp1["m,w,E1,U"] += Gamma1["UV"] * V["m,V,w,E1"];
    temp1["m,w,e1,u"] += Gamma1["uv"] * V["v,w,e1,m"];
    temp1["m,w,E1,U"] += Gamma1["UV"] * V["w,V,m,E1"];

    temp1["m,w,u,v"] += V["vwum"];
    temp1["m,w,U,V"] += V["wVmU"];

    temp1["m,w,e1,m1"] -= V["m1,m,e1,u"] * Gamma1["uw"];
    temp1["m,w,E1,M1"] -= V["m,M1,u,E1"] * Gamma1["uw"];
    temp1["m,w,e1,m1"] -= V["m1,u,e1,m"] * Gamma1["uw"];
    temp1["m,w,E1,M1"] -= V["u,M1,m,E1"] * Gamma1["uw"];

    temp1["m,w,n1,w1"] -= F["m,n1"] * I["w,w1"];

    temp1["m,w,n1,u"] -= V["u,v,n1,m"] * Gamma1["wv"];
    temp1["m,w,N1,U"] -= V["v,U,m,N1"] * Gamma1["wv"];
    temp1["m,w,n1,u"] -= V["u,m,n1,v"] * Gamma1["wv"];
    temp1["m,w,N1,U"] -= V["m,U,v,N1"] * Gamma1["wv"];

    temp1["m,w,n1,u"] += H["m,n1"] * Gamma1["uw"];
    temp1["m,w,n1,u"] += V["m,m2,n1,n2"] * Gamma1["uw"] * I["m2,n2"];
    temp1["m,w,n1,u"] += V["m,M2,n1,N2"] * Gamma1["uw"] * I["M2,N2"];
    temp1["m,w,n1,u"] += 0.5 * V["x,y,n1,m"] * Gamma2["u,w,x,y"];
    temp1["m,w,N1,U"] += V["y,X,m,N1"] * Gamma2["w,U,y,X"];
    temp1["m,w,n1,u"] += V["m,y,n1,v"] * Gamma2["u,v,w,y"];
    temp1["m,w,n1,u"] += V["m,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp1["m,w,N1,U"] += V["m,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp1["m,w,e1,u"] -= H["m,e1"] * Gamma1["uw"];
    temp1["m,w,e1,u"] -= V["m,m2,e1,n2"] * Gamma1["uw"] * I["m2,n2"];
    temp1["m,w,e1,u"] -= V["m,M2,e1,N2"] * Gamma1["uw"] * I["M2,N2"];
    temp1["m,w,e1,u"] -= 0.5 * V["x,y,e1,m"] * Gamma2["u,w,x,y"];
    temp1["m,w,E1,U"] -= V["y,X,m,E1"] * Gamma2["w,U,y,X"];
    temp1["m,w,e1,u"] -= V["m,y,e1,v"] * Gamma2["u,v,w,y"];
    temp1["m,w,e1,u"] -= V["m,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp1["m,w,E1,U"] -= V["m,Y,v,E1"] * Gamma2["v,U,w,Y"];

    temp1["m,w,u1,v1"] -= V["v1,v,u1,m"] * Gamma1["wv"];
    temp1["m,w,U1,V1"] -= V["v,V1,m,U1"] * Gamma1["wv"];

    temp1["m,w,w1,v"] -= F["vm"] * I["w,w1"];


    // VIRTUAL-ACTIVE
    temp1["e,w,e1,m1"] += F["m1,w"] * I["e,e1"];

    temp1["e,w,e1,u"] += H["vw"] * Gamma1["uv"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["v,m,w,m1"] * Gamma1["uv"] * I["m,m1"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["v,M,w,M1"] * Gamma1["uv"] * I["M,M1"] * I["e,e1"];
    temp1["e,w,e1,u"] += 0.5 * V["xywv"] * Gamma2["uvxy"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["xYwV"] * Gamma2["uVxY"] * I["e,e1"];

    temp1["e,w,e1,m1"] -= V["m1,e,e1,u"] * Gamma1["uw"];
    temp1["e,w,E1,M1"] -= V["e,M1,u,E1"] * Gamma1["uw"];
    temp1["e,w,e1,m1"] -= V["m1,u,e1,e"] * Gamma1["uw"];
    temp1["e,w,E1,M1"] -= V["u,M1,e,E1"] * Gamma1["uw"];

    temp1["e,w,n1,u"] -= V["u,v,n1,e"] * Gamma1["wv"];
    temp1["e,w,N1,U"] -= V["v,U,e,N1"] * Gamma1["wv"];
    temp1["e,w,n1,u"] -= V["u,e,n1,v"] * Gamma1["wv"];
    temp1["e,w,N1,U"] -= V["e,U,v,N1"] * Gamma1["wv"];

    temp1["e,w,n1,u"] += H["e,n1"] * Gamma1["uw"];
    temp1["e,w,n1,u"] += V["e,m,n1,m1"] * Gamma1["uw"] * I["m,m1"];
    temp1["e,w,n1,u"] += V["e,M,n1,M1"] * Gamma1["uw"] * I["M,M1"];
    temp1["e,w,n1,u"] += 0.5 * V["x,y,n1,e"] * Gamma2["u,w,x,y"];
    temp1["e,w,N1,U"] += V["y,X,e,N1"] * Gamma2["w,U,y,X"];
    temp1["e,w,n1,u"] += V["e,y,n1,v"] * Gamma2["u,v,w,y"];
    temp1["e,w,n1,u"] += V["e,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp1["e,w,N1,U"] += V["e,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp1["e,w,u1,v1"] -= V["v1,v,u1,e"] * Gamma1["wv"];
    temp1["e,w,U1,V1"] -= V["v,V1,e,U1"] * Gamma1["wv"];

    temp1["e,w,w1,v"] -= F["ve"] * I["w,w1"];

    temp1["e,w,e1,u"] -= H["e,e1"] * Gamma1["uw"];
    temp1["e,w,e1,u"] -= V["e,m,e1,m1"] * Gamma1["uw"] * I["m,m1"];
    temp1["e,w,e1,u"] -= V["e,M,e1,M1"] * Gamma1["uw"] * I["M,M1"];
    temp1["e,w,e1,u"] -= 0.5 * V["x,y,e1,e"] * Gamma2["u,w,x,y"];
    temp1["e,w,E1,U"] -= V["y,X,e,E1"] * Gamma2["w,U,y,X"];
    temp1["e,w,e1,u"] -= V["e,y,e1,v"] * Gamma2["u,v,w,y"];
    temp1["e,w,e1,u"] -= V["e,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp1["e,w,E1,U"] -= V["e,Y,v,E1"] * Gamma2["v,U,w,Y"];


    // ACTIVE-ACTIVE
    temp1["w,z,w1,z1"] += Delta1["zw"] * I["w,w1"] * I["z,z1"];

    temp1["w,z,e1,m1"] -= V["m1,w,e1,u"] * Gamma1["uz"];
    temp1["w,z,E1,M1"] -= V["w,M1,u,E1"] * Gamma1["uz"];
    temp1["w,z,e1,m1"] -= V["m1,u,e1,w"] * Gamma1["uz"];
    temp1["w,z,E1,M1"] -= V["u,M1,w,E1"] * Gamma1["uz"];

    temp1["w,z,e1,m1"] += V["m1,z,e1,u"] * Gamma1["uw"];
    temp1["w,z,E1,M1"] += V["z,M1,u,E1"] * Gamma1["uw"];
    temp1["w,z,e1,m1"] += V["m1,u,e1,z"] * Gamma1["uw"];
    temp1["w,z,E1,M1"] += V["u,M1,z,E1"] * Gamma1["uw"];

    temp1["w,z,n1,z1"] -= F["w,n1"] * I["z,z1"];
    temp1["w,z,n1,w1"] += F["z,n1"] * I["w,w1"];

    temp1["w,z,n1,u"] -= V["u,v,n1,w"] * Gamma1["zv"];
    temp1["w,z,N1,U"] -= V["v,U,w,N1"] * Gamma1["zv"];
    temp1["w,z,n1,u"] -= V["u,w,n1,v"] * Gamma1["zv"];
    temp1["w,z,N1,U"] -= V["w,U,v,N1"] * Gamma1["zv"];

    temp1["w,z,n1,u"] += V["u,v,n1,z"] * Gamma1["wv"];
    temp1["w,z,N1,U"] += V["v,U,z,N1"] * Gamma1["wv"];
    temp1["w,z,n1,u"] += V["u,z,n1,v"] * Gamma1["wv"];
    temp1["w,z,N1,U"] += V["z,U,v,N1"] * Gamma1["wv"];

    temp1["w,z,n1,u"] += H["w,n1"] * Gamma1["uz"];
    temp1["w,z,n1,u"] += V["w,m1,n1,m"] * Gamma1["uz"] * I["m1,m"];
    temp1["w,z,n1,u"] += V["w,M1,n1,M"] * Gamma1["uz"] * I["M1,M"];
    temp1["w,z,n1,u"] += 0.5 * V["x,y,n1,w"] * Gamma2["u,z,x,y"];
    temp1["w,z,N1,U"] += V["y,X,w,N1"] * Gamma2["z,U,y,X"];
    temp1["w,z,n1,u"] += V["w,y,n1,v"] * Gamma2["u,v,z,y"];
    temp1["w,z,n1,u"] += V["w,Y,n1,V"] * Gamma2["u,V,z,Y"];
    temp1["w,z,N1,U"] += V["w,Y,v,N1"] * Gamma2["v,U,z,Y"];

    temp1["w,z,n1,u"] -= H["z,n1"] * Gamma1["uw"];
    temp1["w,z,n1,u"] -= V["z,m1,n1,m"] * Gamma1["uw"] * I["m1,m"];
    temp1["w,z,n1,u"] -= V["z,M1,n1,M"] * Gamma1["uw"] * I["M1,M"];
    temp1["w,z,n1,u"] -= 0.5 * V["x,y,n1,z"] * Gamma2["u,w,x,y"];
    temp1["w,z,N1,U"] -= V["y,X,z,N1"] * Gamma2["w,U,y,X"];
    temp1["w,z,n1,u"] -= V["z,y,n1,v"] * Gamma2["u,v,w,y"];
    temp1["w,z,n1,u"] -= V["z,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp1["w,z,N1,U"] -= V["z,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp1["w,z,e1,u"] -= H["w,e1"] * Gamma1["uz"];
    temp1["w,z,e1,u"] -= V["w,m1,e1,m"] * Gamma1["uz"] * I["m1,m"];
    temp1["w,z,e1,u"] -= V["w,M1,e1,M"] * Gamma1["uz"] * I["M1,M"];
    temp1["w,z,e1,u"] -= 0.5 * V["x,y,e1,w"] * Gamma2["u,z,x,y"];
    temp1["w,z,E1,U"] -= V["y,X,w,E1"] * Gamma2["z,U,y,X"];
    temp1["w,z,e1,u"] -= V["w,y,e1,v"] * Gamma2["u,v,z,y"];
    temp1["w,z,e1,u"] -= V["w,Y,e1,V"] * Gamma2["u,V,z,Y"];
    temp1["w,z,E1,U"] -= V["w,Y,v,E1"] * Gamma2["v,U,z,Y"];
    
    temp1["w,z,e1,u"] += H["z,e1"] * Gamma1["uw"];
    temp1["w,z,e1,u"] += V["z,m1,e1,m"] * Gamma1["uw"] * I["m1,m"];
    temp1["w,z,e1,u"] += V["z,M1,e1,M"] * Gamma1["uw"] * I["M1,M"];
    temp1["w,z,e1,u"] += 0.5 * V["x,y,e1,z"] * Gamma2["u,w,x,y"];
    temp1["w,z,E1,U"] += V["y,X,z,E1"] * Gamma2["w,U,y,X"];
    temp1["w,z,e1,u"] += V["z,y,e1,v"] * Gamma2["u,v,w,y"];
    temp1["w,z,e1,u"] += V["z,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp1["w,z,E1,U"] += V["z,Y,v,E1"] * Gamma2["v,U,w,Y"];

    temp1["w,z,u1,v1"] -= V["v1,v,u1,w"] * Gamma1["zv"];
    temp1["w,z,U1,V1"] -= V["v,V1,w,U1"] * Gamma1["zv"];

    temp1["w,z,u1,v1"] += V["v1,v,u1,z"] * Gamma1["wv"];
    temp1["w,z,U1,V1"] += V["v,V1,z,U1"] * Gamma1["wv"];

    for (const std::string& row : {"vc","ca","va","aa"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];

        for (const std::string& col : {"vc","VC","ca","CA","va","VA","aa","AA"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];
            if (row != "aa" && col != "aa" && col != "AA") {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0] * idx1 + i[1]) 
                                 + dim * (pre2 + i[2] * idx2 + i[3]);
                    A.at(index) += value;
                });
            }
            else if (row == "aa" && col != "aa" && col != "AA") {       
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    if (i[0] > i[1] ) {
                        int index = (pre1 + i[0] * (i[0] - 1) / 2 + i[1])
                                     + dim * (pre2 + i[2] * idx2 + i[3]); 
                        A.at(index) += value;    
                    }
                });
            }
            else if (row != "aa" && (col == "aa" || col == "AA")) {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i2 = i[2] > i[3]? i[2]: i[3],
                        i3 = i[2] > i[3]? i[3]: i[2];
                    if (i2 != i3 ) {
                        int index = (pre1 + i[0] * idx1 + i[1]) 
                                     + dim * (pre2 + i2 * (i2 - 1) / 2 + i3);
                        A.at(index) += value;    
                    }
                });   
            }
            else {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i2 = i[2] > i[3]? i[2]: i[3],
                        i3 = i[2] > i[3]? i[3]: i[2];
                    if (i[0] > i[1] && i2 != i3) {
                        int index = (pre1 + i[0] * (i[0] - 1) / 2 + i[1])
                                     + dim * (pre2 + i2 * (i2 - 1) / 2 + i3);
                        A.at(index) += value;    
                    }       
                });
            }
        }
    } 

    // CI contribution
    auto cc = coupling_coefficients_;
    auto ci = ci_vectors_[0];
    auto cc1a_ = cc.cc1a();
    auto cc1b_ = cc.cc1b();
    auto cc2aa_ = cc.cc2aa();
    auto cc2bb_ = cc.cc2bb();
    auto cc2ab_ = cc.cc2ab();

    ci.print();

    auto ci_vc = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{vc}", {ndets, nvirt_, ncore_});
    auto ci_ca = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{ca}", {ndets, ncore_, na_});
    auto ci_va = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{va}", {ndets, nvirt_, na_});
    auto ci_aa = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{aa}", {ndets, na_, na_});


    // CI contribution to Z{VC}
    ci_vc("Iem") -= H.block("vc")("em") * ci("I");
    ci_vc("Iem") -= V_N_Alpha.block("cv")("me") * ci("I");
    ci_vc("Iem") -= V_N_Beta.block("cv")("me") * ci("I");
    ci_vc("Iem") -= V.block("acav")("umve") * cc1a_("IJuv") * ci("J");
    ci_vc("Iem") -= V.block("cAvA")("mUeV") * cc1b_("IJUV") * ci("J");


    // CI contribution to Z{CA}
    ci_ca("Imw") -= 0.5 * H.block("ac")("vm") * cc1a_("IJwv") * ci("J");
    ci_ca("Imw") -= 0.5 * H.block("ca")("mu") * cc1a_("IJuw") * ci("J");

    ci_ca("Imw") -= 0.5 * V_N_Alpha.block("ac")("um") * cc1a_("IJuw") * ci("J");
    ci_ca("Imw") -= 0.5 * V_N_Beta.block("ac")("um") * cc1a_("IJuw") * ci("J");

    ci_ca("Imw") -= 0.5 * V_N_Alpha.block("ca")("mv") * cc1a_("IJwv") * ci("J"); 
    ci_ca("Imw") -= 0.5 * V_N_Beta.block("ca")("mv") * cc1a_("IJwv") * ci("J"); 

    ci_ca("Imw") -= 0.125 * V.block("caaa")("mvxy") * cc2aa_("IJwvxy") * ci("J");
    ci_ca("Imw") -= 0.250 * V.block("cAaA")("mVxY") * cc2ab_("IJwVxY") * ci("J");

    ci_ca("Imw") -= 0.125 * V.block("acaa")("umxy") * cc2aa_("IJuwxy") * ci("J");
    ci_ca("Imw") -= 0.250 * V.block("cAaA")("mUxY") * cc2ab_("IJwUxY") * ci("J");

    ci_ca("Imw") -= 0.125 * V.block("aaca")("uvmy") * cc2aa_("IJuvwy") * ci("J");
    ci_ca("Imw") -= 0.250 * V.block("aAcA")("uVmY") * cc2ab_("IJuVwY") * ci("J");

    ci_ca("Imw") -= 0.125 * V.block("aaac")("uvxm") * cc2aa_("IJuvxw") * ci("J");
    ci_ca("Imw") -= 0.250 * V.block("aAcA")("uVmX") * cc2ab_("IJuVwX") * ci("J");

    ci_ca("Imw") += H.block("ac")("wm") * ci("I");
    ci_ca("Imw") += V_N_Alpha.block("ca")("mw") * ci("I");
    ci_ca("Imw") += V_N_Beta.block("ca")("mw") * ci("I");
    ci_ca("Imw") += V.block("acaa")("umvw") * cc1a_("IJuv") * ci("J");
    ci_ca("Imw") += V.block("cAaA")("mUwV") * cc1b_("IJUV") * ci("J");

    // CI contribution to Z{VA}
    ci_va("Iew") -= 0.5 * H.block("av")("ve") * cc1a_("IJwv") * ci("J");
    ci_va("Iew") -= 0.5 * H.block("va")("eu") * cc1a_("IJuw") * ci("J");

    ci_va("Iew") -= 0.5 * V_N_Alpha.block("av")("ue") * cc1a_("IJuw") * ci("J");
    ci_va("Iew") -= 0.5 * V_N_Beta.block("av")("ue") * cc1a_("IJuw") * ci("J");
    ci_va("Iew") -= 0.5 * V_N_Alpha.block("va")("ev") * cc1a_("IJwv") * ci("J");
    ci_va("Iew") -= 0.5 * V_N_Beta.block("va")("ev") * cc1a_("IJwv") * ci("J");

    ci_va("Iew") -= 0.125 * V.block("vaaa")("evxy") * cc2aa_("IJwvxy") * ci("J");
    ci_va("Iew") -= 0.250 * V.block("vAaA")("eVxY") * cc2ab_("IJwVxY") * ci("J");

    ci_va("Iew") -= 0.125 * V.block("avaa")("uexy") * cc2aa_("IJuwxy") * ci("J");
    ci_va("Iew") -= 0.250 * V.block("vAaA")("eUxY") * cc2ab_("IJwUxY") * ci("J");

    ci_va("Iew") -= 0.125 * V.block("aava")("uvey") * cc2aa_("IJuvwy") * ci("J");
    ci_va("Iew") -= 0.250 * V.block("aAvA")("uVeY") * cc2ab_("IJuVwY") * ci("J");

    ci_va("Iew") -= 0.125 * V.block("aaav")("uvxe") * cc2aa_("IJuvxw") * ci("J");
    ci_va("Iew") -= 0.250 * V.block("aAvA")("uVeX") * cc2ab_("IJuVwX") * ci("J");

    // CI contribution to Z{AA}
    ci_aa("Iwz") -= 0.5 * H.block("aa")("vw") * cc1a_("IJzv") * ci("J");
    ci_aa("Iwz") -= 0.5 * H.block("aa")("wu") * cc1a_("IJuz") * ci("J");
    ci_aa("Iwz") += 0.5 * H.block("aa")("vz") * cc1a_("IJwv") * ci("J");
    ci_aa("Iwz") += 0.5 * H.block("aa")("zu") * cc1a_("IJuw") * ci("J");

    ci_aa("Iwz") -= 0.5 * V_N_Alpha.block("aa")("uw") * cc1a_("IJuz") * ci("J");
    ci_aa("Iwz") -= 0.5 * V_N_Beta.block("aa")("uw") * cc1a_("IJuz") * ci("J");
    ci_aa("Iwz") -= 0.5 * V_N_Alpha.block("aa")("wv") * cc1a_("IJzv") * ci("J");
    ci_aa("Iwz") -= 0.5 * V_N_Beta.block("aa")("wv") * cc1a_("IJzv") * ci("J");

    ci_aa("Iwz") += 0.5 * V_N_Alpha.block("aa")("uz") * cc1a_("IJuw") * ci("J");
    ci_aa("Iwz") += 0.5 * V_N_Beta.block("aa")("uz") * cc1a_("IJuw") * ci("J");
    ci_aa("Iwz") += 0.5 * V_N_Alpha.block("aa")("zv") * cc1a_("IJwv") * ci("J");
    ci_aa("Iwz") += 0.5 * V_N_Beta.block("aa")("zv") * cc1a_("IJwv") * ci("J");

    ci_aa("Iwz") -= 0.125 * V.block("aaaa")("wvxy") * cc2aa_("IJzvxy") * ci("J");
    ci_aa("Iwz") -= 0.250 * V.block("aAaA")("wVxY") * cc2ab_("IJzVxY") * ci("J");

    ci_aa("Iwz") -= 0.125 * V.block("aaaa")("uwxy") * cc2aa_("IJuzxy") * ci("J");
    ci_aa("Iwz") -= 0.250 * V.block("aAaA")("wUxY") * cc2ab_("IJzUxY") * ci("J");

    ci_aa("Iwz") -= 0.125 * V.block("aaaa")("uvwy") * cc2aa_("IJuvzy") * ci("J");
    ci_aa("Iwz") -= 0.250 * V.block("aAaA")("uVwY") * cc2ab_("IJuVzY") * ci("J");

    ci_aa("Iwz") -= 0.125 * V.block("aaaa")("uvxw") * cc2aa_("IJuvxz") * ci("J");
    ci_aa("Iwz") -= 0.250 * V.block("aAaA")("uVwX") * cc2ab_("IJuVzX") * ci("J");

    ci_aa("Iwz") += 0.125 * V.block("aaaa")("zvxy") * cc2aa_("IJwvxy") * ci("J");
    ci_aa("Iwz") += 0.250 * V.block("aAaA")("zVxY") * cc2ab_("IJwVxY") * ci("J");

    ci_aa("Iwz") += 0.125 * V.block("aaaa")("uzxy") * cc2aa_("IJuwxy") * ci("J");
    ci_aa("Iwz") += 0.250 * V.block("aAaA")("zUxY") * cc2ab_("IJwUxY") * ci("J");

    ci_aa("Iwz") += 0.125 * V.block("aaaa")("uvzy") * cc2aa_("IJuvwy") * ci("J");
    ci_aa("Iwz") += 0.250 * V.block("aAaA")("uVzY") * cc2ab_("IJuVwY") * ci("J");

    ci_aa("Iwz") += 0.125 * V.block("aaaa")("uvxz") * cc2aa_("IJuvxw") * ci("J");
    ci_aa("Iwz") += 0.250 * V.block("aAaA")("uVzX") * cc2ab_("IJuVwX") * ci("J");


    for (const std::string& row : {"vc","ca","va","aa"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];
        auto temp_ci = ci_vc;
        if (row == "ca") temp_ci = ci_ca;
        else if (row == "va") temp_ci = ci_va;
        else if (row == "aa") temp_ci = ci_aa;

        for (const std::string& col : {"ci"}) {
            int pre2 = preidx[col];
            if (row != "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[1] * idx1 + i[2]) 
                                 + dim * (pre2 + i[0]);
                    A.at(index) += value;
                });
            }
            else if (row == "aa") {       
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    if (i[1] > i[2] ) {
                        int index = (pre1 + i[1] * (i[1] - 1) / 2 + i[2])
                                     + dim * (pre2 + i[0]); 
                        A.at(index) += value;    
                    }
                });
            }
        }
    } 

    auto b_ck  = ambit::Tensor::build(ambit::CoreTensor, "ci equations b part", {ndets});

    b_ck("K") -= 2.0 * H.block("aa")("vu") * cc1a_("KJuv") * ci("J");
    b_ck("K") -= 2.0 * H.block("AA")("VU") * cc1b_("KJUV") * ci("J");

    b_ck("K") -= 2.0 * V_N_Alpha.block("aa")("vu") * cc1a_("KJuv") * ci("J");
    b_ck("K") -= 2.0 * V_N_Beta.block("aa")("vu") * cc1a_("KJuv") * ci("J");

    b_ck("K") -= 2.0 * V_all_Beta.block("AA")("VU") * cc1b_("KJUV") * ci("J");
    b_ck("K") -= 2.0 * V_R_Beta.block("AA")("VU") * cc1b_("KJUV") * ci("J");

    b_ck("K") -= 0.5 * V.block("aaaa")("xyuv") * cc2aa_("KJuvxy") * ci("J");
    b_ck("K") -= 0.5 * V.block("AAAA")("XYUV") * cc2bb_("KJUVXY") * ci("J");
    b_ck("K") -= 2.0 * V.block("aAaA")("xYuV") * cc2ab_("KJuVxY") * ci("J");


    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"aa","AA"});

    temp["uv"] += 0.25 * T2_["vjab"] * V_["cdul"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    temp["uv"] += 0.50 * T2_["vJaB"] * V_["cDuL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp["UV"] += 0.25 * T2_["VJAB"] * V_["CDUL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    temp["UV"] += 0.50 * T2_["jVaB"] * V_["cDlU"] * Gamma1["lj"] * Eta1["ac"] * Eta1["BD"];


    temp["uv"] += 0.25 * T2_["ujab"] * V_["cdvl"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    temp["uv"] += 0.50 * T2_["uJaB"] * V_["cDvL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp["UV"] += 0.25 * T2_["UJAB"] * V_["CDVL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    temp["UV"] += 0.50 * T2_["jUaB"] * V_["cDlV"] * Gamma1["lj"] * Eta1["ac"] * Eta1["BD"];


    temp["uv"] += 0.25 * T2_["ivab"] * V_["cdku"] * Gamma1["ki"] * Eta1["ac"] * Eta1["bd"];
    temp["uv"] += 0.50 * T2_["vIaB"] * V_["cDuK"] * Gamma1["KI"] * Eta1["ac"] * Eta1["BD"];
    temp["UV"] += 0.25 * T2_["IVAB"] * V_["CDKU"] * Gamma1["KI"] * Eta1["AC"] * Eta1["BD"];
    temp["UV"] += 0.50 * T2_["iVaB"] * V_["cDkU"] * Gamma1["ki"] * Eta1["ac"] * Eta1["BD"];


    temp["uv"] += 0.25 * T2_["iuab"] * V_["cdkv"] * Gamma1["ki"] * Eta1["ac"] * Eta1["bd"];
    temp["uv"] += 0.50 * T2_["uIaB"] * V_["cDvK"] * Gamma1["KI"] * Eta1["ac"] * Eta1["BD"];
    temp["UV"] += 0.25 * T2_["IUAB"] * V_["CDKV"] * Gamma1["KI"] * Eta1["AC"] * Eta1["BD"];
    temp["UV"] += 0.50 * T2_["iUaB"] * V_["cDkV"] * Gamma1["ki"] * Eta1["ac"] * Eta1["BD"];


    temp["uv"] -= 0.25 * T2_["ijub"] * V_["vdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    temp["uv"] -= 0.50 * T2_["iJuB"] * V_["vDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp["UV"] -= 0.25 * T2_["IJUB"] * V_["VDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    temp["UV"] -= 0.50 * T2_["iJbU"] * V_["dVkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["bd"];


    temp["uv"] -= 0.25 * T2_["ijvb"] * V_["udkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    temp["uv"] -= 0.50 * T2_["iJvB"] * V_["uDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp["UV"] -= 0.25 * T2_["IJVB"] * V_["UDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    temp["UV"] -= 0.50 * T2_["iJbV"] * V_["dUkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["bd"];


    temp["uv"] -= 0.25 * T2_["ijau"] * V_["cvkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"];
    temp["uv"] -= 0.50 * T2_["iJuA"] * V_["vCkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["AC"];
    temp["UV"] -= 0.25 * T2_["IJAU"] * V_["CVKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"];
    temp["UV"] -= 0.50 * T2_["iJaU"] * V_["cVkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"];


    temp["uv"] -= 0.25 * T2_["ijav"] * V_["cukl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"];
    temp["uv"] -= 0.50 * T2_["iJvA"] * V_["uCkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["AC"];
    temp["UV"] -= 0.25 * T2_["IJAV"] * V_["CUKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"];
    temp["UV"] -= 0.50 * T2_["iJaV"] * V_["cUkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"];

    b_ck("K") -= temp.block("aa")("uv") * cc1a_("KJuv") * ci("J"); 
    b_ck("K") -= temp.block("AA")("UV") * cc1b_("KJUV") * ci("J");

    // TODO: need to plug in alpha terms
    Alpha = 0.0;

    Alpha += H["vu"] * Gamma1["uv"];
    Alpha += H["VU"] * Gamma1["UV"];

    Alpha += V_N_Alpha["v,u"] * Gamma1["uv"];
    Alpha += V_N_Beta["v,u"] * Gamma1["uv"];
    Alpha += V["V,M,U,M1"] * Gamma1["UV"] * I["M,M1"];
    Alpha += V["m,V,m1,U"] * Gamma1["UV"] * I["m,m1"];

    Alpha += 0.25 * V["xyuv"] * Gamma2["uvxy"];
    Alpha += 0.25 * V["XYUV"] * Gamma2["UVXY"];
    Alpha += V["xYuV"] * Gamma2["uVxY"];


    Alpha += 0.125 * T2_["vjab"] * V_["cdul"] * Gamma1["uv"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Alpha += 0.125 * T2_["VJAB"] * V_["CDUL"] * Gamma1["UV"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    Alpha += 0.250 * T2_["vJaB"] * V_["cDuL"] * Gamma1["uv"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    Alpha += 0.250 * T2_["jVaB"] * V_["cDlU"] * Gamma1["UV"] * Gamma1["lj"] * Eta1["ac"] * Eta1["BD"];

    Alpha += 0.125 * T2_["ujab"] * V_["cdvl"] * Gamma1["uv"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Alpha += 0.125 * T2_["UJAB"] * V_["CDVL"] * Gamma1["UV"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    Alpha += 0.5 * T2_["uJaB"] * V_["cDvL"] * Gamma1["uv"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    Alpha += 0.125 * T2_["ivab"] * V_["cdku"] * Gamma1["ki"] * Gamma1["uv"] * Eta1["ac"] * Eta1["bd"];
    Alpha += 0.125 * T2_["IVAB"] * V_["CDKU"] * Gamma1["KI"] * Gamma1["UV"] * Eta1["AC"] * Eta1["BD"];
    Alpha += 0.5 * T2_["iVaB"] * V_["cDkU"] * Gamma1["ki"] * Gamma1["UV"] * Eta1["ac"] * Eta1["BD"];

    Alpha += 0.125 * T2_["iuab"] * V_["cdkv"] * Gamma1["ki"] * Gamma1["uv"] * Eta1["ac"] * Eta1["bd"];
    Alpha += 0.125 * T2_["IUAB"] * V_["CDKV"] * Gamma1["KI"] * Gamma1["UV"] * Eta1["AC"] * Eta1["BD"];
    Alpha += 0.5 * T2_["iUaB"] * V_["cDkV"] * Gamma1["ki"] * Gamma1["UV"] * Eta1["ac"] * Eta1["BD"];

    Alpha -= 0.125 * T2_["ijub"] * V_["vdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["uv"] * Eta1["bd"];
    Alpha -= 0.125 * T2_["IJUB"] * V_["VDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["UV"] * Eta1["BD"];
    Alpha -= 0.5 * T2_["iJuB"] * V_["vDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["uv"] * Eta1["BD"];

    Alpha -= 0.125 * T2_["ijvb"] * V_["udkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["uv"] * Eta1["bd"];
    Alpha -= 0.125 * T2_["IJVB"] * V_["UDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["UV"] * Eta1["BD"];
    Alpha -= 0.5 * T2_["iJvB"] * V_["uDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["uv"] * Eta1["BD"];

    Alpha -= 0.125 * T2_["ijau"] * V_["cvkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["uv"];
    Alpha -= 0.125 * T2_["IJAU"] * V_["CVKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["UV"];
    Alpha -= 0.5 * T2_["iJaU"] * V_["cVkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["UV"];

    Alpha -= 0.125 * T2_["ijav"] * V_["cukl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["uv"];
    Alpha -= 0.125 * T2_["IJAV"] * V_["CUKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["UV"];
    Alpha -= 0.5 * T2_["iJaV"] * V_["cUkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["UV"];

    Alpha += Z["mn"] * V["m,v1,n,u1"] * Gamma1["u1,v1"];
    Alpha += Z["mn"] * V["m,V1,n,U1"] * Gamma1["U1,V1"];
    Alpha += Z["MN"] * V["M,V1,N,U1"] * Gamma1["U1,V1"];
    Alpha += Z["MN"] * V["v1,M,u1,N"] * Gamma1["u1,v1"];

    Alpha += Z["ef"] * V["e,v1,f,u1"] * Gamma1["u1,v1"];
    Alpha += Z["ef"] * V["e,V1,f,U1"] * Gamma1["U1,V1"];
    Alpha += Z["EF"] * V["E,V1,F,U1"] * Gamma1["U1,V1"];
    Alpha += Z["EF"] * V["v1,E,u1,F"] * Gamma1["u1,v1"];

    temp3.zero();
    temp3["uv"] += Z["uv"] * I["uv"];
    temp3["UV"] += Z["UV"] * I["UV"];

    Alpha += temp3["uv"] * V["u,v1,v,u1"] * Gamma1["u1,v1"];
    Alpha += temp3["uv"] * V["u,V1,v,U1"] * Gamma1["U1,V1"];
    Alpha += temp3["UV"] * V["U,V1,V,U1"] * Gamma1["U1,V1"];
    Alpha += temp3["UV"] * V["v1,U,u1,V"] * Gamma1["u1,v1"];

    b_ck("K") += 2 * Alpha * ci("K");
    std::cout<<"alpha = " << Alpha << std::endl;

    temp.zero();
    temp["uv"] += 2 * Z["mn"] * V["mvnu"];
    temp["uv"] += 2 * Z["MN"] * V["vMuN"];

    temp["uv"] += 2 * Z["u1,v1"] * V["u1,v,v1,u"] * I["u1,v1"];
    temp["uv"] += 2 * Z["U1,V1"] * V["v,U1,u,V1"] * I["U1,V1"];

    temp["uv"] += 2 * Z["ef"] * V["evfu"];
    temp["uv"] += 2 * Z["EF"] * V["vEuF"];

    temp["UV"] += 2 * Z["MN"] * V["MVNU"];
    temp["UV"] += 2 * Z["mn"] * V["mVnU"];

    temp["UV"] += 2 * Z["U1,V1"] * V["U1,V,V1,U"] * I["U1,V1"];
    temp["UV"] += 2 * Z["u1,v1"] * V["u1,V,v1,U"] * I["u1,v1"];

    temp["UV"] += 2 * Z["EF"] * V["EVFU"];
    temp["UV"] += 2 * Z["ef"] * V["eVfU"];

    b_ck("K") -= temp.block("aa")("uv") * cc1a_("KJuv") * ci("J"); 
    b_ck("K") -= temp.block("AA")("UV") * cc1b_("KJUV") * ci("J"); 

    for (const std::string& block : {"ci"}) {
        (b_ck).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = preidx[block] + i[0];
            b.at(index) = value;
        });
    } 

    auto ck_vc_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{vc} alpha part", {ndets, nvirt_, ncore_});
    auto ck_ca_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{ca} alpha part", {ndets, ncore_, na_});
    auto ck_va_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{va} alpha part", {ndets, nvirt_, na_});
    auto ck_aa_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{aa} alpha part", {ndets, na_, na_});

    auto ck_vc_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{vc} beta part", {ndets, nvirt_, ncore_});
    auto ck_ca_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{ca} beta part", {ndets, ncore_, na_});
    auto ck_va_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{va} beta part", {ndets, nvirt_, na_});
    auto ck_aa_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{aa} beta part", {ndets, na_, na_});

    // virtual-core
    ck_vc_a("Kem") += 4 * V.block("cava")("mveu") * cc1a_("KJuv") * ci("J");
    ck_vc_a("Kem") += 4 * V.block("cAvA")("mVeU") * cc1b_("KJUV") * ci("J");

    ck_vc_b("KEM") += 4 * V.block("CAVA")("MVEU") * cc1b_("KJUV") * ci("J");
    ck_vc_b("KEM") += 4 * V.block("aCaV")("vMuE") * cc1a_("KJuv") * ci("J");


    /// contribution from Alpha
    ck_vc_a("Kem") += -4 * ci("K") * V.block("cava")("myex") * Gamma1.block("aa")("xy");
    ck_vc_a("Kem") += -4 * ci("K") * V.block("cAvA")("mYeX") * Gamma1.block("AA")("XY");

    ck_vc_b("KEM") += -4 * ci("K") * V.block("CAVA")("MYEX") * Gamma1.block("AA")("XY");
    ck_vc_b("KEM") += -4 * ci("K") * V.block("aCaV")("yMxE") * Gamma1.block("aa")("xy");

    // core-active
    ck_ca_a("Knu") += 4 * V.block("aaca")("uynx") * cc1a_("KJxy") * ci("J");
    ck_ca_a("Knu") += 4 * V.block("aAcA")("uYnX") * cc1b_("KJXY") * ci("J");
    ck_ca_a("Knu") -= 4 * H.block("ac")("vn") * cc1a_("KJuv") * ci("J");
    ck_ca_a("Knu") -= 4 * V_N_Alpha.block("ac")("vn") * cc1a_("KJuv") * ci("J");
    ck_ca_a("Knu") -= 4 * V_N_Beta.block("ac")("vn") * cc1a_("KJuv") * ci("J");
    ck_ca_a("Knu") -= 2 * V.block("aaca")("xynv") * cc2aa_("KJuvxy") * ci("J");
    ck_ca_a("Knu") -= 4 * V.block("aAcA")("xYnV") * cc2ab_("KJuVxY") * ci("J");

    // NOTICE beta
    ck_ca_b("KNU") += 4 * V.block("AACA")("UYNX") * cc1b_("KJXY") * ci("J");
    ck_ca_b("KNU") += 4 * V.block("aAaC")("yUxN") * cc1a_("KJxy") * ci("J");
    ck_ca_b("KNU") -= 4 * H.block("AC")("VN") * cc1b_("KJUV") * ci("J");
    ck_ca_b("KNU") -= 4 * V_all_Beta.block("AC")("VN") * cc1b_("KJUV") * ci("J");
    ck_ca_b("KNU") -= 4 * V_R_Beta.block("AC")("VN") * cc1b_("KJUV") * ci("J");
    ck_ca_b("KNU") -= 2 * V.block("AACA")("XYNV") * cc2bb_("KJUVXY") * ci("J");
    ck_ca_b("KNU") -= 4 * V.block("aAaC")("xYvN") * cc2ab_("KJvUxY") * ci("J");

    /// contribution from Alpha
    ck_ca_a("Knu") += -4 * ci("K") * V.block("aaca")("uynx") * Gamma1.block("aa")("xy");
    ck_ca_a("Knu") += -4 * ci("K") * V.block("aAcA")("uYnX") * Gamma1.block("AA")("XY");
    ck_ca_a("Knu") +=  4 * ci("K") * H.block("ac")("vn") * Gamma1.block("aa")("uv");
    ck_ca_a("Knu") +=  4 * ci("K") * V_N_Alpha.block("ac")("vn") * Gamma1.block("aa")("uv");
    ck_ca_a("Knu") +=  4 * ci("K") * V_N_Beta.block("ac")("vn") * Gamma1.block("aa")("uv");
    ck_ca_a("Knu") +=  2 * ci("K") * V.block("aaca")("xynv") * Gamma2.block("aaaa")("uvxy");
    ck_ca_a("Knu") +=  4 * ci("K") * V.block("aAcA")("xYnV") * Gamma2.block("aAaA")("uVxY");

    // NOTICE beta
    ck_ca_b("KNU") += -4 * ci("K") * V.block("AACA")("UYNX") * Gamma1.block("AA")("XY");
    ck_ca_b("KNU") += -4 * ci("K") * V.block("aAaC")("yUxN") * Gamma1.block("aa")("xy");
    ck_ca_b("KNU") +=  4 * ci("K") * H.block("AC")("VN") * Gamma1.block("AA")("UV");
    ck_ca_b("KNU") +=  4 * ci("K") * V_all_Beta.block("AC")("VN") * Gamma1.block("AA")("UV");
    ck_ca_b("KNU") +=  4 * ci("K") * V_R_Beta.block("AC")("VN") * Gamma1.block("AA")("UV");
    ck_ca_b("KNU") +=  2 * ci("K") * V.block("AACA")("XYNV") * Gamma2.block("AAAA")("UVXY");
    ck_ca_b("KNU") +=  4 * ci("K") * V.block("aAaC")("xYvN") * Gamma2.block("aAaA")("vUxY");

    // virtual-active
    ck_va_a("Keu") += 4 * H.block("av")("ve") * cc1a_("KJuv") * ci("J");
    ck_va_a("Keu") += 4 * V_N_Alpha.block("av")("ve") * cc1a_("KJuv") * ci("J");
    ck_va_a("Keu") += 4 * V_N_Beta.block("av")("ve") * cc1a_("KJuv") * ci("J");
    ck_va_a("Keu") += 2 * V.block("aava")("xyev") * cc2aa_("KJuvxy") * ci("J");
    ck_va_a("Keu") += 4 * V.block("aAvA")("xYeV") * cc2ab_("KJuVxY") * ci("J");

    // NOTICE beta
    ck_va_b("KEU") += 4 * H.block("AV")("VE") * cc1b_("KJUV") * ci("J");
    ck_va_b("KEU") += 4 * V_all_Beta.block("AV")("VE") * cc1b_("KJUV") * ci("J");
    ck_va_b("KEU") += 4 * V_R_Beta.block("AV")("VE") * cc1b_("KJUV") * ci("J");
    ck_va_b("KEU") += 2 * V.block("AAVA")("XYEV") * cc2bb_("KJUVXY") * ci("J");
    ck_va_b("KEU") += 4 * V.block("aAaV")("xYvE") * cc2ab_("KJvUxY") * ci("J");

    /// contribution from Alpha
    ck_va_a("Keu") += -4 * ci("K") * H.block("av")("ve") * Gamma1.block("aa")("uv");
    ck_va_a("Keu") += -4 * ci("K") * V_N_Alpha.block("av")("ve") * Gamma1.block("aa")("uv");
    ck_va_a("Keu") += -4 * ci("K") * V_N_Beta.block("av")("ve") * Gamma1.block("aa")("uv");
    ck_va_a("Keu") += -2 * ci("K") * V.block("aava")("xyev") * Gamma2.block("aaaa")("uvxy");
    ck_va_a("Keu") += -4 * ci("K") * V.block("aAvA")("xYeV") * Gamma2.block("aAaA")("uVxY");

    // NOTICE beta
    ck_va_b("KEU") += -4 * ci("K") * H.block("AV")("VE") * Gamma1.block("AA")("UV");
    ck_va_b("KEU") += -4 * ci("K") * V_all_Beta.block("AV")("VE") * Gamma1.block("AA")("UV");
    ck_va_b("KEU") += -4 * ci("K") * V_R_Beta.block("AV")("VE") * Gamma1.block("AA")("UV");
    ck_va_b("KEU") += -2 * ci("K") * V.block("AAVA")("XYEV") * Gamma2.block("AAAA")("UVXY");
    ck_va_b("KEU") += -4 * ci("K") * V.block("aAaV")("xYvE") * Gamma2.block("aAaA")("vUxY");

    // active-active
    ck_aa_a("Kuv") += 2 * V.block("aaaa")("uyvx") * cc1a_("KJxy") * ci("J");
    ck_aa_a("Kuv") += 2 * V.block("aAaA")("uYvX") * cc1b_("KJXY") * ci("J");

    // NOTICE beta
    ck_aa_b("KUV") += 2 * V.block("AAAA")("UYVX") * cc1b_("KJXY") * ci("J");
    ck_aa_b("KUV") += 2 * V.block("aAaA")("yUxV") * cc1a_("KJxy") * ci("J");

    /// contribution from Alpha
    ck_aa_a("Kuv") += -2 * ci("K") * V.block("aaaa")("uyvx") * Gamma1.block("aa")("xy");
    ck_aa_a("Kuv") += -2 * ci("K") * V.block("aAaA")("uYvX") * Gamma1.block("AA")("XY");

    // NOTICE beta
    ck_aa_b("KUV") += -2 * ci("K") * V.block("AAAA")("UYVX") * Gamma1.block("AA")("XY");
    ck_aa_b("KUV") += -2 * ci("K") * V.block("aAaA")("yUxV") * Gamma1.block("aa")("xy");

    // CI equations' contribution to A

    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"vc","ca","va","aa"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];

            auto temp_ci = ck_vc_a;
            if (col == "ca") temp_ci = ck_ca_a;
            else if (col == "va") temp_ci = ck_va_a;
            else if (col == "aa") temp_ci = ck_aa_a;


            if (col != "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0]) + dim * (pre2 + i[1] * idx2 + i[2]);
                    A.at(index) += value;
                });
            }

            else if (col == "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i1 = i[1] > i[2]? i[1]: i[2],
                        i2 = i[1] > i[2]? i[2]: i[1];
                    if (i1 != i2 ) {
                        int index = (pre1 + i[0]) + dim * (pre2 + i1 * (i1 - 1) / 2 + i2);
                        A.at(index) += value;    
                    }
                });   
            }
        }
    } 


    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"VC","CA","VA","AA"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];

            auto temp_ci = ck_vc_b;
            if (col == "CA") temp_ci = ck_ca_b;
            else if (col == "VA") temp_ci = ck_va_b;
            else if (col == "AA") temp_ci = ck_aa_b;


            if (col != "AA") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0]) + dim * (pre2 + i[1] * idx2 + i[2]);
                    A.at(index) += value;
                });
            }

            else if (col == "AA") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i1 = i[1] > i[2]? i[1]: i[2],
                        i2 = i[1] > i[2]? i[2]: i[1];
                    if (i1 != i2 ) {
                        int index = (pre1 + i[0]) + dim * (pre2 + i1 * (i1 - 1) / 2 + i2);
                        A.at(index) += value;    
                    }
                });   
            }
        }
    } 

    auto ck_ci = ambit::Tensor::build(ambit::CoreTensor, "ci equations ci multiplier part", {ndets, ndets});
    auto I_ci = ambit::Tensor::build(ambit::CoreTensor, "identity", {ndets, ndets});
    auto one_ci = ambit::Tensor::build(ambit::CoreTensor, "one", {ndets, ndets});

    I_ci.iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });
    one_ci.iterate([&](const std::vector<size_t>& i, double& value) {
        value = 1.0;
    });


    ck_ci("KI") += H.block("cc")("mn") * I.block("cc")("mn") * I_ci("KI");
    ck_ci("KI") += H.block("CC")("MN") * I.block("CC")("MN") * I_ci("KI");
    ck_ci("KI") += cc1a_("KIuv") * H.block("aa")("uv");
    ck_ci("KI") += cc1b_("KIUV") * H.block("AA")("UV");

    ck_ci("KI") += 0.5 * V["m,n,m1,n1"] * I["m,m1"] * I["n,n1"] * I_ci("KI");
    ck_ci("KI") += 0.5 * V["M,N,M1,N1"] * I["M,M1"] * I["N,N1"] * I_ci("KI");
    ck_ci("KI") +=       V["m,N,m1,N1"] * I["m,m1"] * I["N,N1"] * I_ci("KI");

    ck_ci("KI") += cc1a_("KIuv") * V.block("acac")("umvn") * I.block("cc")("mn");
    ck_ci("KI") += cc1b_("KIUV") * V.block("ACAC")("UMVN") * I.block("CC")("MN");
    
    ck_ci("KI") += cc1a_("KIuv") * V_N_Beta.block("aa")("uv");
    ck_ci("KI") += cc1b_("KIUV") * V_R_Beta.block("AA")("UV");

    ck_ci("KI") += 0.25 * cc2aa_("KIuvxy") * V.block("aaaa")("uvxy");
    ck_ci("KI") += 0.25 * cc2bb_("KIUVXY") * V.block("AAAA")("UVXY");
    ck_ci("KI") += 1.00 * cc2ab_("KIuVxY") * V.block("aAaA")("uVxY");

    // ck_ci("KI") += ints_->nuclear_repulsion_energy() * one_ci("KI");
    std::cout<< "nuclear_repulsion_energy = " << std::setprecision(9) << ints_->nuclear_repulsion_energy() << std::endl;
    // ck_ci.print();


    double Et = 0.0;
    Et += H.block("cc")("mn") * I.block("cc")("mn");
    Et += H.block("CC")("MN") * I.block("CC")("MN");
    Et += cc1a_("KIuv") * H.block("aa")("uv") * ci("I") * ci("K");
    Et += cc1b_("KIUV") * H.block("AA")("UV") * ci("I") * ci("K");

    Et += 0.5 * V["m,n,m1,n1"] * I["m,m1"] * I["n,n1"];
    Et += 0.5 * V["M,N,M1,N1"] * I["M,M1"] * I["N,N1"];
    Et +=       V["m,N,m1,N1"] * I["m,m1"] * I["N,N1"];

    Et += 0.25 * cc2aa_("KIuvxy") * V.block("aaaa")("uvxy") * ci("I") * ci("K");
    Et += 0.25 * cc2bb_("KIUVXY") * V.block("AAAA")("UVXY") * ci("I") * ci("K");
    Et += 1.00 * cc2ab_("KIuVxY") * V.block("aAaA")("uVxY") * ci("I") * ci("K");

    Et += cc1a_("KIuv") * V.block("acac")("umvn") * ci("I") * ci("K") * I.block("cc")("mn");
    Et += cc1b_("KIUV") * V.block("ACAC")("UMVN") * ci("I") * ci("K") * I.block("CC")("MN");
    Et += cc1a_("KIuv") * V.block("aCaC")("uMvN") * ci("I") * ci("K") * I.block("CC")("MN");
    Et += cc1b_("KIUV") * V.block("cAcA")("mUnV") * ci("I") * ci("K") * I.block("cc")("mn");

    // Et += ints_->nuclear_repulsion_energy();
    std::cout<< "E test = " << std::setprecision(9) << Et << std::endl;

    std::cout<< "ndet = " << ndets << std::endl;


    ck_ci("KI") -= Et * I_ci("KI");

    // Just for test, remember to delete
    ck_ci("KI") += 0.0001 * I_ci("KI");

    ck_ci.print(stdout,"%16.14f");

    (ci).iterate([&](const std::vector<size_t>& i, double& value) {
        std::cout << std::setprecision(16) << value << std::endl;
    });


    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"ci"}) {
            int pre2 = preidx[col];

            (ck_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = (pre1 + i[0]) + dim * (pre2 + i[1]);
                A.at(index) += value;
            });

        }
    } 

    outfile->Printf( "\nCI\n" );
    auto pre = preidx["ci"];
    for(int i = 0; i < ndets; i++ ) {
        outfile->Printf( " %.16f ", b[pre + i]);
        outfile->Printf( "\n" );
    }


    outfile->Printf("Done");
    outfile->Printf("\n    Solving Off-diagonal Blocks of Z ................ ");
    int info;
    info = C_DGESV( n, nrhs, &A[0], lda, &ipiv[0], &b[0], ldb);
    std::cout << "info = " << info << std::endl; 

    // Print the solution
    // outfile->Printf( "\nVC\n" );
    // for(int i = 0; i < nvirt_; i++ ) {
    //     for(int j = 0; j < ncore_; j++ ) outfile->Printf( " %.10f ", b[i * ncore_ + j]);
    //     outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\nCA\n" );
    // int pre = preidx["ca"];
    // for(int i = 0; i < ncore_; i++ ) {
    //     for(int j = 0; j < na_; j++ ) outfile->Printf( " %.10f ", b[pre + i * na_ + j]);
    //     outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\nVA\n" );
    // pre = preidx["va"];
    // for(int i = 0; i < nvirt_; i++ ) {
    //     for(int j = 0; j < na_; j++ ) outfile->Printf( " %.10f ", b[pre + i * na_ + j]);
    //     outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\nAA\n" );
    // pre = preidx["aa"];
    // for(int i = 1; i < na_; i++ ) {
    //     for(int j = 0; j < i; j++ ) outfile->Printf( " %.10f ", b[pre + i * (i - 1) / 2 + j]);
    //     outfile->Printf( "\n" );
    // }

    outfile->Printf( "\nCI\n" );
    pre = preidx["ci"];
    for(int i = 0; i < ndets; i++ ) {
        outfile->Printf( " %.16f ", b[pre + i]);
        outfile->Printf( "\n" );
    }

    ci.print();

    for (const std::string& block : {"vc","ca","va","aa"}) {
        int pre = preidx[block],
            idx = block_dim[block];
        if (block != "aa") {
            (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = pre + i[0] * idx + i[1];
                value = b.at(index);
            });
        }
        else {
            (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                int i0 = i[0] > i[1]? i[0]: i[1],
                    i1 = i[0] > i[1]? i[1]: i[0];
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

    // // test need to be deleted
    x_ci.zero();
    double mm = foptions_->get_double("SIGMA");

    std::vector<double> ttt = {0.9269376610043215*mm,0.3752153683044413*mm};

    for (const std::string& block : {"ci"}) {
        (x_ci).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = i[0];
            value = ttt.at(index);
        });
    } 

    scale_ci -= x_ci("K") * ci("K");
    std::cout<< "scale = " << scale_ci << std::endl;

    Z["me"] = Z["em"];
    Z["wm"] = Z["mw"];
    Z["we"] = Z["ew"];

    // Beta part
    for (const std::string& block : {"VC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("vc").data()[i[0] * ncore_ + i[1]];
        });
    }
    for (const std::string& block : {"CA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("ca").data()[i[0] * na_ + i[1]];
        });
    } 
    for (const std::string& block : {"VA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("va").data()[i[0] * na_ + i[1]];
        });
    } 

    for (const std::string& block : {"AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("aa").data()[i[0] * na_ + i[1]];
        });
    } 

    Z["ME"] = Z["EM"];
    Z["WM"] = Z["MW"];
    Z["WE"] = Z["EW"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::tpdm_backtransform() {
    // Backtransform the TPDM
    // NOTICE: This function also appears in the CASSCF gradient code thus can be refined in the future!!
    
    std::vector<std::shared_ptr<psi::MOSpace>> spaces;
    spaces.push_back(psi::MOSpace::all);
    std::shared_ptr<TPDMBackTransform> transform =
        std::shared_ptr<TPDMBackTransform>(new TPDMBackTransform(
            ints_->wfn(), spaces,
            IntegralTransform::TransformationType::Unrestricted, // Transformation type
            IntegralTransform::OutputType::DPDOnly,              // Output buffer
            IntegralTransform::MOOrdering::QTOrder,              // MO ordering
            IntegralTransform::FrozenOrbitals::None));           // Frozen orbitals?
    transform->backtransform_density();
    transform.reset();

    outfile->Printf("\n    TPDM Backtransformation ......................... Done");
}

void DSRG_MRPT2::write_lagrangian() {
    // NOTICE: write the Lagrangian
    outfile->Printf("\n    Writing Lagrangian .............................. ");

    SharedMatrix L(new Matrix("Lagrangian", nirrep_, irrep_vec, irrep_vec));

    for (const std::string& block : {"cc", "CC", "aa", "AA", "ca", "ac", "CA", "AC",
                    "vv", "VV", "av", "cv", "va", "vc", "AV", "CV", "VA", "VC"}) {
        std::vector<std::vector<std::pair<unsigned long, unsigned long>,
                                std::allocator<std::pair<unsigned long, unsigned long>>>>
            spin_pair;
        for (size_t idx : {0, 1}) {
            auto spin = std::tolower(block.at(idx));
            if (spin == 'c') {
                spin_pair.push_back(core_mos_relative);
            } else if (spin == 'a') {
                spin_pair.push_back(actv_mos_relative);
            }
            else if (spin == 'v') {
                spin_pair.push_back(virt_mos_relative);
            }
        }

        (W_.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (spin_pair[0][i[0]].first == spin_pair[1][i[1]].first) {
                L->add(spin_pair[0][i[0]].first, spin_pair[0][i[0]].second,
                       spin_pair[1][i[1]].second, value);
            }
        });
    }

    L->back_transform(ints_->Ca());
    ints_->wfn()->set_Lagrangian(SharedMatrix(new Matrix("Lagrangian", nirrep_, irrep_vec, irrep_vec)));
    ints_->wfn()->Lagrangian()->copy(L);

    outfile->Printf("Done");
}


/**
 * Write spin_dependent one-RDMs coefficients.
 *
 * We force "Da == Db". This function needs be changed if such constraint is revoked.
 */
void DSRG_MRPT2::write_1rdm_spin_dependent() {
    // NOTICE: write spin_dependent one-RDMs coefficients. 
    outfile->Printf("\n    Writing 1RDM Coefficients ....................... ");
    SharedMatrix D1(new Matrix("1rdm coefficients contribution", nirrep_, irrep_vec, irrep_vec));

    (Z.block("vc")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (virt_mos_relative[i[0]].first == core_mos_relative[i[1]].first) {
            D1->set(virt_mos_relative[i[0]].first, virt_mos_relative[i[0]].second,
                core_mos_relative[i[1]].second, value);
            D1->set(virt_mos_relative[i[0]].first, core_mos_relative[i[1]].second,
                virt_mos_relative[i[0]].second, value);
        }
    });

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"ca"});
    temp["nu"] = Z["un"];
    temp["nv"] -= Z["un"] * Gamma1["uv"];

    (temp.block("ca")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->set(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, value);
            D1->set(core_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
                core_mos_relative[i[0]].second, value);
        }
    });

    temp = BTF_->build(CoreTensor, "temporal tensor", {"va"});
    temp["ev"] = Z["eu"] * Gamma1["uv"];

    (temp.block("va")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (virt_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->set(virt_mos_relative[i[0]].first, virt_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, value);
            D1->set(virt_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
                virt_mos_relative[i[0]].second, value);
        }
    });

    (Z.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == core_mos_relative[i[1]].first) {
            D1->set(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                core_mos_relative[i[1]].second, value);
        }
    });

    (Z.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->set(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, value);
        }
    });

    (Z.block("vv")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (virt_mos_relative[i[0]].first == virt_mos_relative[i[1]].first) {
            D1->set(virt_mos_relative[i[0]].first, virt_mos_relative[i[0]].second,
                virt_mos_relative[i[1]].second, value);
        }
    });

    // CASSCF reference
    for (size_t i = 0, size_c = core_mos_relative.size(); i < size_c; ++i) {
        D1->add(core_mos_relative[i].first, core_mos_relative[i].second,
                core_mos_relative[i].second, 1.0);
    }


    (Gamma1_.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                    actv_mos_relative[i[1]].second, value * scale_ci);
        }
    });

    // CI contribution
    auto cc = coupling_coefficients_;
    auto ci = ci_vectors_[0];
    auto cc1a_ = cc.cc1a();
    auto cc1b_ = cc.cc1b();
    auto cc2aa_ = cc.cc2aa();
    auto cc2bb_ = cc.cc2bb();
    auto cc2ab_ = cc.cc2ab();

    auto tp = ambit::Tensor::build(ambit::CoreTensor, "temporal tensor", {na_, na_});

    // it has been tested that cc1a_ and cc1b_ yield the same tensor tp
    tp("uv") = x_ci("I") * cc1a_("IJuv") * ci("J");

    tp.print();

    (tp).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                    actv_mos_relative[i[1]].second, value);
        }
    });

    D1->back_transform(ints_->Ca());
    ints_->wfn()->Da()->copy(D1);
    ints_->wfn()->Db()->copy(D1);

    outfile->Printf("Done");
}

/**
 * Write spin_dependent two-RDMs coefficients using IWL.
 *
 * Coefficients in d2aa and d2bb need be multiplied with additional 1/2!
 * Specifically:
 * If you have v_aa as coefficients before 2-RDMs_alpha_alpha, v_bb before
 * 2-RDMs_beta_beta and v_bb before 2-RDMs_alpha_beta, you need to write
 * 0.5 * v_aa, 0.5 * v_bb and v_ab into the IWL file instead of using
 * the original coefficients v_aa, v_bb and v_ab.
 */
void DSRG_MRPT2::write_2rdm_spin_dependent() {
    // TODO: write spin_dependent two-RDMs coefficients using IWL
    outfile->Printf("\n    Writing 2RDM Coefficients ....................... ");

    auto cc = coupling_coefficients_;
    auto ci = ci_vectors_[0];
    auto cc1a_ = cc.cc1a();
    auto cc1b_ = cc.cc1b();
    auto cc2aa_ = cc.cc2aa();
    auto cc2bb_ = cc.cc2bb();
    auto cc2ab_ = cc.cc2ab();
    auto psio_ = _default_psio_lib_;
    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto m = core_all_[i];
        for (size_t a = 0, size_v = virt_all_.size(); a < size_v; ++a) {
            auto e = virt_all_[a];
            auto idx = a * ncore_ + i;
            auto z_a = Z.block("vc").data()[idx];
            auto z_b = Z.block("VC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(m, e, m1, m1, z_a, 0, "NULL", 0);
                d2bb.write_value(m, e, m1, m1, z_b, 0, "NULL", 0);
                d2aa.write_value(m, m1, m1, e, -z_a, 0, "NULL", 0);
                d2bb.write_value(m, m1, m1, e, -z_b, 0, "NULL", 0);
                
                d2ab.write_value(m, e, m1, m1, 2.0 * (z_a + z_b), 0, "NULL", 0);
            }
        }
    }

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"ac", "AC"});
    temp["un"] = Z["un"];
    temp["un"] -= Z["vn"] * Gamma1["uv"];
    temp["UN"] = Z["UN"];
    temp["UN"] -= Z["VN"] * Gamma1["UV"];

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t a = 0, size_a = actv_all_.size(); a < size_a; ++a) {
            auto u = actv_all_[a];
            auto idx = a * ncore_ + i;
            auto z_a = temp.block("ac").data()[idx];
            auto z_b = temp.block("AC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];              
                if (n != m1) {
                    d2aa.write_value(u, n, m1, m1, z_a, 0, "NULL", 0);
                    d2bb.write_value(u, n, m1, m1, z_b, 0, "NULL", 0);
                    d2aa.write_value(u, m1, m1, n, -z_a, 0, "NULL", 0);
                    d2bb.write_value(u, m1, m1, n, -z_b, 0, "NULL", 0);
                }
                
                d2ab.write_value(u, n, m1, m1, 2.0 * (z_a + z_b), 0, "NULL", 0);
            }
        }
    }

    temp = BTF_->build(CoreTensor, "temporal tensor", {"va", "VA"});

    temp["ev"] = Z["eu"] * Gamma1["uv"];
    temp["EV"] = Z["EU"] * Gamma1["UV"];

    for (size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        auto v = actv_all_[i];
        for (size_t a = 0, size_v = virt_all_.size(); a < size_v; ++a) {
            auto e = virt_all_[a];
            auto idx = a * na_ + i;
            auto z_a = temp.block("va").data()[idx];
            auto z_b = temp.block("VA").data()[idx];
            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(v, e, m1, m1, z_a, 0, "NULL", 0);
                d2bb.write_value(v, e, m1, m1, z_b, 0, "NULL", 0);
                d2aa.write_value(v, m1, m1, e, -z_a, 0, "NULL", 0);
                d2bb.write_value(v, m1, m1, e, -z_b, 0, "NULL", 0);
                
                d2ab.write_value(v, e, m1, m1, 2.0 * (z_a + z_b), 0, "NULL", 0);
            }
        }
    }

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t k = 0; k < size_c; ++k) {
            auto m = core_all_[k];
            auto idx = k * ncore_ + i;
            auto z_a = Z.block("cc").data()[idx];
            auto z_b = Z.block("CC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];

                double v1 = 0.5 * z_a, v2 = 0.5 * z_b, v3 = z_a + z_b;

                if (m == n) {
                    v1 += 0.25;
                    v2 += 0.25;
                    v3 += 1.00;
                }

                if (m != m1) {

                    d2aa.write_value(n, m, m1, m1,  v1, 0, "NULL", 0);
                    d2bb.write_value(n, m, m1, m1,  v2, 0, "NULL", 0);
                    d2aa.write_value(n, m1, m1, m, -v1, 0, "NULL", 0);
                    d2bb.write_value(n, m1, m1, m, -v2, 0, "NULL", 0);
                }
                
                d2ab.write_value(n, m, m1, m1, v3, 0, "NULL", 0);
            }
        }
    }

    auto ci_g1_a = ambit::Tensor::build(ambit::CoreTensor, "effective alpha gamma tensor", {na_, na_});
    auto ci_g1_b = ambit::Tensor::build(ambit::CoreTensor, "effective beta gamma tensor", {na_, na_});

    ci_g1_a("uv") += x_ci("I") * cc1a_("IJuv") * ci("J");
    ci_g1_b("UV") += x_ci("I") * cc1b_("IJUV") * ci("J");

    for (size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        auto v = actv_all_[i];
        for (size_t k = 0; k < size_a; ++k) {
            auto u = actv_all_[k];
            auto idx = k * na_ + i;
            auto z_a = Z.block("aa").data()[idx];
            auto z_b = Z.block("AA").data()[idx];
            auto gamma_a = Gamma1_.block("aa").data()[idx];
            auto gamma_b = Gamma1_.block("AA").data()[idx];
            auto ci_gamma_a = ci_g1_a.data()[idx];
            auto ci_gamma_b = ci_g1_b.data()[idx];
            auto v1 = z_a + scale_ci * gamma_a + ci_gamma_a;
            auto v2 = z_b + scale_ci * gamma_b + ci_gamma_b;

            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(v, u, m1, m1, 0.5 * v1, 0, "NULL", 0);
                d2bb.write_value(v, u, m1, m1, 0.5 * v2, 0, "NULL", 0);
                d2aa.write_value(v, m1, m1, u, -0.5 * v1, 0, "NULL", 0);
                d2bb.write_value(v, m1, m1, u, -0.5 * v2, 0, "NULL", 0);
                
                d2ab.write_value(v, u, m1, m1, (v1 + v2), 0, "NULL", 0);
            }
        }
    }

    for (size_t i = 0, size_v = virt_all_.size(); i < size_v; ++i) {
        auto f = virt_all_[i];
        for (size_t k = 0; k < size_v; ++k) {
            auto e = virt_all_[k];
            auto idx = k * nvirt_ + i;
            auto z_a = Z.block("vv").data()[idx];
            auto z_b = Z.block("VV").data()[idx];
            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(f, e, m1, m1, 0.5 * z_a, 0, "NULL", 0);
                d2bb.write_value(f, e, m1, m1, 0.5 * z_b, 0, "NULL", 0);
                d2aa.write_value(f, m1, m1, e, -0.5 * z_a, 0, "NULL", 0);
                d2bb.write_value(f, m1, m1, e, -0.5 * z_b, 0, "NULL", 0);
                
                d2ab.write_value(f, e, m1, m1, (z_a + z_b), 0, "NULL", 0);
            }
        }
    }

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t j = 0; j < size_c; ++j) {
            auto m = core_all_[j];
            auto idx = j * ncore_ + i;
            auto z_a = Z.block("cc").data()[idx];
            auto z_b = Z.block("CC").data()[idx];
            for (size_t k = 0, size_a = actv_all_.size(); k < size_a; ++k) {
                auto v1 = actv_all_[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto u1 = actv_all_[l];
                    auto idx1 = l * na_ + k;
                    auto g_a = Gamma1.block("aa").data()[idx1];
                    auto g_b = Gamma1.block("AA").data()[idx1];
                
                    d2aa.write_value(n, m, v1, u1, 0.5 * z_a * g_a, 0, "NULL", 0);
                    d2bb.write_value(n, m, v1, u1, 0.5 * z_b * g_b, 0, "NULL", 0);
                    d2aa.write_value(n, u1, v1, m, -0.5 * z_a * g_a, 0, "NULL", 0);
                    d2bb.write_value(n, u1, v1, m, -0.5 * z_b * g_b, 0, "NULL", 0);
                    
                    d2ab.write_value(n, m, v1, u1, (z_a * g_b + z_b * g_a), 0, "NULL", 0);
                }
            }
        }
    }

    for (size_t i = 0, size_v = virt_all_.size(); i < size_v; ++i) {
        auto f = virt_all_[i];
        for (size_t j = 0; j < size_v; ++j) {
            auto e = virt_all_[j];
            auto idx = j * nvirt_ + i;
            auto z_a = Z.block("vv").data()[idx];
            auto z_b = Z.block("VV").data()[idx];
            for (size_t k = 0, size_a = actv_all_.size(); k < size_a; ++k) {
                auto v1 = actv_all_[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto u1 = actv_all_[l];
                    auto idx1 = l * na_ + k;
                    auto g_a = Gamma1.block("aa").data()[idx1];
                    auto g_b = Gamma1.block("AA").data()[idx1];
                
                    d2aa.write_value(f, e, v1, u1, 0.5 * z_a * g_a, 0, "NULL", 0);
                    d2bb.write_value(f, e, v1, u1, 0.5 * z_b * g_b, 0, "NULL", 0);
                    d2aa.write_value(f, u1, v1, e, -0.5 * z_a * g_a, 0, "NULL", 0);
                    d2bb.write_value(f, u1, v1, e, -0.5 * z_b * g_b, 0, "NULL", 0);
                    
                    d2ab.write_value(f, e, v1, u1, (z_a * g_b + z_b * g_a), 0, "NULL", 0);
                }
            }
        }
    }

    // terms with overlap
    temp = BTF_->build(CoreTensor, "temporal tensor", {"pphh", "PPHH", "pPhH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"phph","phPH"});

    if (PT2_TERM) {
        temp["abij"] += Tau1["ijab"];
        temp["ABIJ"] += Tau1["IJAB"];
        temp["aBiJ"] += Tau1["iJaB"];

        temp["cdkl"] += Kappa["klcd"] * Eeps2_p["klcd"];
        temp["CDKL"] += Kappa["KLCD"] * Eeps2_p["KLCD"];
        temp["cDkL"] += Kappa["kLcD"] * Eeps2_p["kLcD"];
    }

    temp["xynv"] -= Z["un"] * Gamma2["uvxy"]; 
    temp["XYNV"] -= Z["UN"] * Gamma2["UVXY"]; 
    temp["xYnV"] -= Z["un"] * Gamma2["uVxY"]; 

    temp["evxy"] += Z["eu"] * Gamma2["uvxy"];
    temp["EVXY"] += Z["EU"] * Gamma2["UVXY"];
    temp["eVxY"] += Z["eu"] * Gamma2["uVxY"];

    // CASSCF reference
    temp["xyuv"] += 0.25 * scale_ci * Gamma2["uvxy"];
    temp["XYUV"] += 0.25 * scale_ci * Gamma2["UVXY"];
    temp["xYuV"] += 0.25 * scale_ci * Gamma2["uVxY"];

    // CI contribution
    temp.block("aaaa")("xyuv") += 0.25 * cc2aa_("IJuvxy") * x_ci("I") * ci("J");
    temp.block("AAAA")("XYUV") += 0.25 * cc2bb_("IJUVXY") * x_ci("I") * ci("J");
    temp.block("aAaA")("xYuV") += 0.25 * cc2ab_("IJuVxY") * x_ci("I") * ci("J");

    // all-alpha and all-beta
    temp2["ckdl"] += temp["cdkl"];
    temp2["cldk"] -= temp["cdkl"];
    // alpha-beta
    temp2["ckDL"] += 2.0 * temp["cDkL"];
    temp2["clDK"] += 2.0 * temp["cDlK"];
    temp.zero();

    temp["eumv"] += 2.0 * Z["em"] * Gamma1["uv"];
    temp["EUMV"] += 2.0 * Z["EM"] * Gamma1["UV"];
    temp["eUmV"] += 2.0 * Z["em"] * Gamma1["UV"];

    temp["u,v1,n,u1"] += 2.0 * Z["un"] * Gamma1["u1,v1"]; 
    temp["U,V1,N,U1"] += 2.0 * Z["UN"] * Gamma1["U1,V1"];
    temp["u,V1,n,U1"] += 2.0 * Z["un"] * Gamma1["U1,V1"]; 

    temp["v,v1,u,u1"] += Z["uv"] * Gamma1["u1,v1"];
    temp["V,V1,U,U1"] += Z["UV"] * Gamma1["U1,V1"];
    temp["v,V1,u,U1"] += Z["uv"] * Gamma1["U1,V1"];

    // all-alpha and all-beta
    temp2["ckdl"] += temp["cdkl"];
    temp2["cldk"] -= temp["cdkl"];
    // alpha-beta
    temp2["ckDL"] += 2.0 * temp["cDkL"];

    temp2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (std::fabs(value) > 1e-12) {
            if (spin[2] == AlphaSpin) {
                d2aa.write_value(i[0], i[1], i[2], i[3], 0.5 * value, 0, "NULL", 0);          
                d2bb.write_value(i[0], i[1], i[2], i[3], 0.5 * value, 0, "NULL", 0);          
            }
            else {
                d2ab.write_value(i[0], i[1], i[2], i[3], value, 0, "NULL", 0); 
            }
        }
    });

    d2aa.flush(1);
    d2bb.flush(1);
    d2ab.flush(1);

    d2aa.set_keep_flag(1);
    d2bb.set_keep_flag(1);
    d2ab.set_keep_flag(1);

    d2aa.close();
    d2bb.close();
    d2ab.close();

    outfile->Printf("Done");
}

SharedMatrix DSRG_MRPT2::compute_gradient() {
    // NOTICE: compute the DSRG_MRPT2 gradient 
    print_method_banner({"DSRG-MRPT2 Gradient", "Shuhe Wang"});
    set_all_variables();
    set_multiplier();
    write_lagrangian();
    write_1rdm_spin_dependent();
    write_2rdm_spin_dependent();
    tpdm_backtransform();

    outfile->Printf("\n    Computing Gradient .............................. Done\n");

    return std::make_shared<Matrix>("nullptr", 0, 0);
}

} 