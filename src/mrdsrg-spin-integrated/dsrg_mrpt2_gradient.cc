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
    // set_alpha();
    set_z();
    // set_CI();
    outfile->Printf("\n    Solving Entries of W ............................ ");
    set_w();   
    outfile->Printf("Done");
}


void DSRG_MRPT2::set_z() {
    outfile->Printf("\n    Initializing Diagonal Entries of Z .............. ");
    set_z_cc();  
    set_z_vv();
    set_z_aa_diag();
    outfile->Printf("Done");
    // Jacobi iterative solver
    // iter_z();
    // LAPACK solver
    solve_z();
}

void DSRG_MRPT2::set_CI() {

    // Set the CI-based one-body coefficients for Z-vector equations
    CI_1["uv"] += 2.0 * H["vu"];
    CI_1["uv"] += 2.0 * V_N_Alpha["v,u"];
    CI_1["uv"] += 2.0 * V_N_Beta["v,u"];


    CI_1["uv"] += 0.5 * V_["cdul"] * T2_["vjab"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    CI_1["uv"] += V_["cDuL"] * T2_["vJaB"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    CI_1["uv"] += 0.50 * V_["cdku"] * T2_["ivab"] * Gamma1["ki"] * Eta1["ac"] * Eta1["bd"];
    CI_1["uv"] += V_["dCuK"] * T2_["vIbA"] * Gamma1["KI"] * Eta1["AC"] * Eta1["bd"];

    CI_1["uv"] -= 0.50 * V_["vdkl"] * T2_["ijub"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    CI_1["uv"] -= V_["vDkL"] * T2_["iJuB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];


    CI_1["uv"] -= 0.50 * V_["cvkl"] * T2_["ijau"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"];
    CI_1["uv"] -= V_["vClK"] * T2_["jIuA"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["AC"];

    CI_1["uv"] += 4.0 * Z["em"] * V["mveu"];
    CI_1["uv"] += 4.0 * Z["EM"] * V["vMuE"];

    CI_1["uv"] += 2.0 * Z["mn"] * V["mvnu"];
    CI_1["uv"] += 2.0 * Z["MN"] * V["vMuN"];

    CI_1["uv"] += 2.0 * Z["xy"] * V["xvyu"];
    CI_1["uv"] += 2.0 * Z["XY"] * V["vXuY"];

    CI_1["uv"] += 2.0 * Z["ef"] * V["evfu"];
    CI_1["uv"] += 2.0 * Z["EF"] * V["vEuF"];

    CI_1["uv"] += 4.0 * Z["xn"] * V["xvnu"];
    CI_1["uv"] += 4.0 * Z["XN"] * V["vXuN"];

    CI_1["uv"] -= 4.0 * Z["un"] * H["vn"];
    CI_1["uv"] -= 4.0 * Z["un"] * V_N_Alpha["v,n"];
    CI_1["uv"] -= 4.0 * Z["un"] * V_N_Beta["v,n"];


    CI_1["uv"] += 4.0 * Z["eu"] * H["ve"];
    CI_1["uv"] += 4.0 * Z["eu"] * V_N_Alpha["v,e"];
    CI_1["uv"] += 4.0 * Z["eu"] * V_N_Beta["v,e"];


    // Set the CI-based two-body coefficients for Z-vector equations
    CI_2["uvxy"] += 0.5 * V["xyuv"];
    CI_2["uvxy"] -= 2.0 * Z["un"] * V["xynv"];
    CI_2["uvxy"] += 2.0 * Z["eu"] * V["xyev"];

    CI_2["uVxY"] += 2.0 * V["xYuV"];
    CI_2["uVxY"] -= 8.0 * Z["un"] * V["xYnV"];
    CI_2["uVxY"] += 8.0 * Z["eu"] * V["xYeV"];
}

void DSRG_MRPT2::set_alpha() {
    outfile->Printf("\n    Setting the Multiplier Alpha .................... ");

    Alpha += H["vu"] * Gamma1["uv"];
    Alpha += H["VU"] * Gamma1["UV"];

    Alpha += V_N_Alpha["v,u"] * Gamma1["uv"];
    Alpha += V_N_Beta["v,u"] * Gamma1["uv"];
    Alpha += V["V,M,U,M1"] * Gamma1["UV"] * I["M,M1"];
    Alpha += V["m,V,m1,U"] * Gamma1["UV"] * I["m,m1"];

    Alpha += 0.25 * V["xyuv"] * Gamma2["uvxy"];
    Alpha += 0.25 * V["XYUV"] * Gamma2["UVXY"];
    Alpha += V["xYuV"] * Gamma2["uVxY"];

    Alpha += 0.50 * V_["cdkl"] * T2_["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Alpha += 0.50 * V_["CDKL"] * T2_["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    Alpha += 2.00 * V_["cDkL"] * T2_["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    Alpha -= 0.25 * V_["cdkl"] * T2_["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Gamma1["ac"] * Eta1["bd"];
    Alpha -= 0.25 * V_["CDKL"] * T2_["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Gamma1["AC"] * Eta1["BD"];
    Alpha -= V_["cDkL"] * T2_["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Gamma1["ac"] * Eta1["BD"];

    Alpha -= 0.25 * V_["cdkl"] * T2_["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Gamma1["bd"];
    Alpha -= 0.25 * V_["CDKL"] * T2_["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Gamma1["BD"];
    Alpha -= V_["cDkL"] * T2_["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Gamma1["BD"];

    Alpha += 2.00 * Z["em"] * V["m,v1,e,u1"] * Gamma1["u1,v1"];
    Alpha += 2.00 * Z["em"] * V["m,V1,e,U1"] * Gamma1["U1,V1"];
    Alpha += 2.00 * Z["EM"] * V["M,V1,E,U1"] * Gamma1["U1,V1"];
    Alpha += 2.00 * Z["EM"] * V["v1,M,u1,E"] * Gamma1["u1,v1"];

    Alpha += Z["mn"] * V["m,v1,n,u1"] * Gamma1["u1,v1"];
    Alpha += Z["mn"] * V["m,V1,n,U1"] * Gamma1["U1,V1"];
    Alpha += Z["MN"] * V["M,V1,N,U1"] * Gamma1["U1,V1"];
    Alpha += Z["MN"] * V["v1,M,u1,N"] * Gamma1["u1,v1"];

    Alpha += Z["uv"] * V["u,v1,v,u1"] * Gamma1["u1,v1"];
    Alpha += Z["uv"] * V["u,V1,v,U1"] * Gamma1["U1,V1"];
    Alpha += Z["UV"] * V["U,V1,V,U1"] * Gamma1["U1,V1"];
    Alpha += Z["UV"] * V["v1,U,u1,V"] * Gamma1["u1,v1"];

    Alpha += Z["ef"] * V["e,v1,f,u1"] * Gamma1["u1,v1"];
    Alpha += Z["ef"] * V["e,V1,f,U1"] * Gamma1["U1,V1"];
    Alpha += Z["EF"] * V["E,V1,F,U1"] * Gamma1["U1,V1"];
    Alpha += Z["EF"] * V["v1,E,u1,F"] * Gamma1["u1,v1"];

    Alpha += 2.00 * Z["un"] * V["u,v1,n,u1"] * Gamma1["u1,v1"];
    Alpha += 2.00 * Z["un"] * V["u,V1,n,U1"] * Gamma1["U1,V1"];
    Alpha += 2.00 * Z["UN"] * V["U,V1,N,U1"] * Gamma1["U1,V1"];
    Alpha += 2.00 * Z["UN"] * V["v1,U,u1,N"] * Gamma1["u1,v1"];

    Alpha -= 2.00 * Z["un"] * H["vn"] * Gamma1["uv"];
    Alpha -= 2.00 * Z["UN"] * H["VN"] * Gamma1["UV"];

    Alpha -= 2.00 * Z["un"] * V["v,m,n,m1"] * Gamma1["uv"] * I["m,m1"];
    Alpha -= 2.00 * Z["un"] * V["v,M,n,M1"] * Gamma1["uv"] * I["M,M1"];
    Alpha -= 2.00 * Z["UN"] * V["V,M,N,M1"] * Gamma1["UV"] * I["M,M1"];
    Alpha -= 2.00 * Z["UN"] * V["m,V,m1,N"] * Gamma1["UV"] * I["m,m1"];

    Alpha -= Z["un"] * V["xynv"] * Gamma2["uvxy"];
    Alpha -= 2.00 * Z["un"] * V["xYnV"] * Gamma2["uVxY"];
    Alpha -= Z["UN"] * V["XYNV"] * Gamma2["UVXY"];
    Alpha -= 2.00 * Z["UN"] * V["yXvN"] * Gamma2["vUyX"];


    Alpha += 2.00 * Z["eu"] * H["ve"] * Gamma1["uv"];
    Alpha += 2.00 * Z["EU"] * H["VE"] * Gamma1["UV"];

    Alpha += 2.00 * Z["eu"] * V["v,m,e,m1"] * Gamma1["uv"] * I["m,m1"];
    Alpha += 2.00 * Z["eu"] * V["v,M,e,M1"] * Gamma1["uv"] * I["M,M1"];
    Alpha += 2.00 * Z["EU"] * V["V,M,E,M1"] * Gamma1["UV"] * I["M,M1"];
    Alpha += 2.00 * Z["EU"] * V["m,V,m1,E"] * Gamma1["UV"] * I["m,m1"];

    Alpha += Z["eu"] * V["xyev"] * Gamma2["uvxy"];
    Alpha += 2.00 * Z["eu"] * V["xYeV"] * Gamma2["uVxY"];
    Alpha += Z["EU"] * V["XYEV"] * Gamma2["UVXY"];
    Alpha += 2.00 * Z["EU"] * V["yXvE"] * Gamma2["vUyX"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::change_w(BlockedTensor& temp1,
        BlockedTensor& temp2, BlockedTensor& W_, const std::string block) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    if (block == "vg") {
        temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
        temp4["cdij"] = temp3["cdkj"] * Gamma1["ki"];
        temp3.zero();
        temp3["cbij"] = temp4["cdij"] * Eta1["bd"];
        temp4.zero();
        temp4["ebij"] = temp3["cbij"] * Eta1["ec"];
        W_["pe"] += 0.25 * temp4["ebij"] * temp2["ijpeb"];
        temp3.zero();
        temp4.zero();

        temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
        temp4["cDiJ"] = temp3["cDkJ"] * Gamma1["ki"];
        temp3.zero();
        temp3["cBiJ"] = temp4["cDiJ"] * Eta1["BD"];
        temp4.zero();
        temp4["eBiJ"] = temp3["cBiJ"] * Eta1["ec"];
        W_["pe"] += 0.50 * temp4["eBiJ"] * temp2["iJpeB"];
    }
    else if (block == "ch") {
        temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
        temp4["cdmj"] = temp3["cdkj"] * Gamma1["km"];
        temp3.zero();
        temp3["cbmj"] = temp4["cdmj"] * Eta1["bd"];
        temp4.zero();
        temp4["abmj"] = temp3["cbmj"] * Eta1["ac"];
        W_["im"] += 0.25 * temp4["abmj"] * temp2["abimj"];
        temp3.zero();
        temp4.zero();

        temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
        temp4["cDmJ"] = temp3["cDkJ"] * Gamma1["km"];
        temp3.zero();
        temp3["cBmJ"] = temp4["cDmJ"] * Eta1["BD"];
        temp4.zero();
        temp4["aBmJ"] = temp3["cBmJ"] * Eta1["ac"];
        W_["im"] += 0.50 * temp4["aBmJ"] * temp2["aBimJ"];
    }
    else if (block == "aa1") {
        temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
        temp4["cdij"] = temp3["cdkj"] * Gamma1["ki"];
        temp3.zero();
        temp3["cbij"] = temp4["cdij"] * Eta1["bd"];
        temp4.zero();
        temp4["wbij"] = temp3["cbij"] * Eta1["wc"];
        W_["zw"] += 0.25 * temp4["wbij"] * temp2["zwbij"];
        temp3.zero();
        temp4.zero();

        temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
        temp4["cDiJ"] = temp3["cDkJ"] * Gamma1["ki"];
        temp3.zero();
        temp3["cBiJ"] = temp4["cDiJ"] * Eta1["BD"];
        temp4.zero();
        temp4["wBiJ"] = temp3["cBiJ"] * Eta1["wc"];
        W_["zw"] += 0.50 * temp4["wBiJ"] * temp2["zwBiJ"];
    }
    else if (block == "aa2") {
        temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
        temp4["cdwj"] = temp3["cdkj"] * Gamma1["kw"];
        temp3.zero();
        temp3["cbwj"] = temp4["cdwj"] * Eta1["bd"];
        temp4.zero();
        temp4["abwj"] = temp3["cbwj"] * Eta1["ac"];
        W_["zw"] += 0.25 * temp4["abwj"] * temp2["abzwj"];
        temp3.zero();
        temp4.zero();

        temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
        temp4["cDwJ"] = temp3["cDkJ"] * Gamma1["kw"];
        temp3.zero();
        temp3["cBwJ"] = temp4["cDwJ"] * Eta1["BD"];
        temp4.zero();
        temp4["aBwJ"] = temp3["cBwJ"] * Eta1["ac"];
        W_["zw"] += 0.50 * temp4["aBwJ"] * temp2["aBzwJ"];
    }

    temp1.zero();
    temp2.zero();
}



void DSRG_MRPT2::set_w() {
    //NOTICE: w for {virtual-general}
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"hhgvp", "hHgvP"});

    if (PT2_TERM) {
        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        temp2["ijpeb"] = V["pbij"] * Eeps2_m1["ijeb"];
        temp2["iJpeB"] = V["pBiJ"] * Eeps2_m1["iJeB"];
        change_w(temp1, temp2, W_, "vg");

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        temp2["ijpeb"] = V["pbij"] * Eeps2_p["ijeb"];
        temp2["iJpeB"] = V["pBiJ"] * Eeps2_p["iJeB"];
        change_w(temp1, temp2, W_, "vg");
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
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"pphch", "pPhcH"});

    if (PT2_TERM) {
        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        temp2["abimj"] = V["abij"] * Eeps2_m1["mjab"];
        temp2["aBimJ"] = V["aBiJ"] * Eeps2_m1["mJaB"];
        change_w(temp1, temp2, W_, "ch");

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        temp2["abimj"] = V["abij"] * Eeps2_p["mjab"];
        temp2["aBimJ"] = V["aBiJ"] * Eeps2_p["mJaB"];
        change_w(temp1, temp2, W_, "ch");
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

    W_["mu"] = W_["um"];

    //NOTICE: w for {active-active}
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"aaphh", "aaPhH"});

    if (PT2_TERM) {
        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        temp2["zwbij"] = V["zbij"] * Eeps2_m1["ijwb"];
        temp2["zwBiJ"] = V["zBiJ"] * Eeps2_m1["iJwB"];
        change_w(temp1, temp2, W_, "aa1");

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        temp2["zwbij"] = V["zbij"] * Eeps2_p["ijwb"];
        temp2["zwBiJ"] = V["zBiJ"] * Eeps2_p["iJwB"];
        change_w(temp1, temp2, W_, "aa1");

        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppaah", "pPaaH"});

        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        temp2["abzwj"] = V["abzj"] * Eeps2_m1["wjab"];
        temp2["aBzwJ"] = V["aBzJ"] * Eeps2_m1["wJaB"];
        change_w(temp1, temp2, W_, "aa2");

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        temp2["abzwj"] = V["abzj"] * Eeps2_p["wjab"];
        temp2["aBzwJ"] = V["aBzJ"] * Eeps2_p["wJaB"];
        change_w(temp1, temp2, W_, "aa2");
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

    auto temp_aa = ambit::Tensor::build(ambit::CoreTensor, "temp_aa", {na_, na_});
    auto cc = coupling_coefficients_;
    auto ci = ci_vectors_[0];
    auto cc1a_ = cc.cc1a();
    auto cc1b_ = cc.cc1b();
    auto cc2aa_ = cc.cc2aa();
    auto cc2bb_ = cc.cc2bb();
    auto cc2ab_ = cc.cc2ab();
    
    temp_aa("zw") += x_ci("I") * H.block("aa")("vz") * cc1a_("IJwv") * ci("J");
    temp_aa("zw") += x_ci("I") * H.block("aa")("zu") * cc1a_("IJuw") * ci("J");


    temp_aa("zw") += 0.25 * x_ci("I") * V.block("aaaa")("zvxy") * cc2aa_("IJwvxy") * ci("J");
    temp_aa("zw") += 0.50 * x_ci("I") * V.block("aAaA")("zVxY") * cc2ab_("IJwVxY") * ci("J");

    temp_aa("zw") += 0.25 * x_ci("I") * V.block("aaaa")("uzxy") * cc2aa_("IJuwxy") * ci("J");
    temp_aa("zw") += 0.50 * x_ci("I") * V.block("aAaA")("zUxY") * cc2ab_("IJwUxY") * ci("J");

    temp_aa("zw") += 0.25 * x_ci("I") * V.block("aaaa")("uvzy") * cc2aa_("IJuvwy") * ci("J");
    temp_aa("zw") += 0.50 * x_ci("I") * V.block("aAaA")("uVzY") * cc2ab_("IJuVwY") * ci("J");

    temp_aa("zw") += 0.25 * x_ci("I") * V.block("aaaa")("uvxz") * cc2aa_("IJuvxw") * ci("J");
    temp_aa("zw") += 0.50 * x_ci("I") * V.block("aAaA")("uVzX") * cc2ab_("IJuVwX") * ci("J");

    // (W_.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
    //     value += temp_aa.data()[i[0] * na_ + i[1]];
    // });

    (temp_aa).iterate([&](const std::vector<size_t>& i, double& value) {
        W_.block("aa").data()[i[0] * na_ + i[1]] += value;
    });




    // CASSCF reference
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"gg"}));

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
}

void DSRG_MRPT2::change_val1(BlockedTensor& temp1, BlockedTensor& temp2, BlockedTensor& val1) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["abml"] = temp1["abmj"] * Gamma1["lj"];
    temp4["adml"] = temp3["abml"] * Eta1["bd"];
    temp3.zero();
    temp3["cdml"] = temp4["adml"] * Eta1["ac"];
    temp4.zero();
    temp4["cdml"] = temp2["cdkl"] * Gamma1["km"];
    val1["m"] += temp3["cdml"] * temp4["cdml"];
    temp3.zero();
    temp4.zero();

    temp3["aBmL"] = temp1["aBmJ"] * Gamma1["LJ"];
    temp4["aDmL"] = temp3["aBmL"] * Eta1["BD"];
    temp3.zero();
    temp3["cDmL"] = temp4["aDmL"] * Eta1["ac"];
    temp4.zero();
    temp4["cDmL"] = temp2["cDkL"] * Gamma1["km"];
    val1["m"] += 2.0 * temp3["cDmL"] * temp4["cDmL"];

    temp1.zero();
    temp2.zero();
}

void DSRG_MRPT2::change_zmn_normal(BlockedTensor& temp1, 
        BlockedTensor& temp2, BlockedTensor& zmn, bool reverse_mn) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["cdnj"] = temp1["cdnl"] * Gamma1["lj"];
    temp4["cbnj"] = temp3["cdnj"] * Eta1["bd"];
    temp3.zero();
    temp3["abnj"] = temp4["cbnj"] * Eta1["ac"];
    if (reverse_mn) { zmn["nm"] -= 0.25 * temp3["abnj"] * temp2["abmnj"];}
    else { zmn["mn"] += 0.25 * temp3["abnj"] * temp2["abmnj"];}  
    temp3.zero();
    temp4.zero();

    temp3["cDnJ"] = temp1["cDnL"] * Gamma1["LJ"];
    temp4["cBnJ"] = temp3["cDnJ"] * Eta1["BD"];
    temp3.zero();
    temp3["aBnJ"] = temp4["cBnJ"] * Eta1["ac"];
    if (reverse_mn) { zmn["nm"] -= 0.50 * temp3["aBnJ"] * temp2["aBmnJ"];}
    else { zmn["mn"] += 0.50 * temp3["aBnJ"] * temp2["aBmnJ"];}
    
    temp1.zero();
    temp2.zero();
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
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"pphh"}));
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", spin_cases({"pphh"}));
    // core-core diagonal entries
    if (PT2_TERM) {
        // Alpha
        temp1["abmj"] = -2.0 * s * V["abmj"] * Delta2["mjab"] * Eeps2["mjab"];
        temp1["aBmJ"] = -2.0 * s * V["aBmJ"] * Delta2["mJaB"] * Eeps2["mJaB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        change_val1(temp1, temp2, val1);

        temp1["abmj"] = 2.0 * s * V["abmj"] * Eeps2["mjab"];
        temp1["aBmJ"] = 2.0 * s * V["aBmJ"] * Eeps2["mJaB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_val1(temp1, temp2, val1);

        temp1["abmj"] = -1.0 * V["abmj"] * Eeps2_m2["mjab"];
        temp1["aBmJ"] = -1.0 * V["aBmJ"] * Eeps2_m2["mJaB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_val1(temp1, temp2, val1);
    }

    BlockedTensor zmn = BTF_->build(CoreTensor, "z{mn} normal", {"cc"});
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppcch", "pPccH"});
    // core-core block entries within normal conditions
    if (PT2_TERM) {
        // Alpha
        temp1["cdnl"] = V["cdnl"] * Eeps2_p["nlcd"];
        temp1["cDnL"] = V["cDnL"] * Eeps2_p["nLcD"];
        temp2["abmnj"] = V["abmj"] * Eeps2_m1["njab"];
        temp2["aBmnJ"] = V["aBmJ"] * Eeps2_m1["nJaB"];
        change_zmn_normal(temp1, temp2, zmn, false);

        temp1["cdnl"] = V["cdnl"] * Eeps2_m1["nlcd"];
        temp1["cDnL"] = V["cDnL"] * Eeps2_m1["nLcD"];
        temp2["abmnj"] = V["abmj"] * Eeps2_p["njab"];
        temp2["aBmnJ"] = V["aBmJ"] * Eeps2_p["nJaB"];
        change_zmn_normal(temp1, temp2, zmn, false);

        temp1["cdml"] = V["cdml"] * Eeps2_p["mlcd"];
        temp1["cDmL"] = V["cDmL"] * Eeps2_p["mLcD"];
        temp2["abnmj"] = V["abnj"] * Eeps2_m1["mjab"];
        temp2["aBnmJ"] = V["aBnJ"] * Eeps2_m1["mJaB"];
        change_zmn_normal(temp1, temp2, zmn, true);

        temp1["cdml"] = V["cdml"] * Eeps2_m1["mlcd"];
        temp1["cDmL"] = V["cDmL"] * Eeps2_m1["mLcD"];
        temp2["abnmj"] = V["abnj"] * Eeps2_p["mjab"];
        temp2["aBnmJ"] = V["aBnJ"] * Eeps2_p["mJaB"];
        change_zmn_normal(temp1, temp2, zmn, true);
    }

    BlockedTensor zmn_d = BTF_->build(CoreTensor, "z{mn} degenerate orbital case", {"cc"});
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppcch", "pPccH"});
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
            if (i[0] == i[1]) { value = 0.5 * val1.block("c").data()[i[0]];}
            else {
                auto dmt = Delta1.block("cc").data()[i[1] * ncore_ + i[0]];
                if (std::fabs(dmt) > 1e-12) { value = zmn.block("cc").data()[i[0] * ncore_ + i[1]] / dmt;}
                else { value = zmn_d.block("cc").data()[i[0] * ncore_ + i[1]];}
            }       
        });
    }  
}


void DSRG_MRPT2::change_val2(BlockedTensor& temp1, BlockedTensor& temp2, BlockedTensor& val2) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["ebil"] = temp1["ebij"] * Gamma1["lj"];
    temp4["ebkl"] = temp3["ebil"] * Gamma1["ki"];
    temp3.zero();
    temp3["edkl"] = temp4["ebkl"] * Eta1["bd"];
    temp4.zero();
    temp4["edkl"] = temp2["cdkl"] * Eta1["ec"];
    val2["e"] += temp3["edkl"] * temp4["edkl"];
    temp3.zero();
    temp4.zero();

    temp3["eBiL"] = temp1["eBiJ"] * Gamma1["LJ"];
    temp4["eBkL"] = temp3["eBiL"] * Gamma1["ki"];
    temp3.zero();
    temp3["eDkL"] = temp4["eBkL"] * Eta1["BD"];
    temp4.zero();
    temp4["eDkL"] = temp2["cDkL"] * Eta1["ec"];
    val2["e"] += 2.0 * temp3["eDkL"] * temp4["eDkL"];

    temp1.zero();
    temp2.zero();
}

void DSRG_MRPT2::change_zef_normal(BlockedTensor& temp1, 
        BlockedTensor& temp2, BlockedTensor& zef, bool reverse_ef) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["fdkj"] = temp1["fdkl"] * Gamma1["lj"];
    temp4["fdij"] = temp3["fdkj"] * Gamma1["ki"];
    temp3.zero();
    temp3["fbij"] = temp4["fdij"] * Eta1["bd"];
    if (reverse_ef) { zef["fe"] -= 0.25 * temp3["fbij"] * temp2["efbij"];}
    else { zef["ef"] += 0.25 * temp3["fbij"] * temp2["efbij"];}  
    temp3.zero();
    temp4.zero();

    temp3["fDkJ"] = temp1["fDkL"] * Gamma1["LJ"];
    temp4["fDiJ"] = temp3["fDkJ"] * Gamma1["ki"];
    temp3.zero();
    temp3["fBiJ"] = temp4["fDiJ"] * Eta1["BD"];
    if (reverse_ef) { zef["fe"] -= 0.50 * temp3["fBiJ"] * temp2["efBiJ"];}
    else { zef["ef"] += 0.50 * temp3["fBiJ"] * temp2["efBiJ"];} 

    temp1.zero();
    temp2.zero();
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
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"pphh"}));
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", spin_cases({"pphh"}));
    // virtual-virtual diagonal entries
    if (PT2_TERM) {
        // Alpha
        temp1["ebij"] = 2.0 * s * V["ebij"] * Delta2["ijeb"] * Eeps2["ijeb"];
        temp1["eBiJ"] = 2.0 * s * V["eBiJ"] * Delta2["iJeB"] * Eeps2["iJeB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        change_val2(temp1, temp2, val2);

        temp1["ebij"] = -2.0 * s * V["ebij"] * Eeps2["ijeb"];
        temp1["eBiJ"] = -2.0 * s * V["eBiJ"] * Eeps2["iJeB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_val2(temp1, temp2, val2);

        temp1["ebij"] = V["ebij"] * Eeps2_m2["ijeb"];
        temp1["eBiJ"] = V["eBiJ"] * Eeps2_m2["iJeB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_val2(temp1, temp2, val2);
    }
   
    BlockedTensor zef = BTF_->build(CoreTensor, "z{ef} normal", {"vv"});
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"vvphh", "vvPhH"});
    // virtual-virtual block entries within normal conditions
    if (PT2_TERM) {
        // Alpha
        temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_m1["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_m1["iJfB"];
        change_zef_normal(temp1, temp2, zef, false);

        temp1["fdkl"] = V["fdkl"] * Eeps2_m1["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_m1["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_p["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_p["iJfB"];
        change_zef_normal(temp1, temp2, zef, false);

        temp1["edkl"] = V["edkl"] * Eeps2_p["kled"];
        temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
        temp2["febij"] = V["fbij"] * Eeps2_m1["ijeb"];
        temp2["feBiJ"] = V["fBiJ"] * Eeps2_m1["iJeB"];
        change_zef_normal(temp1, temp2, zef, true);

        temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"];
        temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
        temp2["febij"] = V["fbij"] * Eeps2_p["ijeb"];
        temp2["feBiJ"] = V["fBiJ"] * Eeps2_p["iJeB"];
        change_zef_normal(temp1, temp2, zef, true);
    }

    BlockedTensor zef_d = BTF_->build(CoreTensor, "z{ef} degenerate orbital case", {"vv"});
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"vvphh", "vvPhH"});
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
            if (i[0] == i[1]) { value = 0.5 * val2.block("v").data()[i[0]];}
            else {
                auto dmt = Delta1.block("vv").data()[i[1] * nvirt_ + i[0]];
                if (std::fabs(dmt) > 1e-12) { value = zef.block("vv").data()[i[0] * nvirt_ + i[1]] / dmt;}
                else { value = zef_d.block("vv").data()[i[0] * nvirt_ + i[1]];}
            }       
        });
    }  
}

void DSRG_MRPT2::change_val3(BlockedTensor& temp1,
        BlockedTensor& temp2, BlockedTensor& val3, bool first_parts) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    if (first_parts) {
        temp3["abul"] = temp1["abuj"] * Gamma1["lj"];
        temp4["adul"] = temp3["abul"] * Eta1["bd"];
        temp3.zero();
        temp3["cdul"] = temp4["adul"] * Eta1["ac"];
        temp4.zero();
        temp4["cdul"] = temp2["cdkl"] * Gamma1["ku"];
        val3["u"] += temp3["cdul"] * temp4["cdul"];
        temp3.zero();
        temp4.zero();

        temp3["aBuL"] = temp1["aBuJ"] * Gamma1["LJ"];
        temp4["aDuL"] = temp3["aBuL"] * Eta1["BD"];
        temp3.zero();
        temp3["cDuL"] = temp4["aDuL"] * Eta1["ac"];
        temp4.zero();
        temp4["cDuL"] = temp2["cDkL"] * Gamma1["ku"];
        val3["u"] += 2.0 * temp3["cDuL"] * temp4["cDuL"];
    }
    else {
        temp3["ubil"] = temp1["ubij"] * Gamma1["lj"];
        temp4["ubkl"] = temp3["ubil"] * Gamma1["ki"];
        temp3.zero();
        temp3["udkl"] = temp4["ubkl"] * Eta1["bd"];
        temp4.zero();
        temp4["udkl"] = temp2["cdkl"] * Eta1["uc"];
        val3["u"] += temp3["udkl"] * temp4["udkl"];
        temp3.zero();
        temp4.zero();

        temp3["uBiL"] = temp1["uBiJ"] * Gamma1["LJ"];
        temp4["uBkL"] = temp3["uBiL"] * Gamma1["ki"];
        temp3.zero();
        temp3["uDkL"] = temp4["uBkL"] * Eta1["BD"];
        temp4.zero();
        temp4["uDkL"] = temp2["cDkL"] * Eta1["uc"];
        val3["u"] += 2.0 * temp3["uDkL"] * temp4["uDkL"];
    }
    temp1.zero();
    temp2.zero();
}

void DSRG_MRPT2::set_z_aa_diag() {
    BlockedTensor val3 = BTF_->build(CoreTensor, "val3", {"a"});
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"pphh"}));
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", spin_cases({"pphh"}));
    // virtual-virtual diagonal entries
    if (PT2_TERM) {
        // Alpha
        temp1["abuj"] = -2.0 * s * V["abuj"] * Delta2["ujab"] * Eeps2["ujab"];
        temp1["aBuJ"] = -2.0 * s * V["aBuJ"] * Delta2["uJaB"] * Eeps2["uJaB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        change_val3(temp1, temp2, val3, true);  

        temp1["abuj"] = 2.0 * s * V["abuj"] * Eeps2["ujab"];
        temp1["aBuJ"] = 2.0 * s * V["aBuJ"] * Eeps2["uJaB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_val3(temp1, temp2, val3, true);

        temp1["abuj"] = -1.0 * V["abuj"] * Eeps2_m2["ujab"];
        temp1["aBuJ"] = -1.0 * V["aBuJ"] * Eeps2_m2["uJaB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_val3(temp1, temp2, val3, true);

        temp1["ubij"] = 2.0 * s * V["ubij"] * Delta2["ijub"] * Eeps2["ijub"];
        temp1["uBiJ"] = 2.0 * s * V["uBiJ"] * Delta2["iJuB"] * Eeps2["iJuB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        change_val3(temp1, temp2, val3, false);

        temp1["ubij"] = -2.0 * s * V["ubij"] * Eeps2["ijub"];
        temp1["uBiJ"] = -2.0 * s * V["uBiJ"] * Eeps2["iJuB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_val3(temp1, temp2, val3, false);

        temp1["ubij"] = V["ubij"] * Eeps2_m2["ijub"];
        temp1["uBiJ"] = V["uBiJ"] * Eeps2_m2["iJuB"];
        temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_val3(temp1, temp2, val3, false);
    }
  
    for (const std::string& block : {"aa", "AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = 0.5 * val3.block("a").data()[i[0]];}
        });
    } 
}


void DSRG_MRPT2::change_b(BlockedTensor& temp1,
        BlockedTensor& temp2, BlockedTensor& Z_b, const std::string block) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    if (block == "cv1") {
        temp3["cdmj"] = temp1["cdml"] * Gamma1["lj"];
        temp4["cbmj"] = temp3["cdmj"] * Eta1["bd"];
        temp3.zero();
        temp3["abmj"] = temp4["cbmj"] * Eta1["ac"];
        Z_b["em"] += 0.25 * temp3["abmj"] * temp2["abemj"];
        temp3.zero();
        temp4.zero();

        temp3["cDmJ"] = temp1["cDmL"] * Gamma1["LJ"];
        temp4["cBmJ"] = temp3["cDmJ"] * Eta1["BD"];
        temp3.zero();
        temp3["aBmJ"] = temp4["cBmJ"] * Eta1["ac"];
        Z_b["em"] += 0.50 * temp3["aBmJ"] * temp2["aBemJ"];
    }
    else if (block == "cv2") {
        temp3["edkj"] = temp1["edkl"] * Gamma1["lj"];
        temp4["edij"] = temp3["edkj"] * Gamma1["ki"];
        temp3.zero();
        temp3["ebij"] = temp4["edij"] * Eta1["bd"];
        Z_b["em"] -= 0.25 * temp3["ebij"] * temp2["mebij"];
        temp3.zero();
        temp4.zero();

        temp3["eDkJ"] = temp1["eDkL"] * Gamma1["LJ"];
        temp4["eDiJ"] = temp3["eDkJ"] * Gamma1["ki"];
        temp3.zero();
        temp3["eBiJ"] = temp4["eDiJ"] * Eta1["BD"];
        Z_b["em"] -= 0.50 * temp3["eBiJ"] * temp2["meBiJ"];
    }
    else if (block == "aa1") { 
        temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
        temp4["cdij"] = temp3["cdkj"] * Gamma1["ki"];
        temp3.zero();
        temp3["cbij"] = temp4["cdij"] * Eta1["bd"];
        temp4.zero();
        temp4["zbij"] = temp3["cbij"] * Eta1["zc"];
        Z_b["wz"] += 0.25 * temp4["zbij"] * temp2["wzbij"];
        temp3.zero();
        temp4.zero();

        temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
        temp4["cDiJ"] = temp3["cDkJ"] * Gamma1["ki"];
        temp3.zero();
        temp3["cBiJ"] = temp4["cDiJ"] * Eta1["BD"];
        temp4.zero();
        temp4["zBiJ"] = temp3["cBiJ"] * Eta1["zc"];
        Z_b["wz"] += 0.50 * temp4["zBiJ"] * temp2["wzBiJ"];
    }
    else if (block == "aa2") { 
        temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
        temp4["cdzj"] = temp3["cdkj"] * Gamma1["kz"];
        temp3.zero();
        temp3["cbzj"] = temp4["cdzj"] * Eta1["bd"];
        temp4.zero();
        temp4["abzj"] = temp3["cbzj"] * Eta1["ac"];
        Z_b["wz"] += 0.25 * temp4["abzj"] * temp2["abwzj"];
        temp3.zero();
        temp4.zero();

        temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
        temp4["cDzJ"] = temp3["cDkJ"] * Gamma1["kz"];
        temp3.zero();
        temp3["cBzJ"] = temp4["cDzJ"] * Eta1["BD"];
        temp4.zero();
        temp4["aBzJ"] = temp3["cBzJ"] * Eta1["ac"];
        Z_b["wz"] += 0.50 * temp4["aBzJ"] * temp2["aBwzJ"];
    }
    else if (block == "aa3") { 
        temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
        temp4["cdij"] = temp3["cdkj"] * Gamma1["ki"];
        temp3.zero();
        temp3["cbij"] = temp4["cdij"] * Eta1["bd"];
        temp4.zero();
        temp4["wbij"] = temp3["cbij"] * Eta1["wc"];
        Z_b["wz"] -= 0.25 * temp4["wbij"] * temp2["zwbij"];
        temp3.zero();
        temp4.zero();

        temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
        temp4["cDiJ"] = temp3["cDkJ"] * Gamma1["ki"];
        temp3.zero();
        temp3["cBiJ"] = temp4["cDiJ"] * Eta1["BD"];
        temp4.zero();
        temp4["wBiJ"] = temp3["cBiJ"] * Eta1["wc"];
        Z_b["wz"] -= 0.50 * temp4["wBiJ"] * temp2["zwBiJ"];
    }
    else if (block == "aa4") { 
        temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
        temp4["cdwj"] = temp3["cdkj"] * Gamma1["kw"];
        temp3.zero();
        temp3["cbwj"] = temp4["cdwj"] * Eta1["bd"];
        temp4.zero();
        temp4["abwj"] = temp3["cbwj"] * Eta1["ac"];
        Z_b["wz"] -= 0.25 * temp4["abwj"] * temp2["abzwj"];
        temp3.zero();
        temp4.zero();

        temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
        temp4["cDwJ"] = temp3["cDkJ"] * Gamma1["kw"];
        temp3.zero();
        temp3["cBwJ"] = temp4["cDwJ"] * Eta1["BD"];
        temp4.zero();
        temp4["aBwJ"] = temp3["cBwJ"] * Eta1["ac"];
        Z_b["wz"] -= 0.50 * temp4["aBwJ"] * temp2["aBzwJ"];
    }
    else if (block == "va1") {
        temp3["udkj"] = temp1["udkl"] * Gamma1["lj"];
        temp4["udij"] = temp3["udkj"] * Gamma1["ki"];
        temp3.zero();
        temp3["ubij"] = temp4["udij"] * Eta1["bd"];
        temp4.zero();
        temp4["wbij"] = temp3["ubij"] * Eta1["wu"];
        Z_b["ew"] += 0.25 * temp4["wbij"] * temp2["ewbij"];
        temp3.zero();
        temp4.zero();

        temp3["uDkJ"] = temp1["uDkL"] * Gamma1["LJ"];
        temp4["uDiJ"] = temp3["uDkJ"] * Gamma1["ki"];
        temp3.zero();
        temp3["uBiJ"] = temp4["uDiJ"] * Eta1["BD"];
        temp4.zero();
        temp4["wBiJ"] = temp3["uBiJ"] * Eta1["wu"];
        Z_b["ew"] += 0.50 * temp4["wBiJ"] * temp2["ewBiJ"];
    }
    else if (block == "va2") {
        temp3["cduj"] = temp1["cdul"] * Gamma1["lj"];
        temp4["cdwj"] = temp3["cduj"] * Gamma1["uw"];
        temp3.zero();
        temp3["cbwj"] = temp4["cdwj"] * Eta1["bd"];
        temp4.zero();
        temp4["abwj"] = temp3["cbwj"] * Eta1["ac"];
        Z_b["ew"] += 0.25 * temp4["abwj"] * temp2["abewj"];
        temp3.zero();
        temp4.zero();

        temp3["cDuJ"] = temp1["cDuL"] * Gamma1["LJ"];
        temp4["cDwJ"] = temp3["cDuJ"] * Gamma1["uw"];
        temp3.zero();
        temp3["cBwJ"] = temp4["cDwJ"] * Eta1["BD"];
        temp4.zero();
        temp4["aBwJ"] = temp3["cBwJ"] * Eta1["ac"];
        Z_b["ew"] += 0.50 * temp4["aBwJ"] * temp2["aBewJ"];        
    }
    else if (block == "va3") {
        temp3["edkj"] = temp1["edkl"] * Gamma1["lj"];
        temp4["edij"] = temp3["edkj"] * Gamma1["ki"];
        temp3.zero();
        temp3["ebij"] = temp4["edij"] * Eta1["bd"];
        Z_b["ew"] -= 0.25 * temp3["ebij"] * temp2["webij"];
        temp3.zero();
        temp4.zero();

        temp3["eDkJ"] = temp1["eDkL"] * Gamma1["LJ"];
        temp4["eDiJ"] = temp3["eDkJ"] * Gamma1["ki"];
        temp3.zero();
        temp3["eBiJ"] = temp4["eDiJ"] * Eta1["BD"];
        Z_b["ew"] -= 0.50 * temp3["eBiJ"] * temp2["weBiJ"];
    }
    else if (block == "ca1") {
        temp3["udkj"] = temp1["udkl"] * Gamma1["lj"];
        temp4["udij"] = temp3["udkj"] * Gamma1["ki"];
        temp3.zero();
        temp3["ubij"] = temp4["udij"] * Eta1["bd"];
        temp4.zero();
        temp4["wbij"] = temp3["ubij"] * Eta1["wu"];
        Z_b["mw"] += 0.25 * temp4["wbij"] * temp2["mwbij"];
        temp3.zero();
        temp4.zero();

        temp3["uDkJ"] = temp1["uDkL"] * Gamma1["LJ"];
        temp4["uDiJ"] = temp3["uDkJ"] * Gamma1["ki"];
        temp3.zero();
        temp3["uBiJ"] = temp4["uDiJ"] * Eta1["BD"];
        temp4.zero();
        temp4["wBiJ"] = temp3["uBiJ"] * Eta1["wu"];
        Z_b["mw"] += 0.50 * temp4["wBiJ"] * temp2["mwBiJ"];    
    }
    else if (block == "ca2") {
        temp3["cduj"] = temp1["cdul"] * Gamma1["lj"];
        temp4["cdwj"] = temp3["cduj"] * Gamma1["uw"];
        temp3.zero();
        temp3["cbwj"] = temp4["cdwj"] * Eta1["bd"];
        temp4.zero();
        temp4["abwj"] = temp3["cbwj"] * Eta1["ac"];
        Z_b["mw"] += 0.25 * temp4["abwj"] * temp2["abmwj"];
        temp3.zero();
        temp4.zero();

        temp3["cDuJ"] = temp1["cDuL"] * Gamma1["LJ"];
        temp4["cDwJ"] = temp3["cDuJ"] * Gamma1["uw"];
        temp3.zero();
        temp3["cBwJ"] = temp4["cDwJ"] * Eta1["BD"];
        temp4.zero();
        temp4["aBwJ"] = temp3["cBwJ"] * Eta1["ac"];
        Z_b["mw"] += 0.50 * temp4["aBwJ"] * temp2["aBmwJ"];
    }
    else if (block == "ca3") {
        temp3["cdmj"] = temp1["cdml"] * Gamma1["lj"];
        temp4["cbmj"] = temp3["cdmj"] * Eta1["bd"];
        temp3.zero();
        temp3["abmj"] = temp4["cbmj"] * Eta1["ac"];
        Z_b["mw"] -= 0.25 * temp3["abmj"] * temp2["abwmj"];
        temp3.zero();
        temp4.zero();

        temp3["cDmJ"] = temp1["cDmL"] * Gamma1["LJ"];
        temp4["cBmJ"] = temp3["cDmJ"] * Eta1["BD"];
        temp3.zero();
        temp3["aBmJ"] = temp4["cBmJ"] * Eta1["ac"];
        Z_b["mw"] -= 0.50 * temp3["aBmJ"] * temp2["aBwmJ"];
    }
    temp1.zero();
    temp2.zero();
}


void DSRG_MRPT2::set_b() {
    outfile->Printf("\n    Initializing b of the Linear System ............. ");
    //NOTICE: constant b for z{core-virtual}
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppvch", "pPvcH"});

    if (PT2_TERM) {
        temp1["cdml"] = V["cdml"] * Eeps2_p["mlcd"];
        temp1["cDmL"] = V["cDmL"] * Eeps2_p["mLcD"];
        temp2["abemj"] = V["abej"] * Eeps2_m1["mjab"];
        temp2["aBemJ"] = V["aBeJ"] * Eeps2_m1["mJaB"];
        change_b(temp1, temp2, Z_b, "cv1");

        temp1["cdml"] = V["cdml"] * Eeps2_m1["mlcd"];
        temp1["cDmL"] = V["cDmL"] * Eeps2_m1["mLcD"];
        temp2["abemj"] = V["abej"] * Eeps2_p["mjab"];
        temp2["aBemJ"] = V["aBeJ"] * Eeps2_p["mJaB"];
        change_b(temp1, temp2, Z_b, "cv1");

        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"cvphh", "cvPhH"});

        temp1["edkl"] = V["edkl"] * Eeps2_p["kled"];
        temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
        temp2["mebij"] = V["mbij"] * Eeps2_m1["ijeb"];
        temp2["meBiJ"] = V["mBiJ"] * Eeps2_m1["iJeB"];
        change_b(temp1, temp2, Z_b, "cv2");

        temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"];
        temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
        temp2["mebij"] = V["mbij"] * Eeps2_p["ijeb"];
        temp2["meBiJ"] = V["mBiJ"] * Eeps2_p["iJeB"];
        change_b(temp1, temp2, Z_b, "cv2");
    }

    Z_b["em"] += Z["m1,n1"] * V["n1,e,m1,m"];
    Z_b["em"] += Z["M1,N1"] * V["e,N1,m,M1"];

    Z_b["em"] += Z["e1,f"] * V["f,e,e1,m"];
    Z_b["em"] += Z["E1,F"] * V["e,F,m,E1"];

    //NOTICE: constant b for z{active-active}
    if (PT2_TERM) {
        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"aaphh", "aaPhH"});

        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        temp2["wzbij"] = V["wbij"] * Eeps2_m1["ijzb"];
        temp2["wzBiJ"] = V["wBiJ"] * Eeps2_m1["iJzB"];
        change_b(temp1, temp2, Z_b, "aa1");

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        temp2["wzbij"] = V["wbij"] * Eeps2_p["ijzb"];
        temp2["wzBiJ"] = V["wBiJ"] * Eeps2_p["iJzB"];
        change_b(temp1, temp2, Z_b, "aa1"); 

        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppaah", "pPaaH"});

        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        temp2["abwzj"] = V["abwj"] * Eeps2_m1["zjab"];
        temp2["aBwzJ"] = V["aBwJ"] * Eeps2_m1["zJaB"];
        change_b(temp1, temp2, Z_b, "aa2"); 

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        temp2["abwzj"] = V["abwj"] * Eeps2_p["zjab"];
        temp2["aBwzJ"] = V["aBwJ"] * Eeps2_p["zJaB"];
        change_b(temp1, temp2, Z_b, "aa2");
          
        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"aaphh", "aaPhH"});

        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        temp2["zwbij"] = V["zbij"] * Eeps2_m1["ijwb"];
        temp2["zwBiJ"] = V["zBiJ"] * Eeps2_m1["iJwB"];
        change_b(temp1, temp2, Z_b, "aa3");

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        temp2["zwbij"] = V["zbij"] * Eeps2_p["ijwb"];
        temp2["zwBiJ"] = V["zBiJ"] * Eeps2_p["iJwB"];
        change_b(temp1, temp2, Z_b, "aa3");

        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppaah", "pPaaH"});

        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        temp2["abzwj"] = V["abzj"] * Eeps2_m1["wjab"];
        temp2["aBzwJ"] = V["aBzJ"] * Eeps2_m1["wJaB"];
        change_b(temp1, temp2, Z_b, "aa4");

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        temp2["abzwj"] = V["abzj"] * Eeps2_p["wjab"];
        temp2["aBzwJ"] = V["aBzJ"] * Eeps2_p["wJaB"];
        change_b(temp1, temp2, Z_b, "aa4");
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
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"aphh", "aPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"vaphh", "vaPhH"});

    if (PT2_TERM) {
        temp1["udkl"] = V["udkl"] * Eeps2_p["klud"];
        temp1["uDkL"] = V["uDkL"] * Eeps2_p["kLuD"];
        temp2["ewbij"] = V["ebij"] * Eeps2_m1["ijwb"];
        temp2["ewBiJ"] = V["eBiJ"] * Eeps2_m1["iJwB"];
        change_b(temp1, temp2, Z_b, "va1");

        temp1["udkl"] = V["udkl"] * Eeps2_m1["klud"];
        temp1["uDkL"] = V["uDkL"] * Eeps2_m1["kLuD"];
        temp2["ewbij"] = V["ebij"] * Eeps2_p["ijwb"];
        temp2["ewBiJ"] = V["eBiJ"] * Eeps2_p["iJwB"];
        change_b(temp1, temp2, Z_b, "va1"); 

        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppah", "pPaH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppvah", "pPvaH"}); 

        temp1["cdul"] = V["cdul"] * Eeps2_p["ulcd"];
        temp1["cDuL"] = V["cDuL"] * Eeps2_p["uLcD"];
        temp2["abewj"] = V["abej"] * Eeps2_m1["wjab"];
        temp2["aBewJ"] = V["aBeJ"] * Eeps2_m1["wJaB"];
        change_b(temp1, temp2, Z_b, "va2"); 

        temp1["cdul"] = V["cdul"] * Eeps2_m1["ulcd"];
        temp1["cDuL"] = V["cDuL"] * Eeps2_m1["uLcD"];
        temp2["abewj"] = V["abej"] * Eeps2_p["wjab"];
        temp2["aBewJ"] = V["aBeJ"] * Eeps2_p["wJaB"];
        change_b(temp1, temp2, Z_b, "va2");

        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"avphh", "avPhH"});

        temp1["edkl"] = V["edkl"] * Eeps2_p["kled"]; 
        temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
        temp2["webij"] = V["wbij"] * Eeps2_m1["ijeb"]; 
        temp2["weBiJ"] = V["wBiJ"] * Eeps2_m1["iJeB"]; 
        change_b(temp1, temp2, Z_b, "va3"); 

        temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"]; 
        temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
        temp2["webij"] = V["wbij"] * Eeps2_p["ijeb"]; 
        temp2["weBiJ"] = V["wBiJ"] * Eeps2_p["iJeB"];    
        change_b(temp1, temp2, Z_b, "va3");
    }

    Z_b["ew"] -= Z["e,f1"] * F["f1,w"];
    Z_b["ew"] += Z["m1,n1"] * V["n1,v,m1,e"] * Gamma1["wv"];
    Z_b["ew"] += Z["M1,N1"] * V["v,N1,e,M1"] * Gamma1["wv"];
    Z_b["ew"] += Z["e1,f1"] * V["f1,v,e1,e"] * Gamma1["wv"];
    Z_b["ew"] += Z["E1,F1"] * V["v,F1,e,E1"] * Gamma1["wv"];

    //NOTICE: constant b for z{core-active}

    if (PT2_TERM) {
        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"aphh", "aPhH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"caphh", "caPhH"});

        temp1["udkl"] = V["udkl"] * Eeps2_p["klud"];
        temp1["uDkL"] = V["uDkL"] * Eeps2_p["kLuD"];
        temp2["mwbij"] = V["mbij"] * Eeps2_m1["ijwb"];
        temp2["mwBiJ"] = V["mBiJ"] * Eeps2_m1["iJwB"];
        change_b(temp1, temp2, Z_b, "ca1");

        temp1["udkl"] = V["udkl"] * Eeps2_m1["klud"];
        temp1["uDkL"] = V["uDkL"] * Eeps2_m1["kLuD"];
        temp2["mwbij"] = V["mbij"] * Eeps2_p["ijwb"];
        temp2["mwBiJ"] = V["mBiJ"] * Eeps2_p["iJwB"];
        change_b(temp1, temp2, Z_b, "ca1");

        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppah", "pPaH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppcah", "pPcaH"});

        temp1["cdul"] = V["cdul"] * Eeps2_p["ulcd"];
        temp1["cDuL"] = V["cDuL"] * Eeps2_p["uLcD"];
        temp2["abmwj"] = V["abmj"] * Eeps2_m1["wjab"];
        temp2["aBmwJ"] = V["aBmJ"] * Eeps2_m1["wJaB"];
        change_b(temp1, temp2, Z_b, "ca2");

        temp1["cdul"] = V["cdul"] * Eeps2_m1["ulcd"];
        temp1["cDuL"] = V["cDuL"] * Eeps2_m1["uLcD"];
        temp2["abmwj"] = V["abmj"] * Eeps2_p["wjab"];
        temp2["aBmwJ"] = V["aBmJ"] * Eeps2_p["wJaB"];
        change_b(temp1, temp2, Z_b, "ca2");

        temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
        temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppach", "pPacH"});

        temp1["cdml"] = V["cdml"] * Eeps2_p["mlcd"];
        temp1["cDmL"] = V["cDmL"] * Eeps2_p["mLcD"];
        temp2["abwmj"] = V["abwj"] * Eeps2_m1["mjab"];
        temp2["aBwmJ"] = V["aBwJ"] * Eeps2_m1["mJaB"];
        change_b(temp1, temp2, Z_b, "ca3");

        temp1["cdml"] = V["cdml"] * Eeps2_m1["mlcd"];
        temp1["cDmL"] = V["cDmL"] * Eeps2_m1["mLcD"];
        temp2["abwmj"] = V["abwj"] * Eeps2_p["mjab"];
        temp2["aBwmJ"] = V["aBwJ"] * Eeps2_p["mJaB"];
        change_b(temp1, temp2, Z_b, "ca3");
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


void DSRG_MRPT2::iter_z() {
    bool converged = false;
    int iter = 1;
    int maxiter = 4000;
    double convergence = 1e-8;
    BlockedTensor Zold = BTF_->build(CoreTensor, "Old Z Matrix", spin_cases({"gg"}));

    set_b();
    outfile->Printf("Done");

    while (iter <= maxiter) {
        Zold["pq"] = Z["pq"];
        compute_z_cv();
        compute_z_aa();
        compute_z_av();
        compute_z_ca();
        Zold["pq"] -= Z["pq"];

        double Znorm = Zold.norm(0);
        if (Znorm < convergence) {
            converged = true;
            break;
        }
        iter++;
        if (iter%5==0) {outfile->Printf("\niter=%d    Znorm                     =  %.8f",iter, Znorm);}
    }
    outfile->Printf("\n    iterations                     =  %d", iter);
}


void DSRG_MRPT2::compute_z_cv() {
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"vc"});

    temp["em"] = Z_b["em"];

    temp["em"] += Z["mu"] * F["ue"];

    temp["em"] += Z["n1,u"] * V["u,e,n1,m"];
    temp["em"] += Z["N1,U"] * V["e,U,m,N1"];
    temp["em"] += Z["n1,u"] * V["u,m,n1,e"];
    temp["em"] += Z["N1,U"] * V["m,U,e,N1"];

    temp["em"] -= Z["n1,u"] * Gamma1["uv"] * V["v,e,n1,m"];
    temp["em"] -= Z["N1,U"] * Gamma1["UV"] * V["e,V,m,N1"];
    temp["em"] -= Z["n1,u"] * Gamma1["uv"] * V["v,m,n1,e"];
    temp["em"] -= Z["N1,U"] * Gamma1["UV"] * V["m,V,e,N1"];

    temp["em"] += Z["e1,u"] * Gamma1["uv"] * V["v,m,e1,e"];
    temp["em"] += Z["E1,U"] * Gamma1["UV"] * V["m,V,e,E1"];
    temp["em"] += Z["e1,u"] * Gamma1["uv"] * V["v,e,e1,m"];
    temp["em"] += Z["E1,U"] * Gamma1["UV"] * V["e,V,m,E1"]; 

    temp["em"] += Z["uv"] * V["veum"];
    temp["em"] += Z["UV"] * V["eVmU"];

    temp["em"] -= Z["eu"] * F["um"];

    temp["em"] += Z["e1,m1"] * V["m1,m,e1,e"];
    temp["em"] += Z["E1,M1"] * V["m,M1,e,E1"];  
    temp["em"] += Z["e1,m1"] * V["m1,e,e1,m"];
    temp["em"] += Z["E1,M1"] * V["e,M1,m,E1"]; 

    // Denominator
    BlockedTensor dnt = BTF_->build(CoreTensor, "temporal denominator", {"vc"});
    dnt["em"] -= V["m1,m,e1,e"] * I["m1,m"] * I["e1,e"];
    dnt["em"] -= V["m1,e,e1,m"] * I["m1,m"] * I["e1,e"];

    // Move z{em} terms to the other side of the equation
    temp["em"] += Z["em"] * dnt["em"];

    // Denominator
    dnt["em"] += Delta1["me"];

    BlockedTensor temp_p = BTF_->build(CoreTensor, "temporal tensor plus Z_vc", {"vc"});
    BlockedTensor temp_m = BTF_->build(CoreTensor, "temporal tensor subtract Z_vc", {"vc"});

    temp_p["em"] = temp["em"];
    temp_p["em"] += Z["em"];
    temp_m["em"] = temp["em"];
    temp_m["em"] -= Z["em"];

    for (const std::string& block : {"vc"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            double dn = dnt.block(block).data()[i[0] * ncore_ + i[1]];
            if (std::fabs(dn) > 1.0) {
                value = temp.block(block).data()[i[0] * ncore_ + i[1]] / dn;
            }
            else if (dn >= 0) {
                value = temp_p.block(block).data()[i[0] * ncore_ + i[1]] / (dn + 1.0);
            }
            else {
                value = temp_m.block(block).data()[i[0] * ncore_ + i[1]] / (dn - 1.0);
            }
        });
    } 

    Z["me"] = Z["em"];

    // Beta part
    for (const std::string& block : {"VC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("vc").data()[i[0] * ncore_ + i[1]];
        });
    } 
    Z["ME"] = Z["EM"];
}


void DSRG_MRPT2::compute_z_av() {
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"va"});

    temp["ew"] = Z_b["ew"];

    temp["ew"] -= Z["e,m1"] * F["m1,w"];

    temp["ew"] += Z["e1,m1"] * V["m1,e,e1,u"] * Gamma1["uw"];
    temp["ew"] += Z["E1,M1"] * V["e,M1,u,E1"] * Gamma1["uw"];
    temp["ew"] += Z["e1,m1"] * V["m1,u,e1,e"] * Gamma1["uw"];
    temp["ew"] += Z["E1,M1"] * V["u,M1,e,E1"] * Gamma1["uw"];

    temp["ew"] += Z["n1,u"] * V["u,v,n1,e"] * Gamma1["wv"];
    temp["ew"] += Z["N1,U"] * V["v,U,e,N1"] * Gamma1["wv"];
    temp["ew"] += Z["n1,u"] * V["u,e,n1,v"] * Gamma1["wv"];
    temp["ew"] += Z["N1,U"] * V["e,U,v,N1"] * Gamma1["wv"];

    temp["ew"] -= Z["n1,u"] * H["e,n1"] * Gamma1["uw"];
    temp["ew"] -= Z["n1,u"] * V["e,m,n1,m1"] * Gamma1["uw"] * I["m,m1"];
    temp["ew"] -= Z["n1,u"] * V["e,M,n1,M1"] * Gamma1["uw"] * I["M,M1"];
    temp["ew"] -= 0.5 * Z["n1,u"] * V["x,y,n1,e"] * Gamma2["u,w,x,y"];
    temp["ew"] -= Z["N1,U"] * V["y,X,e,N1"] * Gamma2["w,U,y,X"];
    temp["ew"] -= Z["n1,u"] * V["e,y,n1,v"] * Gamma2["u,v,w,y"];
    temp["ew"] -= Z["n1,u"] * V["e,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp["ew"] -= Z["N1,U"] * V["e,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp["ew"] += Z["u1,v1"] * V["v1,v,u1,e"] * Gamma1["wv"];
    temp["ew"] += Z["U1,V1"] * V["v,V1,e,U1"] * Gamma1["wv"];

    temp["ew"] += Z["wv"] * F["ve"];

    // Terms need to be separated
    temp["ew"] += Z["e1,u"] * H["e,e1"] * Gamma1["uw"];
    temp["ew"] += Z["e1,u"] * V["e,m,e1,m1"] * Gamma1["uw"] * I["m,m1"];
    temp["ew"] += Z["e1,u"] * V["e,M,e1,M1"] * Gamma1["uw"] * I["M,M1"];
    temp["ew"] += 0.5 * Z["e1,u"] * V["x,y,e1,e"] * Gamma2["u,w,x,y"];
    temp["ew"] += Z["E1,U"] * V["y,X,e,E1"] * Gamma2["w,U,y,X"];
    temp["ew"] += Z["e1,u"] * V["e,y,e1,v"] * Gamma2["u,v,w,y"];
    temp["ew"] += Z["e1,u"] * V["e,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp["ew"] += Z["E1,U"] * V["e,Y,v,E1"] * Gamma2["v,U,w,Y"];

    temp["ew"] -= Z["eu"] * H["vw"] * Gamma1["uv"];
    temp["ew"] -= Z["eu"] * V["v,m,w,m1"] * Gamma1["uv"] * I["m,m1"];
    temp["ew"] -= Z["eu"] * V["v,M,w,M1"] * Gamma1["uv"] * I["M,M1"];
    temp["ew"] -= 0.5 * Z["eu"] * V["xywv"] * Gamma2["uvxy"];
    temp["ew"] -= Z["eu"] * V["xYwV"] * Gamma2["uVxY"];

    // move terms to the left side
    temp["ew"] -= Z["ew"] * H["e,e1"] * Gamma1["w,w1"] * I["e,e1"] * I["w,w1"];
    temp["ew"] -= Z["ew"] * V["e,m,e1,m1"] * Gamma1["w,w1"] * I["e,e1"] * I["w,w1"] * I["m,m1"];
    temp["ew"] -= Z["ew"] * V["e,M,e1,M1"] * Gamma1["w,w1"] * I["e,e1"] * I["w,w1"] * I["M,M1"];
    temp["ew"] -= 0.5 * Z["ew"] * V["x,y,e,e1"] * Gamma2["w,w1,x,y"] * I["e,e1"] * I["w,w1"];
    temp["ew"] -= Z["ew"] * V["e,y,e1,v"] * Gamma2["w,v,w1,y"] * I["e,e1"] * I["w,w1"];
    temp["ew"] -= Z["ew"] * V["e,Y,e1,V"] * Gamma2["w,V,w1,Y"] * I["e,e1"] * I["w,w1"];

    temp["ew"] += Z["ew"] * H["vw"] * Gamma1["wv"];
    temp["ew"] += Z["ew"] * V["v,m,w,m1"] * Gamma1["wv"] * I["m,m1"];
    temp["ew"] += Z["ew"] * V["v,M,w,M1"] * Gamma1["wv"] * I["M,M1"];
    temp["ew"] += 0.5 * Z["ew"] * V["xywv"] * Gamma2["wvxy"];
    temp["ew"] += Z["ew"] * V["xYwV"] * Gamma2["wVxY"];

    // Denominator
    BlockedTensor dnt = BTF_->build(CoreTensor, "temporal denominator", {"va"});
    dnt["ew"] -= H["e,e1"] * Gamma1["w,w1"] * I["e,e1"] * I["w,w1"];
    dnt["ew"] -= V["e,m,e1,m1"] * Gamma1["w,w1"] * I["e,e1"] * I["w,w1"] * I["m,m1"];
    dnt["ew"] -= V["e,M,e1,M1"] * Gamma1["w,w1"] * I["e,e1"] * I["w,w1"] * I["M,M1"];
    dnt["ew"] -= 0.5 * V["x,y,e,e1"] * Gamma2["w,w1,x,y"] * I["e,e1"] * I["w,w1"];
    dnt["ew"] -= V["e,y,e1,v"] * Gamma2["w,v,w1,y"] * I["e,e1"] * I["w,w1"];
    dnt["ew"] -= V["e,Y,e1,V"] * Gamma2["w,V,w1,Y"] * I["e,e1"] * I["w,w1"];

    BlockedTensor dnt1 = BTF_->build(CoreTensor, "temporal denominator", {"a"});
    dnt1["w"] += H["vw"] * Gamma1["wv"];
    dnt1["w"] += V["v,m,w,m1"] * Gamma1["wv"] * I["m,m1"];
    dnt1["w"] += V["v,M,w,M1"] * Gamma1["wv"] * I["M,M1"];
    dnt1["w"] += 0.5 * V["xywv"] * Gamma2["wvxy"];
    dnt1["w"] += V["xYwV"] * Gamma2["wVxY"];

    BlockedTensor temp_p = BTF_->build(CoreTensor, "temporal tensor plus Z_va", {"va"});
    BlockedTensor temp_m = BTF_->build(CoreTensor, "temporal tensor subtract Z_va", {"va"});

    temp_p["ew"] = temp["ew"];
    temp_p["ew"] += 15 * Z["ew"];
    temp_m["ew"] = temp["ew"];
    temp_m["ew"] -= 15 * Z["ew"];

    for (const std::string& block : {"va"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            double dn = dnt.block(block).data()[i[0] * na_ + i[1]] + dnt1.block("a").data()[i[1]];
            if (std::fabs(dn) > 15.0) {
                value = temp.block(block).data()[i[0] * na_ + i[1]] / dn;
            }
            else if (dn >= 0) {
                value = temp_p.block(block).data()[i[0] * na_ + i[1]] / (dn + 15.0);
            }
            else {
                value = temp_m.block(block).data()[i[0] * na_ + i[1]] / (dn - 15.0);
            }
        });
    } 

    Z["we"] = Z["ew"];

    // Beta part
    for (const std::string& block : {"VA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("va").data()[i[0] * na_ + i[1]];
        });
    } 
    Z["WE"] = Z["EW"];
}




void DSRG_MRPT2::compute_z_ca() {
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"ca"});

    temp["mw"] = Z_b["mw"];

    temp["mw"] += Z["e1,m1"] * V["m1,m,e1,u"] * Gamma1["uw"];
    temp["mw"] += Z["E1,M1"] * V["m,M1,u,E1"] * Gamma1["uw"];
    temp["mw"] += Z["e1,m1"] * V["m1,u,e1,m"] * Gamma1["uw"];
    temp["mw"] += Z["E1,M1"] * V["u,M1,m,E1"] * Gamma1["uw"];

    temp["mw"] += Z["e1,u"] * H["m,e1"] * Gamma1["uw"];
    temp["mw"] += Z["e1,u"] * V["m,m1,e1,n1"] * Gamma1["uw"] * I["m1,n1"];
    temp["mw"] += Z["e1,u"] * V["m,M1,e1,N1"] * Gamma1["uw"] * I["M1,N1"];
    temp["mw"] += 0.5 * Z["e1,u"] * V["x,y,e1,m"] * Gamma2["u,w,x,y"];
    temp["mw"] += Z["E1,U"] * V["y,X,m,E1"] * Gamma2["w,U,y,X"];
    temp["mw"] += Z["e1,u"] * V["m,y,e1,v"] * Gamma2["u,v,w,y"];
    temp["mw"] += Z["e1,u"] * V["m,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp["mw"] += Z["E1,U"] * V["m,Y,v,E1"] * Gamma2["v,U,w,Y"];

    temp["mw"] += Z["u1,v1"] * V["v1,v,u1,m"] * Gamma1["wv"];
    temp["mw"] += Z["U1,V1"] * V["v,V1,m,U1"] * Gamma1["wv"];

    temp["mw"] += Z["wv"] * F["vm"];

    temp["mw"] -= Z["e1,m"] * F["w,e1"];

    temp["mw"] -= Z["e1,m1"] * V["m1,m,e1,w"];
    temp["mw"] -= Z["E1,M1"] * V["m,M1,w,E1"];
    temp["mw"] -= Z["e1,m1"] * V["m1,w,e1,m"];
    temp["mw"] -= Z["E1,M1"] * V["w,M1,m,E1"];

    temp["mw"] -= Z["e1,u"] * Gamma1["uv"] * V["v,m,e1,w"];
    temp["mw"] -= Z["E1,U"] * Gamma1["UV"] * V["m,V,w,E1"];
    temp["mw"] -= Z["e1,u"] * Gamma1["uv"] * V["v,w,e1,m"];
    temp["mw"] -= Z["E1,U"] * Gamma1["UV"] * V["w,V,m,E1"];

    temp["mw"] -= Z["uv"] * V["vwum"];
    temp["mw"] -= Z["UV"] * V["wVmU"];

    // Terms need to be separated
    temp["mw"] += Z["n1,u"] * V["u,v,n1,m"] * Gamma1["wv"];
    temp["mw"] += Z["N1,U"] * V["v,U,m,N1"] * Gamma1["wv"];
    temp["mw"] += Z["n1,u"] * V["u,m,n1,v"] * Gamma1["wv"];
    temp["mw"] += Z["N1,U"] * V["m,U,v,N1"] * Gamma1["wv"];

    temp["mw"] -= Z["n1,u"] * H["m,n1"] * Gamma1["uw"];
    temp["mw"] -= Z["n1,u"] * V["m,m1,n1,n"] * Gamma1["uw"] * I["m1,n"];
    temp["mw"] -= Z["n1,u"] * V["m,M1,n1,N"] * Gamma1["uw"] * I["M1,N"];
    temp["mw"] -= 0.5 * Z["n1,u"] * V["x,y,n1,m"] * Gamma2["u,w,x,y"];
    temp["mw"] -= Z["N1,U"] * V["y,X,m,N1"] * Gamma2["w,U,y,X"];
    temp["mw"] -= Z["n1,u"] * V["m,y,n1,v"] * Gamma2["u,v,w,y"];
    temp["mw"] -= Z["n1,u"] * V["m,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp["mw"] -= Z["N1,U"] * V["m,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp["mw"] -= Z["n1,u"] * V["u,w,n1,m"];
    temp["mw"] -= Z["N1,U"] * V["w,U,m,N1"];
    temp["mw"] -= Z["n1,u"] * V["u,m,n1,w"];
    temp["mw"] -= Z["N1,U"] * V["m,U,w,N1"];

    temp["mw"] += Z["n1,u"] * Gamma1["uv"] * V["v,w,n1,m"];
    temp["mw"] += Z["N1,U"] * Gamma1["UV"] * V["w,V,m,N1"];
    temp["mw"] += Z["n1,u"] * Gamma1["uv"] * V["v,m,n1,w"];
    temp["mw"] += Z["N1,U"] * Gamma1["UV"] * V["m,V,w,N1"];

    temp["mw"] += Z["mu"] * H["vw"] * Gamma1["uv"];
    temp["mw"] += Z["mu"] * V["v,m1,w,n1"] * Gamma1["uv"] * I["m1,n1"];
    temp["mw"] += Z["mu"] * V["v,M1,w,N1"] * Gamma1["uv"] * I["M1,N1"];
    temp["mw"] += 0.5 * Z["mu"] * V["xywv"] * Gamma2["uvxy"];
    temp["mw"] += Z["mu"] * V["xYwV"] * Gamma2["uVxY"];

    // move terms to the left side
    // TODO: need check
    temp["mw"] -= Z["mw"] * H["vw"] * Gamma1["wv"];
    temp["mw"] -= Z["mw"] * V["v,m1,w,n1"] * Gamma1["wv"] * I["m1,n1"];
    temp["mw"] -= Z["mw"] * V["v,M1,w,N1"] * Gamma1["wv"] * I["M1,N1"];
    temp["mw"] -= 0.5 * Z["mw"] * V["xywv"] * Gamma2["wvxy"];
    temp["mw"] -= Z["mw"] * V["xYwV"] * Gamma2["wVxY"];

    temp["mw"] += Z["mw"] * V["w,w1,m,m1"] * I["w,w1"] * I["m,m1"];
    temp["mw"] += Z["mw"] * V["w,m,m1,w1"] * I["w,w1"] * I["m,m1"];

    temp["mw"] -= 2.0 * Z["mw"] * Gamma1["wv"] * V["v,w,m,m1"] * I["m,m1"];
    temp["mw"] -= 2.0 * Z["mw"] * Gamma1["wv"] * V["v,m,m1,w"] * I["m,m1"];

    temp["mw"] += Z["mw"] * H["m,m1"] * Gamma1["w,w1"] * I["m,m1"] * I["w,w1"];
    temp["mw"] += Z["mw"] * V["m,n,m1,n1"] * Gamma1["w,w1"] * I["m,m1"] * I["n,n1"] * I["w,w1"];
    temp["mw"] += Z["mw"] * V["m,N,m1,N1"] * Gamma1["w,w1"] * I["m,m1"] * I["N,N1"] * I["w,w1"];
    temp["mw"] += 0.5 * Z["mw"] * V["x,y,m,m1"] * Gamma2["w,w1,x,y"] * I["m,m1"] * I["w,w1"];
    temp["mw"] += Z["mw"] * V["m,y,m1,v"] * Gamma2["w,v,w1,y"] * I["m,m1"] * I["w,w1"];
    temp["mw"] += Z["mw"] * V["m,Y,m1,V"] * Gamma2["w,V,w1,Y"] * I["m,m1"] * I["w,w1"];

    // Denominator
    BlockedTensor dnt = BTF_->build(CoreTensor, "temporal denominator", {"ca"});
    dnt["mw"] += V["w,w1,m,m1"] * I["w,w1"] * I["m,m1"];
    dnt["mw"] += V["w,m,m1,w1"] * I["w,w1"] * I["m,m1"];
    dnt["mw"] -= 2.0 * Gamma1["wv"] * V["v,w,m,m1"] * I["m,m1"];
    dnt["mw"] -= 2.0 * Gamma1["wv"] * V["v,m,m1,w"] * I["m,m1"];
    dnt["mw"] += H["m,m1"] * Gamma1["w,w1"] * I["m,m1"] * I["w,w1"];
    dnt["mw"] += V["m,n,m1,n1"] * Gamma1["w,w1"] * I["m,m1"] * I["n,n1"] * I["w,w1"];
    dnt["mw"] += V["m,N,m1,N1"] * Gamma1["w,w1"] * I["m,m1"] * I["N,N1"] * I["w,w1"];
    dnt["mw"] += 0.5 * V["x,y,m,m1"] * Gamma2["w,w1,x,y"] * I["m,m1"] * I["w,w1"];
    dnt["mw"] += V["m,y,m1,v"] * Gamma2["w,v,w1,y"] * I["m,m1"] * I["w,w1"];
    dnt["mw"] += V["m,Y,m1,V"] * Gamma2["w,V,w1,Y"] * I["m,m1"] * I["w,w1"];

    dnt["mw"] += Delta1["wm"];

    BlockedTensor dnt1 = BTF_->build(CoreTensor, "temporal denominator", {"a"});

    dnt1["w"] -= H["vw"] * Gamma1["wv"];
    dnt1["w"] -= V["v,m1,w,n1"] * Gamma1["wv"] * I["m1,n1"];
    dnt1["w"] -= V["v,M1,w,N1"] * Gamma1["wv"] * I["M1,N1"];
    dnt1["w"] -= 0.5 * V["xywv"] * Gamma2["wvxy"];
    dnt1["w"] -= V["xYwV"] * Gamma2["wVxY"];

    BlockedTensor temp_p = BTF_->build(CoreTensor, "temporal tensor plus Z_ca", {"ca"});
    BlockedTensor temp_m = BTF_->build(CoreTensor, "temporal tensor subtract Z_ca", {"ca"});

    temp_p["mw"] = temp["mw"];
    temp_p["mw"] += Z["mw"];
    temp_m["mw"] = temp["mw"];
    temp_m["mw"] -= Z["mw"];

    for (const std::string& block : {"ca"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            double dn = dnt.block(block).data()[i[0] * na_ + i[1]] + dnt1.block("a").data()[i[1]];
            if (fabs(dn) > 1.0) {
                value = temp.block(block).data()[i[0] * na_ + i[1]] / dn;
            }
            else if (dn >= 0) {
                value = temp_p.block(block).data()[i[0] * na_ + i[1]] / (dn + 1.0);
            }
            else {
                value = temp_m.block(block).data()[i[0] * na_ + i[1]] / (dn - 1.0);
            }
        });
    } 

    Z["wm"] = Z["mw"];

    // Beta part
    for (const std::string& block : {"CA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("ca").data()[i[0] * na_ + i[1]];
        });
    } 
    Z["WM"] = Z["MW"];
}


void DSRG_MRPT2::compute_z_aa() {
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"aa"});

    temp["wz"] = Z_b["wz"];

    temp["wz"] += Z["e1,m1"] * V["m1,w,e1,u"] * Gamma1["uz"];
    temp["wz"] += Z["E1,M1"] * V["w,M1,u,E1"] * Gamma1["uz"];
    temp["wz"] += Z["e1,m1"] * V["m1,u,e1,w"] * Gamma1["uz"];
    temp["wz"] += Z["E1,M1"] * V["u,M1,w,E1"] * Gamma1["uz"];
    temp["wz"] -= Z["e1,m1"] * V["m1,z,e1,u"] * Gamma1["uw"];
    temp["wz"] -= Z["E1,M1"] * V["z,M1,u,E1"] * Gamma1["uw"];
    temp["wz"] -= Z["e1,m1"] * V["m1,u,e1,z"] * Gamma1["uw"];
    temp["wz"] -= Z["E1,M1"] * V["u,M1,z,E1"] * Gamma1["uw"];

    temp["wz"] += Z["n1,z"] * F["w,n1"];
    temp["wz"] -= Z["n1,w"] * F["z,n1"];

    temp["wz"] += Z["n1,u"] * V["u,v,n1,w"] * Gamma1["zv"];
    temp["wz"] += Z["N1,U"] * V["v,U,w,N1"] * Gamma1["zv"];
    temp["wz"] += Z["n1,u"] * V["u,w,n1,v"] * Gamma1["zv"];
    temp["wz"] += Z["N1,U"] * V["w,U,v,N1"] * Gamma1["zv"];
    temp["wz"] -= Z["n1,u"] * V["u,v,n1,z"] * Gamma1["wv"];
    temp["wz"] -= Z["N1,U"] * V["v,U,z,N1"] * Gamma1["wv"];
    temp["wz"] -= Z["n1,u"] * V["u,z,n1,v"] * Gamma1["wv"];
    temp["wz"] -= Z["N1,U"] * V["z,U,v,N1"] * Gamma1["wv"];

    temp["wz"] -= Z["n1,u"] * H["w,n1"] * Gamma1["uz"];
    temp["wz"] -= Z["n1,u"] * V["w,m1,n1,m"] * Gamma1["uz"] * I["m1,m"];
    temp["wz"] -= Z["n1,u"] * V["w,M1,n1,M"] * Gamma1["uz"] * I["M1,M"];
    temp["wz"] -= 0.5 * Z["n1,u"] * V["x,y,n1,w"] * Gamma2["u,z,x,y"];
    temp["wz"] -= Z["N1,U"] * V["y,X,w,N1"] * Gamma2["z,U,y,X"];
    temp["wz"] -= Z["n1,u"] * V["w,y,n1,v"] * Gamma2["u,v,z,y"];
    temp["wz"] -= Z["n1,u"] * V["w,Y,n1,V"] * Gamma2["u,V,z,Y"];
    temp["wz"] -= Z["N1,U"] * V["w,Y,v,N1"] * Gamma2["v,U,z,Y"];

    temp["wz"] += Z["e1,u"] * H["w,e1"] * Gamma1["uz"];
    temp["wz"] += Z["e1,u"] * V["w,m1,e1,m"] * Gamma1["uz"] * I["m1,m"];
    temp["wz"] += Z["e1,u"] * V["w,M1,e1,M"] * Gamma1["uz"] * I["M1,M"];
    temp["wz"] += 0.5 * Z["e1,u"] * V["x,y,e1,w"] * Gamma2["u,z,x,y"];
    temp["wz"] += Z["E1,U"] * V["y,X,w,E1"] * Gamma2["z,U,y,X"];
    temp["wz"] += Z["e1,u"] * V["w,y,e1,v"] * Gamma2["u,v,z,y"];
    temp["wz"] += Z["e1,u"] * V["w,Y,e1,V"] * Gamma2["u,V,z,Y"];
    temp["wz"] += Z["E1,U"] * V["w,Y,v,E1"] * Gamma2["v,U,z,Y"];

    temp["wz"] -= Z["e1,u"] * H["z,e1"] * Gamma1["uw"];
    temp["wz"] -= Z["e1,u"] * V["z,m1,e1,m"] * Gamma1["uw"] * I["m1,m"];
    temp["wz"] -= Z["e1,u"] * V["z,M1,e1,M"] * Gamma1["uw"] * I["M1,M"];
    temp["wz"] -= 0.5 * Z["e1,u"] * V["x,y,e1,z"] * Gamma2["u,w,x,y"];
    temp["wz"] -= Z["E1,U"] * V["y,X,z,E1"] * Gamma2["w,U,y,X"];
    temp["wz"] -= Z["e1,u"] * V["z,y,e1,v"] * Gamma2["u,v,w,y"];
    temp["wz"] -= Z["e1,u"] * V["z,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp["wz"] -= Z["E1,U"] * V["z,Y,v,E1"] * Gamma2["v,U,w,Y"];

    temp["wz"] += Z["n1,u"] * H["z,n1"] * Gamma1["uw"];
    temp["wz"] += Z["n1,u"] * V["z,m1,n1,m"] * Gamma1["uw"] * I["m1,m"];
    temp["wz"] += Z["n1,u"] * V["z,M1,n1,M"] * Gamma1["uw"] * I["M1,M"];
    temp["wz"] += 0.5 * Z["n1,u"] * V["x,y,n1,z"] * Gamma2["u,w,x,y"];
    temp["wz"] += Z["N1,U"] * V["y,X,z,N1"] * Gamma2["w,U,y,X"];
    temp["wz"] += Z["n1,u"] * V["z,y,n1,v"] * Gamma2["u,v,w,y"];
    temp["wz"] += Z["n1,u"] * V["z,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp["wz"] += Z["N1,U"] * V["z,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp["wz"] += Z["u1,v1"] * V["v1,v,u1,w"] * Gamma1["zv"];
    temp["wz"] += Z["U1,V1"] * V["v,V1,w,U1"] * Gamma1["zv"];

    temp["wz"] -= Z["u1,v1"] * V["v1,v,u1,z"] * Gamma1["wv"];
    temp["wz"] -= Z["U1,V1"] * V["v,V1,z,U1"] * Gamma1["wv"];

    // move terms to the left side
    temp["wz"] -= Z["zw"] * V["w,v,z,u1"] * I["w,u1"] * Gamma1["zv"];
    temp["wz"] -= Z["zw"] * V["z,v,w,u1"] * I["w,u1"] * Gamma1["zv"];
    temp["wz"] += Z["zw"] * V["w,v,z,u1"] * I["z,u1"] * Gamma1["wv"];
    temp["wz"] += Z["zw"] * V["z,v,w,u1"] * I["z,u1"] * Gamma1["wv"];

    // Denominator
    BlockedTensor dnt = BTF_->build(CoreTensor, "temporal denominator", {"aa"});
    dnt["wz"] = Delta1["zw"];
    dnt["wz"] -= V["w,v,z,u1"] * I["w,u1"] * Gamma1["zv"];
    dnt["wz"] -= V["z,v,w,u1"] * I["w,u1"] * Gamma1["zv"];
    dnt["wz"] += V["w,v,z,u1"] * I["z,u1"] * Gamma1["wv"];
    dnt["wz"] += V["z,v,w,u1"] * I["z,u1"] * Gamma1["wv"];

    BlockedTensor temp_p = BTF_->build(CoreTensor, "temporal tensor plus Z_aa", {"aa"});
    BlockedTensor temp_m = BTF_->build(CoreTensor, "temporal tensor subtract Z_aa", {"aa"});

    temp_p["wz"] = temp["wz"];
    temp_p["wz"] += Z["wz"];
    temp_m["wz"] = temp["wz"];
    temp_m["wz"] -= Z["wz"];

    // different conditions to guarantee the convergence 
    // NOTICE: can be further accelerated
    for (const std::string& block : {"aa"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] != i[1]) {
                double dn = dnt.block(block).data()[i[0] * na_ + i[1]];
                if (fabs(dn) > 1.0) {
                    value = temp.block(block).data()[i[0] * na_ + i[1]] / dn;
                }
                else if (dn >= 0) {
                    value = temp_p.block(block).data()[i[0] * na_ + i[1]] / (dn + 1.0);
                }
                else {
                    value = temp_m.block(block).data()[i[0] * na_ + i[1]] / (dn - 1.0);
                }
              
            }
        });
    }     

    // Beta part
    for (const std::string& block : {"AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            // if (i[0] != i[1]) {
                value = Z.block("aa").data()[i[0] * na_ + i[1]];
            // }
        });
    } 
}


void DSRG_MRPT2::solve_z() {
    set_b();

    size_t ndets = ci_vectors_[0].dims()[0];
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


    auto ci_vc = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{vc}", {ndets, nvirt_, ncore_});
    auto ci_ca = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{ca}", {ndets, ncore_, na_});
    auto ci_va = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{va}", {ndets, nvirt_, na_});
    auto ci_aa = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{aa}", {ndets, na_, na_});


    // CI contribution to Z{VC}
    ci_vc("Iem") += 2 * H.block("vc")("em") * ci("I");
    ci_vc("Iem") += 2 * V_N_Alpha.block("cv")("me") * ci("I");
    ci_vc("Iem") += 2 * V_N_Beta.block("cv")("me") * ci("I");
    ci_vc("Iem") += 2 * V.block("acav")("umve") * cc1a_("IJuv") * ci("J");
    ci_vc("Iem") += 2 * V.block("cAvA")("mUeV") * cc1b_("IJUV") * ci("J");


    // CI contribution to Z{CA}
    ci_ca("Imw") += H.block("ac")("vm") * cc1a_("IJwv") * ci("J");
    ci_ca("Imw") += H.block("ca")("mu") * cc1a_("IJuw") * ci("J");

    ci_ca("Imw") += V_N_Alpha.block("ac")("um") * cc1a_("IJuw") * ci("J");
    ci_ca("Imw") += V_N_Beta.block("ac")("um") * cc1a_("IJuw") * ci("J");

    ci_ca("Imw") += V_N_Alpha.block("ca")("mv") * cc1a_("IJwv") * ci("J"); 
    ci_ca("Imw") += V_N_Beta.block("ca")("mv") * cc1a_("IJwv") * ci("J"); 

    ci_ca("Imw") += 0.25 * V.block("caaa")("mvxy") * cc2aa_("IJwvxy") * ci("J");
    ci_ca("Imw") += 0.50 * V.block("cAaA")("mVxY") * cc2ab_("IJwVxY") * ci("J");

    ci_ca("Imw") += 0.25 * V.block("acaa")("umxy") * cc2aa_("IJuwxy") * ci("J");
    ci_ca("Imw") += 0.50 * V.block("cAaA")("mUxY") * cc2ab_("IJwUxY") * ci("J");

    ci_ca("Imw") += 0.25 * V.block("aaca")("uvmy") * cc2aa_("IJuvwy") * ci("J");
    ci_ca("Imw") += 0.50 * V.block("aAcA")("uVmY") * cc2ab_("IJuVwY") * ci("J");

    ci_ca("Imw") += 0.25 * V.block("aaac")("uvxm") * cc2aa_("IJuvxw") * ci("J");
    ci_ca("Imw") += 0.50 * V.block("aAcA")("uVmX") * cc2ab_("IJuVwX") * ci("J");

    ci_ca("Imw") -= 2 * H.block("ac")("wm") * ci("I");
    ci_ca("Imw") -= 2 * V_N_Alpha.block("ca")("mw") * ci("I");
    ci_ca("Imw") -= 2 * V_N_Beta.block("ca")("mw") * ci("I");
    ci_ca("Imw") -= 2 * V.block("acaa")("umvw") * cc1a_("IJuv") * ci("J");
    ci_ca("Imw") -= 2 * V.block("cAaA")("mUwV") * cc1b_("IJUV") * ci("J");

    // CI contribution to Z{VA}
    ci_va("Iew") += H.block("av")("ve") * cc1a_("IJwv") * ci("J");
    ci_va("Iew") += H.block("va")("eu") * cc1a_("IJuw") * ci("J");

    ci_va("Iew") += V_N_Alpha.block("av")("ue") * cc1a_("IJuw") * ci("J");
    ci_va("Iew") += V_N_Beta.block("av")("ue") * cc1a_("IJuw") * ci("J");
    ci_va("Iew") += V_N_Alpha.block("va")("ev") * cc1a_("IJwv") * ci("J");
    ci_va("Iew") += V_N_Beta.block("va")("ev") * cc1a_("IJwv") * ci("J");

    ci_va("Iew") += 0.25 * V.block("vaaa")("evxy") * cc2aa_("IJwvxy") * ci("J");
    ci_va("Iew") += 0.50 * V.block("vAaA")("eVxY") * cc2ab_("IJwVxY") * ci("J");

    ci_va("Iew") += 0.25 * V.block("avaa")("uexy") * cc2aa_("IJuwxy") * ci("J");
    ci_va("Iew") += 0.50 * V.block("vAaA")("eUxY") * cc2ab_("IJwUxY") * ci("J");

    ci_va("Iew") += 0.25 * V.block("aava")("uvey") * cc2aa_("IJuvwy") * ci("J");
    ci_va("Iew") += 0.50 * V.block("aAvA")("uVeY") * cc2ab_("IJuVwY") * ci("J");

    ci_va("Iew") += 0.25 * V.block("aaav")("uvxe") * cc2aa_("IJuvxw") * ci("J");
    ci_va("Iew") += 0.50 * V.block("aAvA")("uVeX") * cc2ab_("IJuVwX") * ci("J");

    // CI contribution to Z{AA}
    ci_aa("Iwz") += H.block("aa")("vw") * cc1a_("IJzv") * ci("J");
    ci_aa("Iwz") += H.block("aa")("wu") * cc1a_("IJuz") * ci("J");
    ci_aa("Iwz") -= H.block("aa")("vz") * cc1a_("IJwv") * ci("J");
    ci_aa("Iwz") -= H.block("aa")("zu") * cc1a_("IJuw") * ci("J");

    ci_aa("Iwz") += V_N_Alpha.block("aa")("uw") * cc1a_("IJuz") * ci("J");
    ci_aa("Iwz") += V_N_Beta.block("aa")("uw") * cc1a_("IJuz") * ci("J");
    ci_aa("Iwz") += V_N_Alpha.block("aa")("wv") * cc1a_("IJzv") * ci("J");
    ci_aa("Iwz") += V_N_Beta.block("aa")("wv") * cc1a_("IJzv") * ci("J");

    ci_aa("Iwz") -= V_N_Alpha.block("aa")("uz") * cc1a_("IJuw") * ci("J");
    ci_aa("Iwz") -= V_N_Beta.block("aa")("uz") * cc1a_("IJuw") * ci("J");
    ci_aa("Iwz") -= V_N_Alpha.block("aa")("zv") * cc1a_("IJwv") * ci("J");
    ci_aa("Iwz") -= V_N_Beta.block("aa")("zv") * cc1a_("IJwv") * ci("J");

    ci_aa("Iwz") += 0.25 * V.block("aaaa")("wvxy") * cc2aa_("IJzvxy") * ci("J");
    ci_aa("Iwz") += 0.50 * V.block("aAaA")("wVxY") * cc2ab_("IJzVxY") * ci("J");

    ci_aa("Iwz") += 0.25 * V.block("aaaa")("uwxy") * cc2aa_("IJuzxy") * ci("J");
    ci_aa("Iwz") += 0.50 * V.block("aAaA")("wUxY") * cc2ab_("IJzUxY") * ci("J");

    ci_aa("Iwz") += 0.25 * V.block("aaaa")("uvwy") * cc2aa_("IJuvzy") * ci("J");
    ci_aa("Iwz") += 0.50 * V.block("aAaA")("uVwY") * cc2ab_("IJuVzY") * ci("J");

    ci_aa("Iwz") += 0.25 * V.block("aaaa")("uvxw") * cc2aa_("IJuvxz") * ci("J");
    ci_aa("Iwz") += 0.50 * V.block("aAaA")("uVwX") * cc2ab_("IJuVzX") * ci("J");

    ci_aa("Iwz") -= 0.25 * V.block("aaaa")("zvxy") * cc2aa_("IJwvxy") * ci("J");
    ci_aa("Iwz") -= 0.50 * V.block("aAaA")("zVxY") * cc2ab_("IJwVxY") * ci("J");

    ci_aa("Iwz") -= 0.25 * V.block("aaaa")("uzxy") * cc2aa_("IJuwxy") * ci("J");
    ci_aa("Iwz") -= 0.50 * V.block("aAaA")("zUxY") * cc2ab_("IJwUxY") * ci("J");

    ci_aa("Iwz") -= 0.25 * V.block("aaaa")("uvzy") * cc2aa_("IJuvwy") * ci("J");
    ci_aa("Iwz") -= 0.50 * V.block("aAaA")("uVzY") * cc2ab_("IJuVwY") * ci("J");

    ci_aa("Iwz") -= 0.25 * V.block("aaaa")("uvxz") * cc2aa_("IJuvxw") * ci("J");
    ci_aa("Iwz") -= 0.50 * V.block("aAaA")("uVzX") * cc2ab_("IJuVwX") * ci("J");


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

    b_ck("K") += 0.5 * T2_["mjab"] * V_["cdml"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"] * ci("K");
    b_ck("K") += 0.5 * T2_["MJAB"] * V_["CDML"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"] * ci("K");
    b_ck("K") += 2.0 * T2_["mJaB"] * V_["cDmL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"] * ci("K");

    b_ck("K") += 0.5 * T2_["imab"] * V_["cdkm"] * Gamma1["ki"] * Eta1["ac"] * Eta1["bd"] * ci("K");
    b_ck("K") += 0.5 * T2_["IMAB"] * V_["CDKM"] * Gamma1["KI"] * Eta1["AC"] * Eta1["BD"] * ci("K");
    b_ck("K") += 2.0 * T2_["iMaB"] * V_["cDkM"] * Gamma1["ki"] * Eta1["ac"] * Eta1["BD"] * ci("K");






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
    Alpha += 0.5 * T2_["vJaB"] * V_["cDuL"] * Gamma1["uv"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

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

    Alpha += 0.25 * T2_["mjab"] * V_["cdml"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Alpha += 0.25 * T2_["MJAB"] * V_["CDML"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    Alpha += T2_["mJaB"] * V_["cDmL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    Alpha += 0.25 * T2_["imab"] * V_["cdkm"] * Gamma1["ki"] * Eta1["ac"] * Eta1["bd"];
    Alpha += 0.25 * T2_["IMAB"] * V_["CDKM"] * Gamma1["KI"] * Eta1["AC"] * Eta1["BD"];
    Alpha += T2_["iMaB"] * V_["cDkM"] * Gamma1["ki"] * Eta1["ac"] * Eta1["BD"];



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
    x_ci = ambit::Tensor::build(ambit::CoreTensor, "solution for ci multipliers", {ndets});

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

    ck_ci("KI") += ints_->frozen_core_energy() * one_ci("KI");
    ck_ci("KI") += ints_->nuclear_repulsion_energy() * one_ci("KI");
    ck_ci.print();





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

    Et += ints_->nuclear_repulsion_energy();
    std::cout<< "E test = " << std::setprecision(9) << Et << std::endl;

    std::cout<< "ndet = " << ndets << std::endl;


    ck_ci("KI") -= Et * I_ci("KI");

    ck_ci.print();




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
        outfile->Printf( " %.10f ", b[pre + i]);
        outfile->Printf( "\n" );
    }




    outfile->Printf("Done");
    outfile->Printf("\n    Solving Off-diagonal Blocks of Z ................ ");

    C_DGESV( n, nrhs, &A[0], lda, &ipiv[0], &b[0], ldb);

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
        outfile->Printf( " %.10f ", b[pre + i]);
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


    scale_ci = 1.0;
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







//NOTICE Only for test use, need to delete when done
void DSRG_MRPT2::compute_test_energy() {

    double casscf_energy = ints_->nuclear_repulsion_energy();

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"gg"}));
    BlockedTensor I = BTF_->build(CoreTensor, "identity matrix", spin_cases({"gg"}));
    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });

    casscf_energy += H["m,n"] * I["m,n"];
    casscf_energy += H["M,N"] * I["M,N"];


    casscf_energy += 0.5 * V["m,n,m1,n1"] * I["m,m1"] * I["n,n1"];
    casscf_energy += 0.5 * V["M,N,M1,N1"] * I["M,M1"] * I["N,N1"];
    casscf_energy +=       V["m,N,m1,N1"] * I["m,m1"] * I["N,N1"];

    temp["uv"]  = H["uv"];
    temp["uv"] += V["umvn"] * I["mn"];
    temp["uv"] += V["uMvN"] * I["MN"];

    casscf_energy += temp["uv"] * Gamma1["vu"];

    temp["UV"]  = H["UV"];
    temp["UV"] += V["mUnV"] * I["mn"];
    temp["UV"] += V["UMVN"] * I["MN"];

    casscf_energy += temp["UV"] * Gamma1["VU"];

    casscf_energy += 0.25 * V["uvxy"] * Gamma2["xyuv"];
    casscf_energy += 0.25 * V["UVXY"] * Gamma2["XYUV"];
    casscf_energy +=        V["uVxY"] * Gamma2["xYuV"];

    outfile->Printf("\n    E0 (reference)                 =  %6.12lf", casscf_energy);

    double x1_energy = 0.0;

    /************ NOTICE My Test ************/
    BlockedTensor f_t = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"gg"}));
    temp.zero();
   
    temp["xu"] = Gamma1_["xu"] * Delta1["xu"];
    temp["XU"] = Gamma1_["XU"] * Delta1["XU"];
    
    f_t["ia"]  = F["ia"]; 
    f_t["ia"] += F["ia"] * Eeps1["ia"];
    f_t["ia"] += T2_["iuax"] * temp["xu"] * Eeps1["ia"];
    f_t["ia"] += T2_["iUaX"] * temp["XU"] * Eeps1["ia"];
    f_t["IA"]  = F["IA"]; 
    f_t["IA"] += F["IA"] * Eeps1["IA"];
    f_t["IA"] += T2_["uIxA"] * temp["xu"] * Eeps1["IA"];
    f_t["IA"] += T2_["IUAX"] * temp["XU"] * Eeps1["IA"];

    x1_energy += f_t["me"] * T1_["me"];
    x1_energy += f_t["my"] * T1_["mx"] * Eta1["xy"];
    x1_energy += f_t["ye"] * T1_["xe"] * Gamma1["yx"];

    x1_energy += f_t["ME"] * T1_["ME"];
    x1_energy += f_t["MY"] * T1_["MX"] * Eta1["XY"];
    x1_energy += f_t["YE"] * T1_["XE"] * Gamma1["YX"];
    /************ NOTICE My Test ************/


    /************ NOTICE Using York's code ************/
    // x1_energy += F_["em"] * T1_["me"];
    // x1_energy += F_["ym"] * T1_["mx"] * Eta1["xy"];
    // x1_energy += F_["ey"] * T1_["xe"] * Gamma1["yx"];
    // x1_energy += F_["EM"] * T1_["ME"];
    // x1_energy += F_["YM"] * T1_["MX"] * Eta1["XY"];
    // x1_energy += F_["EY"] * T1_["XE"] * Gamma1["YX"];
    /************ NOTICE Using York's code ************/

    outfile->Printf("\n    <[F, T1]>                      =  %6.12lf", x1_energy);

    double x2_energy = 0.0;

    x2_energy += 0.5 * F_["ex"] * T2_["uvey"] * Lambda2_["xyuv"];
    x2_energy +=       F_["ex"] * T2_["uVeY"] * Lambda2_["xYuV"];
    x2_energy += 0.5 * F_["EX"] * T2_["UVEY"] * Lambda2_["XYUV"];
    x2_energy +=       F_["EX"] * T2_["vUyE"] * Lambda2_["yXvU"];

    x2_energy -= 0.5 * F_["vm"] * T2_["umxy"] * Lambda2_["xyuv"];
    x2_energy -=       F_["vm"] * T2_["mUyX"] * Lambda2_["yXvU"];
    x2_energy -= 0.5 * F_["VM"] * T2_["UMXY"] * Lambda2_["XYUV"];
    x2_energy -=       F_["VM"] * T2_["uMxY"] * Lambda2_["xYuV"];

    outfile->Printf("\n    <[F, T2]>                      =  %6.12lf", x2_energy);

    /************ NOTICE The MP2-like term ************/
    double E = 0.0;

    E += 0.25 * V_["efmn"] * T2_["mnef"];
    E += 0.25 * V_["EFMN"] * T2_["MNEF"];
    E += V_["eFmN"] * T2_["mNeF"];

    temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"aa"}));
    temp["vu"] += 0.5 * V_["efmu"] * T2_["mvef"];
    temp["vu"] += V_["fEuM"] * T2_["vMfE"];
    temp["VU"] += 0.5 * V_["EFMU"] * T2_["MVEF"];
    temp["VU"] += V_["eFmU"] * T2_["mVeF"];
    E += temp["vu"] * Gamma1["uv"];
    E += temp["VU"] * Gamma1["UV"];

    temp.zero();
    temp["vu"] += 0.5 * V_["vemn"] * T2_["mnue"];
    temp["vu"] += V_["vEmN"] * T2_["mNuE"];
    temp["VU"] += 0.5 * V_["VEMN"] * T2_["MNUE"];
    temp["VU"] += V_["eVnM"] * T2_["nMeU"];
    E += temp["vu"] * Eta1["uv"];
    E += temp["VU"] * Eta1["UV"];


    E += 0.25 * V_["efxu"] * T2_["yvef"] * Gamma1["xy"] * Gamma1["uv"];
    E += 0.25 * V_["EFXU"] * T2_["YVEF"] * Gamma1["XY"] * Gamma1["UV"];
    E += V_["eFxU"] * T2_["yVeF"] * Gamma1["xy"] * Gamma1["UV"];

    E += 0.25 * V_["xumn"] * T2_["mnyv"] * Eta1["vu"] * Eta1["yx"];
    E += 0.25 * V_["XUMN"] * T2_["MNYV"] * Eta1["VU"] * Eta1["YX"];
    E += V_["xUmN"] * T2_["mNyV"] * Eta1["VU"] * Eta1["yx"];


    temp = BTF_->build(CoreTensor, "temp", spin_cases({"aaaa"}));
    
    temp["yvxu"] += V_["evmx"] * T2_["myeu"];
    temp["yvxu"] += V_["vExM"] * T2_["yMuE"];
    E += temp["yvxu"] * Gamma1["xy"] * Eta1["uv"];

    temp["YVXU"] += V_["EVMX"] * T2_["MYEU"];
    temp["YVXU"] += V_["eVmX"] * T2_["mYeU"];
    E += temp["YVXU"] * Gamma1["XY"] * Eta1["UV"];
   
    E += V_["eVxM"] * T2_["yMeU"] * Gamma1["xy"] * Eta1["UV"];
    E += V_["vEmX"] * T2_["mYuE"] * Gamma1["XY"] * Eta1["uv"];

    // York's code
    temp.zero();
    temp["yvxu"] += 0.5 * Gamma1["wz"] * V_["vexw"] * T2_["yzue"];
    temp["yvxu"] += Gamma1["WZ"] * V_["vExW"] * T2_["yZuE"];
    temp["yvxu"] += 0.5 * Eta1["wz"] * T2_["myuw"] * V_["vzmx"];
    temp["yvxu"] += Eta1["WZ"] * T2_["yMuW"] * V_["vZxM"];
    E += temp["yvxu"] * Gamma1["xy"] * Eta1["uv"];

    temp["YVXU"] += 0.5 * Gamma1["WZ"] * V_["VEXW"] * T2_["YZUE"];
    temp["YVXU"] += Gamma1["wz"] * V_["eVwX"] * T2_["zYeU"];
    temp["YVXU"] += 0.5 * Eta1["WZ"] * T2_["MYUW"] * V_["VZMX"];
    temp["YVXU"] += Eta1["wz"] * V_["zVmX"] * T2_["mYwU"];
    E += temp["YVXU"] * Gamma1["XY"] * Eta1["UV"];


    outfile->Printf("\n    MP2-like term                  =  %6.12lf", E);

    //NOTICE we are not saving the memory at this time
    double Em = 0.0;
    Em += 0.25 * V_["cdkl"] * T2_["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] *Eta1["bd"];
    Em += 0.25 * V_["CDKL"] * T2_["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] *Eta1["BD"];
    Em += V_["cDkL"] * T2_["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    outfile->Printf("\n    MP2-like term more memory      =  %6.12lf", Em);

    // Test for Eps2 (PASSED)
    // temp = BTF_->build(CoreTensor, "temp", spin_cases({"pphh"}));
    // temp["cdkl"] = V["cdkl"];
    // temp["cdkl"] += V["cdkl"] * Eeps2["klcd"];
    // temp["CDKL"] = V["CDKL"];
    // temp["CDKL"] += V["CDKL"] * Eeps2["KLCD"];
    // temp["cDkL"] = V["cDkL"];
    // temp["cDkL"] += V["cDkL"] * Eeps2["kLcD"];  
    // double Eg = 0.0;
    // Eg += 0.25 * temp["cdkl"] * T2_["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] *Eta1["bd"];
    // Eg += 0.25 * temp["CDKL"] * T2_["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] *Eta1["BD"];
    // Eg += temp["cDkL"] * T2_["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    // outfile->Printf("\n    Energy tested.                 =  %6.12lf", Eg);

    temp = BTF_->build(CoreTensor, "temp", spin_cases({"pphh"}));
    temp["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];  
    double E_lag = 0.0;
    E_lag += 0.25 * temp["cdkl"] * T2_["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] *Eta1["bd"];
    E_lag += 0.25 * temp["CDKL"] * T2_["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] *Eta1["BD"];
    E_lag += temp["cDkL"] * T2_["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    outfile->Printf("\n    Energy in Lagrangian           =  %6.12lf", E_lag);
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

void DSRG_MRPT2::change_2rdm(BlockedTensor& temp1,
        BlockedTensor& temp2, BlockedTensor& temp) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["cdkj"] = temp1["cdkl"] * Gamma1["lj"];
    temp4["cdij"] = temp3["cdkj"] * Gamma1["ki"];
    temp3.zero();
    temp3["cbij"] = temp4["cdij"] * Eta1["bd"];
    temp4.zero();
    temp4["abij"] = temp3["cbij"] * Eta1["ac"];
    temp["abij"] += 0.25 * temp4["abij"] * temp2["ijab"];
    temp3.zero();
    temp4.zero();

    temp3["CDKJ"] = temp1["CDKL"] * Gamma1["LJ"];
    temp4["CDIJ"] = temp3["CDKJ"] * Gamma1["KI"];
    temp3.zero();
    temp3["CBIJ"] = temp4["CDIJ"] * Eta1["BD"];
    temp4.zero();
    temp4["ABIJ"] = temp3["CBIJ"] * Eta1["AC"];
    temp["ABIJ"] += 0.25 * temp4["ABIJ"] * temp2["IJAB"];
    temp3.zero();
    temp4.zero();

    temp3["cDkJ"] = temp1["cDkL"] * Gamma1["LJ"];
    temp4["cDiJ"] = temp3["cDkJ"] * Gamma1["ki"];
    temp3.zero();
    temp3["cBiJ"] = temp4["cDiJ"] * Eta1["BD"];
    temp4.zero();
    temp4["aBiJ"] = temp3["cBiJ"] * Eta1["ac"];
    temp["aBiJ"] += 0.25 * temp4["aBiJ"] * temp2["iJaB"];

    temp1.zero();
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
                    v1 += 0.25 * scale_ci;
                    v2 += 0.25 * scale_ci;
                    v3 += 1.00 * scale_ci;
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

    for (size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        auto v = actv_all_[i];
        for (size_t k = 0; k < size_a; ++k) {
            auto u = actv_all_[k];
            auto idx = k * na_ + i;
            auto z_a = Z.block("aa").data()[idx];
            auto z_b = Z.block("AA").data()[idx];
            auto gamma_a = Gamma1_.block("aa").data()[idx];
            auto gamma_b = Gamma1_.block("AA").data()[idx];

            auto v1 = z_a + scale_ci * gamma_a;
            auto v2 = z_b + scale_ci * gamma_b;

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
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "PPHH", "pPhH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"phph","phPH"});

    if (PT2_TERM) {
        temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
        temp1["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
        change_2rdm(temp1, Eeps2_m1, temp);

        temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
        temp1["CDKL"] = V["CDKL"] * Eeps2_m1["KLCD"];
        temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
        change_2rdm(temp1, Eeps2_p, temp);
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
    //NOTICE Just for test
    // compute_test_energy();

    outfile->Printf("\n    Computing Gradient .............................. Done\n");

    return std::make_shared<Matrix>("nullptr", 0, 0);
}


} 

