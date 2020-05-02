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

void DSRG_MRPT2::set_all_variables() {
	// TODO: set global variables for future use.
	// NOTICE: This function may better be merged into "dsrg_mrpt2.cc" in the future!!

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

    // // Set MO spaces.
    // set_ambit_space();

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


    //NOTICE: The dimension may be further reduced.
    Z = BTF_->build(CoreTensor, "Z Matrix", spin_cases({"gg"}));
    Z_b = BTF_->build(CoreTensor, "b(AX=b)", spin_cases({"gg"}));

    I = BTF_->build(CoreTensor, "identity matrix", spin_cases({"gg"}));
    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });



    set_tensor();

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
    set_z();
    set_w();
    Z.print();
    // W_.print();
    // F.print();
// for(size_t i=0; i<nmo_; i++){
//     outfile->Printf("\n%d   %.6f",i, Fa_[i]);
// }

    
}


void DSRG_MRPT2::set_z() {
    set_z_cc();  
    set_z_vv();
    set_z_aa_diag();
    iter_z();
}


void DSRG_MRPT2::set_w() {

    //NOTICE: w for {virtual-general}
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"hhgvp", "hHgvP"});

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp2["ijpeb"] = V["pbij"] * Eeps2_m1["ijeb"];
    temp2["iJpeB"] = V["pBiJ"] * Eeps2_m1["iJeB"];
    W_["pe"] += 0.25 * temp1["cdkl"] * temp2["ijpeb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ec"] * Eta1["bd"];
    W_["pe"] += 0.50 * temp1["cDkL"] * temp2["iJpeB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ec"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp2["ijpeb"] = V["pbij"] * Eeps2_p["ijeb"];
    temp2["iJpeB"] = V["pBiJ"] * Eeps2_p["iJeB"];
    W_["pe"] += 0.25 * temp1["cdkl"] * temp2["ijpeb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ec"] * Eta1["bd"];
    W_["pe"] += 0.50 * temp1["cDkL"] * temp2["iJpeB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ec"] * Eta1["BD"];

    W_["pe"] += Z["e,m1"] * F["m1,p"];

    W_["pe"] += Z["eu"] * H["vp"] * Gamma1["uv"];
    W_["pe"] += Z["eu"] * V["v,m1,p,n1"] * Gamma1["uv"] * I["m1,n1"];
    W_["pe"] += Z["eu"] * V["v,M1,p,N1"] * Gamma1["uv"] * I["M1,N1"];
    W_["pe"] += 0.5 * Z["eu"] * V["xypv"] * Gamma2["uvxy"];
    W_["pe"] += Z["eu"] * V["xYpV"] * Gamma2["uVxY"];

    W_["pe"] += Z["e,f1"] * F["f1,p"];


    W_["ei"] = W_["ie"];

    //NOTICE: w for {core-hole}
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"pphch", "pPhcH"});

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp2["abimj"] = V["abij"] * Eeps2_m1["mjab"];
    temp2["aBimJ"] = V["aBiJ"] * Eeps2_m1["mJaB"];
    W_["im"] += 0.25 * temp1["cdkl"] * temp2["abimj"] * Gamma1["km"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    W_["im"] += 0.50 * temp1["cDkL"] * temp2["aBimJ"] * Gamma1["km"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp2["abimj"] = V["abij"] * Eeps2_p["mjab"];
    temp2["aBimJ"] = V["aBiJ"] * Eeps2_p["mJaB"];
    W_["im"] += 0.25 * temp1["cdkl"] * temp2["abimj"] * Gamma1["km"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    W_["im"] += 0.50 * temp1["cDkL"] * temp2["aBimJ"] * Gamma1["km"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    W_["im"] += Z["e1,m"] * F["i,e1"];

    W_["im"] += Z["e1,m1"] * V["m1,m,e1,i"];
    W_["im"] += Z["E1,M1"] * V["m,M1,i,E1"];
    W_["im"] += Z["e1,m1"] * V["m1,i,e1,m"];
    W_["im"] += Z["E1,M1"] * V["i,M1,m,E1"];

    W_["im"] += Z["mu"] * F["ui"];
    W_["im"] -= Z["mu"] * H["vi"] * Gamma1["uv"];
    W_["im"] -= Z["mu"] * V["v,m1,i,n1"] * Gamma1["uv"] * I["m1,n1"];
    W_["im"] -= Z["mu"] * V["v,M1,i,N1"] * Gamma1["uv"] * I["M1,N1"];
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

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp2["zwbij"] = V["zbij"] * Eeps2_m1["ijwb"];
    temp2["zwBiJ"] = V["zBiJ"] * Eeps2_m1["iJwB"];
    W_["zw"] += 0.25 * temp1["cdkl"] * temp2["zwbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["wc"] * Eta1["bd"];
    W_["zw"] += 0.50 * temp1["cDkL"] * temp2["zwBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["wc"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp2["zwbij"] = V["zbij"] * Eeps2_p["ijwb"];
    temp2["zwBiJ"] = V["zBiJ"] * Eeps2_p["iJwB"];
    W_["zw"] += 0.25 * temp1["cdkl"] * temp2["zwbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["wc"] * Eta1["bd"];
    W_["zw"] += 0.50 * temp1["cDkL"] * temp2["zwBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["wc"] * Eta1["BD"];

    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppaah", "pPaaH"});

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp2["abzwj"] = V["abzj"] * Eeps2_m1["wjab"];
    temp2["aBzwJ"] = V["aBzJ"] * Eeps2_m1["wJaB"];
    W_["zw"] += 0.25 * temp1["cdkl"] * temp2["abzwj"] * Gamma1["kw"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    W_["zw"] += 0.50 * temp1["cDkL"] * temp2["aBzwJ"] * Gamma1["kw"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp2["abzwj"] = V["abzj"] * Eeps2_p["wjab"];
    temp2["aBzwJ"] = V["aBzJ"] * Eeps2_p["wJaB"];
    W_["zw"] += 0.25 * temp1["cdkl"] * temp2["abzwj"] * Gamma1["kw"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    W_["zw"] += 0.50 * temp1["cDkL"] * temp2["aBzwJ"] * Gamma1["kw"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

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
    W_["zw"] -= Z["n1,u"] * V["z,m1,n1,m"] * Gamma1["uw"] * I["m1,m"];
    W_["zw"] -= Z["n1,u"] * V["z,M1,n1,M"] * Gamma1["uw"] * I["M1,M"];
    W_["zw"] -= 0.5 * Z["n1,u"] * V["x,y,n1,z"] * Gamma2["u,w,x,y"];
    W_["zw"] -= Z["N1,U"] * V["y,X,z,N1"] * Gamma2["w,U,y,X"];
    W_["zw"] -= Z["n1,u"] * V["z,y,n1,v"] * Gamma2["u,v,w,y"];
    W_["zw"] -= 2.0 * Z["n1,u"] * V["z,Y,n1,V"] * Gamma2["u,V,w,Y"];

    W_["zw"] += Z["e1,u"] * H["z,e1"] * Gamma1["uw"];
    W_["zw"] += Z["e1,u"] * V["z,m1,e1,m"] * Gamma1["uw"] * I["m1,m"];
    W_["zw"] += Z["e1,u"] * V["z,M1,e1,M"] * Gamma1["uw"] * I["M1,M"];
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



void DSRG_MRPT2::set_z_cc() {

    //NOTICE: core diag alpha-alpha
    BlockedTensor val1 = BTF_->build(CoreTensor, "val1", {"c", "C"});
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"pphh"}));
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", spin_cases({"pphh"}));

    temp1["abmj"] = -2.0 * s * V["abmj"] * Delta2["mjab"] * Eeps2["mjab"];
    temp1["aBmJ"] = -2.0 * s * V["aBmJ"] * Delta2["mJaB"] * Eeps2["mJaB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];

    val1["m"] += temp1["abmj"] * temp2["cdkl"] * Gamma1["km"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"]; 
    val1["m"] += 2.0 * temp1["aBmJ"] * temp2["cDkL"] * Gamma1["km"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["abmj"] = 2.0 * s * V["abmj"] * Eeps2["mjab"];
    temp1["aBmJ"] = 2.0 * s * V["aBmJ"] * Eeps2["mJaB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];

    val1["m"] += temp1["abmj"] * temp2["cdkl"] * Gamma1["km"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"]; 
    val1["m"] += 2.0 * temp1["aBmJ"] * temp2["cDkL"] * Gamma1["km"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["abmj"] = -1.0 * V["abmj"] * Eeps2_m2["mjab"];
    temp1["aBmJ"] = -1.0 * V["aBmJ"] * Eeps2_m2["mJaB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];

    val1["m"] += temp1["abmj"] * temp2["cdkl"] * Gamma1["km"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"]; 
    val1["m"] += 2.0 * temp1["aBmJ"] * temp2["cDkL"] * Gamma1["km"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 



    //NOTICE: core diag beta-beta
    temp1.zero();
    temp2.zero();

    temp1["ABMJ"] = -2.0 * s * V["ABMJ"] * Delta2["MJAB"] * Eeps2["MJAB"];
    temp1["bAjM"] = -2.0 * s * V["bAjM"] * Delta2["jMbA"] * Eeps2["jMbA"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_m1["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_m1["lKdC"];

    val1["M"] += temp1["ABMJ"] * temp2["CDKL"] * Gamma1["KM"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"]; 
    val1["M"] += 2.0 * temp1["bAjM"] * temp2["dClK"] * Gamma1["KM"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"]; 
    temp1.zero();
    temp2.zero();

    temp1["ABMJ"] = 2.0 * s * V["ABMJ"] * Eeps2["MJAB"];
    temp1["bAjM"] = 2.0 * s * V["bAjM"] * Eeps2["jMbA"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_p["lKdC"];

    val1["M"] += temp1["ABMJ"] * temp2["CDKL"] * Gamma1["KM"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"]; 
    val1["M"] += 2.0 * temp1["bAjM"] * temp2["dClK"] * Gamma1["KM"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"]; 
    temp1.zero();
    temp2.zero();

    temp1["ABMJ"] = -1.0 * V["ABMJ"] * Eeps2_m2["MJAB"];
    temp1["bAjM"] = -1.0 * V["bAjM"] * Eeps2_m2["jMbA"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_p["lKdC"];

    val1["M"] += temp1["ABMJ"] * temp2["CDKL"] * Gamma1["KM"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"]; 
    val1["M"] += 2.0 * temp1["bAjM"] * temp2["dClK"] * Gamma1["KM"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"]; 



    //NOTICE: core-core alpha-alpha normal
    BlockedTensor zmn = BTF_->build(CoreTensor, "z{mn} normal", {"cc", "CC"});
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppcch", "pPccH"});

    temp1["cdnl"] = V["cdnl"] * Eeps2_p["nlcd"];
    temp1["cDnL"] = V["cDnL"] * Eeps2_p["nLcD"];
    temp2["abmnj"] = V["abmj"] * Eeps2_m1["njab"];
    temp2["aBmnJ"] = V["aBmJ"] * Eeps2_m1["nJaB"];
    zmn["mn"] += 0.25 * temp1["cdnl"] * temp2["abmnj"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn["mn"] += 0.50 * temp1["cDnL"] * temp2["aBmnJ"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdnl"] = V["cdnl"] * Eeps2_m1["nlcd"];
    temp1["cDnL"] = V["cDnL"] * Eeps2_m1["nLcD"];
    temp2["abmnj"] = V["abmj"] * Eeps2_p["njab"];
    temp2["aBmnJ"] = V["aBmJ"] * Eeps2_p["nJaB"];
    zmn["mn"] += 0.25 * temp1["cdnl"] * temp2["abmnj"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn["mn"] += 0.50 * temp1["cDnL"] * temp2["aBmnJ"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdml"] = V["cdml"] * Eeps2_p["mlcd"];
    temp1["cDmL"] = V["cDmL"] * Eeps2_p["mLcD"];
    temp2["abnmj"] = V["abnj"] * Eeps2_m1["mjab"];
    temp2["aBnmJ"] = V["aBnJ"] * Eeps2_m1["mJaB"];
    zmn["mn"] -= 0.25 * temp1["cdml"] * temp2["abnmj"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn["mn"] -= 0.50 * temp1["cDmL"] * temp2["aBnmJ"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdml"] = V["cdml"] * Eeps2_m1["mlcd"];
    temp1["cDmL"] = V["cDmL"] * Eeps2_m1["mLcD"];
    temp2["abnmj"] = V["abnj"] * Eeps2_p["mjab"];
    temp2["aBnmJ"] = V["aBnJ"] * Eeps2_p["mJaB"];
    zmn["mn"] -= 0.25 * temp1["cdml"] * temp2["abnmj"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn["mn"] -= 0.50 * temp1["cDmL"] * temp2["aBnmJ"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];


    //NOTICE: core-core beta-beta normal
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"PPCH", "pPhC"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"PPCCH", "pPCCh"});

    temp1["CDNL"] = V["CDNL"] * Eeps2_p["NLCD"];
    temp1["dClN"] = V["dClN"] * Eeps2_p["lNdC"];
    temp2["ABMNJ"] = V["ABMJ"] * Eeps2_m1["NJAB"];
    temp2["bAMNj"] = V["bAjM"] * Eeps2_m1["jNbA"];
    zmn["MN"] += 0.25 * temp1["CDNL"] * temp2["ABMNJ"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn["MN"] += 0.50 * temp1["dClN"] * temp2["bAMNj"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["CDNL"] = V["CDNL"] * Eeps2_m1["NLCD"];
    temp1["dClN"] = V["dClN"] * Eeps2_m1["lNdC"];
    temp2["ABMNJ"] = V["ABMJ"] * Eeps2_p["NJAB"];
    temp2["bAMNj"] = V["bAjM"] * Eeps2_p["jNbA"];
    zmn["MN"] += 0.25 * temp1["CDNL"] * temp2["ABMNJ"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn["MN"] += 0.50 * temp1["dClN"] * temp2["bAMNj"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["CDML"] = V["CDML"] * Eeps2_p["MLCD"];
    temp1["dClM"] = V["dClM"] * Eeps2_p["lMdC"];
    temp2["ABNMJ"] = V["ABNJ"] * Eeps2_m1["MJAB"];
    temp2["bANMj"] = V["bAjN"] * Eeps2_m1["jMbA"];
    zmn["MN"] -= 0.25 * temp1["CDML"] * temp2["ABNMJ"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn["MN"] -= 0.50 * temp1["dClM"] * temp2["bANMj"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["CDML"] = V["CDML"] * Eeps2_m1["MLCD"];
    temp1["dClM"] = V["dClM"] * Eeps2_m1["lMdC"];
    temp2["ABNMJ"] = V["ABNJ"] * Eeps2_p["MJAB"];
    temp2["bANMj"] = V["bAjN"] * Eeps2_p["jMbA"];
    zmn["MN"] -= 0.25 * temp1["CDML"] * temp2["ABNMJ"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn["MN"] -= 0.50 * temp1["dClM"] * temp2["bANMj"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();


    //NOTICE: core-core alpha-alpha degenerate orbital case
    BlockedTensor zmn_d = BTF_->build(CoreTensor, "z{mn} degenerate orbital case", {"cc", "CC"});
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppcch", "pPccH"});

    temp1["abmj"] = V["abmj"] * Delta2["mjab"] * Eeps2["mjab"];
    temp1["aBmJ"] = V["aBmJ"] * Delta2["mJaB"] * Eeps2["mJaB"];
    temp2["cdmnl"] = V["cdnl"] * Eeps2_m1["mlcd"];
    temp2["cDmnL"] = V["cDnL"] * Eeps2_m1["mLcD"];
    zmn_d["mn"] -= 0.5 * s * temp1["abmj"] * temp2["cdmnl"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn_d["mn"] -= s * temp1["aBmJ"] * temp2["cDmnL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["abmj"] = V["abmj"] * Eeps2_m1["mjab"];
    temp1["aBmJ"] = V["aBmJ"] * Eeps2_m1["mJaB"];
    temp2["cdmnl"] = V["cdnl"] * Delta2["mlcd"] * Eeps2["mlcd"];
    temp2["cDmnL"] = V["cDnL"] * Delta2["mLcD"] * Eeps2["mLcD"];
    zmn_d["mn"] -= 0.5 * s * temp1["abmj"] * temp2["cdmnl"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn_d["mn"] -= s * temp1["aBmJ"] * temp2["cDmnL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["abmj"] = V["abmj"] * Eeps2_p["mjab"];
    temp1["aBmJ"] = V["aBmJ"] * Eeps2_p["mJaB"];
    temp2["cdmnl"] = V["cdnl"] * Eeps2["mlcd"];
    temp2["cDmnL"] = V["cDnL"] * Eeps2["mLcD"];
    zmn_d["mn"] += 0.5 * s * temp1["abmj"] * temp2["cdmnl"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn_d["mn"] += s * temp1["aBmJ"] * temp2["cDmnL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["abmj"] = V["abmj"] * Eeps2["mjab"];
    temp1["aBmJ"] = V["aBmJ"] * Eeps2["mJaB"];
    temp2["cdmnl"] = V["cdnl"] * Eeps2_p["mlcd"];
    temp2["cDmnL"] = V["cDnL"] * Eeps2_p["mLcD"];
    zmn_d["mn"] += 0.5 * s * temp1["abmj"] * temp2["cdmnl"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn_d["mn"] += s * temp1["aBmJ"] * temp2["cDmnL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["abmj"] = V["abmj"] * Eeps2_p["mjab"];
    temp1["aBmJ"] = V["aBmJ"] * Eeps2_p["mJaB"];
    temp2["cdmnl"] = V["cdnl"] * Eeps2_m2["mlcd"];
    temp2["cDmnL"] = V["cDnL"] * Eeps2_m2["mLcD"];
    zmn_d["mn"] -= 0.25 * temp1["abmj"] * temp2["cdmnl"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn_d["mn"] -= 0.50 * temp1["aBmJ"] * temp2["cDmnL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["abmj"] = V["abmj"] * Eeps2_m2["mjab"];
    temp1["aBmJ"] = V["aBmJ"] * Eeps2_m2["mJaB"];
    temp2["cdmnl"] = V["cdnl"] * Eeps2_p["mlcd"];
    temp2["cDmnL"] = V["cDnL"] * Eeps2_p["mLcD"];
    zmn_d["mn"] -= 0.25 * temp1["abmj"] * temp2["cdmnl"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    zmn_d["mn"] -= 0.50 * temp1["aBmJ"] * temp2["cDmnL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];


    //NOTICE: core-core beta-beta degenerate orbital case
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"PPCH", "pPhC"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"PPCCH", "pPCCh"});

    temp1["ABMJ"] = V["ABMJ"] * Delta2["MJAB"] * Eeps2["MJAB"];
    temp1["bAjM"] = V["bAjM"] * Delta2["jMbA"] * Eeps2["jMbA"];
    temp2["CDMNL"] = V["CDNL"] * Eeps2_m1["MLCD"];
    temp2["dCMNl"] = V["dClN"] * Eeps2_m1["lMdC"];
    zmn_d["MN"] -= 0.5 * s * temp1["ABMJ"] * temp2["CDMNL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn_d["MN"] -= s * temp1["bAjM"] * temp2["dCMNl"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["ABMJ"] = V["ABMJ"] * Eeps2_m1["MJAB"];
    temp1["bAjM"] = V["bAjM"] * Eeps2_m1["jMbA"];
    temp2["CDMNL"] = V["CDNL"] * Delta2["MLCD"] * Eeps2["MLCD"];
    temp2["dCMNl"] = V["dClN"] * Delta2["lMdC"] * Eeps2["lMdC"];
    zmn_d["MN"] -= 0.5 * s * temp1["ABMJ"] * temp2["CDMNL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn_d["MN"] -= s * temp1["bAjM"] * temp2["dCMNl"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["ABMJ"] = V["ABMJ"] * Eeps2_p["MJAB"];
    temp1["bAjM"] = V["bAjM"] * Eeps2_p["jMbA"];
    temp2["CDMNL"] = V["CDNL"] * Eeps2["MLCD"];
    temp2["dCMNl"] = V["dClN"] * Eeps2["lMdC"];
    zmn_d["MN"] += 0.5 * s * temp1["ABMJ"] * temp2["CDMNL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn_d["MN"] += s * temp1["bAjM"] * temp2["dCMNl"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["ABMJ"] = V["ABMJ"] * Eeps2["MJAB"];
    temp1["bAjM"] = V["bAjM"] * Eeps2["jMbA"];
    temp2["CDMNL"] = V["CDNL"] * Eeps2_p["MLCD"];
    temp2["dCMNl"] = V["dClN"] * Eeps2_p["lMdC"];
    zmn_d["MN"] += 0.5 * s * temp1["ABMJ"] * temp2["CDMNL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn_d["MN"] += s * temp1["bAjM"] * temp2["dCMNl"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["ABMJ"] = V["ABMJ"] * Eeps2_p["MJAB"];
    temp1["bAjM"] = V["bAjM"] * Eeps2_p["jMbA"];
    temp2["CDMNL"] = V["CDNL"] * Eeps2_m2["MLCD"];
    temp2["dCMNl"] = V["dClN"] * Eeps2_m2["lMdC"];
    zmn_d["MN"] -= 0.25 * temp1["ABMJ"] * temp2["CDMNL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn_d["MN"] -= 0.50 * temp1["bAjM"] * temp2["dCMNl"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["ABMJ"] = V["ABMJ"] * Eeps2_m2["MJAB"];
    temp1["bAjM"] = V["bAjM"] * Eeps2_m2["jMbA"];
    temp2["CDMNL"] = V["CDNL"] * Eeps2_p["MLCD"];
    temp2["dCMNl"] = V["dClN"] * Eeps2_p["lMdC"];
    zmn_d["MN"] -= 0.25 * temp1["ABMJ"] * temp2["CDMNL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    zmn_d["MN"] -= 0.50 * temp1["bAjM"] * temp2["dCMNl"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"];


    for (const std::string& block : {"cc", "CC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block == "cc" && i[0] == i[1]) {              
                value = 0.5 * val1.block("c").data()[i[0]];
            }
            else if (block == "CC" && i[0] == i[1]) {
                value = 0.5 * val1.block("C").data()[i[0]];
            }

            else if (block == "cc") {
                if (fabs(Delta1.block("cc").data()[i[1] * ncore_ + i[0]])>1e-7){
                    value = zmn.block("cc").data()[i[0] * ncore_ + i[1]] / Delta1.block("cc").data()[i[1] * ncore_ + i[0]];
                }
                else{
                    value = zmn_d.block("cc").data()[i[0] * ncore_ + i[1]];
                }
            }
            else if (block == "CC") {
                if (fabs(Delta1.block("CC").data()[i[1] * ncore_ + i[0]])>1e-7){
                    value = zmn.block("CC").data()[i[0] * ncore_ + i[1]] / Delta1.block("CC").data()[i[1] * ncore_ + i[0]];
                }
                else{
                    value = zmn_d.block("CC").data()[i[0] * ncore_ + i[1]];
                }
            }
        });
    }   
}




void DSRG_MRPT2::set_z_vv() {

    //NOTICE: virtual diag alpha-alpha 
    BlockedTensor val2 = BTF_->build(CoreTensor, "val2", {"v", "V"});
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"pphh"}));
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", spin_cases({"pphh"}));

    temp1["ebij"] = 2.0 * s * V["ebij"] * Delta2["ijeb"] * Eeps2["ijeb"];
    temp1["eBiJ"] = 2.0 * s * V["eBiJ"] * Delta2["iJeB"] * Eeps2["iJeB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];

    val2["e"] += temp1["ebij"] * temp2["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ec"] * Eta1["bd"]; 
    val2["e"] += 2.0 * temp1["eBiJ"] * temp2["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ec"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["ebij"] = -2.0 * s * V["ebij"] * Eeps2["ijeb"];
    temp1["eBiJ"] = -2.0 * s * V["eBiJ"] * Eeps2["iJeB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];

    val2["e"] += temp1["ebij"] * temp2["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ec"] * Eta1["bd"]; 
    val2["e"] += 2.0 * temp1["eBiJ"] * temp2["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ec"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["ebij"] = V["ebij"] * Eeps2_m2["ijeb"];
    temp1["eBiJ"] = V["eBiJ"] * Eeps2_m2["iJeB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];

    val2["e"] += temp1["ebij"] * temp2["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ec"] * Eta1["bd"]; 
    val2["e"] += 2.0 * temp1["eBiJ"] * temp2["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ec"] * Eta1["BD"];

    temp1.zero();
    temp2.zero();

    
    //NOTICE: virtual diag beta-beta
    temp1["EBIJ"] = 2.0 * s * V["EBIJ"] * Delta2["IJEB"] * Eeps2["IJEB"];
    temp1["bEjI"] = 2.0 * s * V["bEjI"] * Delta2["jIbE"] * Eeps2["jIbE"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_m1["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_m1["lKdC"];

    val2["E"] += temp1["EBIJ"] * temp2["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["EC"] * Eta1["BD"]; 
    val2["E"] += 2.0 * temp1["bEjI"] * temp2["dClK"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["EC"] * Eta1["bd"]; 
    temp1.zero();
    temp2.zero();

    temp1["EBIJ"] = -2.0 * s * V["EBIJ"] * Eeps2["IJEB"];
    temp1["bEjI"] = -2.0 * s * V["bEjI"] * Eeps2["jIbE"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_p["lKdC"];

    val2["E"] += temp1["EBIJ"] * temp2["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["EC"] * Eta1["BD"]; 
    val2["E"] += 2.0 * temp1["bEjI"] * temp2["dClK"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["EC"] * Eta1["bd"]; 
    temp1.zero();
    temp2.zero();

    temp1["EBIJ"] = V["EBIJ"] * Eeps2_m2["IJEB"];
    temp1["bEjI"] = V["bEjI"] * Eeps2_m2["jIbE"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_p["lKdC"];

    val2["E"] += temp1["EBIJ"] * temp2["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["EC"] * Eta1["BD"]; 
    val2["E"] += 2.0 * temp1["bEjI"] * temp2["dClK"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["EC"] * Eta1["bd"]; 



    //NOTICE: virtual-virtual alpha-alpha normal (york's idea on contractions)
    BlockedTensor zef = BTF_->build(CoreTensor, "z{ef} normal", {"vv", "VV"});
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"vvphh", "vvPhH"});
    temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
    temp2["efbij"] = V["ebij"] * Eeps2_m1["ijfb"];
    temp2["efBiJ"] = V["eBiJ"] * Eeps2_m1["iJfB"];
    zef["ef"] += 0.25 * temp1["fdkl"] * temp2["efbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef["ef"] += 0.50 * temp1["fDkL"] * temp2["efBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];


    temp1.zero();
    temp2.zero();
    temp1["fdkl"] = V["fdkl"] * Eeps2_m1["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2_m1["kLfD"];
    temp2["efbij"] = V["ebij"] * Eeps2_p["ijfb"];
    temp2["efBiJ"] = V["eBiJ"] * Eeps2_p["iJfB"];
    zef["ef"] += 0.25 * temp1["fdkl"] * temp2["efbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef["ef"] += 0.50 * temp1["fDkL"] * temp2["efBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];

    temp1.zero();
    temp2.zero();
    temp1["edkl"] = V["edkl"] * Eeps2_p["kled"];
    temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
    temp2["febij"] = V["fbij"] * Eeps2_m1["ijeb"];
    temp2["feBiJ"] = V["fBiJ"] * Eeps2_m1["iJeB"];
    zef["ef"] -= 0.25 * temp1["edkl"] * temp2["febij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef["ef"] -= 0.50 * temp1["eDkL"] * temp2["feBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];

    temp1.zero();
    temp2.zero();
    temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"];
    temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
    temp2["febij"] = V["fbij"] * Eeps2_p["ijeb"];
    temp2["feBiJ"] = V["fBiJ"] * Eeps2_p["iJeB"];
    zef["ef"] -= 0.25 * temp1["edkl"] * temp2["febij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef["ef"] -= 0.50 * temp1["eDkL"] * temp2["feBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];



    //NOTICE: virtual-virtual beta-beta normal
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"VPHH", "pVhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"VVPHH", "VpVhH"});
    temp1["FDKL"] = V["FDKL"] * Eeps2_p["KLFD"];
    temp1["dFlK"] = V["dFlK"] * Eeps2_p["lKdF"];
    temp2["EFBIJ"] = V["EBIJ"] * Eeps2_m1["IJFB"];
    temp2["EbFjI"] = V["bEjI"] * Eeps2_m1["jIbF"];
    zef["EF"] += 0.25 * temp1["FDKL"] * temp2["EFBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef["EF"] += 0.50 * temp1["dFlK"] * temp2["EbFjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];


    temp1.zero();
    temp2.zero();
    temp1["FDKL"] = V["FDKL"] * Eeps2_m1["KLFD"];
    temp1["dFlK"] = V["dFlK"] * Eeps2_m1["lKdF"];
    temp2["EFBIJ"] = V["EBIJ"] * Eeps2_p["IJFB"];
    temp2["EbFjI"] = V["bEjI"] * Eeps2_p["jIbF"];
    zef["EF"] += 0.25 * temp1["FDKL"] * temp2["EFBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef["EF"] += 0.50 * temp1["dFlK"] * temp2["EbFjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];

    temp1.zero();
    temp2.zero();
    temp1["EDKL"] = V["EDKL"] * Eeps2_p["KLED"];
    temp1["dElK"] = V["dElK"] * Eeps2_p["lKdE"];
    temp2["FEBIJ"] = V["FBIJ"] * Eeps2_m1["IJEB"];
    temp2["FbEjI"] = V["bFjI"] * Eeps2_m1["jIbE"];
    zef["EF"] -= 0.25 * temp1["EDKL"] * temp2["FEBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef["EF"] -= 0.50 * temp1["dElK"] * temp2["FbEjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];

    temp1.zero();
    temp2.zero();
    temp1["EDKL"] = V["EDKL"] * Eeps2_m1["KLED"];
    temp1["dElK"] = V["dElK"] * Eeps2_m1["lKdE"];
    temp2["FEBIJ"] = V["FBIJ"] * Eeps2_p["IJEB"];
    temp2["FbEjI"] = V["bFjI"] * Eeps2_p["jIbE"];
    zef["EF"] -= 0.25 * temp1["EDKL"] * temp2["FEBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef["EF"] -= 0.50 * temp1["dElK"] * temp2["FbEjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];


    //NOTICE: virtual-virtual alpha-alpha degenerate orbital case
    BlockedTensor zef_d = BTF_->build(CoreTensor, "z{ef} degenerate orbital case", {"vv", "VV"});
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"vvphh", "vvPhH"});

    temp1["fdkl"] = V["fdkl"] * Eeps2_m1["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2_m1["kLfD"];
    temp2["efbij"] = V["ebij"] * Delta2["ijfb"] * Eeps2["ijfb"];
    temp2["efBiJ"] = V["eBiJ"] * Delta2["iJfB"] * Eeps2["iJfB"];
    zef_d["ef"] += 0.5 * s * temp1["fdkl"] * temp2["efbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef_d["ef"] += s * temp1["fDkL"] * temp2["efBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["fdkl"] = V["fdkl"] * Delta2["klfd"] * Eeps2["klfd"];
    temp1["fDkL"] = V["fDkL"] * Delta2["kLfD"] * Eeps2["kLfD"];
    temp2["efbij"] = V["ebij"] * Eeps2_m1["ijfb"];
    temp2["efBiJ"] = V["eBiJ"] * Eeps2_m1["iJfB"];
    zef_d["ef"] += 0.5 * s * temp1["fdkl"] * temp2["efbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef_d["ef"] += s * temp1["fDkL"] * temp2["efBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
    temp2["efbij"] = V["ebij"] * Eeps2["ijfb"];
    temp2["efBiJ"] = V["eBiJ"] * Eeps2["iJfB"];
    zef_d["ef"] -= 0.5 * s * temp1["fdkl"] * temp2["efbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef_d["ef"] -= s * temp1["fDkL"] * temp2["efBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["fdkl"] = V["fdkl"] * Eeps2["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2["kLfD"];
    temp2["efbij"] = V["ebij"] * Eeps2_p["ijfb"];
    temp2["efBiJ"] = V["eBiJ"] * Eeps2_p["iJfB"];
    zef_d["ef"] -= 0.5 * s * temp1["fdkl"] * temp2["efbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef_d["ef"] -= s * temp1["fDkL"] * temp2["efBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
    temp2["efbij"] = V["ebij"] * Eeps2_m2["ijfb"];
    temp2["efBiJ"] = V["eBiJ"] * Eeps2_m2["iJfB"];
    zef_d["ef"] += 0.25 * temp1["fdkl"] * temp2["efbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef_d["ef"] += 0.50 * temp1["fDkL"] * temp2["efBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["fdkl"] = V["fdkl"] * Eeps2_m2["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2_m2["kLfD"];
    temp2["efbij"] = V["ebij"] * Eeps2_p["ijfb"];
    temp2["efBiJ"] = V["eBiJ"] * Eeps2_p["iJfB"];
    zef_d["ef"] += 0.25 * temp1["fdkl"] * temp2["efbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    zef_d["ef"] += 0.50 * temp1["fDkL"] * temp2["efBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];


    //NOTICE: virtual-virtual beta-beta degenerate orbital case
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"VPHH", "pVhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"VVPHH", "VpVhH"});

    temp1["FDKL"] = V["FDKL"] * Eeps2_m1["KLFD"];
    temp1["dFlK"] = V["dFlK"] * Eeps2_m1["lKdF"];
    temp2["EFBIJ"] = V["EBIJ"] * Delta2["IJFB"] * Eeps2["IJFB"];
    temp2["EbFjI"] = V["bEjI"] * Delta2["jIbF"] * Eeps2["jIbF"];
    zef_d["EF"] += 0.5 * s * temp1["FDKL"] * temp2["EFBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef_d["EF"] += s * temp1["dFlK"] * temp2["EbFjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["FDKL"] = V["FDKL"] * Delta2["KLFD"] * Eeps2["KLFD"];
    temp1["dFlK"] = V["dFlK"] * Delta2["lKdF"] * Eeps2["lKdF"];
    temp2["EFBIJ"] = V["EBIJ"] * Eeps2_m1["IJFB"];
    temp2["EbFjI"] = V["bEjI"] * Eeps2_m1["jIbF"];
    zef_d["EF"] += 0.5 * s * temp1["FDKL"] * temp2["EFBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef_d["EF"] += s * temp1["dFlK"] * temp2["EbFjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["FDKL"] = V["FDKL"] * Eeps2_p["KLFD"];
    temp1["dFlK"] = V["dFlK"] * Eeps2_p["lKdF"];
    temp2["EFBIJ"] = V["EBIJ"] * Eeps2["IJFB"];
    temp2["EbFjI"] = V["bEjI"] * Eeps2["jIbF"];
    zef_d["EF"] -= 0.5 * s * temp1["FDKL"] * temp2["EFBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef_d["EF"] -= s * temp1["dFlK"] * temp2["EbFjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["FDKL"] = V["FDKL"] * Eeps2["KLFD"];
    temp1["dFlK"] = V["dFlK"] * Eeps2["lKdF"];
    temp2["EFBIJ"] = V["EBIJ"] * Eeps2_p["IJFB"];
    temp2["EbFjI"] = V["bEjI"] * Eeps2_p["jIbF"];
    zef_d["EF"] -= 0.5 * s * temp1["FDKL"] * temp2["EFBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef_d["EF"] -= s * temp1["dFlK"] * temp2["EbFjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["FDKL"] = V["FDKL"] * Eeps2_p["KLFD"];
    temp1["dFlK"] = V["dFlK"] * Eeps2_p["lKdF"];
    temp2["EFBIJ"] = V["EBIJ"] * Eeps2_m2["IJFB"];
    temp2["EbFjI"] = V["bEjI"] * Eeps2_m2["jIbF"];
    zef_d["EF"] += 0.25 * temp1["FDKL"] * temp2["EFBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef_d["EF"] += 0.50 * temp1["dFlK"] * temp2["EbFjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];
    temp1.zero();
    temp2.zero();

    temp1["FDKL"] = V["FDKL"] * Eeps2_m2["KLFD"];
    temp1["dFlK"] = V["dFlK"] * Eeps2_m2["lKdF"];
    temp2["EFBIJ"] = V["EBIJ"] * Eeps2_p["IJFB"];
    temp2["EbFjI"] = V["bEjI"] * Eeps2_p["jIbF"];
    zef_d["EF"] += 0.25 * temp1["FDKL"] * temp2["EFBIJ"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
    zef_d["EF"] += 0.50 * temp1["dFlK"] * temp2["EbFjI"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["bd"];



    for (const std::string& block : {"vv", "VV"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block == "vv" && i[0] == i[1]) { 
                value = 0.5 * val2.block("v").data()[i[0]];
            }
            else if (block == "VV" && i[0] == i[1]) {      
                value = 0.5 * val2.block("V").data()[i[0]];
            }

            else if (block == "vv") {
                if (fabs(Delta1.block("vv").data()[i[1] * nvirt_ + i[0]])>1e-7){
                    value = zef.block("vv").data()[i[0] * nvirt_ + i[1]] / Delta1.block("vv").data()[i[1] * nvirt_ + i[0]];
                }
                else{
                    value = zef_d.block("vv").data()[i[0] * nvirt_ + i[1]];
                }
            }
            else if (block == "VV") {
                if (fabs(Delta1.block("VV").data()[i[1] * nvirt_ + i[0]])>1e-7){
                    value = zef.block("VV").data()[i[0] * nvirt_ + i[1]] / Delta1.block("VV").data()[i[1] * nvirt_ + i[0]];
                }
                else{
                    value = zef_d.block("VV").data()[i[0] * nvirt_ + i[1]];
                }

            }
            if (fabs(value)<1e-7) {value = 0;}
        });
    } 


}



void DSRG_MRPT2::set_z_aa_diag() {

    //NOTICE: active diag alpha-alpha
    BlockedTensor val3 = BTF_->build(CoreTensor, "val3", {"a", "A"});
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"pphh"}));
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", spin_cases({"pphh"}));

    temp1["abuj"] = -2.0 * s * V["abuj"] * Delta2["ujab"] * Eeps2["ujab"];
    temp1["aBuJ"] = -2.0 * s * V["aBuJ"] * Delta2["uJaB"] * Eeps2["uJaB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];

    val3["u"] += temp1["abuj"] * temp2["cdkl"] * Gamma1["ku"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"]; 
    val3["u"] += 2.0 * temp1["aBuJ"] * temp2["cDkL"] * Gamma1["ku"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["abuj"] = 2.0 * s * V["abuj"] * Eeps2["ujab"];
    temp1["aBuJ"] = 2.0 * s * V["aBuJ"] * Eeps2["uJaB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];

    val3["u"] += temp1["abuj"] * temp2["cdkl"] * Gamma1["ku"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"]; 
    val3["u"] += 2.0 * temp1["aBuJ"] * temp2["cDkL"] * Gamma1["ku"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["abuj"] = -1.0 * V["abuj"] * Eeps2_m2["ujab"];
    temp1["aBuJ"] = -1.0 * V["aBuJ"] * Eeps2_m2["uJaB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];

    val3["u"] += temp1["abuj"] * temp2["cdkl"] * Gamma1["ku"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"]; 
    val3["u"] += 2.0 * temp1["aBuJ"] * temp2["cDkL"] * Gamma1["ku"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 


    //NOTICE: active diag beta-beta
    temp1.zero();
    temp2.zero();

    temp1["ABUJ"] = -2.0 * s * V["ABUJ"] * Delta2["UJAB"] * Eeps2["UJAB"];
    temp1["bAjU"] = -2.0 * s * V["bAjU"] * Delta2["jUbA"] * Eeps2["jUbA"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_m1["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_m1["lKdC"];

    val3["U"] += temp1["ABUJ"] * temp2["CDKL"] * Gamma1["KU"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"]; 
    val3["U"] += 2.0 * temp1["bAjU"] * temp2["dClK"] * Gamma1["KU"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"]; 
    temp1.zero();
    temp2.zero();

    temp1["ABUJ"] = 2.0 * s * V["ABUJ"] * Eeps2["UJAB"];
    temp1["bAjU"] = 2.0 * s * V["bAjU"] * Eeps2["jUbA"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_p["lKdC"];

    val3["U"] += temp1["ABUJ"] * temp2["CDKL"] * Gamma1["KU"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"]; 
    val3["U"] += 2.0 * temp1["bAjU"] * temp2["dClK"] * Gamma1["KU"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"]; 
    temp1.zero();
    temp2.zero();

    temp1["ABUJ"] = -1.0 * V["ABUJ"] * Eeps2_m2["UJAB"];
    temp1["bAjU"] = -1.0 * V["bAjU"] * Eeps2_m2["jUbA"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_p["lKdC"];

    val3["U"] += temp1["ABUJ"] * temp2["CDKL"] * Gamma1["KU"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"]; 
    val3["U"] += 2.0 * temp1["bAjU"] * temp2["dClK"] * Gamma1["KU"] * Gamma1["lj"] * Eta1["AC"] * Eta1["bd"]; 




    //NOTICE: active diag alpha-alpha
    temp1.zero();
    temp2.zero();

    temp1["ubij"] = 2.0 * s * V["ubij"] * Delta2["ijub"] * Eeps2["ijub"];
    temp1["uBiJ"] = 2.0 * s * V["uBiJ"] * Delta2["iJuB"] * Eeps2["iJuB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];

    val3["u"] += temp1["ubij"] * temp2["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["uc"] * Eta1["bd"]; 
    val3["u"] += 2.0 * temp1["uBiJ"] * temp2["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["uc"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["ubij"] = -2.0 * s * V["ubij"] * Eeps2["ijub"];
    temp1["uBiJ"] = -2.0 * s * V["uBiJ"] * Eeps2["iJuB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];

    val3["u"] += temp1["ubij"] * temp2["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["uc"] * Eta1["bd"]; 
    val3["u"] += 2.0 * temp1["uBiJ"] * temp2["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["uc"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["ubij"] = V["ubij"] * Eeps2_m2["ijub"];
    temp1["uBiJ"] = V["uBiJ"] * Eeps2_m2["iJuB"];
    temp2["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp2["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];

    val3["u"] += temp1["ubij"] * temp2["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["uc"] * Eta1["bd"]; 
    val3["u"] += 2.0 * temp1["uBiJ"] * temp2["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["uc"] * Eta1["BD"];

    temp1.zero();
    temp2.zero();

    
    //NOTICE: active diag beta-beta
    temp1["UBIJ"] = 2.0 * s * V["UBIJ"] * Delta2["IJUB"] * Eeps2["IJUB"];
    temp1["bUjI"] = 2.0 * s * V["bUjI"] * Delta2["jIbU"] * Eeps2["jIbU"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_m1["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_m1["lKdC"];

    val3["U"] += temp1["UBIJ"] * temp2["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["UC"] * Eta1["BD"]; 
    val3["U"] += 2.0 * temp1["bUjI"] * temp2["dClK"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["UC"] * Eta1["bd"]; 
    temp1.zero();
    temp2.zero();

    temp1["UBIJ"] = -2.0 * s * V["UBIJ"] * Eeps2["IJUB"];
    temp1["bUjI"] = -2.0 * s * V["bUjI"] * Eeps2["jIbU"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_p["lKdC"];

    val3["U"] += temp1["UBIJ"] * temp2["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["UC"] * Eta1["BD"]; 
    val3["U"] += 2.0 * temp1["bUjI"] * temp2["dClK"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["UC"] * Eta1["bd"]; 
    temp1.zero();
    temp2.zero();

    temp1["UBIJ"] = V["UBIJ"] * Eeps2_m2["IJUB"];
    temp1["bUjI"] = V["bUjI"] * Eeps2_m2["jIbU"];
    temp2["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp2["dClK"] = V["dClK"] * Eeps2_p["lKdC"];

    val3["U"] += temp1["UBIJ"] * temp2["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["UC"] * Eta1["BD"]; 
    val3["U"] += 2.0 * temp1["bUjI"] * temp2["dClK"] * Gamma1["KI"] * Gamma1["lj"] * Eta1["UC"] * Eta1["bd"]; 


    for (const std::string& block : {"aa", "AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block == "aa" && i[0] == i[1]) {              
                value = 0.5 * val3.block("a").data()[i[0]];
            }
            else if (block == "AA" && i[0] == i[1]) {
                value = 0.5 * val3.block("A").data()[i[0]];
            }
        });
    } 

}

void DSRG_MRPT2::iter_z() {
    bool converged = false;
    int iter = 1;
    int maxiter = 4000;
    double convergence = 1e-8;

    //NOTICE: constant b for z{core-virtual}
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppvch", "pPvcH"});

    temp1["cdml"] = V["cdml"] * Eeps2_p["mlcd"];
    temp1["cDmL"] = V["cDmL"] * Eeps2_p["mLcD"];
    temp2["abemj"] = V["abej"] * Eeps2_m1["mjab"];
    temp2["aBemJ"] = V["aBeJ"] * Eeps2_m1["mJaB"];
    Z_b["em"] += 0.25 * temp1["cdml"] * temp2["abemj"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"]; 
    Z_b["em"] += 0.50 * temp1["cDmL"] * temp2["aBemJ"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["cdml"] = V["cdml"] * Eeps2_m1["mlcd"];
    temp1["cDmL"] = V["cDmL"] * Eeps2_m1["mLcD"];
    temp2["abemj"] = V["abej"] * Eeps2_p["mjab"];
    temp2["aBemJ"] = V["aBeJ"] * Eeps2_p["mJaB"];
    Z_b["em"] += 0.25 * temp1["cdml"] * temp2["abemj"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"]; 
    Z_b["em"] += 0.50 * temp1["cDmL"] * temp2["aBemJ"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 

    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"cvphh", "cvPhH"});

    temp1["edkl"] = V["edkl"] * Eeps2_p["kled"];
    temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
    temp2["mebij"] = V["mbij"] * Eeps2_m1["ijeb"];
    temp2["meBiJ"] = V["mBiJ"] * Eeps2_m1["iJeB"];
    Z_b["em"] -= 0.25 * temp1["edkl"] * temp2["mebij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    Z_b["em"] -= 0.50 * temp1["eDkL"] * temp2["meBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"];
    temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
    temp2["mebij"] = V["mbij"] * Eeps2_p["ijeb"];
    temp2["meBiJ"] = V["mBiJ"] * Eeps2_p["iJeB"];
    Z_b["em"] -= 0.25 * temp1["edkl"] * temp2["mebij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    Z_b["em"] -= 0.50 * temp1["eDkL"] * temp2["meBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];

    Z_b["em"] += Z["m1,n1"] * V["n1,e,m1,m"];
    Z_b["em"] += Z["M1,N1"] * V["e,N1,m,M1"];

    Z_b["em"] += Z["e1,f"] * V["f,e,e1,m"];
    Z_b["em"] += Z["E1,F"] * V["e,F,m,E1"];

    Z_b["me"] = Z_b["em"];

    //NOTICE: constant b for z{active-active}
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"aaphh", "aaPhH"});

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp2["wzbij"] = V["wbij"] * Eeps2_m1["ijzb"];
    temp2["wzBiJ"] = V["wBiJ"] * Eeps2_m1["iJzB"];
    Z_b["wz"] += 0.25 * temp1["cdkl"] * temp2["wzbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["zc"] * Eta1["bd"]; 
    Z_b["wz"] += 0.50 * temp1["cDkL"] * temp2["wzBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["zc"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp2["wzbij"] = V["wbij"] * Eeps2_p["ijzb"];
    temp2["wzBiJ"] = V["wBiJ"] * Eeps2_p["iJzB"];
    Z_b["wz"] += 0.25 * temp1["cdkl"] * temp2["wzbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["zc"] * Eta1["bd"]; 
    Z_b["wz"] += 0.50 * temp1["cDkL"] * temp2["wzBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["zc"] * Eta1["BD"]; 

    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppaah", "pPaaH"});

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp2["abwzj"] = V["abwj"] * Eeps2_m1["zjab"];
    temp2["aBwzJ"] = V["aBwJ"] * Eeps2_m1["zJaB"];
    Z_b["wz"] += 0.25 * temp1["cdkl"] * temp2["abwzj"] * Gamma1["kz"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["wz"] += 0.50 * temp1["cDkL"] * temp2["aBwzJ"] * Gamma1["kz"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp2["abwzj"] = V["abwj"] * Eeps2_p["zjab"];
    temp2["aBwzJ"] = V["aBwJ"] * Eeps2_p["zJaB"];
    Z_b["wz"] += 0.25 * temp1["cdkl"] * temp2["abwzj"] * Gamma1["kz"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["wz"] += 0.50 * temp1["cDkL"] * temp2["aBwzJ"] * Gamma1["kz"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];   

    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"aaphh", "aaPhH"});

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp2["zwbij"] = V["zbij"] * Eeps2_m1["ijwb"];
    temp2["zwBiJ"] = V["zBiJ"] * Eeps2_m1["iJwB"];
    Z_b["wz"] -= 0.25 * temp1["cdkl"] * temp2["zwbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["wc"] * Eta1["bd"]; 
    Z_b["wz"] -= 0.50 * temp1["cDkL"] * temp2["zwBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["wc"] * Eta1["BD"]; 
    temp1.zero();
    temp2.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp2["zwbij"] = V["zbij"] * Eeps2_p["ijwb"];
    temp2["zwBiJ"] = V["zBiJ"] * Eeps2_p["iJwB"];
    Z_b["wz"] -= 0.25 * temp1["cdkl"] * temp2["zwbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["wc"] * Eta1["bd"]; 
    Z_b["wz"] -= 0.50 * temp1["cDkL"] * temp2["zwBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["wc"] * Eta1["BD"]; 
 
    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"pphh", "pPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppaah", "pPaaH"});

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp2["abzwj"] = V["abzj"] * Eeps2_m1["wjab"];
    temp2["aBzwJ"] = V["aBzJ"] * Eeps2_m1["wJaB"];
    Z_b["wz"] -= 0.25 * temp1["cdkl"] * temp2["abzwj"] * Gamma1["kw"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["wz"] -= 0.50 * temp1["cDkL"] * temp2["aBzwJ"] * Gamma1["kw"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp2["abzwj"] = V["abzj"] * Eeps2_p["wjab"];
    temp2["aBzwJ"] = V["aBzJ"] * Eeps2_p["wJaB"];
    Z_b["wz"] -= 0.25 * temp1["cdkl"] * temp2["abzwj"] * Gamma1["kw"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["wz"] -= 0.50 * temp1["cDkL"] * temp2["aBzwJ"] * Gamma1["kw"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"]; 

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

    temp1["udkl"] = V["udkl"] * Eeps2_p["klud"];
    temp1["uDkL"] = V["uDkL"] * Eeps2_p["kLuD"];
    temp2["ewbij"] = V["ebij"] * Eeps2_m1["ijwb"];
    temp2["ewBiJ"] = V["eBiJ"] * Eeps2_m1["iJwB"];
    Z_b["ew"] += 0.25 * temp1["udkl"] * temp2["ewbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["wu"] * Eta1["bd"];
    Z_b["ew"] += 0.50 * temp1["uDkL"] * temp2["ewBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["wu"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["udkl"] = V["udkl"] * Eeps2_m1["klud"];
    temp1["uDkL"] = V["uDkL"] * Eeps2_m1["kLuD"];
    temp2["ewbij"] = V["ebij"] * Eeps2_p["ijwb"];
    temp2["ewBiJ"] = V["eBiJ"] * Eeps2_p["iJwB"];
    Z_b["ew"] += 0.25 * temp1["udkl"] * temp2["ewbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["wu"] * Eta1["bd"];
    Z_b["ew"] += 0.50 * temp1["uDkL"] * temp2["ewBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["wu"] * Eta1["BD"];  

    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppah", "pPaH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppvah", "pPvaH"}); 

    temp1["cdul"] = V["cdul"] * Eeps2_p["ulcd"];
    temp1["cDuL"] = V["cDuL"] * Eeps2_p["uLcD"];
    temp2["abewj"] = V["abej"] * Eeps2_m1["wjab"];
    temp2["aBewJ"] = V["aBeJ"] * Eeps2_m1["wJaB"];
    Z_b["ew"] += 0.25 * temp1["cdul"] * temp2["abewj"] * Gamma1["uw"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["ew"] += 0.50 * temp1["cDuL"] * temp2["aBewJ"] * Gamma1["uw"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();  

    temp1["cdul"] = V["cdul"] * Eeps2_m1["ulcd"];
    temp1["cDuL"] = V["cDuL"] * Eeps2_m1["uLcD"];
    temp2["abewj"] = V["abej"] * Eeps2_p["wjab"];
    temp2["aBewJ"] = V["aBeJ"] * Eeps2_p["wJaB"];
    Z_b["ew"] += 0.25 * temp1["cdul"] * temp2["abewj"] * Gamma1["uw"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["ew"] += 0.50 * temp1["cDuL"] * temp2["aBewJ"] * Gamma1["uw"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"avphh", "avPhH"});

    temp1["edkl"] = V["edkl"] * Eeps2_p["kled"]; 
    temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
    temp2["webij"] = V["wbij"] * Eeps2_m1["ijeb"]; 
    temp2["weBiJ"] = V["wBiJ"] * Eeps2_m1["iJeB"]; 
    Z_b["ew"] -= 0.25 * temp1["edkl"] * temp2["webij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    Z_b["ew"] -= 0.50 * temp1["eDkL"] * temp2["weBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    temp1.zero();
    temp2.zero(); 

    temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"]; 
    temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
    temp2["webij"] = V["wbij"] * Eeps2_p["ijeb"]; 
    temp2["weBiJ"] = V["wBiJ"] * Eeps2_p["iJeB"];    
    Z_b["ew"] -= 0.25 * temp1["edkl"] * temp2["webij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    Z_b["ew"] -= 0.50 * temp1["eDkL"] * temp2["weBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];

    Z_b["ew"] -= Z["e,f1"] * F["f1,w"];

    Z_b["ew"] += Z["m1,n1"] * V["n1,v,m1,e"] * Gamma1["wv"];
    Z_b["ew"] += Z["M1,N1"] * V["v,N1,e,M1"] * Gamma1["wv"];

    Z_b["ew"] += Z["e1,f1"] * V["f1,v,e1,e"] * Gamma1["wv"];
    Z_b["ew"] += Z["E1,F1"] * V["v,F1,e,E1"] * Gamma1["wv"];

    Z_b["we"] = Z_b["ew"];


    //NOTICE: constant b for z{core-active}


    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"aphh", "aPhH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"caphh", "caPhH"});

    temp1["udkl"] = V["udkl"] * Eeps2_p["klud"];
    temp1["uDkL"] = V["uDkL"] * Eeps2_p["kLuD"];
    temp2["mwbij"] = V["mbij"] * Eeps2_m1["ijwb"];
    temp2["mwBiJ"] = V["mBiJ"] * Eeps2_m1["iJwB"];
    Z_b["mw"] += 0.25 * temp1["udkl"] * temp2["mwbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["wu"] * Eta1["bd"];
    Z_b["mw"] += 0.50 * temp1["uDkL"] * temp2["mwBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["wu"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["udkl"] = V["udkl"] * Eeps2_m1["klud"];
    temp1["uDkL"] = V["uDkL"] * Eeps2_m1["kLuD"];
    temp2["mwbij"] = V["mbij"] * Eeps2_p["ijwb"];
    temp2["mwBiJ"] = V["mBiJ"] * Eeps2_p["iJwB"];
    Z_b["mw"] += 0.25 * temp1["udkl"] * temp2["mwbij"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["wu"] * Eta1["bd"];
    Z_b["mw"] += 0.50 * temp1["uDkL"] * temp2["mwBiJ"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["wu"] * Eta1["BD"];

    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppah", "pPaH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppcah", "pPcaH"});

    temp1["cdul"] = V["cdul"] * Eeps2_p["ulcd"];
    temp1["cDuL"] = V["cDuL"] * Eeps2_p["uLcD"];
    temp2["abmwj"] = V["abmj"] * Eeps2_m1["wjab"];
    temp2["aBmwJ"] = V["aBmJ"] * Eeps2_m1["wJaB"];
    Z_b["mw"] += 0.25 * temp1["cdul"] * temp2["abmwj"] * Gamma1["uw"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["mw"] += 0.50 * temp1["cDuL"] * temp2["aBmwJ"] * Gamma1["uw"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdul"] = V["cdul"] * Eeps2_m1["ulcd"];
    temp1["cDuL"] = V["cDuL"] * Eeps2_m1["uLcD"];
    temp2["abmwj"] = V["abmj"] * Eeps2_p["wjab"];
    temp2["aBmwJ"] = V["aBmJ"] * Eeps2_p["wJaB"];
    Z_b["mw"] += 0.25 * temp1["cdul"] * temp2["abmwj"] * Gamma1["uw"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["mw"] += 0.50 * temp1["cDuL"] * temp2["aBmwJ"] * Gamma1["uw"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppach", "pPacH"});

    temp1["cdml"] = V["cdml"] * Eeps2_p["mlcd"];
    temp1["cDmL"] = V["cDmL"] * Eeps2_p["mLcD"];
    temp2["abwmj"] = V["abwj"] * Eeps2_m1["mjab"];
    temp2["aBwmJ"] = V["aBwJ"] * Eeps2_m1["mJaB"];
    Z_b["mw"] -= 0.25 * temp1["cdml"] * temp2["abwmj"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["mw"] -= 0.50 * temp1["cDmL"] * temp2["aBwmJ"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();
    temp2.zero();

    temp1["cdml"] = V["cdml"] * Eeps2_m1["mlcd"];
    temp1["cDmL"] = V["cDmL"] * Eeps2_m1["mLcD"];
    temp2["abwmj"] = V["abwj"] * Eeps2_p["mjab"];
    temp2["aBwmJ"] = V["aBwJ"] * Eeps2_p["mJaB"];
    Z_b["mw"] -= 0.25 * temp1["cdml"] * temp2["abwmj"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    Z_b["mw"] -= 0.50 * temp1["cDmL"] * temp2["aBwmJ"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    Z_b["mw"] += Z["m1,n1"] * V["n1,v,m1,m"] * Gamma1["wv"];
    Z_b["mw"] += Z["M1,N1"] * V["v,N1,m,M1"] * Gamma1["wv"];

    Z_b["mw"] += Z["e1,f1"] * V["f1,v,e1,m"] * Gamma1["wv"];
    Z_b["mw"] += Z["E1,F1"] * V["v,F1,m,E1"] * Gamma1["wv"];

    Z_b["mw"] -= Z["m,n1"] * F["n1,w"];

    Z_b["mw"] -= Z["m1,n1"] * V["n1,w,m1,m"];
    Z_b["mw"] -= Z["M1,N1"] * V["w,N1,m,M1"];

    Z_b["mw"] -= Z["e1,f"] * V["f,w,e1,m"];
    Z_b["mw"] -= Z["E1,F"] * V["w,F,m,E1"];

    Z_b["wm"] = Z_b["mw"];

    // For test use
    //Z_b.print();

    BlockedTensor Zold = BTF_->build(CoreTensor, "Old Z Matrix", spin_cases({"gg"}));
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


void DSRG_MRPT2::math_test() {

    // int N=5;
    // int NRHS=1, LDA=N,LDB=N;
    // int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    // int ipiv[5];

    // std::vector<double> a{
    //     6.80, -2.11,  5.66,  5.97,  8.23,
    //    -6.05, -3.30,  5.36, -4.44,  1.08,
    //    -0.45,  2.58, -2.70,  0.27,  9.04,
    //     8.32,  2.71,  4.35, -7.17,  2.14,
    //    -9.67, -5.14, -7.26,  6.08, -6.87
    // };

    // std::vector<double> b{
    //     4.02,  6.19, -8.22, -7.57, -3.03
    // };

    // int i, j;
    // for( i = 0; i < lda; i++ ) {
    //         for( j = 0; j < lda; j++ ) outfile->Printf( " %6.2f", a[i+j*lda] );
    //         outfile->Printf( "\n" );
    // }

    // outfile->Printf( "\n\n" );


    // for( i = 0; i < lda; i++ ) {
    //         for( j = 0; j < NRHS; j++ ) outfile->Printf( " %6.2f", b[i+j*lda] );
    //         outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\n\n" );

    // C_DGESV( n, nrhs, &a[0], lda, ipiv, &b[0], ldb);


    // for( i = 0; i < lda; i++ ) {
    //         for( j = 0; j < NRHS; j++ ) outfile->Printf( " %6.2f", b[i+j*lda] );
    //         outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\n\n" );

    // for( i = 0; i < lda; i++ ) {
    //         outfile->Printf( " %d", ipiv[i] );
    // }
    // outfile->Printf( "\n\n" );

   

    // // NOTICE:CORE-VIRTUAL
    // int dim = nvirt_ * ncore_;
    // int N=dim;
    // int NRHS=1, LDA=N,LDB=N;
    // int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    // std::vector<int> ipiv(N);

    // BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vcvc","vcVC"});
    // temp1["e,m,e1,m1"] -= V["m1,m,e1,e"];
    // temp1["e,m,e1,m1"] -= V["m1,e,e1,m"];
    // temp1["e,m,e1,m1"] += Delta1["m1,e1"] * I["e1,e"] * I["m1,m"];

    // temp1["e,m,E1,M1"] -= V["m,M1,e,E1"];
    // temp1["e,m,E1,M1"] -= V["e,M1,m,E1"];


    // std::vector<double> at(nvirt_ * ncore_ * nvirt_ * ncore_);
    // std::vector<double> bt(nvirt_ * ncore_);

    // for (const std::string& block : {"vc"}) {
    //     (Z_b.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
    //         bt[i[0] * ncore_ + i[1]] = value;
    //     });
    // } 

    // for (const std::string& block : {"vcvc","vcVC"}) {
    //     (temp1.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
    //         int index = (i[0] * ncore_ + i[1]) * dim + (i[2] * ncore_ + i[3]);
    //         at[index] += value;
    //     });
    // } 


    // for(int i = 0; i < lda; i++ ) {
    //         for(int j = 0; j < NRHS; j++ ) outfile->Printf( " %.7f", bt[i+j*lda] );
    //         outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\n\n" );

    // C_DGESV( n, nrhs, &at[0], lda, &ipiv[0], &bt[0], ldb);

    // for(int i = 0; i < nvirt_; i++ ) {
    //         for(int j = 0; j < ncore_; j++ ) outfile->Printf( " %.10f", bt[i*ncore_+j] );
    //         outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\n\n" );


    //NOTICE:ACTIVE-ACTIVE
    // int row = na_, col = na_-1;
    // int dim = row * col;
    // int N=dim;
    // int NRHS=1, LDA=N,LDB=N;
    // int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    // std::vector<int> ipiv(N);

    // BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"aaaa","aaAA"});

    // temp1["z,w,u1,v1"] -= V["v1,v,u1,w"] * Gamma1["zv"];
    // temp1["z,w,U1,V1"] -= V["v,V1,w,U1"] * Gamma1["zv"];
    // temp1["z,w,u1,v1"] += V["v1,v,u1,z"] * Gamma1["wv"];
    // temp1["z,w,U1,V1"] += V["v,V1,z,U1"] * Gamma1["wv"];
    // temp1["z,w,u1,v1"] += Delta1["z,w"] * I["u1,z"] * I["v1,w"];



    // std::vector<double> at(na_ * (na_-1) * na_ * (na_-1));
    // std::vector<double> bt(na_ * (na_-1));

    // BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"aa"});
    // BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", {"aa","AA"});
    // temp3["uv"] += Z["uv"] * I["uv"];
    // temp3["UV"] += Z["UV"] * I["UV"];
    // temp2["zw"] += Z_b["wz"];
    // temp2["zw"] += temp3["v1,u1"] * V["u1,v,v1,w"] * Gamma1["zv"];
    // temp2["zw"] += temp3["V1,U1"] * V["v,U1,w,V1"] * Gamma1["zv"];
    // temp2["zw"] -= temp3["v1,u1"] * V["u1,v,v1,z"] * Gamma1["wv"];
    // temp2["zw"] -= temp3["V1,U1"] * V["v,U1,z,V1"] * Gamma1["wv"];


    // for (const std::string& block : {"aa"}) {
    //     (temp2.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
    //         if (i[0]>i[1]){
    //             int index = i[0] * col + i[1];
    //             bt.at(index) = value;
    //         }
    //         else if (i[0]<i[1]){
    //             int index = i[0] * col + i[1] - 1;
    //             bt.at(index) = value;
    //         }
    //     });
    // } 

    // for (const std::string& block : {"aaaa","aaAA"}) {
    //     (temp1.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
    //         if (i[0]>i[1] && i[2]>i[3]){
    //             int index = (i[0] * col + i[1]) * dim + (i[2] * col + i[3]);
    //             at.at(index) += value;
    //         }
    //         else if (i[0]<i[1] && i[2]>i[3]){
    //             int index = (i[0] * col + i[1]-1) * dim + (i[2] * col + i[3]);
    //             at.at(index) += value;
    //         }
    //         else if (i[0]>i[1] && i[2]<i[3]){
    //             int index = (i[0] * col + i[1]) * dim + (i[2] * col + i[3]-1);
    //             at.at(index) += value;
    //         }
    //         else if (i[0]<i[1] && i[2]<i[3]){
    //             int index = (i[0] * col + i[1]-1) * dim + (i[2] * col + i[3]-1);
    //             at.at(index) += value;
    //         }          
    //     });
    // } 

    // outfile->Printf( "\n\n" );
    // for(int i = 0; i < na_; i++ ) {
    //         for(int j = 0; j < na_-1; j++ ) outfile->Printf( " %.10f", bt[i*(na_-1)+j] );
    //         outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\n\n" );

    // C_DGESV( n, nrhs, &at[0], lda, &ipiv[0], &bt[0], ldb);

    // for(int i = 0; i < na_; i++ ) {
    //         for(int j = 0; j < na_-1; j++ ) outfile->Printf( " %.10f", bt[i*(na_-1)+j] );
    //         outfile->Printf( "\n" );
    // }
    // outfile->Printf( "\n\n" );




    //NOTICE:All
    int dim_vc = nvirt_ * ncore_,
        dim_ca = ncore_ * na_,
        dim_va = nvirt_ * na_,
        dim_aa = na_ * (na_ - 1); 
    int dim = dim_vc + dim_ca + dim_va + dim_aa;
    int N=dim;
    int NRHS=1, LDA=N,LDB=N;
    int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    std::vector<int> ipiv(N);

    std::vector<double> A(dim * dim);
    std::vector<double> b(dim);

    std::map<string,int> preidx = {
        {"vc", 0}, {"VC", 0}, {"ca", dim_vc}, {"CA", dim_vc},
        {"va", dim_vc + dim_ca}, {"VA", dim_vc + dim_ca},
        {"aa", dim_vc + dim_ca + dim_va}, {"AA", dim_vc + dim_ca + dim_va} 
    };

    std::map<string,int> block_dim = {
        {"vc", ncore_}, {"VC", ncore_}, {"ca", na_}, {"CA", na_},
        {"va", na_}, {"VA", na_}, {"aa", na_-1}, {"AA", na_-1} 
    };

    //TODO:b

    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"gg"});
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", {"aa","AA"});

    temp3["uv"] += Z["uv"] * I["uv"];
    temp3["UV"] += Z["UV"] * I["UV"];

    //NOTICE:VIRTUAL-CORE

    temp2["em"] += Z_b["em"];
    temp2["em"] += temp3["uv"] * V["veum"];
    temp2["em"] += temp3["UV"] * V["eVmU"];

    //NOTICE:CORE-ACTIVE

    temp2["mw"] += Z_b["mw"];
    temp2["mw"] += temp3["u1,v1"] * V["v1,v,u1,m"] * Gamma1["w,v"];
    temp2["mw"] += temp3["U1,V1"] * V["v,V1,m,U1"] * Gamma1["w,v"];
    temp2["mw"] += temp3["wv"] * F["vm"];
    temp2["mw"] -= temp3["uv"] * V["vwum"];
    temp2["mw"] -= temp3["UV"] * V["wVmU"];

    //NOTICE:VIRTUAL-ACTIVE

    temp2["ew"] += Z_b["ew"];
    temp2["ew"] += temp3["u1,v1"] * V["v1,v,u1,e"] * Gamma1["wv"];
    temp2["ew"] += temp3["U1,V1"] * V["v,V1,e,U1"] * Gamma1["wv"];
    temp2["ew"] += temp3["wv"] * F["ve"];

    //NOTICE:ACTIVE-ACTIVE

    temp2["wz"] += Z_b["wz"];
    temp2["wz"] += temp3["v1,u1"] * V["u1,v,v1,w"] * Gamma1["zv"];
    temp2["wz"] += temp3["V1,U1"] * V["v,U1,w,V1"] * Gamma1["zv"];
    temp2["wz"] -= temp3["v1,u1"] * V["u1,v,v1,z"] * Gamma1["wv"];
    temp2["wz"] -= temp3["V1,U1"] * V["v,U1,z,V1"] * Gamma1["wv"];

    for (const std::string& block : {"vc", "ca", "va", "aa"}) {
        (temp2.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = preidx[block] + i[0] * block_dim[block] + i[1];
            if (block == "aa" && i[0] < i[1]) { --index;}
            if (block != "aa" || i[0] != i[1]) { b.at(index) = value;}            
        });
    } 


    // Print b

    outfile->Printf( "\nVC\n" );
    for(int i = 0; i < nvirt_; i++ ) {
        for(int j = 0; j < ncore_; j++ ) outfile->Printf( " %.10f ", b[i*ncore_+j]);
        outfile->Printf( "\n" );
    }
    outfile->Printf( "\nCA\n" );
    for(int i = 0; i < ncore_; i++ ) {
        int preidx = dim_vc;
        for(int j = 0; j < na_; j++ ) outfile->Printf( " %.10f ", b[preidx + i*na_+j]);
        outfile->Printf( "\n" );
    }
    outfile->Printf( "\nVA\n" );
    for(int i = 0; i < nvirt_; i++ ) {
        int preidx = dim_vc + dim_ca;
        for(int j = 0; j < na_; j++ ) outfile->Printf( " %.10f ", b[preidx + i*na_+j]);
        outfile->Printf( "\n" );
    }
    outfile->Printf( "\nAA\n" );
    for(int i = 0; i < na_; i++ ) {
        int preidx = dim_vc + dim_ca + dim_va;
        for(int j = 0; j < na_ -1; j++ ) outfile->Printf( " %.10f ", b[preidx + i*(na_-1)+j]);
        outfile->Printf( "\n" );
    }

    Z_b.print();

    //TODO:A
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"gggg","ggGG"});

    //NOTICE:VIRTUAL-CORE

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

    //NOTICE:CORE-ACTIVE

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

    temp1["m,w,e1,u"] -= Gamma1["uv"] * V["v,m,e1,w"];
    temp1["m,w,E1,U"] -= Gamma1["UV"] * V["m,V,w,E1"];
    temp1["m,w,e1,u"] -= Gamma1["uv"] * V["v,w,e1,m"];
    temp1["m,w,E1,U"] -= Gamma1["UV"] * V["w,V,m,E1"];

    temp1["m,w,u,v"] -= V["vwum"];
    temp1["m,w,U,V"] -= V["wVmU"];

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


    //NOTICE:VIRTUAL-ACTIVE

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
    temp1["e,w,e1,u"] -= V["y,X,e,E1"] * Gamma2["w,U,y,X"];
    temp1["e,w,e1,u"] -= V["e,y,e1,v"] * Gamma2["u,v,w,y"];
    temp1["e,w,e1,u"] -= V["e,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp1["e,w,e1,u"] -= V["e,Y,v,E1"] * Gamma2["v,U,w,Y"];


    //NOTICE:ACTIVE-ACTIVE

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





    // std::vector<double> at(nvirt_ * ncore_ * nvirt_ * ncore_);
    // std::vector<double> bt(nvirt_ * ncore_);

    // for (const std::string& block : {"vc"}) {
    //     (Z_b.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
    //         bt[i[0] * ncore_ + i[1]] = value;
    //     });
    // } 

    // for (const std::string& block : {"vcvc","vcVC"}) {
    //     (temp1.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
    //         int index = (i[0] * ncore_ + i[1]) * dim + (i[2] * ncore_ + i[3]);
    //         at[index] += value;
    //     });
    // } 

    // std::map<string,int> block_dim = {
    //     {"vc", ncore_}, {"VC", ncore_}, {"ca", na_}, {"CA", na_},
    //     {"va", na_}, {"VA", na_}, {"aa", na_}, {"AA", na_} 
    // };


    for (const std::string& row : {"vc","ca","va","aa"}) {
        int idx1 = block_dim[row];

        for (const std::string& col : {"vc","VC","ca","CA","va","VA","aa","AA"}) {
            int idx2 = block_dim[col];
            (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = (i[0] * idx1 + i[1]) * 2 + (i[2] * idx2 + i[3]);
                A[index] += value;
            });
        }
    } 






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


SharedMatrix DSRG_MRPT2::compute_gradient() {


	// TODO: compute the DSRG_MRPT2 gradient 
    print_method_banner({"DSRG-MRPT2 Gradient", "Shuhe Wang"});
    set_all_variables();
    set_multiplier();

    //NOTICE Just for test
    math_test();

    write_lagrangian();
    write_1rdm_spin_dependent();
    write_2rdm_spin_dependent();
    tpdm_backtransform();

    //NOTICE Just for test
    // compute_test_energy();


    outfile->Printf("\n    Computing Gradient .............................. Done\n");


    return std::make_shared<Matrix>("nullptr", 0, 0);
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

    //L->print();

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


    D1->print();

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
                if (m != m1) {
                    d2aa.write_value(n, m, m1, m1, 0.5 * z_a, 0, "NULL", 0);
                    d2bb.write_value(n, m, m1, m1, 0.5 * z_b, 0, "NULL", 0);
                    d2aa.write_value(n, m1, m1, m, -0.5 * z_a, 0, "NULL", 0);
                    d2bb.write_value(n, m1, m1, m, -0.5 * z_b, 0, "NULL", 0);
                }
                
                d2ab.write_value(n, m, m1, m1, (z_a + z_b), 0, "NULL", 0);
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
            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(v, u, m1, m1, 0.5 * z_a, 0, "NULL", 0);
                d2bb.write_value(v, u, m1, m1, 0.5 * z_b, 0, "NULL", 0);
                d2aa.write_value(v, m1, m1, u, -0.5 * z_a, 0, "NULL", 0);
                d2bb.write_value(v, m1, m1, u, -0.5 * z_b, 0, "NULL", 0);
                
                d2ab.write_value(v, u, m1, m1, (z_a + z_b), 0, "NULL", 0);
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

    temp1["cdkl"] = V["cdkl"] * Eeps2_p["klcd"];
    temp1["CDKL"] = V["CDKL"] * Eeps2_p["KLCD"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_p["kLcD"];
    temp["abij"] += 0.25 * temp1["cdkl"] * Eeps2_m1["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    temp["ABIJ"] += 0.25 * temp1["CDKL"] * Eeps2_m1["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    temp["aBiJ"] += 0.25 * temp1["cDkL"] * Eeps2_m1["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    temp1.zero();

    temp1["cdkl"] = V["cdkl"] * Eeps2_m1["klcd"];
    temp1["CDKL"] = V["CDKL"] * Eeps2_m1["KLCD"];
    temp1["cDkL"] = V["cDkL"] * Eeps2_m1["kLcD"];
    temp["abij"] += 0.25 * temp1["cdkl"] * Eeps2_p["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
    temp["ABIJ"] += 0.25 * temp1["CDKL"] * Eeps2_p["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
    temp["aBiJ"] += 0.25 * temp1["cDkL"] * Eeps2_p["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];

    temp["xynv"] -= Z["un"] * Gamma2["uvxy"]; 
    temp["XYNV"] -= Z["UN"] * Gamma2["UVXY"]; 
    temp["xYnV"] -= Z["un"] * Gamma2["uVxY"]; 

    temp["evxy"] += Z["eu"] * Gamma2["uvxy"];
    temp["EVXY"] += Z["EU"] * Gamma2["UVXY"];
    temp["eVxY"] += Z["eu"] * Gamma2["uVxY"];

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



}

