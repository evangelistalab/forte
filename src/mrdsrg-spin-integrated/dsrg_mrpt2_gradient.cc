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
    core_mos_relative = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    actv_mos_relative = mo_space_info_->get_relative_mo("ACTIVE");
    irrep_vec = mo_space_info_->get_dimension("ALL");


    na_ = mo_space_info_->size("ACTIVE");
    ncore_ = mo_space_info_->size("RESTRICTED_DOCC");
    nvirt_ = mo_space_info_->size("RESTRICTED_UOCC");

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
   
    Delta1 = BTF_->build(CoreTensor, "Delta1", spin_cases({"hp"}));
    Delta2 = BTF_->build(CoreTensor, "Delta2", spin_cases({"hhpp"}));

    //NOTICE: The dimension may be further reduced.
    Z = BTF_->build(CoreTensor, "Z Matrix", spin_cases({"gg"}));

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
}


void DSRG_MRPT2::set_z() {
    set_z_cc();  
    set_z_vv();
    set_z_aa();
}


void DSRG_MRPT2::set_w() {

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

    for (const std::string& block : {"cc", "CC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block == "cc" && i[0] == i[1]) {              
                value = 0.5 * val1.block("c").data()[i[0]];
            }
            else if (block == "CC" && i[0] == i[1]) {
                value = 0.5 * val1.block("C").data()[i[0]];
            }

            else if (block == "cc") {

            }
            else if (block == "CC") {

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


    //NOTICE: virtual-virtual alpha-alpha
    BlockedTensor zef = BTF_->build(CoreTensor, "z{ef} normal", {"vv", "VV"});
    BlockedTensor I = BTF_->build(CoreTensor, "I", {"vv", "VV"});


    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });

    temp1.zero();
    temp2.zero();

    temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
    temp2["fbij"] = temp1["fdkl"] * Eeps2_m1["ijfb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    temp2["fBiJ"] = temp1["fDkL"] * Eeps2_m1["iJfB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    zef["ef"] += 0.25 * V["ebij"] * temp2["fbij"];
    zef["ef"] += 0.50 * V["eBiJ"] * temp2["fBiJ"];


    temp1.zero();
    temp2.zero();
    temp1["fdkl"] = V["fdkl"] * Eeps2_m1["klfd"];
    temp1["fDkL"] = V["fDkL"] * Eeps2_m1["kLfD"];
    temp2["fbij"] = temp1["fdkl"] * Eeps2_p["ijfb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    temp2["fBiJ"] = temp1["fDkL"] * Eeps2_p["iJfB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    zef["ef"] += 0.25 * V["ebij"] * temp2["fbij"]; 
    zef["ef"] += 0.50 * V["eBiJ"] * temp2["fBiJ"];

    temp1.zero();
    temp2.zero();
    temp1["edkl"] = V["edkl"] * Eeps2_p["kled"];
    temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
    temp2["ebij"] = temp1["edkl"] * Eeps2_m1["ijeb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    temp2["eBiJ"] = temp1["eDkL"] * Eeps2_m1["iJeB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"]; 
    zef["ef"] -= 0.25 * V["fbij"] * temp2["ebij"];
    zef["ef"] -= 0.50 * V["fBiJ"] * temp2["eBiJ"];

    temp1.zero();
    temp2.zero();
    temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"];
    temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
    temp2["ebij"] = temp1["edkl"] * Eeps2_p["ijeb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    temp2["eBiJ"] = temp1["eDkL"] * Eeps2_p["iJeB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    zef["ef"] -= 0.25 * V["fbij"] * temp2["ebij"];
    zef["ef"] -= 0.50 * V["fBiJ"] * temp2["eBiJ"];
    temp1.zero();
    temp2.zero();

    auto xxx1 = BTF_->build(CoreTensor, "xxx1", spin_cases({"pp"}));
    xxx1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = 1.0/(Fa_[i[0]] - Fa_[i[1]]);}
            else { value = 1.0/(Fb_[i[0]] - Fb_[i[1]]);}
        }
    );

    BlockedTensor zeft = BTF_->build(CoreTensor, "z{ef} normal", {"vv", "VV"});
    zeft["ef"] = zef["ef"] * xxx1["fe"];


    for (const std::string& block : {"vv", "VV"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block == "vv" && i[0] == i[1]) { 
                value = 0.5 * val2.block("v").data()[i[0]];
            }
            else if (block == "VV" && i[0] == i[1]) {      
                value = 0.5 * val2.block("V").data()[i[0]];
            }

            else if (block == "vv") {
                // value = zeft.block("vv").data()[i[0] * nvirt_ + i[1]];
                value = zef.block("vv").data()[i[0] * nvirt_ + i[1]]/(Fa_[i[1]] - Fa_[i[0]]);
            }
            else if (block == "VV") {
                value = zeft.block("vv").data()[i[0] * nvirt_ + i[1]];
            }
        });
    } 



    // temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
    // temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
    // zef["ef"] += 0.25 * V["ebij"] * temp1["fdkl"] * Eeps2_m1["ijfb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    // zef["ef"] += 0.50 * V["eBiJ"] * temp1["fDkL"] * Eeps2_m1["iJfB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];


    // temp1.zero();
    // temp1["fdkl"] = V["fdkl"] * Eeps2_m1["klfd"];
    // temp1["fDkL"] = V["fDkL"] * Eeps2_m1["kLfD"];
    // zef["ef"] += 0.25 * V["ebij"] * temp1["fdkl"] * Eeps2_p["ijfb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    // zef["ef"] += 0.50 * V["eBiJ"] * temp1["fDkL"] * Eeps2_p["iJfB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];

    // temp1.zero();
    // temp1["edkl"] = V["edkl"] * Eeps2_p["kled"];
    // temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
    // zef["ef"] -= 0.25 * V["fbij"] * temp1["edkl"] * Eeps2_m1["ijeb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    // zef["ef"] -= 0.50 * V["fBiJ"] * temp1["eDkL"] * Eeps2_m1["iJeB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];

    // temp1.zero();
    // temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"];
    // temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
    // zef["ef"] -= 0.25 * V["fbij"] * temp1["edkl"] * Eeps2_p["ijeb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    // zef["ef"] -= 0.50 * V["fBiJ"] * temp1["eDkL"] * Eeps2_p["iJeB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    // temp1.zero();


}



void DSRG_MRPT2::set_z_aa() {

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

            else if (block == "aa") {

            }
            else if (block == "AA") {

            }
        });
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






void DSRG_MRPT2::set_lagrangian() {
	// TODO: set coefficients before the overlap integral

    // Create a temporal container and an identity matrix
    // BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"gg"}));
    // BlockedTensor I = BTF_->build(CoreTensor, "identity matrix", spin_cases({"gg"}));
    // I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
    //     value = (i[0] == i[1]) ? 1.0 : 0.0;
    // });

    // // Set core-core and core-active block entries of Lagrangian.
    // // Alpha.
    // W_["mp"] = F_["mp"];
    // // Beta.
    // W_["MP"] = F_["MP"];

    // // Set active-active block entries of Lagrangian.
    // // Alpha.
    // temp["vp"] = Hoei_["vp"];
    // temp["vp"] += V_["vmpn"] * I["mn"];
    // temp["vp"] += V_["vMpN"] * I["MN"];
    // W_["up"] += temp["vp"] * Gamma1_["uv"];
    // W_["up"] += 0.5 * V_["xypv"] * Gamma2_["uvxy"];
    // W_["up"] += V_["xYpV"] * Gamma2_["uVxY"];
    // // Beta.
    // temp["VP"] = Hoei_["VP"];
    // temp["VP"] += V_["mVnP"] * I["mn"];
    // temp["VP"] += V_["VMPN"] * I["MN"];
    // W_["UP"] += temp["VP"] * Gamma1_["UV"];
    // W_["UP"] += 0.5 * V_["XYPV"] * Gamma2_["UVXY"];
    // W_["UP"] += V_["yXvP"] * Gamma2_["vUyX"];
    // No need to set the rest symmetric blocks since they are 0
}

// It's not necessary to define set_tensor

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
    // write_lagrangian();
    // write_1rdm_spin_dependent();
    // write_2rdm_spin_dependent();
    // tpdm_backtransform();



    //NOTICE Just for test
    compute_test_energy();

    outfile->Printf("\n    Computing Gradient .............................. Done\n");


    return std::make_shared<Matrix>("nullptr", 0, 0);
}




void DSRG_MRPT2::write_lagrangian() {
	// TODO: write the Lagrangian
    outfile->Printf("\n    Writing Lagrangian .............................. ");

    set_lagrangian();


    outfile->Printf("Done");
}



void DSRG_MRPT2::write_1rdm_spin_dependent() {
	// TODO: write spin_dependent one-RDMs coefficients. 
    outfile->Printf("\n    Writing 1RDM Coefficients ....................... ");


    outfile->Printf("Done");
}


void DSRG_MRPT2::write_2rdm_spin_dependent() {
	// TODO: write spin_dependent two-RDMs coefficients using IWL
    outfile->Printf("\n    Writing 2RDM Coefficients ....................... ");

    auto psio_ = _default_psio_lib_;
    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);


	// TODO: write coefficients here

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






















