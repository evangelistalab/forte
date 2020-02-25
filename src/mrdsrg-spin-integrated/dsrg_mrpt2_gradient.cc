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
    for(size_t i=0; i<nmo_; i++){
outfile->Printf("\n%d   %.6f",i, Fa_[i]);
    }
    
}


void DSRG_MRPT2::set_z() {
    set_z_cc();  
    set_z_vv();
    set_z_aa_diag();
    iter_z();
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


    // //NOTICE: virtual-virtual alpha-alpha (as a backup)
    // BlockedTensor zef = BTF_->build(CoreTensor, "z{ef} normal", {"vv", "VV"});
    // temp1.zero();
    // temp2.zero();

    // temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
    // temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
    // temp2["fbij"] = temp1["fdkl"] * Eeps2_m1["ijfb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    // temp2["fBiJ"] = temp1["fDkL"] * Eeps2_m1["iJfB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    // zef["ef"] += 0.25 * V["ebij"] * temp2["fbij"];
    // zef["ef"] += 0.50 * V["eBiJ"] * temp2["fBiJ"];


    // temp1.zero();
    // temp2.zero();
    // temp1["fdkl"] = V["fdkl"] * Eeps2_m1["klfd"];
    // temp1["fDkL"] = V["fDkL"] * Eeps2_m1["kLfD"];
    // temp2["fbij"] = temp1["fdkl"] * Eeps2_p["ijfb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    // temp2["fBiJ"] = temp1["fDkL"] * Eeps2_p["iJfB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    // zef["ef"] += 0.25 * V["ebij"] * temp2["fbij"]; 
    // zef["ef"] += 0.50 * V["eBiJ"] * temp2["fBiJ"];

    // temp1.zero();
    // temp2.zero();
    // temp1["edkl"] = V["edkl"] * Eeps2_p["kled"];
    // temp1["eDkL"] = V["eDkL"] * Eeps2_p["kLeD"];
    // temp2["ebij"] = temp1["edkl"] * Eeps2_m1["ijeb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    // temp2["eBiJ"] = temp1["eDkL"] * Eeps2_m1["iJeB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"]; 
    // zef["ef"] -= 0.25 * V["fbij"] * temp2["ebij"];
    // zef["ef"] -= 0.50 * V["fBiJ"] * temp2["eBiJ"];

    // temp1.zero();
    // temp2.zero();
    // temp1["edkl"] = V["edkl"] * Eeps2_m1["kled"];
    // temp1["eDkL"] = V["eDkL"] * Eeps2_m1["kLeD"];
    // temp2["ebij"] = temp1["edkl"] * Eeps2_p["ijeb"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
    // temp2["eBiJ"] = temp1["eDkL"] * Eeps2_p["iJeB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
    // zef["ef"] -= 0.25 * V["fbij"] * temp2["ebij"];
    // zef["ef"] -= 0.50 * V["fBiJ"] * temp2["eBiJ"];
    // temp1.zero();
    // temp2.zero();




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
    int maxiter = 50;
    double convergence = 1e-6;

    //TODO: beta-beta part not done yet

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
    temp1.zero();
    temp2.zero();    

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
    temp1.zero();
    temp2.zero(); 

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





    //NOTICE: constant b for z{core-active}







    Z_b.print();


    BlockedTensor Zold = BTF_->build(CoreTensor, "Old Z Matrix", spin_cases({"gg"}));
    while (iter <= maxiter) {
        Zold["pq"] = Z["pq"];

        compute_z_cv();
        compute_z_av();
        compute_z_ca();
        compute_z_aa();

        Zold["pq"] -= Z["pq"];


        double Znorm = Zold.norm();

        if (Znorm < convergence) {
            converged = true;
            break;
        }
        iter++;
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

    for (const std::string& block : {"vc"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = temp.block(block).data()[i[0] * ncore_ + i[1]] / dnt.block(block).data()[i[0] * ncore_ + i[1]];
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










void DSRG_MRPT2::compute_z_av() {}
void DSRG_MRPT2::compute_z_ca() {}



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
    temp["wz"] -= 2.0 * Z["n1,u"] * V["w,Y,n1,V"] * Gamma2["u,V,z,Y"];

    temp["wz"] += Z["e1,u"] * H["w,e1"] * Gamma1["uz"];
    temp["wz"] += Z["e1,u"] * V["w,m1,e1,m"] * Gamma1["uz"] * I["m1,m"];
    temp["wz"] += Z["e1,u"] * V["w,M1,e1,M"] * Gamma1["uz"] * I["M1,M"];
    temp["wz"] += 0.5 * Z["e1,u"] * V["x,y,e1,w"] * Gamma2["u,z,x,y"];
    temp["wz"] += Z["E1,U"] * V["y,X,w,E1"] * Gamma2["z,U,y,X"];
    temp["wz"] += Z["e1,u"] * V["w,y,e1,v"] * Gamma2["u,v,z,y"];
    temp["wz"] += 2.0 * Z["e1,u"] * V["w,Y,e1,V"] * Gamma2["u,V,z,Y"];

    temp["wz"] += Z["n1,u"] * H["z,n1"] * Gamma1["uw"];
    temp["wz"] += Z["n1,u"] * V["z,m1,n1,m"] * Gamma1["uw"] * I["m1,m"];
    temp["wz"] += Z["n1,u"] * V["z,M1,n1,M"] * Gamma1["uw"] * I["M1,M"];
    temp["wz"] += 0.5 * Z["n1,u"] * V["x,y,n1,z"] * Gamma2["u,w,x,y"];
    temp["wz"] += Z["N1,U"] * V["y,X,z,N1"] * Gamma2["w,U,y,X"];
    temp["wz"] += Z["n1,u"] * V["z,y,n1,v"] * Gamma2["u,v,w,y"];
    temp["wz"] += 2.0 * Z["n1,u"] * V["z,Y,n1,V"] * Gamma2["u,V,w,Y"];

    temp["wz"] -= Z["e1,u"] * H["z,e1"] * Gamma1["uw"];
    temp["wz"] -= Z["e1,u"] * V["z,m1,e1,m"] * Gamma1["uw"] * I["m1,m"];
    temp["wz"] -= Z["e1,u"] * V["z,M1,e1,M"] * Gamma1["uw"] * I["M1,M"];
    temp["wz"] -= 0.5 * Z["e1,u"] * V["x,y,e1,z"] * Gamma2["u,w,x,y"];
    temp["wz"] -= Z["E1,U"] * V["y,X,z,E1"] * Gamma2["w,U,y,X"];
    temp["wz"] -= Z["e1,u"] * V["z,y,e1,v"] * Gamma2["u,v,w,y"];
    temp["wz"] -= 2.0 * Z["e1,u"] * V["z,Y,e1,V"] * Gamma2["u,V,w,Y"];

    temp["wz"] += Z["u1,v1"] * V["v1,v,u1,w"] * Gamma1["zv"];
    temp["wz"] += Z["U1,V1"] * V["v,V1,w,U1"] * Gamma1["zv"];

    temp["wz"] -= Z["u1,v1"] * V["v1,v,u1,z"] * Gamma1["wv"];
    temp["wz"] -= Z["U1,V1"] * V["v,V1,z,U1"] * Gamma1["wv"];

    // move to the left side
    temp["wz"] -= Z["zw"] * V["w,v,z,u1"] * I["w,u1"] * Gamma1["zv"];
    temp["wz"] -= Z["zw"] * V["z,v,w,u1"] * I["w,u1"] * Gamma1["zv"];
    temp["wz"] += Z["zw"] * V["w,v,z,u1"] * I["z,u1"] * Gamma1["wv"];
    temp["wz"] += Z["zw"] * V["z,v,w,u1"] * I["z,u1"] * Gamma1["wv"];

    // Denominator
    BlockedTensor dnt = BTF_->build(CoreTensor, "temporal denominator 1", {"aa"});
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
            if (i[0] != i[1]) {
                value = Z.block("aa").data()[i[0] * na_ + i[1]];
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






















