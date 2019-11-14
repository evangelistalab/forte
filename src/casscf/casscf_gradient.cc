#include "casscf/casscf.h"
#include "casscf/backtransform_tpdm.h"
#include "ambit/tensor.h"
#include "ambit/blocked_tensor.h"

#include "integrals/integrals.h"
#include "base_classes/rdms.h"

#include "psi4/libfock/jk.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"
#include "psi4/psifiles.h"

#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "helpers/timer.h"
#include "sci/aci.h"
#include "fci/fci_solver.h"
#include "base_classes/active_space_solver.h"

#include "sci/fci_mo.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/orbitaloptimizer.h"
#include "orbital-helpers/semi_canonicalize.h"

#ifdef HAVE_CHEMPS2
#include "dmrg/dmrgsolver.h"
#endif
#include "psi4/libdiis/diisentry.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libmints/factory.h"


#include "psi4/psifiles.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

using namespace ambit;
using namespace psi;

namespace forte {


void CASSCF::set_ambit_space() {

/****************************/
/*                          */
/*  set up ambit MO space   */
/*                          */
/****************************/

    outfile->Printf("\n    Setting ambit MO space .......................... ");
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    core_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    virt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    acore_label_ = "c";
    aactv_label_ = "a";
    avirt_label_ = "v";
    bcore_label_ = "C";
    bactv_label_ = "A";
    bvirt_label_ = "V";

    // add Ambit index labels
    BTF_->add_mo_space(acore_label_, "mn", core_mos_, AlphaSpin);
    BTF_->add_mo_space(bcore_label_, "MN", core_mos_, BetaSpin);
    BTF_->add_mo_space(aactv_label_, "uvwxyz123", actv_mos_, AlphaSpin);
    BTF_->add_mo_space(bactv_label_, "UVWXYZ!@#", actv_mos_, BetaSpin);
    BTF_->add_mo_space(avirt_label_, "ef", virt_mos_, AlphaSpin);
    BTF_->add_mo_space(bvirt_label_, "EF", virt_mos_, BetaSpin);

    // define composite spaces
    BTF_->add_composite_mo_space("h", "ijkl", {acore_label_, aactv_label_});
    BTF_->add_composite_mo_space("H", "IJKL", {bcore_label_, bactv_label_});
    BTF_->add_composite_mo_space("p", "abcd", {aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("P", "ABCD", {bactv_label_, bvirt_label_});
    BTF_->add_composite_mo_space("g", "pqrsto456", {acore_label_, aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("G", "PQRSTO789", {bcore_label_, bactv_label_, bvirt_label_});

    outfile->Printf("Done");	
}



void CASSCF::init_density() {

    outfile->Printf("\n    Initializing density tensors ......... ");
    Gamma1_ = BTF_->build(CoreTensor, "Gamma1", spin_cases({"aa"}));
    Gamma2_ = BTF_->build(CoreTensor, "Gamma2", spin_cases({"aaaa"}));
    fill_density();
    outfile->Printf("Done");    
}

void CASSCF::fill_density() {
    // 1-body density 
    Gamma1_.block("aa")("pq") = cas_ref_.g1a()("pq");
    Gamma1_.block("AA")("pq") = cas_ref_.g1b()("pq");

    // 2-body density 
    Gamma2_.block("aaaa")("pqrs") = cas_ref_.g2aa()("pqrs");
    Gamma2_.block("aAaA")("pqrs") = cas_ref_.g2ab()("pqrs");
    Gamma2_.block("AAAA")("pqrs") = cas_ref_.g2bb()("pqrs");
}



void CASSCF::init_h() {

    outfile->Printf("\n    Building Hamiltonian ............................ ");
    H_ = BTF_->build(ambit::CoreTensor, "Hamiltonian", spin_cases({"gg"}));

    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->oei_a(i[0], i[1]);
        } 
        else {
            value = ints_->oei_b(i[0], i[1]);
        }
    });

    outfile->Printf("Done");
}


void CASSCF::init_v() {

    outfile->Printf("\n    Building electron repulsion integrals ............................ ");
    V_ = BTF_->build(ambit::CoreTensor, "Hamiltonian", spin_cases({"gggg"}));

    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        } 
        else if (spin[0] == AlphaSpin && spin[1] == BetaSpin) {
            value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
        }
        else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
            value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
        }
    });

    outfile->Printf("Done");
}



void CASSCF::init_fock() {

/*****************************/
/*                           */
/*  Initialize fock matrix   */
/*                           */
/*****************************/

    outfile->Printf("\n    Building Fock matrix ............................ ");

    SharedMatrix Da(new psi::Matrix("Da", nmo_, nmo_));
    SharedMatrix Db(new psi::Matrix("Db", nmo_, nmo_));
    auto L1a = tensor_to_matrix(cas_ref_.g1a(), na_);
    auto L1b = tensor_to_matrix(cas_ref_.g1b(), na_);

    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
    rdocc_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    actv_ = mo_space_info_->get_dimension("ACTIVE");

    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        // core block (diagonal)
        for (size_t i = 0; i < rdocc_[h]; ++i) {
            Da->set(offset + i, offset + i, 1.0);
            Db->set(offset + i, offset + i, 1.0);
        }

        offset += rdocc_[h];

        // active block
        for (size_t u = 0; u < actv_[h]; ++u) {
            for (size_t v = 0; v < actv_[h]; ++v) {
                Da->set(offset + u, offset + v, L1a->get(h, u, v));
                Db->set(offset + u, offset + v, L1b->get(h, u, v));
            }
        }

        offset += ncmopi_[h] - rdocc_[h];
    }

    ints_->make_fock_matrix(Da, Db);

    F_ = BTF_->build(ambit::CoreTensor, "Fock", spin_cases({"gg"}));

    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->get_fock_a(i[0], i[1]);
        } 
        else {
            value = ints_->get_fock_b(i[0], i[1]);
        }
    });    

    outfile->Printf("Done");

}


void CASSCF::set_lagrangian() {

    outfile->Printf("\n    Building Lagrangian ............................ ");
    W_ = BTF_->build(CoreTensor, "Lagrangian", spin_cases({"gg"}));
    set_lagrangian_1();
    set_lagrangian_2();
    outfile->Printf("Done");
}

void CASSCF::set_lagrangian_1() {

/*********************************************************************/
/*                                                                   */
/*  set up omega matrix entries of core-core and core-active blocks  */
/*                                                                   */
/*********************************************************************/

    W_["mp"] = F_["mp"];
    W_["MP"] = F_["MP"];

}

void CASSCF::set_lagrangian_2() {

/**********************************************/
/*                                            */
/*  compute omega of the active-active block  */
/*                                            */
/**********************************************/

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"gg"}));
    BlockedTensor I = BTF_->build(CoreTensor, "identity matrix", spin_cases({"gg"}));

    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& , double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });

    temp["vp"] = H_["vp"];
    temp["vp"] += V_["vmpn"] * I["mn"];
    temp["vp"] += V_["vMpN"] * I["MN"];
    W_["up"] -= temp["vp"] * Gamma1_["uv"];
    W_["up"] -= V_["xypv"] * Gamma2_["uvxy"];
    W_["up"] -= V_["xYpV"] * Gamma2_["uVxY"];
    W_["up"] -= V_["XypV"] * Gamma2_["uVXy"];

    temp["VP"] = H_["VP"];
    temp["VP"] += V_["VmPn"] * I["mn"];
    temp["VP"] += V_["VMPN"] * I["MN"];
    W_["UP"] -= temp["VP"] * Gamma1_["UV"];
    W_["UP"] -= V_["XYPV"] * Gamma2_["UVXY"];
    W_["UP"] -= V_["XyPv"] * Gamma2_["UvXy"];
    W_["UP"] -= V_["xYPv"] * Gamma2_["UvxY"];

}



// SharedMatrix CASSCF::convert_1rdm_so2mo(SharedMatrix D) {

//     SharedMatrix Dmo(new Matrix(D->name() + " MO", nmo_, nmo_));
//     for (int p = 0; p < nmo_; ++p) {
//         for (int q = 0; q < nmo_; ++q) {
//             double v1 = D->get(2 * p, 2 * q);
//             double v2 = D->get(2 * p + 1, 2 * q + 1);
//             Dmo->set(p, q, v1 + v2);
//         }
//     }
//     return Dmo;
// }












// void CASSCF::compute_Lagrangian() {
//     outfile->Printf("\n  Computing Lagrangian ... ");

//     SharedMatrix L(new Matrix("Lagrangian Matrix", nso_, nso_));
//     compute_Lagrangian_CX(L);
//     compute_Lagrangian_AA(L);


//     // convert spin-orbital Lagrangian to MO
//     SharedMatrix Lmo = convert_1rdm_so2mo(L);

//     // add reference part
//     for (int i = 0; i < nfrzc_ + ncore_; ++i) {
//         Lmo->add(i, i, 2.0 * epsilon_a_->get(i));
//     }

//     // transform to AO and pass to Lagrangian_
//     Lagrangian_ = Lmo->clone();
//     Lagrangian_->set_name("Lagrangian Matrix AO");
//     Lagrangian_->back_transform(Ca_);

//     outfile->Printf("Done.");
// }




void CASSCF::set_parameters() {
    init_density();
    fill_density();
    init_h();
    init_v();
    init_fock();
}






SharedMatrix CASSCF::compute_gradient() {

	set_ambit_space();
    set_parameters();

    // NEED to overwrite the Da_ and Db_ (here)
    // more codes required


    compute_lagrangian();
    write_2rdm_spin_dependent();


    return std::make_shared<Matrix>("nullptr", 0, 0);
}


void CASSCF::compute_lagrangian() {

    set_lagrangian();

    outfile->Printf("\n  Computing Lagrangian ... ");
    //backtransform

    outfile->Printf("Done");

}






void CASSCF::write_2rdm_spin_dependent() {

    outfile->Printf("\n  Writing 2RDM on disk");

    auto psio_ = _default_psio_lib_;
    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);

/*************************************************************************/
/*!!!  coefficients in d2aa and d2bb need to multiply additional 1/2  !!!*/
/*************************************************************************/
  
    std::vector<size_t> core_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_DOCC");
    std::vector<size_t> actv_all_ = mo_space_info_->get_absolute_mo("ACTIVE");
    std::vector<size_t> virt_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_UOCC");

    for(size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        for(size_t j = 0; j < size_c; ++j) {

            auto m = core_all_[i];
            auto n = core_all_[j];

            d2aa.write_value(m, m, n, n, 0.25, 0, "NULL", 0);
            d2ab.write_value(m, m, n, n, 0.50, 0, "NULL", 0);
            d2bb.write_value(m, m, n, n, 0.25, 0, "NULL", 0);

            d2aa.write_value(m, n, n, m, -0.25, 0, "NULL", 0);
            d2ab.write_value(m, n, n, m, -0.50, 0, "NULL", 0);
            d2bb.write_value(m, n, n, m, -0.25, 0, "NULL", 0);

        }
    }



    for(size_t i = 0, size_c = core_all_.size(), size_a = actv_all_.size(); i < size_a; ++i) {
        for(size_t j = 0; j < size_a; ++j) {

            auto idx = actv_mos_[i] * na_ + actv_mos_[j];
            auto u = actv_all_[i];
            auto v = actv_all_[j];
            
            auto gamma_a = Gamma1_.block("aa").data()[idx];
            auto gamma_b = Gamma1_.block("AA").data()[idx];

            for(size_t k = 0; k < size_c; ++k) {

                auto m = core_all_[k];

                d2aa.write_value(v, u, m, m, 0.5 * gamma_a, 0, "NULL", 0);
                d2bb.write_value(v, u, m, m, 0.5 * gamma_b, 0, "NULL", 0);

                d2aa.write_value(v, m, m, u, -0.5 * gamma_a, 0, "NULL", 0);
                d2bb.write_value(v, m, m, u, -0.5 * gamma_b, 0, "NULL", 0);

                /// this need to be checked, inconsistency exists
                d2ab.write_value(v, u, m, m, (gamma_a + gamma_b), 0, "NULL", 0);
            }
        }
    }

    for(size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        for(size_t j = 0; j < size_a; ++j) {
            for(size_t k = 0; k < size_a; ++k) {
                for(size_t l = 0; l < size_a; ++l) {

                    auto u = actv_all_[i];
                    auto v = actv_all_[j];
                    auto x = actv_all_[k];
                    auto y = actv_all_[l];
                    auto idx = u * na_ * na_ * na_ + v * na_ * na_ + x * na_ + y;
                    auto gamma_aa = Gamma2_.block("aaaa").data()[idx];
                    auto gamma_bb = Gamma1_.block("AAAA").data()[idx];
                    auto gamma_ab = Gamma1_.block("aAaA").data()[idx];

                    
                    d2aa.write_value(x, u, y, v, 0.25 * gamma_aa, 0, "NULL", 0);
                    d2bb.write_value(x, u, y, v, 0.25 * gamma_bb, 0, "NULL", 0);

                    d2ab.write_value(x, u, y, v, gamma_ab, 0, "NULL", 0);
                }
            }
        }
    }
  
    outfile->Printf("\n Done");

    d2aa.flush(1);
    d2bb.flush(1);
    d2ab.flush(1);

    d2aa.set_keep_flag(1);
    d2bb.set_keep_flag(1);
    d2ab.set_keep_flag(1);

    d2aa.close();
    d2bb.close();
    d2ab.close();
}


}



