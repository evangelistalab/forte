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
    BTF_= std::make_shared<BlockedTensorFactory>();
    BTF_->add_mo_space(acore_label_, "mn$%", core_mos_, AlphaSpin);
    BTF_->add_mo_space(bcore_label_, "MN<>", core_mos_, BetaSpin);
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

    outfile->Printf("\n    Initializing density tensors .................... ");
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

    outfile->Printf("\n    Building electron repulsion integrals ........... ");
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

    F_ = BTF_->build(ambit::CoreTensor, "Fock", spin_cases({"gg"}));

    // SharedMatrix Da(new psi::Matrix("Da", nmo_, nmo_));
    // SharedMatrix Db(new psi::Matrix("Db", nmo_, nmo_));
    // auto L1a = tensor_to_matrix(cas_ref_.g1a(), na_);
    // auto L1b = tensor_to_matrix(cas_ref_.g1b(), na_);

    // ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
    // rdocc_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    // actv_ = mo_space_info_->get_dimension("ACTIVE");

    // for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
    //     // core block (diagonal)
    //     for (size_t i = 0; i < rdocc_[h]; ++i) {
    //         Da->set(offset + i, offset + i, 1.0);
    //         Db->set(offset + i, offset + i, 1.0);
    //     }

        // offset += rdocc_[h];
        // // active block
        // for (size_t u = 0; u < actv_[h]; ++u) {
        //     for (size_t v = 0; v < actv_[h]; ++v) {
        //         Da->set(offset + u, offset + v, L1a->get(u, v));
        //         Db->set(offset + u, offset + v, L1b->get(u, v));
        //     }
        // }

    //     offset += ncmopi_[h] - rdocc_[h];
    // }


    psi::SharedMatrix D1a(new psi::Matrix("D1a", nmo_, nmo_));
    psi::SharedMatrix D1b(new psi::Matrix("D1b", nmo_, nmo_));
    for (size_t m = 0, ncore = core_mos_.size(); m < ncore; m++) {
        D1a->set(core_mos_[m], core_mos_[m], 1.0);
        D1b->set(core_mos_[m], core_mos_[m], 1.0);
    }

    Gamma1_.block("aa").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1a->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });
    Gamma1_.block("AA").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1b->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });



    ints_->make_fock_matrix(D1a, D1b);

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

    outfile->Printf("\n    Building Lagrangian ............................. ");
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

    W_["mp"] -= F_["mp"];
    W_["MP"] -= F_["MP"];


    // BlockedTensor I = BTF_->build(CoreTensor, "identity matrix", spin_cases({"gg"}));

    // I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& , double& value) {
    //     value = (i[0] == i[1]) ? 1.0 : 0.0;
    // });

    // BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"gg"}));

    // W_["mp"] = -H_["mp"];
    // W_["mp"] -= V_["pnm$"] * I["n$"];
    // W_["mp"] -= V_["pNm<"] * I["N<"];
    // W_["mp"] -= V_["pvmu"] * Gamma1_["uv"];
    // W_["mp"] -= V_["pVmU"] * Gamma1_["UV"];



    // W_["MP"] = -H_["MP"];
    // W_["MP"] -= V_["nP$M"] * I["n$"];
    // W_["MP"] -= V_["PNM<"] * I["N<"];
    // W_["MP"] -= V_["vPuM"] * Gamma1_["uv"];
    // W_["MP"] -= V_["PVMU"] * Gamma1_["UV"];


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
    W_["up"] -= 0.5 * V_["xypv"] * Gamma2_["uvxy"];
    W_["up"] -= V_["xYpV"] * Gamma2_["uVxY"];
    // W_["up"] -= 0.5 * V_["XypV"] * Gamma2_["uVXy"];

    temp["VP"] = H_["VP"];
    temp["VP"] += V_["mVnP"] * I["mn"];
    temp["VP"] += V_["VMPN"] * I["MN"];
    W_["UP"] -= temp["VP"] * Gamma1_["UV"];
    W_["UP"] -= 0.5 * V_["XYPV"] * Gamma2_["UVXY"];
    // W_["UP"] -= 0.5 * V_["XyPv"] * Gamma2_["UvXy"];
    W_["UP"] -= V_["yXvP"] * Gamma2_["vUyX"];
    // W_["UP"] -= 0.5 * V_["xYPv"] * Gamma2_["UvxY"];

    //need to add symmetric parts !!!!!!!




    double energy = ints_->nuclear_repulsion_energy();

    energy += H_["mn"] * I["mn"];
    energy += H_["MN"] * I["MN"];

    energy += 0.5 * V_["mn$%"] * I["m$"] * I["n%"];
    energy += 0.5 * V_["MN<>"] * I["M<"] * I["N>"];
    energy += V_["mN$>"] * I["m$"] * I["N>"];

    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"gg"}));

    temp["vu"] = H_["vu"];
    temp["vu"] += V_["vmun"] * I["mn"];
    temp["vu"] += V_["vMuN"] * I["MN"];
    energy += temp["vu"] * Gamma1_["uv"];   
    
    temp["VU"] = H_["VU"];
    temp["VU"] += V_["mVnU"] * I["mn"];
    temp["VU"] += V_["VMUN"] * I["MN"];
    energy += temp["VU"] * Gamma1_["UV"];
    
    energy += 0.25 * V_["xyuv"] * Gamma2_["uvxy"];
    energy += 0.25 * V_["XYUV"] * Gamma2_["UVXY"];
    energy += V_["xYuV"] * Gamma2_["uVxY"];

    outfile->Printf("\n\n    My stupid energy = %.12f\n\n", energy);

}






void CASSCF::set_parameters() {
    init_density();
    init_h();
    init_v();
    init_fock();
}






SharedMatrix CASSCF::compute_gradient() {

	set_ambit_space();
    set_parameters();

    // NEED to overwrite the Da_ and Db_ (here)
    // more codes required

    compute_1rdm_coeff();
    compute_lagrangian();

    write_2rdm_spin_dependent();
    



    ints_->wfn()->set_reference_wavefunction(ints_->wfn());
    auto temp = ints_->wfn()->reference_wavefunction();
    if(!temp) outfile->Printf("XXXXXXXXXX");






    std::vector<std::shared_ptr<psi::MOSpace> > spaces;
    spaces.push_back(psi::MOSpace::all);
    std::shared_ptr<TPDMBackTransform> transform = std::shared_ptr<TPDMBackTransform>(
    new TPDMBackTransform(ints_->wfn(), spaces,
                IntegralTransform::TransformationType::Unrestricted, // Transformation type
                IntegralTransform::OutputType::DPDOnly,              // Output buffer
                IntegralTransform::MOOrdering::QTOrder,              // MO ordering
                IntegralTransform::FrozenOrbitals::None));           // Frozen orbitals?
    
    // transform->set_psio(ints_->wfn()->psio());
    transform->backtransform_density();
    transform.reset();
    outfile->Printf("\n    TPD Backtransformation .......................... Done");
    outfile->Printf("\n    Computing gradients ............................. Done\n");

    return std::make_shared<Matrix>("nullptr", 0, 0);
}




void CASSCF::compute_1rdm_coeff() {

    outfile->Printf("\n    Computing 1rdm coefficients ..................... ");

    SharedMatrix D1(new Matrix("1rdm coefficients contribution", nmo_, nmo_));

    // We force "Da == Db". The code below need changed if this constraint is revoked.

    std::vector<size_t> core_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_DOCC");
    std::vector<size_t> actv_all_ = mo_space_info_->get_absolute_mo("ACTIVE");
    std::vector<size_t> virt_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_UOCC");

    for(size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto m = core_all_[i];
        D1->set(m, m, 1.0);
    }

    for(size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        for(size_t j = 0; j < size_a; ++j) {
            auto u = actv_all_[i];
            auto v = actv_all_[j];
            D1->set(u, v, Gamma1_.block("aa").data()[i * na_ + j]);
        }
    }



    D1->back_transform(ints_->Ca());

    ints_->wfn()->Da()->copy(D1->clone());
    ints_->wfn()->Db()->copy(ints_->wfn()->Da());

    outfile->Printf("Done");
}




void CASSCF::compute_lagrangian() {

    set_lagrangian();

    outfile->Printf("\n    Computing Lagrangian ............................ ");
    //backtransform

    SharedMatrix L(new Matrix("Lagrangian", nmo_, nmo_));

    // The code below need be changed if frozen approximation is considered

    W_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {           
        L->add(i[0], i[1], value);
    }); 

    L->back_transform(ints_->Ca());
    ints_->wfn()->Lagrangian()->copy(L);
    outfile->Printf("Done");
}






void CASSCF::write_2rdm_spin_dependent() {

    outfile->Printf("\n    Writing 2RDM into disk .......................... ");

    auto psio_ = _default_psio_lib_;

    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);

/*************************************************************************/
/*!!!  coefficients in d2aa and d2bb need to multiply additional 1/2  !!!*/
/*************************************************************************/
  
    std::vector<size_t> core_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_DOCC");
    std::vector<size_t> actv_all_ = mo_space_info_->get_absolute_mo("ACTIVE");
    // std::vector<size_t> virt_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_UOCC");

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

            auto idx = i * na_ + j;
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
                    auto idx = i * na_ * na_ * na_ + j * na_ * na_ + k * na_ + l;
                    auto gamma_aa = Gamma2_.block("aaaa").data()[idx];
                    auto gamma_bb = Gamma2_.block("AAAA").data()[idx];
                    auto gamma_ab = Gamma2_.block("aAaA").data()[idx];

                    
                    d2aa.write_value(x, u, y, v, 0.25 * gamma_aa, 0, "NULL", 0);
                    d2bb.write_value(x, u, y, v, 0.25 * gamma_bb, 0, "NULL", 0);

                    d2ab.write_value(x, u, y, v, gamma_ab, 0, "NULL", 0);
                }
            }
        }
    }
  
    outfile->Printf("Done");

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



