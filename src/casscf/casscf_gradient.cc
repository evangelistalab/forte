#include "casscf/casscf.h"
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





    // for (auto u : active_) {
    //     for (auto v : active_) {

    //         double temp = 0.0;
    //         for(auto v1 : active_) {

    //             double temp_1 = 0.0;
    //             for(auto m : core_) {
    //                 temp_1 += v[v, m, v1, m];
    //             }

    //             temp += (h[v, v1] + temp_1) * gamma_[v1, u];
    //         }

    //         for(auto v1 : active_) {
    //             for(auto x : active_) {
    //                 for(auto y : active_) {

    //                     temp += 0.5 * v[v, v1, x, y] * gamma_[x, y, u, v1]; 
    //                 }
    //             }
    //         }

    //         double omega = - temp; 
    //         L->set(u, v, omega);
    //         L->set(v, u, omega);
    //     }
    // }

	// ambit::Tensor omega_AA_a = 
	//     ambit::Tensor::build(ambit::CoreTensor, "omega active-active alpha", {na_, na_} );
	// ambit::Tensor omega_AA_b = 
	//     ambit::Tensor::build(ambit::CoreTensor, "omega active-active beta", {na_, na_} );

	// nmo2_ = nmo_ * nmo_;
	// nmo3_ = nmo2_ * nmo_;

	// ambit::Tensor v_temp_aa = 
	//     ambit::Tensor::build(ambit::CoreTensor, "v_temp_aa", {actv_, nmo_} );
	
// 	for(int v = core_, uplim = core_ + actv_; v < uplim; ++v) {
// 		for(int p = 0; p < nmo_; ++p) {
// 			for(int m = 0; m < core_; ++m) {
// 				v_temp_aa.data()[(v - core_) * nmo_ + p] += ints_->aptei_aa().data()[v * nmo3_ + m * nmo2_ + p * nmo_ + m];
// 			}	
// 		}	
// 	}	

// 	ambit::Tensor temp_1 = 
// 		ambit::Tensor::build(ambit::CoreTensor, "temp_1", {actv_, nmo_} );

// 	for(int v = core_, uplim = core_ + actv_; v < uplim; ++v) {
// 		for(int p = 0; p < nmo_; ++p) {
// 			temp_1.data()[(v - core_) * nmo_ + p] = ints_->oei_a().data()[v * nmo_ + p];	
// 		}	
// 	}	

// 	temp_1 += v_temp_aa;

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











// double CASSCF::compute_gradient() {

// 	setup_DensityAndFock();

//     compute_Lagrangian();

//     write_2rdm_spin_dependent();

//     return std::make_shared<Matrix>("nullptr", 0, 0);

// }




// void CASSCF::write_2rdm_spin_dependent(SharedMatrix D1) {
//     outfile->Printf("\n  Writing D2 on disk");


//     psio_ = _default_psio_lib_;
//     IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
//     IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
//     IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);
  

//     for (auto p : core_1) {
//         for (auto q : core_1) {
//             for (auto r : virt_1) {
//                 for (auto s : virt_1) {
//                     double t1 = T2_aa->get(p * nmo_ + q, r * nmo_ + s)*(1.0+pow(e,-s_const*eps_aa->get(p * nmo_ + q, r * nmo_ + s)*eps_aa->get(p * nmo_ + q, r * nmo_ + s)));
//                     double t2 = T2_bb->get(p * nmo_ + q, r * nmo_ + s)*(1.0+pow(e,-s_const*eps_bb->get(p * nmo_ + q, r * nmo_ + s)*eps_bb->get(p * nmo_ + q, r * nmo_ + s)));
//                     double t3 = T2_ab->get(p * nmo_ + q, r * nmo_ + s)*(1.0+pow(e,-s_const*eps_ab->get(p * nmo_ + q, r * nmo_ + s)*eps_ab->get(p * nmo_ + q, r * nmo_ + s)));
                    
//                     double vaa = t1;
//                     double vab = t3;
//                     double vbb = t2;

//                     d2aa.write_value(p, r, q, s, 0.5 * vaa, 0, "NULL", 0);
//                     d2ab.write_value(p, r, q, s, 2.0 * vab, 0, "NULL", 0);
//                     d2bb.write_value(p, r, q, s, 0.5 * vbb, 0, "NULL", 0);
//                 }
//             }
//         }
//     }

    
   
//   for (int p = 0; p < nmo_; ++p) {
//         auto pa = 2 * p;
//         auto pb = 2 * p + 1;

//         for (int q = 0; q < nmo_; ++q) {
//             auto qa = 2 * q;
//             auto qb = 2 * q + 1;

//             double va = D1_->get(pa, qa);
//             double vb = D1_->get(pb, qb);
//             if ((p == q) and (p < ndocc_)) {
//                 va += 0.5;
//                 vb += 0.5;
//             }

//             for (int i = 0; i < ndocc_; ++i) {
//                 if (i != q) {
//                     d2aa.write_value(p, q, i, i, 0.5 * va, 0, "NULL", 0);
//                     d2bb.write_value(p, q, i, i, 0.5 * vb, 0, "NULL", 0);

//                     d2aa.write_value(p, i, i, q, -0.5 * va, 0, "NULL", 0);
//                     d2bb.write_value(p, i, i, q, -0.5 * vb, 0, "NULL", 0);
//                 }

//                 d2ab.write_value(p, q, i, i, 2.0 * va, 0, "NULL", 0);
//             }
//         }
//     }


//     outfile->Printf("\n Done");

//     d2aa.flush(1);
//     d2bb.flush(1);
//     d2ab.flush(1);

//     d2aa.set_keep_flag(1);
//     d2bb.set_keep_flag(1);
//     d2ab.set_keep_flag(1);

//     d2aa.close();
//     d2bb.close();
//     d2ab.close();
// }


}



