#include "casscf/casscf.h"



void CASSCF::set_ambit_space() {

    c = 0

    outfile->Printf("\n    Setting ambit MO space .......................... ");
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

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

    // map space labels to mo spaces
    label_to_spacemo_[acore_label_[0]] = core_mos_;
    label_to_spacemo_[bcore_label_[0]] = core_mos_;
    label_to_spacemo_[aactv_label_[0]] = actv_mos_;
    label_to_spacemo_[bactv_label_[0]] = actv_mos_;
    label_to_spacemo_[avirt_label_[0]] = virt_mos_;
    label_to_spacemo_[bvirt_label_[0]] = virt_mos_;

    // define composite spaces
    BTF_->add_composite_mo_space("h", "ijkl", {acore_label_, aactv_label_});
    BTF_->add_composite_mo_space("H", "IJKL", {bcore_label_, bactv_label_});
    BTF_->add_composite_mo_space("p", "abcd", {aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("P", "ABCD", {bactv_label_, bvirt_label_});
    BTF_->add_composite_mo_space("g", "pqrsto456", {acore_label_, aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("G", "PQRSTO789", {bcore_label_, bactv_label_, bvirt_label_});

    outfile->Printf("Done");	
}




SharedMatrix CASSCF::convert_1rdm_so2mo(SharedMatrix D) {

    SharedMatrix Dmo(new Matrix(D->name() + " MO", nmo_, nmo_));
    for (int p = 0; p < nmo_; ++p) {
        for (int q = 0; q < nmo_; ++q) {
            double v1 = D->get(2 * p, 2 * q);
            double v2 = D->get(2 * p + 1, 2 * q + 1);
            Dmo->set(p, q, v1 + v2);
        }
    }
    return Dmo;
}


void CASSCF::setup_DensityAndFock() {

	// ambit::Tensor fock_ = 
	//     ambit::Tensor::build(ambit::CoreTensor, "Fock Tensor", {nmo_, nmo_} );

    psi::SharedMatrix Da(new psi::Matrix("Da", ncmo_, ncmo_));
    psi::SharedMatrix Db(new psi::Matrix("Db", ncmo_, ncmo_));
    auto L1a = tensor_to_matrix(cas_ref_.g1a(), actv_);
    auto L1b = tensor_to_matrix(cas_ref_.g1b(), actv_);

    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        // core block (diagonal)
        for (int i = 0; i < rdocc_[h]; ++i) {
            Da->set(offset + i, offset + i, 1.0);
            Db->set(offset + i, offset + i, 1.0);
        }

        offset += rdocc_[h];

        // active block
        for (int u = 0; u < actv_[h]; ++u) {
            for (int v = 0; v < actv_[h]; ++v) {
                Da->set(offset + u, offset + v, L1a->get(h, u, v));
                Db->set(offset + u, offset + v, L1b->get(h, u, v));
            }
        }

        offset += ncmopi_[h] - rdocc_[h];
    }

    ints_->make_fock_matrix(Da, Db);

	ambit::Tensor fock_a = 
	    ambit::Tensor::build(ambit::CoreTensor, "Fock alpha Tensor", {nmo_, nmo_} );   
	ambit::Tensor fock_b = 
	    ambit::Tensor::build(ambit::CoreTensor, "Fock beta Tensor", {nmo_, nmo_} ); 
	
	for(int p = 0; p < nmo_; ++p) {
		for(int q = 0; q < nmo_; ++q) {
			fock_a.data()[p * nmo_ + q] = ints_->get_fock_a(p, q);
			fock_b.data()[p * nmo_ + q] = ints_->get_fock_b(p, q);
		}
	} 

}





void CASSCF::compute_Lagrangian_CX(SharedMatrix L, SharedMatrix D1) {

/*******************************************************/
/*                                                     */
/*  compute omega of core-core and core-active blocks  */
/*                                                     */
/*******************************************************/

    // for (auto m : core_) {
    //     for (auto n : core_) {
    //         double omega = - f[m,n]; 

    //         L->set(m, n, omega);
    //         L->set(n, m, omega);
    //     }
    // }

    // for (auto m : core_) {
    //     for (auto u : active_) {
    //         double omega = - f[m,u]; 

    //         L->set(m, u, omega);
    //         L->set(u, m, omega);
    //     }
    // }


	ambit::Tensor omega_CX_a = 
	    ambit::Tensor::build(ambit::CoreTensor, "omega core-x alpha", {core_ + actv_, core_} );
	ambit::Tensor omega_CX_b = 
	    ambit::Tensor::build(ambit::CoreTensor, "omega core-x beta", {core_ + actv_, core_} );

	for(int m = 0; m < core_; ++m) {
		for(int p = 0, uplim = core_ + actv_; p < uplim; ++p) {
			omega_CX_a.data()[m * nmo_ + p] = - fock_a.data()[m * nmo_ + p];
			omega_CX_b.data()[m * nmo_ + p] = - fock_b.data()[m * nmo_ + p];
		}
	}



}

void CASSCF::compute_Lagrangian_AA(SharedMatrix L, SharedMatrix D1) {

/**********************************************/
/*                                            */
/*  compute omega of the active-active block  */
/*                                            */
/**********************************************/

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

	ambit::Tensor omega_AA_a = 
	    ambit::Tensor::build(ambit::CoreTensor, "omega active-active alpha", {actv_, actv_} );
	ambit::Tensor omega_AA_b = 
	    ambit::Tensor::build(ambit::CoreTensor, "omega active-active beta", {actv_, actv_} );

	nmo2_ = nmo_ * nmo_;
	nmo3_ = nmo2_ * nmo_;

	ambit::Tensor v_temp_aa = 
	    ambit::Tensor::build(ambit::CoreTensor, "v_temp_aa", {actv_, nmo_} );
	
	for(int v = core_, uplim = core_ + actv_; v < uplim; ++v) {
		for(int p = 0; p < nmo_; ++p) {
			for(int m = 0; m < core_; ++m) {
				v_temp_aa.data()[(v - core_) * nmo_ + p] += ints_->aptei_aa().data()[v * nmo3_ + m * nmo2_ + p * nmo_ + m];
			}	
		}	
	}	

	ambit::Tensor temp_1 = 
		ambit::Tensor::build(ambit::CoreTensor, "temp_1", {actv_, nmo_} );

	for(int v = core_, uplim = core_ + actv_; v < uplim; ++v) {
		for(int p = 0; p < nmo_; ++p) {
			temp_1.data()[(v - core_) * nmo_ + p] = ints_->oei_a().data()[v * nmo_ + p];	
		}	
	}	

	temp_1 += v_temp_aa;






}






void CASSCF::compute_Lagrangian() {
    outfile->Printf("\n  Computing Lagrangian ... ");

    SharedMatrix L(new Matrix("Lagrangian Matrix", nso_, nso_));
    compute_Lagrangian_CX(L);
    compute_Lagrangian_AA(L);


    // convert spin-orbital Lagrangian to MO
    SharedMatrix Lmo = convert_1rdm_so2mo(L);

    // add reference part
    for (int i = 0; i < nfrzc_ + ncore_; ++i) {
        Lmo->add(i, i, 2.0 * epsilon_a_->get(i));
    }

    // transform to AO and pass to Lagrangian_
    Lagrangian_ = Lmo->clone();
    Lagrangian_->set_name("Lagrangian Matrix AO");
    Lagrangian_->back_transform(Ca_);

    outfile->Printf("Done.");
}











double CASSCF::compute_gradient() {

	setup_DensityAndFock();

    compute_Lagrangian();

    write_2rdm_spin_dependent();

    return std::make_shared<Matrix>("nullptr", 0, 0);

}




void CASSCF::write_2rdm_spin_dependent(SharedMatrix D1) {
    outfile->Printf("\n  Writing D2 on disk");


    psio_ = _default_psio_lib_;
    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);
  

    for (auto p : core_1) {
        for (auto q : core_1) {
            for (auto r : virt_1) {
                for (auto s : virt_1) {
                    double t1 = T2_aa->get(p * nmo_ + q, r * nmo_ + s)*(1.0+pow(e,-s_const*eps_aa->get(p * nmo_ + q, r * nmo_ + s)*eps_aa->get(p * nmo_ + q, r * nmo_ + s)));
                    double t2 = T2_bb->get(p * nmo_ + q, r * nmo_ + s)*(1.0+pow(e,-s_const*eps_bb->get(p * nmo_ + q, r * nmo_ + s)*eps_bb->get(p * nmo_ + q, r * nmo_ + s)));
                    double t3 = T2_ab->get(p * nmo_ + q, r * nmo_ + s)*(1.0+pow(e,-s_const*eps_ab->get(p * nmo_ + q, r * nmo_ + s)*eps_ab->get(p * nmo_ + q, r * nmo_ + s)));
                    
                    double vaa = t1;
                    double vab = t3;
                    double vbb = t2;

                    d2aa.write_value(p, r, q, s, 0.5 * vaa, 0, "NULL", 0);
                    d2ab.write_value(p, r, q, s, 2.0 * vab, 0, "NULL", 0);
                    d2bb.write_value(p, r, q, s, 0.5 * vbb, 0, "NULL", 0);
                }
            }
        }
    }

    
   
  for (int p = 0; p < nmo_; ++p) {
        auto pa = 2 * p;
        auto pb = 2 * p + 1;

        for (int q = 0; q < nmo_; ++q) {
            auto qa = 2 * q;
            auto qb = 2 * q + 1;

            double va = D1_->get(pa, qa);
            double vb = D1_->get(pb, qb);
            if ((p == q) and (p < ndocc_)) {
                va += 0.5;
                vb += 0.5;
            }

            for (int i = 0; i < ndocc_; ++i) {
                if (i != q) {
                    d2aa.write_value(p, q, i, i, 0.5 * va, 0, "NULL", 0);
                    d2bb.write_value(p, q, i, i, 0.5 * vb, 0, "NULL", 0);

                    d2aa.write_value(p, i, i, q, -0.5 * va, 0, "NULL", 0);
                    d2bb.write_value(p, i, i, q, -0.5 * vb, 0, "NULL", 0);
                }

                d2ab.write_value(p, q, i, i, 2.0 * va, 0, "NULL", 0);
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






