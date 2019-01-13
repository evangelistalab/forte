/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <algorithm>
#include <cmath>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"

#include "embedding.h"
//#include "avas.h"

using namespace psi;

namespace forte {

void set_EMBEDDING_options(ForteOptions& foptions) {
	foptions.add_str("CUTOFF_BY", "THRESHOLD", "Cut off by: threshold or number.");
	foptions.add_int("NUM_OCC", 0, "Number of (restricted) occpied in system A");
	foptions.add_int("NUM_VIR", 0, "Number of (restricted) virtual in system A");
	foptions.add_double("THRESHOLD", 0.5, "Projector eigenvalue threshold, 0.5 as default");
	foptions.add_str("REFERENCE", "HF", "HF, ROHF, UHF(not implemented), MCSCF, CASSCF, CINO, CINOACTV");
	foptions.add_bool("WRITE_FREEZE_MO", true,
		"Pass orbital space information automatically or manually");
	foptions.add_bool("SEMICANON", true, "Perform semi-canonicalization or not in the end");
	foptions.add_int("FROZEN_SYS_DOCC", 0, "Freeze system occ orbitals");
	foptions.add_int("FROZEN_SYS_UOCC", 0, "Freeze system vir orbitals");
}

void make_embedding(psi::SharedWavefunction ref_wfn, psi::Options& options, psi::SharedMatrix Pf) {
	if (Pf) {
		Dimension noccpi = ref_wfn->doccpi();
		Dimension zeropi = nmopi - nmopi;
		Dimension nvirpi = nmopi - noccpi;
		Dimension sys_mo = nmopi;
		int num_actv = 0;
		int num_rdocc = 0;
		int num_docc = 0;
		int num_fo = 0;
		int num_fv = 0;
		Dimension actv_a = zeropi;
		Dimension res_docc_ori = zeropi;
		Dimension docc_ori = zeropi;
		SharedMatrix Ca_t = ref_wfn->Ca();

		if (options_.get_str("REFERENCE") == "CINO" || options_.get_str("REFERENCE") == "CINOACTV") {
			outfile->Printf("\n Clear the previous frozen orbs when doing NO rotation\n");
			options_["FROZEN_DOCC"][0].assign(0);
			options_["FROZEN_UOCC"][0].assign(0);
		}

		if (options_.get_str("REFERENCE") == "CASSCF" || options_.get_str("REFERENCE") == "CINOACTV") {

			num_actv = options_["ACTIVE"][0].to_integer();
			num_rdocc = options_["RESTRICTED_DOCC"][0].to_integer();
			num_docc = options_["DOCC"][0].to_integer();

			actv_a[0] = num_actv;
			res_docc_ori[0] = num_rdocc;
			docc_ori[0] = num_docc;

			// Change the following part when symmetry is applied
		}

		if (options_.get_str("FREEZE_CORE") ==
			"TRUE") {
			outfile->Printf("\n Read frozen core info: \n");
			num_fo = options_["FROZEN_DOCC"][0].to_integer();
			num_fv = options_["FROZEN_UOCC"][0].to_integer();
			outfile->Printf("fo: %d, fv: %d \n", num_fo, num_fv);
		}

		int num_actv_docc = num_docc - num_rdocc - num_fo;
		int num_actv_vir = num_actv - num_actv_docc;
		outfile->Printf(
			"\n The reference has %d active occupied, %d active virtual, they will be assigned to A "
			"(without any change);\n %d frozen core and %d frozen virtual will be assigned directly to B \n",
			num_actv_docc, num_actv_vir, num_fo, num_fv);

		//transform Pf to MO basis Pf_pq
		Pf->transform(Ca_t);

		// Diagonalize Pf_pq for occ and vir part, respectively.
		SharedMatrix P_oo = S_sys_in_all->get_block(occ, occ);
		SharedMatrix Uo(new Matrix("Uo", nirrep, nroccpi, nroccpi));
		SharedVector lo(new Vector("lo", nirrep, nroccpi));
		P_oo->diagonalize(Uo, lo, descending);
		// lo->print();

		SharedMatrix P_vv = S_sys_in_all->get_block(vir, vir);
		SharedMatrix Uv(new Matrix("Uv", nirrep, nrvirpi, nrvirpi));
		SharedVector lv(new Vector("lv", nirrep, nrvirpi));
		P_vv->diagonalize(Uv, lv, descending);
		// lv->print();

		SharedMatrix U_all(new Matrix("U with Pab", nirrep, nmopi, nmopi));
		U_all->set_block(occ, occ, Uo);
		U_all->set_block(vir, vir, Uv);

		// Based on threshold or num_occ/num_vir, decide the partition (change to a function soon)
		std::vector<int> index_trace_occ = {};
		std::vector<int> index_trace_vir = {};
		if (options_.get_str("CUTOFF_BY") == "THRESHOLD") {
			for (int i = 0; i < nroccpi[0]; i++) {
				if (lo->get(0, i) > thresh) {
					index_trace_occ.push_back(i);
					outfile->Printf("\n Occupied orbital %d is partitioned to A with eigenvalue %8.8f",
						i, lo->get(0, i));
				}
			}
			for (int i = 0; i < nrvirpi[0]; i++) {
				if (lv->get(0, i) > thresh) {
					index_trace_vir.push_back(i);
					outfile->Printf("\n Virtual orbital %d is partitioned to A with eigenvalue %8.8f",
						i, lv->get(0, i));
				}
			}
		}

		if (options_.get_str("CUTOFF_BY") == "NUMBER") {
			for (int i = 0; i < num_occ; i++) {
				index_trace_occ.push_back(i);
				outfile->Printf("\n Occupied orbital %d is partitioned to A with eigenvalue %8.8f", i,
					lo->get(0, i));
			}
			for (int i = 0; i < num_vir; i++) {
				index_trace_vir.push_back(i);
				outfile->Printf("\n Occupied orbital %d is partitioned to A with eigenvalue %8.8f", i,
					lv->get(0, i));
			}
		}

		// Rotate Bocc to first block, rotate Bvir to last block, in order to freeze them
		// Change to block operation soon
		SharedMatrix Ca_Rt(Ca_t->clone());

		// Write Bocc
		int sizeBO = noccpi[0] - index_trace_occ.size() - num_actv_docc;
		int sizeBV = nvirpi[0] - index_trace_vir.size() - num_actv_vir;
		int sizeAO = index_trace_occ.size(); // AO and AV will not include AA
		int sizeAV = index_trace_vir.size();
		outfile->Printf("\n sizeBO: %d, sizeAO: %d, sizeAV: %d, sizeBV: %d \n", sizeBO, sizeAO, sizeAV,
			sizeBV);

		Dimension AO = nmopi;
		AO[0] = sizeAO;
		Dimension AV = nmopi;
		AV[0] = sizeAV;
		Dimension BO = nmopi;
		BO[0] = sizeBO;
		Dimension BV = nmopi;
		BV[0] = sizeBV;
		Dimension AO_BO = nmopi;
		AO_BO[0] = sizeAO + sizeBO;
		Dimension AO_BO_A = nmopi;
		AO_BO_A[0] = sizeAO + sizeBO + num_actv;
		Dimension AO_BO_A_AV = nmopi;
		AO_BO_A_AV[0] = sizeAO + sizeBO + num_actv + sizeAV;

		// Rotate MO coeffs
		Ca_->copy(Matrix::doublet(Ca_t, U_all, false, false)); // Structure becomes AO-BO-0-AV-BV

		if (options_.get_str("REFERENCE") == "CASSCF" || options_.get_str("REFERENCE") == "CINOACTV") {
			// SharedMatrix Ua(new Matrix("Uv", nirrep, actv_a, actv_a));
			// Ua->identity();
			for (int i = 0; i < num_actv; ++i) {
				Ca_->set_column(
					0, nroccpi[0] + i,
					Ca_Rt->get_column(0, nroccpi[0] + i)); // Structure becomes AO-BO-A-AV-BV
			}
		}

		Ca_Rt->zero();
		// Write Bocc
		for (int i = 0; i < sizeBO; ++i) {
			Ca_Rt->set_column(0, i, Ca_->get_column(0, sizeAO + i));
		}

		// Write Aocc
		for (int i = 0; i < sizeAO; ++i) {
			Ca_Rt->set_column(0, i + sizeBO, Ca_->get_column(0, i));
		}

		// Write Avir
		for (int i = 0; i < sizeAV; ++i) {
			Ca_Rt->set_column(0, i + AO_BO_A[0], Ca_->get_column(0, i + AO_BO_A[0]));
		}

		// Write Bvir
		for (int i = 0; i < sizeBV; ++i) {
			Ca_Rt->set_column(0, i + AO_BO_A_AV[0], Ca_->get_column(0, AO_BO_A_AV[0] + i));
		}

		// Update Ca_
		if (options_.get_str("REFERENCE") == "CASSCF" || options_.get_str("REFERENCE") == "CINOACTV") { // Ca_: AO-BO-A-AV-BV
																										// copy original columns in active space from Ca_
			for (int i = 0; i < num_actv; ++i) {
				Ca_Rt->set_column(0, i + nroccpi[0], Ca_->get_column(0, nroccpi[0] + i));
			}
		}
		Ca_->copy(Ca_Rt); // Ca_: BO-AO-A-AV-BV

		if (options_.get_bool("SEMICANON") == true) {
			outfile->Printf("\n *** Semi-canonicalization *** \n");

			Slice BOs(zeropi, BO);
			Slice AOs(BO, AO_BO);
			Slice AVs(AO_BO_A, AO_BO_A_AV);
			Slice BVs(AO_BO_A_AV, nmopi);

			// Build Fock in localized basis
			SharedMatrix Fa_loc = Matrix::triplet(Ca_Rt, Fa_, Ca_Rt, true, false, false);
			// outfile->Printf("\n Fock matrix in localized basis: \n");
			// Fa_loc->print();
			SharedMatrix Fa_AOAO = Fa_loc->get_block(AOs, AOs);
			SharedMatrix Fa_AVAV = Fa_loc->get_block(AVs, AVs);
			SharedMatrix Fa_AAAA = Fa_loc->get_block(actv, actv);

			// AO[0] -= num_actv_docc;
			SharedMatrix Uao(new Matrix("Uoo", nirrep, AO, AO));
			SharedVector lao(new Vector("loo", nirrep, AO));
			Fa_AOAO->diagonalize(Uao, lao, ascending);
			Fa_AOAO->zero();
			for (int i = 0; i < AO[0]; ++i) {
				Fa_AOAO->set(0, i, i, lao->get(0, i));
			}
			// Fa_AOAO->print();

			// AV[0] -= num_actv_vir;
			SharedMatrix Uav(new Matrix("Uvv", nirrep, AV, AV));
			SharedVector lav(new Vector("lvv", nirrep, AV));
			Fa_AVAV->diagonalize(Uav, lav, ascending);
			Fa_AVAV->zero();
			for (int i = 0; i < AV[0]; ++i) {
				Fa_AVAV->set(0, i, i, lav->get(0, i));
			}
			// Fa_AVAV->print();

			// Build transformation matrix
			SharedMatrix U_all_2(new Matrix("U with Pab", nirrep, nmopi, nmopi));
			SharedMatrix Ubo(new Matrix("Ubo", nirrep, BO, BO));
			SharedMatrix Ubv(new Matrix("Ubv", nirrep, BV, BV));
			Ubo->identity();
			Ubv->identity();
			U_all_2->set_block(AOs, AOs, Uao);
			U_all_2->set_block(AVs, AVs, Uav);
			U_all_2->set_block(BOs, BOs, Ubo);
			U_all_2->set_block(BVs, BVs, Ubv);

			Fa_loc->set_block(AOs, AOs, Fa_AOAO);
			Fa_loc->set_block(AVs, AVs, Fa_AVAV);

			if (options_.get_str("REFERENCE") == "CASSCF" || options_.get_str("REFERENCE") == "CINOACTV") {
				SharedMatrix Uaa(new Matrix("Uaa", nirrep, actv_a, actv_a));
				SharedVector laa(new Vector("laa", nirrep, actv_a));
				Fa_AAAA->diagonalize(Uaa, laa, ascending);
				Fa_AAAA->zero();
				for (int i = 0; i < actv_a[0]; ++i) {
					Fa_AAAA->set(0, i, i, laa->get(0, i));
				}
				// Fa_AAAA->print();

				U_all_2->set_block(actv, actv, Uaa);
				Fa_loc->set_block(actv, actv, Fa_AAAA);
			}

			Fa_->copy(Fa_loc);

			Ca_->copy(Matrix::doublet(Ca_Rt, U_all_2, false, false));
		}

		//Apply frozen system core/virtual
		sizeBO += frz_sys_docc;
		sizeAO -= frz_sys_docc;
		sizeAV -= frz_sys_uocc;
		sizeBV += frz_sys_uocc;

		// Write MO space info and print
		outfile->Printf("\n  FROZEN_DOCC     = %d", sizeBO);
		outfile->Printf("\n  RESTRICTED_DOCC     = %d", sizeAO);
		outfile->Printf("\n  ACTIVE     = %d", num_actv);
		outfile->Printf("\n  RESTRICTED_UOCC     = %d", sizeAV);
		outfile->Printf("\n  FROZEN_UOCC	 = %d", sizeBV);
		outfile->Printf("\n");

		if (options_.get_bool("WRITE_FREEZE_MO") == true) {
			if (options_.get_str("FREEZE_CORE") ==
				"TRUE") { // If the initial calculation includes freeze_core, add them to environment
				outfile->Printf("\n Clear the previous frozen cores\n");
				options_["FROZEN_DOCC"][0].assign(0);
				options_["FROZEN_UOCC"][0].assign(0);
			}
			else {
				outfile->Printf("\n Create frozen B \n");
				options_["FROZEN_DOCC"].add(0);
				options_["FROZEN_UOCC"].add(0);
			}
			options_["FROZEN_DOCC"][0].assign(sizeBO);
			options_["FROZEN_UOCC"][0].assign(sizeBV);

			if (options_.get_str("REFERENCE") == "CASSCF" || options_.get_str("REFERENCE") == "CINOACTV") {
				// options_["RESTRICTED_DOCC"].add(0);
				// options_["ACTIVE"].add(0);
				options_["RESTRICTED_UOCC"].add(0);
				options_["RESTRICTED_DOCC"][0].assign(sizeAO);
				options_["ACTIVE"][0].assign(num_actv);
				options_["RESTRICTED_UOCC"][0].assign(sizeAV);
			}
		}
	}
}

psi::SharedMatrix semicanonicalize_block(psi::SharedWavefunction ref_wfn, psi::SharedMatrix C_tilde,
                                    std::vector<int>& mos, int offset) {
    //Function copy from avas.cc
    int nso = ref_wfn->nso();
    int nmo_block = mos.size();
    auto C_block = std::make_shared<psi::Matrix>("C block", nso, nmo_block);

    int mo_count = 0;
    for (int i : mos) {
        for (int mu = 0; mu < nso; ++mu) {
            double value = C_tilde->get(mu, i + offset);
            C_block->set(mu, mo_count, value);
        }
        mo_count += 1;
    }
    // compute (C_block)^T F C_block
    auto Foi = psi::Matrix::triplet(C_block, ref_wfn->Fa(), C_block, true, false, false);

    auto U_block = std::make_shared<psi::Matrix>("U block", nmo_block, nmo_block);
    auto epsilon_block = std::make_shared<Vector>("epsilon block", nmo_block);
    Foi->diagonalize(U_block, epsilon_block);
    auto C_block_prime = psi::Matrix::doublet(C_block, U_block);
    return C_block_prime;
}
}

