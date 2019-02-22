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
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"
#include "embedding.h"
//#include "avas.h"

using namespace psi;

namespace forte {

void make_embedding(psi::SharedWavefunction ref_wfn, psi::Options& options, psi::SharedMatrix Pf) {
	outfile->Printf("\n ------ Orbital Localization and Embedding ------ \n");

	// 1. Get necessary information
	double thresh = options.get_double("THRESHOLD");
	int num_occ = options.get_int("NUM_OCC");
	int num_vir = options.get_int("NUM_VIR");
	int frz_sys_docc = options.get_int("FROZEN_SYS_DOCC");
	int frz_sys_uocc = options.get_int("FROZEN_SYS_UOCC");

	std::shared_ptr<PSIO> psio(_default_psio_lib_);
	if (!ref_wfn)
		throw PSIEXCEPTION("SCF has not been run yet!");

	// 2. Apply projector
	if (Pf) {
		Dimension nmopi = ref_wfn->nmopi();
		int nirrep = ref_wfn->nirrep();
		if (nirrep > 1) {
			throw PSIEXCEPTION("Fragment projection works only without symmetry! (symmetry C1)");
		}
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

		if (options.get_str("REFERENCE") == "CINO" || options.get_str("REFERENCE") == "CINOACTV") {
			outfile->Printf("\n Clear the previous frozen orbs when doing NO rotation\n");
			options["FROZEN_DOCC"][0].assign(0);
			options["FROZEN_UOCC"][0].assign(0);
		}

		if (options.get_str("REFERENCE") == "CASSCF" || options.get_str("REFERENCE") == "CINOACTV") {

			num_actv = options["ACTIVE"][0].to_integer();
			num_rdocc = options["RESTRICTED_DOCC"][0].to_integer();
			num_docc = options["DOCC"][0].to_integer();

			actv_a[0] = num_actv;
			res_docc_ori[0] = num_rdocc;
			docc_ori[0] = num_docc;
		}

		if (options.get_str("FREEZE_CORE") ==
			"TRUE") {
			outfile->Printf("\n Read frozen core info: \n");
			num_fo = options["FROZEN_DOCC"][0].to_integer();
			num_fv = options["FROZEN_UOCC"][0].to_integer();
			outfile->Printf("fo: %d, fv: %d \n", num_fo, num_fv);
		}

		if (options.get_str("REFERENCE") == "RHF") {

			num_docc = options["DOCC"][0].to_integer();
			num_rdocc = num_docc - num_fo;

			actv_a[0] = num_actv;
			res_docc_ori[0] = num_rdocc;
			docc_ori[0] = num_docc;
		}

		int num_actv_docc = num_docc - num_rdocc - num_fo;
		int num_actv_vir = num_actv - num_actv_docc;
		outfile->Printf(
			"\n The reference has %d active occupied, %d active virtual, they will be assigned to A "
			"(without any change);\n %d frozen core and %d frozen virtual will be assigned directly to B \n",
			num_actv_docc, num_actv_vir, num_fo, num_fv);

		Dimension nroccpi = noccpi;
		nroccpi[0] = noccpi[0] - num_actv_docc;
		Dimension nrvirpi = noccpi;
		nrvirpi[0] = nvirpi[0] - num_actv_vir;

		Slice occ(zeropi, nroccpi);
		Slice vir(nroccpi + actv_a, nmopi);
		Slice actv(nroccpi, nroccpi + actv_a);

		//transform Pf to MO basis Pf_pq
		Pf->transform(Ca_t);

		// Diagonalize Pf_pq for occ and vir part, respectively.
		SharedMatrix P_oo = Pf->get_block(occ, occ);
		SharedMatrix Uo(new Matrix("Uo", nirrep, nroccpi, nroccpi));
		SharedVector lo(new Vector("lo", nirrep, nroccpi));
		P_oo->diagonalize(Uo, lo, descending);
		// lo->print();

		SharedMatrix P_vv = Pf->get_block(vir, vir);
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
		if (options.get_str("CUTOFF_BY") == "THRESHOLD") {
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

                if (options.get_str("CUTOFF_BY") == "CUM_THRESHOLD") {
			double tmp = 0.0;
			double sum_lo = 0.0;
			double sum_lv = 0.0;
			for (int i = 0; i < nroccpi[0]; i++) {
				sum_lo += lo->get(0, i);
			}
			for (int i = 0; i < nrvirpi[0]; i++) {
                                sum_lv += lv->get(0, i);
                        }
			
			double cum_l_o = 0.0;
                        for (int i = 0; i < nroccpi[0]; i++) {
				tmp += lo->get(0, i);
				cum_l_o = tmp/sum_lo;
                                if (cum_l_o > thresh) {
                                        index_trace_occ.push_back(i);
                                        outfile->Printf("\n Occupied orbital %d is partitioned to A with cumulative eigenvalue %8.8f",
                                                i, cum_l_o);
                                }
                        }
			tmp = 0.0;
			double cum_l_v = 0.0;
                        for (int i = 0; i < nrvirpi[0]; i++) {
				tmp += lv->get(0, i);
				cum_l_v = tmp/sum_lv;
                                if (cum_l_v > thresh) {
                                        index_trace_vir.push_back(i);
                                        outfile->Printf("\n Virtual orbital %d is partitioned to A with cumulative eigenvalue %8.8f",
                                                i, cum_l_v);
                                }
                        }
                }

		if (options.get_str("CUTOFF_BY") == "NUMBER") {
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
		ref_wfn->Ca()->copy(Matrix::doublet(Ca_t, U_all, false, false)); // Structure becomes AO-BO-0-AV-BV

		if (options.get_str("REFERENCE") == "CASSCF" || options.get_str("REFERENCE") == "CINOACTV") {
			// SharedMatrix Ua(new Matrix("Uv", nirrep, actv_a, actv_a));
			// Ua->identity();
			for (int i = 0; i < num_actv; ++i) {
				ref_wfn->Ca()->set_column(
					0, nroccpi[0] + i,
					Ca_Rt->get_column(0, nroccpi[0] + i)); // Structure becomes AO-BO-A-AV-BV
			}
		}

		Ca_Rt->zero();
		// Write Bocc
		for (int i = 0; i < sizeBO; ++i) {
			Ca_Rt->set_column(0, i, ref_wfn->Ca()->get_column(0, sizeAO + i));
		}

		// Write Aocc
		for (int i = 0; i < sizeAO; ++i) {
			Ca_Rt->set_column(0, i + sizeBO, ref_wfn->Ca()->get_column(0, i));
		}

		// Write Avir
		for (int i = 0; i < sizeAV; ++i) {
			Ca_Rt->set_column(0, i + AO_BO_A[0], ref_wfn->Ca()->get_column(0, i + AO_BO_A[0]));
		}

		// Write Bvir
		for (int i = 0; i < sizeBV; ++i) {
			Ca_Rt->set_column(0, i + AO_BO_A_AV[0], ref_wfn->Ca()->get_column(0, AO_BO_A_AV[0] + i));
		}

		// Update Ca_
		if (options.get_str("REFERENCE") == "CASSCF" || options.get_str("REFERENCE") == "CINOACTV") { // Ca_: AO-BO-A-AV-BV
																										// copy original columns in active space from Ca_
			for (int i = 0; i < num_actv; ++i) {
				Ca_Rt->set_column(0, i + nroccpi[0], ref_wfn->Ca()->get_column(0, nroccpi[0] + i));
			}
		}
		ref_wfn->Ca()->copy(Ca_Rt); // Ca_: BO-AO-A-AV-BV

		if (options.get_bool("SEMICANON") == true) {
			outfile->Printf("\n *** Semi-canonicalization *** \n");

			Slice BOs(zeropi, BO);
			Slice AOs(BO, AO_BO);
			Slice AVs(AO_BO_A, AO_BO_A_AV);
			Slice BVs(AO_BO_A_AV, nmopi);

			// Build Fock in localized basis
			SharedMatrix Fa_loc = Matrix::triplet(Ca_Rt, ref_wfn->Fa(), Ca_Rt, true, false, false);
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

			if (options.get_str("REFERENCE") == "CASSCF" || options.get_str("REFERENCE") == "CINOACTV") {
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

			ref_wfn->Fa()->copy(Fa_loc);

			ref_wfn->Ca()->copy(Matrix::doublet(Ca_Rt, U_all_2, false, false));
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

		if (options.get_bool("WRITE_FREEZE_MO") == true) {
			if (options.get_str("FREEZE_CORE") ==
				"TRUE") { // If the initial calculation includes freeze_core, add them to environment
				outfile->Printf("\n Clear the previous frozen cores\n");
				options["FROZEN_DOCC"][0].assign(0);
				options["FROZEN_UOCC"][0].assign(0);
			}
			else {
				outfile->Printf("\n Create frozen B \n");
				options["FROZEN_DOCC"].add(0);
				options["FROZEN_UOCC"].add(0);
			}
			options["FROZEN_DOCC"][0].assign(sizeBO);
			options["FROZEN_UOCC"][0].assign(sizeBV);

			if (options.get_str("REFERENCE") == "CASSCF" || options.get_str("REFERENCE") == "CINOACTV") {
				// options["RESTRICTED_DOCC"].add(0);
				// options["ACTIVE"].add(0);
				options["RESTRICTED_UOCC"].add(0);
				options["RESTRICTED_DOCC"][0].assign(sizeAO);
				options["ACTIVE"][0].assign(num_actv);
				options["RESTRICTED_UOCC"][0].assign(sizeAV);
			}
		}
	}
	else {
		throw PSIEXCEPTION("No projector (matrix) found!");
	}
}

/*
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
*/
}

