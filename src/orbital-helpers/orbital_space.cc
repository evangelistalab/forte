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
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"

#include "orbital_space.h"

using namespace psi;

namespace forte {

psi::SharedMatrix semicanonicalize_block(psi::SharedWavefunction ref_wfn, psi::SharedMatrix C_tilde,
                                         std::vector<int>& mos, int offset);

void make_avas(psi::SharedWavefunction ref_wfn, psi::Options& options, psi::SharedMatrix Ps) {
    if (Ps) {
        outfile->Printf("\n  Generating AVAS orbitals\n");
        int nocc = ref_wfn->nalpha();
        int nso = ref_wfn->nso();
        int nmo = ref_wfn->nmo();
        int nvir = nmo - nocc;

        int avas_num_active = options.get_int("AVAS_NUM_ACTIVE");
        size_t avas_num_active_occ = options.get_int("AVAS_NUM_ACTIVE_OCC");
        size_t avas_num_active_vir = options.get_int("AVAS_NUM_ACTIVE_VIR");
        double avas_sigma = options.get_double("AVAS_SIGMA");

        outfile->Printf("\n  ==> AVAS Options <==");
        outfile->Printf("\n    Number of AVAV MOs:                 %6d", avas_num_active);
        outfile->Printf("\n    Number of active occupied AVAS MOs: %6d", avas_num_active_occ);
        outfile->Printf("\n    Number of active virtual AVAS  MOs: %6d", avas_num_active_vir);
        outfile->Printf("\n    AVAS sigma (cumulative threshold):       %5f", avas_sigma);
        outfile->Printf("\n    Number of occupied MOs:             %6d", nocc);
        outfile->Printf("\n    Number of virtual MOs:              %6d", nvir);
        outfile->Printf("\n");

        // Allocate a matrix for the occupied block
        auto Socc = std::make_shared<psi::Matrix>("S occupied block", nocc, nocc);
        auto Svir = std::make_shared<psi::Matrix>("S virtual block", nvir, nvir);
        psi::SharedMatrix CPsC = Ps->clone();
        CPsC->transform(ref_wfn->Ca());
        // No diagonalization Socc and Svir
        bool diagonalize_s = options.get_bool("AVAS_DIAGONALIZE");

        auto Uocc = std::make_shared<psi::Matrix>("U occupied block", nocc, nocc);
        auto sigmaocc = std::make_shared<Vector>("sigma occupied block", nocc);

        auto Uvir = std::make_shared<psi::Matrix>("U virtual block", nvir, nvir);
        auto sigmavir = std::make_shared<Vector>("sigma virtual block", nvir);

        auto U = std::make_shared<psi::Matrix>("U", nmo, nmo);
        // diagonalize S
        if (diagonalize_s) {
            outfile->Printf(
                "\n  Diagonalizing the occupied/virtual blocks of the projector matrix");
            // Grab the occupied block and diagonalize it
            for (int i = 0; i < nocc; i++) {
                for (int j = 0; j < nocc; j++) {
                    double value = CPsC->get(i, j);
                    Socc->set(i, j, value);
                }
            }

            Socc->diagonalize(Uocc, sigmaocc, descending);
            // Grab the virtual block and diagonalize it
            for (int a = 0; a < nvir; a++) {
                for (int b = 0; b < nvir; b++) {
                    double value = CPsC->get(nocc + a, nocc + b);
                    Svir->set(a, b, value);
                }
            }
            Svir->diagonalize(Uvir, sigmavir, descending);
            // Form the full matrix U
            for (int i = 0; i < nocc; i++) {
                for (int j = 0; j < nocc; j++) {
                    double value = Uocc->get(i, j);
                    U->set(i, j, value);
                }
            }
            for (int a = 0; a < nvir; a++) {
                for (int b = 0; b < nvir; b++) {
                    double value = Uvir->get(a, b);
                    U->set(a + nocc, b + nocc, value);
                }
            }
        } else {
            outfile->Printf("\n  Skipping diagonalization of the projector matrix.\n  Orbitals "
                            "will be sorted instead of being rotated.");
            // Socc
            for (int i = 0; i < nocc; i++) {
                for (int j = 0; j < nocc; j++) {
                    double value = CPsC->get(i, j);
                    Socc->set(i, j, value);
                }
            }

            for (int i = 0; i < nocc; i++) {
                double value = Socc->get(i, i);
                sigmaocc->set(i, value);
            }
            // Svir
            for (int a = 0; a < nvir; a++) {
                for (int b = 0; b < nvir; b++) {
                    double value = CPsC->get(nocc + a, nocc + b);
                    Svir->set(a, b, value);
                }
            }

            for (int a = 0; a < nvir; a++) {
                double value = Svir->get(a, a);
                sigmavir->set(a, value);
            }
            for (int p = 0; p < nmo; p++) {
                for (int q = 0; q < nmo; q++) {
                    U->set(p, q, p == q ? 1.0 : 0.0);
                }
            }
        } // end options of dia
        auto Ca_tilde = psi::linalg::doublet(ref_wfn->Ca(), U);

        // sum of the eigenvalues (occ + vir)
        double s_sum = 0.0;
        for (int i = 0; i < nocc; i++) {
            s_sum += sigmaocc->get(i);
        }
        for (int a = 0; a < nvir; a++) {
            s_sum += sigmavir->get(a);
        }
        outfile->Printf("\n  Sum of eigenvalues: %f\n", s_sum);

        std::vector<std::tuple<double, bool, int>> sorted_mos;
        for (int i = 0; i < nocc; i++) {
            sorted_mos.push_back(std::make_tuple(sigmaocc->get(i), true, i));
        }
        for (int a = 0; a < nvir; a++) {
            sorted_mos.push_back(std::make_tuple(sigmavir->get(a), false, a));
        }
        std::sort(sorted_mos.rbegin(), sorted_mos.rend());

        std::vector<int> occ_inact, occ_act, vir_inact, vir_act;

        if (avas_num_active_occ + avas_num_active_vir > 0) {
            outfile->Printf(
                "\n  AVAS selection based on number of occupied/virtual MOs requested\n");
            for (const auto& mo_tuple : sorted_mos) {
                bool is_occ = std::get<1>(mo_tuple);
                int p = std::get<2>(mo_tuple);
                if (is_occ) {
                    if (occ_act.size() < avas_num_active_occ) {
                        occ_act.push_back(p);
                    } else {
                        occ_inact.push_back(p);
                    }
                } else {
                    if (vir_act.size() < avas_num_active_vir) {
                        vir_act.push_back(p);
                    } else {
                        vir_inact.push_back(p);
                    }
                }
            }
        } else if (avas_num_active > 0) {
            outfile->Printf("\n  AVAS selection based on number of MOs requested\n");
            for (int n = 0; n < avas_num_active; ++n) {
                bool is_occ = std::get<1>(sorted_mos[n]);
                int p = std::get<2>(sorted_mos[n]);
                if (is_occ) {
                    occ_act.push_back(p);
                } else {
                    vir_act.push_back(p);
                }
            }
            for (int n = avas_num_active; n < nmo; ++n) {
                bool is_occ = std::get<1>(sorted_mos[n]);
                int p = std::get<2>(sorted_mos[n]);
                if (is_occ) {
                    occ_inact.push_back(p);
                } else {
                    vir_inact.push_back(p);
                }
            }
        } else {
            // tollerance on the sum of singular values (for border cases, e.g. sigma = 1.0)
            double sum_tollerance = 1.0e-9;
            // threshold for including an orbital
            double include_threshold = 1.0e-6;
            outfile->Printf("\n  AVAS selection based cumulative threshold (sigma)\n");
            double s_act_sum = 0.0;
            for (const auto& mo_tuple : sorted_mos) {
                double sigma = std::get<0>(mo_tuple);
                bool is_occ = std::get<1>(mo_tuple);
                int p = std::get<2>(mo_tuple);

                s_act_sum += sigma;
                double fraction = s_act_sum / s_sum;

                // decide if this is orbital is active depending on the ratio of
                // the
                // partial sum of singular values and the total sum of singular
                // values
                if ((fraction <= avas_sigma + sum_tollerance) and
                    (std::fabs(sigma) > include_threshold)) {
                    if (is_occ) {
                        occ_act.push_back(p);
                    } else {
                        vir_act.push_back(p);
                    }
                } else {
                    if (is_occ) {
                        occ_inact.push_back(p);
                    } else {
                        vir_inact.push_back(p);
                    }
                }
            }
        }

        outfile->Printf("\n  ==> AVAS MOs Information <==");
        outfile->Printf("\n    Number of inactive occupied MOs: %6d", occ_inact.size());
        outfile->Printf("\n    Number of active occupied MOs:   %6d", occ_act.size());
        outfile->Printf("\n    Number of active virtual MOs:    %6d", vir_act.size());
        outfile->Printf("\n    Number of inactive virtual MOs:  %6d", vir_inact.size());
        outfile->Printf("\n");
        outfile->Printf("\n    restricted_docc = [%d]", occ_inact.size());
        outfile->Printf("\n    active          = [%d]", occ_act.size() + vir_act.size());
        outfile->Printf("\n");

        outfile->Printf("\n  Atomic Valence MOs:\n");
        outfile->Printf("    ============================\n");
        outfile->Printf("    Occupation  MO   <phi|P|phi>\n");
        outfile->Printf("    ----------------------------\n");
        for (int i : occ_act) {
            outfile->Printf("      %1d       %4d    %.6f\n", 2, i + 1, sigmaocc->get(i));
        }
        for (int i : vir_act) {
            outfile->Printf("      %1d       %4d    %.6f\n", 0, nocc + i + 1, sigmavir->get(i));
        }
        outfile->Printf("    ============================\n");

        // occupied inactive
        auto Coi = semicanonicalize_block(ref_wfn, Ca_tilde, occ_inact, 0);
        auto Coa = semicanonicalize_block(ref_wfn, Ca_tilde, occ_act, 0);
        auto Cvi = semicanonicalize_block(ref_wfn, Ca_tilde, vir_inact, nocc);
        auto Cva = semicanonicalize_block(ref_wfn, Ca_tilde, vir_act, nocc);

        auto Ca_tilde_prime = std::make_shared<psi::Matrix>("C tilde prime", nso, nmo);

        int offset = 0;
        for (auto& C_block : {Coi, Coa, Cva, Cvi}) {
            int nmo_block = C_block->ncol();
            for (int i = 0; i < nmo_block; ++i) {
                for (int mu = 0; mu < nso; ++mu) {
                    double value = C_block->get(mu, i);
                    Ca_tilde_prime->set(mu, offset, value);
                }
                offset += 1;
            }
        }

        psi::SharedMatrix Fa = ref_wfn->Fa(); // get Fock matrix
        psi::SharedMatrix Fa_mo = Fa->clone();
        Fa_mo->transform(Ca_tilde_prime);

        // Update both the alpha and beta orbitals
        // This assumes a restricted MO set
        // TODO: generalize to unrestricted references
        ref_wfn->Ca()->copy(Ca_tilde_prime);
        ref_wfn->Cb()->copy(Ca_tilde_prime);
    }
}

void make_embedding(psi::SharedWavefunction ref_wfn, psi::Options& options, psi::SharedMatrix Pf, std::shared_ptr<MOSpaceInfo> mo_space_info) {
	outfile->Printf("\n ------ Orbital Localization and Embedding ------ \n");

	// 1. Get necessary information
	outfile->Printf("\n Reading options ------ \n");
	double thresh = options.get_double("THRESHOLD");
	int frz_sys_docc = options.get_int("FROZEN_SYS_DOCC");
	int frz_sys_uocc = options.get_int("FROZEN_SYS_UOCC");

	std::shared_ptr<PSIO> psio(_default_psio_lib_);
	if (!ref_wfn)
		throw PSIEXCEPTION("SCF has not been run yet!");

	// 2. Apply projector to rotate the orbitals
	if (Pf) {
		outfile->Printf("\n Find fragment projector Pf ------ \n");
		Dimension nmopi = ref_wfn->nmopi();
		Dimension zeropi = nmopi - nmopi;
		int nirrep = ref_wfn->nirrep();
		if (nirrep > 1) {
			throw PSIEXCEPTION("Fragment projection works only without symmetry! (symmetry C1)");
		}

		// Get information of rocc, actv and rvir from MOSpaceInfo
		Dimension frzopi = mo_space_info->get_dimension("FROZEN_DOCC");
		Dimension nroccpi = mo_space_info->get_dimension("RESTRICTED_DOCC");
		Dimension actv_a = mo_space_info->get_dimension("ACTIVE");
		Dimension nrvirpi = mo_space_info->get_dimension("RESTRICTED_UOCC");
		Dimension frzvpi = mo_space_info->get_dimension("FROZEN_UOCC");

		outfile->Printf("\n MOSpaceInfo read ------ \n");

		// Define corresponding blocks (slices)
		nroccpi = nroccpi + frzopi; // Merge frozen and restricted occupied orbitals to take care of initial frozen core
		Slice occ(zeropi, nroccpi);
		Slice vir(nroccpi + actv_a, nmopi);
		Slice actv(nroccpi, nroccpi + actv_a);

		// Transform Pf to MO basis
		SharedMatrix Ca_ori = ref_wfn->Ca();
		Pf->transform(Ca_ori);

		outfile->Printf("\n Projector transformed ------ \n");

		// Diagonalize Pf_pq for occ and vir part, respectively.
		SharedMatrix P_oo = Pf->get_block(occ, occ);
		SharedMatrix Uo(new Matrix("Uo", nirrep, nroccpi, nroccpi));
		SharedVector lo(new Vector("lo", nirrep, nroccpi));
		P_oo->diagonalize(Uo, lo, descending);
		lo->print();

		SharedMatrix P_vv = Pf->get_block(vir, vir);
		SharedMatrix Uv(new Matrix("Uv", nirrep, nrvirpi, nrvirpi));
		SharedVector lv(new Vector("lv", nirrep, nrvirpi));
		P_vv->diagonalize(Uv, lv, descending);
		lv->print();

		SharedMatrix U_all(new Matrix("U with Pab", nirrep, nmopi, nmopi));
		U_all->set_block(occ, occ, Uo);
		U_all->set_block(vir, vir, Uv);

		// Rotate MOs
		ref_wfn->Ca()->copy(psi::linalg::doublet(Ca_ori, U_all, false, false));
		outfile->Printf("\n MOs rotated ------ \n");

		// Based on threshold or num_occ/num_vir, decide the partition
		std::vector<int> index_A_occ = {};
		std::vector<int> index_A_vir = {};
		std::vector<int> index_B_occ = {};
		std::vector<int> index_B_vir = {};
		std::vector<int> index_actv = {};

		if (options.get_str("REFERENCE") == "CASSCF") {
			for (int i = 0; i < actv_a[0]; ++i) {
				index_actv.push_back(nroccpi[0] + i);
			}
		}

		if (options.get_str("CUTOFF_BY") == "THRESHOLD") {
			for (int i = 0; i < nroccpi[0]; i++) {
				if (lo->get(0, i) > thresh) {
					index_A_occ.push_back(i);
					outfile->Printf("\n Occupied orbital %d is partitioned to A with eigenvalue %8.8f",
						i, lo->get(0, i));
				}
				else {
					index_B_occ.push_back(i);
				}
			}
			for (int i = 0; i < nrvirpi[0]; i++) {
				if (lv->get(0, i) > thresh) {
					index_A_vir.push_back(i);
					outfile->Printf("\n Virtual orbital %d is partitioned to A with eigenvalue %8.8f",
						i, lv->get(0, i));
				}
				else {
					index_B_vir.push_back(i);
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
				cum_l_o = tmp / sum_lo;
				if (cum_l_o < thresh) {
					index_A_occ.push_back(i);
					outfile->Printf("\n Occupied orbital %d is partitioned to A with cumulative eigenvalue %8.8f",
						i, cum_l_o);
				}
				else {
					index_B_occ.push_back(i);
				}
			}
			tmp = 0.0;
			double cum_l_v = 0.0;
			for (int i = 0; i < nrvirpi[0]; i++) {
				tmp += lv->get(0, i);
				cum_l_v = tmp / sum_lv;
				if (cum_l_v < thresh) {
					index_A_vir.push_back(i);
					outfile->Printf("\n Virtual orbital %d is partitioned to A with cumulative eigenvalue %8.8f",
						i, cum_l_v);
				}
				else {
					index_B_vir.push_back(i);
				}
			}
		}

		int num_Ao = index_A_occ.size();
		int num_Bo = index_B_occ.size();
		int num_Av = index_A_vir.size();
		int num_Bv = index_B_vir.size();

		outfile->Printf("\n Build and semi-canonocalize new orbitals ------ \n");
		SharedMatrix Ca_tilde(ref_wfn->Ca()->clone());

		// Build and semi-canonicalize BO, AO, AV and BV blocks
		auto C_bo = semicanonicalize_block(ref_wfn, Ca_tilde, index_B_occ, 0);
		auto C_ao = semicanonicalize_block(ref_wfn, Ca_tilde, index_A_occ, 0);
		auto C_av = semicanonicalize_block(ref_wfn, Ca_tilde, index_A_vir, 0);
		auto C_bv = semicanonicalize_block(ref_wfn, Ca_tilde, index_B_vir, 0);

		SharedMatrix C_A(new Matrix("Active_coeff_block", nirrep, nmopi, actv_a));

		// Build the active block
		if (options.get_str("REFERENCE") == "CASSCF") {
			if (options.get_bool("SEMICANON") == true) {
				// Read active orbitals from original Ca and semi-canonicalize
				C_A->copy(semicanonicalize_block(ref_wfn, Ca_ori, index_actv, 0));
			}
			else {
				// Read active orbitals from original Ca
				for (int i = 0; i < actv_a[0]; ++i) {
					C_A->set_column(0, i, Ca_ori->get_column(0, nroccpi[0] + i));
				}
			}
		}

		// Form new C matrix
		SharedMatrix Ca_Rt(new Matrix("Ca rotated tilde", nirrep, nmopi, nmopi));

		int offset = 0;
		for (auto& C_block : { C_bo, C_ao, C_A, C_av, C_bv }) {
			int nmo_block = C_block->ncol();
			for (int i = 0; i < nmo_block; ++i) {
				for (int mu = 0; mu < nmopi[0]; ++mu) {
					double value = C_block->get(mu, i);
					Ca_Rt->set(mu, offset, value);
				}
				offset += 1;
			}
		}

		psi::SharedMatrix Fa = ref_wfn->Fa(); // get AO basis Fock matrix
		psi::SharedMatrix Fa_mo = Fa->clone();
		Fa_mo->transform(Ca_Rt);

		// Update both the alpha and beta orbitals
		ref_wfn->Ca()->copy(Ca_Rt);
		ref_wfn->Cb()->copy(Ca_Rt);

		// Write new MOSpaceInfo
		std::map<std::string, std::vector<size_t>> mo_space_map;

		// Frozen docc space
		size_t freeze_o = static_cast<size_t>(num_Bo + frz_sys_docc); // Add the original frozen core to B
		std::vector<size_t> fo_vec;
		fo_vec.push_back(freeze_o);
		mo_space_map["FROZEN_DOCC"] = fo_vec;

		// Restricted docc space
		size_t ro = static_cast<size_t>(num_Ao);
		std::vector<size_t> ro_vec;
		ro_vec.push_back(ro);
		mo_space_map["RESTRICTED_DOCC"] = ro_vec;

		// Active space
		size_t a = static_cast<size_t>(actv_a[0]);
		std::vector<size_t> a_vec;
		a_vec.push_back(a);
		mo_space_map["ACTIVE"] = a_vec;

		// Restricted uocc space
		size_t rv = static_cast<size_t>(num_Av);
		std::vector<size_t> rv_vec;
		rv_vec.push_back(rv);
		mo_space_map["RESTRICTED_UOCC"] = rv_vec;

		// Frozen uocc space
		size_t freeze_v = static_cast<size_t>(num_Bv + frz_sys_uocc); // Add the original frozen core to B
		std::vector<size_t> fv_vec;
		fv_vec.push_back(freeze_v);
		mo_space_map["FROZEN_UOCC"] = fv_vec;

		// Write new MOSpaceInfo
		std::vector<size_t> reorder;
		mo_space_info = make_mo_space_info_map(ref_wfn, mo_space_map, reorder);

		// Print summary of embedding MO spaces
		outfile->Printf("    ============================\n");
		outfile->Printf("\n  ==> EMBEDDING MO space Information <==");
		outfile->Printf("\n    frozen_docc     = [%d]", mo_space_info->get_dimension("FROZEN_DOCC")[0]);
		outfile->Printf("\n    restricted_docc = [%d]", mo_space_info->get_dimension("RESTRICTED_DOCC")[0]);
		outfile->Printf("\n    active          = [%d]", mo_space_info->get_dimension("ACTIVE")[0]);
		outfile->Printf("\n    restricted_uocc = [%d]", mo_space_info->get_dimension("RESTRICTED_UOCC")[0]);
		outfile->Printf("\n    frozen_uocc     = [%d]", mo_space_info->get_dimension("FROZEN_UOCC")[0]);
		outfile->Printf("\n");
		outfile->Printf("    ============================\n");
	}
	else {
		throw PSIEXCEPTION("No projector (matrix) found!");
	}
}

psi::SharedMatrix semicanonicalize_block(psi::SharedWavefunction ref_wfn, psi::SharedMatrix C_tilde,
                                         std::vector<int>& mos, int offset) {
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
    auto Foi = psi::linalg::triplet(C_block, ref_wfn->Fa(), C_block, true, false, false);

    auto U_block = std::make_shared<psi::Matrix>("U block", nmo_block, nmo_block);
    auto epsilon_block = std::make_shared<Vector>("epsilon block", nmo_block);
    Foi->diagonalize(U_block, epsilon_block);
    auto C_block_prime = psi::linalg::doublet(C_block, U_block);
    return C_block_prime;
}
} // namespace forte

// outfile->Printf("\n  Orbital overlap with ao subspace:\n");
// outfile->Printf("    ========================\n");
// outfile->Printf("    Irrep   MO   <phi|P|phi>\n");
// outfile->Printf("    ------------------------\n");
// for (int i = 0; i < nocc; i++) {
//    outfile->Printf("      %1d   %4d    %.6f\n", 2, i + 1,
//                    sigmaocc->get(i));
//}
// for (int i = 0; i < nvir; i++) {
//    outfile->Printf("      %1d   %4d    %.6f\n", 0, nocc + i + 1,
//                    sigmavir->get(i));
//}
// outfile->Printf("    ========================\n");
