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

void make_embedding(psi::SharedWavefunction ref_wfn, psi::Options& options, psi::SharedMatrix Pf, MOSpaceInfo mo_space_info) {
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

	// 2. Apply projector to rotate the orbitals
        if (Pf) {
			Dimension nmopi = ref_wfn->nmopi();
			Dimension zeropi = nmopi - nmopi;
			int nirrep = ref_wfn->nirrep();
			if (nirrep > 1) {
				throw PSIEXCEPTION("Fragment projection works only without symmetry! (symmetry C1)");
			}

			// Get information of rocc, actv and rvir from MOSpaceInfo
			Dimension nroccpi = mo_space_info.get_dimension("RESTRICTED_DOCC");
			Dimension actv_a = mo_space_info.get_dimension("ACTIVE");
			Dimension nrvirpi = mo_space_info.get_dimension("RESTRICTED_UOCC");;

			// Create corresponding blocks (slices)
			Slice occ(zeropi, nroccpi);
			Slice vir(nroccpi + actv_a, nmopi);
			Slice actv(nroccpi, nroccpi + actv_a);

			// Transform Pf to MO basis
			SharedMatrix Ca_t = ref_wfn->Ca();
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

			// Based on threshold or num_occ/num_vir, decide the partition
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
					cum_l_o = tmp / sum_lo;
					if (cum_l_o < thresh) {
						index_trace_occ.push_back(i);
						outfile->Printf("\n Occupied orbital %d is partitioned to A with cumulative eigenvalue %8.8f",
							i, cum_l_o);
					}
				}
				tmp = 0.0;
				double cum_l_v = 0.0;
				for (int i = 0; i < nrvirpi[0]; i++) {
					tmp += lv->get(0, i);
					cum_l_v = tmp / sum_lv;
					if (cum_l_v < thresh) {
						index_trace_vir.push_back(i);
						outfile->Printf("\n Virtual orbital %d is partitioned to A with cumulative eigenvalue %8.8f",
							i, cum_l_v);
					}
				}
			}

			// Build new Ca with the selected blocks

			// Semi-canonicalize with functions

			// Write new MOSpaceInfo

        }
	else {
		throw PSIEXCEPTION("No projector (matrix) found!");
	}
}
}
