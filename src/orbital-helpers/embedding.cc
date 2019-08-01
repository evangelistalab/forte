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

	// 2. Apply projector
        if (Pf) {
                //Embedding codes will be here
        }
	else {
		throw PSIEXCEPTION("No projector (matrix) found!");
	}
}
}
