/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libmints/molecule.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"

#include "psi4/libpsi4util/PsiOutStream.h"

#include "psi4/libmints/local.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"

#include "unpaired_density.h"

namespace psi {
namespace forte {

UPDensity::UPDensity(std::shared_ptr<Wavefunction> wfn, std::shared_ptr<MOSpaceInfo> mo_space_info, SharedMatrix Ua, SharedMatrix Ub)
    : wfn_(wfn), mo_space_info_(mo_space_info), Uas_(Ua), Ubs_(Ub) {}

void UPDensity::compute_unpaired_density(std::vector<double>& oprdm_a,
                                         std::vector<double>& oprdm_b) {

    Dimension nactpi = mo_space_info_->get_dimension("ACTIVE");
    Dimension nmopi = wfn_->nmopi();
    Dimension ncmopi = mo_space_info_->get_dimension("CORRELATED");
    size_t nirrep = wfn_->nirrep();
    Dimension rdocc = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    Dimension fdocc = mo_space_info_->get_dimension("FROZEN_DOCC");

    size_t nact = nactpi.sum();

    // First compute natural orbitals
    std::shared_ptr<Matrix> opdm_a(new Matrix("OPDM_A", nirrep, nactpi, nactpi));
    std::shared_ptr<Matrix> opdm_b(new Matrix("OPDM_B", nirrep, nactpi, nactpi));

    int offset = 0;
    for (int h = 0; h < nirrep; ++h) {
        for (int u = 0; u < nactpi[h]; ++u) {
            for (int v = 0; v < nactpi[h]; ++v) {
                opdm_a->set(h, u, v, oprdm_a[(u + offset) * nact + v + offset]);
                opdm_b->set(h, u, v, oprdm_b[(u + offset) * nact + v + offset]);
            }
        }
        offset += nactpi[h];
    }
//    opdm_a->transform(Uas_);
//    opdm_b->transform(Ubs_);

    SharedVector OCC_A(new Vector("ALPHA NOCC", nirrep, nactpi));
    SharedVector OCC_B(new Vector("BETA NOCC", nirrep, nactpi));
    SharedMatrix NO_A(new Matrix(nirrep, nactpi, nactpi));
    SharedMatrix NO_B(new Matrix(nirrep, nactpi, nactpi));

    opdm_a->diagonalize(NO_A, OCC_A, descending);
    opdm_b->diagonalize(NO_B, OCC_B, descending);

    // Build the transformation matrix
    // Only build density for active orbitals
    SharedMatrix Ua(new Matrix("Ua", nmopi, nmopi));
    SharedMatrix Ub(new Matrix("Ub", nmopi, nmopi));

//    Ua->identity();
//    Ub->identity();
    Ua->zero();
    Ub->zero();

    for (int h = 0; h < nirrep; ++h) {
        size_t irrep_offset = fdocc[h] + rdocc[h];

        for (int p = 0; p < nactpi[h]; ++p) {
            for (int q = 0; q < nactpi[h]; ++q) {
                Ua->set(h, p + irrep_offset, q + irrep_offset, NO_A->get(h, p, q));
                Ub->set(h, p + irrep_offset, q + irrep_offset, NO_B->get(h, p, q));
            }
        }
    }

    
    SharedMatrix tmp_a(new Matrix("ta", nmopi, nmopi));
    SharedMatrix tmp_b(new Matrix("tb", nmopi, nmopi));

    tmp_a->gemm(true,false,1.0,Uas_,Ua,0.0);
    tmp_b->gemm(true,false,1.0,Ubs_,Ub,0.0);

    // Transform the orbital coefficients
    SharedMatrix Ca = wfn_->Ca();
    SharedMatrix Cb = wfn_->Cb();

    SharedMatrix Ca_new(new Matrix("Ca_new", nmopi, nmopi));
    SharedMatrix Cb_new(new Matrix("Cb_new", nmopi, nmopi));

    Ca_new->zero();
    Cb_new->zero();

    Ca_new->gemm(false, false, 1.0, Ca, tmp_a, 0.0);
    Cb_new->gemm(false, false, 1.0, Cb, tmp_b, 0.0);

//    Ca->copy(Ca_new);
//    Cb->copy(Cb_new);
    // Scale active NOs by unpaired-e criteria
    for (int h = 0; h < nirrep; ++h) {
        int offset = fdocc[h] + rdocc[h];
        for (int p = 0; p < nactpi[h]; ++p) {
            double n_p = OCC_A->get(p) + OCC_B->get(p);

            double up_el = n_p * (2.0 - n_p);
//            double up_el = n_p * n_p * (2.0 - n_p ) * (2.0 - n_p);
            outfile->Printf("\n  Weight for orbital (%d,%d): %1.5f", h, p, up_el);
            Ca_new->scale_column(h, offset + p, up_el);
            Cb_new->scale_column(h, offset + p, up_el);
        }
    }

    // Form the new density

    SharedMatrix Da = wfn_->Da();
    SharedMatrix Db = wfn_->Db();

    SharedMatrix Da_new(new Matrix("Da_new", nmopi, nmopi));
    SharedMatrix Db_new(new Matrix("Db_new", nmopi, nmopi));

    Da_new->gemm(false, true, 1.0, Ca_new, Ca_new, 0.0);
    Db_new->gemm(false, true, 1.0, Cb_new, Cb_new, 0.0);

    Da->copy(Da_new);
    Db->copy(Db_new);
}

UPDensity::~UPDensity() {}
}
} // End Namespaces
