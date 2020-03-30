/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "psi4/libmints/local.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"


#include "base_classes/forte_options.h"
#include "unpaired_density.h"
#include "localize.h"

using namespace psi;

namespace forte {

UPDensity::UPDensity(std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ForteOptions> options, psi::SharedMatrix Ua,
                     psi::SharedMatrix Ub)
    : options_(options), ints_(ints), mo_space_info_(mo_space_info), Uas_(Ua), Ubs_(Ub) {}

void UPDensity::compute_unpaired_density(std::vector<double>& oprdm_a,
                                         std::vector<double>& oprdm_b) {
    // TODO: re-enable this code
    //    psi::Dimension nactpi = mo_space_info_->dimension("ACTIVE");
    //    psi::Dimension nmopi = mo_space_info_->dimension("ALL");
    //    psi::Dimension ncmopi = mo_space_info_->dimension("CORRELATED");
    //    size_t nirrep = ints_->nirrep();
    //    psi::Dimension rdocc = mo_space_info_->dimension("RESTRICTED_DOCC");
    //    psi::Dimension fdocc = mo_space_info_->dimension("FROZEN_DOCC");
    //
    //    size_t nact = nactpi.sum();
    //
    //    // First compute natural orbitals
    //    std::shared_ptr<psi::Matrix> opdm_a(new psi::Matrix("OPDM_A", nirrep, nactpi, nactpi));
    //    std::shared_ptr<psi::Matrix> opdm_b(new psi::Matrix("OPDM_B", nirrep, nactpi, nactpi));
    //
    //    // Put 1-RDM into Shared matrix
    //    int offset = 0;
    //    for (size_t h = 0; h < nirrep; ++h) {
    //        for (int u = 0; u < nactpi[h]; ++u) {
    //            for (int v = 0; v < nactpi[h]; ++v) {
    //                opdm_a->set(h, u, v, oprdm_a[(u + offset) * nact + v + offset]);
    //                opdm_b->set(h, u, v, oprdm_b[(u + offset) * nact + v + offset]);
    //            }
    //        }
    //        offset += nactpi[h];
    //    }
    //    //    opdm_a->transform(Uas_);
    //    //    opdm_b->transform(Ubs_);
    //
    //    // Diagonalize the 1-RDMs
    //    psi::SharedVector OCC_A(new psi::Vector("ALPHA NOCC", nirrep, nactpi));
    //    psi::SharedVector OCC_B(new psi::Vector("BETA NOCC", nirrep, nactpi));
    //    psi::SharedMatrix NO_A(new psi::Matrix(nirrep, nactpi, nactpi));
    //    psi::SharedMatrix NO_B(new psi::Matrix(nirrep, nactpi, nactpi));
    //
    //    opdm_a->diagonalize(NO_A, OCC_A, descending);
    //    opdm_b->diagonalize(NO_B, OCC_B, descending);
    //
    //    // Build the transformation matrix
    //    // Only build density for active orbitals
    //    psi::SharedMatrix Ua(new psi::Matrix("Ua", nmopi, nmopi));
    //    psi::SharedMatrix Ub(new psi::Matrix("Ub", nmopi, nmopi));
    //
    //    Ua->zero();
    //    Ub->zero();
    //
    //    // This Ua/Ub build will ensure that the density only includes active orbitals
    //    // If natural orbitals are desired, change the 1.0 to NO_a->get(p,q)
    //    for (size_t h = 0; h < nirrep; ++h) {
    //        size_t irrep_offset = fdocc[h] + rdocc[h];
    //        for (int p = 0; p < nactpi[h]; ++p) {
    //            // for (int q = 0; q < nactpi[h]; ++q) {
    //            Ua->set(h, p + irrep_offset, p + irrep_offset, 1.0);
    //            Ub->set(h, p + irrep_offset, p + irrep_offset, 1.0);
    //            //  }
    //        }
    //    }
    //
    //    /// ** Compute atom-based unpaired contributions
    //
    //    // ** This will be done in a completely localized basis
    //    // ** This code has only been tested for pz (pi) orbitals, beware!
    //
    //    psi::SharedMatrix Ua_act(new psi::Matrix(nact, nact));
    //
    //    // relocalize to atoms
    //
    //    // Grab matrix that takes the transforms from the NO basis to our local basis
    //    auto loc = std::make_shared<LOCALIZE>(options_, ints_, mo_space_info_);
    //    loc->full_localize();
    //    Ua_act = loc->get_U()->clone();
    //    psi::SharedMatrix Noinv(NO_A->clone());
    //    Noinv->invert();
    //    psi::SharedMatrix Ua_act_r = psi::linalg::doublet(Noinv, Ua_act, false, false);
    //
    //    // Compute sum(p,i) n_i * ( 1 - n_i ) * (U_p,i)^2
    //    double total = 0.0;
    //    std::vector<double> scales(nact);
    //    for (size_t i = 0; i < nact; ++i) {
    //        double value = 0.0;
    //        for (size_t h = 0; h < nirrep; ++h) {
    //            for (int p = 0; p < nactpi[h]; ++p) {
    //                //                double n_p = OCC_A->get(p) + OCC_B->get(p);
    //                double n_p = OCC_A->get(p);
    //                double up_el = n_p * (1.0 - n_p);
    //
    //                value += up_el * Ua_act_r->get(p, i) * Ua_act_r->get(p, i);
    //            }
    //        }
    //        scales[i] = value;
    //        total += value;
    //        outfile->Printf("\n  MO %d:  %1.6f", i, value);
    //    }
    //    outfile->Printf("\n  Total unpaired electrons: %1.4f", total);
    //
    //    // Build the density using scaled columns of C
    //
    //    psi::SharedMatrix Ca = ints_->Ca();
    //    psi::SharedMatrix Cb = ints_->Cb();
    //
    //    psi::SharedMatrix Ca_new = psi::linalg::doublet(Ca->clone(), Ua, false, false);
    //    psi::SharedMatrix Cb_new = psi::linalg::doublet(Cb->clone(), Ub, false, false);
    //
    //    for (size_t h = 0; h < nirrep; ++h) {
    //        int offset = fdocc[h] + rdocc[h];
    //        for (int p = 0; p < nactpi[h]; ++p) {
    //            // double n_p = OCC_A->get(p) + OCC_B->get(p);
    //            // double up_el = n_p * (2.0 - n_p);
    //            double up_el = scales[p];
    //            //            double up_el = n_p * n_p * (2.0 - n_p ) * (2.0 - n_p);
    //            //            outfile->Printf("\n  Weight for orbital (%d,%d): %1.5f", h, p,
    //            up_el); Ca_new->scale_column(h, offset + p, up_el); Cb_new->scale_column(h, offset
    //            + p, up_el);
    //        }
    //    }

    psi::Dimension nactpi = mo_space_info_->dimension("ACTIVE");
    psi::Dimension nmopi = mo_space_info_->dimension("ALL");
    psi::Dimension ncmopi = mo_space_info_->dimension("CORRELATED");
    size_t nirrep = ints_->nirrep();
    psi::Dimension rdocc = mo_space_info_->dimension("RESTRICTED_DOCC");
    psi::Dimension fdocc = mo_space_info_->dimension("FROZEN_DOCC");

    size_t nact = nactpi.sum();

    // First compute natural orbitals
    std::shared_ptr<psi::Matrix> opdm_a(new psi::Matrix("OPDM_A", nirrep, nactpi, nactpi));
    std::shared_ptr<psi::Matrix> opdm_b(new psi::Matrix("OPDM_B", nirrep, nactpi, nactpi));

    // Put 1-RDM into Shared matrix
    int offset = 0;
    for (size_t h = 0; h < nirrep; ++h) {
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

    // Diagonalize the 1-RDMs
    psi::SharedVector OCC_A(new psi::Vector("ALPHA NOCC", nirrep, nactpi));
    psi::SharedVector OCC_B(new psi::Vector("BETA NOCC", nirrep, nactpi));
    psi::SharedMatrix NO_A(new psi::Matrix(nirrep, nactpi, nactpi));
    psi::SharedMatrix NO_B(new psi::Matrix(nirrep, nactpi, nactpi));

    opdm_a->diagonalize(NO_A, OCC_A, descending);
    opdm_b->diagonalize(NO_B, OCC_B, descending);

    // Build the transformation matrix
    // Only build density for active orbitals
    psi::SharedMatrix Ua(new psi::Matrix("Ua", nmopi, nmopi));
    psi::SharedMatrix Ub(new psi::Matrix("Ub", nmopi, nmopi));

    Ua->zero();
    Ub->zero();

    // This Ua/Ub build will ensure that the density only includes active orbitals
    // If natural orbitals are desired, change the 1.0 to NO_a->get(p,q)
    for (size_t h = 0; h < nirrep; ++h) {
        size_t irrep_offset = fdocc[h] + rdocc[h];
        for (int p = 0; p < nactpi[h]; ++p) {
            // for (int q = 0; q < nactpi[h]; ++q) {
            Ua->set(h, p + irrep_offset, p + irrep_offset, 1.0);
            Ub->set(h, p + irrep_offset, p + irrep_offset, 1.0);
            //  }
        }
    }

    /// ** Compute atom-based unpaired contributions

    // ** This will be done in a completely localized basis
    // ** This code has only been tested for pz (pi) orbitals, beware!

    psi::SharedMatrix Ua_act(new psi::Matrix(nact, nact));

    // relocalize to atoms

    // Grab matrix that takes the transforms from the NO basis to our local basis
    auto loc = std::make_shared<Localize>(options_, ints_, mo_space_info_);

    std::vector<size_t> actmo = mo_space_info_->absolute_mo("ACTIVE");
    std::vector<int> loc_mo(2);
    loc_mo[0] = static_cast<int>(actmo[0]);
    loc_mo[1] = static_cast<int>(actmo.back());
    loc->set_orbital_space(loc_mo);

    loc->compute_transformation();
    Ua_act = loc->get_Ua()->clone();
    psi::SharedMatrix Noinv(NO_A->clone());
    Noinv->invert();
    psi::SharedMatrix Ua_act_r = psi::linalg::doublet(Noinv, Ua_act, false, false);

    // Compute sum(p,i) n_i * ( 1 - n_i ) * (U_p,i)^2
    double total = 0.0;
    std::vector<double> scales(nact);
    for (size_t i = 0; i < nact; ++i) {
        double value = 0.0;
        for (size_t h = 0; h < nirrep; ++h) {
            for (int p = 0; p < nactpi[h]; ++p) {
                //                double n_p = OCC_A->get(p) + OCC_B->get(p);
                double n_p = OCC_A->get(p);
                double up_el = n_p * (1.0 - n_p);

                value += up_el * Ua_act_r->get(p, i) * Ua_act_r->get(p, i);
            }
        }
        scales[i] = value;
        total += value;
        outfile->Printf("\n  MO %d:  %1.6f", i, value);
    }
    outfile->Printf("\n  Total unpaired electrons: %1.4f", total);

    // Build the density using scaled columns of C

    psi::SharedMatrix Ca = ints_->Ca();
    psi::SharedMatrix Cb = ints_->Cb();

    psi::SharedMatrix Ca_new = psi::linalg::doublet(Ca->clone(), Ua, false, false);
    psi::SharedMatrix Cb_new = psi::linalg::doublet(Cb->clone(), Ub, false, false);

    for (size_t h = 0; h < nirrep; ++h) {
        int offset = fdocc[h] + rdocc[h];
        for (int p = 0; p < nactpi[h]; ++p) {
            // double n_p = OCC_A->get(p) + OCC_B->get(p);
            // double up_el = n_p * (2.0 - n_p);
            double up_el = scales[p];
            //            double up_el = n_p * n_p * (2.0 - n_p ) * (2.0 - n_p);
            //            outfile->Printf("\n  Weight for orbital (%d,%d): %1.5f", h, p, up_el);
            Ca_new->scale_column(h, offset + p, up_el);
            Cb_new->scale_column(h, offset + p, up_el);
        }
    }

    psi::SharedMatrix Da = ints_->wfn()->Da();
    psi::SharedMatrix Db = ints_->wfn()->Db();

    // psi::SharedMatrix Da_new(new psi::Matrix("Da_new", nmopi, nmopi));
    // psi::SharedMatrix Db_new(new psi::Matrix("Db_new", nmopi, nmopi));

    // Da_new->gemm(false, true, 1.0, Ca_new, Ca_new, 0.0);
    // Db_new->gemm(false, true, 1.0, Cb_new, Cb_new, 0.0);

    // Da->copy(Da_new);
    // Db->copy(Db_new);

    // This is for IAOs, don't really need it
    //    std::shared_ptr<IAOBuilder> IAO =
    //            IAOBuilder::build(wfn_->basisset(),
    //                              wfn_->get_basisset("MINAO_BASIS"), Ca, options_);
    //    outfile->Printf("\n  Computing IAOs\n");
    //    std::map<std::string, psi::SharedMatrix> iao_info = IAO->build_iaos();
    //    psi::SharedMatrix iao_orbs(iao_info["A"]->clone());
    //
    //    psi::SharedMatrix Cainv(Ca->clone());
    //    Cainv->invert();
    //    psi::SharedMatrix iao_coeffs = psi::linalg::doublet(Cainv, iao_orbs, false, false);
    //
    //    size_t new_dim = iao_orbs->colspi()[0];
    //    size_t new_dim2 = new_dim * new_dim;
    //    size_t new_dim3 = new_dim2 * new_dim;
    //
    //    auto labels = IAO->print_IAO(iao_orbs, new_dim, nmo, wfn_);
    //
    //    outfile->Printf("\n label size: %zu", labels.size());
    //
    //    std::vector<int> IAO_inds;
    //    for (int i = 0; i < labels.size(); ++i) {
    //        std::string label = labels[i];
    //        if (label.find("z") != std::string::npos) {
    //            IAO_inds.push_back(i);
    //        }
    //    }
    //    std::vector<size_t> active_mo = mo_space_info_->absolute_mo("ACTIVE");
    //    for (int i = 0; i < nact; ++i) {
    //        int idx = IAO_inds[i];
    //        outfile->Printf("\n Using IAO %d", idx);
    //        for (int j = 0; j < nact; ++j) {
    //            int mo = active_mo[j];
    //            Ua_act->set(j, i, iao_coeffs->get(mo, idx));
    //        }
    //    }
}

UPDensity::~UPDensity() {}
} // namespace forte
