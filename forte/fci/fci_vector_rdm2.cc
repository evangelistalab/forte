/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libqt/qt.h"

#include "base_classes/mo_space_info.h"
#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"
#include "sparse_ci/determinant.h"

#include "fci_vector.h"
#include "fci_solver.h"
#include "binary_graph.hpp"
#include "string_lists.h"
#include "string_address.h"

extern int fci_debug_level;

using namespace psi;

namespace forte {

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
static FCIVector::compute_rdms(FCIVector& C_left, FCIVector& C_right, int max_order) {
    std::vector<double> rdm_timing;

    size_t na = alfa_address_->nones();
    size_t nb = beta_address_->nones();

    if (max_order >= 1) {
        local_timer t;
        if (na >= 1)
            compute_1rdm(C_left, C_right, opdm_a_, true);
        if (nb >= 1)
            compute_1rdm(C_left, C_right, opdm_b_, false);
        rdm_timing.push_back(t.get());
    }

    if (max_order >= 2) {
        local_timer t;
        if (na >= 2)
            compute_2rdm_aa(C_left, C_right, tpdm_aa_, true);
        if (nb >= 2)
            compute_2rdm_aa(C_left, C_right, tpdm_bb_, false);
        if ((na >= 1) and (nb >= 1))
            compute_2rdm_ab(C_left, C_right, tpdm_ab_);
        rdm_timing.push_back(t.get());
    }

    if (max_order >= 3) {
        local_timer t;
        if (na >= 3)
            compute_3rdm_aaa(C_left, C_right, tpdm_aaa_, true);
        if (nb >= 3)
            compute_3rdm_aaa(C_left, C_right, tpdm_bbb_, false);
        if ((na >= 2) and (nb >= 1))
            compute_3rdm_aab(C_left, C_right, tpdm_aab_);
        if ((na >= 1) and (nb >= 2))
            compute_3rdm_abb(C_left, C_right, tpdm_abb_);
        rdm_timing.push_back(t.get());
    }

    if (max_order >= 4) {
    }

    // Print RDM timings
    if (print_ > 0) {
        for (size_t n = 0; n < rdm_timing.size(); ++n) {
            outfile->Printf("\n    Timing for %d-RDM: %.3f s", n + 1, rdm_timing[n]);
        }
    }
}

void fill_C_block(FCIVector& C, std::shared_ptr<psi::Matrix> M, int irrep, bool alfa) {
    // if alfa is true just return the block
    if (alfa) {
        C.C(irrep) = M;
    } else {
        // if alfa is false, transpose the block
        double** m = M->pointer();
        C_left.alfa_address->strpi(irrep);
    }
}

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIVector::compute_1rdm(FCIVector& C_left, FCIVector& C_right, std::vector<double>& rdm,
                             bool alfa) {
    rdm.assign(ncmo_ * ncmo_, 0.0);

    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        if (detpi_[alfa_sym] > 0) {
            auto Cl = alfa ? C_left.C(alfa_sym) : C1;
            auto Cr = alfa ? C_right.C(alfa_sym) : Y1;
            double** Clh = Cl->pointer();
            double** Crh = Cr->pointer();

            if (!alfa) {
                Cl->zero();
                Cr->zero();
                size_t maxIa = alfa_address_->strpi(alfa_sym);
                size_t maxIb = beta_address_->strpi(beta_sym);

                double** C0lh = C_left.C(alfa_sym)->pointer();
                double** C0rh = C_right.C(alfa_sym)->pointer();

                // Copy C0 transposed in C1
                for (size_t Ia = 0; Ia < maxIa; ++Ia)
                    for (size_t Ib = 0; Ib < maxIb; ++Ib)
                        Clh[Ib][Ia] = C0lh[Ia][Ib];
                for (size_t Ia = 0; Ia < maxIa; ++Ia)
                    for (size_t Ib = 0; Ib < maxIb; ++Ib)
                        Crh[Ib][Ia] = C0lr[Ia][Ib];
            }

            size_t maxL = alfa ? beta_address_->strpi(beta_sym) : alfa_address_->strpi(alfa_sym);

            for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
                int q_sym = p_sym; // Select the totat symmetric irrep
                for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                    for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                        int p_abs = p_rel + cmopi_offset_[p_sym];
                        int q_abs = q_rel + cmopi_offset_[q_sym];
                        std::vector<StringSubstitution>& vo =
                            alfa ? lists_->get_alfa_vo_list(p_abs, q_abs, alfa_sym)
                                 : lists_->get_beta_vo_list(p_abs, q_abs, beta_sym);
                        size_t maxss = vo.size();
                        double rdm_element = 0.0;
                        for (size_t ss = 0; ss < maxss; ++ss) {
                            double H = static_cast<double>(vo[ss].sign);
                            double* y = &(Clh[vo[ss].J][0]);
                            double* c = &(Crh[vo[ss].I][0]);
                            for (size_t L = 0; L < maxL; ++L) {
                                rdm_element += c[L] * y[L] * H;
                            }
                        }
                        rdm[p_abs * ncmo_ + q_abs] = rdm_element;
                    }
                }
            }
        }
    } // End loop over h

#if 0
    outfile->Printf("\n OPDM:");
    for (int p = 0; p < no_; ++p) {
        outfile->Printf("\n");
        for (int q = 0; q < no_; ++q) {
            outfile->Printf("%15.12f ",rdm[oei_index(p,q)]);
        }
    }
#endif
}

} // namespace forte
