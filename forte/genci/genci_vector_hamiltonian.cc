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

#include "psi4/libqt/qt.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"
#include "genci_vector.h"
#include "genci_string_lists.h"
#include "genci_string_address.h"

using namespace psi;

namespace forte {

/**
 * Apply the Hamiltonian to the wave function
 * @param result Wave function object which stores the resulting vector
 */
void GenCIVector::Hamiltonian(GenCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    //    check_temp_space();
    result.zero();

    // H0
    { H0(result, fci_ints); }
    // H1_aa
    {
        local_timer t;
        H1(result, fci_ints, true);
        h1_aa_timer += t.get();
    }
    // H1_bb
    {
        local_timer t;
        H1(result, fci_ints, false);
        h1_bb_timer += t.get();
    }
    // H2_aabb
    {
        local_timer t;
        H2_aabb(result, fci_ints);
        h2_aabb_timer += t.get();
    }
    // H2_aaaa
    {
        local_timer t;
        H2_aaaa2(result, fci_ints, true);
        h2_aaaa_timer += t.get();
    }
    // H2_bbbb
    {
        local_timer t;
        H2_aaaa2(result, fci_ints, false);
        h2_bbbb_timer += t.get();
    }
}

void GenCIVector::H0(GenCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    double core_energy = fci_ints->scalar_energy() + fci_ints->frozen_core_energy() +
                         fci_ints->nuclear_repulsion_energy();
    for (const auto& [n, _1, _2] : lists_->determinant_classes()) {
        result.C_[n]->copy(C_[n]);
        result.C_[n]->scale(core_energy);
    }
}

void GenCIVector::H1(GenCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                     bool alfa) {
    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_->determinant_classes()) {
        if (lists_->detpblk(nI) == 0)
            continue;
        auto Cr =
            this->gather_C_block(CR, alfa, alfa_address_, beta_address_, class_Ia, class_Ib, false);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_->determinant_classes()) {

            // If we act on the alpha string, the beta string classes of the result must be the same
            if (alfa and (class_Ib != class_Jb))
                continue;
            // If we act on the beta string, the alpha string classes of the result must be the same
            if (not alfa and (class_Ia != class_Ja))
                continue;

            if (lists_->detpblk(nJ) == 0)
                continue;

            auto Cl = result.gather_C_block(CL, alfa, alfa_address_, beta_address_, class_Ja,
                                            class_Jb, !alfa);

            size_t maxL =
                alfa ? beta_address_->strpcls(class_Ib) : alfa_address_->strpcls(class_Ia);

            const auto& pq_vo_list = alfa ? lists_->get_alfa_vo_list(class_Ia, class_Ja)
                                          : lists_->get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                const double Hpq = alfa ? fci_ints->oei_a(p, q) : fci_ints->oei_b(p, q);
                for (const auto& [sign, I, J] : vo_list) {
                    C_DAXPY(maxL, sign * Hpq, Cr[I], 1, Cl[J], 1);
                }
            }
            result.scatter_C_block(Cl, alfa, alfa_address_, beta_address_, class_Ja, class_Jb);
        }
    }
} // End loop over h

void GenCIVector::H2_aaaa2(GenCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                           bool alfa) {
    for (const auto& [nI, class_Ia, class_Ib] : lists_->determinant_classes()) {
        if (lists_->detpblk(nI) == 0)
            continue;

        const auto Cr =
            this->gather_C_block(CR, alfa, alfa_address_, beta_address_, class_Ia, class_Ib, false);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_->determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                continue;
            if (lists_->detpblk(nJ) == 0)
                continue;

            auto Cl = result.gather_C_block(CL, alfa, alfa_address_, beta_address_, class_Ja,
                                            class_Jb, !alfa);

            // get the size of the string of spin opposite to the one we are acting on
            size_t maxL =
                alfa ? beta_address_->strpcls(class_Ib) : alfa_address_->strpcls(class_Ia);

            if ((class_Ia == class_Ja) and (class_Ib == class_Jb)) {
                // OO terms
                // Loop over (p>q) == (p>q)
                const auto& pq_oo_list =
                    alfa ? lists_->get_alfa_oo_list(class_Ia) : lists_->get_beta_oo_list(class_Ib);
                for (const auto& [pq, oo_list] : pq_oo_list) {
                    const auto& [p, q] = pq;
                    const double integral =
                        alfa ? fci_ints->tei_aa(p, q, p, q) : fci_ints->tei_bb(p, q, p, q);
                    for (const auto& I : oo_list) {
                        C_DAXPY(maxL, integral, Cr[I], 1, Cl[I], 1);
                    }
                }
            }

            // VVOO terms
            const auto& pqrs_vvoo_list = alfa ? lists_->get_alfa_vvoo_list(class_Ia, class_Ja)
                                              : lists_->get_beta_vvoo_list(class_Ib, class_Jb);
            for (const auto& [pqrs, vvoo_list] : pqrs_vvoo_list) {
                const auto& [p, q, r, s] = pqrs;
                const double integral1 =
                    alfa ? fci_ints->tei_aa(p, q, r, s) : fci_ints->tei_bb(p, q, r, s);
                for (const auto& [sign, I, J] : vvoo_list) {
                    C_DAXPY(maxL, sign * integral1, Cr[I], 1, Cl[J], 1);
                }
            }
            result.scatter_C_block(Cl, alfa, alfa_address_, beta_address_, class_Ja, class_Jb);
        }
    }
}

void GenCIVector::H2_aabb(GenCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    const auto& mo_sym = lists_->string_class()->mo_sym();
    const auto Cr = CR->pointer();
    auto Cl = CL->pointer();
    // Loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_->determinant_classes()) {
        if (lists_->detpblk(nI) == 0)
            continue;

        auto h_Ib = lists_->string_class()->beta_string_classes()[class_Ib].second;
        const size_t maxIa = alfa_address_->strpcls(class_Ia);
        const auto C = C_[nI]->pointer();

        for (const auto& [nJ, class_Ja, class_Jb] : lists_->determinant_classes()) {
            if (lists_->detpblk(nJ) == 0)
                continue;

            auto h_Jb = lists_->string_class()->beta_string_classes()[class_Jb].second;
            const size_t maxJa = alfa_address_->strpcls(class_Ja);
            auto HC = result.C_[nJ]->pointer();

            const auto& pq_vo_alfa = lists_->get_alfa_vo_list(class_Ia, class_Ja);
            const auto& rs_vo_beta = lists_->get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [rs, vo_beta_list] : rs_vo_beta) {
                const size_t beta_list_size = vo_beta_list.size();
                if (beta_list_size == 0)
                    continue;

                const auto& [r, s] = rs;
                const auto rs_sym = mo_sym[r] ^ mo_sym[s];

                // Make sure that the symmetry of the J beta string is the same as the symmetry of
                // the I beta string times the symmetry of the rs product
                if (h_Jb != (h_Ib ^ rs_sym))
                    continue;

                // Zero the block of CL used to store the result
                // this should be faster than CL->zero(); when beta_list_size is smaller than
                // the number of columns of CL
                for (size_t Ja = 0; Ja < maxJa; ++Ja) {
                    const auto cl = Cl[Ja];
                    std::fill(cl, cl + beta_list_size, 0.0);
                }

                // Gather cols of C into CR with the correct sign
                for (size_t Ia = 0; Ia < maxIa; ++Ia) {
                    const auto c = C[Ia];
                    auto cr = Cr[Ia];
                    for (size_t idx{0}; const auto& [sign, I, _] : vo_beta_list) {
                        cr[idx] = c[I] * sign;
                        idx++;
                    }
                }

                for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
                    const auto& [p, q] = pq;
                    const auto pq_sym = mo_sym[p] ^ mo_sym[q];
                    // ensure that the product pqrs is totally symmetric
                    if (pq_sym != rs_sym)
                        continue;

                    // Grab the integral
                    const double integral = fci_ints->tei_ab(p, r, q, s);

                    for (const auto& [sign, I, J] : vo_alfa_list) {
                        C_DAXPY(beta_list_size, integral * sign, Cr[I], 1, Cl[J], 1);
                    }
                } // End loop over p,q

                // Scatter cols of CL into HC (the sign was included before in the gathering)
                for (size_t Ja = 0; Ja < maxJa; ++Ja) {
                    auto hc = HC[Ja];
                    const auto cl = Cl[Ja];
                    for (size_t idx{0}; const auto& [_1, _2, J] : vo_beta_list) {
                        hc[J] += cl[idx];
                        idx++;
                    }
                }
            }
        }
    }
}
} // namespace forte
