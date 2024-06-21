/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"
#include "fci_vector.h"
#include "fci_string_lists.h"
#include "fci_string_address.h"

using namespace psi;

namespace forte {

/**
 * Apply the Hamiltonian to the wave function
 * @param result Wave function object which stores the resulting vector
 */
void FCIVector::Hamiltonian(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
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

void FCIVector::H0(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    double core_energy = fci_ints->scalar_energy() + fci_ints->frozen_core_energy() +
                         fci_ints->nuclear_repulsion_energy();
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        result.C_[alfa_sym]->copy(C_[alfa_sym]);
        result.C_[alfa_sym]->scale(core_energy);
    }
}

void FCIVector::H1(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints, bool alfa) {
    for (int h_Ia = 0; h_Ia < nirrep_; ++h_Ia) {
        int h_Ib = h_Ia ^ symmetry_;
        if (detpi_[h_Ia] > 0) {
            auto Cr =
                gather_C_block(*this, CR, alfa, alfa_address_, beta_address_, h_Ia, h_Ib, false);
            auto Cl =
                gather_C_block(result, CL, alfa, alfa_address_, beta_address_, h_Ia, h_Ib, !alfa);

            size_t maxL = alfa ? beta_address_->strpcls(h_Ib) : alfa_address_->strpcls(h_Ia);

            for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
                int q_sym = p_sym; // Select the totat symmetric irrep
                for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                    for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                        const int p_abs = p_rel + cmopi_offset_[p_sym];
                        const int q_abs = q_rel + cmopi_offset_[q_sym];
                        const double Hpq =
                            alfa ? fci_ints->oei_a(p_abs, q_abs) : fci_ints->oei_b(p_abs, q_abs);
                        const auto& vo_list = alfa ? lists_->get_alfa_vo_list(p_abs, q_abs, h_Ia)
                                                   : lists_->get_beta_vo_list(p_abs, q_abs, h_Ib);
                        for (const auto& [sign, I, J] : vo_list) {
                            C_DAXPY(maxL, sign * Hpq, Cr[I], 1, Cl[J], 1);
                        }
                    }
                }
            }
            scatter_C_block(result, Cl, alfa, alfa_address_, beta_address_, h_Ia, h_Ib);
        }
    } // End loop over h
}

void FCIVector::H2_aaaa2(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                         bool alfa) {
    // Notation
    // h_Ia - symmetry of alpha strings
    // h_Ib - symmetry of beta strings
    for (int h_Ia = 0; h_Ia < nirrep_; ++h_Ia) {
        int h_Ib = h_Ia ^ symmetry_;
        if (detpi_[h_Ia] > 0) {
            auto Cr =
                gather_C_block(*this, CR, alfa, alfa_address_, beta_address_, h_Ia, h_Ib, false);
            auto Cl =
                gather_C_block(result, CL, alfa, alfa_address_, beta_address_, h_Ia, h_Ib, !alfa);

            size_t maxL = alfa ? beta_address_->strpcls(h_Ib) : alfa_address_->strpcls(h_Ia);
            // Loop over (p>q) == (p>q)
            for (int pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
                size_t max_pq = lists_->pairpi(pq_sym);
                for (size_t pq = 0; pq < max_pq; ++pq) {
                    const auto& [p_abs, q_abs] = lists_->get_pair_list(pq_sym, pq);

                    const double integral = alfa ? fci_ints->tei_aa(p_abs, q_abs, p_abs, q_abs)
                                                 : fci_ints->tei_bb(p_abs, q_abs, p_abs, q_abs);

                    const auto& OO_list = alfa ? lists_->get_alfa_oo_list(pq_sym, pq, h_Ia)
                                               : lists_->get_beta_oo_list(pq_sym, pq, h_Ib);

                    for (const auto& [sign, I, J] : OO_list) {
                        C_DAXPY(maxL, sign * integral, Cr[I], 1, Cl[J], 1);
                    }
                }
            }
            // Loop over (p>q) > (r>s)
            for (int pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
                size_t max_pq = lists_->pairpi(pq_sym);
                for (size_t pq = 0; pq < max_pq; ++pq) {
                    const Pair& pq_pair = lists_->get_pair_list(pq_sym, pq);
                    int p_abs = pq_pair.first;
                    int q_abs = pq_pair.second;
                    for (size_t rs = 0; rs < pq; ++rs) {
                        const auto& [r_abs, s_abs] = lists_->get_pair_list(pq_sym, rs);
                        const double integral = alfa ? fci_ints->tei_aa(p_abs, q_abs, r_abs, s_abs)
                                                     : fci_ints->tei_bb(p_abs, q_abs, r_abs, s_abs);

                        {
                            const auto& VVOO_list =
                                alfa ? lists_->get_alfa_vvoo_list(p_abs, q_abs, r_abs, s_abs, h_Ia)
                                     : lists_->get_beta_vvoo_list(p_abs, q_abs, r_abs, s_abs, h_Ib);
                            for (const auto& [sign, I, J] : VVOO_list) {
                                C_DAXPY(maxL, sign * integral, Cr[I], 1, Cl[J], 1);
                            }
                            {
                                const auto& VVOO_list =
                                    alfa ? lists_->get_alfa_vvoo_list(r_abs, s_abs, p_abs, q_abs,
                                                                      h_Ia)
                                         : lists_->get_beta_vvoo_list(r_abs, s_abs, p_abs, q_abs,
                                                                      h_Ib);
                                for (const auto& [sign, I, J] : VVOO_list) {
                                    C_DAXPY(maxL, sign * integral, Cr[I], 1, Cl[J], 1);
                                }
                            }
                        }
                    }
                }
            }
            scatter_C_block(result, Cl, alfa, alfa_address_, beta_address_, h_Ia, h_Ib);
        }
    } // End loop over h
}

void FCIVector::H2_aabb(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    // Loop over blocks of matrix C
    for (int h_Ia = 0; h_Ia < nirrep_; ++h_Ia) {
        const size_t maxIa = alfa_address_->strpcls(h_Ia);
        const int h_Ib = h_Ia ^ symmetry_;
        const auto C = C_[h_Ia]->pointer();

        // Loop over all r,s
        for (int rs_sym = 0; rs_sym < nirrep_; ++rs_sym) {
            const int h_Jb = h_Ib ^ rs_sym;
            const int h_Ja = h_Jb ^ symmetry_;

            const size_t maxJa = alfa_address_->strpcls(h_Ja);
            auto HC = result.C_[h_Ja]->pointer();
            for (int r_sym = 0; r_sym < nirrep_; ++r_sym) {
                const int s_sym = rs_sym ^ r_sym;

                for (int r_rel = 0; r_rel < cmopi_[r_sym]; ++r_rel) {
                    for (int s_rel = 0; s_rel < cmopi_[s_sym]; ++s_rel) {
                        const int r_abs = r_rel + cmopi_offset_[r_sym];
                        const int s_abs = s_rel + cmopi_offset_[s_sym];

                        // Grab list (r,s,h_Ib)
                        const auto& vo_beta = lists_->get_beta_vo_list(r_abs, s_abs, h_Ib);
                        const size_t maxSSb = vo_beta.size();

                        if (maxSSb == 0)
                            continue;

                        CR->zero();
                        CL->zero();
                        auto Cr = CR->pointer();
                        auto Cl = CL->pointer();

                        // Gather cols of C into CR
                        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
                            const auto c = C[Ia];
                            auto cr = Cr[Ia];
                            for (size_t SSb = 0; SSb < maxSSb; ++SSb) {
                                cr[SSb] = c[vo_beta[SSb].I] * vo_beta[SSb].sign;
                            }
                        }

                        // Loop over all p,q
                        int pq_sym = rs_sym;
                        for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
                            int q_sym = pq_sym ^ p_sym;
                            for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                                for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                                    int p_abs = p_rel + cmopi_offset_[p_sym];
                                    int q_abs = q_rel + cmopi_offset_[q_sym];
                                    // Grab the integral
                                    const double integral =
                                        fci_ints->tei_ab(p_abs, r_abs, q_abs, s_abs);

                                    const auto& vo_alfa =
                                        lists_->get_alfa_vo_list(p_abs, q_abs, h_Ia);

                                    for (const auto& [sign, I, J] : vo_alfa) {
                                        C_DAXPY(maxSSb, integral * sign, Cr[I], 1, Cl[J], 1);
                                    }
                                }
                            }
                        } // End loop over p,q

                        // Scatter cols of CL into HC
                        for (size_t Ja = 0; Ja < maxJa; ++Ja) {
                            const auto hc = HC[Ja];
                            auto cl = Cl[Ja];
                            for (size_t SSb = 0; SSb < maxSSb; ++SSb) {
                                hc[vo_beta[SSb].J] += cl[SSb];
                            }
                        }
                    }
                } // End loop over r_rel,s_rel
            }
        }
    }
}
} // namespace forte
