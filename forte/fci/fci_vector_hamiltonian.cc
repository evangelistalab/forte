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

#include "psi4/libqt/qt.h"
#include "psi4/libmints/matrix.h"

#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"
#include "fci_vector.h"
#include "binary_graph.hpp"
#include "string_lists.h"

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

/**
 * Apply the scalar part of the Hamiltonian to the wave function
 */
void FCIVector::H0(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    double core_energy = fci_ints->scalar_energy() + fci_ints->frozen_core_energy() +
                         fci_ints->nuclear_repulsion_energy();
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        result.C_[alfa_sym]->copy(C_[alfa_sym]);
        result.C_[alfa_sym]->scale(core_energy);
    }
}

/**
 * Apply the one-particle Hamiltonian to the wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIVector::H1(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints, bool alfa) {
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        if (detpi_[alfa_sym] > 0) {
            psi::SharedMatrix C = alfa ? C_[alfa_sym] : C1;
            psi::SharedMatrix Y = alfa ? result.C_[alfa_sym] : Y1;
            double** Ch = C->pointer();
            double** Yh = Y->pointer();

            if (!alfa) {
                C->zero();
                Y->zero();
                size_t maxIa = alfa_graph_->strpi(alfa_sym);
                size_t maxIb = beta_graph_->strpi(beta_sym);

                double** C0h = C_[alfa_sym]->pointer();

                // Copy C0 transposed in C1
                for (size_t Ia = 0; Ia < maxIa; ++Ia)
                    for (size_t Ib = 0; Ib < maxIb; ++Ib)
                        Ch[Ib][Ia] = C0h[Ia][Ib];
            }

            size_t maxL = alfa ? beta_graph_->strpi(beta_sym) : alfa_graph_->strpi(alfa_sym);

            for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
                int q_sym = p_sym; // Select the totat symmetric irrep
                for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                    for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                        int p_abs = p_rel + cmopi_offset_[p_sym];
                        int q_abs = q_rel + cmopi_offset_[q_sym];

                        double Hpq = alfa ? fci_ints->oei_a(p_abs, q_abs)
                                          : fci_ints->oei_b(p_abs,
                                                            q_abs); // Grab the integral
                        std::vector<StringSubstitution>& vo =
                            alfa ? lists_->get_alfa_vo_list(p_abs, q_abs, alfa_sym)
                                 : lists_->get_beta_vo_list(p_abs, q_abs, beta_sym);
                        // TODO loop in a differen way
                        int maxss = vo.size();

                        for (int ss = 0; ss < maxss; ++ss) {
#if CAPRICCIO_USE_DAXPY
                            C_DAXPY(maxL, static_cast<double>(vo[ss].sign) * Hpq,
                                    &(Ch[vo[ss].I][0]), 1, &(Yh[vo[ss].J][0]), 1);
#else
                            double H = static_cast<double>(vo[ss].sign) * Hpq;
                            double* y = &Y[vo[ss].J][0];
                            double* c = &C[vo[ss].I][0];
                            for (size_t L = 0; L < maxL; ++L)
                                y[L] += c[L] * H;
#endif
                        }
                    }
                }
            }
            if (!alfa) {
                size_t maxIa = alfa_graph_->strpi(alfa_sym);
                size_t maxIb = beta_graph_->strpi(beta_sym);

                double** HC = result.C_[alfa_sym]->pointer();
                // Add Y1 transposed to Y
                for (size_t Ia = 0; Ia < maxIa; ++Ia)
                    for (size_t Ib = 0; Ib < maxIb; ++Ib)
                        HC[Ia][Ib] += Yh[Ib][Ia];
            }
        }
    } // End loop over h
}

/**
 * Apply the same-spin two-particle Hamiltonian to the wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIVector::H2_aaaa2(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                         bool alfa) {
    // Notation
    // ha - symmetry of alpha strings
    // hb - symmetry of beta strings
    for (int ha = 0; ha < nirrep_; ++ha) {
        int hb = ha ^ symmetry_;
        if (detpi_[ha] > 0) {
            psi::SharedMatrix C = alfa ? C_[ha] : C1;
            psi::SharedMatrix Y = alfa ? result.C_[ha] : Y1;
            double** Ch = C->pointer();
            double** Yh = Y->pointer();

            if (!alfa) {
                C->zero();
                Y->zero();
                size_t maxIa = alfa_graph_->strpi(ha);
                size_t maxIb = beta_graph_->strpi(hb);

                double** C0h = C_[ha]->pointer();

                // Copy C0 transposed in C1
                for (size_t Ia = 0; Ia < maxIa; ++Ia)
                    for (size_t Ib = 0; Ib < maxIb; ++Ib)
                        Ch[Ib][Ia] = C0h[Ia][Ib];
            }

            size_t maxL = alfa ? beta_graph_->strpi(hb) : alfa_graph_->strpi(ha);
            // Loop over (p>q) == (p>q)
            for (int pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
                size_t max_pq = lists_->pairpi(pq_sym);
                for (size_t pq = 0; pq < max_pq; ++pq) {
                    const Pair& pq_pair = lists_->get_nn_list_pair(pq_sym, pq);
                    int p_abs = pq_pair.first;
                    int q_abs = pq_pair.second;

                    double integral = alfa ? fci_ints->tei_aa(p_abs, q_abs, p_abs, q_abs)
                                           : fci_ints->tei_bb(p_abs, q_abs, p_abs, q_abs);

                    std::vector<StringSubstitution>& OO =
                        alfa ? lists_->get_alfa_oo_list(pq_sym, pq, ha)
                             : lists_->get_beta_oo_list(pq_sym, pq, hb);

                    size_t maxss = OO.size();
                    for (size_t ss = 0; ss < maxss; ++ss)
                        C_DAXPY(maxL, static_cast<double>(OO[ss].sign) * integral,
                                &(C->pointer()[OO[ss].I][0]), 1, &(Y->pointer()[OO[ss].J][0]), 1);
                }
            }
            // Loop over (p>q) > (r>s)
            for (int pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
                size_t max_pq = lists_->pairpi(pq_sym);
                for (size_t pq = 0; pq < max_pq; ++pq) {
                    const Pair& pq_pair = lists_->get_nn_list_pair(pq_sym, pq);
                    int p_abs = pq_pair.first;
                    int q_abs = pq_pair.second;
                    for (size_t rs = 0; rs < pq; ++rs) {
                        const Pair& rs_pair = lists_->get_nn_list_pair(pq_sym, rs);
                        int r_abs = rs_pair.first;
                        int s_abs = rs_pair.second;
                        double integral = alfa ? fci_ints->tei_aa(p_abs, q_abs, r_abs, s_abs)
                                               : fci_ints->tei_bb(p_abs, q_abs, r_abs, s_abs);

                        {
                            std::vector<StringSubstitution>& VVOO =
                                alfa ? lists_->get_alfa_vvoo_list(p_abs, q_abs, r_abs, s_abs, ha)
                                     : lists_->get_beta_vvoo_list(p_abs, q_abs, r_abs, s_abs, hb);
                            // TODO loop in a differen way
                            size_t maxss = VVOO.size();
                            for (size_t ss = 0; ss < maxss; ++ss)
                                C_DAXPY(maxL, static_cast<double>(VVOO[ss].sign) * integral,
                                        &(C->pointer()[VVOO[ss].I][0]), 1,
                                        &(Y->pointer()[VVOO[ss].J][0]), 1);
                        }
                        {
                            std::vector<StringSubstitution>& VVOO =
                                alfa ? lists_->get_alfa_vvoo_list(r_abs, s_abs, p_abs, q_abs, ha)
                                     : lists_->get_beta_vvoo_list(r_abs, s_abs, p_abs, q_abs, hb);
                            // TODO loop in a differen way
                            size_t maxss = VVOO.size();
                            for (size_t ss = 0; ss < maxss; ++ss)
                                C_DAXPY(maxL, static_cast<double>(VVOO[ss].sign) * integral,
                                        &(C->pointer()[VVOO[ss].I][0]), 1,
                                        &(Y->pointer()[VVOO[ss].J][0]), 1);
                        }
                    }
                }
            }
            if (!alfa) {
                size_t maxIa = alfa_graph_->strpi(ha);
                size_t maxIb = beta_graph_->strpi(hb);

                double** HC = result.C_[ha]->pointer();

                // Add Y1 transposed to Y
                for (size_t Ia = 0; Ia < maxIa; ++Ia)
                    for (size_t Ib = 0; Ib < maxIb; ++Ib)
                        HC[Ia][Ib] += Yh[Ib][Ia];
            }
        }
    } // End loop over h
}

/**
 * Apply the different-spin component of two-particle Hamiltonian to the wave
 * function
 */
void FCIVector::H2_aabb(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    // Loop over blocks of matrix C
    for (int Ia_sym = 0; Ia_sym < nirrep_; ++Ia_sym) {
        size_t maxIa = alfa_graph_->strpi(Ia_sym);
        int Ib_sym = Ia_sym ^ symmetry_;
        double** C = C_[Ia_sym]->pointer();

        // Loop over all r,s
        for (int rs_sym = 0; rs_sym < nirrep_; ++rs_sym) {
            int Jb_sym = Ib_sym ^ rs_sym;    // <- Looks like it should fail for
                                             // states with symmetry != A1  URGENT
            int Ja_sym = Jb_sym ^ symmetry_; // <- Looks like it should fail for
                                             // states with symmetry != A1
                                             // URGENT
            //            int Ja_sym = Ia_sym ^ rs_sym; // <- Looks like it
            //            should fail for states with symmetry != A1  URGENT

            size_t maxJa = alfa_graph_->strpi(Ja_sym);
            double** Y = result.C_[Ja_sym]->pointer();
            for (int r_sym = 0; r_sym < nirrep_; ++r_sym) {
                int s_sym = rs_sym ^ r_sym;

                for (int r_rel = 0; r_rel < cmopi_[r_sym]; ++r_rel) {
                    for (int s_rel = 0; s_rel < cmopi_[s_sym]; ++s_rel) {
                        int r_abs = r_rel + cmopi_offset_[r_sym];
                        int s_abs = s_rel + cmopi_offset_[s_sym];

                        // Grab list (r,s,Ib_sym)
                        std::vector<StringSubstitution>& vo_beta =
                            lists_->get_beta_vo_list(r_abs, s_abs, Ib_sym);
                        size_t maxSSb = vo_beta.size();

                        C1->zero();
                        Y1->zero();

                        // Gather cols of C into C1
                        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
                            if (maxSSb > 0) {
                                double* c1 = &(C1->pointer()[Ia][0]); //&C1[Ia][0];
                                double* c = &(C[Ia][0]);
                                for (size_t SSb = 0; SSb < maxSSb; ++SSb) {
                                    c1[SSb] =
                                        c[vo_beta[SSb].I] * static_cast<double>(vo_beta[SSb].sign);
                                }
                            }
                        }

                        // Loop over all p,q
                        int pq_sym = rs_sym;
                        for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
                            int q_sym = pq_sym ^ p_sym;
                            for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                                int p_abs = p_rel + cmopi_offset_[p_sym];
                                for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                                    int q_abs = q_rel + cmopi_offset_[q_sym];
                                    // Grab the integral
                                    double integral = fci_ints->tei_ab(p_abs, r_abs, q_abs, s_abs);

                                    std::vector<StringSubstitution>& vo_alfa =
                                        lists_->get_alfa_vo_list(p_abs, q_abs, Ia_sym);

                                    // ORIGINAL CODE
                                    size_t maxSSa = vo_alfa.size();
                                    for (size_t SSa = 0; SSa < maxSSa; ++SSa) {
#if CAPRICCIO_USE_DAXPY
                                        C_DAXPY(maxSSb,
                                                integral * static_cast<double>(vo_alfa[SSa].sign),
                                                &(C1->pointer()[vo_alfa[SSa].I][0]), 1,
                                                &(Y1->pointer()[vo_alfa[SSa].J][0]), 1);
#else
                                        double V =
                                            integral * static_cast<double>(vo_alfa[SSa].sign);
                                        for (size_t SSb = 0; SSb < maxSSb; ++SSb) {
                                            Y1[vo_alfa[SSa].J][SSb] += C1[vo_alfa[SSa].I][SSb] * V;
                                        }
#endif
                                    }
                                }
                            }
                        } // End loop over p,q
                        // Scatter cols of Y1 into Y
                        for (size_t Ja = 0; Ja < maxJa; ++Ja) {
                            if (maxSSb > 0) {
                                double* y = &Y[Ja][0];
                                double* y1 = &(Y1->pointer()[Ja][0]);
                                for (size_t SSb = 0; SSb < maxSSb; ++SSb) {
                                    y[vo_beta[SSb].J] += y1[SSb];
                                }
                            }
                        }
                    }
                } // End loop over r_rel,s_rel
            }
        }
    }
}
} // namespace forte
