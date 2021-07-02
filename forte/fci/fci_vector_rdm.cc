/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

extern int fci_debug_level;

using namespace psi;

namespace forte {

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIVector::compute_rdms(int max_order) {
    std::vector<double> rdm_timing;

    size_t na = alfa_graph_->nones();
    size_t nb = beta_graph_->nones();

    if (max_order >= 1) {
        local_timer t;
        if (na >= 1)
            compute_1rdm(opdm_a_, true);
        if (nb >= 1)
            compute_1rdm(opdm_b_, false);
        rdm_timing.push_back(t.get());
    }

    if (max_order >= 2) {
        local_timer t;
        if (na >= 2)
            compute_2rdm_aa(tpdm_aa_, true);
        if (nb >= 2)
            compute_2rdm_aa(tpdm_bb_, false);
        if ((na >= 1) and (nb >= 1))
            compute_2rdm_ab(tpdm_ab_);
        rdm_timing.push_back(t.get());
    }

    if (max_order >= 3) {
        local_timer t;
        if (na >= 3)
            compute_3rdm_aaa(tpdm_aaa_, true);
        if (nb >= 3)
            compute_3rdm_aaa(tpdm_bbb_, false);
        if ((na >= 2) and (nb >= 1))
            compute_3rdm_aab(tpdm_aab_);
        if ((na >= 1) and (nb >= 2))
            compute_3rdm_abb(tpdm_abb_);
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

double FCIVector::energy_from_rdms(std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    // Compute the energy from the 1-RDM and 2-RDM
    size_t na = alfa_graph_->nones();
    size_t nb = beta_graph_->nones();

    double nuclear_repulsion_energy =
        psi::Process::environment.molecule()->nuclear_repulsion_energy({{0, 0, 0}});

    double scalar_energy = fci_ints->frozen_core_energy() + fci_ints->scalar_energy();
    double energy_1rdm = 0.0;
    double energy_2rdm = 0.0;

    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            energy_1rdm += opdm_a_[ncmo_ * p + q] * fci_ints->oei_a(p, q);
            energy_1rdm += opdm_b_[ncmo_ * p + q] * fci_ints->oei_b(p, q);
        }
    }

    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    if (na >= 2)
                        energy_2rdm +=
                            0.25 * tpdm_aa_[tei_index(p, q, r, s)] * fci_ints->tei_aa(p, q, r, s);
                    if ((na >= 1) and (nb >= 1))
                        energy_2rdm +=
                            tpdm_ab_[tei_index(p, q, r, s)] * fci_ints->tei_ab(p, q, r, s);
                    if (nb >= 2)
                        energy_2rdm +=
                            0.25 * tpdm_bb_[tei_index(p, q, r, s)] * fci_ints->tei_bb(p, q, r, s);
                }
            }
        }
    }
    double total_energy = nuclear_repulsion_energy + scalar_energy + energy_1rdm + energy_2rdm;
    outfile->Printf("\n    Total Energy: %25.15f\n", total_energy);
    outfile->Printf("\n scalar_energy = %8.8f", scalar_energy);
    outfile->Printf("\n energy_1rdm = %8.8f", energy_1rdm);
    outfile->Printf("\n energy_2rdm = %8.8f", energy_2rdm);
    outfile->Printf("\n nuc_repulsion_energy = %8.8f", nuclear_repulsion_energy);
    return total_energy;
}

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIVector::compute_1rdm(std::vector<double>& rdm, bool alfa) {
    rdm.assign(ncmo_ * ncmo_, 0.0);

    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        if (detpi_[alfa_sym] > 0) {
            psi::SharedMatrix C = alfa ? C_[alfa_sym] : C1;
            double** Ch = C->pointer();

            if (!alfa) {
                C->zero();
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
                        std::vector<StringSubstitution>& vo =
                            alfa ? lists_->get_alfa_vo_list(p_abs, q_abs, alfa_sym)
                                 : lists_->get_beta_vo_list(p_abs, q_abs, beta_sym);
                        int maxss = vo.size();
                        for (int ss = 0; ss < maxss; ++ss) {
                            double H = static_cast<double>(vo[ss].sign);
                            double* y = &(Ch[vo[ss].J][0]);
                            double* c = &(Ch[vo[ss].I][0]);
                            for (size_t L = 0; L < maxL; ++L) {
                                rdm[p_abs * ncmo_ + q_abs] += c[L] * y[L] * H;
                            }
                        }
                    }
                }
            }
        }
    } // End loop over h

#if 0
    outfile->Printf("\n OPDM:");
    for (int p = 0; p < ncmo_; ++p) {
        outfile->Printf("\n");
        for (int q = 0; q < ncmo_; ++q) {
            outfile->Printf("%15.12f ",rdm[oei_index(p,q)]);
        }
    }
#endif
}

/**
 * Compute the aa/bb two-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
void FCIVector::compute_2rdm_aa(std::vector<double>& rdm, bool alfa) {
    rdm.assign(ncmo_ * ncmo_ * ncmo_ * ncmo_, 0.0);
    // Notation
    // ha - symmetry of alpha strings
    // hb - symmetry of beta strings
    for (int ha = 0; ha < nirrep_; ++ha) {
        int hb = ha ^ symmetry_;
        if (detpi_[ha] > 0) {
            psi::SharedMatrix C = alfa ? C_[ha] : C1;
            double** Ch = C->pointer();

            if (!alfa) {
                C->zero();
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

                    std::vector<StringSubstitution>& OO =
                        alfa ? lists_->get_alfa_oo_list(pq_sym, pq, ha)
                             : lists_->get_beta_oo_list(pq_sym, pq, hb);
                    double rdm_element = 0.0;
                    size_t maxss = OO.size();
                    for (size_t ss = 0; ss < maxss; ++ss) {
                        double H = static_cast<double>(OO[ss].sign);
                        double* y = &(Ch[OO[ss].J][0]);
                        double* c = &(Ch[OO[ss].I][0]);
                        for (size_t L = 0; L < maxL; ++L) {
                            rdm_element += c[L] * y[L] * H;
                        }
                    }

                    rdm[tei_index(p_abs, q_abs, p_abs, q_abs)] += rdm_element;
                    rdm[tei_index(p_abs, q_abs, q_abs, p_abs)] -= rdm_element;
                    rdm[tei_index(q_abs, p_abs, p_abs, q_abs)] -= rdm_element;
                    rdm[tei_index(q_abs, p_abs, q_abs, p_abs)] += rdm_element;
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

                        double rdm_element = 0.0;
                        std::vector<StringSubstitution>& VVOO =
                            alfa ? lists_->get_alfa_vvoo_list(p_abs, q_abs, r_abs, s_abs, ha)
                                 : lists_->get_beta_vvoo_list(p_abs, q_abs, r_abs, s_abs, hb);

                        // TODO loop in a differen way
                        size_t maxss = VVOO.size();
                        for (size_t ss = 0; ss < maxss; ++ss) {
                            double H = static_cast<double>(VVOO[ss].sign);
                            double* y = &(Ch[VVOO[ss].J][0]);
                            double* c = &(Ch[VVOO[ss].I][0]);
                            for (size_t L = 0; L < maxL; ++L) {
                                rdm_element += c[L] * y[L] * H;
                            }
                        }

                        rdm[tei_index(p_abs, q_abs, r_abs, s_abs)] += rdm_element;
                        rdm[tei_index(q_abs, p_abs, r_abs, s_abs)] -= rdm_element;
                        rdm[tei_index(p_abs, q_abs, s_abs, r_abs)] -= rdm_element;
                        rdm[tei_index(q_abs, p_abs, s_abs, r_abs)] += rdm_element;
                        rdm[tei_index(r_abs, s_abs, p_abs, q_abs)] += rdm_element;
                        rdm[tei_index(r_abs, s_abs, q_abs, p_abs)] -= rdm_element;
                        rdm[tei_index(s_abs, r_abs, p_abs, q_abs)] -= rdm_element;
                        rdm[tei_index(s_abs, r_abs, q_abs, p_abs)] += rdm_element;
                    }
                }
            }
        }
    } // End loop over h
#if 0
    outfile->Printf("\n TPDM:");
    for (int p = 0; p < ncmo_; ++p) {
        for (int q = 0; q <= p; ++q) {
            for (int r = 0; r < ncmo_; ++r) {
                for (int s = 0; s <= r; ++s) {
                    if (std::fabs(rdm[tei_index(p,q,r,s)]) > 1.0e-12){
                        outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.12lf", p,q,r,s, rdm[tei_index(p,q,r,s)]);

                    }
                }
            }
        }
    }
#endif
}

/**
 * Compute the ab two-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
void FCIVector::compute_2rdm_ab(std::vector<double>& rdm) {
    rdm.assign(ncmo_ * ncmo_ * ncmo_ * ncmo_, 0.0);

    // Loop over blocks of matrix C
    for (int Ia_sym = 0; Ia_sym < nirrep_; ++Ia_sym) {
        int Ib_sym = Ia_sym ^ symmetry_;
        double** C = C_[Ia_sym]->pointer();

        // Loop over all r,s
        for (int rs_sym = 0; rs_sym < nirrep_; ++rs_sym) {
            int Jb_sym = Ib_sym ^ rs_sym;
            int Ja_sym = Jb_sym ^ symmetry_;
            double** Y = C_[Ja_sym]->pointer();
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

                        // Loop over all p,q
                        int pq_sym = rs_sym;
                        for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
                            int q_sym = pq_sym ^ p_sym;
                            for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                                int p_abs = p_rel + cmopi_offset_[p_sym];
                                for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                                    int q_abs = q_rel + cmopi_offset_[q_sym];

                                    std::vector<StringSubstitution>& vo_alfa =
                                        lists_->get_alfa_vo_list(p_abs, q_abs, Ia_sym);

                                    size_t maxSSa = vo_alfa.size();
                                    for (size_t SSa = 0; SSa < maxSSa; ++SSa) {
                                        for (size_t SSb = 0; SSb < maxSSb; ++SSb) {
                                            double V = static_cast<double>(vo_alfa[SSa].sign *
                                                                           vo_beta[SSb].sign);
                                            rdm[tei_index(p_abs, r_abs, q_abs, s_abs)] +=
                                                Y[vo_alfa[SSa].J][vo_beta[SSb].J] *
                                                C[vo_alfa[SSa].I][vo_beta[SSb].I] * V;
                                        }
                                    }
                                }
                            }
                        } // End loop over p,q
                    }
                } // End loop over r_rel,s_rel
            }
        }
    }
#if 0
    outfile->Printf("\n TPDM (ab):");
    for (int p = 0; p < ncmo_; ++p) {
        for (int q = 0; q < ncmo_; ++q) {
            for (int r = 0; r < ncmo_; ++r) {
                for (int s = 0; s < ncmo_; ++s) {
                    if (std::fabs(rdm[tei_index(p,q,r,s)]) > 1.0e-12){
                        outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.12lf", p,q,r,s, rdm[tei_index(p,q,r,s)]);

                    }
                }
            }
        }
    }
#endif
}

void FCIVector::compute_3rdm_aaa(std::vector<double>& rdm, bool alfa) {
    rdm.assign(ncmo_ * ncmo_ * ncmo_ * ncmo_ * ncmo_ * ncmo_, 0.0);

    for (int h_K = 0; h_K < nirrep_; ++h_K) {
        size_t maxK =
            alfa ? lists_->alfa_graph_3h()->strpi(h_K) : lists_->beta_graph_3h()->strpi(h_K);
        for (int h_I = 0; h_I < nirrep_; ++h_I) {
            int h_Ib = h_I ^ symmetry_;
            int h_J = h_I;
            psi::SharedMatrix C = alfa ? C_[h_J] : C1;
            double** Ch = C->pointer();

            if (!alfa) {
                C->zero();
                size_t maxIa = alfa_graph_->strpi(h_I);
                size_t maxIb = beta_graph_->strpi(h_Ib);

                double** C0h = C_[h_I]->pointer();

                // Copy C0 transposed in C1
                for (size_t Ia = 0; Ia < maxIa; ++Ia)
                    for (size_t Ib = 0; Ib < maxIb; ++Ib)
                        Ch[Ib][Ia] = C0h[Ia][Ib];
            }

            size_t maxL = alfa ? beta_graph_->strpi(h_Ib) : alfa_graph_->strpi(h_I);
            if (maxL > 0) {
                for (size_t K = 0; K < maxK; ++K) {
                    std::vector<H3StringSubstitution>& Klist =
                        alfa ? lists_->get_alfa_3h_list(h_K, K, h_I)
                             : lists_->get_beta_3h_list(h_K, K, h_Ib);
                    for (const auto& Kel : Klist) {
                        size_t p = Kel.p;
                        size_t q = Kel.q;
                        size_t r = Kel.r;
                        size_t I = Kel.J;
                        for (const auto& Lel : Klist) {
                            size_t s = Lel.p;
                            size_t t = Lel.q;
                            size_t u = Lel.r;
                            short sign = Kel.sign * Lel.sign;
                            size_t J = Lel.J;

                            double rdm_value = 0.0;
                            rdm_value = C_DDOT(maxL, &(Ch[J][0]), 1, &(Ch[I][0]), 1);

                            rdm_value *= sign;

                            rdm[six_index(p, q, r, s, t, u)] += rdm_value;
                        }
                    }
                }
            }
        }
    }
}

void FCIVector::compute_3rdm_aab(std::vector<double>& rdm) {
    rdm.assign(ncmo_ * ncmo_ * ncmo_ * ncmo_ * ncmo_ * ncmo_, 0.0);

    for (int h_K = 0; h_K < nirrep_; ++h_K) {
        size_t maxK = lists_->alfa_graph_2h()->strpi(h_K);
        for (int h_L = 0; h_L < nirrep_; ++h_L) {
            size_t maxL = lists_->beta_graph_1h()->strpi(h_L);
            // I and J refer to the 2h part of the operator
            for (int h_Ia = 0; h_Ia < nirrep_; ++h_Ia) {
                int h_Mb = h_Ia ^ symmetry_;
                double** C_I_p = C_[h_Ia]->pointer();
                for (int h_Ja = 0; h_Ja < nirrep_; ++h_Ja) {
                    int h_Nb = h_Ja ^ symmetry_;
                    double** C_J_p = C_[h_Ja]->pointer();
                    for (size_t K = 0; K < maxK; ++K) {
                        std::vector<H2StringSubstitution>& Ilist =
                            lists_->get_alfa_2h_list(h_K, K, h_Ia);
                        std::vector<H2StringSubstitution>& Jlist =
                            lists_->get_alfa_2h_list(h_K, K, h_Ja);
                        for (size_t L = 0; L < maxL; ++L) {
                            std::vector<H1StringSubstitution>& Mlist =
                                lists_->get_beta_1h_list(h_L, L, h_Mb);
                            std::vector<H1StringSubstitution>& Nlist =
                                lists_->get_beta_1h_list(h_L, L, h_Nb);
                            for (const auto& Iel : Ilist) {
                                size_t q = Iel.p;
                                size_t p = Iel.q;
                                size_t I = Iel.J;
                                for (const auto& Jel : Jlist) {
                                    size_t t = Jel.p;
                                    size_t s = Jel.q;
                                    size_t J = Jel.J;
                                    for (const auto& Mel : Mlist) {
                                        size_t r = Mel.p;
                                        size_t M = Mel.J;
                                        for (const auto& Nel : Nlist) {
                                            size_t a = Nel.p;
                                            size_t N = Nel.J;
                                            short sign = Iel.sign * Jel.sign * Mel.sign * Nel.sign;
                                            rdm[six_index(p, q, r, s, t, a)] +=
                                                sign * C_I_p[I][M] * C_J_p[J][N];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void FCIVector::compute_3rdm_abb(std::vector<double>& rdm) {
    rdm.assign(ncmo_ * ncmo_ * ncmo_ * ncmo_ * ncmo_ * ncmo_, 0.0);

    for (int h_K = 0; h_K < nirrep_; ++h_K) {
        size_t maxK = lists_->alfa_graph_1h()->strpi(h_K);
        for (int h_L = 0; h_L < nirrep_; ++h_L) {
            size_t maxL = lists_->beta_graph_2h()->strpi(h_L);
            // I and J refer to the 1h part of the operator
            for (int h_Ia = 0; h_Ia < nirrep_; ++h_Ia) {
                int h_Mb = h_Ia ^ symmetry_;
                double** C_I_p = C_[h_Ia]->pointer();
                for (int h_Ja = 0; h_Ja < nirrep_; ++h_Ja) {
                    int h_Nb = h_Ja ^ symmetry_;
                    double** C_J_p = C_[h_Ja]->pointer();
                    for (size_t K = 0; K < maxK; ++K) {
                        std::vector<H1StringSubstitution>& Ilist =
                            lists_->get_alfa_1h_list(h_K, K, h_Ia);
                        std::vector<H1StringSubstitution>& Jlist =
                            lists_->get_alfa_1h_list(h_K, K, h_Ja);
                        for (size_t L = 0; L < maxL; ++L) {
                            std::vector<H2StringSubstitution>& Mlist =
                                lists_->get_beta_2h_list(h_L, L, h_Mb);
                            std::vector<H2StringSubstitution>& Nlist =
                                lists_->get_beta_2h_list(h_L, L, h_Nb);
                            for (size_t Iel = 0; Iel < Ilist.size(); Iel++) {
                                size_t p = Ilist[Iel].p;
                                size_t I = Ilist[Iel].J;
                                for (const auto& Mel : Mlist) {
                                    size_t q = Mel.p;
                                    size_t r = Mel.q;
                                    size_t M = Mel.J;
                                    for (const auto& Jel : Jlist) {
                                        size_t s = Jel.p;
                                        size_t J = Jel.J;
                                        for (const auto& Nel : Nlist) {
                                            size_t t = Nel.p;
                                            size_t a = Nel.q;
                                            size_t N = Nel.J;
                                            short sign =
                                                Ilist[Iel].sign * Jel.sign * Mel.sign * Nel.sign;
                                            rdm[six_index(p, q, r, s, t, a)] +=
                                                sign * C_I_p[I][M] * C_J_p[J][N];
                                            //}//End of if statement
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void FCIVector::rdm_test() {
    bool* Ia = new bool[ncmo_];
    bool* Ib = new bool[ncmo_];

    // Generate the strings 1111100000
    //                      { k }{n-k}

    size_t na = lists_->na();
    size_t nb = lists_->nb();

    for (size_t i = 0; i < ncmo_ - na; ++i)
        Ia[i] = false; // 0
    for (size_t i = ncmo_ - na; i < ncmo_; ++i)
        Ia[i] = true; // 1

    for (size_t i = 0; i < ncmo_ - nb; ++i)
        Ib[i] = false; // 0
    for (size_t i = ncmo_ - nb; i < ncmo_; ++i)
        Ib[i] = true; // 1

    std::vector<Determinant> dets;
    std::map<Determinant, size_t> dets_map;

    std::vector<double> C;
    std::vector<bool> a_occ(ncmo_);
    std::vector<bool> b_occ(ncmo_);

    size_t num_det = 0;
    do {
        for (size_t i = 0; i < ncmo_; ++i)
            a_occ[i] = Ia[i];
        do {
            for (size_t i = 0; i < ncmo_; ++i)
                b_occ[i] = Ib[i];
            if ((alfa_graph_->sym(Ia) ^ beta_graph_->sym(Ib)) == static_cast<int>(symmetry_)) {
                Determinant d(a_occ, b_occ);
                dets.push_back(d);
                double c = C_[alfa_graph_->sym(Ia)]->get(alfa_graph_->rel_add(Ia),
                                                         beta_graph_->rel_add(Ib));
                C.push_back(c);
                dets_map[d] = num_det;
                num_det++;
            }
        } while (std::next_permutation(Ib, Ib + ncmo_));
    } while (std::next_permutation(Ia, Ia + ncmo_));

    Determinant I; // <- xsize (ncmo_);

    bool test_2rdm_aa = true;
    bool test_2rdm_bb = true;
    bool test_2rdm_ab = true;
    bool test_3rdm_aaa = true;
    bool test_3rdm_bbb = true;
    bool test_3rdm_aab = true;
    bool test_3rdm_abb = true;

    outfile->Printf("\n\n==> RDMs Test <==\n");
    if (test_2rdm_aa) {
        double error_2rdm_aa = 0.0;
        for (size_t p = 0; p < ncmo_; ++p) {
            for (size_t q = 0; q < ncmo_; ++q) {
                for (size_t r = 0; r < ncmo_; ++r) {
                    for (size_t s = 0; s < ncmo_; ++s) {
                        double rdm = 0.0;
                        for (size_t i = 0; i < dets.size(); ++i) {
                            I = dets[i];
                            double sign = 1.0;
                            sign *= I.destroy_alfa_bit(r);
                            sign *= I.destroy_alfa_bit(s);
                            sign *= I.create_alfa_bit(q);
                            sign *= I.create_alfa_bit(p);
                            if (sign != 0) {
                                if (dets_map.count(I) != 0) {
                                    rdm += sign * C[i] * C[dets_map[I]];
                                }
                            }
                        }
                        if (std::fabs(rdm) > 1.0e-12) {
                            //                            outfile->Printf("\n
                            //                            D2(aaaa)[%3lu][%3lu][%3lu][%3lu] =
                            //                            %18.12lf "
                            //                                            "(%18.12lf,%18.12lf)",
                            //                                            p, q, r, s, rdm -
                            //                                            tpdm_aa_[tei_index(p, q,
                            //                                            r,
                            //                                            s)], rdm,
                            //                                            tpdm_aa_[tei_index(p, q,
                            //                                            r,
                            //                                            s)]);
                            error_2rdm_aa += std::fabs(rdm - tpdm_aa_[tei_index(p, q, r, s)]);
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["AAAA 2-RDM ERROR"] = error_2rdm_aa;
        outfile->Printf("\n    AAAA 2-RDM Error :   %+e", error_2rdm_aa);
    }

    if (test_2rdm_bb) {
        double error_2rdm_bb = 0.0;
        for (size_t p = 0; p < ncmo_; ++p) {
            for (size_t q = 0; q < ncmo_; ++q) {
                for (size_t r = 0; r < ncmo_; ++r) {
                    for (size_t s = 0; s < ncmo_; ++s) {
                        double rdm = 0.0;
                        for (size_t i = 0; i < dets.size(); ++i) {
                            I = dets[i];
                            double sign = 1.0;
                            sign *= I.destroy_beta_bit(r);
                            sign *= I.destroy_beta_bit(s);
                            sign *= I.create_beta_bit(q);
                            sign *= I.create_beta_bit(p);
                            if (sign != 0) {
                                if (dets_map.count(I) != 0) {
                                    rdm += sign * C[i] * C[dets_map[I]];
                                }
                            }
                        }
                        if (std::fabs(rdm) > 1.0e-12) {
                            //                            outfile->Printf("\n
                            //                            D2(bbbb)[%3lu][%3lu][%3lu][%3lu]
                            //                            = %18.12lf
                            //                            (%18.12lf,%18.12lf)",
                            //                            p,q,r,s,rdm-tpdm_bb_[tei_index(p,q,r,s)],rdm,tpdm_bb_[tei_index(p,q,r,s)]);
                            error_2rdm_bb += std::fabs(rdm - tpdm_bb_[tei_index(p, q, r, s)]);
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["BBBB 2-RDM ERROR"] = error_2rdm_bb;
        outfile->Printf("\n    BBBB 2-RDM Error :   %+e", error_2rdm_bb);
    }

    if (test_2rdm_ab) {
        double error_2rdm_ab = 0.0;
        for (size_t p = 0; p < ncmo_; ++p) {
            for (size_t q = 0; q < ncmo_; ++q) {
                for (size_t r = 0; r < ncmo_; ++r) {
                    for (size_t s = 0; s < ncmo_; ++s) {
                        double rdm = 0.0;
                        for (size_t i = 0; i < dets.size(); ++i) {
                            I = dets[i];
                            double sign = 1.0;
                            sign *= I.destroy_alfa_bit(r);
                            sign *= I.destroy_beta_bit(s);
                            sign *= I.create_beta_bit(q);
                            sign *= I.create_alfa_bit(p);
                            if (sign != 0) {
                                if (dets_map.count(I) != 0) {
                                    rdm += sign * C[i] * C[dets_map[I]];
                                }
                            }
                        }
                        if (std::fabs(rdm) > 1.0e-12) {
                            //                            outfile->Printf("\n
                            //                            D2(abab)[%3lu][%3lu][%3lu][%3lu]
                            //                            = %18.12lf
                            //                            (%18.12lf,%18.12lf)",
                            //                            p,q,r,s,rdm-tpdm_ab_[tei_index(p,q,r,s)],rdm,tpdm_ab_[tei_index(p,q,r,s)]);
                            error_2rdm_ab += std::fabs(rdm - tpdm_ab_[tei_index(p, q, r, s)]);
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["ABAB 2-RDM ERROR"] = error_2rdm_ab;
        outfile->Printf("\n    ABAB 2-RDM Error :   %+e", error_2rdm_ab);
    }

    if (test_3rdm_aab) {
        double error_3rdm_aab = 0.0;
        for (size_t p = 0; p < ncmo_; ++p) {
            for (size_t q = p + 1; q < ncmo_; ++q) {
                for (size_t r = 0; r < ncmo_; ++r) {
                    for (size_t s = 0; s < ncmo_; ++s) {
                        for (size_t t = s + 1; t < ncmo_; ++t) {
                            for (size_t a = 0; a < ncmo_; ++a) {
                                double rdm = 0.0;
                                for (size_t i = 0; i < dets.size(); ++i) {
                                    I = dets[i];
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_alfa_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_alfa_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (dets_map.count(I) != 0) {
                                            rdm += sign * C[i] * C[dets_map[I]];
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12) {
                                    size_t index = six_index(p, q, r, s, t, a);
                                    double rdm_comp = tpdm_aab_[index];
                                    //                                    outfile->Printf("\n
                                    //                                    D3(aabaab)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu]
                                    //                                    =
                                    //                                    %18.12lf
                                    //                                    (%18.12lf,%18.12lf)",
                                    //                                                    p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
                                    error_3rdm_aab += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["AABAAB 3-RDM ERROR"] = error_3rdm_aab;
        outfile->Printf("\n    AABAAB 3-RDM Error : %+e", error_3rdm_aab);
    }

    if (test_3rdm_abb) {
        double error_3rdm_abb = 0.0;
        for (size_t p = 0; p < ncmo_; ++p) {
            for (size_t q = p + 1; q < ncmo_; ++q) {
                for (size_t r = 0; r < ncmo_; ++r) {
                    for (size_t s = 0; s < ncmo_; ++s) {
                        for (size_t t = s + 1; t < ncmo_; ++t) {
                            for (size_t a = 0; a < ncmo_; ++a) {
                                double rdm = 0.0;
                                for (size_t i = 0; i < dets.size(); ++i) {
                                    I = dets[i];
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_beta_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_beta_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (dets_map.count(I) != 0) {
                                            rdm += sign * C[i] * C[dets_map[I]];
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12) {
                                    size_t index = six_index(p, q, r, s, t, a);
                                    double rdm_comp = tpdm_abb_[index];
                                    //                                    outfile->Printf("\n
                                    //                                    D3(abbabb)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu]
                                    //                                    =
                                    //                                    %18.12lf
                                    //                                    (%18.12lf,%18.12lf)",
                                    //                                                    p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
                                    error_3rdm_abb += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["ABBABB 3-RDM ERROR"] = error_3rdm_abb;
        outfile->Printf("\n    ABBABB 3-RDM Error : %+e", error_3rdm_abb);
    }

    if (test_3rdm_aaa) {
        double error_3rdm_aaa = 0.0;
        //        for (size_t p = 0; p < ncmo_; ++p){
        for (size_t p = 0; p < 1; ++p) {
            for (size_t q = p + 1; q < ncmo_; ++q) {
                for (size_t r = q + 1; r < ncmo_; ++r) {
                    for (size_t s = 0; s < ncmo_; ++s) {
                        for (size_t t = s + 1; t < ncmo_; ++t) {
                            for (size_t a = t + 1; a < ncmo_; ++a) {
                                double rdm = 0.0;
                                for (size_t i = 0; i < dets.size(); ++i) {
                                    I = dets[i];
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_alfa_bit(t);
                                    sign *= I.destroy_alfa_bit(a);
                                    sign *= I.create_alfa_bit(r);
                                    sign *= I.create_alfa_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (dets_map.count(I) != 0) {
                                            rdm += sign * C[i] * C[dets_map[I]];
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12) {
                                    size_t index = six_index(p, q, r, s, t, a);
                                    double rdm_comp = tpdm_aaa_[index];
                                    //                                    outfile->Printf("\n
                                    //                                    D3(aaaaaa)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu]
                                    //                                    =
                                    //                                    %18.12lf
                                    //                                    (%18.12lf,%18.12lf)",
                                    //                                                    p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
                                    error_3rdm_aaa += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["AAAAAA 3-RDM ERROR"] = error_3rdm_aaa;
        outfile->Printf("\n    AAAAAA 3-RDM Error : %+e", error_3rdm_aaa);
    }

    if (test_3rdm_bbb) {
        double error_3rdm_bbb = 0.0;
        for (size_t p = 0; p < 1; ++p) {
            //            for (size_t p = 0; p < ncmo_; ++p){
            for (size_t q = p + 1; q < ncmo_; ++q) {
                for (size_t r = q + 1; r < ncmo_; ++r) {
                    for (size_t s = 0; s < ncmo_; ++s) {
                        for (size_t t = s + 1; t < ncmo_; ++t) {
                            for (size_t a = t + 1; a < ncmo_; ++a) {
                                double rdm = 0.0;
                                for (size_t i = 0; i < dets.size(); ++i) {
                                    I = dets[i];
                                    double sign = 1.0;
                                    sign *= I.destroy_beta_bit(s);
                                    sign *= I.destroy_beta_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_beta_bit(q);
                                    sign *= I.create_beta_bit(p);
                                    if (sign != 0) {
                                        if (dets_map.count(I) != 0) {
                                            rdm += sign * C[i] * C[dets_map[I]];
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12) {
                                    size_t index = six_index(p, q, r, s, t, a);
                                    double rdm_comp = tpdm_bbb_[index];
                                    //                                    outfile->Printf("\n
                                    //                                    D3(bbbbbb)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu]
                                    //                                    =
                                    //                                    %18.12lf
                                    //                                    (%18.12lf,%18.12lf)",
                                    //                                                    p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
                                    error_3rdm_bbb += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["BBBBBB 3-RDM ERROR"] = error_3rdm_bbb;
        outfile->Printf("\n    BBBBBB 3-RDM Error : %+e", error_3rdm_bbb);
    }
    delete[] Ia;
    delete[] Ib;
}

double FCIVector::compute_spin2() {
    double spin2 = 0.0;
    // Loop over blocks of matrix C
    for (int Ia_sym = 0; Ia_sym < nirrep_; ++Ia_sym) {
        const int Ib_sym = Ia_sym ^ symmetry_;
        double** C = C_[Ia_sym]->pointer();

        // Loop over all r,s
        for (int rs_sym = 0; rs_sym < nirrep_; ++rs_sym) {
            const int Jb_sym = Ib_sym ^ rs_sym;
            const int Ja_sym = Jb_sym ^ symmetry_;
            double** Y = C_[Ja_sym]->pointer();
            for (int r_sym = 0; r_sym < nirrep_; ++r_sym) {
                int s_sym = rs_sym ^ r_sym;

                for (int r_rel = 0; r_rel < cmopi_[r_sym]; ++r_rel) {
                        const int r_abs = r_rel + cmopi_offset_[r_sym];
                    for (int s_rel = 0; s_rel < cmopi_[s_sym]; ++s_rel) {
                        const int s_abs = s_rel + cmopi_offset_[s_sym];

                        // Grab list (r,s,Ib_sym)
                        const auto& vo_alfa =
                            lists_->get_alfa_vo_list(s_abs, r_abs, Ia_sym);
                        const auto& vo_beta =
                            lists_->get_beta_vo_list(r_abs, s_abs, Ib_sym);

                        const size_t maxSSa = vo_alfa.size();
                        const size_t maxSSb = vo_beta.size();

                        for (size_t SSa = 0; SSa < maxSSa; ++SSa) {
                            for (size_t SSb = 0; SSb < maxSSb; ++SSb) {
                                spin2 += Y[vo_alfa[SSa].J][vo_beta[SSb].J] *
                                         C[vo_alfa[SSa].I][vo_beta[SSb].I] * static_cast<double>(vo_alfa[SSa].sign * vo_beta[SSb].sign);
                            }
                        }
                    }
                } // End loop over r_rel,s_rel
            }
        }
    }
    double na = alfa_graph_->nones();
    double nb = beta_graph_->nones();
    return -spin2 + 0.25 * std::pow(na - nb, 2.0) + 0.5 * (na + nb);
}

} // namespace forte
