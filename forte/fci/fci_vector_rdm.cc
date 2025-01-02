/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/matrix.h"

#include "fci_string_lists.h"
#include "fci_string_address.h"

#include "fci_vector.h"

namespace forte {

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
std::shared_ptr<RDMs> FCIVector::compute_rdms(FCIVector& C_left, FCIVector& C_right,
                                              int max_rdm_level, RDMsType type) {
    std::vector<double> rdm_timing;

    size_t na = C_left.alfa_address()->nones();
    size_t nb = C_left.beta_address()->nones();
    size_t nmo = C_left.ncmo_;

    ambit::Tensor g1a, g1b;
    ambit::Tensor g2aa, g2ab, g2bb;
    ambit::Tensor g3aaa, g3aab, g3abb, g3bbb;

    if (max_rdm_level >= 1) {
        local_timer t;
        g1a = compute_1rdm_same_irrep(C_left, C_right, true);
        g1b = compute_1rdm_same_irrep(C_left, C_right, false);
        rdm_timing.push_back(t.get());
    }

    if (max_rdm_level >= 2) {
        local_timer t;
        g2aa = compute_2rdm_aa_same_irrep(C_left, C_right, true);
        g2bb = compute_2rdm_aa_same_irrep(C_left, C_right, false);
        g2ab = compute_2rdm_ab_same_irrep(C_left, C_right);
        rdm_timing.push_back(t.get());
    }

    if (max_rdm_level >= 3) {
        local_timer t;
        if (na >= 3) {
            g3aaa = compute_3rdm_aaa_same_irrep(C_left, C_right, true);
        } else {
            g3aaa =
                ambit::Tensor::build(ambit::CoreTensor, "g3aaa", {nmo, nmo, nmo, nmo, nmo, nmo});
            g3aaa.zero();
        }
        if (nb >= 3) {
            g3bbb = compute_3rdm_aaa_same_irrep(C_left, C_right, false);
        } else {
            g3bbb =
                ambit::Tensor::build(ambit::CoreTensor, "g3bbb", {nmo, nmo, nmo, nmo, nmo, nmo});
            g3bbb.zero();
        }

        if ((na >= 2) and (nb >= 1)) {
            g3aab = compute_3rdm_aab_same_irrep(C_left, C_right);
        } else {
            g3aab =
                ambit::Tensor::build(ambit::CoreTensor, "g3aab", {nmo, nmo, nmo, nmo, nmo, nmo});
            g3aab.zero();
        }

        if ((na >= 1) and (nb >= 2)) {
            g3abb = compute_3rdm_abb_same_irrep(C_left, C_right);
        } else {
            g3abb =
                ambit::Tensor::build(ambit::CoreTensor, "g3abb", {nmo, nmo, nmo, nmo, nmo, nmo});
            g3abb.zero();
        }
        rdm_timing.push_back(t.get());
    }

    // for (size_t n = 0; n < rdm_timing.size(); ++n) {
    //     psi::outfile->Printf("\n    Timing for %d-RDM: %.3f s", n + 1, rdm_timing[n]);
    // }

    if (type == RDMsType::spin_dependent) {
        if (max_rdm_level == 1) {
            return std::make_shared<RDMsSpinDependent>(g1a, g1b);
        }
        if (max_rdm_level == 2) {
            return std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb);
        }
        if (max_rdm_level == 3) {
            return std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab,
                                                       g3abb, g3bbb);
        }
    } else {
        g1a("pq") += g1b("pq");

        if (max_rdm_level > 1) {
            g2aa("pqrs") += g2ab("pqrs") + g2ab("qpsr");
            g2aa("pqrs") += g2bb("pqrs");
        }
        if (max_rdm_level > 2) {
            g3aaa("pqrstu") += g3aab("pqrstu") + g3aab("prqsut") + g3aab("qrptus");
            g3aaa("pqrstu") += g3abb("pqrstu") + g3abb("qprtsu") + g3abb("rpqust");
            g3aaa("pqrstu") += g3bbb("pqrstu");
        }
        if (max_rdm_level == 1)
            return std::make_shared<RDMsSpinFree>(g1a);
        if (max_rdm_level == 2)
            return std::make_shared<RDMsSpinFree>(g1a, g2aa);
        if (max_rdm_level == 3)
            return std::make_shared<RDMsSpinFree>(g1a, g2aa, g3aaa);
    }

    if (max_rdm_level >= 4) {
        throw std::runtime_error("RDMs of order 4 or higher are not implemented in FCISolver (and "
                                 "more generally in Forte).");
    }
    return std::make_shared<RDMsSpinDependent>();
}

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
ambit::Tensor FCIVector::compute_1rdm_same_irrep(FCIVector& C_left, FCIVector& C_right, bool alfa) {
    size_t ncmo = C_left.ncmo_;
    size_t nirrep = C_left.nirrep_;
    size_t symmetry = C_left.symmetry_;
    const auto& detpi = C_left.detpi_;
    const auto& alfa_address = C_left.alfa_address_;
    const auto& beta_address = C_left.beta_address_;
    const auto& cmopi = C_left.cmopi_;
    const auto& cmopi_offset = C_left.cmopi_offset_;
    const auto& lists = C_left.lists_;

    auto rdm = ambit::Tensor::build(ambit::CoreTensor, alfa ? "1RDM_A" : "1RDM_B", {ncmo, ncmo});

    auto na = alfa_address->nones();
    auto nb = beta_address->nones();
    if ((alfa and (na < 1)) or ((!alfa) and (nb < 1)))
        return rdm;

    auto& rdm_data = rdm.data();

    for (size_t h_Ia = 0; h_Ia < nirrep; ++h_Ia) {
        int h_Ib = h_Ia ^ symmetry;
        if (detpi[h_Ia] > 0) {
            // Get a pointer to the correct block of matrix C
            auto Cl =
                gather_C_block(C_left, CL, alfa, alfa_address, beta_address, h_Ia, h_Ib, false);
            auto Cr =
                gather_C_block(C_right, CR, alfa, alfa_address, beta_address, h_Ia, h_Ib, false);

            const size_t maxL = alfa ? beta_address->strpcls(h_Ib) : alfa_address->strpcls(h_Ia);
            for (size_t p_sym = 0; p_sym < nirrep; ++p_sym) {
                int q_sym = p_sym; // Select the totat symmetric irrep
                for (int p_rel = 0; p_rel < cmopi[p_sym]; ++p_rel) {
                    for (int q_rel = 0; q_rel < cmopi[q_sym]; ++q_rel) {
                        int p_abs = p_rel + cmopi_offset[p_sym];
                        int q_abs = q_rel + cmopi_offset[q_sym];
                        const auto& vo = alfa ? lists->get_alfa_vo_list(p_abs, q_abs, h_Ia)
                                              : lists->get_beta_vo_list(p_abs, q_abs, h_Ib);
                        double rdm_element = 0.0;
                        for (const auto& [sign, I, J] : vo) {
                            rdm_element += sign * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
                        }
                        rdm_data[p_abs * ncmo + q_abs] += rdm_element;
                    }
                }
            }
        }
    } // End loop over h
    return rdm;
}

/**
 * Compute the aa/bb two-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
ambit::Tensor FCIVector::compute_2rdm_aa_same_irrep(FCIVector& C_left, FCIVector& C_right,
                                                    bool alfa) {
    int nirrep = C_left.nirrep_;
    size_t ncmo = C_left.ncmo_;
    size_t symmetry = C_left.symmetry_;
    const auto& detpi = C_left.detpi_;
    const auto& alfa_address = C_left.alfa_address_;
    const auto& beta_address = C_left.beta_address_;
    const auto& lists = C_left.lists_;

    auto rdm = ambit::Tensor::build(ambit::CoreTensor, alfa ? "2RDM_AA" : "2RDM_BB",
                                    {ncmo, ncmo, ncmo, ncmo});

    auto na = alfa_address->nones();
    auto nb = beta_address->nones();
    if ((alfa and (na < 2)) or ((!alfa) and (nb < 2)))
        return rdm;

    auto& rdm_data = rdm.data();

    // Notation
    // h_Ia - symmetry of alpha strings
    // h_Ib - symmetry of beta strings
    for (int h_Ia = 0; h_Ia < nirrep; ++h_Ia) {
        int h_Ib = h_Ia ^ symmetry;
        if (detpi[h_Ia] > 0) {
            // Get a pointer to the correct block of matrix C
            auto Cl =
                gather_C_block(C_left, CL, alfa, alfa_address, beta_address, h_Ia, h_Ib, false);
            auto Cr =
                gather_C_block(C_right, CR, alfa, alfa_address, beta_address, h_Ia, h_Ib, false);

            size_t maxL = alfa ? beta_address->strpcls(h_Ib) : alfa_address->strpcls(h_Ia);
            // Loop over (p>q) == (p>q)
            for (int pq_sym = 0; pq_sym < nirrep; ++pq_sym) {
                size_t max_pq = lists->pairpi(pq_sym);
                for (size_t pq = 0; pq < max_pq; ++pq) {
                    const auto& [p_abs, q_abs] = lists->get_pair_list(pq_sym, pq);

                    std::vector<StringSubstitution>& OO =
                        alfa ? lists->get_alfa_oo_list(pq_sym, pq, h_Ia)
                             : lists->get_beta_oo_list(pq_sym, pq, h_Ib);

                    double rdm_element = 0.0;
                    for (const auto& [sign, I, J] : OO) {
                        rdm_element += sign * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
                    }

                    rdm_data[tei_index(p_abs, q_abs, p_abs, q_abs, ncmo)] += rdm_element;
                    rdm_data[tei_index(p_abs, q_abs, q_abs, p_abs, ncmo)] -= rdm_element;
                    rdm_data[tei_index(q_abs, p_abs, p_abs, q_abs, ncmo)] -= rdm_element;
                    rdm_data[tei_index(q_abs, p_abs, q_abs, p_abs, ncmo)] += rdm_element;
                }
            }
            // Loop over (p>q) > (r>s)
            for (int pq_sym = 0; pq_sym < nirrep; ++pq_sym) {
                size_t max_pq = lists->pairpi(pq_sym);
                for (size_t pq = 0; pq < max_pq; ++pq) {
                    const auto& [p_abs, q_abs] = lists->get_pair_list(pq_sym, pq);
                    for (size_t rs = 0; rs < pq; ++rs) {
                        const auto& [r_abs, s_abs] = lists->get_pair_list(pq_sym, rs);

                        const auto& VVOO =
                            alfa ? lists->get_alfa_vvoo_list(p_abs, q_abs, r_abs, s_abs, h_Ia)
                                 : lists->get_beta_vvoo_list(p_abs, q_abs, r_abs, s_abs, h_Ib);

                        double rdm_element = 0.0;
                        for (const auto& [sign, I, J] : VVOO) {
                            rdm_element += sign * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
                        }

                        rdm_data[tei_index(p_abs, q_abs, r_abs, s_abs, ncmo)] += rdm_element;
                        rdm_data[tei_index(q_abs, p_abs, r_abs, s_abs, ncmo)] -= rdm_element;
                        rdm_data[tei_index(p_abs, q_abs, s_abs, r_abs, ncmo)] -= rdm_element;
                        rdm_data[tei_index(q_abs, p_abs, s_abs, r_abs, ncmo)] += rdm_element;
                        rdm_data[tei_index(r_abs, s_abs, p_abs, q_abs, ncmo)] += rdm_element;
                        rdm_data[tei_index(r_abs, s_abs, q_abs, p_abs, ncmo)] -= rdm_element;
                        rdm_data[tei_index(s_abs, r_abs, p_abs, q_abs, ncmo)] -= rdm_element;
                        rdm_data[tei_index(s_abs, r_abs, q_abs, p_abs, ncmo)] += rdm_element;
                    }
                }
            }
        }
    } // End loop over h
#if 0
    psi::outfile->Printf("\n TPDM:");
    for (int p = 0; p < no_; ++p) {
        for (int q = 0; q <= p; ++q) {
            for (int r = 0; r < no_; ++r) {
                for (int s = 0; s <= r; ++s) {
                    if (std::fabs(rdm[tei_index(p,q,r,s)]) > 1.0e-12){
                        psi::outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.12lf", p,q,r,s, rdm[tei_index(p,q,r,s)]);

                    }
                }
            }
        }
    }
#endif
    return rdm;
}

ambit::Tensor FCIVector::compute_2rdm_ab_same_irrep(FCIVector& C_left, FCIVector& C_right) {
    int nirrep = C_left.nirrep_;
    size_t ncmo = C_left.ncmo_;
    size_t symmetry = C_left.symmetry_;
    const auto& alfa_address = C_left.alfa_address_;
    const auto& beta_address = C_left.beta_address_;
    const auto& cmopi = C_left.cmopi_;
    const auto& cmopi_offset = C_left.cmopi_offset_;
    const auto& lists = C_left.lists_;

    auto rdm = ambit::Tensor::build(ambit::CoreTensor, "2RDM_AB", {ncmo, ncmo, ncmo, ncmo});

    auto na = alfa_address->nones();
    auto nb = beta_address->nones();
    if ((na < 1) or (nb < 1))
        return rdm;

    auto& rdm_data = rdm.data();

    // Loop over blocks of matrix C
    for (int Ia_sym = 0; Ia_sym < nirrep; ++Ia_sym) {
        int Ib_sym = Ia_sym ^ symmetry;
        const auto Cr = C_right.C(Ia_sym)->pointer();

        // Loop over all r,s
        for (int rs_sym = 0; rs_sym < nirrep; ++rs_sym) {
            int Jb_sym = Ib_sym ^ rs_sym;
            int Ja_sym = Jb_sym ^ symmetry;
            const auto Cl = C_left.C(Ja_sym)->pointer();
            for (int r_sym = 0; r_sym < nirrep; ++r_sym) {
                int s_sym = rs_sym ^ r_sym;

                for (int r_rel = 0; r_rel < cmopi[r_sym]; ++r_rel) {
                    for (int s_rel = 0; s_rel < cmopi[s_sym]; ++s_rel) {
                        int r_abs = r_rel + cmopi_offset[r_sym];
                        int s_abs = s_rel + cmopi_offset[s_sym];

                        // Grab list (r,s,Ib_sym)
                        const auto& vo_beta = lists->get_beta_vo_list(r_abs, s_abs, Ib_sym);

                        // Loop over all p,q
                        int pq_sym = rs_sym;
                        for (int p_sym = 0; p_sym < nirrep; ++p_sym) {
                            int q_sym = pq_sym ^ p_sym;
                            for (int p_rel = 0; p_rel < cmopi[p_sym]; ++p_rel) {
                                int p_abs = p_rel + cmopi_offset[p_sym];
                                for (int q_rel = 0; q_rel < cmopi[q_sym]; ++q_rel) {
                                    int q_abs = q_rel + cmopi_offset[q_sym];

                                    const auto& vo_alfa =
                                        lists->get_alfa_vo_list(p_abs, q_abs, Ia_sym);

                                    double rdm_element = 0.0;
                                    for (const auto& [sign_a, Ia, Ja] : vo_alfa) {
                                        for (const auto& [sign_b, Ib, Jb] : vo_beta) {
                                            rdm_element +=
                                                Cl[Ja][Jb] * Cr[Ia][Ib] * sign_a * sign_b;
                                        }
                                    }
                                    rdm_data[tei_index(p_abs, r_abs, q_abs, s_abs, ncmo)] +=
                                        rdm_element;
                                }
                            }
                        } // End loop over p,q
                    }
                } // End loop over r_rel,s_rel
            }
        }
    }
#if 0
    psi::outfile->Printf("\n TPDM (ab):");
    for (int p = 0; p < no_; ++p) {
        for (int q = 0; q < no_; ++q) {
            for (int r = 0; r < no_; ++r) {
                for (int s = 0; s < no_; ++s) {
                    if (std::fabs(rdm[tei_index(p,q,r,s)]) > 1.0e-12){
                        psi::outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.12lf", p,q,r,s, rdm[tei_index(p,q,r,s)]);

                    }
                }
            }
        }
    }
#endif
    return rdm;
}

ambit::Tensor FCIVector::compute_3rdm_aaa_same_irrep(FCIVector& C_left, FCIVector& C_right,
                                                     bool alfa) {
    int nirrep = C_left.nirrep_;
    size_t ncmo = C_left.ncmo_;
    size_t symmetry = C_left.symmetry_;
    const auto& alfa_address = C_left.alfa_address_;
    const auto& beta_address = C_left.beta_address_;
    const auto& lists = C_left.lists_;

    auto rdm = ambit::Tensor::build(ambit::CoreTensor, alfa ? "3RDM_AAA" : "3RDM_BBB",
                                    {ncmo, ncmo, ncmo, ncmo, ncmo, ncmo});

    auto na = alfa_address->nones();
    auto nb = beta_address->nones();
    if ((alfa and (na < 3)) or ((!alfa) and (nb < 3)))
        return rdm;

    auto& rdm_data = rdm.data();

    for (int h_K = 0; h_K < nirrep; ++h_K) {
        size_t maxK =
            alfa ? lists->alfa_address_3h()->strpcls(h_K) : lists->beta_address_3h()->strpcls(h_K);
        for (int h_Ia = 0; h_Ia < nirrep; ++h_Ia) {
            int h_Ib = h_Ia ^ symmetry;
            // Get a pointer to the correct block of matrix C
            auto Cl =
                gather_C_block(C_left, CL, alfa, alfa_address, beta_address, h_Ia, h_Ib, false);
            auto Cr =
                gather_C_block(C_right, CR, alfa, alfa_address, beta_address, h_Ia, h_Ib, false);

            size_t maxL = alfa ? beta_address->strpcls(h_Ib) : alfa_address->strpcls(h_Ia);
            if (maxL > 0) {
                for (size_t K = 0; K < maxK; ++K) {
                    std::vector<H3StringSubstitution>& Klist =
                        alfa ? lists->get_alfa_3h_list(h_K, K, h_Ia)
                             : lists->get_beta_3h_list(h_K, K, h_Ib);
                    for (const auto& [sign_K, p, q, r, I] : Klist) {
                        for (const auto& [sign_L, s, t, u, J] : Klist) {
                            rdm_data[six_index(p, q, r, s, t, u, ncmo)] +=
                                sign_K * sign_L * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

ambit::Tensor FCIVector::compute_3rdm_aab_same_irrep(FCIVector& C_left, FCIVector& C_right) {
    int nirrep = C_left.nirrep_;
    size_t ncmo = C_left.ncmo_;
    size_t symmetry = C_left.symmetry_;
    const auto& lists = C_left.lists_;

    auto g3 = ambit::Tensor::build(ambit::CoreTensor, "g2", {ncmo, ncmo, ncmo, ncmo, ncmo, ncmo});
    g3.zero();
    auto& rdm = g3.data();

    for (int h_K = 0; h_K < nirrep; ++h_K) {
        size_t maxK = lists->alfa_address_2h()->strpcls(h_K);
        for (int h_L = 0; h_L < nirrep; ++h_L) {
            size_t maxL = lists->beta_address_1h()->strpcls(h_L);
            // I and J refer to the 2h part of the operator
            for (int h_Ia = 0; h_Ia < nirrep; ++h_Ia) {
                int h_Mb = h_Ia ^ symmetry;
                double** C_I_p = C_right.C(h_Ia)->pointer();
                for (int h_Ja = 0; h_Ja < nirrep; ++h_Ja) {
                    int h_Nb = h_Ja ^ symmetry;
                    double** C_J_p = C_left.C(h_Ja)->pointer();
                    for (size_t K = 0; K < maxK; ++K) {
                        std::vector<H2StringSubstitution>& Ilist =
                            lists->get_alfa_2h_list(h_K, K, h_Ia);
                        std::vector<H2StringSubstitution>& Jlist =
                            lists->get_alfa_2h_list(h_K, K, h_Ja);
                        for (size_t L = 0; L < maxL; ++L) {
                            std::vector<H1StringSubstitution>& Mlist =
                                lists->get_beta_1h_list(h_L, L, h_Mb);
                            std::vector<H1StringSubstitution>& Nlist =
                                lists->get_beta_1h_list(h_L, L, h_Nb);
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
                                            rdm[six_index(p, q, r, s, t, a, ncmo)] +=
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
    return g3;
}

ambit::Tensor FCIVector::compute_3rdm_abb_same_irrep(FCIVector& C_left, FCIVector& C_right) {
    int nirrep = C_left.nirrep_;
    size_t ncmo = C_left.ncmo_;
    size_t symmetry = C_left.symmetry_;
    const auto& lists = C_left.lists_;

    auto g3 = ambit::Tensor::build(ambit::CoreTensor, "g2", {ncmo, ncmo, ncmo, ncmo, ncmo, ncmo});
    g3.zero();
    auto& rdm = g3.data();

    for (int h_K = 0; h_K < nirrep; ++h_K) {
        size_t maxK = lists->alfa_address_1h()->strpcls(h_K);
        for (int h_L = 0; h_L < nirrep; ++h_L) {
            size_t maxL = lists->beta_address_2h()->strpcls(h_L);
            // I and J refer to the 1h part of the operator
            for (int h_Ia = 0; h_Ia < nirrep; ++h_Ia) {
                int h_Mb = h_Ia ^ symmetry;
                double** C_I_p = C_right.C(h_Ia)->pointer();
                for (int h_Ja = 0; h_Ja < nirrep; ++h_Ja) {
                    int h_Nb = h_Ja ^ symmetry;
                    double** C_J_p = C_left.C(h_Ja)->pointer();
                    for (size_t K = 0; K < maxK; ++K) {
                        std::vector<H1StringSubstitution>& Ilist =
                            lists->get_alfa_1h_list(h_K, K, h_Ia);
                        std::vector<H1StringSubstitution>& Jlist =
                            lists->get_alfa_1h_list(h_K, K, h_Ja);
                        for (size_t L = 0; L < maxL; ++L) {
                            std::vector<H2StringSubstitution>& Mlist =
                                lists->get_beta_2h_list(h_L, L, h_Mb);
                            std::vector<H2StringSubstitution>& Nlist =
                                lists->get_beta_2h_list(h_L, L, h_Nb);
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
                                            rdm[six_index(p, q, r, s, t, a, ncmo)] +=
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
    return g3;
}

void FCIVector::test_rdms(FCIVector& Cl, FCIVector& Cr, int max_rdm_level, RDMsType type,
                          std::shared_ptr<RDMs> rdms) {

    String Ia, Ib;
    // Generate the strings 1111100000
    //                      { k }{n-k}

    size_t ncmo = Cl.ncmo_;

    std::vector<Determinant> left_dets, right_dets;
    std::map<Determinant, size_t> left_dets_map, right_dets_map;
    std::vector<double> left_C, right_C;

    // left
    {
        int symmetry_left = Cl.symmetry_;
        auto alfa_address_left = Cl.alfa_address_;
        auto beta_address_left = Cl.beta_address_;
        size_t left_na = Cl.lists_->na();
        size_t left_nb = Cl.lists_->nb();
        size_t num_det = 0;
        for (size_t i = 0; i < ncmo - left_na; ++i)
            Ia[i] = false; // 0
        for (size_t i = ncmo - left_na; i < ncmo; ++i)
            Ia[i] = true; // 1

        for (size_t i = 0; i < ncmo - left_nb; ++i)
            Ib[i] = false; // 0
        for (size_t i = ncmo - left_nb; i < ncmo; ++i)
            Ib[i] = true; // 1
        do {
            do {
                if ((alfa_address_left->sym(Ia) ^ beta_address_left->sym(Ib)) == symmetry_left) {
                    Determinant d(Ia, Ib);
                    left_dets.push_back(d);
                    double c = Cl.C(alfa_address_left->sym(Ia))
                                   ->get(alfa_address_left->add(Ia), beta_address_left->add(Ib));
                    left_C.push_back(c);
                    left_dets_map[d] = num_det;
                    num_det++;
                }
            } while (std::next_permutation(Ib.begin(), Ib.begin() + ncmo));
        } while (std::next_permutation(Ia.begin(), Ia.begin() + ncmo));
    }

    // right
    {
        int symmetry_right = Cr.symmetry_;
        auto alfa_address_right = Cr.alfa_address_;
        auto beta_address_right = Cr.beta_address_;
        size_t right_na = Cr.lists_->na();
        size_t right_nb = Cr.lists_->nb();
        size_t num_det = 0;
        for (size_t i = 0; i < ncmo - right_na; ++i)
            Ia[i] = false; // 0
        for (size_t i = ncmo - right_na; i < ncmo; ++i)
            Ia[i] = true; // 1

        for (size_t i = 0; i < ncmo - right_nb; ++i)
            Ib[i] = false; // 0
        for (size_t i = ncmo - right_nb; i < ncmo; ++i)
            Ib[i] = true; // 1
        do {
            do {
                if ((alfa_address_right->sym(Ia) ^ beta_address_right->sym(Ib)) == symmetry_right) {
                    Determinant d(Ia, Ib);
                    right_dets.push_back(d);
                    double c = Cr.C(alfa_address_right->sym(Ia))
                                   ->get(alfa_address_right->add(Ia), beta_address_right->add(Ib));
                    right_C.push_back(c);
                    right_dets_map[d] = num_det;
                    num_det++;
                }
            } while (std::next_permutation(Ib.begin(), Ib.begin() + ncmo));
        } while (std::next_permutation(Ia.begin(), Ia.begin() + ncmo));
    }

    Determinant I; // <- xsize (no_);

    psi::outfile->Printf("\n\n==> RDMs Test <==\n");

    if (max_rdm_level >= 1) {
        // compute the reference 1-RDM_A
        auto g1a_ref = ambit::Tensor::build(ambit::CoreTensor, "1RDM_A", {ncmo, ncmo});
        for (size_t p = 0; p < ncmo; p++) {
            for (size_t q = 0; q < ncmo; ++q) {
                double rdm = 0.0;
                for (size_t i = 0; i < right_dets.size(); ++i) {
                    I = right_dets[i];
                    double sign = 1.0;
                    sign *= I.destroy_alfa_bit(q);
                    sign *= I.create_alfa_bit(p);
                    if (sign != 0) {
                        if (left_dets_map.count(I) != 0) {
                            rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                        }
                    }
                }
                g1a_ref.at({p, q}) = rdm;
            }
        }

        // compute the reference 1-RDM_B
        auto g1b_ref = ambit::Tensor::build(ambit::CoreTensor, "1RDM_B", {ncmo, ncmo});
        for (size_t p = 0; p < ncmo; p++) {
            for (size_t q = 0; q < ncmo; ++q) {
                double rdm = 0.0;
                for (size_t i = 0; i < right_dets.size(); ++i) {
                    I = right_dets[i];
                    double sign = 1.0;
                    sign *= I.destroy_beta_bit(q);
                    sign *= I.create_beta_bit(p);
                    if (sign != 0) {
                        if (left_dets_map.count(I) != 0) {
                            rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                        }
                    }
                }
                g1b_ref.at({p, q}) = rdm;
            }
        }

        if (type == RDMsType::spin_dependent) {
            // test the 1-RDM
            auto g1a = rdms->g1a();
            g1a_ref("pq") -= g1a("pq");
            auto error_1rdm_a = g1a_ref.norm();
            psi::outfile->Printf("\n    AA 1-RDM Error :   %+e", error_1rdm_a);
            psi::Process::environment.globals["AA 1-RDM ERROR"] = error_1rdm_a;

            auto g1b = rdms->g1b();
            g1b_ref("pq") -= g1b("pq");
            auto error_1rdm_b = g1b_ref.norm();
            psi::outfile->Printf("\n    BB 1-RDM Error :   %+e", error_1rdm_b);
            psi::Process::environment.globals["BB 1-RDM ERROR"] = error_1rdm_b;
        } else if (type == RDMsType::spin_free) {
            // test the 1-RDM
            auto G1 = rdms->SF_G1();
            auto G1_ref = ambit::Tensor::build(ambit::CoreTensor, "1RDM", {ncmo, ncmo});
            G1_ref("pq") += g1a_ref("pq");
            G1_ref("pq") += g1b_ref("pq");
            G1_ref("pq") -= G1("pq");

            auto error_1rdm_SF = G1_ref.norm();
            psi::outfile->Printf("\n    SF 1-RDM Error :   %+e", error_1rdm_SF);
            psi::Process::environment.globals["SF 1-RDM ERROR"] = error_1rdm_SF;
        }
    }

    if (max_rdm_level >= 2) {
        auto g2aa = rdms->g2aa();
        double error_2rdm_aa = 0.0;
        for (size_t p = 0; p < ncmo; ++p) {
            for (size_t q = 0; q < ncmo; ++q) {
                for (size_t r = 0; r < ncmo; ++r) {
                    for (size_t s = 0; s < ncmo; ++s) {
                        double rdm = 0.0;
                        for (size_t i = 0; i < right_dets.size(); ++i) {
                            I = right_dets[i];
                            double sign = 1.0;
                            sign *= I.destroy_alfa_bit(r);
                            sign *= I.destroy_alfa_bit(s);
                            sign *= I.create_alfa_bit(q);
                            sign *= I.create_alfa_bit(p);
                            if (sign != 0) {
                                if (left_dets_map.count(I) != 0) {
                                    rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                                }
                            }
                        }
                        if (std::fabs(rdm) > 1.0e-12) {
                            error_2rdm_aa += std::fabs(rdm - g2aa.at({p, q, r, s}));
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["AAAA 2-RDM ERROR"] = error_2rdm_aa;
        psi::outfile->Printf("\n    AAAA 2-RDM Error :   %+e", error_2rdm_aa);

        auto g2bb = rdms->g2bb();
        double error_2rdm_bb = 0.0;
        for (size_t p = 0; p < ncmo; ++p) {
            for (size_t q = 0; q < ncmo; ++q) {
                for (size_t r = 0; r < ncmo; ++r) {
                    for (size_t s = 0; s < ncmo; ++s) {
                        double rdm = 0.0;
                        for (size_t i = 0; i < right_dets.size(); ++i) {
                            I = right_dets[i];
                            double sign = 1.0;
                            sign *= I.destroy_beta_bit(r);
                            sign *= I.destroy_beta_bit(s);
                            sign *= I.create_beta_bit(q);
                            sign *= I.create_beta_bit(p);
                            if (sign != 0) {
                                if (left_dets_map.count(I) != 0) {
                                    rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                                }
                            }
                        }
                        if (std::fabs(rdm) > 1.0e-12) {
                            error_2rdm_bb += std::fabs(rdm - g2bb.at({p, q, r, s}));
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["BBBB 2-RDM ERROR"] = error_2rdm_bb;
        psi::outfile->Printf("\n    BBBB 2-RDM Error :   %+e", error_2rdm_bb);

        auto g2ab = rdms->g2ab();
        double error_2rdm_ab = 0.0;
        for (size_t p = 0; p < ncmo; ++p) {
            for (size_t q = 0; q < ncmo; ++q) {
                for (size_t r = 0; r < ncmo; ++r) {
                    for (size_t s = 0; s < ncmo; ++s) {
                        double rdm = 0.0;
                        for (size_t i = 0; i < right_dets.size(); ++i) {
                            I = right_dets[i];
                            double sign = 1.0;
                            sign *= I.destroy_alfa_bit(r);
                            sign *= I.destroy_beta_bit(s);
                            sign *= I.create_beta_bit(q);
                            sign *= I.create_alfa_bit(p);
                            if (sign != 0) {
                                if (left_dets_map.count(I) != 0) {
                                    rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                                }
                            }
                        }
                        if (std::fabs(rdm) > 1.0e-12) {
                            error_2rdm_ab += std::fabs(rdm - g2ab.at({p, q, r, s}));
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["ABAB 2-RDM ERROR"] = error_2rdm_ab;
        psi::outfile->Printf("\n    ABAB 2-RDM Error :   %+e", error_2rdm_ab);
    }

    if (max_rdm_level >= 3) {
        auto g3aaa = rdms->g3aaa();
        double error_3rdm_aaa = 0.0;
        //        for (size_t p = 0; p < no_; ++p){
        for (size_t p = 0; p < 1; ++p) {
            for (size_t q = p + 1; q < ncmo; ++q) {
                for (size_t r = q + 1; r < ncmo; ++r) {
                    for (size_t s = 0; s < ncmo; ++s) {
                        for (size_t t = s + 1; t < ncmo; ++t) {
                            for (size_t a = t + 1; a < ncmo; ++a) {
                                double rdm = 0.0;
                                for (size_t i = 0; i < right_dets.size(); ++i) {
                                    I = right_dets[i];
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_alfa_bit(t);
                                    sign *= I.destroy_alfa_bit(a);
                                    sign *= I.create_alfa_bit(r);
                                    sign *= I.create_alfa_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (left_dets_map.count(I) != 0) {
                                            rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12) {
                                    double rdm_comp = g3aaa.at({p, q, r, s, t, a});
                                    error_3rdm_aaa += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["AAAAAA 3-RDM ERROR"] = error_3rdm_aaa;
        psi::outfile->Printf("\n    AAAAAA 3-RDM Error : %+e", error_3rdm_aaa);

        auto g3aab = rdms->g3aab();
        double error_3rdm_aab = 0.0;
        for (size_t p = 0; p < ncmo; ++p) {
            for (size_t q = p + 1; q < ncmo; ++q) {
                for (size_t r = 0; r < ncmo; ++r) {
                    for (size_t s = 0; s < ncmo; ++s) {
                        for (size_t t = s + 1; t < ncmo; ++t) {
                            for (size_t a = 0; a < ncmo; ++a) {
                                double rdm = 0.0;
                                for (size_t i = 0; i < right_dets.size(); ++i) {
                                    I = right_dets[i];
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_alfa_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_alfa_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (left_dets_map.count(I) != 0) {
                                            rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12) {
                                    double rdm_comp = g3aab.at({p, q, r, s, t, a});
                                    error_3rdm_aab += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["AABAAB 3-RDM ERROR"] = error_3rdm_aab;
        psi::outfile->Printf("\n    AABAAB 3-RDM Error : %+e", error_3rdm_aab);

        auto g3abb = rdms->g3abb();
        double error_3rdm_abb = 0.0;
        for (size_t p = 0; p < ncmo; ++p) {
            for (size_t q = p + 1; q < ncmo; ++q) {
                for (size_t r = 0; r < ncmo; ++r) {
                    for (size_t s = 0; s < ncmo; ++s) {
                        for (size_t t = s + 1; t < ncmo; ++t) {
                            for (size_t a = 0; a < ncmo; ++a) {
                                double rdm = 0.0;
                                for (size_t i = 0; i < right_dets.size(); ++i) {
                                    I = right_dets[i];
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_beta_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_beta_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (left_dets_map.count(I) != 0) {
                                            rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12) {
                                    double rdm_comp = g3abb.at({p, q, r, s, t, a});
                                    error_3rdm_abb += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["ABBABB 3-RDM ERROR"] = error_3rdm_abb;
        psi::outfile->Printf("\n    ABBABB 3-RDM Error : %+e", error_3rdm_abb);

        auto g3bbb = rdms->g3bbb();
        double error_3rdm_bbb = 0.0;
        for (size_t p = 0; p < 1; ++p) {
            //            for (size_t p = 0; p < no_; ++p){
            for (size_t q = p + 1; q < ncmo; ++q) {
                for (size_t r = q + 1; r < ncmo; ++r) {
                    for (size_t s = 0; s < ncmo; ++s) {
                        for (size_t t = s + 1; t < ncmo; ++t) {
                            for (size_t a = t + 1; a < ncmo; ++a) {
                                double rdm = 0.0;
                                for (size_t i = 0; i < right_dets.size(); ++i) {
                                    I = right_dets[i];
                                    double sign = 1.0;
                                    sign *= I.destroy_beta_bit(s);
                                    sign *= I.destroy_beta_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_beta_bit(q);
                                    sign *= I.create_beta_bit(p);
                                    if (sign != 0) {
                                        if (left_dets_map.count(I) != 0) {
                                            rdm += sign * left_C[left_dets_map[I]] * right_C[i];
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12) {
                                    double rdm_comp = g3bbb.at({p, q, r, s, t, a});
                                    error_3rdm_bbb += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        psi::Process::environment.globals["BBBBBB 3-RDM ERROR"] = error_3rdm_bbb;
        psi::outfile->Printf("\n    BBBBBB 3-RDM Error : %+e", error_3rdm_bbb);
    }
}

} // namespace forte
