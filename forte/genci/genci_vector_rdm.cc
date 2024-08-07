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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/matrix.h"

#include "genci_string_lists.h"
#include "genci_string_address.h"

#include "genci_vector.h"

namespace forte {

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
std::shared_ptr<RDMs> GenCIVector::compute_rdms(GenCIVector& C_left, GenCIVector& C_right,
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
            local_timer t_aaa;
            g3aaa = compute_3rdm_aaa_same_irrep(C_left, C_right, true);
            psi::outfile->Printf("\n    Timing for 3-RDM (aaa): %.3f s", t_aaa.get());
        } else {
            g3aaa =
                ambit::Tensor::build(ambit::CoreTensor, "g3aaa", {nmo, nmo, nmo, nmo, nmo, nmo});
            g3aaa.zero();
        }
        if (nb >= 3) {
            local_timer t_bbb;
            g3bbb = compute_3rdm_aaa_same_irrep(C_left, C_right, false);
            psi::outfile->Printf("\n    Timing for 3-RDM (bbb): %.3f s", t_bbb.get());
        } else {
            g3bbb =
                ambit::Tensor::build(ambit::CoreTensor, "g3bbb", {nmo, nmo, nmo, nmo, nmo, nmo});
            g3bbb.zero();
        }

        if ((na >= 2) and (nb >= 1)) {
            local_timer t_aab;
            g3aab = compute_3rdm_aab_same_irrep(C_left, C_right);
            psi::outfile->Printf("\n    Timing for 3-RDM (aab): %.3f s", t_aab.get());
        } else {
            g3aab =
                ambit::Tensor::build(ambit::CoreTensor, "g3aab", {nmo, nmo, nmo, nmo, nmo, nmo});
            g3aab.zero();
        }

        if ((na >= 1) and (nb >= 2)) {
            local_timer t_abb;
            g3abb = compute_3rdm_abb_same_irrep(C_left, C_right);
            psi::outfile->Printf("\n    Timing for 3-RDM (abb): %.3f s", t_abb.get());
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
        throw std::runtime_error(
            "RDMs of order 4 or higher are not implemented in GenCISolver (and "
            "more generally in Forte).");
    }
    return std::make_shared<RDMsSpinDependent>();
}

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
ambit::Tensor GenCIVector::compute_1rdm_same_irrep(GenCIVector& C_left, GenCIVector& C_right,
                                                   bool alfa) {
    size_t ncmo = C_left.ncmo_;
    const auto& alfa_address = C_left.alfa_address_;
    const auto& beta_address = C_left.beta_address_;
    const auto& lists = C_left.lists_;

    auto rdm = ambit::Tensor::build(ambit::CoreTensor, alfa ? "1RDM_A" : "1RDM_B", {ncmo, ncmo});

    auto na = alfa_address->nones();
    auto nb = beta_address->nones();
    if ((alfa and (na < 1)) or ((!alfa) and (nb < 1)))
        return rdm;

    auto& rdm_data = rdm.data();

    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists->determinant_classes()) {
        if (lists->detpblk(nI) == 0)
            continue;

        auto Cr =
            C_right.gather_C_block(CR, alfa, alfa_address, beta_address, class_Ia, class_Ib, false);

        for (const auto& [nJ, class_Ja, class_Jb] : lists->determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                continue;
            if (lists->detpblk(nJ) == 0)
                continue;

            auto Cl = C_left.gather_C_block(CL, alfa, alfa_address, beta_address, class_Ja,
                                            class_Jb, false);

            size_t maxL = alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

            const auto& pq_vo_list = alfa ? lists->get_alfa_vo_list(class_Ia, class_Ja)
                                          : lists->get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                double rdm_element = 0.0;
                for (const auto& [sign, I, J] : vo_list) {
                    rdm_element += sign * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
                }
                rdm_data[p * ncmo + q] += rdm_element;
            }
        }
    }

    return rdm;
}

/**
 * Compute the aa/bb two-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
ambit::Tensor GenCIVector::compute_2rdm_aa_same_irrep(GenCIVector& C_left, GenCIVector& C_right,
                                                      bool alfa) {
    size_t ncmo = C_left.ncmo_;
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

    for (const auto& [nI, class_Ia, class_Ib] : lists->determinant_classes()) {
        if (lists->detpblk(nI) == 0)
            continue;

        const auto Cr =
            C_right.gather_C_block(CR, alfa, alfa_address, beta_address, class_Ia, class_Ib, false);

        for (const auto& [nJ, class_Ja, class_Jb] : lists->determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                continue;
            if (lists->detpblk(nJ) == 0)
                continue;

            const auto Cl = C_left.gather_C_block(CL, alfa, alfa_address, beta_address, class_Ja,
                                                  class_Jb, false);

            // get the size of the string of spin opposite to the one we are acting on
            size_t maxL = alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

            if ((class_Ia == class_Ja) and (class_Ib == class_Jb)) {
                // OO terms
                // Loop over (p>q) == (p>q)
                const auto& pq_oo_list =
                    alfa ? lists->get_alfa_oo_list(class_Ia) : lists->get_beta_oo_list(class_Ib);
                for (const auto& [pq, oo_list] : pq_oo_list) {
                    const auto& [p, q] = pq;
                    double rdm_element = 0.0;
                    for (const auto& I : oo_list) {
                        rdm_element += psi::C_DDOT(maxL, Cl[I], 1, Cr[I], 1);
                    }
                    rdm_data[tei_index(p, q, p, q, ncmo)] += rdm_element;
                    rdm_data[tei_index(p, q, q, p, ncmo)] -= rdm_element;
                    rdm_data[tei_index(q, p, p, q, ncmo)] -= rdm_element;
                    rdm_data[tei_index(q, p, q, p, ncmo)] += rdm_element;
                }
            }

            // VVOO terms
            const auto& pqrs_vvoo_list = alfa ? lists->get_alfa_vvoo_list(class_Ia, class_Ja)
                                              : lists->get_beta_vvoo_list(class_Ib, class_Jb);
            for (const auto& [pqrs, vvoo_list] : pqrs_vvoo_list) {
                const auto& [p, q, r, s] = pqrs;

                double rdm_element = 0.0;
                for (const auto& [sign, I, J] : vvoo_list) {
                    rdm_element += sign * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
                }
                rdm_data[tei_index(p, q, r, s, ncmo)] += rdm_element;
                rdm_data[tei_index(q, p, r, s, ncmo)] -= rdm_element;
                rdm_data[tei_index(p, q, s, r, ncmo)] -= rdm_element;
                rdm_data[tei_index(q, p, s, r, ncmo)] += rdm_element;
            }
        }
    }
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

ambit::Tensor GenCIVector::compute_2rdm_ab_same_irrep(GenCIVector& C_left, GenCIVector& C_right) {
    size_t ncmo = C_left.ncmo_;
    const auto& alfa_address = C_left.alfa_address_;
    const auto& beta_address = C_left.beta_address_;
    const auto& lists = C_left.lists_;

    auto rdm = ambit::Tensor::build(ambit::CoreTensor, "2RDM_AB", {ncmo, ncmo, ncmo, ncmo});

    auto na = alfa_address->nones();
    auto nb = beta_address->nones();
    if ((na < 1) or (nb < 1))
        return rdm;

    auto& rdm_data = rdm.data();

    const auto& mo_sym = lists->string_class()->mo_sym();
    // Loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists->determinant_classes()) {
        if (lists->detpblk(nI) == 0)
            continue;

        auto h_Ib = lists->string_class()->beta_string_classes()[class_Ib].second;
        const auto Cr = C_right.C_[nI]->pointer();

        for (const auto& [nJ, class_Ja, class_Jb] : lists->determinant_classes()) {
            if (lists->detpblk(nJ) == 0)
                continue;

            auto h_Jb = lists->string_class()->beta_string_classes()[class_Jb].second;
            const auto Cl = C_left.C_[nJ]->pointer();

            const auto& pq_vo_alfa = lists->get_alfa_vo_list(class_Ia, class_Ja);
            const auto& rs_vo_beta = lists->get_beta_vo_list(class_Ib, class_Jb);

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

                for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
                    const auto& [p, q] = pq;
                    const auto pq_sym = mo_sym[p] ^ mo_sym[q];
                    // ensure that the product pqrs is totally symmetric
                    if (pq_sym != rs_sym)
                        continue;

                    double rdm_element = 0.0;
                    for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
                        for (const auto& [sign_b, Ib, Jb] : vo_beta_list) {
                            rdm_element += Cl[Ja][Jb] * Cr[Ia][Ib] * sign_a * sign_b;
                        }
                    }
                    rdm_data[tei_index(p, r, q, s, ncmo)] += rdm_element;
                } // End loop over p,q
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

ambit::Tensor GenCIVector::compute_3rdm_aaa_same_irrep(GenCIVector& C_left, GenCIVector& C_right,
                                                       bool alfa) {
    size_t ncmo = C_left.ncmo_;
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

    int num_3h_classes =
        alfa ? lists->alfa_address_3h()->nclasses() : lists->beta_address_3h()->nclasses();

    for (int class_K = 0; class_K < num_3h_classes; ++class_K) {
        size_t maxK = alfa ? lists->alfa_address_3h()->strpcls(class_K)
                           : lists->beta_address_3h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists->determinant_classes()) {
            if (lists->detpblk(nI) == 0)
                continue;

            auto Cr = C_right.gather_C_block(CR, alfa, alfa_address, beta_address, class_Ia,
                                             class_Ib, false);

            for (const auto& [nJ, class_Ja, class_Jb] : lists->determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                    continue;
                if (lists->detpblk(nJ) == 0)
                    continue;

                // Get a pointer to the correct block of matrix C
                auto Cl = C_left.gather_C_block(CL, alfa, alfa_address, beta_address, class_Ja,
                                                class_Jb, false);

                size_t maxL =
                    alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);
                if (maxL > 0) {
                    for (size_t K = 0; K < maxK; ++K) {
                        std::vector<H3StringSubstitution>& Krlist =
                            alfa ? lists->get_alfa_3h_list(class_K, K, class_Ia)
                                 : lists->get_beta_3h_list(class_K, K, class_Ib);
                        std::vector<H3StringSubstitution>& Kllist =
                            alfa ? lists->get_alfa_3h_list(class_K, K, class_Ja)
                                 : lists->get_beta_3h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, q, r, I] : Krlist) {
                            for (const auto& [sign_L, s, t, u, J] : Kllist) {
                                rdm_data[six_index(p, q, r, s, t, u, ncmo)] +=
                                    sign_K * sign_L * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
                            }
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

ambit::Tensor GenCIVector::compute_3rdm_aab_same_irrep(GenCIVector& C_left, GenCIVector& C_right) {
    // this threshold is used to avoid computing very small contributions to the 3-RDM
    // here we screen for the absolute value of the elements of the C vector, and since the vector
    // is normalized, we can guarantee that the neglected contribution to the 3-RDM is smaller or
    // equal than this value
    // const double c_threshold = 1.0e-14;

    size_t ncmo = C_left.ncmo_;
    const auto& lists = C_left.lists_;

    auto rdm =
        ambit::Tensor::build(ambit::CoreTensor, "3RDM_AAB", {ncmo, ncmo, ncmo, ncmo, ncmo, ncmo});
    rdm.zero();
    auto& rdm_data = rdm.data();

    int num_2h_class_Ka = lists->alfa_address_2h()->nclasses();
    int num_1h_class_Kb = lists->beta_address_1h()->nclasses();

    for (int class_Ka = 0; class_Ka < num_2h_class_Ka; ++class_Ka) {
        size_t maxKa = lists->alfa_address_2h()->strpcls(class_Ka);

        for (int class_Kb = 0; class_Kb < num_1h_class_Kb; ++class_Kb) {
            size_t maxKb = lists->beta_address_1h()->strpcls(class_Kb);

            // loop over blocks of matrix C
            for (const auto& [nI, class_Ia, class_Ib] : lists->determinant_classes()) {
                if (lists->detpblk(nI) == 0)
                    continue;

                const auto Cr = C_right.C_[nI]->pointer();

                for (const auto& [nJ, class_Ja, class_Jb] : lists->determinant_classes()) {
                    if (lists->detpblk(nJ) == 0)
                        continue;

                    // Get a pointer to the correct block of matrix C
                    const auto Cl = C_left.C_[nJ]->pointer();

                    for (size_t Ka = 0; Ka < maxKa; ++Ka) {
                        auto& Ka_right_list = lists->get_alfa_2h_list(class_Ka, Ka, class_Ia);
                        auto& Ka_left_list = lists->get_alfa_2h_list(class_Ka, Ka, class_Ja);
                        for (size_t Kb = 0; Kb < maxKb; ++Kb) {
                            auto& Kb_right_list = lists->get_beta_1h_list(class_Kb, Kb, class_Ib);
                            auto& Kb_left_list = lists->get_beta_1h_list(class_Kb, Kb, class_Jb);
                            for (const auto& [sign_uv, u, v, Ja] : Ka_left_list) {
                                for (const auto& [sign_w, w, Jb] : Kb_left_list) {
                                    const double ClJ = sign_uv * sign_w * Cl[Ja][Jb];
                                    // if (std::fabs(ClJ) < c_threshold)
                                    //     continue;
                                    for (const auto& [sign_xy, x, y, Ia] : Ka_right_list) {
                                        const auto CrIa = Cr[Ia];
                                        for (const auto& [sign_z, z, Ib] : Kb_right_list) {
                                            rdm_data[six_index(u, v, w, x, y, z, ncmo)] +=
                                                sign_xy * sign_z * ClJ * CrIa[Ib];
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
    for (size_t u = 0; u < ncmo; ++u) {
        for (size_t v = 0; v < u; ++v) {
            for (size_t w = 0; w < ncmo; ++w) {
                for (size_t x = 0; x < ncmo; ++x) {
                    for (size_t y = 0; y < x; ++y) {
                        for (size_t z = 0; z < ncmo; ++z) {
                            const double rdm_element = rdm_data[six_index(u, v, w, x, y, z, ncmo)];
                            rdm_data[six_index(u, v, w, y, x, z, ncmo)] = -rdm_element;
                            rdm_data[six_index(v, u, w, x, y, z, ncmo)] = -rdm_element;
                            rdm_data[six_index(v, u, w, y, x, z, ncmo)] = +rdm_element;
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

ambit::Tensor GenCIVector::compute_3rdm_abb_same_irrep(GenCIVector& C_left, GenCIVector& C_right) {
    // this threshold is used to avoid computing very small contributions to the 3-RDM
    // here we screen for the absolute value of the elements of the C vector, and since the vector
    // is normalized, we can guarantee that the neglected contribution to the 3-RDM is smaller or
    // equal than this value
    // const double c_threshold = 1.0e-14;

    size_t ncmo = C_left.ncmo_;
    const auto& lists = C_left.lists_;

    auto rdm =
        ambit::Tensor::build(ambit::CoreTensor, "3RDM_ABB", {ncmo, ncmo, ncmo, ncmo, ncmo, ncmo});
    rdm.zero();
    auto& rdm_data = rdm.data();

    int num_1h_class_Ka = lists->alfa_address_1h()->nclasses();
    int num_2h_class_Kb = lists->beta_address_2h()->nclasses();

    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        size_t maxKa = lists->alfa_address_1h()->strpcls(class_Ka);

        for (int class_Kb = 0; class_Kb < num_2h_class_Kb; ++class_Kb) {
            size_t maxKb = lists->beta_address_2h()->strpcls(class_Kb);

            // loop over blocks of matrix C
            for (const auto& [nI, class_Ia, class_Ib] : lists->determinant_classes()) {
                if (lists->detpblk(nI) == 0)
                    continue;

                const auto Cr = C_right.C_[nI]->pointer();

                for (const auto& [nJ, class_Ja, class_Jb] : lists->determinant_classes()) {
                    if (lists->detpblk(nJ) == 0)
                        continue;

                    // Get a pointer to the correct block of matrix C
                    const auto Cl = C_left.C_[nJ]->pointer();

                    for (size_t Ka = 0; Ka < maxKa; ++Ka) {
                        auto& Ka_right_list = lists->get_alfa_1h_list(class_Ka, Ka, class_Ia);
                        auto& Ka_left_list = lists->get_alfa_1h_list(class_Ka, Ka, class_Ja);
                        for (size_t Kb = 0; Kb < maxKb; ++Kb) {
                            auto& Kb_right_list = lists->get_beta_2h_list(class_Kb, Kb, class_Ib);
                            auto& Kb_left_list = lists->get_beta_2h_list(class_Kb, Kb, class_Jb);
                            for (const auto& [sign_u, u, Ja] : Ka_left_list) {
                                for (const auto& [sign_vw, v, w, Jb] : Kb_left_list) {
                                    const double ClJ = sign_u * sign_vw * Cl[Ja][Jb];
                                    // if (std::fabs(ClJ) < c_threshold)
                                    //     continue;
                                    for (const auto& [sign_x, x, Ia] : Ka_right_list) {
                                        const auto CrIa = Cr[Ia];
                                        for (const auto& [sign_yz, y, z, Ib] : Kb_right_list) {
                                            rdm_data[six_index(u, v, w, x, y, z, ncmo)] +=
                                                sign_x * sign_yz * ClJ * CrIa[Ib];
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
    for (size_t u = 0; u < ncmo; ++u) {
        for (size_t v = 0; v < ncmo; ++v) {
            for (size_t w = 0; w < v; ++w) {
                for (size_t x = 0; x < ncmo; ++x) {
                    for (size_t y = 0; y < ncmo; ++y) {
                        for (size_t z = 0; z < y; ++z) {
                            const double rdm_element = rdm_data[six_index(u, v, w, x, y, z, ncmo)];
                            rdm_data[six_index(u, v, w, x, z, y, ncmo)] = -rdm_element;
                            rdm_data[six_index(u, w, v, x, y, z, ncmo)] = -rdm_element;
                            rdm_data[six_index(u, w, v, x, z, y, ncmo)] = +rdm_element;
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

void GenCIVector::test_rdms(GenCIVector& Cl, GenCIVector& Cr, int max_rdm_level, RDMsType type,
                            std::shared_ptr<RDMs> rdms) {
    size_t ncmo = Cl.ncmo_;
    auto state_vector_l = Cl.as_state_vector();
    auto state_vector_r = Cr.as_state_vector();

    Determinant J;
    psi::outfile->Printf("\n\n==> RDMs Test (max level = %d)<==\n", max_rdm_level);

    if (max_rdm_level >= 1) {
        // compute the reference 1-RDM_A
        auto g1a_ref = ambit::Tensor::build(ambit::CoreTensor, "1RDM_A", {ncmo, ncmo});
        for (size_t p = 0; p < ncmo; p++) {
            for (size_t q = 0; q < ncmo; ++q) {
                double rdm = 0.0;
                for (const auto& [I, c_I] : state_vector_r) {
                    J = I;
                    double sign = 1.0;
                    sign *= J.destroy_alfa_bit(q);
                    sign *= J.create_alfa_bit(p);
                    if (sign != 0) {
                        if (state_vector_l.count(J) != 0) {
                            rdm += sign * to_double(state_vector_l[J] * c_I);
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
                for (const auto& [I, c_I] : state_vector_r) {
                    J = I;
                    double sign = 1.0;
                    sign *= J.destroy_beta_bit(q);
                    sign *= J.create_beta_bit(p);
                    if (sign != 0) {
                        if (state_vector_l.count(J) != 0) {
                            rdm += sign * to_double(state_vector_l[J] * c_I);
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
                        for (const auto& [I, c_I] : state_vector_r) {
                            J = I;
                            double sign = 1.0;
                            sign *= J.destroy_alfa_bit(r);
                            sign *= J.destroy_alfa_bit(s);
                            sign *= J.create_alfa_bit(q);
                            sign *= J.create_alfa_bit(p);
                            if (sign != 0) {
                                if (state_vector_l.count(J) != 0) {
                                    rdm += sign * to_double(state_vector_l[J] * c_I);
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
                        for (const auto& [I, c_I] : state_vector_r) {
                            J = I;
                            double sign = 1.0;
                            sign *= J.destroy_beta_bit(r);
                            sign *= J.destroy_beta_bit(s);
                            sign *= J.create_beta_bit(q);
                            sign *= J.create_beta_bit(p);
                            if (sign != 0) {
                                if (state_vector_l.count(J) != 0) {
                                    rdm += sign * to_double(state_vector_l[J] * c_I);
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
                        for (const auto& [I, c_I] : state_vector_r) {
                            J = I;
                            double sign = 1.0;
                            sign *= J.destroy_alfa_bit(r);
                            sign *= J.destroy_beta_bit(s);
                            sign *= J.create_beta_bit(q);
                            sign *= J.create_alfa_bit(p);
                            if (sign != 0) {
                                if (state_vector_l.count(J) != 0) {
                                    rdm += sign * to_double(state_vector_l[J] * c_I);
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
                                for (const auto& [I, c_I] : state_vector_r) {
                                    J = I;
                                    double sign = 1.0;
                                    sign *= J.destroy_alfa_bit(s);
                                    sign *= J.destroy_alfa_bit(t);
                                    sign *= J.destroy_alfa_bit(a);
                                    sign *= J.create_alfa_bit(r);
                                    sign *= J.create_alfa_bit(q);
                                    sign *= J.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (state_vector_l.count(J) != 0) {
                                            rdm += sign * to_double(state_vector_l[J] * c_I);
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
                                for (const auto& [I, c_I] : state_vector_r) {
                                    J = I;
                                    double sign = 1.0;
                                    sign *= J.destroy_alfa_bit(s);
                                    sign *= J.destroy_alfa_bit(t);
                                    sign *= J.destroy_beta_bit(a);
                                    sign *= J.create_beta_bit(r);
                                    sign *= J.create_alfa_bit(q);
                                    sign *= J.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (state_vector_l.count(J) != 0) {
                                            rdm += sign * to_double(state_vector_l[J] * c_I);
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
                                for (const auto& [I, c_I] : state_vector_r) {
                                    J = I;
                                    double sign = 1.0;
                                    sign *= J.destroy_alfa_bit(s);
                                    sign *= J.destroy_beta_bit(t);
                                    sign *= J.destroy_beta_bit(a);
                                    sign *= J.create_beta_bit(r);
                                    sign *= J.create_beta_bit(q);
                                    sign *= J.create_alfa_bit(p);
                                    if (sign != 0) {
                                        if (state_vector_l.count(J) != 0) {
                                            rdm += sign * to_double(state_vector_l[J] * c_I);
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
                                for (const auto& [I, c_I] : state_vector_r) {
                                    J = I;
                                    double sign = 1.0;
                                    sign *= J.destroy_beta_bit(s);
                                    sign *= J.destroy_beta_bit(t);
                                    sign *= J.destroy_beta_bit(a);
                                    sign *= J.create_beta_bit(r);
                                    sign *= J.create_beta_bit(q);
                                    sign *= J.create_beta_bit(p);
                                    if (sign != 0) {
                                        if (state_vector_l.count(J) != 0) {
                                            rdm += sign * to_double(state_vector_l[J] * c_I);
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
