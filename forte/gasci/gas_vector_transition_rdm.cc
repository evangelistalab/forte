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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/matrix.h"

#include "gas_string_lists.h"
#include "gas_string_address.h"

#include "gas_vector.h"

namespace forte {

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
std::shared_ptr<RDMs> compute_transition_rdms(GASVector& C_left, GASVector& C_right,
                                              int max_rdm_level, RDMsType type) {
    size_t na_left = C_left.alfa_address()->nones();
    size_t nb_left = C_left.beta_address()->nones();
    size_t na_right = C_right.alfa_address()->nones();
    size_t nb_right = C_right.beta_address()->nones();

    if (C_left.ncmo() != C_right.ncmo()) {
        throw std::runtime_error(
            "FCI transition RDMs: The number of MOs must be the same in the two wave functions.");
    }

    if (na_left != na_right or nb_left != nb_right) {
        throw std::runtime_error(
            "FCI transition RDMs: The number of alfa and beta electrons must be the same in the "
            "two wave functions.");
    }

    ambit::Tensor g1a, g1b;

    if (max_rdm_level >= 1) {
        local_timer t;
        g1a = compute_1rdm_different_irrep(C_left, C_right, true);
        g1b = compute_1rdm_different_irrep(C_left, C_right, false);
    }

    if (type == RDMsType::spin_dependent) {
        if (max_rdm_level == 1) {
            auto rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b);
            GASVector::test_rdms(C_left, C_right, 1, type, rdms);
            return rdms;
        }
    } else {
        g1a("pq") += g1b("pq");
        if (max_rdm_level == 1) {
            auto rdms = std::make_shared<RDMsSpinFree>(g1a);
            GASVector::test_rdms(C_left, C_right, 1, type, rdms);
            return rdms;
        }
    }

    if (max_rdm_level >= 2) {
        throw std::runtime_error(
            "Transition RDMs of order 2 or higher are not implemented in GASCISolver (and "
            "more generally in Forte).");
    }
    return std::make_shared<RDMsSpinDependent>();
}

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
ambit::Tensor compute_1rdm_different_irrep(GASVector& C_left, GASVector& C_right, bool alfa) {
    size_t ncmo = C_left.ncmo();
    auto rdm = ambit::Tensor::build(ambit::CoreTensor, alfa ? "1RDM_A" : "1RDM_B", {ncmo, ncmo});
    rdm.zero();

    auto na_left = C_left.alfa_address()->nones();
    auto nb_left = C_left.beta_address()->nones();
    auto na_right = C_right.alfa_address()->nones();
    auto nb_right = C_right.beta_address()->nones();

    if (na_left != na_right or nb_left != nb_right) {
        throw std::runtime_error(
            "FCI transition RDMs: The number of alfa and beta electrons must be the same in the "
            "two wave functions.");
    }

    // Check if there are electrons in the wave function
    if ((alfa and (na_left < 1)) or ((!alfa) and (nb_left < 1)))
        return rdm;

    const auto& alfa_address_left = C_left.alfa_address();
    const auto& beta_address_left = C_left.beta_address();
    const auto& alfa_address_right = C_right.alfa_address();
    const auto& beta_address_right = C_right.beta_address();

    const auto& lists_right = C_right.lists();
    const auto& lists_left = C_left.lists();

    // const auto& alfa_strings_right = lists_right->alfa_strings();
    // const auto& beta_strings_right = lists_right->beta_strings();
    // const auto& alfa_strings_left = lists_left->alfa_strings();
    // const auto& beta_strings_left = lists_left->beta_strings();

    // Compute the lists that map the strings of the right and left wave functions. We only need
    // strings for the part that is left untouched by the a^+_p a_q operator.
    auto string_list = find_string_map(*lists_left, *lists_right, alfa);

    VOListMap vo_list = find_ov_string_map(*lists_left, *lists_right, alfa);

    // auto det_classes_right = lists_right->determinant_classes();
    // auto det_classes_left = lists_left->determinant_classes();
    // if (det_classes_right != det_classes_left) {
    //     throw std::runtime_error(
    //         "FCI transition RDMs: The determinant classes must be the same in the two wave "
    //         "functions.");
    // }

    // here we assume that the two wave functions have the same number of alpha and beta
    // electrons

    auto& rdm_data = rdm.data();

    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_right->determinant_classes()) {
        if (lists_right->detpblk(nI) == 0)
            continue;

        auto Cr = C_right.gather_C_block(GASVector::get_CR(), alfa, alfa_address_right,
                                         beta_address_right, class_Ia, class_Ib, false);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_left->determinant_classes()) {
            // check if the string class on which we don't act is the same for I and J
            // here we cannot assume that the two classes must coincide. So we just check if there
            // are elements in the string list for the given pair of classes
            if (alfa) {
                if (string_list.count(std::make_pair(class_Ib, class_Jb)) == 0)
                    continue;
            } else {
                if (string_list.count(std::make_pair(class_Ia, class_Ja)) == 0)
                    continue;
            }

            if (lists_left->detpblk(nJ) == 0)
                continue;

            auto Cl = C_left.gather_C_block(GASVector::get_CL(), alfa, alfa_address_left,
                                            beta_address_left, class_Ja, class_Jb, false);

            // size_t maxL = alfa ? beta_address_right->strpcls(class_Ib)
            //                    : alfa_address_right->strpcls(class_Ia);

            const auto& string_list_block = alfa ? string_list[std::make_pair(class_Ib, class_Jb)]
                                                 : string_list[std::make_pair(class_Ia, class_Ja)];

            const auto& pq_vo_list = alfa ? vo_list[std::make_pair(class_Ia, class_Ja)]
                                          : vo_list[std::make_pair(class_Ib, class_Jb)];

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                double rdm_element = 0.0;
                for (const auto& [sign, I, J] : vo_list) {
                    for (const auto& [Ip, Jp] : string_list_block)
                        rdm_element += sign * Cl[J][Jp] * Cr[I][Ip];
                }
                rdm_data[p * ncmo + q] += rdm_element;
            }
        }
    }

    // Here we compute the RDMs for the case of different irreps
    // <Ja|a^{+}_p a_q|Ia> CL_{Ja,K} CR_{Ia,K}

    // auto alfa_classes_right = lists_right->alfa_string_classes();
    // auto beta_classes_right = lists_right->beta_string_classes();
    // auto alfa_classes_left = lists_left->alfa_string_classes();
    // auto beta_classes_left = lists_left->beta_string_classes();
    // if (alfa_classes_right.size() != alfa_classes_left.size() or
    //     beta_classes_right.size() != beta_classes_left.size()) {
    //     throw std::runtime_error("FCI transition RDMs: The number of alpha and beta string "
    //                              "classes must be the same in "
    //                              "the two wave functions.");
    // }

    // // loop over blocks of matrix C
    // for (const auto& [nI, class_Ia, class_Ib] : lists_right->determinant_classes()) {
    //     if (lists_right->detpblk(nI) == 0)
    //         continue;

    //     auto Cr = C_right.gather_C_block(GASVector::get_CR(), alfa, alfa_address_right,
    //                                      beta_address_right, class_Ia, class_Ib, false);

    //     for (const auto& [nJ, class_Ja, class_Jb] : lists_left->determinant_classes()) {
    //         // The string class on which we don't act must be the same for I and J
    //         if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
    //             continue;
    //         if (lists_left->detpblk(nJ) == 0)
    //             continue;

    //         auto Cl = C_left.gather_C_block(GASVector::get_CL(), alfa, alfa_address_left,
    //                                         beta_address_left, class_Ja, class_Jb, false);

    //         size_t maxL = alfa ? beta_address_right->strpcls(class_Ib)
    //                            : alfa_address_right->strpcls(class_Ia);

    //         const auto& pq_vo_list = alfa ? lists_right->get_alfa_vo_list(class_Ia, class_Ja)
    //                                       : lists_right->get_beta_vo_list(class_Ib,
    //                                       class_Jb);

    //         for (const auto& [pq, vo_list] : pq_vo_list) {
    //             const auto& [p, q] = pq;
    //             double rdm_element = 0.0;
    //             for (const auto& [sign, I, J] : vo_list) {
    //                 rdm_element += sign * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
    //             }
    //             rdm_data[p * ncmo + q] += rdm_element;
    //         }
    //     }
    // }

    // // Loop over the irreps of the right string
    // for (size_t h_Ia = 0; h_Ia < nirrep; ++h_Ia) {
    //     // The beta right string symmetry is fixed by the symmetry of the right state
    //     int h_Ib = h_Ia ^ symmetry_right;
    //     // the alpha right string symmetry depends on the operators, if they act on the alpha
    //     // string
    //     // then the beta left/right strings have to be the same and so their symmetry.
    //     int h_Ja = alfa ? h_Ib ^ symmetry_left : h_Ia;
    //     // The beta left string symmetry is fixed by the symmetry of the left state
    //     int h_Jb = h_Ja ^ symmetry_left;

    //     if ((detpi_left[h_Ja] > 0) and (detpi_right[h_Ia] > 0)) {
    //         // Fill CR with the correct block
    //         auto Cl = C_left.gather_C_block(GASVector::get_CL(), alfa, alfa_address_left,
    //                                         beta_address_left, h_Ja, h_Jb, false);
    //         auto Cr = C_right.gather_C_block(GASVector::get_CR(), alfa, alfa_address_right,
    //                                          beta_address_right, h_Ia, h_Ib, false);
    //         const size_t maxL =
    //             alfa ? beta_address_right->strpcls(h_Ib) : alfa_address_right->strpcls(h_Ia);
    //         for (size_t p_sym = 0; p_sym < nirrep; ++p_sym) {
    //             int q_sym =
    //                 p_sym ^ symmetry_left ^ symmetry_right; // Select the pair pq that makes
    //                 the
    //                                                         // matrix element total symmetric
    //             for (int p_rel = 0; p_rel < cmopi[p_sym]; ++p_rel) {
    //                 for (int q_rel = 0; q_rel < cmopi[q_sym]; ++q_rel) {
    //                     int p_abs = p_rel + cmopi_offset[p_sym];
    //                     int q_abs = q_rel + cmopi_offset[q_sym];

    //                     // const auto& vo =
    //                     //     alfa ? lists_right->get_alfa_vo_list(p_abs, q_abs, h_Ia, h_Ja)
    //                     //          : lists_right->get_beta_vo_list(p_abs, q_abs, h_Ib,
    //                     h_Jb);
    //                     // double rdm_element = 0.0;
    //                     // for (const auto& [sign, I, J] : vo) {
    //                     //     rdm_element += sign * psi::C_DDOT(maxL, Cl[J], 1, Cr[I], 1);
    //                     // }
    //                     // rdm_data[p_abs * ncmo + q_abs] += rdm_element;
    //                 }
    //             }
    //         }
    //     }
    // } // End loop over h
    return rdm;
}

} // namespace forte
