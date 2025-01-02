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

#include "genci_string_lists.h"
#include "genci_string_address.h"

#include "genci_vector.h"

namespace forte {

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
std::shared_ptr<RDMs> compute_transition_rdms(GenCIVector& C_left, GenCIVector& C_right,
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
            return rdms;
        }
    } else {
        g1a("pq") += g1b("pq");
        if (max_rdm_level == 1) {
            auto rdms = std::make_shared<RDMsSpinFree>(g1a);
            return rdms;
        }
    }

    if (max_rdm_level >= 2) {
        throw std::runtime_error(
            "Transition RDMs of order 2 or higher are not implemented in GenCISolver (and "
            "more generally in Forte).");
    }
    return std::make_shared<RDMsSpinDependent>();
}

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
ambit::Tensor compute_1rdm_different_irrep(GenCIVector& C_left, GenCIVector& C_right, bool alfa) {

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
    size_t ncmo = C_left.ncmo();
    auto rdm = ambit::Tensor::build(ambit::CoreTensor, alfa ? "1RDM_A" : "1RDM_B", {ncmo, ncmo});
    if ((alfa and (na_left < 1)) or ((!alfa) and (nb_left < 1)))
        return rdm;

    const auto& alfa_address_left = C_left.alfa_address();
    const auto& beta_address_left = C_left.beta_address();
    const auto& alfa_address_right = C_right.alfa_address();
    const auto& beta_address_right = C_right.beta_address();

    const auto& lists_right = C_right.lists();
    const auto& lists_left = C_left.lists();

    // Compute the lists that map the strings of the right and left wave functions. We only need
    // strings for the part that is left untouched by the a^+_p a_q operator.
    auto string_list = find_string_map(*lists_left, *lists_right, alfa);
    // Compute the VO lists that map the strings of the right and left wave functions. We only need
    // strings for the part that is affected by the a^+_p a_q operator.
    VOListMap vo_list = find_ov_string_map(*lists_left, *lists_right, alfa);

    rdm.zero();
    auto& rdm_data = rdm.data();

    // Here we compute the RDMs for the case of different irreps
    // <Ja|a^{+}_p a_q|Ia> CL_{Ja,K} CR_{Ia,K}

    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_right->determinant_classes()) {
        if (lists_right->detpblk(nI) == 0)
            continue;

        auto Cr = C_right.gather_C_block(GenCIVector::get_CR(), alfa, alfa_address_right,
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

            auto Cl = C_left.gather_C_block(GenCIVector::get_CL(), alfa, alfa_address_left,
                                            beta_address_left, class_Ja, class_Jb, false);

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
    return rdm;
}

} // namespace forte
