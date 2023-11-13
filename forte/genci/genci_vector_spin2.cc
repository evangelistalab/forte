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

#include "psi4/libmints/matrix.h"

#include "genci_string_lists.h"
#include "genci_string_address.h"

#include "genci_vector.h"

namespace forte {

double GenCIVector::compute_spin2() {
    double spin2 = 0.0;
    for (const auto& [nI, class_Ia, class_Ib] : lists_->determinant_classes()) {
        // The pq product is totally symmetric so the classes if the result are the same as the
        // classes of the input
        size_t block_size = alfa_address_->strpcls(class_Ia) * beta_address_->strpcls(class_Ib);
        if (block_size == 0)
            continue;
        const auto Cr = C_[nI]->pointer();
        for (const auto& [nJ, class_Ja, class_Jb] : lists_->determinant_classes()) {
            auto Cl = C_[nJ]->pointer();
            const auto& pq_vo_alfa = lists_->get_alfa_vo_list(class_Ia, class_Ja);
            const auto& pq_vo_beta = lists_->get_beta_vo_list(class_Ib, class_Jb);
            // loop over the alfa (p,q) pairs
            for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
                const auto& [p, q] = pq;
                // the correspoding beta pair will be (q,p)
                // check if the pair (q,p) is in the list, if not continue
                if (pq_vo_beta.count(std::make_tuple(q, p)) == 0)
                    continue;
                const auto& vo_beta_list = pq_vo_beta.at(std::make_tuple(q, p));
                for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
                    for (const auto& [sign_b, Ib, Jb] : vo_beta_list) {
                        spin2 += Cl[Ja][Jb] * Cr[Ia][Ib] * sign_a * sign_b;
                    }
                }
            }
        }
    }
    double na = alfa_address_->nones();
    double nb = beta_address_->nones();
    return -spin2 + 0.25 * std::pow(na - nb, 2.0) + 0.5 * (na + nb);
}

} // namespace forte
