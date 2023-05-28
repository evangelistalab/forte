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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "fci/string_address.h"

#include "string_lists.h"

namespace forte {

/**
 */
std::vector<StringSubstitution>& StringLists::get_alfa_vvoo_list(size_t p, size_t q, size_t r,
                                                                 size_t s, int h) {
    std::tuple<size_t, size_t, size_t, size_t, int> pqrs_pair(p, q, r, s, h);
    return alfa_vvoo_list[pqrs_pair];
}

/**
 */
std::vector<StringSubstitution>& StringLists::get_beta_vvoo_list(size_t p, size_t q, size_t r,
                                                                 size_t s, int h) {
    std::tuple<size_t, size_t, size_t, size_t, int> pqrs_pair(p, q, r, s, h);
    return beta_vvoo_list[pqrs_pair];
}

void StringLists::make_vvoo_list(std::shared_ptr<StringAddress> addresser, VVOOList& list) {
    // Loop over irreps of the pair pq
    for (int pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
        int rs_sym = pq_sym;
        // Loop over irreps of p,r
        for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
            int q_sym = pq_sym ^ p_sym;
            for (int r_sym = 0; r_sym < nirrep_; ++r_sym) {
                int s_sym = rs_sym ^ r_sym;
                for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                    for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                        for (int r_rel = 0; r_rel < cmopi_[r_sym]; ++r_rel) {
                            for (int s_rel = 0; s_rel < cmopi_[s_sym]; ++s_rel) {
                                int p_abs = p_rel + cmopi_offset_[p_sym];
                                int q_abs = q_rel + cmopi_offset_[q_sym];
                                int r_abs = r_rel + cmopi_offset_[r_sym];
                                int s_abs = s_rel + cmopi_offset_[s_sym];
                                if ((p_abs > q_abs) && (r_abs > s_abs)) {
                                    // Avoid
                                    if (not((p_abs == r_abs) and (q_abs == s_abs))) {
                                        make_vvoo(addresser, list, p_abs, q_abs, r_abs, s_abs);
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

void StringLists::make_vvoo(std::shared_ptr<StringAddress> addresser, VVOOList& list, int p, int q,
                            int r, int s) {
    // Sort pqrs
    int a[4];
    a[0] = s;
    a[1] = r;
    a[2] = q;
    a[3] = p;

    //  if (a[0] > a[1]) std::swap(a[0], a[1]);
    if (a[1] > a[2])
        std::swap(a[1], a[2]);
    //  if (a[2] > a[3]) std::swap(a[2], a[3]);

    if (a[0] > a[1])
        std::swap(a[0], a[1]);
    if (a[1] > a[2])
        std::swap(a[1], a[2]);
    if (a[2] > a[3])
        std::swap(a[2], a[3]);

    if (a[0] > a[1])
        std::swap(a[0], a[1]);
    if (a[1] > a[2])
        std::swap(a[1], a[2]);
    if (a[2] > a[3])
        std::swap(a[2], a[3]);

    bool overlap = false;
    if ((a[0] == a[1]) || (a[1] == a[2]) || (a[2] == a[3]))
        overlap = true;

    int n = addresser->nbits() - 4 + (overlap ? 1 : 0);
    int k = addresser->nones() - 2;

    if (k >= 0 and n >= 0 and (n >= k)) {
        std::vector<int8_t> b(n);
        String I, J;

        for (size_t h = 0; h < nirrep_; ++h) {
            // Create the key to the map
            std::tuple<size_t, size_t, size_t, size_t, int> pqrs_pair(p, q, r, s, h);

            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                b[i] = false; // 0
            for (int i = n - k; i < n; ++i)
                b[i] = true; // 1
            do {
                I[p] = false;
                I[q] = false;
                I[s] = true;
                I[r] = true;
                // Form the string I with r and s true
                int k = 0;
                for (int i = 0; i < a[0]; ++i) {
                    I[i] = b[k];
                    k++;
                }
                for (int i = a[0] + 1; i < a[1]; ++i) {
                    I[i] = b[k];
                    k++;
                }
                for (int i = a[1] + 1; i < a[2]; ++i) {
                    I[i] = b[k];
                    k++;
                }
                for (int i = a[2] + 1; i < a[3]; ++i) {
                    I[i] = b[k];
                    k++;
                }
                for (int i = a[3] + 1; i < static_cast<int>(ncmo_); ++i) {
                    I[i] = b[k];
                    k++;
                }
                if (string_class_->symmetry(I) == h) {
                    J = I;
                    short sign = 1;
                    // Apply a^{+}_p a^{+}_q a_s a_r to I
                    for (int i = s; i < r; ++i)
                        if (J[i])
                            sign *= -1;
                    J[r] = false;
                    J[s] = false;
                    if (!J[q]) { // q = 0
                        J[q] = true;
                        sign *= J.slater_sign(q);
                        if (!J[p]) { // p = 0
                            J[p] = true;
                            sign *= J.slater_sign(p);
                            list[pqrs_pair].push_back(
                                StringSubstitution(sign, addresser->add(I), addresser->add(J)));
                        }
                    }
                }
            } while (std::next_permutation(b.begin(), b.end()));
        } // End loop over h
    }
}

} // namespace forte
