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

/*
 *  string_vo_list.cc
 *  Capriccio
 *
 *  Created by Francesco Evangelista on 3/18/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <algorithm>

#include "psi4/psi4-dec.h"
#include "fci/string_address.h"

#include "string_lists.h"

namespace forte {

std::vector<H1StringSubstitution>& StringLists::get_alfa_1h_list(int h_I, size_t add_I, int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_1h_list[I_tuple];
}

std::vector<H1StringSubstitution>& StringLists::get_beta_1h_list(int h_I, size_t add_I, int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_1h_list[I_tuple];
}

void StringLists::make_1h_list(std::shared_ptr<StringAddress> addresser,
                               std::shared_ptr<StringAddress> addresser_1h, H1List& list) {
    int n = addresser->nbits();
    int k = addresser->nones();
    String I, J;
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (size_t h_I = 0; h_I < nirrep_; ++h_I) {
            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                I[i] = false; // 0
            for (int i = std::max(0, n - k); i < n; ++i)
                I[i] = true; // 1
            do {
                if (string_class_->symmetry(I) == h_I) {
                    size_t add_I = addresser->add(I);
                    for (size_t p = 0; p < ncmo_; ++p) {
                        if (I[p]) {
                            J = I;
                            J[p] = false;
                            short sign = J.slater_sign(p);

                            int h_J = addresser_1h->sym(J);
                            size_t add_J = addresser_1h->add(J);

                            std::tuple<int, size_t, int> I_tuple(h_J, add_J, h_I);
                            list[I_tuple].push_back(H1StringSubstitution(sign, p, add_I));
                        }
                    }
                }
            } while (std::next_permutation(I.begin(), I.begin() + n));
        }
    } // End loop over h
}

std::vector<H2StringSubstitution>& StringLists::get_alfa_2h_list(int h_I, size_t add_I, int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_2h_list[I_tuple];
}

std::vector<H2StringSubstitution>& StringLists::get_beta_2h_list(int h_I, size_t add_I, int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_2h_list[I_tuple];
}

void StringLists::make_2h_list(std::shared_ptr<StringAddress> addresser,
                               std::shared_ptr<StringAddress> addresser_2h, H2List& list) {
    int n = addresser->nbits();
    int k = addresser->nones();
    String I, J;

    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (size_t h_I = 0; h_I < nirrep_; ++h_I) {
            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                I[i] = false; // 0
            for (int i = std::max(0, n - k); i < n; ++i)
                I[i] = true; // 1
            do {
                if (string_class_->symmetry(I) == h_I) {
                    size_t add_I = addresser->add(I);
                    for (size_t q = 0; q < ncmo_; ++q) {
                        for (size_t p = q + 1; p < ncmo_; ++p) {
                            if (I[q] and I[p]) {
                                J = I;
                                J[q] = false;
                                short q_sign = J.slater_sign(q);
                                J[p] = false;
                                short p_sign = J.slater_sign(p);

                                short sign = p_sign * q_sign;

                                int h_J = addresser_2h->sym(J);
                                size_t add_J = addresser_2h->add(J);

                                std::tuple<int, size_t, int> I_tuple(h_J, add_J, h_I);
                                list[I_tuple].push_back(H2StringSubstitution(sign, p, q, add_I));
                                list[I_tuple].push_back(H2StringSubstitution(-sign, q, p, add_I));
                            }
                        }
                    }
                }
            } while (std::next_permutation(I.begin(), I.begin() + n));
        }
    } // End loop over h
}

std::vector<H3StringSubstitution>& StringLists::get_alfa_3h_list(int h_I, size_t add_I, int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_3h_list[I_tuple];
}

std::vector<H3StringSubstitution>& StringLists::get_beta_3h_list(int h_I, size_t add_I, int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_3h_list[I_tuple];
}

/**
 * Generate all the pairs of strings I,J connected
 * by a^{+}_p a_q
 * that is: J = ± a^{+}_p a_q I. p and q are
 * absolute indices and I belongs to the irrep h.
 */
void StringLists::make_3h_list(std::shared_ptr<StringAddress> addresser,
                               std::shared_ptr<StringAddress> addresser_3h, H3List& list) {
    int n = addresser->nbits();
    int k = addresser->nones();
    String I, J;

    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (size_t h_I = 0; h_I < nirrep_; ++h_I) {
            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                I[i] = false; // 0
            for (int i = std::max(0, n - k); i < n; ++i)
                I[i] = true; // 1
            do {
                if (string_class_->symmetry(I) == h_I) {
                    size_t add_I = addresser->add(I);

                    // apply a_r I
                    for (size_t r = 0; r < ncmo_; ++r) {
                        for (size_t q = r + 1; q < ncmo_; ++q) {
                            for (size_t p = q + 1; p < ncmo_; ++p) {
                                if (I[r] and I[q] and I[p]) {
                                    J = I;
                                    J[r] = false;
                                    short r_sign = J.slater_sign(r);
                                    J[q] = false;
                                    short q_sign = J.slater_sign(q);
                                    J[p] = false;
                                    short p_sign = J.slater_sign(p);
                                    short sign = p_sign * q_sign * r_sign;

                                    int h_J = addresser_3h->sym(J);
                                    size_t add_J = addresser_3h->add(J);

                                    std::tuple<int, size_t, int> I_tuple(h_J, add_J, h_I);
                                    list[I_tuple].push_back(
                                        H3StringSubstitution(+sign, p, q, r, add_I));
                                    list[I_tuple].push_back(
                                        H3StringSubstitution(-sign, p, r, q, add_I));
                                    list[I_tuple].push_back(
                                        H3StringSubstitution(-sign, q, p, r, add_I));
                                    list[I_tuple].push_back(
                                        H3StringSubstitution(+sign, q, r, p, add_I));
                                    list[I_tuple].push_back(
                                        H3StringSubstitution(-sign, r, q, p, add_I));
                                    list[I_tuple].push_back(
                                        H3StringSubstitution(+sign, r, p, q, add_I));
                                }
                            }
                        }
                    }
                }
            } while (std::next_permutation(I.begin(), I.begin() + n));
        } // End loop over h
    }
}
} // namespace forte
