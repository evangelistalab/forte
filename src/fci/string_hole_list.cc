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

void StringLists::make_1h_list(GraphPtr graph, GraphPtr graph_1h, H1List& list) {
    int n = graph->nbits();
    int k = graph->nones();
    bool* I = new bool[ncmo_];
    bool* J = new bool[ncmo_];

    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (int h_I = 0; h_I < nirrep_; ++h_I) {
            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                I[i] = false; // 0
            for (int i = std::max(0, n - k); i < n; ++i)
                I[i] = true; // 1
            do {
                if (graph->sym(I) == h_I) {
                    size_t add_I = graph->rel_add(I);
                    for (size_t p = 0; p < ncmo_; ++p) {
                        // copy I to J
                        for (int i = 0; i < n; ++i)
                            J[i] = I[i];
                        if (J[p]) {
                            J[p] = false;
                            short sign = string_sign(J, p);

                            int h_J = graph_1h->sym(J);
                            size_t add_J = graph_1h->rel_add(J);

                            std::tuple<int, size_t, int> I_tuple(h_J, add_J, h_I);
                            list[I_tuple].push_back(H1StringSubstitution(sign, p, add_I));
                        }
                    }
                }
            } while (std::next_permutation(I, I + n));
        }
    } // End loop over h
    delete[] J;
    delete[] I;
}

std::vector<H2StringSubstitution>& StringLists::get_alfa_2h_list(int h_I, size_t add_I, int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_2h_list[I_tuple];
}

std::vector<H2StringSubstitution>& StringLists::get_beta_2h_list(int h_I, size_t add_I, int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_2h_list[I_tuple];
}

void StringLists::make_2h_list(GraphPtr graph, GraphPtr graph_2h, H2List& list) {
    int n = graph->nbits();
    int k = graph->nones();
    bool* I = new bool[ncmo_];
    bool* J = new bool[ncmo_];

    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (int h_I = 0; h_I < nirrep_; ++h_I) {
            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                I[i] = false; // 0
            for (int i = std::max(0, n - k); i < n; ++i)
                I[i] = true; // 1
            do {
                if (graph->sym(I) == h_I) {
                    size_t add_I = graph->rel_add(I);
                    for (size_t q = 0; q < ncmo_; ++q) {
                        for (size_t p = 0; p < ncmo_; ++p) {
                            if (p != q) {
                                // copy I to J
                                for (int i = 0; i < n; ++i)
                                    J[i] = I[i];
                                if (J[q]) {
                                    J[q] = false;
                                    short q_sign = string_sign(J, q);
                                    if (J[p]) {
                                        J[p] = false;
                                        short p_sign = string_sign(J, p);

                                        short sign = p_sign * q_sign;

                                        int h_J = graph_2h->sym(J);
                                        size_t add_J = graph_2h->rel_add(J);

                                        std::tuple<int, size_t, int> I_tuple(h_J, add_J, h_I);
                                        list[I_tuple].push_back(
                                            H2StringSubstitution(sign, p, q, add_I));
                                    }
                                }
                            }
                        }
                    }
                }
            } while (std::next_permutation(I, I + n));
        }
    } // End loop over h
    delete[] J;
    delete[] I;
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
void StringLists::make_3h_list(GraphPtr graph, GraphPtr graph_3h, H3List& list) {
    int n = graph->nbits();
    int k = graph->nones();
    bool* I = new bool[ncmo_];
    bool* J = new bool[ncmo_];

    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (int h_I = 0; h_I < nirrep_; ++h_I) {
            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                I[i] = false; // 0
            for (int i = std::max(0, n - k); i < n; ++i)
                I[i] = true; // 1
            do {
                if (graph->sym(I) == h_I) {
                    size_t add_I = graph->rel_add(I);

                    // apply a_r I
                    for (size_t r = 0; r < ncmo_; ++r) {
                        for (size_t q = 0; q < ncmo_; ++q) {
                            for (size_t p = 0; p < ncmo_; ++p) {
                                if ((p != q) and (p != r) and (q != r)) {
                                    // copy I to J
                                    for (int i = 0; i < n; ++i)
                                        J[i] = I[i];
                                    if (J[r]) {
                                        J[r] = false;
                                        short r_sign = string_sign(J, r);
                                        if (J[q]) {
                                            J[q] = false;
                                            short q_sign = string_sign(J, q);
                                            if (J[p]) {
                                                J[p] = false;
                                                short p_sign = string_sign(J, p);

                                                short sign = p_sign * q_sign * r_sign;

                                                int h_J = graph_3h->sym(J);
                                                size_t add_J = graph_3h->rel_add(J);

                                                std::tuple<int, size_t, int> I_tuple(h_J, add_J,
                                                                                     h_I);
                                                list[I_tuple].push_back(
                                                    H3StringSubstitution(sign, p, q, r, add_I));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } while (std::next_permutation(I, I + n));
        } // End loop over h
    }
    delete[] J;
    delete[] I;
}
}
