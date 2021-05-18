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
#include "psi4/libpsi4util/PsiOutStream.h"

#include "string_lists.h"


namespace forte {

/**
 * Returns a vector of tuples containing the sign, I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
std::vector<StringSubstitution>& StringLists::get_alfa_vo_list(size_t p, size_t q, int h) {
    std::tuple<size_t, size_t, int> pq_pair(p, q, h);
    return alfa_vo_list[pq_pair];
}

/**
 * Returns a vector of tuples containing the sign,I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
std::vector<StringSubstitution>& StringLists::get_beta_vo_list(size_t p, size_t q, int h) {
    std::tuple<size_t, size_t, int> pq_pair(p, q, h);
    return beta_vo_list[pq_pair];
}

void StringLists::make_vo_list(GraphPtr graph, VOList& list) {
    // Loop over irreps of the pair pq
    for (int pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
        // Loop over irreps of p
        for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
            int q_sym = pq_sym ^ p_sym;
            for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                    int p_abs = p_rel + cmopi_offset_[p_sym];
                    int q_abs = q_rel + cmopi_offset_[q_sym];
                    make_vo(graph, list, p_abs, q_abs);
                }
            }
        }
    }
}

/**
 * Generate all the pairs of strings I,J connected by a^{+}_p a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
void StringLists::make_vo(GraphPtr graph, VOList& list, int p, int q) {
    int n = graph->nbits() - 1 - (p == q ? 0 : 1);
    int k = graph->nones() - 1;
    bool* b = new bool[n];
    bool* I = new bool[ncmo_];
    bool* J = new bool[ncmo_];
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (int h = 0; h < nirrep_; ++h) {
            // Create the key to the map
            std::tuple<size_t, size_t, int> pq_pair(p, q, h);

            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                b[i] = false; // 0
            for (int i = std::max(0, n - k); i < n; ++i)
                b[i] = true; // 1
            do {
                int k = 0;
                short sign = 1;
                for (int i = 0; i < std::min(p, q); ++i) {
                    J[i] = I[i] = b[k];
                    k++;
                }
                for (int i = std::min(p, q) + 1; i < std::max(p, q); ++i) {
                    J[i] = I[i] = b[k];
                    if (b[k])
                        sign *= -1;
                    k++;
                }
                for (int i = std::max(p, q) + 1; i < static_cast<int>(ncmo_); ++i) {
                    J[i] = I[i] = b[k];
                    k++;
                }
                I[p] = 0;
                I[q] = 1;
                J[q] = 0;
                J[p] = 1;

                // Add the sting only of irrep(I) is h
                if (graph->sym(I) == h)
                    list[pq_pair].push_back(
                        StringSubstitution(sign, graph->rel_add(I), graph->rel_add(J)));
            } while (std::next_permutation(b, b + n));

        } // End loop over h

        delete[] J;
        delete[] I;
        delete[] b;
    }
}
} // namespace forte


///**
// * Generate all the pairs of strings I,J connected by a^{+}_p a_q
// * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
// * the irrep h.
// */
// void StringLists::make_kh_list(GraphPtr graph, KHList& list) {
//    int n = graph->nbits();
//    int k = graph->nones();
//    outfile->Printf("\n n = %d, k = %d, ncmo_ = %d",n,k,ncmo_);
//    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
//        bool* I = new bool[ncmo_];
//        bool* J = new bool[ncmo_];
//        for (int h_I = 0; h_I < nirrep_; ++h_I) {
//            // Generate the strings 1111100000
//            //                      { k }{n-k}
//            for (int i = 0; i < n - k; ++i)
//                I[i] = false; // 0
//            for (int i = std::max(0, n - k); i < n; ++i)
//                I[i] = true; // 1
//            do {
//                if (graph->sym(I) == h_I) {
//                    size_t add_I = graph->rel_add(I);
//                    // copy I to J
//                    for (int i = 0; i < n; ++i)
//                        J[i] = I[i];

//                    // apply a^{+}_p a_q I
//                    for (int q = 0; q < ncmo_; ++q) {
//                        if (J[q]) {
//                            J[q] = false;
//                            short q_sign = string_sign(J, q);
//                            for (int p = 0; p < ncmo_; ++p) {
//                                if (not J[p]) {
//                                    short pq_sign = q_sign * string_sign(J, p);
//                                    J[p] = true;

//                                    graph->rel_add(J);
//                                    int h_J = graph->sym(J);
//                                    size_t add_J = graph->rel_add(J);

//                                    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);

//                                    list[I_tuple].push_back(
//                                        KHStringSubstitution(pq_sign, p, q, add_J));
//                                }
//                            }
//                        }
//                    }
//                }
//            } while (std::next_permutation(I, I + n));

//        } // End loop over h
//        delete[] J;
//        delete[] I;
//    }
//}
