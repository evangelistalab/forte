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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "gas_string_address.h"
#include "gas_string_lists.h"

namespace forte {

/**
 * Returns the list of alfa strings connected by a^{+}_p a^{+}_q a_q a_p
 * @param pq_sym symmetry of the pq pair
 * @param pq     relative PAIRINDEX of the pq pair
 * @param h      symmetry of the I strings in the list
 */
std::vector<u_int32_t>& GASStringLists::get_alfa_oo_list(int pq_sym, size_t pq, int h) {
    std::tuple<int, size_t, int> pq_pair(pq_sym, pq, h);
    return alfa_oo_list[pq_pair];
}

/**
 * Returns the list of beta strings connected by a^{+}_p a^{+}_q a_q a_p
 * @param pq_sym symmetry of the pq pair
 * @param pq     relative PAIRINDEX of the pq pair
 * @param h      symmetry of the I strings in the list
 */
std::vector<u_int32_t>& GASStringLists::get_beta_oo_list(int pq_sym, size_t pq, int h) {
    std::tuple<int, size_t, int> pq_pair(pq_sym, pq, h);
    return beta_oo_list[pq_pair];
}

void GASStringLists::make_oo_list(const StringList& strings,
                                  std::shared_ptr<StringAddress> addresser, OOList2& list) {
    // Loop over irreps of the pair pq
    for (size_t pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
        size_t max_pq = pairpi_[pq_sym];
        for (size_t pq = 0; pq < max_pq; ++pq) {
            make_oo(strings, addresser, list, pq_sym, pq);
        }
    }
}

void GASStringLists::make_oo(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                             OOList2& list, int pq_sym, size_t pq) {
    int k = addresser->nones() - 2;
    if (k >= 0) {
        auto [p, q] = pair_list_[pq_sym][pq];
        for (int class_I{0}; const auto& string_class : strings) {
            for (u_int32_t add_I{0}; const auto& I : string_class) {
                // find the strings where both p and q are occupied
                if (I[p] and I[q]) {
                    std::tuple<int, size_t, int> pq_pair(pq_sym, pq, class_I);
                    list[pq_pair].push_back(add_I);
                }
                add_I++;
            }
            class_I++;
        }
    }
}

// int n = addresser->nbits() - 2;
// String b, I, J;
// auto b_begin = b.begin();
// auto b_end = b.begin() + n;
// for (size_t h = 0; h < nirrep_; ++h) {
//     // Create the key to the map
//     std::tuple<int, size_t, int> pq_pair(pq_sym, pq, h);

//     // Generate the strings 1111100000
//     //                      { k }{n-k}
//     for (int i = 0; i < n - k; ++i)
//         b[i] = false; // 0
//     for (int i = n - k; i < n; ++i)
//         b[i] = true; // 1
//     do {
//         int k = 0;
//         for (int i = 0; i < q; ++i) {
//             J[i] = I[i] = b[k];
//             k++;
//         }
//         for (int i = q + 1; i < p; ++i) {
//             J[i] = I[i] = b[k];
//             k++;
//         }
//         for (int i = p + 1; i < static_cast<int>(ncmo_); ++i) {
//             J[i] = I[i] = b[k];
//             k++;
//         }
//         I[p] = true;
//         I[q] = true;
//         J[p] = true;
//         J[q] = true;
//         // Add the sting only of irrep(I) is h
//         if (string_class_->symmetry(I) == h)
//             list[pq_pair].push_back(
//                 StringSubstitution(1.0, addresser->add(I), addresser->add(J)));
//     } while (std::next_permutation(b_begin, b_end));
// } // End loop over h

/**
 * Returns a vector of tuples containing the sign, I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
std::vector<StringSubstitution>& GASStringLists::get_alfa_vo_list(size_t p, size_t q, int class_I,
                                                                  int class_J) {
    return alfa_vo_list[std::make_tuple(p, q, class_I, class_J)];
}

/**
 * Returns a vector of tuples containing the sign,I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
std::vector<StringSubstitution>& GASStringLists::get_beta_vo_list(size_t p, size_t q, int class_I,
                                                                  int class_J) {
    return beta_vo_list[std::make_tuple(p, q, class_I, class_J)];
}

void GASStringLists::make_vo_list(const StringList& strings,
                                  std::shared_ptr<StringAddress> addresser, VOList2& list) {
    // Loop over irreps of the pair pq
    for (size_t pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
        // Loop over irreps of p
        for (size_t p_sym = 0; p_sym < nirrep_; ++p_sym) {
            int q_sym = pq_sym ^ p_sym;
            for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                    int p_abs = p_rel + cmopi_offset_[p_sym];
                    int q_abs = q_rel + cmopi_offset_[q_sym];
                    make_vo(strings, addresser, list, p_abs, q_abs);
                }
            }
        }
    }
}

void GASStringLists::make_vo(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                             VOList2& list, int p, int q) {
    for (const auto& string_class : strings) {
        for (const auto& I : string_class) {
            auto J = I;
            double sign = 1.0;
            if (J[p]) {
                sign *= J.slater_sign(p);
                J[p] = false;
                if (!J[q]) {
                    sign *= J.slater_sign(q);
                    J[q] = true;
                    if (auto it = addresser->find(J); it != addresser->end()) {
                        const auto& [add_I, class_I] = addresser->address_and_class(I);
                        const auto& [add_J, class_J] = it->second;

                        list[std::make_tuple(p, q, class_I, class_J)].push_back(
                            StringSubstitution(sign, add_I, add_J));
                    }
                }
            }
        }
    }
}

/**
 */
std::vector<StringSubstitution>& GASStringLists::get_alfa_vvoo_list(size_t p, size_t q, size_t r,
                                                                    size_t s, int h) {
    std::tuple<size_t, size_t, size_t, size_t, int> pqrs_pair(p, q, r, s, h);
    return alfa_vvoo_list[pqrs_pair];
}

/**
 */
std::vector<StringSubstitution>& GASStringLists::get_beta_vvoo_list(size_t p, size_t q, size_t r,
                                                                    size_t s, int h) {
    std::tuple<size_t, size_t, size_t, size_t, int> pqrs_pair(p, q, r, s, h);
    return beta_vvoo_list[pqrs_pair];
}

void GASStringLists::make_vvoo_list(std::shared_ptr<StringAddress> addresser, VVOOList& list) {
    // Loop over irreps of the pair pq
    for (size_t pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
        int rs_sym = pq_sym;
        // Loop over irreps of p,r
        for (size_t p_sym = 0; p_sym < nirrep_; ++p_sym) {
            int q_sym = pq_sym ^ p_sym;
            for (size_t r_sym = 0; r_sym < nirrep_; ++r_sym) {
                size_t s_sym = rs_sym ^ r_sym;
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

void GASStringLists::make_vvoo(std::shared_ptr<StringAddress> addresser, VVOOList& list, int p,
                               int q, int r, int s) {
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

std::vector<H1StringSubstitution>& GASStringLists::get_alfa_1h_list(int h_I, size_t add_I,
                                                                    int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_1h_list[I_tuple];
}

std::vector<H1StringSubstitution>& GASStringLists::get_beta_1h_list(int h_I, size_t add_I,
                                                                    int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_1h_list[I_tuple];
}

void GASStringLists::make_1h_list(std::shared_ptr<StringAddress> addresser,
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

std::vector<H2StringSubstitution>& GASStringLists::get_alfa_2h_list(int h_I, size_t add_I,
                                                                    int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_2h_list[I_tuple];
}

std::vector<H2StringSubstitution>& GASStringLists::get_beta_2h_list(int h_I, size_t add_I,
                                                                    int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_2h_list[I_tuple];
}

void GASStringLists::make_2h_list(std::shared_ptr<StringAddress> addresser,
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

std::vector<H3StringSubstitution>& GASStringLists::get_alfa_3h_list(int h_I, size_t add_I,
                                                                    int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_3h_list[I_tuple];
}

std::vector<H3StringSubstitution>& GASStringLists::get_beta_3h_list(int h_I, size_t add_I,
                                                                    int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_3h_list[I_tuple];
}

/**
 * Generate all the pairs of strings I,J connected
 * by a^{+}_p a_q
 * that is: J = ± a^{+}_p a_q I. p and q are
 * absolute indices and I belongs to the irrep h.
 */
void GASStringLists::make_3h_list(std::shared_ptr<StringAddress> addresser,
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
