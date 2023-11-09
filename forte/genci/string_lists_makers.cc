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

#include "helpers/helpers.h"

#include "genci_string_address.h"
#include "genci_string_lists.h"

#include "string_lists_makers.h"

namespace forte {

/**
 * Generate all the pairs p > q with pq in pq_sym
 * these are stored as pair<int,int> in pair_list[pq_sym][pairpi]
 */
PairList make_pair_list(size_t nirrep, const psi::Dimension& mopi,
                        const std::vector<size_t>& mopi_offset) {
    PairList list(nirrep);
    // Loop over irreps of the pair pq
    for (size_t pq_sym = 0; pq_sym < nirrep; ++pq_sym) {
        // Loop over irreps of p
        for (size_t p_sym = 0; p_sym < nirrep; ++p_sym) {
            int q_sym = pq_sym ^ p_sym;
            for (int p_rel = 0; p_rel < mopi[p_sym]; ++p_rel) {
                for (int q_rel = 0; q_rel < mopi[q_sym]; ++q_rel) {
                    int p_abs = p_rel + mopi_offset[p_sym];
                    int q_abs = q_rel + mopi_offset[q_sym];
                    if (p_abs > q_abs)
                        list[pq_sym].push_back(std::make_pair(p_abs, q_abs));
                }
            }
        }
    }
    return list;
}

StringList make_strings_with_occupation(size_t num_spaces, int nirrep,
                                        const std::vector<int>& space_size,
                                        std::vector<std::vector<size_t>> space_mos,
                                        const std::vector<std::array<int, 6>>& occupations,
                                        std::shared_ptr<StringClass>& string_class) {
    auto list = StringList();
    for (const auto& occupation : occupations) {
        // Something to keep in mind: Here we use ACTIVE as a composite index, which means that we
        // will group the orbitals first by symmetry and then by space. For example, if we have the
        // following GAS: GAS1 = [A1 A1 A1 | A2 | B1 | B2 B2] GAS2 = [A1 | | B1 | B2 ] then the
        // composite space ACTIVE = GAS1 + GAS2 will be: ACTIVE = [A1 A1 A1 A1 | A2 | B1 B1 | B2 B2
        // B2]
        //           G1 G1 G1 G2   G1   G1 G1   G1 G1 G2

        // container to store the strings that generate a give gas space
        std::vector<std::vector<String>> gas_space_string(num_spaces, std::vector<String>{});
        std::vector<std::vector<String>> full_strings(nirrep, std::vector<String>{});

        // enumerate all the possible strings in each GAS space
        for (size_t n = 0; n < num_spaces; n++) {
            String I;
            auto gas_norb = space_size[n];
            auto gas_ne = occupation[n];
            if ((gas_ne >= 0) and (gas_ne <= gas_norb)) {
                const auto I_begin = I.begin();
                const auto I_end = I.begin() + gas_norb;

                I.zero();
                for (int i = std::max(0, gas_norb - gas_ne); i < gas_norb; ++i)
                    I[i] = true; // Generate the string 000011111

                do {
                    String J;
                    J.zero();
                    for (int i = 0; i < gas_norb; ++i) {
                        if (I[i])
                            J[space_mos[n][i]] = true;
                    }
                    gas_space_string[n].push_back(J);
                } while (std::next_permutation(I_begin, I_end));
            }
        }

        auto product_strings = math::cartesian_product(gas_space_string);
        // print product_strings
        // psi::outfile->Printf("\n\n  GAS product strings (size = %d):", product_strings.size());
        for (const auto& strings : product_strings) {
            String I;
            I.zero();
            for (const auto& J : strings) {
                I |= J;
            }
            size_t sym_I = string_class->symmetry(I);
            // psi::outfile->Printf("\n    %s", str(I, ncmo_).c_str());
            full_strings[sym_I].push_back(I);
        }

        for (int h = 0; h < nirrep; h++) {
            list.push_back(full_strings[h]);
        }
    }
    return list;
}

OOListMap make_oo_list(const StringList& strings, std::shared_ptr<StringAddress> addresser) {
    OOListMap list;
    const int nmo = addresser->nbits();
    int k = addresser->nones() - 2;
    if (k >= 0) {
        // Loop over irreps of the pair pq
        for (int p = 0; p < nmo; p++) {
            for (int q = 0; q < p; q++) {
                make_oo(strings, list, p, q);
            }
        }
    }
    return list;
}

void make_oo(const StringList& strings, OOListMap& list, int p, int q) {
    for (int class_I{0}; const auto& string_class : strings) {
        for (u_int32_t add_I{0}; const auto& I : string_class) {
            // find the strings where both p and q are occupied
            if (I[p] and I[q]) {
                auto& list_oo = list[class_I];
                list_oo[std::make_tuple(p, q)].push_back(add_I);
            }
            add_I++;
        }
        class_I++;
    }
}

VOListMap make_vo_list(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
                       const std::shared_ptr<StringAddress>& J_addresser) {
    VOListMap list;
    const int nmo = I_addresser->nbits();
    for (int p = 0; p < nmo; p++) {
        for (int q = 0; q < nmo; q++) {
            make_vo(strings, I_addresser, J_addresser, list, p, q);
        }
    }
    return list;
}

void make_vo(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
             const std::shared_ptr<StringAddress>& J_addresser, VOListMap& list, int p, int q) {
    for (const auto& string_class : strings) {
        for (const auto& I : string_class) {
            const auto& [add_I, class_I] = I_addresser->address_and_class(I);
            auto J = I;
            double sign = 1.0;
            if (J[q]) {
                sign *= J.slater_sign(q);
                J[q] = false;
                if (!J[p]) {
                    sign *= J.slater_sign(p);
                    J[p] = true;
                    if (auto it = J_addresser->find(J); it != J_addresser->end()) {
                        const auto& [add_J, class_J] = it->second;
                        auto& list_IJ = list[std::make_tuple(class_I, class_J)];
                        list_IJ[std::make_tuple(p, q)].push_back(
                            StringSubstitution(sign, add_I, add_J));
                    }
                }
            }
        }
    }
}

VVOOListMap make_vvoo_list(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                           size_t nirrep, const psi::Dimension& mopi,
                           const std::vector<size_t>& mopi_offset) {
    VVOOListMap list;
    // Loop over irreps of the pair pq
    for (size_t pq_sym = 0; pq_sym < nirrep; ++pq_sym) {
        int rs_sym = pq_sym;
        // Loop over irreps of p,r
        for (size_t p_sym = 0; p_sym < nirrep; ++p_sym) {
            int q_sym = pq_sym ^ p_sym;
            for (size_t r_sym = 0; r_sym < nirrep; ++r_sym) {
                size_t s_sym = rs_sym ^ r_sym;
                for (int p_rel = 0; p_rel < mopi[p_sym]; ++p_rel) {
                    for (int q_rel = 0; q_rel < mopi[q_sym]; ++q_rel) {
                        for (int r_rel = 0; r_rel < mopi[r_sym]; ++r_rel) {
                            for (int s_rel = 0; s_rel < mopi[s_sym]; ++s_rel) {
                                int p_abs = p_rel + mopi_offset[p_sym];
                                int q_abs = q_rel + mopi_offset[q_sym];
                                int r_abs = r_rel + mopi_offset[r_sym];
                                int s_abs = s_rel + mopi_offset[s_sym];
                                if ((p_abs > q_abs) && (r_abs > s_abs)) {
                                    // Avoid
                                    if ((not((p_abs == r_abs) and (q_abs == s_abs))) and
                                        (not((p_abs == s_abs) and (q_abs == r_abs)))) {
                                        make_vvoo(strings, addresser, list, p_abs, q_abs, r_abs,
                                                  s_abs);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return list;
}

void make_vvoo(const StringList& strings, std::shared_ptr<StringAddress> addresser,
               VVOOListMap& list, int p, int q, int r, int s) {
    for (const auto& string_class : strings) {
        for (const auto& I : string_class) {
            auto J = I;
            double sign = 1.0;
            // Apply a^{+}_p a^{+}_q a_s a_r to I
            if (J[r]) {
                sign *= J.slater_sign(r);
                J[r] = false;
                if (J[s]) {
                    sign *= J.slater_sign(s);
                    J[s] = false;
                    if (!J[q]) {
                        sign *= J.slater_sign(q);
                        J[q] = true;
                        if (!J[p]) {
                            sign *= J.slater_sign(p);
                            J[p] = true;
                            if (auto it = addresser->find(J); it != addresser->end()) {
                                const auto& [add_I, class_I] = addresser->address_and_class(I);
                                const auto& [add_J, class_J] = it->second;
                                auto& list_IJ = list[std::make_tuple(class_I, class_J)];
                                list_IJ[std::make_tuple(p, q, r, s)].push_back(
                                    StringSubstitution(sign, add_I, add_J));
                            }
                        }
                    }
                }
            }
        }
    }
}

H1List make_1h_list(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                    std::shared_ptr<StringAddress> addresser_1h) {
    H1List list;
    int n = addresser->nbits();
    int k = addresser->nones();
    size_t nmo = addresser->nbits();
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (const auto& string_class : strings) {
            for (const auto& I : string_class) {
                const auto& [add_I, class_I] = addresser->address_and_class(I);
                for (size_t p = 0; p < nmo; ++p) {
                    if (I[p]) {
                        auto J = I;
                        const auto sign = J.slater_sign(p);
                        J[p] = false;
                        if (auto it = addresser_1h->find(J); it != addresser_1h->end()) {
                            const auto& [add_J, class_J] = it->second;
                            std::tuple<int, size_t, int> I_tuple(class_J, add_J, class_I);
                            list[I_tuple].push_back(H1StringSubstitution(sign, p, add_I));
                        }
                    }
                }
            }
        }
    }
    return list;
}

H2List make_2h_list(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                    std::shared_ptr<StringAddress> addresser_2h) {
    H2List list;
    int n = addresser->nbits();
    int k = addresser->nones();
    size_t nmo = addresser->nbits();
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (const auto& string_class : strings) {
            for (const auto& I : string_class) {
                const auto& [add_I, class_I] = addresser->address_and_class(I);
                for (size_t q = 0; q < nmo; ++q) {
                    for (size_t p = q + 1; p < nmo; ++p) {
                        if (I[p] and I[q]) {
                            auto J = I;
                            J[q] = false;
                            const auto q_sign = J.slater_sign(q);
                            J[p] = false;
                            const auto p_sign = J.slater_sign(p);
                            if (auto it = addresser_2h->find(J); it != addresser_2h->end()) {
                                const auto sign = p_sign * q_sign;
                                const auto& [add_J, class_J] = it->second;
                                std::tuple<int, size_t, int> I_tuple(class_J, add_J, class_I);
                                list[I_tuple].push_back(H2StringSubstitution(sign, p, q, add_I));
                                list[I_tuple].push_back(H2StringSubstitution(-sign, q, p, add_I));
                            }
                        }
                    }
                }
            }
        }
    }
    return list;
}

/**
 * Generate all the pairs of strings I,J connected
 * by a^{+}_p a_q
 * that is: J = ± a^{+}_p a_q I. p and q are
 * absolute indices and I belongs to the irrep h.
 */
H3List make_3h_list(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                    std::shared_ptr<StringAddress> addresser_3h) {
    H3List list;
    int n = addresser->nbits();
    int k = addresser->nones();
    size_t nmo = addresser->nbits();
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (const auto& string_class : strings) {
            for (const auto& I : string_class) {
                const auto& [add_I, class_I] = addresser->address_and_class(I);
                for (size_t r = 0; r < nmo; ++r) {
                    for (size_t q = r + 1; q < nmo; ++q) {
                        for (size_t p = q + 1; p < nmo; ++p) {
                            if (I[p] and I[q] and I[r]) {
                                auto J = I;
                                J[r] = false;
                                const auto r_sign = J.slater_sign(r);
                                J[q] = false;
                                const auto q_sign = J.slater_sign(q);
                                J[p] = false;
                                const auto p_sign = J.slater_sign(p);
                                if (auto it = addresser_3h->find(J); it != addresser_3h->end()) {
                                    const auto sign = p_sign * q_sign * r_sign;
                                    const auto& [add_J, class_J] = it->second;
                                    std::tuple<int, size_t, int> I_tuple(class_J, add_J, class_I);
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
            }
        }
    }
    return list;
}

std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>
find_string_map(const GenCIStringLists& list_left, const GenCIStringLists& list_right, bool alfa) {
    std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> m;
    const auto& strings_right = alfa ? list_right.beta_strings() : list_right.alfa_strings();
    const auto& address_left = alfa ? list_left.beta_address() : list_left.alfa_address();
    // loop over all the right string classes (I)
    for (int class_I{0}; const auto& string_class_right : strings_right) {
        // loop over all the right strings (I)
        for (size_t addI{0}; const auto& I : string_class_right) {
            // find the left string class (class_J) and string address (addJ) of the string J = I
            if (auto it = address_left->find(I); it != address_left->end()) {
                const auto& [addJ, class_J] = it->second;
                m[std::make_pair(class_I, class_J)].push_back(std::make_pair(addI, addJ));
            }
            addI++;
        }
        class_I++;
    }
    return m;
}

VOListMap find_ov_string_map(const GenCIStringLists& list_left, const GenCIStringLists& list_right,
                             bool alfa) {
    const auto& strings_right = alfa ? list_right.alfa_strings() : list_right.beta_strings();
    const auto& I_address = alfa ? list_right.alfa_address() : list_right.beta_address();
    const auto& J_address = alfa ? list_left.alfa_address() : list_left.beta_address();
    auto vo_list = make_vo_list(strings_right, I_address, J_address);
    return vo_list;
}

} // namespace forte
