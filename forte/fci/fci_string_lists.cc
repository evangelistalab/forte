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
#include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/printing.h"

#include "fci_string_address.h"

#include "fci_string_lists.h"

using namespace psi;

namespace forte {

FCIStringLists::FCIStringLists(psi::Dimension cmopi, std::vector<size_t> core_mo,
                               std::vector<size_t> cmo_to_mo, size_t na, size_t nb,
                               PrintLevel print)
    : nirrep_(cmopi.n()), ncmo_(cmopi.sum()), cmopi_(cmopi), cmo_to_mo_(cmo_to_mo),
      fomo_to_mo_(core_mo), na_(na), nb_(nb), print_(print) {
    startup();
}

void FCIStringLists::startup() {

    auto start_time_sut = std::clock();

    cmopi_offset_.push_back(0);
    for (size_t h = 1; h < nirrep_; ++h) {
        cmopi_offset_.push_back(cmopi_offset_[h - 1] + cmopi_[h - 1]);
    }

    std::vector<int> cmopi_int;
    for (size_t h = 0; h < nirrep_; ++h) {
        cmopi_int.push_back(cmopi_[h]);
    }

    for (size_t h = 0; h < nirrep_; h++) {
        fill_n(back_inserter(cmo_sym_), cmopi_int[h], h); // insert h irrep_size[h] times
    }

    // this object is used to compute the class of a string (a generalization of the irrep)
    std::vector<int> gas_size = {static_cast<int>(ncmo_)};
    std::vector<size_t> gas_mos(ncmo_);
    std::iota(gas_mos.begin(), gas_mos.end(), 0);
    std::vector<std::vector<size_t>> gas_mos_vector({gas_mos});
    std::vector<std::array<int, 6>> alfa_occupation({{static_cast<int>(na_)}});
    std::vector<std::array<int, 6>> beta_occupation({{static_cast<int>(nb_)}});
    std::vector<std::array<int, 6>> alfa_1h_occupation({{static_cast<int>(na_ - 1)}});
    std::vector<std::array<int, 6>> beta_1h_occupation({{static_cast<int>(nb_ - 1)}});
    std::vector<std::array<int, 6>> alfa_2h_occupation({{static_cast<int>(na_ - 2)}});
    std::vector<std::array<int, 6>> beta_2h_occupation({{static_cast<int>(nb_ - 2)}});
    std::vector<std::array<int, 6>> alfa_3h_occupation({{static_cast<int>(na_ - 3)}});
    std::vector<std::array<int, 6>> beta_3h_occupation({{static_cast<int>(nb_ - 3)}});

    auto stop_time_sut = std::clock();
    auto duration_sut = (stop_time_sut - start_time_sut);
    outfile->Printf("\n\nTag_sut: time = %zu\n", duration_sut);

    string_class_ = std::make_shared<FCIStringClass>(cmopi_int);

    if (na_ >= 1) {
        auto start_time_a1 = std::clock();
        auto alfa_1h_strings = make_fci_strings(ncmo_, na_ - 1);
        alfa_address_1h_ = std::make_shared<FCIStringAddress>(ncmo_, na_ - 1, alfa_1h_strings);
        auto stop_time_a1 = std::clock();
        auto duration_a1 = (stop_time_a1 - start_time_a1);
        outfile->Printf("\nTag_a1: time = %zu\n", duration_a1);
    }
    if (nb_ >= 1) {
        auto start_time_b1 = std::clock();
        auto beta_1h_strings = make_fci_strings(ncmo_, nb_ - 1);
        beta_address_1h_ = std::make_shared<FCIStringAddress>(ncmo_, nb_ - 1, beta_1h_strings);
        auto stop_time_b1 = std::clock();
        auto duration_b1 = (stop_time_b1 - start_time_b1);
        outfile->Printf("\nTag_b1: time = %zu\n", duration_b1);
    }

    if (na_ >= 2) {
        auto start_time_a2 = std::clock();
        auto alfa_2h_strings = make_fci_strings(ncmo_, na_ - 2);
        alfa_address_2h_ = std::make_shared<FCIStringAddress>(ncmo_, na_ - 2, alfa_2h_strings);
        auto stop_time_a2 = std::clock();
        auto duration_a2 = (stop_time_a2 - start_time_a2);
        outfile->Printf("\nTag_a2: time = %zu\n", duration_a2);
    }
    if (nb_ >= 2) {
        auto start_time_b2 = std::clock();
        auto beta_2h_strings = make_fci_strings(ncmo_, nb_ - 2);
        beta_address_2h_ = std::make_shared<FCIStringAddress>(ncmo_, nb_ - 2, beta_2h_strings);
        auto stop_time_b2 = std::clock();
        auto duration_b2 = (stop_time_b2 - start_time_b2);
        outfile->Printf("\nTag_b2: time = %zu\n", duration_b2);
    }

    if (na_ >= 3) {
        auto start_time_a3 = std::clock();
        auto alfa_3h_strings = make_fci_strings(ncmo_, na_ - 3);
        alfa_address_3h_ = std::make_shared<FCIStringAddress>(ncmo_, na_ - 3, alfa_3h_strings);
        auto stop_time_a3 = std::clock();
        auto duration_a3 = (stop_time_a3 - start_time_a3);
        outfile->Printf("\nTag_a3: time = %zu\n", duration_a3);
    }
    if (nb_ >= 3) {
        auto start_time_b3 = std::clock();
        auto beta_3h_strings = make_fci_strings(ncmo_, nb_ - 3);
        beta_address_3h_ = std::make_shared<FCIStringAddress>(ncmo_, nb_ - 3, beta_3h_strings);
        auto stop_time_b3 = std::clock();
        auto duration_b3 = (stop_time_b3 - start_time_b3);
        outfile->Printf("\nTag_b3: time = %zu\n", duration_b3);
    }

    // local_timers
    double str_list_timer = 0.0;
    double vo_list_timer = 0.0;
    double nn_list_timer = 0.0;
    double oo_list_timer = 0.0;
    double h1_list_timer = 0.0;
    double h2_list_timer = 0.0;
    double h3_list_timer = 0.0;
    double vovo_list_timer = 0.0;
    double vvoo_list_timer = 0.0;

    outfile->Printf("\n\n------------------------------------------\n\n");

    {
        auto start_time_mkfci = std::clock();
        local_timer t;
        alfa_strings_ = make_fci_strings(ncmo_, na_);
        beta_strings_ = make_fci_strings(ncmo_, nb_);

        alfa_address_ = std::make_shared<FCIStringAddress>(ncmo_, na_, alfa_strings_);
        beta_address_ = std::make_shared<FCIStringAddress>(ncmo_, nb_, beta_strings_);

        str_list_timer += t.get();
        auto stop_time_mkfci = std::clock();
        auto duration_mkfci = (stop_time_mkfci - start_time_mkfci);
        outfile->Printf("\nTag_mkfci: time = %zu\n", duration_mkfci);
    }

    nas_ = 0;
    nbs_ = 0;
    for (size_t h = 0; h < nirrep_; ++h) {
        auto start_time_strpcls = std::clock();
        nas_ += alfa_address_->strpcls(h);
        nbs_ += beta_address_->strpcls(h);
        auto stop_time_strpcls = std::clock();
        auto duration_strpcls = (stop_time_strpcls - start_time_strpcls);
        outfile->Printf("\nTag_strpcls: time = %zu\n", duration_strpcls);
    }

    {
        auto start_time_pair = std::clock();
        local_timer t;
        make_pair_list(pair_list_);
        nn_list_timer += t.get();
        auto stop_time_pair = std::clock();
        auto duration_pair = (stop_time_pair - start_time_pair);
        outfile->Printf("\nTag_pair: time = %zu\n", duration_pair);
    }
    {
        auto start_time_vo = std::clock();
        local_timer t;
        make_vo_list(alfa_address_, alfa_vo_list);
        make_vo_list(beta_address_, beta_vo_list);
        vo_list_timer += t.get();
        auto stop_time_vo = std::clock();
        auto duration_vo = (stop_time_vo - start_time_vo);
        outfile->Printf("\nTag_vo: time = %zu\n", duration_vo);
    }
    {
        auto start_time_oo = std::clock();
        local_timer t;
        make_oo_list(alfa_address_, alfa_oo_list);
        make_oo_list(beta_address_, beta_oo_list);
        oo_list_timer += t.get();
        auto stop_time_oo = std::clock();
        auto duration_oo = (stop_time_oo - start_time_oo);
        outfile->Printf("\nTag_oo: time = %zu\n", duration_oo);
    }
    {
        auto start_time_1h = std::clock();
        local_timer t;
        make_1h_list(alfa_address_, alfa_address_1h_, alfa_1h_list);
        make_1h_list(beta_address_, beta_address_1h_, beta_1h_list);
        h1_list_timer += t.get();
        auto stop_time_1h = std::clock();
        auto duration_1h = (stop_time_1h - start_time_1h);
        outfile->Printf("\nTag_1h: time = %zu\n", duration_1h);
    }
    {
        auto start_time_2h = std::clock();
        local_timer t;
        make_2h_list(alfa_address_, alfa_address_2h_, alfa_2h_list);
        make_2h_list(beta_address_, beta_address_2h_, beta_2h_list);
        h2_list_timer += t.get();
        auto stop_time_2h = std::clock();
        auto duration_2h = (stop_time_2h - start_time_2h);
        outfile->Printf("\nTag_2h: time = %zu\n", duration_2h);
    }
    {
        auto start_time_3h = std::clock();
        local_timer t;
        make_3h_list(alfa_address_, alfa_address_3h_, alfa_3h_list);
        make_3h_list(beta_address_, beta_address_3h_, beta_3h_list);
        h3_list_timer += t.get();
        auto stop_time_3h = std::clock();
        auto duration_3h = (stop_time_3h - start_time_3h);
        outfile->Printf("\nTag_3h: time = %zu\n", duration_3h);
    }
    {
        auto start_time_vvoo = std::clock();
        local_timer t;
        make_vvoo_list(alfa_address_, alfa_vvoo_list);
        make_vvoo_list(beta_address_, beta_vvoo_list);
        vvoo_list_timer += t.get();
        auto stop_time_vvoo = std::clock();
        auto duration_vvoo = (stop_time_vvoo - start_time_vvoo);
        outfile->Printf("\nTag_vvoo: time = %zu\n", duration_vvoo);
    }

    double total_time = str_list_timer + nn_list_timer + vo_list_timer + oo_list_timer +
                        vvoo_list_timer + vovo_list_timer;

    if (print_ >= PrintLevel::Default) {
        table_printer printer;
        printer.add_int_data({{"number of alpha electrons", na_},
                              {"number of beta electrons", nb_},
                              {"number of alpha strings", nas_},
                              {"number of beta strings", nbs_}});
        if (print_ >= PrintLevel::Verbose) {
            printer.add_timing_data({{"timing for strings", str_list_timer},
                                     {"timing for NN strings", nn_list_timer},
                                     {"timing for VO strings", vo_list_timer},
                                     {"timing for OO strings", oo_list_timer},
                                     {"timing for VVOO strings", vvoo_list_timer},
                                     {"timing for 1-hole strings", h1_list_timer},
                                     {"timing for 2-hole strings", h2_list_timer},
                                     {"timing for 3-hole strings", h3_list_timer},
                                     {"total timing", total_time}});
        }

        std::string table = printer.get_table("String Lists");
        outfile->Printf("%s", table.c_str());
    }
}

/**
 * Generate all the pairs p > q with pq in pq_sym
 * these are stored as pair<int,int> in pair_list[pq_sym][pairpi]
 */
void FCIStringLists::make_pair_list(PairList& list) {
    // Loop over irreps of the pair pq
    for (size_t pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
        list.push_back(std::vector<std::pair<int, int>>(0));
        // Loop over irreps of p
        for (size_t p_sym = 0; p_sym < nirrep_; ++p_sym) {
            int q_sym = pq_sym ^ p_sym;
            for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                    int p_abs = p_rel + cmopi_offset_[p_sym];
                    int q_abs = q_rel + cmopi_offset_[q_sym];
                    if (p_abs > q_abs)
                        list[pq_sym].push_back(std::make_pair(p_abs, q_abs));
                }
            }
        }
        pairpi_.push_back(list[pq_sym].size());
    }
}

void FCIStringLists::make_strings(std::shared_ptr<FCIStringAddress> addresser, StringList& list) {
    for (size_t h = 0; h < nirrep_; ++h) {
        list.push_back(std::vector<String>(addresser->strpcls(h)));
    }

    int n = addresser->nbits();
    int k = addresser->nones();

    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        String I;
        const auto I_begin = I.begin();
        const auto I_end =
            I.begin() + n; // this is important, otherwise we would generate all permutations
        // Generate the strings 1111100000
        //                      { k }{n-k}
        for (int i = 0; i < n - k; ++i)
            I[i] = false; // 0
        for (int i = std::max(0, n - k); i < n; ++i)
            I[i] = true; // 1
        do {
            size_t sym_I = string_class_->symmetry(I);
            size_t add_I = addresser->add(I);
            list[sym_I][add_I] = I;
        } while (std::next_permutation(I_begin, I_end));
    }
}

StringList FCIStringLists::make_fci_strings(const int norb, const int ne) {
    auto list = StringList();
    if ((ne >= 0) and (ne <= norb)) {
        String I;
        const auto I_begin = I.begin();
        const auto I_end = I.begin() + norb;
        // First we count the number of strings in each irrep
        std::vector<size_t> strpcls(nirrep_, 0);
        I.zero();
        for (int i = std::max(0, norb - ne); i < norb; ++i)
            I[i] = true; // Generate the string 000011111
        do {
            size_t sym_I = string_class_->symmetry(I);
            strpcls[sym_I]++;
        } while (std::next_permutation(I_begin, I_end));

        // Then we allocate the memory
        for (size_t h = 0; h < nirrep_; ++h) {
            list.push_back(std::vector<String>(strpcls[h]));
        }

        // Finally we generate the strings and store them in the list
        I.zero();
        for (int i = std::max(0, norb - ne); i < norb; ++i)
            I[i] = true; // Generate the string 000011111

        std::vector<size_t> str_add(nirrep_, 0);
        do {
            size_t sym_I = string_class_->symmetry(I);
            size_t add_I = str_add[sym_I];
            list[sym_I][add_I] = I;
            str_add[sym_I]++;
        } while (std::next_permutation(I_begin, I_end));
    }
    return list;
}

std::vector<Determinant> FCIStringLists::make_determinants(int symmetry) const {
    size_t ndets = 0;
    for (size_t ha = 0; ha < nirrep_; ha++) {
        const int hb = symmetry ^ ha;
        ndets += alfa_strings()[ha].size() * beta_strings()[hb].size();
    }
    std::vector<Determinant> dets(ndets);
    size_t addI = 0;
    // Loop over irreps of alpha
    for (size_t ha = 0; ha < nirrep_; ha++) {
        const int hb = symmetry ^ ha;
        // Loop over alpha strings in this irrep
        for (const auto& Ia : alfa_strings()[ha]) {
            // Loop over beta strings in this irrep
            for (const auto& Ib : beta_strings()[hb]) {
                dets[addI] = Determinant(Ia, Ib);
                addI++;
            }
        }
    }
    return dets;
}

size_t FCIStringLists::determinant_address(const Determinant& d) const {
    const auto Ia = d.get_alfa_bits();
    const auto Ib = d.get_beta_bits();
    const size_t ha = alfa_address_->sym(Ia);
    const size_t hb = beta_address_->sym(Ib);
    const auto symmetry = ha ^ hb;
    const size_t addIa = alfa_address_->add(Ia);
    const size_t addIb = beta_address_->add(Ib);
    size_t addI = addIa * beta_address_->strpcls(hb) + addIb;
    for (size_t h = 0; h < ha; h++) {
        addI += alfa_address_->strpcls(h) * beta_address_->strpcls(symmetry ^ h);
    }
    return addI;
}

Determinant FCIStringLists::determinant(size_t address, size_t symmetry) const {
    // find the irreps of alpha and beta strings
    size_t h = 0;
    size_t addI = 0;
    // keep adding the number of determinants in each irrep until we reach the right one
    for (; h < nirrep_; h++) {
        size_t address_offset = alfa_address_->strpcls(h) * beta_address_->strpcls(symmetry ^ h);
        if (address_offset + addI > address) {
            break;
        }
        addI += address_offset;
    }
    const size_t shift = address - addI;
    const size_t beta_size = beta_address_->strpcls(symmetry ^ h);
    const size_t addIa = shift / beta_size;
    const size_t addIb = shift % beta_size;
    String Ia = alfa_str(h, addIa);
    String Ib = beta_str(symmetry ^ h, addIb);
    return Determinant(Ia, Ib);
}

} // namespace forte