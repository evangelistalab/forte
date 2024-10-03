/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <cmath>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/printing.h"

#include "base_classes/mo_space_info.h"

#include "genci_string_address.h"
#include "ci_occupation.h"
#include "string_lists_makers.h"

#include "genci_string_lists.h"

using namespace psi;

namespace forte {

// Global debug flag
bool debug_gas_strings = true;

// Wrapper function
template <typename Func> void debug(Func func) {
    if (debug_gas_strings) {
        func();
    }
}

GenCIStringLists::GenCIStringLists(std::shared_ptr<MOSpaceInfo> mo_space_info, size_t na, size_t nb,
                                   int symmetry, PrintLevel print, const std::vector<int> gas_min,
                                   const std::vector<int> gas_max)
    : symmetry_(symmetry), nirrep_(mo_space_info->nirrep()), ncmo_(mo_space_info->size("ACTIVE")),
      na_(na), nb_(nb), print_(print), gas_min_(gas_min), gas_max_(gas_max) {
    startup(mo_space_info);
}

void GenCIStringLists::startup(std::shared_ptr<MOSpaceInfo> mo_space_info) {

    // Get the number of correlated molecular orbitals in each irrep
    cmopi_ = mo_space_info->dimension("ACTIVE");
    // Get the number of orbitals in each irrep
    cmo_sym_ = mo_space_info->symmetry("ACTIVE");

    cmopi_offset_.push_back(0);
    for (size_t h = 1; h < nirrep_; ++h) {
        cmopi_offset_.push_back(cmopi_offset_[h - 1] + cmopi_[h - 1]);
    }

    std::vector<int> cmopi_int;
    for (size_t h = 0; h < nirrep_; ++h) {
        cmopi_int.push_back(cmopi_[h]);
    }

    auto gas_space_names = mo_space_info->composite_space_names()["ACTIVE"];

    for (size_t n = 0; n < gas_space_names.size(); ++n) {
        const std::string& space = gas_space_names[n];
        auto pos = mo_space_info->pos_in_space(space, "ACTIVE");
        gas_mos_.push_back(pos);
        gas_size_.push_back(mo_space_info->size(space));
    }

    // find the number of GAS spaces
    for (int i{0}; i < 6; ++i) {
        if (gas_size_[i] > 0) {
            ngas_spaces_ = i + 1;
        }
    }

    // debug([&]() {
    //     psi::outfile->Printf("\n    GAS space sizes: %s",
    //     container_to_string(gas_size_).c_str()); for (size_t n = 0; n < ngas_spaces_; ++n) {
    //         psi::outfile->Printf("\n    GAS%d MOs: %s", n + 1,
    //                              container_to_string(gas_mos_[n]).c_str());
    //     }
    // });

    std::tie(ngas_spaces_, gas_alfa_occupations_, gas_beta_occupations_, gas_occupations_) =
        get_ci_occupation_patterns(na_, nb_, gas_min_, gas_max_, gas_size_);

    if (print_ >= PrintLevel::Default) {
        print_h2("Possible Electron Occupations");
        auto table = occupation_table(ngas_spaces_, gas_alfa_occupations_, gas_beta_occupations_,
                                      gas_occupations_);
        outfile->Printf("%s", table.c_str());
    }

    // local_timers
    double str_list_timer = 0.0;
    double vo_list_timer = 0.0;
    double nn_list_timer = 0.0;
    double oo_list_timer = 0.0;
    double h1_list_timer = 0.0;
    double h2_list_timer = 0.0;
    double h3_list_timer = 0.0;
    double h4_list_timer = 0.0;
    double vovo_list_timer = 0.0;
    double vvoo_list_timer = 0.0;

    // this object is used to compute the class of a string (a generalization of the irrep)
    string_class_ =
        std::make_shared<StringClass>(symmetry_, cmopi_int, gas_mos_, gas_alfa_occupations_,
                                      gas_beta_occupations_, gas_occupations_);

    // Build the string lists and the addresser
    {
        local_timer t;
        alfa_strings_ = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                     gas_alfa_occupations_, string_class_);
        beta_strings_ = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                     gas_beta_occupations_, string_class_);

        alfa_address_ = std::make_shared<StringAddress>(gas_size_, na_, alfa_strings_);
        beta_address_ = std::make_shared<StringAddress>(gas_size_, nb_, beta_strings_);

        str_list_timer += t.get();
    }

    // from here down the code has to be rewritten to use the new StringAddress class

    gas_alfa_1h_occupations_ = generate_1h_occupations(gas_alfa_occupations_);
    gas_beta_1h_occupations_ = generate_1h_occupations(gas_beta_occupations_);
    gas_alfa_2h_occupations_ = generate_1h_occupations(gas_alfa_1h_occupations_);
    gas_beta_2h_occupations_ = generate_1h_occupations(gas_beta_1h_occupations_);
    gas_alfa_3h_occupations_ = generate_1h_occupations(gas_alfa_2h_occupations_);
    gas_beta_3h_occupations_ = generate_1h_occupations(gas_beta_2h_occupations_);
    gas_alfa_4h_occupations_ = generate_1h_occupations(gas_alfa_3h_occupations_);
    gas_beta_4h_occupations_ = generate_1h_occupations(gas_beta_3h_occupations_);

    if (na_ >= 1) {
        auto alfa_1h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_alfa_1h_occupations_, string_class_);
        alfa_address_1h_ = std::make_shared<StringAddress>(gas_size_, na_ - 1, alfa_1h_strings);
    }
    if (nb_ >= 1) {
        auto beta_1h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_beta_1h_occupations_, string_class_);
        beta_address_1h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 1, beta_1h_strings);
    }

    if (na_ >= 2) {
        auto alfa_2h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_alfa_2h_occupations_, string_class_);
        alfa_address_2h_ = std::make_shared<StringAddress>(gas_size_, na_ - 2, alfa_2h_strings);
    }
    if (nb_ >= 2) {
        auto beta_2h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_beta_2h_occupations_, string_class_);
        beta_address_2h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 2, beta_2h_strings);
    }
    if (na_ >= 3) {
        auto alfa_3h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_alfa_3h_occupations_, string_class_);
        alfa_address_3h_ = std::make_shared<StringAddress>(gas_size_, na_ - 3, alfa_3h_strings);
    }
    if (nb_ >= 3) {
        auto beta_3h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_beta_3h_occupations_, string_class_);
        beta_address_3h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 3, beta_3h_strings);
    }
    if (na_ >= 4) {
        auto alfa_4h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_alfa_4h_occupations_, string_class_);
        alfa_address_4h_ = std::make_shared<StringAddress>(gas_size_, na_ - 4, alfa_4h_strings);
    }
    if (nb_ >= 4) {
        auto beta_4h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_beta_4h_occupations_, string_class_);
        beta_address_4h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 4, beta_4h_strings);
    }

    nas_ = 0;
    nbs_ = 0;

    for (int class_Ia = 0; class_Ia < alfa_address_->nclasses(); ++class_Ia) {
        nas_ += alfa_address_->strpcls(class_Ia);
    }
    for (int class_Ib = 0; class_Ib < beta_address_->nclasses(); ++class_Ib) {
        nbs_ += beta_address_->strpcls(class_Ib);
    }

    ndet_ = 0;
    for (const auto& [n, class_Ia, class_Ib] : determinant_classes()) {
        const auto nIa = alfa_address_->strpcls(class_Ia);
        const auto nIb = beta_address_->strpcls(class_Ib);
        const auto nI = nIa * nIb;
        detpblk_.push_back(nI);
        detpblk_offset_.push_back(ndet_);
        ndet_ += nI;
    }

    {
        local_timer t;
        pair_list_ = make_pair_list(nirrep_, cmopi_, cmopi_offset_);
        nn_list_timer += t.get();
    }
    {
        local_timer t;
        alfa_vo_list = make_vo_list(alfa_strings_, alfa_address_, alfa_address_);
        beta_vo_list = make_vo_list(beta_strings_, beta_address_, beta_address_);
        vo_list_timer += t.get();
    }
    {
        local_timer t;
        alfa_oo_list = make_oo_list(alfa_strings_, alfa_address_);
        beta_oo_list = make_oo_list(beta_strings_, beta_address_);
        oo_list_timer += t.get();
    }
    {
        local_timer t;
        alfa_vvoo_list =
            make_vvoo_list(alfa_strings_, alfa_address_, nirrep_, cmopi_, cmopi_offset_);
        beta_vvoo_list =
            make_vvoo_list(beta_strings_, beta_address_, nirrep_, cmopi_, cmopi_offset_);
        vvoo_list_timer += t.get();
    }
    {
        local_timer t;
        alfa_1h_list = make_1h_list(alfa_strings_, alfa_address_, alfa_address_1h_);
        beta_1h_list = make_1h_list(beta_strings_, beta_address_, beta_address_1h_);
        h1_list_timer += t.get();
    }
    {
        local_timer t;
        alfa_2h_list = make_2h_list(alfa_strings_, alfa_address_, alfa_address_2h_);
        beta_2h_list = make_2h_list(beta_strings_, beta_address_, beta_address_2h_);
        h2_list_timer += t.get();
    }
    {
        local_timer t;
        alfa_3h_list = make_3h_list(alfa_strings_, alfa_address_, alfa_address_3h_);
        beta_3h_list = make_3h_list(beta_strings_, beta_address_, beta_address_3h_);
        h3_list_timer += t.get();
    }
    {
        local_timer t;
        alfa_4h_list = make_4h_list(alfa_strings_, alfa_address_, alfa_address_4h_);
        beta_4h_list = make_4h_list(beta_strings_, beta_address_, beta_address_4h_);
        h4_list_timer += t.get();
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
                                     {"timing for 4-hole strings", h4_list_timer},
                                     {"total timing", total_time}});
        }
        std::string table = printer.get_table("String Lists");
        outfile->Printf("%s", table.c_str());
    }
}

std::vector<Determinant> GenCIStringLists::make_determinants() const {
    std::vector<Determinant> dets(ndet_);
    for (size_t add_I{0}; const auto& [n, class_Ia, class_Ib] : determinant_classes()) {
        const auto nIa = alfa_address_->strpcls(class_Ia);
        const auto nIb = beta_address_->strpcls(class_Ib);
        for (size_t Ia = 0; Ia < nIa; ++Ia) {
            for (size_t Ib = 0; Ib < nIb; ++Ib) {
                Determinant I(alfa_str(class_Ia, Ia), beta_str(class_Ib, Ib));
                dets[add_I] = I;
                add_I++;
            }
        }
    }
    return dets;
}

size_t GenCIStringLists::determinant_address(const Determinant& d) const {
    const auto Ia = d.get_alfa_bits();
    const auto Ib = d.get_beta_bits();

    const auto& [addIa, class_Ia] = alfa_address_->address_and_class(Ia);
    const auto& [addIb, class_Ib] = beta_address_->address_and_class(Ib);
    size_t addI = addIa * beta_address_->strpcls(class_Ib) + addIb;
    int n = string_class_->block_index(class_Ia, class_Ib);
    addI += detpblk_offset_[n];
    return addI;
}

Determinant GenCIStringLists::determinant(size_t address) const {
    // find the irreps of alpha and beta strings
    size_t n = 0;
    size_t addI = 0;
    // keep adding the number of determinants in each irrep until we reach the right one
    for (size_t maxh = determinant_classes().size(); n < maxh; n++) {
        if (addI + detpblk_[n] > address) {
            break;
        }
        addI += detpblk_[n];
    }
    const size_t shift = address - addI;
    const auto& [_, class_Ia, class_Ib] = determinant_classes().at(n);
    const size_t beta_size = beta_address_->strpcls(class_Ib);
    const size_t addIa = shift / beta_size;
    const size_t addIb = shift % beta_size;
    String Ia = alfa_str(class_Ia, addIa);
    String Ib = beta_str(class_Ib, addIb);
    return Determinant(Ia, Ib);
}

const OOListElement& GenCIStringLists::get_alfa_oo_list(int class_I) const {
    // check if the key exists, if not return an empty list
    if (auto it = alfa_oo_list.find(class_I); it != alfa_oo_list.end()) {
        return it->second;
    }
    return empty_oo_list;
}

const OOListElement& GenCIStringLists::get_beta_oo_list(int class_I) const {
    // check if the key exists, if not return an empty list
    if (auto it = beta_oo_list.find(class_I); it != beta_oo_list.end()) {
        return it->second;
    }
    return empty_oo_list;
}

/**
 * Returns a vector of tuples containing the sign, I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
const VOListElement& GenCIStringLists::get_alfa_vo_list(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = alfa_vo_list.find(std::make_pair(class_I, class_J)); it != alfa_vo_list.end()) {
        return it->second;
    }
    return empty_vo_list;
}

/**
 * Returns a vector of tuples containing the sign,I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
const VOListElement& GenCIStringLists::get_beta_vo_list(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = beta_vo_list.find(std::make_pair(class_I, class_J)); it != beta_vo_list.end()) {
        return it->second;
    }
    return empty_vo_list;
}

const VVOOListElement& GenCIStringLists::get_alfa_vvoo_list(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = alfa_vvoo_list.find(std::make_pair(class_I, class_J));
        it != alfa_vvoo_list.end()) {
        return it->second;
    }
    return empty_vvoo_list;
}

const VVOOListElement& GenCIStringLists::get_beta_vvoo_list(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = beta_vvoo_list.find(std::make_pair(class_I, class_J));
        it != beta_vvoo_list.end()) {
        return it->second;
    }
    return empty_vvoo_list;
}

std::vector<H1StringSubstitution>& GenCIStringLists::get_alfa_1h_list(int class_I, size_t add_I,
                                                                      int class_J) {
    std::tuple<int, size_t, int> I_tuple(class_I, add_I, class_J);
    return alfa_1h_list[I_tuple];
}

std::vector<H1StringSubstitution>& GenCIStringLists::get_beta_1h_list(int class_I, size_t add_I,
                                                                      int class_J) {
    std::tuple<int, size_t, int> I_tuple(class_I, add_I, class_J);
    return beta_1h_list[I_tuple];
}

std::vector<H2StringSubstitution>& GenCIStringLists::get_alfa_2h_list(int h_I, size_t add_I,
                                                                      int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_2h_list[I_tuple];
}

std::vector<H2StringSubstitution>& GenCIStringLists::get_beta_2h_list(int h_I, size_t add_I,
                                                                      int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_2h_list[I_tuple];
}

std::vector<H3StringSubstitution>& GenCIStringLists::get_alfa_3h_list(int h_I, size_t add_I,
                                                                      int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_3h_list[I_tuple];
}

std::vector<H3StringSubstitution>& GenCIStringLists::get_beta_3h_list(int h_I, size_t add_I,
                                                                      int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_3h_list[I_tuple];
}

std::vector<H4StringSubstitution>& GenCIStringLists::get_alfa_4h_list(int h_I, size_t add_I,
                                                                      int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return alfa_4h_list[I_tuple];
}

std::vector<H4StringSubstitution>& GenCIStringLists::get_beta_4h_list(int h_I, size_t add_I,
                                                                      int h_J) {
    std::tuple<int, size_t, int> I_tuple(h_I, add_I, h_J);
    return beta_4h_list[I_tuple];
}

} // namespace forte