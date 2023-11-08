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
#include <cmath>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/printing.h"
#include "helpers/helpers.h"

#include "base_classes/mo_space_info.h"

#include "gas_string_address.h"

#include "gas_string_lists.h"

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

GASStringLists::GASStringLists(std::shared_ptr<MOSpaceInfo> mo_space_info, size_t na, size_t nb,
                               int symmetry, int print, const std::vector<int> gas_min,
                               const std::vector<int> gas_max)
    : symmetry_(symmetry), nirrep_(mo_space_info->nirrep()), ncmo_(mo_space_info->size("ACTIVE")),
      na_(na), nb_(nb), print_(print), gas_min_(gas_min), gas_max_(gas_max) {
    startup(mo_space_info);
}

void GASStringLists::startup(std::shared_ptr<MOSpaceInfo> mo_space_info) {

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

    get_gas_occupation();

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

    // this object is used to compute the class of a string (a generalization of the irrep)
    string_class_ =
        std::make_shared<StringClass>(symmetry_, cmopi_int, gas_mos_, gas_alfa_occupations_,
                                      gas_beta_occupations_, gas_occupations_);

    // Build the string lists and the addresser
    {
        local_timer t;
        alfa_strings_ = make_gas_strings(gas_size_, gas_alfa_occupations_);
        beta_strings_ = make_gas_strings(gas_size_, gas_beta_occupations_);

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

    if (na_ >= 1) {
        auto alfa_1h_strings = make_gas_strings(gas_size_, gas_alfa_1h_occupations_);
        alfa_address_1h_ = std::make_shared<StringAddress>(gas_size_, na_ - 1, alfa_1h_strings);
    }
    if (nb_ >= 1) {
        auto beta_1h_strings = make_gas_strings(gas_size_, gas_beta_1h_occupations_);
        beta_address_1h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 1, beta_1h_strings);
    }

    if (na_ >= 2) {
        auto alfa_2h_strings = make_gas_strings(gas_size_, gas_alfa_2h_occupations_);
        alfa_address_2h_ = std::make_shared<StringAddress>(gas_size_, na_ - 2, alfa_2h_strings);
    }
    if (nb_ >= 2) {
        auto beta_2h_strings = make_gas_strings(gas_size_, gas_beta_2h_occupations_);
        beta_address_2h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 2, beta_2h_strings);
    }
    if (na_ >= 3) {
        auto alfa_3h_strings = make_gas_strings(gas_size_, gas_alfa_3h_occupations_);
        alfa_address_3h_ = std::make_shared<StringAddress>(gas_size_, na_ - 3, alfa_3h_strings);
    }
    if (nb_ >= 3) {
        auto beta_3h_strings = make_gas_strings(gas_size_, gas_beta_3h_occupations_);
        beta_address_3h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 3, beta_3h_strings);
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
        pair_list_ = make_pair_list();
        nn_list_timer += t.get();
    }
    {
        local_timer t;
        make_vo_list(alfa_strings_, alfa_address_, alfa_address_, alfa_vo_list, ncmo_);
        make_vo_list(beta_strings_, beta_address_, beta_address_, beta_vo_list, ncmo_);
        vo_list_timer += t.get();
    }
    {
        local_timer t;
        make_oo_list(alfa_strings_, alfa_address_, alfa_oo_list);
        make_oo_list(beta_strings_, beta_address_, beta_oo_list);
        oo_list_timer += t.get();
    }
    {
        local_timer t;
        make_1h_list(alfa_strings_, alfa_address_, alfa_address_1h_, alfa_1h_list);
        make_1h_list(beta_strings_, beta_address_, beta_address_1h_, beta_1h_list);
        h1_list_timer += t.get();
    }
    {
        local_timer t;
        make_2h_list(alfa_strings_, alfa_address_, alfa_address_2h_, alfa_2h_list);
        make_2h_list(beta_strings_, beta_address_, beta_address_2h_, beta_2h_list);
        h2_list_timer += t.get();
    }
    {
        local_timer t;
        make_3h_list(alfa_strings_, alfa_address_, alfa_address_3h_, alfa_3h_list);
        make_3h_list(beta_strings_, beta_address_, beta_address_3h_, beta_3h_list);
        h3_list_timer += t.get();
    }
    {
        local_timer t;
        make_vvoo_list(alfa_strings_, alfa_address_, alfa_vvoo_list);
        make_vvoo_list(beta_strings_, beta_address_, beta_vvoo_list);
        vvoo_list_timer += t.get();
    }

    double total_time = str_list_timer + nn_list_timer + vo_list_timer + oo_list_timer +
                        vvoo_list_timer + vovo_list_timer;

    if (print_) {
        table_printer printer;
        printer.add_int_data({{"number of alpha electrons", na_},
                              {"number of beta electrons", nb_},
                              {"number of alpha strings", nas_},
                              {"number of beta strings", nbs_}});
        printer.add_timing_data({{"timing for strings", str_list_timer},
                                 {"timing for NN strings", nn_list_timer},
                                 {"timing for VO strings", vo_list_timer},
                                 {"timing for OO strings", oo_list_timer},
                                 {"timing for VVOO strings", vvoo_list_timer},
                                 {"timing for 1-hole strings", h1_list_timer},
                                 {"timing for 2-hole strings", h2_list_timer},
                                 {"timing for 3-hole strings", h3_list_timer},
                                 {"total timing", total_time}});

        std::string table = printer.get_table("String Lists");
        outfile->Printf("%s", table.c_str());
    }
}

// implement the code above commented out in a recursive way
void GASStringLists::recursive_gas_generation(const std::vector<int>& gas_mine,
                                              const std::vector<int>& gas_maxe, size_t gas_num,
                                              int& n_config, std::vector<int> gas_configuration,
                                              size_t gas_count) {
    outfile->Printf("\n    %6d ", ++n_config);
    for (size_t i = 0; i < 2 * gas_num; i++) {
        outfile->Printf(" %4d", gas_configuration[i]);
    }
    if (gas_count == 0) {
        int gas1_na = gas_configuration[0];
        int gas1_nb = gas_configuration[1];
        int gas1_max = std::max(gas1_na, gas1_nb);
        int gas1_min = std::min(gas1_na, gas1_nb);
        int gas1_total = gas1_na + gas1_nb;
        if (gas1_total <= gas_maxe[0] and gas1_max <= gas_size_[0] and gas1_min >= 0 and
            gas1_total >= gas_mine[0]) {
            outfile->Printf("\n    %6d ", ++n_config);
            for (size_t i = 0; i < 2 * gas_num; i++) {
                outfile->Printf(" %4d", gas_configuration[i]);
            }
        }
    } else {
        for (int gas_na = std::max(0, gas_mine[gas_count] - gas_size_[gas_count]);
             gas_na <= std::min(gas_maxe[gas_count], gas_size_[gas_count]); gas_na++) {
            for (int gas_nb = std::max(0, gas_mine[gas_count] - gas_na);
                 gas_nb <= std::min(gas_maxe[gas_count] - gas_na, gas_size_[gas_count]); gas_nb++) {
                gas_configuration[2 * gas_count] = gas_na;
                gas_configuration[2 * gas_count + 1] = gas_nb;
                recursive_gas_generation(gas_mine, gas_maxe, gas_num, n_config, gas_configuration,
                                         gas_count - 1);
            }
        }
    }
}

void GASStringLists::get_gas_occupation() {
    // gas_electrons_.clear();

    print_h2("Number of Electrons in GAS");
    outfile->Printf("\n    GAS  MAX  MIN");
    outfile->Printf("\n    -------------");

    // The vectors of maximum number of electrons, minimum number of electrons,
    // and the number of orbitals
    std::vector<int> gas_maxe;
    std::vector<int> gas_mine;
    size_t gas_num_ = 0;
    for (size_t gas_count = 0; gas_count < 6; gas_count++) {
        std::string space = "GAS" + std::to_string(gas_count + 1);
        size_t gasn_size = gas_size_[gas_count];
        if (gasn_size) {
            outfile->Printf("\n    %3d", gas_count + 1);
            // define max_e_number to be the largest possible number of electrons in the GASn
            int max_e_number = std::min(gasn_size * 2, na_ + nb_);
            // but if we can read its value, do so
            if (gas_max_.size() > gas_count) {
                // If the defined maximum number of electrons exceed number of orbitals,
                // redefine the maximum number of electrons
                max_e_number = std::min(gas_max_[gas_count], max_e_number);
            }
            gas_maxe.push_back(max_e_number);

            // define min_e_number to be the smallest possible number of electrons in the GASn
            size_t min_e_number = 0;
            // but if we can read its value, do so
            if (gas_min_.size() > gas_count) {
                min_e_number = gas_min_[gas_count];
            }
            gas_mine.push_back(min_e_number);

            outfile->Printf(" %4d %4d", max_e_number, min_e_number);
            gas_num_ = gas_num_ + 1;
        } else {
            gas_maxe.push_back(0);
            gas_mine.push_back(0);
        }
    }
    outfile->Printf("\n    -------------");

    // std::vector<int> gas_configuration(12, 0);
    // recursive_gas_generation(gas_mine, gas_maxe, gas_num_, n_config, gas_configuration, 6);

    for (int gas6_na = std::max(0, gas_mine[5] - gas_size_[5]);
         gas6_na <= std::min(gas_maxe[5], gas_size_[5]); gas6_na++) {
        for (int gas6_nb = std::max(0, gas_mine[5] - gas6_na);
             gas6_nb <= std::min(gas_maxe[5] - gas6_na, gas_size_[5]); gas6_nb++) {
            for (int gas5_na = std::max(0, gas_mine[4] - gas_size_[4]);
                 gas5_na <= std::min(gas_maxe[4], gas_size_[4]); gas5_na++) {
                for (int gas5_nb = std::max(0, gas_mine[4] - gas5_na);
                     gas5_nb <= std::min(gas_maxe[4] - gas5_na, gas_size_[4]); gas5_nb++) {
                    for (int gas4_na = std::max(0, gas_mine[3] - gas_size_[3]);
                         gas4_na <= std::min(gas_maxe[3], gas_size_[3]); gas4_na++) {
                        for (int gas4_nb = std::max(0, gas_mine[3] - gas4_na);
                             gas4_nb <= std::min(gas_maxe[3] - gas4_na, gas_size_[3]); gas4_nb++) {
                            for (int gas3_na = std::max(0, gas_mine[2] - gas_size_[2]);
                                 gas3_na <= std::min(gas_maxe[2], gas_size_[2]); gas3_na++) {
                                for (int gas3_nb = std::max(0, gas_mine[2] - gas3_na);
                                     gas3_nb <= std::min(gas_maxe[2] - gas3_na, gas_size_[2]);
                                     gas3_nb++) {
                                    for (int gas2_na = std::max(0, gas_mine[1] - gas_size_[1]);
                                         gas2_na <= std::min(gas_maxe[1], gas_size_[1]);
                                         gas2_na++) {
                                        for (int gas2_nb = std::max(0, gas_mine[1] - gas2_na);
                                             gas2_nb <=
                                             std::min(gas_maxe[1] - gas2_na, gas_size_[1]);
                                             gas2_nb++) {
                                            int gas1_na = na_ - gas2_na - gas3_na - gas4_na -
                                                          gas5_na - gas6_na;
                                            int gas1_nb = nb_ - gas2_nb - gas3_nb - gas4_nb -
                                                          gas5_nb - gas6_nb;
                                            int gas1_max = std::max(gas1_na, gas1_nb);
                                            int gas1_min = std::min(gas1_na, gas1_nb);
                                            int gas1_total = gas1_na + gas1_nb;
                                            if (gas1_total <= gas_maxe[0] and
                                                gas1_max <= gas_size_[0] and gas1_min >= 0 and
                                                gas1_total >= gas_mine[0]) {
                                                std::array<int, 6> alfa_occ = {gas1_na, gas2_na,
                                                                               gas3_na, gas4_na,
                                                                               gas5_na, gas6_na};
                                                std::array<int, 6> beta_occ = {gas1_nb, gas2_nb,
                                                                               gas3_nb, gas4_nb,
                                                                               gas5_nb, gas6_nb};
                                                // check if alfa_occ is contained in
                                                // gas_alfa_occupations_, if not, add it and
                                                // grab its index, otherwise grab its index
                                                size_t alfa_index;
                                                if (auto alfa_it = std::find(
                                                        gas_alfa_occupations_.begin(),
                                                        gas_alfa_occupations_.end(), alfa_occ);
                                                    alfa_it == gas_alfa_occupations_.end()) {
                                                    gas_alfa_occupations_.push_back(alfa_occ);
                                                    alfa_index = gas_alfa_occupations_.size() - 1;
                                                } else {
                                                    alfa_index = std::distance(
                                                        gas_alfa_occupations_.begin(), alfa_it);
                                                }
                                                // check if beta_occ is contained in
                                                // gas_beta_occupations_, if not, add it and
                                                // grab its index, otherwise grab its index
                                                size_t beta_index;
                                                if (auto beta_it = std::find(
                                                        gas_beta_occupations_.begin(),
                                                        gas_beta_occupations_.end(), beta_occ);
                                                    beta_it == gas_beta_occupations_.end()) {
                                                    gas_beta_occupations_.push_back(beta_occ);
                                                    beta_index = gas_beta_occupations_.size() - 1;
                                                } else {
                                                    beta_index = std::distance(
                                                        gas_beta_occupations_.begin(), beta_it);
                                                }
                                                gas_occupations_.push_back(
                                                    std::make_pair(alfa_index, beta_index));
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
    }

    // print_h2("Possible Electron Occupations of alpha/beta strings in GAS");

    // outfile->Printf("\n    Alfa Occ.");

    // std::vector<std::string> gas_alfa_name = {"GAS1_A", "GAS2_A", "GAS3_A",
    //                                           "GAS4_A", "GAS5_A", "GAS6_A"};
    // for (size_t i = 0; i < gas_num_; i++) {
    //     std::string name = gas_alfa_name.at(i).substr(3, 3);
    //     outfile->Printf("  %s", name.c_str());
    //     ndash += 5;
    // }
    // for (int n{0}; const auto& aocc : gas_alfa_occupations_) {
    //     outfile->Printf("\n    %6d ", ++n);
    //     for (size_t i = 0; i < gas_num_; i++) {
    //         outfile->Printf(" %4d", aocc[i]);
    //     }
    // }
    // outfile->Printf("\n    Beta Occ.");
    // for (size_t i = 0; i < gas_num_; i++) {
    //     std::string name = gas_alfa_name.at(i).substr(3, 3);
    //     outfile->Printf("  %s", name.c_str());
    //     ndash += 5;
    // }
    // for (int n{0}; const auto& bocc : gas_beta_occupations_) {
    //     outfile->Printf("\n    %6d ", ++n);
    //     for (size_t i = 0; i < gas_num_; i++) {
    //         outfile->Printf(" %4d", bocc[i]);
    //     }
    // }

    print_h2("Possible Electron Occupations in GAS");

    int ndash = 7;
    outfile->Printf("\n    Config.");
    std::vector<std::string> gas_electron_name = {"GAS1_A", "GAS1_B", "GAS2_A", "GAS2_B",
                                                  "GAS3_A", "GAS3_B", "GAS4_A", "GAS4_B",
                                                  "GAS5_A", "GAS5_B", "GAS6_A", "GAS6_B"};
    for (size_t i = 0; i < 2 * gas_num_; i++) {
        std::string name = gas_electron_name.at(i).substr(3, 3);
        outfile->Printf("  %s", name.c_str());
        ndash += 5;
    }
    outfile->Printf("  Alfa Conf.  Beta Conf.");

    std::string dash(ndash, '-');
    outfile->Printf("\n    %s", dash.c_str());

    int n_config = 0;
    n_config = 0;
    for (const auto& [aocc_idx, bocc_idx] : gas_occupations_) {
        const auto& aocc = gas_alfa_occupations_[aocc_idx];
        const auto& bocc = gas_beta_occupations_[bocc_idx];
        outfile->Printf("\n    %6d ", ++n_config);
        for (size_t i = 0; i < gas_num_; i++) {
            outfile->Printf(" %4d", aocc[i]);
            outfile->Printf(" %4d", bocc[i]);
        }
        outfile->Printf(" %4d %4d", aocc_idx, bocc_idx);
    }
}

std::vector<std::array<int, 6>>
GASStringLists::generate_1h_occupations(const std::vector<std::array<int, 6>>& gas_occupations) {
    std::vector<std::array<int, 6>> one_hole_occ;
    // Loop over all the GAS alpha/beta occupations
    for (const auto& gas_occupation : gas_occupations) {
        for (size_t n = 0; n < ngas_spaces_; n++) {
            // Check if we can remove an electron from the GAS
            if (gas_occupation[n] >= 1) {
                // If so, remove it and store the new occupation
                std::array<int, 6> new_occ = gas_occupation;
                new_occ[n] -= 1;
                one_hole_occ.push_back(new_occ);
            }
        }
    }
    return one_hole_occ;
}

/**
 * Generate all the pairs p > q with pq in pq_sym
 * these are stored as pair<int,int> in pair_list[pq_sym][pairpi]
 */
PairList GASStringLists::make_pair_list() {
    PairList list(nirrep_);
    // Loop over irreps of the pair pq
    for (size_t pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
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
    return list;
}

StringList
GASStringLists::make_gas_strings(const std::vector<int>& gas_size,
                                 const std::vector<std::array<int, 6>>& gas_occupations) {
    auto list = StringList();
    for (const auto& gas_occupation : gas_occupations) {
        make_gas_strings_with_occupation(list, gas_size, gas_occupation);
    }
    return list;
}

void GASStringLists::make_gas_strings_with_occupation(StringList& list,
                                                      const std::vector<int>& gas_size,
                                                      const std::array<int, 6>& gas_occupation) {

    // Something to keep in mind: Here we use ACTIVE as a composite index, which means that we
    // will group the orbitals first by symmetry and then by space. For example, if we have the
    // following GAS: GAS1 = [A1 A1 A1 | A2 | B1 | B2 B2] GAS2 = [A1 | | B1 | B2 ] then the
    // composite space ACTIVE = GAS1 + GAS2 will be: ACTIVE = [A1 A1 A1 A1 | A2 | B1 B1 | B2 B2
    // B2]
    //           G1 G1 G1 G2   G1   G1 G1   G1 G1 G2

    // container to store the strings that generate a give gas space
    std::vector<std::vector<String>> gas_space_string(ngas_spaces_, std::vector<String>{});
    std::vector<std::vector<String>> full_strings(nirrep_, std::vector<String>{});
    // print gas_occupation
    // psi::outfile->Printf("\n  GAS occupation: %s", container_to_string(gas_occupation).c_str());

    // enumerate all the possible strings in each GAS space
    for (size_t n = 0; n < ngas_spaces_; n++) {
        String I;
        auto gas_norb = gas_size[n];
        auto gas_ne = gas_occupation[n];
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
                        J[gas_mos_[n][i]] = true;
                }
                gas_space_string[n].push_back(J);
                // psi::outfile->Printf("\n    %s", str(J, ncmo_).c_str());
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
        size_t sym_I = string_class_->symmetry(I);
        // psi::outfile->Printf("\n    %s", str(I, ncmo_).c_str());
        full_strings[sym_I].push_back(I);
    }

    for (size_t h = 0; h < nirrep_; h++) {
        list.push_back(full_strings[h]);
    }
}

std::vector<Determinant> GASStringLists::make_determinants() const {
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

size_t GASStringLists::determinant_address(const Determinant& d) const {
    const auto Ia = d.get_alfa_bits();
    const auto Ib = d.get_beta_bits();

    const auto& [addIa, class_Ia] = alfa_address_->address_and_class(Ia);
    const auto& [addIb, class_Ib] = beta_address_->address_and_class(Ib);
    size_t addI = addIa * beta_address_->strpcls(class_Ib) + addIb;
    int n = string_class_->block_index(class_Ia, class_Ib);
    addI += detpblk_offset_[n];
    return addI;
}

Determinant GASStringLists::determinant(size_t address) const {
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

} // namespace forte