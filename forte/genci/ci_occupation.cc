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

// #include <algorithm>
// #include <numeric>
// #include <cmath>
#include <string>

#define FMT_HEADER_ONLY
#include "lib/fmt/core.h"

// #include "psi4/psi4-dec.h"
// #include "psi4/libpsi4util/PsiOutStream.h"

// #include "helpers/printing.h"
// #include "helpers/helpers.h"

// #include "base_classes/mo_space_info.h"

// #include "genci_string_address.h"

// #include "genci_string_lists.h"
#include "ci_occupation.h"

namespace forte {
// // implement the code above commented out in a recursive way
// void GenCIStringLists::recursive_gas_generation(const std::vector<int>& gas_mine,
//                                               const std::vector<int>& gas_maxe, size_t gas_num,
//                                               int& n_config, std::vector<int> gas_configuration,
//                                               size_t gas_count) {
//     outfile->Printf("\n    %6d ", ++n_config);
//     for (size_t i = 0; i < 2 * gas_num; i++) {
//         outfile->Printf(" %4d", gas_configuration[i]);
//     }
//     if (gas_count == 0) {
//         int gas1_na = gas_configuration[0];
//         int gas1_nb = gas_configuration[1];
//         int gas1_max = std::max(gas1_na, gas1_nb);
//         int gas1_min = std::min(gas1_na, gas1_nb);
//         int gas1_total = gas1_na + gas1_nb;
//         if (gas1_total <= gas_maxe[0] and gas1_max <= gas_size[0] and gas1_min >= 0 and
//             gas1_total >= gas_mine[0]) {
//             outfile->Printf("\n    %6d ", ++n_config);
//             for (size_t i = 0; i < 2 * gas_num; i++) {
//                 outfile->Printf(" %4d", gas_configuration[i]);
//             }
//         }
//     } else {
//         for (int gas_na = std::max(0, gas_mine[gas_count] - gas_size[gas_count]);
//              gas_na <= std::min(gas_maxe[gas_count], gas_size[gas_count]); gas_na++) {
//             for (int gas_nb = std::max(0, gas_mine[gas_count] - gas_na);
//                  gas_nb <= std::min(gas_maxe[gas_count] - gas_na, gas_size[gas_count]); gas_nb++)
//                  {
//                 gas_configuration[2 * gas_count] = gas_na;
//                 gas_configuration[2 * gas_count + 1] = gas_nb;
//                 recursive_gas_generation(gas_mine, gas_maxe, gas_num, n_config,
//                 gas_configuration,
//                                          gas_count - 1);
//             }
//         }
//     }
// }

std::tuple<size_t, std::vector<std::array<int, 6>>, std::vector<std::array<int, 6>>,
           std::vector<std::pair<size_t, size_t>>>
get_gas_occupation(size_t na, size_t nb, const std::vector<int>& gas_min,
                   const std::vector<int>& gas_max, const std::vector<int>& gas_size) {
    std::vector<std::array<int, 6>> gas_alfa_occupations;
    std::vector<std::array<int, 6>> gas_beta_occupations;
    std::vector<std::pair<size_t, size_t>> gas_occupations;

    // The vectors of maximum number of electrons, minimum number of electrons,
    // and the number of orbitals
    std::vector<int> gas_maxe;
    std::vector<int> gas_mine;
    size_t gas_num = 0;
    for (size_t gas_count = 0; gas_count < 6; gas_count++) {
        std::string space = "GAS" + std::to_string(gas_count + 1);
        size_t gasn_size = gas_size[gas_count];
        if (gasn_size) {
            // define max_e_number to be the largest possible number of electrons in the GASn
            int max_e_number = std::min(gasn_size * 2, na + nb);
            // but if we can read its value, do so
            if (gas_max.size() > gas_count) {
                // If the defined maximum number of electrons exceed number of orbitals,
                // redefine the maximum number of electrons
                max_e_number = std::min(gas_max[gas_count], max_e_number);
            }
            gas_maxe.push_back(max_e_number);

            // define min_e_number to be the smallest possible number of electrons in the GASn
            size_t min_e_number = 0;
            // but if we can read its value, do so
            if (gas_min.size() > gas_count) {
                min_e_number = gas_min[gas_count];
            }
            gas_mine.push_back(min_e_number);
            gas_num = gas_num + 1;
        } else {
            gas_maxe.push_back(0);
            gas_mine.push_back(0);
        }
    }

    for (int gas6_na = std::max(0, gas_mine[5] - gas_size[5]);
         gas6_na <= std::min(gas_maxe[5], gas_size[5]); gas6_na++) {
        for (int gas6_nb = std::max(0, gas_mine[5] - gas6_na);
             gas6_nb <= std::min(gas_maxe[5] - gas6_na, gas_size[5]); gas6_nb++) {
            for (int gas5_na = std::max(0, gas_mine[4] - gas_size[4]);
                 gas5_na <= std::min(gas_maxe[4], gas_size[4]); gas5_na++) {
                for (int gas5_nb = std::max(0, gas_mine[4] - gas5_na);
                     gas5_nb <= std::min(gas_maxe[4] - gas5_na, gas_size[4]); gas5_nb++) {
                    for (int gas4_na = std::max(0, gas_mine[3] - gas_size[3]);
                         gas4_na <= std::min(gas_maxe[3], gas_size[3]); gas4_na++) {
                        for (int gas4_nb = std::max(0, gas_mine[3] - gas4_na);
                             gas4_nb <= std::min(gas_maxe[3] - gas4_na, gas_size[3]); gas4_nb++) {
                            for (int gas3_na = std::max(0, gas_mine[2] - gas_size[2]);
                                 gas3_na <= std::min(gas_maxe[2], gas_size[2]); gas3_na++) {
                                for (int gas3_nb = std::max(0, gas_mine[2] - gas3_na);
                                     gas3_nb <= std::min(gas_maxe[2] - gas3_na, gas_size[2]);
                                     gas3_nb++) {
                                    for (int gas2_na = std::max(0, gas_mine[1] - gas_size[1]);
                                         gas2_na <= std::min(gas_maxe[1], gas_size[1]); gas2_na++) {
                                        for (int gas2_nb = std::max(0, gas_mine[1] - gas2_na);
                                             gas2_nb <=
                                             std::min(gas_maxe[1] - gas2_na, gas_size[1]);
                                             gas2_nb++) {
                                            int gas1_na = na - gas2_na - gas3_na - gas4_na -
                                                          gas5_na - gas6_na;
                                            int gas1_nb = nb - gas2_nb - gas3_nb - gas4_nb -
                                                          gas5_nb - gas6_nb;
                                            int gas1_max = std::max(gas1_na, gas1_nb);
                                            int gas1_min = std::min(gas1_na, gas1_nb);
                                            int gas1_total = gas1_na + gas1_nb;
                                            if (gas1_total <= gas_maxe[0] and
                                                gas1_max <= gas_size[0] and gas1_min >= 0 and
                                                gas1_total >= gas_mine[0]) {
                                                std::array<int, 6> alfa_occ = {gas1_na, gas2_na,
                                                                               gas3_na, gas4_na,
                                                                               gas5_na, gas6_na};
                                                std::array<int, 6> beta_occ = {gas1_nb, gas2_nb,
                                                                               gas3_nb, gas4_nb,
                                                                               gas5_nb, gas6_nb};
                                                // check if alfa_occ is contained in
                                                // gas_alfa_occupations, if not, add it and
                                                // grab its index, otherwise grab its index
                                                size_t alfa_index;
                                                if (auto alfa_it = std::find(
                                                        gas_alfa_occupations.begin(),
                                                        gas_alfa_occupations.end(), alfa_occ);
                                                    alfa_it == gas_alfa_occupations.end()) {
                                                    gas_alfa_occupations.push_back(alfa_occ);
                                                    alfa_index = gas_alfa_occupations.size() - 1;
                                                } else {
                                                    alfa_index = std::distance(
                                                        gas_alfa_occupations.begin(), alfa_it);
                                                }
                                                // check if beta_occ is contained in
                                                // gas_beta_occupations, if not, add it and
                                                // grab its index, otherwise grab its index
                                                size_t beta_index;
                                                if (auto beta_it = std::find(
                                                        gas_beta_occupations.begin(),
                                                        gas_beta_occupations.end(), beta_occ);
                                                    beta_it == gas_beta_occupations.end()) {
                                                    gas_beta_occupations.push_back(beta_occ);
                                                    beta_index = gas_beta_occupations.size() - 1;
                                                } else {
                                                    beta_index = std::distance(
                                                        gas_beta_occupations.begin(), beta_it);
                                                }
                                                gas_occupations.push_back(
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
    return std::make_tuple(gas_num, gas_alfa_occupations, gas_beta_occupations, gas_occupations);
}

std::vector<std::array<int, 6>>
generate_1h_occupations(const std::vector<std::array<int, 6>>& gas_occupations) {
    std::vector<std::array<int, 6>> one_hole_occ;
    // Loop over all the GAS alpha/beta occupations
    for (const auto& gas_occupation : gas_occupations) {
        for (size_t n = 0; n < 6; n++) {
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

std::string occupation_table(size_t num_spaces,
                             const std::vector<std::array<int, 6>>& alfa_occupation,
                             const std::vector<std::array<int, 6>>& beta_occupation,
                             const std::vector<std::pair<size_t, size_t>>& occupation_pairs) {
    std::string s;
    s += "\n    Config.";
    for (size_t i = 0; i < num_spaces; i++) {
        s += fmt::format("   Space {:1}", i + 1);
    }
    s += "\n            ";
    for (size_t i = 0; i < num_spaces; i++) {
        s += fmt::format("   α    β ");
    }
    int ndash = 7 + 10 * num_spaces;
    std::string dash(ndash, '-');
    s += fmt::format("\n    {}", dash);
    for (size_t num_conf{0}; const auto& [aocc_idx, bocc_idx] : occupation_pairs) {
        num_conf += 1;
        const auto& aocc = alfa_occupation[aocc_idx];
        const auto& bocc = beta_occupation[bocc_idx];
        s += fmt::format("\n    {:6d} ", num_conf);
        for (size_t i = 0; i < num_spaces; i++) {
            s += fmt::format(" {:4d} {:4d}", aocc[i], bocc[i]);
        }
    }
    return s;
}

} // namespace forte
