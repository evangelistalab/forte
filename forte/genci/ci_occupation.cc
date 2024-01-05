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

#include <string>

#define FMT_HEADER_ONLY
#include "lib/fmt/core.h"

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/printing.h"

#include "ci_occupation.h"

namespace forte {

void add_occupation_pair(const occupation_t& alfa_occ, const occupation_t& beta_occ,
                         std::vector<occupation_t>& alfa_occupations,
                         std::vector<occupation_t>& beta_occupations,
                         std::vector<std::pair<size_t, size_t>>& occupation_pairs) {
    size_t alfa_index;
    if (auto alfa_it = std::find(alfa_occupations.begin(), alfa_occupations.end(), alfa_occ);
        alfa_it == alfa_occupations.end()) {
        alfa_occupations.push_back(alfa_occ);
        alfa_index = alfa_occupations.size() - 1;
    } else {
        alfa_index = std::distance(alfa_occupations.begin(), alfa_it);
    }
    size_t beta_index;
    if (auto beta_it = std::find(beta_occupations.begin(), beta_occupations.end(), beta_occ);
        beta_it == beta_occupations.end()) {
        beta_occupations.push_back(beta_occ);
        beta_index = beta_occupations.size() - 1;
    } else {
        beta_index = std::distance(beta_occupations.begin(), beta_it);
    }
    occupation_pairs.push_back(std::make_pair(alfa_index, beta_index));
}

void recursive_ormas_generation(int na, int nb, const occupation_t& min_occ,
                                const occupation_t& max_occ, const occupation_t& space_size,
                                size_t num_spaces, size_t space_count, occupation_t alfa_occ,
                                occupation_t beta_occ, std::vector<occupation_t>& alfa_occupations,
                                std::vector<occupation_t>& beta_occupations,
                                std::vector<std::pair<size_t, size_t>>& occupation_pairs) {
    // if we reach the last space
    if (space_count == num_spaces - 1) {
        int occ_max = std::max(na, nb);
        int occ_min = std::min(na, nb);
        int occ_total = na + nb;
        // ensure that the number of electrons in the last space is within the bounds
        if (occ_total <= max_occ[space_count] and occ_total >= min_occ[space_count] and
            occ_max <= space_size[space_count] and occ_min >= 0) {
            // add the last space to the occupation
            alfa_occ[space_count] = na;
            beta_occ[space_count] = nb;
            add_occupation_pair(alfa_occ, beta_occ, alfa_occupations, beta_occupations,
                                occupation_pairs);
        }
    } else {
        // the minimum number of alpha electrons (na_min) is the total number of electrons minus
        // the maximum number that can filled in the beta levels of this space. To ensure that the
        // number is positive, we take the maximum of this number and zero.
        int na_min = std::max(min_occ[space_count] - space_size[space_count], 0);
        // the maximum number of alpha electrons (na_max) is the total number of electrons, or
        // at most the size of this space
        int na_max = std::min(max_occ[space_count], space_size[space_count]);
        for (int space_na = na_min; space_na <= na_max; space_na++) {
            // the minimum number of beta electrons (nb_min) is the total number of electrons
            // minus those allocated to the alpha levels of this space. To ensure that the number is
            // positive, we take the maximum of this number and zero.
            int nb_min = std::max(min_occ[space_count] - space_na, 0);
            // the maximum number of beta electrons (nb_max) is the total number of electrons
            // minus the number of alpha electrons, or at most the size of this space
            int nb_max = std::min(max_occ[space_count] - space_na, space_size[space_count]);
            for (int space_nb = nb_min; space_nb <= nb_max; space_nb++) {
                int occ_total = space_na + space_nb;
                // this next check might be useless
                if (occ_total <= max_occ[space_count] and occ_total >= min_occ[space_count]) {
                    // add the current space to the occupation
                    alfa_occ[space_count] = space_na;
                    beta_occ[space_count] = space_nb;
                    recursive_ormas_generation(na - space_na, nb - space_nb, min_occ, max_occ,
                                               space_size, num_spaces, space_count + 1, alfa_occ,
                                               beta_occ, alfa_occupations, beta_occupations,
                                               occupation_pairs);
                }
            }
        }
    }
}

std::tuple<size_t, std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
generate_ormas_occupations(size_t na, size_t nb, const occupation_t& min_occ,
                           const occupation_t& max_occ, const occupation_t& space_size,
                           size_t num_spaces) {
    occupation_t alfa_occ{};
    occupation_t beta_occ{};
    std::vector<occupation_t> alfa_occupations;
    std::vector<occupation_t> beta_occupations;
    std::vector<std::pair<size_t, size_t>> occupation_pairs;
    recursive_ormas_generation(na, nb, min_occ, max_occ, space_size, num_spaces, 0, alfa_occ,
                               beta_occ, alfa_occupations, beta_occupations, occupation_pairs);
    return std::make_tuple(num_spaces, alfa_occupations, beta_occupations, occupation_pairs);
}

std::tuple<size_t, std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
get_ci_occupation_patterns(size_t na, size_t nb, const std::vector<int>& occ_min,
                           const std::vector<int>& occ_max, const std::vector<int>& gas_size) {
    // The vectors of maximum number of electrons, minimum number of electrons,
    // and the number of orbitals
    occupation_t max_occ{};
    occupation_t min_occ{};
    occupation_t space_size{};

    size_t num_spaces = 0;
    for (size_t n = 0; n < max_active_spaces; n++) {
        size_t space_n_size = n < gas_size.size() ? gas_size[n] : 0;
        if (space_n_size) {
            // find the largest possible number of electrons in this space
            int max_e_number = std::min(space_n_size * 2, na + nb);
            // but if we can read its value, do so and make sure this number is within the bounds
            if (occ_max.size() > n) {
                max_e_number = std::min(occ_max[n], max_e_number);
            }
            // find the smallest possible number of electrons in this space
            size_t min_e_number = 0;
            // but if we can read its value, do so
            if (occ_min.size() > n) {
                min_e_number = occ_min[n];
            }
            max_occ[n] = max_e_number;
            min_occ[n] = min_e_number;
            space_size[n] = space_n_size;
            num_spaces += 1;
        }
    }
    // the exception that breaks the rule
    if (num_spaces == 0) {
        std::vector<occupation_t> empty_occ(1, occupation_t{});
        std::vector<std::pair<size_t, size_t>> empty_pairs(1, {0, 0});
        return std::make_tuple(1, empty_occ, empty_occ, empty_pairs);
    }
    return generate_ormas_occupations(na, nb, min_occ, max_occ, space_size, num_spaces);
}

std::tuple<std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
generate_gas_occupations(int na, int nb, const occupation_t& min_occ, const occupation_t& max_occ,
                         const occupation_t& gas_size) {
    std::vector<occupation_t> gas_alfa_occupations;
    std::vector<occupation_t> gas_beta_occupations;
    std::vector<std::pair<size_t, size_t>> gas_occupations;

    for (int gas6_na = std::max(0, min_occ[5] - gas_size[5]);
         gas6_na <= std::min(max_occ[5], gas_size[5]); gas6_na++) {
        for (int gas6_nb = std::max(0, min_occ[5] - gas6_na);
             gas6_nb <= std::min(max_occ[5] - gas6_na, gas_size[5]); gas6_nb++) {
            for (int gas5_na = std::max(0, min_occ[4] - gas_size[4]);
                 gas5_na <= std::min(max_occ[4], gas_size[4]); gas5_na++) {
                for (int gas5_nb = std::max(0, min_occ[4] - gas5_na);
                     gas5_nb <= std::min(max_occ[4] - gas5_na, gas_size[4]); gas5_nb++) {
                    for (int gas4_na = std::max(0, min_occ[3] - gas_size[3]);
                         gas4_na <= std::min(max_occ[3], gas_size[3]); gas4_na++) {
                        for (int gas4_nb = std::max(0, min_occ[3] - gas4_na);
                             gas4_nb <= std::min(max_occ[3] - gas4_na, gas_size[3]); gas4_nb++) {
                            for (int gas3_na = std::max(0, min_occ[2] - gas_size[2]);
                                 gas3_na <= std::min(max_occ[2], gas_size[2]); gas3_na++) {
                                for (int gas3_nb = std::max(0, min_occ[2] - gas3_na);
                                     gas3_nb <= std::min(max_occ[2] - gas3_na, gas_size[2]);
                                     gas3_nb++) {
                                    for (int gas2_na = std::max(0, min_occ[1] - gas_size[1]);
                                         gas2_na <= std::min(max_occ[1], gas_size[1]); gas2_na++) {
                                        for (int gas2_nb = std::max(0, min_occ[1] - gas2_na);
                                             gas2_nb <= std::min(max_occ[1] - gas2_na, gas_size[1]);
                                             gas2_nb++) {
                                            int gas1_na = na - gas2_na - gas3_na - gas4_na -
                                                          gas5_na - gas6_na;
                                            int gas1_nb = nb - gas2_nb - gas3_nb - gas4_nb -
                                                          gas5_nb - gas6_nb;
                                            int gas1_max = std::max(gas1_na, gas1_nb);
                                            int gas1_min = std::min(gas1_na, gas1_nb);
                                            int gas1_total = gas1_na + gas1_nb;
                                            if (gas1_total <= max_occ[0] and
                                                gas1_max <= gas_size[0] and gas1_min >= 0 and
                                                gas1_total >= min_occ[0]) {
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
    return std::make_tuple(gas_alfa_occupations, gas_beta_occupations, gas_occupations);
}

std::tuple<size_t, std::vector<std::array<int, 6>>, std::vector<std::array<int, 6>>,
           std::vector<std::pair<size_t, size_t>>>
get_gas_occupation(size_t na, size_t nb, const std::vector<int>& gas_min,
                   const std::vector<int>& gas_max, const std::vector<int>& gas_size) {
    std::vector<std::array<int, 6>> gas_alfa_occupations;
    std::vector<std::array<int, 6>> gas_beta_occupations;
    std::vector<std::pair<size_t, size_t>> gas_occupations;

    // The vectors of maximum number of electrons, minimum number of electrons,
    // and the number of orbitals
    std::vector<int> gas_max_el(6, 0);
    std::vector<int> gas_min_el(6, 0);
    size_t num_gas_spaces = 0;
    for (size_t n = 0; n < 6; n++) {
        std::string space = "GAS" + std::to_string(n + 1);
        size_t gasn_size = n < gas_size.size() ? gas_size[n] : 0;
        if (gasn_size) {
            // define max_e_number to be the largest possible number of electrons in the
            // GASn
            int max_e_number = std::min(gasn_size * 2, na + nb);
            // but if we can read its value, do so
            if (gas_max.size() > n) {
                // If the defined maximum number of electrons exceed number of orbitals,
                // redefine the maximum number of electrons
                max_e_number = std::min(gas_max[n], max_e_number);
            }
            gas_max_el[n] = max_e_number;

            // define min_e_number to be the smallest possible number of electrons in the
            // GASn
            size_t min_e_number = 0;
            // but if we can read its value, do so
            if (gas_min.size() > n) {
                min_e_number = gas_min[n];
            }
            gas_min_el[n] = min_e_number;
            num_gas_spaces += 1;
        }
    }

    for (int gas6_na = std::max(0, gas_min_el[5] - gas_size[5]);
         gas6_na <= std::min(gas_max_el[5], gas_size[5]); gas6_na++) {
        for (int gas6_nb = std::max(0, gas_min_el[5] - gas6_na);
             gas6_nb <= std::min(gas_max_el[5] - gas6_na, gas_size[5]); gas6_nb++) {
            for (int gas5_na = std::max(0, gas_min_el[4] - gas_size[4]);
                 gas5_na <= std::min(gas_max_el[4], gas_size[4]); gas5_na++) {
                for (int gas5_nb = std::max(0, gas_min_el[4] - gas5_na);
                     gas5_nb <= std::min(gas_max_el[4] - gas5_na, gas_size[4]); gas5_nb++) {
                    for (int gas4_na = std::max(0, gas_min_el[3] - gas_size[3]);
                         gas4_na <= std::min(gas_max_el[3], gas_size[3]); gas4_na++) {
                        for (int gas4_nb = std::max(0, gas_min_el[3] - gas4_na);
                             gas4_nb <= std::min(gas_max_el[3] - gas4_na, gas_size[3]); gas4_nb++) {
                            for (int gas3_na = std::max(0, gas_min_el[2] - gas_size[2]);
                                 gas3_na <= std::min(gas_max_el[2], gas_size[2]); gas3_na++) {
                                for (int gas3_nb = std::max(0, gas_min_el[2] - gas3_na);
                                     gas3_nb <= std::min(gas_max_el[2] - gas3_na, gas_size[2]);
                                     gas3_nb++) {
                                    for (int gas2_na = std::max(0, gas_min_el[1] - gas_size[1]);
                                         gas2_na <= std::min(gas_max_el[1], gas_size[1]);
                                         gas2_na++) {
                                        for (int gas2_nb = std::max(0, gas_min_el[1] - gas2_na);
                                             gas2_nb <=
                                             std::min(gas_max_el[1] - gas2_na, gas_size[1]);
                                             gas2_nb++) {
                                            int gas1_na = na - gas2_na - gas3_na - gas4_na -
                                                          gas5_na - gas6_na;
                                            int gas1_nb = nb - gas2_nb - gas3_nb - gas4_nb -
                                                          gas5_nb - gas6_nb;
                                            int gas1_max = std::max(gas1_na, gas1_nb);
                                            int gas1_min = std::min(gas1_na, gas1_nb);
                                            int gas1_total = gas1_na + gas1_nb;
                                            if (gas1_total <= gas_max_el[0] and
                                                gas1_max <= gas_size[0] and gas1_min >= 0 and
                                                gas1_total >= gas_min_el[0]) {
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
    return std::make_tuple(num_gas_spaces, gas_alfa_occupations, gas_beta_occupations,
                           gas_occupations);
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
