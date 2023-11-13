/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#define FMT_HEADER_ONLY
#include "fmt/core.h"

#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "helpers/determinant_helpers.h"
#include "helpers/timer.h"

#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"

#include "sparse_initial_guess.h"

namespace forte {

std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
compute_s2_transformed_hamiltonian_matrix(const std::vector<Determinant>& dets,
                                          std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    size_t num_dets = dets.size();
    // Form the S^2 operator matrix and diagonalize it
    auto S2 = make_s2_matrix(dets);

    auto S2evecs = std::make_shared<psi::Matrix>("S^2", num_dets, num_dets);
    auto S2evals = std::make_shared<psi::Vector>("S^2", num_dets);
    S2->diagonalize(S2evecs, S2evals);

    // Form the Hamiltonian
    auto H = make_hamiltonian_matrix(dets, as_ints);

    // Project H onto the spin-adapted subspace
    H->transform(S2evecs);

    return std::make_tuple(H, S2evals, S2evecs);
}

std::pair<sparse_mat, sparse_mat>
find_initial_guess_det(const std::vector<Determinant>& guess_dets,
                       const std::vector<size_t>& guess_dets_pos, size_t num_guess_states,
                       const std::shared_ptr<ActiveSpaceIntegrals>& as_ints, int multiplicity,
                       bool do_spin_project, bool print,
                       const std::vector<std::vector<std::pair<size_t, double>>>& user_guess) {
    size_t num_guess_dets = guess_dets.size();

    if (print) {
        print_h2("Initial Guess");
        psi::outfile->Printf("\n  Initial guess determinants:         %zu", guess_dets.size());
    }

    auto [HS2full, S2evals, S2evecs] =
        compute_s2_transformed_hamiltonian_matrix(guess_dets, as_ints);

    // Convert the S2evals to a std::vector
    std::vector<double> multp_vec(num_guess_dets);
    for (size_t i = 0; i < num_guess_dets; ++i) {
        multp_vec[i] = std::sqrt(1.0 + 4.0 * S2evals->get(i));
    }

    auto groups = find_integer_groups(multp_vec, 1.0e-6);

    std::vector<bool> is_integer_root(num_guess_dets, false);
    for (const auto& [m, start, end] : groups) {
        for (size_t i = start; i < end; ++i)
            is_integer_root[i] = true;
    }

    if (print) {
        psi::outfile->Printf("\n\n  Classification of the initial guess solutions");
        psi::outfile->Printf("\n\n  Number   2S+1   Selected");
        psi::outfile->Printf("\n  ------------------------");
        for (const auto& [m, start, end] : groups) {
            psi::outfile->Printf("\n %5d    %4d       %c", end - start, m,
                                 m == multiplicity ? '*' : ' ');
        }
        psi::outfile->Printf("\n  ------------------------");
    }

    // count the number of false in is_integer_root
    size_t num_non_integer_roots =
        std::count(is_integer_root.begin(), is_integer_root.end(), false);

    if (num_non_integer_roots > 0) {
        psi::outfile->Printf("\n\n  The following guess solutions are not close to integer spin "
                             "and will be ignored");
        for (size_t i = 0; i < num_guess_dets; ++i) {
            if (!is_integer_root[i])
                psi::outfile->Printf("\n %5d    %4.2f", i, multp_vec[i]);
        }
    }

    std::map<int,
             std::tuple<std::vector<double>, std::vector<double>, std::shared_ptr<psi::Matrix>>>
        guess_info;

    double E0 = as_ints->nuclear_repulsion_energy() + as_ints->frozen_core_energy() +
                as_ints->scalar_energy();

    // Loop over the groups of roots with the same multiplicity and compute the guess vectors
    for (const auto& [m, start, end] : groups) {
        // setup dimension objects
        auto start_dim = psi::Dimension(1);
        start_dim[0] = start;
        auto end_dim = psi::Dimension(1);
        end_dim[0] = end;
        auto block_dim = psi::Dimension(1);
        block_dim[0] = end - start;
        auto col_dim = psi::Dimension(1);
        col_dim[0] = num_guess_dets;
        auto zero_dim = psi::Dimension(1);
        zero_dim[0] = 0;

        auto block_slice = psi::Slice(start_dim, end_dim);
        auto col_slice = psi::Slice(zero_dim, col_dim);
        auto HS2_block = HS2full->get_block(block_slice, block_slice);
        psi::Vector HS2evals_block("HS2", block_dim);
        psi::Matrix HS2evecs_block("HS2", block_dim, block_dim);
        HS2_block->diagonalize(HS2evecs_block, HS2evals_block);
        auto S2evecs_block = S2evecs->get_block(col_slice, block_slice);
        auto S2evals_block = S2evals->get_block(block_slice);

        std::vector<double> energies = Vector_to_vector_double(HS2evals_block);
        for (auto& e : energies) {
            e += E0;
        }
        std::vector<double> s2 = Vector_to_vector_double(S2evals_block);

        auto C_block = std::make_shared<psi::Matrix>("C", col_dim, block_dim);
        C_block->gemm(false, false, 1.0, S2evecs_block, HS2evecs_block, 0.0);

        guess_info[m] =
            std::tuple<std::vector<double>, std::vector<double>, std::shared_ptr<psi::Matrix>>(
                energies, s2, C_block);
    }

    // keep track of the maximum energy of the guess states with the correct multiplicity
    double guess_max_energy = std::numeric_limits<double>::lowest();

    // keep track of the table of guess states
    std::vector<std::pair<double, std::string>> table;
    // the guess vectors stored as a vector of pairs of determinant address and coefficient
    sparse_mat guesses;

    // Add the user guess vectors if any are passed in
    if (user_guess.size() > 0) {
        // Use previous solution as guess
        if (print)
            psi::outfile->Printf("\n  Adding %zu guess vectors passed in by the user",
                                 user_guess.size());
        for (const auto& guess_root : user_guess) {
            guesses.push_back(guess_root);
        }
    } else {
        // Use the initial guess. Here we sort out the roots of correct multiplicity
        // check if multiplicity is in the guess
        if (guess_info.find(multiplicity) == guess_info.end()) {
            throw std::runtime_error(
                "\n\n  No guess with the requested multiplicity was found.\n\n");
        }

        // grab the guess vectors with the correct multiplicity
        auto& [energies, s2, C] = guess_info[multiplicity];

        if (energies.size() < num_guess_states) {
            throw std::runtime_error(
                "\n\n  Found " + std::to_string(energies.size()) +
                " guess(es) with the requested multiplicity but " +
                std::to_string(num_guess_states) +
                " were requested.\n  Increase the value of DL_DETS_PER_GUESS\n\n");
        }

        // Add the guess vectors to list of guesses
        for (size_t r = 0; r < num_guess_states; ++r) {
            auto guess = std::vector<std::pair<size_t, double>>(num_guess_dets);
            for (size_t I = 0; I < num_guess_dets; I++) {
                guess[I] = std::make_pair(guess_dets_pos[I], C->get(I, r));
            }
            guesses.push_back(guess);

            auto guess_energy = energies[r];
            auto guess_s2 = s2[r];
            guess_max_energy = std::max(guess_max_energy, guess_energy);
            auto state_label = s2_label(multiplicity - 1);
            std::string s = fmt::format("   {:>7}  {:>3}  {:>20.12f}  {:+.6f}  added", state_label,
                                        r, guess_energy, guess_s2);
            table.push_back(std::make_pair(guess_energy, s));
        }
    }

    sparse_mat bad_roots;
    if (do_spin_project) {
        // Prepare a list of bad roots to project out and pass them to the solver
        for (auto& [mult, tup] : guess_info) {
            auto& [energies, s2, C] = tup;
            if (mult == multiplicity)
                continue;
            size_t n = energies.size();
            for (size_t r = 0; r < n; r++) {
                auto guess_energy = energies[r];
                auto guess_s2 = s2[r];
                if (guess_energy < guess_max_energy) {
                    std::vector<std::pair<size_t, double>> guess_det_C(num_guess_dets);
                    for (size_t I = 0; I < num_guess_dets; I++) {
                        guess_det_C[I] = std::make_pair(guess_dets_pos[I], C->get(I, r));
                    }
                    bad_roots.push_back(guess_det_C);

                    auto state_label = s2_label(mult - 1);

                    std::string s = fmt::format("   {:>7}  {:>3}  {:>20.12f}  {:+.6f}  removed",
                                                state_label, r, guess_energy, guess_s2);

                    table.push_back(std::make_pair(guess_energy, s));
                }
            }
        }
    }

    std::sort(table.begin(), table.end());
    std::vector<std::string> sorted_table;
    for (const auto& [e, s] : table) {
        sorted_table.push_back(s);
    }
    if (print) {
        psi::outfile->Printf("\n\n    Spin    Root           Energy        <S^2>    Status");
        psi::outfile->Printf("\n  -------------------------------------------------------");
        psi::outfile->Printf("\n%s", join(sorted_table, "\n").c_str());
        psi::outfile->Printf("\n  -------------------------------------------------------");
    }
    return std::make_pair(guesses, bad_roots);
}

sparse_mat find_initial_guess_csf(std::shared_ptr<psi::Vector> diag, size_t num_guess_states,
                                  size_t multiplicity, bool print) {
    local_timer t;

    // Get the list of most important CSFs
    std::vector<std::pair<double, size_t>> lowest_energy(
        num_guess_states, std::make_pair(std::numeric_limits<double>::max(), 0));
    size_t nfound = 0;
    const size_t ncsf = diag->dim();
    for (size_t i = 0; i < ncsf; ++i) {
        double e = diag->get(i);
        if (e < lowest_energy.back().first) {
            nfound += 1;
            lowest_energy.back() = std::make_pair(e, i);
            std::sort(lowest_energy.begin(), lowest_energy.end());
        }
    }
    // number of guess to be used
    size_t num_guess_states_found = std::min(nfound, num_guess_states);

    if (num_guess_states_found != num_guess_states) {
        psi::outfile->Printf("\n  Warning: Found %zu CSF with the requested multiplicity instead "
                             "of the number requested (%zu).\n",
                             num_guess_states_found, num_guess_states);
    }
    if (num_guess_states_found == 0) {
        throw psi::PSIEXCEPTION("\n\n  Found zero FCI guesses with the requested "
                                "multiplicity.\n\n");
    }

    std::vector<size_t> guess;
    for (const auto& [e, i] : lowest_energy) {
        guess.push_back(i);
    }

    // Set the initial guess
    sparse_mat guesses;
    auto temp = std::make_shared<psi::Vector>("temp", ncsf);
    for (size_t g = 0; g < num_guess_states_found; ++g) {
        const auto& [e, i] = lowest_energy[g];
        auto guess = {std::make_pair(i, 1.0)};
        guesses.push_back(guess);
    }

    if (print) {
        print_h2("Initial Guess");
        psi::outfile->Printf("\n  Selected %zu CSF", num_guess_states);
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n    CSF             Energy     <S^2>   Spin");
        psi::outfile->Printf("\n  ---------------------------------------------");
        double S2_target = 0.25 * (multiplicity - 1) * (multiplicity + 1);
        auto label = s2_label(multiplicity - 1);
        for (size_t g = 0; g < num_guess_states_found; ++g) {
            const auto& [e, i] = lowest_energy[g];
            std::string str = fmt::format("  {:>6} {:>20.12f}  {:.3f}  {}", i, e, S2_target, label);
            psi::outfile->Printf("\n%s", str.c_str());
        }
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n  Timing for initial guess  = %10.3f s\n", t.get());
    }
    return guesses;
}

} // namespace forte
