#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "boost/format.hpp"

#include "helpers/helpers.h"
#include "helpers/iterative_solvers.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "helpers/determinant_helpers.h"

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

void find_initial_guess_det(const std::vector<Determinant>& guess_dets,
                            const std::vector<size_t>& guess_dets_pos, size_t num_guess_states,
                            const std::shared_ptr<ActiveSpaceIntegrals>& as_ints,
                            DavidsonLiuSolver& dls, int multiplicity, bool do_spin_project,
                            bool print,
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

    // Add the user guess vectors if any are passed in
    if (user_guess.size() > 0) {
        auto b = std::make_shared<psi::Vector>("b", dls.size());
        // Use previous solution as guess
        if (print)
            psi::outfile->Printf("\n  Adding %zu guess vectors passed in by the user",
                                 user_guess.size());
        for (const auto& guess_root : user_guess) {
            b->zero();
            for (auto& [pos, c] : guess_root) {
                b->set(pos, c);
            }
            double norm = sqrt(1.0 / b->norm());
            b->scale(norm);
            dls.add_guess(b);
        }
    } else {
        // Use the initial guess. Here we sort out the roots of correct multiplicity
        // check if multiplicity is in the guess
        if (guess_info.find(multiplicity) == guess_info.end()) {
            throw std::runtime_error(
                "\n\n  No guess with the requested multiplicity was found.\n\n");
        }
        auto& [energies, s2, C] = guess_info[multiplicity];

        if (energies.size() < num_guess_states) {
            throw std::runtime_error(
                "\n\n  Found " + std::to_string(energies.size()) +
                " guess(es) with the requested multiplicity but " +
                std::to_string(num_guess_states) +
                " were requested.\n  Increase the value of DL_DETS_PER_GUESS\n\n");
        }

        auto b = std::make_shared<psi::Vector>("b", dls.size());

        for (size_t r = 0; r < num_guess_states; ++r) {
            b->zero();
            for (size_t I = 0; I < num_guess_dets; I++) {
                b->set(guess_dets_pos[I], C->get(I, r));
            }
            dls.add_guess(b);

            auto guess_energy = energies[r];
            auto guess_s2 = s2[r];
            guess_max_energy = std::max(guess_max_energy, guess_energy);
            auto state_label = s2_label(multiplicity - 1);

            auto s = boost::str(boost::format("   %7s  %3d  %20.12f  %+.6f  added") %
                                state_label.c_str() % r % guess_energy % guess_s2);

            table.push_back(std::make_pair(guess_energy, s));
        }
    }

    if (do_spin_project) {
        // Prepare a list of bad roots to project out and pass them to the solver
        std::vector<std::vector<std::pair<size_t, double>>> bad_roots;
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

                    auto s = boost::str(boost::format("   %7s  %3d  %20.12f  %+.6f  removed") %
                                        state_label.c_str() % r % guess_energy % guess_s2);

                    table.push_back(std::make_pair(guess_energy, s));
                }
            }
        }
        dls.set_project_out(bad_roots);
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
}
} // namespace forte
  // std::vector<Determinant>&guess_dets,conststd::vector<size_t>&guess_dets_pos,size_tnum_guess_states,DavidsonLiuSolver&dls,intmultiplicity,booldo_spin_project)