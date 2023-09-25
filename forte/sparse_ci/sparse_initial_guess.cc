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

#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"

#include "sparse_initial_guess.h"

namespace forte {

std::tuple<psi::Matrix, psi::Vector, psi::Matrix>
compute_s2_transformed_hamiltonian_matrix(const std::vector<Determinant>& dets,
                                          std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    size_t num_dets = dets.size();
    // Form the S^2 operator matrix and diagonalize it
    psi::Matrix S2("S^2", num_dets, num_dets);
    for (size_t I = 0; I < num_dets; I++) {
        const Determinant& detI = dets[I];
        for (size_t J = I; J < num_dets; J++) {
            const Determinant& detJ = dets[J];
            double S2IJ = spin2(detI, detJ);
            S2.set(I, J, S2IJ);
            S2.set(J, I, S2IJ);
        }
    }
    psi::Matrix S2evecs("S^2", num_dets, num_dets);
    psi::Vector S2evals("S^2", num_dets);
    S2.diagonalize(S2evecs, S2evals);

    // Form the Hamiltonian
    psi::Matrix H("H", num_dets, num_dets);
    for (size_t I = 0; I < num_dets; I++) {
        const Determinant& detI = dets[I];
        for (size_t J = I; J < num_dets; J++) {
            const Determinant& detJ = dets[J];
            double HIJ = as_ints->slater_rules(detI, detJ);
            H.set(I, J, HIJ);
            H.set(J, I, HIJ);
        }
    }

    // Project H onto the spin-adapted subspace
    H.transform(S2evecs);
    return std::make_tuple(H, S2evals, S2evecs);
}

void find_initial_guess_det(const std::vector<Determinant>& guess_dets,
                            const std::vector<size_t>& guess_dets_pos, size_t num_guess_states,
                            const std::shared_ptr<ActiveSpaceIntegrals>& as_ints,
                            DavidsonLiuSolver& dls, int multiplicity, bool do_spin_project,
                            bool print,
                            const std::vector<std::vector<std::pair<size_t, double>>>& user_guess) {
    bool print_details_ = true;

    size_t num_guess_dets = guess_dets.size();
    psi::outfile->Printf("\n  Initial guess determinants:         %zu", guess_dets.size());

    auto [HS2full, S2evals, S2evecs] =
        compute_s2_transformed_hamiltonian_matrix(guess_dets, as_ints);

    // Convert the S2evals to a std::vector
    std::vector<double> multp_vec(num_guess_dets);
    for (size_t i = 0; i < num_guess_dets; ++i) {
        multp_vec[i] = std::sqrt(1.0 + 4.0 * S2evals.get(i));
    }

    auto groups = find_integer_groups(multp_vec, 1.0e-6);

    psi::outfile->Printf("\n\n  Classification of the initial guess solutions");
    psi::outfile->Printf("\n  ========================");
    psi::outfile->Printf("\n  Number   2S+1   Selected");
    psi::outfile->Printf("\n  ------------------------");
    std::vector<bool> is_integer_root(num_guess_dets, false);
    for (const auto& [m, start, end] : groups) {
        psi::outfile->Printf("\n %5d    %4d       %c", end - start, m,
                             m == multiplicity ? '*' : ' ');
        for (size_t i = start; i < end; ++i) {
            is_integer_root[i] = true;
        }
    }
    psi::outfile->Printf("\n  ========================");

    // count the number of false in is_integer_root
    size_t num_non_integer_roots =
        std::count(is_integer_root.begin(), is_integer_root.end(), false);

    if (num_non_integer_roots > 0) {
        psi::outfile->Printf("\n\n  The following solutions are not close to integer spin");
        for (size_t i = 0; i < num_guess_dets; ++i) {
            if (!is_integer_root[i])
                psi::outfile->Printf("\n %5d    %4.2f", i, multp_vec[i]);
        }
    }

    std::vector<std::tuple<int, double, std::vector<std::pair<size_t, double>>>> guess;

    std::vector<std::string> table;

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
        auto HS2_block = HS2full.get_block(block_slice, block_slice);
        psi::Vector HS2evals_block("HS2", block_dim);
        psi::Matrix HS2evecs_block("HS2", block_dim, block_dim);
        HS2_block->diagonalize(HS2evecs_block, HS2evals_block);
        auto S2evecs_block = S2evecs.get_block(col_slice, block_slice);

        psi::Matrix C_block("C", col_dim, block_dim);
        C_block.gemm(false, false, 1.0, S2evecs_block, HS2evecs_block, 0.0);

        for (int r = 0; r < end - start; ++r) {
            std::vector<std::pair<size_t, double>> det_C;
            for (size_t I = 0; I < num_guess_dets; I++) {
                det_C.push_back(std::make_pair(guess_dets_pos[I], C_block.get(I, r)));
            }
            double E = HS2evals_block.get(r);
            guess.push_back(std::make_tuple(m, E, det_C));
        }
        int twice_S = m - 1;
        std::string state_label = s2_label(twice_S);

        for (int r = 0; r < std::min(end - start, num_guess_states); r++) {
            table.push_back(boost::str(boost::format("    %3d  %20.12f  %.3f  %s") % r %
                                       HS2evals_block.get(r) % std::fabs(m) % state_label.c_str()));
        }
    }

    bool print_ = true;
    if (print_) {
        print_h2("Initial Guess");
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n    Root            Energy     <S^2>   Spin");
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n%s", join(table, "\n").c_str());
        psi::outfile->Printf("\n  ---------------------------------------------");
    }

    double guess_max_energy = std::numeric_limits<double>::lowest();

    // Add the user guess vectors if any are passed in
    if (user_guess.size() > 0) {
        auto b = std::make_shared<psi::Vector>("b", dls.size());
        // Use previous solution as guess
        if (print_details_)
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
        auto b = std::make_shared<psi::Vector>("b", dls.size());
        std::vector<int> guess_list;
        for (size_t g = 0; g < guess.size(); ++g) {
            if (std::get<0>(guess[g]) == multiplicity)
                guess_list.push_back(g);
        }

        if (guess_list.size() < num_guess_states) {
            throw std::runtime_error(
                "\n\n  Found " + std::to_string(guess_list.size()) +
                " guess(es) with the requested multiplicity but " +
                std::to_string(num_guess_states) +
                " were requested.\n  Increase the value of DL_DETS_PER_GUESS\n\n");
        }

        for (size_t r = 0; r < num_guess_states; ++r) {
            b->zero();
            for (auto& [pos, c] : std::get<2>(guess[guess_list[r]])) {
                b->set(pos, c);
            }
            int guess_multiplicity = std::get<0>(guess[guess_list[r]]);
            double guess_energy = std::get<1>(guess[guess_list[r]]);

            if (print_details_)
                psi::outfile->Printf("\n  Adding guess %-3d      2S+1 = %-3d  E = %.6f", r,
                                     guess_multiplicity, guess_energy);
            dls.add_guess(b);
            guess_max_energy = std::max(guess_max_energy, guess_energy);
        }
    }

    if (do_spin_project) {
        // Prepare a list of bad roots to project out and pass them to the solver
        std::vector<std::vector<std::pair<size_t, double>>> bad_roots;
        for (auto& g : guess) {
            const auto& [guess_multiplicity, guess_energy, guess_det_C] = g;
            // project out solutions with wrong multiplicity and energy lower than good guesses
            if ((guess_multiplicity != multiplicity) and (guess_energy <= guess_max_energy)) {
                psi::outfile->Printf("\n  Projecting out guess  2S+1 = %-3d  E = %.6f",
                                     guess_multiplicity, guess_energy);
                bad_roots.push_back(guess_det_C);
            }
        }
        psi::outfile->Printf("\n\n  Projecting out %zu solutions", bad_roots.size());
        dls.set_project_out(bad_roots);
    } else {
        psi::outfile->Printf("\n\n  Projecting out no solutions");
    }
}
} // namespace forte
  // std::vector<Determinant>&guess_dets,conststd::vector<size_t>&guess_dets_pos,size_tnum_guess_states,DavidsonLiuSolver&dls,intmultiplicity,booldo_spin_project)