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

#include "boost/format.hpp"

#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant_functions.hpp"
#include "sparse_ci/ci_spin_adaptation.h"
#include "helpers/iterative_solvers.h"

#include "fci_solver.h"
#include "fci_vector.h"
#include "string_lists.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "string_address.h"

#include "sparse_ci/sparse_initial_guess.h"

namespace forte {

std::vector<Determinant> FCISolver::initial_guess_generate_dets(FCIVector& diag,
                                                                size_t num_guess_states) {
    size_t ndets = diag.size();
    // number of guess to be used must be at most as large as the number of determinants
    size_t num_guess_dets = std::min(num_guess_states * ndets_per_guess_, ndets);

    // Get the list of most important determinants in the format
    // std::vector<std::tuple<double, size_t, size_t, size_t>>
    // this list has size exactly num_guess_dets
    auto dets = diag.min_elements(num_guess_dets);

    std::vector<Determinant> guess_dets;

    // Build the full determinants
    size_t nact = active_mo_.size();
    for (const auto& [e, h, add_Ia, add_Ib] : dets) {
        auto Ia = lists_->alfa_str(h, add_Ia);
        auto Ib = lists_->beta_str(h ^ symmetry_, add_Ib);
        guess_dets.emplace_back(Ia, Ib);
    }

    // Make sure that the spin space is complete
    enforce_spin_completeness(guess_dets, nact);
    if (guess_dets.size() > num_guess_dets) {
        if (print_ > 0) {
            psi::outfile->Printf("\n  Initial guess space is incomplete.\n  Adding "
                                 "%d determinant(s).",
                                 guess_dets.size() - num_guess_dets);
        }
    }
    return guess_dets;
}

void FCISolver::initial_guess_det(FCIVector& diag, size_t num_guess_states,
                                  std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                  DavidsonLiuSolver& dls, std::shared_ptr<psi::Vector> temp) {
    local_timer t;

    auto guess_dets = initial_guess_generate_dets(diag, num_guess_states);
    size_t num_guess_dets = guess_dets.size();
    psi::outfile->Printf("\n  Initial guess determinants:         %zu", num_guess_dets);

    auto [H, S2evals, S2evecs] = compute_s2_transformed_hamiltonian_matrix(guess_dets, fci_ints);

    double nuclear_repulsion_energy = fci_ints->nuclear_repulsion_energy();
    double scalar_energy = fci_ints->scalar_energy();
    // psi::Matrix H("H", num_guess_dets, num_guess_dets);
    psi::Matrix evecs("Evecs", num_guess_dets, num_guess_dets);
    psi::Vector evals("Evals", num_guess_dets);

    for (size_t I = 0; I < num_guess_dets; ++I) {
        for (size_t J = I; J < num_guess_dets; ++J) {
            double HIJ = fci_ints->slater_rules(guess_dets[I], guess_dets[J]);
            if (I == J)
                HIJ += scalar_energy;
            H.set(I, J, HIJ);
            H.set(J, I, HIJ);
        }
    }

    H.diagonalize(evecs, evals);

    std::vector<std::pair<int, std::vector<std::tuple<size_t, size_t, size_t, double>>>> guess;

    std::vector<std::string> table;

    for (size_t r = 0; r < num_guess_dets; ++r) {
        double energy = evals.get(r) + nuclear_repulsion_energy;
        double norm = 0.0;
        double S2 = 0.0;
        for (size_t I = 0; I < num_guess_dets; ++I) {
            for (size_t J = 0; J < num_guess_dets; ++J) {
                const double S2IJ = ::forte::spin2(guess_dets[I], guess_dets[J]);
                S2 += evecs.get(I, r) * evecs.get(J, r) * S2IJ;
            }
            norm += std::pow(evecs.get(I, r), 2.0);
        }
        S2 /= norm;
        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        int SS = std::round(S * 2.0);
        int state_multp = SS + 1;
        std::string state_label = s2_label(SS);
        table.push_back(boost::str(boost::format("    %3d  %20.12f  %.3f  %s") % r % energy %
                                   std::fabs(S2) % state_label.c_str()));
        // Save states of the desired multiplicity
        std::vector<std::tuple<size_t, size_t, size_t, double>> solution;
        for (size_t I = 0; I < num_guess_dets; ++I) {
            auto det = guess_dets[I];
            String Ia = det.get_alfa_bits();
            String Ib = det.get_beta_bits();
            const auto h = lists_->alfa_address()->sym(Ia);
            const auto add_Ia = lists_->alfa_address()->add(Ia);
            const auto add_Ib = lists_->beta_address()->add(Ib);
            solution.push_back(std::make_tuple(h, add_Ia, add_Ib, evecs.get(I, r)));
        }
        guess.push_back(std::make_pair(state_multp, solution));
    }
    if (print_) {
        print_h2("FCI Initial Guess");
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n    Root            Energy     <S^2>   Spin");
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n%s", join(table, "\n").c_str());
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n  Timing for initial guess  = %10.3f s\n", t.get());
    }

    // Find the guess with the correct multiplicity
    std::vector<int> guess_list;
    for (size_t g = 0; g < guess.size(); ++g) {
        if (guess[g].first == state().multiplicity())
            guess_list.push_back(g);
    }

    // number of guess to be used
    size_t nguess = std::min(guess_list.size(), num_guess_states);

    if (nguess == 0) {
        throw psi::PSIEXCEPTION("\n\n  Found zero FCI guesses with the requested "
                                "multiplicity.\n\n");
    }

    for (size_t n = 0; n < nguess; ++n) {
        C_->set(guess[guess_list[n]].second);
        C_->copy_to(temp);
        dls.add_guess(temp);
    }

    // Prepare a list of bad roots to project out and pass them to the solver
    std::vector<std::vector<std::pair<size_t, double>>> bad_roots;
    int gr = 0;
    std::vector<std::string> bad_roots_vec;
    for (auto& g : guess) {
        if (g.first != state().multiplicity()) {
            bad_roots_vec.push_back(std::to_string(gr));
            std::vector<std::pair<size_t, double>> bad_root;
            C_->set(g.second);
            C_->copy_to(temp);
            for (size_t I = 0, maxI = C_->size(); I < maxI; ++I) {
                if (std::fabs(temp->get(I)) > 1.0e-12) {
                    bad_root.push_back(std::make_pair(I, temp->get(I)));
                }
            }
            bad_roots.push_back(bad_root);
        }
        gr += 1;
    }
    dls.set_project_out(bad_roots);

    if (print_ > 0) {
        psi::outfile->Printf("\n  Projecting out guess roots: [%s]", join(bad_roots_vec).c_str());
    }
}

void FCISolver::initial_guess_csf(std::shared_ptr<psi::Vector> diag, size_t num_guess_states,
                                  DavidsonLiuSolver& dls, std::shared_ptr<psi::Vector> temp) {
    local_timer t;

    // Get the list of most important CSFs
    std::vector<std::pair<double, size_t>> lowest_energy(
        num_guess_states, std::make_pair(std::numeric_limits<double>::max(), 0));
    size_t nfound = 0;
    const size_t ncsf = spin_adapter_->ncsf();
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
    for (size_t g = 0; g < num_guess_states_found; ++g) {
        const auto& [e, i] = lowest_energy[g];
        temp->zero();
        temp->set(i, 1.0);
        dls.add_guess(temp);
    }

    if (print_) {
        print_h2("FCI Initial Guess");
        psi::outfile->Printf("\n  Selected %zu CSF", num_guess_states);
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n    CSF             Energy     <S^2>   Spin");
        psi::outfile->Printf("\n  ---------------------------------------------");
        double S2_target = 0.25 * (state().multiplicity() - 1) * (state().multiplicity() + 1);
        auto label = s2_label(state().multiplicity() - 1);
        for (size_t g = 0; g < num_guess_states_found; ++g) {
            const auto& [e, i] = lowest_energy[g];
            auto str =
                boost::str(boost::format("  %6d %20.12f  %.3f  %s") % i % e % S2_target % label);
            psi::outfile->Printf("\n%s", str.c_str());
        }
        psi::outfile->Printf("\n  ---------------------------------------------");
        psi::outfile->Printf("\n  Timing for initial guess  = %10.3f s\n", t.get());
    }
}

std::shared_ptr<psi::Vector>
FCISolver::form_Hdiag_csf(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                          std::shared_ptr<SpinAdapter> spin_adapter) {
    auto Hdiag_csf = std::make_shared<psi::Vector>(spin_adapter->ncsf());
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    double E0 = fci_ints->nuclear_repulsion_energy() + fci_ints->scalar_energy();
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    if (spin_adapt_full_preconditioner_) {
        for (size_t i = 0, imax = spin_adapter->ncsf(); i < imax; ++i) {
            double energy = E0;
            int I = 0;
            for (const auto& [det_add_I, c_I] : spin_adapter_->csf(i)) {
                int J = 0;
                for (const auto& [det_add_J, c_J] : spin_adapter_->csf(i)) {
                    if (I == J) {
                        energy += c_I * c_J * fci_ints->energy(dets_[det_add_I]);
                    } else if (I < J) {
                        if (c_I * c_J != 0.0) {
                            energy += 2.0 * c_I * c_J *
                                      fci_ints->slater_rules(dets_[det_add_I], dets_[det_add_J]);
                        }
                    }
                    J++;
                }
                I++;
            }
            Hdiag_csf->set(i, energy);
        }
    } else {
        for (size_t i = 0, imax = spin_adapter->ncsf(); i < imax; ++i) {
            double energy = E0;
            for (const auto& [det_add_I, c_I] : spin_adapter_->csf(i)) {
                energy += c_I * c_I * fci_ints->energy(dets_[det_add_I]);
            }
            Hdiag_csf->set(i, energy);
        }
    }
    return Hdiag_csf;
}
} // namespace forte
