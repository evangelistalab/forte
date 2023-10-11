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

std::pair<sparse_mat, sparse_mat>
FCISolver::initial_guess_det(FCIVector& diag, size_t num_guess_states,
                             std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    auto guess_dets = initial_guess_generate_dets(diag, num_guess_states);
    size_t num_guess_dets = guess_dets.size();

    std::vector<size_t> guess_dets_pos(num_guess_dets);
    for (size_t I = 0; I < num_guess_dets; ++I) {
        guess_dets_pos[I] = lists()->determinant_address(guess_dets[I]);
    }

    // here we use a standard guess procedure
    return find_initial_guess_det(guess_dets, guess_dets_pos, num_guess_states, fci_ints,
                                  state().multiplicity(), true, print_,
                                  std::vector<std::vector<std::pair<size_t, double>>>());
}

sparse_mat FCISolver::initial_guess_csf(std::shared_ptr<psi::Vector> diag,
                                        size_t num_guess_states) {
    return find_initial_guess_csf(diag, num_guess_states, state().multiplicity(), print_);
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
