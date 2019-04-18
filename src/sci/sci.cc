/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "sci.h"
#include "helpers/timer.h"
#include "psi4/libpsi4util/PsiOutStream.h"
namespace forte {
SelectedCIMethod::SelectedCIMethod(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info,
                                   std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : state_(state), nroot_(nroot), mo_space_info_(mo_space_info), as_ints_(as_ints),
      scf_info_(scf_info) {}

double SelectedCIMethod::compute_energy() {
    timer energy_timer("SelectedCIMethod:Energy");

    // Print the banner and starting information.
    print_info();

    // Pre-iter Preparation
    pre_iter_preparation();

    for (cycle_ = 0; cycle_ < max_cycle_; ++cycle_) {

        // Step 1. Diagonalize the Hamiltonian in the P space
        diagonalize_P_space();

        // Step 2. Find determinants in the Q space
        find_q_space();

        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        diagonalize_PQ_space();

        // Step 4. Check convergence and break if needed
        if (check_convergence())
            break;

        // Step 5. Prune the P + Q space to get an updated P space
        prune_PQ_to_P();
    }

    // Active space PT2 correction
    full_mrpt2();

    // Post-iter process
    post_iter_process();

    return 0.0;
}

size_t SelectedCIMethod::get_cycle() { return cycle_; }

} // namespace forte
