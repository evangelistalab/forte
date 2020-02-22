/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>
#include <algorithm>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"

#include "helpers/timer.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "sci.h"

namespace forte {
SelectedCIMethod::SelectedCIMethod(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info,
                                   std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : state_(state), nroot_(nroot), mo_space_info_(mo_space_info), as_ints_(as_ints),
      scf_info_(scf_info), sparse_solver_(std::make_shared<SparseCISolver>()) {}

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

    // Post-iter process
    post_iter_process();

    return 0.0;
}

size_t SelectedCIMethod::get_cycle() { return cycle_; }

SigmaVectorType SelectedCIMethod::sigma_vector_type() const { return sigma_vector_type_; }

size_t SelectedCIMethod::max_memory() const { return max_memory_; }

void SelectedCIMethod::print_wfn(DeterminantHashVec& space, psi::SharedMatrix evecs, int nroot,
                                 size_t max_dets_to_print) {
    std::string state_label;
    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet", "decatet"});

    for (int n = 0; n < nroot; ++n) {
        DeterminantHashVec tmp;
        std::vector<double> tmp_evecs;

        psi::outfile->Printf("\n\n  Most important contributions to root %3d:", n);

        size_t max_dets = std::min(max_dets_to_print, static_cast<size_t>(evecs->nrow()));
        tmp.subspace(space, evecs, tmp_evecs, max_dets, n);

        for (size_t I = 0; I < max_dets; ++I) {
            psi::outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s", I, tmp_evecs[I],
                                 tmp_evecs[I] * tmp_evecs[I], space.get_idx(tmp.get_det(I)),
                                 str(tmp.get_det(I), nact_).c_str());
        }
        auto spin = sparse_solver_->spin();
        double S2 = spin[n];
        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * std::fabs(S2)) - 1.0));
        state_label = s2_labels[std::round(S * 2.0)];
        psi::outfile->Printf("\n\n  Spin state for root %zu: S^2 = %5.6f, S = %5.3f, %s", n, S2, S,
                             state_label.c_str());
    }
}

} // namespace forte
