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

#include "helpers/printing.h"

#include "sparse_ci/determinant.h"
#include "sparse_ci/ci_spin_adaptation.h"
#include "string_lists.h"
#include "fci_vector.h"

#include "fci_solver.h"

namespace forte {

std::vector<std::shared_ptr<RDMs>>
FCISolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list, int max_rdm_level,
                RDMsType type) {
    if (not C_) {
        throw std::runtime_error("FCIVector is not assigned. Cannot compute RDMs.");
    }

    // handle the case of no RDMs
    if (max_rdm_level <= 0) {
        auto nroots = root_list.size();
        if (type == RDMsType::spin_dependent) {
            return std::vector<std::shared_ptr<RDMs>>(nroots,
                                                      std::make_shared<RDMsSpinDependent>());
        } else {
            return std::vector<std::shared_ptr<RDMs>>(nroots, std::make_shared<RDMsSpinFree>());
        }
    }

    std::vector<std::shared_ptr<RDMs>> refs;
    // loop over all the pairs of states
    for (const auto& [root1, root2] : root_list) {
        refs.push_back(compute_rdms_root(root1, root2, max_rdm_level, type));
    }
    return refs;
}

std::shared_ptr<RDMs> FCISolver::compute_rdms_root(size_t root_left, size_t root_right,
                                                   int max_rdm_level, RDMsType type) {
    // make sure the root is valid
    if (std::max(root_left, root_right) >= nroot_) {
        std::string error = "Cannot compute RDMs <" + std::to_string(root_left) + "| ... |" +
                            std::to_string(root_right) +
                            "> (0-based) because nroot = " + std::to_string(nroot_);
        throw std::runtime_error(error);
    }

    // here we will use C_ for the left wave function and T_ for the right wave function

    copy_state_into_fci_vector(root_left, C_);
    if (root_left != root_right) {
        copy_state_into_fci_vector(root_right, T_);
    } else {
        T_->copy(*C_);
    }

    if (print_) {
        std::string title_rdm = "Computing RDMs <" + std::to_string(root_left) + "| ... |" +
                                std::to_string(root_right) + ">";
        print_h2(title_rdm);
    }

    auto rdms = FCIVector::compute_rdms(*C_, *T_, max_rdm_level, type);

    // Optionally, test the RDMs
    // if (test_rdms_) {
    // C_->rdm_test(*C_, *T_, max_rdm_level, type, rdms);
    // }

    // // Print the NO if energy converged
    // if (print_no_ || print_ > 0) {
    //     C_->print_natural_orbitals(mo_space_info_);
    // }
    return rdms;
}

} // namespace forte
