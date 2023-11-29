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

#include "fci_vector.h"
#include "fci_solver.h"

namespace forte {

std::vector<std::shared_ptr<RDMs>>
FCISolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                           std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                           RDMsType rdm_type) {

    if (not C_) {
        throw std::runtime_error("FCIVector is not assigned. Cannot compute RDMs.");
    }
    if (max_rdm_level > 3 || max_rdm_level < 1) {
        throw std::runtime_error("Invalid max_rdm_level, required 1 <= max_rdm_level <= 3.");
    }

    // handle the case of no RDMs
    if (max_rdm_level <= 0) {
        auto nroots = root_list.size();
        if (rdm_type == RDMsType::spin_dependent) {
            return std::vector<std::shared_ptr<RDMs>>(nroots,
                                                      std::make_shared<RDMsSpinDependent>());
        } else {
            return std::vector<std::shared_ptr<RDMs>>(nroots, std::make_shared<RDMsSpinFree>());
        }
    }

    std::vector<std::shared_ptr<RDMs>> refs;
    // loop over all the pairs of states
    for (const auto& [root1, root2] : root_list) {
        refs.push_back(
            compute_transition_rdms_root(root1, root2, method2, max_rdm_level, rdm_type));
    }
    return refs;
}

std::shared_ptr<RDMs>
FCISolver::compute_transition_rdms_root(size_t root_left, size_t root_right,
                                        std::shared_ptr<ActiveSpaceMethod> method2,
                                        int max_rdm_level, RDMsType type) {
    // make sure the root is valid
    if (root_left >= nroot_ or root_right >= method2->nroot()) {
        std::string error = "Cannot compute RDMs <" + std::to_string(root_left) + "| ... |" +
                            std::to_string(root_right) +
                            "> (0-based) because nroot_left = " + std::to_string(nroot_) +
                            " and nroot_right = " + std::to_string(method2->nroot());
        throw std::runtime_error(error);
    }

    // here we will use C_ for the left wave function and method2->T_ for the right wave function
    this->copy_state_into_fci_vector(root_left, C_);
    // cast method2 to FCISolver
    auto method2_fcisolver = std::dynamic_pointer_cast<FCISolver>(method2);
    // copy the right root onto the temporary vector C_
    method2_fcisolver->copy_state_into_fci_vector(root_right, method2_fcisolver->T_);

    if (print_ >= PrintLevel::Verbose) {
        std::string title_rdm = "Computing RDMs <" + std::to_string(root_left) + " " +
                                state().str_minimum() + "| ... |" + std::to_string(root_right) +
                                " " + method2->state().str_minimum() + ">";
        print_h2(title_rdm);
    }

    auto rdms = compute_transition_rdms(*C_, *method2_fcisolver->T_, max_rdm_level, type);

    // Optionally, test the RDMs
    if (test_rdms_) {
        FCIVector::test_rdms(*C_, *method2_fcisolver->T_, 1, type, rdms);
    }

    // // Print the NO if energy converged
    // if (print_no_ || print_ > 1) {
    //     C_->print_natural_orbitals(mo_space_info_);
    // }
    return rdms;
}

} // namespace forte
