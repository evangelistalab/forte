/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <numeric>

#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/process.h"
#include "ambit/tensor.h"

#include "cci_solver.h"
#include "base_classes/active_space_solver.h"
#include "base_classes/active_space_method.h"
#include "base_classes/reference.h"
#include "helpers/helpers.h"
#include "integrals/active_space_integrals.h"

namespace forte {

ContractedCISolver::ContractedCISolver(std::shared_ptr<ActiveSpaceSolver> as_solver,
                                       std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                       int max_rdm_level, int max_body)
    : as_solver_(as_solver), as_ints_(as_ints), max_body_(max_body), max_rdm_level_(max_rdm_level) {
}

void ContractedCISolver::compute_Heff() {
    evals_.resize(as_solver_->get_state_weights_list().size());
    evecs_.resize(as_solver_->get_state_weights_list().size());

    // get useful integrals from ActiveSpaceIntegrals
    auto oei_a = as_ints_->oei_a_vector();
    auto oei_b = as_ints_->oei_b_vector();
    auto tei_aa = as_ints_->tei_aa_vector();
    auto tei_ab = as_ints_->tei_ab_vector();
    auto tei_bb = as_ints_->tei_bb_vector();

    bool do_three_body = (max_body_ == 3 and max_rdm_level_ == 3) ? true : false;
    if (do_three_body) {
        throw psi::PSIEXCEPTION("3-body integrals not implemented in ActiveSpaceIntegrals.");
    }

    auto inner_product = [](std::vector<double>& vec1, std::vector<double>& vec2) {
        return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
    };

    auto method_vec = as_solver_->get_method_vec();
    int i_state = 0;
    for (const auto& state_weights : as_solver_->get_state_weights_list()) {
        const auto& state = state_weights.first;
        const auto& weights = state_weights.second;
        auto method = method_vec[i_state];
        int nroots = weights.size();
        std::string state_name = state.multiplicity_label() + " " + state.irrep_label();

        // form the effective Hamiltonian (assume Hermitian)
        print_h2("Building Effective Hamiltonian for " + state_name);
        psi::Matrix Heff("Heff " + state_name, nroots, nroots);

        for (int A = 0; A < nroots; ++A) {
            for (int B = A; B < nroots; ++B) {
                // just compute transition rdms of <A|sqop|B>
                std::vector<std::pair<size_t, size_t>> root_list{std::make_pair(A, B)};
                RDMs rdms = method->reference(root_list)[0];

                double H_AB = 0.0;
                H_AB += inner_product(oei_a, rdms.g1a().data());
                H_AB += inner_product(oei_b, rdms.g1b().data());

                H_AB += 0.25 * inner_product(tei_aa, rdms.g2aa().data());
                H_AB += 0.25 * inner_product(tei_bb, rdms.g2bb().data());
                H_AB += inner_product(tei_ab, rdms.g2ab().data());

                //                if (do_three_body) {
                //                    H_AB += (1.0 / 36.0) * inner_product(tei_aaa,
                //                    rdms.g3aaa().data()); H_AB += (1.0 / 36.0) *
                //                    inner_product(tei_bbb, rdms.g3bbb().data()); H_AB += 0.25
                //                    * inner_product(tei_aab, rdms.g3aab().data()); H_AB +=
                //                    0.25 * inner_product(tei_abb, rdms.g3abb().data());
                //                }

                if (A == B) {
                    H_AB += as_ints_->nuclear_repulsion_energy() + as_ints_->scalar_energy() +
                            as_ints_->frozen_core_energy();
                    Heff.set(A, B, H_AB);
                } else {
                    Heff.set(A, B, H_AB);
                    Heff.set(B, A, H_AB);
                }
            }
        }

        print_h2("Effective Hamiltonian for " + state_name);
        psi::outfile->Printf("\n");
        Heff.print();
        psi::Matrix U("Eigen Vectors of Heff for " + state_name, nroots, nroots);
        psi::Vector E("Eigen Values of Heff for " + state_name, nroots);
        Heff.diagonalize(U, E);
        U.eivprint(E);

        std::vector<double> energies(nroots);
        for (int i = 0; i < nroots; ++i) {
            energies[i] = E.get(i);
        }
        evals_[i_state] = energies;
        evecs_[i_state] = U;

        i_state++;
    }
}
} // namespace forte
