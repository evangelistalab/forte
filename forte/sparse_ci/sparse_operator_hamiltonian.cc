/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "sparse_ci/sparse_operator_hamiltonian.h"

#include "integrals/active_space_integrals.h"

namespace forte {

SparseOperator sparse_operator_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    SparseOperator H;
    size_t nmo = as_ints->nmo();

    H.add_term_from_str("[]", as_ints->nuclear_repulsion_energy() + as_ints->scalar_energy() +
                                  as_ints->frozen_core_energy());

    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            H.add_term_from_str(std::format("{}a+ {}a-", p, q), as_ints->oei_a(p, q));
            H.add_term_from_str(std::format("{}b+ {}b-", p, q), as_ints->oei_b(p, q));
        }
    }

    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = p + 1; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = r + 1; s < nmo; s++) {
                    H.add_term_from_str(std::format("{}a+ {}a+ {}a- {}a-", p, q, s, r),
                                        as_ints->tei_aa(p, q, r, s));
                    H.add_term_from_str(std::format("{}b+ {}b+ {}b- {}b-", p, q, s, r),
                                        as_ints->tei_bb(p, q, r, s));
                }
            }
        }
    }

    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = 0; s < nmo; s++) {
                    H.add_term_from_str(std::format("{}a+ {}b+ {}b- {}a-", p, q, s, r),
                                        as_ints->tei_ab(p, q, r, s));
                }
            }
        }
    }

    return H;
}

} // namespace forte
