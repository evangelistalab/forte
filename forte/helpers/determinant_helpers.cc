/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/libmints/matrix.h"

#include "integrals/active_space_integrals.h"

#include "determinant_helpers.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

namespace forte {

std::shared_ptr<psi::Matrix> make_s2_matrix(const std::vector<Determinant>& dets) {
    const size_t n = dets.size();
    auto S2 = std::make_shared<psi::Matrix>("S^2", n, n);

    auto threads = omp_get_max_threads();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (size_t I = 0; I < n; I++) {
        const Determinant& detI = dets[I];
        for (size_t J = I; J < n; J++) {
            const Determinant& detJ = dets[J];
            const double S2IJ = spin2(detI, detJ);
            S2->set(I, J, S2IJ);
            S2->set(J, I, S2IJ);
        }
    }
    return S2;
}

std::shared_ptr<psi::Matrix>
make_hamiltonian_matrix(const std::vector<Determinant>& dets,
                        std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    const size_t n = dets.size();
    auto H = std::make_shared<psi::Matrix>("H", n, n);

    // If we are running DiskDF then we need to revert to a single thread loop
    auto threads = (as_ints->get_integral_type() == DiskDF) ? 1 : omp_get_max_threads();

#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (size_t I = 0; I < n; I++) {
        const Determinant& detI = dets[I];
        for (size_t J = I; J < n; J++) {
            const Determinant& detJ = dets[J];
            double HIJ = as_ints->slater_rules(detI, detJ);
            H->set(I, J, HIJ);
            H->set(J, I, HIJ);
        }
    }
    return H;
}
} // namespace forte
