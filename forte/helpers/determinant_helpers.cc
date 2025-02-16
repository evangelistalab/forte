/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER,
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

#include <algorithm>

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

std::vector<std::vector<String>> make_strings(int n, int k, size_t nirrep,
                                              const std::vector<int>& mo_symmetry) {
    // n is the number of orbitals
    // k is the number of electrons
    std::vector<std::vector<String>> strings(nirrep);
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        String I;
        const auto I_begin = I.begin();
        const auto I_end = I.begin() + n;
        // Generate the string 00000001111111
        //                      {n-k}  { k }
        I.zero();
        for (int i = std::max(0, n - k); i < n; ++i)
            I[i] = true; // 1
        do {
            int sym{0};
            for (int i = 0; i < n; ++i) {
                if (I[i])
                    sym ^= mo_symmetry[i];
            }
            strings[sym].push_back(I);
        } while (std::next_permutation(I_begin, I_end));
    }
    return strings;
}

std::vector<Determinant> make_hilbert_space(size_t nmo, size_t na, size_t nb, Determinant ref, int truncation,
                                            size_t nirrep, std::vector<int> mo_symmetry, int symmetry) {
    std::vector<Determinant> dets;
    if (mo_symmetry.size() != nmo) {
        mo_symmetry = std::vector<int>(nmo, 0);
    }
    // find the maximum value in mo_symmetry and check that it is less than nirrep
    int max_sym = *std::max_element(mo_symmetry.begin(), mo_symmetry.end());
    if (max_sym >= static_cast<int>(nirrep)) {
        throw std::runtime_error("The symmetry of the MOs is greater than the number of irreps.");
    }
    // implement other sensible checks, like making sure that symmetry is less than nirrep and na <=
    // nmo, nb <= nmo
    if (symmetry >= static_cast<int>(nirrep)) {
        throw std::runtime_error(
            "The symmetry of the determinants is greater than the number of irreps.");
    }
    if (na > nmo) {
        throw std::runtime_error(
            "The number of alpha electrons is greater than the number of MOs.");
    }
    if (nb > nmo) {
        throw std::runtime_error("The number of beta electrons is greater than the number of MOs.");
    }
    if (truncation < 0 || truncation > static_cast<int>(na + nb)) {
        throw std::runtime_error("The truncation level must an integer between 0 and na + nb.");
    }

    auto strings_a = make_strings(nmo, na, nirrep, mo_symmetry);
    auto strings_b = make_strings(nmo, nb, nirrep, mo_symmetry);
    for (size_t ha = 0; ha < nirrep; ha++) {
        int hb = symmetry ^ ha;
        for (const auto& Ia : strings_a[ha]) {
            Determinant det;
            det.set_alfa_str(Ia);
            for (const auto& Ib : strings_b[hb]) {
                det.set_beta_str(Ib);
                if (det.fast_a_xor_b_count(ref) / 2 <= truncation) {
                    dets.push_back(det);
                } 
            }
        }
    }
    return dets;
}

std::vector<Determinant> make_hilbert_space(size_t nmo, size_t na, size_t nb, size_t nirrep,
                                            std::vector<int> mo_symmetry, int symmetry) {
    std::vector<Determinant> dets;
    if (mo_symmetry.size() != nmo) {
        mo_symmetry = std::vector<int>(nmo, 0);
    }
    // find the maximum value in mo_symmetry and check that it is less than nirrep
    int max_sym = *std::max_element(mo_symmetry.begin(), mo_symmetry.end());
    if (max_sym >= static_cast<int>(nirrep)) {
        throw std::runtime_error("The symmetry of the MOs is greater than the number of irreps.");
    }
    // implement other sensible checks, like making sure that symmetry is less than nirrep and na <=
    // nmo, nb <= nmo
    if (symmetry >= static_cast<int>(nirrep)) {
        throw std::runtime_error(
            "The symmetry of the determinants is greater than the number of irreps.");
    }
    if (na > nmo) {
        throw std::runtime_error(
            "The number of alpha electrons is greater than the number of MOs.");
    }
    if (nb > nmo) {
        throw std::runtime_error("The number of beta electrons is greater than the number of MOs.");
    }

    auto strings_a = make_strings(nmo, na, nirrep, mo_symmetry);
    auto strings_b = make_strings(nmo, nb, nirrep, mo_symmetry);
    for (size_t ha = 0; ha < nirrep; ha++) {
        int hb = symmetry ^ ha;
        for (const auto& Ia : strings_a[ha]) {
            for (const auto& Ib : strings_b[hb]) {
                dets.push_back(Determinant(Ia, Ib));
            }
        }
    }
    return dets;
}

} // namespace forte
