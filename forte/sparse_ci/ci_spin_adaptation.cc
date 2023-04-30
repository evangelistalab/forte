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

#include "ci_spin_adaptation.h"
// #include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
// #include "psi4/libmints/vector.h"

// #include "forte-def.h"
// #include "ci_reference.h"
// #include "base_classes/forte_options.h"
// #include "helpers/helpers.h"
// #include "helpers/printing.h"
// #include "helpers/timer.h"

// #include <algorithm>

namespace forte {

SpinAdapter::SpinAdapter(int na, int nb, int twoS, int twoMs)
    : na_(na), nb_(nb), twoS_(twoS), twoMs_(twoMs) {}

void SpinAdapter::csf_C_to_det_C(const std::vector<double>& csf_C, std::vector<double>& det_C) {}

void SpinAdapter::det_C_to_csf_C(const std::vector<double>& det_C, std::vector<double>& csf_C) {}

/// @brief A function to generate all the CSFs from a configuration
auto SpinAdapter::conf_to_csfs(const Configuration& conf, int twoS, int twoMs)
    -> std::vector<std::vector<double>> {
    std::vector<std::vector<double>> result;

    // number of unpaired electrons
    const auto N = conf.count_socc();

    // make the spin couplings for the CSFs
    const auto spin_couplings = make_spin_couplings(N, twoS_);

    // loop over the spin couplings
    for (const auto& spin_coupling : spin_couplings) {
        psi::outfile->Printf("spin_coupling: %s\n", str(spin_coupling, N).c_str());
    }

    return result;
}

auto make_paths(int N, int twoS) -> std::vector<String> {
    std::vector<String> paths;
    auto nalpha = (N + twoS) / 2;
    auto nbeta = (N - twoS) / 2;
    String path;
    // true = 1 = alpha, false = 0 = beta
    for (int i = 0; i < nbeta; i++)
        path[i] = false;
    for (int i = nbeta; i < N; i++)
        path[i] = true;
    /// Generate all permutations of the path
    do {
        paths.push_back(path);
    } while (std::next_permutation(path.begin(), path.begin() + N));
    return paths;
}

} // namespace forte
