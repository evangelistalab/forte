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

#include "psi4/libmints/matrix.h"

#include "string_address.h"

#include "fci_vector.h"

namespace forte {

std::vector<std::tuple<double, double, size_t, size_t, size_t>>
FCIVector::max_abs_elements(size_t num_dets) {
    num_dets = std::min(num_dets, ndet_);

    std::vector<std::tuple<double, double, size_t, size_t, size_t>> dets(num_dets);

    double emin = 0.0;

    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_address_->strpcls(alfa_sym);
        size_t maxIb = beta_address_->strpcls(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                double e = std::fabs(C_ha[Ia][Ib]);
                if (e > emin) {
                    // Find where to inser this determinant
                    dets.pop_back();
                    auto it = std::find_if(
                        dets.begin(), dets.end(),
                        [&e](const std::tuple<double, double, size_t, size_t, size_t>& t) {
                            return e > std::get<0>(t);
                        });
                    dets.insert(it, std::make_tuple(e, C_ha[Ia][Ib], alfa_sym, Ia, Ib));
                    emin = std::get<0>(dets.back());
                }
            }
        }
    }
    return dets;
}

} // namespace forte
