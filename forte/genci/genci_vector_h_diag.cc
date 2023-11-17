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

#include <cmath>

#include "psi4/libmints/matrix.h"

#include "genci_string_address.h"

#include "genci_vector.h"

namespace forte {

std::vector<std::tuple<double, double, int, int, size_t, size_t>>
GenCIVector::max_abs_elements(size_t num_dets) {
    num_dets = std::min(num_dets, ndet_);

    std::vector<std::tuple<double, double, int, int, size_t, size_t>> dets(num_dets);

    double abs_c_min = 0.0;

    const_for_each_element([&](const size_t& /*n*/, const int& class_Ia, const int& class_Ib,
                               const size_t& Ia, const size_t& Ib, const double& c) {
        auto abs_c = std::fabs(c);
        if (abs_c > abs_c_min) {
            // Find where to insert this determinant
            dets.pop_back();
            auto it =
                std::find_if(dets.begin(), dets.end(),
                             [&](const std::tuple<double, double, int, int, size_t, size_t>& t) {
                                 return abs_c > std::get<0>(t);
                             });
            dets.insert(it, std::make_tuple(abs_c, c, class_Ia, class_Ib, Ia, Ib));
            abs_c_min = std::get<0>(dets.back());
        }
    });
    return dets;
}

} // namespace forte
