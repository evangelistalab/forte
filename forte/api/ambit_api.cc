/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <pybind11/pybind11.h>

#include "helpers/helpers.h"

namespace py = pybind11;

namespace forte {
/// Export the ambit class
void export_ambit(py::module& m) {
    // export ambit::Tensor
    py::class_<ambit::Tensor>(m, "ambitTensor");

    m.def(
        "test_ambit_3d",
        []() {
            size_t n1 = 3;
            size_t n2 = 4;
            size_t n3 = 5;
            auto t = ambit::Tensor::build(ambit::CoreTensor, "L1a_sa", {n1, n2, n3});
            int sum = 0;
            t.iterate([&](const std::vector<size_t>&, double& value) {
                value = sum;
                sum += 1;
            });
            return ambit_to_np(t);
        },
        "Test a 3d ambit tensor");
}
} // namespace forte
