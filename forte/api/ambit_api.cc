/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see LICENSE, AUTHORS).
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
#include <pybind11/stl.h>

#include "helpers/helpers.h"

namespace py = pybind11;

namespace forte {
/// Export the ambit class
void export_ambit(py::module& m) {
    // export ambit::Tensor
    py::class_<ambit::Tensor>(m, "ambitTensor");

    m.def("ambit_doublet",
          [](ambit::Tensor A, ambit::Tensor B, const std::vector<std::string>& pattern) {
            if (pattern.size() != 3)
                throw std::runtime_error("Invalid pattern for contractions between two tensors");
            auto dims_a = A.dims();
            auto dims_b = B.dims();
            std::vector<size_t> dims_c;
            for (auto ic: pattern[2]) {
                auto found_a = pattern[0].find(ic);
                auto found_b = pattern[1].find(ic);
                if (found_a != std::string::npos and found_b == std::string::npos)
                    dims_c.push_back(dims_a[found_a]);
                else if (found_a == std::string::npos and found_b != std::string::npos)
                    dims_c.push_back(dims_b[found_b]);
                else {
                    std::stringstream ss;
                    ss << "Invalid contraction pattern: ";
                    ss << pattern[0] << "," << pattern[1] << "->" << pattern[2];
                    throw std::runtime_error(ss.str());
                }
            }
            auto C = ambit::Tensor::build(A.type(), "C", dims_c);
            C(pattern[2]) = A(pattern[0]) * B(pattern[1]);
            return C;
          });

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
