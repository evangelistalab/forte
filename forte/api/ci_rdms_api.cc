/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see LICENSE, AUTHORS).
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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ci_rdm/ci_rdms.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {
void export_CI_RDMS(py::module& m) {
    py::class_<CI_RDMS, std::shared_ptr<CI_RDMS>>(m, "CI_RDMS")
        .def(py::init<const std::vector<int>&, const std::vector<Determinant>&,
                      std::shared_ptr<psi::Matrix>, int, int>(),
             "mo_symmetry"_a, "det_space"_a, "evecs"_a, "root1"_a, "root2"_a)
        .def(
            "compute_1rdm",
            [](CI_RDMS& rdms) {
                std::vector<double> opdm_a, opdm_b;
                rdms.compute_1rdm(opdm_a, opdm_b);
                return std::make_tuple(opdm_a, opdm_b);
            },
            "Compute the 1RDM")
        .def(
            "compute_2rdm",
            [](CI_RDMS& rdms) {
                std::vector<double> tpdm_aa, tpdm_ab, tpdm_bb;
                rdms.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);
                return std::make_tuple(tpdm_aa, tpdm_ab, tpdm_bb);
            },
            "Compute the 2RDM");
}
} // namespace forte
