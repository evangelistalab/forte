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
#include <pybind11/stl.h>

#include "sparse_ci/sparse_exp.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_SparseExp(py::module& m) {
    py::class_<SparseExp>(m, "SparseExp", "A class to compute the exponential of a sparse operator")
        .def(py::init<int, double>(), "maxk"_a = 19, "screen_thresh"_a = 1.0e-12)
        .def("apply_op",
             py::overload_cast<const SparseOperator&, const SparseState&, double>(
                 &SparseExp::apply_op),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_op",
             py::overload_cast<const SparseOperatorList&, const SparseState&, double>(
                 &SparseExp::apply_op),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_antiherm",
             py::overload_cast<const SparseOperator&, const SparseState&, double>(
                 &SparseExp::apply_antiherm),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_antiherm",
             py::overload_cast<const SparseOperatorList&, const SparseState&, double>(
                 &SparseExp::apply_antiherm),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0);
}

} // namespace forte
