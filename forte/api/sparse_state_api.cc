/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see LICENSE, AUTHORS).
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
#include <pybind11/complex.h>
#include <pybind11/operators.h>

#include "sparse_ci/sparse_state.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_SparseState(py::module& m) {
    py::class_<SparseState, std::shared_ptr<SparseState>>(
        m, "SparseState", "A class to represent a vector of determinants")
        .def(py::init<>())
        .def(py::init<const SparseState&>())
        .def(py::init<const SparseState::container&>())
        .def(
            "items", [](const SparseState& v) { return py::make_iterator(v.begin(), v.end()); },
            py::keep_alive<0, 1>()) // Essential: keep object alive while iterator exists
        .def("str", &SparseState::str)
        .def("size", &SparseState::size)
        .def("norm", &SparseState::norm, "Compute the 2-norm of the vector")
        .def("add", &SparseState::add)
        .def("__iadd__", &SparseState::operator+=, "Add a SparseState to this SparseState")
        .def("__isub__", &SparseState::operator-=, "Subtract a SparseState from this SparseState")
        .def("__imul__", &SparseState::operator*=, "Multiply this SparseState by a scalar")
        .def("__len__", &SparseState::size)
        .def("__eq__", &SparseState::operator==)
        .def("__repr__", [](const SparseState& v) { return v.str(); })
        .def("__str__", [](const SparseState& v) { return v.str(); })
        .def("map", [](const SparseState& v) { return v.elements(); })
        .def("elements", [](const SparseState& v) { return v.elements(); })
        .def("__getitem__", [](SparseState& v, const Determinant& d) { return v[d]; })
        .def("__setitem__",
             [](SparseState& v, const Determinant& d, const sparse_scalar_t val) { v[d] = val; })
        .def("__contains__", [](SparseState& v, const Determinant& d) { return v.count(d); });

    m.def("apply_op", &apply_operator_lin, "sop"_a, "state0"_a, "screen_thresh"_a = 1.0e-12);

    m.def("apply_antiherm", &apply_operator_antiherm, "sop"_a, "state0"_a,
          "screen_thresh"_a = 1.0e-12);

    m.def("apply_number_projector", &apply_number_projector);
    m.def("get_projection", &get_projection);
    m.def("spin2_sparse", &spin2_sparse);
    m.def("overlap", &overlap);
    m.def("normalize", &normalize);
}
} // namespace forte
