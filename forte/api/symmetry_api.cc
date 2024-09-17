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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "helpers/symmetry.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_Symmetry(py::module& m) {
    py::class_<Symmetry>(m, "Symmetry")
        .def(py::init<std::string>())
        .def("point_group_label", &Symmetry::point_group_label,
             "Returns the label of this point group")
        .def("irrep_labels", &Symmetry::irrep_labels, "Returns a vector of irrep labels")
        .def("irrep_label", &Symmetry::irrep_label, "h"_a, "Returns the label of irrep ``h``")
        .def("irrep_label_to_index", &Symmetry::irrep_label_to_index, "label"_a,
             "Returns the index of a given irrep ``label``")
        .def("nirrep", &Symmetry::nirrep, "Returns the number of irreps")
        .def(
            "__repr__",
            [](const Symmetry& sym) { return "Symmetry(" + sym.point_group_label() + ")"; },
            "Returns a representation of this object")
        .def(
            "__str__", [](const Symmetry& sym) { return sym.point_group_label(); },
            "Returns a string representation of this object")
        .def_static("irrep_product", &Symmetry::irrep_product, "h"_a, "g"_a,
                    "Returns the product of irreps ``h`` and ``g``");
}

} // namespace forte
