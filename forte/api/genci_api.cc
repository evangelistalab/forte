/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
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

#include "psi4/libmints/matrix.h"

#include "base_classes/mo_space_info.h"

#include "genci/genci_string_lists.h"
#include "genci/genci_vector.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_GenCIStringLists(py::module& m) {
    py::class_<GenCIStringLists, std::shared_ptr<GenCIStringLists>>(
        m, "GenCIStringLists", "A class to represent the strings of a GAS")
        .def(py::init<std::shared_ptr<MOSpaceInfo>, size_t, size_t, int, PrintLevel,
                      const std::vector<int>, const std::vector<int>>())
        .def("make_determinants", &GenCIStringLists::make_determinants,
             "Return a vector of Determinants");
}

void export_GenCIVector(py::module& m) {
    py::class_<GenCIVector, std::shared_ptr<GenCIVector>>(m, "GenCIVector",
                                                          "A class to represent a GAS vector")
        .def(py::init<std::shared_ptr<GenCIStringLists>>())
        .def("print", &GenCIVector::print, "Print the GAS vector")
        .def("size", &GenCIVector::size, "Return the size of the GAS vector")
        .def("__len__", &GenCIVector::size, "Return the size of the GAS vector")
        .def("as_state_vector", &GenCIVector::as_state_vector, "Return a SparseState object")
        .def("set_to", &GenCIVector::set_to, "Set the GAS vector to a given value");
}

} // namespace forte
