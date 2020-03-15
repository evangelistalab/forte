/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see LICENSE, AUTHORS).
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
#include "helpers/cube_file.h"

namespace py = pybind11;

namespace forte {

/// Export the CubeFile class
void export_ForteCubeFile(py::module& m) {
    py::class_<CubeFile, std::shared_ptr<CubeFile>>(m, "CubeFile")
        .def(py::init<const std::string&>())
        .def("num", &CubeFile::num, "Get the number of grid point in each direction")
        .def("min", &CubeFile::min, "Get the minimum value of a coordinate")
        .def("max", &CubeFile::max, "Get the maximum value of a coordinate")
        .def("inc", &CubeFile::inc, "Get the increment in each direction")
        .def("natoms", &CubeFile::natoms, "The number of atoms")
        .def("atom_numbers", &CubeFile::atom_numbers, "The atomic numbers")
        .def("atom_coords", &CubeFile::atom_coords, "The atomic numbers")
        .def(
            "data", [](CubeFile& cf) { return vector_to_np(cf.data(), cf.num()); }, "Get the data")
        .def("scale", &CubeFile::scale, "The atomic numbers")
        .def("add", &CubeFile::add, "The atomic numbers")
        .def("pointwise_product", &CubeFile::pointwise_product, "The atomic numbers");
}

} // namespace forte
