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

#include "base_classes/active_space_method.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {
void export_ActiveSpaceMethod(py::module& m) {
    py::class_<ActiveSpaceMethod, std::shared_ptr<ActiveSpaceMethod>>(m, "ActiveSpaceMethod")
        .def("compute_energy", &ActiveSpaceMethod::compute_energy)
        .def("set_quiet_mode", &ActiveSpaceMethod::set_quiet_mode)
        .def("dump_wave_function", &ActiveSpaceMethod::dump_wave_function)
        .def("read_wave_function", &ActiveSpaceMethod::read_wave_function);
}

} // namespace forte
