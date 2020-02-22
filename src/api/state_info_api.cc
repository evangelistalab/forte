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

#include "base_classes/state_info.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {
/// export StateInfo
void export_StateInfo(py::module& m) {
    py::class_<StateInfo, std::shared_ptr<StateInfo>>(m, "StateInfo")
        .def(py::init<int, int, int, int, int>(), "na"_a, "nb"_a, "multiplicity"_a, "twice_ms"_a,
             "irrep"_a)
        .def("__str__", &StateInfo::str, "String representation of StateInfo")
        .def("__repr__", &StateInfo::str, "String representation of StateInfo")
        .def("na", &StateInfo::na, "Number of alpha electrons")
        .def("nb", &StateInfo::nb, "Number of beta electrons")
        .def("multiplicity", &StateInfo::multiplicity, "Multiplicity")
        .def("twice_ms", &StateInfo::twice_ms, "Twice of Ms")
        .def("irrep", &StateInfo::irrep, "Irreducible representation")
        .def("multiplicity_label", &StateInfo::multiplicity_label, "Multiplicity label")
        .def("irrep_label", &StateInfo::irrep_label, "Symbol for irreducible representation")
        .def("__eq__", [](const StateInfo& a, const StateInfo& b) { return a == b; })
        .def("__lt__", [](const StateInfo& a, const StateInfo& b) { return a < b; });
}
} // namespace forte
