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

#include "base_classes/mo_space_info.h"

namespace py = pybind11;

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_MOSpaceInfo(py::module& m) {
    py::class_<MOSpaceInfo, std::shared_ptr<MOSpaceInfo>>(m, "MOSpaceInfo")
        .def(py::init<const psi::Dimension&, const std::string&>())
        .def("dimension", &MOSpaceInfo::dimension,
             "Return a psi::Dimension object for the given space")
        .def("absolute_mo", &MOSpaceInfo::absolute_mo,
             "Return the list of the absolute index of the molecular orbitals in a space "
             "excluding "
             "the frozen core/virtual orbitals")
        .def("corr_absolute_mo", &MOSpaceInfo::corr_absolute_mo,
             "Return the list of the absolute index of the molecular orbitals in a correlated "
             "space")
        .def("contained_in_space", &MOSpaceInfo::contained_in_space,
             "Return True if a space is contained in another space")
        .def("pos_in_space", &MOSpaceInfo::pos_in_space,
             "Return the position of the orbitals in a space within another space")
        .def("relative_mo", &MOSpaceInfo::relative_mo, "Return the relative MOs")
        .def("range", &MOSpaceInfo::range, "Return a psi::Slice object for the given space")
        .def("read_from_map", &MOSpaceInfo::read_from_map,
             "Read the space info from a map {spacename -> dimension vector}")
        .def("compute_space_info", &MOSpaceInfo::compute_space_info,
             "Processing current MOSpaceInfo: calculate frozen core, count and assign orbitals")
        .def("size", &MOSpaceInfo::size, "Return the number of orbitals in a space")
        .def("nirrep", &MOSpaceInfo::nirrep, "Return the number of irreps")
        .def("symmetry", &MOSpaceInfo::symmetry, "Return the symmetry of each orbital")
        .def("irrep_labels", &MOSpaceInfo::irrep_labels, "Return the labels of the irreps")
        .def("irrep_label", &MOSpaceInfo::irrep_label, "Return the labels of the irreps")
        .def("point_group_label", &MOSpaceInfo::point_group_label,
             "Return the label of the point group")
        .def("__str__", &MOSpaceInfo::str)
        .def_property_readonly_static(
            "elementary_spaces", [](py::object) { return MOSpaceInfo::elementary_spaces(); },
            "Return the list of elementary spaces")
        .def_property_readonly_static(
            "composite_spaces", [](py::object) { return MOSpaceInfo::composite_spaces(); },
            "Return the list of composite spaces")
        .def_property_readonly_static(
            "composite_spaces_def", [](py::object) { return MOSpaceInfo::composite_spaces_def(); },
            "Return the definition of composite spaces");

    //     m.def("make_mo_space_info", &make_mo_space_info, "Make a MOSpaceInfo object");
    m.def("make_mo_space_info_from_map", &make_mo_space_info_from_map, "nmopi"_a, "point_group"_a,
          "mo_space_map"_a, "Make a MOSpaceInfo object using a dictionary");
}

} // namespace forte
