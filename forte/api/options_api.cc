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

#include "psi4/liboptions/liboptions.h"

#include "base_classes/forte_options.h"

namespace py = pybind11;

namespace forte {

/// Export the ForteOptions class
void export_ForteOptions(py::module& m) {
    py::class_<ForteOptions, std::shared_ptr<ForteOptions>>(m, "ForteOptions")
        .def(py::init<>())
        .def(py::init<const ForteOptions&>())
        .def("set_dict", &ForteOptions::set_dict, "Set the options dictionary")
        .def("dict", &ForteOptions::dict, "Returns the options dictionary")
        .def("set_group", &ForteOptions::set_group, "Set the options group")
        .def("is_none", &ForteOptions::is_none, "Is this variable defined?")
        .def("exists", &ForteOptions::exists, "Does this option exist?")
        .def("add_bool", &ForteOptions::add_bool, "Add a boolean option")
        .def("add_int", &ForteOptions::add_int, "Add an integer option")
        .def("add_double", &ForteOptions::add_double, "Add a double option")
        .def("add_str",
             (void(ForteOptions::*)(const std::string&, py::object, const std::string&)) &
                 ForteOptions::add_str,
             "Add a string option")
        .def("add_str",
             (void(ForteOptions::*)(const std::string&, py::object, const std::vector<std::string>&,
                                    const std::string&)) &
                 ForteOptions::add_str,
             "Add a string option")
        .def("add_int_list", &ForteOptions::add_int_array, "Add a list of integers option")
        .def("add_double_list", &ForteOptions::add_double_array, "Add a list of doubles option")
        .def("add_list", &ForteOptions::add_array, "Add an array option for general elements")
        .def("get_bool", &ForteOptions::get_bool, "Get a boolean option")
        .def("get_int", &ForteOptions::get_int, "Get an integer option")
        .def("get_double", &ForteOptions::get_double, "Get a double option")
        .def("get_str", &ForteOptions::get_str, "Get a string option")
        .def("get_int_list", &ForteOptions::get_int_list, "Get a list of integers option")
        .def("get_double_list", &ForteOptions::get_double_list,
             "Get a vector of doubles (py::float) option")
        .def("get_list", &ForteOptions::get_gen_list, "Get a general list")
        .def("set_bool", &ForteOptions::set_bool, "Set a boolean option")
        .def("set_int", &ForteOptions::set_int, "Set an integer option")
        .def("set_double", &ForteOptions::set_double, "Set a double option")
        .def("set_str", &ForteOptions::set_str, "Set a string option")
        .def("set_int_list", &ForteOptions::set_int_list, "Set a vector of integers option")
        .def("set_double_list", &ForteOptions::set_double_list,
             "Set a vector of doubles (py::float) option")
        .def("set_list", &ForteOptions::set_gen_list,
             "Set a vector of python objects (py::object) option")
        .def("push_options_to_psi4", &ForteOptions::push_options_to_psi4,
             "Push the options list to Psi4")
        .def("get_options_from_psi4", &ForteOptions::get_options_from_psi4,
             "Read the value of options from Psi4")
        .def("set_from_dict", &ForteOptions::set_from_dict,
             "Set options from a dictionary `dict` of labels -> values")
        .def("generate_documentation", &ForteOptions::generate_documentation,
             "Generate documentation from the options list")
        .def("__str__", &ForteOptions::str, "Returns a string represenation of this object")
        .def("__repr__", &ForteOptions::str, "Returns a string represenation of this object");
}

} // namespace forte
