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

#include "psi4/liboptions/liboptions.h"

#include "base_classes/forte_options.h"

namespace py = pybind11;

namespace forte {

/// Export the ForteOptions class
void export_ForteOptions(py::module& m) {
    py::class_<ForteOptions, std::shared_ptr<ForteOptions>>(m, "ForteOptions")
        .def(py::init<>())
        .def("set_dict", &ForteOptions::set_dict)
        .def("dict", &ForteOptions::dict)
        .def("set_group", &ForteOptions::set_group, "Set the options group")
        .def("add_bool", &ForteOptions::add_bool, "Add a boolean option")
        .def("add_int", &ForteOptions::add_int, "Add an integer option")
        .def("add_double", &ForteOptions::add_double, "Add a double option")
        .def("add_str",
             (void (ForteOptions::*)(const std::string&, const std::string&, const std::string&)) &
                 ForteOptions::add_str,
             "Add a string option")
        .def("add_str",
             (void (ForteOptions::*)(const std::string&, const std::string&,
                                     const std::vector<std::string>&, const std::string&)) &
                 ForteOptions::add_str,
             "Add a string option")
        .def("add_int_array", &ForteOptions::add_int_array, "Add an array of integers option")
        .def("add_double_array", &ForteOptions::add_double_array, "Add an array of doubles option")
        .def("add_array", &ForteOptions::add_array, "Add an array option for general elements")
        .def("get_bool", &ForteOptions::get_bool, "Get a boolean option")
        .def("get_int", &ForteOptions::get_int, "Get an integer option")
        .def("get_double", &ForteOptions::get_double, "Get a double option")
        .def("get_str", &ForteOptions::get_str, "Get a string option")
        .def("get_int_vec", &ForteOptions::get_int_vec, "Get a vector of integers option")
        .def("get_double_vec", &ForteOptions::get_double_vec,
             "Get a vector of doubles (py::float) option")
        .def("set_bool", &ForteOptions::set_bool, "Set a boolean option")
        .def("set_int", &ForteOptions::set_int, "Set an integer option")
        .def("set_double", &ForteOptions::set_double, "Set a double option")
        .def("set_str", &ForteOptions::set_str, "Set a string option")
        .def("set_int_vec", &ForteOptions::set_int_vec, "Set a vector of integers option")
        .def("set_double_vec", &ForteOptions::set_double_vec,
             "Set a vector of doubles (py::float) option")
        .def("push_options_to_psi4", &ForteOptions::push_options_to_psi4,
             "Push the options list to Psi4")
        .def("get_options_from_psi4", &ForteOptions::get_options_from_psi4,
             "Read the value of options from Psi4")
        .def("generate_documentation", &ForteOptions::generate_documentation);
}

} // namespace forte
