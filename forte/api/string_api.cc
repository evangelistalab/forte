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

#include "sparse_ci/determinant.h"
#include "fci/fci_string_address.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_String(py::module& m) {
    py::class_<String>(
        m, "String",
        "A class for representing the occupation pattern of alpha or beta spin orbitals. The "
        "number of orbitals is determined at compile time and is set to a multiple of 64.")
        .def(py::init<>(), "Build an empty string")
        .def(
            "__repr__", [](const String& a) { return str(a); },
            "Get the string representation of the string")
        .def(
            "__str__", [](const String& a) { return str(a); },
            "Get the string representation of the string")
        .def(
            "__eq__", [](const String& a, const String& b) { return a == b; },
            "Check if two strings are equal")
        .def(
            "__lt__", [](const String& a, const String& b) { return a < b; },
            "Check if a string is less than another string")
        .def(
            "__hash__", [](const String& a) { return String::Hash()(a); },
            "Get the hash of the string");

    m.def(
        "str",
        [](const std::string& s) {
            size_t nchar = s.size();
            if (nchar > String::nbits) {
                std::string msg = "The forte.str function was passed a string of length greather "
                                  "than the maximum size: " +
                                  std::to_string(String::nbits);
                throw std::runtime_error(msg);
            }
            String str;
            int k = 0;
            for (const char cc : s) {
                const char c = tolower(cc);
                if (c == '1') {
                    str.set_bit(k, true);
                }
                ++k;
            }
            return str;
        },
        "Make a string from a text string (e.g., \'1001\'). 1 = occupied MO, 0 = unoccupied MO. "
        "Orbital occupations are read from left to right.");

    py::class_<FCIStringAddress>(m, "StringAddress", "A class to compute the address of a string")
        .def(py::init<int, int, const std::vector<std::vector<String>>&>(),
             "Construct a StringAddress object from a list of lists of strings")
        .def("add", &FCIStringAddress::add, "Return the address of a string")
        .def("sym", &FCIStringAddress::sym, "Return the symmetry of a string")
        .def("strpcls", &FCIStringAddress::strpcls, "Return the number of strings per class");
}
} // namespace forte
