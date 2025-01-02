/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see LICENSE, AUTHORS).
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

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_Configuration(py::module& m) {

    py::class_<Configuration>(
        m, "Configuration",
        "A class to represent an electron configuration. A configuration stores information "
        "about the doubly and singly occupied orbitals. However, it does not store information "
        "about how the spin of singly occupied orbitals are coupled. The number of orbitals is "
        "determined at compile time and is set to a multiple of 64.")
        .def(py::init<>(), "Build an empty configuration")
        .def(py::init<const Determinant&>(), "Build a configuration from a determinant")
        .def(
            "str", [](const Configuration& a, int n) { return str(a, n); },
            "n"_a = Configuration::norb(),
            "Get the string representation of the Slater determinant")
        .def("is_empt", &Configuration::is_empt, "n"_a, "Is orbital n empty?")
        .def("is_docc", &Configuration::is_docc, "n"_a, "Is orbital n doubly occupied?")
        .def("is_socc", &Configuration::is_socc, "n"_a, "Is orbital n singly occupied?")
        .def("set_occ", &Configuration::set_occ, "n"_a, "value"_a, "Set the value of an alpha bit")
        .def("count_docc", &Configuration::count_docc,
             "Count the number of doubly occupied orbitals")
        .def("count_socc", &Configuration::count_socc,
             "Count the number of singly occupied orbitals")
        .def(
            "get_docc_vec",
            [](const Configuration& c) {
                int dim = c.count_docc();
                std::vector<int> l(dim);
                c.get_docc_vec(Configuration::norb(), l);
                return l;
            },
            "Get a list of the doubly occupied orbitals")
        .def(
            "get_socc_vec",
            [](const Configuration& c) {
                int dim = c.count_socc();
                std::vector<int> l(dim);
                c.get_socc_vec(Configuration::norb(), l);
                return l;
            },
            "Get a list of the singly occupied orbitals")
        .def(
            "__repr__", [](const Configuration& a) { return str(a); },
            "Get the string representation of the configuration")
        .def(
            "__str__", [](const Configuration& a) { return str(a); },
            "Get the string representation of the configuration")
        .def(
            "__eq__", [](const Configuration& a, const Configuration& b) { return a == b; },
            "Check if two configurations are equal")
        .def(
            "__lt__", [](const Configuration& a, const Configuration& b) { return a < b; },
            "Check if a configuration is less than another configuration")
        .def(
            "__hash__", [](const Configuration& a) { return Configuration::Hash()(a); },
            "Get the hash of the configuration");
}
} // namespace forte
