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

#include "sparse_ci/sparse_hamiltonian.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_SparseHamiltonian(py::module& m) {
    py::class_<SparseHamiltonian>(m, "SparseHamiltonian",
                                  "A class to represent a sparse Hamiltonian")
        .def(py::init<std::shared_ptr<ActiveSpaceIntegrals>>())
        .def("compute", &SparseHamiltonian::compute, "state"_a, "screen_thresh"_a = 1.0e-12,
             "Compute the state H|state> using an algorithm that caches the elements of H")
        .def("apply", &SparseHamiltonian::compute, "state"_a, "screen_thresh"_a = 1.0e-12,
             "Compute the state H|state> using an algorithm that caches the elements of H")
        .def(
            "compute_on_the_fly", &SparseHamiltonian::compute_on_the_fly, "state"_a,
            "screen_thresh"_a = 1.0e-12,
            "Compute the state H|state> using an on-the-fly algorithm that has no memory footprint")
        .def("timings", &SparseHamiltonian::timings)
        .def("to_sparse_operator", &SparseHamiltonian::to_sparse_operator,
             "Convert the Hamiltonian to a SparseOperator object");
}
} // namespace forte
