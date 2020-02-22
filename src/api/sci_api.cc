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

#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "sparse_ci/sigma_vector.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "integrals/active_space_integrals.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_SigmaVector(py::module& m) {
    py::class_<SigmaVector, std::shared_ptr<SigmaVector>>(m, "SigmaVector");

    m.def("make_sigma_vector",
          (std::shared_ptr<SigmaVector>(*)(const std::vector<Determinant>& space,
                                           std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                           size_t max_memory, SigmaVectorType sigma_type)) &
              make_sigma_vector,
          "space"_a, "fci_ints"_a, "max_memory"_a, "sigma_type"_a, "Make a SigmaVector object");

    py::enum_<SigmaVectorType>(m, "SigmaVectorType")
        .value("Full", SigmaVectorType::Full)
        .value("Dynamic", SigmaVectorType::Dynamic)
        .value("SparseList", SigmaVectorType::SparseList)
        .export_values();
}

void export_SparseCISolver(py::module& m) {
    py::class_<SparseCISolver, std::shared_ptr<SparseCISolver>>(m, "SparseCISolver")
        .def(py::init<>())
        .def("diagonalize_hamiltonian",
             (std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>(
                 SparseCISolver::*)(const std::vector<Determinant>& space,
                                    std::shared_ptr<SigmaVector> sigma_vec, int nroot,
                                    int multiplicity)) &
                 SparseCISolver::diagonalize_hamiltonian,
             "Diagonalize the Hamiltonian")
        .def("diagonalize_hamiltonian_full", &SparseCISolver::diagonalize_hamiltonian_full,
             "Diagonalize the full Hamiltonian matrix")
        .def("spin", &SparseCISolver::spin,
             "Return a vector with the average of the S^2 operator for each state")
        .def("energy", &SparseCISolver::energy, "Return a vector with the energy of each state");
    /// Define a convenient function that diagonalizes the Hamiltonian
    m.def(
        "diag",
        [](const std::vector<Determinant>& dets, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
           int multiplicity, int nroot, std::string diag_algorithm) {
            SigmaVectorType type = string_to_sigma_vector_type(diag_algorithm);
            SparseCISolver sparse_solver;
            auto sigma_vector = make_sigma_vector(dets, as_ints, 10000000, type);
            auto [evals, evecs] =
                sparse_solver.diagonalize_hamiltonian(dets, sigma_vector, nroot, multiplicity);
            std::vector<double> energy = sparse_solver.energy();
            std::vector<double> spin = sparse_solver.spin();
            return std::make_tuple(energy, evals, evecs, spin);
        },
        "Diagonalize the Hamiltonian in a basis of determinants");
}
} // namespace forte
