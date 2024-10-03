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

#include <vector>
#include <functional>
#include <span>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "helpers/davidson_liu_solver.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

// a utility function to create a sigma builder from a matrix
auto make_sigma_builder(const std::vector<std::vector<double>>& M)
    -> std::function<void(std::span<double>, std::span<double>)> {
    return {[M](std::span<double> b, std::span<double> sigma) {
        auto n = sigma.size();
        for (size_t i = 0; i < n; ++i) {
            auto res = 0.0;
            for (size_t j = 0; j < n; ++j) {
                res += M[i][j] * b[j];
            }
            sigma[i] = res;
        }
    }};
};

void export_DavidsonLiuSolver(py::module& m) {
    py::class_<DavidsonLiuSolver, std::shared_ptr<DavidsonLiuSolver>>(
        m, "DavidsonLiuSolver", "A class to diagonalize hermitian matrices")
        .def(py::init<size_t, size_t, size_t, size_t>(), "Initialize the solver", "size"_a,
             "nroots"_a, "collapse_per_root"_a = 1, "subspace_per_root"_a = 5)
        .def("add_sigma_builder", &DavidsonLiuSolver::add_sigma_builder,
             "Add a function to build the sigma vector", "sigma_builder"_a)
        .def(
            "add_test_sigma_builder",
            [](DavidsonLiuSolver& self, const std::vector<std::vector<double>>& M) {
                self.add_sigma_builder(make_sigma_builder(M));
            },
            "Create a sigma builder from a matrix", "M"_a)
        .def("add_h_diag", &DavidsonLiuSolver::add_h_diag, "Add the diagonal of the Hamiltonian")
        .def("add_guesses", &DavidsonLiuSolver::add_guesses, "Add the initial guesses")
        .def("add_project_out_vectors", &DavidsonLiuSolver::add_project_out_vectors,
             "Add vectors to project out of the subspace")
        .def("set_print_level", &DavidsonLiuSolver::set_print_level, "Set the print level")
        .def("set_e_convergence", &DavidsonLiuSolver::set_e_convergence,
             "Set the energy convergence")
        .def("set_r_convergence", &DavidsonLiuSolver::set_r_convergence,
             "Set the residual convergence")
        .def("solve", &DavidsonLiuSolver::solve, "The main solver function")
        .def("reset", &DavidsonLiuSolver::reset, "Function to reset the solver")
        .def("eigenvalues", &DavidsonLiuSolver::eigenvalues, "Return the eigenvalues")
        .def("eigenvectors", &DavidsonLiuSolver::eigenvectors, "Return the eigenvectors")
        .def("eigenvector", &DavidsonLiuSolver::eigenvector, "Return the n-th eigenvector");
}

} // namespace forte
