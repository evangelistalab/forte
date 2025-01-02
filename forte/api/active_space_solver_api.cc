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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "integrals/one_body_integrals.h"
#include "integrals/active_space_integrals.h"

#include "base_classes/active_space_solver.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_ActiveSpaceSolver(py::module& m) {
    py::class_<ActiveSpaceSolver, std::shared_ptr<ActiveSpaceSolver>>(m, "ActiveSpaceSolver")
        .def("compute_energy", &ActiveSpaceSolver::compute_energy)
        .def("rdms", &ActiveSpaceSolver::rdms)
        .def("compute_contracted_energy", &ActiveSpaceSolver::compute_contracted_energy,
             "as_ints"_a, "max_body"_a,
             "Solve the contracted CI eigenvalue problem using given integrals")
        .def("compute_average_rdms", &ActiveSpaceSolver::compute_average_rdms,
             "Compute the weighted average reference")
        .def("state_energies_map", &ActiveSpaceSolver::state_energies_map,
             "Return a map of StateInfo to the computed nroots of energies")
        .def("set_active_space_integrals", &ActiveSpaceSolver::set_active_space_integrals,
             "Set the active space integrals manually")
        .def("set_Uactv", &ActiveSpaceSolver::set_Uactv,
             "Set unitary matrices for changing orbital basis in RDMs when computing dipoles")
        .def("compute_multipole_moment", &ActiveSpaceSolver::compute_multipole_moment,
             "Compute dipole or quadrupole moment")
        .def("compute_fosc_same_orbs", &ActiveSpaceSolver::compute_fosc_same_orbs,
             "Compute the oscillator strength assuming using same orbitals")
        .def("state_ci_wfn_map", &ActiveSpaceSolver::state_ci_wfn_map,
             "Return a map from StateInfo to CI wave functions (DeterminantHashVec, eigenvectors)")
        .def("state_filename_map", &ActiveSpaceSolver::state_filename_map,
             "Return a map from StateInfo to wave function file names")
        .def("dump_wave_function", &ActiveSpaceSolver::dump_wave_function,
             "Dump wave functions to disk")
        .def("eigenvectors", &ActiveSpaceSolver::eigenvectors, "Return the CI wave functions");

    m.def("compute_average_state_energy", &compute_average_state_energy,
          "Compute the average energy given the energies and weights of each state");
}

} // namespace forte
