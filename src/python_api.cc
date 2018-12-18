/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
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

#ifndef _python_api_h_
#define _python_api_h_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers/mo_space_info.h"
#include "integrals/integrals.h"
#include "orbital-helpers/localize.h"
#include "forte.h"
#include "fci/fci.h"
#include "fci/fci_solver.h"
#include "base_classes/state_info.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

PYBIND11_MODULE(forte, m) {
    m.doc() = "pybind11 Forte module"; // module docstring
    m.def("read_options", &read_options, "Read Forte options");
    m.def("startup", &startup);
    m.def("cleanup", &cleanup);
    m.def("banner", &banner, "Print forte banner");
    m.def("make_mo_space_info", &make_mo_space_info, "Make a MOSpaceInfo object");
    m.def("make_aosubspace_projector", &make_aosubspace_projector, "Make a AOSubspace projector");
    m.def("make_forte_integrals", &make_forte_integrals, "Make Forte integrals");
    m.def("forte_old_methods", &forte_old_methods, "Run Forte methods");
    m.def("make_active_space_solver", &make_active_space_solver, "Make an active space solver");

    // export ForteOptions
    py::class_<ForteOptions, std::shared_ptr<ForteOptions>>(m, "ForteOptions")
        .def(py::init<>())
        .def(py::init<psi::Options&>())
        .def("push_options_to_psi4", &ForteOptions::push_options_to_psi4)
        .def("generate_documentation", &ForteOptions::generate_documentation);

    // export MOSpaceInfo
    py::class_<MOSpaceInfo, std::shared_ptr<MOSpaceInfo>>(m, "MOSpaceInfo")
        .def("size", &MOSpaceInfo::size);

    // export ForteIntegrals
    py::class_<ForteIntegrals, std::shared_ptr<ForteIntegrals>>(m, "ForteIntegrals");

    // export Localize
    py::class_<LOCALIZE, std::shared_ptr<LOCALIZE>>(m, "LOCALIZE")
        .def(py::init<std::shared_ptr<psi::Wavefunction>, psi::Options&,
                      std::shared_ptr<ForteIntegrals>, std::shared_ptr<MOSpaceInfo>>())
        .def("split_localize", &LOCALIZE::split_localize)
        .def("full_localize", &LOCALIZE::split_localize);

    // export StateInfo
    py::class_<StateInfo, std::shared_ptr<StateInfo>>(m, "StateInfo")
        .def(py::init<int, int, int, int, int>(), "na"_a, "nb"_a, "multiplicity"_a, "twice_ms"_a,
             "irrep"_a)
        .def(py::init<psi::SharedWavefunction>());

    py::class_<ActiveSpaceSolver, std::shared_ptr<ActiveSpaceSolver>>(m, "ActiveSpaceSolver")
        .def("compute_energy", &ActiveSpaceSolver::compute_energy);

    // export FCIIntegrals
    py::class_<FCIIntegrals, std::shared_ptr<FCIIntegrals>>(m, "FCIIntegrals")
        .def(py::init<std::shared_ptr<ForteIntegrals>, std::shared_ptr<MOSpaceInfo>>());

    // export FCISolver
    py::class_<FCISolver, std::shared_ptr<FCISolver>>(m, "FCISolver")
        .def(py::init<psi::Dimension, std::vector<size_t>, std::vector<size_t>, StateInfo,
                      std::shared_ptr<ForteIntegrals>, std::shared_ptr<MOSpaceInfo>, size_t, int,
                      psi::Options&>())
        .def("compute_energy", &FCISolver::compute_energy);

    py::class_<FCI, std::shared_ptr<FCI>>(m, "FCI")
        .def(py::init<StateInfo, std::shared_ptr<ForteOptions>, std::shared_ptr<ForteIntegrals>,
                      std::shared_ptr<MOSpaceInfo>>())
        .def("compute_energy", &FCI::compute_energy);
}
} // namespace forte

#endif // _python_api_h_
