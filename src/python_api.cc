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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers.h"
#include "integrals/integrals.h"
#include "orbital-helper/localize.h"
#include "forte.h"

namespace py = pybind11;

namespace psi {
namespace forte {

/**
 * @brief Wrapper of the main forte function for the pybind11 interface
 */
int api_forte_read_options(Options& options) {
    //    Process::environment.options.set_read_globals(true);
    int value = read_forte_options("FORTE", options); // options are not read by default
    //    Process::environment.options.set_read_globals(false);
    return value;
}

int add(int a, int b) { return a + b; }

/**
 * @brief Wrapper of the main forte function for the pybind11 interface
 */
SharedWavefunction api_run_forte(SharedWavefunction ref_wfn, Options& options) {
    options.set_current_module("FORTE");
    return run_forte(ref_wfn, options);
}

PYBIND11_MODULE(forte, m) {
    m.doc() = "pybind11 Forte module"; // module docstring
    m.def("read_forte_options", &api_forte_read_options, "Read Forte options");
    m.def("run_forte", &api_run_forte, "Run Forte plugin");
    m.def("add", &add, "Run Forte plugin");
    m.def("forte_startup", &forte_startup);
    m.def("forte_cleanup", &forte_cleanup);
    m.def("forte_banner", &forte_banner, "Print forte banner");
    m.def("make_mo_space_info", &make_mo_space_info, "Make a MOSpaceInfo object");
    m.def("make_aosubspace_projector", &make_aosubspace_projector, "Make a AOSubspace projector");
    m.def("make_forte_integrals", &make_forte_integrals, "Make Forte integrals");
    m.def("forte_old_methods", &forte_old_methods, "Run Forte methods");
    // export MOSpaceInfo
    py::class_<MOSpaceInfo, std::shared_ptr<MOSpaceInfo>>(m, "MOSpaceInfo")
        .def("size", &MOSpaceInfo::size);

    // export ForteIntegrals
    py::class_<ForteIntegrals, std::shared_ptr<ForteIntegrals>>(m, "ForteIntegrals");

    // export Localize
    py::class_<LOCALIZE, std::shared_ptr<LOCALIZE>>(m, "LOCALIZE")
        .def(py::init<std::shared_ptr<Wavefunction>, Options&, std::shared_ptr<ForteIntegrals>,
                      std::shared_ptr<MOSpaceInfo>>())
        .def("split_localize", &LOCALIZE::split_localize)
        .def("full_localize", &LOCALIZE::split_localize);
}
}
}
#endif // _python_api_h_
