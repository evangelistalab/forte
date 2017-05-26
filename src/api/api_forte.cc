/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
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

#ifndef _api_api_forte_h_
#define _api_api_forte_h_

#include "psi4/libmints/wavefunction.h"

#include "../integrals/integrals.h"
#include "../main.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace psi {
namespace forte {

PYBIND11_PLUGIN(forte) {
    py::module m("forte", "pybind11 Forte plugin");

    m.def("read_forte_options", &api_forte_read_options, "Read Forte options");
    m.def("run_forte", &api_run_forte, "Run Forte plugin");
    return m.ptr();
}
}
}
#endif // _api_api_forte_h_
