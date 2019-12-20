/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see LICENSE, AUTHORS).
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

//#ifndef _python_api_h_
//#define _python_api_h_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "base_classes/rdms.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

py::array_t<double> ambit_to_np(ambit::Tensor t) {
    return py::array_t<double>(t.dims(), &(t.data()[0]));
}

/// Export the RDMs class
void export_RDMs(py::module& m) {
    py::class_<RDMs>(m, "RDMs")
        .def("max_rdm_level", &RDMs::max_rdm_level, "Return the max RDM level")
        .def(
            "g1a_np", [](RDMs& rdm) { return ambit_to_np(rdm.g1a()); },
            "Return the alpha 1RDM as a numpy array")
        .def(
            "g1b_np", [](RDMs& rdm) { return ambit_to_np(rdm.g1b()); },
            "Return the beta 1RDM as a numpy array")
        .def(
            "g2aa_np", [](RDMs& rdm) { return ambit_to_np(rdm.g2aa()); },
            "Return the alpha-alpha 2RDM as a numpy array")
        .def(
            "g2ab_np", [](RDMs& rdm) { return ambit_to_np(rdm.g2ab()); },
            "Return the alpha-beta 2RDM as a numpy array")
        .def(
            "g2bb_np", [](RDMs& rdm) { return ambit_to_np(rdm.g2bb()); },
            "Return the beta-beta 2RDM as a numpy array")
        .def(
            "g3aaa_np", [](RDMs& rdm) { return ambit_to_np(rdm.g3aaa()); },
            "Return the alpha-alpha-alpha 3RDM as a numpy array")
        .def(
            "g3aab_np", [](RDMs& rdm) { return ambit_to_np(rdm.g3aab()); },
            "Return the alpha-alpha-beta 3RDM as a numpy array")
        .def(
            "g3abb_np", [](RDMs& rdm) { return ambit_to_np(rdm.g3abb()); },
            "Return the alpha-beta-beta 3RDM as a numpy array")
        .def(
            "g3bbb_np", [](RDMs& rdm) { return ambit_to_np(rdm.g3bbb()); },
            "Return the beta-beta-beta 3RDM as a numpy array")
        .def(
            "SFg2_data_np", [](RDMs& rdm) { return ambit_to_np(rdm.SFg2()); },
            "Return the spin-free 2-RDM as a numpy array")
        .def(
            "L2aa_np", [](RDMs& rdm) { return ambit_to_np(rdm.L2aa()); },
            "Return the alpha-alpha 2-cumulant as a numpy array")
        .def(
            "L2ab_np", [](RDMs& rdm) { return ambit_to_np(rdm.L2ab()); },
            "Return the alpha-beta 2-cumulant as a numpy array")
        .def(
            "L2bb_np", [](RDMs& rdm) { return ambit_to_np(rdm.L2bb()); },
            "Return the beta-beta 2-cumulant as a numpy array")
        .def(
            "L3aaa_np", [](RDMs& rdm) { return ambit_to_np(rdm.L3aaa()); },
            "Return the alpha-alpha-alpha 3-cumulant as a numpy array")
        .def(
            "L3aab_np", [](RDMs& rdm) { return ambit_to_np(rdm.L3aab()); },
            "Return the alpha-alpha-beta 3-cumulant as a numpy array")
        .def(
            "L3abb_np", [](RDMs& rdm) { return ambit_to_np(rdm.L3abb()); },
            "Return the alpha-beta-beta 3-cumulant as a numpy array")
        .def(
            "L3bbb_np", [](RDMs& rdm) { return ambit_to_np(rdm.L3bbb()); },
            "Return the beta-beta-beta 3-cumulant as a numpy array");
}
} // namespace forte
