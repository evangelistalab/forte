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
#include <pybind11/numpy.h>

#include "helpers/helpers.h"
#include "base_classes/rdms.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

/// Export the RDMs class
void export_RDMs(py::module& m) {
    py::class_<RDMs>(m, "RDMs")
        .def("max_rdm_level", &RDMs::max_rdm_level, "Return the max RDM level")
        .def(
            "g1a", [](RDMs& rdm) { return ambit_to_np(rdm.g1a()); },
            "Return the alpha 1RDM as a numpy array")
        .def(
            "g1b", [](RDMs& rdm) { return ambit_to_np(rdm.g1b()); },
            "Return the beta 1RDM as a numpy array")
        .def(
            "g2aa", [](RDMs& rdm) { return ambit_to_np(rdm.g2aa()); },
            "Return the alpha-alpha 2RDM as a numpy array")
        .def(
            "g2ab", [](RDMs& rdm) { return ambit_to_np(rdm.g2ab()); },
            "Return the alpha-beta 2RDM as a numpy array")
        .def(
            "g2bb", [](RDMs& rdm) { return ambit_to_np(rdm.g2bb()); },
            "Return the beta-beta 2RDM as a numpy array")
        .def(
            "g3aaa", [](RDMs& rdm) { return ambit_to_np(rdm.g3aaa()); },
            "Return the alpha-alpha-alpha 3RDM as a numpy array")
        .def(
            "g3aab", [](RDMs& rdm) { return ambit_to_np(rdm.g3aab()); },
            "Return the alpha-alpha-beta 3RDM as a numpy array")
        .def(
            "g3abb", [](RDMs& rdm) { return ambit_to_np(rdm.g3abb()); },
            "Return the alpha-beta-beta 3RDM as a numpy array")
        .def(
            "g3bbb", [](RDMs& rdm) { return ambit_to_np(rdm.g3bbb()); },
            "Return the beta-beta-beta 3RDM as a numpy array")
        .def(
            "SFg2_data", [](RDMs& rdm) { return ambit_to_np(rdm.SFg2()); },
            "Return the spin-free 2-RDM as a numpy array")
        .def(
            "L2aa", [](RDMs& rdm) { return ambit_to_np(rdm.L2aa()); },
            "Return the alpha-alpha 2-cumulant as a numpy array")
        .def(
            "L2ab", [](RDMs& rdm) { return ambit_to_np(rdm.L2ab()); },
            "Return the alpha-beta 2-cumulant as a numpy array")
        .def(
            "L2bb", [](RDMs& rdm) { return ambit_to_np(rdm.L2bb()); },
            "Return the beta-beta 2-cumulant as a numpy array")
        .def(
            "L3aaa", [](RDMs& rdm) { return ambit_to_np(rdm.L3aaa()); },
            "Return the alpha-alpha-alpha 3-cumulant as a numpy array")
        .def(
            "L3aab", [](RDMs& rdm) { return ambit_to_np(rdm.L3aab()); },
            "Return the alpha-alpha-beta 3-cumulant as a numpy array")
        .def(
            "L3abb", [](RDMs& rdm) { return ambit_to_np(rdm.L3abb()); },
            "Return the alpha-beta-beta 3-cumulant as a numpy array")
        .def(
            "L3bbb", [](RDMs& rdm) { return ambit_to_np(rdm.L3bbb()); },
            "Return the beta-beta-beta 3-cumulant as a numpy array");
}
} // namespace forte
