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

#include "psi4/libmints/matrix.h"

#include "base_classes/orbital_transform.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "integrals/integrals.h"

#include "orbital-helpers/localize.h"
#include "orbital-helpers/semi_canonicalize.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace forte {
void export_OrbitalTransform(py::module& m) {
    py::class_<OrbitalTransform>(m, "OrbitalTransform")
        .def("compute_transformation", &OrbitalTransform::compute_transformation)
        .def("set_print", &OrbitalTransform::set_print, "Set printing level")
        .def("get_Ua", &OrbitalTransform::get_Ua, "Get Ua rotation")
        .def("get_Ub", &OrbitalTransform::get_Ub, "Get Ub rotation");
}

void export_Localize(py::module& m) {
    py::class_<Localize, OrbitalTransform>(m, "Localize")
        .def(py::init<std::shared_ptr<ForteOptions>, std::shared_ptr<ForteIntegrals>,
                      std::shared_ptr<MOSpaceInfo>>())
        .def("set_orbital_space",
             (void(Localize::*)(std::vector<int>&)) & Localize::set_orbital_space,
             "Compute the transformation")
        .def("set_orbital_space",
             (void(Localize::*)(std::vector<std::string>&)) & Localize::set_orbital_space,
             "Compute the transformation");
}

void export_SemiCanonical(py::module& m) {
    py::enum_<ActiveOrbitalType::Value>(m, "ActiveOrbitalType::Value")
        .value("canonical", ActiveOrbitalType::Value::canonical)
        .value("natural", ActiveOrbitalType::Value::natural)
        .value("unspecified", ActiveOrbitalType::Value::unspecified)
        .export_values();

    py::class_<ActiveOrbitalType>(m, "ActiveOrbitalType")
        .def(py::init<std::string>())
        .def(py::init<ActiveOrbitalType::Value>())
        .def_readwrite("value", &ActiveOrbitalType::value_)
        .def("__str__", &ActiveOrbitalType::toString);

    py::class_<SemiCanonical>(m, "SemiCanonical")
        .def(py::init<std::shared_ptr<MOSpaceInfo>, std::shared_ptr<ForteIntegrals>,
                      std::shared_ptr<SCFInfo>, bool, bool, double, bool>(),
             "mo_space_info"_a, "ints"_a, "scf_info"_a, "inactive_mix"_a, "active_mix"_a,
             "threshold"_a = 1.0e-8, "quiet"_a = false)
        .def("semicanonicalize", &SemiCanonical::semicanonicalize, "RDMs"_a, "build_fock"_a = true,
             "orb_type"_a = ActiveOrbitalType(ActiveOrbitalType::Value::canonical),
             "transform"_a = true,
             "Semicanonicalize the orbitals and transform the integrals and reference")
        .def("Ua", &SemiCanonical::Ua, "Return the alpha rotation matrix")
        .def("Ub", &SemiCanonical::Ub, "Return the alpha rotation matrix")
        .def("Ua_t", &SemiCanonical::Ua_t, "Return the alpha rotation matrix in the active space")
        .def("Ub_t", &SemiCanonical::Ub_t, "Return the beta rotation matrix in the active space")
        .def("fix_orbital_success", &SemiCanonical::fix_orbital_success,
             "Return if the orbital ordering and phases are fixed successfully");
}

} // namespace forte
