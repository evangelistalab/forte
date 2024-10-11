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
#include <pybind11/numpy.h>

#include "helpers/helpers.h"
#include "base_classes/rdms.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {
void export_RDMs(py::module& m) {
    py::enum_<RDMsType>(m, "RDMsType")
        .value("spin_dependent", RDMsType::spin_dependent)
        .value("spin_free", RDMsType::spin_free);

    py::class_<RDMs, std::shared_ptr<RDMs>>(m, "RDMs")
        .def("max_rdm_level", &RDMs::max_rdm_level, "Return the max RDM level")
        .def("dim", &RDMs::dim, "Return the dimension of each index")
        .def("type", &RDMs::rdm_type, "Return RDM spin type")
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
            "g4aaaa", [](RDMs& rdm) { return ambit_to_np(rdm.g4aaaa()); },
            "Return the alpha-alpha-alpha-alpha 4RDM as a numpy array")
        .def(
            "g4aaab", [](RDMs& rdm) { return ambit_to_np(rdm.g4aaab()); },
            "Return the alpha-alpha-alpha-beta 4RDM as a numpy array")
        .def(
            "g4aabb", [](RDMs& rdm) { return ambit_to_np(rdm.g4aabb()); },
            "Return the alpha-alpha-beta-beta 4RDM as a numpy array")
        .def(
            "g4abbb", [](RDMs& rdm) { return ambit_to_np(rdm.g4abbb()); },
            "Return the alpha-beta-beta-beta 4RDM as a numpy array")
        .def(
            "g4bbbb", [](RDMs& rdm) { return ambit_to_np(rdm.g4bbbb()); },
            "Return the beta-beta-beta-beta 4RDM as a numpy array")
        .def(
            "L1a", [](RDMs& rdm) { return ambit_to_np(rdm.L1a()); },
            "Return the alpha 1-cumulant as a numpy array")
        .def(
            "L1b", [](RDMs& rdm) { return ambit_to_np(rdm.L1b()); },
            "Return the beta 1-cumulant as a numpy array")
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
            "Return the beta-beta-beta 3-cumulant as a numpy array")
        .def(
            "L4aaaa", [](RDMs& rdm) { return ambit_to_np(rdm.L4aaaa()); },
            "Return the alpha-alpha-alpha-alpha 4-cumulant as a numpy array")
        .def(
            "L4aaab", [](RDMs& rdm) { return ambit_to_np(rdm.L4aaab()); },
            "Return the alpha-alpha-alpha-beta 4-cumulant as a numpy array")
        .def(
            "L4aabb", [](RDMs& rdm) { return ambit_to_np(rdm.L4aabb()); },
            "Return the alpha-alpha-beta-beta 4-cumulant as a numpy array")
        .def(
            "L4abbb", [](RDMs& rdm) { return ambit_to_np(rdm.L4abbb()); },
            "Return the alpha-beta-beta-beta 4-cumulant as a numpy array")
        .def(
            "L4bbbb", [](RDMs& rdm) { return ambit_to_np(rdm.L4bbbb()); },
            "Return the beta-beta-beta-beta 4-cumulant as a numpy array")
        .def(
            "SF_G1", [](RDMs& rdm) { return ambit_to_np(rdm.SF_G1()); },
            "Return the spin-free 1RDM as a numpy array")
        .def(
            "SF_G2", [](RDMs& rdm) { return ambit_to_np(rdm.SF_G2()); },
            "Return the spin-free 2RDM as a numpy array")
        .def(
            "SF_G3", [](RDMs& rdm) { return ambit_to_np(rdm.SF_G3()); },
            "Return the spin-free 3RDM as a numpy array")
        .def("SF_G1mat", py::overload_cast<>(&RDMs::SF_G1mat),
             "Return the spin-free 1RDM as a Psi4 Matrix without symmetry")
        .def("SF_G1mat", py::overload_cast<const psi::Dimension&>(&RDMs::SF_G1mat),
             "Return the spin-free 1RDM as a Psi4 Matrix with symmetry given by input")
        .def(
            "SF_L1", [](RDMs& rdm) { return ambit_to_np(rdm.SF_L1()); },
            "Return the spin-free 1RDM as a numpy array")
        .def(
            "SF_L2", [](RDMs& rdm) { return ambit_to_np(rdm.SF_L2()); },
            "Return the spin-free (Ms-averaged) 2-cumulant as a numpy array")
        .def(
            "SF_L3", [](RDMs& rdm) { return ambit_to_np(rdm.SF_L3()); },
            "Return the spin-free (Ms-averaged) 2-cumulant as a numpy array")
        .def("rotate", &RDMs::rotate, "Ua"_a, "Ub"_a,
             "Rotate RDMs using the input unitary matrices");
}
} // namespace forte
