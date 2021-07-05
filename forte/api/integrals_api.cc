/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see LICENSE, AUTHORS).
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

#include "helpers/helpers.h"
#include "integrals/integrals.h"

namespace py = pybind11;

namespace forte {

/// export ForteIntegrals
void export_ForteIntegrals(py::module& m) {
    py::class_<ForteIntegrals, std::shared_ptr<ForteIntegrals>>(m, "ForteIntegrals")
        .def("rotate_orbitals", &ForteIntegrals::rotate_orbitals, "Rotate MOs during contructor")
        .def("nmo", &ForteIntegrals::nmo, "Return the total number of moleuclar orbitals")
        .def("ncmo", &ForteIntegrals::ncmo, "Return the number of correlated orbitals")
        .def(
            "oei_a_block",
            [](ForteIntegrals& ints, const std::vector<size_t>& p, const std::vector<size_t>& q) {
                return ambit_to_np(ints.oei_a_block(p, q));
            },
            "Return the alpha 1e-integrals")
        .def(
            "oei_b_block",
            [](ForteIntegrals& ints, const std::vector<size_t>& p, const std::vector<size_t>& q) {
                return ambit_to_np(ints.oei_b_block(p, q));
            },
            "Return the beta 1e-integrals")
        .def(
            "tei_aa_block",
            [](ForteIntegrals& ints, const std::vector<size_t>& p, const std::vector<size_t>& q,
               const std::vector<size_t>& r, const std::vector<size_t>& s) {
                return ambit_to_np(ints.aptei_aa_block(p, q, r, s));
            },
            "Return the alpha-alpha 2e-integrals in physicists' notation")
        .def(
            "tei_ab_block",
            [](ForteIntegrals& ints, const std::vector<size_t>& p, const std::vector<size_t>& q,
               const std::vector<size_t>& r, const std::vector<size_t>& s) {
                return ambit_to_np(ints.aptei_ab_block(p, q, r, s));
            },
            "Return the alpha-beta 2e-integrals in physicists' notation")
        .def(
            "tei_bb_block",
            [](ForteIntegrals& ints, const std::vector<size_t>& p, const std::vector<size_t>& q,
               const std::vector<size_t>& r, const std::vector<size_t>& s) {
                return ambit_to_np(ints.aptei_bb_block(p, q, r, s));
            },
            "Return the beta-beta 2e-integrals in physicists' notation")
        .def("set_nuclear_repulsion", &ForteIntegrals::set_nuclear_repulsion,
             "Set the nuclear repulsion energy")
        .def("set_scalar", &ForteIntegrals::set_scalar, "Set the scalar energy")
        .def("set_oei", &ForteIntegrals::set_oei_all, "Set the one-electron integrals")
        .def("set_tei", &ForteIntegrals::set_tei_all, "Set the two-electron integrals")
        .def("initialize", &ForteIntegrals::initialize, "Initialize the integrals")
        .def("print_ints", &ForteIntegrals::print_ints, "Print the integrals");
}
} // namespace forte
