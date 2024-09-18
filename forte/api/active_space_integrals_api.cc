/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
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
#include <pybind11/stl.h>

#include "integrals/active_space_integrals.h"
#include "integrals/one_body_integrals.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

/// export ActiveSpaceIntegrals
void export_ActiveSpaceIntegrals(py::module& m) {

    // export ActiveSpaceIntegrals
    py::class_<ActiveSpaceIntegrals, std::shared_ptr<ActiveSpaceIntegrals>>(m,
                                                                            "ActiveSpaceIntegrals")
        .def("slater_rules", &ActiveSpaceIntegrals::slater_rules,
             "Compute the matrix element of the Hamiltonian between two determinants")
        .def("energy", &ActiveSpaceIntegrals::energy, "Return the energy of a determinant")
        .def("nuclear_repulsion_energy", &ActiveSpaceIntegrals::nuclear_repulsion_energy,
             "Get the nuclear repulsion energy")
        .def("frozen_core_energy", &ActiveSpaceIntegrals::frozen_core_energy,
             "Get the frozen core energy (contribution from FROZEN_DOCC)")
        .def("scalar_energy", &ActiveSpaceIntegrals::scalar_energy,
             "Get the scalar_energy energy (contribution from RESTRICTED_DOCC)")
        .def("nmo", &ActiveSpaceIntegrals::nmo, "Get the number of active orbitals")
        .def("mo_symmetry", &ActiveSpaceIntegrals::active_mo_symmetry,
             "Return the symmetry of the active MOs")
        .def("oei_a", &ActiveSpaceIntegrals::oei_a, "Get the alpha effective one-electron integral")
        .def("oei_b", &ActiveSpaceIntegrals::oei_b, "Get the beta effective one-electron integral")
        .def("tei_aa", &ActiveSpaceIntegrals::tei_aa, "alpha-alpha two-electron integral <pq||rs>")
        .def("tei_ab", &ActiveSpaceIntegrals::tei_ab, "alpha-beta two-electron integral <pq|rs>")
        .def("tei_bb", &ActiveSpaceIntegrals::tei_bb, "beta-beta two-electron integral <pq||rs>")
        .def("add", &ActiveSpaceIntegrals::add, "Add another integrals to this one", "as_ints"_a,
             "factor"_a = 1.0)
        .def("print", &ActiveSpaceIntegrals::print, "Print the integrals (alpha-alpha case)");

    // export ActiveMultipoleIntegrals
    py::class_<ActiveMultipoleIntegrals, std::shared_ptr<ActiveMultipoleIntegrals>>(
        m, "ActiveMultipoleIntegrals")
        .def("compute_electronic_dipole", &ActiveMultipoleIntegrals::compute_electronic_dipole)
        .def("compute_electronic_quadrupole",
             &ActiveMultipoleIntegrals::compute_electronic_quadrupole)
        .def("nuclear_dipole", &ActiveMultipoleIntegrals::nuclear_dipole)
        .def("nuclear_quadrupole", &ActiveMultipoleIntegrals::nuclear_quadrupole)
        .def("set_dipole_name", &ActiveMultipoleIntegrals::set_dp_name)
        .def("set_quadrupole_name", &ActiveMultipoleIntegrals::set_qp_name);
}

} // namespace forte