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
#include <pybind11/stl.h>

#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "base_classes/scf_info.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_SCFInfo(py::module& m) {
    py::class_<SCFInfo, std::shared_ptr<SCFInfo>>(m, "SCFInfo")
        .def(py::init<psi::SharedWavefunction>())
        .def(py::init<const psi::Dimension&, const psi::Dimension&, const psi::Dimension&, double,
                      std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Vector>,
                      std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>())
        .def("__repr__", &SCFInfo::to_string)
        .def("nmopi", &SCFInfo::nmopi, "the number of orbitals per irrep")
        .def("doccpi", &SCFInfo::doccpi, "the number of doubly occupied orbitals per irrep")
        .def("soccpi", &SCFInfo::soccpi, "the number of singly occupied orbitals per irrep")
        .def("reference_energy", &SCFInfo::reference_energy, "the reference energy")
        .def("epsilon_a", &SCFInfo::epsilon_a, "a vector of alpha orbital energy (psi::Vector)")
        .def("epsilon_b", &SCFInfo::epsilon_b, "a vector of beta orbital energy (psi::Vector)")
        .def("Ca", &SCFInfo::Ca, "the alpha MO coefficient matrix (psi::Matrix)")
        .def("Cb", &SCFInfo::Cb, "the beta MO coefficient matrix (psi::Matrix)")
        .def("update_orbitals", &SCFInfo::update_orbitals, "Ca"_a, "Cb"_a,
             "transform_ints"_a = true, "Update orbitals for given orbital coefficients")
        .def("rotate_orbitals", &SCFInfo::rotate_orbitals, "Ua"_a, "Ub"_a,
             "transform_ints"_a = true,
             "Update orbitals with given unitary "
             "transformation matrices")
        .def("reorder_orbitals", &SCFInfo::reorder_orbitals, "perm"_a,
             "Reorder orbitals based on the given permutation (uses symmetry of the molecule)");
}

} // namespace forte
