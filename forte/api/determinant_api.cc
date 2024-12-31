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
#include <pybind11/stl.h>

#include "psi4/libmints/matrix.h"

#include "helpers/determinant_helpers.h"

#include "integrals/active_space_integrals.h"

#include "sparse_ci/determinant.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/sq_operator_string.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_Determinant(py::module& m) {
    py::class_<Determinant>(m, "Determinant",
                            "A class for representing a Slater determinant. The number of orbitals "
                            "is determined at compile time and is set to a multiple of 64.")
        .def(py::init<>())
        .def(py::init<const Determinant&>())
        .def(py::init<const std::vector<bool>&, const std::vector<bool>&>())
        .def("zero", &Determinant::zero, "Set all bits to zero")
        .def("fill_up_to", &Determinant::fill_up_to, "Set all bits up index n to one")
        .def("get_bit", &Determinant::get_bit, "Get the value of a bit")
        .def("set_bit", &Determinant::set_bit, "n"_a, "value"_a, "Set the value of a bit")
        .def("get_alfa_bits", &Determinant::get_alfa_bits, "Get alpha bits")
        .def("get_beta_bits", &Determinant::get_beta_bits, "Get beta bits")
        .def("get_alfa_occ", &Determinant::get_alfa_occ,
             "Get the vector of alpha occupied orbital indices.")
        .def("get_beta_occ", &Determinant::get_beta_occ,
             "Get the vector of beta occupied orbital indices.")
        .def("get_alfa_vir", &Determinant::get_alfa_vir,
             "Get the vector of alpha unoccupied orbital indices.")
        .def("get_beta_vir", &Determinant::get_beta_vir,
             "Get the vector of beta unoccupied orbital indices.")
        .def("nbits", &Determinant::get_nbits, "The number of spin orbitals (twice norb)")
        .def("nspinorb", &Determinant::get_nbits, "The number of spin orbitals (twice norb)")
        .def("norb", &Determinant::norb, "The number of spatial orbitals")
        .def("get_alfa_bit", &Determinant::get_alfa_bit, "n"_a, "Get the value of an alpha bit")
        .def("get_beta_bit", &Determinant::get_beta_bit, "n"_a, "Get the value of a beta bit")
        .def("set_alfa_bit", &Determinant::set_alfa_bit, "n"_a, "value"_a,
             "Set the value of an alpha bit")
        .def("set_beta_bit", &Determinant::set_beta_bit, "n"_a, "value"_a,
             "Set the value of an beta bit")
        .def(
            "slater_sign", [](const Determinant& d, size_t n) { return d.slater_sign(n); },
            "Get the sign of the Slater determinant")
        .def(
            "slater_sign_reverse",
            [](const Determinant& d, size_t n) { return d.slater_sign_reverse(n); },
            "Get the sign of the Slater determinant")
        .def("create_alfa_bit", &Determinant::create_alfa_bit, "n"_a, "Create an alpha bit")
        .def("create_beta_bit", &Determinant::create_beta_bit, "n"_a, "Create a beta bit")
        .def("destroy_alfa_bit", &Determinant::destroy_alfa_bit, "n"_a, "Destroy an alpha bit")
        .def("destroy_beta_bit", &Determinant::destroy_beta_bit, "n"_a, "Destroy a beta bit")
        .def("count_alfa", &Determinant::count_alfa, "Count the number of set alpha bits")
        .def("count_beta", &Determinant::count_beta, "Count the number of set beta bits")
        .def("symmetry", &Determinant::symmetry, "Get the symmetry")
        .def(
            "gen_excitation",
            [](Determinant& d, const std::vector<int>& aann, const std::vector<int>& acre,
               const std::vector<int>& bann,
               const std::vector<int>& bcre) { return gen_excitation(d, aann, acre, bann, bcre); },
            "Apply a generic excitation") // uses gen_excitation() defined in determinant.hpp
        .def("spin_flip", &Determinant::spin_flip, "Get the spin-flip determinant")
        .def(
            "str", [](const Determinant& a, int n) { return str(a, n); },
            "n"_a = Determinant::norb(),
            "Get the string representation of the Slater determinant") // uses str() defined in
                                                                       // determinant.hpp
        .def("__repr__", [](const Determinant& a) { return str(a); })
        .def("__str__", [](const Determinant& a) { return str(a); })
        .def("__eq__", [](const Determinant& a, const Determinant& b) { return a == b; })
        .def("__lt__", [](const Determinant& a, const Determinant& b) { return a < b; })
        .def("__hash__", [](const Determinant& a) { return Determinant::Hash()(a); })
        .def(
            "apply",
            [](Determinant& d, const SQOperatorString op) { return apply_operator_to_det(d, op); },
            "Apply an operator to this determinant. Returns the sign and changes the determinant.");

    m.def(
        "det",
        [](const std::string& s) {
            size_t nchar = s.size();
            if (nchar > Determinant::norb()) {
                std::string msg = "The forte.det function was passed a string of length greather "
                                  "than the maximum determinant size: " +
                                  std::to_string(Determinant::norb());
                throw std::runtime_error(msg);
            }
            Determinant d;
            int k = 0;
            for (const char cc : s) {
                const char c = tolower(cc);
                if ((c == '+') or (c == 'a')) {
                    d.create_alfa_bit(k);
                } else if ((c == '-') or (c == 'b')) {
                    d.create_beta_bit(k);
                } else if ((c == '2') or (c == 'x')) {
                    d.create_alfa_bit(k);
                    d.create_beta_bit(k);
                }
                ++k;
            }
            return d;
        },
        "Make a determinant from a string (e.g., \'2+-0\'). 2 or x/X = doubly occupied MO, "
        "+ or "
        "a/A = alpha, - or b/B = beta. Orbital occupations are read from left to right.");

    m.def(
        "spin2", [](const Determinant& lhs, const Determinant& rhs) { return spin2(lhs, rhs); },
        "Compute a matrix element of the S^2 operator");
    m.def("spin2", &spin2<Determinant::nbits>);
    m.def("hamiltonian_matrix", &make_hamiltonian_matrix, "dets"_a, "as_ints"_a,
          "Make a Hamiltonian matrix (psi::Matrix) from a list of determinants and an "
          "ActiveSpaceIntegrals object");
    m.def("s2_matrix", &make_s2_matrix, "dets"_a,
          "Make a matrix (psi::Matrix) of the S^2 operator from a list of determinants");

    py::class_<DeterminantHashVec>(
        m, "DeterminantHashVec",
        "A vector of determinants and a hash table combined into a single object.")
        .def(py::init<>())
        .def(py::init<const std::vector<Determinant>&>())
        .def(py::init<const det_hashvec&>())
        .def("add", &DeterminantHashVec::add, "Add a determinant")
        .def("size", &DeterminantHashVec::size, "Get the size of the vector")
        .def("determinants", &DeterminantHashVec::determinants, "Return a vector of Determinants")
        .def("get_det", &DeterminantHashVec::get_det, "Return a specific determinant by reference")
        .def("get_idx", &DeterminantHashVec::get_idx, " Return the index of a determinant");

    m.def("hilbert_space", &make_hilbert_space, "nmo"_a, "na"_a, "nb"_a, "nirrep"_a,
          "mo_symmetry"_a, "symmetry"_a,
          "Generate the Hilbert space for a given number of "
          "electrons and orbitals");
}
} // namespace forte
