/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see LICENSE, AUTHORS).
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

#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"

#include "sparse_ci/sigma_vector.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "integrals/active_space_integrals.h"

#include "fci/string_address.h"
#include "sparse_ci/determinant.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/sparse_state_vector.h"
#include "sparse_ci/sparse_operator.h"
#include "sparse_ci/sparse_fact_exp.h"
#include "sparse_ci/sparse_exp.h"
#include "sparse_ci/sparse_hamiltonian.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

/// Export the Determinant class
void export_Determinant(py::module& m) {
    py::class_<Determinant>(m, "Determinant",
                            "A class for representing a Slater determinant. The number of orbitals "
                            "is determined at compile time and is set to a multiple of 64.")
        .def(py::init<>())
        .def(py::init<const Determinant&>())
        .def(py::init<const std::vector<bool>&, const std::vector<bool>&>())
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
        .def(
            "str", [](const Determinant& a, int n) { return str(a, n); },
            "n"_a = Determinant::norb(),
            "Get the string representation of the Slater determinant") // uses str() defined in
                                                                       // determinant.hpp
        .def("__repr__", [](const Determinant& a) { return str(a); })
        .def("__str__", [](const Determinant& a) { return str(a); })
        .def("__eq__", [](const Determinant& a, const Determinant& b) { return a == b; })
        .def("__lt__", [](const Determinant& a, const Determinant& b) { return a < b; })
        .def("__hash__", [](const Determinant& a) { return Determinant::Hash()(a); });

    py::class_<String>(
        m, "String",
        "A class for representing the occupation pattern of alpha or beta spin orbitals. The "
        "number of orbitals is determined at compile time and is set to a multiple of 64.")
        .def(py::init<>(), "Build an empty string")
        .def(
            "__repr__", [](const String& a) { return str(a); },
            "Get the string representation of the string")
        .def(
            "__str__", [](const String& a) { return str(a); },
            "Get the string representation of the string")
        .def(
            "__eq__", [](const String& a, const String& b) { return a == b; },
            "Check if two strings are equal")
        .def(
            "__lt__", [](const String& a, const String& b) { return a < b; },
            "Check if a string is less than another string")
        .def(
            "__hash__", [](const String& a) { return String::Hash()(a); },
            "Get the hash of the string");

    py::class_<Configuration>(
        m, "Configuration",
        "A class to represent an electron configuration. A configuration stores information "
        "about "
        "the doubly and singly occupied orbitals. However, it does not store information about "
        "how "
        "the spin of singly occupied orbitals. The number of orbitals is determined at "
        "compile time and is set to a multiple of 64.")
        .def(py::init<>(), "Build an empty configuration")
        .def(py::init<const Determinant&>(), "Build a configuration from a determinant")
        .def(
            "str", [](const Configuration& a, int n) { return str(a, n); },
            "n"_a = Configuration::norb(),
            "Get the string representation of the Slater determinant")
        .def("is_empt", &Configuration::is_empt, "n"_a, "Is orbital n empty?")
        .def("is_docc", &Configuration::is_docc, "n"_a, "Is orbital n doubly occupied?")
        .def("is_socc", &Configuration::is_socc, "n"_a, "Is orbital n singly occupied?")
        .def("set_occ", &Configuration::set_occ, "n"_a, "value"_a, "Set the value of an alpha bit")
        .def("count_docc", &Configuration::count_docc,
             "Count the number of doubly occupied orbitals")
        .def("count_socc", &Configuration::count_socc,
             "Count the number of singly occupied orbitals")
        .def(
            "get_docc_vec",
            [](const Configuration& c) {
                int dim = c.count_docc();
                std::vector<int> l(dim);
                c.get_docc_vec(Configuration::norb(), l);
                return l;
            },
            "Get a list of the doubly occupied orbitals")
        .def(
            "get_socc_vec",
            [](const Configuration& c) {
                int dim = c.count_socc();
                std::vector<int> l(dim);
                c.get_socc_vec(Configuration::norb(), l);
                return l;
            },
            "Get a list of the singly occupied orbitals")
        .def(
            "__repr__", [](const Configuration& a) { return str(a); },
            "Get the string representation of the configuration")
        .def(
            "__str__", [](const Configuration& a) { return str(a); },
            "Get the string representation of the configuration")
        .def(
            "__eq__", [](const Configuration& a, const Configuration& b) { return a == b; },
            "Check if two configurations are equal")
        .def(
            "__lt__", [](const Configuration& a, const Configuration& b) { return a < b; },
            "Check if a configuration is less than another configuration")
        .def(
            "__hash__", [](const Configuration& a) { return Configuration::Hash()(a); },
            "Get the hash of the configuration");

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
        "str",
        [](const std::string& s) {
            size_t nchar = s.size();
            if (nchar > String::nbits) {
                std::string msg = "The forte.str function was passed a string of length greather "
                                  "than the maximum size: " +
                                  std::to_string(String::nbits);
                throw std::runtime_error(msg);
            }
            String str;
            int k = 0;
            for (const char cc : s) {
                const char c = tolower(cc);
                if (c == '1') {
                    str.set_bit(k, true);
                }
                ++k;
            }
            return str;
        },
        "Make a string from a text string (e.g., \'1001\'). 1 = occupied MO, 0 = unoccupied MO. "
        "Orbital occupations are read from left to right.");

    m.def(
        "spin2", [](const Determinant& lhs, const Determinant& rhs) { return spin2(lhs, rhs); },
        "Compute a matrix element of the S^2 operator");

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

    py::class_<StringAddress>(m, "StringAddress", "A class to compute the address of a string")
        .def(py::init<int, int, const std::vector<std::vector<String>>&>(),
             "Construct a StringAddress object from a list of lists of strings", "nmo"_a, "ne"_a,
             "strings"_a)
        .def("add", &StringAddress::add, "Return the address of a string")
        .def("sym", &StringAddress::sym, "Return the symmetry of a string")
        .def("strpi", &StringAddress::strpi, "Return the number of strings per irrep");

    py::class_<SparseOperator>(m, "SparseOperator", "A class to represent a sparse operator")
        .def(py::init<bool>(), "antihermitian"_a = false)
        .def("add_term",
             py::overload_cast<const std::vector<std::tuple<bool, bool, int>>&, double, bool>(
                 &SparseOperator::add_term),
             "op_list"_a, "value"_a = 0.0, "allow_reordering"_a = false)
        .def("add_term", py::overload_cast<const SQOperator&>(&SparseOperator::add_term))
        .def("add_term_from_str", &SparseOperator::add_term_from_str, "str"_a,
             "coefficient"_a = 0.0, "allow_reordering"_a = false)
        .def("pop_term", &SparseOperator::pop_term)
        .def("term", &SparseOperator::term)
        .def("size", &SparseOperator::size)
        .def("coefficients", &SparseOperator::coefficients)
        .def("set_coefficients", &SparseOperator::set_coefficients)
        .def("set_coefficient", &SparseOperator::set_coefficient)
        .def("op_list", &SparseOperator::op_list)
        .def("str", &SparseOperator::str)
        .def("latex", &SparseOperator::latex)
        .def("adjoint", &SparseOperator::adjoint);

    py::class_<SQOperator>(m, "SQOperator",
                           "A class to represent a string of creation/annihilation operators")
        .def(py::init<double, const Determinant&, const Determinant&>())
        .def("coefficient", &SQOperator::coefficient)
        .def("cre", &SQOperator::cre)
        .def("ann", &SQOperator::ann)
        .def("str", &SQOperator::str)
        .def("latex", &SQOperator::latex)
        .def("adjoint", &SQOperator::adjoint)
        .def("__repr__", [](const SQOperator& sqop) { return sqop.str(); })
        .def("__str__", [](const SQOperator& sqop) { return sqop.str(); });

    py::class_<StateVector>(m, "StateVector", "A class to represent a vector of determinants")
        .def(py::init<const det_hash<double>&>())
        .def(py::init<const StateVector&>())
        .def(
            "items", [](const StateVector& v) { return py::make_iterator(v.begin(), v.end()); },
            py::keep_alive<0, 1>()) // Essential: keep object alive while iterator exists
        .def("str", &StateVector::str)
        .def("__eq__", &StateVector::operator==)
        .def("__repr__", [](const StateVector& v) { return v.str(); })
        .def("__str__", [](const StateVector& v) { return v.str(); })
        .def("__getitem__", [](StateVector& v, const Determinant& d) { return v[d]; })
        .def("__setitem__",
             [](StateVector& v, const Determinant& d, const double val) { v[d] = val; })
        .def("__contains__", [](StateVector& v, const Determinant& d) { return v.map().count(d); });

    py::class_<SparseHamiltonian>(m, "SparseHamiltonian",
                                  "A class to represent a sparse Hamiltonian")
        .def(py::init<std::shared_ptr<ActiveSpaceIntegrals>>())
        .def("compute", &SparseHamiltonian::compute)
        .def("compute_on_the_fly", &SparseHamiltonian::compute_on_the_fly)
        .def("timings", &SparseHamiltonian::timings);

    py::class_<SparseExp>(m, "SparseExp", "A class to compute the exponential of a sparse operator")
        .def(py::init<>())
        .def("compute", &SparseExp::compute, "sop"_a, "state"_a, "algorithm"_a = "cached",
             "scaling_factor"_a = 1.0, "maxk"_a = 19, "screen_thresh"_a = 1.0e-12)
        .def("timings", &SparseExp::timings);

    py::class_<SparseFactExp>(
        m, "SparseFactExp",
        "A class to compute the product exponential of a sparse operator using factorization")
        .def(py::init<bool>(), "phaseless"_a = false)
        .def("compute", &SparseFactExp::compute, "sop"_a, "state"_a, "algorithm"_a = "cached",
             "inverse"_a = false, "screen_thresh"_a = 1.0e-13)
        .def("timings", &SparseFactExp::timings);

    m.def("apply_operator",
          py::overload_cast<SparseOperator&, const StateVector&, double>(&apply_operator), "sop"_a,
          "state0"_a, "screen_thresh"_a = 1.0e-12);

    m.def("apply_operator_safe",
          py::overload_cast<SparseOperator&, const StateVector&>(&apply_operator_safe), "sop"_a,
          "state0"_a);

    m.def("apply_number_projector", &apply_number_projector);
    m.def("get_projection", &get_projection);
    m.def("overlap", &overlap);
    m.def("spin2", &spin2<Determinant::nbits>);
}

void export_SigmaVector(py::module& m) {
    py::class_<SigmaVector, std::shared_ptr<SigmaVector>>(m, "SigmaVector");

    m.def("make_sigma_vector",
          (std::shared_ptr<SigmaVector>(*)(const std::vector<Determinant>& space,
                                           std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                           size_t max_memory, SigmaVectorType sigma_type)) &
              make_sigma_vector,
          "space"_a, "fci_ints"_a, "max_memory"_a, "sigma_type"_a, "Make a SigmaVector object");

    py::enum_<SigmaVectorType>(m, "SigmaVectorType")
        .value("Full", SigmaVectorType::Full)
        .value("Dynamic", SigmaVectorType::Dynamic)
        .value("SparseList", SigmaVectorType::SparseList)
        .export_values();
}

void export_SparseCISolver(py::module& m) {
    py::class_<SparseCISolver, std::shared_ptr<SparseCISolver>>(
        m, "SparseCISolver", "A class to represent a sparse CI solver")
        .def(py::init<>())
        .def("diagonalize_hamiltonian",
             (std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>(
                 SparseCISolver::*)(const std::vector<Determinant>& space,
                                    std::shared_ptr<SigmaVector> sigma_vec, int nroot,
                                    int multiplicity)) &
                 SparseCISolver::diagonalize_hamiltonian,
             "Diagonalize the Hamiltonian")
        .def("diagonalize_hamiltonian_full", &SparseCISolver::diagonalize_hamiltonian_full,
             "Diagonalize the full Hamiltonian matrix")
        .def("spin", &SparseCISolver::spin,
             "Return a vector with the average of the S^2 operator for each state")
        .def("energy", &SparseCISolver::energy, "Return a vector with the energy of each state");
    /// Define a convenient function that diagonalizes the Hamiltonian
    m.def(
        "diag",
        [](const std::vector<Determinant>& dets, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
           int multiplicity, int nroot, std::string diag_algorithm) {
            SigmaVectorType type = string_to_sigma_vector_type(diag_algorithm);
            SparseCISolver sparse_solver;
            auto sigma_vector = make_sigma_vector(dets, as_ints, 10000000, type);
            auto [evals, evecs] =
                sparse_solver.diagonalize_hamiltonian(dets, sigma_vector, nroot, multiplicity);
            std::vector<double> energy = sparse_solver.energy();
            std::vector<double> spin = sparse_solver.spin();
            return std::make_tuple(energy, evals, evecs, spin);
        },
        "Diagonalize the Hamiltonian in a basis of determinants");
}
} // namespace forte
