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
#include <pybind11/complex.h>

#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"

#include "helpers/determinant_helpers.h"
#include "helpers/string_algorithms.h"

#include "sparse_ci/sigma_vector.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "integrals/active_space_integrals.h"

#include "fci/fci_string_address.h"

#include "genci/genci_string_lists.h"
#include "genci/genci_vector.h"

#include "base_classes/mo_space_info.h"

#include "sparse_ci/determinant.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/sparse.h"
#include "sparse_ci/sparse_state.h"
#include "sparse_ci/sparse_operator.h"
#include "sparse_ci/sparse_fact_exp.h"
#include "sparse_ci/sparse_exp.h"
#include "sparse_ci/sparse_hamiltonian.h"
#include "sparse_ci/sq_operator_string.h"
#include "sparse_ci/sq_operator_string_ops.h"

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

    py::class_<FCIStringAddress>(m, "StringAddress", "A class to compute the address of a string")
        .def(py::init<int, int, const std::vector<std::vector<String>>&>(),
             "Construct a StringAddress object from a list of lists of strings")
        .def("add", &FCIStringAddress::add, "Return the address of a string")
        .def("sym", &FCIStringAddress::sym, "Return the symmetry of a string")
        .def("strpcls", &FCIStringAddress::strpcls, "Return the number of strings per class");

    py::class_<SQOperatorString>(m, "SQOperatorString",
                                 "A class to represent a string of creation/annihilation operators")
        .def(py::init<const Determinant&, const Determinant&>())
        .def("cre", &SQOperatorString::cre)
        .def("ann", &SQOperatorString::ann)
        .def("str", &SQOperatorString::str)
        .def("count", &SQOperatorString::count)
        .def("number_component", &SQOperatorString::number_component)
        .def("non_number_component", &SQOperatorString::non_number_component)
        .def("__str__", &SQOperatorString::str)
        .def("__repr___", &SQOperatorString::str)
        .def("latex", &SQOperatorString::latex)
        .def("latex_compact", &SQOperatorString::latex_compact)
        .def("is_identity", &SQOperatorString::is_identity)
        .def("is_nilpotent", &SQOperatorString::is_nilpotent)
        .def("op_tuple", &SQOperatorString::op_tuple)
        .def("__eq__", &SQOperatorString::operator==)
        .def("__lt__", &SQOperatorString::operator<);

    py::enum_<CommutatorType>(m, "CommutatorType")
        .value("commute", CommutatorType::Commute)
        .value("anticommute", CommutatorType::AntiCommute)
        .value("may_not_commute", CommutatorType::MayNotCommute);

    m.def("commutator_type", &commutator_type, "lhs"_a, "rhs"_a,
          "Get the commutator type of two operator strings");

    m.def(
        "sqop",
        [](const std::string& s, bool allow_reordering) {
            return make_sq_operator_string(s, allow_reordering);
        },
        "s"_a, "allow_reordering"_a = false);

    py::class_<SparseOperator>(m, "SparseOperator", "A class to represent a sparse operator")
        .def(py::init<>())
        .def(py::init<SparseOperator>())
        .def("add",
             py::overload_cast<const SQOperatorString&, sparse_scalar_t>(&SparseOperator::add),
             "sqop"_a, "coefficient"_a = sparse_scalar_t(1))
        .def("add",
             py::overload_cast<const std::string&, sparse_scalar_t, bool>(
                 &SparseOperator::add_term_from_str),
             "str"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false)
        .def(
            "__iter__",
            [](const SparseOperator& v) {
                return py::make_iterator(v.elements().begin(), v.elements().end());
            },
            py::keep_alive<0, 1>()) // Essential: keep object alive while iterator exists
        .def(
            "coefficient",
            [](const SparseOperator& op, const std::string& s) {
                const auto [sqop, factor] = make_sq_operator_string(s, false);
                return factor * op[sqop];
            },
            "Get the coefficient of a term")
        .def(
            "__getitem__",
            [](const SparseOperator& op, const std::string& s) {
                const auto [sqop, factor] = make_sq_operator_string(s, false);
                return factor * op[sqop];
            },
            "Get the coefficient of a term")
        .def(
            "set_coefficient",
            [](SparseOperator& op, const std::string& s, sparse_scalar_t value) {
                const auto [sqop, factor] = make_sq_operator_string(s, false);
                op[sqop] = factor * value;
            },
            "Get the coefficient of a term")
        .def(
            "remove",
            [](SparseOperator& op, const std::string& s) {
                const auto [sqop, _] = make_sq_operator_string(s, false);
                op.remove(sqop);
            },
            "Remove a term")
        .def(
            "__matmul__",
            [](const SparseOperator& lhs, const SparseOperator& rhs) { return lhs * rhs; },
            "Multiply two SparseOperator objects")
        .def(
            "commutator",
            [](const SparseOperator& lhs, const SparseOperator& rhs) {
                return commutator(lhs, rhs);
            },
            "Compute the commutator of two SparseOperator objects")
        .def("__iadd__", &SparseOperator::operator+=, "Add a SparseOperator to this SparseOperator")
        .def("__isub__", &SparseOperator::operator-=,
             "Subtract a SparseOperator from this SparseOperator")
        .def("__imul__", &SparseOperator::operator*=, "Multiply this SparseOperator by a scalar")
        .def("__itruediv__", &SparseOperator::operator/=, "Divide this SparseOperator by a scalar")
        .def("__mul__",
             [](const SparseOperator& self, sparse_scalar_t scalar) {
                 return self * scalar; // This uses the operator* we defined
             })
        .def("__rmul__",
             [](const SparseOperator& self, sparse_scalar_t scalar) {
                 // This enables the reversed operation: scalar * SparseOperator
                 return self * scalar; // Reuse the __mul__ logic
             })
        .def("__rdiv__",
             [](const SparseOperator& self, sparse_scalar_t scalar) {
                 return self * (1.0 / scalar); // This uses the operator* we defined
             })
        .def("__add__", &SparseOperator::operator+, "Add two SparseOperators")
        .def("__sub__", &SparseOperator::operator-, "Subtract two SparseOperators")
        .def("copy", &SparseOperator::copy)
        .def("size", &SparseOperator::size)
        .def("__len__", &SparseOperator::size)
        .def("norm", [](const SparseOperator& op) { return op.norm(); })
        // .def("op_list", &SparseOperator::op_list)
        .def("str", &SparseOperator::str)
        .def("latex", &SparseOperator::latex)
        .def("adjoint", [](const SparseOperator& op) { return op.adjoint(); })
        .def("__eq__", &SparseOperator::operator==)
        .def("__repr__", [](const SparseOperator& op) { return join(op.str(), "\n"); })
        .def("__str__", [](const SparseOperator& op) { return join(op.str(), "\n"); });

    py::class_<SparseOperatorList>(m, "SparseOperatorList",
                                   "A class to represent a list of sparse operators")
        .def(py::init<>())
        .def("add", &SparseOperatorList::add)
        .def("add", &SparseOperatorList::add_term_from_str, "str"_a,
             "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false)
        .def("to_operator", &SparseOperatorList::to_operator)
        .def("__len__", &SparseOperatorList::size)
        .def(
            "__iter__",
            [](const SparseOperatorList& v) {
                return py::make_iterator(v.elements().begin(), v.elements().end());
            },
            py::keep_alive<0, 1>())
        .def("size", &SparseOperatorList::size)
        .def("__repr__", [](const SparseOperatorList& op) { return join(op.str(), "\n"); })
        .def("__str__", [](const SparseOperatorList& op) { return join(op.str(), "\n"); })
        .def(
            "__getitem__", [](const SparseOperatorList& op, const size_t n) { return op[n]; },
            "Get the coefficient of a term")
        .def(
            "__setitem__",
            [](SparseOperatorList& op, const size_t n, sparse_scalar_t value) { op[n] = value; },
            "Set the coefficient of a term")
        .def("coefficients",
             [](SparseOperatorList& op) {
                 std::vector<sparse_scalar_t> values(op.size());
                 for (size_t i = 0, max = op.size(); i < max; ++i) {
                     values[i] = op[i];
                 }
                 return values;
             })
        .def("set_coefficients",
             [](SparseOperatorList& op, const std::vector<sparse_scalar_t>& values) {
                 if (op.size() != values.size()) {
                     throw std::invalid_argument(
                         "The size of the list of coefficients must match the "
                         "size of the operator list");
                 }
                 for (size_t i = 0; i < op.size(); ++i) {
                     op[i] = values[i];
                 }
             })
        .def(
            "__call__",
            [](const SparseOperatorList& op, const size_t n) {
                if (n >= op.size()) {
                    throw std::out_of_range("Index out of range");
                }
                return op(n);
            },
            "Get the nth element");

    m.def("new_product", [](const SparseOperator A, const SparseOperator B) {
        SparseOperator C;
        SQOperatorProductComputer computer;
        for (const auto& [op, coeff] : A.elements()) {
            for (const auto& [op2, coeff2] : B.elements()) {
                computer.product(op, op2, coeff * coeff2,
                                 [&C](const SQOperatorString& sqop, const sparse_scalar_t c) {
                                     C.add(sqop, c);
                                 });
            }
        }
        return C;
    });

    m.def("new_product2", [](const SparseOperator A, const SparseOperator B) {
        SparseOperator C;
        for (const auto& [op, coeff] : A.elements()) {
            for (const auto& [op2, coeff2] : B.elements()) {
                new_product2(C, op, op2, coeff * coeff2);
            }
        }
        return C;
    });

    m.def(
        "sparse_operator",
        [](const std::string& s, sparse_scalar_t coefficient, bool allow_reordering) {
            SparseOperator sop;
            sop.add_term_from_str(s, coefficient, allow_reordering);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false);

    m.def(
        "sparse_operator",
        [](const std::vector<std::pair<std::string, sparse_scalar_t>>& list,
           bool allow_reordering) {
            SparseOperator sop;
            for (const auto& [s, coefficient] : list) {
                sop.add_term_from_str(s, coefficient, allow_reordering);
            }
            return sop;
        },
        "list"_a, "allow_reordering"_a = false);

    m.def(
        "operator_list",
        [](const std::string& s, sparse_scalar_t coefficient, bool allow_reordering) {
            SparseOperatorList sop;
            sop.add_term_from_str(s, coefficient, allow_reordering);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false);

    m.def(
        "sim_trans_fact_exc",
        [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
            // time this call and print to std::cout
            sim_trans_fact_op(O, T, reverse, screen_thresh);
        },
        "O"_a, "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12);

    m.def(
        "sim_trans_fact_antiherm",
        [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
            // time this call and print to std::cout
            sim_trans_fact_antiherm(O, T, reverse, screen_thresh);
        },
        "O"_a, "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12);

    m.def(
        "sim_trans_fact_antiherm_grad",
        [](SparseOperator& O, const SparseOperatorList& T, size_t n, bool reverse,
           double screen_thresh) { sim_trans_fact_antiherm_grad(O, T, n, reverse, screen_thresh); },
        "O"_a, "T"_a, "n"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12);

    m.def(
        "sim_trans_fact_imagherm",
        [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
            sim_trans_fact_imagherm(O, T, reverse, screen_thresh);
        },
        "O"_a, "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12);

    py::class_<SparseState, std::shared_ptr<SparseState>>(
        m, "SparseState", "A class to represent a vector of determinants")
        .def(py::init<>())
        .def(py::init<const SparseState&>())
        .def(py::init<const SparseState::container&>())
        .def(
            "items", [](const SparseState& v) { return py::make_iterator(v.begin(), v.end()); },
            py::keep_alive<0, 1>()) // Essential: keep object alive while iterator exists
        .def("str", &SparseState::str)
        .def("size", &SparseState::size)
        .def("norm", &SparseState::norm, "p"_a = 2,
             "Compute the p-norm of the vector (default p = 2, p = -1 for infinity norm)")
        .def("add", &SparseState::add)
        .def("__iadd__", &SparseState::operator+=, "Add a SparseState to this SparseState")
        .def("__isub__", &SparseState::operator-=, "Subtract a SparseState from this SparseState")
        .def("__imul__", &SparseState::operator*=, "Multiply this SparseState by a scalar")
        .def("__len__", &SparseState::size)
        .def("__eq__", &SparseState::operator==)
        .def("__repr__", [](const SparseState& v) { return v.str(); })
        .def("__str__", [](const SparseState& v) { return v.str(); })
        .def("map", [](const SparseState& v) { return v.elements(); })
        .def("elements", [](const SparseState& v) { return v.elements(); })
        .def("__getitem__", [](SparseState& v, const Determinant& d) { return v[d]; })
        .def("__setitem__",
             [](SparseState& v, const Determinant& d, const sparse_scalar_t val) { v[d] = val; })
        .def("__contains__", [](SparseState& v, const Determinant& d) { return v.count(d); });

    py::class_<SparseHamiltonian>(m, "SparseHamiltonian",
                                  "A class to represent a sparse Hamiltonian")
        .def(py::init<std::shared_ptr<ActiveSpaceIntegrals>>())
        .def("compute", &SparseHamiltonian::compute)
        .def("apply", &SparseHamiltonian::compute)
        .def("compute_on_the_fly", &SparseHamiltonian::compute_on_the_fly)
        .def("timings", &SparseHamiltonian::timings);

    py::class_<SparseExp>(m, "SparseExp", "A class to compute the exponential of a sparse operator")
        .def(py::init<int, double>(), "maxk"_a = 19, "screen_thresh"_a = 1.0e-12)
        .def("apply_op",
             py::overload_cast<const SparseOperator&, const SparseState&, double>(
                 &SparseExp::apply_op),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_op",
             py::overload_cast<const SparseOperatorList&, const SparseState&, double>(
                 &SparseExp::apply_op),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_antiherm",
             py::overload_cast<const SparseOperator&, const SparseState&, double>(
                 &SparseExp::apply_antiherm),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_antiherm",
             py::overload_cast<const SparseOperatorList&, const SparseState&, double>(
                 &SparseExp::apply_antiherm),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0);

    py::class_<SparseFactExp>(
        m, "SparseFactExp",
        "A class to compute the product exponential of a sparse operator using factorization")
        .def(py::init<double>(), "screen_thresh"_a = 1.0e-12)
        .def("apply_op", &SparseFactExp::apply_op, "sop"_a, "state"_a, "inverse"_a = false)
        .def("apply_antiherm", &SparseFactExp::apply_antiherm, "sop"_a, "state"_a,
             "inverse"_a = false);

    m.def("apply_op", &apply_operator_lin, "sop"_a, "state0"_a, "screen_thresh"_a = 1.0e-12);

    m.def("apply_antiherm", &apply_operator_antiherm, "sop"_a, "state0"_a,
          "screen_thresh"_a = 1.0e-12);

    m.def("apply_number_projector", &apply_number_projector);
    m.def("get_projection", &get_projection);
    m.def("overlap", &overlap);
    m.def("spin2", &spin2<Determinant::nbits>);
    m.def("hamiltonian_matrix", &make_hamiltonian_matrix, "dets"_a, "as_ints"_a,
          "Make a Hamiltonian matrix (psi::Matrix) from a list of determinants and an "
          "ActiveSpaceIntegrals object");
    m.def("s2_matrix", &make_s2_matrix, "dets"_a,
          "Make a matrix (psi::Matrix) of the S^2 operator from a list of determinants");
}

void export_SigmaVector(py::module& m) {
    py::class_<SigmaVector, std::shared_ptr<SigmaVector>>(m, "SigmaVector");

    m.def("sigma_vector",
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

void export_GAS(py::module& m) {
    py::class_<GenCIStringLists, std::shared_ptr<GenCIStringLists>>(
        m, "GenCIStringLists", "A class to represent the strings of a GAS")
        .def(py::init<std::shared_ptr<MOSpaceInfo>, size_t, size_t, int, PrintLevel,
                      const std::vector<int>, const std::vector<int>>())
        .def("make_determinants", &GenCIStringLists::make_determinants,
             "Return a vector of Determinants");
    py::class_<GenCIVector, std::shared_ptr<GenCIVector>>(m, "GenCIVector",
                                                          "A class to represent a GAS vector")
        .def(py::init<std::shared_ptr<GenCIStringLists>>())
        .def("print", &GenCIVector::print, "Print the GAS vector")
        .def("size", &GenCIVector::size, "Return the size of the GAS vector")
        .def("__len__", &GenCIVector::size, "Return the size of the GAS vector")
        .def("as_state_vector", &GenCIVector::as_state_vector, "Return a SparseState object")
        .def("set_to", &GenCIVector::set_to, "Set the GAS vector to a given value");
}
} // namespace forte
