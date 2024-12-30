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
#include <pybind11/complex.h>
#include <pybind11/operators.h>

#include "helpers/string_algorithms.h"

#include "integrals/active_space_integrals.h"

#include "sparse_ci/sparse_operator.h"
#include "sparse_ci/sq_operator_string_ops.h"
#include "sparse_ci/sparse_operator_hamiltonian.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void export_SparseOperator(py::module& m) {
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
        .def(
            "__imul__",
            [](const SparseOperator self, sparse_scalar_t scalar) {
                return self * scalar; // Call the multiplication operator
            },
            "Multiply this SparseOperator by a scalar")
        .def(
            "__imul__",
            [](SparseOperator self, const SparseOperator& other) {
                SparseOperator C;
                for (const auto& [op, coeff] : self.elements()) {
                    for (const auto& [op2, coeff2] : other.elements()) {
                        new_product2(C, op, op2, coeff * coeff2);
                    }
                }
                self = C;
                return self;
            },
            "Multiply this SparseOperator by another SparseOperator")
        .def(
            "__itruediv__",
            [](SparseOperator& self, sparse_scalar_t scalar) {
                return self /= scalar; // Call the in-place division operator
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const SparseOperator& self, sparse_scalar_t scalar) {
                return self / scalar; // Call the division operator
            },
            py::is_operator())
        .def("__mul__",
             [](const SparseOperator& self, sparse_scalar_t scalar) {
                 return self * scalar; // This uses the operator* we defined
             })
        .def("__rmul__",
             [](const SparseOperator& self, sparse_scalar_t scalar) {
                 // This enables the reversed operation: scalar * SparseOperator
                 return self * scalar; // Reuse the __mul__ logic
             })
        .def("__mul__",
             [](const SparseOperator& self, const SparseOperator& other) {
                 SparseOperator C;
                 for (const auto& [op, coeff] : self.elements()) {
                     for (const auto& [op2, coeff2] : other.elements()) {
                         new_product2(C, op, op2, coeff * coeff2);
                     }
                 }
                 return C;
             })
        .def("__rdiv__",
             [](const SparseOperator& self, sparse_scalar_t scalar) {
                 return self * (1.0 / scalar); // This uses the operator* we defined
             })
        .def("__add__", &SparseOperator::operator+, "Add two SparseOperators")
        .def(py::self - py::self)
        .def(-py::self, "Negate the SparseOperator")
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
        .def("__str__", [](const SparseOperator& op) { return join(op.str(), "\n"); })
        .def(
            "fact_trans_lin",
            [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
                auto O_copy = O;
                fact_trans_lin(O_copy, T, reverse, screen_thresh);
                return O_copy;
            },
            "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
            "Evaluate ... (1 - T1) O (1 + T1) ...")

        .def(
            "fact_unitary_trans_antiherm",
            [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
                auto O_copy = O;
                fact_unitary_trans_antiherm(O_copy, T, reverse, screen_thresh);
                return O_copy;
            },
            "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
            "Evaluate ... exp(T1^dagger - T1) O exp(T1 - T1^dagger) ...")

        .def(
            "fact_unitary_trans_antiherm_grad",
            [](SparseOperator& O, const SparseOperatorList& T, size_t n, bool reverse,
               double screen_thresh) {
                auto O_copy = O;
                fact_unitary_trans_antiherm_grad(O_copy, T, n, reverse, screen_thresh);
                return O_copy;
            },
            "T"_a, "n"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
            "Evaluate the gradient of ... exp(T1^dagger - T1) O exp(T1 - T1^dagger) ...")

        .def(
            "fact_unitary_trans_imagherm",
            [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
                auto O_copy = O;
                fact_unitary_trans_imagherm(O_copy, T, reverse, screen_thresh);
                return O_copy;
            },
            "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
            "Evaluate ... exp(i (T1^dagger + T1)) O exp(-i(T1 + T1^dagger)) ...");

    m.def(
        "sparse_operator",
        [](const std::string& s, sparse_scalar_t coefficient, bool allow_reordering) {
            SparseOperator sop;
            sop.add_term_from_str(s, coefficient, allow_reordering);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false,
        "Create a SparseOperator object from a string and a complex");

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
        "list"_a, "allow_reordering"_a = false,
        "Create a SparseOperator object from a list of Tuple[str, complex]");

    m.def(
        "sparse_operator",
        [](const SQOperatorString& sqop, sparse_scalar_t coefficient) {
            SparseOperator sop;
            sop.add(sqop, coefficient);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1),
        "Create a SparseOperator object from a SQOperatorString and a complex");

    m.def(
        "sparse_operator",
        [](const std::vector<std::pair<SQOperatorString, sparse_scalar_t>>& list) {
            SparseOperator sop;
            for (const auto& [sqop, coefficient] : list) {
                sop.add(sqop, coefficient);
            }
            return sop;
        },
        "list"_a, "Create a SparseOperator object from a list of Tuple[SQOperatorString, complex]");

    m.def("sparse_operator_hamiltonian", &sparse_operator_hamiltonian,
          "Create a SparseOperator object from an ActiveSpaceIntegrals object");

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
}
} // namespace forte
