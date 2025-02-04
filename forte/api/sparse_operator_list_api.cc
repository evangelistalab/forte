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
#include <pybind11/complex.h>
#include <pybind11/operators.h>

#include "helpers/string_algorithms.h"

#include "sparse_ci/sparse_operator.h"
#include "sparse_ci/sparse_state.h"
#include "sparse_ci/sparse_exp.h"
#include "sparse_ci/sparse_fact_exp.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

// enum used to specify the type of the exponential operator
enum class ExpType { Excitation, Antihermitian };

// this struct is used to implement exp(op) in Python
// Note that it creates a copy of the operator
struct ExpOperatorList {
    const SparseOperatorList op;
    ExpType exp_type;
    int maxk;
    double screen_thresh;
    ExpOperatorList(const SparseOperatorList& op, ExpType exp_type, int maxk, double screen_thresh)
        : op(op), exp_type(exp_type), maxk(maxk), screen_thresh(screen_thresh) {}
};

// this struct is used to implement fact_exp(op) in Python
// Note that it creates a copy of the operator
struct ExpOperatorListFact {
    const SparseOperatorList op;
    ExpType exp_type;
    bool inverse;
    double screen_thresh;
    ExpOperatorListFact(const SparseOperatorList& op, ExpType exp_type, bool inverse,
                        double screen_thresh)
        : op(op), exp_type(exp_type), inverse(inverse), screen_thresh(screen_thresh) {}
};

void export_SparseOperatorList(py::module& m) {
    py::class_<SparseOperatorList>(m, "SparseOperatorList",
                                   "A class to represent a list of sparse operators")
        .def(py::init<>())
        .def(py::init<SparseOperatorList>())
        .def("add", &SparseOperatorList::add)
        .def("add", &SparseOperatorList::add_term_from_str, "str"_a,
             "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false)
        // .def("add",
        //      [](SparseOperatorList& op, const, sparse_scalar_t value, bool allow_reordering) {
        //          make_sq_operator_string_from_list op.add(sqop, value);
        //      })
        .def("add_term",
             py::overload_cast<const std::vector<std::tuple<bool, bool, int>>&, double, bool>(
                 &SparseOperatorList::add_term),
             "op_list"_a, "value"_a = 0.0, "allow_reordering"_a = false)
        .def(
            "add",
            [](SparseOperatorList& op, const std::vector<size_t>& acre,
               const std::vector<size_t>& bcre, const std::vector<size_t>& aann,
               const std::vector<size_t>& bann, sparse_scalar_t coeff) {
                op.add(SQOperatorString({acre.begin(), acre.end()}, {bcre.begin(), bcre.end()},
                                        {aann.begin(), aann.end()}, {bann.begin(), bann.end()}),
                       coeff);
            },
            "acre"_a, "bcre"_a, "aann"_a, "bann"_a, "coeff"_a = sparse_scalar_t(1),
            "Add a term to the operator by passing lists of creation and annihilation indices. "
            "This version is faster than the string version and does not check for reordering")
        .def("to_operator", &SparseOperatorList::to_operator)
        .def(
            "remove",
            [](SparseOperatorList& op, const std::string& s) {
                const auto [sqop, _] = make_sq_operator_string(s, false);
                op.remove(sqop);
            },
            "Remove a specific element from the vector space")
        .def("__len__", &SparseOperatorList::size)
        .def(
            "__iter__",
            [](const SparseOperatorList& v) {
                return py::make_iterator(v.elements().begin(), v.elements().end());
            },
            py::keep_alive<0, 1>())
        .def("__repr__", [](const SparseOperatorList& op) { return join(op.str(), "\n"); })
        .def("__str__", [](const SparseOperatorList& op) { return join(op.str(), "\n"); })
        .def(
            "__getitem__", [](const SparseOperatorList& op, const size_t n) { return op[n]; },
            "Get the coefficient of a term")
        .def(
            "__getitem__",
            [](const SparseOperatorList& op, const std::string& s) {
                const auto [sqop, factor] = make_sq_operator_string(s, false);
                return factor * op[sqop];
            },
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
        .def("reverse", &SparseOperatorList::reverse, "Reverse the order of the operators")
        .def(
            "__call__",
            [](const SparseOperatorList& op, const size_t n) {
                if (n >= op.size()) {
                    throw std::out_of_range("Index out of range");
                }
                return op(n);
            },
            "Get the nth operator")
        .def(
            "__matmul__",
            [](const SparseOperatorList& op, const SparseState& st) {
                // form a temporary SparseOperator from the list of operators
                auto sop = op.to_operator();
                return apply_operator_lin(sop, st);
            },
            "Multiply a SparseOperator and a SparseState");

    // Wrapper class that holds a SparseOperator
    // and overloads operator* to apply forte::SparseExp.
    py::class_<struct ExpOperatorList>(m, "ExpOperatorList")
        .def(py::init<const SparseOperatorList&, ExpType, bool, double>())
        .def("__matmul__", [](const ExpOperatorList& self, const SparseState& state) {
            if (self.exp_type == ExpType::Excitation) {
                auto exp_op = SparseExp(self.maxk, self.screen_thresh);
                return exp_op.apply_op(self.op, state);
            }
            auto exp_op = SparseExp(self.maxk, self.screen_thresh);
            return exp_op.apply_antiherm(self.op, state);
        });

    // Define a small wrapper class that holds a SparseOperator
    // and overloads operator* to apply forte::SparseFactExp.
    py::class_<struct ExpOperatorListFact>(m, "ExpOperatorListFact")
        .def(py::init<const SparseOperatorList&, ExpType, bool, double>())
        .def("__matmul__", [](const ExpOperatorListFact& self, const SparseState& state) {
            if (self.exp_type == ExpType::Excitation) {
                auto exp_op = SparseFactExp(self.screen_thresh);
                return exp_op.apply_op(self.op, state, self.inverse);
            }
            auto exp_op = SparseFactExp(self.screen_thresh);
            return exp_op.apply_antiherm(self.op, state, self.inverse);
        });

    // Provide a function "exp" that returns an ExpOperatorList object
    m.def(
        "exp",
        [](const SparseOperatorList& T, int maxk, double screen_thresh) {
            return ExpOperatorList{T, ExpType::Excitation, maxk, screen_thresh};
        },
        "Allow usage of exp(T) * state", "T"_a, "makx"_a = 20, "screen_thresh"_a = 1.0e-12);

    // Provide a function "exp_antiherm" that returns an ExpOperatorList object
    m.def(
        "exp_antiherm",
        [](const SparseOperatorList& T, int maxk, double screen_thresh) {
            return ExpOperatorList{T, ExpType::Antihermitian, maxk, screen_thresh};
        },
        "Allow usage of exp_antiherm(T) * state", "T"_a, "makx"_a = 20,
        "screen_thresh"_a = 1.0e-12);

    // Provide a function "fact_exp" that returns an ExpOperatorListFact object
    m.def(
        "fact_exp",
        [](const SparseOperatorList& T, bool inverse, double screen_thresh) {
            return ExpOperatorListFact{T, ExpType::Excitation, inverse, screen_thresh};
        },
        "Allow usage of fact_exp(T) * state", "T"_a, "inverse"_a = false,
        "screen_thresh"_a = 1.0e-12);

    // Provide a function "fact_exp_antiherm" that returns an ExpOperatorListFact object
    m.def(
        "fact_exp_antiherm",
        [](const SparseOperatorList& T, bool inverse, double screen_thresh) {
            return ExpOperatorListFact{T, ExpType::Antihermitian, inverse, screen_thresh};
        },
        "Allow usage of fact_exp_antiherm(T) * state", "T"_a, "inverse"_a = false,
        "screen_thresh"_a = 1.0e-12);

    m.def(
        "operator_list",
        [](const std::string& s, sparse_scalar_t coefficient, bool allow_reordering) {
            SparseOperatorList sop;
            sop.add_term_from_str(s, coefficient, allow_reordering);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false,
        "Create a SparseOperatorList object from a string and a complex");

    m.def(
        "operator_list",
        [](const std::vector<std::pair<std::string, sparse_scalar_t>>& list,
           bool allow_reordering) {
            SparseOperatorList sop;
            for (const auto& [s, coefficient] : list) {
                sop.add_term_from_str(s, coefficient, allow_reordering);
            }
            return sop;
        },
        "list"_a, "allow_reordering"_a = false,
        "Create a SparseOperatorList object from a list of Tuple[str, complex]");

    m.def(
        "operator_list",
        [](const SQOperatorString& sqop, sparse_scalar_t coefficient) {
            SparseOperatorList sop;
            sop.add(sqop, coefficient);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1),
        "Create a SparseOperatorList object from a SQOperatorString and a complex");

    m.def(
        "operator_list",
        [](const std::vector<std::pair<SQOperatorString, sparse_scalar_t>>& list) {
            SparseOperatorList sop;
            for (const auto& [sqop, coefficient] : list) {
                sop.add(sqop, coefficient);
            }
            return sop;
        },
        "list"_a,
        "Create a SparseOperatorList object from a list of Tuple[SQOperatorString, complex]");
}
} // namespace forte
