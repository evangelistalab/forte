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

#include "sparse_ci/sparse_operator.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

/// Export the Determinant class
void export_SparseOperatorSimTrans(py::module& m) {
    m.def(
        "fact_trans_lin",
        [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
            // time this call and print to std::cout
            fact_trans_lin(O, T, reverse, screen_thresh);
        },
        "O"_a, "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
        "Evaluate ... (1 - T1) O (1 + T1) ...");

    m.def(
        "fact_unitary_trans_antiherm",
        [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
            // time this call and print to std::cout
            fact_unitary_trans_antiherm(O, T, reverse, screen_thresh);
        },
        "O"_a, "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
        "Evaluate ... exp(T1^dagger - T1) O exp(T1 - T1^dagger) ...");

    m.def(
        "fact_unitary_trans_antiherm_grad",
        [](SparseOperator& O, const SparseOperatorList& T, size_t n, bool reverse,
           double screen_thresh) {
            fact_unitary_trans_antiherm_grad(O, T, n, reverse, screen_thresh);
        },
        "O"_a, "T"_a, "n"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
        "Evaluate the gradient of ... exp(T1^dagger - T1) O exp(T1 - T1^dagger) ...");

    m.def(
        "fact_unitary_trans_imagherm",
        [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double screen_thresh) {
            fact_unitary_trans_imagherm(O, T, reverse, screen_thresh);
        },
        "O"_a, "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
        "Evaluate ... exp(i (T1^dagger + T1)) O exp(-i(T1 + T1^dagger)) ...");
}
} // namespace forte
