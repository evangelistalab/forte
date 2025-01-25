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

namespace forte {

// see the files in src/api for the implementation of the following methods
void export_ndarray(py::module& m);

// Base classes
void export_ForteIntegrals(py::module& m);
void export_ForteOptions(py::module& m);
void export_MOSpaceInfo(py::module& m);
void export_StateInfo(py::module& m);
void export_RDMs(py::module& m);
void export_SCFInfo(py::module& m);

// Helpers
void export_Symmetry(py::module& m);

// ActiveSpaceSolver
void export_ActiveSpaceIntegrals(py::module& m);
void export_ActiveSpaceMethod(py::module& m);
void export_ActiveSpaceSolver(py::module& m);
void export_GenCIStringLists(py::module& m);
void export_GenCIVector(py::module& m);

// MCSCF
void export_MCSCF(py::module& m);

// Determinant, Configuration, String
void export_Determinant(py::module& m);
void export_String(py::module& m);
void export_Configuration(py::module& m);
void export_CI_RDMS(py::module& m);

// Sparse Operators, States, and related classes
void export_SQOperatorString(py::module& m);
void export_SparseExp(py::module& m);
void export_SparseFactExp(py::module& m);
void export_SparseOperator(py::module& m);
void export_SparseOperatorList(py::module& m);
void export_SparseOperatorSimTrans(py::module& m);
void export_SparseState(py::module& m);
void export_SigmaVector(py::module& m);
void export_SparseCISolver(py::module& m);

// Additional classes
void export_ForteCubeFile(py::module& m);
void export_OrbitalTransform(py::module& m);
void export_Localize(py::module& m);
void export_SemiCanonical(py::module& m);
void export_DavidsonLiuSolver(py::module& m);

} // namespace forte
