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

#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"

#include "sparse_ci/sigma_vector.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "integrals/active_space_integrals.h"

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
    py::class_<Determinant>(m, "Determinant")
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

    m.def(
        "det",
        [](const std::string& s) {
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
        "Make a determinant from a string (e.g., \'2+-0\'). 2 or x/X = doubly occupied MO, + or "
        "a/A = alpha, - or b/B = beta. Orbital occupations are read from left to right.");

    m.def(
        "spin2", [](const Determinant& lhs, const Determinant& rhs) { return spin2(lhs, rhs); },
        "Compute a matrix element of the S^2 operator");

    py::class_<DeterminantHashVec>(m, "DeterminantHashVec")
        .def(py::init<>())
        .def(py::init<const std::vector<Determinant>&>())
        .def(py::init<const det_hashvec&>())
        .def("add", &DeterminantHashVec::add, "Add a determinant")
        .def("size", &DeterminantHashVec::size, "Get the size of the vector")
        .def("get_det", &DeterminantHashVec::get_det, "Return a specific determinant by reference")
        .def("get_idx", &DeterminantHashVec::get_idx, " Return the index of a determinant");

    py::class_<SparseOperator>(m, "SparseOperator")
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
        .def("latex", &SparseOperator::latex);

    py::class_<SQOperator>(m, "SQOperator")
        .def("coefficient", &SQOperator::coefficient)
        .def("cre", &SQOperator::cre)
        .def("ann", &SQOperator::ann)
        .def("str", &SQOperator::str)
        .def("latex", &SQOperator::latex);

    py::class_<StateVector>(m, "StateVector")
        .def(py::init<const det_hash<double>&>())
        // .def("map", &StateVector::map)
        .def(
            "items", [](const StateVector& v) { return py::make_iterator(v.begin(), v.end()); },
            py::keep_alive<0, 1>()) // Essential: keep object alive while iterator exists
        // .def("items",  [](StateVector& v) { return v.map(); })
        .def("str", &StateVector::str)
        .def("__eq__", &StateVector::operator==)
        .def("__repr__", [](const StateVector& v) { return v.str(); })
        .def("__str__", [](const StateVector& v) { return v.str(); })
        .def("__getitem__", [](StateVector& v, const Determinant& d) { return v[d]; })
        .def("__setitem__",
             [](StateVector& v, const Determinant& d, const double val) { v[d] = val; })
        .def("__contains__", [](StateVector& v, const Determinant& d) { return v.map().count(d); });

    py::class_<SparseHamiltonian>(m, "SparseHamiltonian")
        .def(py::init<std::shared_ptr<ActiveSpaceIntegrals>>())
        .def("compute", &SparseHamiltonian::compute)
        .def("compute_on_the_fly", &SparseHamiltonian::compute_on_the_fly)
        .def("timings", &SparseHamiltonian::timings);

    py::class_<SparseExp>(m, "SparseExp")
        .def(py::init<>())
        .def("compute", &SparseExp::compute, "sop"_a, "state"_a, "algorithm"_a = "cached",
             "scaling_factor"_a = 1.0, "maxk"_a = 19, "screen_thresh"_a = 1.0e-12)
        .def("timings", &SparseExp::timings);

    py::class_<SparseFactExp>(m, "SparseFactExp")
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
    py::class_<SparseCISolver, std::shared_ptr<SparseCISolver>>(m, "SparseCISolver")
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
