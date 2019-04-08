/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see LICENSE, AUTHORS).
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

#ifndef _python_api_h_
#define _python_api_h_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"

#include "base_classes/active_space_solver.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/orbital_transform.h"
#include "integrals/integrals.h"
#include "integrals/make_integrals.h"

#include "orbital-helpers/aosubspace.h"
#include "orbital-helpers/localize.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "orbital-helpers/avas.h"

#include "forte.h"
#include "fci/fci_solver.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"
#include "mrdsrg-helper/run_dsrg.h"
#include "mrdsrg-spin-integrated/master_mrdsrg.h"
#include "sparse_ci/determinant.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

/// Export the ForteOptions class
void export_ForteOptions(py::module& m) {
    py::class_<ForteOptions, std::shared_ptr<ForteOptions>>(m, "ForteOptions")
        .def(py::init<>())
        .def(py::init<psi::Options&>())
        .def("add_bool", &ForteOptions::add_bool, "Add a boolean option")
        .def("add_int", &ForteOptions::add_int, "Add an integer option")
        .def("add_double", &ForteOptions::add_double, "Add a double option")
        .def("add_str",
             (void (ForteOptions::*)(const std::string&, const std::string&, const std::string&)) &
                 ForteOptions::add_str,
             "Add a string option")
        .def("add_str",
             (void (ForteOptions::*)(const std::string&, const std::string&,
                                     const std::vector<std::string>&, const std::string&)) &
                 ForteOptions::add_str,
             "Add a string option")
        .def("add_array", &ForteOptions::add_array, "Add an array option")
        .def("get_bool", &ForteOptions::get_bool, "Get a boolean option")
        .def("get_int", &ForteOptions::get_int, "Get an integer option")
        .def("get_double", &ForteOptions::get_double, "Get a double option")
        .def("get_str", &ForteOptions::get_str, "Get a string option")
        .def("get_int_vec", &ForteOptions::get_int_vec, "Get a vector of integers option")
        .def("push_options_to_psi4", &ForteOptions::push_options_to_psi4)
        .def("update_psi_options", &ForteOptions::update_psi_options)
        .def("generate_documentation", &ForteOptions::generate_documentation);
}

/// Export the ActiveSpaceMethod class
void export_ActiveSpaceMethod(py::module& m) {
    py::class_<ActiveSpaceMethod>(m, "ActiveSpaceMethod")
        .def("compute_energy", &ActiveSpaceMethod::compute_energy);
}

void export_ActiveSpaceSolver(py::module& m) {
    py::class_<ActiveSpaceSolver>(m, "ActiveSpaceSolver")
        .def("compute_energy", &ActiveSpaceSolver::compute_energy)
        .def("rdms", &ActiveSpaceSolver::rdms)
        .def("compute_average_rdms", &ActiveSpaceSolver::compute_average_rdms)
        .def("compute_contracted_energy", &ActiveSpaceSolver::compute_contracted_energy,
             "as_ints"_a, "max_body"_a,
             "Solve the contracted CI eigenvalue problem using given integrals")
        .def("compute_average_rdms", &ActiveSpaceSolver::compute_average_rdms,
             "Compute the weighted average reference");

    m.def("compute_average_state_energy", &compute_average_state_energy,
          "Compute the average energy given the energies and weights of each state");
}

/// Export the OrbitalTransform class
void export_OrbitalTransform(py::module& m) {
    py::class_<OrbitalTransform>(m, "OrbitalTransform")
        .def("compute_transformation", &OrbitalTransform::compute_transformation)
        .def("get_Ua", &OrbitalTransform::get_Ua, "Get Ua rotation")
        .def("get_Ub", &OrbitalTransform::get_Ub, "Get Ub rotation");
}

constexpr int Determinant::num_str_bits;
constexpr int Determinant::num_det_bits;

/// Export the Determinant class
void export_Determinant(py::module& m) {
    py::class_<Determinant>(m, "Determinant")
        .def(py::init<>())
        .def(py::init<const std::vector<bool>&, const std::vector<bool>&>())
        .def("get_alfa_bits", &Determinant::get_alfa_bits, "Get alpha bits")
        .def("get_beta_bits", &Determinant::get_beta_bits, "Get beta bits")
        .def_readonly_static("num_str_bits", &Determinant::num_str_bits)
        .def_readonly_static("num_det_bits", &Determinant::num_det_bits)
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
        .def("str", &Determinant::str, "Get the string representation of the Slater determinant");
}

// TODO: export more classes using the function above
PYBIND11_MODULE(forte, m) {
    m.doc() = "pybind11 Forte module"; // module docstring
    m.def("read_options", &read_options, "Read Forte options");
    m.def("startup", &startup);
    m.def("cleanup", &cleanup);
    m.def("banner", &banner, "Print forte banner");
    m.def("make_mo_space_info", &make_mo_space_info, "Make a MOSpaceInfo object");
    m.def("make_aosubspace_projector", &make_aosubspace_projector, "Make a AOSubspace projector");
    m.def("make_avas", &make_avas, "Make AVAS orbitals");
    m.def("make_forte_integrals", &make_forte_integrals, "Make Forte integrals");
    m.def("forte_old_methods", &forte_old_methods, "Run Forte methods");
    m.def("make_active_space_method", &make_active_space_method, "Make an active space method");
    m.def("make_active_space_solver", &make_active_space_solver, "Make an active space solver");
    m.def("make_orbital_transformation", &make_orbital_transformation,
          "Make an orbital transformation");
    m.def("make_state_info_from_psi_wfn", &make_state_info_from_psi_wfn,
          "Make a state info object from a psi4 Wavefunction");
    m.def("to_state_nroots_map", &to_state_nroots_map,
          "Convert a map of StateInfo to weight lists to a map of StateInfo to number of states.");
    m.def("make_state_weights_map", &make_state_weights_map,
          "Make a list of target states with their weigth");
    m.def("make_active_space_ints", &make_active_space_ints,
          "Make an object that holds the molecular orbital integrals for the active orbitals");
    m.def("make_dynamic_correlation_solver", &make_dynamic_correlation_solver,
          "Make a dynamical correlation solver");
    m.def("make_dsrg_method", &make_dsrg_method, "Make a DSRG method");

    export_ForteOptions(m);

    export_ActiveSpaceMethod(m);
    export_ActiveSpaceSolver(m);

    export_OrbitalTransform(m);

    export_Determinant(m);

    //    export_FCISolver(m);

    // export MOSpaceInfo
    py::class_<MOSpaceInfo, std::shared_ptr<MOSpaceInfo>>(m, "MOSpaceInfo")
        .def("size", &MOSpaceInfo::size);

    // export ForteIntegrals
    py::class_<ForteIntegrals, std::shared_ptr<ForteIntegrals>>(m, "ForteIntegrals")
        .def("rotate_orbitals", &ForteIntegrals::rotate_orbitals)
        .def("nmo", &ForteIntegrals::nmo)
        .def("ncmo", &ForteIntegrals::ncmo);

    // export StateInfo
    py::class_<StateInfo, std::shared_ptr<StateInfo>>(m, "StateInfo")
        .def(py::init<int, int, int, int, int>(), "na"_a, "nb"_a, "multiplicity"_a, "twice_ms"_a,
             "irrep"_a);

    // export SCFInfo
    py::class_<SCFInfo, std::shared_ptr<SCFInfo>>(m, "SCFInfo")
        .def(py::init<psi::SharedWavefunction>());

    // export DynamicCorrelationSolver
    py::class_<DynamicCorrelationSolver, std::shared_ptr<DynamicCorrelationSolver>>(
        m, "DynamicCorrelationSolver")
        .def("compute_energy", &DynamicCorrelationSolver::compute_energy);

    // export ActiveSpaceIntegrals
    py::class_<ActiveSpaceIntegrals, std::shared_ptr<ActiveSpaceIntegrals>>(m,
                                                                            "ActiveSpaceIntegrals")
        .def(py::init<std::shared_ptr<ForteIntegrals>, std::shared_ptr<MOSpaceInfo>>())
        .def("slater_rules", &ActiveSpaceIntegrals::slater_rules,
             "Compute the matrix element of the Hamiltonian between two determinants")
        .def("nuclear_repulsion_energy", &ActiveSpaceIntegrals::nuclear_repulsion_energy,
             "Get the nuclear repulsion energy")
        .def("frozen_core_energy", &ActiveSpaceIntegrals::frozen_core_energy,
             "Get the frozen core energy (contribution from FROZEN_DOCC)")
        .def("scalar_energy", &ActiveSpaceIntegrals::scalar_energy,
             "Get the scalar_energy energy (contribution from RESTRICTED_DOCC)");

    // export SemiCanonical
    py::class_<SemiCanonical>(m, "SemiCanonical")
        .def(py::init<std::shared_ptr<MOSpaceInfo>, std::shared_ptr<ForteIntegrals>,
                      std::shared_ptr<ForteOptions>, bool>(),
             "mo_space_info"_a, "ints"_a, "options"_a, "quiet_banner"_a = false)
        .def("semicanonicalize", &SemiCanonical::semicanonicalize, "reference"_a,
             "max_rdm_level"_a = 3, "build_fock"_a = true, "transform"_a = true,
             "Semicanonicalize the orbitals and transform the integrals and reference")
        .def("transform_rdms", &SemiCanonical::transform_rdms, "Ua"_a, "Ub"_a, "reference"_a,
             "max_rdm_level"_a, "Transform the RDMs by input rotation matrices")
        .def("Ua_t", &SemiCanonical::Ua_t, "Return the alpha rotation matrix in the active space")
        .def("Ub_t", &SemiCanonical::Ub_t, "Return the beta rotation matrix in the active space");

    // export RDMs
    py::class_<RDMs>(m, "RDMs");

    // export ambit::Tensor
    py::class_<ambit::Tensor>(m, "ambitTensor");

    // export MASTER_DSRG
    py::class_<MASTER_DSRG>(m, "MASTER_DSRG")
        .def("compute_energy", &MASTER_DSRG::compute_energy, "Compute the DSRG energy")
        .def("compute_Heff_actv", &MASTER_DSRG::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals")
        .def("deGNO_DMbar_actv", &MASTER_DSRG::deGNO_DMbar_actv,
             "Return the DSRG dressed dipole integrals")
        .def("nuclear_dipole", &MASTER_DSRG::nuclear_dipole,
             "Return nuclear components of dipole moments")
        .def("set_Uactv", &MASTER_DSRG::set_Uactv, "Ua"_a, "Ub"_a,
             "Set active part orbital rotation matrix (from original to semicanonical)");

    // export DressedQuantity for dipole moments
    py::class_<DressedQuantity>(m, "DressedQuantity")
        .def("contract_with_rdms", &DressedQuantity::contract_with_rdms, "reference"_a,
             "Contract densities with quantity");
}

} // namespace forte

#endif // _python_api_h_
