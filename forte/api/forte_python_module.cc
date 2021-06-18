/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see LICENSE, AUTHORS).
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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers/printing.h"
#include "helpers/lbfgs/rosenbrock.h"
#include "helpers/symmetry.h"

#include "base_classes/active_space_solver.h"
#include "base_classes/orbital_transform.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"

#include "integrals/make_integrals.h"

#include "orbital-helpers/aosubspace.h"
#include "orbital-helpers/localize.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "orbital-helpers/orbital_embedding.h"
#include "orbital-helpers/fragment_projector.h"

#include "forte.h"

#include "casscf/casscf.h"
#include "casscf/mcscf_2step.h"
#include "fci/fci_solver.h"
#include "mrdsrg-helper/run_dsrg.h"
#include "mrdsrg-spin-integrated/master_mrdsrg.h"
#include "mrdsrg-spin-adapted/sadsrg.h"
#include "mrdsrg-spin-integrated/mcsrgpt2_mo.h"

#include "post_process/spin_corr.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

// see the files in src/api for the implementation of the following methods
void export_ambit(py::module& m);
void export_ForteIntegrals(py::module& m);
void export_ForteOptions(py::module& m);
void export_MOSpaceInfo(py::module& m);
void export_RDMs(py::module& m);
void export_Determinant(py::module& m);
void export_StateInfo(py::module& m);
void export_SigmaVector(py::module& m);
void export_SparseCISolver(py::module& m);
void export_ForteCubeFile(py::module& m);
void export_OrbitalTransform(py::module& m);
void export_Localize(py::module& m);

void set_master_screen_threshold(double value);
double get_master_screen_threshold();

/// Export the ActiveSpaceMethod class
void export_ActiveSpaceMethod(py::module& m) {
    py::class_<ActiveSpaceMethod>(m, "ActiveSpaceMethod")
        .def("compute_energy", &ActiveSpaceMethod::compute_energy)
        .def("dump_wave_function", &ActiveSpaceMethod::dump_wave_function)
        .def("read_wave_function", &ActiveSpaceMethod::read_wave_function);
}

void export_ActiveSpaceSolver(py::module& m) {
    py::class_<ActiveSpaceSolver>(m, "ActiveSpaceSolver")
        .def("compute_energy", &ActiveSpaceSolver::compute_energy)
        .def("rdms", &ActiveSpaceSolver::rdms)
        .def("compute_contracted_energy", &ActiveSpaceSolver::compute_contracted_energy,
             "as_ints"_a, "max_body"_a,
             "Solve the contracted CI eigenvalue problem using given integrals")
        .def("compute_average_rdms", &ActiveSpaceSolver::compute_average_rdms,
             "Compute the weighted average reference")
        .def("set_active_space_integrals", &ActiveSpaceSolver::set_active_space_integrals,
             "Set the active space integrals manually")
        .def("compute_fosc_same_orbs", &ActiveSpaceSolver::compute_fosc_same_orbs)
        .def("state_filename_map", &ActiveSpaceSolver::state_filename_map)
        .def("dump_wave_function", &ActiveSpaceSolver::dump_wave_function);

    m.def("compute_average_state_energy", &compute_average_state_energy,
          "Compute the average energy given the energies and weights of each state");
}

void export_CASSCF(py::module& m) {
    py::class_<CASSCF>(m, "CASSCF")
        .def("compute_energy", &CASSCF::compute_energy, "Compute the CASSCF energy")
        .def("compute_gradient", &CASSCF::compute_gradient, "Compute the CASSCF gradient");
}

void export_MCSCF_2STEP(py::module& m) {
    py::class_<MCSCF_2STEP>(m, "MCSCF_2STEP")
        .def("compute_energy", &MCSCF_2STEP::compute_energy, "Compute the MCSCF energy");
}

void export_Symmetry(py::module& m) {
    py::class_<Symmetry>(m, "Symmetry")
        .def(py::init<std::string>())
        .def("point_group_label", &Symmetry::point_group_label,
             "Returns the label of this point group")
        .def("irrep_labels", &Symmetry::irrep_labels, "Returns a vector of irrep labels")
        .def("irrep_label", &Symmetry::irrep_label, "h"_a, "Returns the label of irrep ``h``")
        .def("irrep_label_to_index", &Symmetry::irrep_label_to_index, "label"_a,
             "Returns the index of a given irrep ``label``")
        .def("nirrep", &Symmetry::nirrep, "Returns the number of irreps")
        .def(
            "__repr__",
            [](const Symmetry& sym) { return "Symmetry(" + sym.point_group_label() + ")"; },
            "Returns a representation of this object")
        .def(
            "__str__", [](const Symmetry& sym) { return sym.point_group_label(); },
            "Returns a string representation of this object")
        .def_static("irrep_product", &Symmetry::irrep_product, "h"_a, "g"_a,
                    "Returns the product of irreps ``h`` and ``g``");
}

// TODO: export more classes using the function above
PYBIND11_MODULE(forte, m) {
    m.doc() = "pybind11 Forte module"; // module docstring
    m.def("startup", &startup);
    m.def("cleanup", &cleanup);
    m.def("banner", &banner, "Print forte banner");
    m.def("print_method_banner", &print_method_banner, "text"_a, "separator"_a = "-",
          "Print a method banner");
    m.def("make_mo_space_info", &make_mo_space_info, "Make a MOSpaceInfo object");
    m.def("make_mo_space_info_from_map", &make_mo_space_info_from_map, "nmopi"_a, "point_group"_a,
          "mo_space_map"_a, "reorder"_a = std::vector<size_t>(),
          "Make a MOSpaceInfo object using a dictionary");
    m.def("make_aosubspace_projector", &make_aosubspace_projector, "Make a AOSubspace projector");
    m.def("make_avas", &make_avas, "Make AVAS orbitals");
    m.def("make_fragment_projector", &make_fragment_projector,
          "Make a fragment(embedding) projector");
    m.def("make_embedding", &make_embedding, "Apply fragment projector to embed");
    m.def("make_ints_from_psi4", &make_forte_integrals_from_psi4,
          "Make Forte integral object from psi4");
    m.def("make_custom_ints", &make_custom_forte_integrals, "Make a custom integral object");
    m.def("make_active_space_method", &make_active_space_method, "Make an active space method");
    m.def("make_active_space_solver", &make_active_space_solver, "Make an active space solver");
    m.def("make_orbital_transformation", &make_orbital_transformation,
          "Make an orbital transformation");
    m.def("make_state_info_from_psi", &make_state_info_from_psi,
          "Make a state info object from a psi4 Wavefunction");
    m.def("to_state_nroots_map", &to_state_nroots_map,
          "Convert a map of StateInfo to weight lists to a map of StateInfo to number of "
          "states.");
    m.def("make_state_weights_map", &make_state_weights_map,
          "Make a list of target states with their weigth");
    m.def("make_active_space_ints", &make_active_space_ints,
          "Make an object that holds the molecular orbital integrals for the active orbitals");
    m.def("make_dynamic_correlation_solver", &make_dynamic_correlation_solver,
          "Make a dynamical correlation solver");
    m.def("perform_spin_analysis", &perform_spin_analysis, "Do spin analysis");
    m.def("make_dsrg_method", &make_dsrg_method,
          "Make a DSRG method (spin-integrated implementation)");
    m.def("make_sadsrg_method", &make_sadsrg_method,
          "Make a DSRG method (spin-adapted implementation)");
    m.def("make_dsrg_so_y", &make_dsrg_so_y, "Make a DSRG pointer (spin-orbital implementation)");
    m.def("make_dsrg_so_f", &make_dsrg_so_f, "Make a DSRG pointer (spin-orbital implementation)");
    m.def("make_dsrg_spin_adapted", &make_dsrg_spin_adapted,
          "Make a DSRG pointer (spin-adapted implementation)");
    m.def("make_casscf", &make_casscf, "Make a CASSCF object");
    m.def("make_mcscf_two_step", &make_mcscf_two_step, "Make a 2-step MCSCF object");
    m.def("test_lbfgs_rosenbrock", &test_lbfgs_rosenbrock, "Test L-BFGS on Rosenbrock function");

    export_ambit(m);

    export_ForteOptions(m);

    export_ActiveSpaceMethod(m);
    export_ActiveSpaceSolver(m);

    export_CASSCF(m);
    export_MCSCF_2STEP(m);
    export_ForteIntegrals(m);

    export_Symmetry(m);
    export_OrbitalTransform(m);
    export_Localize(m);

    export_Determinant(m);

    export_RDMs(m);

    export_StateInfo(m);

    export_SigmaVector(m);
    export_SparseCISolver(m);

    export_ForteCubeFile(m);

    export_MOSpaceInfo(m);

    // export SCFInfo
    py::class_<SCFInfo, std::shared_ptr<SCFInfo>>(m, "SCFInfo")
        .def(py::init<psi::SharedWavefunction>())
        .def(py::init<const psi::Dimension&, const psi::Dimension&, const psi::Dimension&, double,
                      std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Vector>>())
        .def("nmopi", &SCFInfo::nmopi, "the number of orbitals per irrep")
        .def("doccpi", &SCFInfo::doccpi, "the number of doubly occupied orbitals per irrep")
        .def("soccpi", &SCFInfo::soccpi, "the number of singly occupied orbitals per irrep")
        .def("reference_energy", &SCFInfo::reference_energy, "the reference energy")
        .def("epsilon_a", &SCFInfo::epsilon_a, "a vector of alpha orbital energy (psi::Vector)")
        .def("epsilon_b", &SCFInfo::epsilon_b, "a vector of beta orbital energy (psi::Vector)");

    // export DynamicCorrelationSolver
    py::class_<DynamicCorrelationSolver, std::shared_ptr<DynamicCorrelationSolver>>(
        m, "DynamicCorrelationSolver")
        .def("compute_energy", &DynamicCorrelationSolver::compute_energy);

    // export ActiveSpaceIntegrals
    py::class_<ActiveSpaceIntegrals, std::shared_ptr<ActiveSpaceIntegrals>>(m,
                                                                            "ActiveSpaceIntegrals")
        .def("slater_rules", &ActiveSpaceIntegrals::slater_rules,
             "Compute the matrix element of the Hamiltonian between two determinants")
        .def("nuclear_repulsion_energy", &ActiveSpaceIntegrals::nuclear_repulsion_energy,
             "Get the nuclear repulsion energy")
        .def("frozen_core_energy", &ActiveSpaceIntegrals::frozen_core_energy,
             "Get the frozen core energy (contribution from FROZEN_DOCC)")
        .def("scalar_energy", &ActiveSpaceIntegrals::scalar_energy,
             "Get the scalar_energy energy (contribution from RESTRICTED_DOCC)")
        .def("nmo", &ActiveSpaceIntegrals::nmo, "Get the number of active orbitals")
        .def("oei_a", &ActiveSpaceIntegrals::oei_a, "Get the alpha effective one-electron integral")
        .def("oei_b", &ActiveSpaceIntegrals::oei_b, "Get the beta effective one-electron integral")
        .def("tei_aa", &ActiveSpaceIntegrals::tei_aa, "alpha-alpha two-electron integral <pq||rs>")
        .def("tei_ab", &ActiveSpaceIntegrals::tei_ab, "alpha-beta two-electron integral <pq|rs>")
        .def("tei_bb", &ActiveSpaceIntegrals::tei_bb, "beta-beta two-electron integral <pq||rs>")
        .def("print", &ActiveSpaceIntegrals::print, "Print the integrals (alpha-alpha case)");

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
             "Set active part orbital rotation matrix (from original to semicanonical)")
        .def("set_read_cwd_amps", &MASTER_DSRG::set_read_amps_cwd,
             "Set if reading amplitudes in the current directory or not")
        .def("clean_checkpoints", &MASTER_DSRG::clean_checkpoints,
             "Delete amplitudes checkpoint files");

    // export SADSRG
    py::class_<SADSRG>(m, "SADSRG")
        .def("compute_energy", &SADSRG::compute_energy, "Compute the DSRG energy")
        .def("compute_Heff_actv", &SADSRG::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals")
        .def("set_Uactv", &SADSRG::set_Uactv, "Ua"_a,
             "Set active part orbital rotation matrix (from original to semicanonical)")
        .def("set_read_cwd_amps", &SADSRG::set_read_amps_cwd,
             "Set if reading amplitudes in the current directory or not")
        .def("clean_checkpoints", &SADSRG::clean_checkpoints, "Delete amplitudes checkpoint files");

    // export MRDSRG_SO
    py::class_<MRDSRG_SO>(m, "MRDSRG_SO")
        .def("compute_energy", &MRDSRG_SO::compute_energy, "Compute DSRG energy")
        .def("compute_Heff_actv", &MRDSRG_SO::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals");

    // export SOMRDSRG
    py::class_<SOMRDSRG>(m, "SOMRDSRG")
        .def("compute_energy", &SOMRDSRG::compute_energy, "Compute DSRG energy")
        .def("compute_Heff_actv", &SOMRDSRG::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals");

    // export DSRG_MRPT spin-adapted code
    py::class_<DSRG_MRPT>(m, "DSRG_MRPT")
        .def("compute_energy", &DSRG_MRPT::compute_energy, "Compute DSRG energy")
        .def("compute_Heff_actv", &DSRG_MRPT::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals");

    py::class_<MCSRGPT2_MO>(m, "MCSRGPT2_MO")
        .def(py::init<RDMs, std::shared_ptr<ForteOptions>, std::shared_ptr<ForteIntegrals>,
                      std::shared_ptr<MOSpaceInfo>>())
        .def("compute_energy", &MCSRGPT2_MO::compute_energy, "Compute DSRG energy");

    // export DressedQuantity for dipole moments
    py::class_<DressedQuantity>(m, "DressedQuantity")
        .def("contract_with_rdms", &DressedQuantity::contract_with_rdms, "reference"_a,
             "Contract densities with quantity");
}

} // namespace forte
