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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"

#include "base_classes/active_space_solver.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/orbital_transform.h"
#include "integrals/make_integrals.h"

#include "helpers/printing.h"

#include "orbital-helpers/aosubspace.h"
#include "orbital-helpers/localize.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "orbital-helpers/orbital_embedding.h"
#include "orbital-helpers/fragment_projector.h"

#include "forte.h"
#include "fci/fci_solver.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"
#include "mrdsrg-helper/run_dsrg.h"
#include "mrdsrg-spin-integrated/master_mrdsrg.h"

#include "sparse_ci/determinant.h"
#include "post_process/spin_corr.h"
#include "sparse_ci/determinant_hashvector.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

// see the files in src/api for the implementation of the following methods
void export_ambit(py::module& m);
void export_ForteIntegrals(py::module& m);
void export_RDMs(py::module& m);
void export_StateInfo(py::module& m);
void export_SigmaVector(py::module& m);
void export_SparseCISolver(py::module& m);

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

/// Export the Determinant class
void export_Determinant(py::module& m) {
    py::class_<Determinant>(m, "Determinant")
        .def(py::init<>())
        .def(py::init<const Determinant&>())
        .def(py::init<const std::vector<bool>&, const std::vector<bool>&>())
        .def("get_alfa_bits", &Determinant::get_alfa_bits, "Get alpha bits")
        .def("get_beta_bits", &Determinant::get_beta_bits, "Get beta bits")
        .def("nbits", &Determinant::get_nbits)
        .def("nbits_half", &Determinant::get_nbits_half)
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
        .def(
            "gen_excitation",
            [](Determinant& d, const std::vector<int>& aann, const std::vector<int>& acre,
               const std::vector<int>& bann,
               const std::vector<int>& bcre) { return gen_excitation(d, aann, acre, bann, bcre); },
            "Apply a generic excitation")
        .def(
            "str", [](const Determinant& a, int n) { return str(a, n); }, "n"_a = 64,
            "Get the string representation of the Slater determinant")
        .def("__repr__", [](const Determinant& a) { return str(a); })
        .def("__str__", [](const Determinant& a) { return str(a); })
        .def("__eq__", [](const Determinant& a, const Determinant& b) { return a == b; })
        .def("__lt__", [](const Determinant& a, const Determinant& b) { return a < b; })
        .def("__hash__", [](const Determinant& a) { return Determinant::Hash()(a); });
    py::class_<DeterminantHashVec>(m, "DeterminantHashVec")
        .def(py::init<>())
        .def(py::init<const std::vector<Determinant>&>())
        .def(py::init<const det_hashvec&>())
        .def("add", &DeterminantHashVec::add, "Add a determinant")
        .def("size", &DeterminantHashVec::size, "Get the size of the vector")
        .def("get_det", &DeterminantHashVec::get_det, "Return a specific determinant by reference")
        .def("get_idx", &DeterminantHashVec::get_idx, " Return the index of a determinant");
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
    m.def("make_mo_space_info_from_map", &make_mo_space_info_from_map,
          "Make a MOSpaceInfo object from a map of space name (string) to a vector");
    m.def("make_aosubspace_projector", &make_aosubspace_projector, "Make a AOSubspace projector");
    m.def("make_avas", &make_avas, "Make AVAS orbitals");
    m.def("make_fragment_projector", &make_fragment_projector,
          "Make a fragment(embedding) projector");
    m.def("make_embedding", &make_embedding, "Apply fragment projector to embed");
    m.def("make_forte_integrals", &make_forte_integrals, "Make Forte integrals");
    m.def("forte_old_methods", &forte_old_methods, "Run Forte methods");
    m.def("make_active_space_method", &make_active_space_method, "Make an active space method");
    m.def("make_active_space_solver", &make_active_space_solver, "Make an active space solver");
    m.def("make_orbital_transformation", &make_orbital_transformation,
          "Make an orbital transformation");
    m.def("make_state_info_from_psi_wfn", &make_state_info_from_psi_wfn,
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
    m.def("make_dsrg_so_y", &make_dsrg_so_y, "Make a DSRG pointer (spin-orbital implementation)");
    m.def("make_dsrg_so_f", &make_dsrg_so_f, "Make a DSRG pointer (spin-orbital implementation)");
    m.def("make_dsrg_spin_adapted", &make_dsrg_spin_adapted,
          "Make a DSRG pointer (spin-adapted implementation)");

    export_ambit(m);

    export_ForteOptions(m);

    export_ActiveSpaceMethod(m);
    export_ActiveSpaceSolver(m);
    export_ForteIntegrals(m);

    export_OrbitalTransform(m);

    export_Determinant(m);

    export_RDMs(m);

    export_StateInfo(m);

    export_SigmaVector(m);
    export_SparseCISolver(m);

    // export MOSpaceInfo
    py::class_<MOSpaceInfo, std::shared_ptr<MOSpaceInfo>>(m, "MOSpaceInfo")
        .def("dimension", &MOSpaceInfo::dimension,
             "Return a psi::Dimension object for the given space")
        .def("absolute_mo", &MOSpaceInfo::absolute_mo,
             "Return the list of the absolute index of the molecular orbitals in a space "
             "excluding "
             "the frozen core/virtual orbitals")
        .def("corr_absolute_mo", &MOSpaceInfo::corr_absolute_mo,
             "Return the list of the absolute index of the molecular orbitals in a correlated "
             "space")
        .def("get_relative_mo", &MOSpaceInfo::get_relative_mo, "Return the relative MOs")
        .def("read_options", &MOSpaceInfo::read_options, "Read options")
        .def("read_from_map", &MOSpaceInfo::read_from_map,
             "Read the space info from a map {spacename -> dimension vector}")
        .def("set_reorder", &MOSpaceInfo::set_reorder,
             "Reorder MOs according to the input indexing vector")
        .def("compute_space_info", &MOSpaceInfo::compute_space_info,
             "Processing current MOSpaceInfo: calculate frozen core, count and assign orbitals")
        .def("size", &MOSpaceInfo::size, "Return the number of orbitals in a space")
        .def("nirrep", &MOSpaceInfo::nirrep, "Return the number of irreps")
        .def("symmetry", &MOSpaceInfo::symmetry, "Return the symmetry of each orbital")
        .def("space_names", &MOSpaceInfo::space_names, "Return the names of orbital spaces");

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
        .def("slater_rules", &ActiveSpaceIntegrals::slater_rules,
             "Compute the matrix element of the Hamiltonian between two determinants")
        .def("nuclear_repulsion_energy", &ActiveSpaceIntegrals::nuclear_repulsion_energy,
             "Get the nuclear repulsion energy")
        .def("frozen_core_energy", &ActiveSpaceIntegrals::frozen_core_energy,
             "Get the frozen core energy (contribution from FROZEN_DOCC)")
        .def("scalar_energy", &ActiveSpaceIntegrals::scalar_energy,
             "Get the scalar_energy energy (contribution from RESTRICTED_DOCC)")
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
             "Set active part orbital rotation matrix (from original to semicanonical)");

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

    // export DressedQuantity for dipole moments
    py::class_<DressedQuantity>(m, "DressedQuantity")
        .def("contract_with_rdms", &DressedQuantity::contract_with_rdms, "reference"_a,
             "Contract densities with quantity");
}

} // namespace forte
