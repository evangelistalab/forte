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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/lbfgs/rosenbrock.h"
#include "helpers/symmetry.h"
#include "helpers/spinorbital_helpers.h"

#include "base_classes/active_space_solver.h"
#include "base_classes/orbital_transform.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"

#include "integrals/make_integrals.h"

#include "orbital-helpers/aosubspace.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/orbital_embedding.h"
#include "orbital-helpers/fragment_projector.h"

#include "forte.h"

#include "mcscf/mcscf_2step.h"
#include "fci/fci_solver.h"
#include "mrdsrg-helper/run_dsrg.h"
#include "mrdsrg-spin-adapted/sadsrg.h"
#include "mrdsrg-spin-adapted/sa_mrpt2.h"
#include "mrdsrg-spin-integrated/master_mrdsrg.h"
#include "integrals/one_body_integrals.h"
#include "sci/tdci.h"
#include "genci/ci_occupation.h"

#include "post_process/spin_corr.h"

#include "forte_python_module.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace forte {

void set_master_screen_threshold(double value);
double get_master_screen_threshold();

// TODO: export more classes using the function above
PYBIND11_MODULE(_forte, m) {

    // This line is how pb11 knows what pieces of ambit have already been exposed,
    // and can be sent Py-side by Forte.
    py::module::import("ambit");

    m.doc() = "pybind11 Forte module"; // module docstring

    // export base classes
    export_ndarray(m);

    export_ActiveSpaceIntegrals(m);

    m.def("startup", &startup);
    m.def("cleanup", &cleanup);
    m.def("banner", &banner, "Print forte banner");
    m.def("print_method_banner", &print_method_banner, "text"_a, "separator"_a = "-",
          "Print a method banner");
    m.def("make_aosubspace_projector", &make_aosubspace_projector, "Make a AOSubspace projector");
    m.def("make_avas", &make_avas, "Make AVAS orbitals");
    m.def("make_fragment_projector", &make_fragment_projector,
          "Make a fragment(embedding) projector");
    m.def("make_embedding", &make_embedding, "Apply fragment projector to embed");
    m.def("make_custom_ints", &make_custom_forte_integrals,
          "Make a custom Forte integral object from arrays");
    m.def("make_ints_from_psi4", &make_forte_integrals_from_psi4, "ref_wfn"_a, "options"_a,
          "scf_info"_a, "mo_space_info"_a, "int_type"_a = "",
          "Make a Forte integral object from psi4");
    m.def("make_active_space_method", &make_active_space_method, "Make an active space method");
    m.def("make_active_space_solver", &make_active_space_solver, "Make an active space solver",
          "method"_a, "state_nroots_map"_a, "scf_info"_a, "mo_space_info"_a, "options"_a,
          "as_ints"_a = std::shared_ptr<ActiveSpaceIntegrals>());

    m.def("make_orbital_transformation", &make_orbital_transformation,
          "Make an orbital transformation");
    m.def("make_state_info_from_options", &make_state_info_from_options,
          "Make a state info object from ForteOptions");
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
    m.def("make_dsrg_so", &make_dsrg_so, "Make a DSRG pointer (spin-orbital implementation)");
    m.def("make_dsrg_spin_adapted", &make_dsrg_spin_adapted,
          "Make a DSRG pointer (spin-adapted implementation)");

    m.def("make_mcscf_two_step", &make_mcscf_two_step, "Make a 2-step MCSCF object");
    m.def("make_mcscf", &make_mcscf_two_step, "Make a 2-step MCSCF object");
    m.def("test_lbfgs_rosenbrock", &test_lbfgs_rosenbrock, "Test L-BFGS on Rosenbrock function");

    m.def(
        "spinorbital_oei",
        [](const std::shared_ptr<ForteIntegrals> ints, const std::vector<size_t>& p,
           const std::vector<size_t>& q) { return ambit_to_np(spinorbital_oei(ints, p, q)); },
        "Compute the one-electron integrals in a spinorbital basis. Spinorbitals follow the "
        "ordering abab...");
    m.def(
        "spinorbital_tei",
        [](const std::shared_ptr<ForteIntegrals> ints, const std::vector<size_t>& p,
           const std::vector<size_t>& q, const std::vector<size_t>& r,
           const std::vector<size_t>& s) { return ambit_to_np(spinorbital_tei(ints, p, q, r, s)); },
        "Compute the two-electron integrals in a spinorbital basis. Spinorbitals follow the "
        "ordering abab...");
    m.def(
        "spinorbital_fock",
        [](const std::shared_ptr<ForteIntegrals> ints, const std::vector<size_t>& p,
           const std::vector<size_t>& q, const std::vector<size_t>& occ) {
            return ambit_to_np(spinorbital_fock(ints, p, q, occ));
        },
        "Compute the fock matrix in a spinorbital basis. Spinorbitals follow the ordering abab...");

    m.def(
        "spinorbital_rdms",
        [](std::shared_ptr<RDMs> rdms) {
            auto sordms = spinorbital_rdms(rdms);
            std::vector<py::array_t<double>> pysordms;
            for (const auto& sordm : sordms) {
                pysordms.push_back(ambit_to_np(sordm));
            }
            return pysordms;
        },
        "Return the RDMs in a spinorbital basis. Spinorbitals follow the ordering abab...");

    m.def(
        "spinorbital_cumulants",
        [](std::shared_ptr<RDMs> rdms) {
            auto sordms = spinorbital_cumulants(rdms);
            std::vector<py::array_t<double>> pysordms;
            for (const auto& sordm : sordms) {
                pysordms.push_back(ambit_to_np(sordm));
            }
            return pysordms;
        },
        "Return the cumulants of the RDMs in a spinorbital basis. Spinorbitals follow the ordering "
        "abab...");
    m.def("get_gas_occupation", &get_gas_occupation);
    m.def("get_ci_occupation_patterns", &get_ci_occupation_patterns);

    py::enum_<PrintLevel>(m, "PrintLevel")
        .value("Quiet", PrintLevel::Quiet)
        .value("Brief", PrintLevel::Brief)
        .value("Default", PrintLevel::Default)
        .value("Verbose", PrintLevel::Verbose)
        .value("Debug", PrintLevel::Debug)
        .export_values();

    export_ForteOptions(m);

    export_ActiveSpaceMethod(m);
    export_ActiveSpaceSolver(m);

    export_MCSCF(m);
    export_ForteIntegrals(m);

    export_Symmetry(m);
    export_OrbitalTransform(m);
    export_Localize(m);
    export_SemiCanonical(m);

    export_Determinant(m);
    export_Configuration(m);
    export_String(m);

    export_SQOperatorString(m);
    export_SparseExp(m);
    export_SparseFactExp(m);
    export_SparseHamiltonian(m);
    export_SparseOperator(m);
    export_SparseOperatorList(m);
    export_SparseOperatorSimTrans(m);
    export_SparseState(m);

    export_GenCIStringLists(m);
    export_GenCIVector(m);

    export_RDMs(m);

    export_StateInfo(m);

    export_SigmaVector(m);
    export_SparseCISolver(m);

    export_ForteCubeFile(m);

    export_MOSpaceInfo(m);

    export_DavidsonLiuSolver(m);

    export_SCFInfo(m);

    // export DynamicCorrelationSolver
    py::class_<DynamicCorrelationSolver, std::shared_ptr<DynamicCorrelationSolver>>(
        m, "DynamicCorrelationSolver")
        .def("compute_energy", &DynamicCorrelationSolver::compute_energy)
        .def("set_ci_vectors", &DynamicCorrelationSolver::set_ci_vectors,
             "Set the CI eigenvectors for DSRG-MRPT2 analytic gradients");

    // export MASTER_DSRG
    py::class_<MASTER_DSRG>(m, "MASTER_DSRG")
        .def("compute_energy", &MASTER_DSRG::compute_energy, "Compute the DSRG energy")
        .def("compute_gradient", &MASTER_DSRG::compute_gradient, "Compute the DSRG gradient")
        .def("compute_Heff_actv", &MASTER_DSRG::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals")
        .def("compute_Heff_full", [](MASTER_DSRG& self) {
            const auto Heff = self.compute_Heff_full();
            return py::make_tuple(blockedtensor_to_np(Heff.at(0)), blockedtensor_to_np(Heff.at(1)));
            })
        .def("compute_Heff_full_degno", [](MASTER_DSRG& self) {
            const auto Heff = self.compute_Heff_full_degno();
            return py::make_tuple(blockedtensor_to_np(Heff.at(0)), blockedtensor_to_np(Heff.at(1)));
            })
        .def("compute_Mbar0_full", &MASTER_DSRG::compute_Mbar0_full,
             "Return full transformed zero-body dipole integrals")
        .def("compute_Mbar1_full", [](MASTER_DSRG& self) {
            const auto Mbar1 = self.compute_Mbar1_full();
            return py::make_tuple(blockedtensor_to_np(Mbar1.at(0)), blockedtensor_to_np(Mbar1.at(1)),
                                  blockedtensor_to_np(Mbar1.at(2)));
            })
        .def("compute_Mbar2_full", [](MASTER_DSRG& self) {
            const auto Mbar2 = self.compute_Mbar2_full();
            return py::make_tuple(blockedtensor_to_np(Mbar2.at(0)), blockedtensor_to_np(Mbar2.at(1)),
                                  blockedtensor_to_np(Mbar2.at(2)));
            })
        .def("get_gamma1", [](MASTER_DSRG& self) {
            const auto gamma1 = self.get_gamma1();
            return blockedtensor_to_np(gamma1);
            })
        .def("get_eta1", [](MASTER_DSRG& self) {
            const auto eta1 = self.get_eta1();
            return blockedtensor_to_np(eta1);
            })
        .def("get_lambda2", [](MASTER_DSRG& self) {
               const auto lambda2 = self.get_lambda2();
               return blockedtensor_to_np(lambda2);
               })
        .def("get_lambda3", [](MASTER_DSRG& self) {
               const auto lambda3 = self.get_lambda3();
               py::dict pyrdm;
               pyrdm[py::str("aaaaaa")] = ambit_to_np(lambda3.at(0));
               pyrdm[py::str("aaAaaA")] = ambit_to_np(lambda3.at(1));
               pyrdm[py::str("aAAaAA")] = ambit_to_np(lambda3.at(2));
               pyrdm[py::str("AAAAAA")] = ambit_to_np(lambda3.at(3));
               return pyrdm;
               })
        .def("get_lambda4", [](MASTER_DSRG& self) {
               const auto lambda4 = self.get_lambda4();
               py::dict pyrdm;
               pyrdm[py::str("aaaaaaaa")] = ambit_to_np(lambda4.at(0));
               pyrdm[py::str("aaaAaaaA")] = ambit_to_np(lambda4.at(1));
               pyrdm[py::str("aaAAaaAA")] = ambit_to_np(lambda4.at(2));
               pyrdm[py::str("aAAAaAAA")] = ambit_to_np(lambda4.at(3));
               pyrdm[py::str("AAAAAAAA")] = ambit_to_np(lambda4.at(4));
               return pyrdm;
               })
        .def("deGNO_DMbar_actv", &MASTER_DSRG::deGNO_DMbar_actv,
             "Return the DSRG dressed dipole integrals")
        .def("nuclear_dipole", &MASTER_DSRG::nuclear_dipole,
             "Return nuclear components of dipole moments")
        .def("set_Uactv", &MASTER_DSRG::set_Uactv, "Ua"_a, "Ub"_a,
             "Set active part orbital rotation matrix (from original to semicanonical)")
        .def("set_active_space_solver", &MASTER_DSRG::set_active_space_solver,
             "Set the pointer of ActiveSpaceSolver")
        .def("set_state_weights_map", &MASTER_DSRG::set_state_weights_map,
             "Set the map from state to the weights of all computed roots")
        .def("set_read_cwd_amps", &MASTER_DSRG::set_read_amps_cwd,
             "Set if reading amplitudes in the current directory or not")
        .def("clean_checkpoints", &MASTER_DSRG::clean_checkpoints,
             "Delete amplitudes checkpoint files")
        .def("converged", &MASTER_DSRG::converged, "Return if amplitudes are converged or not")
        .def("set_ci_vectors", &MASTER_DSRG::set_ci_vectors,
             "Set the CI eigenvector for DSRG-MRPT2 analytic gradients")
        .def("set_active_space_solver", &MASTER_DSRG::set_active_space_solver,
             "Set the shared pointer for ActiveSpaceSolver");

    // export SADSRG
    py::class_<SADSRG>(m, "SADSRG")
        .def("compute_energy", &SADSRG::compute_energy, "Compute the DSRG energy")
        .def("compute_Heff_actv", &SADSRG::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals")
        .def("compute_Heff_full", [](SADSRG& self) {
            const auto Heff = self.compute_Heff_full();
            return py::make_tuple(blockedtensor_to_np(Heff.at(0)), blockedtensor_to_np(Heff.at(1)));
            })
        .def("compute_Heff_full_degno", [](SADSRG& self) {
            const auto Heff = self.compute_Heff_full_degno();
            return py::make_tuple(blockedtensor_to_np(Heff.at(0)), blockedtensor_to_np(Heff.at(1)));
            })
        .def("compute_mp_eff_actv", &SADSRG::compute_mp_eff_actv,
             "Return the DSRG dressed ActiveMultipoleIntegrals")
        .def("set_Uactv", &SADSRG::set_Uactv, "Ua"_a,
             "Set active part orbital rotation matrix (from original to semicanonical)")
        .def("set_active_space_solver", &SADSRG::set_active_space_solver,
             "Set the pointer of ActiveSpaceSolver")
        .def("set_state_weights_map", &SADSRG::set_state_weights_map,
             "Set the map from state to the weights of all computed roots")
        .def("set_read_cwd_amps", &SADSRG::set_read_amps_cwd,
             "Set if reading amplitudes in the current directory or not")
        .def("converged", &SADSRG::converged, "Return if amplitudes are converged or not")
        .def("clean_checkpoints", &SADSRG::clean_checkpoints, "Delete amplitudes checkpoint files");

    // export spin-adapted DSRG-MRPT2
    py::class_<SA_MRPT2, SADSRG>(m, "SA_MRPT2")
        .def(
            py::init<std::shared_ptr<RDMs>, std::shared_ptr<SCFInfo>, std::shared_ptr<ForteOptions>,
                     std::shared_ptr<ForteIntegrals>, std::shared_ptr<MOSpaceInfo>>())
        .def("build_fno", &SA_MRPT2::build_fno, "Build DSRG-MRPT2 frozen natural orbitals");

    // export MRDSRG_SO
    py::class_<MRDSRG_SO>(m, "MRDSRG_SO")
        .def("compute_energy", &MRDSRG_SO::compute_energy, "Compute DSRG energy")
        .def("compute_Heff_actv", &MRDSRG_SO::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals")
        .def("compute_Heff_full", [](MRDSRG_SO& self) {
            const auto Heff = self.compute_Heff_full();
            return py::make_tuple(blockedtensor_to_np(Heff.at(0)), blockedtensor_to_np(Heff.at(1)));
            })
        .def("get_gamma1", [](MRDSRG_SO& self) {
            const auto gamma1 = self.get_gamma1();
            return blockedtensor_to_np(gamma1);
            })
        .def("get_eta1", [](MRDSRG_SO& self) {
            const auto eta1 = self.get_eta1();
            return blockedtensor_to_np(eta1);
            })
        .def("get_lambda2", [](MRDSRG_SO& self) {
               const auto lambda2 = self.get_lambda2();
               return blockedtensor_to_np(lambda2);
               })
        .def("get_lambda3", [](MRDSRG_SO& self) {
                const auto lambda3 = self.get_lambda3();
               return blockedtensor_to_np(lambda3);
               });

    // export DSRG_MRPT spin-adapted code
    py::class_<DSRG_MRPT>(m, "DSRG_MRPT")
        .def("compute_energy", &DSRG_MRPT::compute_energy, "Compute DSRG energy")
        .def("compute_Heff_actv", &DSRG_MRPT::compute_Heff_actv,
             "Return the DSRG dressed ActiveSpaceIntegrals");

    // export the time-dependent ACI code
    py::class_<TDCI>(m, "TDCI", "Time-dependent ACI")
        .def(py::init<std::shared_ptr<ActiveSpaceMethod>, std::shared_ptr<SCFInfo>,
                      std::shared_ptr<ForteOptions>, std::shared_ptr<MOSpaceInfo>,
                      std::shared_ptr<ActiveSpaceIntegrals>>())
        .def("compute_energy", &TDCI::compute_energy, "Compute TD-ACI");

    // export DressedQuantity for dipole moments
    py::class_<DressedQuantity>(m, "DressedQuantity")
        .def("contract_with_rdms", &DressedQuantity::contract_with_rdms, "reference"_a,
             "Contract densities with quantity");
}

} // namespace forte
