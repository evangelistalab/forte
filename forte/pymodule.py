#
# @BEGIN LICENSE
#
# forte_inversion by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2016 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import time
import os

import psi4
import psi4.driver.p4util as p4util

import forte
from forte.data import ForteData
from forte.modules import OptionsFactory, ObjectsFactoryFCIDUMP, ObjectsFactoryPsi4, ActiveSpaceIntsFactory
from forte.proc.external_active_space_solver import (
    write_external_active_space_file,
    write_external_rdm_file,
    write_wavefunction,
    read_wavefunction,
    make_hamiltonian,
)
from forte.proc.dsrg import ProcedureDSRG


def forte_driver(data: ForteData):
    """
    Driver to perform a Forte calculation using new solvers.

    :param state_weights_map: dictionary of {state: weights}
    :param scf_info: a SCFInfo object of Forte
    :param options: a ForteOptions object of Forte
    :param ints: a ForteIntegrals object of Forte
    :param mo_space_info: a MOSpaceInfo object of Forte

    :return: the computed energy
    """
    state_weights_map, scf_info, options, ints, mo_space_info = (
        data.state_weights_map,
        data.scf_info,
        data.options,
        data.ints,
        data.mo_space_info,
    )

    # map state to number of roots
    state_map = forte.to_state_nroots_map(state_weights_map)

    # create an active space solver object and compute the energy
    active_space_solver_type = options.get_str("ACTIVE_SPACE_SOLVER")
    data = ActiveSpaceIntsFactory(active="ACTIVE", core=["RESTRICTED_DOCC"]).run(data)
    as_ints = data.as_ints
    active_space_solver = forte.make_active_space_solver(
        active_space_solver_type, state_map, scf_info, mo_space_info, as_ints, options
    )

    if active_space_solver_type == "EXTERNAL":
        write_external_active_space_file(as_ints, state_map, mo_space_info, "as_ints.json")
        msg = "External solver: save active space integrals to as_ints.json"
        print(msg)
        psi4.core.print_out(msg)

        if not os.path.isfile("rdms.json"):
            msg = "External solver: rdms.json file not present, exit."
            print(msg)
            psi4.core.print_out(msg)
            # finish the computation
            exit()
    # if rdms.json exists, then run "external" as_solver to compute energy
    state_energies_list = active_space_solver.compute_energy()

    if options.get_bool("WRITE_RDM"):
        max_rdm_level = 3  # TODO allow the user to change this variable
        write_external_rdm_file(active_space_solver, state_weights_map, max_rdm_level)

    if options.get_bool("SPIN_ANALYSIS"):
        rdms = active_space_solver.compute_average_rdms(state_weights_map, 2, forte.RDMsType.spin_dependent)
        forte.perform_spin_analysis(rdms, options, mo_space_info, as_ints)

    # solver for dynamical correlation from DSRG
    correlation_solver_type = options.get_str("CORRELATION_SOLVER")
    if correlation_solver_type != "NONE":
        dsrg_proc = ProcedureDSRG(active_space_solver, state_weights_map, mo_space_info, ints, options, scf_info)
        return_en = dsrg_proc.compute_energy()
        dsrg_proc.print_summary()
        dsrg_proc.push_to_psi4_environment()

        if options.get_str("DERTYPE") == "FIRST" and active_space_solver_type in ["DETCI", "GENCI"]:
            # Compute coupling coefficients
            # NOTE: 1. Orbitals have to be semicanonicalized already to make sure
            #          DSRG reads consistent CI coefficients before and after SemiCanonical class.
            #       2. This is OK only when running ground-state calculations
            state = list(state_map.keys())[0]
            psi4.core.print_out(f"\n  ==> Coupling Coefficients for {state} <==")
            ci_vectors = active_space_solver.eigenvectors(state)
            dsrg_proc.compute_gradient(ci_vectors)
        else:
            psi4.core.print_out("\n  Semicanonical orbitals must be used!\n")
    else:
        average_energy = forte.compute_average_state_energy(state_energies_list, state_weights_map)
        return_en = average_energy

    return return_en


def run_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('forte')

    """

    # # Start Forte, initialize ambit
    # my_proc_n_nodes = forte.startup()
    # my_proc, n_nodes = my_proc_n_nodes

    # Start timer
    start_pre_ints = time.time()

    # Print the banner
    forte.banner()

    # Build Forte options
    data = OptionsFactory(options=kwargs.get("forte_options")).run()

    job_type = data.options.get_str("JOB_TYPE")
    # Prepare Forte objects
    if "FCIDUMP" in data.options.get_str("INT_TYPE"):
        data = ObjectsFactoryFCIDUMP(options=kwargs).run(data)
    else:
        data = ObjectsFactoryPsi4(**kwargs).run(data)

    start = time.time()

    # Rotate orbitals before computation (e.g. localization, MP2 natural orbitals, etc.)
    orb_type = data.options.get_str("ORBITAL_TYPE")
    if orb_type != "CANONICAL":
        orb_t = forte.make_orbital_transformation(orb_type, data.scf_info, data.options, data.ints, data.mo_space_info)
        orb_t.compute_transformation()
        Ua = orb_t.get_Ua()
        Ub = orb_t.get_Ub()
        data.ints.rotate_orbitals(Ua, Ub, job_type != "NONE")

    # Run a method
    if job_type == "NONE":
        psi4.core.set_scalar_variable("CURRENT ENERGY", 0.0)
        # forte.cleanup()
        return data.psi_wfn

    energy = 0.0

    if data.options.get_bool("CASSCF_REFERENCE") or job_type == "CASSCF":
        if data.options.get_str("INT_TYPE") == "FCIDUMP":
            raise Exception("Forte: the CASSCF code cannot use integrals read from a FCIDUMP file")

        casscf = forte.make_casscf(data.state_weights_map, data.scf_info, data.options, data.mo_space_info, data.ints)
        energy = casscf.compute_energy()

    if job_type == "MCSCF_TWO_STEP":
        casscf = forte.make_mcscf_two_step(
            data.state_weights_map, data.scf_info, data.options, data.mo_space_info, data.ints
        )
        energy = casscf.compute_energy()

    if job_type == "TDCI":
        state = forte.make_state_info_from_psi(data.options)
        data = ActiveSpaceIntsFactory(active="ACTIVE", core=["RESTRICTED_DOCC"]).run(data)
        state_map = forte.to_state_nroots_map(data.state_weights_map)
        active_space_method = forte.make_active_space_method(
            "ACI", state, data.options.get_int("NROOT"), data.scf_info, data.mo_space_info, data.as_ints, data.options
        )
        active_space_method.set_quiet_mode()
        active_space_method.compute_energy()

        tdci = forte.TDCI(active_space_method, data.scf_info, data.options, data.mo_space_info, data.as_ints)
        energy = tdci.compute_energy()

    if job_type == "NEWDRIVER":
        energy = forte_driver(data)
    elif job_type == "MR-DSRG-PT2":
        energy = mr_dsrg_pt2(job_type, data)

    end = time.time()

    # Close ambit, etc.
    # forte.cleanup()

    psi4.core.set_scalar_variable("CURRENT ENERGY", energy)

    psi4.core.print_out(f"\n\n  Time to prepare integrals: {start - start_pre_ints:12.3f} seconds")
    psi4.core.print_out(f"\n  Time to run job          : {end - start:12.3f} seconds")
    psi4.core.print_out(f"\n  Total                    : {end - start_pre_ints:12.3f} seconds\n")

    if "FCIDUMP" not in data.options.get_str("INT_TYPE"):
        if data.options.get_bool("DUMP_ORBITALS"):
            dump_orbitals(data.psi_wfn)
        return data.psi_wfn
    return None


def gradient_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> gradient('forte')
        available for : CASSCF
    """

    # Get the psi4 option object
    optstash = p4util.OptionsState(["GLOBALS", "DERTYPE"])
    psi4.core.set_global_option("DERTYPE", "FIRST")

    # Build Forte options
    data = OptionsFactory(options=kwargs.get("forte_options")).run()

    # Print the banner
    forte.banner()

    # Run a method
    job_type = data.options.get_str("JOB_TYPE")
    int_type = data.options.get_str("INT_TYPE")
    correlation_solver = data.options.get_str("CORRELATION_SOLVER")

    if job_type not in {"CASSCF", "MCSCF_TWO_STEP"} and correlation_solver != "DSRG-MRPT2":
        raise Exception("Analytic energy gradients are only implemented for" " CASSCF, MCSCF_TWO_STEP, or DSRG-MRPT2.")

    # Prepare Forte objects: state_weights_map, mo_space_info, scf_info
    data = ObjectsFactoryPsi4(**kwargs).run(data)

    # Make an integral object
    time_pre_ints = time.time()

    data.ints = forte.make_ints_from_psi4(data.psi_wfn, data.options, data.mo_space_info)

    start = time.time()

    # Rotate orbitals before computation
    orb_type = data.options.get_str("ORBITAL_TYPE")
    if orb_type != "CANONICAL":
        orb_t = forte.make_orbital_transformation(orb_type, data.scf_info, data.options, data.ints, data.mo_space_info)
        orb_t.compute_transformation()
        Ua = orb_t.get_Ua()
        Ub = orb_t.get_Ub()
        ints.rotate_orbitals(Ua, Ub)

    if job_type == "CASSCF":
        casscf = forte.make_casscf(data.state_weights_map, data.scf_info, data.options, data.mo_space_info, data.ints)
        energy = casscf.compute_energy()
        casscf.compute_gradient()

    if job_type == "MCSCF_TWO_STEP":
        casscf = forte.make_mcscf_two_step(
            data.state_weights_map, data.scf_info, data.options, data.mo_space_info, data.ints
        )
        energy = casscf.compute_energy()

    if job_type == "NEWDRIVER" and correlation_solver == "DSRG-MRPT2":
        forte_driver(data)

    time_pre_deriv = time.time()

    derivobj = psi4.core.Deriv(data.psi_wfn)
    derivobj.set_deriv_density_backtransformed(True)
    derivobj.set_ignore_reference(True)
    if int_type == "DF":
        grad = derivobj.compute_df("DF_BASIS_SCF", "DF_BASIS_MP2")
    else:
        grad = derivobj.compute(psi4.core.DerivCalcType.Correlated)
    data.psi_wfn.set_gradient(grad)
    optstash.restore()

    end = time.time()

    # Close ambit, etc.
    # forte.cleanup()

    # Print timings
    psi4.core.print_out("\n\n ==> Forte Timings <==\n")
    times = [
        ("prepare integrals", start - time_pre_ints),
        ("run forte energy", time_pre_deriv - start),
        ("compute derivative integrals", end - time_pre_deriv),
    ]
    max_key_size = max(len(k) for k, v in times)
    for key, value in times:
        psi4.core.print_out(f"\n  Time to {key:{max_key_size}} : {value:12.3f} seconds")
    psi4.core.print_out(f'\n  {"Total":{max_key_size + 8}} : {end - time_pre_ints:12.3f} seconds\n')

    # Dump orbitals if needed
    if data.options.get_bool("DUMP_ORBITALS"):
        dump_orbitals(data.psi_wfn)

    return data.psi_wfn


def mr_dsrg_pt2(job_type, data):
    """
    Driver to perform a MCSRGPT2_MO computation.

    :return: the computed energy
    """
    final_energy = 0.0

    options = data.options
    ref_wfn = data.psi_wfn
    state_weights_map = data.state_weights_map
    mo_space_info = data.mo_space_info
    scf_info = data.scf_info
    ints = data.ints

    state = forte.make_state_info_from_psi(options)
    # generate a list of states with their own weights
    state_map = forte.to_state_nroots_map(state_weights_map)

    cas_type = options.get_str("ACTIVE_SPACE_SOLVER")
    actv_type = options.get_str("FCIMO_ACTV_TYPE")
    if actv_type == "CIS" or actv_type == "CISD":
        raise Exception("Forte: VCIS/VCISD is not supported for MR-DSRG-PT2")
    max_rdm_level = 2 if options.get_str("THREEPDC") == "ZERO" else 3
    data = ActiveSpaceIntsFactory(active="ACTIVE", core=["RESTRICTED_DOCC"]).run(data)
    ci = forte.make_active_space_solver(cas_type, state_map, scf_info, mo_space_info, data.as_ints, options)
    ci.compute_energy()

    rdms = ci.compute_average_rdms(state_weights_map, max_rdm_level, forte.RDMsType.spin_dependent)
    semi = forte.SemiCanonical(mo_space_info, ints, options)
    semi.semicanonicalize(rdms)

    mcsrgpt2_mo = forte.MCSRGPT2_MO(rdms, options, ints, mo_space_info)
    energy = mcsrgpt2_mo.compute_energy()
    return energy


# Integration with driver routines
psi4.driver.procedures["energy"]["forte"] = run_forte
psi4.driver.procedures["gradient"]["forte"] = gradient_forte
