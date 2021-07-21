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

import psi4
import forte
import numpy as np
from forte.proc.dsrg import ProcedureDSRG


def aset2_driver(state_weights_map, scf_info, ref_wfn, mo_space_info, options):
    """
    Driver to run an systematic Active Space Embedding Theory (ASET) computation.

    :param state_weights_map: dictionary of {state: weights}
    :param scf_info: a SCFInfo object of Forte
    :param ref_wfn: a Psi4::Wavefunction object
    :param mo_space_info: a MOSpaceInfo object with embedding partitions (EMBEDDING_ORB)
    :param options: a ForteOptions object of Forte

    :return: the computed ASET energy.
    """

    # Read all options
    frag_as_solver = options.get_str('ACTIVE_SPACE_SOLVER')
    frag_do_fci = options.get_bool('FRAG_DO_FCI')
    frag_corr_solver = options.get_str('FRAG_CORRELATION_SOLVER')
    frag_corr_level = options.get_str('FRAG_CORR_LEVEL')
    env_corr_solver = options.get_str('ENV_CORRELATION_SOLVER')
    env_corr_level = options.get_str('ENV_CORR_LEVEL')
    relax = options.get_str('RELAX_REF')
    semi = options.get_bool('SEMI_CANONICAL')
    int_type_frag = options.get_str('INT_TYPE_FRAG')
    int_type_env = options.get_str('INT_TYPE_ENV')
    do_aset_mf = options.get_bool('EMBEDDING_ASET2_MF_REF')
    frag_density = options.get_str('FRAGMENT_DENSITY')
    ref_type = options.get_str('EMBEDDING_REFERENCE')
    name = frag_corr_solver + '-' + frag_corr_level
    #fold = options.get_bool('DSRG_FOLD')
    
    # Build two option lists for fragment (A) and environment (B) solver
    A_list = [frag_corr_solver, frag_corr_level, int_type_frag, relax, semi]
    B_list = [env_corr_solver, env_corr_level, int_type_env]

    psi4.core.print_out("\n  ")
    psi4.core.print_out("\n  ")
    forte.print_method_banner(["Second-order Active Space Embedding Theory [ASET(2)]", "Nan He"])
    psi4.core.print_out("\n  ===========================ASET(2) Options===========================")
    psi4.core.print_out("\n  Fragment active space solver:             {:s}".format(frag_as_solver))
    psi4.core.print_out("\n  Fragment correlation solver:              {:s}".format(frag_corr_solver))
    psi4.core.print_out("\n  Fragment correlation level:               {:s}".format(frag_corr_level))
    psi4.core.print_out("\n  Fragment integral type:                   {:s}".format(int_type_frag))
    if(frag_do_fci):
        psi4.core.print_out("\n  {:s} will be used on the whole fragment!".format(frag_as_solver))
    psi4.core.print_out("\n  ---------------------------------------------------------------------")
    psi4.core.print_out("\n  Environment correlation solver:           {:s}".format(env_corr_solver))
    psi4.core.print_out("\n  Environment correlation level:            {:s}".format(env_corr_level))
    psi4.core.print_out("\n  Environment integral type:                {:s}".format(int_type_env))
    psi4.core.print_out("\n  Fragment density will be evaluated using: {:s}".format(frag_density))
    psi4.core.print_out("\n  ---------------------------------------------------------------------")
    if(do_aset_mf):
        psi4.core.print_out("\n  Procedure: ASET(MF)-[{:s}] -> {:s} -> ASET(2)-[{:s}]".format(name, frag_density, name))
    else:
        psi4.core.print_out("\n  Procedure: {:s} -> ASET(2)-[{:s}]".format(frag_density, name))
    if(int_type_frag in ["CHOLESKY", "DF", "DISKDF"]):
        psi4.core.print_out("\n  Warning: DF/CD integrals inside the fragment (A) are not supported now.")
        psi4.core.print_out("\n  Will build a Custom_Integral for H_bar instead.")
    if(frag_corr_solver == "THREE-DSRG-MRPT2"):
        psi4.core.print_out("\n  Warning: DF/CD integrals inside the fragment (A) are not supported now.")
        psi4.core.print_out("\n  Will automatically convert to conventional DSRG-MRPT2 for H_bar.")
    if(ref_type == "HF"):
        psi4.core.print_out("\n  Warning: reference may have no information on active orbital space definitions.")
        psi4.core.print_out("\n  Please ensure that they are set manually in the inputs.")
    psi4.core.print_out("\n  =====================================================================")
    psi4.core.print_out("\n  ")
    psi4.core.print_out("\n  ")
    psi4.core.print_out("\n  ")

    # In ASET(2) procedure, We will keep 2 different mo_space_info:
    # mo_space_info: Active = AC + AA + AV, restricted = BO + BV, frozen = F
    # mo_space_info_active: Active = AA, restricted = AC + AV, frozen = BO + BV + F
    mo_space_info_active = forte.build_aset2_spaceinfo(ref_wfn, mo_space_info, options)

    # Build total (A+B) integrals first
    update_environment_options(options, B_list, ref_wfn)
    ints_e = forte.make_ints_from_psi4(ref_wfn, options, mo_space_info)

    # Compute ASET(mf) higher-level with mo_space_info_active(inner) and methods in options, -> E_high(origin)
    energy_high = 0.0
    if(do_aset_mf):
        update_fragment_options(options, A_list, ref_wfn)
        ints_f = forte.make_ints_from_psi4(ref_wfn, options, mo_space_info_active)
        energy_high = forte_driver_fragment(state_weights_map, scf_info, forte.forte_options, ints_f, mo_space_info_active)

    # Compute envrionment (B) amplitudes T1, T2 and energy corrections E_c^B, dress fragment ints
    update_environment_options(options, B_list, ref_wfn)
    energy_env, ints_dressed = forte_driver_environment(state_weights_map, scf_info, ref_wfn,  mo_space_info, mo_space_info_active, ints_e, options)

    # Build dressed fragment integrals, starting from an ints object.
    update_fragment_options(options, A_list, ref_wfn)
    ints_f = None
    options.set_bool('TRUNCATE_MO_SPACE', True)
    mo_space_info_truncated = forte.build_aset2_spaceinfo(ref_wfn, mo_space_info, options)
    # Build custom integrals for dressed H
    nmo = mo_space_info_truncated.size("ALL")
    scalar = 0.0 
    hcore = np.zeros((nmo, nmo))
    eri_aa = np.zeros((nmo, nmo, nmo, nmo))
    eri_ab = np.zeros((nmo, nmo, nmo, nmo))
    eri_bb = np.zeros((nmo, nmo, nmo, nmo))
    ints_f = forte.make_custom_ints(options, mo_space_info_truncated, scalar,
                              hcore.flatten(), hcore.flatten(), eri_aa.flatten(),
                              eri_ab.flatten(), eri_bb.flatten())
    options.set_str('INT_TYPE', "FCIDUMP")

    # Set the scalar for the newly-made integral
    frz1 = ints_f.frozen_core_energy()
    dressed_scalar = ints_dressed.scalar_energy()
    scalar = dressed_scalar - frz1
    #if(int_type_frag != "CONVENTIONAL") or (int_type_frag == "CONVENTIONAL" and frag_do_fci):
    scalar += ints_e.nuclear_repulsion_energy()
    ints_f.set_scalar(scalar)

    # Build new ints for dressed computation
    ints_f.set_ints_from_asints(ints_dressed)

    # For three-dsrg-mrpt2, automatically convert to dsrg-mrpt2 when computing the fragment (A)
    if options.get_str('CORRELATION_SOLVER') == "THREE-DSRG-MRPT2":
        options.set_str('CORRELATION_SOLVER', "MRDSRG")
        options.set_str('CORR_LEVEL', "PT2")

    psi4.core.print_out("\n    Integral dressing successed !  ")

    # Get the number of doccs we should set for the fragment compution   
    docc_B = mo_space_info.dimension("RESTRICTED_DOCC")

    # Reduce na and nb of all states by docc_B
    state_info_list = list(state_weights_map.keys())
    for state_info in state_info_list:
        na = state_info.na() - docc_B[0]
        nb = state_info.nb() - docc_B[0]
        multiplicity = state_info.multiplicity()
        twice_ms = state_info.twice_ms()
        state_info_new = forte.StateInfo(na, nb, multiplicity, twice_ms, 0)
        state_weights_map[state_info_new] = state_weights_map.pop(state_info)

    # Run the fragment (A) computation
    energy_high_dressed = forte_driver_fragment(state_weights_map, scf_info, options, ints_f, mo_space_info_truncated)

    psi4.core.print_out("\n  ")
    psi4.core.print_out("\n  ")
    psi4.core.print_out("\n    ============================ASET(2) Summary========================")
    if(do_aset_mf):
        psi4.core.print_out("\n    ASET(mf) energy                                = {:10.12f}".format(energy_high))
    psi4.core.print_out("\n    E_c^B (Environment correlation, Hbar0)         = {:10.12f}".format(energy_env))
    psi4.core.print_out("\n    E(Hbar) (Fragment energy computed using Hbar)  = {:10.12f}".format(energy_high_dressed))
    psi4.core.print_out("\n    ASET(2) energy                                 = {:10.12f}".format(energy_high_dressed + energy_env))
    psi4.core.print_out("\n    =========================ASET(2) Procedure Done==================== \n")
    psi4.core.print_out("\n  ")

    return energy_high_dressed + energy_env

def forte_driver_fragment(state_weights_map, scf_info, options, ints, mo_space_info):
    """
    Driver to perform a Forte calculation for embedded fragment (A).

    :param state_weights_map: dictionary of {state: weights}
    :param scf_info: a SCFInfo object of Forte
    :param options: a ForteOptions object of Forte
    :param ints: a fragment ForteIntegrals object (size = A)
    :param mo_space_info: a MOSpaceInfo object inside the fragment (A)

    :return: the computed fragment energy (both for undressed or dressed)
    """

    # map state to number of roots
    state_map = forte.to_state_nroots_map(state_weights_map)

    # create an active space solver object and compute the energy
    active_space_solver_type = options.get_str('ACTIVE_SPACE_SOLVER')

    as_ints = forte.make_active_space_ints(mo_space_info, ints, "ACTIVE", ["RESTRICTED_DOCC"])

    active_space_solver = forte.make_active_space_solver(active_space_solver_type, state_map, scf_info,
                                                         mo_space_info, as_ints, options)
    state_energies_list = active_space_solver.compute_energy()

    if options.get_bool('SPIN_ANALYSIS'):
        rdms = active_space_solver.compute_average_rdms(state_weights_map, 2)
        forte.perform_spin_analysis(rdms, options, mo_space_info, as_ints)

    # solver for dynamical correlation from DSRG
    correlation_solver_type = options.get_str('FRAG_CORRELATION_SOLVER')
    if correlation_solver_type != 'NONE':
        dsrg_proc = ProcedureDSRG(active_space_solver, state_weights_map, mo_space_info, ints, options, scf_info)
        return_en = dsrg_proc.compute_energy()
        dsrg_proc.print_summary()
        dsrg_proc.push_to_psi4_environment()
    else:
        average_energy = forte.compute_average_state_energy(state_energies_list, state_weights_map)
        return_en = average_energy

    return return_en

def forte_driver_environment(state_weights_map, scf_info, ref_wfn, mo_space_info, mo_space_info_active, ints_e, options):
    """
    Driver to perform a Forte calculation for embedding environment and downfolding (A+B).

    :param state_weights_map: dictionary of {state: weights}
    :param scf_info: a SCFInfo object of Forte
    :param ref_wfn: a Psi4::Wavefunction object
    :param options: a ForteOptions object of Forte
    :param mo_space_info: a MOSpaceInfo object for A + B
    :param mo_space_info_active: a MOSpaceInfo object for A
    :param ints_e: a ForteIntegrals object of the whole system (size = A + B)
    :param options: a ForteOptions object of Forte

    :return: the computed environment energy E_corr, and dressed integrals ints_dressed
    """

    # Build integrals for fragment density approximation. Int type will be consistent with INT_TYPE_ENV
    ints_d = forte.make_ints_from_psi4(ref_wfn, options, mo_space_info_active)

    # Form an approximate or accurate rdms for A for the MRDSRG (A+B) computation
    density = forte.EMBEDDING_DENSITY(state_weights_map, scf_info, mo_space_info, ints_d, options)

    rdms = None
    rdms_level = 3
    if options.get_str('THREEPDC') == 'ZERO':
        rdms_level = 2

    if options.get_str('fragment_density') == "RHF":
        rdms = density.rhf_rdms()
    if options.get_str('fragment_density') == "CASSCF" or options.get_str('fragment_density') == "CASCI": # Default option
        rdms = density.cas_rdms(mo_space_info_active)

    if options.get_str('fragment_density') == "FULL": # Use active_space_solver to compute the whole fragment (A). Expensive, for benchmarking only
        state_map = forte.to_state_nroots_map(state_weights_map)
        as_ints_full = forte.make_active_space_ints(mo_space_info, ints_e, "ACTIVE", ["RESTRICTED_DOCC"])
        as_solver_full = forte.make_active_space_solver(options.get_str('ACTIVE_SPACE_SOLVER'),
                                                       state_map, scf_info,
                                                       mo_space_info, as_ints_full,
                                                       options)
        state_energies_list = as_solver_full.compute_energy()
        rdms = as_solver_full.compute_average_rdms(state_weights_map, 3)

    # Compute MRDSRG (A+B)
    dsrg = forte.make_dsrg_method(rdms, scf_info, options, ints_e, mo_space_info)
    Edsrg = dsrg.compute_energy()

    # Extract correlation energy E_cB
    psi4.core.set_scalar_variable('UNRELAXED ENERGY', Edsrg)
    E_ref1 = psi4.core.scalar_variable("DSRG REFERENCE ENERGY")
    E_corr = Edsrg - E_ref1

    # Compute Hbar 1, 2, return the dressed integral
    ints_dressed = dsrg.compute_Heff_actv()

    return E_corr, ints_dressed

def update_fragment_options(options, A_list, ref_wfn):

    if ('DF' in A_list[2]):
        aux_basis_frag = psi4.core.BasisSet.build(ref_wfn.molecule(), 'DF_BASIS_MP2',
                                         psi4.core.get_global_option('DF_BASIS_MP2'),
                                         'RIFIT', psi4.core.get_global_option('BASIS'))
        ref_wfn.set_basisset('DF_BASIS_MP2', aux_basis_frag)
    
    options.set_str('CORRELATION_SOLVER', A_list[0])
    options.set_str('CORR_LEVEL', A_list[1])
    options.set_str('INT_TYPE', A_list[2])
    options.set_str('RELAX_REF', A_list[3])
    options.set_bool('SEMI_CANONICAL', False)
    options.set_bool('EMBEDDING_DISABLE_SEMI_CHECK', False)
    options.set_bool('EMBEDDING_ALIGN_SCALAR', False)
    #forte.forte_options.update_psi_options(options) #TODO: check whether we still need this

    # Block folding functionality will be in another PR
    #options.set_bool('DSRG_FOLD', False)
    #options.set_bool('EMBEDDING_ALIGN_FROZEN', False)

    return

def update_environment_options(options, B_list, ref_wfn):

    if ('DF' in B_list[2]):
        aux_basis_env = psi4.core.BasisSet.build(ref_wfn.molecule(), 'DF_BASIS_MP2',
                                         psi4.core.get_global_option('DF_BASIS_MP2'),
                                         'RIFIT', psi4.core.get_global_option('BASIS'))
        ref_wfn.set_basisset('DF_BASIS_MP2', aux_basis_env)

    options.set_str('CORRELATION_SOLVER', B_list[0])
    options.set_str('CORR_LEVEL', B_list[1])
    options.set_str('INT_TYPE', B_list[2])
    options.set_bool('EMBEDDING_DISABLE_SEMI_CHECK', True)
    options.set_str('RELAX_REF', "ONCE") # Setting it here to bypass many relax_ref checks in all DSRG codes
    #options.set_bool('DSRG_FOLD', fold)
    options.set_bool('EMBEDDING_ALIGN_SCALAR', True)

    return
