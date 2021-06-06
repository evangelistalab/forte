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

import psi4
import forte
from forte.proc.dsrg import ProcedureDSRG

def aset2_driver(state_weights_map, scf_info, ref_wfn, mo_space_info, options):

    # Read all options
    frag_corr_solver = options.get_str('FRAG_CORRELATION_SOLVER')
    frag_corr_level = options.get_str('FRAG_CORR_LEVEL')
    env_corr_solver = options.get_str('ENV_CORRELATION_SOLVER')
    env_corr_level = options.get_str('ENV_CORR_LEVEL')
    relax = options.get_str('RELAX_REF')
    semi = options.get_bool('SEMI_CANONICAL')
    int_type_frag = options.get_str('INT_TYPE_FRAG')
    int_type_env = options.get_str('INT_TYPE_ENV')
    do_aset_mf = options.get_bool('EMBEDDING_ASET2_MF_REF')
    #fold = options.get_bool('DSRG_FOLD')
    
    # Build two option lists for fragment (A) and environment (B) solver
    A_list = [frag_corr_solver, frag_corr_level, int_type_frag, relax, semi]
    B_list = [env_corr_solver, env_corr_level, int_type_env]

    # TODO: print computation details here

    # In ASET(2) procedure, We will keep 2 different mo_space_info:
    # mo_space_info: Active = AC + AA + AV, restricted = BO + BV, frozen = F
    # mo_space_info_active: Active = AA, restricted = AC + AV, frozen = BO + BV + F
    mo_space_info_active = forte.build_aset2_fragment(ref_wfn, mo_space_info)

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
    energy_cB, ints_dressed = forte_driver_environment(state_weights_map, scf_info, ref_wfn,  mo_space_info, mo_space_info_active, ints_e, options)

    raise Exception('Breakpoint reached!')

    # Build dressed fragment integrals.
    update_fragment_options(options, A_list, ref_wfn)
    ints_f = None
    if(int_type_frag == "CONVENTIONAL"):
        ints_f = forte.make_ints_from_psi4(ref_wfn, options, mo_space_info_active)
    else:
        #ints_f = build_empty_integral(mo_space_info_active)
        # TODO: if fragment integrals is not conventional, build a custom empty ints here
        raise Exception('Not finished here!')

    # Reset scalar
    frz1 = ints_f.frozen_core_energy()
    scalar = ints_dressed.scalar_energy() - frz1
    ints_f.set_scalar(scalar)
    psi4.core.print_out("\n ints_f shifted scalar".format(ints_dressed.scalar_energy()))

    # Build new ints for dressed computation
    #state_map = forte.to_state_nroots_map(state_weights_map)
    ints_f.build_from_asints(ints_dressed)

    # Compute MRDSRG-in-PT2 energy (folded)
    energy_high_dressed = forte_driver_fragment(state_weights_map, scf_info, forte.forte_options, ints_f, mo_space_info_active)
    # TODO: extract E_0 and E_c^A in forte_driver_fragment, and print them

    psi4.core.print_out("\n ==============ASET(2) Summary==============")
    if(do_aset_mf):
        psi4.core.print_out("\n E(fragment, undressed) = {:10.12f}".format(energy_high))
    psi4.core.print_out("\n E_corr(env correlation) = {:10.12f}".format(energy_cB))
    if(do_aset_mf):
        psi4.core.print_out("\n E(embedding, undressed) = {:10.12f}".format(energy_high + energy_cB))
    psi4.core.print_out("\n E(fragment, dressed) = {:10.12f}".format(energy_high_dressed))
    psi4.core.print_out("\n E(embedding, dressed) = {:10.12f}".format(energy_high_dressed + energy_cB))
    psi4.core.print_out("\n ==============ASET(2) procedure done============== \n")

    return energy_high_dressed + energy_cB

def forte_driver_fragment(state_weights_map, scf_info, options, ints, mo_space_info):
    # TODO Modify this solver to consider frag/env
    """
    Driver to perform a Forte calculation using new solvers.

    :param state_weights_map: dictionary of {state: weights}
    :param scf_info: a SCFInfo object of Forte
    :param options: a ForteOptions object of Forte
    :param ints: a ForteIntegrals object of Forte
    :param mo_space_info: a MOSpaceInfo object of Forte

    :return: the computed energy
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
        options.set_str('INT_TYPE', A_list[2])

    options.set_str('CORRELATION_SOLVER', A_list[0])
    options.set_str('CORR_LEVEL', A_list[1])
    options.set_str('RELAX_REF', A_list[3])
    options.set_bool('SEMI_CANONICAL', A_list[4])
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
        options.set_str('INT_TYPE', B_list[2])

    options.set_str('CORRELATION_SOLVER', B_list[0])
    options.set_str('CORR_LEVEL', B_list[1])
    options.set_bool('EMBEDDING_DISABLE_SEMI_CHECK', True)
    options.set_str('RELAX_REF', "ONCE") # Setting it here to bypass many relax_ref checks in all DSRG codes
    #options.set_bool('DSRG_FOLD', fold)
    options.set_bool('EMBEDDING_ALIGN_SCALAR', True)

    return
