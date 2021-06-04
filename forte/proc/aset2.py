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
from forte.pymodule import forte_driver

def aset2_driver(state_weights_map, scf_info, ref_wfn, mo_space_info, options):
    # TODO:build an assertion function to check all options?    
    # TODO:set EMBEDDING_TYPE to be ASET2

    # options.set_bool('FORTE', 'EMBEDDING', True)
    # options.set_str('FORTE', 'EMBEDDING_CUTOFF_METHOD', 'CORRELATED_BATH')
    # options.set_str('FORTE', 'EMBEDDING_SPECIAL', 'INNER_LAYER')

    frag_corr_level = options.get_str('FRAG_CORR_LEVEL')
    env_corr_level = options.get_str('ENV_CORR_LEVEL')

    # mo_space_info: fragment all active, environment restricted, core frozen
    # mo_space_info_active: fragment-C,A,V, environment and core frozen
    mo_space_info_active = forte.build_inner_space(ref_wfn, options, mo_space_info)
    # TODO: add this function in api

    # Build fragment and f-e integrals
    ints_f = forte.make_forte_integrals_from_psi4(ref_wfn, options, mo_space_info_active)
    # TODO:if custom, build custom integrals?

    psi4.core.print_out("\n Integral test (f original): oei_a(0, 2) = {:10.8f}".format(ints_f.oei_a(0, 2)))

    if ('DF' in options.get_str('INT_TYPE_ENV')):
        aux_basis_env = psi4.core.BasisSet.build(ref_wfn.molecule(), 'DF_BASIS_MP2',
                                         psi4.core.get_global_option('DF_BASIS_MP2'),
                                         'RIFIT', psi4.core.get_global_option('BASIS'))
        ref_wfn.set_basisset('DF_BASIS_MP2', aux_basis_env)
        options.set_str('FORTE', 'INT_TYPE', 'DF')
        forte.forte_options.update_psi_options(options)

    ints_e = forte.make_forte_integrals_from_psi4(ref_wfn, options, mo_space_info)
    
    #psi4.core.print_out("\n Integral test (e original): oei_a(4, 6) = {:10.8f}".format(ints_e.oei_a(4, 6)))
    #ints_f.build_from_another_ints(ints_e, -4) # Make sure this function works! make ints_f elements same with ints_e for corresponding blocks
    #psi4.core.print_out("\n Integral test (f original): oei_a(0, 2) = {:10.8f}".format(ints_e.oei_a(4, 6)))
    psi4.core.print_out("\n Integral test (f original): int_f ncmo: {:d}".format(ints_f.ncmo()))
    psi4.core.print_out("\n Integral test (e original): int_e ncmo: {:d}".format(ints_e.ncmo()))
    # compute higher-level with mo_space_info_active(inner) and methods in options, -> E_high(origin)

    # TODO: add an option manager function
    # TODO: make this ASET(mf) computation optional
    options.set_str('FORTE', 'INT_TYPE', 'CONVENTIONAL')
    options.set_str('FORTE', 'CORR_LEVEL', frag_corr_level)
    fold = options.get_bool('DSRG_FOLD')
    relax = options.get_str('RELAX_REF')
    options.set_bool('FORTE', 'DSRG_FOLD', False)
    options.set_str('FORTE', 'RELAX_REF', relax)
    options.set_bool('FORTE', 'EMBEDDING_DISABLE_SEMI_CHECK', False)
    options.set_bool('FORTE', 'EMBEDDING_ALIGN_FROZEN', False)
    #options.set_bool('FORTE', 'SEMI_CANONICAL', False)
    forte.forte_options.update_psi_options(options)
    energy_high = forte_driver_aset(state_weights_map, scf_info, forte.forte_options, ints_f, mo_space_info_active)
    psi4.core.print_out("\n Integral test (f_1, after ldsrg2): oei_a(0, 2) = {:10.8f}".format(ints_f.oei_a(0, 2)))

    raise Exception('Breakpoint reached!')

    options.set_bool('FORTE', 'DSRG_FOLD', fold)
    options.set_bool('FORTE', 'EMBEDDING_DISABLE_SEMI_CHECK', True)
    options.set_bool('FORTE', 'EMBEDDING_ALIGN_FROZEN', True)
    options.set_str('FORTE', 'RELAX_REF', "ONCE")
    
    # TODO: Build C++ embedding_density class, automate rdm level, rdms = forte.embedding_density(scf_info, mo_space_info, mo_space_info_active, ...)
    rdm_level = 2
    # Form rdms for interaction correlation computation
    rdms = forte.RHF_DENSITY(scf_info, mo_space_info).rhf_rdms()
    # TODO: add option for CASCI density using FCI, consider doing this in python side?
    if options.get_str('fragment_density') == "CASSCF": 
        as_ints = forte.make_active_space_ints(mo_space_info_active, ints_f, "ACTIVE", ["RESTRICTED_DOCC"])
        rdms = forte.build_casscf_density(rdm_level, scf_info, forte.forte_options, mo_space_info_active, mo_space_info, as_ints) # Stateinfo is no longer needed
    if options.get_str('fragment_density') == "FCI": # Expensive, for benchmarking only
        state_map = forte.to_state_nroots_map(state_weights_map)
        as_ints_full = forte.make_active_space_ints(mo_space_info, ints_e, "ACTIVE", ["RESTRICTED_DOCC"])
        as_solver_full = forte.make_active_space_solver(options.get_str('ACTIVE_SPACE_SOLVER'),
                                                       state_map, scf_info,
                                                       mo_space_info, as_ints_full,
                                                       forte.forte_options)
        state_energies_list = as_solver_full.compute_energy()
        rdms = as_solver_full.compute_average_rdms(state_weights_map, 3)
    # if options.get_str('downfold_density') == "MRDSRG": NotImplemented

    # DSRG-MRPT2(mo_space_info(outer), rdms)
    options.set_str('FORTE', 'CORR_LEVEL', env_corr_level)
    forte.forte_options.update_psi_options(options)

    if ('DF' in options.get_str('INT_TYPE_ENV')):
        aux_basis_env = psi4.core.BasisSet.build(ref_wfn.molecule(), 'DF_BASIS_MP2',
                                         psi4.core.get_global_option('DF_BASIS_MP2'),
                                         'RIFIT', psi4.core.get_global_option('BASIS'))
        ref_wfn.set_basisset('DF_BASIS_MP2', aux_basis_env)
        options.set_str('FORTE', 'INT_TYPE', 'DF')
        forte.forte_options.update_psi_options(options)

    dsrg = forte.make_dsrg_method(options.get_str('ENV_CORRELATION_SOLVER'), # TODO: ensure here always run canonical mr-dsrg (modify C++ side)
                                  rdms, scf_info, forte.forte_options, ints_e, mo_space_info)

    Edsrg = dsrg.compute_energy()
    psi4.core.set_scalar_variable('UNRELAXED ENERGY', Edsrg)
    E_ref1 = psi4.core.scalar_variable("DSRG REFERENCE ENERGY")
    E_corr = Edsrg - E_ref1
    # E_corr = Edsrg - E_cas_ref
    # Compute MRDSRG-in-PT2 energy (unfolded)
    # E_emb = E(MRDSRG) + E(Corr)

    # Test new MRDSRG energy (with rotated Heff/ints)
    # eH rotation
    ints_dressed = dsrg.compute_Heff_actv()

    # Debug:
    psi4.core.print_out("\n ints_f frozen core energy: {:10.8f}".format(ints_f.frozen_core_energy()))
    psi4.core.print_out("\n ints_f original scalar: {:10.8f}".format(ints_f.scalar()))
    psi4.core.print_out("\n ints_dressed frozen core energy: {:10.8f}".format(ints_dressed.frozen_core_energy()))
    psi4.core.print_out("\n ints_dressed original scalar: {:10.8f}".format(ints_dressed.scalar_energy()))

    frz1 = ints_f.frozen_core_energy()
    scalar = ints_dressed.scalar_energy() - frz1
    ints_f.set_scalar(scalar)

    psi4.core.print_out("\n ints_f shifted scalar".format(ints_dressed.scalar_energy()))

    # Build new ints for dressed computation
    state_map = forte.to_state_nroots_map(state_weights_map)
    ints_f.build_from_asints(ints_dressed)
    psi4.core.print_out("\n Integral test (f_1 after dressing): oei_a(0, 2) = {:10.8f}".format(ints_f.oei_a(0, 2)))

    # Compute MRDSRG-in-PT2 energy (folded)
    options.set_str('FORTE', 'CORR_LEVEL', frag_corr_level)
    options.set_bool('FORTE', 'DSRG_FOLD', False)
    options.set_str('FORTE', 'RELAX_REF', relax)
    options.set_bool('FORTE', 'EMBEDDING_DISABLE_SEMI_CHECK', False)
    options.set_bool('FORTE', 'EMBEDDING_ALIGN_FROZEN', False)
    options.set_str('FORTE', 'INT_TYPE', 'CONVENTIONAL')
    forte.forte_options.update_psi_options(options)

    # Compute Ec1 dressed
    energy_high_relaxed = forte_driver_aset(state_weights_map, scf_info, forte.forte_options, ints_f, mo_space_info_active)

    psi4.core.print_out("\n Integral test (f_1 after relaxed ldsrg2): oei_a(0, 2) = {:10.8f}".format(ints_f.oei_a(0, 2)))

    psi4.core.print_out("\n ==============Embedding Summary==============")
    psi4.core.print_out("\n E(fragment, undressed) = {:10.12f}".format(energy_high))
    psi4.core.print_out("\n E_corr(env correlation) = {:10.12f}".format(E_corr))
    psi4.core.print_out("\n E(embedding, undressed) = {:10.12f}".format(energy_high + E_corr))
    psi4.core.print_out("\n E(fragment, dressed) = {:10.12f}".format(energy_high_relaxed))
    psi4.core.print_out("\n E(embedding, dressed) = {:10.12f}".format(energy_high_relaxed + E_corr))
    psi4.core.print_out("\n ==============MRDSRG embedding done============== \n")

    return energy_high_relaxed + E_corr

def forte_driver_aset(state_weights_map, scf_info, options, ints, mo_space_info):
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
    correlation_solver_type = options.get_str('CORRELATION_SOLVER')
    if correlation_solver_type != 'NONE':
        dsrg_proc = ProcedureDSRG(active_space_solver, state_weights_map, mo_space_info, ints, options, scf_info)
        return_en = dsrg_proc.compute_energy()
        dsrg_proc.print_summary()
        dsrg_proc.push_to_psi4_environment()
    else:
        average_energy = forte.compute_average_state_energy(state_energies_list, state_weights_map)
        return_en = average_energy

    return return_en
