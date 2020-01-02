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

import timeit
import math
import warnings

import psi4
import forte
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util

def forte_driver(state_weights_map, scf_info, options, ints, mo_space_info):
    max_rdm_level = 2 if options.get_str('THREEPDC') == 'ZERO' else 3
    return_en = 0.0

    state_map = forte.to_state_nroots_map(state_weights_map)

    # Create an active space solver object and compute the energy
    active_space_solver_type = options.get_str('ACTIVE_SPACE_SOLVER')
    as_ints = forte.make_active_space_ints(mo_space_info, ints, "ACTIVE", ["RESTRICTED_DOCC"]);
    active_space_solver = forte.make_active_space_solver(active_space_solver_type,state_map,scf_info,mo_space_info,as_ints,options)
    state_energies_list = active_space_solver.compute_energy()


    # Notes (York):
    #     cases to run active space solver: reference relaxation, state-average dsrg
    #     cases to run contracted ci solver (will be put in ActiveSpaceSolver): contracted state-average dsrg
    Etemp1, Etemp2 = 0.0, 0.0

    # Create a dynamical correlation solver object
    correlation_solver_type = options.get_str('CORRELATION_SOLVER')
    if correlation_solver_type != 'NONE':
        # Grab the reference
        rdms = active_space_solver.compute_average_rdms(state_weights_map, max_rdm_level)

        # Compute unitary matrices Ua and Ub that rotate the orbitals to the semicanonical basis
        semi = forte.SemiCanonical(mo_space_info, ints, options)
        if options.get_bool("SEMI_CANONICAL"):
            semi.semicanonicalize(rdms, max_rdm_level)
        Ua = semi.Ua_t()
        Ub = semi.Ub_t()

        Edsrg, dsrg, Heff_actv_implemented = compute_dsrg_unrelaxed_energy(correlation_solver_type,
                                                                           rdms, scf_info, options,
                                                                           ints, mo_space_info,
                                                                           Ua, Ub)
        if not Heff_actv_implemented:
            return Edsrg

        # dipole moment related
        do_dipole = options.get_bool("DSRG_DIPOLE")
        if do_dipole:
            if correlation_solver_type == 'MRDSRG' or correlation_solver_type == 'THREE-DSRG-MRPT2':
                do_dipole = False
                psi4.core.print_out("\n  !Dipole moment is not implemented for {}.".format(correlation_solver_type))
                warnings.warn("Dipole moment is not implemented for MRDSRG.", UserWarning)
            udm_x = psi4.core.variable('UNRELAXED DIPOLE X')
            udm_y = psi4.core.variable('UNRELAXED DIPOLE Y')
            udm_z = psi4.core.variable('UNRELAXED DIPOLE Z')
            udm_t = psi4.core.variable('UNRELAXED DIPOLE')

        def dipole_routine(dsrg_method, rdms):
            dipole_moments = dsrg_method.nuclear_dipole()
            dipole_dressed = dsrg_method.deGNO_DMbar_actv()
            for i in range(3):
                dipole_moments[i] += dipole_dressed[i].contract_with_rdms(rdms)
            dm_total = math.sqrt(sum([i * i for i in dipole_moments]))
            dipole_moments.append(dm_total)
            return dipole_moments

        # determine the relaxation procedure
        relax_mode = options.get_str("RELAX_REF")
        is_multi_state = True if options.get_str("CALC_TYPE") != "SS" else False

        if relax_mode == 'NONE' and is_multi_state:
            relax_mode = 'ONCE'

        if relax_mode == 'NONE':
            return Edsrg
        elif relax_mode == 'ONCE':
            maxiter = 1
        elif relax_mode == 'TWICE':
            maxiter = 2
        else:
            maxiter = options.get_int('MAXITER_RELAX_REF')

        # filter out some ms-dsrg algorithms
        ms_dsrg_algorithm = options.get_str("DSRG_MULTI_STATE")
        if is_multi_state and ("SA" not in ms_dsrg_algorithm):
            raise NotImplementedError("MS or XMS is disabled due to the reconstruction.")
        if ms_dsrg_algorithm == "SA_SUB" and relax_mode != 'ONCE':
            raise NotImplementedError("Need to figure out how to compute relaxed SA density.")

        # prepare for reference relaxation iteration
        relax_conv = options.get_double("RELAX_E_CONVERGENCE")
        e_conv = options.get_double("E_CONVERGENCE")
        converged = False if relax_mode != 'ONCE' and relax_mode != 'TWICE' else True

        # store (unrelaxed, relaxed) quantities
        dsrg_energies = []
        dsrg_dipoles = []

        for N in range(maxiter):
            # Grab the effective Hamiltonian in the actice space
            ints_dressed = dsrg.compute_Heff_actv()

            # Compute the energy
            if is_multi_state and ms_dsrg_algorithm == "SA_SUB":
                sa_sub_max_rdm = 2 # TODO: This should be 3 if do_hbar3 is true
                state_energies_list = active_space_solver.compute_contracted_energy(ints_dressed, sa_sub_max_rdm)
                Erelax = forte.compute_average_state_energy(state_energies_list,state_weights_map)
                return Erelax
            else:
                # Make a new ActiveSpaceSolver with the new ints
                as_solver_relaxed = forte.make_active_space_solver(active_space_solver_type,
                                                                   state_map,scf_info,
                                                                   mo_space_info,ints_dressed,
                                                                   options)
                state_energies_list = as_solver_relaxed.compute_energy()
                Erelax = forte.compute_average_state_energy(state_energies_list,state_weights_map)

            dsrg_energies.append((Edsrg, Erelax))

            if do_dipole:
                if is_multi_state:
                    psi4.core.print_out("\n  !DSRG transition dipoles are disabled temporarily.")
                    warnings.warn("DSRG transition dipoles are disabled temporarily.", UserWarning)
                else:
                    rdms = as_solver_relaxed.compute_average_rdms(state_weights_map, 3)
                    x, y, z, t = dipole_routine(dsrg, rdms)
                    dsrg_dipoles.append(((udm_x, udm_y, udm_z, udm_t), (x, y, z, t)))
                    psi4.core.print_out("\n\n    {} partially relaxed dipole moment:".format(correlation_solver_type))
                    psi4.core.print_out("\n      X: {:10.6f}  Y: {:10.6f}"
                                        "  Z: {:10.6f}  Total: {:10.6f}\n".format(x, y, z, t))

            # test convergence and break loop
            if abs(Edsrg - Etemp1) < relax_conv and abs(Erelax - Etemp2) < relax_conv \
                and abs(Edsrg - Erelax) < e_conv:
                converged = True
                break

            Etemp1, Etemp2 = Edsrg, Erelax

            # continue iterations
            if N + 1 != maxiter:
                # Compute the rdms in the original basis
                # rdms available if done relaxed dipole
                if do_dipole and (not is_multi_state):
                    rdms = semi.transform_rdms(Ua, Ub, rdms, max_rdm_level)
                else:
                    rdms = semi.transform_rdms(Ua, Ub, as_solver_relaxed.compute_average_rdms(state_weights_map, 3),
                                                         max_rdm_level)

                # Now semicanonicalize the reference and orbitals
                semi.semicanonicalize(rdms, max_rdm_level)
                Ua = semi.Ua_t()
                Ub = semi.Ub_t()

                # Compute DSRG in the semicanonical basis
                dsrg = forte.make_dsrg_method(correlation_solver_type, rdms,
                                              scf_info, options, ints, mo_space_info)
                dsrg.set_Uactv(Ua, Ub)
                Edsrg = dsrg.compute_energy()

                if do_dipole:
                    udm_x = psi4.core.variable('UNRELAXED DIPOLE X')
                    udm_y = psi4.core.variable('UNRELAXED DIPOLE Y')
                    udm_z = psi4.core.variable('UNRELAXED DIPOLE Z')
                    udm_t = psi4.core.variable('UNRELAXED DIPOLE')

        # printing
        if (not is_multi_state) or maxiter > 1:
            psi4.core.print_out("\n\n  => {} Reference Relaxation Energy Summary <=\n".format(correlation_solver_type))
            indent = ' ' * 4
            dash = '-' * 71
            title = indent + "{:5}  {:>31}  {:>31}\n".format(' ', "Fixed Ref. (a.u.)",
                                                             "Relaxed Ref. (a.u.)")
            title += indent + "{}  {}  {}\n".format(' ' * 5, '-' * 31, '-' * 31)
            title += indent + "{:5}  {:>20} {:>10}  {:>20} {:>10}\n".format("Iter.", "Total Energy", "Delta",
                                                                        "Total Energy", "Delta")
            psi4.core.print_out("\n{}".format(title + indent + dash))
            E0_old, E1_old = 0.0, 0.0
            for n, pair in enumerate(dsrg_energies):
                E0, E1 = pair
                psi4.core.print_out("\n{}{:>5}  {:>20.12f} {:>10.3e}"
                                    "  {:>20.12f} {:>10.3e}".format(indent, n + 1,
                                                                    E0, E0 - E0_old, E1, E1 - E1_old))
                E0_old, E1_old = E0, E1

            psi4.core.print_out("\n{}{}".format(indent, dash))

        if do_dipole and (not is_multi_state):
            psi4.core.print_out("\n\n  => {} Reference Relaxation Dipole Summary <=\n".format(correlation_solver_type))
            psi4.core.print_out("\n    {} unrelaxed dipole moment:".format(correlation_solver_type))
            psi4.core.print_out("\n      X: {:10.6f}  Y: {:10.6f}  "
                                "Z: {:10.6f}  Total: {:10.6f}\n".format(*dsrg_dipoles[0][0]))
            psi4.core.set_scalar_variable('UNRELAXED DIPOLE', dsrg_dipoles[0][0][-1])

            psi4.core.print_out("\n    {} partially relaxed dipole moment:".format(correlation_solver_type))
            psi4.core.print_out("\n      X: {:10.6f}  Y: {:10.6f}  "
                                "Z: {:10.6f}  Total: {:10.6f}\n".format(*dsrg_dipoles[0][1]))
            psi4.core.set_scalar_variable('PARTIALLY RELAXED DIPOLE', dsrg_dipoles[0][1][-1])

            if maxiter > 1:
                psi4.core.print_out("\n    {} relaxed dipole moment:".format(correlation_solver_type))
                psi4.core.print_out("\n      X: {:10.6f}  Y: {:10.6f}  "
                                    "Z: {:10.6f}  Total: {:10.6f}\n".format(*dsrg_dipoles[1][0]))
                psi4.core.set_scalar_variable('RELAXED DIPOLE', dsrg_dipoles[1][0][-1])

        # set energies to environment
        psi4.core.set_scalar_variable('PARTIALLY RELAXED ENERGY', dsrg_energies[0][1])
        if maxiter > 1:
            psi4.core.set_scalar_variable('RELAXED ENERGY', dsrg_energies[1][0])

        # throw not converging error if relaxation not converged
        if not converged:
            psi4.core.set_scalar_variable('CURRENT UNRELAXED ENERGY', dsrg_energies[-1][0])
            psi4.core.set_scalar_variable('CURRENT RELAXED ENERGY', dsrg_energies[-1][1])
            psi4.core.set_scalar_variable('CURRENT ENERGY', dsrg_energies[-1][1])
            raise psi4.core.ConvergenceError("DSRG relaxation does not converge in {} cycles".format(maxiter))
        else:
            if relax_mode != 'ONCE' and relax_mode != 'TWICE':
                psi4.core.set_scalar_variable('FULLY RELAXED ENERGY', dsrg_energies[-1][1])
                psi4.core.set_scalar_variable('CURRENT ENERGY', dsrg_energies[-1][1])
                if do_dipole and (not is_multi_state):
                    psi4.core.print_out("\n    {} fully relaxed dipole moment:".format(correlation_solver_type))
                    psi4.core.print_out("\n      X: {:10.6f}  Y: {:10.6f}  "
                                        "Z: {:10.6f}  Total: {:10.6f}\n".format(*dsrg_dipoles[-1][1]))
                    psi4.core.set_scalar_variable('FULLY RELAXED DIPOLE', dsrg_dipoles[-1][1][-1])

        return Erelax
    else:

        average_energy = forte.compute_average_state_energy(state_energies_list,state_weights_map)
        return_en = average_energy

    return return_en

def compute_dsrg_unrelaxed_energy(correlation_solver_type, rdms, scf_info, options,
                                  ints, mo_space_info, Ua, Ub):
    Heff_actv_implemented = False

    if correlation_solver_type == "MRDSRG_SO":
        dsrg = forte.make_dsrg_so_y(rdms, scf_info, options, ints, mo_space_info)
    elif correlation_solver_type == "SOMRDSRG":
        dsrg = forte.make_dsrg_so_f(rdms, scf_info, options, ints, mo_space_info)
    elif correlation_solver_type == "DSRG_MRPT":
        dsrg = forte.make_dsrg_spin_adapted(rdms, scf_info, options, ints, mo_space_info)
    else:
        Heff_actv_implemented = True
        dsrg = forte.make_dsrg_method(correlation_solver_type, rdms, scf_info, options, ints, mo_space_info)
        dsrg.set_Uactv(Ua, Ub)

    Edsrg = dsrg.compute_energy()
    psi4.core.set_scalar_variable('UNRELAXED ENERGY', Edsrg)

    return Edsrg, dsrg, Heff_actv_implemented

def orbital_projection(ref_wfn, options, mo_space_info):
    r"""Functions that pre-rotate orbitals before calculations;
    Requires a set of reference orbitals and mo_space_info.

    AVAS: an automatic active space selection and projection;
    Embedding: simple frozen-orbital embedding with the overlap projector.

    Return a mo_space_info (forte::MOSpaceInfo)
    """

    # Create the AO subspace projector
    ps = forte.make_aosubspace_projector(ref_wfn, options)

    #Apply the projector to rotate orbitals
    if options.get_bool("AVAS"):
        forte.print_method_banner(["Atomic Valence Active Space (AVAS)", "Chenxi Cai"]);
        forte.make_avas(ref_wfn, options, ps)

    # Create the fragment(embedding) projector and apply to rotate orbitals
    if options.get_bool("EMBEDDING"):
        forte.print_method_banner(["Frozen-orbital Embedding", "Nan He"]);
        pf = forte.make_fragment_projector(ref_wfn, options)
        return forte.make_embedding(ref_wfn, options, pf, mo_space_info)
    else:
        return mo_space_info

def adv_embedding_driver(state, state_weights_map, scf_info, ref_wfn, mo_space_info, options, ints):
    # options.set_bool('FORTE', 'EMBEDDING', True)
    # options.set_str('FORTE', 'EMBEDDING_CUTOFF_METHOD', 'CORRELATED_BATH')
    # options.set_str('FORTE', 'EMBEDDING_SPECIAL', 'INNER_LAYER')

    frag_corr_level = options.get_str('FRAG_CORR_LEVEL')
    env_corr_level = options.get_str('ENV_CORR_LEVEL')

    # mo_space_info: fragment all active, environment restricted, core frozen
    # mo_space_info_active: fragment-C,A,V, environment and core frozen
    mo_space_info_active = forte.build_inner_space(ref_wfn, options, mo_space_info)

    # Build fragment and f-e integrals
    ints_f = forte.make_forte_integrals(ref_wfn, options, mo_space_info_active)
    #ints_f_1 = ints_f # TODO: How the hell can I deep copy this!!!!?>??>?
    psi4.core.print_out("\n Integral test (f original): oei_a(4, 13) = {:10.8f}".format(ints_f.oei_a(4, 13)))
    ints_e = forte.make_forte_integrals(ref_wfn, options, mo_space_info)
    psi4.core.print_out("\n Integral test (f original): int_f ncmo: {:d}".format(ints_f.ncmo()))
    psi4.core.print_out("\n Integral test (e original): int_e ncmo: {:d}".format(ints_e.ncmo()))
    # compute higher-level with mo_space_info_active(inner) and methods in options, -> E_high(origin)
    options.set_str('FORTE', 'CORR_LEVEL', frag_corr_level)
    #options.set_bool('FORTE', 'SEMI_CANONICAL', False)
    forte.forte_options.update_psi_options(options)
    energy_high = forte_driver(state_weights_map, scf_info, forte.forte_options, ints_f, mo_space_info_active)
    psi4.core.print_out("\n Integral test (f_1, after ldsrg2): oei_a(4, 13) = {:10.8f}".format(ints_f.oei_a(4, 13)))

    # Form rdms for interaction correlation computation
    rdms = forte.RHF_DENSITY(scf_info, mo_space_info).rhf_rdms()
    if options.get_str('fragment_density') == "CASSCF": 
        as_ints = forte.make_active_space_ints(mo_space_info_active, ints_f, "ACTIVE", ["RESTRICTED_DOCC"])
        rdms = forte.build_casscf_density(state, 2, scf_info, forte.forte_options, mo_space_info_active, mo_space_info, as_ints) # TODO:Fix this function
    if options.get_str('fragment_density') == "FCI":
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

    # Semi-Canonicalize A+B
    # semi = forte.SemiCanonical(mo_space_info, ints_e, forte.forte_options)
    # semi.semicanonicalize(rdms, 2) # TODO: should automatically determine max_rdm_level
    dsrg = forte.make_dsrg_method(options.get_str('ENV_CORRELATION_SOLVER'), # TODO: ensure here always run canonical mr-dsrg
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
    psi4.core.print_out("\n Integral test (f_1 before dressing): oei_a(4, 13) = {:10.8f}".format(ints_f.oei_a(4, 13)))
    ints_dressed = dsrg.compute_Heff_actv()
    state_map = forte.to_state_nroots_map(state_weights_map)
    ints_f.build_from_asints(ints_dressed)
    psi4.core.print_out("\n Integral test (f_1 after dressing): oei_a(4, 13) = {:10.8f}".format(ints_f.oei_a(4, 13)))

    # Compute MRDSRG-in-PT2 energy (folded)
    options.set_str('FORTE', 'CORR_LEVEL', frag_corr_level)
    forte.forte_options.update_psi_options(options)
    energy_high_relaxed = forte_driver(state_weights_map, scf_info, forte.forte_options, ints_f, mo_space_info_active)

    psi4.core.print_out("\n Integral test (f_1 after relaxed ldsrg2): oei_a(4, 13) = {:10.8f}".format(ints_f.oei_a(4, 13)))

    psi4.core.print_out("\n ==============Embedding Summary==============")
    psi4.core.print_out("\n E(fragment, unrelaxed) = {:10.8f}".format(energy_high))
    psi4.core.print_out("\n E_corr(env correlation) = {:10.8f}".format(E_corr))
    psi4.core.print_out("\n E(embedding, unrelaxed) = {:10.8f}".format(energy_high + E_corr))
    psi4.core.print_out("\n E(fragment, Hbar2 relaxed) = {:10.8f}".format(energy_high_relaxed))
    psi4.core.print_out("\n E(embedding, Hbar2 relaxed) = {:10.8f}".format(energy_high_relaxed + E_corr))
    psi4.core.print_out("\n ==============MRDSRG embedding done============== \n")

    # Update RDMs
    ints_f.build_from_asints(ints_dressed)
    psi4.core.print_out("\n Integral test (f_1 recover to -after dressing-): oei_a(4, 13) = {:10.8f}".format(ints_f.oei_a(4, 13)))

    # Update Integrals
    ints_e.build_from_another_ints(ints_f, 44)

    psi4.core.print_out("\n")
    psi4.core.print_out("\n ==============Update and Verify RDMs============== \n")
    rdms = forte.RHF_DENSITY(scf_info, mo_space_info).rhf_rdms()
    if options.get_str('fragment_density') == "CASSCF":
        as_ints = forte.make_active_space_ints(mo_space_info_active, ints_f, "ACTIVE", ["RESTRICTED_DOCC"])
        rdms = forte.build_casscf_density(state, 2, scf_info, forte.forte_options, mo_space_info_active, mo_space_info, as_ints)
    if options.get_str('fragment_density') == "FCI":
        state_map = forte.to_state_nroots_map(state_weights_map)
        as_ints_full = forte.make_active_space_ints(mo_space_info, ints_e, "ACTIVE", ["RESTRICTED_DOCC"])
        as_solver_full = forte.make_active_space_solver(options.get_str('ACTIVE_SPACE_SOLVER'),
                                                       state_map, scf_info,
                                                       mo_space_info, as_ints_full,
                                                       forte.forte_options)
        state_energies_list = as_solver_full.compute_energy()
        rdms = as_solver_full.compute_average_rdms(state_weights_map, 3)

    psi4.core.print_out("\n Integral test (f_1 final): oei_a(4, 13) = {:10.8f}".format(ints_f.oei_a(4, 13)))
    
    # To do iteratively: 
    # ints_e = build_from_fragment_ints(ints_f)
    # Do ENV MRDSRG with ints_e

    # Compute folded casci (should use a general forte_driver instead!)
 #   as_solver_relaxed = forte.make_active_space_solver(options.get_str('ACTIVE_SPACE_SOLVER'),
 #                                                      state_map, scf_info,
 #                                                      mo_space_info_active, ints_dressed,
 #                                                      forte.forte_options)
 #   state_energies_list = as_solver_relaxed.compute_energy()
 #   Erelax = forte.compute_average_state_energy(state_energies_list,state_weights_map)
#
    # Compute relaxed(folded) MRDSRG energy 
 #   rdms_fold = as_solver_relaxed.compute_average_rdms(state_weights_map, 3)
 #   dsrg_high_fold = forte.make_dsrg_method(options.get_str('FRAG_CORRELATION_SOLVER'),
 #                                 rdms_fold, scf_info, forte.forte_options, ints_f, mo_space_info_active) # Should use int_f_dressed here!
 #   energy_high_fold = dsrg_high_fold.compute_energy()
    # Compute MRDSRG-in-PT2 energy (folded)
    # E_emb = E(MRDSRG_folded) + E(Corr)

def forte_sr_downfolding(state_weights_map, scf_info, options, ints, mo_space_info):

    rdms = forte.RHF_DENSITY(scf_info, mo_space_info).rhf_rdms()
#    if options.get_str('downfold_density') == "CASSCF": # Working here
        # Build and run a CASSCF computation
        # Compute CASSCF density

    dsrg = forte.make_dsrg_method(options.get_str('CORRELATION_SOLVER'),
                                  rdms, scf_info, options, ints, mo_space_info)
    Edsrg = dsrg.compute_energy()
    psi4.core.set_scalar_variable('UNRELAXED ENERGY', Edsrg)

    ints_dressed = dsrg.compute_Heff_actv()
    state_map = forte.to_state_nroots_map(state_weights_map)
    as_solver_relaxed = forte.make_active_space_solver(options.get_str('ACTIVE_SPACE_SOLVER'),
                                                       state_map, scf_info,
                                                       mo_space_info, ints_dressed,
                                                       options)
    state_energies_list = as_solver_relaxed.compute_energy()
    Erelax = forte.compute_average_state_energy(state_energies_list,state_weights_map)
    psi4.core.set_scalar_variable('PARTIALLY RELAXED ENERGY', Erelax)
    psi4.core.set_scalar_variable('CURRENT ENERGY', Erelax)
    return Erelax

def run_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('forte')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Get the option object
    options = psi4.core.get_options()
    options.set_current_module('FORTE')
    forte.forte_options.update_psi_options(options)

    if ('DF' in options.get_str('INT_TYPE')):
        aux_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), 'DF_BASIS_MP2',
                                         psi4.core.get_global_option('DF_BASIS_MP2'),
                                         'RIFIT', psi4.core.get_global_option('BASIS'))
        ref_wfn.set_basisset('DF_BASIS_MP2', aux_basis)

    if (options.get_str('MINAO_BASIS')):
        minao_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), 'MINAO_BASIS',
                                               options.get_str('MINAO_BASIS'))
        ref_wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Start Forte, initialize ambit
    my_proc_n_nodes = forte.startup()
    my_proc, n_nodes = my_proc_n_nodes

    # Print the banner
    forte.banner()

    # Create the MOSpaceInfo object
    mo_space_info = forte.make_mo_space_info(ref_wfn, forte.forte_options)

    # Call methods that project the orbitals (AVAS, embedding)
    mo_space_info = orbital_projection(ref_wfn, options, mo_space_info)

    state = forte.make_state_info_from_psi_wfn(ref_wfn)
    scf_info = forte.SCFInfo(ref_wfn)
    state_weights_map = forte.make_state_weights_map(forte.forte_options,ref_wfn)

    # Run a method
    job_type = options.get_str('JOB_TYPE')

    energy = 0.0

    if job_type == 'NONE':
        forte.cleanup()
        return ref_wfn

    start = timeit.timeit()

    # Make an integral object
    ints = forte.make_forte_integrals(ref_wfn, options, mo_space_info)

    # Rotate orbitals before computation
    orb_type = options.get_str("ORBITAL_TYPE")
    if orb_type != 'CANONICAL':
        orb_t = forte.make_orbital_transformation(orb_type, scf_info, forte.forte_options, ints, mo_space_info)
        orb_t.compute_transformation()
        Ua = orb_t.get_Ua()
        Ub = orb_t.get_Ub()

        ints.rotate_orbitals(Ua,Ub)

    # Run a method
    if (job_type == 'NEWDRIVER'):
        energy = forte_driver(state_weights_map, scf_info, forte.forte_options, ints, mo_space_info)
    elif (job_type == 'SR_DOWNFOLDING'):
        options.set_str('FORTE', 'RELAX_REF', 'ONCE')
        options.set_bool('FORTE', 'DSRG_SR_DOWNFOLD', True)
        forte.forte_options.update_psi_options(options)
        energy = forte_sr_downfolding(state_weights_map, scf_info, forte.forte_options, ints, mo_space_info)
    elif (job_type == 'ADV_EMBEDDING'):
        energy_df = adv_embedding_driver(state, state_weights_map, scf_info, ref_wfn, mo_space_info, options, ints)
    else:
        energy = forte.forte_old_methods(ref_wfn, options, ints, mo_space_info)

    end = timeit.timeit()
    #print('\n\n  Your calculation took ', (end - start), ' seconds');

    # Close ambit, etc.
    forte.cleanup()

    psi4.core.set_scalar_variable('CURRENT ENERGY', energy)
    return ref_wfn

# Integration with driver routines
psi4.driver.procedures['energy']['forte'] = run_forte

