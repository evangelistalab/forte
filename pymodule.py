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
    max_rdm_level = 3 # TODO: set this (Francesco)
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
        rdms = active_space_solver.compute_average_rdms(state_weights_map, 3) # TODO: max_rdm_level should be chosen in a smart way

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
    else : 

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
    else:
        energy = forte.forte_old_methods(ref_wfn, options, ints, mo_space_info)

    end = timeit.timeit()
    #print('\n\n  Your calculation took ', (end - start), ' seconds');

    # Close ambit, etc.
    forte.cleanup()

    psi4.core.set_scalar_variable('CURRENT ENERGY', energy)
    return ref_wfn



def gradient_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> gradient('forte') 
        available for : CASSCF

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Get the option object
    optstash = p4util.OptionsState(['GLOBALS', 'DERTYPE'])
    options = psi4.core.get_options()
    options.set_current_module('FORTE')
    forte.forte_options.update_psi_options(options)

    if ('DF' in options.get_str('INT_TYPE')):
        raise Exception('analytic gradient is not implemented for density fitting')

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

    if not job_type == 'CASSCF':
        raise Exception('analytic gradient is only implemented for CASSCF')

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
    # Run gradient computation
    energy = forte.forte_old_methods(ref_wfn, options, ints, mo_space_info)
    derivobj = psi4.core.Deriv(ref_wfn)
    derivobj.set_deriv_density_backtransformed(True)
    derivobj.set_ignore_reference(True)
    grad = derivobj.compute(psi4.core.DerivCalcType.Correlated)
    ref_wfn.set_gradient(grad)    
    optstash.restore()        

    end = timeit.timeit()
    #print('\n\n  Your calculation took ', (end - start), ' seconds');

    # Close ambit, etc.
    forte.cleanup()

    return ref_wfn


# Integration with driver routines
psi4.driver.procedures['energy']['forte'] = run_forte
psi4.driver.procedures['gradient']['forte'] = gradient_forte

