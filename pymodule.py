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
import math
import warnings

import numpy as np
import psi4
import forte
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util
import forte.proc.fcidump


def forte_driver(state_weights_map, scf_info, options, ints, mo_space_info):
    max_rdm_level = 3 if options.get_str("THREEPDC") != "ZERO" else 2
    return_en = 0.0

    state_map = forte.to_state_nroots_map(state_weights_map)

    # Create an active space solver object and compute the energy
    active_space_solver_type = options.get_str('ACTIVE_SPACE_SOLVER')
    as_ints = forte.make_active_space_ints(mo_space_info, ints, "ACTIVE",
                                           ["RESTRICTED_DOCC"])
    active_space_solver = forte.make_active_space_solver(
        active_space_solver_type, state_map, scf_info, mo_space_info, as_ints,
        options)

    state_energies_list = active_space_solver.compute_energy()

    if options.get_bool('SPIN_ANALYSIS'):
        rdms = active_space_solver.compute_average_rdms(state_weights_map, 2)
        forte.perform_spin_analysis(rdms, options, mo_space_info, as_ints)

    # Notes (York):
    #     cases to run active space solver: reference relaxation, state-average dsrg
    #     cases to run contracted ci solver (will be put in ActiveSpaceSolver): contracted state-average dsrg
    Etemp1, Etemp2 = 0.0, 0.0

    # Create a dynamical correlation solver object
    correlation_solver_type = options.get_str('CORRELATION_SOLVER')
    if correlation_solver_type != 'NONE':
        # Grab the reference
        rdms = active_space_solver.compute_average_rdms(
            state_weights_map, max_rdm_level)

        # Compute unitary matrices Ua and Ub that rotate the orbitals to the semicanonical basis
        semi = forte.SemiCanonical(mo_space_info, ints, options)
        if options.get_bool("SEMI_CANONICAL"):
            semi.semicanonicalize(rdms, max_rdm_level)
        Ua = semi.Ua_t()
        Ub = semi.Ub_t()

        Edsrg, dsrg, Heff_actv_implemented = compute_dsrg_unrelaxed_energy(
            correlation_solver_type, rdms, scf_info, options, ints,
            mo_space_info, Ua, Ub)
        if not Heff_actv_implemented:
            return Edsrg

        # dipole moment related
        do_dipole = options.get_bool("DSRG_DIPOLE")
        if do_dipole:
            if correlation_solver_type == 'MRDSRG' or correlation_solver_type == 'THREE-DSRG-MRPT2':
                do_dipole = False
                psi4.core.print_out(
                    "\n  !Dipole moment is not implemented for {}.".format(
                        correlation_solver_type))
                warnings.warn("Dipole moment is not implemented for MRDSRG.",
                              UserWarning)
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
        is_multi_state = True if options.get_str(
            "CALC_TYPE") != "SS" else False

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
            raise NotImplementedError(
                "MS or XMS is disabled due to the reconstruction.")
        if ms_dsrg_algorithm == "SA_SUB" and relax_mode != 'ONCE':
            raise NotImplementedError(
                "Need to figure out how to compute relaxed SA density.")

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
                sa_sub_max_rdm = 2  # TODO: This should be 3 if do_hbar3 is true
                state_energies_list = active_space_solver.compute_contracted_energy(
                    ints_dressed, sa_sub_max_rdm)
                Erelax = forte.compute_average_state_energy(
                    state_energies_list, state_weights_map)
                return Erelax
            else:
                # Make a new ActiveSpaceSolver with the new ints
                as_solver_relaxed = forte.make_active_space_solver(
                    active_space_solver_type, state_map, scf_info,
                    mo_space_info, ints_dressed, options)
                state_energies_list = as_solver_relaxed.compute_energy()
                Erelax = forte.compute_average_state_energy(
                    state_energies_list, state_weights_map)

            dsrg_energies.append((Edsrg, Erelax))

            if do_dipole:
                if is_multi_state:
                    psi4.core.print_out(
                        "\n  !DSRG transition dipoles are disabled temporarily."
                    )
                    warnings.warn(
                        "DSRG transition dipoles are disabled temporarily.",
                        UserWarning)
                else:
                    rdms = as_solver_relaxed.compute_average_rdms(
                        state_weights_map, max_rdm_level)
                    x, y, z, t = dipole_routine(dsrg, rdms)
                    dsrg_dipoles.append(
                        ((udm_x, udm_y, udm_z, udm_t), (x, y, z, t)))
                    psi4.core.print_out(
                        "\n\n    {} partially relaxed dipole moment:".format(
                            correlation_solver_type))
                    psi4.core.print_out(
                        "\n      X: {:10.6f}  Y: {:10.6f}"
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
                    rdms = as_solver_relaxed.compute_average_rdms(
                        state_weights_map, max_rdm_level)
                    rdms = semi.transform_rdms(Ua, Ub, rdms, max_rdm_level)

                # Now semicanonicalize the reference and orbitals
                semi.semicanonicalize(rdms, max_rdm_level)
                Ua = semi.Ua_t()
                Ub = semi.Ub_t()

                # Compute DSRG in the semicanonical basis
                if correlation_solver_type == "SA-MRDSRG":
                    dsrg = forte.make_sadsrg_method(rdms, scf_info, options,
                                                    ints, mo_space_info)
                    dsrg.set_Uactv(Ua)
                else:
                    dsrg = forte.make_dsrg_method(correlation_solver_type,
                                                  rdms, scf_info, options,
                                                  ints, mo_space_info)
                    dsrg.set_Uactv(Ua, Ub)
                Edsrg = dsrg.compute_energy()

                if do_dipole:
                    udm_x = psi4.core.variable('UNRELAXED DIPOLE X')
                    udm_y = psi4.core.variable('UNRELAXED DIPOLE Y')
                    udm_z = psi4.core.variable('UNRELAXED DIPOLE Z')
                    udm_t = psi4.core.variable('UNRELAXED DIPOLE')

        # printing
        if (not is_multi_state) or maxiter > 1:
            psi4.core.print_out(
                "\n\n  => {} Reference Relaxation Energy Summary <=\n".format(
                    correlation_solver_type))
            indent = ' ' * 4
            dash = '-' * 71
            title = indent + "{:5}  {:>31}  {:>31}\n".format(
                ' ', "Fixed Ref. (a.u.)", "Relaxed Ref. (a.u.)")
            title += indent + "{}  {}  {}\n".format(' ' * 5, '-' * 31,
                                                    '-' * 31)
            title += indent + "{:5}  {:>20} {:>10}  {:>20} {:>10}\n".format(
                "Iter.", "Total Energy", "Delta", "Total Energy", "Delta")
            psi4.core.print_out("\n{}".format(title + indent + dash))
            E0_old, E1_old = 0.0, 0.0
            for n, pair in enumerate(dsrg_energies):
                E0, E1 = pair
                psi4.core.print_out("\n{}{:>5}  {:>20.12f} {:>10.3e}"
                                    "  {:>20.12f} {:>10.3e}".format(
                                        indent, n + 1, E0, E0 - E0_old, E1,
                                        E1 - E1_old))
                E0_old, E1_old = E0, E1

            psi4.core.print_out("\n{}{}".format(indent, dash))

        if do_dipole and (not is_multi_state):
            psi4.core.print_out(
                "\n\n  => {} Reference Relaxation Dipole Summary <=\n".format(
                    correlation_solver_type))
            psi4.core.print_out("\n    {} unrelaxed dipole moment:".format(
                correlation_solver_type))
            psi4.core.print_out(
                "\n      X: {:10.6f}  Y: {:10.6f}  "
                "Z: {:10.6f}  Total: {:10.6f}\n".format(*dsrg_dipoles[0][0]))
            psi4.core.set_scalar_variable('UNRELAXED DIPOLE',
                                          dsrg_dipoles[0][0][-1])

            psi4.core.print_out(
                "\n    {} partially relaxed dipole moment:".format(
                    correlation_solver_type))
            psi4.core.print_out(
                "\n      X: {:10.6f}  Y: {:10.6f}  "
                "Z: {:10.6f}  Total: {:10.6f}\n".format(*dsrg_dipoles[0][1]))
            psi4.core.set_scalar_variable('PARTIALLY RELAXED DIPOLE',
                                          dsrg_dipoles[0][1][-1])

            if maxiter > 1:
                psi4.core.print_out("\n    {} relaxed dipole moment:".format(
                    correlation_solver_type))
                psi4.core.print_out("\n      X: {:10.6f}  Y: {:10.6f}  "
                                    "Z: {:10.6f}  Total: {:10.6f}\n".format(
                                        *dsrg_dipoles[1][0]))
                psi4.core.set_scalar_variable('RELAXED DIPOLE',
                                              dsrg_dipoles[1][0][-1])

        # set energies to environment
        psi4.core.set_scalar_variable('PARTIALLY RELAXED ENERGY',
                                      dsrg_energies[0][1])
        if maxiter > 1:
            psi4.core.set_scalar_variable('RELAXED ENERGY',
                                          dsrg_energies[1][0])

        # throw not converging error if relaxation not converged
        if not converged:
            psi4.core.set_scalar_variable('CURRENT UNRELAXED ENERGY',
                                          dsrg_energies[-1][0])
            psi4.core.set_scalar_variable('CURRENT RELAXED ENERGY',
                                          dsrg_energies[-1][1])
            psi4.core.set_scalar_variable('CURRENT ENERGY',
                                          dsrg_energies[-1][1])
            raise psi4.core.ConvergenceError(
                "DSRG relaxation does not converge in {} cycles".format(
                    maxiter))
        else:
            if relax_mode != 'ONCE' and relax_mode != 'TWICE':
                psi4.core.set_scalar_variable('FULLY RELAXED ENERGY',
                                              dsrg_energies[-1][1])
                psi4.core.set_scalar_variable('CURRENT ENERGY',
                                              dsrg_energies[-1][1])
                if do_dipole and (not is_multi_state):
                    psi4.core.print_out(
                        "\n    {} fully relaxed dipole moment:".format(
                            correlation_solver_type))
                    psi4.core.print_out(
                        "\n      X: {:10.6f}  Y: {:10.6f}  "
                        "Z: {:10.6f}  Total: {:10.6f}\n".format(
                            *dsrg_dipoles[-1][1]))
                    psi4.core.set_scalar_variable('FULLY RELAXED DIPOLE',
                                                  dsrg_dipoles[-1][1][-1])

        return Erelax
    else:

        average_energy = forte.compute_average_state_energy(
            state_energies_list, state_weights_map)
        return_en = average_energy

    return return_en


def compute_dsrg_unrelaxed_energy(correlation_solver_type, rdms, scf_info,
                                  options, ints, mo_space_info, Ua, Ub):
    Heff_actv_implemented = False

    if correlation_solver_type == "SA-MRDSRG":
        dsrg = forte.make_sadsrg_method(rdms, scf_info, options, ints,
                                        mo_space_info)
        Heff_actv_implemented = True
        dsrg.set_Uactv(Ua)
    elif correlation_solver_type == "MRDSRG_SO":
        dsrg = forte.make_dsrg_so_y(rdms, scf_info, options, ints,
                                    mo_space_info)
    elif correlation_solver_type == "SOMRDSRG":
        dsrg = forte.make_dsrg_so_f(rdms, scf_info, options, ints,
                                    mo_space_info)
    elif correlation_solver_type == "DSRG_MRPT":
        dsrg = forte.make_dsrg_spin_adapted(rdms, scf_info, options, ints,
                                            mo_space_info)
    else:
        Heff_actv_implemented = True
        dsrg = forte.make_dsrg_method(correlation_solver_type, rdms, scf_info,
                                      options, ints, mo_space_info)
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
        forte.print_method_banner(
            ["Atomic Valence Active Space (AVAS)", "Chenxi Cai"])
        forte.make_avas(ref_wfn, options, ps)

    # Create the fragment(embedding) projector and apply to rotate orbitals
    if options.get_bool("EMBEDDING"):
        forte.print_method_banner(["Frozen-orbital Embedding", "Nan He"])
        fragment_projector, fragment_nbf = forte.make_fragment_projector(
            ref_wfn, options)
        return forte.make_embedding(ref_wfn, options, fragment_projector,
                                    fragment_nbf, mo_space_info)
    else:
        return mo_space_info


#def get_options_from_psi(forte_options, psi_options):
#     forte_dict = forte_options.dict()
#     for key, value in forte_dict.items():
#         v_type = value['type']
#         if v_type == 'bool':
#             forte_dict[key]['value'] = psi_options.get_bool(key)
#         elif v_type == 'int':
#             forte_dict[key]['value'] = psi_options.get_int(key)
#         elif v_type == 'float':
#             forte_dict[key]['value'] = psi_options.get_double(key)
#         elif v_type == 'str':
#             forte_dict[key]['value'] = psi_options.get_str(key)
#         elif v_type == 'int_list':
#             forte_dict[key]['value'] = [i for i in psi_options.get_int_vector(key)]
#         elif v_type == 'float_list':
#             forte_dict[key]['value'] = [i for i in psi_options.get_double_vector(key)]
#         elif v_type == 'gen_list':
#             nitems = psi_options.get(key).size()
#             forte_dict[key]['value'] = [psi_options.get(key)[i].to_string() for i in range(nitems)]
##             forte_dict[key]['value'] = [i for i in psi_options.get_double_vector(key)]
#         else:
#             print(key, value, type(psi_options.get_int_vector(key)))
#             # TODO: need to do this recursively
#     return forte_dict


def prepare_forte_objects_from_psi4_wfn(options, wfn):
    """
    Take a psi4 wavefunction object and prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects

    Parameters
    ----------
    wfn : psi4Wavefunction
        A psi4 Wavefunction object

    Returns
    -------
    tuple(ForteIntegrals, SCFInfo, MOSpaceInfo)
        a tuple containing the ForteIntegrals, SCFInfo, and MOSpaceInfo objects
    """

    if 'DF' in options.get_str('INT_TYPE'):
        aux_basis = psi4.core.BasisSet.build(wfn.molecule(), 'DF_BASIS_MP2',
                                             options.get_str('DF_BASIS_MP2'),
                                             'RIFIT', options.get_str('BASIS'))
        wfn.set_basisset('DF_BASIS_MP2', aux_basis)

    if options.get_str('MINAO_BASIS'):
        minao_basis = psi4.core.BasisSet.build(wfn.molecule(), 'MINAO_BASIS',
                                               options.get_str('MINAO_BASIS'))
        wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Create the MOSpaceInfo object
    nmopi = wfn.nmopi()
    mo_space_info = forte.make_mo_space_info(nmopi, options)

    # Call methods that project the orbitals (AVAS, embedding)
    mo_space_info = orbital_projection(wfn, options, mo_space_info)

    # Averaging spin multiplets if doing spin-adapted computation
    if options.get_str('CORRELATION_SOLVER') == 'SA-MRDSRG':
        options_dict = options.dict()
        options_dict['SPIN_AVG_DENSITY']['value'] = True
        options.set_dict(options_dict)

    scf_info = forte.SCFInfo(wfn)
    state_weights_map = forte.make_state_weights_map(options, mo_space_info)

    return (state_weights_map, mo_space_info, scf_info)


def make_state_info_from_fcidump(fcidump, options):
    nel = fcidump['nelec']
    if not options.is_none("NEL"):
        nel = options.get_int("NEL")

    multiplicity = fcidump['ms2'] + 1
    if not options.is_none("MULTIPLICITY"):
        multiplicity = options.get_int("MULTIPLICITY")

    # If the user did not specify ms determine the value from the input or
    # take the lowest value consistent with the value of "MULTIPLICITY"
    # For example:
    #    singlet: multiplicity = 1 -> twice_ms = 0 (ms = 0)
    #    doublet: multiplicity = 2 -> twice_ms = 1 (ms = 1/2)
    #    triplet: multiplicity = 3 -> twice_ms = 0 (ms = 0)
    twice_ms = (multiplicity + 1) % 2
    if not options.is_none("MS"):
        twice_ms = int(round(2.0 * options.get_double("MS")))

    if (((nel - twice_ms) % 2) != 0):
        raise Exception(f'Forte: the value of MS ({twice_ms}/2) is incompatible with the number of electrons ({nel})')

    na = (nel + twice_ms) // 2
    nb = nel - na

    irrep = fcidump['isym']
    if not options.is_none("ROOT_SYM"):
        irrep = options.get_int("ROOT_SYM")

    return forte.StateInfo(na, nb, multiplicity, twice_ms, irrep)


def prepare_forte_objects_from_fcidump(options):
    filename = options.get_str('FCIDUMP_FILE')
    psi4.core.print_out(
        f'\n  Reading integral information from FCIDUMP file {filename}')
    fcidump = forte.proc.fcidump_from_file(filename, convert_to_psi4=True)

    irrep_size = {
        'c1': 1,
        'ci': 2,
        'c2': 2,
        'cs': 2,
        'd2': 4,
        'c2v': 4,
        'c2h': 4,
        'd2h': 8
    }

    nmo = len(fcidump['orbsym'])
    if 'pntgrp' in fcidump:
        nirrep = irrep_size[fcidump['pntgrp'].lower()]
        nmopi_list = [fcidump['orbsym'].count(x) for x in range(nirrep)]
    else:
        fcidump['isym'] = 0  # shift by -1
        nirrep = 1
        nmopi_list = [nmo]

    nmopi = psi4.core.Dimension(nmopi_list)

    # Create the MOSpaceInfo object
    mo_space_info = forte.make_mo_space_info(nmopi, options)

    # Call methods that project the orbitals (AVAS, embedding)
    # skipped due to lack of functionality

    # Averaging spin multiplets if doing spin-adapted computation
    if options.get_str('CORRELATION_SOLVER') == 'SA-MRDSRG':
        options_dict = options.dict()
        options_dict['SPIN_AVG_DENSITY']['value'] = True
        options.set_dict(options_dict)

    # manufacture a SCFInfo object from the FCIDUMP file (this assumes C1 symmetry)
    nel = fcidump['nelec']
    ms2 = fcidump['ms2']
    na = (nel + ms2) // 2
    nb = nel - na
    doccpi = psi4.core.Dimension([nb])
    soccpi = psi4.core.Dimension([ms2])
    if 'epsilon' in fcidump:
        epsilon = psi4.core.Vector.from_array(fcidump['epsilon'])
    else:
        epsilon = psi4.core.Vector(nmopi)
    scf_info = forte.SCFInfo(doccpi, soccpi, 0.0, epsilon, epsilon)

    state_info = make_state_info_from_fcidump(fcidump, options)
    state_weights_map = {state_info: [1.0]}
    return (state_weights_map, mo_space_info, scf_info, fcidump)


def make_ints_from_fcidump(fcidump, options, mo_space_info):
    # transform two-electron integrals from chemist to physicist notation
    eri = fcidump['eri']
    nmo = fcidump['norb']
    eri_aa = np.zeros((nmo, nmo, nmo, nmo))
    eri_ab = np.zeros((nmo, nmo, nmo, nmo))
    eri_bb = np.zeros((nmo, nmo, nmo, nmo))
    # <ij||kl> = (ik|jl) - (il|jk)
    eri_aa += np.einsum('ikjl->ijkl', eri)
    eri_aa -= np.einsum('iljk->ijkl', eri)
    # <ij|kl> = (ik|jl)
    eri_ab = np.einsum('ikjl->ijkl', eri)
    # <ij||kl> = (ik|jl) - (il|jk)
    eri_bb += np.einsum('ikjl->ijkl', eri)
    eri_bb -= np.einsum('iljk->ijkl', eri)

    return forte.make_custom_ints(options, mo_space_info, fcidump['enuc'],
                                  fcidump['hcore'].flatten(),
                                  fcidump['hcore'].flatten(), eri_aa.flatten(),
                                  eri_ab.flatten(), eri_bb.flatten())


def run_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('forte')

    """

    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Start Forte, initialize ambit
    my_proc_n_nodes = forte.startup()
    my_proc, n_nodes = my_proc_n_nodes

    # Get the option object
    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')

    # Get the forte option object
    options = forte.forte_options
    options.get_options_from_psi4(psi4_options)

    # Print the banner
    forte.banner()

    if 'FCIDUMP' in options.get_str('INT_TYPE'):
        psi4.core.print_out('\n  Preparing forte objects from a custom source')
        state_weights_map, mo_space_info, scf_info, fcidump = prepare_forte_objects_from_fcidump(
            options)
    else:
        psi4.core.print_out(
            '\n  Preparing forte objects from a psi4 Wavefunction object')
        # Compute a SCF reference using psi4 and obtain a wavefunction object
        # which holds the molecule used, orbitals, Fock matrices, and more
        ref_wfn = kwargs.get('ref_wfn', None)
        if ref_wfn is None:
            psi4.core.print_out(
                '\n  No reference wavefunction provided. Computing one with psi4\n'
            )
            ref_wfn = psi4.driver.scf_helper(name, **kwargs)

        state_weights_map, mo_space_info, scf_info = prepare_forte_objects_from_psi4_wfn(
            options, ref_wfn)

    # Run a method
    job_type = options.get_str('JOB_TYPE')

    if job_type == 'NONE':
        forte.cleanup()
        return ref_wfn

    start_pre_ints = time.time()

    if 'FCIDUMP' in options.get_str('INT_TYPE'):
        psi4.core.print_out('\n  Forte will use custom integrals')
        # Make an integral object from the psi4 wavefunction object
        ints = make_ints_from_fcidump(fcidump, options, mo_space_info)
    else:
        psi4.core.print_out('\n  Forte will use psi4 integrals')
        # Make an integral object from the psi4 wavefunction object
        ints = forte.make_ints_from_psi4(ref_wfn, options, mo_space_info)

    start = time.time()

    # Rotate orbitals before computation (e.g. localization, MP2 natural orbitals, etc.)
    orb_type = options.get_str("ORBITAL_TYPE")
    if orb_type != 'CANONICAL':
        orb_t = forte.make_orbital_transformation(orb_type, scf_info, options,
                                                  ints, mo_space_info)
        orb_t.compute_transformation()
        Ua = orb_t.get_Ua()
        Ub = orb_t.get_Ub()
        ints.rotate_orbitals(Ua, Ub)

    # Run a method
    energy = 0.0

    if (options.get_bool("CASSCF_REFERENCE") == True or job_type == "CASSCF"):
        if options.get_str('INT_TYPE') == 'FCIDUMP':
            raise Exception('Forte: the CASSCF code cannot use integrals read from a FCIDUMP file')

        casscf = forte.make_casscf(state_weights_map, scf_info, options, mo_space_info, ints)
        energy = casscf.compute_energy()

    if (job_type == "MCSCF_TWO_STEP"):
        casscf = forte.make_mcscf_two_step(state_weights_map, scf_info, options, mo_space_info, ints)
        energy = casscf.compute_energy()

    if (job_type == 'NEWDRIVER'):
        energy = forte_driver(state_weights_map, scf_info, options, ints,
                              mo_space_info)
    elif (job_type == 'MR-DSRG-PT2'):
        energy = forte.forte_old_methods(ref_wfn, options, ints, mo_space_info)

    end = time.time()

    # Close ambit, etc.
    forte.cleanup()

    psi4.core.set_scalar_variable('CURRENT ENERGY', energy)

    psi4.core.print_out(
        f'\n\n  Time to prepare integrals: {start - start_pre_ints:12.3f} seconds'
    )
    psi4.core.print_out(
        f'\n  Time to run job          : {end - start:12.3f} seconds')
    psi4.core.print_out(
        f'\n  Total                    : {end - start:12.3f} seconds')

    if 'FCIDUMP' not in options.get_str('INT_TYPE'):
        return ref_wfn


def gradient_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> gradient('forte')
        available for : CASSCF

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Get the psi4 option object
    optstash = p4util.OptionsState(['GLOBALS', 'DERTYPE'])
    psi4.core.set_global_option('DERTYPE', 'FIRST')

    # Start Forte, initialize ambit
    my_proc_n_nodes = forte.startup()
    my_proc, n_nodes = my_proc_n_nodes

    # Get the option object
    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')

    # Get the forte option object
    options = forte.forte_options
    options.get_options_from_psi4(psi4_options)

    # Print the banner
    forte.banner()

    if 'FCIDUMP' in options.get_str('INT_TYPE'):
        psi4.core.print_out(
            '\n  Gradients are not implemented for non-psi4 integrals')
        exit()
    else:
        psi4.core.print_out(
            '\n  Preparing forte objects from a psi4 Wavefunction object')
        # Compute a SCF reference using psi4 and obtain a wavefunction object
        # which holds the molecule used, orbitals, Fock matrices, and more
        ref_wfn = kwargs.get('ref_wfn', None)
        if ref_wfn is None:
            psi4.core.print_out(
                '\n  No reference wavefunction provided. Computing one with psi4\n'
            )
            ref_wfn = psi4.driver.scf_helper(name, **kwargs)

        state_weights_map, mo_space_info, scf_info = prepare_forte_objects_from_psi4_wfn(
            options, ref_wfn)

    # Run a method
    job_type = options.get_str('JOB_TYPE')

    energy = 0.0

    if not job_type == 'CASSCF':
        raise Exception('analytic gradient is only implemented for CASSCF')

    start = time.time()

    # Make an integral object
    ints = forte.make_ints_from_psi4(ref_wfn, options, mo_space_info)

    # Rotate orbitals before computation
    orb_type = options.get_str("ORBITAL_TYPE")
    if orb_type != 'CANONICAL':
        orb_t = forte.make_orbital_transformation(orb_type, scf_info, options,
                                                  ints, mo_space_info)
        orb_t.compute_transformation()
        Ua = orb_t.get_Ua()
        Ub = orb_t.get_Ub()

        ints.rotate_orbitals(Ua, Ub)

    # Run gradient computation
    energy = forte.forte_old_methods(ref_wfn, options, ints, mo_space_info)

    casscf = forte.make_casscf(state_weights_map, scf_info, options, mo_space_info, ints)
    energy = casscf.compute_energy()
    casscf.compute_gradient();

    derivobj = psi4.core.Deriv(ref_wfn)
    derivobj.set_deriv_density_backtransformed(True)
    derivobj.set_ignore_reference(True)
    grad = derivobj.compute(psi4.core.DerivCalcType.Correlated)
    ref_wfn.set_gradient(grad)
    optstash.restore()

    end = time.time()
    #print('\n\n  Your calculation took ', (end - start), ' seconds');

    # Close ambit, etc.
    forte.cleanup()

    return ref_wfn


# Integration with driver routines
psi4.driver.procedures['energy']['forte'] = run_forte
psi4.driver.procedures['gradient']['forte'] = gradient_forte
