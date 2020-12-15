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
from forte.proc.dsrg import ProcedureDSRG
from forte.proc.orthogonalize_orbitals import ortho_orbs_forte


def forte_driver(state_weights_map, scf_info, options, ints, mo_space_info):
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


def orbital_projection(ref_wfn, options, mo_space_info):
    r"""Functions that pre-rotate orbitals before calculations;
    Requires a set of reference orbitals and mo_space_info.

    AVAS: an automatic active space selection and projection;
    Embedding: simple frozen-orbital embedding with the overlap projector.

    Return a mo_space_info (forte::MOSpaceInfo)
    """

    # Create the AO subspace projector
    ps = forte.make_aosubspace_projector(ref_wfn, options)

    # Apply the projector to rotate orbitals
    if options.get_bool("AVAS"):
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


def prepare_forte_objects_from_psi4_wfn(options, wfn, mo_space_info):
    """
    Take a psi4 wavefunction object and prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects

    Parameters
    ----------
    options : ForteOptions
        A Forte ForteOptions object
    wfn : psi4 Wavefunction
        A psi4 Wavefunction object
    mo_space_info : the MO space info read from options
        A Forte MOSpaceInfo object

    Returns
    -------
    tuple(ForteIntegrals, SCFInfo, MOSpaceInfo)
        a tuple containing the ForteIntegrals, SCFInfo, and MOSpaceInfo objects
    """

    # Call methods that project the orbitals (AVAS, embedding)
    mo_space_info = orbital_projection(wfn, options, mo_space_info)

    # Build Forte SCFInfo object
    scf_info = forte.SCFInfo(wfn)

    # Build a map from Forte StateInfo to the weights
    state_weights_map = forte.make_state_weights_map(options, mo_space_info)

    return (state_weights_map, mo_space_info, scf_info)


def prepare_psi4_ref_wfn(name, options, **kwargs):
    """
    Prepare a Psi4 Wavefunction as reference for Forte.
    :param name: the name of the module associated with Psi4
    :param options: a ForteOptions object for options
    :param kwargs: named arguments associated with Psi4
    :return: (the processed Psi4 Wavefunction, a Forte MOSpaceInfo object)

    Notes:
        We will create a new Psi4 Wavefunction (wfn_new) if necessary.

        1. For an empty ref_wfn, wfn_new will come from Psi4 SCF or MCSCF.

        2. For a valid ref_wfn, we will test the orbital orthonormality against molecule.
           If the orbitals from ref_wfn are consistent with the active geometry,
           wfn_new will simply be a link to ref_wfn.
           If not, we will rerun a Psi4 SCF and orthogonalize orbitals, where
           wfn_new comes from this new Psi4 SCF computation.
    """
    p4print = psi4.core.print_out

    # grab reference Wavefunction and Molecule from kwargs
    kwargs = p4util.kwargs_lower(kwargs)

    ref_wfn = kwargs.get('ref_wfn', None)

    molecule = kwargs.pop('molecule', psi4.core.get_active_molecule())
    point_group = molecule.point_group().symbol()

    if ref_wfn is None:
        ref_type = options.get_str('REF_TYPE')

        p4print('\n  No reference wave function provided for Forte.'
                f' Computing {ref_type} orbitals with Psi4 ...\n')

        if ref_type == 'SCF':
            # no warning printing for MCSCF
            job_type = options.get_str('JOB_TYPE')
            do_mcscf = (job_type in ["CASSCF", "MCSCF_TWO_STEP"] or
                        options.get_bool("CASSCF_REFERENCE"))

            if not do_mcscf:
                msg = ['Forte is using orbitals from a Psi4 SCF reference.',
                       'This is not the best for multireference computations.',
                       'To use Psi4 CASSCF orbitals, set REF_TYPE to CASSCF.']
                msg = '\n  '.join(msg)
                warnings.warn(f"\n  {msg}\n", UserWarning)

            wfn_new = psi4.driver.scf_helper(name, molecule=molecule, **kwargs)
        elif ref_type == 'CASSCF':
            wfn_new = psi4.proc.run_detcas('casscf', molecule=molecule, **kwargs)
        else:
            raise ValueError(f"Given REF_TYPE ({ref_type}) not available!")

        # build Forte MOSpaceInfo
        nmopi = wfn_new.nmopi()
        mo_space_info = forte.make_mo_space_info(nmopi, point_group, options)
    else:
        basis = options.get_str('BASIS')
        wfn_new = psi4.core.Wavefunction.build(molecule, basis)

        if psi4.core.get_global_option("RELATIVISTIC") in ["X2C", "DKH"]:
            basis_rel = options.get_str("BASIS_RELATIVISTIC")
            puream = wfn_new.basisset().has_puream()
            rel_bas = psi4.core.BasisSet.build(molecule, "BASIS_RELATIVISTIC",
                                               basis_rel, "DECON", basis,
                                               puream=puream)
            wfn_new.set_basisset('BASIS_RELATIVISTIC', rel_bas)

        # build Forte MOSpaceInfo
        nmopi = ref_wfn.nmopi()
        mo_space_info = forte.make_mo_space_info(nmopi, point_group, options)

        p4print("\n\n  Forte testing orbital orthonormality ...")

        # build MO overlap
        mints = wfn_new.mintshelper()
        S = mints.so_overlap()
        Ca = ref_wfn.Ca()
        mo_overlap = psi4.core.triplet(Ca, S, Ca, True, False, False)
        mo_overlap.zero_diagonal()
        absmax = mo_overlap.absmax()

        # test orbital orthonormality
        if absmax > 1.0e-8:
            p4print("\n\n  Forte Warning: ")
            p4print("Orbitals of ref_wfn NOT from current geometry!!!")
            p4print(f"\n  Max value of MO overlap: {absmax:.15f}")
            p4print("\n  Perform new SCF at current geometry ...\n")

            kwargs_copy = {k: v for k, v in kwargs.items() if k != 'ref_wfn'}
            wfn_new = psi4.driver.scf_helper(name, molecule=molecule, **kwargs_copy)

            # orthonormalize orbitals
            wfn_new.Ca().copy(ortho_orbs_forte(wfn_new, mo_space_info, Ca))

            # copy wfn_new to ref_wfn
            ref_wfn.shallow_copy(wfn_new)
        else:
            p4print(" Done (OK)\n\n")

            wfn_new = ref_wfn

            # read orbitals from file
            if options.get_bool('READ_ORBITALS'):
                Cold = read_orbitals()
                if Cold is not False:
                    p4print("\n\n  Forte: Read orbitals from file!\n\n")
                    wfn_new.Ca().copy(Cold)

    # set DF and MINAO basis
    if 'DF' in options.get_str('INT_TYPE'):
        aux_basis = psi4.core.BasisSet.build(molecule, 'DF_BASIS_MP2',
                                             options.get_str('DF_BASIS_MP2'),
                                             'RIFIT', options.get_str('BASIS'))
        wfn_new.set_basisset('DF_BASIS_MP2', aux_basis)

    if options.get_str('MINAO_BASIS'):
        minao_basis = psi4.core.BasisSet.build(molecule, 'MINAO_BASIS',
                                               options.get_str('MINAO_BASIS'))
        wfn_new.set_basisset('MINAO_BASIS', minao_basis)

    return wfn_new, mo_space_info


def read_orbitals():
    """ Read orbitals from file. """
    try:
        with open('forte_Ca.npy', 'rb') as f:
            Ca_array = np.load(f, allow_pickle=True)
        Ca_list = [Ca_array[i] for i in range(len(Ca_array))]  # to list
        Ca_mat = psi4.core.Matrix.from_array(Ca_list)
        return Ca_mat
    except FileNotFoundError:
        return False


def dump_orbitals(wfn):
    """ Dump orbitals to file. """
    Ca = wfn.Ca()
    with open('forte_Ca.npy', 'wb') as f:
        if wfn.nirrep() == 1:
            np.save(f, [Ca.to_array()])
        else:
            np.save(f, Ca.to_array())


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
        raise Exception(
            f'Forte: the value of MS ({twice_ms}/2) is incompatible with the number of electrons ({nel})'
        )

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
        fcidump['pntgrp'] = 'C1'  # set the point group to C1
        fcidump['isym'] = 0  # shift by -1
        nirrep = 1
        nmopi_list = [nmo]

    nmopi = psi4.core.Dimension(nmopi_list)

    # Create the MOSpaceInfo object
    mo_space_info = forte.make_mo_space_info(nmopi, fcidump['pntgrp'], options)

    # Call methods that project the orbitals (AVAS, embedding)
    # skipped due to lack of functionality

    # Averaging spin multiplets if doing spin-adapted computation
    if options.get_str('CORRELATION_SOLVER') == 'SA-MRDSRG':
        options.set_bool('SPIN_AVG_DENSITY', True)

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


def prepare_forte_options():
    """
    Return a ForteOptions object.
    """
    # Get the option object
    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')

    # Get the forte option object
    options = forte.forte_options
    options.get_options_from_psi4(psi4_options)

    # Averaging spin multiplets if doing spin-adapted computation
    if options.get_str('CORRELATION_SOLVER') in ('SA-MRDSRG', 'SA_MRDSRG'):
        options.set_bool('SPIN_AVG_DENSITY', True)

    return options


def prepare_forte_objects(options, name, **kwargs):
    """
    Prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects.
    :param options: the ForteOptions object
    :param name: the name of the module associated with Psi4
    :param kwargs: named arguments associated with Psi4
    :return: a tuple of (Wavefunction, ForteIntegrals, SCFInfo, MOSpaceInfo, FCIDUMP)
    """
    lowername = name.lower().strip()

    if 'FCIDUMP' in options.get_str('INT_TYPE'):
        if 'FIRST' in options.get_str('DERTYPE'):
            raise Exception("Energy gradients NOT available for custom integrals!")

        psi4.core.print_out('\n  Preparing forte objects from a custom source\n')
        forte_objects = prepare_forte_objects_from_fcidump(options)
        state_weights_map, mo_space_info, scf_info, fcidump = forte_objects
        ref_wfn = None
    else:
        psi4.core.print_out('\n  Preparing forte objects from a Psi4 Wavefunction object\n')
        ref_wfn, mo_space_info = prepare_psi4_ref_wfn(lowername, options, **kwargs)
        forte_objects = prepare_forte_objects_from_psi4_wfn(options, ref_wfn, mo_space_info)
        state_weights_map, mo_space_info, scf_info = forte_objects
        fcidump = None

    return ref_wfn, state_weights_map, mo_space_info, scf_info, fcidump


def run_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('forte')

    """

    # Start Forte, initialize ambit
    my_proc_n_nodes = forte.startup()
    my_proc, n_nodes = my_proc_n_nodes

    # Build Forte options
    options = prepare_forte_options()

    # Print the banner
    forte.banner()

    # Prepare Forte objects: state_weights_map, mo_space_info, scf_info
    forte_objects = prepare_forte_objects(options, name, **kwargs)
    ref_wfn, state_weights_map, mo_space_info, scf_info, fcidump = forte_objects

    # Run a method
    job_type = options.get_str('JOB_TYPE')

    if job_type == 'NONE':
        psi4.core.set_scalar_variable('CURRENT ENERGY', 0.0)
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

    if (options.get_bool("CASSCF_REFERENCE") or job_type == "CASSCF"):
        if options.get_str('INT_TYPE') == 'FCIDUMP':
            raise Exception('Forte: the CASSCF code cannot use integrals read'
                            ' from a FCIDUMP file')

        casscf = forte.make_casscf(state_weights_map, scf_info, options,
                                   mo_space_info, ints)
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
        f'\n  Total                    : {end - start_pre_ints:12.3f} seconds')

    if 'FCIDUMP' not in options.get_str('INT_TYPE'):
        if options.get_bool('DUMP_ORBITALS'):
            dump_orbitals(ref_wfn)
        return ref_wfn


def gradient_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> gradient('forte')
        available for : CASSCF
    """

    # Start Forte, initialize ambit
    my_proc_n_nodes = forte.startup()
    my_proc, n_nodes = my_proc_n_nodes

    # Get the psi4 option object
    optstash = p4util.OptionsState(['GLOBALS', 'DERTYPE'])
    psi4.core.set_global_option('DERTYPE', 'FIRST')

    # Build Forte options
    options = prepare_forte_options()

    # Print the banner
    forte.banner()

    # Run a method
    job_type = options.get_str('JOB_TYPE')

    if job_type not in {"CASSCF", "MCSCF_TWO_STEP"}:
        raise Exception('Analytic energy gradients are only implemented for job_types CASSCF and MCSCF_TWO_STEP.')

    # Prepare Forte objects: state_weights_map, mo_space_info, scf_info
    forte_objects = prepare_forte_objects(options, name, **kwargs)
    ref_wfn, state_weights_map, mo_space_info, scf_info, fcidump = forte_objects

    # Make an integral object
    time_pre_ints = time.time()

    ints = forte.make_ints_from_psi4(ref_wfn, options, mo_space_info)

    start = time.time()

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

    if job_type == "CASSCF":
        casscf = forte.make_casscf(state_weights_map, scf_info, options,
                                   mo_space_info, ints)
        energy = casscf.compute_energy()
        casscf.compute_gradient();

    if job_type == "MCSCF_TWO_STEP":
        casscf = forte.make_mcscf_two_step(state_weights_map, scf_info, options,
                                           mo_space_info, ints)
        energy = casscf.compute_energy()

    time_pre_deriv = time.time()

    derivobj = psi4.core.Deriv(ref_wfn)
    derivobj.set_deriv_density_backtransformed(True)
    derivobj.set_ignore_reference(True)
    grad = derivobj.compute(psi4.core.DerivCalcType.Correlated)
    ref_wfn.set_gradient(grad)
    optstash.restore()

    end = time.time()

    # Close ambit, etc.
    forte.cleanup()

    # Print timings
    psi4.core.print_out('\n\n ==> Forte Timings <==\n')
    times = [('prepare integrals', start - time_pre_ints),
             ('run forte energy', time_pre_deriv - start),
             ('compute derivative integrals', end - time_pre_deriv)]
    max_key_size = max(len(k) for k, v in times)
    for key, value in times:
        psi4.core.print_out(f'\n  Time to {key:{max_key_size}} :'
                            f' {value:12.3f} seconds')
    psi4.core.print_out(f'\n  {"Total":{max_key_size + 8}} :'
                        f' {end - time_pre_ints:12.3f} seconds\n')

    # Dump orbitals if needed
    if options.get_bool('DUMP_ORBITALS'):
        dump_orbitals(ref_wfn)

    return ref_wfn


# Integration with driver routines
psi4.driver.procedures['energy']['forte'] = run_forte
psi4.driver.procedures['gradient']['forte'] = gradient_forte
