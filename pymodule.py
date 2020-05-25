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

import psi4
import forte
from forte.proc.dsrg import ProcedureDSRG
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util


def forte_driver(state_weights_map, scf_info, options, ints, mo_space_info):
    # map state to number of roots
    state_map = forte.to_state_nroots_map(state_weights_map)

    # create an active space solver object and compute the energy
    active_space_solver_type = options.get_str('ACTIVE_SPACE_SOLVER')
    as_ints = forte.make_active_space_ints(mo_space_info, ints, "ACTIVE", ["RESTRICTED_DOCC"])
    active_space_solver = forte.make_active_space_solver(active_space_solver_type, state_map, scf_info, mo_space_info,
                                                         as_ints, options)
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
        forte.print_method_banner(["Atomic Valence Active Space (AVAS)", "Chenxi Cai"])
        forte.make_avas(ref_wfn, options, ps)

    # Create the fragment(embedding) projector and apply to rotate orbitals
    if options.get_bool("EMBEDDING"):
        forte.print_method_banner(["Frozen-orbital Embedding", "Nan He"])
        fragment_projector, fragment_nbf = forte.make_fragment_projector(ref_wfn, options)
        return forte.make_embedding(ref_wfn, options, fragment_projector, fragment_nbf, mo_space_info)
    else:
        return mo_space_info


def run_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('forte')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Compute a SCF reference
    # A wavefunction is returned, which holds the molecule used, orbitals, Fock matrices, and more
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Get the option object
    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')

    # Get the forte option object
    options = forte.forte_options
    options.get_options_from_psi4(psi4_options)

    if 'DF' in options.get_str('INT_TYPE'):
        aux_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), 'DF_BASIS_MP2', options.get_str('DF_BASIS_MP2'),
                                             'RIFIT', options.get_str('BASIS'))
        ref_wfn.set_basisset('DF_BASIS_MP2', aux_basis)

    if options.get_str('MINAO_BASIS'):
        minao_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), 'MINAO_BASIS', options.get_str('MINAO_BASIS'))
        ref_wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Start Forte, initialize ambit
    my_proc_n_nodes = forte.startup()
    my_proc, n_nodes = my_proc_n_nodes

    # Print the banner
    forte.banner()

    # Create the MOSpaceInfo object
    mo_space_info = forte.make_mo_space_info(ref_wfn, options)

    # Call methods that project the orbitals (AVAS, embedding)
    mo_space_info = orbital_projection(ref_wfn, options, mo_space_info)

    # Averaging spin multiplets if doing spin-adapted computation
    if options.get_str('CORRELATION_SOLVER') == 'SA-MRDSRG':
        options.set_bool('SPIN_AVG_DENSITY', True)

    state = forte.make_state_info_from_psi_wfn(ref_wfn)
    scf_info = forte.SCFInfo(ref_wfn)
    state_weights_map = forte.make_state_weights_map(options, ref_wfn)

    # Run a method
    job_type = options.get_str('JOB_TYPE')

    if job_type == 'NONE':
        forte.cleanup()
        return ref_wfn

    start_pre_ints = time.time()

    # Make an integral object
    ints = forte.make_forte_integrals(ref_wfn, options, mo_space_info)

    start = time.time()

    # Rotate orbitals before computation (e.g. localization, MP2 natural orbitals, etc.)
    orb_type = options.get_str("ORBITAL_TYPE")
    if orb_type != 'CANONICAL':
        orb_t = forte.make_orbital_transformation(orb_type, scf_info, options, ints, mo_space_info)
        orb_t.compute_transformation()
        Ua = orb_t.get_Ua()
        Ub = orb_t.get_Ub()
        ints.rotate_orbitals(Ua, Ub)

    # Run a method
    if job_type == 'NEWDRIVER':
        energy = forte_driver(state_weights_map, scf_info, options, ints, mo_space_info)
    else:
        energy = forte.forte_old_methods(ref_wfn, options, ints, mo_space_info)

    end = time.time()

    # Close ambit, etc.
    forte.cleanup()

    psi4.core.set_scalar_variable('CURRENT ENERGY', energy)

    psi4.core.print_out(f'\n\n  Time to prepare integrals: {start - start_pre_ints:12.3f} seconds')
    psi4.core.print_out(f'\n  Time to run job          : {end - start:12.3f} seconds')
    psi4.core.print_out(f'\n  Total                    : {end - start:12.3f} seconds')
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

    # Get the psi4 option object
    optstash = p4util.OptionsState(['GLOBALS', 'DERTYPE'])
    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')

    # Get the forte option object
    options = forte.forte_options
    options.get_options_from_psi4(psi4_options)

    if 'DF' in options.get_str('INT_TYPE'):
        raise Exception('analytic gradient is not implemented for density fitting')

    if options.get_str('MINAO_BASIS'):
        minao_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), 'MINAO_BASIS', options.get_str('MINAO_BASIS'))
        ref_wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Start Forte, initialize ambit
    my_proc_n_nodes = forte.startup()
    my_proc, n_nodes = my_proc_n_nodes

    # Print the banner
    forte.banner()

    # Create the MOSpaceInfo object
    mo_space_info = forte.make_mo_space_info(ref_wfn, options)

    # Call methods that project the orbitals (AVAS, embedding)
    mo_space_info = orbital_projection(ref_wfn, options, mo_space_info)

    state = forte.make_state_info_from_psi_wfn(ref_wfn)
    scf_info = forte.SCFInfo(ref_wfn)
    state_weights_map = forte.make_state_weights_map(options, ref_wfn)

    # Run a method
    job_type = options.get_str('JOB_TYPE')

    if not job_type == 'CASSCF':
        raise Exception('analytic gradient is only implemented for CASSCF')

    start = time.time()

    # Make an integral object
    ints = forte.make_forte_integrals(ref_wfn, options, mo_space_info)

    # Rotate orbitals before computation
    orb_type = options.get_str("ORBITAL_TYPE")
    if orb_type != 'CANONICAL':
        orb_t = forte.make_orbital_transformation(orb_type, scf_info, options, ints, mo_space_info)
        orb_t.compute_transformation()
        Ua = orb_t.get_Ua()
        Ub = orb_t.get_Ub()
        ints.rotate_orbitals(Ua, Ub)

    # Run gradient computation
    forte.forte_old_methods(ref_wfn, options, ints, mo_space_info)
    derivobj = psi4.core.Deriv(ref_wfn)
    derivobj.set_deriv_density_backtransformed(True)
    derivobj.set_ignore_reference(True)
    grad = derivobj.compute()
    # psi4.core.DerivCalcType.Correlated
    ref_wfn.set_gradient(grad)    
    optstash.restore()        

    end = time.time()
    psi4.core.print_out(f'\n\n  Your calculation took {end - start:12.3f} seconds.')

    # Close ambit, etc.
    forte.cleanup()

    return ref_wfn


# Integration with driver routines
psi4.driver.procedures['energy']['forte'] = run_forte
psi4.driver.procedures['gradient']['forte'] = gradient_forte
