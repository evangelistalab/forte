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

import psi4
import forte
import ambit
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util

def forte_driver(state_weights_list, scf_info, options, ints, mo_space_info):
    max_rdm_level = 3 # TODO: set this (Francesco)
    return_en = 0.0

    # Create an active space solver object and compute the energy
    active_space_solver_type = options.get_str('ACTIVE_SPACE_SOLVER')
    as_ints = forte.make_active_space_ints(mo_space_info, ints, "ACTIVE", ["RESTRICTED_DOCC"]);
    active_space_solver = forte.make_active_space_solver(active_space_solver_type,state_weights_list,scf_info,mo_space_info,as_ints,options)
    active_space_solver.set_max_rdm_level(max_rdm_level)
    state_energies_list = active_space_solver.compute_energy()

    correlation_solver_type = options.get_str('CORRELATION_SOLVER')
    # Create a dynamical correlation solver object
    if correlation_solver_type != 'NONE':

        reference = active_space_solver.reference()
        semi = forte.SemiCanonical(mo_space_info, ints, options)
        semi.semicanonicalize(reference, max_rdm_level)
        Ua = semi.Ua_t()
        Ub = semi.Ub_t()


        dsrg = forte.make_dsrg_method(correlation_solver_type, reference, scf_info, options, ints, mo_space_info)
        dsrg.set_Uactv(Ua, Ub)
    
        Edsrg = dsrg.compute_energy()

        # store DSRG energy in vector of pair
        # Evec = [ [Edsrg, Erelaxed], ...]
        Evec = []

        do_dipole = options.get_bool("DSRG_DIPOLE")

        # determine the relaxation procedure
        relax_mode = options.get_str('RELAX_REF')

        # set the maxiter  
        maxiter = options.get_int('MAXITER_RELAX_REF')
        if relax_mode == 'NONE':
            if (len(state_weights_list) > 1) or (len(state_weights_list[0][1]) > 1) :
                maxiter = 1
            else:
                maxiter = 0
                Evec.append([Edsrg,Edsrg]) #TODO: fix
        elif relax_mode == 'ONCE':
            maxiter = 1
        elif relax_mode == 'TWICE':
            maxiter = 2

        for N in range(maxiter):

            # Grab the effective Hamiltonian in the actice space
            ints_dressed = dsrg.compute_Heff_actv()

            # Make a new ActiveSpaceSolver with the new ints
            as_solver_relaxed = forte.make_active_space_solver(active_space_solver_type,state_weights_list,scf_info,mo_space_info,ints_dressed,options)
            as_solver_relaxed.set_max_rdm_level(max_rdm_level)

            # Compute the energy
            state_energies_list = as_solver_relaxed.compute_energy()
            Erelax = forte.compute_average_state_energy(state_energies_list,state_weights_list)
            Evec.append([Edsrg,Erelax])

            try:
                if (abs(Evec[N][0] - Evec[N-1][0]) <= econv) and (abs(Evec[N][1] - Evec[N-1][1]) <= econv):
                    psi4.core.set_scalar_variable('FULLY RELAXED ENERGY', Evec[N][1])
                    break
            except:
                pass
            
            # NOTE: This loop ends here on last iteration

            # if we are continuing iterations, 
            # update the reference and recompute the dynamical correlation energy
            if (N+1) != maxiter:
                # Compute the reference in the original semicanonical basis
                reference = semi.transform_reference(Ua, Ub, as_solver_relaxed.reference(), max_rdm_level)
                
                # Now semicanonicalize
                semi.semicanonicalize(reference, max_rdm_level)
                Ua = semi.Ua_t()
                Ub = semi.Ub_t()
                
                # Compute DSRG in this basis
                dsrg = forte.make_dsrg_method(correlation_solver_type, reference, scf_info, options, ints, mo_space_info)
                dsrg.set_Uactv(Ua, Ub)
                Edsrg = dsrg.compute_energy()

            elif do_dipole:
                dipole_moments = dsrg.nuclear_dipole()
                trans_dipole = dsrg.deGNO_DMbar_actv()
                reference = as_solver_relaxed.reference()
                dm_total = 0.0
                for i in range(3):
                    dipole_moments[i] += trans_dipole[i].contract_with_densities(reference)
                    dm_total += dipole_moments[i] * dipole_moments[i]
                dm_total = math.sqrt(dm_total)

                psi4.core.print_out("\n    DSRG-MRPT3 partially relaxed dipole moment:")
                psi4.core.print_out("\n      X: %10.6f  Y: %10.6f  Z: %10.6f  Total: %10.6f\n" % 
                                (dipole_moments[0], dipole_moments[1], dipole_moments[2],
                                dm_total ))

                psi4.core.set_scalar_variable('PARTIALLY RELAXED DIPOLE', dm_total)
        
        psi4.core.set_scalar_variable('UNRELAXED ENERGY', Evec[0][0])
        psi4.core.set_scalar_variable('PARTIALLY RELAXED ENERGY', Evec[0][1])
        # is this 'relaxed energy'?
        psi4.core.set_scalar_variable('RELAXED ENERGY', Evec[-1][0])

        ## TODO: maybe change this
        ## return relaxed energy
        return_en = Evec[-1][1]

    else : 

        average_energy = forte.compute_average_state_energy(state_energies_list,state_weights_list)
        return_en = average_energy

    return return_en


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

    # Create the AO subspace projector
    ps = forte.make_aosubspace_projector(ref_wfn, options)

    state = forte.make_state_info_from_psi_wfn(ref_wfn)
    scf_info = forte.SCFInfo(ref_wfn)
    state_weights_list = forte.make_state_weights_list(forte.forte_options,ref_wfn)

    # Run a method
    job_type = options.get_str('JOB_TYPE')

    energy = 0.0
    if job_type != 'NONE':
        start = timeit.timeit()

        # Make an integral object
        ints = forte.make_forte_integrals(ref_wfn, options, mo_space_info)

        # Rotate orbitals before computation
        # TODO: Have these (and ci no) inherit from a base class
        #       and just call that
        loc_type = options.get_str("LOCALIZE")
        if loc_type == "FULL":
            loc = forte.LOCALIZE(ref_wfn,options,ints,mo_space_info)
            loc.full_localize()
        if loc_type == "SPLIT":
            loc = forte.LOCALIZE(ref_wfn,options,ints,mo_space_info)
            loc.split_localize()
        if options.get_bool("MP2_NOS"):
            forte.MP2_NOS(ref_wfn,options,ints,mo_space_info)

        # Run a method
        if (job_type == 'NEWDRIVER'):
            energy = forte_driver(state_weights_list, scf_info, forte.forte_options, ints, mo_space_info)
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

