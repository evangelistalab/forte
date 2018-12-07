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

import psi4
import forte
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util

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
    my_proc_n_nodes = forte.forte_startup()
    my_proc, n_nodes = my_proc_n_nodes

    # Print the banner
    forte.forte_banner()

    # Create the MOSpaceInfo object
    mo_space_info = forte.make_mo_space_info(ref_wfn, options)

    # Create the AO subspace projector
    ps = forte.make_aosubspace_projector(ref_wfn, options)

    # Run a method
    job_type = options.get_str('JOB_TYPE')
#    job_type = psi4.core.get_option('FORTE','JOB_TYPE')
    if job_type != 'NONE':
        start = timeit.timeit()

        # Make an integral object
        ints = forte.make_forte_integrals(ref_wfn, options, mo_space_info)

        if options.get_bool("LOCALIZE"):
            forte.LOCALIZE(ref_wfn,options,ints,mo_space_info)

        # Run a method
        forte.forte_old_methods(ref_wfn, options, ints, mo_space_info, my_proc)

        end = timeit.timeit()
        #print('\n\n  Your calculation took ', (end - start), ' seconds');

    # Close ambit, etc.
    forte.forte_cleanup()
    return ref_wfn

# Integration with driver routines
psi4.driver.procedures['energy']['forte'] = run_forte

