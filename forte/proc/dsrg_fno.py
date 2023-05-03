#
# @BEGIN LICENSE
#
# Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
# that implements a variety of quantum chemistry methods for strongly
# correlated electrons.
#
# Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
#
#  The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.
#
# @END LICENSE
#

import json
import os
import psi4
import forte
from forte.proc.dsrg import ProcedureDSRG


def frozen_natural_virtual_occupations(cutoff):
    """
    Figure out frozen virtuals from 'NAT_OCC_VIRT' file.
    :param cutoff: threshold smaller than which will be frozen virtuals    
    :return: the occupancy of frozen virtuals
    """
    virt_orbs = []
    with open(f'NAT_OCC_VIRT') as f:
        for _, line in enumerate(f):
            if _ == 0:
                continue
            h, i, n = line.split()
            virt_orbs.append((int(h), int(i), float(n)))

    nirreps = virt_orbs[-1][0] + 1

    frzv = [0] * nirreps
    for i in virt_orbs:
        frzv[i[0]] += 1 if i[2] < cutoff else 0

    return frzv


def dsrg_fno_procrouting(state_weights_map, scf_info, options, ints, mo_space_info,
                         active_space_solver):
    """ Driver for frozen-natural-orbital truncated DSRG. """
    # compute RDMs
    max_rdm_level = 3 if options.get_str("THREEPDC") != "ZERO" else 2
    rdms = active_space_solver.compute_average_rdms(
        state_weights_map, max_rdm_level, forte.RDMsType.spin_free)

    # patch some options
    ccvv_source = options.get_str("CCVV_SOURCE")
    pt2_correction = options.get_bool("DSRG_FNO_PT2_CORRECTION")
    dsrg_s = options.get_double("DSRG_S")
    options.set_double("DSRG_S", options.get_double("DSRG_FNO_PT2_S"))

    # run a DSRG-MRPT2 quasi-natural orbitals
    # only run MRPT2 NOs when the file of natural occupations is not available
    if not os.path.isfile('NAT_OCC_VIRT'):
        psi4.core.print_out(
            "\n\n  ==> DSRG-MRPT2 Frozen Natural Orbitals <==\n")
        options.set_str("CCVV_SOURCE", "ZERO")
        mrpt2_nos = forte.MRPT2_NOS(
            rdms, scf_info, options, ints, mo_space_info)
        Ua = mrpt2_nos.compute_fno()
        psi4.core.print_out(
            "\n\n  ==> Rotate to DSRG-MRPT2 Frozen Natural Orbitals <==\n")
        ints.rotate_orbitals(Ua, Ua, True)  # orbitals to MRPT2 NOs basis
        options.set_str("CCVV_SOURCE", ccvv_source)

    # run a complete DSRG-MRPT2 computation
    dept2 = 0.0
    dhpt2 = None
    if pt2_correction:
        psi4.core.print_out(
            "\n\n  ==> Frozen Natural Orbitals Correction (Untruncated PT2) <==\n")

        pt2_solver = forte.SA_MRPT2(
            rdms, scf_info, options, ints, mo_space_info)
        pt2_solver.set_state_weights_map(state_weights_map)
        pt2_solver.set_active_space_solver(active_space_solver)

        dept2 = pt2_solver.compute_energy()
        if options.get_str("CALC_TYPE") != "SS" or options.get_str("RELAX_REF") != "NONE":
            dhpt2 = pt2_solver.compute_Heff_actv()
        pt2_solver = None

    # truncate virtual orbitals
    fno_cutoff = options.get_double("DSRG_FNO_CUTOFF")
    options.set_int_list(
        "FROZEN_UOCC", frozen_natural_virtual_occupations(fno_cutoff))

    nmopi = mo_space_info.dimension("ALL")
    pg = mo_space_info.point_group_label()
    mo_space_info = forte.make_mo_space_info(nmopi, pg, options)
    ints = forte.make_ints_from_psi4(ints.wfn(), options, mo_space_info)

    # run a truncated DSRG-MRPT2 computation
    if pt2_correction:
        psi4.core.print_out(
            "\n\n  ==> Frozen Natural Orbitals Correction (Truncated PT2) <==\n")
        pt2_solver = forte.SA_MRPT2(
            rdms, scf_info, options, ints, mo_space_info)
        pt2_solver.set_state_weights_map(state_weights_map)
        pt2_solver.set_active_space_solver(active_space_solver)

        dept2 -= pt2_solver.compute_energy()
        if options.get_str("CALC_TYPE") != "SS" or options.get_str("RELAX_REF") != "NONE":
            dhpt2.add(pt2_solver.compute_Heff_actv(), -1.0)

    # reset flow parameter
    options.set_double("DSRG_S", dsrg_s)

    # return
    return mo_space_info, ints, dept2, dhpt2
