#
# @BEGIN LICENSE
#
# Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
# that implements a variety of quantum chemistry methods for strongly
# correlated electrons.
#
# Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
#
# The copyrights for code used from other parties are included in
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

from forte.modules.helpers import make_mo_spaces_from_options


def dsrg_fno_procrouting(state_weights_map, scf_info, options, ints, mo_space_info, active_space_solver, rdms, Ua):
    """Driver for frozen-natural-orbital truncated DSRG."""
    # read options
    pt2_correction = options.get_bool("DSRG_FNO_PT2_CORRECTION")
    dsrg_s = options.get_double("DSRG_S")
    options.set_double("DSRG_S", options.get_double("DSRG_FNO_PT2_S"))

    # PT2 correction variables
    dept2 = 0.0
    dhpt2 = None

    # determine FNO and run DSRG-MRPT2 in full basis
    pt2_solver = forte.SA_MRPT2(rdms, scf_info, options, ints, mo_space_info)
    pt2_solver.set_state_weights_map(state_weights_map)
    pt2_solver.set_active_space_solver(active_space_solver)
    pt2_solver.set_Uactv(Ua)

    if pt2_correction:
        dept2 = pt2_solver.compute_energy()
        if options.get_str("CALC_TYPE") != "SS" or options.get_str("RELAX_REF") != "NONE":
            dhpt2 = pt2_solver.compute_Heff_actv()

    fnopi, Va = pt2_solver.build_fno()
    pt2_solver = None  # clean up

    # rebuild MOSpaceInfo
    options.set_int_list("FROZEN_UOCC", [fnopi[h] for h in range(mo_space_info.nirrep())])
    nmopi = mo_space_info.dimension("ALL")
    pg = mo_space_info.point_group_label()
    mo_spaces = make_mo_spaces_from_options(options)
    mo_space_info = forte.make_mo_space_info_from_map(nmopi, pg, mo_spaces)

    # transform integrals to FNO semicanonical basis
    Ca = ints.wfn().Ca()
    Ca.copy(psi4.core.doublet(Ca, Va, False, False))
    # update the MO space info and SCF info with the new Ca
    # the order here is important, as the SCF info will update
    ints.update_mo_space_info(mo_space_info)
    scf_info.update_orbitals(Ca, Ca, True)

    # run DSRG-MRPT2 in truncated basis
    if pt2_correction:
        pt2_solver = forte.SA_MRPT2(rdms, scf_info, options, ints, mo_space_info)
        pt2_solver.set_state_weights_map(state_weights_map)
        pt2_solver.set_active_space_solver(active_space_solver)
        pt2_solver.set_Uactv(Ua)

        dept2 -= pt2_solver.compute_energy()
        if options.get_str("CALC_TYPE") != "SS" or options.get_str("RELAX_REF") != "NONE":
            dhpt2.add(pt2_solver.compute_Heff_actv(), -1.0)

        pt2_solver = None  # clean up

    # reset flow parameter
    options.set_double("DSRG_S", dsrg_s)

    # return
    return mo_space_info, ints, dept2, dhpt2
