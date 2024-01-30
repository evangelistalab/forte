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


def dsrg_fno_procrouting(state_weights_map, scf_info, options, ints, mo_space_info,
                         active_space_solver, rdms, Ua):
    """ Driver for frozen-natural-orbital truncated DSRG. """
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

        # save orbital energies to file
        epsilon = pt2_solver.epsilon('v')
        with open('epsilon_full.txt', 'w') as w:
            w.write("# Semicanonical orbital energies w/o FNO: irrep, index, value")
            virt_mos = mo_space_info.relative_mo("RESTRICTED_UOCC")
            for (h, i), v in zip(virt_mos, epsilon):
                w.write(f"\n {h} {i:4}  {v:20.15f}")

    fnopi, Va = pt2_solver.build_fno()
    pt2_solver = None  # clean up

    # rebuild MOSpaceInfo
    options.set_int_list("FROZEN_UOCC", [fnopi[h] for h in range(mo_space_info.nirrep())])
    nmopi = mo_space_info.dimension("ALL")
    pg = mo_space_info.point_group_label()
    mo_space_info = forte.make_mo_space_info(nmopi, pg, options)

    # transform integrals to FNO semicanonical basis
    Ca = ints.wfn().Ca()
    Ca.copy(psi4.core.doublet(Ca, Va, False, False))
    ints.update_mo_space_info(mo_space_info)
    ints.update_orbitals(Ca, Ca, True)

    # run DSRG-MRPT2 in truncated basis
    if pt2_correction:
        pt2_solver = forte.SA_MRPT2(rdms, scf_info, options, ints, mo_space_info)
        pt2_solver.set_state_weights_map(state_weights_map)
        pt2_solver.set_active_space_solver(active_space_solver)
        pt2_solver.set_Uactv(Ua)

        dept2 -= pt2_solver.compute_energy()
        if options.get_str("CALC_TYPE") != "SS" or options.get_str("RELAX_REF") != "NONE":
            dhpt2.add(pt2_solver.compute_Heff_actv(), -1.0)

        # save orbital energies to file
        epsilon = pt2_solver.epsilon('v')
        with open('epsilon_fno.txt', 'w') as w:
            w.write("# Semicanonical orbital energies w/ FNO: irrep, index, value")
            virt_mos = mo_space_info.relative_mo("RESTRICTED_UOCC")
            for (h, i), v in zip(virt_mos, epsilon):
                w.write(f"\n {h} {i:4}  {v:20.15f}")

        pt2_solver = None  # clean up

    # reset flow parameter
    options.set_double("DSRG_S", dsrg_s)

    # return
    return mo_space_info, ints, dept2, dhpt2
