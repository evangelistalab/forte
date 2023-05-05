#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2019 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import numpy as np
import warnings
import math
import json
import psi4

import forte
from forte.proc.external_active_space_solver import write_external_active_space_file


class ProcedureDSRG:
    def __init__(self, active_space_solver, state_weights_map, mo_space_info, ints, options, scf_info):
        """
        Procedure to perform a DSRG computation.
        :param active_space_solver: the Forte ActiveSpaceSolver object
        :param state_weights_map: the map between state and its weights
        :param mo_space_info: the Forte MOSpaceInfo object
        :param ints: the ForteIntegral object
        :param options: the ForteOptions object
        :param scf_info: the Forte SCFInfo object
        """

        # Read options
        self.solver_type = options.get_str('CORRELATION_SOLVER')
        if self.solver_type in ["SA-MRDSRG", "SA_MRDSRG", "DSRG_MRPT", "DSRG-MRPT"]:
            self.rdm_type = forte.RDMsType.spin_free
        else:
            self.rdm_type = forte.RDMsType.spin_free if options.get_bool('DSRG_RDM_MS_AVG') \
                else forte.RDMsType.spin_dependent

        self.do_semicanonical = options.get_bool("SEMI_CANONICAL")

        self.do_multi_state = False if options.get_str("CALC_TYPE") == "SS" else True

        self.relax_ref = options.get_str("RELAX_REF")
        if self.relax_ref == "NONE" and self.do_multi_state:
            self.relax_ref = "ONCE"

        self.max_rdm_level = 3 if options.get_str("THREEPDC") != "ZERO" else 2
        if options.get_str("DSRG_3RDM_ALGORITHM") == "DIRECT":
            as_type = options.get_str("ACTIVE_SPACE_SOLVER")
            if as_type == "CAS" and self.solver_type in ["SA-MRDSRG", "SA_MRDSRG"]:
                self.max_rdm_level = 2
            else:
                psi4.core.print_out(f"\n  DSRG 3RDM direct algorithm only available for CAS/SA-MRDSRG")
                psi4.core.print_out(f"\n  Set DSRG_3RDM_ALGORITHM to 'EXPLICIT' (default)")
                options.set_str("DSRG_3RDM_ALGORITHM", "EXPLICIT")

        self.relax_convergence = float('inf')
        self.e_convergence = options.get_double("E_CONVERGENCE")

        self.restart_amps = options.get_bool("DSRG_RESTART_AMPS")

        if self.relax_ref == "NONE":
            self.relax_maxiter = 0
        elif self.relax_ref == "ONCE":
            self.relax_maxiter = 1
        elif self.relax_ref == "TWICE":
            self.relax_maxiter = 2
        else:
            self.relax_maxiter = options.get_int('MAXITER_RELAX_REF')
            self.relax_convergence = options.get_double("RELAX_E_CONVERGENCE")

        self.save_relax_energies = options.get_bool("DSRG_DUMP_RELAXED_ENERGIES")

        # Filter out some ms-dsrg algorithms
        ms_dsrg_algorithm = options.get_str("DSRG_MULTI_STATE")
        if self.do_multi_state and ("SA" not in ms_dsrg_algorithm):
            raise NotImplementedError("MS or XMS is disabled due to the reconstruction.")
        if ms_dsrg_algorithm == "SA_SUB" and self.relax_ref != 'ONCE':
            raise NotImplementedError("SA_SUB only supports relax once at present. Relaxed SA density not implemented.")
        self.multi_state_type = ms_dsrg_algorithm

        # Filter out some methods for computing dipole moments
        do_dipole = options.get_bool("DSRG_DIPOLE")
        if do_dipole and (self.solver_type not in ["DSRG-MRPT2", "DSRG-MRPT3"]):
            do_dipole = False
            psi4.core.print_out(f"\n  Skip computation for dipole moment (not implemented for {self.solver_type})!")
            warnings.warn(f"Dipole moment is not implemented for {self.solver_type}.", UserWarning)
        if do_dipole and self.do_multi_state:
            do_dipole = False
            psi4.core.print_out("\n  !DSRG transition dipoles are disabled temporarily.")
            warnings.warn("DSRG transition dipoles are disabled temporarily.", UserWarning)
        self.do_dipole = do_dipole
        self.dipoles = []

        self.max_dipole_level = options.get_int("DSRG_MAX_DIPOLE_LEVEL")
        self.max_quadrupole_level = options.get_int("DSRG_MAX_QUADRUPOLE_LEVEL")

        # Set up Forte objects
        self.active_space_solver = active_space_solver
        self.state_weights_map = state_weights_map
        self.states = sorted(state_weights_map.keys())
        self.mo_space_info = mo_space_info
        self.ints = ints
        self.options = options
        self.scf_info = scf_info

        # DSRG solver related
        self.dsrg_solver = None
        self.Heff_implemented = False
        self.Meff_implemented = False
        self.converged = False
        self.energies = []  # energies along the relaxation steps
        self.energies_environment = {}  # energies pushed to Psi4 environment globals

        # Compute RDMs from initial ActiveSpaceSolver
        self.rdms = active_space_solver.compute_average_rdms(state_weights_map, self.max_rdm_level, self.rdm_type)

        # Save a copy CI vectors
        try:
            self.state_ci_wfn_map = active_space_solver.state_ci_wfn_map()
        except RuntimeError as err:
            print("Warning DSRG Python driver:", err)
            self.state_ci_wfn_map = None

        # Semi-canonicalize orbitals and rotation matrices
        self.semi = forte.SemiCanonical(mo_space_info, ints, options)
        if self.do_semicanonical:
            self.semi.semicanonicalize(self.rdms)
        self.Ua, self.Ub = self.semi.Ua_t(), self.semi.Ub_t()

    def make_dsrg_solver(self):
        """ Make a DSRG solver. """
        args = (self.rdms, self.scf_info, self.options, self.ints, self.mo_space_info)

        if self.solver_type in ["MRDSRG", "DSRG-MRPT2", "DSRG-MRPT3", "THREE-DSRG-MRPT2"]:
            self.dsrg_solver = forte.make_dsrg_method(*args)
            self.dsrg_solver.set_state_weights_map(self.state_weights_map)
            self.dsrg_solver.set_active_space_solver(self.active_space_solver)
            self.Heff_implemented = True
        elif self.solver_type in ["SA-MRDSRG", "SA_MRDSRG"]:
            self.dsrg_solver = forte.make_sadsrg_method(*args)
            self.dsrg_solver.set_state_weights_map(self.state_weights_map)
            self.dsrg_solver.set_active_space_solver(self.active_space_solver)
            self.Heff_implemented = True
            self.Meff_implemented = True
        elif self.solver_type in ["MRDSRG_SO", "MRDSRG-SO"]:
            self.dsrg_solver = forte.make_dsrg_so_y(*args)
        elif self.solver_type == "SOMRDSRG":
            self.dsrg_solver = forte.make_dsrg_so_f(*args)
        elif self.solver_type in ["DSRG_MRPT", "DSRG-MRPT"]:
            self.dsrg_solver = forte.make_dsrg_spin_adapted(*args)
        else:
            raise NotImplementedError(f"{self.solver_type} not available!")

    def dsrg_setup(self):
        """ Set up DSRG parameters before computations. """

        if self.solver_type in ["SA-MRDSRG", "SA_MRDSRG"]:
            self.dsrg_solver.set_Uactv(self.Ua)

        if self.solver_type in ["MRDSRG", "DSRG-MRPT2", "DSRG-MRPT3", "THREE-DSRG-MRPT2"]:
            self.dsrg_solver.set_Uactv(self.Ua, self.Ub)

    def dsrg_cleanup(self):
        """ Clean up for reference relaxation. """
        if self.Heff_implemented:
            self.dsrg_solver.clean_checkpoints()

    def compute_energy(self):
        """ Compute energy with reference relaxation and return current DSRG energy. """
        # Notes (York):
        # cases to run active space solver: reference relaxation, state-average dsrg
        # cases to run contracted ci solver (will be put in ActiveSpaceSolver): contracted state-average dsrg

        e_relax = 0.0  # initialize this to avoid PyCharm warning
        self.energies = []
        self.dipoles = []

        # Perform the initial un-relaxed DSRG
        self.make_dsrg_solver()
        self.dsrg_setup()
        e_dsrg = self.dsrg_solver.compute_energy()
        psi4.core.set_scalar_variable("UNRELAXED ENERGY", e_dsrg)

        self.energies_environment[0] = {k: v for k, v in psi4.core.variables().items() if 'ROOT' in k}

        # Spit out energy if reference relaxation not implemented
        if not self.Heff_implemented:
            self.relax_maxiter = 0

        # Reference relaxation procedure
        for n in range(self.relax_maxiter):
            # Grab the effective Hamiltonian in the active space
            # Note: The active integrals (ints_dressed) are in the original basis
            #       (before semi-canonicalization in the init function),
            #       so that the CI vectors are comparable before and after DSRG dressing.
            #       However, the ForteIntegrals object and the dipole integrals always refer to the current semi-canonical basis.
            #       so to compute the dipole moment correctly, we need to make the RDMs and orbital basis consistent
            ints_dressed = self.dsrg_solver.compute_Heff_actv()
            if self.Meff_implemented and (self.max_dipole_level > 0 or self.max_quadrupole_level > 0):
                asmpints = self.dsrg_solver.compute_mp_eff_actv()

            if self.options.get_str('ACTIVE_SPACE_SOLVER') == 'EXTERNAL':
                state_map = forte.to_state_nroots_map(self.state_weights_map)
                write_external_active_space_file(ints_dressed, state_map, self.mo_space_info, "dsrg_ints.json")
                msg = 'External solver: save DSRG dressed integrals to dsrg_ints.json'
                print(msg)
                psi4.core.print_out(msg)

                if self.options.get_bool("EXTERNAL_PARTIAL_RELAX"):
                    active_space_solver_2 = forte.make_active_space_solver(
                        self.options.get_str('EXT_RELAX_SOLVER'), state_map, self.scf_info, self.mo_space_info, ints_dressed, self.options)
                    active_space_solver_2.set_Uactv(self.Ua, self.Ub)
                    e_relax = list(active_space_solver_2.compute_energy().values())[0][0]
                self.energies.append((e_dsrg, e_relax))
                break
            
            if self.do_multi_state and self.options.get_bool("SAVE_SA_DSRG_INTS"):
                state_map = forte.to_state_nroots_map(self.state_weights_map)
                write_external_active_space_file(ints_dressed, state_map, self.mo_space_info, "dsrg_ints.json")
                msg = '\n\nSave SA-DSRG dressed integrals to dsrg_ints.json\n\n'
                print(msg)
                psi4.core.print_out(msg)

            # Spit out contracted SA-DSRG energy
            if self.do_multi_state and self.multi_state_type == "SA_SUB":
                max_rdm_level = 3 if self.options.get_bool("FORM_HBAR3") else 2
                state_energies_list = self.active_space_solver.compute_contracted_energy(ints_dressed, max_rdm_level)
                e_relax = forte.compute_average_state_energy(state_energies_list, self.state_weights_map)
                self.energies.append((e_dsrg, e_relax))
                break

            # Call the active space solver using the dressed integrals
            self.active_space_solver.set_active_space_integrals(ints_dressed)
            # pass to the active space solver the unitary transformation between the original basis
            # and the current semi-canonical basis
            self.active_space_solver.set_Uactv(self.Ua, self.Ub)
            state_energies_list = self.active_space_solver.compute_energy()

            if self.Meff_implemented:
                if self.max_dipole_level > 0:
                    self.active_space_solver.compute_dipole_moment(asmpints)
                if self.max_quadrupole_level > 0:
                    self.active_space_solver.compute_quadrupole_moment(asmpints);
                if self.max_dipole_level > 0:
                    self.active_space_solver.compute_fosc_same_orbs(asmpints)

            # Reorder weights if needed
            if self.state_ci_wfn_map is not None:
                state_ci_wfn_map = self.active_space_solver.state_ci_wfn_map()
                self.reorder_weights(state_ci_wfn_map)
                self.state_ci_wfn_map = state_ci_wfn_map

            e_relax = forte.compute_average_state_energy(state_energies_list, self.state_weights_map)
            self.energies.append((e_dsrg, e_relax))

            # Compute relaxed dipole
            if self.do_dipole:
                self.rdms = self.active_space_solver.compute_average_rdms(
                    self.state_weights_map, self.max_rdm_level, self.rdm_type
                )
                dm_u = ProcedureDSRG.grab_dipole_unrelaxed()
                dm_r = self.compute_dipole_relaxed()
                self.dipoles.append((dm_u, dm_r))

            # Save energies that have been pushed to Psi4 environment
            self.energies_environment[n + 1] = {k: v for k, v in psi4.core.variables().items() if 'ROOT' in k}
            self.energies_environment[n + 1]["DSRG FIXED"] = e_dsrg
            self.energies_environment[n + 1]["DSRG RELAXED"] = e_relax

            # Test convergence and break loop
            if self.test_relaxation_convergence(n):
                break

            # Continue to solve DSRG equations

            # - Compute RDMs from the active space solver (the RDMs are already available if we computed the relaxed dipole)
            #   These RDMs are computed in the original basis
            if self.do_multi_state or (not self.do_dipole):
                self.rdms = self.active_space_solver.compute_average_rdms(
                    self.state_weights_map, self.max_rdm_level, self.rdm_type
                )

            # - Transform RDMs to the semi-canonical basis used in the last step (stored in self.Ua/self.Ub)
            #   We do this because the integrals and amplitudes are all expressed in the previous semi-canonical basis
            self.rdms.rotate(self.Ua, self.Ub)

            # - Semi-canonicalize RDMs and orbitals
            if self.do_semicanonical:
                self.semi.semicanonicalize(self.rdms)
                # Do NOT read previous orbitals if fixing orbital ordering and phases failed
                if (not self.semi.fix_orbital_success()) and self.Heff_implemented:
                    psi4.core.print_out(
                        "\n  DSRG checkpoint files removed due to the unsuccessful"
                        " attempt to fix orbital phase and order."
                    )
                    self.dsrg_solver.clean_checkpoints()

                # update the orbital transformation matrix that connects the original orbitals
                # to the current semi-canonical ones. We do this only if we did a semi-canonicalization
                temp = self.Ua.clone()
                self.Ua["ik"] = temp["ij"] * self.semi.Ua_t()["jk"]
                temp.copy(self.Ub)
                self.Ub["ik"] = temp["ij"] * self.semi.Ub_t()["jk"]

            # - Compute the DSRG energy
            self.make_dsrg_solver()
            self.dsrg_setup()
            self.dsrg_solver.set_read_cwd_amps(not self.restart_amps)  # don't read from cwd if checkpoint available
            e_dsrg = self.dsrg_solver.compute_energy()

        self.dsrg_cleanup()

        # dump reference relaxation energies to json file
        if self.save_relax_energies:
            with open('dsrg_relaxed_energies.json', 'w') as w:
                json.dump(self.energies_environment, w, sort_keys=True, indent=4)

        e_current = e_dsrg if len(self.energies) == 0 else e_relax
        psi4.core.set_scalar_variable("CURRENT ENERGY", e_current)

        return e_current

    def compute_dipole_relaxed(self):
        """ Compute dipole moments. """
        dipole_moments = self.dsrg_solver.nuclear_dipole()
        dipole_dressed = self.dsrg_solver.deGNO_DMbar_actv()
        for i in range(3):
            dipole_moments[i] += dipole_dressed[i].contract_with_rdms(self.rdms)
        dm_total = math.sqrt(sum([i * i for i in dipole_moments]))
        dipole_moments.append(dm_total)
        return dipole_moments

    @staticmethod
    def grab_dipole_unrelaxed():
        """ Grab dipole moment from C++ results. """
        dipole = psi4.core.variable('UNRELAXED DIPOLE')
        return dipole[0], dipole[1], dipole[2], np.linalg.norm(dipole)

    def test_relaxation_convergence(self, n):
        """
        Test convergence for reference relaxation.
        :param n: iteration number (start from 0)
        :return: True if converged
        """
        if n == 0 and self.relax_ref == "ONCE":
            self.converged = True

        if n == 1 and self.relax_ref == "TWICE":
            self.converged = True

        if n != 0 and self.relax_ref == "ITERATE":
            e_diff_u = abs(self.energies[-1][0] - self.energies[-2][0])
            e_diff_r = abs(self.energies[-1][1] - self.energies[-2][1])
            e_diff = abs(self.energies[-1][0] - self.energies[-1][1])
            if all(e < self.relax_convergence for e in [e_diff_u, e_diff_r, e_diff]):
                self.converged = True

        return self.converged

    def reorder_weights(self, state_ci_wfn_map):
        """
        Check CI overlap and reorder weights between consecutive relaxation steps.
        :param state_ci_wfn_map: the map to be compared to self.state_ci_wfn_map
        """
        # bypass this check if state to CI vectors map not available
        if self.state_ci_wfn_map is None:
            return

        for state in self.states:
            twice_ms = state.twice_ms()
            if twice_ms < 0:
                continue

            # compute overlap between two sets of CI vectors <this|prior>
            overlap = psi4.core.doublet(state_ci_wfn_map[state], self.state_ci_wfn_map[state], True, False)
            overlap.name = f"CI Overlap of {state}"

            # check overlap and determine if we need to permute states
            overlap_np = np.abs(overlap.to_array())
            max_values = np.max(overlap_np, axis=1)
            permutation = np.argmax(overlap_np, axis=1)
            check_pass = len(permutation) == len(set(permutation)) and np.all(max_values > 0.5)

            if not check_pass:
                msg = "Relaxed states are likely wrong. Please increase the number of roots."
                warnings.warn(f"{msg}", UserWarning)
                psi4.core.print_out(f"\n\n  Forte Warning: {msg}")
                psi4.core.print_out(f"\n\n  ==> Overlap of CI Vectors <this|prior> <==\n\n")
                overlap.print_out()
            else:
                if list(permutation) == list(range(len(permutation))):
                    continue

                msg = "Weights will be permuted to ensure consistency before and after relaxation."
                psi4.core.print_out(f"\n\n  Forte Warning: {msg}\n")

                weights_old = self.state_weights_map[state]
                weights_new = [weights_old[i] for i in permutation]
                self.state_weights_map[state] = weights_new

                psi4.core.print_out(f"\n  ==> Weights for {state} <==\n")
                psi4.core.print_out(f"\n    Root    Old       New")
                psi4.core.print_out(f"\n    {'-' * 24}")
                for i, w_old in enumerate(weights_old):
                    w_new = weights_new[i]
                    psi4.core.print_out(f"\n    {i:4d} {w_old:9.3e} {w_new:9.3e}")
                psi4.core.print_out(f"\n    {'-' * 24}\n")

                # try to fix ms < 0
                if twice_ms > 0:
                    state_spin = forte.StateInfo(
                        state.nb(), state.na(), state.multiplicity(), -twice_ms, state.irrep(), state.irrep_label(),
                        state.gas_min(), state.gas_max()
                    )
                    if state_spin in self.state_weights_map:
                        self.state_weights_map[state_spin] = weights_new

    def print_summary(self):
        """ Print energies and dipole moment to output file. """
        if self.relax_maxiter < 1:
            return

        if (not self.do_multi_state) or self.relax_maxiter > 1:
            psi4.core.print_out(f"\n\n  => {self.solver_type} Reference Relaxation Energy Summary <=\n")
            indent = ' ' * 4
            dash = '-' * 71
            title = f"{indent}{' ':5}  {'Fixed Ref. (a.u.)':>31}  {'Relaxed Ref. (a.u.)':>31}\n"
            title += f"{indent}{' ' * 5}  {'-' * 31}  {'-' * 31}\n"
            title += f"{indent}{'Iter.':5}  {'Total Energy':>20} {'Delta':>10}  {'Total Energy':>20} {'Delta':>10}\n"
            psi4.core.print_out("\n{}".format(title + indent + dash))

            e0_old, e1_old = 0.0, 0.0
            for n, pair in enumerate(self.energies, 1):
                e0, e1 = pair
                e0_diff, e1_diff = e0 - e0_old, e1 - e1_old
                psi4.core.print_out(f"\n{indent}{n:>5}  {e0:>20.12f} {e0_diff:>10.3e}  {e1:>20.12f} {e1_diff:>10.3e}")
                e0_old, e1_old = e0, e1

            psi4.core.print_out(f"\n{indent}{dash}")

        def print_dipole(name, dipole_xyzt):
            out = f"\n    {self.solver_type} {name.lower()} dipole moment:"
            out += "\n      X: {:10.6f}  Y: {:10.6f}  Z: {:10.6f}  Total: {:10.6f}\n".format(*dipole_xyzt)
            return out

        if self.do_dipole and (not self.do_multi_state):
            psi4.core.print_out(f"\n\n  => {self.solver_type} Reference Relaxation Dipole Summary <=\n")

            psi4.core.print_out(print_dipole("unrelaxed", self.dipoles[0][0]))
            psi4.core.print_out(print_dipole("partially relaxed", self.dipoles[0][1]))

            if self.relax_maxiter > 1:
                psi4.core.print_out(print_dipole("relaxed", self.dipoles[1][0]))

            if self.relax_ref == "ITERATE" and self.converged:
                psi4.core.print_out(print_dipole("fully relaxed", self.dipoles[-1][1]))

    def push_to_psi4_environment(self):
        """ Push results to Psi4 environment. """
        if self.relax_maxiter < 1:
            return

        psi4.core.set_scalar_variable('UNRELAXED ENERGY', self.energies[0][0])
        psi4.core.set_scalar_variable('PARTIALLY RELAXED ENERGY', self.energies[0][1])

        if self.do_dipole and (not self.do_multi_state):
            psi4.core.set_scalar_variable('UNRELAXED DIPOLE', self.dipoles[0][0][-1])
            psi4.core.set_scalar_variable('PARTIALLY RELAXED DIPOLE', self.dipoles[0][1][-1])

        if self.relax_maxiter > 1:
            psi4.core.set_scalar_variable('RELAXED ENERGY', self.energies[1][0])
            if self.do_dipole and (not self.do_multi_state):
                psi4.core.set_scalar_variable('RELAXED DIPOLE', self.dipoles[1][0][-1])

            if self.relax_ref == "ITERATE":
                if not self.converged:
                    psi4.core.set_scalar_variable('CURRENT UNRELAXED ENERGY', self.energies[-1][0])
                    psi4.core.set_scalar_variable('CURRENT RELAXED ENERGY', self.energies[-1][1])
                    raise psi4.p4util.PsiException(f"DSRG relaxation does not converge in {self.relax_maxiter} cycles")
                else:
                    psi4.core.set_scalar_variable('FULLY RELAXED ENERGY', self.energies[-1][1])

                    if self.do_dipole and (not self.do_multi_state):
                        psi4.core.set_scalar_variable('FULLY RELAXED DIPOLE', self.dipoles[-1][1][-1])
