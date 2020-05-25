import psi4
import forte
import warnings
import math


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

        self.do_semicanonical = options.get_bool("SEMI_CANONICAL")

        self.do_multi_state = False if options.get_str("CALC_TYPE") == "SS" else True

        self.relax_ref = options.get_str("RELAX_REF")
        if self.relax_ref == "NONE" and self.do_multi_state:
            self.relax_ref = "ONCE"

        self.max_rdm_level = 3 if options.get_str("THREEPDC") != "ZERO" else 2

        self.relax_convergence = float('inf')
        self.e_convergence = options.get_double("E_CONVERGENCE")

        if self.relax_ref == "NONE":
            self.relax_maxiter = 0
        elif self.relax_ref == "ONCE":
            self.relax_maxiter = 1
        elif self.relax_ref == "TWICE":
            self.relax_maxiter = 2
        else:
            self.relax_maxiter = options.get_int('MAXITER_RELAX_REF')
            self.relax_convergence = options.get_double("RELAX_E_CONVERGENCE")

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

        # Set up Forte objects
        self.active_space_solver = active_space_solver
        self.state_weights_map = state_weights_map
        self.mo_space_info = mo_space_info
        self.ints = ints
        self.options = options
        self.scf_info = scf_info

        # DSRG solver related
        self.dsrg_solver = None
        self.Heff_implemented = False
        self.converged = False
        self.energies = []  # energies along the relaxation steps
        self.t1_file, self.t2_file = "", ""  # amplitudes file names for restart

        # Compute RDMs from initial ActiveSpaceSolver
        self.rdms = active_space_solver.compute_average_rdms(state_weights_map, self.max_rdm_level)

        # Semi-canonicalize orbitals and rotation matrices
        self.semi = forte.SemiCanonical(mo_space_info, ints, options)
        if self.do_semicanonical:
            self.semi.semicanonicalize(self.rdms, self.max_rdm_level)
        self.Ua, self.Ub = self.semi.Ua_t(), self.semi.Ub_t()

    def make_dsrg_solver(self):
        """ Make a DSRG solver. """
        args = (self.rdms, self.scf_info, self.options, self.ints, self.mo_space_info)

        if self.solver_type in ["MRDSRG", "DSRG-MRPT2", "DSRG-MRPT3", "THREE-DSRG-MRPT2"]:
            self.dsrg_solver = forte.make_dsrg_method(*args)
            self.Heff_implemented = True
        elif self.solver_type in ["SA-MRDSRG", "SA_MRDSRG"]:
            self.dsrg_solver = forte.make_sadsrg_method(*args)
            self.Heff_implemented = True
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
            self.dsrg_solver.set_t1_file(self.t1_file)
            self.dsrg_solver.set_t2_file(self.t2_file)

        if self.solver_type in ["MRDSRG", "DSRG-MRPT2", "DSRG-MRPT3", "THREE-DSRG-MRPT2"]:
            self.dsrg_solver.set_Uactv(self.Ua, self.Ub)
            self.dsrg_solver.set_t1_file(self.t1_file)
            self.dsrg_solver.set_t2_file(self.t2_file)

    def dsrg_post_setup(self):
        """ Grab DSRG parameters after a computation. """
        if self.Heff_implemented:
            self.t1_file = self.dsrg_solver.t1_file()
            self.t2_file = self.dsrg_solver.t2_file()

    def dsrg_cleanup(self):
        """ Clean up for reference relaxation. """
        if self.Heff_implemented:
            self.dsrg_solver.clean_checkpoints()

    def compute_energy(self):
        """ Compute energy with reference relaxation and return current DSRG energy. """
        self.dipoles = []
        self.energies = []

        # Perform the initial un-relaxed DSRG
        self.make_dsrg_solver()
        self.dsrg_setup()
        Edsrg = self.dsrg_solver.compute_energy()
        self.dsrg_post_setup()
        psi4.core.set_scalar_variable("UNRELAXED ENERGY", Edsrg)

        # Spit out energy if reference relaxation not implemented
        if not self.Heff_implemented:
            self.relax_maxiter = 0

        # Reference relaxation procedure
        for n in range(self.relax_maxiter):
            # Grab effective Hamiltonian in the active space
            # These active integrals are in the original basis (before semi-canonicalize in the init function),
            # so that the CI coefficients are comparable before and after DSRG dressing.
            ints_dressed = self.dsrg_solver.compute_Heff_actv()

            # Spit out contracted SA-DSRG energy
            if self.do_multi_state and self.multi_state_type == "SA_SUB":
                max_rdm_level = 3 if self.options.get_bool("FORM_HBAR3") else 2
                state_energies_list = self.active_space_solver.compute_contracted_energy(ints_dressed, max_rdm_level)
                Erelax = forte.compute_average_state_energy(state_energies_list, self.state_weights_map)
                self.energies.append((Edsrg, Erelax))
                break

            # Solver active space using dressed integrals
            self.active_space_solver.set_active_space_integrals(ints_dressed)
            state_energies_list = self.active_space_solver.compute_energy()
            Erelax = forte.compute_average_state_energy(state_energies_list, self.state_weights_map)
            self.energies.append((Edsrg, Erelax))

            # Compute relaxed dipole
            if self.do_dipole:
                self.rdms = self.active_space_solver.compute_average_rdms(self.state_weights_map, self.max_rdm_level)
                dm_u = ProcedureDSRG.grab_dipole_unrelaxed()
                dm_r = self.compute_dipole_relaxed()
                self.dipoles.append((dm_u, dm_r))

            # Test convergence and break loop
            if self.test_relaxation_convergence(n):
                break

            # Continue to solve DSRG equations

            # - Compute RDMs (RDMs available if done relaxed dipole)
            if self.do_multi_state or (not self.do_dipole):
                self.rdms = self.active_space_solver.compute_average_rdms(self.state_weights_map, self.max_rdm_level)

            # - Transform RDMs to the semi-canonical orbitals of last step
            self.rdms = self.semi.transform_rdms(self.Ua, self.Ub, self.rdms, self.max_rdm_level)

            # - Semi-canonicalize RDMs and orbitals
            if self.do_semicanonical:
                self.semi.semicanonicalize(self.rdms, self.max_rdm_level)
            self.Ua, self.Ub = self.semi.Ua_t(), self.semi.Ub_t()

            # - Compute DSRG energy
            self.make_dsrg_solver()
            self.dsrg_setup()
            Edsrg = self.dsrg_solver.compute_energy()
            self.dsrg_post_setup()

        self.dsrg_cleanup()

        psi4.core.set_scalar_variable("CURRENT ENERGY", Edsrg)

        return Erelax if len(self.energies) else Edsrg

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
        x = psi4.core.variable('UNRELAXED DIPOLE X')
        y = psi4.core.variable('UNRELAXED DIPOLE Y')
        z = psi4.core.variable('UNRELAXED DIPOLE Z')
        t = psi4.core.variable('UNRELAXED DIPOLE')
        return x, y, z, t

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
            Ediff_u = abs(self.energies[-1][0] - self.energies[-2][0])
            Ediff_r = abs(self.energies[-1][1] - self.energies[-2][1])
            Ediff = abs(self.energies[-1][0] - self.energies[-1][1])
            if all(e < self.relax_convergence for e in [Ediff_u, Ediff_r]) and Ediff < self.e_convergence:
                self.converged = True

        return self.converged

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

            E0_old, E1_old = 0.0, 0.0
            for n, pair in enumerate(self.energies, 1):
                E0, E1 = pair
                E0_diff, E1_diff = E0 - E0_old, E1 - E1_old
                psi4.core.print_out(f"\n{indent}{n:>5}  {E0:>20.12f} {E0_diff:>10.3e}  {E1:>20.12f} {E1_diff:>10.3e}")
                E0_old, E1_old = E0, E1

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
