# import psi4
import logging

from forte.solvers.solver import Solver
from forte.forte import StateInfo
from forte import to_state_nroots_map
from forte import forte_options
from forte.forte import make_active_space_ints, make_active_space_solver
# from forte.solvers.hf import HF
# from forte.model import Model
# from forte import prepare_forte_options, make_ints_from_psi4, make_mcscf_two_step, make_mcscf


class ActiveSpaceSolver(Solver):
    """
    A class to diagonalize the Hamiltonian in a subset of molecular orbitals.
    """
    def __init__(
        self,
        type,
        mos,
        states,
        active=None,
        restricted_docc=None,
        frozen_docc=None,
        e_convergence=1.0e-10,
        r_convergence=1.0e-6
    ):
        """
        Initialize a FCI object

        Parameters
        ----------
        type: str
            The type of solver (set by the derived classes)
        mos: Solver
            The solver that will provide the molecular orbitals
        states: dict()
            A dictionary of StateInfo -> list(float) of all the states that will be computed.
        active: list(int)
            The number of active MOs per irrep
        restricted_docc: list(int)
            The number of restricted doubly occupied MOs per irrep
        frozen_docc: list(int)
            The number of frozen doubly occupied MOs per irrep            
        e_convergence: float
            energy convergence criterion
        r_convergence: float
            residual convergence criterion
        """
        super().__init__()
        self._type = type.upper()
        self._mos = mos
        # allow passing a single StateInfo object
        if isinstance(states, StateInfo):
            self._states = {states: [1.0]}
        else:
            self._states = states
        self._data = mos._data
        self._mo_space_info_map = self._mo_space_info_map(
            frozen_docc=frozen_docc, restricted_docc=restricted_docc, active=active
        )
        self._e_convergence = e_convergence
        self._r_convergence = r_convergence

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'{self._type}(mo_space={self._mo_space_info_map},e_convergence={self._e_convergence},r_convergence={self._r_convergence})'

    def __str__(self):
        """
        return a string representation of this object
        """
        return repr(self)

    def run(self):
        # compute the guess orbitals
        if not self._mos.executed:
            self._mos.run()

        # make the state_map
        state_map = to_state_nroots_map(self._states)

        # make the mo_space_info object
        self.make_mo_space_info(self._mo_space_info_map)

        # make the integral objects
        # options = self.prepare_forte_options()  # TODO: this is a hack
        self.ints = self.model.ints(self.data, forte_options)

        # Make an active space integral object
        as_ints = make_active_space_ints(self.mo_space_info, self.ints, "ACTIVE", ["RESTRICTED_DOCC"])

        # create an active space solver object and compute the energy
        active_space_solver = make_active_space_solver(
            self._type, state_map, self.scf_info, self.mo_space_info, as_ints, forte_options
        )
        state_energies_list = active_space_solver.compute_energy()
        self._results.add('active space energy', state_energies_list, 'Active space energy', 'Eh')

        return self
