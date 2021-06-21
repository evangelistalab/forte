# import psi4
import logging

from forte.solvers.solver import Solver

from forte.solvers.callback_handler import CallbackHandler
from forte.forte import StateInfo
from forte.forte import ForteOptions
from forte import to_state_nroots_map
from forte import forte_options
from forte.forte import make_active_space_ints, make_active_space_solver


class ActiveSpaceSolver(Solver):
    """
    A class to diagonalize the Hamiltonian in a subset of molecular orbitals.
    """
    def __init__(
        self,
        mo_solver,
        type,
        states,
        active=None,
        restricted_docc=None,
        frozen_docc=None,
        e_convergence=1.0e-10,
        r_convergence=1.0e-6,
        options=None,
        cbh=None
    ):
        """
        Initialize a FCI object

        Parameters
        ----------
        type: str
            The type of solver (set by the derived classes)
        mo_solver: Solver
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
        options: dict()
            Additional options passed to control the active space solver
        cbh: CallbackHandler
            A callback object used to inject code into the HF class
        """
        super().__init__()
        self._type = type.upper()
        self._mo_solver = mo_solver
        # allow passing a single StateInfo object
        if isinstance(states, StateInfo):
            self._states = {states: [1.0]}
        else:
            self._states = states
        self._data = mo_solver._data
        self._mo_space_info_map = self._mo_space_info_map(
            frozen_docc=frozen_docc, restricted_docc=restricted_docc, active=active
        )
        self._e_convergence = e_convergence
        self._r_convergence = r_convergence
        self._options = {} if options is None else options
        self._cbh = CallbackHandler() if cbh is None else cbh

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

    @property
    def e_convergence(self):
        return self._e_convergence

    @property
    def r_convergence(self):
        return self._r_convergence

    def run(self):
        """Run an active space solver computation"""

        logging.info('ActiveSpaceSolver: entering run()')

        # compute the guess orbitals
        if not self._mo_solver.executed:
            self._mo_solver.run()

        # make the state_map
        state_map = to_state_nroots_map(self._states)

        # make the mo_space_info object
        self.make_mo_space_info(self._mo_space_info_map)

        # prepare the options
        options = {'E_CONVERGENCE': self.e_convergence, 'R_CONVERGENCE': self.r_convergence}

        # values from self._options (user specified) replace those from options
        full_options = {**options, **self._options}

        local_options = ForteOptions(forte_options)
        local_options.set_from_dict(full_options)

        self.ints = self.model.ints(self.data, local_options)

        # Make an active space integral object
        as_ints = make_active_space_ints(self.mo_space_info, self.ints, "ACTIVE", ["RESTRICTED_DOCC"])

        # create an active space solver object and compute the energy
        active_space_solver = make_active_space_solver(
            self._type, state_map, self.scf_info, self.mo_space_info, as_ints, local_options
        )
        state_energies_list = active_space_solver.compute_energy()
        self._results.add('active space energy', state_energies_list, 'Active space energy', 'Eh')

        logging.info('ActiveSpaceSolver: exiting run()')

        return self
