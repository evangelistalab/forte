from forte.core import flog

from forte.solvers.solver import Feature, Solver

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
        input,
        states,
        type,
        active=None,
        restricted_docc=None,
        frozen_docc=None,
        gas1=None,
        gas2=None,
        gas3=None,
        gas4=None,
        gas5=None,
        gas6=None,
        e_convergence=1.0e-10,
        r_convergence=1.0e-6,
        options=None,
        cbh=None
    ):
        """
        Initialize a FCI object

        Parameters
        ----------
        input: Solver
            The solver that will provide the molecular orbitals
        states: dict()
            A dictionary of StateInfo -> list(float) of all the states that will be computed.
        type: str
            The type of solver (e.g. one of 'FCI', 'ACI', ...)
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
        # initialize the base class
        super().__init__(
            input=input,
            needs=[Feature.MODEL, Feature.ORBITALS],
            provides=[Feature.MODEL, Feature.ORBITALS, Feature.RDMS],
            options=options,
            cbh=cbh
        )
        self._data = self.input[0].data

        # parse the states parameter
        self._states = self._parse_states(states)

        self._type = type.upper()

        self._mo_space_info_map = self._make_mo_space_info_map(
            frozen_docc=frozen_docc,
            restricted_docc=restricted_docc,
            active=active,
            gas1=gas1,
            gas2=gas2,
            gas3=gas3,
            gas4=gas4,
            gas5=gas5,
            gas6=gas6
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

    @property
    def e_convergence(self):
        return self._e_convergence

    @property
    def r_convergence(self):
        return self._r_convergence

    def _run(self):
        """Run an active space solver computation"""

        # compute the guess orbitals
        if not self.input[0].executed:
            flog('info', 'ActiveSpaceSolver: MOs not available in mo_solver. Calling mo_solver run()')
            self.input[0].run()
        else:
            flog('info', 'ActiveSpaceSolver: MOs read from mo_solver object')

        # make the state_map
        state_map = to_state_nroots_map(self._states)

        # make the mo_space_info object
        self.make_mo_space_info(self._mo_space_info_map)

        # prepare the options
        options = {'E_CONVERGENCE': self.e_convergence, 'R_CONVERGENCE': self.r_convergence}

        # values from self._options (user specified) replace those from options
        full_options = {**options, **self._options}

        flog('info', 'ActiveSpaceSolver: adding options')
        local_options = ForteOptions(forte_options)
        local_options.set_from_dict(full_options)

        flog('info', 'ActiveSpaceSolver: getting integral from the model object')
        self.ints = self.model.ints(self.data, local_options)

        # Make an active space integral object
        flog('info', 'ActiveSpaceSolver: making active space integrals')
        self.as_ints = make_active_space_ints(self.mo_space_info, self.ints, "ACTIVE", ["RESTRICTED_DOCC"])

        # create an active space solver object and compute the energy
        flog('info', 'ActiveSpaceSolver: creating active space solver object')
        self._active_space_solver = make_active_space_solver(
            self._type, state_map, self.scf_info, self.mo_space_info, self.as_ints, local_options
        )

        flog('info', 'ActiveSpaceSolver: calling compute_energy() on active space solver object')
        state_energies_list = self._active_space_solver.compute_energy()
        flog('info', 'ActiveSpaceSolver: compute_energy() done')

        flog('info', f'ActiveSpaceSolver: active space energy = {state_energies_list}')
        self._results.add('active space energy', state_energies_list, 'Active space energy', 'Eh')

        return self
