from forte.core import flog

from forte.solvers.solver import Feature, Solver

from forte.forte import ForteOptions
from forte import to_state_nroots_map
from forte import forte_options
from forte.forte import make_active_space_ints, make_active_space_solver


class ActiveSpaceSolver(Solver):
    """
    A class to diagonalize the Hamiltonian in a subset of molecular orbitals.

    Multi-state computations need a dict[StateInfo,list(float)] object,
    which can be cumbersome to prepare for all computations.
    Therefore, this class accepts various inputs, for example:

        >>> # single state (one eigenstate of state ``state``)
        >>> state = StateInfo(...)
        >>> _parse_states(state)

        >>> # list of states (one eigenstate of ``state_1`` and one of ``state_2``)
        >>> state_1 = StateInfo(...)
        >>> state_2 = StateInfo(...)
        >>> _parse_states([state_1,state_2])

        >>> # dict of states (5 eigenstate of ``state_1`` and 3 of ``state_2``)
        >>> state_info_1 = StateInfo(...)
        >>> state_info_2 = StateInfo(...)
        >>> _parse_states({state_info_1: 5,state_info_2: 3})

        >>> # dict of states with weights (5 eigenstate of ``state_1`` and 3 of ``state_2``)
        >>> state_info_1 = StateInfo(...)
        >>> state_info_2 = StateInfo(...)
        >>> _parse_states({state_info_1: [1.0,1.0,0.5,0.5,0.5],state_info_2: [0.25,0.25,0.25]})
    """
    def __init__(
        self,
        input_nodes,
        states,
        type,
        mo_spaces=None,
        e_convergence=1.0e-10,
        r_convergence=1.0e-6,
        options=None,
        cbh=None
    ):
        """
        Initialize an ActiveSpaceSolver object

        Parameters
        ----------
        input_nodes: Solver
            The solver that will provide the molecular orbitals
        states: StateInfo, or list(StateInfo), or dict[StateInfo,int], or dict[StateInfo,list(float)]
            The state(s) to be computed passed in one of the following ways:
                1. A single state
                2. A list of single states (will compute one level for each type of state)
                3. A dictionary that maps StateInfo objects to the number of states to compute
                4. A dictionary that maps StateInfo objects to a list of weights for the states to compute
            If explicit weights are passed, these are used in procedures that average properties
            over states (e.g., state-averaged CASSCF)
        type: {'FCI','ACI','CAS','DETCI','ASCI','PCI'}
            The type of solver
        mo_spaces: dict[str,list(int)]
            A dictionary that specifies the number of MOs per irrep that belong to a given orbital space.
            The available spaces are: frozen_docc, restricted_docc, active, restricted_uocc, frozen_uocc, and gas1-gas6.
            Please consult the manual for the meaning of these orbital spaces.
            A convenience function is provided by the Input class [mo_spaces()] to facilitate the creation of this dict.
            Note that an empty dict implies that all orbitals are treated as active.
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
            input_nodes=input_nodes,
            needs=[Feature.MODEL, Feature.ORBITALS],
            provides=[Feature.MODEL, Feature.ORBITALS, Feature.RDMS],
            options=options,
            cbh=cbh
        )
        self._data = self.input_nodes[0].data

        # parse the states parameter
        self._states = self._parse_states(states)

        self._type = type.upper()

        self._mo_space_info_map = {} if mo_spaces is None else mo_spaces
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
        if not self.input_nodes[0].executed:
            flog('info', 'ActiveSpaceSolver: MOs not available in mo_solver. Calling mo_solver run()')
            self.input_nodes[0].run()
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
