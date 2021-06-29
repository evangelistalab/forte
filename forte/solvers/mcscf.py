from forte.core import flog

from forte.solvers.feature import Feature
from forte.solvers.solver import Solver

# from forte.forte import ForteOptions
# from forte import to_state_nroots_map
# from forte import forte_options
# from forte.forte import make_active_space_ints, make_active_space_solver


class MCSCF(Solver):
    """
    A class to perform orbital optimization
    """
    def __init__(self, input_nodes, e_convergence=1.0e-10, r_convergence=1.0e-6, options=None, cbh=None):
        """
        Initialize an ActiveSpaceSolver object

        Parameters
        ----------
        input_nodes: Solver
            The solver that will provide the active space functionality and the starting molecular orbitals
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
            needs=[Feature.MODEL, Feature.ORBITALS, Feature.ACTIVESPACESOLVER],
            provides=[Feature.MODEL, Feature.ORBITALS, Feature.ACTIVESPACESOLVER, Feature.RDMS],
            options=options,
            cbh=cbh
        )
        self._data = self.input_nodes[0].data
        # parse the states parameter
        self._e_convergence = e_convergence
        self._r_convergence = r_convergence

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'MCSCF(e_convergence={self._e_convergence},r_convergence={self._r_convergence})'

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
        """Run an MCSCF computation"""
        # make sure the active space solver executed
        if not self.input_nodes[0].executed:
            flog('info', 'ActiveSpaceSolver: MOs not available in mo_solver. Calling mo_solver run()')
            self.input_nodes[0].run()
        else:
            flog('info', 'ActiveSpaceSolver: MOs read from mo_solver object')

        # options = prepare_forte_options()

        # ints = make_ints_from_psi4(self.guess.psi_wfn, options, mo_space_info)

        # # pipe output to the file self._output_file
        # psi4.core.set_output_file(self._output_file, True)

        # casscf = make_mcscf_two_step(self._states, self.guess.scf_info, options, mo_space_info, ints)
        # energy = casscf.compute_energy()
        # self._results.add('mcscf energy', [energy], 'MCSCF energy', 'Eh')

        return self
