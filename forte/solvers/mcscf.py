from forte.core import flog

from forte.solvers.feature import Feature
from forte.solvers.solver import Solver

from forte.forte import ForteOptions
from forte import forte_options
from forte.forte import make_mcscf_two_step


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
        # grab a pointer to the data from the input node
        self._data = self.input_nodes[0].data
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
            flog('info', 'MCSCF: reference not available. Calling run() on input node')
            self.input_nodes[0].run()
        else:
            flog('info', 'MCSCF: reference read from input node')

        # prepare the options
        options = {'E_CONVERGENCE': self.e_convergence, 'R_CONVERGENCE': self.r_convergence}

        # values from self._options (user specified) replace those from options
        full_options = {**options, **self._options}

        flog('info', 'MCSCF: adding options')
        local_options = ForteOptions(forte_options)
        local_options.set_from_dict(full_options)

        flog('info', 'MCSCF: making the mcscf object')
        mcscf = make_mcscf_two_step(
            self.input_nodes[0]._states, local_options, self.ints, self.input_nodes[0].active_space_solver
        )
        flog('info', 'MCSCF: computing the energy')
        average_energy, energies = mcscf.compute_energy()
        flog('info', f'MCSCF: mcscf average energy = {average_energy}')
        flog('info', f'MCSCF: mcscf energy = {energies}')

        self._results.add('mcscf energy', energies, 'MCSCF energy', 'Eh')

        return self
