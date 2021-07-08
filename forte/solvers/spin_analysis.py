from forte.core import flog

from forte import forte_options
from forte import ForteOptions
from forte.solvers.solver import Feature, Solver
from forte.solvers.callback_handler import CallbackHandler
from forte.forte import perform_spin_analysis


class SpinAnalysis(Solver):
    """
    A class to perform spin analysis of an active space solver wave function
    """
    def __init__(self, input_nodes, options=None, cbh=None):
        """
        initialize a SpinAnalysis object

        Parameters
        ----------
        input_nodes: Solver
            the object that provides information about this computation
        options: dict()
            Additional options passed to control psi4
        cbh: CallbackHandler
            A callback object used to inject code into the HF class
        """
        # initialize common objects
        super().__init__(input_nodes=input_nodes, needs=[Feature.RDMS], provides=[], options=options, cbh=cbh)
        self._data = self.input_nodes[0].data

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'SpinAnalysis(options={self._options})'

    def __str__(self):
        """
        return a string representation of this object
        """
        return repr(self)

    def _run(self):
        """Run the spin analysis"""
        if not self.input_nodes[0].executed:
            flog(
                'info',
                f'{__class__.__name__}: active space solver not available in parent_solver. Calling parent_solver run()'
            )
            self.input_nodes[0].run()
        else:
            flog('info', f'{__class__.__name__}: MOs read from mo_solver object')

        # prepare the options
        flog('info', 'ActiveSpaceSolver: adding options')
        local_options = ForteOptions(forte_options)
        local_options.set_from_dict(self._options)

        flog('info', f'{__class__.__name__}: preparing the 1- and 2-body reduced density matrices')
        rdms = self.input_nodes[0]._active_space_solver.compute_average_rdms(self.input_nodes[0]._states, 2)
        perform_spin_analysis(rdms, local_options, self.mo_space_info, self.as_ints)

        return self
