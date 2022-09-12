from abc import abstractmethod

from forte import StateInfo
from forte.core import flog, increase_log_depth
from forte.solvers.callback_handler import CallbackHandler
from forte.results import Results

import forte

from forte.solvers.node import Node


class Solver(Node):
    """
    Represents a node on a computational graph that performs
    an operation.

    A Solver is a node in a graph that takes zero or more input
    solvers. This class is used to implement quantum chemistry
    methods and other operations.
    This class implements a ``run()`` method that can be called
    to start the evaluation of the computational graph.

    To check if the validity of a graph, each node provides a
    list of features that are needed and provided.
    The features that are needed must be part of the input node(s).

    Solver stores Forte base objects in a data attribute
    and a results object.
    """
    def __init__(self, needs, provides, input_nodes=None, data=None, options=None, cbh=None):
        """
        Parameters
        ----------
        needs: list(str)
            a list of features required by this solver
        provides: list(str)
            a list of features provided by this solver
        input: list(Solver)
            a list of input nodes to this node
        options: dict(str -> obj)
            a dictionary of options to pass to the forte modules
        cbh: CallbackHandler
            a callback handler object
        """
        super().__init__(needs, provides, input_nodes, data)

        self._options = {} if options is None else options
        self._cbh = CallbackHandler() if cbh is None else cbh
        self._executed = False
        self._results = Results()
        # set the default psi4 output file
        self._output_file = 'output.dat'
        # self._output_file = f'output.{time.strftime("%Y-%m-%d-%H:%M:%S")}.dat'

    # decorate to increase the log depth
    @increase_log_depth
    def run(self):
        """
        A general solver interface.

        This method is common to all solvers, and in turn it is routed to
        the method ``_run()`` implemented differently in each solver.

        Return
        ------
            A data object
        """
        # log call to run()
        flog('info', f'{type(self).__name__}: calling run()')

        # call derived class implementation of _run()
        self._run()

        # log end of run()
        flog('info', f'{type(self).__name__}: run() finished executing')

        # set executed flag
        self._executed = True

        return self.data

    @abstractmethod
    def _run():
        """The actual run function implemented by each method"""
        pass

    @property
    def results(self):
        return self._results

    @property
    def executed(self):
        return self._executed

    @property
    def psi_wfn(self):
        return self.data.psi_wfn

    @psi_wfn.setter
    def psi_wfn(self, val):
        self.data.psi_wfn = val

    @property
    def scf_info(self):
        return self.data.scf_info

    @scf_info.setter
    def scf_info(self, val):
        self.data.scf_info = val

    @property
    def mo_space_info(self):
        return self.data.mo_space_info

    @mo_space_info.setter
    def mo_space_info(self, val):
        self.data.mo_space_info = val

    @property
    def ints(self):
        return self.data.ints

    @ints.setter
    def ints(self, val):
        self.data.ints = val

    @property
    def as_ints(self):
        return self.data.as_ints

    @as_ints.setter
    def as_ints(self, val):
        self.data.as_ints = val

    @property
    def model(self):
        return self.data.model

    @property
    def output_file(self):
        return self._output_file

    @output_file.setter
    def output_file(self, val):
        self._output_file = val

    def value(self, label):
        return self._results.value(label)

    def state(self, *args, **kwargs):
        return self.data.model.state(*args, **kwargs)

    def _parse_states(self, states):
        """
        This function converts the input of a user into the standard
        input for a multi-state computation.

        Parameters
        ----------
        states: StateInfo, or list(StateInfo), or dict[StateInfo,int], or dict[StateInfo,list(float)]
            The user input can be one of four options:
            1. A single state
            2. A list of single states (will compute one level for each type of state)
            3. A dictionary that maps StateInfo objects to the number of states to compute
            4. A dictionary that maps StateInfo objects to a list of weights for the states to compute
        """
        parsed_states = {}
        if isinstance(states, StateInfo):
            parsed_states[states] = [1.0]
        elif isinstance(states, list):
            for state in states:
                parsed_states[state] = [1.0]
        elif isinstance(states, dict):
            for k, v in states.items():
                if isinstance(v, int):
                    parsed_states[k] = [1.0] * v
                elif isinstance(v, list):
                    parsed_states[k] = v
                else:
                    raise ValueError(f'could not parse stats input {states}')
        else:
            raise ValueError(f'could not parse stats input {states}')
        return parsed_states

    def make_mo_space_info(self, mo_spaces, reorder=None):
        """
        Make a MOSpaceInfo object from a dictionary

        Parameters
        ----------
        mo_spaces: dict(str -> list(int))
            A dictionary of orbital space labels to a list of number of orbitals per irrep
        reorder: list(int)
            A list used to reorder the MOs. If not provided, use Pitzer order as obtained
            from psi4
        Return
        ------
            A MOSpaceInfo object
        """
        nmopi = self.data.scf_info.nmopi()
        point_group = self.model.point_group
        reorder = [] if reorder is None else reorder
        self.data.mo_space_info = forte.make_mo_space_info_from_map(nmopi, point_group, mo_spaces, reorder)

    def prepare_forte_options(self):
        """
        Return a ForteOptions object
        """
        import psi4
        # Get the option object
        psi4_options = psi4.core.get_options()
        psi4_options.set_current_module('FORTE')

        # Get the forte option object
        options = forte.forte_options
        options.get_options_from_psi4(psi4_options)

        return options
