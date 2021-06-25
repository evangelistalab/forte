# import time

from enum import Enum, auto
from abc import ABC, abstractmethod

from forte.core import flog, increase_log_depth
from forte.solvers.callback_handler import CallbackHandler
from forte.data import Data
from forte.results import Results

import forte


class Feature(Enum):
    """
    This enum class is used to store all possible Features needed
    or provided by a solver.
    """
    MODEL = auto()
    ORBITALS = auto()
    RDMS = auto()


class Solver(ABC):
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
    def __init__(self, needs, provides, input=None, options=None, cbh=None):
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
        self._needs = needs
        self._provides = provides
        # input can be None, a single element, or a list
        input = [] if input is None else input
        self._input = [input] if type(input) is not list else input
        self._check_input()

        self._options = {} if options is None else options
        self._cbh = CallbackHandler() if cbh is None else cbh
        self._executed = False
        self._data = Data()
        self._results = Results()
        # the default psi4 output file
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

    @property
    def input(self):
        return self._input

    @property
    def needs(self):
        return self._needs

    @property
    def provides(self):
        return self._provides

    @property
    def results(self):
        return self._results

    @property
    def data(self):
        return self._data

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

    def _make_mo_space_info_map(
        self,
        frozen_docc=None,
        restricted_docc=None,
        active=None,
        restricted_uocc=None,
        frozen_uocc=None,
        gas1=None,
        gas2=None,
        gas3=None,
        gas4=None,
        gas5=None,
        gas6=None
    ):
        """
        Returns a dictionary with the size of each orbital space defined.

        This function acts mainly as an interface. This is a good place
        for future implementation of error checking.
        """
        mo_space = {}
        if frozen_docc is not None:
            mo_space['FROZEN_DOCC'] = frozen_docc
        if restricted_docc is not None:
            mo_space['RESTRICTED_DOCC'] = restricted_docc
        if active is not None:
            mo_space['ACTIVE'] = active
        if gas1 is not None:
            mo_space['GAS1'] = gas1
        if gas2 is not None:
            mo_space['GAS2'] = gas2
        if gas3 is not None:
            mo_space['GAS3'] = gas3
        if gas4 is not None:
            mo_space['GAS4'] = gas4
        if gas5 is not None:
            mo_space['GAS5'] = gas5
        if gas6 is not None:
            mo_space['GAS6'] = gas6
        if restricted_uocc is not None:
            mo_space['RESTRICTED_UOCC'] = restricted_uocc
        if frozen_uocc is not None:
            mo_space['FROZEN_UOCC'] = frozen_uocc

        return mo_space

    def make_mo_space_info(self, mo_spaces):
        """
        Make a MOSpaceInfo object from a dictionary

        Parameters
        ----------
        mo_spaces: dict(str -> list(int))
            A dictionary of orbital space labels to a list of number of orbitals per irrep
        Return
        ------
            A MOSpaceInfo object
        """
        nmopi = self.data.scf_info.nmopi()
        point_group = self.model.point_group
        reorder = []  # TODO: enable reorder
        self.data.mo_space_info = forte.make_mo_space_info_from_map(nmopi, point_group, mo_spaces, reorder)

    def _check_input(self):
        # verify that this solver can get all that it needs from its inputs
        for need in self.needs:
            need_met = False
            for input in self.input:
                if need in input.provides:
                    need_met = True
            if not need_met:
                raise AssertionError(
                    f'\n\n  ** The computational graph is inconsistent ** \n\n{self.computational_graph()}'
                    f'\n\n  The solver {self.__class__.__name__} cannot get a feature ({need}) from its input solver: '
                    + ','.join([input.__class__.__name__ for input in self.input]) + '\n'
                )

    def computational_graph(self):
        graph = f'{self.__class__.__name__}'
        if len(self.input) > 0:
            graph += '\n |\n'
            for input in self.input:
                graph += input.computational_graph()
        return graph

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

        # Averaging spin multiplets if doing spin-adapted computation
        if options.get_str('CORRELATION_SOLVER') in ('SA-MRDSRG', 'SA_MRDSRG'):
            options.set_bool('SPIN_AVG_DENSITY', True)

        return options
