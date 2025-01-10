from abc import ABC, abstractmethod
from typing import List, Type, Union

from forte.data import ForteData
from forte.core import flog, increase_log_depth


class Module(ABC):
    """
    Represents a module that can be used in a workflow.
    """

    def __init__(
        self,
        options=None,
    ):
        """
        Parameters
        ----------
        input_modules: Union[Module, List[Module]]
            A single input module or a list of input modules. Defaults to None.
        options: dict
            A dictionary of options. Defaults to None.
        """
        self._options = options
        self._executed = False

    @property
    def options(self) -> dict:
        return self._options

    # decorate to increase the log depth
    @increase_log_depth
    def run(self, data: ForteData = None) -> ForteData:
        """
        A general solver interface.

        This method is common to all solvers, and in turn it is routed to
        the method ``_run()`` implemented differently in each solver.

        Return
        ------
            A data object
        """
        # log call to run()
        flog("info", f"{type(self).__name__}: calling run()")

        # call derived class implementation of _run()
        data = self._run(data)

        # log end of run()
        flog("info", f"{type(self).__name__}: run() finished executing")

        # set executed flag
        self._executed = True

        return data

    @abstractmethod
    def _run(self, data: ForteData = None) -> ForteData:
        """The actual run function implemented by each method"""
        NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}"
