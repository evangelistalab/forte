from abc import abstractmethod
from typing import List, Type, Union

from forte.data import ForteData
from forte.core import flog, increase_log_depth


class Module:
    """
    Represents a module in a computational graph.

    A `Module` can store a list of input modules and the features needed and provided.
    """

    def __init__(self, input_modules: Union["Module", List["Module"]] = None, options=None):
        """
        Parameters
        ----------
        input_modules: Union[Module, List[Module]]
            A single input module or a list of input modules. Defaults to None.
        options: dict
            A dictionary of options. Defaults to None.
        """
        if input_modules is None:
            self._input_modules = []
        elif isinstance(input_modules, Module):
            self._input_modules = [input_modules]
        else:
            self._input_modules = input_modules
        self._options = options

    @property
    def input_modules(self) -> List["Module"]:
        return self._input_modules

    @property
    def options(self) -> dict:
        return self._options

    def _check_input(self):
        """
        Verify that this module can get all that it needs from its inputs.
        """
        for need in self.needs:
            if not any(need in input_module.provides for input_module in self.input_modules):
                missing_feature_error = (
                    f"{self.__class__.__name__} needs {need}, " "which is not provided by input modules."
                )
                raise AssertionError(missing_feature_error)

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

        for input_module in self.input_modules:
            data = input_module.run(data)

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
