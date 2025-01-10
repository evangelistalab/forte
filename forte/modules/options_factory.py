import psi4
import forte

from forte.data import ForteData
from .module import Module

from forte.register_forte_options import register_forte_options


class OptionsFactory(Module):
    """
    A module to generate the ForteOptions object
    """

    def __init__(self, options: dict = None):
        """
        Parameters
        ----------
        options: dict
            A dictionary of options. Defaults to None, in which case the options are read from psi4.
        """
        super().__init__(options=options)

    def _run(self, data: ForteData = None) -> ForteData:
        if data is None:
            data = ForteData()
        if self.options is None or isinstance(self._options, dict):
            psi4.core.print_out(
                "\n  Forte will use options passed as a dictionary. Option read from psi4 will be ignored\n"
            )
            data.options = forte.ForteOptions()
            register_forte_options(data.options)
            # if options is a dictionary then set the options from the dictionary
            if isinstance(self._options, dict):
                data.options.set_from_dict(self._options)
        else:
            # Copy globals into a new object
            data.options = forte.ForteOptions(forte.forte_options)
            psi4_options = psi4.core.get_options()
            psi4_options.set_current_module("FORTE")

            # Get the forte option object
            data.options.get_options_from_psi4(psi4_options)

        return data
