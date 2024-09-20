from forte.data import ForteData
from .module import Module
from .workflow import Workflow
from .options_factory import OptionsFactory
from .objects_factory_psi4 import ObjectsFromPsi4
from .active_space_ints import ActiveSpaceInts


class ObjectsUtilPsi4(Module):
    """
    A utility module to prepare options through active space integrals from a Psi4 Wavefunction object
    """

    def __init__(self, active="ACTIVE", core=["RESTRICTED_DOCC"], options: dict = None, **kwargs):
        """
        Parameters
        ----------
        options: dict
            A dictionary of options. Defaults to None, in which case the options are read from psi4.
        """
        super().__init__(options=options)
        self.job = Workflow(
            [
                OptionsFactory(options=options),
                ObjectsFromPsi4(options=options, **kwargs),
                ActiveSpaceInts(active, core),
            ]
        )

    def _run(self, data: ForteData = None) -> ForteData:
        return self.job.run(data)
