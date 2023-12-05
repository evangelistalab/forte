from forte.data import ForteData
from .module import Module
from .sequential import Sequential
from .options_factory import OptionsFactory
from .objects_factory_psi4 import ObjectsFactoryPsi4
from .active_space_ints_factory import ActiveSpaceIntsFactory


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
        self.seq = Sequential(
            [
                OptionsFactory(options=options),
                ObjectsFactoryPsi4(options=options, **kwargs),
                ActiveSpaceIntsFactory(active, core),
            ]
        )

    def _run(self, data: ForteData = None) -> ForteData:
        return self.seq.run(data)
