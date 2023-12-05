from typing import List
from .module import Module
from forte.data import ForteData


class ActiveSpaceIntsFactory(Module):
    """
    A module to prepare an ActiveSpaceIntegral
    """

    def __init__(self, active: str, core: List[str]):
        """
        Parameters
        ----------
        active : str
            The label of the space to be used for the active space
        core : List[str]
            A list of labels of the spaces to be used for the core space
        """
        super().__init__()
        self.active = active
        self.core = core

    def _run(self, data: ForteData) -> ForteData:
        import forte

        data.as_ints = forte.make_active_space_ints(data.mo_space_info, data.ints, self.active, self.core)
        return data
