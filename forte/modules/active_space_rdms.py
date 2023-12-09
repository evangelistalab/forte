from typing import List
from .module import Module
from forte.data import ForteData
from forte._forte import RDMsType


class ActiveSpaceRDMs(Module):

    """
    A module to prepare an ActiveSpaceIntegral
    """

    def __init__(self, max_rdm_level: int, rdms_type=RDMsType.spin_dependent):
        """
        Parameters
        ----------
        max_rdm_level: int
            The maximum level of RDMs to be computed.
        """
        super().__init__()
        self.max_rdm_level = max_rdm_level
        self.rdms_type = rdms_type

    def _run(self, data: ForteData) -> ForteData:
        import forte

        data.rdms = data.active_space_solver.compute_average_rdms(
            data.state_weights_map, self.max_rdm_level, self.rdms_type
        )

        return data
