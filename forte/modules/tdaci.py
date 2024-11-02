from typing import List

from forte.data import ForteData
from forte._forte import to_state_nroots_map, make_active_space_method, TDCI

from .module import Module
from .active_space_ints import ActiveSpaceInts


class TDACI(Module):

    """
    A module to perform real-time time dependent ACI calculations.
    """

    def __init__(self):
        """
        Parameters
        ----------
        """
        super().__init__()

    def _run(self, data: ForteData) -> ForteData:
        data = ActiveSpaceInts(active="ACTIVE", core=["RESTRICTED_DOCC"]).run(data)
        # As long as TDCI takes ActiveSpaceMethod rather than ActiveSpaceSolver, this error message must stay.
        if len(data.state_weights_map) != 1:
            raise("TDACI does not support multi-state computations yet. File an issue if you need this!")
        else:
            state = next(iter(data.state_weights_map))
        state_map = to_state_nroots_map(data.state_weights_map)
        active_space_method = make_active_space_method(
            "ACI", state, data.options.get_int("NROOT"), data.scf_info, data.mo_space_info, data.as_ints, data.options
        )
        active_space_method.set_quiet_mode()
        active_space_method.compute_energy()

        tdci = TDCI(active_space_method, data.scf_info, data.options, data.mo_space_info, data.as_ints)
        energy = tdci.compute_energy()
        data.results.add("energy", energy, "TDACI energy", "hartree")

        return data
