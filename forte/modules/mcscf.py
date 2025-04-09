# import forte
from typing import List
from .module import Module
from forte.data import ForteData
from forte._forte import (
    to_state_nroots_map,
    make_active_space_solver,
    make_mcscf_two_step,
    MOSpaceInfo,
    make_mo_space_info_from_map,
)


class MCSCF(Module):
    """
    A module to perform MCSCF calculations.
    """

    def __init__(self, solver_type: str = "FCI"):
        """
        Parameters
        ----------
        solver_type: str
            The type of the active space solver.
        """
        super().__init__()
        self.solver_type = solver_type

    def _run(self, data: ForteData) -> ForteData:
        state_map = to_state_nroots_map(data.state_weights_map)

        data.active_space_solver = make_active_space_solver(
            self.solver_type, state_map, data.scf_info, data.mo_space_info, data.options
        )
        mcscf = make_mcscf_two_step(
            data.active_space_solver, data.state_weights_map, data.scf_info, data.options, data.mo_space_info, data.ints
        )

        average_energy = mcscf.compute_energy()
        data.results.add("mcscf energy", average_energy, "MCSCF energy", "hartree")

        # Store the energy of each state
        state_index = 0
        for state, energies in sorted(data.active_space_solver.state_energies_map().items()):
            for energy in energies:
                data.results.add(f"mcscf energy [{state_index}]", energy, f"MCSCF energy of state {state}", "hartree")
                state_index += 1

        return data
