from os.path import isfile
from typing import List

from psi4.core import print_out

from forte._forte import to_state_nroots_map, make_active_space_solver

from forte.data import ForteData
from forte.proc.external_active_space_solver import write_external_active_space_file

from .module import Module


class ActiveSpaceSolver(Module):
    """
    A module to prepare an ActiveSpaceIntegral
    """

    def __init__(self, solver_type: str):
        """
        Parameters
        ----------
        solver_type: str
            The type of the active space solver.
        """
        super().__init__()
        self.solver_type = solver_type
        self.state_energies_list = None

    def _run(self, data: ForteData) -> ForteData:
        state_map = to_state_nroots_map(data.state_weights_map)

        data.active_space_solver = make_active_space_solver(
            self.solver_type, state_map, data.scf_info, data.mo_space_info, data.options, data.as_ints
        )

        if self.solver_type == "EXTERNAL":
            write_external_active_space_file(data.as_ints, state_map, data.mo_space_info, "as_ints.json")
            msg = "External solver: save active space integrals to as_ints.json"
            print(msg)
            print_out(msg)

            if not isfile("rdms.json"):
                msg = "External solver: rdms.json file not present, exit."
                print(msg)
                print_out(msg)
                # finish the computation
                exit()
        # if rdms.json exists, then run "external" as_solver to compute energy
        data.state_energies_list = data.active_space_solver.compute_energy()

        return data
