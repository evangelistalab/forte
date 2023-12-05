from typing import List
from .module import Module
from forte.data import ForteData

from forte.proc.external_active_space_solver import write_external_active_space_file


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
        import forte

        state_map = forte.to_state_nroots_map(data.state_weights_map)

        data.active_space_solver = forte.make_active_space_solver(
            self.solver_type, state_map, data.scf_info, data.mo_space_info, data.as_ints, data.options
        )

        if self.solver_type == "EXTERNAL":
            write_external_active_space_file(data.as_ints, state_map, data.mo_space_info, "as_ints.json")
            msg = "External solver: save active space integrals to as_ints.json"
            print(msg)
            psi4.core.print_out(msg)

            if not os.path.isfile("rdms.json"):
                msg = "External solver: rdms.json file not present, exit."
                print(msg)
                psi4.core.print_out(msg)
                # finish the computation
                exit()
        # if rdms.json exists, then run "external" as_solver to compute energy
        data.state_energies_list = data.active_space_solver.compute_energy()

        return data
