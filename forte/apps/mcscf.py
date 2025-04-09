from .apps_helpers import parse_active_space
from .hf import hf
from forte.modules.workflow import Workflow
from forte.modules.objects_factory_psi4 import ObjectsFromPsi4
from forte.modules.active_space_selector import ActiveSpaceSelector
from forte.modules.mcscf import MCSCF
from forte.modules.options_factory import OptionsFactory
from forte.modules.state_factory import MultiStateFactory


def mcscf(data, active_space, solver_type="FCI", options=None, states=None):
    """
    Run a MCSCF calculation using the Forte workflow.

    Args:
        active_space (str): Active space specification (e.g., "6-31G*").
        solver_type (str): Type of solver to be used (e.g., "FCI").
        options (dict, optional): Additional options for the calculation.
        data: Precomputed data from a previous calculation.
        geom (str, optional): Geometry of the molecule in XYZ format.
        basis (str, optional): Basis set to be used.
        state (dict, optional): Electronic state of the system.
            e.g., {"charge": 0, "multiplicity": 1, "sym": "ag"}.
    Returns:
        data: The result of the MCSCF calculation.
    """
    # Setup a workflow to run MCSCF
    mcscf_workflow = Workflow(
        [
            # Parse the active space before defining the MOSpaceInfo object
            ActiveSpaceSelector(active_space),
            # Generate the necessary objects from psi4
            ObjectsFromPsi4(ref_wfn=data.psi_wfn),
            # Parse the multiple states if provided
            MultiStateFactory(states),
            # Run the MCSCF calculation
            MCSCF(solver_type=solver_type),
        ],
        name="MCSCF Workflow",
    )

    # Run the workflow
    data = mcscf_workflow.run(data)

    return data
