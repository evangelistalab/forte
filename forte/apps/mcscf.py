from .apps_helpers import parse_active_space
from .hf import hf
from forte.modules.workflow import Workflow
from forte.modules.objects_factory_psi4 import ObjectsFromPsi4
from forte.modules.active_space_selector import ActiveSpaceSelector
from forte.modules.mcscf import MCSCF
from forte.modules.options_factory import OptionsFactory


def mcscf(active_space, solver_type="FCI", options=None, data=None, geom=None, basis=None, state=None):
    if data is None and geom is not None and basis is not None:
        # If data is None, create a new data object
        data = hf(geom, basis, state, options=options)
    elif data is None:
        raise ValueError("Either data or geom and basis must be provided.")

    # Setup a workflow to run MCSCF
    mcscf_workflow = Workflow(
        [
            # Parse the active space before defining the MOSpaceInfo object
            ActiveSpaceSelector(active_space),
            # Generate the necessary objects from psi4
            ObjectsFromPsi4(ref_wfn=data.psi_wfn),
            # Run the MCSCF calculation
            MCSCF(solver_type=solver_type),
        ],
        name="MCSCF Workflow",
    )

    # Run the workflow
    data = mcscf_workflow.run(data)

    return data
