from .apps_helpers import parse_active_space
from .hf import run_hf
from forte.modules.workflow import Workflow
from forte.modules.objects_factory_psi4 import ObjectsFromPsi4
from forte.modules.active_space_selector import ActiveSpaceSelector
from forte.modules.mcscf import MCSCF


def run_mcscf(geom, basis, state, active_space, solver_type="FCI"):
    data = run_hf(geom, basis, state)
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
    data = mcscf_workflow.run(data)
    return data
