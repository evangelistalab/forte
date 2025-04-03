from .apps_helpers import parse_state
from forte.modules.workflow import Workflow
from forte.modules.options_factory import OptionsFactory
from forte.modules.molecule_factory import MoleculeFactory
from forte.modules.state_factory import StateFactory
from forte.modules.hf import HF


def hf(geom, basis, state, e_convergence=1e-8, d_convergence=1e-8, options=None):
    charge, multiplicity, sym = parse_state(state)
    hf_workflow = Workflow(
        [
            # Pass the options to the ForteOptions object
            OptionsFactory(options),
            # Generate the molecule
            MoleculeFactory(geom),
            # Generate the state
            StateFactory(charge=charge, multiplicity=multiplicity, sym=sym),
            # Run the HF calculation
            HF(basis=basis, e_convergence=e_convergence, d_convergence=d_convergence),
        ],
        name="HF Workflow",
    )
    data = hf_workflow.run()
    data.options.set_str("basis", basis)
    return data
