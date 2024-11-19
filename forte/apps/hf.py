from .apps_helpers import parse_state
from forte.modules.workflow import Workflow
from forte.modules.options_factory import OptionsFactory
from forte.modules.molecule_factory import MoleculeFactory
from forte.modules.state_factory import StateFactory
from forte.modules.hf import HF


def run_hf(geom, basis, state):
    charge, multiplicity, sym = parse_state(state)
    hf_workflow = Workflow(
        [
            OptionsFactory(),
            MoleculeFactory(geom),
            StateFactory(charge=charge, multiplicity=multiplicity, sym=sym),
            HF(basis=basis),
        ],
        name="HF Workflow",
    )
    data = hf_workflow.run()
    return data
