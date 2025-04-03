from .apps_helpers import parse_state
from forte.modules.workflow import Workflow
from forte.modules.options_factory import OptionsFactory
from forte.modules.molecule_factory import MoleculeFactory
from forte.modules.state_factory import StateFactory
from forte.modules.hf import HF


def hf(geom, basis, state, e_convergence=1e-8, d_convergence=1e-6, options=None):
    """
    Run a Hartree-Fock calculation using the Forte workflow.

    Args:
        geom (str): Geometry of the molecule in XYZ format.
        basis (str): Basis set to be used.
        state (dict): Electronic state of the system (e.g., {"charge": 0, "multiplicity": 1, "sym": "ag"},).
        e_convergence (float): Energy convergence threshold.
        d_convergence (float): Density convergence threshold.
        options (dict, optional): Additional options for the calculation.

    Returns:
        data: The result of the Hartree-Fock calculation.
    """
    # Parse the state string to get charge, multiplicity, and symmetry
    charge, multiplicity, sym = parse_state(state)

    # Set up the workflow to run HF
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

    # Run the workflow
    data = hf_workflow.run()

    return data
