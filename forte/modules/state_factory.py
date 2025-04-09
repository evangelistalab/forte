from dataclasses import dataclass
from typing import List

from psi4 import geometry

from .module import Module
from forte.data import ForteData
from forte._forte import Symmetry, StateInfo
from .validators import Feature, module_validation


@dataclass
class StateFactory(Module):
    """
    A module used to create a StateInfo object. It checks for potential errors.

    Parameters
    ----------
    charge: int
        total charge
    multiplicity: int
        the spin multiplicity of the state
    ms: float
        projection of spin on the z axis (e.g. 0.5, 2.0).
        (default = lowest value consistent with multiplicity)
    sym: str
        the state irrep label (e.g., 'C2v')
    gasmin: list(int)
        the minimum number of electrons in each GAS space
    gasmax: list(int)
        the maximum number of electrons in each GAS space
    """

    charge: int
    multiplicity: int
    ms: float = None
    sym: str = None
    gasmin: List[int] = None
    gasmax: List[int] = None

    def __post_init__(self):
        super().__init__()

    @module_validation(needs=[Feature.MOLECULE])
    def _run(self, data: ForteData) -> ForteData:
        if self.ms is None:
            # If ms = None take the lowest value consistent with multiplicity
            # For example:
            #   singlet: multiplicity = 1 -> twice_ms = 0 (ms = 0)
            #   doublet: multiplicity = 2 -> twice_ms = 1 (ms = 1/2)
            #   triplet: multiplicity = 3 -> twice_ms = 0 (ms = 0)
            twice_ms = (self.multiplicity + 1) % 2
        else:
            twice_ms = round(2.0 * ms)

        # compute the number of electrons
        if data.molecule is None:
            raise ValueError(f"(MolecularModel) The molecule object is not initialized.")
        molecule = data.molecule
        natom = molecule.natom()
        nel = round(sum([molecule.Z(i) for i in range(natom)])) - self.charge

        if (nel - twice_ms) % 2 != 0:
            raise ValueError(
                f"(MolecularModel) The value of M_S ({twice_ms / 2.0}) is incompatible with the number of electrons ({nel})"
            )

        # compute the number of alpha/beta electrons
        na = (nel + twice_ms) // 2  # from ms = (na - nb) / 2
        nb = nel - na

        # compute the irrep index and produce a standard label
        if self.sym is None:
            if data.symmetry.nirrep() == 1:
                # in this case there is only one possible choice
                sym = "A"
            else:
                raise ValueError(
                    f"(MolecularModel) The value of sym ({self.sym}) is invalid."
                    f" Please specify a valid symmetry label."
                )

        # get the irrep index from the symbol
        irrep = data.symmetry.irrep_label_to_index(self.sym)

        self.gasmin = [] if self.gasmin is None else self.gasmin
        self.gasmax = [] if self.gasmax is None else self.gasmax

        state = StateInfo(na, nb, self.multiplicity, twice_ms, irrep, self.sym, self.gasmin, self.gasmax)
        data.state_weights_map = {state: [1.0]}
        return data


@dataclass
class MultiStateFactory(Module):
    """
    A module used to create a dictionary of StateInfo objects. It checks for potential errors.

    Args:
        state (dict): A dictionary that specifies the state information. [{"charge": 0, "multiplicity": 1, "sym": "ag", "weights": [0.5, 0.5]}]

        states (list): A list of dictionaries that specify the state information.
    """

    states: List[dict] = None

    def __post_init__(self):
        super().__init__()

    @module_validation(needs=[Feature.MOLECULE])
    def _run(self, data: ForteData) -> ForteData:
        if self.states is None:
            if data.state_weights_map is None:
                raise ValueError("The option states must be provided.")
            else:
                return data
        # if states is a dictionary, insert it into a list
        if isinstance(self.states, dict):
            self.states = [self.states]

        state_weights_map = {}
        for state in self.states:
            self.process(state, state_weights_map, data)

        data.state_weights_map = state_weights_map

        return data

    def process(self, state, state_weights_map, data):
        # check that dictionary has the right keys
        if not all(key in state for key in ["charge", "multiplicity", "sym"]):
            raise ValueError("State dictionary must contain 'charge', 'multiplicity', and 'sym' keys.")

        # check if weights are provided
        if "weights" in state:
            if not isinstance(state["weights"], list):
                raise ValueError("Weights must be a list.")
            weights = state.get("weights", [1.0])
        # alternatively, check if nstates is provided and use equal weights
        elif "nstates" in state:
            weights = [1 / nstates for i in range(nstates)]
        # if no weights are provided, assume one state with weight 1.0
        else:
            weights = [1.0]

        charge = int(state["charge"])
        multiplicity = int(state["multiplicity"])
        sym = state["sym"]

        ms = state.get("ms", None)

        if ms is None:
            # If ms = None take the lowest value consistent with multiplicity
            # For example:
            #   singlet: multiplicity = 1 -> twice_ms = 0 (ms = 0)
            #   doublet: multiplicity = 2 -> twice_ms = 1 (ms = 1/2)
            #   triplet: multiplicity = 3 -> twice_ms = 0 (ms = 0)
            twice_ms = (multiplicity + 1) % 2
        else:
            twice_ms = round(2.0 * ms)

        # compute the number of electrons
        if data.molecule is None:
            raise ValueError(f"(MolecularModel) The molecule object is not initialized.")
        molecule = data.molecule
        natom = molecule.natom()
        nel = round(sum([molecule.Z(i) for i in range(natom)])) - charge

        if (nel - twice_ms) % 2 != 0:
            raise ValueError(
                f"(MolecularModel) The value of M_S ({twice_ms / 2.0}) is incompatible with the number of electrons ({nel})"
            )

        # compute the number of alpha/beta electrons
        na = (nel + twice_ms) // 2  # from ms = (na - nb) / 2
        nb = nel - na

        # compute the irrep index and produce a standard label
        if sym is None:
            if data.symmetry.nirrep() == 1:
                # in this case there is only one possible choice
                sym = "A"
            else:
                raise ValueError(
                    f"(MolecularModel) The value of sym ({sym}) is invalid." f" Please specify a valid symmetry label."
                )

        # get the irrep index from the symbol
        irrep = data.symmetry.irrep_label_to_index(sym)

        gasmin = state.get("gasmin", [])
        gasmax = state.get("gasmax", [])

        state_info = StateInfo(na, nb, multiplicity, twice_ms, irrep, sym, gasmin, gasmax)
        state_weights_map[state_info] = weights
