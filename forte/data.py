from dataclasses import dataclass

from psi4.core import Wavefunction

from ._forte import (
    ForteOptions,
    SCFInfo,
    MOSpaceInfo,
    ForteIntegrals,
    ActiveSpaceIntegrals,
    StateInfo,
    ActiveSpaceSolver,
    RDMs,
)
from forte import Model


@dataclass
class ForteData:
    """Dataclass for holding data passed around in forte.

    This class is the container for all the data that is passed around in forte from one module to another.
    """

    options: ForteOptions = None
    state_weights_map: dict[StateInfo] = None
    model: Model = None
    scf_info: SCFInfo = None
    mo_space_info: MOSpaceInfo = None
    ints: ForteIntegrals = None
    as_ints: ActiveSpaceIntegrals = None
    psi_wfn: Wavefunction = None
    active_space_solver: ActiveSpaceSolver = None
    rdms: rdms = None
