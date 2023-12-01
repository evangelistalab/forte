from dataclasses import dataclass

from psi4.core import Wavefunction
from forte import Model
from ._forte import ForteOptions, SCFInfo, MOSpaceInfo, ForteIntegrals, ActiveSpaceIntegrals, StateInfo


@dataclass
class ForteData:
    """Dataclass for holding data passed around in forte."""

    options: ForteOptions = None
    state_weights_map: dict[StateInfo] = None
    model: Model = None
    scf_info: SCFInfo = None
    mo_space_info: MOSpaceInfo = None
    ints: ForteIntegrals = None
    as_ints: ActiveSpaceIntegrals = None
    psi_wfn: Wavefunction = None
