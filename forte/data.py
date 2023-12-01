from dataclasses import dataclass

import psi4
from forte import Model
from ._forte import ForteOptions, SCFInfo, MOSpaceInfo, ForteIntegrals, ActiveSpaceIntegrals, StateInfo


@dataclass
class ForteData:
    """Dataclass for holding data used by forte."""

    options: ForteOptions = None
    state_weights_map: dict[StateInfo] = None
    model: Model = None
    scf_info: SCFInfo = None
    mo_space_info: MOSpaceInfo = None
    ints: ForteIntegrals = None
    as_ints: ActiveSpaceIntegrals = None
    psi_wfn: psi4.core.Wavefunction = None
