from dataclasses import dataclass

import psi4
import forte


@dataclass
class Data:
    model: forte.Model = None
    scf_info: forte.SCFInfo = None
    mo_space_info: forte.MOSpaceInfo = None
    ints: forte.ForteIntegrals = None
    as_ints: forte.ActiveSpaceIntegrals = None
    psi_wfn: psi4.core.Wavefunction = None
