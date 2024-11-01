from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from psi4.core import Wavefunction, Molecule

from ._forte import (
    ForteOptions,
    SCFInfo,
    MOSpaceInfo,
    ForteIntegrals,
    ActiveSpaceIntegrals,
    StateInfo,
    ActiveSpaceSolver,
    RDMs,
    Symmetry,
)
from forte import Model, Results


@dataclass
class ForteData:
    """Dataclass for holding objects passed around in Forte.

    This class is the container for all the data that is passed around in forte from one module to another.

    Attributes
    ----------
    options: ForteOptions
        The options for the current run.
    state_weights_map: dict[StateInfo]
        A dictionary with the weights for each state.
    molecule: Molecule
        The molecule object (psi4.core.Molecule).
    model: Model
        The model object.
    scf_info: SCFInfo
        The SCF information.
    mo_space_info: MOSpaceInfo
        The molecular orbital space information.
    ints: ForteIntegrals
        The integrals object.
    as_ints: ActiveSpaceIntegrals
        The active space integrals object.
    psi_wfn: Wavefunction
        The psi4 Wavefunction object.
    active_space_solver: ActiveSpaceSolver
        The active space solver object.
    rdms: rdms
        The reduced density matrices.
    results: Results
        The results object.
    """

    options: ForteOptions = None
    state_weights_map: dict[StateInfo] = None
    symmetry: Symmetry = None
    molecule: Optional[Molecule] = None
    model: Model = None
    scf_info: SCFInfo = None
    mo_space_info: MOSpaceInfo = None
    ints: ForteIntegrals = None
    as_ints: ActiveSpaceIntegrals = None
    psi_wfn: Wavefunction = None
    active_space_solver: ActiveSpaceSolver = None
    rdms: RDMs = None
    results: Results = field(default_factory=Results)
