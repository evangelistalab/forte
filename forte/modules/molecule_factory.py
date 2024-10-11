from typing import List
from dataclasses import dataclass

from psi4 import geometry

from .module import Module
from forte.data import ForteData
from forte._forte import Symmetry


@dataclass
class MoleculeFactory(Module):
    """
    A module to define the molecule in a ForteData object
    """

    molecule: str

    def __post_init__(self):
        # This module initializes the molecule object in ForteData, so it does not require a ForteData object as input
        super().__init__()

    def _run(self, data: ForteData = None) -> ForteData:
        data = ForteData() if data is None else data
        data.molecule = geometry(self.molecule)
        data.symmetry = Symmetry(data.molecule.point_group().symbol().capitalize())

        return data
