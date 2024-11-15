# import forte
from typing import List
from .module import Module
from forte.data import ForteData
from forte._forte import RDMsType, make_orbital_transformation


class OrbitalTransformation(Module):
    """
    A module to transform the orbitals and the integrals to a new basis.
    """

    def __init__(self, orb_type: str, transform_ints: bool = True):
        """
        Parameters
        ----------
        orb_type: str
            The type of orbital transformation to perform
        """
        super().__init__()
        self.orb_type = orb_type
        self.transform_ints = transform_ints

    def _run(self, data: ForteData) -> ForteData:
        orb_t = make_orbital_transformation(self.orb_type, data.scf_info, data.options, data.ints, data.mo_space_info)
        orb_t.compute_transformation()
        Ua = orb_t.get_Ua()
        Ub = orb_t.get_Ub()
        data.scf_info.rotate_orbitals(Ua, Ub, self.transform_ints)
        if data.as_ints is not None:
            data.as_ints.rotate_orbitals(Ua, Ub, self.transform_ints)

        return data
