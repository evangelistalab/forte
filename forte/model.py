from abc import ABC, abstractmethod
from forte.molecule import Molecule
from forte.basis import Basis
from forte.forte import StateInfo
from forte.forte import Symmetry


class Model(ABC):
    """
    An abstract base class used to implement a model.

    Attributes
    ----------
    """
    @abstractmethod
    def point_group(self) -> str:
        """The model point group"""


class MolecularModel(Model):
    """
    A class used to handle molecules.

    This class is used to compute the integrals once a model has been defined.

    Attributes
    ----------
    """
    def __init__(self, molecule: Molecule, basis: Basis):
        self._molecule = molecule
        self._basis = basis
        self.symmetry = Symmetry(molecule.molecule.point_group().symbol().capitalize())

    def __repr__(self):
        """
        return a string representation of this object
        """

    def __str__(self):
        """
        return a string representation of this object
        """

    @property
    def molecule(self):
        return self._molecule.molecule

    @property
    def basis(self):
        return self._basis.basis

    @property
    def point_group(self) -> str:
        return self.symmetry.point_group_label()

    def state(self, charge: int, multiplicity: int, ms: float = None, sym: str = None):
        """This function is ued to create a StateInfo object. It checks for potential errors.""""
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
        molecule = self.molecule
        natom = molecule.natom()
        nel = round(sum([molecule.Z(i) for i in range(natom)])) - charge

        if (nel - twice_ms) % 2 != 0:
            raise ValueError(
                f'(MolecularModel) The value of M_S ({ms}) is incompatible with the number of electrons ({nel})'
            )

        # compute the number of alpha/beta electrons
        na = (nel + twice_ms) // 2  # from ms = (na - nb) / 2
        nb = nel - na

        # compute the irrep index and produce a standard label
        if sym is None:
            if sym.nirrep() == 1:
                # in this case there is only one possible choice
                sym = 'A'
            else:
                raise ValueError(
                    f'(MolecularModel) The value of sym ({sym}) is invalid.'
                    f' Please specify a valid symmetry label.'
                )

        # get the irrep index from the symbol
        irrep = self.symmetry.irrep_label_to_index(sym)
        return StateInfo(na, nb, multiplicity, twice_ms, irrep, sym)
