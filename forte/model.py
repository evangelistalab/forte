from abc import ABC, abstractmethod
from forte.molecule import Molecule
from forte.basis import Basis


class Model(ABC):
    """
    A class used to implement a model.

    Attributes
    ----------
    """


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
        return self._molecule

    @property
    def basis(self):
        return self._basis.basis
