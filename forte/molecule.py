from pathlib import Path
from psi4 import geometry


class Molecule:
    """
    A class used to represent a molecule. It stores a psi4.core.Molecule object

    Attributes
    ----------
    molecule : psi4.core.Molecule object
        a molecule object
    """
    def __init__(self, molecule=None):
        """
        initialize a Molecule object

        Parameters
        ----------
        molecule : psi4.core.Molecule object
            a molecule object
        """
        self._molecule = molecule

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f"Molecule({self._molecule.to_string(dtype='xyz')})"

    def __str__(self):
        """
        return a string representation of this object
        """
        return self.__repr__()

    @property
    def molecule(self):
        return self._molecule

    @staticmethod
    def from_geom(geom):
        """
        make a Molecule object from a string

        Parameters
        ----------
        geom : str
            a string representing a molecule
        """
        return Molecule(geometry(geom))  # <- call psi4.geometry()

    @staticmethod
    def from_geom_file(filename, path='.'):
        """
        make a Molecule object from a file named path/filename

        Parameters
        ----------
        filename : str
            the name of the file
        path : str
            the path of the file (default = '.')
        """
        with open(Path(path) / filename, 'r') as file:
            geom = file.read()
            return Molecule.from_geom(geom)
