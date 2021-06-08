from abc import ABC, abstractmethod
from forte.molecule import Molecule
from forte.basis import Basis
from forte.forte import StateInfo


class Model(ABC):
    """
    A class used to implement a model.

    Attributes
    ----------
    """
    @abstractmethod
    def point_group(self) -> str:
        """The model point group"""


def label_to_irrep_num_(irrep_label, pg_symbol):
    pg_irrep_labels = {
        'C1': ['A'],
        'Cs': ['Ap', 'App'],
        'Ci': ['Ag', 'Au'],
        'C2': ['A', 'B'],
        'C2h': ['Ag', 'Bg', 'Au', 'Bu'],
        'C2v': ['A1', 'A2', 'B1', 'B2'],
        'D2': ['A', 'B1', 'B2', 'B3'],
        'D2h': ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']
    }

    pg_symbol = pg_symbol.capitalize()
    pg_labels_dict = {}
    for i, label in enumerate(pg_irrep_labels[pg_symbol]):
        pg_labels_dict[label.capitalize()] = i

    irrep_label = irrep_label.capitalize()
    return pg_labels_dict[irrep_label]


def irrep_num_to_label_(h, pg_symbol):
    pg_symbol = pg_symbol.capitalize()
    pg_irrep_labels = {
        'C1': ['A'],
        'Cs': ['Ap', 'App'],
        'Ci': ['Ag', 'Au'],
        'C2': ['A', 'B'],
        'C2h': ['Ag', 'Bg', 'Au', 'Bu'],
        'C2v': ['A1', 'A2', 'B1', 'B2'],
        'D2': ['A', 'B1', 'B2', 'B3'],
        'D2h': ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']
    }
    return pg_irrep_labels[pg_symbol][h]


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
        return self._molecule.molecule

    @property
    def basis(self):
        return self._basis.basis

    @property
    def point_group(self) -> str:
        return self.molecule.point_group().symbol()

    def state(self, charge: int, multiplicity: int, ms: float = None, sym: str = None):
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
            raise ValueError(f'The value of M_S ({ms}) is incompatible with the number of electrons ({nel})')

        # compute the number of alpha/beta electrons
        na = (nel + twice_ms) // 2  # from ms = (na - nb) / 2
        nb = nel - na

        # compute the irrep index and produce a standard label
        pg_symbol = molecule.point_group().symbol().capitalize()
        irrep = label_to_irrep_num_(sym, pg_symbol)
        irrep_label = irrep_num_to_label_(irrep, pg_symbol)
        return StateInfo(na, nb, multiplicity, twice_ms, irrep, irrep_label)