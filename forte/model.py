from abc import ABC, abstractmethod

from forte.core import flog
from forte.molecule import Molecule
from forte.basis import Basis
from ._forte import StateInfo
from ._forte import Symmetry
from ._forte import make_ints_from_psi4


class Model(ABC):
    """
    An abstract base class used to implement a model.

    A model contains the information necessary to compute
    the Hamiltonian for a system (number of orbitals,
    symmetry information, basis set).
    This allows to deal with molecules, model Hamiltonians
    (e.g. Hubbard), effective Hamiltonians, in a unified way.
    """
    @abstractmethod
    def point_group(self) -> str:
        """The model point group"""

    @abstractmethod
    def ints(self, mo_space_info):
        """Make a ForteIntegral object"""
        pass


class MolecularModel(Model):
    """
    A class used to handle molecules

    This class knows the molecular structure and the basis,
    and is responsible for providing integrals over molecular orbitals.
    """
    def __init__(
        self,
        molecule: Molecule,
        basis: Basis,
        int_type: str = None,
        scf_aux_basis: Basis = None,
        corr_aux_basis: Basis = None
    ):
        """
        Initialize a MolecularModel object

        Parameters
        ----------
        molecule: Molecule
            the molecule information
        int_type: {'CONVENTIONAL', 'DF, 'CD', 'DISKDF'}
            the type of integrals used
        basis: Basis
            the computational basis
        scf_aux_basis: Basis
            the auxiliary basis set used in density-fitted SCF computations
        corr_aux_basis: Basis
            the auxiliary basis set used in density-fitted correlated computations
        """
        self._molecule = molecule
        self._basis = basis
        self._int_type = 'CONVENTIONAL' if int_type is None else int_type.upper()
        self._scf_aux_basis = scf_aux_basis
        self._corr_aux_basis = corr_aux_basis
        self.symmetry = Symmetry(molecule.molecule.point_group().symbol().capitalize())

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f"MolecularModel(\n{repr(self._molecule)},\n{repr(self._basis)},\n{self._int_type})"

    def __str__(self):
        """
        return a string representation of this object
        """
        return self.__repr__()

    @property
    def molecule(self):
        return self._molecule.molecule

    @property
    def int_type(self):
        return self._int_type

    @property
    def basis(self):
        return self._basis.__str__()

    @property
    def scf_aux_basis(self):
        if self._scf_aux_basis is None:
            return None
        return self._scf_aux_basis.__str__()

    @property
    def corr_aux_basis(self):
        if self._corr_aux_basis is None:
            return None
        return self._corr_aux_basis.__str__()

    @property
    def point_group(self) -> str:
        return self.symmetry.point_group_label()

    def state(self, charge: int, multiplicity: int, ms: float = None, sym: str = None, gasmin=None, gasmax=None):
        """This function is used to create a StateInfo object.
        It checks for potential errors.

        Parameters
        ----------
        charge: int
            total charge
        multiplicity: int
            the spin multiplicity of the state
        ms: float
            projection of spin on the z axis (e.g. 0.5, 2.0,).
            (default = lowest value consistent with multiplicity)
        sym: str
            the state irrep label (e.g., 'C2v')
        gasmin: list(int)
            the minimum number of electrons in each GAS space
        gasmax: list(int)
            the maximum number of electrons in each GAS space
        """
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
                f'(MolecularModel) The value of M_S ({twice_ms / 2.0}) is incompatible with the number of electrons ({nel})'
            )

        # compute the number of alpha/beta electrons
        na = (nel + twice_ms) // 2  # from ms = (na - nb) / 2
        nb = nel - na

        # compute the irrep index and produce a standard label
        if sym is None:
            if self.symmetry.nirrep() == 1:
                # in this case there is only one possible choice
                sym = 'A'
            else:
                raise ValueError(
                    f'(MolecularModel) The value of sym ({sym}) is invalid.'
                    f' Please specify a valid symmetry label.'
                )

        # get the irrep index from the symbol
        irrep = self.symmetry.irrep_label_to_index(sym)

        gasmin = [] if gasmin is None else gasmin
        gasmax = [] if gasmax is None else gasmax
        return StateInfo(na, nb, multiplicity, twice_ms, irrep, sym, gasmin, gasmax)

    def ints(self, data, options):
        flog('info', 'MolecularModel: preparing integrals from psi4')
        # if we do DF, we need to make sure that psi4's wavefunction object
        # has a DF_BASIS_MP2 basis registered
        if self.int_type == 'DF':
            import psi4
            aux_basis = psi4.core.BasisSet.build(
                self.molecule, 'DF_BASIS_MP2', self.corr_aux_basis, 'RIFIT', self.basis,
                puream=data.psi_wfn.basisset().has_puream()
            )
            data.psi_wfn.set_basisset('DF_BASIS_MP2', aux_basis)
        # get the appropriate integral object
        return make_ints_from_psi4(data.psi_wfn, options, data.mo_space_info, self._int_type)
