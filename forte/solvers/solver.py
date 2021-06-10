import psi4
from abc import ABC, abstractmethod
from forte.data import Data
from forte.model import Model, MolecularModel
from forte.results import Results
from forte.forte import MOSpaceInfo
from forte.molecule import Molecule
from forte.basis import Basis
import forte


class Solver(ABC):
    """
    A class used to implement a quantum chemistry solver.

    Attributes
    ----------
    """
    def __init__(self):
        self._results = Results()
        self._data = Data()
        self._output_file = 'output.dat'
        self._executed = False

    @abstractmethod
    def run(self):
        """Interface for computing the energy"""

    @property
    def results(self):
        return self._results

    @property
    def data(self):
        return self._data

    @property
    def executed(self):
        return self._executed

    # @property
    # def psi_wfn(self):
    #     return self._psi_wfn

    # @psi_wfn.setter
    # def psi_wfn(self, val):
    #     self._psi_wfn = val

    # @property
    # def scf_info(self):
    #     return self._scf_info

    # @scf_info.setter
    # def scf_info(self, val):
    #     self._scf_info = val

    # @property
    # def ints(self):
    #     return self._ints

    # @ints.setter
    # def ints(self, val):
    #     self._ints = val

    # @property
    # def mo_space_info(self):
    #     return self._mo_space_info

    # @mo_space_info.setter
    # def mo_space_info(self, val):
    #     self._mo_space_info = val

    # @property
    # def model(self):
    #     return self._model

    # @model.setter
    # def model(self, val):
    #     self._model = val

    @property
    def output_file(self):
        return self._output_file

    @output_file.setter
    def output_file(self, val):
        self._output_file = val

    def value(self, label):
        return self._results.value(label)

    def make_mo_space_info(self, mo_spaces):
        nmopi = self.scf_info.nmopi()
        return forte.make_mo_space_info_from_map(nmopi, self.model.point_group, mo_spaces)


class BasicSolver(Solver):
    """A simple solver"""
    def __init__(self):
        super().__init__()

    def run(self):
        """Nothing to run"""


def molecular_model(molecule: Molecule, basis: Basis):
    solver = BasicSolver()
    solver.data.model = MolecularModel(molecule=molecule, basis=basis)
    return solver
