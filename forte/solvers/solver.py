from abc import ABC, abstractmethod
import logging

from forte.data import Data
from forte.model import MolecularModel
from forte.results import Results
from forte.molecule import Molecule
from forte.basis import Basis

import forte


class Solver(ABC):
    """
    A class used to implement a quantum chemistry solver.

    Solver stores Forte base objects in a data attribute
    and a results object.
    """
    def __init__(self):
        self._executed = False
        self._data = Data()
        self._results = Results()
        self._output_file = 'output.dat'

    @abstractmethod
    def run(self):
        """Interface for running the solver."""

    @property
    def results(self):
        return self._results

    @property
    def data(self):
        return self._data

    @property
    def executed(self):
        return self._executed

    @property
    def psi_wfn(self):
        return self.data.psi_wfn

    @psi_wfn.setter
    def psi_wfn(self, val):
        self.data.psi_wfn = val

    @property
    def scf_info(self):
        return self.data.scf_info

    @scf_info.setter
    def scf_info(self, val):
        self.data.scf_info = val

    @property
    def mo_space_info(self):
        return self.data.mo_space_info

    @mo_space_info.setter
    def mo_space_info(self, val):
        self.data.mo_space_info = val

    @property
    def model(self):
        return self.data.model

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

    def state(self, charge: int, multiplicity: int, ms: float = None, sym: str = None):
        return self.data.model.state(charge, multiplicity, ms, sym)


class BasicSolver(Solver):
    """A simple solver used to instantiate a new job"""
    def __init__(self):
        super().__init__()

    def run(self):
        pass


def solver_factory(molecule, basis, scf_aux_basis=None, corr_aux_basis=None):
    """A factory to build a basic solver object"""
    logging.info('Calling solver factory')

    # TODO: generalize to other type of models (e.g. if molecule/basis are not provided)

    # convert string arguments to objects if necessary
    if isinstance(molecule, str):
        molecule = Molecule.from_geom(molecule)
    if isinstance(basis, str):
        basis = Basis(basis)
    if isinstance(scf_aux_basis, str):
        scf_aux_basis = Basis(scf_aux_basis)
    if isinstance(corr_aux_basis, str):
        corr_aux_basis = Basis(corr_aux_basis)

    # create an empty solver and pass the model in
    solver = BasicSolver()
    solver.data.model = MolecularModel(
        molecule=molecule, basis=basis, scf_aux_basis=scf_aux_basis, corr_aux_basis=corr_aux_basis
    )
    return solver
