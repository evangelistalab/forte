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

    Solver stores Forte base objects in a data attribute
    and a results object
    """
    def __init__(self):
        self._executed = False
        self._data = Data()
        self._results = Results()
        self._output_file = 'output.dat'

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

    @property
    def psi_wfn(self):
        return self.data.psi_wfn

    @psi_wfn.setter
    def psi_wfn(self, val):
        self.data.psi_wfn = val

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

    def state(self, charge: int, multiplicity: int, ms: float = None, sym: str = None):
        return self.data.model.state(charge, multiplicity, ms, sym)


class BasicSolver(Solver):
    """A simple solver used to instantiate a new job"""
    def __init__(self):
        super().__init__()

    def run(self):
        """Nothing to run"""


class CallbackHandler():
    """
    This class stores a list of callback functions labeled by an ID

    Use:
    > ch = CallbackHandler()
    > def func(state):
    >     <do something with state>   
    > ch.add_callback(id='post',func=func) # define callback with id='post'
    > ...
    > ch.callback('pre',mystate) # id 'pre' is not defined, skip
    > ch.callback('post',mystate) # calls func(mystate)
    """
    def __init__(self):
        self._callback_list = {}

    def add_callback(self, id, func):
        """Add a callback function labeled by an ID"""
        self._callback_list[id] = func

    def callback(self, id, state):
        """Call the function ID on a given state"""
        if id in self._callback_list:
            self._callback_list[id](state)


def solver_factory(molecule, basis):
    """A factory to build a basic solver object"""
    if isinstance(molecule, str):
        molecule = Molecule.from_geom(molecule)
    if isinstance(basis, str):
        basis = Basis(basis)
    solver = BasicSolver()
    solver.data.model = MolecularModel(molecule=molecule, basis=basis)
    return solver
