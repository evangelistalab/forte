from abc import ABC, abstractmethod
from forte.basis import Basis
from forte.model import Model
from forte.results import Results


class Method(ABC):
    """
    A class used to implement methods.

    Attributes
    ----------
    """
    @abstractmethod
    def energy(self, model: Model, results: Results):
        """Interface for computing the energy"""
