from enum import Enum, auto


class Feature(Enum):
    """
    This enum class is used to store all possible Features needed
    or provided by a node in the computational graph
    """
    MODEL = auto()
    ORBITALS = auto()
    RDMS = auto()
    ACTIVESPACESOLVER = auto()
