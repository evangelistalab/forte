from enum import Enum, auto


class Feature(Enum):
    """
    This enum class is used to store all types of features
    needed or provided by a node in the computational graph
    """
    # the node can probide a model
    MODEL = auto()
    # the node can probide molecular orbitals
    ORBITALS = auto()
    # the node can probide reduced density matrices
    RDMS = auto()
    # the node can probide an active space solver
    ACTIVESPACESOLVER = auto()
