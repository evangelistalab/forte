from forte.solvers.solver import Feature, Node
from forte.data import Data


class Input(Node):
    """
    This solver class is used as a starting point of computations.

    When initialized, the resulting object does not contain any information.
    It is used by the function `solver_factory` which fills it with
    an object from a class derived from the Model class.
    """
    def __init__(self):
        super().__init__(input_nodes=None, needs=[], provides=[Feature.MODEL])
        self._data = Data()

    def _run(self):
        pass

    def state(self, *args, **kwargs):
        return self.data.model.state(*args, **kwargs)

    def mo_spaces(
        self,
        frozen_docc=None,
        restricted_docc=None,
        active=None,
        restricted_uocc=None,
        frozen_uocc=None,
        gas1=None,
        gas2=None,
        gas3=None,
        gas4=None,
        gas5=None,
        gas6=None
    ):
        """
        Returns a dictionary with the size of each orbital space defined.

        This function acts mainly as an interface. This is a good place
        for future implementation of error checking.

        active: list(int)
            The number of active MOs per irrep
        restricted_docc: list(int)
            The number of restricted doubly occupied MOs per irrep
        frozen_docc: list(int)
            The number of frozen doubly occupied MOs per irrep
        """
        mo_space = {}
        if frozen_docc is not None:
            mo_space['FROZEN_DOCC'] = frozen_docc
        if restricted_docc is not None:
            mo_space['RESTRICTED_DOCC'] = restricted_docc
        if active is not None:
            mo_space['ACTIVE'] = active
        if gas1 is not None:
            mo_space['GAS1'] = gas1
        if gas2 is not None:
            mo_space['GAS2'] = gas2
        if gas3 is not None:
            mo_space['GAS3'] = gas3
        if gas4 is not None:
            mo_space['GAS4'] = gas4
        if gas5 is not None:
            mo_space['GAS5'] = gas5
        if gas6 is not None:
            mo_space['GAS6'] = gas6
        if restricted_uocc is not None:
            mo_space['RESTRICTED_UOCC'] = restricted_uocc
        if frozen_uocc is not None:
            mo_space['FROZEN_UOCC'] = frozen_uocc
        return mo_space
