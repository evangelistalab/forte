from forte.solvers.solver import Feature, Solver


class Input(Solver):
    """
    This solver class is used as a starting point of computations.

    When initialized, the resulting object does not contain any information.
    It is used by the function `solver_factory` which fills it with
    an object from a class derived from the Model class.
    """
    def __init__(self, options=None, cbh=None):
        super().__init__(input=None, needs=[], provides=[Feature.MODEL], options=options, cbh=cbh)

    def _run(self):
        pass
