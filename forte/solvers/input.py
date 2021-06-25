from forte.solvers.solver import Feature, Solver


class InputSolver(Solver):
    """
    This solver class is used as a starting point of computations.

    When initialized, this solver does not contain any information.
    It is used by the function `solver_factory` which fills it with
    information about a model.
    """
    def __init__(self, options=None, cbh=None):
        super().__init__(input=None, needs=[], provides=[Feature.MODEL], options=options, cbh=cbh)

    def _run(self):
        pass
