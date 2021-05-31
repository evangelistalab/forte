from forte import MCSCF, Computation


def mcscf_solver(model):
    """Run a MCSCF computation."""

    # create a HF method object
    mcscf = MCSCF()

    # assemble a computation
    return Computation(model=model, method=mcscf)
