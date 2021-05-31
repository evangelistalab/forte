from forte import HF, Computation


def hf_solver(model):
    """Run a Hartree-Fock computation."""

    # create a HF method object
    hf = HF()

    # assemble a computation
    return Computation(model=model, method=hf)
