import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver, SpinAnalysis


def test_solver_2():
    """Test RHF on H2."""

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    # create a molecular model
    input = solver_factory(molecule=xyz, basis='cc-pVDZ')

    # specify the electronic state
    state = input.state(charge=0, multiplicity=1, sym='ag')

    # create a HF object and run
    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, state, 'FCI')
    spin = SpinAnalysis(fci)
    test_graph = """SpinAnalysis
 |
ActiveSpaceSolver
 |
HF
 |
Input"""
    assert spin.computational_graph() == test_graph


if __name__ == "__main__":
    test_solver_2()
