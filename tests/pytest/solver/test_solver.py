"""Test the HF solver."""

import pytest

from forte.solvers import solver_factory, HF, SpinAnalysis


def test_solver():
    """Test RHF on H2."""

    ref_energy = -1.10015376479352

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    # create a molecular model
    input = solver_factory(molecule=xyz, basis='cc-pVDZ')

    # create a HF object and run
    with pytest.raises(AssertionError):
        spin = SpinAnalysis(input)


if __name__ == "__main__":
    test_solver()
