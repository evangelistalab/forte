"""Test the HF solver."""

import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_fci():
    """Test FCI on H2."""

    ref_energy = -1.108337719536

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    # create a molecular model
    root = solver_factory(molecule=xyz, basis='cc-pVDZ')

    # specify the electronic state
    state = root.state(charge=0, multiplicity=1, sym='ag')

    # create a HF object and run
    hf = HF(root, state=state)
    fci = ActiveSpaceSolver(type='FCI', mos=hf, states=state, active=[1, 0, 0, 0, 0, 1, 0, 0])
    fci.run()
    assert list(fci.value('active space energy').items())[0][1][0] == pytest.approx(ref_energy, 1.0e-10)


if __name__ == "__main__":
    test_fci()
