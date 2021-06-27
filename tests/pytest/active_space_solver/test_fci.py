"""Test the HF solver."""

import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_fci():
    """Test FCI on H2."""

    ref_energy = [-1.1083377195359851, -0.2591786932627466]

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    # create a molecular model
    input = solver_factory(molecule=xyz, basis='cc-pVDZ')

    # specify the electronic state and the active orbitals
    state = input.state(charge=0, multiplicity=1, sym='ag')
    mo_spaces = input.mo_spaces(active=[1, 0, 0, 0, 0, 1, 0, 0])

    # create a HF object
    hf = HF(input, state=state)
    # create a FCI object that grabs the MOs from the HF object (hf)
    fci = ActiveSpaceSolver(hf, type='FCI', states={state: 2}, mo_spaces=mo_spaces)
    # run the computation
    fci.run()
    # check the FCI energy
    assert fci.value('active space energy')[state] == pytest.approx(ref_energy, 1.0e-10)


if __name__ == "__main__":
    test_fci()
