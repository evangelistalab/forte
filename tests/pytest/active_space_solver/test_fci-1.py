"""Test the HF solver."""

import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_fci_1():
    """Test FCI on Li2/STO-3G. Reproduces the test fci-1"""

    ref_hf_energy = -14.54873910108353
    ref_fci_energy = -14.595808852754054

    xyz = """
    Li
    Li 1 3.0
    units bohr
    """

    # create a molecular model
    root = solver_factory(molecule=xyz, basis='sto-3g')

    # specify the electronic state
    state = root.state(charge=0, multiplicity=1, sym='ag')

    # create a HF object and run
    hf = HF(root, state=state)
    fci = ActiveSpaceSolver(type='FCI', mos=hf, states=state)
    fci.run()
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert list(fci.value('active space energy').items())[0][1][0] == pytest.approx(ref_fci_energy, 1.0e-10)


if __name__ == "__main__":
    test_fci_1()
