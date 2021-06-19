"""Test the HF solver."""

import pytest

from forte.solvers import solver_factory, HF


def test_df_rhf():
    """Test DF-RHF on HF."""

    ref_energy = -100.04775218911111

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    # create a molecular model
    root = solver_factory(molecule=xyz, basis='cc-pVTZ')

    # specify the electronic state
    state = root.state(charge=0, multiplicity=1, sym='a1')

    # create a HF object and run
    hf = HF(root, state=state, int_type='df')
    hf.run()

    assert hf.value('hf energy') == pytest.approx(ref_energy, 1.0e-10)


def test_df_rhf_select_aux():
    """Test DF-RHF on HF."""

    ref_energy = -100.04775602524956

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    # create a molecular model
    root = solver_factory(molecule=xyz, basis='cc-pVTZ', scf_aux_basis='cc-pVQZ-JKFIT')

    # specify the electronic state
    state = root.state(charge=0, multiplicity=1, sym='a1')

    # create a HF object and run
    hf = HF(root, state=state, int_type='df')
    hf.run()

    assert hf.value('hf energy') == pytest.approx(ref_energy, 1.0e-10)


if __name__ == "__main__":
    test_df_rhf()
    test_df_rhf_select_aux()
