"""Test the HF solver."""

import pytest

from forte import Molecule, Basis
from forte.solvers import molecular_model, HF


def test_uhf():
    """Test UHF computation of the B1 state of methylene."""

    ref_energy = -38.92655952236967

    # create a molecule from a string
    mol = Molecule.from_geom("""
    C
    H 1 1.085
    H 1 1.085 2 135.5
    """)

    # create a basis object
    basis = Basis('cc-pVDZ')

    # create a molecular model
    root = molecular_model(molecule=mol, basis=basis)

    # specify the electronic state
    state = root.data.model.state(charge=0, multiplicity=3, sym='b1')

    # define a HF object
    hf = HF(root, state=state, restricted=False)
    hf.run()

    assert hf.value('hf energy') == pytest.approx(ref_energy, 1.0e-10)


def test_uhf_wrong_sym():
    """Test UHF computation of the B1 state of methylene."""

    # create a molecule from a string
    mol = Molecule.from_geom("""
    C
    H 1 1.085
    H 1 1.085 2 135.5
    """)

    # create a basis object
    basis = Basis('cc-pVDZ')

    # create a molecular model
    root = molecular_model(molecule=mol, basis=basis)

    # specify the electronic state
    state = root.data.model.state(charge=0, multiplicity=3, sym='a1')

    # define a HF object
    hf = HF(root, state=state, docc=[2, 0, 0, 1], socc=[1, 0, 1, 0], restricted=False)

    # should raise an error since we get a state with different symmetry
    with pytest.raises(RuntimeError):
        hf.run()


if __name__ == "__main__":
    test_uhf()
    test_uhf_wrong_sym()