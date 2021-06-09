"""Test the HF solver."""

import pytest

from forte import Molecule, Basis, MolecularModel
from forte.solvers import HF, molecular_model


def test_hf():
    """Test RHF."""

    ref_energy = -1.10015376479352

    # create a molecule from a string
    mol = Molecule.from_geom("""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """)

    # create a basis object
    basis = Basis('cc-pVDZ')

    # create a molecular model
    root = molecular_model(molecule=mol, basis=basis)

    # specify the electronic state
    state = root.data.model.state(charge=0, multiplicity=1, sym='ag')

    hf = HF(root, state=state)
    hf.run()

    assert hf.value('hf energy') == pytest.approx(ref_energy, 1.0e-10)


if __name__ == "__main__":
    test_hf()
