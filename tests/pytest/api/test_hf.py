"""Test the HF solver."""

import pytest

from forte import Molecule, Basis, MolecularModel
from forte.solvers import hf_solver


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
    model = MolecularModel(molecule=mol, basis=basis)

    hf = hf_solver(model)

    results = hf.energy()

    assert results.value('hf energy') == pytest.approx(ref_energy, 1.0e-10)


if __name__ == "__main__":
    test_hf()
