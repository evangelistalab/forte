import pytest

from forte import Molecule, Basis
from forte.solvers import solver_factory, HF


def test_rohf():
    """Test ROHF computation of the B1 state of methylene."""

    ref_energy = -38.92107482288969

    # create a molecule from a string
    mol = Molecule.from_geom("""
    C
    H 1 1.085
    H 1 1.085 2 135.5
    """)

    # create a basis object
    basis = Basis('cc-pVDZ')

    # create a molecular model
    input = solver_factory(molecule=mol, basis=basis)

    # specify the electronic state
    state = input.state(charge=0, multiplicity=3, sym='b1')

    # define a HF object
    hf = HF(input, state=state)
    hf.run()

    assert hf.value('hf energy') == pytest.approx(ref_energy, 1.0e-10)


if __name__ == "__main__":
    test_rohf()
