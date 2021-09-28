import pytest

from forte import Molecule, Basis
from forte.solvers import solver_factory, HF


def test_rhf():
    """Test RHF on H2."""

    ref_energy = -1.10015376479352

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
    hf.run()

    assert hf.value('hf energy') == pytest.approx(ref_energy, 1.0e-10)


def test_rhf_docc():
    """Test RHF on LiH using an occupation pattern that is not optimal.
    This example shows an alternative way to build the molecule/basis."""

    ref_energy = -7.40425598707951

    # create a molecule from a string
    mol = Molecule.from_geom("""
    Li 0.0 0.0 0.0
    H  0.0 0.0 1.6
    """)

    # create a basis object
    basis = Basis('cc-pVDZ')

    # create a molecular model
    input = solver_factory(molecule=mol, basis=basis)

    # specify the electronic state
    state = input.state(charge=0, multiplicity=1, sym='a1')

    hf = HF(input, state=state, docc=[1, 0, 1, 0])
    hf.run()

    assert hf.value('hf energy') == pytest.approx(ref_energy, 1.0e-10)


if __name__ == "__main__":
    test_rhf()
    test_rhf_docc()
