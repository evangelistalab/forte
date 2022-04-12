from forte import Basis
from forte import Molecule
from forte import MolecularModel


def test_molecular_model():
    """Test molecular model creation."""

    mol = Molecule.from_geom("""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """)

    basis = Basis('cc-pVDZ')
    mm = MolecularModel(basis=basis, molecule=mol)

    test_mm_str = """MolecularModel(
Molecule(2
0 1 H2
H                     0.000000000000     0.000000000000    -0.500000000000
H                     0.000000000000     0.000000000000     0.500000000000
),
Basis('cc-pVDZ'),
CONVENTIONAL)"""

    assert str(mm) == test_mm_str


if __name__ == "__main__":
    test_molecular_model()
