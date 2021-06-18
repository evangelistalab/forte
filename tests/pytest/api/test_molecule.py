"""Test the Molecule class."""

import os.path
from pathlib import Path
import pytest
from forte import Molecule


def test_molecule_from_geom():
    """Test molecule creation."""

    # create a molecule from a string
    mol = Molecule.from_geom("""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """)

    # get the nuclear repulsion energy
    nuclear_repulsion_energy = mol.molecule.nuclear_repulsion_energy()

    # verify the nuclear repulsion energy
    assert nuclear_repulsion_energy == pytest.approx(0.5291772106699999, 1.0e-12)


def test_molecule_from_geom_file():
    """Test molecule creation from a file."""

    # create a molecule from a file (in 'data/geom.xyz')
    mol = Molecule.from_geom_file('geom.xyz', Path(os.path.dirname(__file__)) / 'data')

    # get the nuclear repulsion energy
    nuclear_repulsion_energy = mol.molecule.nuclear_repulsion_energy()

    # verify the nuclear repulsion energy
    assert nuclear_repulsion_energy == pytest.approx(1.5875316320100001, 1.0e-12)


def test_str():
    """Test molecule creation from a file."""

    # create a molecule from a string
    mol = Molecule.from_geom("""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """)

    # get the string representation
    mol_str = str(mol)
    test_str = '''Molecule(2
0 1 H2
H                     0.000000000000     0.000000000000    -0.500000000000
H                     0.000000000000     0.000000000000     0.500000000000
)'''

    # verify the string representation
    assert mol_str == test_str


if __name__ == "__main__":
    test_molecule_from_geom()
    test_molecule_from_geom_file()
    test_str()
