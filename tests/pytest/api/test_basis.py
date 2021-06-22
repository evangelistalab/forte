"""Test the Basis class."""

import pytest

from forte import Basis

def test_basis():
    """Test basis creation."""

    # create a basis from a string
    basis = Basis('cc-pVDZ')

    assert repr(basis) == 'Basis(\'cc-pVDZ\')'
    assert str(basis) == 'cc-pVDZ'

    # test raising if no basis is provided
    with pytest.raises(TypeError):
        basis = Basis()

if __name__ == "__main__":
    test_basis()
