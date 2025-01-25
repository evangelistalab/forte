#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_sparse_operator2():
    """Test the SparseHamiltonian class"""

    import pytest
    import forte
    import forte.utils
    import psi4
    from forte import det

    molecule = psi4.geometry(
        """
     H
     H 1 1.0
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ").run()

    as_ints = data.as_ints  # forte_objs["as_ints"]

    ham = forte.sparse_operator_hamiltonian(as_ints)

    ref = forte.SparseState({det("20"): 1.0})
    Href1 = ham @ ref
    assert Href1[det("20")] == pytest.approx(-1.094572, abs=1e-6)

    psi4.core.clean()


if __name__ == "__main__":
    test_sparse_operator2()
