#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_sparse_operator2():
    """Test the SparseHamiltonian class"""

    import pytest
    import forte
    import forte.utils
    import psi4
    from forte import det

    geom = """
     H
     H 1 1.0
    """
    scf_energy, psi4_wfn = forte.utils.psi4_scf(geom, basis='DZ', reference='RHF')
    forte_objs = forte.utils.prepare_forte_objects(psi4_wfn, mo_spaces={})

    as_ints = forte_objs[1]

    ham_op = forte.SparseHamiltonian(as_ints)

    ref = forte.StateVector({det("20"): 1.0})
    Href1 = ham_op.compute(ref, 0.0)
    Href2 = ham_op.compute_on_the_fly(ref, 0.0)
    assert Href1[det("20")] == pytest.approx(-1.094572, abs=1e-6)
    assert Href2[det("20")] == pytest.approx(-1.094572, abs=1e-6)

    psi4.core.clean()


if __name__ == "__main__":
    test_sparse_operator2()
