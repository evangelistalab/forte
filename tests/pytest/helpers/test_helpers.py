#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_helpers():
    import math
    import pytest
    import psi4
    import forte
    import forte.utils

    psi4.core.clean()

    geom = """
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692    
    symmetry c2v
    """
    (E_scf, wfn_scf) = forte.utils.psi4_scf(geom,basis='6-31g',reference='rhf')
    assert math.isclose(E_scf, -75.98397447271522)

    (E_casscf, wfn_casscf) = forte.utils.psi4_casscf(geom,basis='6-31g',reference='rhf', restricted_docc=[2,0,0,1], active=[2,0,1,1])
    assert math.isclose(E_casscf, -75.9998515885993)

    res = forte.utils.prepare_forte_objects(wfn_casscf, {"RESTRICTED_DOCC": [2,0,0,1], "ACTIVE": [2,0,1,1]})
    mo_space_info = res["mo_space_info"]
    # mo_space_info.corr_absolute_mo("RESTRICTED_DOCC") == [0, 1, 9]
    assert mo_space_info.corr_absolute_mo("RESTRICTED_DOCC")[2] == 9
    
    psi4.core.clean()

if __name__ == "__main__":
    test_helpers()
