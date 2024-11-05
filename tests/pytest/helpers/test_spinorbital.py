#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_spinorbital_mp2():
    import math
    import pytest
    import psi4
    import forte
    import forte.utils
    from math import isclose

    import numpy as np

    psi4.core.clean()

    molecule = psi4.geometry(
        """0 1
    H
    F 1 1.0
    symmetry c1
    """
    )

    data = forte.modules.ObjectsUtilPsi4(
        molecule=molecule,
        basis="6-31G",
        mo_spaces={"RESTRICTED_DOCC": [5], "ACTIVE": [0]},
        options={"E_CONVERGENCE": 1e-12},
    ).run()

    mo_space_info = data.mo_space_info
    core = mo_space_info.corr_absolute_mo("RESTRICTED_DOCC")
    virt = mo_space_info.corr_absolute_mo("RESTRICTED_UOCC")

    ints = data.ints
    H = {
        "cc": forte.spinorbital_oei(ints, core, core),
        "cccc": forte.spinorbital_tei(ints, core, core, core, core),
        "ccvv": forte.spinorbital_tei(ints, core, core, virt, virt),
    }

    Eref_test = ints.nuclear_repulsion_energy()
    Eref_test += np.einsum("mm->", H["cc"])
    Eref_test += 0.5 * np.einsum("mnmn->", H["cccc"])
    assert math.isclose(Eref_test, -99.97763667846159)

    Fc = forte.spinorbital_fock(ints, core, core, core).diagonal()
    Fv = forte.spinorbital_fock(ints, virt, virt, core).diagonal()
    ncoreso = 2 * len(core)
    nvirtso = 2 * len(virt)
    d = np.zeros((ncoreso, ncoreso, nvirtso, nvirtso))
    for i in range(ncoreso):
        for j in range(ncoreso):
            for a in range(nvirtso):
                for b in range(nvirtso):
                    d[i][j][a][b] = 1.0 / (Fc[i] + Fc[j] - Fv[a] - Fv[b])
    T = {"ccvv": np.einsum("ijab,ijab->ijab", d, H["ccvv"])}
    E = 0.25 * np.einsum("ijab,ijab->", T["ccvv"], H["ccvv"])

    assert math.isclose(E, -0.13305567213152, abs_tol=1e-09)

    psi4.core.clean()


if __name__ == "__main__":
    test_spinorbital_mp2()
