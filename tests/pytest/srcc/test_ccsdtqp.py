#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


@pytest.mark.skip(reason="This is a long test")
def test_ccsdtqp():
    """Test CCSDTQP on Ne using RHF/cc-pVDZ orbitals"""

    import forte.proc.scc as scc
    import forte
    import psi4

    ref_energy = -128.679025538  # from Evangelista, J. Chem. Phys. 134, 224102 (2011).

    molecule = psi4.geometry(
        """
     Ne 0.0 0.0 0.0"""
    )

    data = forte.modules.ObjectsUtilPsi4(
        molecule=molecule, basis="cc-pVDZ", mo_spaces={"FROZEN_DOCC": [1, 0, 0, 0, 0, 0, 0, 0]}
    ).run()
    cc = forte.modules.generalCC(cc_type="cc", max_exc=5, e_convergence=1.0e-10)
    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("energy")

    print(energy - ref_energy)
    assert energy == pytest.approx(ref_energy, 1.0e-9)


if __name__ == "__main__":
    test_ccsdtqp()
