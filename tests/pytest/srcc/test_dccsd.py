#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_dccsd():
    """Test CCSD on H2 using RHF/DZ orbitals"""

    import pytest

    import forte
    import psi4

    ref_energy = -1.126712715716011  # CCSD = FCI energy from psi4

    molecule = psi4.geometry(
        """
    H
    H 1 1.0
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ").run()
    scf_energy = data.psi_wfn.energy()
    cc = forte.modules.GeneralCC(cc_type="dcc", max_exc=2, e_convergence=1.0e-11)
    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("energy")

    print(f"  HF energy:   {scf_energy}")
    print(f"  CCSD energy: {energy}")
    print(f"  E - Eref:    {energy - ref_energy}")

    assert energy == pytest.approx(ref_energy, 1.0e-11)


if __name__ == "__main__":
    test_dccsd()
