#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_uccsd_6():
    """Test projective linearized unliked UCCSD = CISD on H4 using RHF/STO-3G orbitals"""

    import pytest
    import forte.proc.scc as scc
    import forte
    import psi4

    ref_energy = -1.981824023630  # from CISD

    molecule = psi4.geometry(
        """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    H 0.0 0.0 3.0
    H 0.0 0.0 4.5     
    """
    )


    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="sto-3g").run()
    scf_energy = data.psi_wfn.energy()

    cc = forte.modules.GeneralCC(cc_type="ucc", max_exc=2, e_convergence=1.0e-10, linked = False, maxk=1)
    data = cc.run()

    psi4.core.clean()

    energy = data.results.value("energy")

    print(f"  HF energy:    {scf_energy}")
    print(f"  CCSD energy:  {energy}")
    print(f"  corr. energy: {energy - scf_energy}")

    assert energy == pytest.approx(ref_energy, 1.0e-9)


if __name__ == "__main__":
    test_uccsd_6()
