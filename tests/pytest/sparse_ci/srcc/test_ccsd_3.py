#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_ccsd_3():
    """Test CCSD on H4 using RHF/DZ orbitals"""

    import pytest
    import forte.proc.scc as scc
    import forte
    import psi4

    ref_energy = -2.225059801642  # from psi4

    geom = """
     H 0.0 0.0 0.0
     H 0.0 0.0 1.0
     H 0.0 0.0 2.0
     H 0.0 0.0 3.0     
    """

    scf_energy, psi4_wfn = forte.utils.psi4_scf(geom, basis="DZ", reference="RHF")
    data = forte.modules.ObjectsUtilPsi4(ref_wnf=psi4_wfn).run()
    calc_data = scc.run_cc(data.as_ints, data.scf_info, data.mo_space_info, cc_type="cc", max_exc=2)

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f"  HF energy:   {scf_energy}")
    print(f"  CCSD energy: {energy}")
    print(f"  E - Eref:    {energy - ref_energy}")

    assert energy == pytest.approx(ref_energy, 1.0e-11)


if __name__ == "__main__":
    test_ccsd_3()
