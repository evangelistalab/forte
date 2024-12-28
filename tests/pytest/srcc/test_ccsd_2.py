#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest


@pytest.mark.skip(reason="This is a long test")
def test_ccsd2():
    """Test CCSD on H2 using RHF/DZ orbitals"""

    import forte
    import psi4

    ref_energy = -76.237730204702288  # CCSD energy from psi4

    molecule = psi4.geometry(
        """
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ").run()
    scf_energy = data.psi_wfn.energy()
    calc_data = scc.run_cc(
        data.as_ints,
        data.scf_info,
        data.mo_space_info,
        cc_type="cc",
        max_exc=2,
        e_convergence=1.0e-6,
        r_convergence=1.0e-4,
        compute_threshold=1.0e-6,
    )

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f"  HF energy:   {scf_energy}")
    print(f"  CCSD energy: {energy}")
    print(f"  E - Eref:    {energy - ref_energy}")

    assert energy == pytest.approx(ref_energy, 1.0e-11)


if __name__ == "__main__":
    test_ccsd2()
