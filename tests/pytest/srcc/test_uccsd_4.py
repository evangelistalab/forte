#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_uccsd_4():
    """Test projective UCCSD on Ne using RHF/cc-pVDZ orbitals"""

    import pytest
    import forte.proc.scc as scc
    import forte
    import psi4

    ref_energy = -107.655681875111

    molecule = psi4.geometry(
        """
    N
    N 1 1.3
    symmetry c1
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="sto-3g", mo_spaces={"frozen_docc": [2]}).run()

    scf_energy = data.psi_wfn.energy()

    calc_data = scc.run_cc(
        data.as_ints, data.scf_info, data.mo_space_info, cc_type="ucc", max_exc=2, e_convergence=1.0e-10
    )

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f"  HF energy:    {scf_energy}")
    print(f"  CCSD energy:  {energy}")
    print(f"  corr. energy: {energy - scf_energy}")

    assert energy == pytest.approx(ref_energy, 1.0e-9)


if __name__ == "__main__":
    test_uccsd_4()
