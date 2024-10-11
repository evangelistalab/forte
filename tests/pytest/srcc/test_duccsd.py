#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_duccsd():
    """Test projective factorized UCCSDT on Ne using RHF/cc-pVDZ orbitals"""

    import pytest
    import forte.proc.scc as scc
    import forte
    import psi4

    ref_energy = -128.677997285129
    molecule = psi4.geometry(
        """
    Ne 0.0 0.0 0.0"""
    )

    data = forte.modules.ObjectsUtilPsi4(
        molecule=molecule, basis="cc-pVDZ", mo_spaces={"FROZEN_DOCC": [1, 0, 0, 0, 0, 0, 0, 0]}
    ).run()

    scf_energy = data.psi_wfn.energy()

    calc_data = scc.run_cc(
        data.as_ints,
        data.scf_info,
        data.mo_space_info,
        cc_type="ducc",
        max_exc=2,
        e_convergence=1.0e-10,
    )

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f"  HF energy:     {scf_energy}")
    print(f"  DUCCSD energy: {energy}")
    print(f"  E - Eref:      {energy - ref_energy}")

    assert energy == pytest.approx(ref_energy, 1.0e-11)


if __name__ == "__main__":
    test_duccsd()
