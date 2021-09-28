#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_duccsd():
    """Test projective factorized UCCSDT on Ne using RHF/cc-pVDZ orbitals"""

    import pytest
    import forte.proc.scc as scc
    import forte
    import psi4

    ref_energy = -128.677997285129

    geom = "Ne"

    scf_energy, psi4_wfn = forte.utils.psi4_scf(geom, basis='cc-pVDZ', reference='RHF')
    forte_objs = forte.utils.prepare_forte_objects(psi4_wfn, mo_spaces={'FROZEN_DOCC': [1, 0, 0, 0, 0, 0, 0, 0]})
    calc_data = scc.run_cc(
        forte_objs[1], forte_objs[2], forte_objs[3], cc_type='ducc', max_exc=2, e_convergence=1.0e-10
    )

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f'  HF energy:     {scf_energy}')
    print(f'  DUCCSD energy: {energy}')
    print(f'  E - Eref:      {energy - ref_energy}')

    assert energy == pytest.approx(ref_energy, 1.0e-11)


if __name__ == "__main__":
    test_duccsd()
