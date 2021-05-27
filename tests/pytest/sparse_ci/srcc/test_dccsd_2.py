#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_dccsd_2():
    """Test CCSD on H4 using RHF/DZ orbitals"""

    import pytest
    import forte.proc.scc as scc
    import forte
    import psi4

    forte.startup()

    ref_energy = -2.225059801642  # from psi4

    geom = """
     H 0.0 0.0 0.0
     H 0.0 0.0 1.0
     H 0.0 0.0 2.0
     H 0.0 0.0 3.0     
    """

    scf_energy, psi4_wfn = forte.utils.psi4_scf(geom, basis='DZ', reference='RHF')
    forte_objs = forte.utils.prepare_forte_objects(psi4_wfn, mo_spaces={})
    calc_data = scc.run_cc(forte_objs[1], forte_objs[2], forte_objs[3], cc_type='dcc', max_exc=2, on_the_fly=True)

    forte.cleanup()
    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f'  HF energy:   {scf_energy}')
    print(f'  CCSD energy: {energy}')
    print(f'  E - Eref:    {energy - ref_energy}')

    assert energy == pytest.approx(ref_energy, 1.0e-11)


if __name__ == "__main__":
    test_dccsd_2()
