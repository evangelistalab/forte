#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest


@pytest.mark.skip(reason="This is a long test")
def test_ccsd2():
    """Test CCSD on H2 using RHF/DZ orbitals"""

    import forte.proc.scc as scc
    import forte
    import psi4

    ref_energy = -76.237730204702288  # CCSD energy from psi4

    geom = """
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """

    scf_energy, psi4_wfn = forte.utils.psi4_scf(geom, basis='cc-pVDZ', reference='RHF')
    forte_objs = forte.utils.prepare_forte_objects(psi4_wfn, mo_spaces={})
    calc_data = scc.run_cc(
        forte_objs[1],
        forte_objs[2],
        forte_objs[3],
        cc_type='cc',
        max_exc=2,
        e_convergence=1.0e-6,
        r_convergence=1.0e-4,
        compute_threshold=1.0e-6,
        on_the_fly=True
    )

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f'  HF energy:   {scf_energy}')
    print(f'  CCSD energy: {energy}')
    print(f'  E - Eref:    {energy - ref_energy}')

    #assert energy == pytest.approx(ref_energy, 1.0e-11)


if __name__ == "__main__":
    test_ccsd2()
