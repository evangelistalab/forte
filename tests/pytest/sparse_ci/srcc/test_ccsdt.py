#!/usr/bin/env python
# -*- coding: utf-8 -*-


def nest_ccsdt():
    """Test CCSDT on H6 using RHF/sto-3g orbitals"""

    import pytest
    import forte.proc.scc as scc
    import forte
    import psi4

    geom = """
H 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 0.0 2.0
H 0.0 0.0 3.0
H 0.0 0.0 4.0
H 0.0 0.0 5.1
symmetry c1
"""

    scf_energy, psi4_wfn = forte.utils.psi4_scf(geom, basis='sto-3g', reference='RHF')
    forte_objs = forte.utils.prepare_forte_objects(psi4_wfn, mo_spaces={})
    calc_data = scc.run_cc(
        forte_objs['as_ints'], forte_objs['scf_info'], forte_objs['mo_space_info'], cc_type='cc', max_exc=3
    )

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f'  HF energy:     {scf_energy}')
    print(f'  CCSDT energy:  {energy}')

    assert energy == pytest.approx(ref_energy, 1.0e-11)


if __name__ == "__main__":
    nest_ccsdt()
