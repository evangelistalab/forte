#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_uccsd_2():
    """Test projective UCCSD on Ne using RHF/cc-pVDZ orbitals"""
    
    import pytest    
    import scc
    import forte
    import psi4

    forte.startup()

    ref_energy = -128.677999887 # from Evangelista, J. Chem. Phys. 134, 224102 (2011).

    geom = "Ne"
    
    scf_energy, psi4_wfn = forte.utils.psi4_scf(geom, basis='cc-pVDZ', reference='RHF')
    forte_objs = scc.make_forte_objs(psi4_wfn,mo_spaces={'FROZEN_DOCC' :[1,0,0,0,0,0,0,0]})
    calc_data = scc.run_cc(forte_objs,psi4_wfn,cc_type='ucc',max_exc=2,e_convergence = 1.0e-10)

    forte.cleanup()
    psi4.core.clean()
    
    energy = calc_data[-1][1]

    print(f'  HF energy:    {scf_energy}')
    print(f'  UCCSD energy: {energy}')
    print(f'  E - Eref:     {energy - ref_energy}')    

    assert energy == pytest.approx(ref_energy, 1.0e-11)
    
if __name__ == "__main__":
    test_uccsd_2()