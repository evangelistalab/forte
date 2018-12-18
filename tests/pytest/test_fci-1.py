#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_fci1():
    import psi4
    import forte
    from forte import forte_options

    h2o = psi4.geometry("""
     O
     H 1 0.96
     H 1 0.96 2 104.5
    """)

    psi4.set_options({'basis': "sto-3g"})
    E_scf, wfn = psi4.energy('scf', return_wfn=True)
    state = forte.StateInfo(na=5, nb=5, multiplicity=1, twice_ms=0, irrep=0)
    dim = psi4.core.Dimension([4, 0, 1, 2])

    options = psi4.core.get_options()
    options.set_current_module('FORTE')

    forte.startup()
    forte.banner()
    mo_space_info = forte.make_mo_space_info(wfn, options)
    ints = forte.make_forte_integrals(wfn, options, mo_space_info)
    solver = forte.FCI(state,forte_options,ints,mo_space_info)
    energy = solver.compute_energy()
    print(energy)
    forte.cleanup()

test_fci1()
