#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_fci1():
    import math
    import psi4
    import forte
    from forte import forte_options

    ref_fci = -75.01315470154653
    rel_tol = 1e-9
    abs_tol = 1e-8

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
    forte_options = forte.ForteOptions(options)
    solver = forte.make_active_space_solver('FCI',state,forte_options,ints,mo_space_info)
    energy = solver.compute_energy()

    assert math.isclose(energy,ref_fci,abs_tol=abs_tol, rel_tol=rel_tol)

    print("\n\nFCI Energy = {}".format(energy))
    forte.cleanup()

test_fci1()
