#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_fci1():
    import psi4
#    import forte

    h2o = psi4.geometry("""
     O
     H 1 0.96
     H 1 0.96 2 104.5
    """)

    psi4.set_options({'basis': "sto-3g"})
    E_scf, wfn = psi4.energy('scf', return_wfn=True)
    state = forte.StateInfo(na=5, nb=5, multiplicity=1, twice_ms=0, irrep=0)
    dim = psi4.core.Dimension([4, 0, 1, 2])

    forte.startup()
    forte.banner()
    options = psi4.core.get_options()
    options.set_current_module('FORTE')
    mo_space_info = forte.make_mo_space_info(wfn, options)
    ints = forte.make_forte_integrals(wfn, options, mo_space_info)
    solver = forte.FCISolver(dim, [], list(range(7)), state, ints,
                             mo_space_info, 10, 1, options)
    energy = solver.compute_energy()
    forte.cleanup()
