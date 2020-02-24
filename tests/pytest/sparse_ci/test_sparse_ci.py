#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_sparse_ci():
    import math
    import psi4
    import forte
    import itertools
    import numpy as np
    import pytest
    from forte import forte_options

    ref_fci = -1.101150330132956

    psi4.core.clean()

    h2o = psi4.geometry("""
     H
     H 1 1.0
    """)

    psi4.set_options({'basis': 'sto-3g'})
    E_scf, wfn = psi4.energy('scf', return_wfn=True)
    na = wfn.nalpha()
    nb = wfn.nbeta()
    nirrep = wfn.nirrep()
    wfn_symmetry = 0

    forte.startup()
    forte.banner()

    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')
    forte_options.get_options_from_psi4(psi4_options)

    # Setup forte and prepare the active space integral class
    mo_space_info = forte.make_mo_space_info(wfn, forte_options)
    ints = forte.make_forte_integrals(wfn, forte_options, mo_space_info)
    as_ints = forte.make_active_space_ints(mo_space_info, ints, 'ACTIVE', ['RESTRICTED_DOCC'])
    as_ints.print()

    print('\n\n  => Sparse FCI Test <=')
    print('  Number of irreps: {}'.format(nirrep))
    nmo = wfn.nmo()
    nmopi = [wfn.nmopi()[h] for h in range(nirrep)]
    nmopi_str = [str(wfn.nmopi()[h]) for h in range(nirrep)]
    mo_sym = []
    for h in range(nirrep):
        for i in range(nmopi[h]):
            mo_sym.append(h)

    print('  Number of orbitals per irreps: [{}]'.format(','.join(nmopi_str)))
    print('  Symmetry of the MOs: ',mo_sym)

    hf_reference = forte.Determinant()
    hf_reference.create_alfa_bit(0)
    hf_reference.create_beta_bit(0)
    print('  Hartree-Fock determinant: {}'.format(hf_reference.str(2)))

    # Compute the HF energy
    hf_energy = as_ints.nuclear_repulsion_energy() + as_ints.slater_rules(hf_reference,hf_reference)
    print('  Nuclear repulsion energy: {}'.format(as_ints.nuclear_repulsion_energy()))
    print('  Reference energy: {}'.format(hf_energy))

    # Build a list of determinants
    orblist = [i for i in range(nmo)]
    dets = []
    for astr in itertools.combinations(orblist,na):
        for bstr in itertools.combinations(orblist,nb):
            sym = 0
            d = forte.Determinant()
            for a in astr:
                d.create_alfa_bit(a)
                sym = sym ^ mo_sym[a]
            for b in bstr:
                d.create_beta_bit(b)
                sym = sym ^ mo_sym[b]
            if (sym == wfn_symmetry):
                dets.append(d)
                print('  Determinant {} has symmetry {}'.format(d.str(nmo),sym))

    # Build the Hamiltonian matrix using 'slater_rules'
    nfci = len(dets)
    H = np.ndarray((nfci,nfci))
    for I in range(nfci):
        # off-diagonal terms
        for J in range(I + 1, nfci):
            HIJ = as_ints.slater_rules(dets[I],dets[J])
            H[I][J] = H[J][I] = HIJ
        # diagonal term
        H[I][I] = as_ints.nuclear_repulsion_energy() + as_ints.slater_rules(dets[I],dets[I])

    # Find the lowest eigenvalue
    efci = np.linalg.eigh(H)[0][0]

    print('\n  FCI Energy: {}\n'.format(efci))

    assert efci == pytest.approx(ref_fci, 1.0e-9)

    # Clean up forte (necessary)
    forte.cleanup()

test_sparse_ci()
