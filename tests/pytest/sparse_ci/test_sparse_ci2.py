#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_sparse_ci2():
    import math
    import psi4
    import forte
    import itertools
    import numpy as np
    import pytest
    from forte import forte_options

    ref_fci = -5.623851783330647

    psi4.core.clean()
    # need to clean the options otherwise this job will interfere
    forte.clean_options()

    h2o = psi4.geometry("""
     He
     He 1 1.0
    """)

    psi4.set_options({'basis': 'cc-pVDZ'})
    _, wfn = psi4.energy('scf', return_wfn=True)
    na = wfn.nalpha()
    nb = wfn.nbeta()
    nirrep = wfn.nirrep()
    wfn_symmetry = 0

    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')
    forte_options.get_options_from_psi4(psi4_options)

    # Setup forte and prepare the active space integral class
    nmopi = wfn.nmopi()
    point_group = wfn.molecule().point_group().symbol()
    mo_space_info = forte.make_mo_space_info(nmopi, point_group, forte_options)
    ints = forte.make_ints_from_psi4(wfn, forte_options, mo_space_info)
    as_ints = forte.make_active_space_ints(mo_space_info, ints, 'ACTIVE', ['RESTRICTED_DOCC'])

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
    print('  Symmetry of the MOs: ', mo_sym)

    hf_reference = forte.Determinant()
    hf_reference.create_alfa_bit(0)
    hf_reference.create_beta_bit(0)
    print('  Hartree-Fock determinant: {}'.format(hf_reference.str(10)))

    # Compute the HF energy
    hf_energy = as_ints.nuclear_repulsion_energy() + as_ints.slater_rules(hf_reference, hf_reference)
    print('  Nuclear repulsion energy: {}'.format(as_ints.nuclear_repulsion_energy()))
    print('  Reference energy: {}'.format(hf_energy))

    # Build a list of determinants
    orblist = [i for i in range(nmo)]
    dets = []
    for astr in itertools.combinations(orblist, na):
        for bstr in itertools.combinations(orblist, nb):
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
                print('  Determinant {} has symmetry {}'.format(d.str(nmo), sym))

    print(f'\n  Size of the derminant basis: {len(dets)}')

    energy, evals, evecs, spin = forte.diag(dets, as_ints, 1, 1, "FULL")

    print(energy)

    efci = energy[0] + as_ints.nuclear_repulsion_energy()

    print('\n  FCI Energy: {}\n'.format(efci))

    assert efci == pytest.approx(ref_fci, abs=1e-9)


if __name__ == "__main__":
    test_sparse_ci2()
