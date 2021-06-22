#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_input_simple():
    """Test input for a simple computation using the `ROOT_SYM` keyword."""
    import forte
    import psi4
    import pytest

    psi4.geometry("""
    Li
    H 1 3.0
    units bohr
    """)

    ref_efci_0 = -8.008550659909
    ref_efci_1 = -7.396156201698
    ref_efci_2 = -7.853436217184
    ref_efci_3 = -7.853436217157

    # need to clean the options otherwise this job will interfere
    forte.clean_options()

    psi4.set_options(
        {
            'basis': 'DZ',
            'scf_type': 'pk',
            'e_convergence': 12,
            'forte__active_space_solver': 'fci',
            'forte__active': [8, 0, 2, 2],
            'forte__restricted_docc': [0, 0, 0, 0],
            'forte__root_sym': 0,
            'forte__multiplicity': 1,
            'forte__ms': 0.0
        }
    )
    psi4.core.set_output_file('debug.txt', False)
    efci = psi4.energy('forte')
    print('Test 1')
    assert efci == pytest.approx(ref_efci_0, 1.0e-9)

    print('Test 2')
    psi4.set_options({'forte__root_sym': 1})
    efci = psi4.energy('forte')
    assert efci == pytest.approx(ref_efci_1, 1.0e-9)

    psi4.set_options({'forte__root_sym': 2})
    efci = psi4.energy('forte')
    assert efci == pytest.approx(ref_efci_2, 1.0e-9)

    psi4.set_options({'forte__root_sym': 3})
    efci = psi4.energy('forte')
    assert efci == pytest.approx(ref_efci_3, 1.0e-9)


if __name__ == "__main__":
    test_input_simple()
