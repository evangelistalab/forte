#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_input_example_1():
    """Run a FCI computation on methylene using ROHF orbitals optimized for the 3B1 state.
       Computes the lowest 3B1 state and the lowest two 1A1 states."""
    import math
    import psi4
    import forte
    import pytest

    ref_e_3b1 = -38.924726774489
    ref_e_1a1 = -38.866616413802
    ref_e_1a1_ex = -38.800424868719

    psi4.geometry("""
    0 3
    C
    H 1 1.085
    H 1 1.085 2 135.5
    """)

    psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12, 'reference': 'rohf'})
    psi4.set_module_options('FORTE', {
        'active_space_solver': 'fci',
        'restricted_docc': [1, 0, 0, 0],
        'active': [3, 0, 2, 2],        
        'multiplicity': 3,
        'root_sym': 2,
    })
    efci = psi4.energy('forte')
    assert efci == pytest.approx(ref_e_3b1, 1.0e-9)

    psi4.set_module_options('FORTE', {
        'active_space_solver': 'fci',
        'restricted_docc': [1, 0, 0, 0],
        'active': [3, 0, 2, 2],        
        'multiplicity': 1,
        'root_sym': 0,
        'nroot' : 2
    })
    efci = psi4.energy('forte')
    assert efci == pytest.approx(ref_e_1a1, 1.0e-9)

    psi4.set_module_options('FORTE', {
        'active_space_solver': 'fci',
        'restricted_docc': [1, 0, 0, 0],
        'active': [3, 0, 2, 2],        
        'multiplicity': 1,
        'root_sym': 0,
        'nroot' : 2,
        'root' : 1
    })
    efci = psi4.energy('forte')
    assert efci == pytest.approx(ref_e_1a1_ex, 1.0e-9)    

if __name__ == "__main__":
    test_input_example_1()
