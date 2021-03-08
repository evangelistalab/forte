#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_input_example_2():
    """Run a FCISCF computation on methylene using ROHF orbitals optimized for the 3B1 state.
       Computes both the lowest 3B1 and 1A1 states."""
    import math
    import psi4
    import forte
    import pytest

    ref_e_1a1 = -38.900217662950
    ref_e_3b1 = -38.960623289646

    psi4.geometry("""
    0 3
    C
    H 1 1.085
    H 1 1.085 2 135.5
    """)

    psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12, 'reference': 'rohf'})
    psi4.set_module_options(
        'FORTE', {
            'job_type': 'mcscf_two_step',
            'active_space_solver': 'fci',
            'restricted_docc': [1, 0, 0, 0],
            'active': [3, 0, 2, 2],
            'avg_state': [[2, 3, 1], [0, 1, 1]]
        }
    )
    psi4.energy('forte')
    assert psi4.core.variable('ENERGY ROOT 0 3B1') == pytest.approx(ref_e_3b1, 1.0e-9)
    assert psi4.core.variable('ENERGY ROOT 0 1A1') == pytest.approx(ref_e_1a1, 1.0e-9)


if __name__ == "__main__":
    test_input_example_2()
