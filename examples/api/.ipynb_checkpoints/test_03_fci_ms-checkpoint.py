def test_03_fci_ms():
    return
    """Run a FCI computation on methylene using ROHF orbitals optimized for the 3B1 state.
       Computes both the lowest 3B1 and 1A1 states."""
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

    psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12, 'reference': 'rohf',
            'forte__job_type': 'mcscf_two_step',
            'forte__active_space_solver': 'fci',
            'forte__restricted_docc': [1, 0, 0, 0],
            'forte__active': [3, 0, 2, 2],
            'forte__avg_state': [[2, 3, 1], [0, 1, 1]]
        }
    )
    psi4.energy('forte')
    assert psi4.core.variable('ENERGY ROOT 0 3B1') == pytest.approx(ref_e_3b1, 1.0e-9)
    assert psi4.core.variable('ENERGY ROOT 0 1A1') == pytest.approx(ref_e_1a1, 1.0e-9)

#     forte.cleanup()
        
if __name__ == "__main__":
    test_03_fci_ms()

