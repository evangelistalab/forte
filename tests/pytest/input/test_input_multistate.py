import psi4
import forte
import pytest


def test_input_multistate():
    """Test input for a multi-state computation using the `AVG_STATE` keyword."""

    psi4.geometry("""
    Li
    H 1 3.0
    units bohr
    """)

    ref_energies = {
        '1A1': [-8.008550659909, -7.890309067618, -7.776162395969],
        '1A2': [-7.396156201690, -7.321587400358, -7.262899599968],
        '1B1': [-7.853436217153, -7.749808606728, -7.382069881110],
        '1B2': [-7.853436217170, -7.749808606704, -7.382069881071],
    }

    # need to clean the options otherwise this job will interfere
    forte.clean_options()

    psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12})
    psi4.set_module_options(
        'FORTE', {
            'active_space_solver': 'fci',
            'active': [8, 0, 2, 2],
            'restricted_docc': [0, 0, 0, 0],
            'avg_state': [[0, 1, 3], [1, 1, 3], [2, 1, 3], [3, 1, 3]],
            'ms': 0.0
        }
    )
    efci = psi4.energy('forte')
    for k, vals in ref_energies.items():
        for i in range(3):
            assert psi4.core.variable(f'ENERGY ROOT {i} {k}') == pytest.approx(vals[i], 1.0e-9)


if __name__ == "__main__":
    test_input_multistate()
