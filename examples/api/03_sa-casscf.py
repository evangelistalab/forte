# examples/api/03_ms-casscf.py
"""State-averaged CASSCF computation on the triplet B1 state and singlet A1 electronic states of methylene"""

import psi4
import forte

psi4.geometry("""
0 3
C
H 1 1.085
H 1 1.085 2 135.5
""")

psi4.set_options(
    {
        'basis': 'DZ',
        'scf_type': 'pk',
        'e_convergence': 12,
        'reference': 'rohf',
        'forte__job_type': 'mcscf_two_step',
        'forte__active_space_solver': 'fci',
        'forte__restricted_docc': [1, 0, 0, 0],
        'forte__active': [3, 0, 2, 2],
        'forte__avg_state': [[2, 3, 1], [0, 1, 1]]
        # [(B1, triplet, 1 state), (A1,singlet,1 state)]
    }
)

psi4.energy('forte')
