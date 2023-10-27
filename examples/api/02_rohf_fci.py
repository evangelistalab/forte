# examples/api/02_rohf_fci.py
"""Example of a FCI computation on the triplet state of methylene using ROHF orbitals"""

import numpy as np
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
        'forte__active_space_solver': 'fci',
        'forte__restricted_docc': [1, 0, 0, 0],
        'forte__active': [3, 0, 2, 2],
        'forte__multiplicity': 3,
        'forte__root_sym': 2,
    }
)

efci = psi4.energy('forte')
np.isclose(efci,-38.924726774489)

psi4.set_options({'forte__multiplicity': 1, 'forte__root_sym': 0, 'forte__nroot': 2})
efci = psi4.energy('forte')
np.isclose(efci,-38.866616413802)

psi4.set_options({'forte__multiplicity': 1, 'forte__root_sym': 0, 'forte__nroot': 2, 'forte__root': 1})
efci = psi4.energy('forte')
np.isclose(efci,-38.800424868719)
