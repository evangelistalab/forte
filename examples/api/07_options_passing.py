# examples/api/07_options_passing.py
"""Example of passing options as a dictionary in an energy call"""

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

efci = psi4.energy('forte',forte_options={'multiplicity': 1, 'root_sym': 0, 'nroot': 2})
np.isclose(efci,-38.866616413802)

efci = psi4.energy('forte',forte_options={'multiplicity': 1, 'root_sym': 0, 'nroot': 2, 'root' : 1})
np.isclose(efci,-38.800424868719)
