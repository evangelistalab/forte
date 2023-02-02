# examples/api/07_options_passing.py
"""Example of passing options as a dictionary in an energy call"""

import psi4
import forte

psi4.geometry("""
0 3
C
H 1 1.085
H 1 1.085 2 135.5
""")

psi4.set_options({
    'basis': 'DZ',
    'scf_type': 'pk',
    'e_convergence': 12,
    'reference': 'rohf',
})

forte_options = {
    'active_space_solver': 'fci',
    'restricted_docc': [1, 0, 0, 0],
    'active': [3, 0, 2, 2],
    'multiplicity': 3,
    'root_sym': 2,
}

efci1 = psi4.energy('forte', forte_options=forte_options)

forte_options['multiplicity'] = 1
forte_options['root_sym'] = 0
forte_options['nroot'] = 2
forte_options['root'] = 1

efci2 = psi4.energy('forte', forte_options=forte_options)

# check the results
from numpy import isclose
isclose(efci1, -38.924726774489)
isclose(efci2, -38.800424868719)
