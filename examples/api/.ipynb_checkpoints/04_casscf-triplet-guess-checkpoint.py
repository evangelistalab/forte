# examples/api/04_casscf-triplet-guess.py
"""Example of a CASSCF computation on singlet methylene starting from triplet ROHF orbitals (from psi4)"""

import psi4
import forte

psi4.geometry("""
0 3
C
H 1 1.085
H 1 1.085 2 135.5
""")

psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12, 'reference': 'rohf'})

e, wfn = psi4.energy('scf',return_wfn=True)

psi4.set_options({
        'forte__job_type': 'mcscf_two_step',
        'forte__multiplicity' : 1, # <-- to override multiplicity = 2 assumed from geometry
        'forte__active_space_solver': 'fci',
        'forte__restricted_docc': [1, 0, 0, 0],
        'forte__active': [3, 0, 2, 2],
    }
)

psi4.energy('forte',ref_wfn=wfn)


