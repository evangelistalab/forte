# examples/api/01_fci.py

import psi4
import forte

psi4.geometry("""
0 1
Li 0.0 0.0 0.0
Li 0.0 0.0 3.0
units bohr
""")

psi4.set_options({
    'basis': 'sto-3g',                    # <-- set the basis set
    'scf_type': 'pk',                     # <-- request conventional two-electron integrals
    'e_convergence': 10,                  # <-- set the energy convergence
    'forte__active_space_solver' : 'fci'} # <-- specify the active space solver
    )

psi4.energy('forte')
