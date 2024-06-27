# FCI computation on Li2 using RHF orbitals

import psi4
import forte

mol = psi4.geometry("""
Li 
Li 1 1.6
""")

psi4.set_options(
    {
        'basis': 'cc-pVDZ',
        'scf_type': 'pk',
        'forte__active_space_solver': 'fci',
        'forte__mcscf_reference' : False
    } 
)

psi4.energy('forte',molecule=mol)
