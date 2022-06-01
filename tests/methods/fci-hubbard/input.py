import psi4
from psi4 import *
from psi4.core import *
#from psi4.driver.diatomic import anharmonicity
#from psi4.driver.gaussian_n import *
#from psi4.driver.frac import ip_fitting, frac_traverse
#from psi4.driver.aliases import *
#from psi4.driver.driver_cbs import *
#from psi4.driver.wrapper_database import database, db, DB_RGT, DB_RXN
#from psi4.driver.wrapper_autofrag import auto_fragments
psi4_io = core.IOManager.shared_object()
#geometry("""
#0 1
#H 0 0 0
#H 0.74 0 0
#""","blank_molecule_psi4_yo")

import forte
reffci = -0.472135955000
core.set_local_option("FORTE", "ACTIVE_SPACE_SOLVER", "fci")
core.set_local_option("FORTE", "INT_TYPE", "fcidump")
core.set_local_option("FORTE", "E_CONVERGENCE", 12)
forte.run_forte('forte')
compare_values(reffci, variable("CURRENT ENERGY"),9, "FCI energy")
