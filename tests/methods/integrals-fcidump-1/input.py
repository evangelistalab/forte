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

import forte
reffci = -112.74446815362198
core.set_local_option("FORTE", "ACTIVE_SPACE_SOLVER", "fci")
core.set_local_option("FORTE", "INT_TYPE", "fcidump")
core.set_local_option("FORTE", "FROZEN_DOCC", [2 ,0 ,0 ,0])
core.set_local_option("FORTE", "RESTRICTED_DOCC", [2 ,0 ,0 ,0])
core.set_local_option("FORTE", "ACTIVE", [2 ,2 ,2 ,2])
core.set_local_option("FORTE", "E_CONVERGENCE", 12)
forte.run_forte('forte')
compare_values(reffci, variable("CURRENT ENERGY"),9, "FCI energy")
