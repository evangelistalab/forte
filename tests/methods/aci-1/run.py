import psi4
from psi4 import *
from psi4.core import *
from psi4.driver.diatomic import anharmonicity
from psi4.driver.gaussian_n import *
from psi4.driver.frac import ip_fitting, frac_traverse
from psi4.driver.aliases import *
from psi4.driver.driver_cbs import *
from psi4.driver.wrapper_database import database, db, DB_RGT, DB_RXN
from psi4.driver.wrapper_autofrag import auto_fragments
psi4_io = core.IOManager.shared_object()
geometry("""
0 1
H 0 0 0
H 0.74 0 0
""","blank_molecule_psi4_yo")

refscf = -14.839846512738
refaci = -14.889166993726
refacipt2 = -14.890166618934
li2 = geometry("""
0 1
   Li
   Li 1 2.0000
""","li2")
core.IO.set_default_namespace("li2")
core.set_global_option("BASIS", "DZ")
core.set_global_option("E_CONVERGENCE", 10)
core.set_global_option("D_CONVERGENCE", 8)
core.set_global_option("GUESS", "gwh")
core.set_local_option("SCF", "SCF_TYPE", "pk")
core.set_local_option("SCF", "REFERENCE", "rohf")
energy('scf')
#compare_values(refscf, variable("CURRENT ENERGY"),9, "SCF energy")
import forte
core.set_local_option("FORTE", "ACTIVE_SPACE_SOLVER", "aci")
