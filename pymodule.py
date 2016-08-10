import psi4
import re
import os
import inputparser
import math
import warnings
import driver 
from molutil import *
import p4util
from p4util.exceptions import *
import procedures
from procedures.proc import *

plugdir = os.path.split(os.path.abspath(__file__))[0]
sofile = plugdir + "/forte.so"

def run_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('forte')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = driver.scf_helper(name, **kwargs)

    # Run FORTE
    returnvalue = psi4.plugin(sofile,ref_wfn)

    return returnvalue

def run_forte_mcscf(name, **kwargs):
    molecule = kwargs.pop('molecule', psi4.get_active_molecule())
    molecule.update_geometry()
    molecule.print_out()
    psi4.set_active_molecule(molecule)

    mcscf_wfn = run_mcscf('mcscf', return_wfn=True)
    returnvalue = psi4.plugin(sofile, mcscf_wfn)

    return returnvalue

def run_forte_casscf(name, **kwargs):
    molecule = kwargs.pop('molecule', psi4.get_active_molecule())
    molecule.update_geometry()
    molecule.print_out()
    psi4.set_active_molecule(molecule)

    cas_wfn = run_detcas('casscf', return_wfn=True)
    returnvalue = psi4.plugin(sofile, cas_wfn)

    return returnvalue

# Integration with driver routines
driver.procedures['energy']['forte'] = run_forte
driver.procedures['energy']['forte_mcscf'] = run_forte_mcscf
driver.procedures['energy']['forte_casscf'] = run_forte_casscf
