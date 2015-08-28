import psi4
import re
import os
import inputparser
import math
import warnings
from driver import *
from wrappers import *
from molutil import *
import p4util
from p4xcpt import *

plugdir = os.path.split(os.path.abspath(__file__))[0]
sofile = os.path.split(plugdir)[1] + '.so'

def run_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('forte')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    #psi4.set_local_option('FORTE', 'PRINT', 1)
    #scf_helper(name, **kwargs)
    returnvalue = psi4.plugin(sofile)
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

# Integration with driver routines
procedures['energy']['forte'] = run_forte
procedures['energy']['forte-fci'] = run_forte_fci
procedures['energy']['aci'] = run_adaptive_ci
procedures['energy']['ex-aci'] = run_ex_aci
procedures['energy']['apici'] = run_adaptive_pici
procedures['energy']['fapici'] = run_fast_adaptive_pici
procedures['energy']['fciqmc'] = run_fciqmc
procedures['energy']['fno-apifci'] = run_fno_apifci
procedures['energy']['ct'] = run_ct
procedures['energy']['srg'] = run_srg
procedures['energy']['ct-ci'] = run_ct_ci
procedures['energy']['sr-lctsd'] = run_sr_lctsd
procedures['energy']['sr-srgsd'] = run_sr_srgsd
procedures['energy']['sr-dsrgsd'] = run_sr_dsrgsd
procedures['energy']['sr-dsrg-aci'] = run_sr_dsrg_aci
procedures['energy']['mr-dsrg-pt2'] = run_mrdsrgpt2
procedures['energy']['mrdsrg'] = run_mrdsrg
procedures['energy']['mrdsrg_so'] = run_mrdsrg_so
procedures['energy']['dsrg-mrpt2'] = run_dsrg_mrpt2
procedures['energy']['three-dsrg-mrpt2'] = run_three_dsrg_mrpt2
