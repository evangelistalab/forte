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
from psiexceptions import *


def run_libadaptive(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('libadaptive')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    #psi4.set_local_option('LIBADAPTIVE', 'PRINT', 1)
    #scf_helper(name, **kwargs)
    returnvalue = psi4.plugin('libadaptive.so')
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

def run_sr_lctsd(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('sr-lctsd')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'SRG')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','CT')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_sr_srgsd(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('sr-srgsd')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE','SRG')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','SRG')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

# Integration with driver routines
procedures['energy']['libadaptive'] = run_libadaptive
procedures['energy']['sr-lctsd'] = run_sr_lctsd
procedures['energy']['sr-srgsd'] = run_sr_srgsd


