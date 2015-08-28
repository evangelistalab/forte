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
    returnvalue = psi4.plugin(sofile)
#    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

# Integration with driver routines
procedures['energy']['forte'] = run_forte
