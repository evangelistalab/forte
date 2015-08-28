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
sofile = plugdir + "/src/src.so"

def run_forte(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    forte can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('forte')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Run FORTE
    returnvalue = psi4.plugin(sofile)

    return returnvalue

# Integration with driver routines
procedures['energy']['forte'] = run_forte
