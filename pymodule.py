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
sofile = plugdir + "/forte.so"

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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def save_timing(name,timing):
    r"""Function to save execution timing
    """
    with open("timing.txt", "w+") as myfile:
        filler = " " * (64 - len(name))
        str = "        %-s%s%.6f" % (name,filler,timing)
        myfile.write(str + "\n")
        print str
