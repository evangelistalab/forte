"""Plugin docstring.

"""
__version__ = '0.1'
__author__  = 'Psi4 Developer'

# Load Python modules
from pymodule import *

# Load C++ plugin
import os
import psi4
plugdir = os.path.split(os.path.abspath(__file__))[0]
sofile = plugdir + '/src/src.so'
print "PLUGDIR",plugdir
print "SOFILE",sofile
psi4.plugin_load(sofile)
