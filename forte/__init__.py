#
# @BEGIN LICENSE
#
# Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
# that implements a variety of quantum chemistry methods for strongly
# correlated electrons.
#
# Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.
#
# @END LICENSE
#
"""Plugin docstring.

"""

import sys

# Load Python modules
from .molecule import Molecule
from .basis import Basis
from .results import Results
from .model import Model, MolecularModel

from .pymodule import *
from .register_forte_options import *
from .core import ForteManager, clean_options
from .forte import *

__version__ = '0.2.3'
__author__ = 'Forte Developers'

# Create a ForteOptions object (stores all options)
forte_options = forte.ForteOptions()

# Register options defined in Forte in the forte_options object
register_forte_options(forte_options)

# create a singleton to handle startup and cleanup of forte
ForteManager()

# If we are running psi4, push the options defined in forte_options to psi
if 'psi4' in sys.modules:
    psi_options = psi4.core.get_options()
    psi_options.set_current_module('FORTE')
    forte_options.push_options_to_psi4(psi_options)
