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
__version__ = '1.0'
__author__ = 'Forte Developers'

import sys

# Load Python modules
from .pymodule import *

from .register_forte_options import *

# Load C++ plugin
from .forte import *

# Create a ForteOptions object (stores all options)
forte_options = forte.ForteOptions()

# Register options defined in Forte in the forte_options object
register_forte_options(forte_options)  # py-side
forte.read_options(forte_options)  # c++-side

# If we are running psi4, push the options defined in forte_options to psi
if 'psi4' in sys.modules:
    psi_options = psi4.core.get_options()
    psi_options.set_current_module('FORTE')

#   for key, value in forte_options.dict().items():
#       v_type = value['type']
#       v_default = value['default_value']
#       if v_type == 'bool':
#           psi_options.add_bool(key, v_default)
#       elif v_type == 'int':
#           psi_options.add_int(key, v_default)
#       elif v_type == 'float':
#           psi_options.set_double('FORTE', key, v_default)
#       elif v_type == 'str':
#           allowed = ''
#           if 'allowed_values' in value:
#               allowed = ' '.join(value['allowed_values'])
#           psi_options.add_str(key, v_default, allowed)
#       else:
#           psi_options.add_array(key)

    forte_options.push_options_to_psi4(psi_options)
