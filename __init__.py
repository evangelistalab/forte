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
__version__ = '0.1'
__author__  = 'Psi4 Developer'

# Load Python modules
from .pymodule import *

# Load C++ plugin
import os
import psi4
plugdir = os.path.split(os.path.abspath(__file__))[0]
sofile = plugdir + '/' + os.path.split(plugdir)[1] + '.so'
psi4.core.plugin_load(sofile)

