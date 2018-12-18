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
__author__  = 'Forte Developers'

import sys

# Load Python modules
from .pymodule import *

# Load C++ plugin
from .forte import *


forte_options = forte.ForteOptions();
#forte_options.add_int()

if 'psi4' in sys.modules:
    # Register options with psi
    options = psi4.core.get_options()
    options.set_current_module('FORTE')
    forte.read_options(forte_options)
    forte_options.push_options_to_psi4(psi4.core.get_options())

#    /*- Number of frozen occupied orbitals per irrep (in Cotton order) -*/
    options.add_int("PRINT",0)
    options.add_array("FROZEN_DOCC");

#    /*- Number of restricted doubly occupied orbitals per irrep (in Cotton
#     * order) -*/
    options.add_array("RESTRICTED_DOCC");

#    /*- Number of active orbitals per irrep (in Cotton order) -*/
    options.add_array("ACTIVE");

#    /*- Number of restricted unoccupied orbitals per irrep (in Cotton order)
#     * -*/
    options.add_array("RESTRICTED_UOCC");

#    /*- Number of frozen unoccupied orbitals per irrep (in Cotton order) -*/
#    options.add_array("FROZEN_UOCC");
#    /*- Molecular orbitals to swap -
#     *  Swap mo_1 with mo_2 in irrep symmetry
#     *  Swap mo_3 with mo_4 in irrep symmetry
#     *  Format: [irrep, mo_1, mo_2, irrep, mo_3, mo_4]
#     *          Irrep and MO indices are 1-based (NOT 0-based)!
#    -*/
    options.add_array("ROTATE_MOS");

#    //////////////////////////////////////////////////////////////
#    /// OPTIONS FOR STATE-AVERAGE CASCI/CASSCF
#    //////////////////////////////////////////////////////////////
#    /*- An array of states [[irrep1, multi1, nstates1], [irrep2, multi2, nstates2], ...] -*/
    options.add_array("AVG_STATE");
#    /*- An array of weights [[w1_1, w1_2, ..., w1_n], [w2_1, w2_2, ..., w2_n], ...] -*/
    options.add_array("AVG_WEIGHT");
#    /*- Number of roots per irrep (in Cotton order) -*/
    options.add_array("NROOTPI");

#    # forte -> psi4





