#
# @BEGIN LICENSE
#
# Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
# that implements a variety of quantum chemistry methods for strongly
# correlated electrons.
#
# Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import psi4
import forte

def clean_options():
    """
    A function to clear the options object

    This class also clears the psi4 options object
    """
    # clear options
    psi4.core.clean_options()
    forte.forte_options.reset()

    # register options defined in Forte in the forte_options object
    forte.register_forte_options(forte.forte_options)

    # Push the options defined in forte_options to psi
    psi_options = psi4.core.get_options()
    psi_options.set_current_module('FORTE')
    forte.forte_options.push_options_to_psi4(psi_options)
