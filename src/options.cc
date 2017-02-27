/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "forte_options.h"
#include "fci/fci.h"
#include "avas.h"

namespace psi { namespace forte {

void forte_options(std::string name, ForteOptions& foptions) {

    set_FCI_options(foptions);
    // set_ACI_options(foptions);
    // set_PCI_options(foptions);
    // ..
    set_AVAS_options(foptions);

    foptions.add_str("MINAO_BASIS", "STO-3G",
                     "The basis used to define an orbital subspace");

    foptions.add_array("SUBSPACE",
                       "A list of orbital subspaces");
}

}
}


// if (name == "FORTE" || options.read_globals()) {

//    /*- The basis used to define an orbital subspace -*/
//    options.add_str("MIN_BASIS", "STO-3G");

//    /*- Selects a subspace of atomic orbitals
//    *
//    *  Syntax: ["<element1><range1><ao set1>","<element2><range2><ao
// * set2>",...]
//    *
//    *  Each list entry of the form "<element><range><ao set>" specifies a
// * set
//    *  of atomic orbitals for atoms of a give type.
//    *
//    *  <element> - the symbol of the element, e.g. 'Fe', 'C'
//    *
//    *  <range>   - the range of the atoms selected.  Possible choices are:
//    *              1) '' (empty): all atoms that match <element> are
// * selected
//    *              2) 'i'       : select the i-th atom of type <element>
//    *              3) 'i-j'     : select atoms i through j (included) of
// * type <element>
//    *
//    *  <ao set>  - the set of atomic orbitals to select.  Possible choices
// * are:
// *              1) '' (empty): select all basis functions
// *              2) '(nl)'    : select the n-th level with angular momentum l
// *                             e.g. '(1s)', '(2s)', '(2p)',...
// *                             n = 1, 2, 3, ...
// *                             l = 's', 'p', 'd', 'f', 'g', ...
// *              3) '(nlm)'   : select the n-th level with angular momentum l
// * and component m
// *                             e.g. '(2pz)', '(3dzz)', '(3dxx-yy)'
// *                             n = 1, 2, 3, ...
// *                             l = 's', 'p', 'd', 'f', 'g', ...
// *                             m = 'x', 'y', 'z', 'xy', 'xz', 'yz', 'zz',
// * 'xx-yy'
// *
// *  Valid options include:
// *
// *  ["C"] - all carbon atoms
// *  ["C","N"] - all carbon and nitrogen atoms
// *  ["C1"] - carbon atom #1
// *  ["C1-3"] - carbon atoms #1, #2, #3
// *  ["C(2p)"] - the 2p subset of all carbon atoms
// *  ["C(1s,2s)"] - the 1s/2s subsets of all carbon atoms
// *  ["C1-3(2s)"] - the 2s subsets of carbon atoms #1, #2, #3
// *
// * -*/
//    options.add("SUBSPACE", new ArrayType());
//}
