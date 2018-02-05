/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "aci/aci.h"
#include "avas.h"
#include "ci-no/ci-no.h"
#include "ci-no/mrci-no.h"
#include "fci/fci_solver.h"
#include "fci/fci.h"
#include "fci_mo.h"
#include "forte_options.h"
#include "integrals/integrals.h"
#include "pci/pci.h"
#include "reference.h"
#include "../mrdsrg-helper/run_dsrg.h"
#include "../mrdsrg-spin-integrated/dwms_mrpt2.h"

namespace psi {
namespace forte {

void forte_options(std::string name, ForteOptions& foptions) {

    // Method-specific options
    set_FCI_options(foptions);
    set_ACI_options(foptions);
    set_PCI_options(foptions);
    set_INT_options(foptions);
    set_PT2_options(foptions);
    set_AVAS_options(foptions);
    set_CINO_options(foptions);
    set_MRCINO_options(foptions);
    set_FCI_MO_options(foptions);
    set_DSRG_options(foptions);
    set_DWMS_options(foptions);

    // General options

    /*- The job type
     *  - NONE Do not run methods (may transform orbitals)
     *  - FCI Full configuration interaction (Francesco's code)
     *  - CAS Full configuration interaction (York's code)
     *  - ACI Adaptive configuration interaction
     *  - PCI Projector CI
     *  - DSRG-MRPT2 Tensor-based DSRG-MRPT2 code
     *  - THREE-DSRG-MRPT2 A DF/CD based DSRG-MRPT2 code.  Very fast
     *  - CASSCF A AO based CASSCF code by Kevin Hannon
    -*/
    foptions.add_str("JOB_TYPE", "NONE", {"NONE",
                                          "ACI",
                                          "PCI",
                                          "CAS",
                                          "DMRG",
                                          "SR-DSRG",
                                          "SR-DSRG-ACI",
                                          "SR-DSRG-PCI",
                                          "TENSORSRG",
                                          "TENSORSRG-CI",
                                          "DSRG-MRPT2",
                                          "DSRG-MRPT3",
                                          "MR-DSRG-PT2",
                                          "THREE-DSRG-MRPT2",
                                          "SOMRDSRG",
                                          "MRDSRG",
                                          "MRDSRG_SO",
                                          "CASSCF",
                                          "ACTIVE-DSRGPT2",
                                          "DWMS-DSRGPT2",
                                          "DSRG_MRPT",
                                          "TASKS",
                                          "CC",
                                          "NOJOB"},
                     "Specify the job type");

    foptions.add_str("MINAO_BASIS", "STO-3G", "The basis used to define an orbital subspace");

    foptions.add_array("SUBSPACE", "A list of orbital subspaces");

    foptions.add_double("MS", 0.0, "Projection of spin onto the z axis");

    foptions.add_str("ACTIVE_REF_TYPE", "CAS", "Initial guess for active space wave functions");
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
