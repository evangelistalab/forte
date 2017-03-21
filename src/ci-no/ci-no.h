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

#ifndef _ci_no_h_
#define _ci_no_h_

//#include <fstream>
//#include <iomanip>

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
//#include "psi4/physconst.h"

#include "../forte_options.h"
#include "../helpers.h"
#include "../integrals/integrals.h"

//#include "../ci_rdms.h"
//#include "../determinant_map.h"
//#include "../fci/fci_integrals.h"
//#include "../operator.h"
//#include "../sparse_ci_solver.h"
//#include "../stl_bitset_determinant.h"

namespace psi {
namespace forte {

/// Set the CI-NO options
void set_CINO_options(ForteOptions& foptions);

/**
 * @brief The CINO class
 * This class implements natural orbitals for CI wave functions
 */
class CINO : public Wavefunction {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    CINO(SharedWavefunction ref_wfn, Options& options,
         std::shared_ptr<ForteIntegrals> ints,
         std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~CINO();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

  private:
    // ==> Class data <==

    /// The molecular integrals required by Explorer
    std::shared_ptr<ForteIntegrals> ints_;
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
//    /// Pointer to FCI integrals
//    std::shared_ptr<FCIIntegrals> fci_ints_;

    // ==> Class functions <==

};
}
} // End Namespaces

#endif // _ci_no_h_
