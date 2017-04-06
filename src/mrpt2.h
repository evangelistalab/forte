

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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "ci_rdms.h"
#include "helpers.h"
#include "reference.h"
#include "stl_bitset_determinant.h"
#include "integrals/integrals.h"
#include "fci/fci_integrals.h"
#include "determinant_map.h"
#include "operator.h"
#include "sparse_ci_solver.h"

namespace psi { namespace forte {

class MRPT2 : public Wavefunction {
  public:

    //Class constructor and destructor
    MRPT2( SharedWavefunction ref_wfn, Options& options,
        std::shared_ptr<ForteIntegrals> ints, 
        std::shared_ptr<MOSpaceInfo> mo_space_info,
        DeterminantMap& reference, SharedMatrix evecs,
        SharedVector evals);
    
    ~MRPT2();

    std::shared_ptr<ForteIntegrals> ints_;
    DeterminantMap& reference_;

    double compute_energy(); 

  private:
    std::shared_ptr<FCIIntegrals> fci_ints_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_; 
    SharedMatrix evecs_;
    SharedVector evals_;

    void startup();

    double compute_pt2_energy();

    std::vector<int> mo_symmetry_;

    int nroot_;
    int multiplicity_;
    DiagonalizationMethod diag_method_;

    double screen_thresh_;

};
}}
