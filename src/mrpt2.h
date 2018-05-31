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

#ifndef _mrpt2_h_
#define _mrpt2_h_

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "ci_rdm/ci_rdms.h"
#include "helpers.h"
#include "reference.h"
#include "sparse_ci/determinant.h"
#include "integrals/integrals.h"
#include "fci/fci_integrals.h"
#include "determinant_hashvector.h"
#include "operator.h"
#include "sparse_ci/sparse_ci_solver.h"

namespace psi {
namespace forte {

void set_PT2_options(ForteOptions& foptions);

class MRPT2 : public Wavefunction {
  public:
    // Class constructor and destructor
    MRPT2(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
          std::shared_ptr<MOSpaceInfo> mo_space_info, DeterminantHashVec& reference,
          SharedMatrix evecs, SharedVector evals);

    ~MRPT2();

    std::shared_ptr<ForteIntegrals> ints_;
    DeterminantHashVec& reference_;

    double compute_energy();

  private:
    std::shared_ptr<FCIIntegrals> fci_ints_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    SharedMatrix evecs_;
    SharedVector evals_;

    void startup();

    double compute_pt2_energy();
    double energy_kernel(int bin, int nbin);

    std::vector<int> mo_symmetry_;

    int nroot_;
    int multiplicity_;
    DiagonalizationMethod diag_method_;

    double screen_thresh_;
};
}
}

#endif // _mrpt2_h_
