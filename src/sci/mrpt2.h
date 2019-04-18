/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "sparse_ci/determinant.h"
#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/operator.h"
#include "sparse_ci/sparse_ci_solver.h"


namespace forte {


class MRPT2 {
  public:
    // Class constructor and destructor
    MRPT2(std::shared_ptr<ForteOptions> options, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
          std::shared_ptr<MOSpaceInfo> mo_space_info, DeterminantHashVec& reference,
          psi::SharedMatrix evecs, psi::SharedVector evals, int nroot);

    ~MRPT2();

    DeterminantHashVec& reference_;

    std::vector<double> compute_energy();

  private:
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;
    std::shared_ptr<ForteOptions> options_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    psi::SharedMatrix evecs_;
    psi::SharedVector evals_;

    void startup();

    double compute_pt2_energy(int root);
    double energy_kernel(int bin, int nbin, int root);

    std::vector<int> mo_symmetry_;

    int nroot_;
    int multiplicity_;

    double screen_thresh_;
};
}

#endif // _mrpt2_h_
