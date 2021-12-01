/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "base_classes/mo_space_info.h"
#include "sparse_ci/determinant.h"
#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/determinant_substitution_lists.h"

namespace forte {

class MRPT2 {
  public:
    // This class performs an Epstein--Nesbet second-order energy correction
    // to any sCI wavefunction. This correction spans only the active orbitals
    // in the standard CI+PT way

    // Class constructor and destructor
    MRPT2(std::shared_ptr<ForteOptions> options, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
          std::shared_ptr<MOSpaceInfo> mo_space_info, DeterminantHashVec& reference,
          psi::SharedMatrix evecs, psi::SharedVector evals, int nroot);

    ~MRPT2();

    // The determinants
    DeterminantHashVec& reference_;

    // Computes the PT2 energy correction
    std::vector<double> compute_energy();

  private:
    // The active space integrals
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;
    // The options (needed only for memory/binning)
    std::shared_ptr<ForteOptions> options_;
    // MoSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    // The sCI expansion coefficients
    psi::SharedMatrix evecs_;
    // The sCI energies
    psi::SharedVector evals_;
    // the orbital symmetry labels
    std::vector<int> mo_symmetry_;
    // Number of reference roots
    int nroot_;

    // Computes the total energy correction for a given root
    double compute_pt2_energy(int root);
    // Computes the energy contribution from a subset of excited
    // determinants
    double energy_kernel(int bin, int nbin, int root);
};
} // namespace forte

#endif // _mrpt2_h_
