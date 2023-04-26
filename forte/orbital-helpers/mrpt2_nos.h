/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _mrpt2_nos_h_
#define _mrpt2_nos_h_

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "base_classes/mo_space_info.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"
#include "base_classes/orbital_transform.h"
#include "mrdsrg-spin-adapted/sa_mrpt2.h"

namespace forte {

class MRPT2_NOS : public OrbitalTransform {
  public:
    // => Constructor <= //
    MRPT2_NOS(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
              std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
              std::shared_ptr<MOSpaceInfo> mo_space_info);

    void compute_transformation();

  private:
    /// Pointer to ForteOptions
    std::shared_ptr<ForteOptions> options_;

    /// DSRG-MRPT2 method
    std::shared_ptr<SA_MRPT2> mrpt2_;

    /// DSRG-MRPT2 1-RDM CC part
    psi::SharedMatrix D1c_;
    /// DSRG-MRPT2 1-RDM VV part
    psi::SharedMatrix D1v_;
    /// DSRG-MRPT2 1-RDM AA part
    psi::SharedMatrix D1a_;

    /// Suggest active space
    std::vector<std::vector<std::pair<int, int>>>
    suggest_active_space(const psi::Vector& D1c_evals, const psi::Vector& D1v_evals,
                         const psi::Vector& D1a_evals);
};
} // namespace forte

#endif // _mrpt2_nos_h_