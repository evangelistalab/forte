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

#ifndef _spin_corr_h_
#define _spin_corr_h_

#include <fstream>
#include <iomanip>

#include "base_classes/forte_options.h"
#include "ci_rdm/ci_rdms.h"
#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"
#include "orbital-helpers/iao_builder.h"
#include "orbital-helpers/localize.h"

namespace forte {

/**
 * @brief The SpinCorr class
 * This class computes the spin correlation from RDMs
 */
class SpinCorr {

  public:
    SpinCorr(std::shared_ptr<RDMs> rdms, std::shared_ptr<ForteOptions> options,
             std::shared_ptr<MOSpaceInfo> mo_space_info,
             std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    std::pair<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>> compute_nos();

    void spin_analysis();

  private:
    std::shared_ptr<RDMs> rdms_;

    std::shared_ptr<ForteOptions> options_;

    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    size_t nact_;

    size_t nirrep_;

    psi::Dimension nactpi_;
};

void perform_spin_analysis(std::shared_ptr<RDMs> rdms, std::shared_ptr<ForteOptions> options,
                           std::shared_ptr<MOSpaceInfo> mo_space_info,
                           std::shared_ptr<ActiveSpaceIntegrals> as_ints);

} // namespace forte

#endif // _spin_corr_h_
