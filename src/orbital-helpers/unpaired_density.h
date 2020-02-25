/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _updensity_h_
#define _updensity_h_


#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/local.h"

#include "base_classes/mo_space_info.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"
#include "iao_builder.h"

namespace forte {

class UPDensity {
  public:
    UPDensity(std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
              std::shared_ptr<ForteOptions> options, psi::SharedMatrix Ua, psi::SharedMatrix Ub);

    ~UPDensity();

    void compute_unpaired_density(std::vector<double>& ordm_a, std::vector<double>& ordm_b);

  private:
    std::shared_ptr<ForteOptions> options_;
    std::shared_ptr<ForteIntegrals> ints_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    psi::SharedMatrix Uas_;
    psi::SharedMatrix Ubs_;
};
} // namespace forte

#endif
