/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#pragma once

#include "ambit/tensor.h"

#include "base_classes/rdms.h"

namespace forte {
class DressedQuantity {
  public:
    DressedQuantity();
    DressedQuantity(double scalar);
    DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b);
    DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b, ambit::Tensor aa,
                    ambit::Tensor ab, ambit::Tensor bb);
    DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b, ambit::Tensor aa,
                    ambit::Tensor ab, ambit::Tensor bb, ambit::Tensor aaa, ambit::Tensor aab,
                    ambit::Tensor abb, ambit::Tensor bbb);
    double contract_with_rdms(std::shared_ptr<RDMs> rdms);

  private:
    size_t max_body_;
    double scalar_;
    ambit::Tensor a_;
    ambit::Tensor b_;
    ambit::Tensor aa_;
    ambit::Tensor ab_;
    ambit::Tensor bb_;
    ambit::Tensor aaa_;
    ambit::Tensor aab_;
    ambit::Tensor abb_;
    ambit::Tensor bbb_;
};
} // namespace forte
