/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "helpers/timer.h"
#include "sa_mrdsrg.h"
using namespace psi;
namespace forte {
ambit::BlockedTensor SA_MRDSRG::compute_eom_hbar() {
    // IP singles. Should add foptions.
    EOM_Hbar_ = BTF_->build(tensor_type_, "EOM-Hbar", {"hh"}, true);

    EOM_Hbar["mn"] = -Hbar1_["mn"];
    E0M_Hbar["mv"] = -Hbar1_["mu"] * L1_["uv"] + Hbar2_["mwuv"] * L2_["uvwx"];
    EOM_Hbar["wx"] = -Hbar1_["vu"] * L1_["ux"] * L1_["wv"] + Hbar1_["vu"] * L2_["uwvx"] +
                     0.5 * Hbar2_["wxuv"] * L1_["yw"] * L2_["uvxz"] -
                     0.5 * Hbar2_["wxuv"] * L1_["uz"] * L2_["uywx"] +
                     0.25 * Hbar2_["wxuv"] * L3_["uvywxz"];
}
} // namespace forte