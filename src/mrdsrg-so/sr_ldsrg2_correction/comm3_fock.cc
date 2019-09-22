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

#include <algorithm>
#include <map>
#include <vector>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "mrdsrg-so/mrdsrg_so.h"

using namespace psi;

namespace forte {

void MRDSRG_SO::sr_ldsrg2star_comm3_fock(BlockedTensor &C1, BlockedTensor &C2) {
    C1["c0,v0"] += (-1.0 / 12.0) * F["c1,v1"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["c0,v0"] += (-1.0 / 12.0) * F["c1,v1"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["c0,v0"] += (-1.0 / 12.0) * V["v1,v2,v0,c1"] * T2["c0,c1,v1,v3"] * T2["c2,c3,v2,v4"] * T2["c2,c3,v3,v4"];
    C1["c0,v0"] += (-1.0 / 24.0) * V["v1,v2,v0,c1"] * T2["c0,c2,v1,v2"] * T2["c1,c3,v3,v4"] * T2["c2,c3,v3,v4"];
    C1["c0,v0"] += (1.0 / 12.0) * V["v1,c0,v0,c1"] * T1["c1,v2"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["c0,v0"] += (1.0 / 12.0) * V["v1,c0,v0,c1"] * T1["c2,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["c0,v0"] += (-1.0 / 12.0) * V["v1,c2,v0,c1"] * T1["c0,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["c0,v0"] += (1.0 / 12.0) * V["v2,v3,v0,v1"] * T1["c0,v2"] * T2["c1,c2,v1,v4"] * T2["c1,c2,v3,v4"];
    C1["c0,v0"] += (-1.0 / 12.0) * V["v2,c1,v1,c0"] * T1["c1,v0"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["c0,v0"] += (-1.0 / 12.0) * V["v3,c1,v1,v2"] * T2["c0,c1,v0,v1"] * T2["c2,c3,v2,v4"] * T2["c2,c3,v3,v4"];
    C1["c0,v0"] += (-1.0 / 12.0) * V["c0,c1,v0,v1"] * T1["c1,v2"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["c0,v0"] += (-1.0 / 12.0) * V["c0,c1,v0,v1"] * T1["c2,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["c0,v0"] += (-1.0 / 24.0) * V["c1,c2,v1,c0"] * T2["c1,c2,v0,v2"] * T2["c3,c4,v1,v3"] * T2["c3,c4,v2,v3"];
    C1["c0,v0"] += (-1.0 / 12.0) * V["c1,c2,v1,c0"] * T2["c1,c3,v0,v1"] * T2["c2,c4,v2,v3"] * T2["c3,c4,v2,v3"];
    C1["c0,v0"] += (-1.0 / 12.0) * V["c2,c3,v1,c1"] * T2["c0,c2,v0,v1"] * T2["c1,c4,v2,v3"] * T2["c3,c4,v2,v3"];
    C1["c0,v0"] += (1.0 / 12.0) * V["c2,c3,c0,c1"] * T1["c2,v0"] * T2["c1,c4,v1,v2"] * T2["c3,c4,v1,v2"];

    C1["v0,c0"] += (-1.0 / 6.0) * F["c1,v1"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (1.0 / 12.0) * F["c1,v1"] * T2["c0,c1,v2,v3"] * T2["c2,c3,v0,v1"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 6.0) * F["c1,v1"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 3.0) * F["c1,v1"] * T2["c0,c3,v1,v3"] * T2["c1,c2,v0,v2"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["v1,v2,v0,c1"] * T2["c0,c1,v1,v3"] * T2["c2,c3,v2,v4"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (1.0 / 48.0) * V["v1,v2,v0,c1"] * T2["c0,c1,v3,v4"] * T2["c2,c3,v1,v2"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (-1.0 / 24.0) * V["v1,v2,v0,c1"] * T2["c0,c2,v1,v2"] * T2["c1,c3,v3,v4"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (1.0 / 6.0) * V["v1,v2,v0,c1"] * T2["c0,c2,v1,v3"] * T2["c1,c3,v2,v4"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (1.0 / 12.0) * V["v1,c0,v0,c1"] * T1["c1,v2"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (1.0 / 12.0) * V["v1,c0,v0,c1"] * T1["c2,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["v1,c2,v0,c1"] * T1["c0,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 6.0) * V["v1,c2,v0,c1"] * T1["c1,v2"] * T2["c0,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["v1,c2,v0,c1"] * T1["c3,v1"] * T2["c0,c1,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (1.0 / 12.0) * V["v2,v3,v0,v1"] * T1["c0,v2"] * T2["c1,c2,v1,v4"] * T2["c1,c2,v3,v4"];
    C1["v0,c0"] += (-1.0 / 6.0) * V["v2,v3,v0,v1"] * T1["c1,v2"] * T2["c0,c2,v3,v4"] * T2["c1,c2,v1,v4"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["v2,c1,v1,c0"] * T1["c1,v0"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["v2,c1,v1,c0"] * T1["c1,v3"] * T2["c2,c3,v0,v1"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 6.0) * V["v2,c1,v1,c0"] * T1["c2,v1"] * T2["c1,c3,v0,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (1.0 / 6.0) * V["v2,c2,v1,c1"] * T1["c0,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v0,v3"];
    C1["v0,c0"] += (1.0 / 6.0) * V["v2,c2,v1,c1"] * T1["c1,v0"] * T2["c0,c3,v2,v3"] * T2["c2,c3,v1,v3"];
    C1["v0,c0"] += (1.0 / 6.0) * V["v2,c2,v1,c1"] * T1["c1,v3"] * T2["c0,c3,v0,v2"] * T2["c2,c3,v1,v3"];
    C1["v0,c0"] += (1.0 / 6.0) * V["v2,c2,v1,c1"] * T1["c3,v1"] * T2["c0,c2,v0,v3"] * T2["c1,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 24.0) * V["v3,v4,v1,v2"] * T1["c0,v1"] * T2["c1,c2,v0,v2"] * T2["c1,c2,v3,v4"];
    C1["v0,c0"] += (1.0 / 12.0) * V["v3,v4,v1,v2"] * T1["c1,v1"] * T2["c0,c2,v0,v2"] * T2["c1,c2,v3,v4"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["v3,c1,v1,v2"] * T2["c0,c1,v0,v1"] * T2["c2,c3,v2,v4"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (-1.0 / 24.0) * V["v3,c1,v1,v2"] * T2["c0,c1,v0,v4"] * T2["c2,c3,v1,v2"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["v3,c1,v1,v2"] * T2["c0,c1,v2,v4"] * T2["c2,c3,v0,v1"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (1.0 / 6.0) * V["v3,c1,v1,v2"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v4"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["v3,c1,v1,v2"] * T2["c0,c3,v1,v2"] * T2["c1,c2,v0,v4"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (-1.0 / 6.0) * V["v3,c1,v1,v2"] * T2["c0,c3,v2,v4"] * T2["c1,c2,v0,v1"] * T2["c2,c3,v3,v4"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["c0,c1,v0,v1"] * T1["c1,v2"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["c0,c1,v0,v1"] * T1["c2,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (1.0 / 24.0) * V["c0,c1,v1,v2"] * T1["c1,v3"] * T2["c2,c3,v0,v3"] * T2["c2,c3,v1,v2"];
    C1["v0,c0"] += (-1.0 / 6.0) * V["c0,c1,v1,v2"] * T1["c2,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v0,v3"];
    C1["v0,c0"] += (-1.0 / 6.0) * V["c1,c2,v0,v1"] * T1["c1,v2"] * T2["c0,c3,v2,v3"] * T2["c2,c3,v1,v3"];
    C1["v0,c0"] += (1.0 / 24.0) * V["c1,c2,v0,v1"] * T1["c3,v1"] * T2["c0,c3,v2,v3"] * T2["c1,c2,v2,v3"];
    C1["v0,c0"] += (1.0 / 12.0) * V["c1,c2,v1,v2"] * T1["c1,v3"] * T2["c0,c3,v0,v3"] * T2["c2,c3,v1,v2"];
    C1["v0,c0"] += (1.0 / 12.0) * V["c1,c2,v1,v2"] * T1["c3,v1"] * T2["c0,c3,v0,v3"] * T2["c1,c2,v2,v3"];
    C1["v0,c0"] += (-1.0 / 24.0) * V["c1,c2,v1,c0"] * T2["c1,c2,v0,v2"] * T2["c3,c4,v1,v3"] * T2["c3,c4,v2,v3"];
    C1["v0,c0"] += (1.0 / 48.0) * V["c1,c2,v1,c0"] * T2["c1,c2,v2,v3"] * T2["c3,c4,v0,v1"] * T2["c3,c4,v2,v3"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["c1,c2,v1,c0"] * T2["c1,c3,v0,v1"] * T2["c2,c4,v2,v3"] * T2["c3,c4,v2,v3"];
    C1["v0,c0"] += (1.0 / 6.0) * V["c1,c2,v1,c0"] * T2["c1,c3,v0,v2"] * T2["c2,c4,v1,v3"] * T2["c3,c4,v2,v3"];
    C1["v0,c0"] += (-1.0 / 12.0) * V["c2,c3,v1,c1"] * T2["c0,c2,v0,v1"] * T2["c1,c4,v2,v3"] * T2["c3,c4,v2,v3"];
    C1["v0,c0"] += (1.0 / 6.0) * V["c2,c3,v1,c1"] * T2["c0,c2,v0,v2"] * T2["c1,c4,v2,v3"] * T2["c3,c4,v1,v3"];
    C1["v0,c0"] += (-1.0 / 6.0) * V["c2,c3,v1,c1"] * T2["c0,c3,v1,v3"] * T2["c1,c4,v2,v3"] * T2["c2,c4,v0,v2"];
    C1["v0,c0"] += (1.0 / 12.0) * V["c2,c3,v1,c1"] * T2["c0,c3,v2,v3"] * T2["c1,c4,v2,v3"] * T2["c2,c4,v0,v1"];
    C1["v0,c0"] += (-1.0 / 24.0) * V["c2,c3,v1,c1"] * T2["c0,c4,v0,v1"] * T2["c1,c4,v2,v3"] * T2["c2,c3,v2,v3"];
    C1["v0,c0"] += (1.0 / 12.0) * V["c2,c3,v1,c1"] * T2["c0,c4,v1,v3"] * T2["c1,c4,v2,v3"] * T2["c2,c3,v0,v2"];
    C1["v0,c0"] += (1.0 / 12.0) * V["c2,c3,c0,c1"] * T1["c2,v0"] * T2["c1,c4,v1,v2"] * T2["c3,c4,v1,v2"];
    C1["v0,c0"] += (-1.0 / 6.0) * V["c2,c3,c0,c1"] * T1["c2,v1"] * T2["c1,c4,v1,v2"] * T2["c3,c4,v0,v2"];
    C1["v0,c0"] += (-1.0 / 24.0) * V["c3,c4,c1,c2"] * T1["c1,v0"] * T2["c0,c2,v1,v2"] * T2["c3,c4,v1,v2"];
    C1["v0,c0"] += (1.0 / 12.0) * V["c3,c4,c1,c2"] * T1["c1,v1"] * T2["c0,c2,v0,v2"] * T2["c3,c4,v1,v2"];

    BlockedTensor temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * F["v3,v2"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v4"] * T2["c2,c3,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * F["c0,v2"] * T1["c2,v3"] * T2["c1,c3,v0,v1"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * F["c2,c0"] * T2["c1,c4,v2,v3"] * T2["c2,c3,v0,v1"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * F["c3,c2"] * T2["c0,c2,v0,v1"] * T2["c1,c4,v2,v3"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * F["c3,c2"] * T2["c0,c4,v0,v1"] * T2["c1,c2,v2,v3"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 24.0) * V["v2,v3,v0,v1"] * T2["c0,c2,v2,v3"] * T2["c1,c3,v4,v5"] * T2["c2,c3,v4,v5"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["v2,c0,v0,v1"] * T1["c2,v2"] * T2["c1,c3,v3,v4"] * T2["c2,c3,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * V["v2,c2,v0,v1"] * T1["c0,v2"] * T2["c1,c3,v3,v4"] * T2["c2,c3,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * V["v3,c2,v2,c0"] * T1["c2,v2"] * T1["c3,v3"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["v3,c2,v2,c0"] * T1["c3,v2"] * T1["c3,v3"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["v3,c3,v2,c2"] * T2["c0,c2,v0,v1"] * T2["c1,c4,v3,v4"] * T2["c3,c4,v2,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * V["v3,c3,v2,c2"] * T2["c0,c4,v0,v1"] * T2["c1,c3,v2,v4"] * T2["c2,c4,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 24.0) * V["v4,v5,v2,v3"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"] * T2["c2,c3,v4,v5"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["v4,c0,v2,v3"] * T1["c2,v4"] * T2["c1,c3,v0,v1"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["v4,c2,v2,v3"] * T1["c3,v4"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * V["v4,c2,v2,v3"] * T1["c3,v4"] * T2["c0,c3,v0,v1"] * T2["c1,c2,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["c0,c3,v2,c2"] * T1["c2,v3"] * T2["c1,c4,v0,v1"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * V["c0,c3,v2,c2"] * T1["c4,v3"] * T2["c1,c2,v0,v1"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * V["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c2,v3"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["c3,c4,v2,c2"] * T1["c2,v3"] * T2["c0,c3,v0,v1"] * T2["c1,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["c3,c4,c0,c2"] * T1["c2,v2"] * T1["c3,v2"] * T2["c1,c4,v0,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 24.0) * V["c3,c4,c0,c2"] * T2["c1,c5,v2,v3"] * T2["c2,c5,v2,v3"] * T2["c3,c4,v0,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 24.0) * V["c4,c5,c2,c3"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"] * T2["c4,c5,v2,v3"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * F["v2,v0"] * T2["c0,c1,v2,v4"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * F["v3,v2"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v4"] * T2["c2,c3,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * F["v3,v2"] * T2["c0,c1,v1,v4"] * T2["c2,c3,v0,v2"] * T2["c2,c3,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * F["c2,v0"] * T1["c3,v2"] * T2["c0,c1,v1,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * F["c3,c2"] * T2["c0,c1,v0,v2"] * T2["c2,c4,v1,v3"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * V["v2,c3,v0,c2"] * T1["c2,v2"] * T1["c3,v3"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["v2,c3,v0,c2"] * T1["c2,v3"] * T1["c3,v3"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["v3,v4,v0,v2"] * T1["c2,v2"] * T1["c2,v3"] * T2["c0,c1,v1,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 24.0) * V["v3,v4,v0,v2"] * T2["c0,c1,v3,v4"] * T2["c2,c3,v1,v5"] * T2["c2,c3,v2,v5"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["v3,c2,v0,v2"] * T1["c3,v3"] * T2["c0,c1,v1,v4"] * T2["c2,c3,v2,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * V["v3,c2,v0,v2"] * T1["c3,v4"] * T2["c0,c1,v1,v3"] * T2["c2,c3,v2,v4"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["v3,c3,v2,c2"] * T2["c0,c1,v0,v2"] * T2["c2,c4,v3,v4"] * T2["c3,c4,v1,v4"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["v3,c3,v2,c2"] * T2["c0,c1,v1,v4"] * T2["c2,c4,v3,v4"] * T2["c3,c4,v0,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 24.0) * V["v4,v5,v2,v3"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"] * T2["c2,c3,v4,v5"];
    temp["c0,c1,v0,v1"] += (1.0 / 6.0) * V["v4,c2,v2,v3"] * T1["c3,v4"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["c0,c1,v0,c2"] * T1["c2,v2"] * T2["c3,c4,v1,v3"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * V["c0,c1,v2,c2"] * T1["c2,v0"] * T2["c3,c4,v1,v3"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 6.0) * V["c2,c3,v2,v3"] * T1["c2,v0"] * T1["c3,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 24.0) * V["c2,c3,c0,c1"] * T2["c2,c3,v0,v2"] * T2["c4,c5,v1,v3"] * T2["c4,c5,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["c3,c4,v0,c2"] * T1["c2,v2"] * T2["c0,c1,v1,v3"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["c3,c4,v2,c2"] * T1["c2,v3"] * T2["c0,c1,v0,v2"] * T2["c3,c4,v1,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["c3,c4,v2,c2"] * T1["c2,v3"] * T2["c0,c1,v1,v3"] * T2["c3,c4,v0,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 24.0) * V["c4,c5,c2,c3"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"] * T2["c4,c5,v2,v3"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * F["v2,v0"] * T2["c0,c2,v1,v2"] * T2["c1,c3,v3,v4"] * T2["c2,c3,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * F["c2,c0"] * T2["c1,c2,v0,v2"] * T2["c3,c4,v1,v3"] * T2["c3,c4,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["v2,c0,v0,c2"] * T2["c1,c2,v2,v4"] * T2["c3,c4,v1,v3"] * T2["c3,c4,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * V["v2,c0,v0,c2"] * T2["c1,c4,v3,v4"] * T2["c2,c3,v1,v2"] * T2["c3,c4,v3,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * V["v2,c3,v0,c2"] * T2["c0,c2,v1,v2"] * T2["c1,c4,v3,v4"] * T2["c3,c4,v3,v4"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["v3,c0,v0,v2"] * T1["c1,v3"] * T2["c2,c3,v1,v4"] * T2["c2,c3,v2,v4"];
    temp["c0,c1,v0,v1"] += (-1.0 / 12.0) * V["v3,c2,v2,c0"] * T2["c1,c2,v0,v2"] * T2["c3,c4,v1,v4"] * T2["c3,c4,v3,v4"];
    temp["c0,c1,v0,v1"] += (1.0 / 12.0) * V["c0,c3,v0,c2"] * T1["c2,v1"] * T2["c1,c4,v2,v3"] * T2["c3,c4,v2,v3"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcc"});
    temp["v0,v1,c0,c1"] += (-1.0 / 3.0) * F["c2,v2"] * T1["c3,v3"] * T2["c0,c3,v0,v2"] * T2["c1,c2,v1,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,v3,v0,c2"] * T1["c3,v4"] * T2["c0,c2,v1,v2"] * T2["c1,c3,v3,v4"];
    temp["v0,v1,c0,c1"] += (1.0 / 12.0) * V["v2,v3,v0,c2"] * T1["c3,v4"] * T2["c0,c2,v1,v4"] * T2["c1,c3,v2,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["v2,v3,v0,c2"] * T1["c3,v4"] * T2["c0,c3,v1,v2"] * T2["c1,c2,v3,v4"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,c0,v0,c2"] * T1["c1,v2"] * T1["c3,v3"] * T2["c2,c3,v1,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,c0,v0,c2"] * T1["c2,v1"] * T1["c3,v3"] * T2["c1,c3,v2,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,c0,v0,c2"] * T1["c2,v3"] * T1["c3,v3"] * T2["c1,c3,v1,v2"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,c0,v0,c2"] * T1["c3,v2"] * T1["c3,v3"] * T2["c1,c2,v1,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["v2,c3,v0,c2"] * T1["c0,v2"] * T1["c3,v3"] * T2["c1,c2,v1,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v3,v4,v0,v2"] * T1["c0,v3"] * T1["c2,v2"] * T2["c1,c2,v1,v4"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["v3,c2,v2,c0"] * T1["c2,v0"] * T1["c3,v3"] * T2["c1,c3,v1,v2"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["v4,c2,v2,v3"] * T1["c3,v4"] * T2["c0,c2,v0,v2"] * T2["c1,c3,v1,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 3.0) * V["c0,c2,v0,v2"] * T1["c2,v3"] * T1["c3,v2"] * T2["c1,c3,v1,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c2,c3,v2,c0"] * T1["c4,v3"] * T2["c1,c2,v0,v2"] * T2["c3,c4,v1,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["c2,c3,v2,c0"] * T1["c4,v3"] * T2["c1,c3,v1,v3"] * T2["c2,c4,v0,v2"];
    temp["v0,v1,c0,c1"] += (1.0 / 12.0) * V["c2,c3,v2,c0"] * T1["c4,v3"] * T2["c1,c4,v0,v2"] * T2["c2,c3,v1,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["c3,c4,v2,c2"] * T1["c2,v3"] * T2["c0,c3,v0,v2"] * T2["c1,c4,v1,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c3,c4,c0,c2"] * T1["c2,v2"] * T1["c3,v0"] * T2["c1,c4,v1,v2"];
    C2["v0,v1,c0,c1"] += temp["v0,v1,c0,c1"];
    C2["v0,v1,c1,c0"] -= temp["v0,v1,c0,c1"];
    C2["v1,v0,c0,c1"] -= temp["v0,v1,c0,c1"];
    C2["v1,v0,c1,c0"] += temp["v0,v1,c0,c1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcc"});
    temp["v0,v1,c0,c1"] += (-1.0 / 3.0) * F["c2,v2"] * T1["c3,v3"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["v2,v3,v0,c2"] * T1["c3,v4"] * T2["c0,c1,v1,v2"] * T2["c2,c3,v3,v4"];
    temp["v0,v1,c0,c1"] += (-1.0 / 12.0) * V["v2,v3,v0,c2"] * T1["c3,v4"] * T2["c0,c1,v2,v3"] * T2["c2,c3,v1,v4"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["v2,v3,v0,c2"] * T1["c3,v4"] * T2["c0,c1,v3,v4"] * T2["c2,c3,v1,v2"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,c3,v0,c2"] * T1["c2,v1"] * T1["c3,v3"] * T2["c0,c1,v2,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,c3,v0,c2"] * T1["c2,v3"] * T1["c3,v3"] * T2["c0,c1,v1,v2"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v3,v4,v0,v2"] * T1["c2,v2"] * T1["c2,v3"] * T2["c0,c1,v1,v4"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["v3,c3,v2,c2"] * T1["c2,v0"] * T1["c3,v2"] * T2["c0,c1,v1,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v4,c2,v2,v3"] * T1["c3,v4"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["c0,c1,v0,v2"] * T1["c2,v2"] * T1["c3,v3"] * T2["c2,c3,v1,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["c2,c3,v0,v2"] * T1["c2,v2"] * T1["c3,v3"] * T2["c0,c1,v1,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c2,c3,c0,c1"] * T1["c2,v0"] * T1["c4,v2"] * T2["c3,c4,v1,v2"];
    temp["v0,v1,c0,c1"] += (1.0 / 12.0) * V["c3,c4,v2,c2"] * T1["c2,v3"] * T2["c0,c1,v0,v2"] * T2["c3,c4,v1,v3"];
    C2["v0,v1,c0,c1"] += temp["v0,v1,c0,c1"];
    C2["v1,v0,c0,c1"] -= temp["v0,v1,c0,c1"];

    C2["v0,v1,c0,c1"] += (1.0 / 3.0) * F["c2,v2"] * T1["c3,v3"] * T2["c0,c1,v2,v3"] * T2["c2,c3,v0,v1"];
    C2["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,v3,v0,v1"] * T1["c2,v2"] * T1["c2,v4"] * T2["c0,c1,v3,v4"];
    C2["v0,v1,c0,c1"] += (-1.0 / 12.0) * V["v4,c2,v2,v3"] * T1["c3,v4"] * T2["c0,c1,v2,v3"] * T2["c2,c3,v0,v1"];
    C2["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c0,c1,v2,v3"] * T1["c2,v2"] * T1["c3,v3"] * T2["c2,c3,v0,v1"];
    C2["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c2,c3,v0,v1"] * T1["c2,v2"] * T1["c3,v3"] * T2["c0,c1,v2,v3"];
    C2["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c2,c3,c0,c1"] * T1["c2,v2"] * T1["c4,v2"] * T2["c3,c4,v0,v1"];
    C2["v0,v1,c0,c1"] += (-1.0 / 12.0) * V["c3,c4,v2,c2"] * T1["c2,v3"] * T2["c0,c1,v2,v3"] * T2["c3,c4,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcc"});
    temp["v0,v1,c0,c1"] += (-1.0 / 3.0) * F["c2,v2"] * T1["c3,v3"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v2,v3,v0,v1"] * T1["c0,v2"] * T1["c2,v4"] * T2["c1,c2,v3,v4"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v3,c2,v2,c0"] * T1["c1,v2"] * T1["c3,v3"] * T2["c2,c3,v0,v1"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["v3,c2,v2,c0"] * T1["c3,v2"] * T1["c3,v3"] * T2["c1,c2,v0,v1"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["v3,c3,v2,c2"] * T1["c0,v2"] * T1["c2,v3"] * T2["c1,c3,v0,v1"];
    temp["v0,v1,c0,c1"] += (1.0 / 12.0) * V["v4,c2,v2,v3"] * T1["c3,v4"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["c0,c2,v0,v1"] * T1["c2,v2"] * T1["c3,v3"] * T2["c1,c3,v2,v3"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["c0,c2,v2,v3"] * T1["c2,v2"] * T1["c3,v3"] * T2["c1,c3,v0,v1"];
    temp["v0,v1,c0,c1"] += (-1.0 / 6.0) * V["c2,c3,v2,c0"] * T1["c4,v3"] * T2["c1,c2,v0,v1"] * T2["c3,c4,v2,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c2,c3,v2,c0"] * T1["c4,v3"] * T2["c1,c3,v2,v3"] * T2["c2,c4,v0,v1"];
    temp["v0,v1,c0,c1"] += (-1.0 / 12.0) * V["c2,c3,v2,c0"] * T1["c4,v3"] * T2["c1,c4,v2,v3"] * T2["c2,c3,v0,v1"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c3,c4,v2,c2"] * T1["c2,v3"] * T2["c0,c3,v0,v1"] * T2["c1,c4,v2,v3"];
    temp["v0,v1,c0,c1"] += (1.0 / 6.0) * V["c3,c4,c0,c2"] * T1["c2,v2"] * T1["c3,v2"] * T2["c1,c4,v0,v1"];
    C2["v0,v1,c0,c1"] += temp["v0,v1,c0,c1"];
    C2["v0,v1,c1,c0"] -= temp["v0,v1,c0,c1"];
}

} // namespace forte
