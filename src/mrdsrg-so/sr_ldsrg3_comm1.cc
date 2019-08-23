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
#include "mrdsrg_so.h"

using namespace psi;

namespace forte {

void MRDSRG_SO::comm_H_A_3_sr_fock(double factor, BlockedTensor& H1, BlockedTensor& H2,
                                   BlockedTensor& H3, BlockedTensor& T1, BlockedTensor& T2,
                                   BlockedTensor& T3, double& C0, BlockedTensor& C1,
                                   BlockedTensor& C2, BlockedTensor& C3) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    C3.zero();
    BlockedTensor temp;

    C0 += 1.0 * H1["c0,v0"] * T1["c0,v0"];
    C0 += (1.0 / 4.0) * H2["c0,c1,v0,v1"] * T2["c0,c1,v0,v1"];

    C1["c0,v0"] += 1.0 * H1["v1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += 1.0 * H1["c1,v1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,c0"] * T1["c1,v0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v1,v2,v0,c1"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * T1["c1,v1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v1,c0"] * T2["c1,c2,v0,v1"];

    C1["v0,v1"] += -1.0 * H1["c0,v0"] * T1["c0,v1"];
    C1["v0,v1"] += 1.0 * H2["v1,c0,v0,v2"] * T1["c0,v2"];
    C1["v0,v1"] += (-1.0 / 2.0) * H2["c0,c1,v0,v2"] * T2["c0,c1,v1,v2"];

    C1["v0,c0"] += 1.0 * H2["c0,c1,v0,v1"] * T1["c1,v1"];

    C1["c0,c1"] += 1.0 * H1["c1,v0"] * T1["c0,v0"];
    C1["c0,c1"] += (1.0 / 2.0) * H2["c1,c2,v0,v1"] * T2["c0,c2,v0,v1"];
    C1["c0,c1"] += -1.0 * H2["c1,c2,v0,c0"] * T1["c2,v0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * T2["c1,c2,v1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H1["v2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * T1["c2,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,c0"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,v1"] * T1["c1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * T2["c2,c3,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
    temp["c0,c1,v0,c2"] += -1.0 * H2["v1,c0,v0,c2"] * T1["c1,v1"];
    temp["c0,c1,v0,c2"] += 1.0 * H2["c2,c3,v1,c0"] * T2["c1,c3,v0,v1"];
    C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
    C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];
    C2["c1,c0,v0,c2"] -= temp["c0,c1,v0,c2"];
    C2["c1,c0,c2,v0"] += temp["c0,c1,v0,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
    temp["c0,c1,v0,c2"] += 1.0 * H1["c2,v1"] * T2["c0,c1,v0,v1"];
    temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H2["v1,v2,v0,c2"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,c2"] += 1.0 * H2["c2,c3,c0,c1"] * T1["c3,v0"];
    C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
    C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
    temp["v0,c0,v1,v2"] += 1.0 * H2["v1,c1,v0,v3"] * T2["c0,c1,v2,v3"];
    temp["v0,c0,v1,v2"] += -1.0 * H2["v1,c1,v0,c0"] * T1["c1,v2"];
    C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
    C2["v0,c0,v2,v1"] -= temp["v0,c0,v1,v2"];
    C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];
    C2["c0,v0,v2,v1"] += temp["v0,c0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
    temp["v0,c0,v1,v2"] += 1.0 * H1["c1,v0"] * T2["c0,c1,v1,v2"];
    temp["v0,c0,v1,v2"] += 1.0 * H2["v1,v2,v0,v3"] * T1["c0,v3"];
    temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H2["c1,c2,v0,c0"] * T2["c1,c2,v1,v2"];
    C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
    C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
    temp["v0,c0,v1,c1"] += 1.0 * H2["v1,c1,v0,v2"] * T1["c0,v2"];
    temp["v0,c0,v1,c1"] += -1.0 * H2["c1,c2,v0,v2"] * T2["c0,c2,v1,v2"];
    temp["v0,c0,v1,c1"] += 1.0 * H2["c1,c2,v0,c0"] * T1["c2,v1"];
    C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
    C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

    C2["v0,v1,v2,v3"] += (1.0 / 2.0) * H2["c0,c1,v0,v1"] * T2["c0,c1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvv"});
    temp["v0,v1,v2,v3"] += -1.0 * H2["v2,c0,v0,v1"] * T1["c0,v3"];
    C2["v0,v1,v2,v3"] += temp["v0,v1,v2,v3"];
    C2["v0,v1,v3,v2"] -= temp["v0,v1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
    temp["v0,v1,v2,c0"] += 1.0 * H2["c0,c1,v0,v1"] * T1["c1,v2"];
    C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
    C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
    temp["v0,c0,c1,c2"] += 1.0 * H2["c1,c2,v0,v1"] * T1["c0,v1"];
    C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
    C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccc"});
    temp["c0,c1,c2,c3"] += -1.0 * H2["c2,c3,v0,c0"] * T1["c1,v0"];
    C2["c0,c1,c2,c3"] += temp["c0,c1,c2,c3"];
    C2["c1,c0,c2,c3"] -= temp["c0,c1,c2,c3"];

    C2["c0,c1,c2,c3"] += (1.0 / 2.0) * H2["c2,c3,v0,v1"] * T2["c0,c1,v0,v1"];

    // 4th-order
    C0 += (1.0 / 36.0) * H3["c0,c1,c2,v0,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];

    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H3["v1,v2,c0,v0,c1,c2"] * T2["c1,c2,v1,v2"];

    C1["v0,c0"] += (1.0 / 4.0) * H3["c0,c1,c2,v0,v1,v2"] * T2["c1,c2,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * T3["c1,c2,c3,v0,v1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H3["v2,v3,c0,v0,v1,c2"] * T2["c1,c2,v2,v3"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += 1.0 * H1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H3["v2,c0,c1,v0,v1,c2"] * T1["c2,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * T3["c0,c1,c2,v1,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H3["v2,c0,c1,v0,c2,c3"] * T2["c2,c3,v1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    C2["v0,v1,c0,c1"] += 1.0 * H3["c0,c1,c2,v0,v1,v2"] * T1["c2,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H2["c0,c1,v0,c3"] * T2["c2,c3,v1,v2"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H1["v3,v0"] * T3["c0,c1,c2,v1,v2,v3"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += -1.0 * H1["c3,c0"] * T3["c1,c2,c3,v0,v1,v2"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H2["v3,c0,v0,v1"] * T2["c1,c2,v2,v3"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
    temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H2["c2,c3,v0,c0"] * T2["c1,c3,v1,v2"];
    C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c1,c0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c1,c0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c1,c0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,v0,c0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,v0,c0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,v0,c0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,c0,v0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,c0,v0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,c0,v0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
    temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H2["v1,c2,v0,v3"] * T2["c0,c1,v2,v3"];
    C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
    temp["v0,c0,c1,v1,c2,c3"] += -1.0 * H2["c2,c3,v0,v2"] * T2["c0,c1,v1,v2"];
    C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
    temp["v0,v1,c0,v2,v3,c1"] += 1.0 * H2["c1,c2,v0,v1"] * T2["c0,c2,v2,v3"];
    C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];

    if (ldsrg3_fock_order_ > 4) {
        C1["c0,v0"] += (1.0 / 12.0) * H3["v1,v2,v3,v0,c1,c2"] * T3["c0,c1,c2,v1,v2,v3"];
        C1["c0,v0"] += (-1.0 / 12.0) * H3["c1,c2,c3,v1,v2,c0"] * T3["c1,c2,c3,v0,v1,v2"];

        C1["v0,v1"] += (1.0 / 4.0) * H3["v1,c0,c1,v0,v2,v3"] * T2["c0,c1,v2,v3"];

        C1["c0,c1"] += (1.0 / 4.0) * H3["c1,c2,c3,v0,v1,c0"] * T2["c2,c3,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
        temp["c0,c1,v0,v1"] += (1.0 / 4.0) * H3["v2,v3,c0,v0,c2,c3"] * T3["c1,c2,c3,v1,v2,v3"];
        C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
        C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
        C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
        C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

        C2["c0,c1,v0,v1"] += (1.0 / 6.0) * H3["v2,v3,v4,v0,v1,c2"] * T3["c0,c1,c2,v2,v3,v4"];
        C2["c0,c1,v0,v1"] += (1.0 / 6.0) * H3["c2,c3,c4,v2,c0,c1"] * T3["c2,c3,c4,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
        temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H3["v1,v2,c0,v0,c2,c3"] * T2["c1,c3,v1,v2"];
        C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
        C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];
        C2["c1,c0,v0,c2"] -= temp["c0,c1,v0,c2"];
        C2["c1,c0,c2,v0"] += temp["c0,c1,v0,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
        temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H2["c2,c3,v1,v2"] * T3["c0,c1,c3,v0,v1,v2"];
        temp["c0,c1,v0,c2"] += 1.0 * H3["v1,c0,c1,v0,c2,c3"] * T1["c3,v1"];
        temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H3["c2,c3,c4,v1,c0,c1"] * T2["c3,c4,v0,v1"];
        C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
        C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
        temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H3["v1,c1,c2,v0,v3,c0"] * T2["c1,c2,v2,v3"];
        C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
        C2["v0,c0,v2,v1"] -= temp["v0,c0,v1,v2"];
        C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];
        C2["c0,v0,v2,v1"] += temp["v0,c0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
        temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H2["c1,c2,v0,v3"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H3["v1,v2,c1,v0,v3,v4"] * T2["c0,c1,v3,v4"];
        temp["v0,c0,v1,v2"] += -1.0 * H3["v1,v2,c1,v0,v3,c0"] * T1["c1,v3"];
        C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
        C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
        temp["v0,c0,v1,c1"] += (1.0 / 2.0) * H3["v1,c1,c2,v0,v2,v3"] * T2["c0,c2,v2,v3"];
        temp["v0,c0,v1,c1"] += -1.0 * H3["v1,c1,c2,v0,v2,c0"] * T1["c2,v2"];
        temp["v0,c0,v1,c1"] += (-1.0 / 2.0) * H3["c1,c2,c3,v0,v2,c0"] * T2["c2,c3,v1,v2"];
        C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
        C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvv"});
        temp["v0,v1,v2,v3"] += (-1.0 / 2.0) * H3["v2,c0,c1,v0,v1,v4"] * T2["c0,c1,v3,v4"];
        C2["v0,v1,v2,v3"] += temp["v0,v1,v2,v3"];
        C2["v0,v1,v3,v2"] -= temp["v0,v1,v2,v3"];

        C2["v0,v1,v2,v3"] += 1.0 * H3["v2,v3,c0,v0,v1,v4"] * T1["c0,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
        temp["v0,v1,v2,c0"] += 1.0 * H3["v2,c0,c1,v0,v1,v3"] * T1["c1,v3"];
        temp["v0,v1,v2,c0"] += (1.0 / 2.0) * H3["c0,c1,c2,v0,v1,v3"] * T2["c1,c2,v2,v3"];
        C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
        C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
        temp["v0,c0,c1,c2"] += (1.0 / 2.0) * H3["c1,c2,c3,v0,v1,v2"] * T2["c0,c3,v1,v2"];
        temp["v0,c0,c1,c2"] += -1.0 * H3["c1,c2,c3,v0,v1,c0"] * T1["c3,v1"];
        C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
        C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

        C2["c0,c1,c2,c3"] += 1.0 * H3["c2,c3,c4,v0,c0,c1"] * T1["c4,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccc"});
        temp["c0,c1,c2,c3"] += (1.0 / 2.0) * H3["c2,c3,c4,v0,v1,c0"] * T2["c1,c4,v0,v1"];
        C2["c0,c1,c2,c3"] += temp["c0,c1,c2,c3"];
        C2["c1,c0,c2,c3"] -= temp["c0,c1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H3["c0,c1,c2,v0,c3,c4"] * T2["c3,c4,v1,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H2["v3,v4,v0,v1"] * T3["c0,c1,c2,v2,v3,v4"];
        temp["c0,c1,c2,v0,v1,v2"] += -1.0 * H3["c0,c1,c2,v0,v1,c3"] * T1["c3,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += -1.0 * H2["v3,c0,v0,c3"] * T3["c1,c2,c3,v1,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H2["c3,c4,c0,c1"] * T3["c2,c3,c4,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H3["v3,c0,c1,v0,v1,v2"] * T1["c2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H3["v3,v4,c0,v0,v1,v2"] * T2["c1,c2,v3,v4"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H3["v3,c0,c1,v0,v1,c3"] * T2["c2,c3,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += 1.0 * H2["c3,c4,c0,c1"] * T2["c2,c4,v0,v1"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += -1.0 * H2["v2,c0,v0,c3"] * T2["c1,c2,v1,v2"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v1,v0,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v1,c3,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,c3,v1,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += 1.0 * H2["c3,c4,v1,c0"] * T2["c1,c2,v0,v1"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,v0,c3,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,v0,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,c4,v0"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += 1.0 * H2["v1,c2,v0,c0"] * T2["c1,c2,v2,v3"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H2["v1,v2,v0,v4"] * T2["c0,c1,v3,v4"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H3["v1,c2,c3,v0,c0,c1"] * T1["c3,v2"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H2["c2,c3,v0,v3"] * T3["c0,c1,c3,v1,v2,v3"];
        temp["v0,c0,c1,v1,v2,c2"] += (1.0 / 2.0) * H3["v1,v2,c2,v0,v3,v4"] * T2["c0,c1,v3,v4"];
        temp["v0,c0,c1,v1,v2,c2"] += (1.0 / 2.0) * H3["c2,c3,c4,v0,c0,c1"] * T2["c3,c4,v1,v2"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += -1.0 * H3["v1,v2,c2,v0,v3,c0"] * T1["c1,v3"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H3["v1,c2,c3,v0,v3,c0"] * T2["c1,c3,v2,v3"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
        temp["v0,c0,c1,v1,c2,c3"] += -1.0 * H3["v1,c2,c3,v0,v2,c0"] * T1["c1,v2"];
        temp["v0,c0,c1,v1,c2,c3"] += -1.0 * H3["c2,c3,c4,v0,v2,c0"] * T2["c1,c4,v1,v2"];
        C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c1,c0,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c1,c0,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c1,c0,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,v0,c0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,v0,c0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,v0,c0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,c0,v0,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,c0,v0,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,c0,v0,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
        temp["v0,c0,c1,v1,c2,c3"] += (1.0 / 2.0) * H3["v1,c2,c3,v0,v2,v3"] * T2["c0,c1,v2,v3"];
        temp["v0,c0,c1,v1,c2,c3"] += -1.0 * H3["c2,c3,c4,v0,c0,c1"] * T1["c4,v1"];
        C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += 1.0 * H2["v2,c1,v0,v1"] * T2["c0,c1,v3,v4"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v2,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v4,v2"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
        temp["v0,v1,c0,v2,v3,c1"] += -1.0 * H3["v2,c1,c2,v0,v1,v4"] * T2["c0,c2,v3,v4"];
        temp["v0,v1,c0,v2,v3,c1"] += 1.0 * H3["v2,c1,c2,v0,v1,c0"] * T1["c2,v3"];
        C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v3,v2,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v3,c1,v2"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,c1,v3,v2"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v3,v2,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v3,c1,v2"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,c1,v3,v2"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v3,v2,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v3,c1,v2"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,c1,v3,v2"] -= temp["v0,v1,c0,v2,v3,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
        temp["v0,v1,c0,v2,v3,c1"] += 1.0 * H3["v2,v3,c1,v0,v1,v4"] * T1["c0,v4"];
        temp["v0,v1,c0,v2,v3,c1"] += (1.0 / 2.0) * H3["c1,c2,c3,v0,v1,c0"] * T2["c2,c3,v2,v3"];
        C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvcc"});
        temp["v0,v1,c0,v2,c1,c2"] += 1.0 * H3["v2,c1,c2,v0,v1,v3"] * T1["c0,v3"];
        temp["v0,v1,c0,v2,c1,c2"] += 1.0 * H3["c1,c2,c3,v0,v1,v3"] * T2["c0,c3,v2,v3"];
        temp["v0,v1,c0,v2,c1,c2"] += -1.0 * H3["c1,c2,c3,v0,v1,c0"] * T1["c3,v2"];
        C3["v0,v1,c0,v2,c1,c2"] += temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,v1,c0,c1,v2,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,v1,c0,c1,c2,v2"] += temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,c0,v1,v2,c1,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,c0,v1,c1,v2,c2"] += temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,c0,v1,c1,c2,v2"] -= temp["v0,v1,c0,v2,c1,c2"];
        C3["c0,v0,v1,v2,c1,c2"] += temp["v0,v1,c0,v2,c1,c2"];
        C3["c0,v0,v1,c1,v2,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
        C3["c0,v0,v1,c1,c2,v2"] += temp["v0,v1,c0,v2,c1,c2"];
    }

    if (ldsrg3_fock_order_ > 5) {
        C1["v0,v1"] += (-1.0 / 12.0) * H3["c0,c1,c2,v0,v2,v3"] * T3["c0,c1,c2,v1,v2,v3"];

        C1["c0,c1"] += (1.0 / 12.0) * H3["c1,c2,c3,v0,v1,v2"] * T3["c0,c2,c3,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
        temp["c0,c1,v0,c2"] += (-1.0 / 4.0) * H3["c2,c3,c4,v1,v2,c0"] * T3["c1,c3,c4,v0,v1,v2"];
        C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
        C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];
        C2["c1,c0,v0,c2"] -= temp["c0,c1,v0,c2"];
        C2["c1,c0,c2,v0"] += temp["c0,c1,v0,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
        temp["c0,c1,v0,c2"] += (1.0 / 6.0) * H3["v1,v2,v3,v0,c2,c3"] * T3["c0,c1,c3,v1,v2,v3"];
        C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
        C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
        temp["v0,c0,v1,v2"] += (-1.0 / 6.0) * H3["c1,c2,c3,v0,v3,c0"] * T3["c1,c2,c3,v1,v2,v3"];
        C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
        C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
        temp["v0,c0,v1,v2"] += (1.0 / 4.0) * H3["v1,c1,c2,v0,v3,v4"] * T3["c0,c1,c2,v2,v3,v4"];
        C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
        C2["v0,c0,v2,v1"] -= temp["v0,c0,v1,v2"];
        C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];
        C2["c0,v0,v2,v1"] += temp["v0,c0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
        temp["v0,c0,v1,c1"] += (-1.0 / 4.0) * H3["c1,c2,c3,v0,v2,v3"] * T3["c0,c2,c3,v1,v2,v3"];
        C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
        C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

        C2["v0,v1,v2,v3"] += (1.0 / 6.0) * H3["c0,c1,c2,v0,v1,v4"] * T3["c0,c1,c2,v2,v3,v4"];

        C2["c0,c1,c2,c3"] += (1.0 / 6.0) * H3["c2,c3,c4,v0,v1,v2"] * T3["c0,c1,c4,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H3["v3,c0,c1,v0,c3,c4"] * T3["c2,c3,c4,v1,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        C3["c0,c1,c2,v0,v1,v2"] += (1.0 / 6.0) * H3["v3,v4,v5,v0,v1,v2"] * T3["c0,c1,c2,v3,v4,v5"];
        C3["c0,c1,c2,v0,v1,v2"] += (-1.0 / 6.0) * H3["c3,c4,c5,c0,c1,c2"] * T3["c3,c4,c5,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (-1.0 / 2.0) * H3["v3,v4,c0,v0,v1,c3"] * T3["c1,c2,c3,v2,v3,v4"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += -1.0 * H3["v2,c0,c1,v0,c3,c4"] * T2["c2,c4,v1,v2"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v1,v0,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v1,c3,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v1,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += 1.0 * H1["c3,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 2.0) * H3["c3,c4,c5,c0,c1,c2"] * T2["c4,c5,v0,v1"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += (-1.0 / 2.0) * H2["v2,v3,v0,c3"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["c0,c1,c2,v0,v1,c3"] += 1.0 * H3["c0,c1,c2,v0,c3,c4"] * T1["c4,v1"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += -1.0 * H2["c3,c4,v2,c0"] * T3["c1,c2,c4,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 2.0) * H3["v2,v3,c0,v0,v1,c3"] * T2["c1,c2,v2,v3"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += 1.0 * H3["v2,c0,c1,v0,v1,c3"] * T1["c2,v2"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += 1.0 * H3["v1,c0,c1,v0,c3,c4"] * T1["c2,v1"];
        temp["c0,c1,c2,v0,c3,c4"] += 1.0 * H3["c3,c4,c5,v1,c0,c1"] * T2["c2,c5,v0,v1"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c2,c1,v0,c3,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c2,c1,c3,v0,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c2,c1,c3,c4,v0"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c2,c0,c1,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c2,c0,c1,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c2,c0,c1,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += (1.0 / 2.0) * H2["c3,c4,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["c0,c1,c2,v0,c3,c4"] += -1.0 * H3["c3,c4,c5,c0,c1,c2"] * T1["c5,v0"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += (1.0 / 2.0) * H3["v1,v2,c0,v0,c3,c4"] * T2["c1,c2,v1,v2"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,v0,c3,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,v0,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,c4,v0"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H3["v1,v2,c2,v0,c0,c1"] * T1["c2,v3"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H1["c2,v0"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["v0,c0,c1,v1,v2,v3"] += (1.0 / 2.0) * H3["v1,v2,v3,v0,v4,v5"] * T2["c0,c1,v4,v5"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += 1.0 * H2["v1,c2,v0,v4"] * T3["c0,c1,c2,v2,v3,v4"];
        temp["v0,c0,c1,v1,v2,v3"] += (1.0 / 2.0) * H3["v1,c2,c3,v0,c0,c1"] * T2["c2,c3,v2,v3"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += (1.0 / 2.0) * H2["c2,c3,v0,c0"] * T3["c1,c2,c3,v1,v2,v3"];
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H3["v1,v2,v3,v0,v4,c0"] * T1["c1,v4"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H3["v1,v2,c2,v0,v4,c0"] * T2["c1,c2,v3,v4"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += (-1.0 / 2.0) * H3["c2,c3,c4,v0,v3,c0"] * T3["c1,c3,c4,v1,v2,v3"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += (1.0 / 2.0) * H3["v1,c2,c3,v0,v3,v4"] * T3["c0,c1,c3,v2,v3,v4"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
        temp["v0,c0,c1,v1,c2,c3"] += (-1.0 / 2.0) * H3["c2,c3,c4,v0,v2,v3"] * T3["c0,c1,c4,v1,v2,v3"];
        C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += (1.0 / 2.0) * H3["v2,c1,c2,v0,v1,c0"] * T2["c1,c2,v3,v4"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v2,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v4,v2"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += (1.0 / 2.0) * H2["c1,c2,v0,v1"] * T3["c0,c1,c2,v2,v3,v4"];
        temp["v0,v1,c0,v2,v3,v4"] += 1.0 * H3["v2,v3,v4,v0,v1,v5"] * T1["c0,v5"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += 1.0 * H3["v2,v3,c1,v0,v1,v5"] * T2["c0,c1,v4,v5"];
        temp["v0,v1,c0,v2,v3,v4"] += -1.0 * H3["v2,v3,c1,v0,v1,c0"] * T1["c1,v4"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v2,v4,v3"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v4,v2,v3"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v4,v3"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v4,v2,v3"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v4,v3"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v4,v2,v3"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
        temp["v0,v1,c0,v2,v3,c1"] += (1.0 / 2.0) * H3["c1,c2,c3,v0,v1,v4"] * T3["c0,c2,c3,v2,v3,v4"];
        C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvvc"});
        temp["v0,v1,v2,v3,v4,c0"] += (1.0 / 2.0) * H3["c0,c1,c2,v0,v1,v2"] * T2["c1,c2,v3,v4"];
        C3["v0,v1,v2,v3,v4,c0"] += temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,v3,c0,v4"] -= temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,c0,v3,v4"] += temp["v0,v1,v2,v3,v4,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvvc"});
        temp["v0,v1,v2,v3,v4,c0"] += 1.0 * H3["v3,c0,c1,v0,v1,v2"] * T1["c1,v4"];
        C3["v0,v1,v2,v3,v4,c0"] += temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,v3,c0,v4"] -= temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,v4,v3,c0"] -= temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,v4,c0,v3"] += temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,c0,v3,v4"] += temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,c0,v4,v3"] -= temp["v0,v1,v2,v3,v4,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvcc"});
        temp["v0,v1,v2,v3,c0,c1"] += -1.0 * H3["c0,c1,c2,v0,v1,v2"] * T1["c2,v3"];
        C3["v0,v1,v2,v3,c0,c1"] += temp["v0,v1,v2,v3,c0,c1"];
        C3["v0,v1,v2,c0,v3,c1"] -= temp["v0,v1,v2,v3,c0,c1"];
        C3["v0,v1,v2,c0,c1,v3"] += temp["v0,v1,v2,v3,c0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcccc"});
        temp["v0,v1,c0,c1,c2,c3"] += 1.0 * H3["c1,c2,c3,v0,v1,v2"] * T1["c0,v2"];
        C3["v0,v1,c0,c1,c2,c3"] += temp["v0,v1,c0,c1,c2,c3"];
        C3["v0,c0,v1,c1,c2,c3"] -= temp["v0,v1,c0,c1,c2,c3"];
        C3["c0,v0,v1,c1,c2,c3"] += temp["v0,v1,c0,c1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccccc"});
        temp["v0,c0,c1,c2,c3,c4"] += -1.0 * H3["c2,c3,c4,v0,v1,c0"] * T1["c1,v1"];
        C3["v0,c0,c1,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
        C3["v0,c1,c0,c2,c3,c4"] -= temp["v0,c0,c1,c2,c3,c4"];
        C3["c0,v0,c1,c2,c3,c4"] -= temp["v0,c0,c1,c2,c3,c4"];
        C3["c0,c1,v0,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
        C3["c1,v0,c0,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
        C3["c1,c0,v0,c2,c3,c4"] -= temp["v0,c0,c1,c2,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccccc"});
        temp["v0,c0,c1,c2,c3,c4"] += (1.0 / 2.0) * H3["c2,c3,c4,v0,v1,v2"] * T2["c0,c1,v1,v2"];
        C3["v0,c0,c1,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
        C3["c0,v0,c1,c2,c3,c4"] -= temp["v0,c0,c1,c2,c3,c4"];
        C3["c0,c1,v0,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
    }

    if (ldsrg3_fock_order_ > 6) {
        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 2.0) * H3["c3,c4,c5,v2,c0,c1"] * T3["c2,c4,c5,v0,v1,v2"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 6.0) * H3["v2,v3,v4,v0,v1,c3"] * T3["c0,c1,c2,v2,v3,v4"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 2.0) * H3["v2,v3,c0,v0,c3,c4"] * T3["c1,c2,c4,v1,v2,v3"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v1,v0,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v1,c3,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,c3,v1,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += (-1.0 / 2.0) * H3["c3,c4,c5,v1,v2,c0"] * T3["c1,c2,c5,v0,v1,v2"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,v0,c3,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,v0,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,c4,v0"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += (1.0 / 6.0) * H3["v1,v2,v3,v0,c3,c4"] * T3["c0,c1,c2,v1,v2,v3"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += (-1.0 / 6.0) * H3["c2,c3,c4,v0,c0,c1"] * T3["c2,c3,c4,v1,v2,v3"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += (-1.0 / 2.0) * H3["v1,v2,c2,v0,v4,v5"] * T3["c0,c1,c2,v3,v4,v5"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += (-1.0 / 2.0) * H3["v1,c2,c3,v0,v4,c0"] * T3["c1,c2,c3,v2,v3,v4"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += (-1.0 / 6.0) * H3["c1,c2,c3,v0,v1,c0"] * T3["c1,c2,c3,v2,v3,v4"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += (1.0 / 2.0) * H3["v2,c1,c2,v0,v1,v5"] * T3["c0,c1,c2,v3,v4,v5"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v2,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v4,v2"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvvv"});
        temp["v0,v1,v2,v3,v4,v5"] += (1.0 / 2.0) * H3["v3,c0,c1,v0,v1,v2"] * T2["c0,c1,v4,v5"];
        C3["v0,v1,v2,v3,v4,v5"] += temp["v0,v1,v2,v3,v4,v5"];
        C3["v0,v1,v2,v4,v3,v5"] -= temp["v0,v1,v2,v3,v4,v5"];
        C3["v0,v1,v2,v4,v5,v3"] += temp["v0,v1,v2,v3,v4,v5"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvvv"});
        temp["v0,v1,v2,v3,v4,v5"] += -1.0 * H3["v3,v4,c0,v0,v1,v2"] * T1["c0,v5"];
        C3["v0,v1,v2,v3,v4,v5"] += temp["v0,v1,v2,v3,v4,v5"];
        C3["v0,v1,v2,v3,v5,v4"] -= temp["v0,v1,v2,v3,v4,v5"];
        C3["v0,v1,v2,v5,v3,v4"] += temp["v0,v1,v2,v3,v4,v5"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccccc"});
        temp["c0,c1,c2,c3,c4,c5"] += 1.0 * H3["c3,c4,c5,v0,c0,c1"] * T1["c2,v0"];
        C3["c0,c1,c2,c3,c4,c5"] += temp["c0,c1,c2,c3,c4,c5"];
        C3["c0,c2,c1,c3,c4,c5"] -= temp["c0,c1,c2,c3,c4,c5"];
        C3["c2,c0,c1,c3,c4,c5"] += temp["c0,c1,c2,c3,c4,c5"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccccc"});
        temp["c0,c1,c2,c3,c4,c5"] += (1.0 / 2.0) * H3["c3,c4,c5,v0,v1,c0"] * T2["c1,c2,v0,v1"];
        C3["c0,c1,c2,c3,c4,c5"] += temp["c0,c1,c2,c3,c4,c5"];
        C3["c1,c0,c2,c3,c4,c5"] -= temp["c0,c1,c2,c3,c4,c5"];
        C3["c1,c2,c0,c3,c4,c5"] += temp["c0,c1,c2,c3,c4,c5"];
    }

    if (ldsrg3_fock_order_ > 7) {
        C3["v0,v1,v2,v3,v4,v5"] += (-1.0 / 6.0) * H3["c0,c1,c2,v0,v1,v2"] * T3["c0,c1,c2,v3,v4,v5"];

        C3["c0,c1,c2,c3,c4,c5"] += (1.0 / 6.0) * H3["c3,c4,c5,v0,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    }

    // scale by factor
    C0 *= factor;
    C1.scale(factor);
    C2.scale(factor);
    C3.scale(factor);

    // add T dagger
    C0 *= 2.0;
    H1["pq"] = C1["pq"];
    C1["pq"] += H1["qp"];
    H2["pqrs"] = C2["pqrs"];
    C2["pqrs"] += H2["rspq"];
    H3["pqrsto"] = C3["pqrsto"];
    C3["pqrsto"] += H3["stopqr"];
}

void MRDSRG_SO::comm_H_A_3_sr_fink(double factor, BlockedTensor& H1, BlockedTensor& H2,
                                   BlockedTensor& H3, BlockedTensor& T1, BlockedTensor& T2,
                                   BlockedTensor& T3, double& C0, BlockedTensor& C1,
                                   BlockedTensor& C2, BlockedTensor& C3) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    C3.zero();
    BlockedTensor temp;

    C0 += 1.0 * H1["c0,v0"] * T1["c0,v0"];
    C0 += (1.0 / 4.0) * H2["c0,c1,v0,v1"] * T2["c0,c1,v0,v1"];

    C1["c0,v0"] += 1.0 * H1["v1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += 1.0 * H1["c1,v1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,c0"] * T1["c1,v0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v1,v2,v0,c1"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * T1["c1,v1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v1,c0"] * T2["c1,c2,v0,v1"];

    C1["v0,v1"] += -1.0 * H1["c0,v0"] * T1["c0,v1"];
    C1["v0,v1"] += 1.0 * H2["v1,c0,v0,v2"] * T1["c0,v2"];
    C1["v0,v1"] += (-1.0 / 2.0) * H2["c0,c1,v0,v2"] * T2["c0,c1,v1,v2"];

    C1["v0,c0"] += 1.0 * H2["c0,c1,v0,v1"] * T1["c1,v1"];

    C1["c0,c1"] += 1.0 * H1["c1,v0"] * T1["c0,v0"];
    C1["c0,c1"] += (1.0 / 2.0) * H2["c1,c2,v0,v1"] * T2["c0,c2,v0,v1"];
    C1["c0,c1"] += -1.0 * H2["c1,c2,v0,c0"] * T1["c2,v0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * T2["c1,c2,v1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H1["v2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * T1["c2,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,c0"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,v1"] * T1["c1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * T2["c2,c3,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
    temp["c0,c1,v0,c2"] += -1.0 * H2["v1,c0,v0,c2"] * T1["c1,v1"];
    temp["c0,c1,v0,c2"] += 1.0 * H2["c2,c3,v1,c0"] * T2["c1,c3,v0,v1"];
    C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
    C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];
    C2["c1,c0,v0,c2"] -= temp["c0,c1,v0,c2"];
    C2["c1,c0,c2,v0"] += temp["c0,c1,v0,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
    temp["c0,c1,v0,c2"] += 1.0 * H1["c2,v1"] * T2["c0,c1,v0,v1"];
    temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H2["v1,v2,v0,c2"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,c2"] += 1.0 * H2["c2,c3,c0,c1"] * T1["c3,v0"];
    C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
    C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
    temp["v0,c0,v1,v2"] += 1.0 * H2["v1,c1,v0,v3"] * T2["c0,c1,v2,v3"];
    temp["v0,c0,v1,v2"] += -1.0 * H2["v1,c1,v0,c0"] * T1["c1,v2"];
    C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
    C2["v0,c0,v2,v1"] -= temp["v0,c0,v1,v2"];
    C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];
    C2["c0,v0,v2,v1"] += temp["v0,c0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
    temp["v0,c0,v1,v2"] += 1.0 * H1["c1,v0"] * T2["c0,c1,v1,v2"];
    temp["v0,c0,v1,v2"] += 1.0 * H2["v1,v2,v0,v3"] * T1["c0,v3"];
    temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H2["c1,c2,v0,c0"] * T2["c1,c2,v1,v2"];
    C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
    C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
    temp["v0,c0,v1,c1"] += 1.0 * H2["v1,c1,v0,v2"] * T1["c0,v2"];
    temp["v0,c0,v1,c1"] += -1.0 * H2["c1,c2,v0,v2"] * T2["c0,c2,v1,v2"];
    temp["v0,c0,v1,c1"] += 1.0 * H2["c1,c2,v0,c0"] * T1["c2,v1"];
    C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
    C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

    C2["v0,v1,v2,v3"] += (1.0 / 2.0) * H2["c0,c1,v0,v1"] * T2["c0,c1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvv"});
    temp["v0,v1,v2,v3"] += -1.0 * H2["v2,c0,v0,v1"] * T1["c0,v3"];
    C2["v0,v1,v2,v3"] += temp["v0,v1,v2,v3"];
    C2["v0,v1,v3,v2"] -= temp["v0,v1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
    temp["v0,v1,v2,c0"] += 1.0 * H2["c0,c1,v0,v1"] * T1["c1,v2"];
    C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
    C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
    temp["v0,c0,c1,c2"] += 1.0 * H2["c1,c2,v0,v1"] * T1["c0,v1"];
    C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
    C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccc"});
    temp["c0,c1,c2,c3"] += -1.0 * H2["c2,c3,v0,c0"] * T1["c1,v0"];
    C2["c0,c1,c2,c3"] += temp["c0,c1,c2,c3"];
    C2["c1,c0,c2,c3"] -= temp["c0,c1,c2,c3"];

    C2["c0,c1,c2,c3"] += (1.0 / 2.0) * H2["c2,c3,v0,v1"] * T2["c0,c1,v0,v1"];

    // 4th order
    C0 += (1.0 / 36.0) * H3["c0,c1,c2,v0,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];

    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["c0,v0"] += (1.0 / 12.0) * H3["v1,v2,v3,v0,c1,c2"] * T3["c0,c1,c2,v1,v2,v3"];
    C1["c0,v0"] += (1.0 / 4.0) * H3["v1,v2,c0,v0,c1,c2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (-1.0 / 12.0) * H3["c1,c2,c3,v1,v2,c0"] * T3["c1,c2,c3,v0,v1,v2"];

    C1["v0,v1"] += (1.0 / 4.0) * H3["v1,c0,c1,v0,v2,v3"] * T2["c0,c1,v2,v3"];

    C1["v0,c0"] += (1.0 / 4.0) * H3["c0,c1,c2,v0,v1,v2"] * T2["c1,c2,v1,v2"];

    C1["c0,c1"] += (1.0 / 4.0) * H3["c1,c2,c3,v0,v1,c0"] * T2["c2,c3,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * T3["c1,c2,c3,v0,v1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H3["v2,v3,c0,v0,v1,c2"] * T2["c1,c2,v2,v3"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += 1.0 * H1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H3["v2,c0,c1,v0,v1,c2"] * T1["c2,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * T3["c0,c1,c2,v1,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H3["v2,c0,c1,v0,c2,c3"] * T2["c2,c3,v1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
    temp["v0,c0,v1,c1"] += (1.0 / 2.0) * H3["v1,c1,c2,v0,v2,v3"] * T2["c0,c2,v2,v3"];
    temp["v0,c0,v1,c1"] += (-1.0 / 2.0) * H3["c1,c2,c3,v0,v2,c0"] * T2["c2,c3,v1,v2"];
    C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
    C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvv"});
    temp["v0,v1,v2,v3"] += (-1.0 / 2.0) * H3["v2,c0,c1,v0,v1,v4"] * T2["c0,c1,v3,v4"];
    C2["v0,v1,v2,v3"] += temp["v0,v1,v2,v3"];
    C2["v0,v1,v3,v2"] -= temp["v0,v1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
    temp["v0,v1,v2,c0"] += 1.0 * H3["v2,c0,c1,v0,v1,v3"] * T1["c1,v3"];
    C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
    C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

    C2["v0,v1,c0,c1"] += 1.0 * H3["c0,c1,c2,v0,v1,v2"] * T1["c2,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
    temp["v0,c0,c1,c2"] += -1.0 * H3["c1,c2,c3,v0,v1,c0"] * T1["c3,v1"];
    C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
    C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccc"});
    temp["c0,c1,c2,c3"] += (1.0 / 2.0) * H3["c2,c3,c4,v0,v1,c0"] * T2["c1,c4,v0,v1"];
    C2["c0,c1,c2,c3"] += temp["c0,c1,c2,c3"];
    C2["c1,c0,c2,c3"] -= temp["c0,c1,c2,c3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H2["c3,c4,c0,c1"] * T3["c2,c3,c4,v0,v1,v2"];
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H3["v3,c0,c1,v0,v1,v2"] * T1["c2,v3"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H1["v3,v0"] * T3["c0,c1,c2,v1,v2,v3"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += -1.0 * H1["c3,c0"] * T3["c1,c2,c3,v0,v1,v2"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H2["v3,v4,v0,v1"] * T3["c0,c1,c2,v2,v3,v4"];
    temp["c0,c1,c2,v0,v1,v2"] += -1.0 * H3["c0,c1,c2,v0,v1,c3"] * T1["c3,v2"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H2["v3,c0,v0,v1"] * T2["c1,c2,v2,v3"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += -1.0 * H2["v3,c0,v0,c3"] * T3["c1,c2,c3,v1,v2,v3"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c0,c2,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c1,c2,c0,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H2["c0,c1,v0,c3"] * T2["c2,c3,v1,v2"];
    C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    C3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    C3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
    temp["c0,c1,c2,v0,v1,c3"] += 1.0 * H2["c3,c4,c0,c1"] * T2["c2,c4,v0,v1"];
    C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c2,c1,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c2,c1,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c2,c1,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c2,c0,c1,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c2,c0,c1,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c2,c0,c1,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
    temp["c0,c1,c2,v0,v1,c3"] += -1.0 * H2["v2,c0,v0,c3"] * T2["c1,c2,v1,v2"];
    C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c1,c2,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c1,c2,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c0,c1,c2,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c0,c2,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c0,c2,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c0,c2,v1,v0,c3"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c0,c2,v1,c3,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c0,c2,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c0,c2,c3,v1,v0"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c2,c0,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c2,c0,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c2,c0,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c2,c0,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c2,c0,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
    C3["c1,c2,c0,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
    temp["v0,c0,c1,v1,v2,v3"] += 1.0 * H2["v1,c2,v0,c0"] * T2["c1,c2,v2,v3"];
    C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["v0,c0,c1,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["v0,c0,c1,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["v0,c1,c0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["v0,c1,c0,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["v0,c1,c0,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,v0,c1,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,v0,c1,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,c1,v0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,c1,v0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c1,v0,c0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c1,v0,c0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c1,v0,c0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c1,c0,v0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c1,c0,v0,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c1,c0,v0,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
    temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H2["v1,v2,v0,v4"] * T2["c0,c1,v3,v4"];
    C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["v0,c0,c1,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["v0,c0,c1,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,v0,c1,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,v0,c1,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,c1,v0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
    C3["c0,c1,v0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
    temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H2["c2,c3,v0,c0"] * T2["c1,c3,v1,v2"];
    C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c1,c0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c1,c0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c1,c0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,v0,c0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,v0,c0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,v0,c0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,c0,v0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,c0,v0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c1,c0,v0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
    temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H2["v1,c2,v0,v3"] * T2["c0,c1,v2,v3"];
    C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["v0,c0,c1,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,v0,c1,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
    C3["c0,c1,v0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
    temp["v0,c0,c1,v1,c2,c3"] += -1.0 * H3["c2,c3,c4,v0,v2,c0"] * T2["c1,c4,v1,v2"];
    C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c1,c0,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c1,c0,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c1,c0,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c1,v0,c0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c1,v0,c0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c1,v0,c0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c1,c0,v0,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c1,c0,v0,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c1,c0,v0,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
    temp["v0,c0,c1,v1,c2,c3"] += -1.0 * H2["c2,c3,v0,v2"] * T2["c0,c1,v1,v2"];
    temp["v0,c0,c1,v1,c2,c3"] += (1.0 / 2.0) * H3["v1,c2,c3,v0,v2,v3"] * T2["c0,c1,v2,v3"];
    C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
    C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
    temp["v0,v1,c0,v2,v3,c1"] += -1.0 * H3["v2,c1,c2,v0,v1,v4"] * T2["c0,c2,v3,v4"];
    C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,v3,v2,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,v3,c1,v2"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,c1,v3,v2"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,v3,v2,c1"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,v3,c1,v2"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,c1,v3,v2"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,v3,v2,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,v3,c1,v2"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,c1,v3,v2"] -= temp["v0,v1,c0,v2,v3,c1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
    temp["v0,v1,c0,v2,v3,c1"] += 1.0 * H2["c1,c2,v0,v1"] * T2["c0,c2,v2,v3"];
    temp["v0,v1,c0,v2,v3,c1"] += (1.0 / 2.0) * H3["c1,c2,c3,v0,v1,c0"] * T2["c2,c3,v2,v3"];
    C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
    C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvcc"});
    temp["v0,v1,c0,v2,c1,c2"] += 1.0 * H3["v2,c1,c2,v0,v1,v3"] * T1["c0,v3"];
    temp["v0,v1,c0,v2,c1,c2"] += -1.0 * H3["c1,c2,c3,v0,v1,c0"] * T1["c3,v2"];
    C3["v0,v1,c0,v2,c1,c2"] += temp["v0,v1,c0,v2,c1,c2"];
    C3["v0,v1,c0,c1,v2,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
    C3["v0,v1,c0,c1,c2,v2"] += temp["v0,v1,c0,v2,c1,c2"];
    C3["v0,c0,v1,v2,c1,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
    C3["v0,c0,v1,c1,v2,c2"] += temp["v0,v1,c0,v2,c1,c2"];
    C3["v0,c0,v1,c1,c2,v2"] -= temp["v0,v1,c0,v2,c1,c2"];
    C3["c0,v0,v1,v2,c1,c2"] += temp["v0,v1,c0,v2,c1,c2"];
    C3["c0,v0,v1,c1,v2,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
    C3["c0,v0,v1,c1,c2,v2"] += temp["v0,v1,c0,v2,c1,c2"];

    if (ldsrg3_fink_order_ > 4) {
        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
        temp["c0,c1,v0,v1"] += (1.0 / 4.0) * H3["v2,v3,c0,v0,c2,c3"] * T3["c1,c2,c3,v1,v2,v3"];
        C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
        C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
        C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
        C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

        C2["c0,c1,v0,v1"] += (1.0 / 6.0) * H3["v2,v3,v4,v0,v1,c2"] * T3["c0,c1,c2,v2,v3,v4"];
        C2["c0,c1,v0,v1"] += (1.0 / 6.0) * H3["c2,c3,c4,v2,c0,c1"] * T3["c2,c3,c4,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
        temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H3["v1,v2,c0,v0,c2,c3"] * T2["c1,c3,v1,v2"];
        temp["c0,c1,v0,c2"] += (-1.0 / 4.0) * H3["c2,c3,c4,v1,v2,c0"] * T3["c1,c3,c4,v0,v1,v2"];
        C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
        C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];
        C2["c1,c0,v0,c2"] -= temp["c0,c1,v0,c2"];
        C2["c1,c0,c2,v0"] += temp["c0,c1,v0,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
        temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H2["c2,c3,v1,v2"] * T3["c0,c1,c3,v0,v1,v2"];
        temp["c0,c1,v0,c2"] += (1.0 / 6.0) * H3["v1,v2,v3,v0,c2,c3"] * T3["c0,c1,c3,v1,v2,v3"];
        temp["c0,c1,v0,c2"] += 1.0 * H3["v1,c0,c1,v0,c2,c3"] * T1["c3,v1"];
        temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H3["c2,c3,c4,v1,c0,c1"] * T2["c3,c4,v0,v1"];
        C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
        C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
        temp["v0,c0,v1,v2"] += (1.0 / 4.0) * H3["v1,c1,c2,v0,v3,v4"] * T3["c0,c1,c2,v2,v3,v4"];
        temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H3["v1,c1,c2,v0,v3,c0"] * T2["c1,c2,v2,v3"];
        C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
        C2["v0,c0,v2,v1"] -= temp["v0,c0,v1,v2"];
        C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];
        C2["c0,v0,v2,v1"] += temp["v0,c0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
        temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H2["c1,c2,v0,v3"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H3["v1,v2,c1,v0,v3,v4"] * T2["c0,c1,v3,v4"];
        temp["v0,c0,v1,v2"] += -1.0 * H3["v1,v2,c1,v0,v3,c0"] * T1["c1,v3"];
        temp["v0,c0,v1,v2"] += (-1.0 / 6.0) * H3["c1,c2,c3,v0,v3,c0"] * T3["c1,c2,c3,v1,v2,v3"];
        C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
        C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
        temp["v0,c0,v1,c1"] += -1.0 * H3["v1,c1,c2,v0,v2,c0"] * T1["c2,v2"];
        C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
        C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

        C2["v0,v1,v2,v3"] += 1.0 * H3["v2,v3,c0,v0,v1,v4"] * T1["c0,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
        temp["v0,v1,v2,c0"] += (1.0 / 2.0) * H3["c0,c1,c2,v0,v1,v3"] * T2["c1,c2,v2,v3"];
        C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
        C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
        temp["v0,c0,c1,c2"] += (1.0 / 2.0) * H3["c1,c2,c3,v0,v1,v2"] * T2["c0,c3,v1,v2"];
        C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
        C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

        C2["c0,c1,c2,c3"] += 1.0 * H3["c2,c3,c4,v0,c0,c1"] * T1["c4,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H3["c0,c1,c2,v0,c3,c4"] * T2["c3,c4,v1,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H3["v3,v4,c0,v0,v1,v2"] * T2["c1,c2,v3,v4"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += 1.0 * H3["v3,c0,c1,v0,v1,c3"] * T2["c2,c3,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += 1.0 * H2["c3,c4,v1,c0"] * T2["c1,c2,v0,v1"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,v0,c3,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,v0,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,c4,v0"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H3["v1,c2,c3,v0,v3,c0"] * T2["c1,c3,v2,v3"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H2["c2,c3,v0,v3"] * T3["c0,c1,c3,v1,v2,v3"];
        temp["v0,c0,c1,v1,v2,c2"] += (1.0 / 2.0) * H3["v1,v2,c2,v0,v3,v4"] * T2["c0,c1,v3,v4"];
        temp["v0,c0,c1,v1,v2,c2"] += (1.0 / 2.0) * H3["c2,c3,c4,v0,c0,c1"] * T2["c3,c4,v1,v2"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += -1.0 * H3["v1,v2,c2,v0,v3,c0"] * T1["c1,v3"];
        temp["v0,c0,c1,v1,v2,c2"] += (-1.0 / 2.0) * H3["c2,c3,c4,v0,v3,c0"] * T3["c1,c3,c4,v1,v2,v3"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c1,c0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,v0,c0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c1,c0,v0,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvc"});
        temp["v0,c0,c1,v1,v2,c2"] += (1.0 / 2.0) * H3["v1,c2,c3,v0,v3,v4"] * T3["c0,c1,c3,v2,v3,v4"];
        temp["v0,c0,c1,v1,v2,c2"] += 1.0 * H3["v1,c2,c3,v0,c0,c1"] * T1["c3,v2"];
        C3["v0,c0,c1,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["v0,c0,c1,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,v2,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v1,c2,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,v1,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,v2,c2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v1,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,v0,c1,c2,v2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,v2,c2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v1,c2,v2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,v1,c2"] -= temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,v2,c2,v1"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v1,v2"] += temp["v0,c0,c1,v1,v2,c2"];
        C3["c0,c1,v0,c2,v2,v1"] -= temp["v0,c0,c1,v1,v2,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
        temp["v0,c0,c1,v1,c2,c3"] += -1.0 * H3["c2,c3,c4,v0,c0,c1"] * T1["c4,v1"];
        C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
        temp["v0,c0,c1,v1,c2,c3"] += -1.0 * H3["v1,c2,c3,v0,v2,c0"] * T1["c1,v2"];
        C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c1,c0,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c1,c0,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c1,c0,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,v0,c0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,v0,c0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,v0,c0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,c0,v0,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,c0,v0,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c1,c0,v0,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += 1.0 * H2["v2,c1,v0,v1"] * T2["c0,c1,v3,v4"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v2,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v4,v2"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
        temp["v0,v1,c0,v2,v3,c1"] += 1.0 * H3["v2,c1,c2,v0,v1,c0"] * T1["c2,v3"];
        C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v3,v2,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v3,c1,v2"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,c1,v3,v2"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v3,v2,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v3,c1,v2"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,c1,v3,v2"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v3,v2,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v3,c1,v2"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,c1,v3,v2"] -= temp["v0,v1,c0,v2,v3,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
        temp["v0,v1,c0,v2,v3,c1"] += 1.0 * H3["v2,v3,c1,v0,v1,v4"] * T1["c0,v4"];
        C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvcc"});
        temp["v0,v1,c0,v2,c1,c2"] += 1.0 * H3["c1,c2,c3,v0,v1,v3"] * T2["c0,c3,v2,v3"];
        C3["v0,v1,c0,v2,c1,c2"] += temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,v1,c0,c1,v2,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,v1,c0,c1,c2,v2"] += temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,c0,v1,v2,c1,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,c0,v1,c1,v2,c2"] += temp["v0,v1,c0,v2,c1,c2"];
        C3["v0,c0,v1,c1,c2,v2"] -= temp["v0,v1,c0,v2,c1,c2"];
        C3["c0,v0,v1,v2,c1,c2"] += temp["v0,v1,c0,v2,c1,c2"];
        C3["c0,v0,v1,c1,v2,c2"] -= temp["v0,v1,c0,v2,c1,c2"];
        C3["c0,v0,v1,c1,c2,v2"] += temp["v0,v1,c0,v2,c1,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvvc"});
        temp["v0,v1,v2,v3,v4,c0"] += 1.0 * H3["v3,c0,c1,v0,v1,v2"] * T1["c1,v4"];
        C3["v0,v1,v2,v3,v4,c0"] += temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,v3,c0,v4"] -= temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,v4,v3,c0"] -= temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,v4,c0,v3"] += temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,c0,v3,v4"] += temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,c0,v4,v3"] -= temp["v0,v1,v2,v3,v4,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccccc"});
        temp["v0,c0,c1,c2,c3,c4"] += -1.0 * H3["c2,c3,c4,v0,v1,c0"] * T1["c1,v1"];
        C3["v0,c0,c1,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
        C3["v0,c1,c0,c2,c3,c4"] -= temp["v0,c0,c1,c2,c3,c4"];
        C3["c0,v0,c1,c2,c3,c4"] -= temp["v0,c0,c1,c2,c3,c4"];
        C3["c0,c1,v0,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
        C3["c1,v0,c0,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
        C3["c1,c0,v0,c2,c3,c4"] -= temp["v0,c0,c1,c2,c3,c4"];
    }

    if (ldsrg3_fink_order_ > 5) {
        C1["v0,v1"] += (-1.0 / 12.0) * H3["c0,c1,c2,v0,v2,v3"] * T3["c0,c1,c2,v1,v2,v3"];

        C1["c0,c1"] += (1.0 / 12.0) * H3["c1,c2,c3,v0,v1,v2"] * T3["c0,c2,c3,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
        temp["v0,c0,v1,c1"] += (-1.0 / 4.0) * H3["c1,c2,c3,v0,v2,v3"] * T3["c0,c2,c3,v1,v2,v3"];
        C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
        C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

        C2["v0,v1,v2,v3"] += (1.0 / 6.0) * H3["c0,c1,c2,v0,v1,v4"] * T3["c0,c1,c2,v2,v3,v4"];

        C2["c0,c1,c2,c3"] += (1.0 / 6.0) * H3["c2,c3,c4,v0,v1,v2"] * T3["c0,c1,c4,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * H3["v3,c0,c1,v0,c3,c4"] * T3["c2,c3,c4,v1,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        C3["c0,c1,c2,v0,v1,v2"] += (1.0 / 6.0) * H3["v3,v4,v5,v0,v1,v2"] * T3["c0,c1,c2,v3,v4,v5"];
        C3["c0,c1,c2,v0,v1,v2"] += (-1.0 / 6.0) * H3["c3,c4,c5,c0,c1,c2"] * T3["c3,c4,c5,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v0,v1,v2"] += (-1.0 / 2.0) * H3["v3,v4,c0,v0,v1,c3"] * T3["c1,c2,c3,v2,v3,v4"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += -1.0 * H3["v2,c0,c1,v0,c3,c4"] * T2["c2,c4,v1,v2"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v1,v0,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v1,c3,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v1,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += 1.0 * H1["c3,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 2.0) * H3["c3,c4,c5,c0,c1,c2"] * T2["c4,c5,v0,v1"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += (-1.0 / 2.0) * H2["v2,v3,v0,c3"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["c0,c1,c2,v0,v1,c3"] += 1.0 * H3["c0,c1,c2,v0,c3,c4"] * T1["c4,v1"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += -1.0 * H2["c3,c4,v2,c0"] * T3["c1,c2,c4,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 2.0) * H3["v2,v3,c0,v0,v1,c3"] * T2["c1,c2,v2,v3"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += 1.0 * H3["v2,c0,c1,v0,v1,c3"] * T1["c2,v2"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += 1.0 * H3["v1,c0,c1,v0,c3,c4"] * T1["c2,v1"];
        temp["c0,c1,c2,v0,c3,c4"] += 1.0 * H3["c3,c4,c5,v1,c0,c1"] * T2["c2,c5,v0,v1"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c2,c1,v0,c3,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c2,c1,c3,v0,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c2,c1,c3,c4,v0"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c2,c0,c1,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c2,c0,c1,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c2,c0,c1,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += (1.0 / 2.0) * H2["c3,c4,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["c0,c1,c2,v0,c3,c4"] += (1.0 / 6.0) * H3["v1,v2,v3,v0,c3,c4"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["c0,c1,c2,v0,c3,c4"] += -1.0 * H3["c3,c4,c5,c0,c1,c2"] * T1["c5,v0"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvcc"});
        temp["c0,c1,c2,v0,c3,c4"] += (1.0 / 2.0) * H3["v1,v2,c0,v0,c3,c4"] * T2["c1,c2,v1,v2"];
        temp["c0,c1,c2,v0,c3,c4"] += (-1.0 / 2.0) * H3["c3,c4,c5,v1,v2,c0"] * T3["c1,c2,c5,v0,v1,v2"];
        C3["c0,c1,c2,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c0,c1,c2,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,v0,c3,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,v0,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c0,c2,c3,c4,v0"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,v0,c3,c4"] += temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,v0,c4"] -= temp["c0,c1,c2,v0,c3,c4"];
        C3["c1,c2,c0,c3,c4,v0"] += temp["c0,c1,c2,v0,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H3["v1,v2,c2,v0,c0,c1"] * T1["c2,v3"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H1["c2,v0"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["v0,c0,c1,v1,v2,v3"] += (1.0 / 2.0) * H3["v1,v2,v3,v0,v4,v5"] * T2["c0,c1,v4,v5"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += 1.0 * H2["v1,c2,v0,v4"] * T3["c0,c1,c2,v2,v3,v4"];
        temp["v0,c0,c1,v1,v2,v3"] += (1.0 / 2.0) * H3["v1,c2,c3,v0,c0,c1"] * T2["c2,c3,v2,v3"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += (1.0 / 2.0) * H2["c2,c3,v0,c0"] * T3["c1,c2,c3,v1,v2,v3"];
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H3["v1,v2,v3,v0,v4,c0"] * T1["c1,v4"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += -1.0 * H3["v1,v2,c2,v0,v4,c0"] * T2["c1,c2,v3,v4"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvcc"});
        temp["v0,c0,c1,v1,c2,c3"] += (-1.0 / 2.0) * H3["c2,c3,c4,v0,v2,v3"] * T3["c0,c1,c4,v1,v2,v3"];
        C3["v0,c0,c1,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["v0,c0,c1,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,v1,c2,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,v1,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,v0,c1,c2,c3,v1"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,v1,c2,c3"] += temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,v1,c3"] -= temp["v0,c0,c1,v1,c2,c3"];
        C3["c0,c1,v0,c2,c3,v1"] += temp["v0,c0,c1,v1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += (1.0 / 2.0) * H3["v2,c1,c2,v0,v1,v5"] * T3["c0,c1,c2,v3,v4,v5"];
        temp["v0,v1,c0,v2,v3,v4"] += (1.0 / 2.0) * H3["v2,c1,c2,v0,v1,c0"] * T2["c1,c2,v3,v4"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v2,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v3,v4,v2"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v2,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v3,v4,v2"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += (1.0 / 2.0) * H2["c1,c2,v0,v1"] * T3["c0,c1,c2,v2,v3,v4"];
        temp["v0,v1,c0,v2,v3,v4"] += 1.0 * H3["v2,v3,v4,v0,v1,v5"] * T1["c0,v5"];
        temp["v0,v1,c0,v2,v3,v4"] += (-1.0 / 6.0) * H3["c1,c2,c3,v0,v1,c0"] * T3["c1,c2,c3,v2,v3,v4"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvv"});
        temp["v0,v1,c0,v2,v3,v4"] += 1.0 * H3["v2,v3,c1,v0,v1,v5"] * T2["c0,c1,v4,v5"];
        temp["v0,v1,c0,v2,v3,v4"] += -1.0 * H3["v2,v3,c1,v0,v1,c0"] * T1["c1,v4"];
        C3["v0,v1,c0,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v2,v4,v3"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,v1,c0,v4,v2,v3"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v3,v4"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v2,v4,v3"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["v0,c0,v1,v4,v2,v3"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v3,v4"] += temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v2,v4,v3"] -= temp["v0,v1,c0,v2,v3,v4"];
        C3["c0,v0,v1,v4,v2,v3"] += temp["v0,v1,c0,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcvvc"});
        temp["v0,v1,c0,v2,v3,c1"] += (1.0 / 2.0) * H3["c1,c2,c3,v0,v1,v4"] * T3["c0,c2,c3,v2,v3,v4"];
        C3["v0,v1,c0,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,v1,c0,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,v3,c1"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,v2,c1,v3"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["v0,c0,v1,c1,v2,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,v3,c1"] += temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,v2,c1,v3"] -= temp["v0,v1,c0,v2,v3,c1"];
        C3["c0,v0,v1,c1,v2,v3"] += temp["v0,v1,c0,v2,v3,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvvv"});
        temp["v0,v1,v2,v3,v4,v5"] += (1.0 / 2.0) * H3["v3,c0,c1,v0,v1,v2"] * T2["c0,c1,v4,v5"];
        C3["v0,v1,v2,v3,v4,v5"] += temp["v0,v1,v2,v3,v4,v5"];
        C3["v0,v1,v2,v4,v3,v5"] -= temp["v0,v1,v2,v3,v4,v5"];
        C3["v0,v1,v2,v4,v5,v3"] += temp["v0,v1,v2,v3,v4,v5"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvvc"});
        temp["v0,v1,v2,v3,v4,c0"] += (1.0 / 2.0) * H3["c0,c1,c2,v0,v1,v2"] * T2["c1,c2,v3,v4"];
        C3["v0,v1,v2,v3,v4,c0"] += temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,v3,c0,v4"] -= temp["v0,v1,v2,v3,v4,c0"];
        C3["v0,v1,v2,c0,v3,v4"] += temp["v0,v1,v2,v3,v4,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvcc"});
        temp["v0,v1,v2,v3,c0,c1"] += -1.0 * H3["c0,c1,c2,v0,v1,v2"] * T1["c2,v3"];
        C3["v0,v1,v2,v3,c0,c1"] += temp["v0,v1,v2,v3,c0,c1"];
        C3["v0,v1,v2,c0,v3,c1"] -= temp["v0,v1,v2,v3,c0,c1"];
        C3["v0,v1,v2,c0,c1,v3"] += temp["v0,v1,v2,v3,c0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcccc"});
        temp["v0,v1,c0,c1,c2,c3"] += 1.0 * H3["c1,c2,c3,v0,v1,v2"] * T1["c0,v2"];
        C3["v0,v1,c0,c1,c2,c3"] += temp["v0,v1,c0,c1,c2,c3"];
        C3["v0,c0,v1,c1,c2,c3"] -= temp["v0,v1,c0,c1,c2,c3"];
        C3["c0,v0,v1,c1,c2,c3"] += temp["v0,v1,c0,c1,c2,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccccc"});
        temp["v0,c0,c1,c2,c3,c4"] += (1.0 / 2.0) * H3["c2,c3,c4,v0,v1,v2"] * T2["c0,c1,v1,v2"];
        C3["v0,c0,c1,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];
        C3["c0,v0,c1,c2,c3,c4"] -= temp["v0,c0,c1,c2,c3,c4"];
        C3["c0,c1,v0,c2,c3,c4"] += temp["v0,c0,c1,c2,c3,c4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccccc"});
        temp["c0,c1,c2,c3,c4,c5"] += (1.0 / 2.0) * H3["c3,c4,c5,v0,v1,c0"] * T2["c1,c2,v0,v1"];
        C3["c0,c1,c2,c3,c4,c5"] += temp["c0,c1,c2,c3,c4,c5"];
        C3["c1,c0,c2,c3,c4,c5"] -= temp["c0,c1,c2,c3,c4,c5"];
        C3["c1,c2,c0,c3,c4,c5"] += temp["c0,c1,c2,c3,c4,c5"];
    }

    if (ldsrg3_fink_order_ > 6) {
        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 2.0) * H3["c3,c4,c5,v2,c0,c1"] * T3["c2,c4,c5,v0,v1,v2"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c2,c1,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c2,c0,c1,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 6.0) * H3["v2,v3,v4,v0,v1,c3"] * T3["c0,c1,c2,v2,v3,v4"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvc"});
        temp["c0,c1,c2,v0,v1,c3"] += (1.0 / 2.0) * H3["v2,v3,c0,v0,c3,c4"] * T3["c1,c2,c4,v1,v2,v3"];
        C3["c0,c1,c2,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c0,c1,c2,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,v1,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v0,c3,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v1,v0,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,v1,c3,v0"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,c3,v0,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c0,c2,c3,v1,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,v1,c3"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v0,c3,v1"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v1,v0,c3"] -= temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,v1,c3,v0"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,c3,v0,v1"] += temp["c0,c1,c2,v0,v1,c3"];
        C3["c1,c2,c0,c3,v1,v0"] -= temp["c0,c1,c2,v0,v1,c3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += (-1.0 / 6.0) * H3["c2,c3,c4,v0,c0,c1"] * T3["c2,c3,c4,v1,v2,v3"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += (-1.0 / 2.0) * H3["v1,v2,c2,v0,v4,v5"] * T3["c0,c1,c2,v3,v4,v5"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v3,v2"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v3,v1,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v3,v2"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v3,v1,v2"] += temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccvvv"});
        temp["v0,c0,c1,v1,v2,v3"] += (-1.0 / 2.0) * H3["v1,c2,c3,v0,v4,c0"] * T3["c1,c2,c3,v2,v3,v4"];
        C3["v0,c0,c1,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c0,c1,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["v0,c1,c0,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,v0,c1,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c0,c1,v0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v1,v2,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v2,v1,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,v0,c0,v2,v3,v1"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v1,v2,v3"] -= temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v2,v1,v3"] += temp["v0,c0,c1,v1,v2,v3"];
        C3["c1,c0,v0,v2,v3,v1"] -= temp["v0,c0,c1,v1,v2,v3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvvvv"});
        temp["v0,v1,v2,v3,v4,v5"] += -1.0 * H3["v3,v4,c0,v0,v1,v2"] * T1["c0,v5"];
        C3["v0,v1,v2,v3,v4,v5"] += temp["v0,v1,v2,v3,v4,v5"];
        C3["v0,v1,v2,v3,v5,v4"] -= temp["v0,v1,v2,v3,v4,v5"];
        C3["v0,v1,v2,v5,v3,v4"] += temp["v0,v1,v2,v3,v4,v5"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccccc"});
        temp["c0,c1,c2,c3,c4,c5"] += 1.0 * H3["c3,c4,c5,v0,c0,c1"] * T1["c2,v0"];
        C3["c0,c1,c2,c3,c4,c5"] += temp["c0,c1,c2,c3,c4,c5"];
        C3["c0,c2,c1,c3,c4,c5"] -= temp["c0,c1,c2,c3,c4,c5"];
        C3["c2,c0,c1,c3,c4,c5"] += temp["c0,c1,c2,c3,c4,c5"];

    }

    if (ldsrg3_fink_order_ > 7) {
        C3["v0,v1,v2,v3,v4,v5"] += (-1.0 / 6.0) * H3["c0,c1,c2,v0,v1,v2"] * T3["c0,c1,c2,v3,v4,v5"];

        C3["c0,c1,c2,c3,c4,c5"] += (1.0 / 6.0) * H3["c3,c4,c5,v0,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    }

    // scale by factor
    C0 *= factor;
    C1.scale(factor);
    C2.scale(factor);
    C3.scale(factor);

    // add T dagger
    C0 *= 2.0;
    H1["pq"] = C1["pq"];
    C1["pq"] += H1["qp"];
    H2["pqrs"] = C2["pqrs"];
    C2["pqrs"] += H2["rspq"];
    H3["pqrsto"] = C3["pqrsto"];
    C3["pqrsto"] += H3["stopqr"];
}

void MRDSRG_SO::comm_H_A_3_sr_2(double factor, BlockedTensor& H1, BlockedTensor& H2,
                                BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& T3, double& C0,
                                BlockedTensor& C1, BlockedTensor& C2) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    BlockedTensor temp;

    C0 += 1.0 * H1["c0,v0"] * T1["c0,v0"];
    C0 += (1.0 / 4.0) * H2["c0,c1,v0,v1"] * T2["c0,c1,v0,v1"];

    C1["c0,g0"] += 1.0 * H1["v0,g0"] * T1["c0,v0"];
    C1["c0,g0"] += (1.0 / 2.0) * H2["v0,v1,g0,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += 1.0 * H1["c1,v1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["g0,g1"] += 1.0 * H2["g1,c0,g0,v0"] * T1["c0,v0"];
    C1["g0,v0"] += -1.0 * H1["c0,g0"] * T1["c0,v0"];
    C1["g0,v0"] += (-1.0 / 2.0) * H2["c0,c1,g0,v1"] * T2["c0,c1,v0,v1"];

    C2["c0,c1,g0,g1"] += (1.0 / 2.0) * H2["v0,v1,g0,g1"] * T2["c0,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["g0,g1,v0,v1"] += (1.0 / 2.0) * H2["c0,c1,g0,g1"] * T2["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccgv"});
    temp["c0,c1,g0,v0"] += -1.0 * H1["v1,g0"] * T2["c0,c1,v0,v1"];
    temp["c0,c1,g0,v0"] += (-1.0 / 2.0) * H2["v1,v2,g0,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,c1,g0,v0"] += temp["c0,c1,g0,v0"];
    C2["c0,c1,v0,g0"] -= temp["c0,c1,g0,v0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgg"});
    temp["g0,c0,g1,g2"] += 1.0 * H2["g1,g2,g0,v0"] * T1["c0,v0"];
    C2["g0,c0,g1,g2"] += temp["g0,c0,g1,g2"];
    C2["c0,g0,g1,g2"] -= temp["g0,c0,g1,g2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcvv"});
    temp["g0,c0,v0,v1"] += 1.0 * H1["c1,g0"] * T2["c0,c1,v0,v1"];
    temp["g0,c0,v0,v1"] += (1.0 / 2.0) * H2["c1,c2,g0,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["g0,c0,v0,v1"] += temp["g0,c0,v0,v1"];
    C2["c0,g0,v0,v1"] -= temp["g0,c0,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggv"});
    temp["g0,g1,g2,v0"] += -1.0 * H2["g2,c0,g0,g1"] * T1["c0,v0"];
    C2["g0,g1,g2,v0"] += temp["g0,g1,g2,v0"];
    C2["g0,g1,v0,g2"] -= temp["g0,g1,g2,v0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgv"});
    temp["g0,c0,g1,v0"] += 1.0 * H2["g1,c1,g0,v1"] * T2["c0,c1,v0,v1"];
    C2["g0,c0,g1,v0"] += temp["g0,c0,g1,v0"];
    C2["g0,c0,v0,g1"] -= temp["g0,c0,g1,v0"];
    C2["c0,g0,g1,v0"] -= temp["g0,c0,g1,v0"];
    C2["c0,g0,v0,g1"] += temp["g0,c0,g1,v0"];

    // scale by factor
    C0 *= factor;
    C1.scale(factor);
    C2.scale(factor);

    // add T dagger
    C0 *= 2.0;
    H1["pq"] = C1["pq"];
    C1["pq"] += H1["qp"];
    H2["pqrs"] = C2["pqrs"];
    C2["pqrs"] += H2["rspq"];
}

} // namespace forte
