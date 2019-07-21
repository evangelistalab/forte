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

void MRDSRG_SO::comm2_l3(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                         BlockedTensor& T3, double& C0, BlockedTensor& C1, BlockedTensor& C2) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    BlockedTensor temp;

    C0 += (1.0 / 24.0) * H1["v1,v0"] * T3["c0,c1,c2,v0,v2,v3"] * T3["c0,c1,c2,v1,v2,v3"];
    C0 += (-1.0 / 24.0) * H1["c1,c0"] * T3["c0,c2,c3,v0,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
    C0 += (-1.0 / 8.0) * H2["v2,c0,v0,v1"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v3"];
    C0 += (-1.0 / 8.0) * H2["c1,c2,v0,c0"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];

    C1["g0,c0"] += (1.0 / 2.0) * H2["v1,c1,g0,v0"] * T2["c1,c2,v0,v2"] * T2["c0,c2,v1,v2"];
    C1["g0,c0"] += (1.0 / 8.0) * H2["c2,c3,g0,c1"] * T2["c0,c1,v0,v1"] * T2["c2,c3,v0,v1"];
    C1["g0,g1"] += (1.0 / 4.0) * H2["g1,v1,g0,v0"] * T2["c0,c1,v0,v2"] * T2["c0,c1,v1,v2"];
    C1["g0,g1"] += (-1.0 / 4.0) * H2["g1,c1,g0,c0"] * T2["c0,c2,v0,v1"] * T2["c1,c2,v0,v1"];
    C1["v0,g0"] += (1.0 / 2.0) * H2["v1,c1,g0,c0"] * T2["c1,c2,v0,v2"] * T2["c0,c2,v1,v2"];
    C1["v0,g0"] += (-1.0 / 8.0) * H2["v2,v3,g0,v1"] * T2["c0,c1,v0,v1"] * T2["c0,c1,v2,v3"];
    C1["v0,c0"] += (1.0 / 4.0) * H1["v2,v1"] * T2["c1,c2,v1,v3"] * T3["c0,c1,c2,v0,v2,v3"];
    C1["v0,c0"] += (-1.0 / 4.0) * H1["c2,c1"] * T2["c1,c3,v1,v2"] * T3["c0,c2,c3,v0,v1,v2"];
    C1["v0,c0"] += (-1.0 / 4.0) * H2["v3,c1,v1,v2"] * T2["c0,c2,v0,v3"] * T2["c1,c2,v1,v2"];
    C1["v0,c0"] += (-1.0 / 4.0) * H2["c2,c3,v1,c1"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v2"];

    if (ldsrg3_level_ == 1) {
        C1["v0,c0"] += (1.0 / 8.0) * H1["v1,v0"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v1,v2,v3"];
        C1["v0,c0"] += (-1.0 / 8.0) * H1["c1,c0"] * T2["c2,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];

        C2["g0,g1,c0,c1"] += (-1.0 / 2.0) * H2["v0,c2,g0,g1"] * T1["c2,v1"] * T2["c0,c1,v0,v1"];
        C2["v0,v1,c0,c1"] += (1.0 / 2.0) * H1["v3,v2"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v3"];
        C2["v0,v1,c0,c1"] += (-1.0 / 2.0) * H1["c3,c2"] * T1["c2,v2"] * T3["c0,c1,c3,v0,v1,v2"];
        C2["v0,v1,g0,g1"] += (-1.0 / 2.0) * H2["v2,c0,g0,g1"] * T1["c1,v2"] * T2["c0,c1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccgv"});
        temp["c0,c1,g0,v0"] += (1.0 / 2.0) * H2["v1,v2,g0,c2"] * T1["c2,v1"] * T2["c0,c1,v0,v2"];
        C2["c0,c1,g0,v0"] += temp["c0,c1,g0,v0"];
        C2["c0,c1,v0,g0"] -= temp["c0,c1,g0,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
        temp["c0,c1,v0,v1"] +=
            (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
        C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
        C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
        temp["c0,c1,v0,v1"] +=
            (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
        C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
        C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gccc"});
        temp["g0,c0,c1,c2"] +=
            (-1.0 / 2.0) * H2["v1,c3,g0,v0"] * T2["c0,c3,v0,v2"] * T2["c1,c2,v1,v2"];
        C2["g0,c0,c1,c2"] += temp["g0,c0,c1,c2"];
        C2["c0,g0,c1,c2"] -= temp["g0,c0,c1,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgg"});
        temp["g0,c0,g1,g2"] +=
            (-1.0 / 4.0) * H2["g1,g2,g0,c1"] * T2["c0,c2,v0,v1"] * T2["c1,c2,v0,v1"];
        C2["g0,c0,g1,g2"] += temp["g0,c0,g1,g2"];
        C2["c0,g0,g1,g2"] -= temp["g0,c0,g1,g2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcvv"});
        temp["g0,c0,v0,v1"] += (-1.0 / 2.0) * H2["c1,c2,g0,v2"] * T1["c1,v2"] * T2["c0,c2,v0,v1"];
        C2["g0,c0,v0,v1"] += temp["g0,c0,v0,v1"];
        C2["c0,g0,v0,v1"] -= temp["g0,c0,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggc"});
        temp["g0,g1,g2,c0"] += (1.0 / 2.0) * H2["g2,v0,g0,g1"] * T1["c1,v1"] * T2["c0,c1,v0,v1"];
        C2["g0,g1,g2,c0"] += temp["g0,g1,g2,c0"];
        C2["g0,g1,c0,g2"] -= temp["g0,g1,g2,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggv"});
        temp["g0,g1,g2,v0"] +=
            (-1.0 / 4.0) * H2["g2,v1,g0,g1"] * T2["c0,c1,v0,v2"] * T2["c0,c1,v1,v2"];
        C2["g0,g1,g2,v0"] += temp["g0,g1,g2,v0"];
        C2["g0,g1,v0,g2"] -= temp["g0,g1,g2,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggvc"});
        temp["g0,g1,v0,c0"] +=
            (1.0 / 2.0) * H2["v1,c1,g0,g1"] * T2["c1,c2,v0,v2"] * T2["c0,c2,v1,v2"];
        C2["g0,g1,v0,c0"] += temp["g0,g1,v0,c0"];
        C2["g0,g1,c0,v0"] -= temp["g0,g1,v0,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvcc"});
        temp["g0,v0,c0,c1"] += (-1.0 / 2.0) * H2["v2,c2,g0,v1"] * T1["c2,v1"] * T2["c0,c1,v0,v2"];
        C2["g0,v0,c0,c1"] += temp["g0,v0,c0,c1"];
        C2["v0,g0,c0,c1"] -= temp["g0,v0,c0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvgg"});
        temp["g0,v0,g1,g2"] += (-1.0 / 2.0) * H2["g1,g2,g0,c0"] * T1["c1,v1"] * T2["c0,c1,v0,v1"];
        C2["g0,v0,g1,g2"] += temp["g0,v0,g1,g2"];
        C2["v0,g0,g1,g2"] -= temp["g0,v0,g1,g2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
        temp["v0,c0,c1,c2"] +=
            (1.0 / 4.0) * H1["v1,v0"] * T2["c0,c3,v2,v3"] * T3["c1,c2,c3,v1,v2,v3"];
        temp["v0,c0,c1,c2"] +=
            (1.0 / 2.0) * H1["v2,v1"] * T2["c0,c3,v1,v3"] * T3["c1,c2,c3,v0,v2,v3"];
        temp["v0,c0,c1,c2"] +=
            (-1.0 / 4.0) * H1["c4,c3"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c4,v0,v1,v2"];
        temp["v0,c0,c1,c2"] +=
            (1.0 / 4.0) * H2["v3,c3,v1,v2"] * T2["c1,c2,v0,v3"] * T2["c0,c3,v1,v2"];
        C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
        C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcgg"});
        temp["v0,c0,g0,g1"] +=
            (1.0 / 2.0) * H2["v1,c1,g0,g1"] * T2["c1,c2,v0,v2"] * T2["c0,c2,v1,v2"];
        C2["v0,c0,g0,g1"] += temp["v0,c0,g0,g1"];
        C2["c0,v0,g0,g1"] -= temp["v0,c0,g0,g1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcc"});
        temp["v0,v1,c0,c1"] += (-1.0 / 2.0) * H1["v2,v0"] * T1["c2,v3"] * T3["c0,c1,c2,v1,v2,v3"];
        C2["v0,v1,c0,c1"] += temp["v0,v1,c0,c1"];
        C2["v1,v0,c0,c1"] -= temp["v0,v1,c0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvcc"});
        temp["v0,v1,c0,c1"] += (1.0 / 2.0) * H1["c2,c0"] * T1["c3,v2"] * T3["c1,c2,c3,v0,v1,v2"];
        C2["v0,v1,c0,c1"] += temp["v0,v1,c0,c1"];
        C2["v0,v1,c1,c0"] -= temp["v0,v1,c0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvgc"});
        temp["v0,v1,g0,c0"] += (-1.0 / 2.0) * H2["v2,c2,g0,c1"] * T1["c1,v2"] * T2["c0,c2,v0,v1"];
        C2["v0,v1,g0,c0"] += temp["v0,v1,g0,c0"];
        C2["v0,v1,c0,g0"] -= temp["v0,v1,g0,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvgv"});
        temp["v0,v1,g0,v2"] +=
            (1.0 / 2.0) * H2["v3,c1,g0,c0"] * T2["c1,c2,v0,v1"] * T2["c0,c2,v2,v3"];
        C2["v0,v1,g0,v2"] += temp["v0,v1,g0,v2"];
        C2["v0,v1,v2,g0"] -= temp["v0,v1,g0,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
        temp["v0,v1,v2,c0"] +=
            (1.0 / 4.0) * H1["v4,v3"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v4"];
        temp["v0,v1,v2,c0"] +=
            (-1.0 / 4.0) * H1["c1,c0"] * T2["c2,c3,v2,v3"] * T3["c1,c2,c3,v0,v1,v3"];
        temp["v0,v1,v2,c0"] +=
            (-1.0 / 2.0) * H1["c2,c1"] * T2["c1,c3,v2,v3"] * T3["c0,c2,c3,v0,v1,v3"];
        temp["v0,v1,v2,c0"] +=
            (1.0 / 4.0) * H2["c2,c3,v3,c1"] * T2["c0,c1,v0,v1"] * T2["c2,c3,v2,v3"];
        C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
        C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgc"});
        temp["g0,c0,g1,c1"] +=
            (1.0 / 2.0) * H2["g1,v1,g0,v0"] * T2["c0,c2,v0,v2"] * T2["c1,c2,v1,v2"];
        temp["g0,c0,g1,c1"] +=
            (-1.0 / 4.0) * H2["g1,c3,g0,c2"] * T2["c0,c3,v0,v1"] * T2["c1,c2,v0,v1"];
        C2["g0,c0,g1,c1"] += temp["g0,c0,g1,c1"];
        C2["g0,c0,c1,g1"] -= temp["g0,c0,g1,c1"];
        C2["c0,g0,g1,c1"] -= temp["g0,c0,g1,c1"];
        C2["c0,g0,c1,g1"] += temp["g0,c0,g1,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvgc"});
        temp["g0,v0,g1,c0"] += (1.0 / 2.0) * H2["g1,v2,g0,v1"] * T1["c1,v1"] * T2["c0,c1,v0,v2"];
        temp["g0,v0,g1,c0"] += (-1.0 / 2.0) * H2["g1,c2,g0,c1"] * T1["c2,v1"] * T2["c0,c1,v0,v1"];
        C2["g0,v0,g1,c0"] += temp["g0,v0,g1,c0"];
        C2["g0,v0,c0,g1"] -= temp["g0,v0,g1,c0"];
        C2["v0,g0,g1,c0"] -= temp["g0,v0,g1,c0"];
        C2["v0,g0,c0,g1"] += temp["g0,v0,g1,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvgv"});
        temp["g0,v0,g1,v1"] +=
            (-1.0 / 4.0) * H2["g1,v3,g0,v2"] * T2["c0,c1,v0,v3"] * T2["c0,c1,v1,v2"];
        temp["g0,v0,g1,v1"] +=
            (1.0 / 2.0) * H2["g1,c1,g0,c0"] * T2["c0,c2,v0,v2"] * T2["c1,c2,v1,v2"];
        C2["g0,v0,g1,v1"] += temp["g0,v0,g1,v1"];
        C2["g0,v0,v1,g1"] -= temp["g0,v0,g1,v1"];
        C2["v0,g0,g1,v1"] -= temp["g0,v0,g1,v1"];
        C2["v0,g0,v1,g1"] += temp["g0,v0,g1,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvvc"});
        temp["g0,v0,v1,c0"] +=
            (1.0 / 2.0) * H2["v3,c1,g0,v2"] * T2["c0,c2,v0,v3"] * T2["c1,c2,v1,v2"];
        temp["g0,v0,v1,c0"] +=
            (1.0 / 4.0) * H2["c2,c3,g0,c1"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v2"];
        C2["g0,v0,v1,c0"] += temp["g0,v0,v1,c0"];
        C2["g0,v0,c0,v1"] -= temp["g0,v0,v1,c0"];
        C2["v0,g0,v1,c0"] -= temp["g0,v0,v1,c0"];
        C2["v0,g0,c0,v1"] += temp["g0,v0,v1,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
        temp["v0,c0,c1,c2"] +=
            (1.0 / 4.0) * H1["c3,c1"] * T2["c0,c4,v1,v2"] * T3["c2,c3,c4,v0,v1,v2"];
        C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
        C2["v0,c0,c2,c1"] -= temp["v0,c0,c1,c2"];
        C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];
        C2["c0,v0,c2,c1"] += temp["v0,c0,c1,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcgc"});
        temp["v0,c0,g0,c1"] +=
            (1.0 / 2.0) * H2["v1,c3,g0,c2"] * T2["c1,c3,v0,v2"] * T2["c0,c2,v1,v2"];
        temp["v0,c0,g0,c1"] +=
            (-1.0 / 4.0) * H2["v2,v3,g0,v1"] * T2["c1,c2,v0,v1"] * T2["c0,c2,v2,v3"];
        C2["v0,c0,g0,c1"] += temp["v0,c0,g0,c1"];
        C2["v0,c0,c1,g0"] -= temp["v0,c0,g0,c1"];
        C2["c0,v0,g0,c1"] -= temp["v0,c0,g0,c1"];
        C2["c0,v0,c1,g0"] += temp["v0,c0,g0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
        temp["v0,v1,v2,c0"] +=
            (-1.0 / 4.0) * H1["v3,v0"] * T2["c1,c2,v2,v4"] * T3["c0,c1,c2,v1,v3,v4"];
        C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
        C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];
        C2["v1,v0,v2,c0"] -= temp["v0,v1,v2,c0"];
        C2["v1,v0,c0,v2"] += temp["v0,v1,v2,c0"];
    } else {
        C0 += (-1.0 / 8.0) * H2["v1,c1,v0,c0"] * T3["c1,c2,c3,v0,v2,v3"] * T3["c0,c2,c3,v1,v2,v3"];
        C0 += (1.0 / 48.0) * H2["v2,v3,v0,v1"] * T3["c0,c1,c2,v0,v1,v4"] * T3["c0,c1,c2,v2,v3,v4"];
        C0 += (1.0 / 48.0) * H2["c2,c3,c0,c1"] * T3["c0,c1,c4,v0,v1,v2"] * T3["c2,c3,c4,v0,v1,v2"];

        C1["c0,c1"] +=
            (1.0 / 8.0) * H1["v1,v0"] * T3["c0,c2,c3,v0,v2,v3"] * T3["c1,c2,c3,v1,v2,v3"];
        C1["c0,c1"] +=
            (-1.0 / 12.0) * H1["c3,c2"] * T3["c0,c2,c4,v0,v1,v2"] * T3["c1,c3,c4,v0,v1,v2"];
        C1["c0,c1"] +=
            (-1.0 / 4.0) * H2["v1,c3,v0,c2"] * T3["c0,c3,c4,v0,v2,v3"] * T3["c1,c2,c4,v1,v2,v3"];
        C1["c0,c1"] +=
            (1.0 / 16.0) * H2["v2,v3,v0,v1"] * T3["c0,c2,c3,v0,v1,v4"] * T3["c1,c2,c3,v2,v3,v4"];
        C1["c0,c1"] +=
            (1.0 / 4.0) * H2["v2,c2,v0,v1"] * T2["c1,c3,v2,v3"] * T3["c0,c2,c3,v0,v1,v3"];
        C1["c0,c1"] +=
            (1.0 / 8.0) * H2["c3,c4,v0,c2"] * T2["c1,c2,v1,v2"] * T3["c0,c3,c4,v0,v1,v2"];
        C1["c0,c1"] +=
            (1.0 / 48.0) * H2["c4,c5,c2,c3"] * T3["c0,c2,c3,v0,v1,v2"] * T3["c1,c4,c5,v0,v1,v2"];

        C1["c0,g0"] +=
            (-1.0 / 24.0) * H1["c1,g0"] * T3["c0,c2,c3,v0,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
        C1["c0,g0"] +=
            (-1.0 / 4.0) * H2["v0,c2,g0,c1"] * T2["c2,c3,v1,v2"] * T3["c0,c1,c3,v0,v1,v2"];
        C1["c0,g0"] +=
            (1.0 / 8.0) * H2["v1,v2,g0,v0"] * T2["c1,c2,v0,v3"] * T3["c0,c1,c2,v1,v2,v3"];
        C1["c0,g0"] +=
            (1.0 / 8.0) * H2["v1,c1,g0,v0"] * T3["c1,c2,c3,v0,v2,v3"] * T3["c0,c2,c3,v1,v2,v3"];
        C1["c0,g0"] +=
            (1.0 / 24.0) * H2["c2,c3,g0,c1"] * T3["c0,c1,c4,v0,v1,v2"] * T3["c2,c3,c4,v0,v1,v2"];

        C1["g0,c0"] += (1.0 / 8.0) * H1["v0,g0"] * T2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        C1["g0,c0"] +=
            (-1.0 / 4.0) * H2["v0,c2,g0,c1"] * T2["c2,c3,v1,v2"] * T3["c0,c1,c3,v0,v1,v2"];
        C1["g0,c0"] +=
            (1.0 / 8.0) * H2["v1,v2,g0,v0"] * T2["c1,c2,v0,v3"] * T3["c0,c1,c2,v1,v2,v3"];

        C1["g0,g1"] +=
            (1.0 / 8.0) * H2["g1,v0,g0,c0"] * T2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];

        C1["g0,v0"] +=
            (-1.0 / 24.0) * H1["v1,g0"] * T3["c0,c1,c2,v0,v2,v3"] * T3["c0,c1,c2,v1,v2,v3"];
        C1["g0,v0"] +=
            (1.0 / 8.0) * H2["v1,c1,g0,c0"] * T3["c1,c2,c3,v0,v2,v3"] * T3["c0,c2,c3,v1,v2,v3"];
        C1["g0,v0"] +=
            (-1.0 / 24.0) * H2["v2,v3,g0,v1"] * T3["c0,c1,c2,v0,v1,v4"] * T3["c0,c1,c2,v2,v3,v4"];
        C1["g0,v0"] +=
            (1.0 / 4.0) * H2["v2,c0,g0,v1"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v3"];
        C1["g0,v0"] +=
            (1.0 / 8.0) * H2["c1,c2,g0,c0"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];

        C1["v0,c0"] +=
            (-1.0 / 2.0) * H2["v2,c2,v1,c1"] * T2["c2,c3,v1,v3"] * T3["c0,c1,c3,v0,v2,v3"];
        C1["v0,c0"] +=
            (1.0 / 16.0) * H2["v3,v4,v1,v2"] * T2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v3,v4"];
        C1["v0,c0"] +=
            (1.0 / 16.0) * H2["c3,c4,c1,c2"] * T2["c1,c2,v1,v2"] * T3["c0,c3,c4,v0,v1,v2"];

        C1["v0,g0"] += (-1.0 / 8.0) * H1["c0,g0"] * T2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        C1["v0,g0"] +=
            (1.0 / 4.0) * H2["v2,c0,g0,v1"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v3"];
        C1["v0,g0"] +=
            (1.0 / 8.0) * H2["c1,c2,g0,c0"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];

        C1["v0,v1"] +=
            (-1.0 / 12.0) * H1["v3,v2"] * T3["c0,c1,c2,v0,v2,v4"] * T3["c0,c1,c2,v1,v3,v4"];
        C1["v0,v1"] +=
            (1.0 / 8.0) * H1["c1,c0"] * T3["c0,c2,c3,v0,v2,v3"] * T3["c1,c2,c3,v1,v2,v3"];
        C1["v0,v1"] +=
            (1.0 / 4.0) * H2["v3,c1,v2,c0"] * T3["c1,c2,c3,v0,v2,v4"] * T3["c0,c2,c3,v1,v3,v4"];
        C1["v0,v1"] +=
            (-1.0 / 48.0) * H2["v4,v5,v2,v3"] * T3["c0,c1,c2,v0,v2,v3"] * T3["c0,c1,c2,v1,v4,v5"];
        C1["v0,v1"] +=
            (-1.0 / 8.0) * H2["v4,c0,v2,v3"] * T2["c1,c2,v0,v4"] * T3["c0,c1,c2,v1,v2,v3"];
        C1["v0,v1"] +=
            (-1.0 / 4.0) * H2["c1,c2,v2,c0"] * T2["c0,c3,v0,v3"] * T3["c1,c2,c3,v1,v2,v3"];
        C1["v0,v1"] +=
            (-1.0 / 16.0) * H2["c2,c3,c0,c1"] * T3["c0,c1,c4,v0,v2,v3"] * T3["c2,c3,c4,v1,v2,v3"];

        C2["c0,c1,c2,c3"] +=
            (1.0 / 4.0) * H1["v1,v0"] * T3["c0,c1,c4,v0,v2,v3"] * T3["c2,c3,c4,v1,v2,v3"];
        C2["c0,c1,c2,c3"] +=
            (-1.0 / 12.0) * H1["c5,c4"] * T3["c0,c1,c4,v0,v1,v2"] * T3["c2,c3,c5,v0,v1,v2"];
        C2["c0,c1,c2,c3"] +=
            (-1.0 / 4.0) * H2["v1,c5,v0,c4"] * T3["c0,c1,c5,v0,v2,v3"] * T3["c2,c3,c4,v1,v2,v3"];
        C2["c0,c1,c2,c3"] +=
            (1.0 / 8.0) * H2["v2,v3,v0,v1"] * T3["c0,c1,c4,v0,v1,v4"] * T3["c2,c3,c4,v2,v3,v4"];
        C2["c0,c1,c2,c3"] +=
            (-1.0 / 4.0) * H2["v2,c4,v0,v1"] * T2["c2,c3,v2,v3"] * T3["c0,c1,c4,v0,v1,v3"];

        C2["c0,c1,g0,g1"] +=
            (-1.0 / 4.0) * H2["v0,c2,g0,g1"] * T2["c2,c3,v1,v2"] * T3["c0,c1,c3,v0,v1,v2"];
        C2["c0,c1,g0,g1"] +=
            (1.0 / 24.0) * H2["c2,c3,g0,g1"] * T3["c0,c1,c4,v0,v1,v2"] * T3["c2,c3,c4,v0,v1,v2"];

        C2["c0,c1,v0,v1"] +=
            (1.0 / 2.0) * H2["c2,c3,v2,v3"] * T1["c2,v2"] * T3["c0,c1,c3,v0,v1,v3"];

        C2["g0,g1,c0,c1"] +=
            (1.0 / 4.0) * H2["v0,v1,g0,g1"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        C2["g0,g1,c0,c1"] += (-1.0 / 2.0) * H2["v0,c2,g0,g1"] * T1["c2,v1"] * T2["c0,c1,v0,v1"];

        C2["g0,g1,v0,v1"] +=
            (1.0 / 24.0) * H2["v2,v3,g0,g1"] * T3["c0,c1,c2,v0,v1,v4"] * T3["c0,c1,c2,v2,v3,v4"];
        C2["g0,g1,v0,v1"] +=
            (-1.0 / 4.0) * H2["v2,c0,g0,g1"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v3"];

        C2["v0,v1,c0,c1"] += (1.0 / 2.0) * H1["v3,v2"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v3"];
        C2["v0,v1,c0,c1"] += (-1.0 / 2.0) * H1["c3,c2"] * T1["c2,v2"] * T3["c0,c1,c3,v0,v1,v2"];
        C2["v0,v1,c0,c1"] +=
            (-1.0 / 2.0) * H2["v3,c3,v2,c2"] * T1["c3,v2"] * T3["c0,c1,c2,v0,v1,v3"];

        C2["v0,v1,g0,g1"] += (-1.0 / 2.0) * H2["v2,c0,g0,g1"] * T1["c1,v2"] * T2["c0,c1,v0,v1"];
        C2["v0,v1,g0,g1"] +=
            (1.0 / 4.0) * H2["c0,c1,g0,g1"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];

        C2["v0,v1,v2,v3"] +=
            (1.0 / 12.0) * H1["v5,v4"] * T3["c0,c1,c2,v0,v1,v4"] * T3["c0,c1,c2,v2,v3,v5"];
        C2["v0,v1,v2,v3"] +=
            (-1.0 / 4.0) * H1["c1,c0"] * T3["c0,c2,c3,v0,v1,v4"] * T3["c1,c2,c3,v2,v3,v4"];
        C2["v0,v1,v2,v3"] +=
            (-1.0 / 4.0) * H2["v5,c1,v4,c0"] * T3["c1,c2,c3,v0,v1,v4"] * T3["c0,c2,c3,v2,v3,v5"];
        C2["v0,v1,v2,v3"] +=
            (-1.0 / 4.0) * H2["c1,c2,v4,c0"] * T2["c0,c3,v0,v1"] * T3["c1,c2,c3,v2,v3,v4"];
        C2["v0,v1,v2,v3"] +=
            (1.0 / 8.0) * H2["c2,c3,c0,c1"] * T3["c0,c1,c4,v0,v1,v4"] * T3["c2,c3,c4,v2,v3,v4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccgc"});
        temp["c0,c1,g0,c2"] +=
            (1.0 / 12.0) * H1["c3,g0"] * T3["c0,c1,c4,v0,v1,v2"] * T3["c2,c3,c4,v0,v1,v2"];
        temp["c0,c1,g0,c2"] +=
            (-1.0 / 4.0) * H2["v0,c4,g0,c3"] * T2["c2,c4,v1,v2"] * T3["c0,c1,c3,v0,v1,v2"];
        temp["c0,c1,g0,c2"] +=
            (1.0 / 4.0) * H2["v1,v2,g0,v0"] * T2["c2,c3,v0,v3"] * T3["c0,c1,c3,v1,v2,v3"];
        temp["c0,c1,g0,c2"] +=
            (-1.0 / 4.0) * H2["v1,c3,g0,v0"] * T3["c2,c3,c4,v0,v2,v3"] * T3["c0,c1,c4,v1,v2,v3"];
        temp["c0,c1,g0,c2"] +=
            (-1.0 / 24.0) * H2["c4,c5,g0,c3"] * T3["c0,c1,c3,v0,v1,v2"] * T3["c2,c4,c5,v0,v1,v2"];
        C2["c0,c1,g0,c2"] += temp["c0,c1,g0,c2"];
        C2["c0,c1,c2,g0"] -= temp["c0,c1,g0,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccgv"});
        temp["c0,c1,g0,v0"] += (1.0 / 2.0) * H2["v1,v2,g0,c2"] * T1["c2,v1"] * T2["c0,c1,v0,v2"];
        C2["c0,c1,g0,v0"] += temp["c0,c1,g0,v0"];
        C2["c0,c1,v0,g0"] -= temp["c0,c1,g0,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
        temp["c0,c1,v0,v1"] +=
            (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
        C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
        C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
        temp["c0,c1,v0,v1"] +=
            (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
        C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
        C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gccc"});
        temp["g0,c0,c1,c2"] +=
            (1.0 / 4.0) * H1["v0,g0"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
        temp["g0,c0,c1,c2"] +=
            (-1.0 / 4.0) * H2["v0,c4,g0,c3"] * T2["c0,c4,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
        temp["g0,c0,c1,c2"] +=
            (1.0 / 4.0) * H2["v1,v2,g0,v0"] * T2["c0,c3,v0,v3"] * T3["c1,c2,c3,v1,v2,v3"];
        temp["g0,c0,c1,c2"] +=
            (-1.0 / 2.0) * H2["v1,c3,g0,v0"] * T2["c0,c3,v0,v2"] * T2["c1,c2,v1,v2"];
        C2["g0,c0,c1,c2"] += temp["g0,c0,c1,c2"];
        C2["c0,g0,c1,c2"] -= temp["g0,c0,c1,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgg"});
        temp["g0,c0,g1,g2"] +=
            (-1.0 / 4.0) * H2["g1,g2,g0,c1"] * T2["c0,c2,v0,v1"] * T2["c1,c2,v0,v1"];
        C2["g0,c0,g1,g2"] += temp["g0,c0,g1,g2"];
        C2["c0,g0,g1,g2"] -= temp["g0,c0,g1,g2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcvv"});
        temp["g0,c0,v0,v1"] += (-1.0 / 2.0) * H2["c1,c2,g0,v2"] * T1["c1,v2"] * T2["c0,c2,v0,v1"];
        C2["g0,c0,v0,v1"] += temp["g0,c0,v0,v1"];
        C2["c0,g0,v0,v1"] -= temp["g0,c0,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggc"});
        temp["g0,g1,g2,c0"] += (1.0 / 2.0) * H2["g2,v0,g0,g1"] * T1["c1,v1"] * T2["c0,c1,v0,v1"];
        C2["g0,g1,g2,c0"] += temp["g0,g1,g2,c0"];
        C2["g0,g1,c0,g2"] -= temp["g0,g1,g2,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggv"});
        temp["g0,g1,g2,v0"] +=
            (-1.0 / 4.0) * H2["g2,v1,g0,g1"] * T2["c0,c1,v0,v2"] * T2["c0,c1,v1,v2"];
        C2["g0,g1,g2,v0"] += temp["g0,g1,g2,v0"];
        C2["g0,g1,v0,g2"] -= temp["g0,g1,g2,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggvc"});
        temp["g0,g1,v0,c0"] +=
            (1.0 / 8.0) * H2["v1,v2,g0,g1"] * T2["c1,c2,v0,v3"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["g0,g1,v0,c0"] +=
            (1.0 / 2.0) * H2["v1,c1,g0,g1"] * T2["c1,c2,v0,v2"] * T2["c0,c2,v1,v2"];
        C2["g0,g1,v0,c0"] += temp["g0,g1,v0,c0"];
        C2["g0,g1,c0,v0"] -= temp["g0,g1,v0,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvcc"});
        temp["g0,v0,c0,c1"] += (-1.0 / 2.0) * H1["v1,g0"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["g0,v0,c0,c1"] +=
            (1.0 / 2.0) * H2["v1,c3,g0,c2"] * T1["c3,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["g0,v0,c0,c1"] +=
            (-1.0 / 4.0) * H2["v2,v3,g0,v1"] * T1["c2,v1"] * T3["c0,c1,c2,v0,v2,v3"];
        temp["g0,v0,c0,c1"] += (-1.0 / 2.0) * H2["v2,c2,g0,v1"] * T1["c2,v1"] * T2["c0,c1,v0,v2"];
        C2["g0,v0,c0,c1"] += temp["g0,v0,c0,c1"];
        C2["v0,g0,c0,c1"] -= temp["g0,v0,c0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvgg"});
        temp["g0,v0,g1,g2"] += (-1.0 / 2.0) * H2["g1,g2,g0,c0"] * T1["c1,v1"] * T2["c0,c1,v0,v1"];
        C2["g0,v0,g1,g2"] += temp["g0,v0,g1,g2"];
        C2["v0,g0,g1,g2"] -= temp["g0,v0,g1,g2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvvv"});
        temp["g0,v0,v1,v2"] +=
            (-1.0 / 12.0) * H1["v3,g0"] * T3["c0,c1,c2,v0,v3,v4"] * T3["c0,c1,c2,v1,v2,v4"];
        temp["g0,v0,v1,v2"] +=
            (1.0 / 4.0) * H2["v3,c1,g0,c0"] * T3["c0,c2,c3,v0,v3,v4"] * T3["c1,c2,c3,v1,v2,v4"];
        temp["g0,v0,v1,v2"] +=
            (-1.0 / 24.0) * H2["v4,v5,g0,v3"] * T3["c0,c1,c2,v0,v4,v5"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["g0,v0,v1,v2"] +=
            (-1.0 / 4.0) * H2["v4,c0,g0,v3"] * T2["c1,c2,v0,v4"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["g0,v0,v1,v2"] +=
            (-1.0 / 4.0) * H2["c1,c2,g0,c0"] * T2["c0,c3,v0,v3"] * T3["c1,c2,c3,v1,v2,v3"];
        C2["g0,v0,v1,v2"] += temp["g0,v0,v1,v2"];
        C2["v0,g0,v1,v2"] -= temp["g0,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
        temp["v0,c0,c1,c2"] +=
            (1.0 / 2.0) * H1["v2,v1"] * T2["c0,c3,v1,v3"] * T3["c1,c2,c3,v0,v2,v3"];
        temp["v0,c0,c1,c2"] +=
            (-1.0 / 4.0) * H1["c4,c3"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c4,v0,v1,v2"];
        temp["v0,c0,c1,c2"] +=
            (-1.0 / 2.0) * H2["v2,c4,v1,c3"] * T2["c0,c4,v1,v3"] * T3["c1,c2,c3,v0,v2,v3"];
        temp["v0,c0,c1,c2"] +=
            (1.0 / 8.0) * H2["v3,v4,v1,v2"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v3,v4"];
        temp["v0,c0,c1,c2"] +=
            (1.0 / 4.0) * H2["v3,c3,v1,v2"] * T2["c1,c2,v0,v3"] * T2["c0,c3,v1,v2"];
        C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
        C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcgg"});
        temp["v0,c0,g0,g1"] +=
            (1.0 / 2.0) * H2["v1,c1,g0,g1"] * T2["c1,c2,v0,v2"] * T2["c0,c2,v1,v2"];
        temp["v0,c0,g0,g1"] +=
            (1.0 / 8.0) * H2["c1,c2,g0,g1"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
        C2["v0,c0,g0,g1"] += temp["v0,c0,g0,g1"];
        C2["c0,v0,g0,g1"] -= temp["v0,c0,g0,g1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvgc"});
        temp["v0,v1,g0,c0"] += (1.0 / 2.0) * H1["c1,g0"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["v0,v1,g0,c0"] += (-1.0 / 2.0) * H2["v2,c2,g0,c1"] * T1["c1,v2"] * T2["c0,c2,v0,v1"];
        temp["v0,v1,g0,c0"] +=
            (-1.0 / 2.0) * H2["v3,c1,g0,v2"] * T1["c2,v3"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["v0,v1,g0,c0"] +=
            (-1.0 / 4.0) * H2["c2,c3,g0,c1"] * T1["c1,v2"] * T3["c0,c2,c3,v0,v1,v2"];
        C2["v0,v1,g0,c0"] += temp["v0,v1,g0,c0"];
        C2["v0,v1,c0,g0"] -= temp["v0,v1,g0,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvgv"});
        temp["v0,v1,g0,v2"] +=
            (1.0 / 4.0) * H1["c0,g0"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v3"];
        temp["v0,v1,g0,v2"] +=
            (1.0 / 2.0) * H2["v3,c1,g0,c0"] * T2["c1,c2,v0,v1"] * T2["c0,c2,v2,v3"];
        temp["v0,v1,g0,v2"] +=
            (-1.0 / 4.0) * H2["v4,c0,g0,v3"] * T2["c1,c2,v2,v4"] * T3["c0,c1,c2,v0,v1,v3"];
        temp["v0,v1,g0,v2"] +=
            (-1.0 / 4.0) * H2["c1,c2,g0,c0"] * T2["c0,c3,v2,v3"] * T3["c1,c2,c3,v0,v1,v3"];
        C2["v0,v1,g0,v2"] += temp["v0,v1,g0,v2"];
        C2["v0,v1,v2,g0"] -= temp["v0,v1,g0,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
        temp["v0,v1,v2,c0"] +=
            (1.0 / 4.0) * H1["v4,v3"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v4"];
        temp["v0,v1,v2,c0"] +=
            (-1.0 / 2.0) * H1["c2,c1"] * T2["c1,c3,v2,v3"] * T3["c0,c2,c3,v0,v1,v3"];
        temp["v0,v1,v2,c0"] +=
            (-1.0 / 2.0) * H2["v4,c2,v3,c1"] * T2["c2,c3,v2,v3"] * T3["c0,c1,c3,v0,v1,v4"];
        temp["v0,v1,v2,c0"] +=
            (1.0 / 4.0) * H2["c2,c3,v3,c1"] * T2["c0,c1,v0,v1"] * T2["c2,c3,v2,v3"];
        temp["v0,v1,v2,c0"] +=
            (1.0 / 8.0) * H2["c3,c4,c1,c2"] * T2["c1,c2,v2,v3"] * T3["c0,c3,c4,v0,v1,v3"];
        C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
        C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgc"});
        temp["g0,c0,g1,c1"] +=
            (-1.0 / 4.0) * H2["g1,v0,g0,c2"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
        temp["g0,c0,g1,c1"] +=
            (1.0 / 2.0) * H2["g1,v1,g0,v0"] * T2["c0,c2,v0,v2"] * T2["c1,c2,v1,v2"];
        temp["g0,c0,g1,c1"] +=
            (-1.0 / 4.0) * H2["g1,c3,g0,c2"] * T2["c0,c3,v0,v1"] * T2["c1,c2,v0,v1"];
        C2["g0,c0,g1,c1"] += temp["g0,c0,g1,c1"];
        C2["g0,c0,c1,g1"] -= temp["g0,c0,g1,c1"];
        C2["c0,g0,g1,c1"] -= temp["g0,c0,g1,c1"];
        C2["c0,g0,c1,g1"] += temp["g0,c0,g1,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgv"});
        temp["g0,c0,g1,v0"] +=
            (1.0 / 8.0) * H2["g1,v1,g0,c1"] * T3["c0,c2,c3,v0,v2,v3"] * T3["c1,c2,c3,v1,v2,v3"];
        temp["g0,c0,g1,v0"] +=
            (1.0 / 4.0) * H2["g1,v2,g0,v1"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v3"];
        temp["g0,c0,g1,v0"] +=
            (-1.0 / 4.0) * H2["g1,c2,g0,c1"] * T2["c1,c3,v1,v2"] * T3["c0,c2,c3,v0,v1,v2"];
        C2["g0,c0,g1,v0"] += temp["g0,c0,g1,v0"];
        C2["g0,c0,v0,g1"] -= temp["g0,c0,g1,v0"];
        C2["c0,g0,g1,v0"] -= temp["g0,c0,g1,v0"];
        C2["c0,g0,v0,g1"] += temp["g0,c0,g1,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcvc"});
        temp["g0,c0,v0,c1"] +=
            (-1.0 / 8.0) * H1["v1,g0"] * T3["c0,c2,c3,v0,v2,v3"] * T3["c1,c2,c3,v1,v2,v3"];
        temp["g0,c0,v0,c1"] +=
            (1.0 / 4.0) * H2["v1,c3,g0,c2"] * T3["c0,c3,c4,v0,v2,v3"] * T3["c1,c2,c4,v1,v2,v3"];
        temp["g0,c0,v0,c1"] +=
            (-1.0 / 8.0) * H2["v2,v3,g0,v1"] * T3["c0,c2,c3,v0,v1,v4"] * T3["c1,c2,c3,v2,v3,v4"];
        temp["g0,c0,v0,c1"] +=
            (-1.0 / 2.0) * H2["v2,c2,g0,v1"] * T2["c1,c3,v2,v3"] * T3["c0,c2,c3,v0,v1,v3"];
        temp["g0,c0,v0,c1"] +=
            (-1.0 / 8.0) * H2["c3,c4,g0,c2"] * T2["c1,c2,v1,v2"] * T3["c0,c3,c4,v0,v1,v2"];
        C2["g0,c0,v0,c1"] += temp["g0,c0,v0,c1"];
        C2["g0,c0,c1,v0"] -= temp["g0,c0,v0,c1"];
        C2["c0,g0,v0,c1"] -= temp["g0,c0,v0,c1"];
        C2["c0,g0,c1,v0"] += temp["g0,c0,v0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvgc"});
        temp["g0,v0,g1,c0"] +=
            (1.0 / 2.0) * H2["g1,v1,g0,c1"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        temp["g0,v0,g1,c0"] += (1.0 / 2.0) * H2["g1,v2,g0,v1"] * T1["c1,v1"] * T2["c0,c1,v0,v2"];
        temp["g0,v0,g1,c0"] += (-1.0 / 2.0) * H2["g1,c2,g0,c1"] * T1["c2,v1"] * T2["c0,c1,v0,v1"];
        C2["g0,v0,g1,c0"] += temp["g0,v0,g1,c0"];
        C2["g0,v0,c0,g1"] -= temp["g0,v0,g1,c0"];
        C2["v0,g0,g1,c0"] -= temp["g0,v0,g1,c0"];
        C2["v0,g0,c0,g1"] += temp["g0,v0,g1,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvgv"});
        temp["g0,v0,g1,v1"] +=
            (1.0 / 4.0) * H2["g1,v2,g0,c0"] * T2["c1,c2,v1,v3"] * T3["c0,c1,c2,v0,v2,v3"];
        temp["g0,v0,g1,v1"] +=
            (-1.0 / 4.0) * H2["g1,v3,g0,v2"] * T2["c0,c1,v0,v3"] * T2["c0,c1,v1,v2"];
        temp["g0,v0,g1,v1"] +=
            (1.0 / 2.0) * H2["g1,c1,g0,c0"] * T2["c0,c2,v0,v2"] * T2["c1,c2,v1,v2"];
        C2["g0,v0,g1,v1"] += temp["g0,v0,g1,v1"];
        C2["g0,v0,v1,g1"] -= temp["g0,v0,g1,v1"];
        C2["v0,g0,g1,v1"] -= temp["g0,v0,g1,v1"];
        C2["v0,g0,v1,g1"] += temp["g0,v0,g1,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gvvc"});
        temp["g0,v0,v1,c0"] +=
            (-1.0 / 4.0) * H1["v2,g0"] * T2["c1,c2,v1,v3"] * T3["c0,c1,c2,v0,v2,v3"];
        temp["g0,v0,v1,c0"] +=
            (1.0 / 2.0) * H2["v2,c2,g0,c1"] * T2["c2,c3,v1,v3"] * T3["c0,c1,c3,v0,v2,v3"];
        temp["g0,v0,v1,c0"] +=
            (-1.0 / 8.0) * H2["v3,v4,g0,v2"] * T2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v3,v4"];
        temp["g0,v0,v1,c0"] +=
            (1.0 / 2.0) * H2["v3,c1,g0,v2"] * T2["c0,c2,v0,v3"] * T2["c1,c2,v1,v2"];
        temp["g0,v0,v1,c0"] +=
            (1.0 / 4.0) * H2["c2,c3,g0,c1"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v2"];
        C2["g0,v0,v1,c0"] += temp["g0,v0,v1,c0"];
        C2["g0,v0,c0,v1"] -= temp["g0,v0,v1,c0"];
        C2["v0,g0,v1,c0"] -= temp["g0,v0,v1,c0"];
        C2["v0,g0,c0,v1"] += temp["g0,v0,v1,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcgc"});
        temp["v0,c0,g0,c1"] +=
            (1.0 / 4.0) * H1["c2,g0"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
        temp["v0,c0,g0,c1"] +=
            (1.0 / 2.0) * H2["v1,c3,g0,c2"] * T2["c1,c3,v0,v2"] * T2["c0,c2,v1,v2"];
        temp["v0,c0,g0,c1"] +=
            (-1.0 / 4.0) * H2["v2,v3,g0,v1"] * T2["c1,c2,v0,v1"] * T2["c0,c2,v2,v3"];
        temp["v0,c0,g0,c1"] +=
            (-1.0 / 2.0) * H2["v2,c2,g0,v1"] * T2["c0,c3,v2,v3"] * T3["c1,c2,c3,v0,v1,v3"];
        temp["v0,c0,g0,c1"] +=
            (-1.0 / 8.0) * H2["c3,c4,g0,c2"] * T2["c0,c2,v1,v2"] * T3["c1,c3,c4,v0,v1,v2"];
        C2["v0,c0,g0,c1"] += temp["v0,c0,g0,c1"];
        C2["v0,c0,c1,g0"] -= temp["v0,c0,g0,c1"];
        C2["c0,v0,g0,c1"] -= temp["v0,c0,g0,c1"];
        C2["c0,v0,c1,g0"] += temp["v0,c0,g0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcgv"});
        temp["v0,c0,g0,v1"] +=
            (-1.0 / 8.0) * H1["c1,g0"] * T3["c1,c2,c3,v0,v2,v3"] * T3["c0,c2,c3,v1,v2,v3"];
        temp["v0,c0,g0,v1"] +=
            (1.0 / 2.0) * H2["v2,c2,g0,c1"] * T2["c2,c3,v0,v3"] * T3["c0,c1,c3,v1,v2,v3"];
        temp["v0,c0,g0,v1"] +=
            (-1.0 / 8.0) * H2["v3,v4,g0,v2"] * T2["c1,c2,v0,v2"] * T3["c0,c1,c2,v1,v3,v4"];
        temp["v0,c0,g0,v1"] +=
            (1.0 / 4.0) * H2["v3,c1,g0,v2"] * T3["c1,c2,c3,v0,v2,v4"] * T3["c0,c2,c3,v1,v3,v4"];
        temp["v0,c0,g0,v1"] +=
            (1.0 / 8.0) * H2["c2,c3,g0,c1"] * T3["c2,c3,c4,v0,v2,v3"] * T3["c0,c1,c4,v1,v2,v3"];
        C2["v0,c0,g0,v1"] += temp["v0,c0,g0,v1"];
        C2["v0,c0,v1,g0"] -= temp["v0,c0,g0,v1"];
        C2["c0,v0,g0,v1"] -= temp["v0,c0,g0,v1"];
        C2["c0,v0,v1,g0"] += temp["v0,c0,g0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
        temp["v0,c0,v1,c1"] +=
            (-1.0 / 4.0) * H1["v3,v2"] * T3["c1,c2,c3,v0,v2,v4"] * T3["c0,c2,c3,v1,v3,v4"];
        temp["v0,c0,v1,c1"] +=
            (1.0 / 4.0) * H1["c3,c2"] * T3["c1,c2,c4,v0,v2,v3"] * T3["c0,c3,c4,v1,v2,v3"];
        temp["v0,c0,v1,c1"] +=
            (1.0 / 2.0) * H2["v3,c3,v2,c2"] * T3["c1,c3,c4,v0,v2,v4"] * T3["c0,c2,c4,v1,v3,v4"];
        temp["v0,c0,v1,c1"] +=
            (-1.0 / 16.0) * H2["v4,v5,v2,v3"] * T3["c1,c2,c3,v0,v2,v3"] * T3["c0,c2,c3,v1,v4,v5"];
        temp["v0,c0,v1,c1"] +=
            (1.0 / 4.0) * H2["v4,c2,v2,v3"] * T2["c1,c3,v0,v4"] * T3["c0,c2,c3,v1,v2,v3"];
        temp["v0,c0,v1,c1"] +=
            (1.0 / 4.0) * H2["c3,c4,v2,c2"] * T2["c1,c2,v0,v3"] * T3["c0,c3,c4,v1,v2,v3"];
        temp["v0,c0,v1,c1"] +=
            (-1.0 / 16.0) * H2["c4,c5,c2,c3"] * T3["c1,c2,c3,v0,v2,v3"] * T3["c0,c4,c5,v1,v2,v3"];
        C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
        C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
        C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];
    }

    // add T dagger
    C0 *= 2.0;

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gg"});
    temp["pq"] = C1["pq"];
    C1["pq"] += temp["qp"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggg"});
    temp["pqrs"] = C2["pqrs"];
    C2["pqrs"] += temp["rspq"];
}

} // namespace forte
