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

void MRDSRG_SO::comm_H_A_3_sr(double factor, BlockedTensor& H1, BlockedTensor& H2,
                              BlockedTensor& H3, BlockedTensor& T1, BlockedTensor& T2,
                              BlockedTensor& T3, double& C0, BlockedTensor& C1, BlockedTensor& C2,
                              BlockedTensor& C3) {
    int level = 3;
    if (foptions_->get_str("CORR_LEVEL") == "LDSRG3_2")
        level = 2;
    if (foptions_->get_str("CORR_LEVEL") == "LDSRG3_1")
        level = 1;

    C0 = 0.0;
    C1.zero();
    C2.zero();
    C3.zero();
    BlockedTensor temp;

    C0 += 1.0 * H1["c0,v0"] * T1["c0,v0"];
    C0 += (1.0 / 4.0) * H2["c0,c1,v0,v1"] * T2["c0,c1,v0,v1"];
    C0 += (1.0 / 36.0) * H3["c0,c1,c2,v0,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];

    C1["c0,g0"] += 1.0 * H1["v0,g0"] * T1["c0,v0"];
    C1["c0,g0"] += (1.0 / 2.0) * H2["v0,v1,g0,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += 1.0 * H1["c1,v1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["g0,g1"] += 1.0 * H2["g1,c0,g0,v0"] * T1["c0,v0"];
    C1["g0,g1"] += (1.0 / 4.0) * H3["g1,c0,c1,g0,v0,v1"] * T2["c0,c1,v0,v1"];
    C1["g0,v0"] += -1.0 * H1["c0,g0"] * T1["c0,v0"];
    C1["g0,v0"] += (-1.0 / 2.0) * H2["c0,c1,g0,v1"] * T2["c0,c1,v0,v1"];

    C2["c0,c1,g0,g1"] += (1.0 / 2.0) * H2["v0,v1,g0,g1"] * T2["c0,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["g0,g1,g2,g3"] += 1.0 * H3["g2,g3,c0,g0,g1,v0"] * T1["c0,v0"];
    C2["g0,g1,v0,v1"] += (1.0 / 2.0) * H2["c0,c1,g0,g1"] * T2["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccgv"});
    temp["c0,c1,g0,v0"] += -1.0 * H1["v1,g0"] * T2["c0,c1,v0,v1"];
    temp["c0,c1,g0,v0"] += (-1.0 / 2.0) * H2["v1,v2,g0,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,c1,g0,v0"] += temp["c0,c1,g0,v0"];
    C2["c0,c1,v0,g0"] -= temp["c0,c1,g0,v0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgg"});
    temp["g0,c0,g1,g2"] += 1.0 * H2["g1,g2,g0,v0"] * T1["c0,v0"];
    temp["g0,c0,g1,g2"] += (1.0 / 2.0) * H3["g1,g2,c1,g0,v0,v1"] * T2["c0,c1,v0,v1"];
    C2["g0,c0,g1,g2"] += temp["g0,c0,g1,g2"];
    C2["c0,g0,g1,g2"] -= temp["g0,c0,g1,g2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcvv"});
    temp["g0,c0,v0,v1"] += 1.0 * H1["c1,g0"] * T2["c0,c1,v0,v1"];
    temp["g0,c0,v0,v1"] += (1.0 / 2.0) * H2["c1,c2,g0,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["g0,c0,v0,v1"] += temp["g0,c0,v0,v1"];
    C2["c0,g0,v0,v1"] -= temp["g0,c0,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggv"});
    temp["g0,g1,g2,v0"] += -1.0 * H2["g2,c0,g0,g1"] * T1["c0,v0"];
    temp["g0,g1,g2,v0"] += (-1.0 / 2.0) * H3["g2,c0,c1,g0,g1,v1"] * T2["c0,c1,v0,v1"];
    C2["g0,g1,g2,v0"] += temp["g0,g1,g2,v0"];
    C2["g0,g1,v0,g2"] -= temp["g0,g1,g2,v0"];

    if (level == 1) {
        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgv"});
        temp["g0,c0,g1,v0"] += 1.0 * H2["g1,c1,g0,v1"] * T2["c0,c1,v0,v1"];
        C2["g0,c0,g1,v0"] += temp["g0,c0,g1,v0"];
        C2["g0,c0,v0,g1"] -= temp["g0,c0,g1,v0"];
        C2["c0,g0,g1,v0"] -= temp["g0,c0,g1,v0"];
        C2["c0,g0,v0,g1"] += temp["g0,c0,g1,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
        temp["c0,c1,c2,v3,v0,v1"] = 1.0 * H1["v2,v3"] * T3["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v2,v0,v1"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v2,v0,v1"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v2,v0,v1"];

        temp["c3,c0,c1,v0,v1,v2"] = -1.0 * H1["c2,c3"] * T3["c2,c0,c1,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c2,c0,c1,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c2,c0,c1,v0,v1,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c2,c0,c1,v0,v1,v2"];
    } else {
        C1["c0,g0"] += (1.0 / 12.0) * H3["v0,v1,v2,g0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
        C1["g0,v0"] += (-1.0 / 12.0) * H3["c0,c1,c2,g0,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];

        C2["c0,c1,g0,g1"] += (1.0 / 6.0) * H3["v0,v1,v2,g0,g1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
        C2["g0,g1,v0,v1"] += (1.0 / 6.0) * H3["c0,c1,c2,g0,g1,v2"] * T3["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gcgv"});
        temp["g0,c0,g1,v0"] += 1.0 * H2["g1,c1,g0,v1"] * T2["c0,c1,v0,v1"];
        temp["g0,c0,g1,v0"] += (1.0 / 4.0) * H3["g1,c1,c2,g0,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        C2["g0,c0,g1,v0"] += temp["g0,c0,g1,v0"];
        C2["g0,c0,v0,g1"] -= temp["g0,c0,g1,v0"];
        C2["c0,g0,g1,v0"] -= temp["g0,c0,g1,v0"];
        C2["c0,g0,v0,g1"] += temp["g0,c0,g1,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccggv"});
        temp["c0,c1,c2,g0,g1,v0"] += (1.0 / 2.0) * H2["v1,v2,g0,g1"] * T3["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,g0,g1,v0"] += temp["c0,c1,c2,g0,g1,v0"];
        C3["c0,c1,c2,g0,v0,g1"] -= temp["c0,c1,c2,g0,g1,v0"];
        C3["c0,c1,c2,v0,g0,g1"] += temp["c0,c1,c2,g0,g1,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccgvv"});
        temp["c0,c1,c2,g0,v0,v1"] += 1.0 * H1["v2,g0"] * T3["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,g0,v0,v1"] += temp["c0,c1,c2,g0,v0,v1"];
        C3["c0,c1,c2,v0,g0,v1"] -= temp["c0,c1,c2,g0,v0,v1"];
        C3["c0,c1,c2,v0,v1,g0"] += temp["c0,c1,c2,g0,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gccggg"});
        temp["g0,c0,c1,g1,g2,g3"] += (1.0 / 2.0) * H3["g1,g2,g3,g0,v0,v1"] * T2["c0,c1,v0,v1"];
        C3["g0,c0,c1,g1,g2,g3"] += temp["g0,c0,c1,g1,g2,g3"];
        C3["c0,g0,c1,g1,g2,g3"] -= temp["g0,c0,c1,g1,g2,g3"];
        C3["c0,c1,g0,g1,g2,g3"] += temp["g0,c0,c1,g1,g2,g3"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gccvvv"});
        temp["g0,c0,c1,v0,v1,v2"] += -1.0 * H1["c2,g0"] * T3["c2,c0,c1,v0,v1,v2"];
        C3["g0,c0,c1,v0,v1,v2"] += temp["g0,c0,c1,v0,v1,v2"];
        C3["c0,g0,c1,v0,v1,v2"] -= temp["g0,c0,c1,v0,v1,v2"];
        C3["c0,c1,g0,v0,v1,v2"] += temp["g0,c0,c1,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggcggg"});
        temp["g0,g1,c0,g2,g3,g4"] += 1.0 * H3["g2,g3,g4,g0,g1,v0"] * T1["c0,v0"];
        C3["g0,g1,c0,g2,g3,g4"] += temp["g0,g1,c0,g2,g3,g4"];
        C3["g0,c0,g1,g2,g3,g4"] -= temp["g0,g1,c0,g2,g3,g4"];
        C3["c0,g0,g1,g2,g3,g4"] += temp["g0,g1,c0,g2,g3,g4"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggcvvv"});
        temp["g0,g1,c0,v0,v1,v2"] += (1.0 / 2.0) * H2["c1,c2,g0,g1"] * T3["c0,c1,c2,v0,v1,v2"];
        C3["g0,g1,c0,v0,v1,v2"] += temp["g0,g1,c0,v0,v1,v2"];
        C3["g0,c0,g1,v0,v1,v2"] -= temp["g0,g1,c0,v0,v1,v2"];
        C3["c0,g0,g1,v0,v1,v2"] += temp["g0,g1,c0,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggggv"});
        temp["g0,g1,g2,g3,g4,v0"] += -1.0 * H3["g3,g4,c0,g0,g1,g2"] * T1["c0,v0"];
        C3["g0,g1,g2,g3,g4,v0"] += temp["g0,g1,g2,g3,g4,v0"];
        C3["g0,g1,g2,g3,v0,g4"] -= temp["g0,g1,g2,g3,g4,v0"];
        C3["g0,g1,g2,v0,g3,g4"] += temp["g0,g1,g2,g3,g4,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggggvv"});
        temp["g0,g1,g2,g3,v0,v1"] += (1.0 / 2.0) * H3["g3,c0,c1,g0,g1,g2"] * T2["c0,c1,v0,v1"];
        C3["g0,g1,g2,g3,v0,v1"] += temp["g0,g1,g2,g3,v0,v1"];
        C3["g0,g1,g2,v0,g3,v1"] -= temp["g0,g1,g2,g3,v0,v1"];
        C3["g0,g1,g2,v0,v1,g3"] += temp["g0,g1,g2,g3,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gccgvv"});
        temp["g0,c0,c1,g1,v0,v1"] += 1.0 * H2["g1,c2,g0,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        C3["g0,c0,c1,g1,v0,v1"] += temp["g0,c0,c1,g1,v0,v1"];
        C3["g0,c0,c1,v0,g1,v1"] -= temp["g0,c0,c1,g1,v0,v1"];
        C3["g0,c0,c1,v0,v1,g1"] += temp["g0,c0,c1,g1,v0,v1"];
        C3["c0,g0,c1,g1,v0,v1"] -= temp["g0,c0,c1,g1,v0,v1"];
        C3["c0,g0,c1,v0,g1,v1"] += temp["g0,c0,c1,g1,v0,v1"];
        C3["c0,g0,c1,v0,v1,g1"] -= temp["g0,c0,c1,g1,v0,v1"];
        C3["c0,c1,g0,g1,v0,v1"] += temp["g0,c0,c1,g1,v0,v1"];
        C3["c0,c1,g0,v0,g1,v1"] -= temp["g0,c0,c1,g1,v0,v1"];
        C3["c0,c1,g0,v0,v1,g1"] += temp["g0,c0,c1,g1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggcggv"});
        temp["g0,g1,c0,g2,g3,v0"] += 1.0 * H3["g2,g3,c1,g0,g1,v1"] * T2["c0,c1,v0,v1"];
        C3["g0,g1,c0,g2,g3,v0"] += temp["g0,g1,c0,g2,g3,v0"];
        C3["g0,g1,c0,g2,v0,g3"] -= temp["g0,g1,c0,g2,g3,v0"];
        C3["g0,g1,c0,v0,g2,g3"] += temp["g0,g1,c0,g2,g3,v0"];
        C3["g0,c0,g1,g2,g3,v0"] -= temp["g0,g1,c0,g2,g3,v0"];
        C3["g0,c0,g1,g2,v0,g3"] += temp["g0,g1,c0,g2,g3,v0"];
        C3["g0,c0,g1,v0,g2,g3"] -= temp["g0,g1,c0,g2,g3,v0"];
        C3["c0,g0,g1,g2,g3,v0"] += temp["g0,g1,c0,g2,g3,v0"];
        C3["c0,g0,g1,g2,v0,g3"] -= temp["g0,g1,c0,g2,g3,v0"];
        C3["c0,g0,g1,v0,g2,g3"] += temp["g0,g1,c0,g2,g3,v0"];
    }

    if (level == 3) {
        C3["c0,c1,c2,g0,g1,g2"] += (1.0 / 6.0) * H3["v0,v1,v2,g0,g1,g2"] * T3["c0,c1,c2,v0,v1,v2"];
        C3["g0,g1,g2,v0,v1,v2"] += (-1.0 / 6.0) * H3["c0,c1,c2,g0,g1,g2"] * T3["c0,c1,c2,v0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gccggv"});
        temp["g0,c0,c1,g1,g2,v0"] += -1.0 * H2["g1,g2,g0,v1"] * T2["c0,c1,v0,v1"];
        temp["g0,c0,c1,g1,g2,v0"] +=
            (-1.0 / 2.0) * H3["g1,g2,c2,g0,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        C3["g0,c0,c1,g1,g2,v0"] += temp["g0,c0,c1,g1,g2,v0"];
        C3["g0,c0,c1,g1,v0,g2"] -= temp["g0,c0,c1,g1,g2,v0"];
        C3["g0,c0,c1,v0,g1,g2"] += temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,g0,c1,g1,g2,v0"] -= temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,g0,c1,g1,v0,g2"] += temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,g0,c1,v0,g1,g2"] -= temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,c1,g0,g1,g2,v0"] += temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,c1,g0,g1,v0,g2"] -= temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,c1,g0,v0,g1,g2"] += temp["g0,c0,c1,g1,g2,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggcgvv"});
        temp["g0,g1,c0,g2,v0,v1"] += 1.0 * H2["g2,c1,g0,g1"] * T2["c0,c1,v0,v1"];
        temp["g0,g1,c0,g2,v0,v1"] +=
            (1.0 / 2.0) * H3["g2,c1,c2,g0,g1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
        C3["g0,g1,c0,g2,v0,v1"] += temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,g1,c0,v0,g2,v1"] -= temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,g1,c0,v0,v1,g2"] += temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,c0,g1,g2,v0,v1"] -= temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,c0,g1,v0,g2,v1"] += temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,c0,g1,v0,v1,g2"] -= temp["g0,g1,c0,g2,v0,v1"];
        C3["c0,g0,g1,g2,v0,v1"] += temp["g0,g1,c0,g2,v0,v1"];
        C3["c0,g0,g1,v0,g2,v1"] -= temp["g0,g1,c0,g2,v0,v1"];
        C3["c0,g0,g1,v0,v1,g2"] += temp["g0,g1,c0,g2,v0,v1"];
    } else {
        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gccggv"});
        temp["g0,c0,c1,g1,g2,v0"] += -1.0 * H2["g1,g2,g0,v1"] * T2["c0,c1,v0,v1"];
        C3["g0,c0,c1,g1,g2,v0"] += temp["g0,c0,c1,g1,g2,v0"];
        C3["g0,c0,c1,g1,v0,g2"] -= temp["g0,c0,c1,g1,g2,v0"];
        C3["g0,c0,c1,v0,g1,g2"] += temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,g0,c1,g1,g2,v0"] -= temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,g0,c1,g1,v0,g2"] += temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,g0,c1,v0,g1,g2"] -= temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,c1,g0,g1,g2,v0"] += temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,c1,g0,g1,v0,g2"] -= temp["g0,c0,c1,g1,g2,v0"];
        C3["c0,c1,g0,v0,g1,g2"] += temp["g0,c0,c1,g1,g2,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ggcgvv"});
        temp["g0,g1,c0,g2,v0,v1"] += 1.0 * H2["c1,g2,g0,g1"] * T2["c1,c0,v0,v1"];
        C3["g0,g1,c0,g2,v0,v1"] += temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,g1,c0,v0,g2,v1"] -= temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,g1,c0,v0,v1,g2"] += temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,c0,g1,g2,v0,v1"] -= temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,c0,g1,v0,g2,v1"] += temp["g0,g1,c0,g2,v0,v1"];
        C3["g0,c0,g1,v0,v1,g2"] -= temp["g0,g1,c0,g2,v0,v1"];
        C3["c0,g0,g1,g2,v0,v1"] += temp["g0,g1,c0,g2,v0,v1"];
        C3["c0,g0,g1,v0,g2,v1"] -= temp["g0,g1,c0,g2,v0,v1"];
        C3["c0,g0,g1,v0,v1,g2"] += temp["g0,g1,c0,g2,v0,v1"];
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
