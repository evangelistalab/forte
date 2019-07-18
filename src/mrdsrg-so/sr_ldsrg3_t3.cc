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

void MRDSRG_SO::direct_t3() {
    int level = 3;
    if (foptions_->get_str("CORR_LEVEL") == "LDSRG3_2")
        level = 2;
    if (foptions_->get_str("CORR_LEVEL") == "LDSRG3_1")
        level = 1;

    BlockedTensor C3 = ambit::BlockedTensor::build(ambit::CoreTensor, "T3 new", {"cccvvv"});
    BlockedTensor temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});

    if (ncomm_3body_ == 1) {
        temp["c0,c1,c2,v0,v1,v2"] = 1.0 * F["v3,v0"] * T3["c0,c1,c2,v1,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        temp["c0,c1,c2,v0,v1,v2"] = -1.0 * F["c3,c0"] * T3["c1,c2,c3,v0,v1,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

        temp["c0,c1,c2,v0,v1,v2"] = 1.0 * V["v3,c0,v0,v1"] * T2["c1,c2,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

        temp["c0,c1,c2,v0,v1,v2"] = 1.0 * V["c0,c1,v0,c3"] * T2["c2,c3,v1,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        if (level > 1) {
            temp["c0,c1,c2,v0,v1,v2"] = (1.0 / 2.0) * V["v3,v4,v0,v1"] * T3["c0,c1,c2,v2,v3,v4"];
            C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

            temp["c0,c1,c2,v0,v1,v2"] = (1.0 / 2.0) * V["c3,c4,c0,c1"] * T3["c2,c3,c4,v0,v1,v2"];
            C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

            temp["c0,c1,c2,v0,v1,v2"] = -1.0 * V["v3,c0,v0,c3"] * T3["c1,c2,c3,v1,v2,v3"];
            C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c1,c0,c2,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c1,c0,c2,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c1,c2,c0,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c1,c2,c0,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        }
    } else {
        double factor = (level == 1 ? -0.5 : -1.0);

        temp.zero();
        temp["c0,c1,c2,v0,v1,v2"] += 1.0 * F["v3,v0"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (-1.0 / 2.0) * F["c3,v0"] * T1["c3,v3"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["c0,c1,c2,v0,v1,v2"] += factor * F["c3,v3"] * T1["c3,v0"] * T3["c0,c1,c2,v1,v2,v3"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (-1.0 / 2.0) * V["v3,v4,v0,c3"] * T1["c3,v3"] * T3["c0,c1,c2,v1,v2,v4"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (1.0 / 2.0) * V["v4,c3,v0,v3"] * T1["c3,v3"] * T3["c0,c1,c2,v1,v2,v4"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (1.0 / 4.0) * V["c3,c4,v0,v3"] * T2["c3,c4,v3,v4"] * T3["c0,c1,c2,v1,v2,v4"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (1.0 / 4.0) * V["c3,c4,v3,v4"] * T2["c3,c4,v0,v3"] * T3["c0,c1,c2,v1,v2,v4"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        temp.zero();
        temp["c0,c1,c2,v0,v1,v2"] += -1.0 * F["c3,c0"] * T3["c1,c2,c3,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (-1.0 / 2.0) * F["c0,v3"] * T1["c3,v3"] * T3["c1,c2,c3,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,v2"] += factor * F["c3,v3"] * T1["c0,v3"] * T3["c1,c2,c3,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (1.0 / 4.0) * V["c0,c3,v3,v4"] * T2["c3,c4,v3,v4"] * T3["c1,c2,c4,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (1.0 / 2.0) * V["c0,c4,v3,c3"] * T1["c4,v3"] * T3["c1,c2,c3,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (1.0 / 4.0) * V["c3,c4,v3,v4"] * T2["c0,c3,v3,v4"] * T3["c1,c2,c4,v0,v1,v2"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (-1.0 / 2.0) * V["c3,c4,v3,c0"] * T1["c3,v3"] * T3["c1,c2,c4,v0,v1,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

        if (level > 1) {
            temp.zero();
            temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * V["v3,v4,v0,v1"] * T3["c0,c1,c2,v2,v3,v4"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (-1.0 / 2.0) * V["v3,c3,v0,v1"] * T1["c3,v4"] * T3["c0,c1,c2,v2,v3,v4"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (1.0 / 8.0) * V["c3,c4,v0,v1"] * T2["c3,c4,v3,v4"] * T3["c0,c1,c2,v2,v3,v4"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (1.0 / 4.0) * V["c3,c4,v3,v4"] * T2["c3,c4,v0,v1"] * T3["c0,c1,c2,v2,v3,v4"];
            C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

            temp.zero();
            temp["c0,c1,c2,v0,v1,v2"] += (1.0 / 2.0) * V["c3,c4,c0,c1"] * T3["c2,c3,c4,v0,v1,v2"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (1.0 / 8.0) * V["c0,c1,v3,v4"] * T2["c3,c4,v3,v4"] * T3["c2,c3,c4,v0,v1,v2"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (-1.0 / 2.0) * V["c0,c1,v3,c3"] * T1["c4,v3"] * T3["c2,c3,c4,v0,v1,v2"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (1.0 / 4.0) * V["c3,c4,v3,v4"] * T2["c0,c1,v3,v4"] * T3["c2,c3,c4,v0,v1,v2"];
            C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];

            temp.zero();
            temp["c0,c1,c2,v0,v1,v2"] +=
                (-1.0 / 2.0) * V["v3,v4,v0,c3"] * T1["c3,v1"] * T3["c0,c1,c2,v2,v3,v4"];
            C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c1,c2,v2,v1,v0"] -= temp["c0,c1,c2,v0,v1,v2"];

            temp.zero();
            temp["c0,c1,c2,v0,v1,v2"] +=
                (-1.0 / 2.0) * V["c3,c4,v3,c0"] * T1["c1,v3"] * T3["c2,c3,c4,v0,v1,v2"];
            C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
            C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
            C3["c2,c1,c0,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        }

        temp.zero();
        temp["c0,c1,c2,v0,v1,v2"] += 1.0 * V["v3,c0,v0,v1"] * T2["c1,c2,v2,v3"];
        temp["c0,c1,c2,v0,v1,v2"] += 1.0 * F["c3,v3"] * T2["c0,c3,v0,v1"] * T2["c1,c2,v2,v3"];
        temp["c0,c1,c2,v0,v1,v2"] += factor * V["v3,v4,v0,v1"] * T1["c0,v3"] * T2["c1,c2,v2,v4"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (1.0 / 2.0) * V["c0,c3,v0,v1"] * T1["c3,v3"] * T2["c1,c2,v2,v3"];
        if (level == 3) {
            temp["c0,c1,c2,v0,v1,v2"] +=
                (1.0 / 4.0) * V["c0,c3,v0,v1"] * T2["c3,c4,v3,v4"] * T3["c1,c2,c4,v2,v3,v4"];
        }
        temp["c0,c1,c2,v0,v1,v2"] += (level == 3 ? 0.5 : 0.25) * V["c3,c4,v3,v4"] *
                                     T2["c0,c3,v0,v1"] * T3["c1,c2,c4,v2,v3,v4"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (level != 1 ? 0.5 : 0.25) * V["c3,c4,v3,c0"] * T2["c3,c4,v0,v1"] * T2["c1,c2,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

        temp.zero();
        temp["c0,c1,c2,v0,v1,v2"] +=
            (level != 1 ? 0.5 : 0.25) * V["v3,v4,v0,c3"] * T2["c0,c3,v1,v2"] * T2["c1,c2,v3,v4"];
        if (level > 1) {
            temp["c0,c1,c2,v0,v1,v2"] += -1.0 * V["v3,c0,v0,c3"] * T3["c1,c2,c3,v1,v2,v3"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                1.0 * V["v3,v4,v0,c3"] * T1["c0,v3"] * T3["c1,c2,c3,v1,v2,v4"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (-1.0 / 2.0) * V["v4,c0,v0,v3"] * T1["c3,v3"] * T3["c1,c2,c3,v1,v2,v4"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (1.0 / 2.0) * V["c0,c3,v0,v3"] * T2["c3,c4,v3,v4"] * T3["c1,c2,c4,v1,v2,v4"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                (-1.0 / 2.0) * V["c0,c4,v0,c3"] * T1["c4,v3"] * T3["c1,c2,c3,v1,v2,v3"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                1.0 * V["c3,c4,v3,v4"] * T2["c0,c3,v0,v3"] * T3["c1,c2,c4,v1,v2,v4"];
            temp["c0,c1,c2,v0,v1,v2"] +=
                1.0 * V["c3,c4,v3,c0"] * T1["c3,v0"] * T3["c1,c2,c4,v1,v2,v3"];
        }
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        temp.zero();
        temp["c0,c1,c2,v0,v1,v2"] += 1.0 * V["c0,c1,v0,c3"] * T2["c2,c3,v1,v2"];
        temp["c0,c1,c2,v0,v1,v2"] +=
            (1.0 / 2.0) * V["c0,c1,v0,v3"] * T1["c3,v3"] * T2["c2,c3,v1,v2"];
        if (level == 3) {
            temp["c0,c1,c2,v0,v1,v2"] +=
                (1.0 / 4.0) * V["c0,c1,v0,v3"] * T2["c3,c4,v3,v4"] * T3["c2,c3,c4,v1,v2,v4"];
        }
        temp["c0,c1,c2,v0,v1,v2"] += (level == 3 ? 0.5 : 0.25) * V["c3,c4,v3,v4"] *
                                     T2["c0,c1,v0,v3"] * T3["c2,c3,c4,v1,v2,v4"];
        temp["c0,c1,c2,v0,v1,v2"] += factor * V["c3,c4,c0,c1"] * T1["c3,v0"] * T2["c2,c4,v1,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

        temp["c0,c1,c2,v0,v1,v2"] = factor * V["v3,v4,v0,c3"] * T2["c0,c1,v1,v3"] * T2["c2,c3,v2,v4"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v1,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v2,v1,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v2,v1,v0"] -= temp["c0,c1,c2,v0,v1,v2"];

        temp["c0,c1,c2,v0,v1,v2"] = factor * V["v3,c0,v0,c3"] * T1["c1,v3"] * T2["c2,c3,v1,v2"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c1,c0,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c1,c0,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c1,c0,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];

        temp["c0,c1,c2,v0,v1,v2"] = factor * V["v3,c0,v0,c3"] * T1["c3,v1"] * T2["c1,c2,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v1,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v2,v1,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v2,v1,v0"] -= temp["c0,c1,c2,v0,v1,v2"];

        temp["c0,c1,c2,v0,v1,v2"] = factor * V["c3,c4,v3,c0"] * T2["c1,c3,v0,v1"] * T2["c2,c4,v2,v3"];
        C3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c0,c2,c1,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c0,c1,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c1,c0,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c1,c0,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
        C3["c2,c1,c0,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    }

    C3.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] + Fd[i[2]] - Fd[i[3]] - Fd[i[4]] -
                Fd[i[5]]);
    });

    temp["ijkabc"] = T3["ijkabc"];

    T3.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double delta = Fd[i[0]] + Fd[i[1]] + Fd[i[2]] - Fd[i[3]] - Fd[i[4]] - Fd[i[5]];
        value *= std::exp(-s_ * delta * delta);
    });

    C3["ijkabc"] -= T3["ijkabc"];
    rms_t3 = C3.norm();

    T3["ijkabc"] = C3["ijkabc"] + temp["ijkabc"];

    // norm and max
    T3max = T3.norm(0), T3norm = T3.norm();
}

} // namespace forte
