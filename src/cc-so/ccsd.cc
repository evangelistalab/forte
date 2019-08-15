#include "cc.h"

using namespace psi;

namespace forte {

void CC_SO::compute_ccsd_amp(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                             BlockedTensor& T2, double& C0, BlockedTensor& C1, BlockedTensor& C2) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    BlockedTensor temp;

    C0 += 1.0 * H1["c0,v0"] * T1["c0,v0"];
    C0 += (1.0 / 4.0) * H2["c0,c1,v0,v1"] * T2["c0,c1,v0,v1"];
    C0 += (1.0 / 2.0) * H2["c0,c1,v0,v1"] * T1["c0,v0"] * T1["c1,v1"];

    C1["c0,v0"] += 1.0 * H1["v1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += 1.0 * H1["c1,v1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,c0"] * T1["c1,v0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v1,v2,v0,c1"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * T1["c1,v1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v1,c0"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,v1"] * T1["c0,v1"] * T1["c1,v0"];
    C1["c0,v0"] += 1.0 * H2["v1,v2,v0,c1"] * T1["c0,v1"] * T1["c1,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["c1,c2,v1,v2"] * T1["c0,v1"] * T2["c1,c2,v0,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["c1,c2,v1,v2"] * T1["c1,v0"] * T2["c0,c2,v1,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,v2"] * T1["c1,v1"] * T2["c0,c2,v0,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,c0"] * T1["c1,v0"] * T1["c2,v1"];
    C1["c0,v0"] += -1.0 * H2["c1,c2,v1,v2"] * T1["c0,v1"] * T1["c1,v0"] * T1["c2,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * T1["c0,v2"] * T2["c1,c2,v1,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,c0,v0,c2"] * T1["c1,v2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,v2,v3"] * T2["c0,c2,v0,v2"] * T2["c1,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * T1["c2,v0"] * T2["c1,c3,v1,v2"];
    temp["c0,c1,v0,v1"] +=
        (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * T1["c0,v2"] * T1["c1,v3"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c2,v0"] * T2["c1,c3,v1,v3"];
    temp["c0,c1,v0,v1"] +=
        (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * T1["c1,v2"] * T1["c2,v0"] * T1["c3,v1"];
    temp["c0,c1,v0,v1"] +=
        (1.0 / 4.0) * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c1,v3"] * T1["c2,v0"] * T1["c3,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["c2,c3,v2,v3"] * T2["c0,c1,v2,v3"] * T2["c2,c3,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,c0"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,v2"] * T1["c0,v2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * T1["c0,v2"] * T1["c1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,v3"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * T1["c1,v2"] * T2["c2,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * T1["c2,v2"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] +=
        (1.0 / 4.0) * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c1,v3"] * T2["c2,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c2,v3"] * T2["c1,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H1["v2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,v2"] * T1["c2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * T1["c2,v1"] * T2["c0,c1,v2,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * T1["c2,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,v3"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * T1["c2,v0"] * T1["c3,v1"];
    temp["c0,c1,v0,v1"] +=
        (1.0 / 4.0) * H2["c2,c3,v2,v3"] * T1["c2,v0"] * T1["c3,v1"] * T2["c0,c1,v2,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c2,c3,v2,v3"] * T1["c2,v0"] * T1["c3,v2"] * T2["c0,c1,v1,v3"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
}

} // namespace forte
