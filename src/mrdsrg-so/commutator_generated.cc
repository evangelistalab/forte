#include "ambit/blocked_tensor.h"

#include "mrdsrg-so/mrdsrg_so.h"

using namespace psi;

namespace forte {
void MRDSRG_SO::commutator_H_A_3_sr(double factor, BlockedTensor& H1, BlockedTensor& H2,
                                    BlockedTensor& H3, BlockedTensor& T1, BlockedTensor& T2,
                                    BlockedTensor& T3, double& C0, BlockedTensor& C1,
                                    BlockedTensor& C2, BlockedTensor& C3) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    C3.zero();

    C0 += 1.0 * H1["v0,c0"] * T1["c0,v0"];
    C0 += (1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["c0,c1,v0,v1"];
    C0 += (1.0 / 36.0) * H3["v0,v1,v2,c0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];

    C1["g1,g0"] += 1.0 * H2["g1,v0,g0,c0"] * T1["c0,v0"];
    C1["g1,g0"] += (1.0 / 4.0) * H3["g1,v0,v1,g0,c0,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,g0"] += 1.0 * H1["v0,g0"] * T1["c0,v0"];
    C1["c0,g0"] += (1.0 / 2.0) * H2["v0,v1,g0,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,g0"] += (1.0 / 12.0) * H3["v0,v1,v2,g0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["g0,v0"] += -1.0 * H1["g0,c0"] * T1["c0,v0"];
    C1["g0,v0"] += (-1.0 / 2.0) * H2["g0,v1,c0,c1"] * T2["c0,c1,v0,v1"];
    C1["g0,v0"] += (-1.0 / 12.0) * H3["g0,v1,v2,c0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["c0,v0"] += 1.0 * H1["v1,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["v1,v2,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];

    C2["g2,g3,g0,g1"] += 1.0 * H3["g2,g3,v0,g0,g1,c0"] * T1["c0,v0"];
    auto temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gcgg"});
    temp["g2,c0,g0,g1"] += 1.0 * H2["g2,v0,g0,g1"] * T1["c0,v0"];
    temp["g2,c0,g0,g1"] += (1.0 / 2.0) * H3["g2,v0,v1,g0,g1,c1"] * T2["c0,c1,v0,v1"];
    C2["c0,g2,g0,g1"] -= temp["g2,c0,g0,g1"];
    C2["g2,c0,g0,g1"] += temp["g2,c0,g0,g1"];
    C2["c0,c1,g0,g1"] += (1.0 / 2.0) * H2["v0,v1,g0,g1"] * T2["c0,c1,v0,v1"];
    C2["c0,c1,g0,g1"] += (1.0 / 6.0) * H3["v0,v1,v2,g0,g1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gggv"});
    temp["g1,g2,g0,v0"] += -1.0 * H2["g1,g2,g0,c0"] * T1["c0,v0"];
    temp["g1,g2,g0,v0"] += (-1.0 / 2.0) * H3["g1,g2,v1,g0,c0,c1"] * T2["c0,c1,v0,v1"];
    C2["g1,g2,g0,v0"] += temp["g1,g2,g0,v0"];
    C2["g1,g2,v0,g0"] -= temp["g1,g2,g0,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gcgv"});
    temp["g1,c0,g0,v0"] += 1.0 * H2["g1,v1,g0,c1"] * T2["c0,c1,v0,v1"];
    temp["g1,c0,g0,v0"] += (1.0 / 4.0) * H3["g1,v1,v2,g0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,g1,g0,v0"] -= temp["g1,c0,g0,v0"];
    C2["g1,c0,g0,v0"] += temp["g1,c0,g0,v0"];
    C2["c0,g1,v0,g0"] += temp["g1,c0,g0,v0"];
    C2["g1,c0,v0,g0"] -= temp["g1,c0,g0,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ccgv"});
    temp["c0,c1,g0,v0"] += -1.0 * H1["v1,g0"] * T2["c0,c1,v0,v1"];
    temp["c0,c1,g0,v0"] += (-1.0 / 2.0) * H2["v1,v2,g0,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,c1,g0,v0"] += temp["c0,c1,g0,v0"];
    C2["c0,c1,v0,g0"] -= temp["c0,c1,g0,v0"];
    C2["g0,g1,v0,v1"] += (1.0 / 2.0) * H2["g0,g1,c0,c1"] * T2["c0,c1,v0,v1"];
    C2["g0,g1,v0,v1"] += (1.0 / 6.0) * H3["g0,g1,v2,c0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gcvv"});
    temp["g0,c0,v0,v1"] += 1.0 * H1["g0,c1"] * T2["c0,c1,v0,v1"];
    temp["g0,c0,v0,v1"] += (1.0 / 2.0) * H2["g0,v2,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,g0,v0,v1"] -= temp["g0,c0,v0,v1"];
    C2["g0,c0,v0,v1"] += temp["g0,c0,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H1["v2,c2"] * T3["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ggcggg"});
    temp["g3,g4,c0,g0,g1,g2"] += 1.0 * H3["g3,g4,v0,g0,g1,g2"] * T1["c0,v0"];
    C3["c0,g3,g4,g0,g1,g2"] += temp["g3,g4,c0,g0,g1,g2"];
    C3["g3,c0,g4,g0,g1,g2"] -= temp["g3,g4,c0,g0,g1,g2"];
    C3["g3,g4,c0,g0,g1,g2"] += temp["g3,g4,c0,g0,g1,g2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gccggg"});
    temp["g3,c0,c1,g0,g1,g2"] += (1.0 / 2.0) * H3["g3,v0,v1,g0,g1,g2"] * T2["c0,c1,v0,v1"];
    C3["c0,c1,g3,g0,g1,g2"] += temp["g3,c0,c1,g0,g1,g2"];
    C3["c0,g3,c1,g0,g1,g2"] -= temp["g3,c0,c1,g0,g1,g2"];
    C3["g3,c0,c1,g0,g1,g2"] += temp["g3,c0,c1,g0,g1,g2"];
    C3["c0,c1,c2,g0,g1,g2"] += (1.0 / 6.0) * H3["v0,v1,v2,g0,g1,g2"] * T3["c0,c1,c2,v0,v1,v2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gggggv"});
    temp["g2,g3,g4,g0,g1,v0"] += -1.0 * H3["g2,g3,g4,g0,g1,c0"] * T1["c0,v0"];
    C3["g2,g3,g4,g0,g1,v0"] += temp["g2,g3,g4,g0,g1,v0"];
    C3["g2,g3,g4,g0,v0,g1"] -= temp["g2,g3,g4,g0,g1,v0"];
    C3["g2,g3,g4,v0,g0,g1"] += temp["g2,g3,g4,g0,g1,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ggcggv"});
    temp["g2,g3,c0,g0,g1,v0"] += 1.0 * H3["g2,g3,v1,g0,g1,c1"] * T2["c0,c1,v0,v1"];
    C3["c0,g2,g3,g0,g1,v0"] += temp["g2,g3,c0,g0,g1,v0"];
    C3["g2,c0,g3,g0,g1,v0"] -= temp["g2,g3,c0,g0,g1,v0"];
    C3["g2,g3,c0,g0,g1,v0"] += temp["g2,g3,c0,g0,g1,v0"];
    C3["c0,g2,g3,g0,v0,g1"] -= temp["g2,g3,c0,g0,g1,v0"];
    C3["g2,c0,g3,g0,v0,g1"] += temp["g2,g3,c0,g0,g1,v0"];
    C3["g2,g3,c0,g0,v0,g1"] -= temp["g2,g3,c0,g0,g1,v0"];
    C3["c0,g2,g3,v0,g0,g1"] += temp["g2,g3,c0,g0,g1,v0"];
    C3["g2,c0,g3,v0,g0,g1"] -= temp["g2,g3,c0,g0,g1,v0"];
    C3["g2,g3,c0,v0,g0,g1"] += temp["g2,g3,c0,g0,g1,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gccggv"});
    temp["g2,c0,c1,g0,g1,v0"] += -1.0 * H2["g2,v1,g0,g1"] * T2["c0,c1,v0,v1"];
    temp["g2,c0,c1,g0,g1,v0"] += (-1.0 / 2.0) * H3["g2,v1,v2,g0,g1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,g2,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,g0,g1,v0"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,c1,g2,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,g0,v0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,c1,g2,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,v0,g0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"cccggv"});
    temp["c0,c1,c2,g0,g1,v0"] += (1.0 / 2.0) * H2["v1,v2,g0,g1"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,g0,g1,v0"] += temp["c0,c1,c2,g0,g1,v0"];
    C3["c0,c1,c2,g0,v0,g1"] -= temp["c0,c1,c2,g0,g1,v0"];
    C3["c0,c1,c2,v0,g0,g1"] += temp["c0,c1,c2,g0,g1,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ggggvv"});
    temp["g1,g2,g3,g0,v0,v1"] += (1.0 / 2.0) * H3["g1,g2,g3,g0,c0,c1"] * T2["c0,c1,v0,v1"];
    C3["g1,g2,g3,g0,v0,v1"] += temp["g1,g2,g3,g0,v0,v1"];
    C3["g1,g2,g3,v0,g0,v1"] -= temp["g1,g2,g3,g0,v0,v1"];
    C3["g1,g2,g3,v0,v1,g0"] += temp["g1,g2,g3,g0,v0,v1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ggcgvv"});
    temp["g1,g2,c0,g0,v0,v1"] += 1.0 * H2["g1,g2,g0,c1"] * T2["c0,c1,v0,v1"];
    temp["g1,g2,c0,g0,v0,v1"] += (1.0 / 2.0) * H3["g1,g2,v2,g0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,g1,g2,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,g0,v0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["c0,g1,g2,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,v0,g0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["c0,g1,g2,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,v0,v1,g0"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gccgvv"});
    temp["g1,c0,c1,g0,v0,v1"] += 1.0 * H2["g1,v2,g0,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,g1,g0,v0,v1"] += temp["g1,c0,c1,g0,v0,v1"];
    C3["c0,g1,c1,g0,v0,v1"] -= temp["g1,c0,c1,g0,v0,v1"];
    C3["g1,c0,c1,g0,v0,v1"] += temp["g1,c0,c1,g0,v0,v1"];
    C3["c0,c1,g1,v0,g0,v1"] -= temp["g1,c0,c1,g0,v0,v1"];
    C3["c0,g1,c1,v0,g0,v1"] += temp["g1,c0,c1,g0,v0,v1"];
    C3["g1,c0,c1,v0,g0,v1"] -= temp["g1,c0,c1,g0,v0,v1"];
    C3["c0,c1,g1,v0,v1,g0"] += temp["g1,c0,c1,g0,v0,v1"];
    C3["c0,g1,c1,v0,v1,g0"] -= temp["g1,c0,c1,g0,v0,v1"];
    C3["g1,c0,c1,v0,v1,g0"] += temp["g1,c0,c1,g0,v0,v1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"cccgvv"});
    temp["c0,c1,c2,g0,v0,v1"] += 1.0 * H1["v2,g0"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,g0,v0,v1"] += temp["c0,c1,c2,g0,v0,v1"];
    C3["c0,c1,c2,v0,g0,v1"] -= temp["c0,c1,c2,g0,v0,v1"];
    C3["c0,c1,c2,v0,v1,g0"] += temp["c0,c1,c2,g0,v0,v1"];
    C3["g0,g1,g2,v0,v1,v2"] += (-1.0 / 6.0) * H3["g0,g1,g2,c0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ggcvvv"});
    temp["g0,g1,c0,v0,v1,v2"] += (1.0 / 2.0) * H2["g0,g1,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,g0,g1,v0,v1,v2"] += temp["g0,g1,c0,v0,v1,v2"];
    C3["g0,c0,g1,v0,v1,v2"] -= temp["g0,g1,c0,v0,v1,v2"];
    C3["g0,g1,c0,v0,v1,v2"] += temp["g0,g1,c0,v0,v1,v2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gccvvv"});
    temp["g0,c0,c1,v0,v1,v2"] += -1.0 * H1["g0,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,g0,v0,v1,v2"] += temp["g0,c0,c1,v0,v1,v2"];
    C3["c0,g0,c1,v0,v1,v2"] -= temp["g0,c0,c1,v0,v1,v2"];
    C3["g0,c0,c1,v0,v1,v2"] += temp["g0,c0,c1,v0,v1,v2"];

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

void MRDSRG_SO::commutator_H_A_3_sr_1(double factor, BlockedTensor& H1, BlockedTensor& H2,
                                      BlockedTensor& H3, BlockedTensor& T1, BlockedTensor& T2,
                                      BlockedTensor& T3, double& C0, BlockedTensor& C1,
                                      BlockedTensor& C2, BlockedTensor& C3) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    C3.zero();

    C0 += 1.0 * H1["v0,c0"] * T1["c0,v0"];
    C0 += (1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["c0,c1,v0,v1"];
    C0 += (1.0 / 36.0) * H3["v0,v1,v2,c0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];

    C1["g1,g0"] += 1.0 * H2["g1,v0,g0,c0"] * T1["c0,v0"];
    C1["g1,g0"] += (1.0 / 4.0) * H3["g1,v0,v1,g0,c0,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,g0"] += 1.0 * H1["v0,g0"] * T1["c0,v0"];
    C1["c0,g0"] += (1.0 / 2.0) * H2["v0,v1,g0,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,g0"] += (1.0 / 12.0) * H3["v0,v1,v2,g0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["g0,v0"] += -1.0 * H1["g0,c0"] * T1["c0,v0"];
    C1["g0,v0"] += (-1.0 / 2.0) * H2["g0,v1,c0,c1"] * T2["c0,c1,v0,v1"];
    C1["g0,v0"] += (-1.0 / 12.0) * H3["g0,v1,v2,c0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["c0,v0"] += 1.0 * H1["v1,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["v1,v2,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];

    C2["g2,g3,g0,g1"] += 1.0 * H3["g2,g3,v0,g0,g1,c0"] * T1["c0,v0"];
    auto temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gcgg"});
    temp["g2,c0,g0,g1"] += 1.0 * H2["g2,v0,g0,g1"] * T1["c0,v0"];
    temp["g2,c0,g0,g1"] += (1.0 / 2.0) * H3["g2,v0,v1,g0,g1,c1"] * T2["c0,c1,v0,v1"];
    C2["c0,g2,g0,g1"] -= temp["g2,c0,g0,g1"];
    C2["g2,c0,g0,g1"] += temp["g2,c0,g0,g1"];
    C2["c0,c1,g0,g1"] += (1.0 / 2.0) * H2["v0,v1,g0,g1"] * T2["c0,c1,v0,v1"];
    C2["c0,c1,g0,g1"] += (1.0 / 6.0) * H3["v0,v1,v2,g0,g1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gggv"});
    temp["g1,g2,g0,v0"] += -1.0 * H2["g1,g2,g0,c0"] * T1["c0,v0"];
    temp["g1,g2,g0,v0"] += (-1.0 / 2.0) * H3["g1,g2,v1,g0,c0,c1"] * T2["c0,c1,v0,v1"];
    C2["g1,g2,g0,v0"] += temp["g1,g2,g0,v0"];
    C2["g1,g2,v0,g0"] -= temp["g1,g2,g0,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gcgv"});
    temp["g1,c0,g0,v0"] += 1.0 * H2["g1,v1,g0,c1"] * T2["c0,c1,v0,v1"];
    temp["g1,c0,g0,v0"] += (1.0 / 4.0) * H3["g1,v1,v2,g0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,g1,g0,v0"] -= temp["g1,c0,g0,v0"];
    C2["g1,c0,g0,v0"] += temp["g1,c0,g0,v0"];
    C2["c0,g1,v0,g0"] += temp["g1,c0,g0,v0"];
    C2["g1,c0,v0,g0"] -= temp["g1,c0,g0,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ccgv"});
    temp["c0,c1,g0,v0"] += -1.0 * H1["v1,g0"] * T2["c0,c1,v0,v1"];
    temp["c0,c1,g0,v0"] += (-1.0 / 2.0) * H2["v1,v2,g0,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,c1,g0,v0"] += temp["c0,c1,g0,v0"];
    C2["c0,c1,v0,g0"] -= temp["c0,c1,g0,v0"];
    C2["g0,g1,v0,v1"] += (1.0 / 2.0) * H2["g0,g1,c0,c1"] * T2["c0,c1,v0,v1"];
    C2["g0,g1,v0,v1"] += (1.0 / 6.0) * H3["g0,g1,v2,c0,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gcvv"});
    temp["g0,c0,v0,v1"] += 1.0 * H1["g0,c1"] * T2["c0,c1,v0,v1"];
    temp["g0,c0,v0,v1"] += (1.0 / 2.0) * H2["g0,v2,c1,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,g0,v0,v1"] -= temp["g0,c0,v0,v1"];
    C2["g0,c0,v0,v1"] += temp["g0,c0,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H1["v2,c2"] * T3["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gccggv"});
    temp["g2,c0,c1,g0,g1,v0"] += -1.0 * H2["g2,v1,g0,g1"] * T2["c0,c1,v0,v1"];
    C3["c0,c1,g2,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,g0,g1,v0"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,c1,g2,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,g0,v0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,c1,g2,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,v0,g0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ggcgvv"});
    temp["g1,g2,c0,g0,v0,v1"] += 1.0 * H2["g1,g2,g0,c1"] * T2["c0,c1,v0,v1"];
    C3["c0,g1,g2,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,g0,v0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["c0,g1,g2,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,v0,g0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["c0,g1,g2,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,v0,v1,g0"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"cccgvv"});
    temp["c0,c1,c2,g0,v0,v1"] += 1.0 * H1["v2,g0"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,c2,g0,v0,v1"] += temp["c0,c1,c2,g0,v0,v1"];
    C3["c0,c1,c2,v0,g0,v1"] -= temp["c0,c1,c2,g0,v0,v1"];
    C3["c0,c1,c2,v0,v1,g0"] += temp["c0,c1,c2,g0,v0,v1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gccvvv"});
    temp["g0,c0,c1,v0,v1,v2"] += -1.0 * H1["g0,c2"] * T3["c0,c1,c2,v0,v1,v2"];
    C3["c0,c1,g0,v0,v1,v2"] += temp["g0,c0,c1,v0,v1,v2"];
    C3["c0,g0,c1,v0,v1,v2"] -= temp["g0,c0,c1,v0,v1,v2"];
    C3["g0,c0,c1,v0,v1,v2"] += temp["g0,c0,c1,v0,v1,v2"];


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

void MRDSRG_SO::commutator_H_A_2(double factor, BlockedTensor& H1, BlockedTensor& H2,
                                 BlockedTensor& T1, BlockedTensor& T2, double& C0,
                                 BlockedTensor& C1, BlockedTensor& C2) {
    C0 = 0.0;
    C1.zero();
    C2.zero();

    C0 += 1.0 * H1["p0,h0"] * T1["h1,p0"] * L1["h0,h1"];
    C0 += (1.0 / 2.0) * H1["p0,a0"] * T2["a1,a2,p0,a3"] * L2["a0,a3,a1,a2"];
    C0 += -1.0 * H1["a0,h0"] * T1["h0,a1"] * L1["a1,a0"];
    C0 += (-1.0 / 2.0) * H1["a0,h0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C0 += (1.0 / 8.0) * H2["p0,p1,a0,a1"] * T2["a2,a3,p0,p1"] * L2["a0,a1,a2,a3"];
    C0 += (-1.0 / 2.0) * H2["p0,a0,a1,a2"] * T1["a3,p0"] * L2["a1,a2,a0,a3"];
    C0 += (-1.0 / 4.0) * H2["p0,a0,a1,a2"] * T2["a3,a4,p0,a5"] * L3["a1,a2,a5,a0,a3,a4"];
    C0 += (-1.0 / 8.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C0 += (1.0 / 2.0) * H2["a0,a1,h0,a2"] * T1["h0,a3"] * L2["a2,a3,a0,a1"];
    C0 += (1.0 / 4.0) * H2["a0,a1,h0,a2"] * T2["h0,a3,a4,a5"] * L3["a2,a4,a5,a0,a1,a3"];
    C0 += (1.0 / 4.0) * H2["p0,p1,h0,h1"] * T2["h2,h3,p0,p1"] * L1["h0,h2"] * L1["h1,h3"];
    C0 += 1.0 * H2["p0,a0,h0,a1"] * T2["h1,a2,p0,a3"] * L1["h0,h1"] * L2["a1,a3,a0,a2"];
    C0 += (-1.0 / 4.0) * H2["p0,a0,a1,a2"] * T2["a3,a4,p0,a5"] * L1["a5,a0"] * L2["a1,a2,a3,a4"];
    C0 += (-1.0 / 4.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C0 += (1.0 / 4.0) * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h1,h2"] * L2["a2,a3,a0,a1"];
    C0 += -1.0 * H2["a0,a1,h0,a2"] * T2["h0,a3,a4,a5"] * L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C0 += (-1.0 / 2.0) * H2["p0,a0,h0,h1"] * T2["h2,h3,p0,a1"] * L1["h0,h2"] * L1["h1,h3"] *
          L1["a1,a0"];
    C0 += (1.0 / 2.0) * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h1,h2"] * L1["a2,a0"] *
          L1["a3,a1"];

    C1["g1,g0"] += 1.0 * H2["g1,p0,g0,h0"] * T1["h1,p0"] * L1["h0,h1"];
    C1["g1,g0"] += (1.0 / 2.0) * H2["g1,p0,g0,a0"] * T2["a1,a2,p0,a3"] * L2["a0,a3,a1,a2"];
    C1["g1,g0"] += -1.0 * H2["g1,a0,g0,h0"] * T1["h0,a1"] * L1["a1,a0"];
    C1["g1,g0"] += (-1.0 / 2.0) * H2["g1,a0,g0,h0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] += 1.0 * H1["p0,g0"] * T1["h0,p0"];
    C1["h0,g0"] += (1.0 / 2.0) * H2["p0,p1,g0,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    C1["h0,g0"] += 1.0 * H2["p0,a0,g0,a1"] * T2["h0,a2,p0,a3"] * L2["a1,a3,a0,a2"];
    C1["h0,g0"] += (1.0 / 4.0) * H2["a0,a1,g0,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] += -1.0 * H2["p0,a0,g0,h1"] * T2["h0,h2,p0,a1"] * L1["h1,h2"] * L1["a1,a0"];
    C1["h0,g0"] += (1.0 / 2.0) * H2["a0,a1,g0,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C1["g0,p0"] += -1.0 * H1["g0,h0"] * T1["h0,p0"];
    C1["g0,p0"] += (-1.0 / 4.0) * H2["g0,p1,a0,a1"] * T2["a2,a3,p0,p1"] * L2["a0,a1,a2,a3"];
    C1["g0,p0"] += (-1.0 / 2.0) * H2["g0,a0,h0,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C1["g0,p0"] += -1.0 * H2["g0,a0,h0,a1"] * T2["h0,a2,p0,a3"] * L2["a1,a3,a0,a2"];
    C1["g0,p0"] += (-1.0 / 2.0) * H2["g0,p1,h0,h1"] * T2["h2,h3,p0,p1"] * L1["h0,h2"] * L1["h1,h3"];
    C1["g0,p0"] += 1.0 * H2["g0,a0,h0,h1"] * T2["h0,h2,p0,a1"] * L1["h1,h2"] * L1["a1,a0"];
    C1["h0,p0"] += 1.0 * H1["p1,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    C1["h0,p0"] += -1.0 * H1["a0,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C1["h0,p0"] += (-1.0 / 2.0) * H2["p1,a0,a1,a2"] * T2["h0,a3,p0,p1"] * L2["a1,a2,a0,a3"];
    C1["h0,p0"] += (1.0 / 2.0) * H2["a0,a1,h1,a2"] * T2["h0,h1,p0,a3"] * L2["a2,a3,a0,a1"];

    auto temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghgg"});
    temp["g2,h0,g0,g1"] += 1.0 * H2["g2,p0,g0,g1"] * T1["h0,p0"];
    C2["g2,h0,g0,g1"] += temp["g2,h0,g0,g1"];
    C2["h0,g2,g0,g1"] -= temp["g2,h0,g0,g1"];
    C2["h0,h1,g0,g1"] += (1.0 / 2.0) * H2["p0,p1,g0,g1"] * T2["h0,h1,p0,p1"];
    C2["h0,h1,g0,g1"] += -1.0 * H2["p0,a0,g0,g1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gggp"});
    temp["g1,g2,g0,p0"] += -1.0 * H2["g1,g2,g0,h0"] * T1["h0,p0"];
    C2["g1,g2,g0,p0"] += temp["g1,g2,g0,p0"];
    C2["g1,g2,p0,g0"] -= temp["g1,g2,g0,p0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghgp"});
    temp["g1,h0,g0,p0"] += 1.0 * H2["g1,p1,g0,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    temp["g1,h0,g0,p0"] += -1.0 * H2["g1,a0,g0,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C2["g1,h0,g0,p0"] += temp["g1,h0,g0,p0"];
    C2["h0,g1,g0,p0"] -= temp["g1,h0,g0,p0"];
    C2["g1,h0,p0,g0"] -= temp["g1,h0,g0,p0"];
    C2["h0,g1,p0,g0"] += temp["g1,h0,g0,p0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"hhgp"});
    temp["h0,h1,g0,p0"] += -1.0 * H1["p1,g0"] * T2["h0,h1,p0,p1"];
    C2["h0,h1,g0,p0"] += temp["h0,h1,g0,p0"];
    C2["h0,h1,p0,g0"] -= temp["h0,h1,g0,p0"];
    C2["g0,g1,p0,p1"] += (-1.0 / 2.0) * H2["g0,g1,h0,h1"] * T2["h0,h1,p0,p1"];
    C2["g0,g1,p0,p1"] += 1.0 * H2["g0,g1,h0,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghpp"});
    temp["g0,h0,p0,p1"] += 1.0 * H1["g0,h1"] * T2["h0,h1,p0,p1"];
    C2["g0,h0,p0,p1"] += temp["g0,h0,p0,p1"];
    C2["h0,g0,p0,p1"] -= temp["g0,h0,p0,p1"];

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

void MRDSRG_SO::commutator_H_A_3(double factor, BlockedTensor& H1, BlockedTensor& H2,
                                 BlockedTensor& H3, BlockedTensor& T1, BlockedTensor& T2,
                                 BlockedTensor& T3, double& C0, BlockedTensor& C1,
                                 BlockedTensor& C2, BlockedTensor& C3) {
    C0 = 0.0;
    C1.zero();
    C2.zero();
    C3.zero();

    C0 += 1.0 * H1["p0,h0"] * T1["h1,p0"] * L1["h0,h1"];
    C0 += (1.0 / 2.0) * H1["p0,a0"] * T2["a1,a2,p0,a3"] * L2["a0,a3,a1,a2"];
    C0 += (1.0 / 12.0) * H1["p0,a0"] * T3["a1,a2,a3,p0,a4,a5"] * L3["a0,a4,a5,a1,a2,a3"];
    C0 += -1.0 * H1["a0,h0"] * T1["h0,a1"] * L1["a1,a0"];
    C0 += (-1.0 / 2.0) * H1["a0,h0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 12.0) * H1["a0,h0"] * T3["h0,a1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (1.0 / 8.0) * H2["p0,p1,a0,a1"] * T2["a2,a3,p0,p1"] * L2["a0,a1,a2,a3"];
    C0 += (1.0 / 24.0) * H2["p0,p1,a0,a1"] * T3["a2,a3,a4,p0,p1,a5"] * L3["a0,a1,a5,a2,a3,a4"];
    C0 += (-1.0 / 2.0) * H2["p0,a0,a1,a2"] * T1["a3,p0"] * L2["a1,a2,a0,a3"];
    C0 += (-1.0 / 4.0) * H2["p0,a0,a1,a2"] * T2["a3,a4,p0,a5"] * L3["a1,a2,a5,a0,a3,a4"];
    C0 += (-1.0 / 8.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 24.0) * H2["a0,a1,h0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (1.0 / 2.0) * H2["a0,a1,h0,a2"] * T1["h0,a3"] * L2["a2,a3,a0,a1"];
    C0 += (1.0 / 4.0) * H2["a0,a1,h0,a2"] * T2["h0,a3,a4,a5"] * L3["a2,a4,a5,a0,a1,a3"];
    C0 +=
        (1.0 / 216.0) * H3["p0,p1,p2,a0,a1,a2"] * T3["a3,a4,a5,p0,p1,p2"] * L3["a0,a1,a2,a3,a4,a5"];
    C0 += (1.0 / 24.0) * H3["p0,p1,a0,a1,a2,a3"] * T2["a4,a5,p0,p1"] * L3["a1,a2,a3,a0,a4,a5"];
    C0 += (1.0 / 12.0) * H3["p0,a0,a1,a2,a3,a4"] * T1["a5,p0"] * L3["a2,a3,a4,a0,a1,a5"];
    C0 += (-1.0 / 216.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] *
          L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 24.0) * H3["a0,a1,a2,h0,h1,a3"] * T2["h0,h1,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 12.0) * H3["a0,a1,a2,h0,a3,a4"] * T1["h0,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (1.0 / 4.0) * H2["p0,p1,h0,h1"] * T2["h2,h3,p0,p1"] * L1["h0,h2"] * L1["h1,h3"];
    C0 +=
        (1.0 / 4.0) * H2["p0,p1,h0,a0"] * T3["h1,a1,a2,p0,p1,a3"] * L1["h0,h1"] * L2["a0,a3,a1,a2"];
    C0 += 1.0 * H2["p0,a0,h0,a1"] * T2["h1,a2,p0,a3"] * L1["h0,h1"] * L2["a1,a3,a0,a2"];
    C0 += (1.0 / 4.0) * H2["p0,a0,h0,a1"] * T3["h1,a2,a3,p0,a4,a5"] * L1["h0,h1"] *
          L3["a1,a4,a5,a0,a2,a3"];
    C0 += (-1.0 / 4.0) * H2["p0,a0,a1,a2"] * T2["a3,a4,p0,a5"] * L1["a5,a0"] * L2["a1,a2,a3,a4"];
    C0 += (-1.0 / 12.0) * H2["p0,a0,a1,a2"] * T3["a3,a4,a5,p0,a6,a7"] * L1["a6,a0"] *
          L3["a1,a2,a7,a3,a4,a5"];
    C0 += (-1.0 / 8.0) * H2["p0,a0,a1,a2"] * T3["a3,a4,a5,p0,a6,a7"] * L2["a1,a2,a4,a5"] *
          L2["a6,a7,a0,a3"];
    C0 += (1.0 / 2.0) * H2["p0,a0,a1,a2"] * T3["a3,a4,a5,p0,a6,a7"] * L2["a1,a6,a3,a4"] *
          L2["a2,a7,a0,a5"];
    C0 += (-1.0 / 4.0) * H2["a0,a1,h0,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C0 += (1.0 / 4.0) * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h1,h2"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 4.0) * H2["a0,a1,h0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L1["a3,a0"] *
          L2["a4,a5,a1,a2"];
    C0 += (1.0 / 12.0) * H2["a0,a1,h0,h1"] * T3["h0,h2,a2,a3,a4,a5"] * L1["h1,h2"] *
          L3["a3,a4,a5,a0,a1,a2"];
    C0 += -1.0 * H2["a0,a1,h0,a2"] * T2["h0,a3,a4,a5"] * L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C0 += (-1.0 / 4.0) * H2["a0,a1,h0,a2"] * T3["h0,a3,a4,a5,a6,a7"] * L1["a5,a0"] *
          L3["a2,a6,a7,a1,a3,a4"];
    C0 += (1.0 / 8.0) * H2["a0,a1,h0,a2"] * T3["h0,a3,a4,a5,a6,a7"] * L2["a2,a5,a3,a4"] *
          L2["a6,a7,a0,a1"];
    C0 += (-1.0 / 2.0) * H2["a0,a1,h0,a2"] * T3["h0,a3,a4,a5,a6,a7"] * L2["a2,a7,a1,a4"] *
          L2["a5,a6,a0,a3"];
    C0 += (1.0 / 24.0) * H3["p0,p1,p2,h0,a0,a1"] * T3["h1,a2,a3,p0,p1,p2"] * L1["h0,h1"] *
          L2["a0,a1,a2,a3"];
    C0 += (-1.0 / 4.0) * H3["p0,p1,a0,h0,a1,a2"] * T2["h1,a3,p0,p1"] * L1["h0,h1"] *
          L2["a1,a2,a0,a3"];
    C0 += (-1.0 / 8.0) * H3["p0,p1,a0,h0,a1,a2"] * T3["h1,a3,a4,p0,p1,a5"] * L1["h0,h1"] *
          L3["a1,a2,a5,a0,a3,a4"];
    C0 += (-1.0 / 72.0) * H3["p0,p1,a0,a1,a2,a3"] * T3["a4,a5,a6,p0,p1,a7"] * L1["a7,a0"] *
          L3["a1,a2,a3,a4,a5,a6"];
    C0 += (1.0 / 8.0) * H3["p0,p1,a0,a1,a2,a3"] * T3["a4,a5,a6,p0,p1,a7"] * L2["a1,a2,a4,a5"] *
          L2["a3,a7,a0,a6"];
    C0 += (1.0 / 8.0) * H3["p0,p1,a0,a1,a2,a3"] * T3["a4,a5,a6,p0,p1,a7"] * L2["a1,a7,a4,a5"] *
          L2["a2,a3,a0,a6"];
    C0 += (1.0 / 4.0) * H3["p0,a0,a1,h0,a2,a3"] * T2["h1,a4,p0,a5"] * L1["h0,h1"] *
          L3["a2,a3,a5,a0,a1,a4"];
    C0 += (-1.0 / 12.0) * H3["p0,a0,a1,a2,a3,a4"] * T2["a5,a6,p0,a7"] * L1["a7,a0"] *
          L3["a2,a3,a4,a1,a5,a6"];
    C0 += (1.0 / 8.0) * H3["p0,a0,a1,a2,a3,a4"] * T2["a5,a6,p0,a7"] * L2["a2,a3,a5,a6"] *
          L2["a4,a7,a0,a1"];
    C0 += (-1.0 / 2.0) * H3["p0,a0,a1,a2,a3,a4"] * T2["a5,a6,p0,a7"] * L2["a2,a7,a0,a5"] *
          L2["a3,a4,a1,a6"];
    C0 += (-1.0 / 8.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L2["a2,a3,a0,a5"] *
          L3["a4,a8,a9,a1,a6,a7"];
    C0 += (1.0 / 16.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L2["a2,a3,a5,a6"] *
          L3["a4,a8,a9,a0,a1,a7"];
    C0 += (1.0 / 24.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L2["a2,a8,a0,a1"] *
          L3["a3,a4,a9,a5,a6,a7"];
    C0 += (-1.0 / 4.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L2["a2,a8,a0,a5"] *
          L3["a3,a4,a9,a1,a6,a7"];
    C0 += (1.0 / 8.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L2["a2,a8,a5,a6"] *
          L3["a3,a4,a9,a0,a1,a7"];
    C0 += (1.0 / 144.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L2["a8,a9,a0,a1"] *
          L3["a2,a3,a4,a5,a6,a7"];
    C0 += (-1.0 / 24.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L2["a8,a9,a0,a5"] *
          L3["a2,a3,a4,a1,a6,a7"];
    C0 += (-1.0 / 24.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L1["a3,a0"] *
          L2["a4,a5,a1,a2"];
    C0 += (1.0 / 72.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h2,h3"] *
          L3["a3,a4,a5,a0,a1,a2"];
    C0 +=
        (1.0 / 4.0) * H3["a0,a1,a2,h0,h1,a3"] * T2["h0,h1,a4,a5"] * L1["a4,a0"] * L2["a3,a5,a1,a2"];
    C0 += (1.0 / 12.0) * H3["a0,a1,a2,h0,h1,a3"] * T2["h0,h2,a4,a5"] * L1["h1,h2"] *
          L3["a3,a4,a5,a0,a1,a2"];
    C0 += (1.0 / 8.0) * H3["a0,a1,a2,h0,h1,a3"] * T3["h0,h1,a4,a5,a6,a7"] * L1["a5,a0"] *
          L3["a3,a6,a7,a1,a2,a4"];
    C0 += (-1.0 / 8.0) * H3["a0,a1,a2,h0,h1,a3"] * T3["h0,h1,a4,a5,a6,a7"] * L2["a3,a5,a0,a4"] *
          L2["a6,a7,a1,a2"];
    C0 += (-1.0 / 8.0) * H3["a0,a1,a2,h0,h1,a3"] * T3["h0,h1,a4,a5,a6,a7"] * L2["a3,a7,a1,a2"] *
          L2["a5,a6,a0,a4"];
    C0 += (-1.0 / 4.0) * H3["a0,a1,a2,h0,a3,a4"] * T2["h0,a5,a6,a7"] * L1["a6,a0"] *
          L3["a3,a4,a7,a1,a2,a5"];
    C0 += (-1.0 / 8.0) * H3["a0,a1,a2,h0,a3,a4"] * T2["h0,a5,a6,a7"] * L2["a3,a4,a2,a5"] *
          L2["a6,a7,a0,a1"];
    C0 += (1.0 / 2.0) * H3["a0,a1,a2,h0,a3,a4"] * T2["h0,a5,a6,a7"] * L2["a3,a6,a0,a5"] *
          L2["a4,a7,a1,a2"];
    C0 += (-1.0 / 24.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a3,a4,a0,a5"] *
          L3["a7,a8,a9,a1,a2,a6"];
    C0 += (-1.0 / 144.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a3,a4,a5,a6"] *
          L3["a7,a8,a9,a0,a1,a2"];
    C0 += (1.0 / 8.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a3,a7,a0,a1"] *
          L3["a4,a8,a9,a2,a5,a6"];
    C0 += (1.0 / 4.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a3,a7,a0,a5"] *
          L3["a4,a8,a9,a1,a2,a6"];
    C0 += (1.0 / 24.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a3,a7,a5,a6"] *
          L3["a4,a8,a9,a0,a1,a2"];
    C0 += (-1.0 / 16.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a7,a8,a0,a1"] *
          L3["a3,a4,a9,a2,a5,a6"];
    C0 += (-1.0 / 8.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L2["a7,a8,a0,a5"] *
          L3["a3,a4,a9,a1,a2,a6"];
    C0 += (-1.0 / 2.0) * H2["p0,a0,h0,h1"] * T2["h2,h3,p0,a1"] * L1["h0,h2"] * L1["h1,h3"] *
          L1["a1,a0"];
    C0 += (-1.0 / 4.0) * H2["p0,a0,h0,h1"] * T3["h2,h3,a1,p0,a2,a3"] * L1["h0,h2"] * L1["h1,h3"] *
          L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 2.0) * H2["p0,a0,h0,a1"] * T3["h1,a2,a3,p0,a4,a5"] * L1["h0,h1"] * L1["a4,a0"] *
          L2["a1,a5,a2,a3"];
    C0 += (1.0 / 2.0) * H2["a0,a1,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h1,h2"] * L1["a2,a0"] *
          L1["a3,a1"];
    C0 += (1.0 / 2.0) * H2["a0,a1,h0,h1"] * T3["h0,h2,a2,a3,a4,a5"] * L1["h1,h2"] * L1["a3,a0"] *
          L2["a4,a5,a1,a2"];
    C0 += (1.0 / 4.0) * H2["a0,a1,h0,a2"] * T3["h0,a3,a4,a5,a6,a7"] * L1["a5,a0"] * L1["a6,a1"] *
          L2["a2,a7,a3,a4"];
    C0 += (1.0 / 36.0) * H3["p0,p1,p2,h0,h1,h2"] * T3["h3,h4,h5,p0,p1,p2"] * L1["h0,h3"] *
          L1["h1,h4"] * L1["h2,h5"];
    C0 += (1.0 / 4.0) * H3["p0,p1,a0,h0,h1,a1"] * T3["h2,h3,a2,p0,p1,a3"] * L1["h0,h2"] *
          L1["h1,h3"] * L2["a1,a3,a0,a2"];
    C0 += (-1.0 / 8.0) * H3["p0,p1,a0,h0,a1,a2"] * T3["h1,a3,a4,p0,p1,a5"] * L1["h0,h1"] *
          L1["a5,a0"] * L2["a1,a2,a3,a4"];
    C0 += (1.0 / 4.0) * H3["p0,a0,a1,h0,h1,a2"] * T2["h2,h3,p0,a3"] * L1["h0,h2"] * L1["h1,h3"] *
          L2["a2,a3,a0,a1"];
    C0 += (1.0 / 8.0) * H3["p0,a0,a1,h0,h1,a2"] * T3["h2,h3,a3,p0,a4,a5"] * L1["h0,h2"] *
          L1["h1,h3"] * L3["a2,a4,a5,a0,a1,a3"];
    C0 += (1.0 / 2.0) * H3["p0,a0,a1,h0,a2,a3"] * T2["h1,a4,p0,a5"] * L1["h0,h1"] * L1["a5,a0"] *
          L2["a2,a3,a1,a4"];
    C0 += (1.0 / 4.0) * H3["p0,a0,a1,h0,a2,a3"] * T3["h1,a4,a5,p0,a6,a7"] * L1["h0,h1"] *
          L1["a6,a0"] * L3["a2,a3,a7,a1,a4,a5"];
    C0 += (-1.0 / 4.0) * H3["p0,a0,a1,h0,a2,a3"] * T3["h1,a4,a5,p0,a6,a7"] * L1["h0,h1"] *
          L2["a2,a3,a1,a5"] * L2["a6,a7,a0,a4"];
    C0 += (1.0 / 16.0) * H3["p0,a0,a1,h0,a2,a3"] * T3["h1,a4,a5,p0,a6,a7"] * L1["h0,h1"] *
          L2["a2,a3,a4,a5"] * L2["a6,a7,a0,a1"];
    C0 += (1.0 / 2.0) * H3["p0,a0,a1,h0,a2,a3"] * T3["h1,a4,a5,p0,a6,a7"] * L1["h0,h1"] *
          L2["a2,a6,a0,a4"] * L2["a3,a7,a1,a5"];
    C0 += (-1.0 / 4.0) * H3["p0,a0,a1,h0,a2,a3"] * T3["h1,a4,a5,p0,a6,a7"] * L1["h0,h1"] *
          L2["a2,a6,a4,a5"] * L2["a3,a7,a0,a1"];
    C0 += (1.0 / 72.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L1["a8,a0"] *
          L1["a9,a1"] * L3["a2,a3,a4,a5,a6,a7"];
    C0 += (-1.0 / 4.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L1["a8,a0"] *
          L2["a2,a3,a5,a6"] * L2["a4,a9,a1,a7"];
    C0 += (-1.0 / 4.0) * H3["p0,a0,a1,a2,a3,a4"] * T3["a5,a6,a7,p0,a8,a9"] * L1["a8,a0"] *
          L2["a2,a9,a5,a6"] * L2["a3,a4,a1,a7"];
    C0 += (-1.0 / 36.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L1["a3,a0"] *
          L1["a4,a1"] * L1["a5,a2"];
    C0 += (1.0 / 8.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h2,h3"] *
          L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C0 += (-1.0 / 72.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h3,h4,a3,a4,a5"] * L1["h1,h3"] *
          L1["h2,h4"] * L3["a3,a4,a5,a0,a1,a2"];
    C0 += (-1.0 / 2.0) * H3["a0,a1,a2,h0,h1,a3"] * T2["h0,h2,a4,a5"] * L1["h1,h2"] * L1["a4,a0"] *
          L2["a3,a5,a1,a2"];
    C0 += (-1.0 / 4.0) * H3["a0,a1,a2,h0,h1,a3"] * T3["h0,h1,a4,a5,a6,a7"] * L1["a5,a0"] *
          L1["a6,a1"] * L2["a3,a7,a2,a4"];
    C0 += (-1.0 / 4.0) * H3["a0,a1,a2,h0,h1,a3"] * T3["h0,h2,a4,a5,a6,a7"] * L1["h1,h2"] *
          L1["a5,a0"] * L3["a3,a6,a7,a1,a2,a4"];
    C0 += (1.0 / 4.0) * H3["a0,a1,a2,h0,h1,a3"] * T3["h0,h2,a4,a5,a6,a7"] * L1["h1,h2"] *
          L2["a3,a5,a0,a4"] * L2["a6,a7,a1,a2"];
    C0 += (1.0 / 4.0) * H3["a0,a1,a2,h0,h1,a3"] * T3["h0,h2,a4,a5,a6,a7"] * L1["h1,h2"] *
          L2["a3,a7,a1,a2"] * L2["a5,a6,a0,a4"];
    C0 += (-1.0 / 4.0) * H3["a0,a1,a2,h0,a3,a4"] * T2["h0,a5,a6,a7"] * L1["a6,a0"] * L1["a7,a1"] *
          L2["a3,a4,a2,a5"];
    C0 += (-1.0 / 8.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a0"] *
          L1["a8,a1"] * L3["a3,a4,a9,a2,a5,a6"];
    C0 += (1.0 / 4.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a0"] *
          L2["a3,a4,a2,a6"] * L2["a8,a9,a1,a5"];
    C0 += (-1.0 / 16.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a0"] *
          L2["a3,a4,a5,a6"] * L2["a8,a9,a1,a2"];
    C0 += (-1.0 / 2.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a0"] *
          L2["a3,a8,a1,a5"] * L2["a4,a9,a2,a6"];
    C0 += (1.0 / 4.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a0"] *
          L2["a3,a8,a5,a6"] * L2["a4,a9,a1,a2"];
    C0 += (-1.0 / 12.0) * H3["p0,p1,a0,h0,h1,h2"] * T3["h3,h4,h5,p0,p1,a1"] * L1["h0,h3"] *
          L1["h1,h4"] * L1["h2,h5"] * L1["a1,a0"];
    C0 += (1.0 / 24.0) * H3["p0,a0,a1,h0,h1,h2"] * T3["h3,h4,h5,p0,a2,a3"] * L1["h0,h3"] *
          L1["h1,h4"] * L1["h2,h5"] * L2["a2,a3,a0,a1"];
    C0 += (-1.0 / 2.0) * H3["p0,a0,a1,h0,h1,a2"] * T3["h2,h3,a3,p0,a4,a5"] * L1["h0,h2"] *
          L1["h1,h3"] * L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C0 += (1.0 / 8.0) * H3["p0,a0,a1,h0,a2,a3"] * T3["h1,a4,a5,p0,a6,a7"] * L1["h0,h1"] *
          L1["a6,a0"] * L1["a7,a1"] * L2["a2,a3,a4,a5"];
    C0 += (1.0 / 12.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h2,h3"] *
          L1["a3,a0"] * L1["a4,a1"] * L1["a5,a2"];
    C0 += (-1.0 / 8.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h3,h4,a3,a4,a5"] * L1["h1,h3"] *
          L1["h2,h4"] * L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C0 += (1.0 / 2.0) * H3["a0,a1,a2,h0,h1,a3"] * T3["h0,h2,a4,a5,a6,a7"] * L1["h1,h2"] *
          L1["a5,a0"] * L1["a6,a1"] * L2["a3,a7,a2,a4"];
    C0 += (-1.0 / 24.0) * H3["a0,a1,a2,h0,a3,a4"] * T3["h0,a5,a6,a7,a8,a9"] * L1["a7,a0"] *
          L1["a8,a1"] * L1["a9,a2"] * L2["a3,a4,a5,a6"];
    C0 += (1.0 / 12.0) * H3["p0,a0,a1,h0,h1,h2"] * T3["h3,h4,h5,p0,a2,a3"] * L1["h0,h3"] *
          L1["h1,h4"] * L1["h2,h5"] * L1["a2,a0"] * L1["a3,a1"];
    C0 += (-1.0 / 12.0) * H3["a0,a1,a2,h0,h1,h2"] * T3["h0,h3,h4,a3,a4,a5"] * L1["h1,h3"] *
          L1["h2,h4"] * L1["a3,a0"] * L1["a4,a1"] * L1["a5,a2"];

    C1["g1,g0"] += 1.0 * H2["g1,p0,g0,h0"] * T1["h1,p0"] * L1["h0,h1"];
    C1["g1,g0"] += (1.0 / 2.0) * H2["g1,p0,g0,a0"] * T2["a1,a2,p0,a3"] * L2["a0,a3,a1,a2"];
    C1["g1,g0"] +=
        (1.0 / 12.0) * H2["g1,p0,g0,a0"] * T3["a1,a2,a3,p0,a4,a5"] * L3["a0,a4,a5,a1,a2,a3"];
    C1["g1,g0"] += -1.0 * H2["g1,a0,g0,h0"] * T1["h0,a1"] * L1["a1,a0"];
    C1["g1,g0"] += (-1.0 / 2.0) * H2["g1,a0,g0,h0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["g1,g0"] +=
        (-1.0 / 12.0) * H2["g1,a0,g0,h0"] * T3["h0,a1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["g1,g0"] += (1.0 / 8.0) * H3["g1,p0,p1,g0,a0,a1"] * T2["a2,a3,p0,p1"] * L2["a0,a1,a2,a3"];
    C1["g1,g0"] +=
        (1.0 / 24.0) * H3["g1,p0,p1,g0,a0,a1"] * T3["a2,a3,a4,p0,p1,a5"] * L3["a0,a1,a5,a2,a3,a4"];
    C1["g1,g0"] += (-1.0 / 2.0) * H3["g1,p0,a0,g0,a1,a2"] * T1["a3,p0"] * L2["a1,a2,a0,a3"];
    C1["g1,g0"] +=
        (-1.0 / 4.0) * H3["g1,p0,a0,g0,a1,a2"] * T2["a3,a4,p0,a5"] * L3["a1,a2,a5,a0,a3,a4"];
    C1["g1,g0"] += (-1.0 / 8.0) * H3["g1,a0,a1,g0,h0,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["g1,g0"] +=
        (-1.0 / 24.0) * H3["g1,a0,a1,g0,h0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["g1,g0"] += (1.0 / 2.0) * H3["g1,a0,a1,g0,h0,a2"] * T1["h0,a3"] * L2["a2,a3,a0,a1"];
    C1["g1,g0"] +=
        (1.0 / 4.0) * H3["g1,a0,a1,g0,h0,a2"] * T2["h0,a3,a4,a5"] * L3["a2,a4,a5,a0,a1,a3"];
    C1["g1,g0"] +=
        (1.0 / 4.0) * H3["g1,p0,p1,g0,h0,h1"] * T2["h2,h3,p0,p1"] * L1["h0,h2"] * L1["h1,h3"];
    C1["g1,g0"] += (1.0 / 4.0) * H3["g1,p0,p1,g0,h0,a0"] * T3["h1,a1,a2,p0,p1,a3"] * L1["h0,h1"] *
                   L2["a0,a3,a1,a2"];
    C1["g1,g0"] +=
        1.0 * H3["g1,p0,a0,g0,h0,a1"] * T2["h1,a2,p0,a3"] * L1["h0,h1"] * L2["a1,a3,a0,a2"];
    C1["g1,g0"] += (1.0 / 4.0) * H3["g1,p0,a0,g0,h0,a1"] * T3["h1,a2,a3,p0,a4,a5"] * L1["h0,h1"] *
                   L3["a1,a4,a5,a0,a2,a3"];
    C1["g1,g0"] += (-1.0 / 4.0) * H3["g1,p0,a0,g0,a1,a2"] * T2["a3,a4,p0,a5"] * L1["a5,a0"] *
                   L2["a1,a2,a3,a4"];
    C1["g1,g0"] += (-1.0 / 12.0) * H3["g1,p0,a0,g0,a1,a2"] * T3["a3,a4,a5,p0,a6,a7"] * L1["a6,a0"] *
                   L3["a1,a2,a7,a3,a4,a5"];
    C1["g1,g0"] += (-1.0 / 8.0) * H3["g1,p0,a0,g0,a1,a2"] * T3["a3,a4,a5,p0,a6,a7"] *
                   L2["a1,a2,a4,a5"] * L2["a6,a7,a0,a3"];
    C1["g1,g0"] += (1.0 / 2.0) * H3["g1,p0,a0,g0,a1,a2"] * T3["a3,a4,a5,p0,a6,a7"] *
                   L2["a1,a6,a3,a4"] * L2["a2,a7,a0,a5"];
    C1["g1,g0"] +=
        (-1.0 / 4.0) * H3["g1,a0,a1,g0,h0,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C1["g1,g0"] +=
        (1.0 / 4.0) * H3["g1,a0,a1,g0,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h1,h2"] * L2["a2,a3,a0,a1"];
    C1["g1,g0"] += (-1.0 / 4.0) * H3["g1,a0,a1,g0,h0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L1["a3,a0"] *
                   L2["a4,a5,a1,a2"];
    C1["g1,g0"] += (1.0 / 12.0) * H3["g1,a0,a1,g0,h0,h1"] * T3["h0,h2,a2,a3,a4,a5"] * L1["h1,h2"] *
                   L3["a3,a4,a5,a0,a1,a2"];
    C1["g1,g0"] +=
        -1.0 * H3["g1,a0,a1,g0,h0,a2"] * T2["h0,a3,a4,a5"] * L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C1["g1,g0"] += (-1.0 / 4.0) * H3["g1,a0,a1,g0,h0,a2"] * T3["h0,a3,a4,a5,a6,a7"] * L1["a5,a0"] *
                   L3["a2,a6,a7,a1,a3,a4"];
    C1["g1,g0"] += (1.0 / 8.0) * H3["g1,a0,a1,g0,h0,a2"] * T3["h0,a3,a4,a5,a6,a7"] *
                   L2["a2,a5,a3,a4"] * L2["a6,a7,a0,a1"];
    C1["g1,g0"] += (-1.0 / 2.0) * H3["g1,a0,a1,g0,h0,a2"] * T3["h0,a3,a4,a5,a6,a7"] *
                   L2["a2,a7,a1,a4"] * L2["a5,a6,a0,a3"];
    C1["g1,g0"] += (-1.0 / 2.0) * H3["g1,p0,a0,g0,h0,h1"] * T2["h2,h3,p0,a1"] * L1["h0,h2"] *
                   L1["h1,h3"] * L1["a1,a0"];
    C1["g1,g0"] += (-1.0 / 4.0) * H3["g1,p0,a0,g0,h0,h1"] * T3["h2,h3,a1,p0,a2,a3"] * L1["h0,h2"] *
                   L1["h1,h3"] * L2["a2,a3,a0,a1"];
    C1["g1,g0"] += (-1.0 / 2.0) * H3["g1,p0,a0,g0,h0,a1"] * T3["h1,a2,a3,p0,a4,a5"] * L1["h0,h1"] *
                   L1["a4,a0"] * L2["a1,a5,a2,a3"];
    C1["g1,g0"] += (1.0 / 2.0) * H3["g1,a0,a1,g0,h0,h1"] * T2["h0,h2,a2,a3"] * L1["h1,h2"] *
                   L1["a2,a0"] * L1["a3,a1"];
    C1["g1,g0"] += (1.0 / 2.0) * H3["g1,a0,a1,g0,h0,h1"] * T3["h0,h2,a2,a3,a4,a5"] * L1["h1,h2"] *
                   L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C1["g1,g0"] += (1.0 / 4.0) * H3["g1,a0,a1,g0,h0,a2"] * T3["h0,a3,a4,a5,a6,a7"] * L1["a5,a0"] *
                   L1["a6,a1"] * L2["a2,a7,a3,a4"];
    C1["h0,g0"] += 1.0 * H1["p0,g0"] * T1["h0,p0"];
    C1["h0,g0"] += (1.0 / 2.0) * H2["p0,p1,g0,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    C1["h0,g0"] += (1.0 / 4.0) * H2["p0,p1,g0,a0"] * T3["h0,a1,a2,p0,p1,a3"] * L2["a0,a3,a1,a2"];
    C1["h0,g0"] += 1.0 * H2["p0,a0,g0,a1"] * T2["h0,a2,p0,a3"] * L2["a1,a3,a0,a2"];
    C1["h0,g0"] +=
        (1.0 / 4.0) * H2["p0,a0,g0,a1"] * T3["h0,a2,a3,p0,a4,a5"] * L3["a1,a4,a5,a0,a2,a3"];
    C1["h0,g0"] += (1.0 / 4.0) * H2["a0,a1,g0,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] +=
        (1.0 / 12.0) * H2["a0,a1,g0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,g0"] +=
        (1.0 / 24.0) * H3["p0,p1,p2,g0,a0,a1"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a0,a1,a2,a3"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["p0,p1,a0,g0,a1,a2"] * T2["h0,a3,p0,p1"] * L2["a1,a2,a0,a3"];
    C1["h0,g0"] +=
        (-1.0 / 8.0) * H3["p0,p1,a0,g0,a1,a2"] * T3["h0,a3,a4,p0,p1,a5"] * L3["a1,a2,a5,a0,a3,a4"];
    C1["h0,g0"] +=
        (1.0 / 4.0) * H3["p0,a0,a1,g0,a2,a3"] * T2["h0,a4,p0,a5"] * L3["a2,a3,a5,a0,a1,a4"];
    C1["h0,g0"] +=
        (1.0 / 72.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,g0"] +=
        (1.0 / 12.0) * H3["a0,a1,a2,g0,h1,a3"] * T2["h0,h1,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,g0"] += -1.0 * H2["p0,a0,g0,h1"] * T2["h0,h2,p0,a1"] * L1["h1,h2"] * L1["a1,a0"];
    C1["h0,g0"] += (-1.0 / 2.0) * H2["p0,a0,g0,h1"] * T3["h0,h2,a1,p0,a2,a3"] * L1["h1,h2"] *
                   L2["a2,a3,a0,a1"];
    C1["h0,g0"] += (-1.0 / 2.0) * H2["p0,a0,g0,a1"] * T3["h0,a2,a3,p0,a4,a5"] * L1["a4,a0"] *
                   L2["a1,a5,a2,a3"];
    C1["h0,g0"] += (1.0 / 2.0) * H2["a0,a1,g0,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C1["h0,g0"] +=
        (1.0 / 2.0) * H2["a0,a1,g0,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C1["h0,g0"] += (1.0 / 12.0) * H3["p0,p1,p2,g0,h1,h2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h1,h3"] *
                   L1["h2,h4"];
    C1["h0,g0"] += (1.0 / 2.0) * H3["p0,p1,a0,g0,h1,a1"] * T3["h0,h2,a2,p0,p1,a3"] * L1["h1,h2"] *
                   L2["a1,a3,a0,a2"];
    C1["h0,g0"] += (-1.0 / 8.0) * H3["p0,p1,a0,g0,a1,a2"] * T3["h0,a3,a4,p0,p1,a5"] * L1["a5,a0"] *
                   L2["a1,a2,a3,a4"];
    C1["h0,g0"] +=
        (1.0 / 2.0) * H3["p0,a0,a1,g0,h1,a2"] * T2["h0,h2,p0,a3"] * L1["h1,h2"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["p0,a0,a1,g0,h1,a2"] * T3["h0,h2,a3,p0,a4,a5"] * L1["h1,h2"] *
                   L3["a2,a4,a5,a0,a1,a3"];
    C1["h0,g0"] +=
        (1.0 / 2.0) * H3["p0,a0,a1,g0,a2,a3"] * T2["h0,a4,p0,a5"] * L1["a5,a0"] * L2["a2,a3,a1,a4"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["p0,a0,a1,g0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] * L1["a6,a0"] *
                   L3["a2,a3,a7,a1,a4,a5"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["p0,a0,a1,g0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] *
                   L2["a2,a3,a1,a5"] * L2["a6,a7,a0,a4"];
    C1["h0,g0"] += (1.0 / 16.0) * H3["p0,a0,a1,g0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] *
                   L2["a2,a3,a4,a5"] * L2["a6,a7,a0,a1"];
    C1["h0,g0"] += (1.0 / 2.0) * H3["p0,a0,a1,g0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] *
                   L2["a2,a6,a0,a4"] * L2["a3,a7,a1,a5"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["p0,a0,a1,g0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] *
                   L2["a2,a6,a4,a5"] * L2["a3,a7,a0,a1"];
    C1["h0,g0"] += (1.0 / 8.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L1["a3,a0"] *
                   L2["a4,a5,a1,a2"];
    C1["h0,g0"] += (-1.0 / 36.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h2,h3"] *
                   L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,g0"] += (-1.0 / 2.0) * H3["a0,a1,a2,g0,h1,a3"] * T2["h0,h1,a4,a5"] * L1["a4,a0"] *
                   L2["a3,a5,a1,a2"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["a0,a1,a2,g0,h1,a3"] * T3["h0,h1,a4,a5,a6,a7"] * L1["a5,a0"] *
                   L3["a3,a6,a7,a1,a2,a4"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["a0,a1,a2,g0,h1,a3"] * T3["h0,h1,a4,a5,a6,a7"] *
                   L2["a3,a5,a0,a4"] * L2["a6,a7,a1,a2"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["a0,a1,a2,g0,h1,a3"] * T3["h0,h1,a4,a5,a6,a7"] *
                   L2["a3,a7,a1,a2"] * L2["a5,a6,a0,a4"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["p0,p1,a0,g0,h1,h2"] * T3["h0,h3,h4,p0,p1,a1"] * L1["h1,h3"] *
                   L1["h2,h4"] * L1["a1,a0"];
    C1["h0,g0"] += (1.0 / 8.0) * H3["p0,a0,a1,g0,h1,h2"] * T3["h0,h3,h4,p0,a2,a3"] * L1["h1,h3"] *
                   L1["h2,h4"] * L2["a2,a3,a0,a1"];
    C1["h0,g0"] += -1.0 * H3["p0,a0,a1,g0,h1,a2"] * T3["h0,h2,a3,p0,a4,a5"] * L1["h1,h2"] *
                   L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C1["h0,g0"] += (1.0 / 8.0) * H3["p0,a0,a1,g0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] * L1["a6,a0"] *
                   L1["a7,a1"] * L2["a2,a3,a4,a5"];
    C1["h0,g0"] += (1.0 / 12.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L1["a3,a0"] *
                   L1["a4,a1"] * L1["a5,a2"];
    C1["h0,g0"] += (-1.0 / 4.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h2,h3"] *
                   L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C1["h0,g0"] += (1.0 / 2.0) * H3["a0,a1,a2,g0,h1,a3"] * T3["h0,h1,a4,a5,a6,a7"] * L1["a5,a0"] *
                   L1["a6,a1"] * L2["a3,a7,a2,a4"];
    C1["h0,g0"] += (1.0 / 4.0) * H3["p0,a0,a1,g0,h1,h2"] * T3["h0,h3,h4,p0,a2,a3"] * L1["h1,h3"] *
                   L1["h2,h4"] * L1["a2,a0"] * L1["a3,a1"];
    C1["h0,g0"] += (-1.0 / 6.0) * H3["a0,a1,a2,g0,h1,h2"] * T3["h0,h1,h3,a3,a4,a5"] * L1["h2,h3"] *
                   L1["a3,a0"] * L1["a4,a1"] * L1["a5,a2"];
    C1["g0,p0"] += -1.0 * H1["g0,h0"] * T1["h0,p0"];
    C1["g0,p0"] += (-1.0 / 4.0) * H2["g0,p1,a0,a1"] * T2["a2,a3,p0,p1"] * L2["a0,a1,a2,a3"];
    C1["g0,p0"] +=
        (-1.0 / 12.0) * H2["g0,p1,a0,a1"] * T3["a2,a3,a4,p0,p1,a5"] * L3["a0,a1,a5,a2,a3,a4"];
    C1["g0,p0"] += (-1.0 / 2.0) * H2["g0,a0,h0,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C1["g0,p0"] += (-1.0 / 4.0) * H2["g0,a0,h0,h1"] * T3["h0,h1,a1,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] += -1.0 * H2["g0,a0,h0,a1"] * T2["h0,a2,p0,a3"] * L2["a1,a3,a0,a2"];
    C1["g0,p0"] +=
        (-1.0 / 4.0) * H2["g0,a0,h0,a1"] * T3["h0,a2,a3,p0,a4,a5"] * L3["a1,a4,a5,a0,a2,a3"];
    C1["g0,p0"] +=
        (-1.0 / 72.0) * H3["g0,p1,p2,a0,a1,a2"] * T3["a3,a4,a5,p0,p1,p2"] * L3["a0,a1,a2,a3,a4,a5"];
    C1["g0,p0"] +=
        (-1.0 / 12.0) * H3["g0,p1,a0,a1,a2,a3"] * T2["a4,a5,p0,p1"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["g0,p0"] +=
        (-1.0 / 24.0) * H3["g0,a0,a1,h0,h1,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["g0,a0,a1,h0,h1,a2"] * T2["h0,h1,p0,a3"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] +=
        (1.0 / 8.0) * H3["g0,a0,a1,h0,h1,a2"] * T3["h0,h1,a3,p0,a4,a5"] * L3["a2,a4,a5,a0,a1,a3"];
    C1["g0,p0"] +=
        (-1.0 / 4.0) * H3["g0,a0,a1,h0,a2,a3"] * T2["h0,a4,p0,a5"] * L3["a2,a3,a5,a0,a1,a4"];
    C1["g0,p0"] += (-1.0 / 2.0) * H2["g0,p1,h0,h1"] * T2["h2,h3,p0,p1"] * L1["h0,h2"] * L1["h1,h3"];
    C1["g0,p0"] += (-1.0 / 2.0) * H2["g0,p1,h0,a0"] * T3["h1,a1,a2,p0,p1,a3"] * L1["h0,h1"] *
                   L2["a0,a3,a1,a2"];
    C1["g0,p0"] += 1.0 * H2["g0,a0,h0,h1"] * T2["h0,h2,p0,a1"] * L1["h1,h2"] * L1["a1,a0"];
    C1["g0,p0"] +=
        (1.0 / 2.0) * H2["g0,a0,h0,h1"] * T3["h0,h2,a1,p0,a2,a3"] * L1["h1,h2"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] +=
        (1.0 / 2.0) * H2["g0,a0,h0,a1"] * T3["h0,a2,a3,p0,a4,a5"] * L1["a4,a0"] * L2["a1,a5,a2,a3"];
    C1["g0,p0"] += (-1.0 / 8.0) * H3["g0,p1,p2,h0,a0,a1"] * T3["h1,a2,a3,p0,p1,p2"] * L1["h0,h1"] *
                   L2["a0,a1,a2,a3"];
    C1["g0,p0"] +=
        (1.0 / 2.0) * H3["g0,p1,a0,h0,a1,a2"] * T2["h1,a3,p0,p1"] * L1["h0,h1"] * L2["a1,a2,a0,a3"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["g0,p1,a0,h0,a1,a2"] * T3["h1,a3,a4,p0,p1,a5"] * L1["h0,h1"] *
                   L3["a1,a2,a5,a0,a3,a4"];
    C1["g0,p0"] += (1.0 / 36.0) * H3["g0,p1,a0,a1,a2,a3"] * T3["a4,a5,a6,p0,p1,a7"] * L1["a7,a0"] *
                   L3["a1,a2,a3,a4,a5,a6"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["g0,p1,a0,a1,a2,a3"] * T3["a4,a5,a6,p0,p1,a7"] *
                   L2["a1,a2,a4,a5"] * L2["a3,a7,a0,a6"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["g0,p1,a0,a1,a2,a3"] * T3["a4,a5,a6,p0,p1,a7"] *
                   L2["a1,a7,a4,a5"] * L2["a2,a3,a0,a6"];
    C1["g0,p0"] += (-1.0 / 12.0) * H3["g0,a0,a1,h0,h1,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L1["a2,a0"] *
                   L1["a3,a1"];
    C1["g0,p0"] += (1.0 / 8.0) * H3["g0,a0,a1,h0,h1,h2"] * T3["h0,h1,h3,p0,a2,a3"] * L1["h2,h3"] *
                   L2["a2,a3,a0,a1"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["g0,a0,a1,h0,h1,a2"] * T2["h0,h2,p0,a3"] * L1["h1,h2"] *
                   L2["a2,a3,a0,a1"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["g0,a0,a1,h0,h1,a2"] * T3["h0,h1,a3,p0,a4,a5"] * L1["a4,a0"] *
                   L2["a2,a5,a1,a3"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["g0,a0,a1,h0,h1,a2"] * T3["h0,h2,a3,p0,a4,a5"] * L1["h1,h2"] *
                   L3["a2,a4,a5,a0,a1,a3"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["g0,a0,a1,h0,a2,a3"] * T2["h0,a4,p0,a5"] * L1["a5,a0"] *
                   L2["a2,a3,a1,a4"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["g0,a0,a1,h0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] * L1["a6,a0"] *
                   L3["a2,a3,a7,a1,a4,a5"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["g0,a0,a1,h0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] *
                   L2["a2,a3,a1,a5"] * L2["a6,a7,a0,a4"];
    C1["g0,p0"] += (-1.0 / 16.0) * H3["g0,a0,a1,h0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] *
                   L2["a2,a3,a4,a5"] * L2["a6,a7,a0,a1"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["g0,a0,a1,h0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] *
                   L2["a2,a6,a0,a4"] * L2["a3,a7,a1,a5"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["g0,a0,a1,h0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] *
                   L2["a2,a6,a4,a5"] * L2["a3,a7,a0,a1"];
    C1["g0,p0"] += (-1.0 / 12.0) * H3["g0,p1,p2,h0,h1,h2"] * T3["h3,h4,h5,p0,p1,p2"] * L1["h0,h3"] *
                   L1["h1,h4"] * L1["h2,h5"];
    C1["g0,p0"] += (-1.0 / 2.0) * H3["g0,p1,a0,h0,h1,a1"] * T3["h2,h3,a2,p0,p1,a3"] * L1["h0,h2"] *
                   L1["h1,h3"] * L2["a1,a3,a0,a2"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["g0,p1,a0,h0,a1,a2"] * T3["h1,a3,a4,p0,p1,a5"] * L1["h0,h1"] *
                   L1["a5,a0"] * L2["a1,a2,a3,a4"];
    C1["g0,p0"] += (1.0 / 4.0) * H3["g0,a0,a1,h0,h1,h2"] * T3["h0,h1,h3,p0,a2,a3"] * L1["h2,h3"] *
                   L1["a2,a0"] * L1["a3,a1"];
    C1["g0,p0"] += (-1.0 / 8.0) * H3["g0,a0,a1,h0,h1,h2"] * T3["h0,h3,h4,p0,a2,a3"] * L1["h1,h3"] *
                   L1["h2,h4"] * L2["a2,a3,a0,a1"];
    C1["g0,p0"] += 1.0 * H3["g0,a0,a1,h0,h1,a2"] * T3["h0,h2,a3,p0,a4,a5"] * L1["h1,h2"] *
                   L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C1["g0,p0"] += (-1.0 / 8.0) * H3["g0,a0,a1,h0,a2,a3"] * T3["h0,a4,a5,p0,a6,a7"] * L1["a6,a0"] *
                   L1["a7,a1"] * L2["a2,a3,a4,a5"];
    C1["g0,p0"] += (1.0 / 6.0) * H3["g0,p1,a0,h0,h1,h2"] * T3["h3,h4,h5,p0,p1,a1"] * L1["h0,h3"] *
                   L1["h1,h4"] * L1["h2,h5"] * L1["a1,a0"];
    C1["g0,p0"] += (-1.0 / 4.0) * H3["g0,a0,a1,h0,h1,h2"] * T3["h0,h3,h4,p0,a2,a3"] * L1["h1,h3"] *
                   L1["h2,h4"] * L1["a2,a0"] * L1["a3,a1"];
    C1["h0,p0"] += 1.0 * H1["p1,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    C1["h0,p0"] += (1.0 / 2.0) * H1["p1,a0"] * T3["h0,a1,a2,p0,p1,a3"] * L2["a0,a3,a1,a2"];
    C1["h0,p0"] += -1.0 * H1["a0,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C1["h0,p0"] += (-1.0 / 2.0) * H1["a0,h1"] * T3["h0,h1,a1,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] += (1.0 / 8.0) * H2["p1,p2,a0,a1"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a0,a1,a2,a3"];
    C1["h0,p0"] += (-1.0 / 2.0) * H2["p1,a0,a1,a2"] * T2["h0,a3,p0,p1"] * L2["a1,a2,a0,a3"];
    C1["h0,p0"] +=
        (-1.0 / 4.0) * H2["p1,a0,a1,a2"] * T3["h0,a3,a4,p0,p1,a5"] * L3["a1,a2,a5,a0,a3,a4"];
    C1["h0,p0"] += (-1.0 / 8.0) * H2["a0,a1,h1,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] += (1.0 / 2.0) * H2["a0,a1,h1,a2"] * T2["h0,h1,p0,a3"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] +=
        (1.0 / 4.0) * H2["a0,a1,h1,a2"] * T3["h0,h1,a3,p0,a4,a5"] * L3["a2,a4,a5,a0,a1,a3"];
    C1["h0,p0"] +=
        (1.0 / 24.0) * H3["p1,p2,a0,a1,a2,a3"] * T3["h0,a4,a5,p0,p1,p2"] * L3["a1,a2,a3,a0,a4,a5"];
    C1["h0,p0"] +=
        (1.0 / 12.0) * H3["p1,a0,a1,a2,a3,a4"] * T2["h0,a5,p0,p1"] * L3["a2,a3,a4,a0,a1,a5"];
    C1["h0,p0"] +=
        (-1.0 / 24.0) * H3["a0,a1,a2,h1,h2,a3"] * T3["h0,h1,h2,p0,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,p0"] +=
        (-1.0 / 12.0) * H3["a0,a1,a2,h1,a3,a4"] * T2["h0,h1,p0,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,p0"] +=
        (1.0 / 4.0) * H2["p1,p2,h1,h2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h1,h3"] * L1["h2,h4"];
    C1["h0,p0"] +=
        1.0 * H2["p1,a0,h1,a1"] * T3["h0,h2,a2,p0,p1,a3"] * L1["h1,h2"] * L2["a1,a3,a0,a2"];
    C1["h0,p0"] += (-1.0 / 4.0) * H2["p1,a0,a1,a2"] * T3["h0,a3,a4,p0,p1,a5"] * L1["a5,a0"] *
                   L2["a1,a2,a3,a4"];
    C1["h0,p0"] +=
        (-1.0 / 4.0) * H2["a0,a1,h1,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    C1["h0,p0"] +=
        (1.0 / 4.0) * H2["a0,a1,h1,h2"] * T3["h0,h1,h3,p0,a2,a3"] * L1["h2,h3"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] +=
        -1.0 * H2["a0,a1,h1,a2"] * T3["h0,h1,a3,p0,a4,a5"] * L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C1["h0,p0"] += (-1.0 / 4.0) * H3["p1,p2,a0,h1,a1,a2"] * T3["h0,h2,a3,p0,p1,p2"] * L1["h1,h2"] *
                   L2["a1,a2,a0,a3"];
    C1["h0,p0"] += (1.0 / 4.0) * H3["p1,a0,a1,h1,a2,a3"] * T3["h0,h2,a4,p0,p1,a5"] * L1["h1,h2"] *
                   L3["a2,a3,a5,a0,a1,a4"];
    C1["h0,p0"] += (-1.0 / 12.0) * H3["p1,a0,a1,a2,a3,a4"] * T3["h0,a5,a6,p0,p1,a7"] * L1["a7,a0"] *
                   L3["a2,a3,a4,a1,a5,a6"];
    C1["h0,p0"] += (1.0 / 8.0) * H3["p1,a0,a1,a2,a3,a4"] * T3["h0,a5,a6,p0,p1,a7"] *
                   L2["a2,a3,a5,a6"] * L2["a4,a7,a0,a1"];
    C1["h0,p0"] += (-1.0 / 2.0) * H3["p1,a0,a1,a2,a3,a4"] * T3["h0,a5,a6,p0,p1,a7"] *
                   L2["a2,a7,a0,a5"] * L2["a3,a4,a1,a6"];
    C1["h0,p0"] += (1.0 / 4.0) * H3["a0,a1,a2,h1,h2,a3"] * T3["h0,h1,h2,p0,a4,a5"] * L1["a4,a0"] *
                   L2["a3,a5,a1,a2"];
    C1["h0,p0"] += (1.0 / 12.0) * H3["a0,a1,a2,h1,h2,a3"] * T3["h0,h1,h3,p0,a4,a5"] * L1["h2,h3"] *
                   L3["a3,a4,a5,a0,a1,a2"];
    C1["h0,p0"] += (-1.0 / 4.0) * H3["a0,a1,a2,h1,a3,a4"] * T3["h0,h1,a5,p0,a6,a7"] * L1["a6,a0"] *
                   L3["a3,a4,a7,a1,a2,a5"];
    C1["h0,p0"] += (-1.0 / 8.0) * H3["a0,a1,a2,h1,a3,a4"] * T3["h0,h1,a5,p0,a6,a7"] *
                   L2["a3,a4,a2,a5"] * L2["a6,a7,a0,a1"];
    C1["h0,p0"] += (1.0 / 2.0) * H3["a0,a1,a2,h1,a3,a4"] * T3["h0,h1,a5,p0,a6,a7"] *
                   L2["a3,a6,a0,a5"] * L2["a4,a7,a1,a2"];
    C1["h0,p0"] += (-1.0 / 2.0) * H2["p1,a0,h1,h2"] * T3["h0,h3,h4,p0,p1,a1"] * L1["h1,h3"] *
                   L1["h2,h4"] * L1["a1,a0"];
    C1["h0,p0"] += (1.0 / 2.0) * H2["a0,a1,h1,h2"] * T3["h0,h1,h3,p0,a2,a3"] * L1["h2,h3"] *
                   L1["a2,a0"] * L1["a3,a1"];
    C1["h0,p0"] += (1.0 / 4.0) * H3["p1,a0,a1,h1,h2,a2"] * T3["h0,h3,h4,p0,p1,a3"] * L1["h1,h3"] *
                   L1["h2,h4"] * L2["a2,a3,a0,a1"];
    C1["h0,p0"] += (1.0 / 2.0) * H3["p1,a0,a1,h1,a2,a3"] * T3["h0,h2,a4,p0,p1,a5"] * L1["h1,h2"] *
                   L1["a5,a0"] * L2["a2,a3,a1,a4"];
    C1["h0,p0"] += (-1.0 / 2.0) * H3["a0,a1,a2,h1,h2,a3"] * T3["h0,h1,h3,p0,a4,a5"] * L1["h2,h3"] *
                   L1["a4,a0"] * L2["a3,a5,a1,a2"];
    C1["h0,p0"] += (-1.0 / 4.0) * H3["a0,a1,a2,h1,a3,a4"] * T3["h0,h1,a5,p0,a6,a7"] * L1["a6,a0"] *
                   L1["a7,a1"] * L2["a3,a4,a2,a5"];

    C2["g2,g3,g0,g1"] += 1.0 * H3["g2,g3,p0,g0,g1,h0"] * T1["h1,p0"] * L1["h0,h1"];
    C2["g2,g3,g0,g1"] +=
        (1.0 / 2.0) * H3["g2,g3,p0,g0,g1,a0"] * T2["a1,a2,p0,a3"] * L2["a0,a3,a1,a2"];
    C2["g2,g3,g0,g1"] +=
        (1.0 / 12.0) * H3["g2,g3,p0,g0,g1,a0"] * T3["a1,a2,a3,p0,a4,a5"] * L3["a0,a4,a5,a1,a2,a3"];
    C2["g2,g3,g0,g1"] += -1.0 * H3["g2,g3,a0,g0,g1,h0"] * T1["h0,a1"] * L1["a1,a0"];
    C2["g2,g3,g0,g1"] +=
        (-1.0 / 2.0) * H3["g2,g3,a0,g0,g1,h0"] * T2["h0,a1,a2,a3"] * L2["a2,a3,a0,a1"];
    C2["g2,g3,g0,g1"] +=
        (-1.0 / 12.0) * H3["g2,g3,a0,g0,g1,h0"] * T3["h0,a1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    auto temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghgg"});
    temp["g2,h0,g0,g1"] += 1.0 * H2["g2,p0,g0,g1"] * T1["h0,p0"];
    temp["g2,h0,g0,g1"] += (1.0 / 2.0) * H3["g2,p0,p1,g0,g1,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    temp["g2,h0,g0,g1"] +=
        (1.0 / 4.0) * H3["g2,p0,p1,g0,g1,a0"] * T3["h0,a1,a2,p0,p1,a3"] * L2["a0,a3,a1,a2"];
    temp["g2,h0,g0,g1"] += 1.0 * H3["g2,p0,a0,g0,g1,a1"] * T2["h0,a2,p0,a3"] * L2["a1,a3,a0,a2"];
    temp["g2,h0,g0,g1"] +=
        (1.0 / 4.0) * H3["g2,p0,a0,g0,g1,a1"] * T3["h0,a2,a3,p0,a4,a5"] * L3["a1,a4,a5,a0,a2,a3"];
    temp["g2,h0,g0,g1"] +=
        (1.0 / 4.0) * H3["g2,a0,a1,g0,g1,h1"] * T2["h0,h1,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["g2,h0,g0,g1"] +=
        (1.0 / 12.0) * H3["g2,a0,a1,g0,g1,h1"] * T3["h0,h1,a2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    temp["g2,h0,g0,g1"] +=
        -1.0 * H3["g2,p0,a0,g0,g1,h1"] * T2["h0,h2,p0,a1"] * L1["h1,h2"] * L1["a1,a0"];
    temp["g2,h0,g0,g1"] += (-1.0 / 2.0) * H3["g2,p0,a0,g0,g1,h1"] * T3["h0,h2,a1,p0,a2,a3"] *
                           L1["h1,h2"] * L2["a2,a3,a0,a1"];
    temp["g2,h0,g0,g1"] += (-1.0 / 2.0) * H3["g2,p0,a0,g0,g1,a1"] * T3["h0,a2,a3,p0,a4,a5"] *
                           L1["a4,a0"] * L2["a1,a5,a2,a3"];
    temp["g2,h0,g0,g1"] +=
        (1.0 / 2.0) * H3["g2,a0,a1,g0,g1,h1"] * T2["h0,h1,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    temp["g2,h0,g0,g1"] += (1.0 / 2.0) * H3["g2,a0,a1,g0,g1,h1"] * T3["h0,h1,a2,a3,a4,a5"] *
                           L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C2["g2,h0,g0,g1"] += temp["g2,h0,g0,g1"];
    C2["h0,g2,g0,g1"] -= temp["g2,h0,g0,g1"];
    C2["h0,h1,g0,g1"] += (1.0 / 2.0) * H2["p0,p1,g0,g1"] * T2["h0,h1,p0,p1"];
    C2["h0,h1,g0,g1"] += -1.0 * H2["p0,a0,g0,g1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    C2["h0,h1,g0,g1"] +=
        (-1.0 / 2.0) * H2["p0,a0,g0,g1"] * T3["h0,h1,a1,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C2["h0,h1,g0,g1"] +=
        (1.0 / 6.0) * H3["p0,p1,p2,g0,g1,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h2,h3"];
    C2["h0,h1,g0,g1"] +=
        (1.0 / 2.0) * H3["p0,p1,a0,g0,g1,a1"] * T3["h0,h1,a2,p0,p1,a3"] * L2["a1,a3,a0,a2"];
    C2["h0,h1,g0,g1"] +=
        (1.0 / 2.0) * H3["p0,a0,a1,g0,g1,a2"] * T2["h0,h1,p0,a3"] * L2["a2,a3,a0,a1"];
    C2["h0,h1,g0,g1"] +=
        (1.0 / 4.0) * H3["p0,a0,a1,g0,g1,a2"] * T3["h0,h1,a3,p0,a4,a5"] * L3["a2,a4,a5,a0,a1,a3"];
    C2["h0,h1,g0,g1"] +=
        (-1.0 / 36.0) * H3["a0,a1,a2,g0,g1,h2"] * T3["h0,h1,h2,a3,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    C2["h0,h1,g0,g1"] += (-1.0 / 2.0) * H3["p0,p1,a0,g0,g1,h2"] * T3["h0,h1,h3,p0,p1,a1"] *
                         L1["h2,h3"] * L1["a1,a0"];
    C2["h0,h1,g0,g1"] += (1.0 / 4.0) * H3["p0,a0,a1,g0,g1,h2"] * T3["h0,h1,h3,p0,a2,a3"] *
                         L1["h2,h3"] * L2["a2,a3,a0,a1"];
    C2["h0,h1,g0,g1"] +=
        -1.0 * H3["p0,a0,a1,g0,g1,a2"] * T3["h0,h1,a3,p0,a4,a5"] * L1["a4,a0"] * L2["a2,a5,a1,a3"];
    C2["h0,h1,g0,g1"] += (-1.0 / 4.0) * H3["a0,a1,a2,g0,g1,h2"] * T3["h0,h1,h2,a3,a4,a5"] *
                         L1["a3,a0"] * L2["a4,a5,a1,a2"];
    C2["h0,h1,g0,g1"] += (1.0 / 2.0) * H3["p0,a0,a1,g0,g1,h2"] * T3["h0,h1,h3,p0,a2,a3"] *
                         L1["h2,h3"] * L1["a2,a0"] * L1["a3,a1"];
    C2["h0,h1,g0,g1"] += (-1.0 / 6.0) * H3["a0,a1,a2,g0,g1,h2"] * T3["h0,h1,h2,a3,a4,a5"] *
                         L1["a3,a0"] * L1["a4,a1"] * L1["a5,a2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gggp"});
    temp["g1,g2,g0,p0"] += -1.0 * H2["g1,g2,g0,h0"] * T1["h0,p0"];
    temp["g1,g2,g0,p0"] +=
        (-1.0 / 4.0) * H3["g1,g2,p1,g0,a0,a1"] * T2["a2,a3,p0,p1"] * L2["a0,a1,a2,a3"];
    temp["g1,g2,g0,p0"] +=
        (-1.0 / 12.0) * H3["g1,g2,p1,g0,a0,a1"] * T3["a2,a3,a4,p0,p1,a5"] * L3["a0,a1,a5,a2,a3,a4"];
    temp["g1,g2,g0,p0"] += (-1.0 / 2.0) * H3["g1,g2,a0,g0,h0,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp["g1,g2,g0,p0"] +=
        (-1.0 / 4.0) * H3["g1,g2,a0,g0,h0,h1"] * T3["h0,h1,a1,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["g1,g2,g0,p0"] += -1.0 * H3["g1,g2,a0,g0,h0,a1"] * T2["h0,a2,p0,a3"] * L2["a1,a3,a0,a2"];
    temp["g1,g2,g0,p0"] +=
        (-1.0 / 4.0) * H3["g1,g2,a0,g0,h0,a1"] * T3["h0,a2,a3,p0,a4,a5"] * L3["a1,a4,a5,a0,a2,a3"];
    temp["g1,g2,g0,p0"] +=
        (-1.0 / 2.0) * H3["g1,g2,p1,g0,h0,h1"] * T2["h2,h3,p0,p1"] * L1["h0,h2"] * L1["h1,h3"];
    temp["g1,g2,g0,p0"] += (-1.0 / 2.0) * H3["g1,g2,p1,g0,h0,a0"] * T3["h1,a1,a2,p0,p1,a3"] *
                           L1["h0,h1"] * L2["a0,a3,a1,a2"];
    temp["g1,g2,g0,p0"] +=
        1.0 * H3["g1,g2,a0,g0,h0,h1"] * T2["h0,h2,p0,a1"] * L1["h1,h2"] * L1["a1,a0"];
    temp["g1,g2,g0,p0"] += (1.0 / 2.0) * H3["g1,g2,a0,g0,h0,h1"] * T3["h0,h2,a1,p0,a2,a3"] *
                           L1["h1,h2"] * L2["a2,a3,a0,a1"];
    temp["g1,g2,g0,p0"] += (1.0 / 2.0) * H3["g1,g2,a0,g0,h0,a1"] * T3["h0,a2,a3,p0,a4,a5"] *
                           L1["a4,a0"] * L2["a1,a5,a2,a3"];
    C2["g1,g2,g0,p0"] += temp["g1,g2,g0,p0"];
    C2["g1,g2,p0,g0"] -= temp["g1,g2,g0,p0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghgp"});
    temp["g1,h0,g0,p0"] += 1.0 * H2["g1,p1,g0,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    temp["g1,h0,g0,p0"] +=
        (1.0 / 2.0) * H2["g1,p1,g0,a0"] * T3["h0,a1,a2,p0,p1,a3"] * L2["a0,a3,a1,a2"];
    temp["g1,h0,g0,p0"] += -1.0 * H2["g1,a0,g0,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp["g1,h0,g0,p0"] +=
        (-1.0 / 2.0) * H2["g1,a0,g0,h1"] * T3["h0,h1,a1,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["g1,h0,g0,p0"] +=
        (1.0 / 8.0) * H3["g1,p1,p2,g0,a0,a1"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a0,a1,a2,a3"];
    temp["g1,h0,g0,p0"] +=
        (-1.0 / 2.0) * H3["g1,p1,a0,g0,a1,a2"] * T2["h0,a3,p0,p1"] * L2["a1,a2,a0,a3"];
    temp["g1,h0,g0,p0"] +=
        (-1.0 / 4.0) * H3["g1,p1,a0,g0,a1,a2"] * T3["h0,a3,a4,p0,p1,a5"] * L3["a1,a2,a5,a0,a3,a4"];
    temp["g1,h0,g0,p0"] +=
        (-1.0 / 8.0) * H3["g1,a0,a1,g0,h1,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["g1,h0,g0,p0"] +=
        (1.0 / 2.0) * H3["g1,a0,a1,g0,h1,a2"] * T2["h0,h1,p0,a3"] * L2["a2,a3,a0,a1"];
    temp["g1,h0,g0,p0"] +=
        (1.0 / 4.0) * H3["g1,a0,a1,g0,h1,a2"] * T3["h0,h1,a3,p0,a4,a5"] * L3["a2,a4,a5,a0,a1,a3"];
    temp["g1,h0,g0,p0"] +=
        (1.0 / 4.0) * H3["g1,p1,p2,g0,h1,h2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h1,h3"] * L1["h2,h4"];
    temp["g1,h0,g0,p0"] +=
        1.0 * H3["g1,p1,a0,g0,h1,a1"] * T3["h0,h2,a2,p0,p1,a3"] * L1["h1,h2"] * L2["a1,a3,a0,a2"];
    temp["g1,h0,g0,p0"] += (-1.0 / 4.0) * H3["g1,p1,a0,g0,a1,a2"] * T3["h0,a3,a4,p0,p1,a5"] *
                           L1["a5,a0"] * L2["a1,a2,a3,a4"];
    temp["g1,h0,g0,p0"] += (-1.0 / 4.0) * H3["g1,a0,a1,g0,h1,h2"] * T3["h0,h1,h2,p0,a2,a3"] *
                           L1["a2,a0"] * L1["a3,a1"];
    temp["g1,h0,g0,p0"] += (1.0 / 4.0) * H3["g1,a0,a1,g0,h1,h2"] * T3["h0,h1,h3,p0,a2,a3"] *
                           L1["h2,h3"] * L2["a2,a3,a0,a1"];
    temp["g1,h0,g0,p0"] +=
        -1.0 * H3["g1,a0,a1,g0,h1,a2"] * T3["h0,h1,a3,p0,a4,a5"] * L1["a4,a0"] * L2["a2,a5,a1,a3"];
    temp["g1,h0,g0,p0"] += (-1.0 / 2.0) * H3["g1,p1,a0,g0,h1,h2"] * T3["h0,h3,h4,p0,p1,a1"] *
                           L1["h1,h3"] * L1["h2,h4"] * L1["a1,a0"];
    temp["g1,h0,g0,p0"] += (1.0 / 2.0) * H3["g1,a0,a1,g0,h1,h2"] * T3["h0,h1,h3,p0,a2,a3"] *
                           L1["h2,h3"] * L1["a2,a0"] * L1["a3,a1"];
    C2["g1,h0,g0,p0"] += temp["g1,h0,g0,p0"];
    C2["h0,g1,g0,p0"] -= temp["g1,h0,g0,p0"];
    C2["g1,h0,p0,g0"] -= temp["g1,h0,g0,p0"];
    C2["h0,g1,p0,g0"] += temp["g1,h0,g0,p0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"hhgp"});
    temp["h0,h1,g0,p0"] += -1.0 * H1["p1,g0"] * T2["h0,h1,p0,p1"];
    temp["h0,h1,g0,p0"] += (-1.0 / 2.0) * H2["p1,p2,g0,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h2,h3"];
    temp["h0,h1,g0,p0"] += -1.0 * H2["p1,a0,g0,a1"] * T3["h0,h1,a2,p0,p1,a3"] * L2["a1,a3,a0,a2"];
    temp["h0,h1,g0,p0"] +=
        (-1.0 / 4.0) * H2["a0,a1,g0,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["h0,h1,g0,p0"] +=
        (1.0 / 4.0) * H3["p1,p2,a0,g0,a1,a2"] * T3["h0,h1,a3,p0,p1,p2"] * L2["a1,a2,a0,a3"];
    temp["h0,h1,g0,p0"] +=
        (-1.0 / 4.0) * H3["p1,a0,a1,g0,a2,a3"] * T3["h0,h1,a4,p0,p1,a5"] * L3["a2,a3,a5,a0,a1,a4"];
    temp["h0,h1,g0,p0"] +=
        (-1.0 / 12.0) * H3["a0,a1,a2,g0,h2,a3"] * T3["h0,h1,h2,p0,a4,a5"] * L3["a3,a4,a5,a0,a1,a2"];
    temp["h0,h1,g0,p0"] +=
        1.0 * H2["p1,a0,g0,h2"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h2,h3"] * L1["a1,a0"];
    temp["h0,h1,g0,p0"] +=
        (-1.0 / 2.0) * H2["a0,a1,g0,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    temp["h0,h1,g0,p0"] += (-1.0 / 2.0) * H3["p1,a0,a1,g0,h2,a2"] * T3["h0,h1,h3,p0,p1,a3"] *
                           L1["h2,h3"] * L2["a2,a3,a0,a1"];
    temp["h0,h1,g0,p0"] += (-1.0 / 2.0) * H3["p1,a0,a1,g0,a2,a3"] * T3["h0,h1,a4,p0,p1,a5"] *
                           L1["a5,a0"] * L2["a2,a3,a1,a4"];
    temp["h0,h1,g0,p0"] += (1.0 / 2.0) * H3["a0,a1,a2,g0,h2,a3"] * T3["h0,h1,h2,p0,a4,a5"] *
                           L1["a4,a0"] * L2["a3,a5,a1,a2"];
    C2["h0,h1,g0,p0"] += temp["h0,h1,g0,p0"];
    C2["h0,h1,p0,g0"] -= temp["h0,h1,g0,p0"];
    C2["g0,g1,p0,p1"] += (-1.0 / 2.0) * H2["g0,g1,h0,h1"] * T2["h0,h1,p0,p1"];
    C2["g0,g1,p0,p1"] += 1.0 * H2["g0,g1,h0,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    C2["g0,g1,p0,p1"] +=
        (1.0 / 2.0) * H2["g0,g1,h0,a0"] * T3["h0,a1,a2,p0,p1,a3"] * L2["a0,a3,a1,a2"];
    C2["g0,g1,p0,p1"] +=
        (1.0 / 36.0) * H3["g0,g1,p2,a0,a1,a2"] * T3["a3,a4,a5,p0,p1,p2"] * L3["a0,a1,a2,a3,a4,a5"];
    C2["g0,g1,p0,p1"] +=
        (-1.0 / 6.0) * H3["g0,g1,a0,h0,h1,h2"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    C2["g0,g1,p0,p1"] +=
        (-1.0 / 2.0) * H3["g0,g1,a0,h0,h1,a1"] * T3["h0,h1,a2,p0,p1,a3"] * L2["a1,a3,a0,a2"];
    C2["g0,g1,p0,p1"] +=
        (-1.0 / 2.0) * H3["g0,g1,a0,h0,a1,a2"] * T2["h0,a3,p0,p1"] * L2["a1,a2,a0,a3"];
    C2["g0,g1,p0,p1"] +=
        (-1.0 / 4.0) * H3["g0,g1,a0,h0,a1,a2"] * T3["h0,a3,a4,p0,p1,a5"] * L3["a1,a2,a5,a0,a3,a4"];
    C2["g0,g1,p0,p1"] += (1.0 / 4.0) * H3["g0,g1,p2,h0,a0,a1"] * T3["h1,a2,a3,p0,p1,p2"] *
                         L1["h0,h1"] * L2["a0,a1,a2,a3"];
    C2["g0,g1,p0,p1"] +=
        (1.0 / 2.0) * H3["g0,g1,a0,h0,h1,h2"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h2,h3"] * L1["a1,a0"];
    C2["g0,g1,p0,p1"] +=
        1.0 * H3["g0,g1,a0,h0,h1,a1"] * T3["h0,h2,a2,p0,p1,a3"] * L1["h1,h2"] * L2["a1,a3,a0,a2"];
    C2["g0,g1,p0,p1"] += (-1.0 / 4.0) * H3["g0,g1,a0,h0,a1,a2"] * T3["h0,a3,a4,p0,p1,a5"] *
                         L1["a5,a0"] * L2["a1,a2,a3,a4"];
    C2["g0,g1,p0,p1"] += (1.0 / 6.0) * H3["g0,g1,p2,h0,h1,h2"] * T3["h3,h4,h5,p0,p1,p2"] *
                         L1["h0,h3"] * L1["h1,h4"] * L1["h2,h5"];
    C2["g0,g1,p0,p1"] += (-1.0 / 2.0) * H3["g0,g1,a0,h0,h1,h2"] * T3["h0,h3,h4,p0,p1,a1"] *
                         L1["h1,h3"] * L1["h2,h4"] * L1["a1,a0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghpp"});
    temp["g0,h0,p0,p1"] += 1.0 * H1["g0,h1"] * T2["h0,h1,p0,p1"];
    temp["g0,h0,p0,p1"] +=
        (1.0 / 4.0) * H2["g0,p2,a0,a1"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a0,a1,a2,a3"];
    temp["g0,h0,p0,p1"] += (1.0 / 2.0) * H2["g0,a0,h1,h2"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    temp["g0,h0,p0,p1"] += 1.0 * H2["g0,a0,h1,a1"] * T3["h0,h1,a2,p0,p1,a3"] * L2["a1,a3,a0,a2"];
    temp["g0,h0,p0,p1"] +=
        (1.0 / 12.0) * H3["g0,p2,a0,a1,a2,a3"] * T3["h0,a4,a5,p0,p1,p2"] * L3["a1,a2,a3,a0,a4,a5"];
    temp["g0,h0,p0,p1"] +=
        (-1.0 / 4.0) * H3["g0,a0,a1,h1,h2,a2"] * T3["h0,h1,h2,p0,p1,a3"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,p0,p1"] +=
        (1.0 / 4.0) * H3["g0,a0,a1,h1,a2,a3"] * T3["h0,h1,a4,p0,p1,a5"] * L3["a2,a3,a5,a0,a1,a4"];
    temp["g0,h0,p0,p1"] +=
        (1.0 / 2.0) * H2["g0,p2,h1,h2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h1,h3"] * L1["h2,h4"];
    temp["g0,h0,p0,p1"] +=
        -1.0 * H2["g0,a0,h1,h2"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h2,h3"] * L1["a1,a0"];
    temp["g0,h0,p0,p1"] += (-1.0 / 2.0) * H3["g0,p2,a0,h1,a1,a2"] * T3["h0,h2,a3,p0,p1,p2"] *
                           L1["h1,h2"] * L2["a1,a2,a0,a3"];
    temp["g0,h0,p0,p1"] += (1.0 / 2.0) * H3["g0,a0,a1,h1,h2,a2"] * T3["h0,h1,h3,p0,p1,a3"] *
                           L1["h2,h3"] * L2["a2,a3,a0,a1"];
    temp["g0,h0,p0,p1"] += (1.0 / 2.0) * H3["g0,a0,a1,h1,a2,a3"] * T3["h0,h1,a4,p0,p1,a5"] *
                           L1["a5,a0"] * L2["a2,a3,a1,a4"];
    C2["g0,h0,p0,p1"] += temp["g0,h0,p0,p1"];
    C2["h0,g0,p0,p1"] -= temp["g0,h0,p0,p1"];
    C2["h0,h1,p0,p1"] += 1.0 * H1["p2,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h2,h3"];
    C2["h0,h1,p0,p1"] += -1.0 * H1["a0,h2"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    C2["h0,h1,p0,p1"] +=
        (-1.0 / 2.0) * H2["p2,a0,a1,a2"] * T3["h0,h1,a3,p0,p1,p2"] * L2["a1,a2,a0,a3"];
    C2["h0,h1,p0,p1"] +=
        (1.0 / 2.0) * H2["a0,a1,h2,a2"] * T3["h0,h1,h2,p0,p1,a3"] * L2["a2,a3,a0,a1"];
    C2["h0,h1,p0,p1"] +=
        (1.0 / 12.0) * H3["p2,a0,a1,a2,a3,a4"] * T3["h0,h1,a5,p0,p1,p2"] * L3["a2,a3,a4,a0,a1,a5"];
    C2["h0,h1,p0,p1"] +=
        (-1.0 / 12.0) * H3["a0,a1,a2,h2,a3,a4"] * T3["h0,h1,h2,p0,p1,a5"] * L3["a3,a4,a5,a0,a1,a2"];

    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gghggg"});
    temp["g3,g4,h0,g0,g1,g2"] += 1.0 * H3["g3,g4,p0,g0,g1,g2"] * T1["h0,p0"];
    C3["g3,g4,h0,g0,g1,g2"] += temp["g3,g4,h0,g0,g1,g2"];
    C3["g3,h0,g4,g0,g1,g2"] -= temp["g3,g4,h0,g0,g1,g2"];
    C3["h0,g3,g4,g0,g1,g2"] += temp["g3,g4,h0,g0,g1,g2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghhggg"});
    temp["g3,h0,h1,g0,g1,g2"] += (1.0 / 2.0) * H3["g3,p0,p1,g0,g1,g2"] * T2["h0,h1,p0,p1"];
    temp["g3,h0,h1,g0,g1,g2"] += -1.0 * H3["g3,p0,a0,g0,g1,g2"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp["g3,h0,h1,g0,g1,g2"] +=
        (-1.0 / 2.0) * H3["g3,p0,a0,g0,g1,g2"] * T3["h0,h1,a1,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C3["g3,h0,h1,g0,g1,g2"] += temp["g3,h0,h1,g0,g1,g2"];
    C3["h0,g3,h1,g0,g1,g2"] -= temp["g3,h0,h1,g0,g1,g2"];
    C3["h0,h1,g3,g0,g1,g2"] += temp["g3,h0,h1,g0,g1,g2"];
    C3["h0,h1,h2,g0,g1,g2"] += (1.0 / 6.0) * H3["p0,p1,p2,g0,g1,g2"] * T3["h0,h1,h2,p0,p1,p2"];
    C3["h0,h1,h2,g0,g1,g2"] +=
        (-1.0 / 2.0) * H3["p0,p1,a0,g0,g1,g2"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    C3["h0,h1,h2,g0,g1,g2"] +=
        (1.0 / 4.0) * H3["p0,a0,a1,g0,g1,g2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C3["h0,h1,h2,g0,g1,g2"] +=
        (1.0 / 2.0) * H3["p0,a0,a1,g0,g1,g2"] * T3["h0,h1,h2,p0,a2,a3"] * L1["a2,a0"] * L1["a3,a1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gggggp"});
    temp["g2,g3,g4,g0,g1,p0"] += -1.0 * H3["g2,g3,g4,g0,g1,h0"] * T1["h0,p0"];
    C3["g2,g3,g4,g0,g1,p0"] += temp["g2,g3,g4,g0,g1,p0"];
    C3["g2,g3,g4,g0,p0,g1"] -= temp["g2,g3,g4,g0,g1,p0"];
    C3["g2,g3,g4,p0,g0,g1"] += temp["g2,g3,g4,g0,g1,p0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gghggp"});
    temp["g2,g3,h0,g0,g1,p0"] += 1.0 * H3["g2,g3,p1,g0,g1,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    temp["g2,g3,h0,g0,g1,p0"] +=
        (1.0 / 2.0) * H3["g2,g3,p1,g0,g1,a0"] * T3["h0,a1,a2,p0,p1,a3"] * L2["a0,a3,a1,a2"];
    temp["g2,g3,h0,g0,g1,p0"] += -1.0 * H3["g2,g3,a0,g0,g1,h1"] * T2["h0,h1,p0,a1"] * L1["a1,a0"];
    temp["g2,g3,h0,g0,g1,p0"] +=
        (-1.0 / 2.0) * H3["g2,g3,a0,g0,g1,h1"] * T3["h0,h1,a1,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    C3["g2,g3,h0,g0,g1,p0"] += temp["g2,g3,h0,g0,g1,p0"];
    C3["g2,h0,g3,g0,g1,p0"] -= temp["g2,g3,h0,g0,g1,p0"];
    C3["h0,g2,g3,g0,g1,p0"] += temp["g2,g3,h0,g0,g1,p0"];
    C3["g2,g3,h0,g0,p0,g1"] -= temp["g2,g3,h0,g0,g1,p0"];
    C3["g2,h0,g3,g0,p0,g1"] += temp["g2,g3,h0,g0,g1,p0"];
    C3["h0,g2,g3,g0,p0,g1"] -= temp["g2,g3,h0,g0,g1,p0"];
    C3["g2,g3,h0,p0,g0,g1"] += temp["g2,g3,h0,g0,g1,p0"];
    C3["g2,h0,g3,p0,g0,g1"] -= temp["g2,g3,h0,g0,g1,p0"];
    C3["h0,g2,g3,p0,g0,g1"] += temp["g2,g3,h0,g0,g1,p0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghhggp"});
    temp["g2,h0,h1,g0,g1,p0"] += -1.0 * H2["g2,p1,g0,g1"] * T2["h0,h1,p0,p1"];
    temp["g2,h0,h1,g0,g1,p0"] +=
        (-1.0 / 2.0) * H3["g2,p1,p2,g0,g1,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h2,h3"];
    temp["g2,h0,h1,g0,g1,p0"] +=
        -1.0 * H3["g2,p1,a0,g0,g1,a1"] * T3["h0,h1,a2,p0,p1,a3"] * L2["a1,a3,a0,a2"];
    temp["g2,h0,h1,g0,g1,p0"] +=
        (-1.0 / 4.0) * H3["g2,a0,a1,g0,g1,h2"] * T3["h0,h1,h2,p0,a2,a3"] * L2["a2,a3,a0,a1"];
    temp["g2,h0,h1,g0,g1,p0"] +=
        1.0 * H3["g2,p1,a0,g0,g1,h2"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h2,h3"] * L1["a1,a0"];
    temp["g2,h0,h1,g0,g1,p0"] += (-1.0 / 2.0) * H3["g2,a0,a1,g0,g1,h2"] * T3["h0,h1,h2,p0,a2,a3"] *
                                 L1["a2,a0"] * L1["a3,a1"];
    C3["g2,h0,h1,g0,g1,p0"] += temp["g2,h0,h1,g0,g1,p0"];
    C3["h0,g2,h1,g0,g1,p0"] -= temp["g2,h0,h1,g0,g1,p0"];
    C3["h0,h1,g2,g0,g1,p0"] += temp["g2,h0,h1,g0,g1,p0"];
    C3["g2,h0,h1,g0,p0,g1"] -= temp["g2,h0,h1,g0,g1,p0"];
    C3["h0,g2,h1,g0,p0,g1"] += temp["g2,h0,h1,g0,g1,p0"];
    C3["h0,h1,g2,g0,p0,g1"] -= temp["g2,h0,h1,g0,g1,p0"];
    C3["g2,h0,h1,p0,g0,g1"] += temp["g2,h0,h1,g0,g1,p0"];
    C3["h0,g2,h1,p0,g0,g1"] -= temp["g2,h0,h1,g0,g1,p0"];
    C3["h0,h1,g2,p0,g0,g1"] += temp["g2,h0,h1,g0,g1,p0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"hhhggp"});
    temp["h0,h1,h2,g0,g1,p0"] += (1.0 / 2.0) * H2["p1,p2,g0,g1"] * T3["h0,h1,h2,p0,p1,p2"];
    temp["h0,h1,h2,g0,g1,p0"] += -1.0 * H2["p1,a0,g0,g1"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    temp["h0,h1,h2,g0,g1,p0"] +=
        (1.0 / 2.0) * H3["p1,a0,a1,g0,g1,a2"] * T3["h0,h1,h2,p0,p1,a3"] * L2["a2,a3,a0,a1"];
    C3["h0,h1,h2,g0,g1,p0"] += temp["h0,h1,h2,g0,g1,p0"];
    C3["h0,h1,h2,g0,p0,g1"] -= temp["h0,h1,h2,g0,g1,p0"];
    C3["h0,h1,h2,p0,g0,g1"] += temp["h0,h1,h2,g0,g1,p0"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ggggpp"});
    temp["g1,g2,g3,g0,p0,p1"] += (-1.0 / 2.0) * H3["g1,g2,g3,g0,h0,h1"] * T2["h0,h1,p0,p1"];
    temp["g1,g2,g3,g0,p0,p1"] += 1.0 * H3["g1,g2,g3,g0,h0,h1"] * T2["h0,h2,p0,p1"] * L1["h1,h2"];
    temp["g1,g2,g3,g0,p0,p1"] +=
        (1.0 / 2.0) * H3["g1,g2,g3,g0,h0,a0"] * T3["h0,a1,a2,p0,p1,a3"] * L2["a0,a3,a1,a2"];
    C3["g1,g2,g3,g0,p0,p1"] += temp["g1,g2,g3,g0,p0,p1"];
    C3["g1,g2,g3,p0,g0,p1"] -= temp["g1,g2,g3,g0,p0,p1"];
    C3["g1,g2,g3,p0,p1,g0"] += temp["g1,g2,g3,g0,p0,p1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gghgpp"});
    temp["g1,g2,h0,g0,p0,p1"] += 1.0 * H2["g1,g2,g0,h1"] * T2["h0,h1,p0,p1"];
    temp["g1,g2,h0,g0,p0,p1"] +=
        (1.0 / 4.0) * H3["g1,g2,p2,g0,a0,a1"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a0,a1,a2,a3"];
    temp["g1,g2,h0,g0,p0,p1"] +=
        (1.0 / 2.0) * H3["g1,g2,a0,g0,h1,h2"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    temp["g1,g2,h0,g0,p0,p1"] +=
        1.0 * H3["g1,g2,a0,g0,h1,a1"] * T3["h0,h1,a2,p0,p1,a3"] * L2["a1,a3,a0,a2"];
    temp["g1,g2,h0,g0,p0,p1"] +=
        (1.0 / 2.0) * H3["g1,g2,p2,g0,h1,h2"] * T3["h0,h3,h4,p0,p1,p2"] * L1["h1,h3"] * L1["h2,h4"];
    temp["g1,g2,h0,g0,p0,p1"] +=
        -1.0 * H3["g1,g2,a0,g0,h1,h2"] * T3["h0,h1,h3,p0,p1,a1"] * L1["h2,h3"] * L1["a1,a0"];
    C3["g1,g2,h0,g0,p0,p1"] += temp["g1,g2,h0,g0,p0,p1"];
    C3["g1,h0,g2,g0,p0,p1"] -= temp["g1,g2,h0,g0,p0,p1"];
    C3["h0,g1,g2,g0,p0,p1"] += temp["g1,g2,h0,g0,p0,p1"];
    C3["g1,g2,h0,p0,g0,p1"] -= temp["g1,g2,h0,g0,p0,p1"];
    C3["g1,h0,g2,p0,g0,p1"] += temp["g1,g2,h0,g0,p0,p1"];
    C3["h0,g1,g2,p0,g0,p1"] -= temp["g1,g2,h0,g0,p0,p1"];
    C3["g1,g2,h0,p0,p1,g0"] += temp["g1,g2,h0,g0,p0,p1"];
    C3["g1,h0,g2,p0,p1,g0"] -= temp["g1,g2,h0,g0,p0,p1"];
    C3["h0,g1,g2,p0,p1,g0"] += temp["g1,g2,h0,g0,p0,p1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghhgpp"});
    temp["g1,h0,h1,g0,p0,p1"] += 1.0 * H2["g1,p2,g0,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h2,h3"];
    temp["g1,h0,h1,g0,p0,p1"] += -1.0 * H2["g1,a0,g0,h2"] * T3["h0,h1,h2,p0,p1,a1"] * L1["a1,a0"];
    temp["g1,h0,h1,g0,p0,p1"] +=
        (-1.0 / 2.0) * H3["g1,p2,a0,g0,a1,a2"] * T3["h0,h1,a3,p0,p1,p2"] * L2["a1,a2,a0,a3"];
    temp["g1,h0,h1,g0,p0,p1"] +=
        (1.0 / 2.0) * H3["g1,a0,a1,g0,h2,a2"] * T3["h0,h1,h2,p0,p1,a3"] * L2["a2,a3,a0,a1"];
    C3["g1,h0,h1,g0,p0,p1"] += temp["g1,h0,h1,g0,p0,p1"];
    C3["h0,g1,h1,g0,p0,p1"] -= temp["g1,h0,h1,g0,p0,p1"];
    C3["h0,h1,g1,g0,p0,p1"] += temp["g1,h0,h1,g0,p0,p1"];
    C3["g1,h0,h1,p0,g0,p1"] -= temp["g1,h0,h1,g0,p0,p1"];
    C3["h0,g1,h1,p0,g0,p1"] += temp["g1,h0,h1,g0,p0,p1"];
    C3["h0,h1,g1,p0,g0,p1"] -= temp["g1,h0,h1,g0,p0,p1"];
    C3["g1,h0,h1,p0,p1,g0"] += temp["g1,h0,h1,g0,p0,p1"];
    C3["h0,g1,h1,p0,p1,g0"] -= temp["g1,h0,h1,g0,p0,p1"];
    C3["h0,h1,g1,p0,p1,g0"] += temp["g1,h0,h1,g0,p0,p1"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"hhhgpp"});
    temp["h0,h1,h2,g0,p0,p1"] += 1.0 * H1["p2,g0"] * T3["h0,h1,h2,p0,p1,p2"];
    C3["h0,h1,h2,g0,p0,p1"] += temp["h0,h1,h2,g0,p0,p1"];
    C3["h0,h1,h2,p0,g0,p1"] -= temp["h0,h1,h2,g0,p0,p1"];
    C3["h0,h1,h2,p0,p1,g0"] += temp["h0,h1,h2,g0,p0,p1"];
    C3["g0,g1,g2,p0,p1,p2"] += (-1.0 / 6.0) * H3["g0,g1,g2,h0,h1,h2"] * T3["h0,h1,h2,p0,p1,p2"];
    C3["g0,g1,g2,p0,p1,p2"] +=
        (1.0 / 2.0) * H3["g0,g1,g2,h0,h1,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h2,h3"];
    C3["g0,g1,g2,p0,p1,p2"] +=
        (-1.0 / 4.0) * H3["g0,g1,g2,h0,a0,a1"] * T3["h0,a2,a3,p0,p1,p2"] * L2["a0,a1,a2,a3"];
    C3["g0,g1,g2,p0,p1,p2"] += (-1.0 / 2.0) * H3["g0,g1,g2,h0,h1,h2"] * T3["h0,h3,h4,p0,p1,p2"] *
                               L1["h1,h3"] * L1["h2,h4"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"gghppp"});
    temp["g0,g1,h0,p0,p1,p2"] += (-1.0 / 2.0) * H2["g0,g1,h1,h2"] * T3["h0,h1,h2,p0,p1,p2"];
    temp["g0,g1,h0,p0,p1,p2"] += 1.0 * H2["g0,g1,h1,h2"] * T3["h0,h1,h3,p0,p1,p2"] * L1["h2,h3"];
    temp["g0,g1,h0,p0,p1,p2"] +=
        (-1.0 / 2.0) * H3["g0,g1,a0,h1,a1,a2"] * T3["h0,h1,a3,p0,p1,p2"] * L2["a1,a2,a0,a3"];
    C3["g0,g1,h0,p0,p1,p2"] += temp["g0,g1,h0,p0,p1,p2"];
    C3["g0,h0,g1,p0,p1,p2"] -= temp["g0,g1,h0,p0,p1,p2"];
    C3["h0,g0,g1,p0,p1,p2"] += temp["g0,g1,h0,p0,p1,p2"];
    temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"ghhppp"});
    temp["g0,h0,h1,p0,p1,p2"] += -1.0 * H1["g0,h2"] * T3["h0,h1,h2,p0,p1,p2"];
    C3["g0,h0,h1,p0,p1,p2"] += temp["g0,h0,h1,p0,p1,p2"];
    C3["h0,g0,h1,p0,p1,p2"] -= temp["g0,h0,h1,p0,p1,p2"];
    C3["h0,h1,g0,p0,p1,p2"] += temp["g0,h0,h1,p0,p1,p2"];

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
    H3["g0,g1,g2,g3,g4,g5"] = C3["g0,g1,g2,g3,g4,g5"];
    C3["g0,g1,g2,g3,g4,g5"] += H3["g3,g4,g5,g0,g1,g2"];
}
} // namespace forte
