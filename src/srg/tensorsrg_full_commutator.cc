/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "helpers/timer.h"
#include "helpers.h"
#include "srg/tensorsrg.h"

using namespace psi;

namespace psi {
namespace forte {

extern double time_comm_A1_B1_C0;
extern double time_comm_A1_B1_C1;
extern double time_comm_A1_B2_C0;
extern double time_comm_A1_B2_C1;
extern double time_comm_A1_B2_C2;
extern double time_comm_A2_B2_C0;
extern double time_comm_A2_B2_C1;
extern double time_comm_A2_B2_C2;

void TensorSRG::full_commutator_A_B_C(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                      BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                      BlockedTensor& C1, BlockedTensor& C2) {
    if (options_.get_str("SRG_COMM") == "STANDARD") {
        full_commutator_A_B_C_SRC(factor, A1, A2, B1, B2, C0, C1, C2);
    } else if (options_.get_str("SRG_COMM") == "FO") {
        full_commutator_A_B_C_SRC_fourth_order(factor, A1, A2, B1, B2, C0, C1, C2);
    }
}

void TensorSRG::full_commutator_A_B_C_SRC(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                          BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                          BlockedTensor& C1, BlockedTensor& C2) {
    // => Compute C = [A,B]_12 <= //

    full_commutator_A1_B1_C0(A1, B1, +factor, C0);
    //    full_commutator_A1_B2_C0(A1,B2,+factor,C0);
    //    full_commutator_A1_B2_C0(B1,A2,-factor,C0);
    full_commutator_A2_B2_C0(A2, B2, +factor, C0);

    full_commutator_A1_B1_C1(A1, B1, +factor, C1);
    full_commutator_A1_B2_C1(A1, B2, +factor, C1);
    full_commutator_A1_B2_C1(B1, A2, -factor, C1);
    full_commutator_A2_B2_C1(A2, B2, +factor, C1);

    full_commutator_A1_B2_C2(A1, B2, +factor, C2);
    full_commutator_A1_B2_C2(B1, A2, -factor, C2);
    full_commutator_A2_B2_C2(A2, B2, +factor, C2);
}

void TensorSRG::full_commutator_A_B_C_SRC_fourth_order(double factor, BlockedTensor& A1,
                                                       BlockedTensor& A2, BlockedTensor& B1,
                                                       BlockedTensor& B2, double& C0,
                                                       BlockedTensor& C1, BlockedTensor& C2) {
    // => Compute C = [A,B]_12 <= //

    full_commutator_A1_B1_C0(A1, B1, +factor, C0);
    //    commutator_A1_B2_C0(A1,B2,+factor,C0);
    //    commutator_A1_B2_C0(B1,A2,-factor,C0);
    full_commutator_A2_B2_C0(A2, B2, +factor, C0);

    full_commutator_A1_B1_C1(A1, B1, +factor, C1);
    full_commutator_A1_B2_C1(A1, B2, +factor, C1);
    full_commutator_A1_B2_C1(B1, A2, +factor, C1);
    full_commutator_A2_B2_C1(A2, B2, +2.0 * factor, C1);

    full_commutator_A1_B2_C2(A1, B2, +factor, C2);
    full_commutator_A1_B2_C2(B1, A2, -factor, C2);
    full_commutator_A2_B2_C2(A2, B2, +factor, C2);
}

void TensorSRG::full_commutator_A1_B1_C0(BlockedTensor& A, BlockedTensor& B, double alpha,
                                         double& C) {
    local_timer t;
    C += alpha * A["qi"] * B["iq"];
    C -= alpha * B["qi"] * A["iq"];
    C += alpha * A["QI"] * B["IQ"];
    C -= alpha * B["QI"] * A["IQ"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B1] -> C0 : %.4f", t.get());
    }
    //    time_comm_A1_B1_C0 += t.get();
}

void TensorSRG::full_commutator_A1_B1_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                         BlockedTensor& C) {
    local_timer t;

    C["qp"] += alpha * A["rp"] * B["qr"];
    C["qp"] -= alpha * B["rp"] * A["qr"];

    C["QP"] += alpha * A["RP"] * B["QR"];
    C["QP"] -= alpha * B["RP"] * A["QR"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B1] -> C1 : %.4f", t.get());
    }
    //    time_comm_A1_B1_C1 += t.get();
}

void TensorSRG::full_commutator_A1_B2_C0(BlockedTensor& A, BlockedTensor& B, double alpha,
                                         double& C) {}

void TensorSRG::full_commutator_A1_B2_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                         BlockedTensor& C) {
    local_timer t;
    C["qp"] += alpha * A["ai"] * B["qipa"];
    C["qp"] -= alpha * A["ia"] * B["qapi"];
    C["qp"] += alpha * A["AI"] * B["qIpA"];
    C["qp"] -= alpha * A["IA"] * B["qApI"];

    C["QP"] += alpha * A["ai"] * B["iQaP"];
    C["QP"] -= alpha * A["ia"] * B["aQiP"];
    C["QP"] += alpha * A["AI"] * B["IQAP"];
    C["QP"] -= alpha * A["IA"] * B["AQIP"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B2] -> C1 : %.4f", t.get());
    }
    time_comm_A1_B2_C1 += t.get();
}

void TensorSRG::full_commutator_A1_B2_C2(BlockedTensor& A, BlockedTensor& B, double alpha,
                                         BlockedTensor& C) {
    local_timer t;

    C["rspq"] += alpha * A["tp"] * B["rstq"];
    C["rspq"] += alpha * A["tq"] * B["rspt"];
    C["rspq"] -= alpha * A["rt"] * B["tspq"];
    C["rspq"] -= alpha * A["st"] * B["rtpq"];

    C["rSpQ"] += alpha * A["tp"] * B["rStQ"];
    C["rSpQ"] += alpha * A["TQ"] * B["rSpT"];
    C["rSpQ"] -= alpha * A["rt"] * B["tSpQ"];
    C["rSpQ"] -= alpha * A["ST"] * B["rTpQ"];

    C["RSPQ"] += alpha * A["TP"] * B["RSTQ"];
    C["RSPQ"] += alpha * A["TQ"] * B["RSPT"];
    C["RSPQ"] -= alpha * A["RT"] * B["TSPQ"];
    C["RSPQ"] -= alpha * A["ST"] * B["RTPQ"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B2] -> C2 : %.4f", t.get());
    }
    time_comm_A1_B2_C2 += t.get();
}

void TensorSRG::full_commutator_A2_B2_C0(BlockedTensor& A, BlockedTensor& B, double alpha,
                                         double& C) {
    local_timer t;

    C += alpha * 0.25 * A["abij"] * B["ijab"];
    C += alpha * 1.0 * A["aBiJ"] * B["iJaB"];
    C += alpha * 0.25 * A["ABIJ"] * B["IJAB"];
    C -= alpha * 0.25 * B["abij"] * A["ijab"];
    C -= alpha * 1.0 * B["aBiJ"] * A["iJaB"];
    C -= alpha * 0.25 * B["ABIJ"] * A["IJAB"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C0 : %.4f", t.get());
    }
    time_comm_A2_B2_C0 += t.get();
}

void TensorSRG::full_commutator_A2_B2_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                         BlockedTensor& C) {
    local_timer t;

    C["qp"] += +0.5 * alpha * A["ijap"] * B["aqij"];
    C["qp"] -= +0.5 * alpha * B["ijap"] * A["aqij"];

    C["qp"] += +1.0 * alpha * A["iJpA"] * B["qAiJ"];
    C["qp"] -= +1.0 * alpha * B["iJpA"] * A["qAiJ"];

    C["qp"] += +0.5 * alpha * A["abip"] * B["iqab"];
    C["qp"] -= +0.5 * alpha * B["abip"] * A["iqab"];

    C["qp"] += +1.0 * alpha * A["bApI"] * B["qIbA"];
    C["qp"] -= +1.0 * alpha * B["bApI"] * A["qIbA"];

    C["QP"] += +0.5 * alpha * A["IJAP"] * B["AQIJ"];
    C["QP"] -= +0.5 * alpha * B["IJAP"] * A["AQIJ"];

    C["QP"] += +1.0 * alpha * A["iJaP"] * B["aQiJ"];
    C["QP"] -= +1.0 * alpha * B["iJaP"] * A["aQiJ"];

    C["QP"] += +0.5 * alpha * A["ABIP"] * B["IQAB"];
    C["QP"] -= +0.5 * alpha * B["ABIP"] * A["IQAB"];

    C["QP"] += +1.0 * alpha * A["aBiP"] * B["iQaB"];
    C["QP"] -= +1.0 * alpha * B["aBiP"] * A["iQaB"];
    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C1 : %.4f", t.get());
    }
    time_comm_A2_B2_C1 += t.get();
}

void TensorSRG::full_commutator_A2_B2_C2(BlockedTensor& A, BlockedTensor& B, double alpha,
                                         BlockedTensor& C) {
    local_timer t;

    // AAAA case (these work only in the single-reference case)
    // Term I
    C["rspq"] += 0.5 * alpha * A["abpq"] * B["rsab"];
    C["rspq"] -= 0.5 * alpha * B["abpq"] * A["rsab"];

    C["rspq"] -= 0.5 * alpha * A["ijpq"] * B["rsij"];
    C["rspq"] += 0.5 * alpha * B["ijpq"] * A["rsij"];

    // Term II
    C["rspq"] += alpha * A["rapi"] * B["siqa"];
    C["rspq"] -= alpha * A["raqi"] * B["sipa"];
    C["rspq"] -= alpha * A["sapi"] * B["riqa"];
    C["rspq"] += alpha * A["saqi"] * B["ripa"];
    C["rspq"] -= alpha * A["ripa"] * B["saqi"];
    C["rspq"] += alpha * A["riqa"] * B["sapi"];
    C["rspq"] += alpha * A["sipa"] * B["raqi"];
    C["rspq"] -= alpha * A["siqa"] * B["rapi"];

    C["rspq"] += alpha * A["rApI"] * B["sIqA"];
    C["rspq"] -= alpha * A["rAqI"] * B["sIpA"];
    C["rspq"] -= alpha * A["sApI"] * B["rIqA"];
    C["rspq"] += alpha * A["sAqI"] * B["rIpA"];
    C["rspq"] -= alpha * A["rIpA"] * B["sAqI"];
    C["rspq"] += alpha * A["rIqA"] * B["sApI"];
    C["rspq"] += alpha * A["sIpA"] * B["rAqI"];
    C["rspq"] -= alpha * A["sIqA"] * B["rApI"];

    // ABAB case (these work only in the single-reference case)
    // Term I
    C["rSpQ"] += alpha * A["aBpQ"] * B["rSaB"];
    C["rSpQ"] -= alpha * B["aBpQ"] * A["rSaB"];

    C["rSpQ"] -= alpha * A["iJpQ"] * B["rSiJ"];
    C["rSpQ"] += alpha * B["iJpQ"] * A["rSiJ"];

    // Term II
    C["rSpQ"] += alpha * A["rapi"] * B["iSaQ"];
    C["rSpQ"] -= alpha * A["ripa"] * B["aSiQ"];
    C["rSpQ"] += alpha * A["rApI"] * B["ISAQ"];
    C["rSpQ"] -= alpha * A["rIpA"] * B["ASIQ"];

    C["rSpQ"] -= alpha * A["rAiQ"] * B["iSpA"];
    C["rSpQ"] += alpha * A["rIaQ"] * B["aSpI"];

    C["rSpQ"] -= alpha * A["aSpI"] * B["rIaQ"];
    C["rSpQ"] += alpha * A["iSpA"] * B["rAiQ"];

    C["rSpQ"] += alpha * B["ripa"] * A["aSiQ"];
    C["rSpQ"] -= alpha * B["rapi"] * A["iSaQ"];
    C["rSpQ"] += alpha * B["rIpA"] * A["ASIQ"];
    C["rSpQ"] -= alpha * B["rApI"] * A["ISAQ"];

    // BBBB case (these work only in the single-reference case)
    // Term I
    C["RSPQ"] += 0.5 * alpha * A["ABPQ"] * B["RSAB"];
    C["RSPQ"] -= 0.5 * alpha * B["ABPQ"] * A["RSAB"];

    C["RSPQ"] -= 0.5 * alpha * A["IJPQ"] * B["RSIJ"];
    C["RSPQ"] += 0.5 * alpha * B["IJPQ"] * A["RSIJ"];

    // Term II
    C["RSPQ"] += alpha * A["RAPI"] * B["SIQA"];
    C["RSPQ"] -= alpha * A["RAQI"] * B["SIPA"];
    C["RSPQ"] -= alpha * A["SAPI"] * B["RIQA"];
    C["RSPQ"] += alpha * A["SAQI"] * B["RIPA"];
    C["RSPQ"] -= alpha * A["RIPA"] * B["SAQI"];
    C["RSPQ"] += alpha * A["RIQA"] * B["SAPI"];
    C["RSPQ"] += alpha * A["SIPA"] * B["RAQI"];
    C["RSPQ"] -= alpha * A["SIQA"] * B["RAPI"];

    C["RSPQ"] += alpha * A["aRiP"] * B["iSaQ"];
    C["RSPQ"] -= alpha * A["aRiQ"] * B["iSaP"];
    C["RSPQ"] -= alpha * A["aSiP"] * B["iRaQ"];
    C["RSPQ"] += alpha * A["aSiQ"] * B["iRaP"];
    C["RSPQ"] -= alpha * A["iRaP"] * B["aSiQ"];
    C["RSPQ"] += alpha * A["iRaQ"] * B["aSiP"];
    C["RSPQ"] += alpha * A["iSaP"] * B["aRiQ"];
    C["RSPQ"] -= alpha * A["iSaQ"] * B["aRiP"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C2 : %.4f", t.get());
    }
    time_comm_A2_B2_C2 += t.get();
}
}
}
