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

#include "helpers.h"
#include "tensorsrg.h"

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

void TensorSRG::hermitian_commutator_A_B_C(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                           BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                           BlockedTensor& C1, BlockedTensor& C2) {
    if (options_.get_str("SRG_COMM") == "STANDARD") {
        hermitian_commutator_A_B_C_SRC(factor, A1, A2, B1, B2, C0, C1, C2);
    } else if (options_.get_str("SRG_COMM") == "FO") {
        hermitian_commutator_A_B_C_SRC_fourth_order(factor, A1, A2, B1, B2, C0, C1, C2);
    }
}

void TensorSRG::hermitian_commutator_A_B_C_SRC(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                               BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                               BlockedTensor& C1, BlockedTensor& C2) {
    // => Compute C = [A,B]_12 <= //

    //    hermitian_commutator_A1_B1_C0(A1,B1,+factor,C0);
    //    hermitian_commutator_A1_B2_C0(A1,B2,+factor,C0);
    //    hermitian_commutator_A1_B2_C0(B1,A2,-factor,C0);
    hermitian_commutator_A2_B2_C0(A2, B2, +factor, C0);

    //    hermitian_commutator_A1_B1_C1(A1,B1,+factor,C1);
    //    hermitian_commutator_A1_B2_C1(A1,B2,+factor,C1);
    //    hermitian_commutator_A1_B2_C1(B1,A2,+factor,C1);
    //    hermitian_commutator_A2_B2_C1(A2,B2,+factor,C1);

    //    hermitian_commutator_A1_B2_C2(A1,B2,+factor,C2);
    //    hermitian_commutator_A1_B2_C2(B1,A2,-factor,C2);
    hermitian_commutator_A2_B2_C2(A2, B2, +factor, C2);

    // => Add the term  + [B^+,A] <= //
    C0 *= 2.0;

    O1["pq"] = C1["pq"];
    O1["PQ"] = C1["PQ"];
    C1["pq"] += O1["qp"];
    C1["PQ"] += O1["QP"];

    O2["pqrs"] = C2["pqrs"];
    O2["pQrS"] = C2["pQrS"];
    O2["PQRS"] = C2["PQRS"];
    C2["pqrs"] += O2["rspq"];
    C2["pQrS"] += O2["rSpQ"];
    C2["PQRS"] += O2["RSPQ"];
}

void TensorSRG::hermitian_commutator_A_B_C_SRC_fourth_order(double factor, BlockedTensor& A1,
                                                            BlockedTensor& A2, BlockedTensor& B1,
                                                            BlockedTensor& B2, double& C0,
                                                            BlockedTensor& C1, BlockedTensor& C2) {
    // => Compute C = [A,B]_12 <= //

    hermitian_commutator_A1_B1_C0(A1, B1, +factor, C0);
    hermitian_commutator_A1_B2_C0(A1, B2, +factor, C0);
    hermitian_commutator_A1_B2_C0(B1, A2, -factor, C0);
    hermitian_commutator_A2_B2_C0(A2, B2, +factor, C0);

    hermitian_commutator_A1_B1_C1(A1, B1, +factor, C1);
    hermitian_commutator_A1_B2_C1(A1, B2, +factor, C1);
    hermitian_commutator_A1_B2_C1(B1, A2, +factor, C1);
    hermitian_commutator_A2_B2_C1(A2, B2, +2.0 * factor, C1);

    hermitian_commutator_A1_B2_C2(A1, B2, +factor, C2);
    hermitian_commutator_A1_B2_C2(B1, A2, -factor, C2);
    hermitian_commutator_A2_B2_C2(A2, B2, +factor, C2);

    // => Add the term  + [B^+,A] <= //
    C0 *= 2.0;

    O1["pq"] = C1["pq"];
    O1["PQ"] = C1["PQ"];
    C1["pq"] += O1["qp"];
    C1["PQ"] += O1["QP"];

    O2["pqrs"] = C2["pqrs"];
    O2["pQrS"] = C2["pQrS"];
    O2["PQRS"] = C2["PQRS"];
    C2["pqrs"] += O2["rspq"];
    C2["pQrS"] += O2["rSpQ"];
    C2["PQRS"] += O2["RSPQ"];
}

void TensorSRG::hermitian_commutator_A1_B1_C0(BlockedTensor& A, BlockedTensor& B, double alpha,
                                              double& C) {
    ForteTimer t;
    C += alpha * A["ai"] * B["ia"];
    C += alpha * A["AI"] * B["IA"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B1] -> C0 : %.4f", t.elapsed());
    }
    //    time_comm_A1_B1_C0 += t.elapsed();
}

void TensorSRG::hermitian_commutator_A1_B1_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                              BlockedTensor& C) {
    ForteTimer t;

    C["ip"] += alpha * A["ap"] * B["ia"];
    C["qa"] -= alpha * B["ia"] * A["qi"];

    C["IP"] += alpha * A["AP"] * B["IA"];
    C["QA"] -= alpha * B["IA"] * A["QI"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B1] -> C1 : %.4f", t.elapsed());
    }
    //    time_comm_A1_B1_C1 += t.elapsed();
}

void TensorSRG::hermitian_commutator_A1_B2_C0(BlockedTensor& A, BlockedTensor& B, double alpha,
                                              double& C) {}

void TensorSRG::hermitian_commutator_A1_B2_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                              BlockedTensor& C) {
    ForteTimer t;
    C["qp"] += alpha * A["sr"] * B["qrps"];
    C["qp"] += alpha * A["SR"] * B["qRpS"];
    C["QP"] += alpha * A["SR"] * B["QRPS"];
    C["QP"] += alpha * A["sr"] * B["rQsP"];
    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B2] -> C1 : %.4f", t.elapsed());
    }
    //    time_comm_A1_B2_C1 += t.elapsed();
}

void TensorSRG::hermitian_commutator_A1_B2_C2(BlockedTensor& A, BlockedTensor& B, double alpha,
                                              BlockedTensor& C) {
    ForteTimer t;

    C["rspq"] += alpha * A["tp"] * B["rstq"];
    C["rspq"] += alpha * A["tq"] * B["rspt"];
    C["rspq"] -= alpha * A["rt"] * B["tspq"];
    C["rspq"] -= alpha * A["st"] * B["rtpq"];

    //    Could rewrite the contractions about as:
    //    C["rspq"] += alpha * A["tp"] * B["rstq"];
    //    C["rsqp"] -= alpha * A["tp"] * B["rstq"];
    //    C["rspq"] -= alpha * A["rt"] * B["tspq"];
    //    C["srpq"] += alpha * A["rt"] * B["tspq"];

    //    // These are here to account for B^dagger
    //    C["rspq"] -= alpha * A["tp"] * B["tqrs"];
    //    C["rspq"] -= alpha * A["tq"] * B["ptrs"];
    //    C["rspq"] += alpha * A["rt"] * B["pqts"];
    //    C["rspq"] += alpha * A["st"] * B["pqrt"];

    C["rSpQ"] += alpha * A["tp"] * B["rStQ"];
    C["rSpQ"] += alpha * A["TQ"] * B["rSpT"];
    C["rSpQ"] -= alpha * A["rt"] * B["tSpQ"];
    C["rSpQ"] -= alpha * A["ST"] * B["rTpQ"];

    //    C["rSpQ"] -= alpha * A["tp"] * B["tQrS"];
    //    C["rSpQ"] -= alpha * A["TQ"] * B["pTrS"];
    //    C["rSpQ"] += alpha * A["rt"] * B["pQtS"];
    //    C["rSpQ"] += alpha * A["ST"] * B["pQrT"];

    C["RSPQ"] += alpha * A["TP"] * B["RSTQ"];
    C["RSPQ"] += alpha * A["TQ"] * B["RSPT"];
    C["RSPQ"] -= alpha * A["RT"] * B["TSPQ"];
    C["RSPQ"] -= alpha * A["ST"] * B["RTPQ"];

    //    C["RSPQ"] -= alpha * A["TP"] * B["TQRS"];
    //    C["RSPQ"] -= alpha * A["TQ"] * B["PTRS"];
    //    C["RSPQ"] += alpha * A["RT"] * B["PQTS"];
    //    C["RSPQ"] += alpha * A["ST"] * B["PQRT"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B2] -> C2 : %.4f", t.elapsed());
    }
    //    time_comm_A1_B2_C2 += t.elapsed();
}

void TensorSRG::hermitian_commutator_A2_B2_C0(BlockedTensor& A, BlockedTensor& B, double alpha,
                                              double& C) {
    ForteTimer t;

    C += alpha * 0.25 * A["abij"] * B["ijab"];
    C += alpha * 1.00 * A["aBiJ"] * B["iJaB"];
    C += alpha * 0.25 * A["ABIJ"] * B["IJAB"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C0 : %.4f", t.elapsed());
    }
    //    time_comm_A2_B2_C0 += t.elapsed();
}

void TensorSRG::hermitian_commutator_A2_B2_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                              BlockedTensor& C) {
    ForteTimer t;

    C["qp"] += +0.5 * alpha * A["abip"] * B["iqab"];
    C["qp"] += +1.0 * alpha * A["aBpI"] * B["qIaB"];

    C["qp"] += -0.5 * alpha * A["aqij"] * B["ijap"];
    C["qp"] += -1.0 * alpha * A["qAiJ"] * B["iJpA"];

    C["QP"] += +0.5 * alpha * A["ABIP"] * B["IQAB"];
    C["QP"] += +1.0 * alpha * A["aBiP"] * B["iQaB"];

    C["QP"] += -0.5 * alpha * A["AQIJ"] * B["IJAP"];
    C["QP"] += -1.0 * alpha * A["aQiJ"] * B["iJaP"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C1 : %.4f", t.elapsed());
    }
    //    time_comm_A2_B2_C1 += t.elapsed();
}

void TensorSRG::hermitian_commutator_A2_B2_C2(BlockedTensor& A, BlockedTensor& B, double alpha,
                                              BlockedTensor& C) {
    ForteTimer t;

    // AAAA case (these work only in the single-reference case)
    // Term I
    C["ijpq"] += 0.5 * alpha * A["abpq"] * B["ijab"];
    C["pqab"] += 0.5 * alpha * A["pqij"] * B["ijab"];

    I_ioiv["rspq"] = alpha * A["rupv"] * B["svqu"];
    I_ioiv["rspq"] += alpha * A["rUpV"] * B["sVqU"];
    C["rspq"] += I_ioiv["rspq"];
    C["rspq"] -= I_ioiv["rsqp"];
    C["rspq"] -= I_ioiv["srpq"];
    C["rspq"] += I_ioiv["srqp"];

    // ABAB case (these work only in the single-reference case)
    // Term I
    C["iJpQ"] += alpha * A["aBpQ"] * B["iJaB"];
    C["pQaB"] += alpha * A["pQiJ"] * B["iJaB"];

    //    // Term II
    //    C["rSpQ"] += alpha * A["rupv"] * B["vSuQ"];
    //    C["rSpQ"] += alpha * A["rUpV"] * B["VSUQ"];
    //    C["rSpQ"] += -alpha * A["rUvQ"] * B["vSpU"];
    //    C["rSpQ"] += -alpha * A["uSpV"] * B["rVuQ"];
    //    C["rSpQ"] += alpha * A["uSvQ"] * B["rvpu"];
    //    C["rSpQ"] += alpha * A["USVQ"] * B["rVpU"];

    // BBBB case (these work only in the single-reference case)
    // Term I
    C["IJPQ"] += 0.5 * alpha * A["ABPQ"] * B["IJAB"];
    C["PQAB"] += 0.5 * alpha * A["PQIJ"] * B["IJAB"];

    I_ioiv["RSPQ"] = alpha * A["uRvP"] * B["vSuQ"];
    I_ioiv["RSPQ"] += alpha * A["RUPV"] * B["SVQU"];
    C["RSPQ"] += I_ioiv["RSPQ"];
    C["RSPQ"] -= I_ioiv["RSQP"];
    C["RSPQ"] -= I_ioiv["SRPQ"];
    C["RSPQ"] += I_ioiv["SRQP"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C2 : %.4f", t.elapsed());
    }
    //    time_comm_A2_B2_C2 += t.elapsed();
}
}
} // EndNamespaces
