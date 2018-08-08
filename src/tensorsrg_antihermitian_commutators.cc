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
#include "psi4/libpsi4util/libpsi4util.h"

#include "helpers.h"
#include "tensorsrg.h"

using namespace psi;

namespace psi {
namespace forte {

double time_comm_A1_B1_C0 = 0;
double time_comm_A1_B1_C1 = 0;
double time_comm_A1_B2_C0 = 0;
double time_comm_A1_B2_C1 = 0;
double time_comm_A1_B2_C2 = 0;
double time_comm_A2_B2_C0 = 0;
double time_comm_A2_B2_C1 = 0;
double time_comm_A2_B2_C2 = 0;
double t_tensor = 0;
double t_four = 0;
int ncalls = 0;

void TensorSRG::commutator_A_B_C(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                 BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                 BlockedTensor& C1, BlockedTensor& C2, int order) {
    if (options_.get_str("SRG_COMM") == "STANDARD") {
        commutator_A_B_C_SRC(factor, A1, A2, B1, B2, C0, C1, C2);
    } else if (options_.get_str("SRG_COMM") == "FO") {
        commutator_A_B_C_SRC_fourth_order(factor, A1, A2, B1, B2, C0, C1, C2);
    } else if (options_.get_str("SRG_COMM") == "FO2") {
        if (order < 2) {
            commutator_A_B_C_SRC(factor, A1, A2, B1, B2, C0, C1, C2);
        } else {
            commutator_A_B_C_SRC_fourth_order2(factor, A1, A2, B1, B2, C0, C1, C2);
        }
    }
    ncalls += 1;
}

void TensorSRG::commutator_A_B_C_SRC(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                     BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                     BlockedTensor& C1, BlockedTensor& C2) {
    // => Compute C = [A,B]_12 <= //

    commutator_A1_B1_C0(A1, B1, +factor, C0);
    commutator_A1_B2_C0(A1, B2, +factor, C0);
    commutator_A1_B2_C0(B1, A2, -factor, C0);
    commutator_A2_B2_C0(A2, B2, +factor, C0);

    commutator_A1_B1_C1(A1, B1, +factor, C1);
    commutator_A1_B2_C1(A1, B2, +factor, C1);
    commutator_A1_B2_C1(B1, A2, +factor, C1);
    commutator_A2_B2_C1_simplified(A2, B2, +factor, C1);

    commutator_A1_B2_C2(A1, B2, +factor, C2);
    commutator_A1_B2_C2(B1, A2, -factor, C2);
    commutator_A2_B2_C2(A2, B2, +factor, C2);

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

void TensorSRG::commutator_A_B_C_SRC_fourth_order(double factor, BlockedTensor& A1,
                                                  BlockedTensor& A2, BlockedTensor& B1,
                                                  BlockedTensor& B2, double& C0, BlockedTensor& C1,
                                                  BlockedTensor& C2) {
    // => Compute C = [A,B]_12 <= //

    commutator_A1_B1_C0(A1, B1, +factor, C0);
    commutator_A1_B2_C0(A1, B2, +factor, C0);
    commutator_A1_B2_C0(B1, A2, -factor, C0);
    commutator_A2_B2_C0(A2, B2, +factor, C0);

    commutator_A1_B1_C1(A1, B1, +factor, C1);
    commutator_A1_B2_C1(A1, B2, +factor, C1);
    commutator_A1_B2_C1(B1, A2, +factor, C1);
    commutator_A2_B2_C1_fo(A2, B2, factor, C1); // <-- use approximate fourth-order

    commutator_A1_B2_C2(A1, B2, +factor, C2);
    commutator_A1_B2_C2(B1, A2, -factor, C2);
    commutator_A2_B2_C2(A2, B2, +factor, C2);

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

void TensorSRG::commutator_A_B_C_SRC_fourth_order2(double factor, BlockedTensor& A1,
                                                   BlockedTensor& A2, BlockedTensor& B1,
                                                   BlockedTensor& B2, double& C0, BlockedTensor& C1,
                                                   BlockedTensor& C2) {
    // => Compute C = [A,B]_12 <= //

    commutator_A1_B1_C0(A1, B1, +factor, C0);
    commutator_A1_B2_C0(A1, B2, +factor, C0);
    commutator_A1_B2_C0(B1, A2, -factor, C0);
    commutator_A2_B2_C0(A2, B2, +factor, C0);

    commutator_A1_B1_C1(A1, B1, +factor, C1);
    commutator_A1_B2_C1(A1, B2, +factor, C1);
    commutator_A1_B2_C1(B1, A2, +factor, C1);
    commutator_A2_B2_C1(A2, B2, factor, C1);

    commutator_A1_B2_C2_fo(A1, B2, +factor, C2);
    commutator_A1_B2_C2(B1, A2, -factor, C2);
    commutator_A2_B2_C2(A2, B2, +factor, C2);

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

void TensorSRG::commutator_A1_B1_C0(BlockedTensor& A, BlockedTensor& B, double alpha, double& C) {
    Timer t;
    C += alpha * A["ai"] * B["ia"];
    C += alpha * A["AI"] * B["IA"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B1] -> C0 : %.4f", t.get());
    }
    time_comm_A1_B1_C0 += t.get();
}

void TensorSRG::commutator_A1_B1_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                    BlockedTensor& C) {
    Timer t;

    C["ip"] += alpha * A["ap"] * B["ia"];
    C["qa"] -= alpha * B["ia"] * A["qi"];

    C["IP"] += alpha * A["AP"] * B["IA"];
    C["QA"] -= alpha * B["IA"] * A["QI"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B1] -> C1 : %.4f", t.get());
    }
    time_comm_A1_B1_C1 += t.get();
}

void TensorSRG::commutator_A1_B2_C0(BlockedTensor& A, BlockedTensor& B, double alpha, double& C) {}

void TensorSRG::commutator_A1_B2_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                    BlockedTensor& C) {
    Timer t;
    C["qp"] += alpha * A["sr"] * B["qrps"];
    C["qp"] += alpha * A["SR"] * B["qRpS"];
    C["QP"] += alpha * A["SR"] * B["QRPS"];
    C["QP"] += alpha * A["sr"] * B["rQsP"];
    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B2] -> C1 : %.4f", t.get());
    }
    time_comm_A1_B2_C1 += t.get();
}

void TensorSRG::commutator_A1_B2_C2(BlockedTensor& A, BlockedTensor& B, double alpha,
                                    BlockedTensor& C) {
    Timer t;

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
        outfile->Printf("\n  Time for [A1,B2] -> C2 : %.4f", t.get());
    }
    time_comm_A1_B2_C2 += t.get();
}

void TensorSRG::commutator_A1_B2_C2_fo(BlockedTensor& A, BlockedTensor& B, double alpha,
                                       BlockedTensor& C) {
    Timer t;

    // The point of this routine is to test a different way to correct for forth
    // order terms missing
    // in the BCS approximation.  The idea behind commutator_A2_B2_C1_fo is to
    // double the one-body
    // term [V,T_2]_1 to simulate the contribution of 1/2 [[V,T_2]_3,T_2]_2,
    // namely
    // [[V,T_2]_1,T_2].
    // Here we explore another route, where instead of modifying this term, we
    // just add twice of its
    // contribution directly to the commutator [[V,T_2]_1,T_2].

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

    C["rskq"] += alpha * A["jk"] * B["rsjq"];
    C["rspk"] += alpha * A["jk"] * B["rspj"];
    C["jspq"] -= alpha * A["jk"] * B["kspq"];
    C["rjpq"] -= alpha * A["jk"] * B["rkpq"];

    C["rSkQ"] += alpha * A["jk"] * B["rSjQ"];
    C["rSpK"] += alpha * A["JK"] * B["rSpJ"];
    C["jSpQ"] -= alpha * A["jk"] * B["kSpQ"];
    C["rJpQ"] -= alpha * A["JK"] * B["rKpQ"];

    C["RSKQ"] += alpha * A["JK"] * B["RSJQ"];
    C["RSPK"] += alpha * A["JK"] * B["RSPJ"];
    C["JSPQ"] -= alpha * A["JK"] * B["KSPQ"];
    C["RJPQ"] -= alpha * A["JK"] * B["RKPQ"];

    C["rscq"] += alpha * A["bc"] * B["rsbq"];
    C["rspc"] += alpha * A["bc"] * B["rspb"];
    C["bspq"] -= alpha * A["bc"] * B["cspq"];
    C["rbpq"] -= alpha * A["bc"] * B["rcpq"];

    C["rScQ"] += alpha * A["bc"] * B["rSbQ"];
    C["rSpC"] += alpha * A["BC"] * B["rSpB"];
    C["bSpQ"] -= alpha * A["bc"] * B["cSpQ"];
    C["rBpQ"] -= alpha * A["BC"] * B["rCpQ"];

    C["RSCQ"] += alpha * A["BC"] * B["RSBQ"];
    C["RSPC"] += alpha * A["BC"] * B["RSPB"];
    C["BSPQ"] -= alpha * A["BC"] * B["CSPQ"];
    C["RBPQ"] -= alpha * A["BC"] * B["RCPQ"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A1,B2] -> C2 : %.4f", t.get());
    }
    time_comm_A1_B2_C2 += t.get();
}

void TensorSRG::commutator_A2_B2_C0(BlockedTensor& A, BlockedTensor& B, double alpha, double& C) {
    Timer t;

    C += alpha * 0.25 * A["abij"] * B["ijab"];
    C += alpha * 1.00 * A["aBiJ"] * B["iJaB"];
    C += alpha * 0.25 * A["ABIJ"] * B["IJAB"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C0 : %.4f", t.get());
    }
    time_comm_A2_B2_C0 += t.get();
}

void TensorSRG::commutator_A2_B2_C1(BlockedTensor& A, BlockedTensor& B, double alpha,
                                    BlockedTensor& C) {
    Timer t;

    C["qp"] += +0.5 * alpha * A["abip"] * B["iqab"];
    C["qp"] += +1.0 * alpha * A["aBpI"] * B["qIaB"];

    C["qp"] += -0.5 * alpha * A["aqij"] * B["ijap"];
    C["qp"] += -1.0 * alpha * A["qAiJ"] * B["iJpA"];

    C["QP"] += +0.5 * alpha * A["ABIP"] * B["IQAB"];
    C["QP"] += +1.0 * alpha * A["aBiP"] * B["iQaB"];

    C["QP"] += -0.5 * alpha * A["AQIJ"] * B["IJAP"];
    C["QP"] += -1.0 * alpha * A["aQiJ"] * B["iJaP"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C1 : %.4f", t.get());
    }
    time_comm_A2_B2_C1 += t.get();
}

void TensorSRG::commutator_A2_B2_C1_simplified(BlockedTensor& A, BlockedTensor& B, double alpha,
                                               BlockedTensor& C) {
    Timer t;

    C["jk"] += +0.5 * alpha * A["abik"] * B["ijab"];
    C["jk"] += +1.0 * alpha * A["aBkI"] * B["jIaB"];

    C["jc"] += +0.5 * alpha * A["abic"] * B["ijab"];
    C["jc"] += +1.0 * alpha * A["aBcI"] * B["jIaB"];

    C["kb"] += -0.5 * alpha * A["akij"] * B["ijab"];
    C["kb"] += -1.0 * alpha * A["kAiJ"] * B["iJbA"];

    C["cb"] += -0.5 * alpha * A["acij"] * B["ijab"];
    C["cb"] += -1.0 * alpha * A["cAiJ"] * B["iJbA"];

    C["JK"] += +0.5 * alpha * A["ABIK"] * B["IJAB"];
    C["JK"] += +1.0 * alpha * A["aBiK"] * B["iJaB"];

    C["JC"] += +0.5 * alpha * A["ABIC"] * B["IJAB"];
    C["JC"] += +1.0 * alpha * A["aBiC"] * B["iJaB"];

    C["KB"] += -0.5 * alpha * A["AKIJ"] * B["IJAB"];
    C["KB"] += -1.0 * alpha * A["aKiJ"] * B["iJaB"];

    C["CB"] += -0.5 * alpha * A["ACIJ"] * B["IJAB"];
    C["CB"] += -1.0 * alpha * A["aCiJ"] * B["iJaB"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C1 : %.4f", t.get());
    }
    time_comm_A2_B2_C1 += t.get();
}

void TensorSRG::commutator_A2_B2_C1_fo(BlockedTensor& A, BlockedTensor& B, double alpha,
                                       BlockedTensor& C) {
    Timer t;

    C["jk"] += +1.0 * alpha * A["abik"] * B["ijab"];
    C["jk"] += +2.0 * alpha * A["aBkI"] * B["jIaB"];

    C["jc"] += +0.5 * alpha * A["abic"] * B["ijab"];
    C["jc"] += +1.0 * alpha * A["aBcI"] * B["jIaB"];

    C["kb"] += -0.5 * alpha * A["akij"] * B["ijab"];
    C["kb"] += -1.0 * alpha * A["kAiJ"] * B["iJbA"];

    C["cb"] += -1.0 * alpha * A["acij"] * B["ijab"];
    C["cb"] += -2.0 * alpha * A["cAiJ"] * B["iJbA"];

    C["JK"] += +1.0 * alpha * A["ABIK"] * B["IJAB"];
    C["JK"] += +2.0 * alpha * A["aBiK"] * B["iJaB"];

    C["JC"] += +0.5 * alpha * A["ABIC"] * B["IJAB"];
    C["JC"] += +1.0 * alpha * A["aBiC"] * B["iJaB"];

    C["KB"] += -0.5 * alpha * A["AKIJ"] * B["IJAB"];
    C["KB"] += -1.0 * alpha * A["aKiJ"] * B["iJaB"];

    C["CB"] += -1.0 * alpha * A["ACIJ"] * B["IJAB"];
    C["CB"] += -2.0 * alpha * A["aCiJ"] * B["iJaB"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C1 : %.4f", t.get());
    }
    time_comm_A2_B2_C1 += t.get();
}

void TensorSRG::commutator_A2_B2_C2(BlockedTensor& A, BlockedTensor& B, double alpha,
                                    BlockedTensor& C) {
    Timer t;

    // AAAA case (these work only in the single-reference case)
    // Term I
    C["ijpq"] += 0.5 * alpha * A["abpq"] * B["ijab"];
    C["pqab"] += 0.5 * alpha * A["pqij"] * B["ijab"];

    //    I_ioiv["rspq"]  = alpha * A["rupv"] * B["svqu"];
    //    I_ioiv["rspq"] += alpha * A["rUpV"] * B["sVqU"];
    //    C["rspq"] += I_ioiv["rspq"];
    //    C["rspq"] -= I_ioiv["rsqp"];
    //    C["rspq"] -= I_ioiv["srpq"];
    //    C["rspq"] += I_ioiv["srqp"];

    I_ioiv["rjpb"] = alpha * A["rapi"] * B["jiba"];
    I_ioiv["rjpb"] += alpha * A["rApI"] * B["jIbA"];

    C["rspq"] += I_ioiv["rspq"];
    C["rspq"] -= I_ioiv["rsqp"];
    C["rspq"] -= I_ioiv["srpq"];
    C["rspq"] += I_ioiv["srqp"];

    // ABAB case (these work only in the single-reference case)
    // Term I
    C["iJpQ"] += alpha * A["aBpQ"] * B["iJaB"];
    C["pQaB"] += alpha * A["pQiJ"] * B["iJaB"];

    // Term II
    C["rSpQ"] += alpha * A["rupv"] * B["vSuQ"];
    C["rSpQ"] += alpha * A["rUpV"] * B["VSUQ"];
    C["rSpQ"] += -alpha * A["rUvQ"] * B["vSpU"];
    C["rSpQ"] += -alpha * A["uSpV"] * B["rVuQ"];
    C["rSpQ"] += alpha * A["uSvQ"] * B["rvpu"];
    C["rSpQ"] += alpha * A["USVQ"] * B["rVpU"];

    // BBBB case (these work only in the single-reference case)
    // Term I
    C["IJPQ"] += 0.5 * alpha * A["ABPQ"] * B["IJAB"];
    C["PQAB"] += 0.5 * alpha * A["PQIJ"] * B["IJAB"];

    I_ioiv["RJPB"] = alpha * A["RAPI"] * B["JIBA"];
    I_ioiv["RJPB"] += alpha * A["aRiP"] * B["iJaB"];
    //    I_ioiv["RSPQ"]  = alpha * A["uRvP"] * B["vSuQ"];
    //    I_ioiv["RSPQ"] += alpha * A["RUPV"] * B["SVQU"];
    C["RSPQ"] += I_ioiv["RSPQ"];
    C["RSPQ"] -= I_ioiv["RSQP"];
    C["RSPQ"] -= I_ioiv["SRPQ"];
    C["RSPQ"] += I_ioiv["SRQP"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C2 : %.4f", t.get());
    }
    time_comm_A2_B2_C2 += t.get();
}

void TensorSRG::modified_commutator_A_B_C(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                          BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                          BlockedTensor& C1, BlockedTensor& C2) {
    // => Compute C = [A,B]_12 <= //

    //    commutator_A1_B1_C0(A1,B1,+factor,C0);
    //    commutator_A1_B2_C0(A1,B2,+factor,C0);
    //    commutator_A1_B2_C0(B1,A2,-factor,C0);
    commutator_A2_B2_C0(A2, B2, +factor, C0);

    //    commutator_A1_B1_C1(A1,B1,+factor,C1);
    //    commutator_A1_B2_C1(A1,B2,+factor,C1);
    //    commutator_A1_B2_C1(B1,A2,+factor,C1);
    //    commutator_A2_B2_C1(A2,B2,+factor,C1);

    //    commutator_A1_B2_C2(A1,B2,+factor,C2);
    //    commutator_A1_B2_C2(B1,A2,-factor,C2);
    commutator_A2_B2_C2(A2, B2, +factor, C2);

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

void TensorSRG::modified_commutator_A2_B2_C0(BlockedTensor& A, BlockedTensor& B, double alpha,
                                             double& C) {
    Timer t;

    C += alpha * 0.25 * A["abij"] * B["ijab"];
    C += alpha * 1.00 * A["aBiJ"] * B["iJaB"];
    C += alpha * 0.25 * A["ABIJ"] * B["IJAB"];

    if (print_ > 2) {
        outfile->Printf("\n  Time for [A2,B2] -> C0 : %.4f", t.get());
    }
    time_comm_A2_B2_C0 += t.get();
}

void TensorSRG::modified_commutator_A2_B2_C2(BlockedTensor& A, BlockedTensor& B, double alpha,
                                             BlockedTensor& C) {
    Timer t;

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

    // Term II
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
        outfile->Printf("\n  Time for [A2,B2] -> C2 : %.4f", t.get());
    }
    time_comm_A2_B2_C2 += t.get();
}

void TensorSRG::print_timings() {
    outfile->Printf("\n\n              ============== TIMINGS ============");
    outfile->Printf("\n              Time for [A1,B1] -> C0 : %10.3f", time_comm_A1_B1_C0);
    outfile->Printf("\n              Time for [A1,B1] -> C1 : %10.3f", time_comm_A1_B1_C1);
    outfile->Printf("\n              Time for [A1,B2] -> C0 : %10.3f", time_comm_A1_B2_C0);
    outfile->Printf("\n              Time for [A1,B2] -> C1 : %10.3f", time_comm_A1_B2_C1);
    outfile->Printf("\n              Time for [A1,B2] -> C2 : %10.3f", time_comm_A1_B2_C2);
    outfile->Printf("\n              Time for [A2,B2] -> C0 : %10.3f", time_comm_A2_B2_C0);
    outfile->Printf("\n              Time for [A2,B2] -> C1 : %10.3f", time_comm_A2_B2_C1);
    outfile->Printf("\n              Time for [A2,B2] -> C2 : %10.3f", time_comm_A2_B2_C2);
    outfile->Printf("\n              ===================================\n");
    outfile->Printf("\n              The commutator was called %d times\n", ncalls);
}
}
} // EndNamespaces

// void TensorSRG::commutator_A2_B2_C1(BlockedTensor& A,BlockedTensor& B,double
// alpha,BlockedTensor C)
//{
//    Timer t;
//    if(use_tensor_class_){
//        loop_mo_p loop_mo_q{
//            D_a(p,q) = (p == q) ? No_.a[p] : 0.0;
//            D_b(p,q) = (p == q) ? No_.b[p] : 0.0;
//            CD_a(p,q) = (p == q) ? 1.0 - No_.a[p] : 0.0;
//            CD_b(p,q) = (p == q) ? 1.0 - No_.b[p] : 0.0;
//        }

//        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
//            A4_aa(p,q,r,s) = A.aaaa[p][q][r][s];
//            B4_aa(p,q,r,s) = B.aaaa[p][q][r][s];
//            A4_ab(p,q,r,s) = A.abab[p][q][r][s];
//            B4_ab(p,q,r,s) = B.abab[p][q][r][s];
//            A4_bb(p,q,r,s) = A.bbbb[p][q][r][s];
//            B4_bb(p,q,r,s) = B.bbbb[p][q][r][s];
//        }

//        C_a.zero();
//        C_b.zero();

//        A4m_aa("cpdb") = A4_aa("cpab") * D_a("ad");
//        A4m_aa("cpde") = A4m_aa("cpdb") * D_a("be");
//        A4m_aa("fpde") = A4m_aa("cpde") * CD_a("fc");
//        C_a("pq") += 0.5 * sign * A4m_aa("fpde") * B4_aa("defq");

//        B4m_aa("cpdb") = B4_aa("cpab") * D_a("ad");
//        B4m_aa("cpde") = B4m_aa("cpdb") * D_a("be");
//        B4m_aa("fpde") = B4m_aa("cpde") * CD_a("fc");
//        C_a("pq") += -0.5 * sign * B4m_aa("fpde") * A4_aa("defq");

//        A4m_aa("cpdb") = A4_aa("cpab") * CD_a("ad");
//        A4m_aa("cpde") = A4m_aa("cpdb") * CD_a("be");
//        A4m_aa("fpde") = A4m_aa("cpde") * D_a("fc");
//        C_a("pq") += 0.5 * sign * A4m_aa("fpde") * B4_aa("defq");

//        B4m_aa("cpdb") = B4_aa("cpab") * CD_a("ad");
//        B4m_aa("cpde") = B4m_aa("cpdb") * CD_a("be");
//        B4m_aa("fpde") = B4m_aa("cpde") * D_a("fc");
//        C_a("pq") += -0.5 * sign * B4m_aa("fpde") * A4_aa("defq");

//        A4m_ab("pCbD") = A4_ab("pCbA") * D_b("AD");
//        A4m_ab("pCeD") = A4m_ab("pCbD") * D_a("be");
//        A4m_ab("pFeD") = A4m_ab("pCeD") * CD_b("FC");
//        C_a("pq") += sign * A4m_ab("pFeD") * B4_ab("eDqF");

//        B4m_ab("pCbD") = B4_ab("pCbA") * D_b("AD");
//        B4m_ab("pCeD") = B4m_ab("pCbD") * D_a("be");
//        B4m_ab("pFeD") = B4m_ab("pCeD") * CD_b("FC");
//        C_a("pq") += -sign * B4m_ab("pFeD") * A4_ab("eDqF");

//        A4m_ab("pCbD") = A4_ab("pCbA") * CD_b("AD");
//        A4m_ab("pCeD") = A4m_ab("pCbD") * CD_a("be");
//        A4m_ab("pFeD") = A4m_ab("pCeD") * D_b("FC");
//        C_a("pq") += sign * A4m_ab("pFeD") * B4_ab("eDqF");

//        B4m_ab("pCbD") = B4_ab("pCbA") * CD_b("AD");
//        B4m_ab("pCeD") = B4m_ab("pCbD") * CD_a("be");
//        B4m_ab("pFeD") = B4m_ab("pCeD") * D_b("FC");
//        C_a("pq") += -sign * B4m_ab("pFeD") * A4_ab("eDqF");

//        A4m_bb("cpdb") = A4_bb("cpab") * D_b("ad");
//        A4m_bb("cpde") = A4m_bb("cpdb") * D_b("be");
//        A4m_bb("fpde") = A4m_bb("cpde") * CD_b("fc");
//        C_b("pq") += 0.5 * sign * A4m_bb("fpde") * B4_bb("defq");

//        B4m_bb("cpdb") = B4_bb("cpab") * D_b("ad");
//        B4m_bb("cpde") = B4m_bb("cpdb") * D_b("be");
//        B4m_bb("fpde") = B4m_bb("cpde") * CD_b("fc");
//        C_b("pq") += -0.5 * sign * B4m_bb("fpde") * A4_bb("defq");

//        A4m_bb("cpdb") = A4_bb("cpab") * CD_b("ad");
//        A4m_bb("cpde") = A4m_bb("cpdb") * CD_b("be");
//        A4m_bb("fpde") = A4m_bb("cpde") * D_b("fc");
//        C_b("pq") += 0.5 * sign * A4m_bb("fpde") * B4_bb("defq");

//        B4m_bb("cpdb") = B4_bb("cpab") * CD_b("ad");
//        B4m_bb("cpde") = B4m_bb("cpdb") * CD_b("be");
//        B4m_bb("fpde") = B4m_bb("cpde") * D_b("fc");
//        C_b("pq") += -0.5 * sign * B4m_bb("fpde") * A4_bb("defq");

//        A4m_ab("cPdB") = A4_ab("cPaB") * D_a("ad");
//        A4m_ab("cPdE") = A4m_ab("cPdB") * D_b("BE");
//        A4m_ab("fPdE") = A4m_ab("cPdE") * CD_a("fc");
//        C_b("PQ") += sign * A4m_ab("fPdE") * B4_ab("dEfQ");

//        B4m_ab("cPdB") = B4_ab("cPaB") * D_a("ad");
//        B4m_ab("cPdE") = B4m_ab("cPdB") * D_b("BE");
//        B4m_ab("fPdE") = B4m_ab("cPdE") * CD_a("fc");
//        C_b("PQ") += -sign * B4m_ab("fPdE") * A4_ab("dEfQ");

//        A4m_ab("cPdB") = A4_ab("cPaB") * CD_a("ad");
//        A4m_ab("cPdE") = A4m_ab("cPdB") * CD_b("BE");
//        A4m_ab("fPdE") = A4m_ab("cPdE") * D_a("fc");
//        C_b("PQ") += sign * A4m_ab("fPdE") * B4_ab("dEfQ");

//        B4m_ab("cPdB") = B4_ab("cPaB") * CD_a("ad");
//        B4m_ab("cPdE") = B4m_ab("cPdB") * CD_b("BE");
//        B4m_ab("fPdE") = B4m_ab("cPdE") * D_a("fc");
//        C_b("PQ") += -sign * B4m_ab("fPdE") * A4_ab("dEfQ");

//        loop_mo_p loop_mo_q{
//            C.aa[p][q] += C_a(p,q);
//            C.bb[p][q] += C_b(p,q);
//        }
//    }else{
//        loop_mo_p loop_mo_q{
//            double sum = 0.0;
//            loop_mo_r loop_mo_s loop_mo_t{
//                sum += 0.5 * (A.aaaa[t][p][r][s] * B.aaaa[r][s][t][q] -
//                B.aaaa[t][p][r][s] * A.aaaa[r][s][t][q])
//                        * (No_.a[r] * No_.a[s] * Nv_.a[t] + Nv_.a[r] *
//                        Nv_.a[s] * No_.a[t]);
//                sum += (A.abab[p][t][r][s] * B.abab[r][s][q][t] -
//                B.abab[p][t][r][s] * A.abab[r][s][q][t])
//                        * (No_.a[r] * No_.b[s] * Nv_.b[t] + Nv_.a[r] *
//                        Nv_.b[s] * No_.b[t]);
//            }
//            C.aa[p][q] += sign * sum;
//        }
//        loop_mo_p loop_mo_q{
//            double sum = 0.0;
//            loop_mo_r loop_mo_s loop_mo_t{
//                sum += 0.5 * (A.bbbb[t][p][r][s] * B.bbbb[r][s][t][q] -
//                B.bbbb[t][p][r][s] * A.bbbb[r][s][t][q])
//                        * (No_.b[r] * No_.b[s] * Nv_.b[t] + Nv_.b[r] *
//                        Nv_.b[s] * No_.b[t]);
//                sum += (A.abab[t][p][r][s] * B.abab[r][s][t][q] -
//                B.abab[t][p][r][s] * A.abab[r][s][t][q])
//                        * (No_.a[r] * No_.b[s] * Nv_.a[t] + Nv_.a[r] *
//                        Nv_.b[s] * No_.a[t]);
//            }
//            C.bb[p][q] += sign * sum;
//        }
//    }
//    if(print_ > 2){
//        outfile->Printf("\n  Time for [A2,B2] -> C1 : %.4f",t.get());
//    }
//    time_comm_A2_B2_C1 += t.get();
//}
