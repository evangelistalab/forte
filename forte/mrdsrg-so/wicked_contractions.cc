#include <algorithm>
#include "psi4/libpsi4util/PsiOutStream.h"
#include "helpers/timer.h"
#include "mrdsrg_so.h"

using namespace psi;

namespace forte {

void MRDSRG_SO::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    // 3 lines
    local_timer timer;
    C0 += alpha * +1.00000000 * H1["u,m"] * T1["m,v"] * Eta1["v,u"];
    C0 += alpha * +1.00000000 * H1["e,m"] * T1["m,e"];
    C0 += alpha * +1.00000000 * H1["e,u"] * T1["v,e"] * Gamma1["u,v"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H1_T1_C0 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    // 2 lines
    local_timer timer;
    C0 += alpha * -0.50000000 * H1["u,m"] * T2["m,v,w,x"] * Lambda2["w,x,u,v"];
    C0 += alpha * -0.50000000 * H1["e,u"] * T2["v,w,x,e"] * Lambda2["u,x,v,w"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H1_T2_C0 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    // 2 lines
    local_timer timer;
    C0 += alpha * +0.50000000 * H2["u,v,m,w"] * T1["m,x"] * Lambda2["w,x,u,v"];
    C0 += alpha * +0.50000000 * H2["u,e,v,w"] * T1["x,e"] * Lambda2["v,w,u,x"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H2_T1_C0 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
    // 17 lines
    local_timer timer;
    C0 += alpha * +0.25000000 * H2["u,v,m,n"] * T2["m,n,w,x"] * Eta1["x,v"] * Eta1["w,u"];
    C0 += alpha * +0.12500000 * H2["u,v,m,n"] * T2["m,n,w,x"] * Lambda2["w,x,u,v"];
    C0 += alpha * +0.50000000 * H2["u,e,m,n"] * T2["m,n,v,e"] * Eta1["v,u"];
    C0 += alpha * +0.25000000 * H2["e,f,m,n"] * T2["m,n,e,f"];
    C0 += alpha * +0.50000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Eta1["z,v"] * Eta1["y,u"] *
          Gamma1["w,x"];
    C0 += alpha * +1.00000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Eta1["z,v"] * Lambda2["w,y,u,x"];
    C0 += alpha * +0.25000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Gamma1["w,x"] * Lambda2["y,z,u,v"];
    C0 += alpha * +0.25000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Lambda3["w,y,z,u,v,x"];
    C0 += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,x,e"] * Eta1["x,u"] * Gamma1["v,w"];
    C0 += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,x,e"] * Lambda2["v,x,u,w"];
    C0 += alpha * +0.50000000 * H2["e,f,m,u"] * T2["m,v,e,f"] * Gamma1["u,v"];
    C0 += alpha * +0.50000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Eta1["z,u"] * Gamma1["w,y"] *
          Gamma1["v,x"];
    C0 += alpha * +0.25000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Eta1["z,u"] * Lambda2["v,w,x,y"];
    C0 += alpha * +1.00000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Gamma1["w,y"] * Lambda2["v,z,u,x"];
    C0 += alpha * -0.25000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Lambda3["v,w,z,u,x,y"];
    C0 += alpha * +0.25000000 * H2["e,f,u,v"] * T2["w,x,e,f"] * Gamma1["v,x"] * Gamma1["u,w"];
    C0 += alpha * +0.12500000 * H2["e,f,u,v"] * T2["w,x,e,f"] * Lambda2["u,v,w,x"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H2_T2_C0 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C1) {
    // 24 lines
    local_timer timer;
    C1["u,v"] += alpha * -1.00000000 * H1["u,m"] * T1["m,v"];
    C1["v,u"] += alpha * +1.00000000 * H1["e,u"] * T1["v,e"];
    C1["u,m"] += alpha * +1.00000000 * H1["e,m"] * T1["u,e"];
    C1["u,e"] += alpha * -1.00000000 * H1["u,m"] * T1["m,e"];
    C1["u,e"] += alpha * -1.00000000 * H1["u,v"] * T1["w,e"] * Eta1["v,w"];
    C1["u,e"] += alpha * -1.00000000 * H1["u,v"] * T1["w,e"] * Gamma1["v,w"];
    C1["u,f"] += alpha * +1.00000000 * H1["e,f"] * T1["u,e"];
    C1["m,u"] += alpha * -1.00000000 * H1["m,n"] * T1["n,u"];
    C1["m,v"] += alpha * +1.00000000 * H1["u,v"] * T1["m,w"] * Eta1["w,u"];
    C1["m,v"] += alpha * +1.00000000 * H1["u,v"] * T1["m,w"] * Gamma1["w,u"];
    C1["m,u"] += alpha * +1.00000000 * H1["e,u"] * T1["m,e"];
    C1["n,m"] += alpha * +1.00000000 * H1["u,m"] * T1["n,v"] * Eta1["v,u"];
    C1["n,m"] += alpha * +1.00000000 * H1["u,m"] * T1["n,v"] * Gamma1["v,u"];
    C1["n,m"] += alpha * +1.00000000 * H1["e,m"] * T1["n,e"];
    C1["m,e"] += alpha * -1.00000000 * H1["m,n"] * T1["n,e"];
    C1["m,e"] += alpha * -1.00000000 * H1["m,u"] * T1["v,e"] * Eta1["u,v"];
    C1["m,e"] += alpha * -1.00000000 * H1["m,u"] * T1["v,e"] * Gamma1["u,v"];
    C1["m,e"] += alpha * +1.00000000 * H1["u,e"] * T1["m,v"] * Eta1["v,u"];
    C1["m,e"] += alpha * +1.00000000 * H1["u,e"] * T1["m,v"] * Gamma1["v,u"];
    C1["m,f"] += alpha * +1.00000000 * H1["e,f"] * T1["m,e"];
    C1["e,u"] += alpha * -1.00000000 * H1["e,m"] * T1["m,u"];
    C1["e,f"] += alpha * -1.00000000 * H1["e,m"] * T1["m,f"];
    C1["e,f"] += alpha * -1.00000000 * H1["e,u"] * T1["v,f"] * Eta1["u,v"];
    C1["e,f"] += alpha * -1.00000000 * H1["e,u"] * T1["v,f"] * Gamma1["u,v"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H1_T1_C1 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    // 18 lines
    local_timer timer;
    C1["v,w"] += alpha * -1.00000000 * H1["u,m"] * T2["m,v,w,x"] * Eta1["x,u"];
    C1["u,v"] += alpha * -1.00000000 * H1["e,m"] * T2["m,u,v,e"];
    C1["v,x"] += alpha * +1.00000000 * H1["e,u"] * T2["v,w,x,e"] * Gamma1["u,w"];
    C1["v,e"] += alpha * +1.00000000 * H1["u,m"] * T2["m,v,w,e"] * Eta1["w,u"];
    C1["u,f"] += alpha * -1.00000000 * H1["e,m"] * T2["m,u,f,e"];
    C1["w,e"] += alpha * -1.00000000 * H1["u,v"] * T2["w,x,y,e"] * Eta1["y,u"] * Gamma1["v,x"];
    C1["w,e"] += alpha * +1.00000000 * H1["u,v"] * T2["w,x,y,e"] * Eta1["v,x"] * Gamma1["y,u"];
    C1["v,f"] += alpha * +1.00000000 * H1["e,u"] * T2["v,w,f,e"] * Gamma1["u,w"];
    C1["n,v"] += alpha * +1.00000000 * H1["u,m"] * T2["n,m,v,w"] * Eta1["w,u"];
    C1["n,u"] += alpha * +1.00000000 * H1["e,m"] * T2["n,m,u,e"];
    C1["m,x"] += alpha * +1.00000000 * H1["u,v"] * T2["m,w,x,y"] * Eta1["y,u"] * Gamma1["v,w"];
    C1["m,x"] += alpha * -1.00000000 * H1["u,v"] * T2["m,w,x,y"] * Eta1["v,w"] * Gamma1["y,u"];
    C1["m,w"] += alpha * +1.00000000 * H1["e,u"] * T2["m,v,w,e"] * Gamma1["u,v"];
    C1["n,e"] += alpha * -1.00000000 * H1["u,m"] * T2["n,m,v,e"] * Eta1["v,u"];
    C1["n,f"] += alpha * +1.00000000 * H1["e,m"] * T2["n,m,f,e"];
    C1["m,e"] += alpha * -1.00000000 * H1["u,v"] * T2["m,w,x,e"] * Eta1["x,u"] * Gamma1["v,w"];
    C1["m,e"] += alpha * +1.00000000 * H1["u,v"] * T2["m,w,x,e"] * Eta1["v,w"] * Gamma1["x,u"];
    C1["m,f"] += alpha * +1.00000000 * H1["e,u"] * T2["m,v,f,e"] * Gamma1["u,v"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H1_T2_C1 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C1) {
    // 27 lines
    local_timer timer;
    C1["u,w"] += alpha * -1.00000000 * H2["u,v,m,w"] * T1["m,x"] * Eta1["x,v"];
    C1["u,v"] += alpha * -1.00000000 * H2["u,e,m,v"] * T1["m,e"];
    C1["u,v"] += alpha * +1.00000000 * H2["u,e,v,w"] * T1["x,e"] * Gamma1["w,x"];
    C1["u,m"] += alpha * +1.00000000 * H2["u,v,m,n"] * T1["n,w"] * Eta1["w,v"];
    C1["u,m"] += alpha * +1.00000000 * H2["u,e,m,n"] * T1["n,e"];
    C1["u,m"] += alpha * +1.00000000 * H2["u,e,m,v"] * T1["w,e"] * Gamma1["v,w"];
    C1["u,e"] += alpha * -1.00000000 * H2["u,v,m,e"] * T1["m,w"] * Eta1["w,v"];
    C1["u,f"] += alpha * -1.00000000 * H2["u,e,m,f"] * T1["m,e"];
    C1["u,f"] += alpha * -1.00000000 * H2["u,e,v,f"] * T1["w,e"] * Gamma1["v,w"];
    C1["m,v"] += alpha * -1.00000000 * H2["m,u,n,v"] * T1["n,w"] * Eta1["w,u"];
    C1["m,u"] += alpha * -1.00000000 * H2["m,e,n,u"] * T1["n,e"];
    C1["m,u"] += alpha * +1.00000000 * H2["m,e,u,v"] * T1["w,e"] * Gamma1["v,w"];
    C1["m,n"] += alpha * +1.00000000 * H2["m,u,n,c0"] * T1["c0,v"] * Eta1["v,u"];
    C1["m,n"] += alpha * +1.00000000 * H2["m,e,n,c0"] * T1["c0,e"];
    C1["m,n"] += alpha * +1.00000000 * H2["m,e,n,u"] * T1["v,e"] * Gamma1["u,v"];
    C1["m,e"] += alpha * -1.00000000 * H2["m,u,n,e"] * T1["n,v"] * Eta1["v,u"];
    C1["m,f"] += alpha * -1.00000000 * H2["m,e,n,f"] * T1["n,e"];
    C1["m,f"] += alpha * -1.00000000 * H2["m,e,u,f"] * T1["v,e"] * Gamma1["u,v"];
    C1["e,v"] += alpha * +1.00000000 * H2["u,e,m,v"] * T1["m,w"] * Eta1["w,u"];
    C1["e,u"] += alpha * -1.00000000 * H2["e,f,m,u"] * T1["m,f"];
    C1["e,u"] += alpha * +1.00000000 * H2["e,f,u,v"] * T1["w,f"] * Gamma1["v,w"];
    C1["e,m"] += alpha * -1.00000000 * H2["u,e,m,n"] * T1["n,v"] * Eta1["v,u"];
    C1["e,m"] += alpha * +1.00000000 * H2["e,f,m,n"] * T1["n,f"];
    C1["e,m"] += alpha * +1.00000000 * H2["e,f,m,u"] * T1["v,f"] * Gamma1["u,v"];
    C1["e,f"] += alpha * +1.00000000 * H2["u,e,m,f"] * T1["m,v"] * Eta1["v,u"];
    C1["e,g"] += alpha * -1.00000000 * H2["e,f,m,g"] * T1["m,f"];
    C1["e,g"] += alpha * -1.00000000 * H2["e,f,u,g"] * T1["v,f"] * Gamma1["u,v"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H2_T1_C1 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    // 158 lines
    local_timer timer;
    C1["u,w"] += alpha * -0.50000000 * H2["u,v,m,n"] * T2["m,n,w,x"] * Eta1["x,v"];
    C1["u,v"] += alpha * -0.50000000 * H2["u,e,m,n"] * T2["m,n,v,e"];
    C1["u,w"] += alpha * +0.50000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Lambda2["y,z,v,x"];
    C1["x,w"] += alpha * +0.50000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Eta1["z,v"] * Eta1["y,u"];
    C1["x,w"] += alpha * +0.25000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Lambda2["y,z,u,v"];
    C1["w,v"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,x,e"] * Eta1["x,u"];
    C1["v,u"] += alpha * +0.50000000 * H2["e,f,m,u"] * T2["m,v,e,f"];
    C1["u,y"] += alpha * -1.00000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Eta1["z,v"] * Gamma1["w,x"];
    C1["u,y"] += alpha * -1.00000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Lambda2["w,z,v,x"];
    C1["u,x"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["m,w,x,e"] * Gamma1["v,w"];
    C1["x,y"] += alpha * -0.50000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Lambda2["w,z,u,v"];
    C1["u,v"] += alpha * -0.50000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Lambda2["w,z,x,y"];
    C1["x,v"] += alpha * +1.00000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Eta1["z,u"] * Gamma1["w,y"];
    C1["x,v"] += alpha * +1.00000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Lambda2["w,z,u,y"];
    C1["w,u"] += alpha * +0.50000000 * H2["e,f,u,v"] * T2["w,x,e,f"] * Gamma1["v,x"];
    C1["u,z"] +=
        alpha * -0.50000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Gamma1["w,y"] * Gamma1["v,x"];
    C1["u,z"] += alpha * -0.25000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Lambda2["v,w,x,y"];
    C1["x,z"] += alpha * +0.50000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Lambda2["v,w,u,y"];
    C1["u,m"] += alpha * -0.50000000 * H2["u,v,m,n"] * T2["n,w,x,y"] * Lambda2["x,y,v,w"];
    C1["w,m"] += alpha * -0.50000000 * H2["u,v,m,n"] * T2["n,w,x,y"] * Eta1["y,v"] * Eta1["x,u"];
    C1["w,m"] += alpha * -0.25000000 * H2["u,v,m,n"] * T2["n,w,x,y"] * Lambda2["x,y,u,v"];
    C1["v,m"] += alpha * -1.00000000 * H2["u,e,m,n"] * T2["n,v,w,e"] * Eta1["w,u"];
    C1["u,m"] += alpha * -0.50000000 * H2["e,f,m,n"] * T2["n,u,e,f"];
    C1["u,m"] += alpha * -0.50000000 * H2["u,e,m,v"] * T2["w,x,y,e"] * Lambda2["v,y,w,x"];
    C1["w,m"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["w,x,y,e"] * Eta1["y,u"] * Gamma1["v,x"];
    C1["w,m"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["w,x,y,e"] * Lambda2["v,y,u,x"];
    C1["v,m"] += alpha * +0.50000000 * H2["e,f,m,u"] * T2["v,w,e,f"] * Gamma1["u,w"];
    C1["u,e"] += alpha * +0.50000000 * H2["u,v,m,n"] * T2["m,n,w,e"] * Eta1["w,v"];
    C1["u,f"] += alpha * -0.50000000 * H2["u,e,m,n"] * T2["m,n,f,e"];
    C1["u,e"] += alpha * +1.00000000 * H2["u,v,m,w"] * T2["m,x,y,e"] * Eta1["y,v"] * Gamma1["w,x"];
    C1["u,e"] += alpha * +1.00000000 * H2["u,v,m,w"] * T2["m,x,y,e"] * Lambda2["w,y,v,x"];
    C1["u,f"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["m,w,f,e"] * Gamma1["v,w"];
    C1["x,e"] += alpha * +0.50000000 * H2["u,v,m,w"] * T2["m,x,y,e"] * Lambda2["w,y,u,v"];
    C1["u,e"] += alpha * +0.50000000 * H2["u,v,m,e"] * T2["m,w,x,y"] * Lambda2["x,y,v,w"];
    C1["w,e"] += alpha * +0.50000000 * H2["u,v,m,e"] * T2["m,w,x,y"] * Eta1["y,v"] * Eta1["x,u"];
    C1["w,e"] += alpha * +0.25000000 * H2["u,v,m,e"] * T2["m,w,x,y"] * Lambda2["x,y,u,v"];
    C1["v,f"] += alpha * +1.00000000 * H2["u,e,m,f"] * T2["m,v,w,e"] * Eta1["w,u"];
    C1["u,g"] += alpha * +0.50000000 * H2["e,f,m,g"] * T2["m,u,e,f"];
    C1["u,e"] += alpha * +0.50000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["a0,v"] *
                 Gamma1["x,z"] * Gamma1["w,y"];
    C1["u,e"] +=
        alpha * +0.25000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["a0,v"] * Lambda2["w,x,y,z"];
    C1["u,e"] += alpha * +0.50000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["x,z"] * Eta1["w,y"] *
                 Gamma1["a0,v"];
    C1["u,e"] +=
        alpha * +1.00000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["x,z"] * Lambda2["w,a0,v,y"];
    C1["u,e"] +=
        alpha * +0.25000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Gamma1["a0,v"] * Lambda2["w,x,y,z"];
    C1["u,e"] +=
        alpha * +1.00000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Gamma1["x,z"] * Lambda2["w,a0,v,y"];
    C1["u,f"] +=
        alpha * -0.50000000 * H2["u,e,v,w"] * T2["x,y,f,e"] * Gamma1["w,y"] * Gamma1["v,x"];
    C1["u,f"] += alpha * -0.25000000 * H2["u,e,v,w"] * T2["x,y,f,e"] * Lambda2["v,w,x,y"];
    C1["y,e"] +=
        alpha * -0.50000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["a0,v"] * Lambda2["w,x,u,z"];
    C1["y,e"] +=
        alpha * +0.50000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["x,z"] * Lambda2["w,a0,u,v"];
    C1["y,e"] +=
        alpha * +0.50000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Gamma1["a0,u"] * Lambda2["w,x,v,z"];
    C1["y,e"] +=
        alpha * +0.50000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Gamma1["x,z"] * Lambda2["w,a0,u,v"];
    C1["x,f"] += alpha * +0.50000000 * H2["u,e,v,w"] * T2["x,y,f,e"] * Lambda2["v,w,u,y"];
    C1["u,f"] += alpha * +0.50000000 * H2["u,e,v,f"] * T2["w,x,y,e"] * Lambda2["v,y,w,x"];
    C1["w,f"] += alpha * -1.00000000 * H2["u,e,v,f"] * T2["w,x,y,e"] * Eta1["y,u"] * Gamma1["v,x"];
    C1["w,f"] += alpha * -1.00000000 * H2["u,e,v,f"] * T2["w,x,y,e"] * Lambda2["v,y,u,x"];
    C1["v,g"] += alpha * -0.50000000 * H2["e,f,u,g"] * T2["v,w,e,f"] * Gamma1["u,w"];
    C1["m,v"] += alpha * -0.50000000 * H2["m,u,n,c0"] * T2["n,c0,v,w"] * Eta1["w,u"];
    C1["m,u"] += alpha * -0.50000000 * H2["m,e,n,c0"] * T2["n,c0,u,e"];
    C1["m,v"] += alpha * +0.50000000 * H2["m,u,n,v"] * T2["n,w,x,y"] * Lambda2["x,y,u,w"];
    C1["n,w"] += alpha * -0.50000000 * H2["u,v,m,w"] * T2["n,m,x,y"] * Eta1["y,v"] * Eta1["x,u"];
    C1["n,w"] += alpha * -0.25000000 * H2["u,v,m,w"] * T2["n,m,x,y"] * Lambda2["x,y,u,v"];
    C1["n,v"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["n,m,w,e"] * Eta1["w,u"];
    C1["n,u"] += alpha * -0.50000000 * H2["e,f,m,u"] * T2["n,m,e,f"];
    C1["m,x"] += alpha * -1.00000000 * H2["m,u,n,v"] * T2["n,w,x,y"] * Eta1["y,u"] * Gamma1["v,w"];
    C1["m,x"] += alpha * -1.00000000 * H2["m,u,n,v"] * T2["n,w,x,y"] * Lambda2["v,y,u,w"];
    C1["m,w"] += alpha * -1.00000000 * H2["m,e,n,u"] * T2["n,v,w,e"] * Gamma1["u,v"];
    C1["n,x"] += alpha * +0.50000000 * H2["u,v,m,w"] * T2["n,m,x,y"] * Lambda2["w,y,u,v"];
    C1["m,u"] += alpha * -0.50000000 * H2["m,e,u,v"] * T2["w,x,y,e"] * Lambda2["v,y,w,x"];
    C1["m,w"] += alpha * +0.50000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["a0,v"] * Eta1["z,u"] *
                 Gamma1["x,y"];
    C1["m,w"] +=
        alpha * +1.00000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["a0,v"] * Lambda2["x,z,u,y"];
    C1["m,w"] += alpha * +0.50000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["x,y"] *
                 Gamma1["a0,v"] * Gamma1["z,u"];
    C1["m,w"] +=
        alpha * +0.25000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["x,y"] * Lambda2["z,a0,u,v"];
    C1["m,w"] +=
        alpha * +1.00000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Gamma1["a0,v"] * Lambda2["x,z,u,y"];
    C1["m,w"] +=
        alpha * +0.25000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Gamma1["x,y"] * Lambda2["z,a0,u,v"];
    C1["m,v"] += alpha * +1.00000000 * H2["u,e,v,w"] * T2["m,x,y,e"] * Eta1["y,u"] * Gamma1["w,x"];
    C1["m,v"] += alpha * +1.00000000 * H2["u,e,v,w"] * T2["m,x,y,e"] * Lambda2["w,y,u,x"];
    C1["m,u"] += alpha * +0.50000000 * H2["e,f,u,v"] * T2["m,w,e,f"] * Gamma1["v,w"];
    C1["m,y"] +=
        alpha * -0.50000000 * H2["m,e,u,v"] * T2["w,x,y,e"] * Gamma1["v,x"] * Gamma1["u,w"];
    C1["m,y"] += alpha * -0.25000000 * H2["m,e,u,v"] * T2["w,x,y,e"] * Lambda2["u,v,w,x"];
    C1["m,z"] +=
        alpha * +0.50000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["a0,v"] * Lambda2["w,x,u,y"];
    C1["m,z"] +=
        alpha * -0.50000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["x,y"] * Lambda2["w,a0,u,v"];
    C1["m,z"] +=
        alpha * -0.50000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Gamma1["a0,u"] * Lambda2["w,x,v,y"];
    C1["m,z"] +=
        alpha * -0.50000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Gamma1["x,y"] * Lambda2["w,a0,u,v"];
    C1["m,y"] += alpha * +0.50000000 * H2["u,e,v,w"] * T2["m,x,y,e"] * Lambda2["v,w,u,x"];
    C1["m,n"] += alpha * -0.50000000 * H2["m,u,n,c0"] * T2["c0,v,w,x"] * Lambda2["w,x,u,v"];
    C1["c0,m"] += alpha * +0.50000000 * H2["u,v,m,n"] * T2["c0,n,w,x"] * Eta1["x,v"] * Eta1["w,u"];
    C1["c0,m"] += alpha * +0.25000000 * H2["u,v,m,n"] * T2["c0,n,w,x"] * Lambda2["w,x,u,v"];
    C1["c0,m"] += alpha * +1.00000000 * H2["u,e,m,n"] * T2["c0,n,v,e"] * Eta1["v,u"];
    C1["c0,m"] += alpha * +0.50000000 * H2["e,f,m,n"] * T2["c0,n,e,f"];
    C1["m,n"] += alpha * -0.50000000 * H2["m,e,n,u"] * T2["v,w,x,e"] * Lambda2["u,x,v,w"];
    C1["n,m"] += alpha * +0.50000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Eta1["z,v"] * Eta1["y,u"] *
                 Gamma1["w,x"];
    C1["n,m"] +=
        alpha * +1.00000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Eta1["z,v"] * Lambda2["w,y,u,x"];
    C1["n,m"] += alpha * +0.50000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Eta1["w,x"] * Gamma1["z,v"] *
                 Gamma1["y,u"];
    C1["n,m"] +=
        alpha * +0.25000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Eta1["w,x"] * Lambda2["y,z,u,v"];
    C1["n,m"] +=
        alpha * +1.00000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Gamma1["z,v"] * Lambda2["w,y,u,x"];
    C1["n,m"] +=
        alpha * +0.25000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Gamma1["w,x"] * Lambda2["y,z,u,v"];
    C1["n,m"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["n,w,x,e"] * Eta1["x,u"] * Gamma1["v,w"];
    C1["n,m"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["n,w,x,e"] * Lambda2["v,x,u,w"];
    C1["n,m"] += alpha * +0.50000000 * H2["e,f,m,u"] * T2["n,v,e,f"] * Gamma1["u,v"];
    C1["m,e"] += alpha * +0.50000000 * H2["m,u,n,c0"] * T2["n,c0,v,e"] * Eta1["v,u"];
    C1["m,f"] += alpha * -0.50000000 * H2["m,e,n,c0"] * T2["n,c0,f,e"];
    C1["m,e"] += alpha * +1.00000000 * H2["m,u,n,v"] * T2["n,w,x,e"] * Eta1["x,u"] * Gamma1["v,w"];
    C1["m,e"] += alpha * +1.00000000 * H2["m,u,n,v"] * T2["n,w,x,e"] * Lambda2["v,x,u,w"];
    C1["m,f"] += alpha * -1.00000000 * H2["m,e,n,u"] * T2["n,v,f,e"] * Gamma1["u,v"];
    C1["n,e"] += alpha * -0.50000000 * H2["u,v,m,w"] * T2["n,m,x,e"] * Lambda2["w,x,u,v"];
    C1["m,e"] += alpha * +0.50000000 * H2["m,u,n,e"] * T2["n,v,w,x"] * Lambda2["w,x,u,v"];
    C1["n,e"] += alpha * -0.50000000 * H2["u,v,m,e"] * T2["n,m,w,x"] * Eta1["x,v"] * Eta1["w,u"];
    C1["n,e"] += alpha * -0.25000000 * H2["u,v,m,e"] * T2["n,m,w,x"] * Lambda2["w,x,u,v"];
    C1["n,f"] += alpha * -1.00000000 * H2["u,e,m,f"] * T2["n,m,v,e"] * Eta1["v,u"];
    C1["n,g"] += alpha * -0.50000000 * H2["e,f,m,g"] * T2["n,m,e,f"];
    C1["m,e"] += alpha * +0.50000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Eta1["z,u"] * Gamma1["w,y"] *
                 Gamma1["v,x"];
    C1["m,e"] +=
        alpha * +0.25000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Eta1["z,u"] * Lambda2["v,w,x,y"];
    C1["m,e"] += alpha * +0.50000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Eta1["w,y"] * Eta1["v,x"] *
                 Gamma1["z,u"];
    C1["m,e"] +=
        alpha * +1.00000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Eta1["w,y"] * Lambda2["v,z,u,x"];
    C1["m,e"] +=
        alpha * +0.25000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Gamma1["z,u"] * Lambda2["v,w,x,y"];
    C1["m,e"] +=
        alpha * +1.00000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Gamma1["w,y"] * Lambda2["v,z,u,x"];
    C1["m,f"] +=
        alpha * -0.50000000 * H2["m,e,u,v"] * T2["w,x,f,e"] * Gamma1["v,x"] * Gamma1["u,w"];
    C1["m,f"] += alpha * -0.25000000 * H2["m,e,u,v"] * T2["w,x,f,e"] * Lambda2["u,v,w,x"];
    C1["m,e"] +=
        alpha * -0.50000000 * H2["u,v,w,x"] * T2["m,y,z,e"] * Eta1["z,v"] * Lambda2["w,x,u,y"];
    C1["m,e"] +=
        alpha * +0.50000000 * H2["u,v,w,x"] * T2["m,y,z,e"] * Eta1["x,y"] * Lambda2["w,z,u,v"];
    C1["m,e"] +=
        alpha * +0.50000000 * H2["u,v,w,x"] * T2["m,y,z,e"] * Gamma1["z,u"] * Lambda2["w,x,v,y"];
    C1["m,e"] +=
        alpha * +0.50000000 * H2["u,v,w,x"] * T2["m,y,z,e"] * Gamma1["x,y"] * Lambda2["w,z,u,v"];
    C1["m,f"] += alpha * +0.50000000 * H2["u,e,v,w"] * T2["m,x,f,e"] * Lambda2["v,w,u,x"];
    C1["m,f"] += alpha * +0.50000000 * H2["m,e,u,f"] * T2["v,w,x,e"] * Lambda2["u,x,v,w"];
    C1["m,e"] += alpha * -0.50000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Eta1["z,v"] * Eta1["y,u"] *
                 Gamma1["w,x"];
    C1["m,e"] +=
        alpha * -1.00000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Eta1["z,v"] * Lambda2["w,y,u,x"];
    C1["m,e"] += alpha * -0.50000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Eta1["w,x"] * Gamma1["z,v"] *
                 Gamma1["y,u"];
    C1["m,e"] +=
        alpha * -0.25000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Eta1["w,x"] * Lambda2["y,z,u,v"];
    C1["m,e"] +=
        alpha * -1.00000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Gamma1["z,v"] * Lambda2["w,y,u,x"];
    C1["m,e"] +=
        alpha * -0.25000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Gamma1["w,x"] * Lambda2["y,z,u,v"];
    C1["m,f"] += alpha * -1.00000000 * H2["u,e,v,f"] * T2["m,w,x,e"] * Eta1["x,u"] * Gamma1["v,w"];
    C1["m,f"] += alpha * -1.00000000 * H2["u,e,v,f"] * T2["m,w,x,e"] * Lambda2["v,x,u,w"];
    C1["m,g"] += alpha * -0.50000000 * H2["e,f,u,g"] * T2["m,v,e,f"] * Gamma1["u,v"];
    C1["e,v"] += alpha * +0.50000000 * H2["u,e,m,n"] * T2["m,n,v,w"] * Eta1["w,u"];
    C1["e,u"] += alpha * -0.50000000 * H2["e,f,m,n"] * T2["m,n,u,f"];
    C1["e,v"] += alpha * -0.50000000 * H2["u,e,m,v"] * T2["m,w,x,y"] * Lambda2["x,y,u,w"];
    C1["e,x"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,x,y"] * Eta1["y,u"] * Gamma1["v,w"];
    C1["e,x"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,x,y"] * Lambda2["v,y,u,w"];
    C1["e,w"] += alpha * -1.00000000 * H2["e,f,m,u"] * T2["m,v,w,f"] * Gamma1["u,v"];
    C1["e,u"] += alpha * -0.50000000 * H2["e,f,u,v"] * T2["w,x,y,f"] * Lambda2["v,y,w,x"];
    C1["e,y"] +=
        alpha * -0.50000000 * H2["e,f,u,v"] * T2["w,x,y,f"] * Gamma1["v,x"] * Gamma1["u,w"];
    C1["e,y"] += alpha * -0.25000000 * H2["e,f,u,v"] * T2["w,x,y,f"] * Lambda2["u,v,w,x"];
    C1["e,m"] += alpha * +0.50000000 * H2["u,e,m,n"] * T2["n,v,w,x"] * Lambda2["w,x,u,v"];
    C1["e,m"] += alpha * -0.50000000 * H2["e,f,m,u"] * T2["v,w,x,f"] * Lambda2["u,x,v,w"];
    C1["e,f"] += alpha * -0.50000000 * H2["u,e,m,n"] * T2["m,n,v,f"] * Eta1["v,u"];
    C1["e,g"] += alpha * -0.50000000 * H2["e,f,m,n"] * T2["m,n,g,f"];
    C1["e,f"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["m,w,x,f"] * Eta1["x,u"] * Gamma1["v,w"];
    C1["e,f"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["m,w,x,f"] * Lambda2["v,x,u,w"];
    C1["e,g"] += alpha * -1.00000000 * H2["e,f,m,u"] * T2["m,v,g,f"] * Gamma1["u,v"];
    C1["e,f"] += alpha * -0.50000000 * H2["u,e,m,f"] * T2["m,v,w,x"] * Lambda2["w,x,u,v"];
    C1["e,f"] += alpha * -0.50000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Eta1["z,u"] * Gamma1["w,y"] *
                 Gamma1["v,x"];
    C1["e,f"] +=
        alpha * -0.25000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Eta1["z,u"] * Lambda2["v,w,x,y"];
    C1["e,f"] += alpha * -0.50000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Eta1["w,y"] * Eta1["v,x"] *
                 Gamma1["z,u"];
    C1["e,f"] +=
        alpha * -1.00000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Eta1["w,y"] * Lambda2["v,z,u,x"];
    C1["e,f"] +=
        alpha * -0.25000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Gamma1["z,u"] * Lambda2["v,w,x,y"];
    C1["e,f"] +=
        alpha * -1.00000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Gamma1["w,y"] * Lambda2["v,z,u,x"];
    C1["e,g"] +=
        alpha * -0.50000000 * H2["e,f,u,v"] * T2["w,x,g,f"] * Gamma1["v,x"] * Gamma1["u,w"];
    C1["e,g"] += alpha * -0.25000000 * H2["e,f,u,v"] * T2["w,x,g,f"] * Lambda2["u,v,w,x"];
    C1["e,g"] += alpha * +0.50000000 * H2["e,f,u,g"] * T2["v,w,x,f"] * Lambda2["u,x,v,w"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H2_T2_C1 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    // 96 lines
    local_timer timer;
    C2["u,v,w,x"] += alpha * -0.50000000 * H1["u,m"] * T2["m,v,w,x"];
    C2["v,w,u,x"] += alpha * -0.50000000 * H1["e,u"] * T2["v,w,x,e"];
    C2["u,v,m,w"] += alpha * -0.50000000 * H1["e,m"] * T2["u,v,w,e"];
    C2["u,v,w,e"] += alpha * -1.00000000 * H1["u,m"] * T2["m,v,w,e"];
    C2["w,x,v,e"] += alpha * +0.50000000 * H1["u,v"] * T2["w,x,y,e"] * Eta1["y,u"];
    C2["w,x,v,e"] += alpha * +0.50000000 * H1["u,v"] * T2["w,x,y,e"] * Gamma1["y,u"];
    C2["v,w,u,f"] += alpha * -0.50000000 * H1["e,u"] * T2["v,w,f,e"];
    C2["u,w,y,e"] += alpha * +1.00000000 * H1["u,v"] * T2["w,x,y,e"] * Eta1["v,x"];
    C2["u,w,y,e"] += alpha * +1.00000000 * H1["u,v"] * T2["w,x,y,e"] * Gamma1["v,x"];
    C2["u,v,w,f"] += alpha * +0.50000000 * H1["e,f"] * T2["u,v,w,e"];
    C2["v,w,m,e"] += alpha * +0.50000000 * H1["u,m"] * T2["v,w,x,e"] * Eta1["x,u"];
    C2["v,w,m,e"] += alpha * +0.50000000 * H1["u,m"] * T2["v,w,x,e"] * Gamma1["x,u"];
    C2["u,v,m,f"] += alpha * -0.50000000 * H1["e,m"] * T2["u,v,f,e"];
    C2["u,v,e,f"] += alpha * -0.50000000 * H1["u,m"] * T2["m,v,e,f"];
    C2["u,w,e,f"] += alpha * +0.50000000 * H1["u,v"] * T2["w,x,e,f"] * Eta1["v,x"];
    C2["u,w,e,f"] += alpha * +0.50000000 * H1["u,v"] * T2["w,x,e,f"] * Gamma1["v,x"];
    C2["v,w,e,f"] += alpha * +0.50000000 * H1["u,e"] * T2["v,w,x,f"] * Eta1["x,u"];
    C2["v,w,e,f"] += alpha * +0.50000000 * H1["u,e"] * T2["v,w,x,f"] * Gamma1["x,u"];
    C2["u,v,f,g"] += alpha * -0.50000000 * H1["e,f"] * T2["u,v,g,e"];
    C2["u,e,v,w"] += alpha * +0.50000000 * H1["e,m"] * T2["m,u,v,w"];
    C2["u,e,v,f"] += alpha * +1.00000000 * H1["e,m"] * T2["m,u,v,f"];
    C2["v,e,x,f"] += alpha * -1.00000000 * H1["e,u"] * T2["v,w,x,f"] * Eta1["u,w"];
    C2["v,e,x,f"] += alpha * -1.00000000 * H1["e,u"] * T2["v,w,x,f"] * Gamma1["u,w"];
    C2["u,e,f,g"] += alpha * +0.50000000 * H1["e,m"] * T2["m,u,f,g"];
    C2["v,e,f,g"] += alpha * -0.50000000 * H1["e,u"] * T2["v,w,f,g"] * Eta1["u,w"];
    C2["v,e,f,g"] += alpha * -0.50000000 * H1["e,u"] * T2["v,w,f,g"] * Gamma1["u,w"];
    C2["m,u,v,w"] += alpha * -0.50000000 * H1["m,n"] * T2["n,u,v,w"];
    C2["n,u,v,w"] += alpha * -0.50000000 * H1["u,m"] * T2["n,m,v,w"];
    C2["m,w,v,x"] += alpha * -1.00000000 * H1["u,v"] * T2["m,w,x,y"] * Eta1["y,u"];
    C2["m,w,v,x"] += alpha * -1.00000000 * H1["u,v"] * T2["m,w,x,y"] * Gamma1["y,u"];
    C2["m,v,u,w"] += alpha * -1.00000000 * H1["e,u"] * T2["m,v,w,e"];
    C2["m,u,x,y"] += alpha * -0.50000000 * H1["u,v"] * T2["m,w,x,y"] * Eta1["v,w"];
    C2["m,u,x,y"] += alpha * -0.50000000 * H1["u,v"] * T2["m,w,x,y"] * Gamma1["v,w"];
    C2["n,v,m,w"] += alpha * -1.00000000 * H1["u,m"] * T2["n,v,w,x"] * Eta1["x,u"];
    C2["n,v,m,w"] += alpha * -1.00000000 * H1["u,m"] * T2["n,v,w,x"] * Gamma1["x,u"];
    C2["n,u,m,v"] += alpha * -1.00000000 * H1["e,m"] * T2["n,u,v,e"];
    C2["m,u,v,e"] += alpha * -1.00000000 * H1["m,n"] * T2["n,u,v,e"];
    C2["n,u,v,e"] += alpha * -1.00000000 * H1["u,m"] * T2["n,m,v,e"];
    C2["m,w,v,e"] += alpha * +1.00000000 * H1["u,v"] * T2["m,w,x,e"] * Eta1["x,u"];
    C2["m,w,v,e"] += alpha * +1.00000000 * H1["u,v"] * T2["m,w,x,e"] * Gamma1["x,u"];
    C2["m,v,u,f"] += alpha * -1.00000000 * H1["e,u"] * T2["m,v,f,e"];
    C2["m,v,x,e"] += alpha * +1.00000000 * H1["m,u"] * T2["v,w,x,e"] * Eta1["u,w"];
    C2["m,v,x,e"] += alpha * +1.00000000 * H1["m,u"] * T2["v,w,x,e"] * Gamma1["u,w"];
    C2["m,u,x,e"] += alpha * -1.00000000 * H1["u,v"] * T2["m,w,x,e"] * Eta1["v,w"];
    C2["m,u,x,e"] += alpha * -1.00000000 * H1["u,v"] * T2["m,w,x,e"] * Gamma1["v,w"];
    C2["m,v,w,e"] += alpha * +1.00000000 * H1["u,e"] * T2["m,v,w,x"] * Eta1["x,u"];
    C2["m,v,w,e"] += alpha * +1.00000000 * H1["u,e"] * T2["m,v,w,x"] * Gamma1["x,u"];
    C2["m,u,v,f"] += alpha * +1.00000000 * H1["e,f"] * T2["m,u,v,e"];
    C2["n,v,m,e"] += alpha * +1.00000000 * H1["u,m"] * T2["n,v,w,e"] * Eta1["w,u"];
    C2["n,v,m,e"] += alpha * +1.00000000 * H1["u,m"] * T2["n,v,w,e"] * Gamma1["w,u"];
    C2["n,u,m,f"] += alpha * -1.00000000 * H1["e,m"] * T2["n,u,f,e"];
    C2["m,u,e,f"] += alpha * -0.50000000 * H1["m,n"] * T2["n,u,e,f"];
    C2["n,u,e,f"] += alpha * -0.50000000 * H1["u,m"] * T2["n,m,e,f"];
    C2["m,v,e,f"] += alpha * +0.50000000 * H1["m,u"] * T2["v,w,e,f"] * Eta1["u,w"];
    C2["m,v,e,f"] += alpha * +0.50000000 * H1["m,u"] * T2["v,w,e,f"] * Gamma1["u,w"];
    C2["m,u,e,f"] += alpha * -0.50000000 * H1["u,v"] * T2["m,w,e,f"] * Eta1["v,w"];
    C2["m,u,e,f"] += alpha * -0.50000000 * H1["u,v"] * T2["m,w,e,f"] * Gamma1["v,w"];
    C2["m,v,e,f"] += alpha * +1.00000000 * H1["u,e"] * T2["m,v,w,f"] * Eta1["w,u"];
    C2["m,v,e,f"] += alpha * +1.00000000 * H1["u,e"] * T2["m,v,w,f"] * Gamma1["w,u"];
    C2["m,u,f,g"] += alpha * -1.00000000 * H1["e,f"] * T2["m,u,g,e"];
    C2["m,c0,u,v"] += alpha * +0.50000000 * H1["m,n"] * T2["c0,n,u,v"];
    C2["m,n,v,w"] += alpha * -0.50000000 * H1["u,v"] * T2["m,n,w,x"] * Eta1["x,u"];
    C2["m,n,v,w"] += alpha * -0.50000000 * H1["u,v"] * T2["m,n,w,x"] * Gamma1["x,u"];
    C2["m,n,u,v"] += alpha * -0.50000000 * H1["e,u"] * T2["m,n,v,e"];
    C2["m,n,w,x"] += alpha * +0.50000000 * H1["m,u"] * T2["n,v,w,x"] * Eta1["u,v"];
    C2["m,n,w,x"] += alpha * +0.50000000 * H1["m,u"] * T2["n,v,w,x"] * Gamma1["u,v"];
    C2["n,c0,m,v"] += alpha * -0.50000000 * H1["u,m"] * T2["n,c0,v,w"] * Eta1["w,u"];
    C2["n,c0,m,v"] += alpha * -0.50000000 * H1["u,m"] * T2["n,c0,v,w"] * Gamma1["w,u"];
    C2["n,c0,m,u"] += alpha * -0.50000000 * H1["e,m"] * T2["n,c0,u,e"];
    C2["m,c0,u,e"] += alpha * +1.00000000 * H1["m,n"] * T2["c0,n,u,e"];
    C2["m,n,v,e"] += alpha * +0.50000000 * H1["u,v"] * T2["m,n,w,e"] * Eta1["w,u"];
    C2["m,n,v,e"] += alpha * +0.50000000 * H1["u,v"] * T2["m,n,w,e"] * Gamma1["w,u"];
    C2["m,n,u,f"] += alpha * -0.50000000 * H1["e,u"] * T2["m,n,f,e"];
    C2["m,n,w,e"] += alpha * +1.00000000 * H1["m,u"] * T2["n,v,w,e"] * Eta1["u,v"];
    C2["m,n,w,e"] += alpha * +1.00000000 * H1["m,u"] * T2["n,v,w,e"] * Gamma1["u,v"];
    C2["m,n,v,e"] += alpha * +0.50000000 * H1["u,e"] * T2["m,n,v,w"] * Eta1["w,u"];
    C2["m,n,v,e"] += alpha * +0.50000000 * H1["u,e"] * T2["m,n,v,w"] * Gamma1["w,u"];
    C2["m,n,u,f"] += alpha * +0.50000000 * H1["e,f"] * T2["m,n,u,e"];
    C2["n,c0,m,e"] += alpha * +0.50000000 * H1["u,m"] * T2["n,c0,v,e"] * Eta1["v,u"];
    C2["n,c0,m,e"] += alpha * +0.50000000 * H1["u,m"] * T2["n,c0,v,e"] * Gamma1["v,u"];
    C2["n,c0,m,f"] += alpha * -0.50000000 * H1["e,m"] * T2["n,c0,f,e"];
    C2["m,c0,e,f"] += alpha * +0.50000000 * H1["m,n"] * T2["c0,n,e,f"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H1["m,u"] * T2["n,v,e,f"] * Eta1["u,v"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H1["m,u"] * T2["n,v,e,f"] * Gamma1["u,v"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H1["u,e"] * T2["m,n,v,f"] * Eta1["v,u"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H1["u,e"] * T2["m,n,v,f"] * Gamma1["v,u"];
    C2["m,n,f,g"] += alpha * -0.50000000 * H1["e,f"] * T2["m,n,g,e"];
    C2["n,e,u,v"] += alpha * -0.50000000 * H1["e,m"] * T2["n,m,u,v"];
    C2["m,e,w,x"] += alpha * -0.50000000 * H1["e,u"] * T2["m,v,w,x"] * Eta1["u,v"];
    C2["m,e,w,x"] += alpha * -0.50000000 * H1["e,u"] * T2["m,v,w,x"] * Gamma1["u,v"];
    C2["n,e,u,f"] += alpha * -1.00000000 * H1["e,m"] * T2["n,m,u,f"];
    C2["m,e,w,f"] += alpha * -1.00000000 * H1["e,u"] * T2["m,v,w,f"] * Eta1["u,v"];
    C2["m,e,w,f"] += alpha * -1.00000000 * H1["e,u"] * T2["m,v,w,f"] * Gamma1["u,v"];
    C2["n,e,f,g"] += alpha * -0.50000000 * H1["e,m"] * T2["n,m,f,g"];
    C2["m,e,f,g"] += alpha * -0.50000000 * H1["e,u"] * T2["m,v,f,g"] * Eta1["u,v"];
    C2["m,e,f,g"] += alpha * -0.50000000 * H1["e,u"] * T2["m,v,f,g"] * Gamma1["u,v"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H1_T2_C2 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C2) {
    // 144 lines
    local_timer timer;
    C2["u,v,w,x"] += alpha * +0.50000000 * H2["u,v,m,w"] * T1["m,x"];
    C2["u,x,v,w"] += alpha * +0.50000000 * H2["u,e,v,w"] * T1["x,e"];
    C2["u,v,m,w"] += alpha * -0.50000000 * H2["u,v,m,n"] * T1["n,w"];
    C2["u,w,m,v"] += alpha * +1.00000000 * H2["u,e,m,v"] * T1["w,e"];
    C2["u,v,m,n"] += alpha * +0.50000000 * H2["u,e,m,n"] * T1["v,e"];
    C2["u,v,w,e"] += alpha * +0.50000000 * H2["u,v,m,w"] * T1["m,e"];
    C2["u,v,w,e"] += alpha * -0.50000000 * H2["u,v,m,e"] * T1["m,w"];
    C2["u,v,w,e"] += alpha * -0.50000000 * H2["u,v,w,x"] * T1["y,e"] * Eta1["x,y"];
    C2["u,v,w,e"] += alpha * -0.50000000 * H2["u,v,w,x"] * T1["y,e"] * Gamma1["x,y"];
    C2["u,w,v,f"] += alpha * +1.00000000 * H2["u,e,v,f"] * T1["w,e"];
    C2["u,v,m,e"] += alpha * -0.50000000 * H2["u,v,m,n"] * T1["n,e"];
    C2["u,v,m,e"] += alpha * -0.50000000 * H2["u,v,m,w"] * T1["x,e"] * Eta1["w,x"];
    C2["u,v,m,e"] += alpha * -0.50000000 * H2["u,v,m,w"] * T1["x,e"] * Gamma1["w,x"];
    C2["u,v,m,f"] += alpha * +1.00000000 * H2["u,e,m,f"] * T1["v,e"];
    C2["u,v,e,f"] += alpha * +0.50000000 * H2["u,v,m,e"] * T1["m,f"];
    C2["u,v,e,f"] += alpha * +0.50000000 * H2["u,v,w,e"] * T1["x,f"] * Eta1["w,x"];
    C2["u,v,e,f"] += alpha * +0.50000000 * H2["u,v,w,e"] * T1["x,f"] * Gamma1["w,x"];
    C2["u,v,f,g"] += alpha * +0.50000000 * H2["u,e,f,g"] * T1["v,e"];
    C2["u,e,v,w"] += alpha * +1.00000000 * H2["u,e,m,v"] * T1["m,w"];
    C2["w,e,u,v"] += alpha * -0.50000000 * H2["e,f,u,v"] * T1["w,f"];
    C2["u,e,m,v"] += alpha * -1.00000000 * H2["u,e,m,n"] * T1["n,v"];
    C2["v,e,m,u"] += alpha * -1.00000000 * H2["e,f,m,u"] * T1["v,f"];
    C2["u,e,m,n"] += alpha * -0.50000000 * H2["e,f,m,n"] * T1["u,f"];
    C2["u,e,v,f"] += alpha * +1.00000000 * H2["u,e,m,v"] * T1["m,f"];
    C2["u,e,v,f"] += alpha * -1.00000000 * H2["u,e,m,f"] * T1["m,v"];
    C2["u,e,v,f"] += alpha * -1.00000000 * H2["u,e,v,w"] * T1["x,f"] * Eta1["w,x"];
    C2["u,e,v,f"] += alpha * -1.00000000 * H2["u,e,v,w"] * T1["x,f"] * Gamma1["w,x"];
    C2["v,e,u,g"] += alpha * -1.00000000 * H2["e,f,u,g"] * T1["v,f"];
    C2["u,e,m,f"] += alpha * -1.00000000 * H2["u,e,m,n"] * T1["n,f"];
    C2["u,e,m,f"] += alpha * -1.00000000 * H2["u,e,m,v"] * T1["w,f"] * Eta1["v,w"];
    C2["u,e,m,f"] += alpha * -1.00000000 * H2["u,e,m,v"] * T1["w,f"] * Gamma1["v,w"];
    C2["u,e,m,g"] += alpha * -1.00000000 * H2["e,f,m,g"] * T1["u,f"];
    C2["u,e,f,g"] += alpha * +1.00000000 * H2["u,e,m,f"] * T1["m,g"];
    C2["u,e,f,g"] += alpha * +1.00000000 * H2["u,e,v,f"] * T1["w,g"] * Eta1["v,w"];
    C2["u,e,f,g"] += alpha * +1.00000000 * H2["u,e,v,f"] * T1["w,g"] * Gamma1["v,w"];
    C2["u,e,g,h"] += alpha * -0.50000000 * H2["e,f,g,h"] * T1["u,f"];
    C2["m,u,v,w"] += alpha * +1.00000000 * H2["m,u,n,v"] * T1["n,w"];
    C2["m,w,u,v"] += alpha * +0.50000000 * H2["m,e,u,v"] * T1["w,e"];
    C2["m,u,w,x"] += alpha * -0.50000000 * H2["u,v,w,x"] * T1["m,y"] * Eta1["y,v"];
    C2["m,u,w,x"] += alpha * -0.50000000 * H2["u,v,w,x"] * T1["m,y"] * Gamma1["y,v"];
    C2["m,u,v,w"] += alpha * -0.50000000 * H2["u,e,v,w"] * T1["m,e"];
    C2["m,u,n,v"] += alpha * -1.00000000 * H2["m,u,n,c0"] * T1["c0,v"];
    C2["m,v,n,u"] += alpha * +1.00000000 * H2["m,e,n,u"] * T1["v,e"];
    C2["n,u,m,w"] += alpha * -1.00000000 * H2["u,v,m,w"] * T1["n,x"] * Eta1["x,v"];
    C2["n,u,m,w"] += alpha * -1.00000000 * H2["u,v,m,w"] * T1["n,x"] * Gamma1["x,v"];
    C2["n,u,m,v"] += alpha * -1.00000000 * H2["u,e,m,v"] * T1["n,e"];
    C2["m,u,n,c0"] += alpha * +0.50000000 * H2["m,e,n,c0"] * T1["u,e"];
    C2["c0,u,m,n"] += alpha * -0.50000000 * H2["u,v,m,n"] * T1["c0,w"] * Eta1["w,v"];
    C2["c0,u,m,n"] += alpha * -0.50000000 * H2["u,v,m,n"] * T1["c0,w"] * Gamma1["w,v"];
    C2["c0,u,m,n"] += alpha * -0.50000000 * H2["u,e,m,n"] * T1["c0,e"];
    C2["m,u,v,e"] += alpha * +1.00000000 * H2["m,u,n,v"] * T1["n,e"];
    C2["m,u,v,e"] += alpha * -1.00000000 * H2["m,u,n,e"] * T1["n,v"];
    C2["m,u,v,e"] += alpha * -1.00000000 * H2["m,u,v,w"] * T1["x,e"] * Eta1["w,x"];
    C2["m,u,v,e"] += alpha * -1.00000000 * H2["m,u,v,w"] * T1["x,e"] * Gamma1["w,x"];
    C2["m,v,u,f"] += alpha * +1.00000000 * H2["m,e,u,f"] * T1["v,e"];
    C2["m,u,w,e"] += alpha * -1.00000000 * H2["u,v,w,e"] * T1["m,x"] * Eta1["x,v"];
    C2["m,u,w,e"] += alpha * -1.00000000 * H2["u,v,w,e"] * T1["m,x"] * Gamma1["x,v"];
    C2["m,u,v,f"] += alpha * -1.00000000 * H2["u,e,v,f"] * T1["m,e"];
    C2["m,u,n,e"] += alpha * -1.00000000 * H2["m,u,n,c0"] * T1["c0,e"];
    C2["m,u,n,e"] += alpha * -1.00000000 * H2["m,u,n,v"] * T1["w,e"] * Eta1["v,w"];
    C2["m,u,n,e"] += alpha * -1.00000000 * H2["m,u,n,v"] * T1["w,e"] * Gamma1["v,w"];
    C2["m,u,n,f"] += alpha * +1.00000000 * H2["m,e,n,f"] * T1["u,e"];
    C2["n,u,m,e"] += alpha * -1.00000000 * H2["u,v,m,e"] * T1["n,w"] * Eta1["w,v"];
    C2["n,u,m,e"] += alpha * -1.00000000 * H2["u,v,m,e"] * T1["n,w"] * Gamma1["w,v"];
    C2["n,u,m,f"] += alpha * -1.00000000 * H2["u,e,m,f"] * T1["n,e"];
    C2["m,u,e,f"] += alpha * +1.00000000 * H2["m,u,n,e"] * T1["n,f"];
    C2["m,u,e,f"] += alpha * +1.00000000 * H2["m,u,v,e"] * T1["w,f"] * Eta1["v,w"];
    C2["m,u,e,f"] += alpha * +1.00000000 * H2["m,u,v,e"] * T1["w,f"] * Gamma1["v,w"];
    C2["m,u,f,g"] += alpha * +0.50000000 * H2["m,e,f,g"] * T1["u,e"];
    C2["m,u,e,f"] += alpha * -0.50000000 * H2["u,v,e,f"] * T1["m,w"] * Eta1["w,v"];
    C2["m,u,e,f"] += alpha * -0.50000000 * H2["u,v,e,f"] * T1["m,w"] * Gamma1["w,v"];
    C2["m,u,f,g"] += alpha * -0.50000000 * H2["u,e,f,g"] * T1["m,e"];
    C2["m,n,u,v"] += alpha * +0.50000000 * H2["m,n,c0,u"] * T1["c0,v"];
    C2["m,n,v,w"] += alpha * +0.50000000 * H2["m,u,v,w"] * T1["n,x"] * Eta1["x,u"];
    C2["m,n,v,w"] += alpha * +0.50000000 * H2["m,u,v,w"] * T1["n,x"] * Gamma1["x,u"];
    C2["m,n,u,v"] += alpha * +0.50000000 * H2["m,e,u,v"] * T1["n,e"];
    C2["m,n,c0,u"] += alpha * -0.50000000 * H2["m,n,c0,c1"] * T1["c1,u"];
    C2["m,c0,n,v"] += alpha * +1.00000000 * H2["m,u,n,v"] * T1["c0,w"] * Eta1["w,u"];
    C2["m,c0,n,v"] += alpha * +1.00000000 * H2["m,u,n,v"] * T1["c0,w"] * Gamma1["w,u"];
    C2["m,c0,n,u"] += alpha * +1.00000000 * H2["m,e,n,u"] * T1["c0,e"];
    C2["m,c1,n,c0"] += alpha * +0.50000000 * H2["m,u,n,c0"] * T1["c1,v"] * Eta1["v,u"];
    C2["m,c1,n,c0"] += alpha * +0.50000000 * H2["m,u,n,c0"] * T1["c1,v"] * Gamma1["v,u"];
    C2["m,c1,n,c0"] += alpha * +0.50000000 * H2["m,e,n,c0"] * T1["c1,e"];
    C2["m,n,u,e"] += alpha * +0.50000000 * H2["m,n,c0,u"] * T1["c0,e"];
    C2["m,n,u,e"] += alpha * -0.50000000 * H2["m,n,c0,e"] * T1["c0,u"];
    C2["m,n,u,e"] += alpha * -0.50000000 * H2["m,n,u,v"] * T1["w,e"] * Eta1["v,w"];
    C2["m,n,u,e"] += alpha * -0.50000000 * H2["m,n,u,v"] * T1["w,e"] * Gamma1["v,w"];
    C2["m,n,v,e"] += alpha * +1.00000000 * H2["m,u,v,e"] * T1["n,w"] * Eta1["w,u"];
    C2["m,n,v,e"] += alpha * +1.00000000 * H2["m,u,v,e"] * T1["n,w"] * Gamma1["w,u"];
    C2["m,n,u,f"] += alpha * +1.00000000 * H2["m,e,u,f"] * T1["n,e"];
    C2["m,n,c0,e"] += alpha * -0.50000000 * H2["m,n,c0,c1"] * T1["c1,e"];
    C2["m,n,c0,e"] += alpha * -0.50000000 * H2["m,n,c0,u"] * T1["v,e"] * Eta1["u,v"];
    C2["m,n,c0,e"] += alpha * -0.50000000 * H2["m,n,c0,u"] * T1["v,e"] * Gamma1["u,v"];
    C2["m,c0,n,e"] += alpha * +1.00000000 * H2["m,u,n,e"] * T1["c0,v"] * Eta1["v,u"];
    C2["m,c0,n,e"] += alpha * +1.00000000 * H2["m,u,n,e"] * T1["c0,v"] * Gamma1["v,u"];
    C2["m,c0,n,f"] += alpha * +1.00000000 * H2["m,e,n,f"] * T1["c0,e"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H2["m,n,c0,e"] * T1["c0,f"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H2["m,n,u,e"] * T1["v,f"] * Eta1["u,v"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H2["m,n,u,e"] * T1["v,f"] * Gamma1["u,v"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H2["m,u,e,f"] * T1["n,v"] * Eta1["v,u"];
    C2["m,n,e,f"] += alpha * +0.50000000 * H2["m,u,e,f"] * T1["n,v"] * Gamma1["v,u"];
    C2["m,n,f,g"] += alpha * +0.50000000 * H2["m,e,f,g"] * T1["n,e"];
    C2["m,e,u,v"] += alpha * +1.00000000 * H2["m,e,n,u"] * T1["n,v"];
    C2["m,e,v,w"] += alpha * +0.50000000 * H2["u,e,v,w"] * T1["m,x"] * Eta1["x,u"];
    C2["m,e,v,w"] += alpha * +0.50000000 * H2["u,e,v,w"] * T1["m,x"] * Gamma1["x,u"];
    C2["m,e,u,v"] += alpha * -0.50000000 * H2["e,f,u,v"] * T1["m,f"];
    C2["m,e,n,u"] += alpha * -1.00000000 * H2["m,e,n,c0"] * T1["c0,u"];
    C2["n,e,m,v"] += alpha * +1.00000000 * H2["u,e,m,v"] * T1["n,w"] * Eta1["w,u"];
    C2["n,e,m,v"] += alpha * +1.00000000 * H2["u,e,m,v"] * T1["n,w"] * Gamma1["w,u"];
    C2["n,e,m,u"] += alpha * -1.00000000 * H2["e,f,m,u"] * T1["n,f"];
    C2["c0,e,m,n"] += alpha * +0.50000000 * H2["u,e,m,n"] * T1["c0,v"] * Eta1["v,u"];
    C2["c0,e,m,n"] += alpha * +0.50000000 * H2["u,e,m,n"] * T1["c0,v"] * Gamma1["v,u"];
    C2["c0,e,m,n"] += alpha * -0.50000000 * H2["e,f,m,n"] * T1["c0,f"];
    C2["m,e,u,f"] += alpha * +1.00000000 * H2["m,e,n,u"] * T1["n,f"];
    C2["m,e,u,f"] += alpha * -1.00000000 * H2["m,e,n,f"] * T1["n,u"];
    C2["m,e,u,f"] += alpha * -1.00000000 * H2["m,e,u,v"] * T1["w,f"] * Eta1["v,w"];
    C2["m,e,u,f"] += alpha * -1.00000000 * H2["m,e,u,v"] * T1["w,f"] * Gamma1["v,w"];
    C2["m,e,v,f"] += alpha * +1.00000000 * H2["u,e,v,f"] * T1["m,w"] * Eta1["w,u"];
    C2["m,e,v,f"] += alpha * +1.00000000 * H2["u,e,v,f"] * T1["m,w"] * Gamma1["w,u"];
    C2["m,e,u,g"] += alpha * -1.00000000 * H2["e,f,u,g"] * T1["m,f"];
    C2["m,e,n,f"] += alpha * -1.00000000 * H2["m,e,n,c0"] * T1["c0,f"];
    C2["m,e,n,f"] += alpha * -1.00000000 * H2["m,e,n,u"] * T1["v,f"] * Eta1["u,v"];
    C2["m,e,n,f"] += alpha * -1.00000000 * H2["m,e,n,u"] * T1["v,f"] * Gamma1["u,v"];
    C2["n,e,m,f"] += alpha * +1.00000000 * H2["u,e,m,f"] * T1["n,v"] * Eta1["v,u"];
    C2["n,e,m,f"] += alpha * +1.00000000 * H2["u,e,m,f"] * T1["n,v"] * Gamma1["v,u"];
    C2["n,e,m,g"] += alpha * -1.00000000 * H2["e,f,m,g"] * T1["n,f"];
    C2["m,e,f,g"] += alpha * +1.00000000 * H2["m,e,n,f"] * T1["n,g"];
    C2["m,e,f,g"] += alpha * +1.00000000 * H2["m,e,u,f"] * T1["v,g"] * Eta1["u,v"];
    C2["m,e,f,g"] += alpha * +1.00000000 * H2["m,e,u,f"] * T1["v,g"] * Gamma1["u,v"];
    C2["m,e,f,g"] += alpha * +0.50000000 * H2["u,e,f,g"] * T1["m,v"] * Eta1["v,u"];
    C2["m,e,f,g"] += alpha * +0.50000000 * H2["u,e,f,g"] * T1["m,v"] * Gamma1["v,u"];
    C2["m,e,g,h"] += alpha * -0.50000000 * H2["e,f,g,h"] * T1["m,f"];
    C2["e,f,u,v"] += alpha * +0.50000000 * H2["e,f,m,u"] * T1["m,v"];
    C2["e,f,m,u"] += alpha * -0.50000000 * H2["e,f,m,n"] * T1["n,u"];
    C2["e,f,u,g"] += alpha * +0.50000000 * H2["e,f,m,u"] * T1["m,g"];
    C2["e,f,u,g"] += alpha * -0.50000000 * H2["e,f,m,g"] * T1["m,u"];
    C2["e,f,u,g"] += alpha * -0.50000000 * H2["e,f,u,v"] * T1["w,g"] * Eta1["v,w"];
    C2["e,f,u,g"] += alpha * -0.50000000 * H2["e,f,u,v"] * T1["w,g"] * Gamma1["v,w"];
    C2["e,f,m,g"] += alpha * -0.50000000 * H2["e,f,m,n"] * T1["n,g"];
    C2["e,f,m,g"] += alpha * -0.50000000 * H2["e,f,m,u"] * T1["v,g"] * Eta1["u,v"];
    C2["e,f,m,g"] += alpha * -0.50000000 * H2["e,f,m,u"] * T1["v,g"] * Gamma1["u,v"];
    C2["e,f,g,h"] += alpha * +0.50000000 * H2["e,f,m,g"] * T1["m,h"];
    C2["e,f,g,h"] += alpha * +0.50000000 * H2["e,f,u,g"] * T1["v,h"] * Eta1["u,v"];
    C2["e,f,g,h"] += alpha * +0.50000000 * H2["e,f,u,g"] * T1["v,h"] * Gamma1["u,v"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H2_T1_C2 : %12.3f", timer.get());
    //}
}

void MRDSRG_SO::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    // 282 lines
    local_timer timer;
    C2["u,v,w,x"] += alpha * +0.12500000 * H2["u,v,m,n"] * T2["m,n,w,x"];
    C2["u,x,w,y"] += alpha * +1.00000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Eta1["z,v"];
    C2["u,w,v,x"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,x,e"];
    C2["u,v,y,z"] += alpha * +0.25000000 * H2["u,v,m,w"] * T2["m,x,y,z"] * Gamma1["w,x"];
    C2["x,y,v,w"] += alpha * +0.25000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Eta1["z,u"];
    C2["w,x,u,v"] += alpha * +0.12500000 * H2["e,f,u,v"] * T2["w,x,e,f"];
    C2["u,x,v,z"] += alpha * +1.00000000 * H2["u,e,v,w"] * T2["x,y,z,e"] * Gamma1["w,y"];
    C2["u,w,m,x"] += alpha * -1.00000000 * H2["u,v,m,n"] * T2["n,w,x,y"] * Eta1["y,v"];
    C2["u,v,m,w"] += alpha * -1.00000000 * H2["u,e,m,n"] * T2["n,v,w,e"];
    C2["w,x,m,v"] += alpha * +0.50000000 * H2["u,e,m,v"] * T2["w,x,y,e"] * Eta1["y,u"];
    C2["v,w,m,u"] += alpha * +0.25000000 * H2["e,f,m,u"] * T2["v,w,e,f"];
    C2["u,w,m,y"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["w,x,y,e"] * Gamma1["v,x"];
    C2["v,w,m,n"] += alpha * +0.25000000 * H2["u,e,m,n"] * T2["v,w,x,e"] * Eta1["x,u"];
    C2["u,v,m,n"] += alpha * +0.12500000 * H2["e,f,m,n"] * T2["u,v,e,f"];
    C2["u,v,w,e"] += alpha * +0.25000000 * H2["u,v,m,n"] * T2["m,n,w,e"];
    C2["u,x,w,e"] += alpha * -1.00000000 * H2["u,v,m,w"] * T2["m,x,y,e"] * Eta1["y,v"];
    C2["u,w,v,f"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,f,e"];
    C2["u,v,y,e"] += alpha * +0.50000000 * H2["u,v,m,w"] * T2["m,x,y,e"] * Gamma1["w,x"];
    C2["u,w,x,e"] += alpha * -1.00000000 * H2["u,v,m,e"] * T2["m,w,x,y"] * Eta1["y,v"];
    C2["u,v,w,f"] += alpha * -1.00000000 * H2["u,e,m,f"] * T2["m,v,w,e"];
    C2["u,y,w,e"] +=
        alpha * -1.00000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["a0,v"] * Gamma1["x,z"];
    C2["u,y,w,e"] +=
        alpha * +1.00000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["x,z"] * Gamma1["a0,v"];
    C2["u,x,v,f"] += alpha * +1.00000000 * H2["u,e,v,w"] * T2["x,y,f,e"] * Gamma1["w,y"];
    C2["w,x,v,f"] += alpha * +0.50000000 * H2["u,e,v,f"] * T2["w,x,y,e"] * Eta1["y,u"];
    C2["v,w,u,g"] += alpha * +0.25000000 * H2["e,f,u,g"] * T2["v,w,e,f"];
    C2["u,v,a0,e"] +=
        alpha * -0.25000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Eta1["x,z"] * Eta1["w,y"];
    C2["u,v,a0,e"] +=
        alpha * +0.25000000 * H2["u,v,w,x"] * T2["y,z,a0,e"] * Gamma1["x,z"] * Gamma1["w,y"];
    C2["u,w,y,f"] += alpha * +1.00000000 * H2["u,e,v,f"] * T2["w,x,y,e"] * Gamma1["v,x"];
    C2["u,w,m,e"] += alpha * +1.00000000 * H2["u,v,m,n"] * T2["n,w,x,e"] * Eta1["x,v"];
    C2["u,v,m,f"] += alpha * -1.00000000 * H2["u,e,m,n"] * T2["n,v,f,e"];
    C2["u,x,m,e"] +=
        alpha * -1.00000000 * H2["u,v,m,w"] * T2["x,y,z,e"] * Eta1["z,v"] * Gamma1["w,y"];
    C2["u,x,m,e"] +=
        alpha * +1.00000000 * H2["u,v,m,w"] * T2["x,y,z,e"] * Eta1["w,y"] * Gamma1["z,v"];
    C2["u,w,m,f"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["w,x,f,e"] * Gamma1["v,x"];
    C2["v,w,m,f"] += alpha * +0.50000000 * H2["u,e,m,f"] * T2["v,w,x,e"] * Eta1["x,u"];
    C2["u,v,m,g"] += alpha * +0.25000000 * H2["e,f,m,g"] * T2["u,v,e,f"];
    C2["u,v,e,f"] += alpha * +0.12500000 * H2["u,v,m,n"] * T2["m,n,e,f"];
    C2["u,v,e,f"] += alpha * +0.25000000 * H2["u,v,m,w"] * T2["m,x,e,f"] * Gamma1["w,x"];
    C2["u,w,e,f"] += alpha * -1.00000000 * H2["u,v,m,e"] * T2["m,w,x,f"] * Eta1["x,v"];
    C2["u,v,f,g"] += alpha * +1.00000000 * H2["u,e,m,f"] * T2["m,v,g,e"];
    C2["u,v,e,f"] +=
        alpha * -0.12500000 * H2["u,v,w,x"] * T2["y,z,e,f"] * Eta1["x,z"] * Eta1["w,y"];
    C2["u,v,e,f"] +=
        alpha * +0.12500000 * H2["u,v,w,x"] * T2["y,z,e,f"] * Gamma1["x,z"] * Gamma1["w,y"];
    C2["u,x,e,f"] +=
        alpha * +1.00000000 * H2["u,v,w,e"] * T2["x,y,z,f"] * Eta1["z,v"] * Gamma1["w,y"];
    C2["u,x,e,f"] +=
        alpha * -1.00000000 * H2["u,v,w,e"] * T2["x,y,z,f"] * Eta1["w,y"] * Gamma1["z,v"];
    C2["u,w,f,g"] += alpha * -1.00000000 * H2["u,e,v,f"] * T2["w,x,g,e"] * Gamma1["v,x"];
    C2["v,w,f,g"] += alpha * +0.25000000 * H2["u,e,f,g"] * T2["v,w,x,e"] * Eta1["x,u"];
    C2["u,v,g,h"] += alpha * +0.12500000 * H2["e,f,g,h"] * T2["u,v,e,f"];
    C2["u,e,v,w"] += alpha * +0.25000000 * H2["u,e,m,n"] * T2["m,n,v,w"];
    C2["w,e,v,x"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,x,y"] * Eta1["y,u"];
    C2["v,e,u,w"] += alpha * -1.00000000 * H2["e,f,m,u"] * T2["m,v,w,f"];
    C2["u,e,x,y"] += alpha * +0.50000000 * H2["u,e,m,v"] * T2["m,w,x,y"] * Gamma1["v,w"];
    C2["w,e,u,y"] += alpha * -1.00000000 * H2["e,f,u,v"] * T2["w,x,y,f"] * Gamma1["v,x"];
    C2["v,e,m,w"] += alpha * -1.00000000 * H2["u,e,m,n"] * T2["n,v,w,x"] * Eta1["x,u"];
    C2["u,e,m,v"] += alpha * +1.00000000 * H2["e,f,m,n"] * T2["n,u,v,f"];
    C2["v,e,m,x"] += alpha * -1.00000000 * H2["e,f,m,u"] * T2["v,w,x,f"] * Gamma1["u,w"];
    C2["u,e,v,f"] += alpha * +0.50000000 * H2["u,e,m,n"] * T2["m,n,v,f"];
    C2["w,e,v,f"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["m,w,x,f"] * Eta1["x,u"];
    C2["v,e,u,g"] += alpha * -1.00000000 * H2["e,f,m,u"] * T2["m,v,g,f"];
    C2["u,e,x,f"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["m,w,x,f"] * Gamma1["v,w"];
    C2["v,e,w,f"] += alpha * -1.00000000 * H2["u,e,m,f"] * T2["m,v,w,x"] * Eta1["x,u"];
    C2["u,e,v,g"] += alpha * +1.00000000 * H2["e,f,m,g"] * T2["m,u,v,f"];
    C2["x,e,v,f"] +=
        alpha * -1.00000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Eta1["z,u"] * Gamma1["w,y"];
    C2["x,e,v,f"] +=
        alpha * +1.00000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Eta1["w,y"] * Gamma1["z,u"];
    C2["w,e,u,g"] += alpha * -1.00000000 * H2["e,f,u,v"] * T2["w,x,g,f"] * Gamma1["v,x"];
    C2["u,e,z,f"] +=
        alpha * -0.50000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Eta1["w,y"] * Eta1["v,x"];
    C2["u,e,z,f"] +=
        alpha * +0.50000000 * H2["u,e,v,w"] * T2["x,y,z,f"] * Gamma1["w,y"] * Gamma1["v,x"];
    C2["v,e,x,g"] += alpha * -1.00000000 * H2["e,f,u,g"] * T2["v,w,x,f"] * Gamma1["u,w"];
    C2["v,e,m,f"] += alpha * +1.00000000 * H2["u,e,m,n"] * T2["n,v,w,f"] * Eta1["w,u"];
    C2["u,e,m,g"] += alpha * +1.00000000 * H2["e,f,m,n"] * T2["n,u,g,f"];
    C2["w,e,m,f"] +=
        alpha * -1.00000000 * H2["u,e,m,v"] * T2["w,x,y,f"] * Eta1["y,u"] * Gamma1["v,x"];
    C2["w,e,m,f"] +=
        alpha * +1.00000000 * H2["u,e,m,v"] * T2["w,x,y,f"] * Eta1["v,x"] * Gamma1["y,u"];
    C2["v,e,m,g"] += alpha * -1.00000000 * H2["e,f,m,u"] * T2["v,w,g,f"] * Gamma1["u,w"];
    C2["u,e,f,g"] += alpha * +0.25000000 * H2["u,e,m,n"] * T2["m,n,f,g"];
    C2["u,e,f,g"] += alpha * +0.50000000 * H2["u,e,m,v"] * T2["m,w,f,g"] * Gamma1["v,w"];
    C2["v,e,f,g"] += alpha * -1.00000000 * H2["u,e,m,f"] * T2["m,v,w,g"] * Eta1["w,u"];
    C2["u,e,g,h"] += alpha * -1.00000000 * H2["e,f,m,g"] * T2["m,u,h,f"];
    C2["u,e,f,g"] +=
        alpha * -0.25000000 * H2["u,e,v,w"] * T2["x,y,f,g"] * Eta1["w,y"] * Eta1["v,x"];
    C2["u,e,f,g"] +=
        alpha * +0.25000000 * H2["u,e,v,w"] * T2["x,y,f,g"] * Gamma1["w,y"] * Gamma1["v,x"];
    C2["w,e,f,g"] +=
        alpha * +1.00000000 * H2["u,e,v,f"] * T2["w,x,y,g"] * Eta1["y,u"] * Gamma1["v,x"];
    C2["w,e,f,g"] +=
        alpha * -1.00000000 * H2["u,e,v,f"] * T2["w,x,y,g"] * Eta1["v,x"] * Gamma1["y,u"];
    C2["v,e,g,h"] += alpha * +1.00000000 * H2["e,f,u,g"] * T2["v,w,h,f"] * Gamma1["u,w"];
    C2["m,u,v,w"] += alpha * +0.25000000 * H2["m,u,n,c0"] * T2["n,c0,v,w"];
    C2["m,w,v,x"] += alpha * +1.00000000 * H2["m,u,n,v"] * T2["n,w,x,y"] * Eta1["y,u"];
    C2["m,v,u,w"] += alpha * +1.00000000 * H2["m,e,n,u"] * T2["n,v,w,e"];
    C2["n,u,w,x"] += alpha * +1.00000000 * H2["u,v,m,w"] * T2["n,m,x,y"] * Eta1["y,v"];
    C2["n,u,v,w"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["n,m,w,e"];
    C2["m,u,x,y"] += alpha * +0.50000000 * H2["m,u,n,v"] * T2["n,w,x,y"] * Gamma1["v,w"];
    C2["m,y,w,x"] +=
        alpha * +0.25000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["a0,v"] * Eta1["z,u"];
    C2["m,y,w,x"] +=
        alpha * -0.25000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Gamma1["a0,v"] * Gamma1["z,u"];
    C2["m,x,v,w"] += alpha * +0.50000000 * H2["u,e,v,w"] * T2["m,x,y,e"] * Eta1["y,u"];
    C2["m,w,u,v"] += alpha * +0.25000000 * H2["e,f,u,v"] * T2["m,w,e,f"];
    C2["m,w,u,y"] += alpha * +1.00000000 * H2["m,e,u,v"] * T2["w,x,y,e"] * Gamma1["v,x"];
    C2["m,u,w,z"] +=
        alpha * -1.00000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["a0,v"] * Gamma1["x,y"];
    C2["m,u,w,z"] +=
        alpha * +1.00000000 * H2["u,v,w,x"] * T2["m,y,z,a0"] * Eta1["x,y"] * Gamma1["a0,v"];
    C2["m,u,v,y"] += alpha * -1.00000000 * H2["u,e,v,w"] * T2["m,x,y,e"] * Gamma1["w,x"];
    C2["m,v,n,w"] += alpha * -1.00000000 * H2["m,u,n,c0"] * T2["c0,v,w,x"] * Eta1["x,u"];
    C2["m,u,n,v"] += alpha * -1.00000000 * H2["m,e,n,c0"] * T2["c0,u,v,e"];
    C2["c0,u,m,w"] += alpha * -1.00000000 * H2["u,v,m,n"] * T2["c0,n,w,x"] * Eta1["x,v"];
    C2["c0,u,m,v"] += alpha * -1.00000000 * H2["u,e,m,n"] * T2["c0,n,v,e"];
    C2["n,x,m,w"] +=
        alpha * +0.50000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Eta1["z,v"] * Eta1["y,u"];
    C2["n,x,m,w"] +=
        alpha * -0.50000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Gamma1["z,v"] * Gamma1["y,u"];
    C2["n,w,m,v"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["n,w,x,e"] * Eta1["x,u"];
    C2["n,v,m,u"] += alpha * +0.50000000 * H2["e,f,m,u"] * T2["n,v,e,f"];
    C2["m,v,n,x"] += alpha * +1.00000000 * H2["m,e,n,u"] * T2["v,w,x,e"] * Gamma1["u,w"];
    C2["n,u,m,y"] +=
        alpha * -1.00000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Eta1["z,v"] * Gamma1["w,x"];
    C2["n,u,m,y"] +=
        alpha * +1.00000000 * H2["u,v,m,w"] * T2["n,x,y,z"] * Eta1["w,x"] * Gamma1["z,v"];
    C2["n,u,m,x"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["n,w,x,e"] * Gamma1["v,w"];
    C2["c0,w,m,n"] +=
        alpha * +0.25000000 * H2["u,v,m,n"] * T2["c0,w,x,y"] * Eta1["y,v"] * Eta1["x,u"];
    C2["c0,w,m,n"] +=
        alpha * -0.25000000 * H2["u,v,m,n"] * T2["c0,w,x,y"] * Gamma1["y,v"] * Gamma1["x,u"];
    C2["c0,v,m,n"] += alpha * +0.50000000 * H2["u,e,m,n"] * T2["c0,v,w,e"] * Eta1["w,u"];
    C2["c0,u,m,n"] += alpha * +0.25000000 * H2["e,f,m,n"] * T2["c0,u,e,f"];
    C2["m,u,v,e"] += alpha * +0.50000000 * H2["m,u,n,c0"] * T2["n,c0,v,e"];
    C2["m,w,v,e"] += alpha * -1.00000000 * H2["m,u,n,v"] * T2["n,w,x,e"] * Eta1["x,u"];
    C2["m,v,u,f"] += alpha * +1.00000000 * H2["m,e,n,u"] * T2["n,v,f,e"];
    C2["n,u,w,e"] += alpha * -1.00000000 * H2["u,v,m,w"] * T2["n,m,x,e"] * Eta1["x,v"];
    C2["n,u,v,f"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["n,m,f,e"];
    C2["m,u,x,e"] += alpha * +1.00000000 * H2["m,u,n,v"] * T2["n,w,x,e"] * Gamma1["v,w"];
    C2["m,v,w,e"] += alpha * -1.00000000 * H2["m,u,n,e"] * T2["n,v,w,x"] * Eta1["x,u"];
    C2["m,u,v,f"] += alpha * -1.00000000 * H2["m,e,n,f"] * T2["n,u,v,e"];
    C2["n,u,w,e"] += alpha * -1.00000000 * H2["u,v,m,e"] * T2["n,m,w,x"] * Eta1["x,v"];
    C2["n,u,v,f"] += alpha * -1.00000000 * H2["u,e,m,f"] * T2["n,m,v,e"];
    C2["m,x,v,e"] +=
        alpha * -1.00000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Eta1["z,u"] * Gamma1["w,y"];
    C2["m,x,v,e"] +=
        alpha * +1.00000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Eta1["w,y"] * Gamma1["z,u"];
    C2["m,w,u,f"] += alpha * +1.00000000 * H2["m,e,u,v"] * T2["w,x,f,e"] * Gamma1["v,x"];
    C2["m,u,w,e"] +=
        alpha * +1.00000000 * H2["u,v,w,x"] * T2["m,y,z,e"] * Eta1["z,v"] * Gamma1["x,y"];
    C2["m,u,w,e"] +=
        alpha * -1.00000000 * H2["u,v,w,x"] * T2["m,y,z,e"] * Eta1["x,y"] * Gamma1["z,v"];
    C2["m,u,v,f"] += alpha * -1.00000000 * H2["u,e,v,w"] * T2["m,x,f,e"] * Gamma1["w,x"];
    C2["m,x,w,e"] +=
        alpha * +0.50000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Eta1["z,v"] * Eta1["y,u"];
    C2["m,x,w,e"] +=
        alpha * -0.50000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Gamma1["z,v"] * Gamma1["y,u"];
    C2["m,w,v,f"] += alpha * +1.00000000 * H2["u,e,v,f"] * T2["m,w,x,e"] * Eta1["x,u"];
    C2["m,v,u,g"] += alpha * +0.50000000 * H2["e,f,u,g"] * T2["m,v,e,f"];
    C2["m,u,z,e"] +=
        alpha * -0.50000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Eta1["w,y"] * Eta1["v,x"];
    C2["m,u,z,e"] +=
        alpha * +0.50000000 * H2["m,u,v,w"] * T2["x,y,z,e"] * Gamma1["w,y"] * Gamma1["v,x"];
    C2["m,v,x,f"] += alpha * +1.00000000 * H2["m,e,u,f"] * T2["v,w,x,e"] * Gamma1["u,w"];
    C2["m,u,y,e"] +=
        alpha * -1.00000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Eta1["z,v"] * Gamma1["w,x"];
    C2["m,u,y,e"] +=
        alpha * +1.00000000 * H2["u,v,w,e"] * T2["m,x,y,z"] * Eta1["w,x"] * Gamma1["z,v"];
    C2["m,u,x,f"] += alpha * -1.00000000 * H2["u,e,v,f"] * T2["m,w,x,e"] * Gamma1["v,w"];
    C2["m,v,n,e"] += alpha * +1.00000000 * H2["m,u,n,c0"] * T2["c0,v,w,e"] * Eta1["w,u"];
    C2["m,u,n,f"] += alpha * -1.00000000 * H2["m,e,n,c0"] * T2["c0,u,f,e"];
    C2["c0,u,m,e"] += alpha * +1.00000000 * H2["u,v,m,n"] * T2["c0,n,w,e"] * Eta1["w,v"];
    C2["c0,u,m,f"] += alpha * -1.00000000 * H2["u,e,m,n"] * T2["c0,n,f,e"];
    C2["m,w,n,e"] +=
        alpha * -1.00000000 * H2["m,u,n,v"] * T2["w,x,y,e"] * Eta1["y,u"] * Gamma1["v,x"];
    C2["m,w,n,e"] +=
        alpha * +1.00000000 * H2["m,u,n,v"] * T2["w,x,y,e"] * Eta1["v,x"] * Gamma1["y,u"];
    C2["m,v,n,f"] += alpha * +1.00000000 * H2["m,e,n,u"] * T2["v,w,f,e"] * Gamma1["u,w"];
    C2["n,u,m,e"] +=
        alpha * +1.00000000 * H2["u,v,m,w"] * T2["n,x,y,e"] * Eta1["y,v"] * Gamma1["w,x"];
    C2["n,u,m,e"] +=
        alpha * -1.00000000 * H2["u,v,m,w"] * T2["n,x,y,e"] * Eta1["w,x"] * Gamma1["y,v"];
    C2["n,u,m,f"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["n,w,f,e"] * Gamma1["v,w"];
    C2["n,w,m,e"] +=
        alpha * +0.50000000 * H2["u,v,m,e"] * T2["n,w,x,y"] * Eta1["y,v"] * Eta1["x,u"];
    C2["n,w,m,e"] +=
        alpha * -0.50000000 * H2["u,v,m,e"] * T2["n,w,x,y"] * Gamma1["y,v"] * Gamma1["x,u"];
    C2["n,v,m,f"] += alpha * +1.00000000 * H2["u,e,m,f"] * T2["n,v,w,e"] * Eta1["w,u"];
    C2["n,u,m,g"] += alpha * +0.50000000 * H2["e,f,m,g"] * T2["n,u,e,f"];
    C2["m,u,e,f"] += alpha * +0.25000000 * H2["m,u,n,c0"] * T2["n,c0,e,f"];
    C2["m,u,e,f"] += alpha * +0.50000000 * H2["m,u,n,v"] * T2["n,w,e,f"] * Gamma1["v,w"];
    C2["m,v,e,f"] += alpha * -1.00000000 * H2["m,u,n,e"] * T2["n,v,w,f"] * Eta1["w,u"];
    C2["m,u,f,g"] += alpha * +1.00000000 * H2["m,e,n,f"] * T2["n,u,g,e"];
    C2["n,u,e,f"] += alpha * -1.00000000 * H2["u,v,m,e"] * T2["n,m,w,f"] * Eta1["w,v"];
    C2["n,u,f,g"] += alpha * +1.00000000 * H2["u,e,m,f"] * T2["n,m,g,e"];
    C2["m,u,e,f"] +=
        alpha * -0.25000000 * H2["m,u,v,w"] * T2["x,y,e,f"] * Eta1["w,y"] * Eta1["v,x"];
    C2["m,u,e,f"] +=
        alpha * +0.25000000 * H2["m,u,v,w"] * T2["x,y,e,f"] * Gamma1["w,y"] * Gamma1["v,x"];
    C2["m,w,e,f"] +=
        alpha * +1.00000000 * H2["m,u,v,e"] * T2["w,x,y,f"] * Eta1["y,u"] * Gamma1["v,x"];
    C2["m,w,e,f"] +=
        alpha * -1.00000000 * H2["m,u,v,e"] * T2["w,x,y,f"] * Eta1["v,x"] * Gamma1["y,u"];
    C2["m,v,f,g"] += alpha * -1.00000000 * H2["m,e,u,f"] * T2["v,w,g,e"] * Gamma1["u,w"];
    C2["m,u,e,f"] +=
        alpha * -1.00000000 * H2["u,v,w,e"] * T2["m,x,y,f"] * Eta1["y,v"] * Gamma1["w,x"];
    C2["m,u,e,f"] +=
        alpha * +1.00000000 * H2["u,v,w,e"] * T2["m,x,y,f"] * Eta1["w,x"] * Gamma1["y,v"];
    C2["m,u,f,g"] += alpha * +1.00000000 * H2["u,e,v,f"] * T2["m,w,g,e"] * Gamma1["v,w"];
    C2["m,w,e,f"] +=
        alpha * +0.25000000 * H2["u,v,e,f"] * T2["m,w,x,y"] * Eta1["y,v"] * Eta1["x,u"];
    C2["m,w,e,f"] +=
        alpha * -0.25000000 * H2["u,v,e,f"] * T2["m,w,x,y"] * Gamma1["y,v"] * Gamma1["x,u"];
    C2["m,v,f,g"] += alpha * +0.50000000 * H2["u,e,f,g"] * T2["m,v,w,e"] * Eta1["w,u"];
    C2["m,u,g,h"] += alpha * +0.25000000 * H2["e,f,g,h"] * T2["m,u,e,f"];
    C2["m,n,u,v"] += alpha * +0.12500000 * H2["m,n,c0,c1"] * T2["c0,c1,u,v"];
    C2["m,c0,v,w"] += alpha * -1.00000000 * H2["m,u,n,v"] * T2["c0,n,w,x"] * Eta1["x,u"];
    C2["m,c0,u,v"] += alpha * -1.00000000 * H2["m,e,n,u"] * T2["c0,n,v,e"];
    C2["m,n,w,x"] += alpha * +0.25000000 * H2["m,n,c0,u"] * T2["c0,v,w,x"] * Gamma1["u,v"];
    C2["m,n,w,x"] +=
        alpha * +0.12500000 * H2["u,v,w,x"] * T2["m,n,y,z"] * Eta1["z,v"] * Eta1["y,u"];
    C2["m,n,w,x"] +=
        alpha * -0.12500000 * H2["u,v,w,x"] * T2["m,n,y,z"] * Gamma1["z,v"] * Gamma1["y,u"];
    C2["m,n,v,w"] += alpha * +0.25000000 * H2["u,e,v,w"] * T2["m,n,x,e"] * Eta1["x,u"];
    C2["m,n,u,v"] += alpha * +0.12500000 * H2["e,f,u,v"] * T2["m,n,e,f"];
    C2["m,n,v,y"] +=
        alpha * +1.00000000 * H2["m,u,v,w"] * T2["n,x,y,z"] * Eta1["z,u"] * Gamma1["w,x"];
    C2["m,n,v,y"] +=
        alpha * -1.00000000 * H2["m,u,v,w"] * T2["n,x,y,z"] * Eta1["w,x"] * Gamma1["z,u"];
    C2["m,n,u,x"] += alpha * +1.00000000 * H2["m,e,u,v"] * T2["n,w,x,e"] * Gamma1["v,w"];
    C2["m,c1,n,v"] += alpha * +1.00000000 * H2["m,u,n,c0"] * T2["c1,c0,v,w"] * Eta1["w,u"];
    C2["m,c1,n,u"] += alpha * +1.00000000 * H2["m,e,n,c0"] * T2["c1,c0,u,e"];
    C2["n,c0,m,w"] +=
        alpha * +0.25000000 * H2["u,v,m,w"] * T2["n,c0,x,y"] * Eta1["y,v"] * Eta1["x,u"];
    C2["n,c0,m,w"] +=
        alpha * -0.25000000 * H2["u,v,m,w"] * T2["n,c0,x,y"] * Gamma1["y,v"] * Gamma1["x,u"];
    C2["n,c0,m,v"] += alpha * +0.50000000 * H2["u,e,m,v"] * T2["n,c0,w,e"] * Eta1["w,u"];
    C2["n,c0,m,u"] += alpha * +0.25000000 * H2["e,f,m,u"] * T2["n,c0,e,f"];
    C2["m,c0,n,x"] +=
        alpha * +1.00000000 * H2["m,u,n,v"] * T2["c0,w,x,y"] * Eta1["y,u"] * Gamma1["v,w"];
    C2["m,c0,n,x"] +=
        alpha * -1.00000000 * H2["m,u,n,v"] * T2["c0,w,x,y"] * Eta1["v,w"] * Gamma1["y,u"];
    C2["m,c0,n,w"] += alpha * +1.00000000 * H2["m,e,n,u"] * T2["c0,v,w,e"] * Gamma1["u,v"];
    C2["c0,c1,m,n"] +=
        alpha * +0.12500000 * H2["u,v,m,n"] * T2["c0,c1,w,x"] * Eta1["x,v"] * Eta1["w,u"];
    C2["c0,c1,m,n"] +=
        alpha * -0.12500000 * H2["u,v,m,n"] * T2["c0,c1,w,x"] * Gamma1["x,v"] * Gamma1["w,u"];
    C2["c0,c1,m,n"] += alpha * +0.25000000 * H2["u,e,m,n"] * T2["c0,c1,v,e"] * Eta1["v,u"];
    C2["c0,c1,m,n"] += alpha * +0.12500000 * H2["e,f,m,n"] * T2["c0,c1,e,f"];
    C2["m,n,u,e"] += alpha * +0.25000000 * H2["m,n,c0,c1"] * T2["c0,c1,u,e"];
    C2["m,c0,v,e"] += alpha * +1.00000000 * H2["m,u,n,v"] * T2["c0,n,w,e"] * Eta1["w,u"];
    C2["m,c0,u,f"] += alpha * -1.00000000 * H2["m,e,n,u"] * T2["c0,n,f,e"];
    C2["m,n,w,e"] += alpha * +0.50000000 * H2["m,n,c0,u"] * T2["c0,v,w,e"] * Gamma1["u,v"];
    C2["m,c0,v,e"] += alpha * +1.00000000 * H2["m,u,n,e"] * T2["c0,n,v,w"] * Eta1["w,u"];
    C2["m,c0,u,f"] += alpha * +1.00000000 * H2["m,e,n,f"] * T2["c0,n,u,e"];
    C2["m,n,v,e"] +=
        alpha * -1.00000000 * H2["m,u,v,w"] * T2["n,x,y,e"] * Eta1["y,u"] * Gamma1["w,x"];
    C2["m,n,v,e"] +=
        alpha * +1.00000000 * H2["m,u,v,w"] * T2["n,x,y,e"] * Eta1["w,x"] * Gamma1["y,u"];
    C2["m,n,u,f"] += alpha * +1.00000000 * H2["m,e,u,v"] * T2["n,w,f,e"] * Gamma1["v,w"];
    C2["m,n,w,e"] +=
        alpha * +0.25000000 * H2["u,v,w,e"] * T2["m,n,x,y"] * Eta1["y,v"] * Eta1["x,u"];
    C2["m,n,w,e"] +=
        alpha * -0.25000000 * H2["u,v,w,e"] * T2["m,n,x,y"] * Gamma1["y,v"] * Gamma1["x,u"];
    C2["m,n,v,f"] += alpha * +0.50000000 * H2["u,e,v,f"] * T2["m,n,w,e"] * Eta1["w,u"];
    C2["m,n,u,g"] += alpha * +0.25000000 * H2["e,f,u,g"] * T2["m,n,e,f"];
    C2["m,n,y,e"] +=
        alpha * -0.25000000 * H2["m,n,u,v"] * T2["w,x,y,e"] * Eta1["v,x"] * Eta1["u,w"];
    C2["m,n,y,e"] +=
        alpha * +0.25000000 * H2["m,n,u,v"] * T2["w,x,y,e"] * Gamma1["v,x"] * Gamma1["u,w"];
    C2["m,n,x,e"] +=
        alpha * +1.00000000 * H2["m,u,v,e"] * T2["n,w,x,y"] * Eta1["y,u"] * Gamma1["v,w"];
    C2["m,n,x,e"] +=
        alpha * -1.00000000 * H2["m,u,v,e"] * T2["n,w,x,y"] * Eta1["v,w"] * Gamma1["y,u"];
    C2["m,n,w,f"] += alpha * +1.00000000 * H2["m,e,u,f"] * T2["n,v,w,e"] * Gamma1["u,v"];
    C2["m,c1,n,e"] += alpha * -1.00000000 * H2["m,u,n,c0"] * T2["c1,c0,v,e"] * Eta1["v,u"];
    C2["m,c1,n,f"] += alpha * +1.00000000 * H2["m,e,n,c0"] * T2["c1,c0,f,e"];
    C2["m,c0,n,e"] +=
        alpha * -1.00000000 * H2["m,u,n,v"] * T2["c0,w,x,e"] * Eta1["x,u"] * Gamma1["v,w"];
    C2["m,c0,n,e"] +=
        alpha * +1.00000000 * H2["m,u,n,v"] * T2["c0,w,x,e"] * Eta1["v,w"] * Gamma1["x,u"];
    C2["m,c0,n,f"] += alpha * +1.00000000 * H2["m,e,n,u"] * T2["c0,v,f,e"] * Gamma1["u,v"];
    C2["n,c0,m,e"] +=
        alpha * +0.25000000 * H2["u,v,m,e"] * T2["n,c0,w,x"] * Eta1["x,v"] * Eta1["w,u"];
    C2["n,c0,m,e"] +=
        alpha * -0.25000000 * H2["u,v,m,e"] * T2["n,c0,w,x"] * Gamma1["x,v"] * Gamma1["w,u"];
    C2["n,c0,m,f"] += alpha * +0.50000000 * H2["u,e,m,f"] * T2["n,c0,v,e"] * Eta1["v,u"];
    C2["n,c0,m,g"] += alpha * +0.25000000 * H2["e,f,m,g"] * T2["n,c0,e,f"];
    C2["m,n,e,f"] += alpha * +0.12500000 * H2["m,n,c0,c1"] * T2["c0,c1,e,f"];
    C2["m,n,e,f"] += alpha * +0.25000000 * H2["m,n,c0,u"] * T2["c0,v,e,f"] * Gamma1["u,v"];
    C2["m,c0,e,f"] += alpha * +1.00000000 * H2["m,u,n,e"] * T2["c0,n,v,f"] * Eta1["v,u"];
    C2["m,c0,f,g"] += alpha * -1.00000000 * H2["m,e,n,f"] * T2["c0,n,g,e"];
    C2["m,n,e,f"] +=
        alpha * -0.12500000 * H2["m,n,u,v"] * T2["w,x,e,f"] * Eta1["v,x"] * Eta1["u,w"];
    C2["m,n,e,f"] +=
        alpha * +0.12500000 * H2["m,n,u,v"] * T2["w,x,e,f"] * Gamma1["v,x"] * Gamma1["u,w"];
    C2["m,n,e,f"] +=
        alpha * +1.00000000 * H2["m,u,v,e"] * T2["n,w,x,f"] * Eta1["x,u"] * Gamma1["v,w"];
    C2["m,n,e,f"] +=
        alpha * -1.00000000 * H2["m,u,v,e"] * T2["n,w,x,f"] * Eta1["v,w"] * Gamma1["x,u"];
    C2["m,n,f,g"] += alpha * -1.00000000 * H2["m,e,u,f"] * T2["n,v,g,e"] * Gamma1["u,v"];
    C2["m,n,e,f"] +=
        alpha * +0.12500000 * H2["u,v,e,f"] * T2["m,n,w,x"] * Eta1["x,v"] * Eta1["w,u"];
    C2["m,n,e,f"] +=
        alpha * -0.12500000 * H2["u,v,e,f"] * T2["m,n,w,x"] * Gamma1["x,v"] * Gamma1["w,u"];
    C2["m,n,f,g"] += alpha * +0.25000000 * H2["u,e,f,g"] * T2["m,n,v,e"] * Eta1["v,u"];
    C2["m,n,g,h"] += alpha * +0.12500000 * H2["e,f,g,h"] * T2["m,n,e,f"];
    C2["m,e,u,v"] += alpha * +0.25000000 * H2["m,e,n,c0"] * T2["n,c0,u,v"];
    C2["n,e,v,w"] += alpha * -1.00000000 * H2["u,e,m,v"] * T2["n,m,w,x"] * Eta1["x,u"];
    C2["n,e,u,v"] += alpha * +1.00000000 * H2["e,f,m,u"] * T2["n,m,v,f"];
    C2["m,e,w,x"] += alpha * +0.50000000 * H2["m,e,n,u"] * T2["n,v,w,x"] * Gamma1["u,v"];
    C2["m,e,v,y"] +=
        alpha * +1.00000000 * H2["u,e,v,w"] * T2["m,x,y,z"] * Eta1["z,u"] * Gamma1["w,x"];
    C2["m,e,v,y"] +=
        alpha * -1.00000000 * H2["u,e,v,w"] * T2["m,x,y,z"] * Eta1["w,x"] * Gamma1["z,u"];
    C2["m,e,u,x"] += alpha * -1.00000000 * H2["e,f,u,v"] * T2["m,w,x,f"] * Gamma1["v,w"];
    C2["c0,e,m,v"] += alpha * +1.00000000 * H2["u,e,m,n"] * T2["c0,n,v,w"] * Eta1["w,u"];
    C2["c0,e,m,u"] += alpha * -1.00000000 * H2["e,f,m,n"] * T2["c0,n,u,f"];
    C2["n,e,m,x"] +=
        alpha * +1.00000000 * H2["u,e,m,v"] * T2["n,w,x,y"] * Eta1["y,u"] * Gamma1["v,w"];
    C2["n,e,m,x"] +=
        alpha * -1.00000000 * H2["u,e,m,v"] * T2["n,w,x,y"] * Eta1["v,w"] * Gamma1["y,u"];
    C2["n,e,m,w"] += alpha * -1.00000000 * H2["e,f,m,u"] * T2["n,v,w,f"] * Gamma1["u,v"];
    C2["m,e,u,f"] += alpha * +0.50000000 * H2["m,e,n,c0"] * T2["n,c0,u,f"];
    C2["n,e,v,f"] += alpha * +1.00000000 * H2["u,e,m,v"] * T2["n,m,w,f"] * Eta1["w,u"];
    C2["n,e,u,g"] += alpha * +1.00000000 * H2["e,f,m,u"] * T2["n,m,g,f"];
    C2["m,e,w,f"] += alpha * +1.00000000 * H2["m,e,n,u"] * T2["n,v,w,f"] * Gamma1["u,v"];
    C2["n,e,v,f"] += alpha * +1.00000000 * H2["u,e,m,f"] * T2["n,m,v,w"] * Eta1["w,u"];
    C2["n,e,u,g"] += alpha * -1.00000000 * H2["e,f,m,g"] * T2["n,m,u,f"];
    C2["m,e,v,f"] +=
        alpha * -1.00000000 * H2["u,e,v,w"] * T2["m,x,y,f"] * Eta1["y,u"] * Gamma1["w,x"];
    C2["m,e,v,f"] +=
        alpha * +1.00000000 * H2["u,e,v,w"] * T2["m,x,y,f"] * Eta1["w,x"] * Gamma1["y,u"];
    C2["m,e,u,g"] += alpha * -1.00000000 * H2["e,f,u,v"] * T2["m,w,g,f"] * Gamma1["v,w"];
    C2["m,e,y,f"] +=
        alpha * -0.50000000 * H2["m,e,u,v"] * T2["w,x,y,f"] * Eta1["v,x"] * Eta1["u,w"];
    C2["m,e,y,f"] +=
        alpha * +0.50000000 * H2["m,e,u,v"] * T2["w,x,y,f"] * Gamma1["v,x"] * Gamma1["u,w"];
    C2["m,e,x,f"] +=
        alpha * +1.00000000 * H2["u,e,v,f"] * T2["m,w,x,y"] * Eta1["y,u"] * Gamma1["v,w"];
    C2["m,e,x,f"] +=
        alpha * -1.00000000 * H2["u,e,v,f"] * T2["m,w,x,y"] * Eta1["v,w"] * Gamma1["y,u"];
    C2["m,e,w,g"] += alpha * -1.00000000 * H2["e,f,u,g"] * T2["m,v,w,f"] * Gamma1["u,v"];
    C2["c0,e,m,f"] += alpha * -1.00000000 * H2["u,e,m,n"] * T2["c0,n,v,f"] * Eta1["v,u"];
    C2["c0,e,m,g"] += alpha * -1.00000000 * H2["e,f,m,n"] * T2["c0,n,g,f"];
    C2["n,e,m,f"] +=
        alpha * -1.00000000 * H2["u,e,m,v"] * T2["n,w,x,f"] * Eta1["x,u"] * Gamma1["v,w"];
    C2["n,e,m,f"] +=
        alpha * +1.00000000 * H2["u,e,m,v"] * T2["n,w,x,f"] * Eta1["v,w"] * Gamma1["x,u"];
    C2["n,e,m,g"] += alpha * -1.00000000 * H2["e,f,m,u"] * T2["n,v,g,f"] * Gamma1["u,v"];
    C2["m,e,f,g"] += alpha * +0.25000000 * H2["m,e,n,c0"] * T2["n,c0,f,g"];
    C2["m,e,f,g"] += alpha * +0.50000000 * H2["m,e,n,u"] * T2["n,v,f,g"] * Gamma1["u,v"];
    C2["n,e,f,g"] += alpha * +1.00000000 * H2["u,e,m,f"] * T2["n,m,v,g"] * Eta1["v,u"];
    C2["n,e,g,h"] += alpha * +1.00000000 * H2["e,f,m,g"] * T2["n,m,h,f"];
    C2["m,e,f,g"] +=
        alpha * -0.25000000 * H2["m,e,u,v"] * T2["w,x,f,g"] * Eta1["v,x"] * Eta1["u,w"];
    C2["m,e,f,g"] +=
        alpha * +0.25000000 * H2["m,e,u,v"] * T2["w,x,f,g"] * Gamma1["v,x"] * Gamma1["u,w"];
    C2["m,e,f,g"] +=
        alpha * +1.00000000 * H2["u,e,v,f"] * T2["m,w,x,g"] * Eta1["x,u"] * Gamma1["v,w"];
    C2["m,e,f,g"] +=
        alpha * -1.00000000 * H2["u,e,v,f"] * T2["m,w,x,g"] * Eta1["v,w"] * Gamma1["x,u"];
    C2["m,e,g,h"] += alpha * +1.00000000 * H2["e,f,u,g"] * T2["m,v,h,f"] * Gamma1["u,v"];
    C2["e,f,u,v"] += alpha * +0.12500000 * H2["e,f,m,n"] * T2["m,n,u,v"];
    C2["e,f,w,x"] += alpha * +0.25000000 * H2["e,f,m,u"] * T2["m,v,w,x"] * Gamma1["u,v"];
    C2["e,f,u,g"] += alpha * +0.25000000 * H2["e,f,m,n"] * T2["m,n,u,g"];
    C2["e,f,w,g"] += alpha * +0.50000000 * H2["e,f,m,u"] * T2["m,v,w,g"] * Gamma1["u,v"];
    C2["e,f,y,g"] +=
        alpha * -0.25000000 * H2["e,f,u,v"] * T2["w,x,y,g"] * Eta1["v,x"] * Eta1["u,w"];
    C2["e,f,y,g"] +=
        alpha * +0.25000000 * H2["e,f,u,v"] * T2["w,x,y,g"] * Gamma1["v,x"] * Gamma1["u,w"];
    C2["e,f,g,h"] += alpha * +0.12500000 * H2["e,f,m,n"] * T2["m,n,g,h"];
    C2["e,f,g,h"] += alpha * +0.25000000 * H2["e,f,m,u"] * T2["m,v,g,h"] * Gamma1["u,v"];
    C2["e,f,g,h"] +=
        alpha * -0.12500000 * H2["e,f,u,v"] * T2["w,x,g,h"] * Eta1["v,x"] * Eta1["u,w"];
    C2["e,f,g,h"] +=
        alpha * +0.12500000 * H2["e,f,u,v"] * T2["w,x,g,h"] * Gamma1["v,x"] * Gamma1["u,w"];

    // if (print_ > 2) {
    // outfile -> Printf("\n    Time for H2_T2_C2 : %12.3f", timer.get());
    //}
}

} // namespace forte