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

void MRDSRG_SO::compute_lambda() {
    Tbar1 = BTF_->build(tensor_type_, "Tbar1", {"cv"});
    Tbar2 = BTF_->build(tensor_type_, "Tbar2", {"ccvv"});

    // lambda equation
    BlockedTensor C1 = BTF_->build(tensor_type_, "C1 Lambda", {"cv"});
    BlockedTensor C2 = BTF_->build(tensor_type_, "C2 Lambda", {"ccvv"});

    Tbar1["ia"] = T1["ia"];
    Tbar2["ijab"] = T2["ijab"];

    // start iteration
    outfile->Printf("\n\n  ==> Lambda Iterations <==\n");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n           Cycle     |L1|    |L2|  |L1 - T1|  |L2 - T2|  max(L1) max(L2) ");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    int maxiter = foptions_->get_int("MAXITER");
    int cycle = 0;
    Tbar1_diff = 0.0;
    Tbar2_diff = 0.0;
    for (int i = 0; i < maxiter; ++i) {
        C1.zero();
        C2.zero();

        compute_lambda_comm1(F, V, Tbar1, Tbar2, C1, C2);
        compute_lambda_comm2(F, V, T1, T2, Tbar1, Tbar2, C1, C2);
        compute_lambda_comm3(F, V, T1, T2, Tbar1, Tbar2, C1, C2);
        compute_lambda_comm4(F, V, T1, T2, Tbar1, Tbar2, C1, C2);
        compute_lambda_comm5(F, V, T1, T2, C1, C2);

        outfile->Printf("\n      @CT %4d %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f", cycle, T1norm,
                        T2norm, Tbar1_diff, Tbar2_diff, T1max, T2max);

        update_lambda(C1, C2);

        double rms = std::max(rms_t1, rms_t2);
        if (rms < foptions_->get_double("R_CONVERGENCE")) {
            break;
        }
        cycle++;
    }

    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    if (cycle == maxiter) {
        outfile->Printf("\n\n\tThe calculation did not converge in %d "
                        "cycles\n\tQuitting.\n",
                        maxiter);
    }
}

void MRDSRG_SO::update_lambda(BlockedTensor& C1, BlockedTensor& C2) {
    BlockedTensor X1 = ambit::BlockedTensor::build(tensor_type_, "X1", {"cv"});
    BlockedTensor X2 = ambit::BlockedTensor::build(tensor_type_, "X2", {"ccvv"});
    X1["ia"] = C1["ia"];
    X2["ijab"] = C2["ijab"];

    X1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double delta = Fd[i[0]] - Fd[i[1]];
        value /= delta;
    });

    X2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double delta = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
        value /= delta;
    });

    T1max = X1.norm(0);
    T2max = X2.norm(0);
    T1norm = X1.norm();
    T2norm = X2.norm();

    C1["ia"] = X1["ia"];
    C2["ijab"] = X2["ijab"];
    C1["ia"] -= Tbar1["ia"];
    C2["ijab"] -= Tbar2["ijab"];
    rms_t1 = C1.norm();
    rms_t2 = C2.norm();

    C1["ia"] = X1["ia"];
    C2["ijab"] = X2["ijab"];
    C1["ia"] -= T1["ia"];
    C2["ijab"] -= T2["ijab"];
    Tbar1_diff = C1.norm();
    Tbar2_diff = C2.norm();

    Tbar1["ia"] = X1["ia"];
    Tbar2["ijab"] = X2["ijab"];
}

void MRDSRG_SO::compute_lambda_comm4(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                     BlockedTensor& T2, BlockedTensor& L1, BlockedTensor& L2,
                                     BlockedTensor& C1, BlockedTensor& C2) {
    compute_lambda_comm4_part1(H1, H2, T1, T2, L1, L2, C1);
    compute_lambda_comm4_part2(H1, H2, T1, T2, L1, L2, C2);
    compute_lambda_comm4_part3(H1, H2, T1, T2, L1, L2, C2);
}

void MRDSRG_SO::compute_lambda_comm1(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& L1,
                                     BlockedTensor& L2, BlockedTensor& C1, BlockedTensor& C2) {
    BlockedTensor temp;

    C1["c0,v0"] += 2.0 * H1["c0,v0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v1,v2,v0,c1"] * L2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * L1["c1,v1"];
    C1["c0,v0"] += 1.0 * H2["c0,c1,v0,v1"] * L1["c1,v1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v1,c0"] * L2["c1,c2,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,v1"] * L1["c1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += 2.0 * H2["c0,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * L2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * L2["c2,c3,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * L1["c2,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["c0,v0"] * L1["c1,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * L2["c1,c2,v1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];
}

void MRDSRG_SO::compute_lambda_comm2(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                     BlockedTensor& T2, BlockedTensor& L1, BlockedTensor& L2,
                                     BlockedTensor& C1, BlockedTensor& C2) {
    C1["c0,v0"] += 2.0 * H1["v1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += 1.0 * H1["c1,v1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += -2.0 * H1["c1,c0"] * T1["c1,v0"];
    C1["c0,v0"] += 1.0 * H2["v1,v2,v0,c1"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -2.0 * H2["v1,c0,v0,c1"] * T1["c1,v1"];
    C1["c0,v0"] += 2.0 * H2["c0,c1,v0,v1"] * T1["c1,v1"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,c0"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += 1.0 * H1["v1,v0"] * L1["c1,v2"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H1["v2,v1"] * L1["c1,v1"] * T2["c0,c1,v0,v2"];
    C1["c0,v0"] += -1.0 * H1["c0,v1"] * L1["c1,v0"] * T1["c1,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c0,v1"] * L1["c1,v1"] * T1["c1,v0"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c0,v1"] * L2["c1,c2,v0,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H1["c0,v1"] * L2["c1,c2,v1,v2"] * T2["c1,c2,v0,v2"];
    C1["c0,v0"] += -1.0 * H1["c1,v0"] * L1["c0,v1"] * T1["c1,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c1,v0"] * L1["c1,v1"] * T1["c0,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c1,v0"] * L2["c0,c2,v1,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H1["c1,v0"] * L2["c1,c2,v1,v2"] * T2["c0,c2,v1,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c1,v1"] * L1["c0,v1"] * T1["c1,v0"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c1,v1"] * L1["c1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,c0"] * L1["c2,v1"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c2,c1"] * L1["c1,v1"] * T2["c0,c2,v0,v1"];
    C1["c0,v0"] += 1.0 * H2["v1,v2,v0,c1"] * L1["c0,v1"] * T1["c1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,v2,v0,c1"] * L1["c1,v1"] * T1["c0,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * L1["c2,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c2,v0,c1"] * L1["c2,v2"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c2,v0,c1"] * L2["c0,c1,v1,v2"] * T1["c2,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * L1["c1,v1"] * T2["c0,c1,v2,v3"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * L2["c0,c1,v2,v3"] * T1["c1,v1"];
    C1["c0,v0"] += -1.0 * H2["v2,c0,v0,v1"] * L1["c1,v1"] * T1["c1,v2"];
    C1["c0,v0"] += -1.0 * H2["v2,c0,v0,v1"] * L1["c1,v2"] * T1["c1,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v2,c0,v0,v1"] * L2["c1,c2,v1,v3"] * T2["c1,c2,v2,v3"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v2,c0,v0,v1"] * L2["c1,c2,v2,v3"] * T2["c1,c2,v1,v3"];
    C1["c0,v0"] += 1.0 * H2["v2,c1,v0,v1"] * L1["c0,v2"] * T1["c1,v1"];
    C1["c0,v0"] += 1.0 * H2["v2,c1,v0,v1"] * L1["c1,v1"] * T1["c0,v2"];
    C1["c0,v0"] += 1.0 * H2["v2,c1,v0,v1"] * L2["c0,c2,v2,v3"] * T2["c1,c2,v1,v3"];
    C1["c0,v0"] += 1.0 * H2["v2,c1,v0,v1"] * L2["c1,c2,v1,v3"] * T2["c0,c2,v2,v3"];
    C1["c0,v0"] += -1.0 * H2["v2,c1,v1,c0"] * L1["c2,v2"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += -1.0 * H2["v2,c1,v1,c0"] * L2["c1,c2,v0,v1"] * T1["c2,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v2,c2,v1,c1"] * L1["c2,v1"] * T2["c0,c1,v0,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["v3,c0,v1,v2"] * L2["c1,c2,v0,v3"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["v3,c0,v1,v2"] * L2["c1,c2,v1,v2"] * T2["c1,c2,v0,v3"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["v3,c1,v1,v2"] * L2["c1,c2,v1,v2"] * T2["c0,c2,v0,v3"];
    C1["c0,v0"] += 1.0 * H2["c0,c1,v0,v1"] * L1["c2,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c0,c1,v1,v2"] * L1["c2,v0"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c0,c1,v1,v2"] * L2["c1,c2,v1,v2"] * T1["c2,v0"];
    C1["c0,v0"] += -1.0 * H2["c0,c2,v0,c1"] * L1["c1,v1"] * T1["c2,v1"];
    C1["c0,v0"] += -1.0 * H2["c0,c2,v0,c1"] * L1["c2,v1"] * T1["c1,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["c0,c2,v0,c1"] * L2["c1,c3,v1,v2"] * T2["c2,c3,v1,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["c0,c2,v0,c1"] * L2["c2,c3,v1,v2"] * T2["c1,c3,v1,v2"];
    C1["c0,v0"] += 1.0 * H2["c0,c2,v1,c1"] * L1["c1,v0"] * T1["c2,v1"];
    C1["c0,v0"] += 1.0 * H2["c0,c2,v1,c1"] * L1["c2,v1"] * T1["c1,v0"];
    C1["c0,v0"] += 1.0 * H2["c0,c2,v1,c1"] * L2["c1,c3,v0,v2"] * T2["c2,c3,v1,v2"];
    C1["c0,v0"] += 1.0 * H2["c0,c2,v1,c1"] * L2["c2,c3,v1,v2"] * T2["c1,c3,v0,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v0,v1"] * L1["c0,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v0,v1"] * L2["c1,c2,v1,v2"] * T1["c0,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["c1,c2,v1,v2"] * L1["c0,v1"] * T2["c1,c2,v0,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["c1,c2,v1,v2"] * L1["c1,v0"] * T2["c0,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v1,v2"] * L1["c1,v1"] * T2["c0,c2,v0,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v1,v2"] * L2["c0,c1,v1,v2"] * T1["c2,v0"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v1,v2"] * L2["c1,c2,v0,v1"] * T1["c0,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,c0"] * L1["c1,v0"] * T1["c2,v1"];
    C1["c0,v0"] += -1.0 * H2["c1,c2,v1,c0"] * L1["c1,v1"] * T1["c2,v0"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c2,c3,v0,c1"] * L2["c0,c1,v1,v2"] * T2["c2,c3,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c2,c3,v0,c1"] * L2["c2,c3,v1,v2"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["c2,c3,v1,c1"] * L2["c2,c3,v1,v2"] * T2["c0,c1,v0,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * L1["c1,v1"] * T2["c2,c3,v0,v1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * L2["c2,c3,v0,v1"] * T1["c1,v1"];

    C2["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,c0,c1"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c2,v0,v1"] * L1["c2,v3"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c2,v0,v1"] * L2["c0,c1,v2,v3"] * T1["c2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["c0,c1,v2,v3"] * L2["c2,c3,v0,v1"] * T2["c2,c3,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 8.0) * H2["c0,c1,v2,v3"] * L2["c2,c3,v2,v3"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v2,c2"] * L1["c3,v2"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v2,c2"] * L2["c2,c3,v0,v1"] * T1["c3,v2"];
    C2["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["c2,c3,v0,v1"] * L2["c0,c1,v2,v3"] * T2["c2,c3,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 8.0) * H2["c2,c3,v0,v1"] * L2["c2,c3,v2,v3"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 8.0) * H2["c2,c3,v2,v3"] * L2["c0,c1,v2,v3"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += (1.0 / 8.0) * H2["c2,c3,v2,v3"] * L2["c2,c3,v0,v1"] * T2["c0,c1,v2,v3"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -2.0 * H1["v2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += -2.0 * H2["c0,c1,v0,c2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c2,v0"] * L1["c2,v2"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,v0"] * L2["c0,c1,v1,v2"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c2,v2"] * L1["c2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H1["c2,v2"] * L2["c0,c1,v0,v2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * L1["c2,v1"] * T2["c0,c1,v2,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * L1["c2,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v0,c2"] * L2["c0,c1,v1,v2"] * T1["c2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * L2["c0,c1,v2,v3"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v3,c2,v0,v2"] * L1["c2,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v3,c2,v0,v2"] * L2["c0,c1,v1,v3"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,v2"] * L1["c2,v1"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c1,v0,v2"] * L1["c2,v2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c1,v0,v2"] * L2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c0,c1,v0,v2"] * L2["c2,c3,v2,v3"] * T2["c2,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * L1["c3,v2"] * T2["c2,c3,v1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v0,v2"] * L2["c0,c1,v1,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v0,v2"] * L2["c2,c3,v2,v3"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * L2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * L2["c2,c3,v0,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,c0,c1"] * L1["c2,v0"] * T1["c3,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["c0,v0"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += -2.0 * H2["v2,c0,v0,c2"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H1["v2,v0"] * L1["c0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["v2,v0"] * L1["c0,v2"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c0,v0"] * L1["c2,v2"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c2,v2"] * L1["c0,v0"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,c0"] * L1["c1,v0"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H1["c2,c0"] * L1["c2,v0"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * L1["c0,v1"] * T2["c1,c2,v2,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * L1["c0,v2"] * T2["c1,c2,v1,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * L2["c0,c2,v1,v2"] * T1["c1,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["v2,v3,v0,c2"] * L2["c0,c2,v2,v3"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * L1["c1,v1"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,c0,v0,c2"] * L1["c1,v2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,c0,v0,c2"] * L1["c2,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,c0,v0,c2"] * L1["c2,v2"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v3,c0,v0,v2"] * L1["c2,v2"] * T2["c1,c2,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v3,c0,v0,v2"] * L2["c1,c2,v1,v3"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c0,c2,v0,v2"] * L1["c1,v1"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c0,c2,v0,v2"] * L1["c2,v2"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c0,c2,v0,v2"] * L2["c1,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c0,c2,v0,v2"] * L2["c2,c3,v2,v3"] * T2["c1,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c3,v0,c2"] * L1["c3,v2"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c3,v0,c2"] * L2["c1,c2,v1,v2"] * T1["c3,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,v2,v3"] * L2["c0,c2,v0,v2"] * T2["c1,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * L1["c1,v0"] * T2["c2,c3,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * L1["c2,v0"] * T2["c1,c3,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * L2["c1,c2,v0,v2"] * T1["c3,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["c2,c3,v2,c0"] * L2["c2,c3,v0,v2"] * T1["c1,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 2.0 * H1["c2,c0"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += -2.0 * H2["v2,c0,v0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c0,v2"] * L1["c2,v2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["c0,v2"] * L2["c1,c2,v0,v1"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c2,v2"] * L1["c0,v2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H1["c2,v2"] * L2["c0,c2,v0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,v1"] * L1["c0,v2"] * T1["c1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,v1"] * L1["c2,v3"] * T2["c1,c2,v2,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c2,v0,v1"] * L1["c1,v2"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c2,v0,v1"] * L1["c2,v2"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c2,v0,v1"] * L2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c0,c2,v0,v1"] * L2["c2,c3,v2,v3"] * T2["c1,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c2,v2,v3"] * L2["c1,c3,v0,v1"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c0,c2,v2,v3"] * L2["c2,c3,v2,v3"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c3,v2,c2"] * L1["c3,v2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c3,v2,c2"] * L2["c1,c2,v0,v1"] * T1["c3,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * L2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * L2["c0,c2,v2,v3"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * L1["c1,v2"] * T2["c2,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * L1["c2,v2"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c2,c3,v2,c0"] * L2["c1,c2,v0,v1"] * T1["c3,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * L2["c2,c3,v0,v1"] * T1["c1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
}

} // namespace forte
