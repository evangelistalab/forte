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

std::vector<double> MRDSRG_SO::E4th_correction(){
    // determine 2nd-order T3
    compute_2nd_order_t3();

    // compute T3 contribution
    double e1 = E4th_correction_t3();

    // determine 3rd-order Hbar1 and Hbar2
    BlockedTensor C1;
    BlockedTensor C2;
    compute_3rd_order_hbar(C1, C2);

    // compute 3rd-order T2 contribution
    double e2 = E4th_correction_t2(C1, C2);

    // compute lambda contribution (T variant)
    double e3 = E4th_correction_lambda_1(C1, C2);
    double e4 = E4th_correction_lambda_2(C1, C2);
    double e5 = 0.0;
    if (foptions_->get_bool("DSRG_LAMBDA_FINDIFF")) {
        e5 = E4th_correction_lambda(C1, C2);
    }

    return {e1, e2, e3, e4, e5};
}

void MRDSRG_SO::compute_2nd_order_t3(){
    T3 = ambit::BlockedTensor::build(tensor_type_, "T3", {"cccvvv"});

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * V["c0,c1,v0,c3"] * T2["c2,c3,v1,v2"];
    T3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    T3["c0,c1,c2,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    T3["c0,c1,c2,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];
    T3["c0,c2,c1,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    T3["c0,c2,c1,v1,v0,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    T3["c0,c2,c1,v1,v2,v0"] -= temp["c0,c1,c2,v0,v1,v2"];
    T3["c2,c0,c1,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    T3["c2,c0,c1,v1,v0,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    T3["c2,c0,c1,v1,v2,v0"] += temp["c0,c1,c2,v0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccvvv"});
    temp["c0,c1,c2,v0,v1,v2"] += 1.0 * V["v3,c0,v0,v1"] * T2["c1,c2,v2,v3"];
    T3["c0,c1,c2,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    T3["c0,c1,c2,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    T3["c0,c1,c2,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];
    T3["c1,c0,c2,v0,v1,v2"] -= temp["c0,c1,c2,v0,v1,v2"];
    T3["c1,c0,c2,v0,v2,v1"] += temp["c0,c1,c2,v0,v1,v2"];
    T3["c1,c0,c2,v2,v0,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    T3["c1,c2,c0,v0,v1,v2"] += temp["c0,c1,c2,v0,v1,v2"];
    T3["c1,c2,c0,v0,v2,v1"] -= temp["c0,c1,c2,v0,v1,v2"];
    T3["c1,c2,c0,v2,v0,v1"] += temp["c0,c1,c2,v0,v1,v2"];

    T3.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] + Fd[i[2]] - Fd[i[3]] - Fd[i[4]] -
                Fd[i[5]]);
    });
}

double MRDSRG_SO::E4th_correction_t2(BlockedTensor& C1, BlockedTensor& C2){
    double C0 = 0.0;

    auto X1 = ambit::BlockedTensor::build(tensor_type_, "X1", {"cv"});
    X1["ia"] = C1["ia"];
    X1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] - Fd[i[1]]);
    });

    auto X2 = ambit::BlockedTensor::build(tensor_type_, "X2", {"ccvv"});
    X2["ijab"] = C2["ijab"];
    X2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]]);
    });

    C0 += 2.0 * F["c0,v0"] * X1["c0,v0"];
    C0 += (1.0 / 2.0) * V["c0,c1,v0,v1"] * X2["c0,c1,v0,v1"];

    C0 += 2.0 * F["v1,v0"] * T1["c0,v0"] * X1["c0,v1"];
    C0 += 1.0 * F["v1,v0"] * T2["c0,c1,v0,v2"] * X2["c0,c1,v1,v2"];
    C0 += -2.0 * F["c1,c0"] * T1["c0,v0"] * X1["c1,v0"];
    C0 += -1.0 * F["c1,c0"] * T2["c0,c2,v0,v1"] * X2["c1,c2,v0,v1"];

    return C0;
}

double MRDSRG_SO::E4th_correction_t3(){
    double C0 = 0.0;

    C0 += (1.0 / 12.0) * F["v1,v0"] * T3["c0,c1,c2,v0,v2,v3"] * T3["c0,c1,c2,v1,v2,v3"];
    C0 += (-1.0 / 12.0) * F["c1,c0"] * T3["c0,c2,c3,v0,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];

    C0 += (1.0 / 4.0) * F["c0,v0"] * T2["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C0 += (-1.0 / 2.0) * V["v2,c0,v0,v1"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v0,v1,v3"];
    C0 += (1.0 / 4.0) * V["c0,c1,v0,v1"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C0 += (-1.0 / 2.0) * V["c1,c2,v0,c0"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];

    C0 += (1.0 / 4.0) * F["v1,v0"] * T1["c0,v0"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v1,v2,v3"];
    C0 += (-1.0 / 2.0) * F["v1,v0"] * T1["c0,v2"] * T2["c1,c2,v0,v3"] * T3["c0,c1,c2,v1,v2,v3"];
    C0 += (-1.0 / 4.0) * F["c1,c0"] * T1["c0,v0"] * T2["c2,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
    C0 += (1.0 / 2.0) * F["c1,c0"] * T1["c2,v0"] * T2["c0,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];

    return C0;
}

void MRDSRG_SO::compute_3rd_order_hbar(BlockedTensor& C1, BlockedTensor& C2){
    C1 = ambit::BlockedTensor::build(tensor_type_, "C1", {"cv"});
    C2 = ambit::BlockedTensor::build(tensor_type_, "C2", {"ccvv"});

    C1["c0,v0"] += (1.0 / 4.0) * V["c1,c2,v1,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C1["c0,v0"] += (1.0 / 8.0) * F["v1,v0"] * T2["c1,c2,v2,v3"] * T3["c0,c1,c2,v1,v2,v3"];
    C1["c0,v0"] += (1.0 / 2.0) * F["v2,v1"] * T2["c1,c2,v1,v3"] * T3["c0,c1,c2,v0,v2,v3"];
    C1["c0,v0"] += (-1.0 / 8.0) * F["c1,c0"] * T2["c2,c3,v1,v2"] * T3["c1,c2,c3,v0,v1,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * F["c2,c1"] * T2["c1,c3,v1,v2"] * T3["c0,c2,c3,v0,v1,v2"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * V["c2,c3,v2,c0"] * T3["c1,c2,c3,v0,v1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * F["c2,c0"] * T1["c3,v2"] * T3["c1,c2,c3,v0,v1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * V["v2,v3,v0,c2"] * T3["c0,c1,c2,v1,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * F["v2,v0"] * T1["c2,v3"] * T3["c0,c1,c2,v1,v2,v3"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += 1.0 * F["c2,v2"] * T3["c0,c1,c2,v0,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * F["v3,v2"] * T1["c2,v2"] * T3["c0,c1,c2,v0,v1,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * F["c3,c2"] * T1["c2,v2"] * T3["c0,c1,c3,v0,v1,v2"];
}

double MRDSRG_SO::E4th_correction_lambda(BlockedTensor& C1, BlockedTensor& C2){
    double C0 = 0.0;
    Tbar1 = BTF_->build(tensor_type_, "Tbar1", {"cv"});
    Tbar2 = BTF_->build(tensor_type_, "Tbar2", {"ccvv"});

    compute_lambda();

    C1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double D = Fd[i[0]] - Fd[i[1]];
        value *= 1.0 - std::exp(-s_ * D * D);
    });

    C2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double D = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
        value *= 1.0 - std::exp(-s_ * D * D);
    });

    C0 += C1["ia"] * Tbar1["ia"];
    C0 += 0.25 * C2["ijab"] * Tbar2["ijab"];

    return C0;
}

double MRDSRG_SO::E4th_correction_lambda_1(BlockedTensor& C1, BlockedTensor& C2){
    double C0 = 0.0;

    Tbar1 = BTF_->build(tensor_type_, "Tbar1", {"cv"});
    Tbar2 = BTF_->build(tensor_type_, "Tbar2", {"ccvv"});

    Tbar1["ia"] = T1["ia"];
    Tbar2["ijab"] = T2["ijab"];

    Tbar1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double D = Fd[i[0]] - Fd[i[1]];
        value *= 1.0 - std::exp(-s_ * D * D);
    });

    Tbar2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double D = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
        value *= 1.0 - std::exp(-s_ * D * D);
    });

    C0 -= 2.0 * C1["ia"] * Tbar1["ia"];
    C0 -= 0.5 * C2["ijab"] * Tbar2["ijab"];

    Tbar1["ia"] = F["ia"];
    Tbar2["ijab"] = V["ijab"];

    Tbar1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] - Fd[i[1]]);
    });

    Tbar2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]]);
    });

    C0 += 2.0 * C1["ia"] * Tbar1["ia"];
    C0 += 0.5 * C2["ijab"] * Tbar2["ijab"];

    return C0;
}

double MRDSRG_SO::E4th_correction_lambda_2(BlockedTensor& C1, BlockedTensor& C2){
    double C0 = 0.0;

    Tbar1 = BTF_->build(tensor_type_, "Tbar1", {"cv"});
    Tbar2 = BTF_->build(tensor_type_, "Tbar2", {"ccvv"});

    Tbar1["ia"] = T1["ia"];
    Tbar2["ijab"] = T2["ijab"];

    Tbar1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double D = Fd[i[0]] - Fd[i[1]];
        value *= std::exp(-s_ * D * D);
    });

    Tbar2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double D = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
        value *= std::exp(-s_ * D * D);
    });

    C0 += 2.0 * C1["ia"] * Tbar1["ia"];
    C0 += 0.5 * C2["ijab"] * Tbar2["ijab"];

    return C0;
}

} // namespace forte
