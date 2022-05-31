/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <cstdio>
#include <sys/stat.h>

#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {

void SA_MRDSRG::guess_t(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1,
                        BlockedTensor& B) {
    print_h2("Build Initial Amplitudes Guesses");

    guess_t2(V, T2, B);
    guess_t1(F, T2, T1);

    analyze_amplitudes("Initial", T1_, T2_);
}

void SA_MRDSRG::update_t() {
    update_t2();
    if (t1_type_ == "PROJECT")
        update_t1_proj();
    else
        update_t1();
}

void SA_MRDSRG::guess_t2(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& B) {
    local_timer timer;

    struct stat buf;
    if (read_amps_cwd_ and (stat(t2_file_cwd_.c_str(), &buf) == 0)) {
        print_contents("Reading T2 from current dir");
        T2.load(t2_file_cwd_);
    } else if (restart_amps_ and (stat(t2_file_chk_.c_str(), &buf) == 0)) {
        print_contents("Reading previous T2 from scratch dir");
        T2.load(t2_file_chk_);
    } else {
        print_contents("Computing T2 amplitudes from PT2");
        if (eri_df_) {
            T2["ijab"] = B["gia"] * B["gjb"];
        } else {
            T2["ijab"] = V["ijab"];
        }
        guess_t2_impl(T2);
    }

    T2max_ = T2.norm(0);
    T2norm_ = T2.norm();
    T2rms_ = 0.0;

    print_done(timer.get());
}

void SA_MRDSRG::guess_t2_impl(BlockedTensor& T2) {
    // transform to semi-canonical basis
    BlockedTensor tempT2;
    if (!semi_canonical_) {
        tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", T2.block_labels());
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
    }

    // special case for CCVV block
    std::vector<std::string> T2blocks(T2.block_labels());
    if (ccvv_source_ == "ZERO") {
        T2blocks.erase(std::remove(T2blocks.begin(), T2blocks.end(), "ccvv"), T2blocks.end());
        T2.block("ccvv").iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = core_mos_[i[0]];
            size_t i1 = core_mos_[i[1]];
            size_t i2 = virt_mos_[i[2]];
            size_t i3 = virt_mos_[i[3]];

            value /= Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];
        });
    }

    for (const std::string& block : T2blocks) {
        T2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            double denom = Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];
            value *= dsrg_source_->compute_renormalized_denominator(denom);
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
    }

    // zero internal amplitudes
    internal_amps_T2(T2);
}

void SA_MRDSRG::guess_t1(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1) {
    local_timer timer;

    struct stat buf;
    if (read_amps_cwd_ and (stat(t1_file_cwd_.c_str(), &buf) == 0)) {
        print_contents("Reading T1 from current dir");
        T1.load(t1_file_cwd_);
    } else if (restart_amps_ and (stat(t1_file_chk_.c_str(), &buf) == 0)) {
        print_contents("Reading previous T1 from scratch dir");
        T1.load(t1_file_chk_);
    } else {
        if (t1_guess_ == "ZERO") {
            print_contents("Zeroing T1 amplitudes as requested");
            T1.zero();
        } else {
            print_contents("Computing T1 amplitudes from PT2");

            T1["ia"] = F["ia"];
            T1["ia"] += T2["ivaw"] * F["wu"] * L1_["uv"];
            T1["ia"] -= 0.5 * T2["ivwa"] * F["wu"] * L1_["uv"];
            T1["ia"] -= T2["iwau"] * F["vw"] * L1_["uv"];
            T1["ia"] += 0.5 * T2["iwua"] * F["vw"] * L1_["uv"];

            // transform to semi-canonical basis
            BlockedTensor tempX;
            if (!semi_canonical_) {
                tempX = ambit::BlockedTensor::build(tensor_type_, "Temp T1", T1.block_labels());
                tempX["jb"] = U_["ji"] * T1["ia"] * U_["ba"];
                T1["ia"] = tempX["ia"];
            }

            // special case for CV block
            std::vector<std::string> T1blocks(T1.block_labels());
            if (ccvv_source_ == "ZERO") {
                T1blocks.erase(std::remove(T1blocks.begin(), T1blocks.end(), "cv"), T1blocks.end());
                T1.block("cv").iterate([&](const std::vector<size_t>& i, double& value) {
                    size_t i0 = core_mos_[i[0]];
                    size_t i1 = virt_mos_[i[1]];

                    value /= Fdiag_[i0] - Fdiag_[i1];
                });
            }

            for (const std::string& block : T1blocks) {
                T1.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
                    size_t i0 = label_to_spacemo_[block[0]][i[0]];
                    size_t i1 = label_to_spacemo_[block[1]][i[1]];
                    value *=
                        dsrg_source_->compute_renormalized_denominator(Fdiag_[i0] - Fdiag_[i1]);
                });
            }

            // transform back to non-canonical basis
            if (!semi_canonical_) {
                tempX["jb"] = U_["ij"] * T1["ia"] * U_["ab"];
                T1["ia"] = tempX["ia"];
            }

            // zero internal amplitudes
            internal_amps_T1(T1);
        }
    }

    // norms
    T1max_ = T1.norm(0);
    T1norm_ = T1.norm();
    T1rms_ = 0.0;

    print_done(timer.get());
}

void SA_MRDSRG::update_t2() {
    // make a copy of the active part of Hbar2 as it will be used as intermediate
    auto Hbar2copy = BlockedTensor::build(tensor_type_, "Hbar2 active copy", {"aaaa"});
    Hbar2copy["uvxy"] = Hbar2_["uvxy"];

    // special case for CCVV block
    std::vector<std::string> T2blocks(T2_.block_labels());
    if (ccvv_source_ == "ZERO") {
        T2blocks.erase(std::remove(T2blocks.begin(), T2blocks.end(), "ccvv"), T2blocks.end());
    }

    /**
     * Update T2 using delta_t algorithm
     * T2(new) = T2(old) + DT2
     * DT2 = Hbar2(old) * (1 - exp(-s*d^2)) / d - T2(old) * exp(-s * d^2)
     *       ----------------------------------   -----------------------
     *                      Step 1                        Step 2
     **/

    // Step 1: work on Hbar2 where DT2 is treated as intermediate

    timer t1("transform Hbar2 to semi-canonical basis");
    // transform Hbar2 to semi-canonical basis
    if (!semi_canonical_) {
        DT2_["klcd"] = U_["ki"] * U_["lj"] * Hbar2_["ijab"] * U_["db"] * U_["ca"];
    } else {
        DT2_["ijab"] = Hbar2_["ijab"];
    }
    t1.stop();

    timer t2("scale Hbar2 by renormalized denominator");
    // scale Hbar2 by renormalized denominator
    if (ccvv_source_ == "ZERO") {
        DT2_.block("ccvv").iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = core_mos_[i[0]];
            size_t i1 = core_mos_[i[1]];
            size_t i2 = virt_mos_[i[2]];
            size_t i3 = virt_mos_[i[3]];
            value /= Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];
        });
    }

    for (const std::string& block : T2blocks) {
        DT2_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            double denom = Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];
            value *= dsrg_source_->compute_renormalized_denominator(denom);
        });
    }
    t2.stop();

    // Step 2: work on T2 where Hbar2 is treated as intermediate

    timer t4("copy T2 to Hbar2");
    // copy T2 to Hbar2
    Hbar2_["ijab"] = T2_["ijab"];
    t4.stop();

    timer t5("transform T2 to semi-canonical basis");
    // transform T2 to semi-canonical basis
    if (!semi_canonical_) {
        auto temp = BlockedTensor::build(tensor_type_, "temp for T2 update", T2blocks);
        temp["klab"] = U_["ki"] * U_["lj"] * T2_["ijab"];
        T2_["ijcd"] = temp["ijab"] * U_["db"] * U_["ca"];
    }
    t5.stop();

    timer t6("scale T2 by delta exponential");
    // scale T2 by delta exponential
    for (const std::string& block : T2blocks) {
        T2_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            double denom = Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];
            value *= dsrg_source_->compute_renormalized(denom);
        });
    }
    if (ccvv_source_ == "ZERO") {
        T2_.block("ccvv").zero();
    }
    t6.stop();

    timer t7("minus the renormalized T2 from renormalized Hbar2");
    // minus the renormalized T2 from renormalized Hbar2
    DT2_["ijab"] -= T2_["ijab"];
    t7.stop();

    timer t8("zero internal amplitudes");
    // zero internal amplitudes
    internal_amps_T2(DT2_);
    t8.stop();

    // compute consecutive T2 difference
    T2rms_ = DT2_.norm();

    timer t9("transform DT2 back to original basis");
    // transform DT2 back to original basis
    if (!semi_canonical_) {
        T2_["klab"] = U_["ik"] * U_["jl"] * DT2_["ijab"];
        DT2_["ijcd"] = T2_["ijab"] * U_["bd"] * U_["ac"];
    }
    t9.stop();

    timer t10("Step 3: update and analyze T2");
    // Step 3: update and analyze T2
    T2_["ijab"] = Hbar2_["ijab"];
    T2_["ijab"] += DT2_["ijab"];

    // compute norm and find maximum
    T2norm_ = T2_.norm(2);
    T2max_ = T2_.norm(0);
    t10.stop();

    // reset the active part of Hbar2
    Hbar2_["uvxy"] = Hbar2copy["uvxy"];
}

void SA_MRDSRG::update_t1() {
    // make a copy of the active part of Hbar2 as it will be used as intermediate
    auto Hbar1copy = BlockedTensor::build(tensor_type_, "Hbar1 active copy", {"aa"});
    Hbar1copy["uv"] = Hbar1_["uv"];

    // special case for CV block
    std::vector<std::string> T1blocks(T1_.block_labels());
    if (ccvv_source_ == "ZERO") {
        T1blocks.erase(std::remove(T1blocks.begin(), T1blocks.end(), "cv"), T1blocks.end());
    }

    /**
     * Update T1 using delta_t algorithm
     * T1(new) = T1(old) + DT1
     * DT1 = Hbar1(old) * (1 - exp(-s*d^2)) / d - T1(old) * exp(-s * d^2)
     *       ----------------------------------   -----------------------
     *                      Step 1                        Step 2
     **/

    // Step 1: work on Hbar1 where DT1 is treated as intermediate

    // transform Hbar1 to semi-canonical basis
    if (!semi_canonical_) {
        DT1_["jb"] = U_["ji"] * Hbar1_["ia"] * U_["ba"];
    } else {
        DT1_["ia"] = Hbar1_["ia"];
    }

    // scale Hbar1 by renormalized denominator
    if (ccvv_source_ == "ZERO") {
        DT1_.block("cv").iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = core_mos_[i[0]];
            size_t i1 = virt_mos_[i[1]];
            value /= Fdiag_[i0] - Fdiag_[i1];
        });
    }

    for (const std::string& block : T1blocks) {
        DT1_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            double denom = Fdiag_[i0] - Fdiag_[i1];
            value *= dsrg_source_->compute_renormalized_denominator(denom);
        });
    }

    // Step 2: work on T1 where Hbar1 is treated as intermediate

    // copy T1 to Hbar1
    Hbar1_["ia"] = T1_["ia"];

    // transform T1 to semi-canonical basis
    if (!semi_canonical_) {
        auto temp =
            ambit::BlockedTensor::build(tensor_type_, "temp for T1 update", T1_.block_labels());
        temp["jb"] = U_["ji"] * T1_["ia"] * U_["ba"];
        T1_["ia"] = temp["ia"];
    }

    // scale T1 by delta exponential
    for (const std::string& block : T1blocks) {
        T1_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            double denom = Fdiag_[i0] - Fdiag_[i1];
            if (block == "ca")
                outfile->Printf("\n  %20.15f %20.15f %20.15f", Fdiag_[i0], Fdiag_[i1], denom);
            value *= dsrg_source_->compute_renormalized(denom);
        });
    }
    if (ccvv_source_ == "ZERO") {
        T1_.block("cv").zero();
    }

    // minus the renormalized T1 from renormalized Hbar1
    DT1_["ia"] -= T1_["ia"];

    // zero internal amplitudes
    internal_amps_T1(DT1_);

    // compute consecutive T1 difference
    T1rms_ = DT1_.norm();

    // transform DT1 back to original basis
    if (!semi_canonical_) {
        T1_["jb"] = U_["ji"] * DT1_["ia"] * U_["ba"];
        DT1_["ia"] = T1_["ia"];
    }

    // Step 3: update and analyze T1
    T1_["ia"] = Hbar1_["ia"];
    T1_["ia"] += DT1_["ia"];

    // compute norm and find maximum
    T1max_ = T1_.norm(0);
    T1norm_ = T1_.norm(2);

    // reset the active part of Hbar2
    Hbar1_["uv"] = Hbar1copy["uv"];

    // test numerator
    auto Nca = ambit::BlockedTensor::build(tensor_type_, "Nca", {"ca"});
    Nca["mz"] += 0.5 * Hbar1_["mu"] * Eta1_["uz"];
    Nca["mz"] -= 0.5 * Hbar2_["xmuv"] * L2_["uvxz"];

    auto Nav = ambit::BlockedTensor::build(tensor_type_, "Nav", {"av"});
    Nav["ze"] += 0.5 * Hbar1_["ue"] * L1_["zu"];
    Nav["ze"] += 0.5 * Hbar2_["uvex"] * L2_["zxuv"];

    // test denominator
    auto Dccaa = ambit::BlockedTensor::build(tensor_type_, "Dccaa", {"ccaa"});
    auto Icc = ambit::BlockedTensor::build(tensor_type_, "Icc", {"cc"});
    Icc.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });
    Dccaa["mnuv"] += 0.5 * F_["vx"] * Eta1_["xu"] * Icc["nm"];
    Dccaa["mnuv"] -= 0.5 * F_["nm"] * Eta1_["vu"];
    Dccaa["mnuv"] += 0.25 * V_["xnmy"] * Eta1_["vx"] * Eta1_["yu"];
    Dccaa["mnuv"] -= 0.25 * V_["nxmy"] * Eta1_["vx"] * Eta1_["yu"];
    Dccaa["mnuv"] += 0.25 * V_["mnxy"] * Eta1_["xv"] * Eta1_["yu"];
    Dccaa["mnuv"] -= 0.25 * V_["nmxy"] * Eta1_["xv"] * Eta1_["yu"];
    Dccaa["mnuv"] -= 0.5 * V_["vwxy"] * L2_["xyuw"] * Icc["mn"];
    Dccaa["mnuv"] -= (1.0 / 6.0) * V_["mnxy"] * L2_["xyuv"];
    Dccaa["mnuv"] += (1.0 / 6.0) * V_["mnxy"] * L2_["xyvu"];
    Dccaa["mnuv"] += 0.5 * V_["nxmy"] * L2_["vyux"];
    Dccaa["mnuv"] -= (1.0 / 6.0) * V_["xnmy"] * L2_["vyux"];
    Dccaa["mnuv"] += (1.0 / 6.0) * V_["xnmy"] * L2_["yvux"];
    Dccaa.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            if (i[0] == i[1] and i[2] == i[3]) {
                outfile->Printf("\n  %30.15f", value);
            }
        });

    auto ncore = core_mos_.size();
    auto nactv = actv_mos_.size();
    auto Dcaa = ambit::Tensor::build(tensor_type_, "Dcaa", {ncore, nactv, nactv});
    Dcaa.iterate([&](const std::vector<size_t>& i, double& value) {
        value =
            Dccaa.block("ccaa")
                .data()[i[0] * nactv * nactv * ncore + i[0] * nactv * nactv + i[1] * nactv + i[2]];
    });
    auto nactv_h = Xca_.dim(1);
    auto Dcaa_h = ambit::Tensor::build(tensor_type_, "Dcaa_h", {ncore, nactv_h, nactv_h});
    Dcaa_h("mxy") = Dcaa("muv") * Xca_("ux") * Xca_("vy");
    auto Dca = ambit::Tensor::build(tensor_type_, "Dca", {ncore, nactv_h});
    Dca.iterate([&](const std::vector<size_t>& i, double& value) {
        value = Dcaa_h.data()[i[0] * nactv * nactv + i[1] * nactv + i[1]];
        outfile->Printf("\n  %20.15f", value);
    });

    auto update_ca = ambit::Tensor::build(tensor_type_, "Nca", {ncore, nactv_h});
    update_ca("mx") = Nca.block("ca")("mu") * Xca_("ux");
    update_ca.iterate([&](const std::vector<size_t>& i, double& value) {
        outfile->Printf("\n  %40.15f", value);
        value /= Dca.data()[i[0] * nactv_h + i[1]];
        outfile->Printf("\n  %45.15f", value);
    });
}

void SA_MRDSRG::update_t1_proj() {
    /// update T1 according to <{a^i_a} Hbar^a_i> = 0

    // core-virt block
    const auto& Hbar1cv_data = Hbar1_.block("cv").data();
    auto dim_v = Hbar1_.block("cv").dim(1);
    T1_.block("cv").iterate([&](const std::vector<size_t>& i, double& value) {
        size_t i0 = label_to_spacemo_['c'][i[0]];
        size_t i1 = label_to_spacemo_['v'][i[1]];
        double denom = Fdiag_[i0] - Fdiag_[i1];
        value += Hbar1cv_data[i[0] * dim_v + i[1]] / denom;
    });

    // core-actv block

    // actv-virt block
}

void SA_MRDSRG::dump_amps_to_disk() {
    // dump to psi4 scratch directory for reference relaxation
    if (restart_amps_ and (relax_ref_ != "NONE")) {
        local_timer lt;
        print_contents("Dumping amplitudes to scratch dir");
        T1_.save(t1_file_chk_);
        T2_.save(t2_file_chk_);
        print_done(lt.get());
    }

    // dump amplitudes to the current directory
    if (dump_amps_cwd_) {
        local_timer lt;
        print_contents("Dumping amplitudes to current dir");
        T1_.save(t1_file_cwd_);
        T2_.save(t2_file_cwd_);
        print_done(lt.get());
    }
}

} // namespace forte
