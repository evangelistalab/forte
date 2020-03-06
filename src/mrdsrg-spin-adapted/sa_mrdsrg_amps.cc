/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/mo_space_info.h"
#include "helpers/timer.h"
#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {

void SA_MRDSRG::guess_t(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1) {
    if (ccvv_source_ == "ZERO") {
        guess_t2_noccvv(V, T2);
        guess_t1_nocv(F, T2, T1);
    } else if (ccvv_source_ == "NORMAL") {
        guess_t2_std(V, T2);
        guess_t1_std(F, T2, T1);
    }
}

void SA_MRDSRG::guess_t_df(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& F,
                           BlockedTensor& T1) {
    if (ccvv_source_ == "ZERO") {
        guess_t2_noccvv_df(B, T2);
        guess_t1_nocv(F, T2, T1);
    } else if (ccvv_source_ == "NORMAL") {
        guess_t2_std_df(B, T2);
        guess_t1_std(F, T2, T1);
    }
}

void SA_MRDSRG::update_t() {
    if (ccvv_source_ == "ZERO") {
        update_t2_noccvv();
        update_t1_nocv();
    } else if (ccvv_source_ == "NORMAL") {
        update_t2_std();
        update_t1_std();
    }
}

void SA_MRDSRG::guess_t2_std(BlockedTensor& V, BlockedTensor& T2) {
    local_timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, T2norm_ = 0.0, T2rms_ = 0.0;

    T2["ijab"] = V["ijab"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", {"hhpp"});
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
    }

    T2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i[0]] + Fdiag_[i[1]] -
                                                                    Fdiag_[i[2]] - Fdiag_[i[3]]);
            T2norm_ += value * value;
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        }
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", {"hhpp"});
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
    }

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>&, double& value) {
        T2norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T2norm_ = std::sqrt(T2norm_);

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void SA_MRDSRG::guess_t2_std_df(BlockedTensor& B, BlockedTensor& T2) {
    local_timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, T2norm_ = 0.0, T2rms_ = 0.0;

    T2["ijab"] = B["gia"] * B["gjb"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", {"hhpp"});
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
    }

    T2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i[0]] + Fdiag_[i[1]] -
                                                                    Fdiag_[i[2]] - Fdiag_[i[3]]);
            T2norm_ += value * value;
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        }
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", {"hhpp"});
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
    }

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>&, double& value) {
        T2norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T2norm_ = std::sqrt(T2norm_);

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void SA_MRDSRG::guess_t1_std(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1) {
    local_timer timer;
    std::string str = "Computing T1 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T1max_ = 0.0, T1norm_ = 0.0, T1rms_ = 0.0;

    BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aa"});
    temp["xu"] = L1_["xu"];
    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempG = ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", {"aa"});
        tempG["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        temp["uv"] = tempG["uv"];
    }
    // scale by delta
    temp.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= Fdiag_[i[0]] - Fdiag_[i[1]];
    });
    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempG = ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", {"aa"});
        tempG["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        temp["uv"] = tempG["uv"];
    }

    T1["ia"] = F["ia"];
    T1["ia"] += temp["xu"] * T2["iuax"];
    T1["ia"] -= 0.5 * temp["xu"] * T2["iuxa"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempT1 = ambit::BlockedTensor::build(tensor_type_, "Temp T1", {"hp"});
        tempT1["jb"] = U_["ji"] * T1["ia"] * U_["ba"];
        T1["ia"] = tempT1["ia"];
    }

    T1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i[0]] - Fdiag_[i[1]]);
            T1norm_ += value * value;
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        }
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempT1 = ambit::BlockedTensor::build(tensor_type_, "Temp T1", {"hp"});
        tempT1["jb"] = U_["ij"] * T1["ia"] * U_["ab"];
        T1["ia"] = tempT1["ia"];
    }

    // zero internal amplitudes
    T1.block("aa").iterate([&](const std::vector<size_t>&, double& value) {
        T1norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T1norm_ = std::sqrt(T1norm_);

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void SA_MRDSRG::guess_t2_noccvv(BlockedTensor& V, BlockedTensor& T2) {
    local_timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, T2norm_ = 0.0, T2rms_ = 0.0;

    T2["ijab"] = V["ijab"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", {"hhpp"});
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
    }

    // labels for ccvv blocks and the rest blocks
    std::vector<std::string> other_blocks(T2.block_labels());
    other_blocks.erase(std::remove(other_blocks.begin(), other_blocks.end(), "ccvv"),
                       other_blocks.end());

    // ccvv blocks
    T2.block("ccvv").iterate([&](const std::vector<size_t>& i, double& value) {
        size_t i0 = core_mos_[i[0]];
        size_t i1 = core_mos_[i[1]];
        size_t i2 = virt_mos_[i[2]];
        size_t i3 = virt_mos_[i[3]];

        value /= Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];

        T2norm_ += value * value;
        if (std::fabs(value) > std::fabs(T2max_))
            T2max_ = value;
    });

    // other blocks
    for (const std::string& block : other_blocks) {
        T2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i0] + Fdiag_[i1] -
                                                                    Fdiag_[i2] - Fdiag_[i3]);

            T2norm_ += value * value;
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", {"hhpp"});
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
    }

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>&, double& value) {
        T2norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T2norm_ = std::sqrt(T2norm_);

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void SA_MRDSRG::guess_t2_noccvv_df(BlockedTensor& B, BlockedTensor& T2) {
    local_timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, T2norm_ = 0.0, T2rms_ = 0.0;

    T2["ijab"] = B["gia"] * B["gjb"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", {"hhpp"});
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
    }

    // labels for ccvv blocks and the rest blocks
    std::vector<std::string> other_blocks(T2.block_labels());
    other_blocks.erase(std::remove(other_blocks.begin(), other_blocks.end(), "ccvv"),
                       other_blocks.end());

    // ccvv blocks
    T2.block("ccvv").iterate([&](const std::vector<size_t>& i, double& value) {
        size_t i0 = core_mos_[i[0]];
        size_t i1 = core_mos_[i[1]];
        size_t i2 = virt_mos_[i[2]];
        size_t i3 = virt_mos_[i[3]];

        value /= Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];

        T2norm_ += value * value;
        if (std::fabs(value) > std::fabs(T2max_))
            T2max_ = value;
    });

    // other blocks
    for (const std::string& block : other_blocks) {
        T2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i0] + Fdiag_[i1] -
                                                                    Fdiag_[i2] - Fdiag_[i3]);

            T2norm_ += value * value;
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempT2 = ambit::BlockedTensor::build(tensor_type_, "Temp T2", {"hhpp"});
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
    }

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>&, double& value) {
        T2norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T2norm_ = std::sqrt(T2norm_);

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void SA_MRDSRG::guess_t1_nocv(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1) {
    local_timer timer;
    std::string str = "Computing T1 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T1max_ = 0.0, T1norm_ = 0.0, T1rms_ = 0.0;

    BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aa"});
    temp["xu"] = L1_["xu"];
    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempG = ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", {"aa"});
        tempG["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        temp["uv"] = tempG["uv"];
    }
    // scale by delta
    temp.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= Fdiag_[i[0]] - Fdiag_[i[1]];
    });
    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempG = ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", {"aa"});
        tempG["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        temp["uv"] = tempG["uv"];
    }

    T1["ia"] = F["ia"];
    T1["ia"] += temp["xu"] * T2["iuax"];
    T1["ia"] -= 0.5 * temp["xu"] * T2["iuxa"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempT1 = ambit::BlockedTensor::build(tensor_type_, "Temp T1", {"hp"});
        tempT1["jb"] = U_["ji"] * T1["ia"] * U_["ba"];
        T1["ia"] = tempT1["ia"];
    }

    // labels for cv blocks and the rest blocks
    std::vector<std::string> other_blocks(T1.block_labels());
    other_blocks.erase(std::remove(other_blocks.begin(), other_blocks.end(), "cv"),
                       other_blocks.end());

    // cv blocks
    T1.block("cv").iterate([&](const std::vector<size_t>& i, double& value) {
        size_t i0 = core_mos_[i[0]];
        size_t i1 = virt_mos_[i[1]];

        value /= Fdiag_[i0] - Fdiag_[i1];

        T1norm_ += value * value;
        if (std::fabs(value) > std::fabs(T1max_))
            T1max_ = value;
    });

    // other blocks
    for (const std::string& block : other_blocks) {
        T1.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i0] - Fdiag_[i1]);

            T1norm_ += value * value;
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempT1 = ambit::BlockedTensor::build(tensor_type_, "Temp T1", {"hp"});
        tempT1["jb"] = U_["ij"] * T1["ia"] * U_["ab"];
        T1["ia"] = tempT1["ia"];
    }

    // zero internal amplitudes
    T1.block("aa").iterate([&](const std::vector<size_t>&, double& value) {
        T1norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T1norm_ = std::sqrt(T1norm_);

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void SA_MRDSRG::update_t2_std() {
    T2max_ = 0.0, T2norm_ = 0.0;

    /**
     * Update T2 using delta_t algorithm
     * T2(new) = T2(old) + DT2
     * DT2 = Hbar2(old) * (1 - exp(-s*d^2)) / d^2 - T2(old) * exp(-s * d^2)
     *       ------------------------------------   -----------------------
     *                      Step 1                          Step 2
     **/

    // Step 1: work on Hbar2 where DT2 is treated as intermediate

    // make a copy of the active part of Hbar2 as it will be used as intermediate
    auto Hbar2copy = BlockedTensor::build(tensor_type_, "Hbar2 active copy", {"aaaa"});
    Hbar2copy["uvxy"] = Hbar2_["uvxy"];

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
    DT2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i[0]] + Fdiag_[i[1]] -
                                                                Fdiag_[i[2]] - Fdiag_[i[3]]);
    });
    t2.stop();

    // Step 2: work on T2 where Hbar2 is treated as intermediate

    timer t4("copy T2 to Hbar2");
    // copy T2 to Hbar2
    Hbar2_["ijab"] = T2_["ijab"];
    t4.stop();

    timer t5("transform T2 to semi-canonical basis");
    // transform T2 to semi-canonical basis
    if (!semi_canonical_) {
        auto temp = BlockedTensor::build(tensor_type_, "temp for T2 update", {"hhpp"});
        temp["klab"] = U_["ki"] * U_["lj"] * T2_["ijab"];
        T2_["ijcd"] = temp["ijab"] * U_["db"] * U_["ca"];
    }
    t5.stop();

    timer t6("scale T2 by delta exponential");
    // scale T2 by delta exponential
    T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized(Fdiag_[i[0]] + Fdiag_[i[1]] - Fdiag_[i[2]] -
                                                    Fdiag_[i[3]]);
    });
    t6.stop();

    timer t7("minus the renormalized T2 from renormalized Hbar2");
    // minus the renormalized T2 from renormalized Hbar2
    DT2_["ijab"] -= T2_["ijab"];
    t7.stop();

    timer t8("zero internal amplitudes");
    // zero internal amplitudes
    DT2_.block("aaaa").iterate([&](const std::vector<size_t>&, double& value) { value = 0.0; });
    t8.stop();

    // compute RMS
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
    T2_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        T2norm_ += value * value;
        if (std::fabs(value) > std::fabs(T2max_))
            T2max_ = value;
    });
    T2norm_ = std::sqrt(T2norm_);
    t10.stop();

    // reset the active part of Hbar2
    Hbar2_["uvxy"] = Hbar2copy["uvxy"];
}

void SA_MRDSRG::update_t1_std() {
    T1max_ = 0.0, T1norm_ = 0.0;

    /**
     * Update T1 using delta_t algorithm
     * T1(new) = T1(old) + DT1
     * DT1 = Hbar1(old) * (1 - exp(-s*d^2)) / d^2 - T1(old) * exp(-s * d^2)
     *       ------------------------------------   -----------------------
     *                      Step 1                          Step 2
     **/

    // Step 1: work on Hbar1 where DT1 is treated as intermediate

    // make a copy of the active part of Hbar2 as it will be used as intermediate
    auto Hbar1copy = BlockedTensor::build(tensor_type_, "Hbar1 active copy", {"aa"});
    Hbar1copy["uv"] = Hbar1_["uv"];

    // transform Hbar1 to semi-canonical basis
    if (!semi_canonical_) {
        DT1_["jb"] = U_["ji"] * Hbar1_["ia"] * U_["ba"];
    } else {
        DT1_["ia"] = Hbar1_["ia"];
    }

    // scale Hbar1 by renormalized denominator
    DT1_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i[0]] - Fdiag_[i[1]]);
    });

    // Step 2: work on T1 where Hbar1 is treated as intermediate

    // copy T1 to Hbar1
    Hbar1_["ia"] = T1_["ia"];

    // transform T1 to semi-canonical basis
    if (!semi_canonical_) {
        auto temp = ambit::BlockedTensor::build(tensor_type_, "temp for T1 update", {"hp"});
        temp["jb"] = U_["ji"] * T1_["ia"] * U_["ba"];
        T1_["ia"] = temp["ia"];
    }

    // scale T1 by delta exponential
    T1_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized(Fdiag_[i[0]] - Fdiag_[i[1]]);
    });

    // minus the renormalized T1 from renormalized Hbar1
    DT1_["ia"] -= T1_["ia"];

    // zero internal amplitudes
    DT1_.block("aa").iterate([&](const std::vector<size_t>&, double& value) { value = 0.0; });

    // compute RMS
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
    T1_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        T1norm_ += value * value;
        if (std::fabs(value) > std::fabs(T1max_))
            T1max_ = value;
    });

    T1norm_ = std::sqrt(T1norm_);

    // reset the active part of Hbar2
    Hbar1_["uv"] = Hbar1copy["uv"];
}

void SA_MRDSRG::update_t2_noccvv() {
    T2max_ = 0.0, T2norm_ = 0.0;

    // create a temporary tensor
    BlockedTensor R2 = ambit::BlockedTensor::build(tensor_type_, "R2", {"hhpp"});
    R2["ijab"] = T2_["ijab"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto tempR2 = ambit::BlockedTensor::build(tensor_type_, "Temp R2", {"hhpp"});
        tempR2["klab"] = U_["ki"] * U_["lj"] * R2["ijab"];
        R2["ijcd"] = tempR2["ijab"] * U_["db"] * U_["ca"];

        auto tempH2 = ambit::BlockedTensor::build(tensor_type_, "Temp Hbar2", {"hhpp"});
        tempH2["klab"] = U_["ki"] * U_["lj"] * Hbar2_["ijab"];
        Hbar2_["ijcd"] = tempH2["ijab"] * U_["db"] * U_["ca"];
    }
    R2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= Fdiag_[i[0]] + Fdiag_[i[1]] - Fdiag_[i[2]] - Fdiag_[i[3]];
    });
    R2["ijab"] += Hbar2_["ijab"];

    // block labels
    std::vector<std::string> other_blocks(R2.block_labels());
    other_blocks.erase(std::remove(other_blocks.begin(), other_blocks.end(), "ccvv"),
                       other_blocks.end());

    // ccvv blocks
    R2.block("ccvv").iterate([&](const std::vector<size_t>& i, double& value) {
        size_t i0 = core_mos_[i[0]];
        size_t i1 = core_mos_[i[1]];
        size_t i2 = virt_mos_[i[2]];
        size_t i3 = virt_mos_[i[3]];

        value /= Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];

        T2norm_ += value * value;
        if (std::fabs(value) > std::fabs(T2max_))
            T2max_ = value;
    });

    // other blocks
    for (const std::string& block : other_blocks) {
        R2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i0] + Fdiag_[i1] -
                                                                    Fdiag_[i2] - Fdiag_[i3]);

            T2norm_ += value * value;
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempR2 = ambit::BlockedTensor::build(tensor_type_, "Temp R2", {"hhpp"});
        tempR2["klab"] = U_["ik"] * U_["jl"] * R2["ijab"];
        R2["ijcd"] = tempR2["ijab"] * U_["bd"] * U_["ac"];
    }

    // zero internal amplitudes
    R2.block("aaaa").iterate([&](const std::vector<size_t>&, double& value) {
        T2norm_ -= value * value;
        value = 0.0;
    });

    // compute RMS
    DT2_["ijab"] = T2_["ijab"] - R2["ijab"];
    T2rms_ = DT2_.norm();

    // copy R2 to T2
    T2_["ijab"] = R2["ijab"];

    // norms
    T2norm_ = std::sqrt(T2norm_);
}

void SA_MRDSRG::update_t1_nocv() {
    T1max_ = 0.0, T1norm_ = 0.0;

    // create a temporary tensor
    BlockedTensor R1 = ambit::BlockedTensor::build(tensor_type_, "R1", {"hp"});
    R1["ia"] = T1_["ia"];
    // transform to semi-canonical basis
    if (!semi_canonical_) {
        auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp R1", {"hp"});
        temp["jb"] = U_["ji"] * R1["ia"] * U_["ba"];
        R1["ia"] = temp["ia"];

        temp["jb"] = U_["ji"] * Hbar1_["ia"] * U_["ba"];
        Hbar1_["ia"] = temp["ia"];
    }
    R1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= Fdiag_[i[0]] - Fdiag_[i[1]];
    });
    R1["ia"] += Hbar1_["ia"];

    // block labels
    std::vector<std::string> other_blocks(R1.block_labels());
    other_blocks.erase(std::remove(other_blocks.begin(), other_blocks.end(), "cv"),
                       other_blocks.end());

    // cv blocks
    R1.block("cv").iterate([&](const std::vector<size_t>& i, double& value) {
        size_t i0 = core_mos_[i[0]];
        size_t i1 = virt_mos_[i[1]];

        value /= Fdiag_[i0] - Fdiag_[i1];

        T1norm_ += value * value;
        if (std::fabs(value) > std::fabs(T1max_))
            T1max_ = value;
    });

    // other blocks
    for (const std::string& block : other_blocks) {
        R1.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i0] - Fdiag_[i1]);

            T1norm_ += value * value;
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        auto tempR1 = ambit::BlockedTensor::build(tensor_type_, "Temp R1", {"hp"});
        tempR1["jb"] = U_["ij"] * R1["ia"] * U_["ab"];
        R1["ia"] = tempR1["ia"];
    }

    // zero internal amplitudes
    R1.block("aa").iterate([&](const std::vector<size_t>&, double& value) {
        T1norm_ -= value * value;
        value = 0.0;
    });

    // compute RMS
    DT1_["ia"] = T1_["ia"] - R1["ia"];
    T1rms_ = DT1_.norm();

    // copy R1 to T1
    T1_["ia"] = R1["ia"];

    // norms
    T1norm_ = std::sqrt(T1norm_);
}
} // namespace forte
