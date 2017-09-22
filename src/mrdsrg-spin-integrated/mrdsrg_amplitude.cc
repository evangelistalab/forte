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

#include <algorithm>
#include <map>
#include <vector>

#include "../helpers.h"
#include "../mini-boost/boost/format.hpp"
#include "mrdsrg.h"

#define TIME_LINE(x) timer_on(#x);x;timer_off(#x)

namespace psi {
namespace forte {

void MRDSRG::guess_t(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1) {
    // if fully decouple core-core-virtual-virtual block
    std::string ccvv_source = options_.get_str("CCVV_SOURCE");

    if (ccvv_source == "ZERO") {
        guess_t2_noccvv(V, T2);
        guess_t1_nocv(F, T2, T1);
    } else if (ccvv_source == "NORMAL") {
        guess_t2_std(V, T2);
        guess_t1_std(F, T2, T1);
    }
}

void MRDSRG::guess_t_df(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1) {
    // if fully decouple core-core-virtual-virtual block
    std::string ccvv_source = options_.get_str("CCVV_SOURCE");

    if (ccvv_source == "ZERO") {
        guess_t2_noccvv_df(B, T2);
        guess_t1_nocv(F, T2, T1);
    } else if (ccvv_source == "NORMAL") {
        guess_t2_std_df(B, T2);
        guess_t1_std(F, T2, T1);
    }
}

void MRDSRG::update_t() {
    // if fully decouple core-core-virtual-virtual block
    std::string ccvv_source = options_.get_str("CCVV_SOURCE");
    if (ccvv_source == "ZERO") {
        TIME_LINE(update_t2_noccvv());
        TIME_LINE(update_t1_nocv());
    } else if (ccvv_source == "NORMAL") {
        TIME_LINE(update_t2_std());
        TIME_LINE(update_t1_std());
    }
}

void MRDSRG::guess_t2_std(BlockedTensor& V, BlockedTensor& T2) {
    Timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, t2aa_norm_ = 0.0, t2ab_norm_ = 0.0, t2bb_norm_ = 0.0;

    T2["ijab"] = V["ijab"];
    T2["iJaB"] = V["iJaB"];
    T2["IJAB"] = V["IJAB"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        tempT2["kLaB"] = U_["ki"] * U_["LJ"] * T2["iJaB"];
        tempT2["KLAB"] = U_["KI"] * U_["LJ"] * T2["IJAB"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
        T2["iJcD"] = tempT2["iJaB"] * U_["DB"] * U_["ca"];
        T2["IJCD"] = tempT2["IJAB"] * U_["DB"] * U_["CA"];
    }

    T2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                        Fa_[i[2]] - Fa_[i[3]]);
                t2aa_norm_ += value * value;
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                        Fa_[i[2]] - Fb_[i[3]]);
                t2ab_norm_ += value * value;
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                        Fb_[i[2]] - Fb_[i[3]]);
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        }
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        tempT2["kLaB"] = U_["ik"] * U_["JL"] * T2["iJaB"];
        tempT2["KLAB"] = U_["IK"] * U_["JL"] * T2["IJAB"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
        T2["iJcD"] = tempT2["iJaB"] * U_["BD"] * U_["ac"];
        T2["IJCD"] = tempT2["IJAB"] * U_["BD"] * U_["AC"];
    }

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>& i, double& value) {
        t2aa_norm_ -= value * value;
        value = 0.0;
    });
    T2.block("aAaA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2ab_norm_ -= value * value;
        value = 0.0;
    });
    T2.block("AAAA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2bb_norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T2norm_ = std::sqrt(t2aa_norm_ + t2bb_norm_ + 4 * t2ab_norm_);
    t2aa_norm_ = std::sqrt(t2aa_norm_);
    t2ab_norm_ = std::sqrt(t2ab_norm_);
    t2bb_norm_ = std::sqrt(t2bb_norm_);
    T2rms_ = 0.0;

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG::guess_t2_std_df(BlockedTensor& B, BlockedTensor& T2) {
    Timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, t2aa_norm_ = 0.0, t2ab_norm_ = 0.0, t2bb_norm_ = 0.0;

    T2["ijab"] = B["gia"] * B["gjb"];
    T2["ijab"] -= B["gib"] * B["gja"];
    T2["iJaB"] = B["gia"] * B["gJB"];
    T2["IJAB"] = B["gIA"] * B["gJB"];
    T2["IJAB"] -= B["gIB"] * B["gJA"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        tempT2["kLaB"] = U_["ki"] * U_["LJ"] * T2["iJaB"];
        tempT2["KLAB"] = U_["KI"] * U_["LJ"] * T2["IJAB"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
        T2["iJcD"] = tempT2["iJaB"] * U_["DB"] * U_["ca"];
        T2["IJCD"] = tempT2["IJAB"] * U_["DB"] * U_["CA"];
    }

    T2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                        Fa_[i[2]] - Fa_[i[3]]);
                t2aa_norm_ += value * value;
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                        Fa_[i[2]] - Fb_[i[3]]);
                t2ab_norm_ += value * value;
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                        Fb_[i[2]] - Fb_[i[3]]);
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        }
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        tempT2["kLaB"] = U_["ik"] * U_["JL"] * T2["iJaB"];
        tempT2["KLAB"] = U_["IK"] * U_["JL"] * T2["IJAB"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
        T2["iJcD"] = tempT2["iJaB"] * U_["BD"] * U_["ac"];
        T2["IJCD"] = tempT2["IJAB"] * U_["BD"] * U_["AC"];
    }

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>& i, double& value) {
        t2aa_norm_ -= value * value;
        value = 0.0;
    });
    T2.block("aAaA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2ab_norm_ -= value * value;
        value = 0.0;
    });
    T2.block("AAAA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2bb_norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T2norm_ = std::sqrt(t2aa_norm_ + t2bb_norm_ + 4 * t2ab_norm_);
    t2aa_norm_ = std::sqrt(t2aa_norm_);
    t2ab_norm_ = std::sqrt(t2ab_norm_);
    t2bb_norm_ = std::sqrt(t2bb_norm_);
    T2rms_ = 0.0;

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG::guess_t1_std(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1) {
    Timer timer;
    std::string str = "Computing T1 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T1max_ = 0.0, t1a_norm_ = 0.0, t1b_norm_ = 0.0;

    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aa"}));
    temp["xu"] = Gamma1_["xu"];
    temp["XU"] = Gamma1_["XU"];
    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempG =
            ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", spin_cases({"aa"}));
        tempG["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        tempG["UV"] = U_["UX"] * temp["XY"] * U_["VY"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }
    // scale by delta
    temp.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value *= Fa_[i[0]] - Fa_[i[1]];
            } else {
                value *= Fb_[i[0]] - Fb_[i[1]];
            }
        });
    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempG =
            ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", spin_cases({"aa"}));
        tempG["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        tempG["UV"] = U_["XU"] * temp["XY"] * U_["YV"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }

    T1["ia"] = F["ia"];
    T1["ia"] += temp["xu"] * T2["iuax"];
    T1["ia"] += temp["XU"] * T2["iUaX"];

    T1["IA"] = F["IA"];
    T1["IA"] += temp["xu"] * T2["uIxA"];
    T1["IA"] += temp["XU"] * T2["IUAX"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T1", spin_cases({"hp"}));
        tempT1["jb"] = U_["ji"] * T1["ia"] * U_["ba"];
        tempT1["JB"] = U_["JI"] * T1["IA"] * U_["BA"];
        T1["ia"] = tempT1["ia"];
        T1["IA"] = tempT1["IA"];
    }

    T1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            if (spin[0] == AlphaSpin) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] - Fa_[i[1]]);
                t1a_norm_ += value * value;
            } else {
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] - Fb_[i[1]]);
                t1b_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        }
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T1", spin_cases({"hp"}));
        tempT1["jb"] = U_["ij"] * T1["ia"] * U_["ab"];
        tempT1["JB"] = U_["IJ"] * T1["IA"] * U_["AB"];
        T1["ia"] = tempT1["ia"];
        T1["IA"] = tempT1["IA"];
    }

    // zero internal amplitudes
    T1.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
        t1a_norm_ -= value * value;
        value = 0.0;
    });
    T1.block("AA").iterate([&](const std::vector<size_t>& i, double& value) {
        t1b_norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T1norm_ = std::sqrt(t1a_norm_ + t1b_norm_);
    t1a_norm_ = std::sqrt(t1a_norm_);
    t1b_norm_ = std::sqrt(t1b_norm_);
    T1rms_ = 0.0;

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG::guess_t2_noccvv(BlockedTensor& V, BlockedTensor& T2) {
    Timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, t2aa_norm_ = 0.0, t2ab_norm_ = 0.0, t2bb_norm_ = 0.0;

    T2["ijab"] = V["ijab"];
    T2["iJaB"] = V["iJaB"];
    T2["IJAB"] = V["IJAB"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        tempT2["kLaB"] = U_["ki"] * U_["LJ"] * T2["iJaB"];
        tempT2["KLAB"] = U_["KI"] * U_["LJ"] * T2["IJAB"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
        T2["iJcD"] = tempT2["iJaB"] * U_["DB"] * U_["ca"];
        T2["IJCD"] = tempT2["IJAB"] * U_["DB"] * U_["CA"];
    }

    // labels for ccvv blocks and the rest blocks
    std::vector<std::string> cv_blocks{acore_label_ + acore_label_ + avirt_label_ + avirt_label_,
                                       acore_label_ + bcore_label_ + avirt_label_ + bvirt_label_,
                                       bcore_label_ + bcore_label_ + bvirt_label_ + bvirt_label_};
    std::vector<std::string> other_blocks(T2.block_labels());
    other_blocks.erase(std::remove_if(other_blocks.begin(), other_blocks.end(),
                                      [&](std::string i) {
                                          return std::find(cv_blocks.begin(), cv_blocks.end(), i) !=
                                                 cv_blocks.end();
                                      }),
                       other_blocks.end());

    // map spin with Fock matrices
    std::map<bool, const std::vector<double>> Fock_spin{{true, Fa_}, {false, Fb_}};

    // ccvv blocks
    for (const std::string& block : cv_blocks) {
        // spin
        bool spin0 = islower(block[0]);
        bool spin1 = islower(block[1]);

        // diagonal Fock matrix elements
        const std::vector<double>& F0 = Fock_spin[spin0];
        const std::vector<double>& F1 = Fock_spin[spin1];

        T2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value /= F0[i0] + F1[i1] - F0[i2] - F0[i3];
            if (spin0 && spin1) {
                t2aa_norm_ += value * value;
            } else if (spin0 && !spin1) {
                t2ab_norm_ += value * value;
            } else if (!spin0 && !spin1) {
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // other blocks
    for (const std::string& block : other_blocks) {
        // spin
        bool spin0 = islower(block[0]);
        bool spin1 = islower(block[1]);

        // diagonal Fock matrix elements
        const std::vector<double>& F0 = Fock_spin[spin0];
        const std::vector<double>& F1 = Fock_spin[spin1];

        T2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value *=
                dsrg_source_->compute_renormalized_denominator(F0[i0] + F1[i1] - F0[i2] - F0[i3]);

            if (spin0 && spin1) {
                t2aa_norm_ += value * value;
            } else if (spin0 && !spin1) {
                t2ab_norm_ += value * value;
            } else if (!spin0 && !spin1) {
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        tempT2["kLaB"] = U_["ik"] * U_["JL"] * T2["iJaB"];
        tempT2["KLAB"] = U_["IK"] * U_["JL"] * T2["IJAB"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
        T2["iJcD"] = tempT2["iJaB"] * U_["BD"] * U_["ac"];
        T2["IJCD"] = tempT2["IJAB"] * U_["BD"] * U_["AC"];
    }

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>& i, double& value) {
        t2aa_norm_ -= value * value;
        value = 0.0;
    });
    T2.block("aAaA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2ab_norm_ -= value * value;
        value = 0.0;
    });
    T2.block("AAAA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2bb_norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T2norm_ = std::sqrt(t2aa_norm_ + t2bb_norm_ + 4 * t2ab_norm_);
    t2aa_norm_ = std::sqrt(t2aa_norm_);
    t2ab_norm_ = std::sqrt(t2ab_norm_);
    t2bb_norm_ = std::sqrt(t2bb_norm_);
    T2rms_ = 0.0;

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG::guess_t2_noccvv_df(BlockedTensor& B, BlockedTensor& T2) {
    Timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, t2aa_norm_ = 0.0, t2ab_norm_ = 0.0, t2bb_norm_ = 0.0;

    T2["ijab"] = B["gia"] * B["gjb"];
    T2["ijab"] -= B["gib"] * B["gja"];
    T2["iJaB"] = B["gia"] * B["gJB"];
    T2["IJAB"] = B["gIA"] * B["gJB"];
    T2["IJAB"] -= B["gIB"] * B["gJA"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2["ijab"];
        tempT2["kLaB"] = U_["ki"] * U_["LJ"] * T2["iJaB"];
        tempT2["KLAB"] = U_["KI"] * U_["LJ"] * T2["IJAB"];
        T2["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
        T2["iJcD"] = tempT2["iJaB"] * U_["DB"] * U_["ca"];
        T2["IJCD"] = tempT2["IJAB"] * U_["DB"] * U_["CA"];
    }

    // labels for ccvv blocks and the rest blocks
    std::vector<std::string> cv_blocks{acore_label_ + acore_label_ + avirt_label_ + avirt_label_,
                                       acore_label_ + bcore_label_ + avirt_label_ + bvirt_label_,
                                       bcore_label_ + bcore_label_ + bvirt_label_ + bvirt_label_};
    std::vector<std::string> other_blocks(T2.block_labels());
    other_blocks.erase(std::remove_if(other_blocks.begin(), other_blocks.end(),
                                      [&](std::string i) {
                                          return std::find(cv_blocks.begin(), cv_blocks.end(), i) !=
                                                 cv_blocks.end();
                                      }),
                       other_blocks.end());

    // map spin with Fock matrices
    std::map<bool, const std::vector<double>> Fock_spin{{true, Fa_}, {false, Fb_}};

    // ccvv blocks
    for (const std::string& block : cv_blocks) {
        // spin
        bool spin0 = islower(block[0]);
        bool spin1 = islower(block[1]);

        // diagonal Fock matrix elements
        const std::vector<double>& F0 = Fock_spin[spin0];
        const std::vector<double>& F1 = Fock_spin[spin1];

        T2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value /= F0[i0] + F1[i1] - F0[i2] - F0[i3];
            if (spin0 && spin1) {
                t2aa_norm_ += value * value;
            } else if (spin0 && !spin1) {
                t2ab_norm_ += value * value;
            } else if (!spin0 && !spin1) {
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // other blocks
    for (const std::string& block : other_blocks) {
        // spin
        bool spin0 = islower(block[0]);
        bool spin1 = islower(block[1]);

        // diagonal Fock matrix elements
        const std::vector<double>& F0 = Fock_spin[spin0];
        const std::vector<double>& F1 = Fock_spin[spin1];

        T2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value *=
                dsrg_source_->compute_renormalized_denominator(F0[i0] + F1[i1] - F0[i2] - F0[i3]);

            if (spin0 && spin1) {
                t2aa_norm_ += value * value;
            } else if (spin0 && !spin1) {
                t2ab_norm_ += value * value;
            } else if (!spin0 && !spin1) {
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2["ijab"];
        tempT2["kLaB"] = U_["ik"] * U_["JL"] * T2["iJaB"];
        tempT2["KLAB"] = U_["IK"] * U_["JL"] * T2["IJAB"];
        T2["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
        T2["iJcD"] = tempT2["iJaB"] * U_["BD"] * U_["ac"];
        T2["IJCD"] = tempT2["IJAB"] * U_["BD"] * U_["AC"];
    }

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>& i, double& value) {
        t2aa_norm_ -= value * value;
        value = 0.0;
    });
    T2.block("aAaA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2ab_norm_ -= value * value;
        value = 0.0;
    });
    T2.block("AAAA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2bb_norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T2norm_ = std::sqrt(t2aa_norm_ + t2bb_norm_ + 4 * t2ab_norm_);
    t2aa_norm_ = std::sqrt(t2aa_norm_);
    t2ab_norm_ = std::sqrt(t2ab_norm_);
    t2bb_norm_ = std::sqrt(t2bb_norm_);
    T2rms_ = 0.0;

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG::guess_t1_nocv(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1) {
    Timer timer;
    std::string str = "Computing T1 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T1max_ = 0.0, t1a_norm_ = 0.0, t1b_norm_ = 0.0;

    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aa"}));
    temp["xu"] = Gamma1_["xu"];
    temp["XU"] = Gamma1_["XU"];
    if (!semi_canonical_) {
        BlockedTensor tempG =
            ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", spin_cases({"aa"}));
        tempG["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        tempG["UV"] = U_["UX"] * temp["XY"] * U_["VY"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }
    temp.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value *= Fa_[i[0]] - Fa_[i[1]];
            } else {
                value *= Fb_[i[0]] - Fb_[i[1]];
            }
        });
    if (!semi_canonical_) {
        BlockedTensor tempG =
            ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", spin_cases({"aa"}));
        tempG["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        tempG["UV"] = U_["XU"] * temp["XY"] * U_["YV"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }

    T1["ia"] = F["ia"];
    T1["ia"] += temp["xu"] * T2["iuax"];
    T1["ia"] += temp["XU"] * T2["iUaX"];

    T1["IA"] = F["IA"];
    T1["IA"] += temp["xu"] * T2["uIxA"];
    T1["IA"] += temp["XU"] * T2["IUAX"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T1", spin_cases({"hp"}));
        tempT1["jb"] = U_["ji"] * T1["ia"] * U_["ba"];
        tempT1["JB"] = U_["JI"] * T1["IA"] * U_["BA"];
        T1["ia"] = tempT1["ia"];
        T1["IA"] = tempT1["IA"];
    }

    // labels for ccvv blocks and the rest blocks
    std::vector<std::string> cv_blocks{acore_label_ + avirt_label_, bcore_label_ + bvirt_label_};
    std::vector<std::string> other_blocks(T1.block_labels());
    other_blocks.erase(std::remove_if(other_blocks.begin(), other_blocks.end(),
                                      [&](std::string i) {
                                          return std::find(cv_blocks.begin(), cv_blocks.end(), i) !=
                                                 cv_blocks.end();
                                      }),
                       other_blocks.end());

    // map spin with Fock matrices
    std::map<bool, const std::vector<double>> Fock_spin{{true, Fa_}, {false, Fb_}};

    // cv blocks
    for (const std::string& block : cv_blocks) {
        bool spin0 = islower(block[0]);
        const std::vector<double>& F0 = Fock_spin[spin0];

        T1.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            value /= F0[i0] - F0[i1];
            if (spin0) {
                t1a_norm_ += value * value;
            } else if (!spin0) {
                t1b_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });
    }

    // other blocks
    for (const std::string& block : other_blocks) {
        bool spin0 = islower(block[0]);
        const std::vector<double>& F0 = Fock_spin[spin0];

        T1.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            value *= dsrg_source_->compute_renormalized_denominator(F0[i0] - F0[i1]);

            if (spin0) {
                t1a_norm_ += value * value;
            } else if (!spin0) {
                t1b_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T1", spin_cases({"hp"}));
        tempT1["jb"] = U_["ij"] * T1["ia"] * U_["ab"];
        tempT1["JB"] = U_["IJ"] * T1["IA"] * U_["AB"];
        T1["ia"] = tempT1["ia"];
        T1["IA"] = tempT1["IA"];
    }

    // zero internal amplitudes
    T1.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
        t1a_norm_ -= value * value;
        value = 0.0;
    });
    T1.block("AA").iterate([&](const std::vector<size_t>& i, double& value) {
        t1b_norm_ -= value * value;
        value = 0.0;
    });

    // norms
    T1norm_ = std::sqrt(t1a_norm_ + t1b_norm_);
    t1a_norm_ = std::sqrt(t1a_norm_);
    t1b_norm_ = std::sqrt(t1b_norm_);
    T1rms_ = 0.0;

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG::update_t2_std() {
    T2max_ = 0.0, t2aa_norm_ = 0.0, t2ab_norm_ = 0.0, t2bb_norm_ = 0.0;

    /**
     * Update T2 using delta_t algorithm
     * T2(new) = T2(old) + DT2
     * DT2 = Hbar2(old) * (1 - exp(-s*d^2)) / d^2 - T2(old) * exp(-s * d^2)
     *       ------------------------------------   -----------------------
     *                      Step 1                          Step 2
     **/


    // Step 1: work on Hbar2 where DT2 is treated as intermediate

    timer t1("transform Hbar2 to semi-canonical basis");
    // transform Hbar2 to semi-canonical basis
    if (!semi_canonical_) {
        DT2_["klab"] = U_["ki"] * U_["lj"] * Hbar2_["ijab"];
        DT2_["kLaB"] = U_["ki"] * U_["LJ"] * Hbar2_["iJaB"];
        DT2_["KLAB"] = U_["KI"] * U_["LJ"] * Hbar2_["IJAB"];
        Hbar2_["ijcd"] = DT2_["ijab"] * U_["db"] * U_["ca"];
        Hbar2_["iJcD"] = DT2_["iJaB"] * U_["DB"] * U_["ca"];
        Hbar2_["IJCD"] = DT2_["IJAB"] * U_["DB"] * U_["CA"];
    }

    DT2_["ijab"] = Hbar2_["ijab"];
    DT2_["iJaB"] = Hbar2_["iJaB"];
    DT2_["IJAB"] = Hbar2_["IJAB"];
    t1.stop();

    timer t2("scale Hbar2 by renormalized denominator");
    // scale Hbar2 by renormalized denominator
    DT2_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                        Fa_[i[2]] - Fa_[i[3]]);
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                        Fa_[i[2]] - Fb_[i[3]]);
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                        Fb_[i[2]] - Fb_[i[3]]);
            }
        });
    t2.stop();

//  timer t3("copy renormalized Hbar2 to DT2");
//  // copy renormalized Hbar2 to DT2
//  DT2_["ijab"] = Hbar2_["ijab"];
//  DT2_["iJaB"] = Hbar2_["iJaB"];
//  DT2_["IJAB"] = Hbar2_["IJAB"];
//  t3.stop();

    // Step 2: work on T2 where Hbar2 is treated as intermediate

    timer t4("copy T2 to Hbar2");
    // copy T2 to Hbar2
    Hbar2_["ijab"] = T2_["ijab"];
    Hbar2_["iJaB"] = T2_["iJaB"];
    Hbar2_["IJAB"] = T2_["IJAB"];
    t4.stop();

    timer t5("transform T2 to semi-canonical basis");
    // transform T2 to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor temp =
            ambit::BlockedTensor::build(tensor_type_, "temp for T2 update", spin_cases({"hhpp"}));
        temp["klab"] = U_["ki"] * U_["lj"] * T2_["ijab"];
        temp["kLaB"] = U_["ki"] * U_["LJ"] * T2_["iJaB"];
        temp["KLAB"] = U_["KI"] * U_["LJ"] * T2_["IJAB"];
        T2_["ijcd"] = temp["ijab"] * U_["db"] * U_["ca"];
        T2_["iJcD"] = temp["iJaB"] * U_["DB"] * U_["ca"];
        T2_["IJCD"] = temp["IJAB"] * U_["DB"] * U_["CA"];
    }
    t5.stop();

    timer t6("scale T2 by delta exponential");
    // scale T2 by delta exponential
    T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
                       double& value) {
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
            value *=
                dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
        } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
            value *=
                dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
        } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
            value *=
                dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
        }
    });
    t6.stop();

    timer t7("minus the renormalized T2 from renormalized Hbar2");
    // minus the renormalized T2 from renormalized Hbar2
    DT2_["ijab"] -= T2_["ijab"];
    DT2_["iJaB"] -= T2_["iJaB"];
    DT2_["IJAB"] -= T2_["IJAB"];
    t7.stop();

    timer t8("zero internal amplitudes");
    // zero internal amplitudes
    for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
        DT2_.block(block).iterate([&](const std::vector<size_t>&, double& value) { value = 0.0; });
    }
    t8.stop();

    // compute RMS
    TIME_LINE(T2rms_ = DT2_.norm());

    timer t9("transform DT2 back to original basis");
    // transform DT2 back to original basis
    if (!semi_canonical_) {
        T2_["klab"] = U_["ik"] * U_["jl"] * DT2_["ijab"];
        T2_["kLaB"] = U_["ik"] * U_["JL"] * DT2_["iJaB"];
        T2_["KLAB"] = U_["IK"] * U_["JL"] * DT2_["IJAB"];
        DT2_["ijcd"] = T2_["ijab"] * U_["bd"] * U_["ac"];
        DT2_["iJcD"] = T2_["iJaB"] * U_["BD"] * U_["ac"];
        DT2_["IJCD"] = T2_["IJAB"] * U_["BD"] * U_["AC"];
    }
    t9.stop();

    timer t10("Step 3: update and analyze T2");
    // Step 3: update and analyze T2
    T2_["ijab"] = Hbar2_["ijab"];
    T2_["iJaB"] = Hbar2_["iJaB"];
    T2_["IJAB"] = Hbar2_["IJAB"];
    T2_["ijab"] += DT2_["ijab"];
    T2_["iJaB"] += DT2_["iJaB"];
    T2_["IJAB"] += DT2_["IJAB"];

    // compute norm and find maximum
    T2_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                t2aa_norm_ += value * value;
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                t2ab_norm_ += value * value;
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });

    T2norm_ = std::sqrt(t2aa_norm_ + t2bb_norm_ + 4 * t2ab_norm_);
    t2aa_norm_ = std::sqrt(t2aa_norm_);
    t2ab_norm_ = std::sqrt(t2ab_norm_);
    t2bb_norm_ = std::sqrt(t2bb_norm_);
    t10.stop();
}

void MRDSRG::update_t1_std() {
    T1max_ = 0.0, t1a_norm_ = 0.0, t1b_norm_ = 0.0;

    /**
     * Update T1 using delta_t algorithm
     * T1(new) = T1(old) + DT1
     * DT1 = Hbar1(old) * (1 - exp(-s*d^2)) / d^2 - T1(old) * exp(-s * d^2)
     *       ------------------------------------   -----------------------
     *                      Step 1                          Step 2
     **/

    // Step 1: work on Hbar1 where DT1 is treated as intermediate

    // transform Hbar1 to semi-canonical basis
    if (!semi_canonical_) {
        DT1_["jb"] = U_["ji"] * Hbar1_["ia"] * U_["ba"];
        DT1_["JB"] = U_["JI"] * Hbar1_["IA"] * U_["BA"];
        Hbar1_["ia"] = DT1_["ia"];
        Hbar1_["IA"] = DT1_["IA"];
    }

    DT1_["ia"] = Hbar1_["ia"];
    DT1_["IA"] = Hbar1_["IA"];

    // scale Hbar1 by renormalized denominator
    DT1_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] - Fa_[i[1]]);
            } else {
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] - Fb_[i[1]]);
            }
        });

//  // copy renormalized Hbar1 to DT1
//  DT1_["ia"] = Hbar1_["ia"];
//  DT1_["IA"] = Hbar1_["IA"];

    // Step 2: work on T1 where Hbar1 is treated as intermediate

    // copy T1 to Hbar1
    Hbar1_["ia"] = T1_["ia"];
    Hbar1_["IA"] = T1_["IA"];

    // transform T1 to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor temp =
            ambit::BlockedTensor::build(tensor_type_, "temp for T1 update", spin_cases({"hp"}));
        temp["jb"] = U_["ji"] * T1_["ia"] * U_["ba"];
        temp["JB"] = U_["JI"] * T1_["IA"] * U_["BA"];
        T1_["ia"] = temp["ia"];
        T1_["IA"] = temp["IA"];
    }

    // scale T1 by delta exponential
    T1_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value *= dsrg_source_->compute_renormalized(Fa_[i[0]] - Fa_[i[1]]);
            } else {
                value *= dsrg_source_->compute_renormalized(Fb_[i[0]] - Fb_[i[1]]);
            }
        });

    // minus the renormalized T1 from renormalized Hbar1
    DT1_["ia"] -= T1_["ia"];
    DT1_["IA"] -= T1_["IA"];

    // zero internal amplitudes
    for (const std::string& block : {"aa", "AA"}) {
        DT1_.block(block).iterate([&](const std::vector<size_t>&, double& value) { value = 0.0; });
    }

    // compute RMS
    T1rms_ = DT1_.norm();

    // transform DT1 back to original basis
    if (!semi_canonical_) {
        T1_["jb"] = U_["ji"] * DT1_["ia"] * U_["ba"];
        T1_["JB"] = U_["JI"] * DT1_["IA"] * U_["BA"];
        DT1_["ia"] = T1_["ia"];
        DT1_["IA"] = T1_["IA"];
    }

    // Step 3: update and analyze T1
    T1_["ia"] = Hbar1_["ia"];
    T1_["IA"] = Hbar1_["IA"];
    T1_["ia"] += DT1_["ia"];
    T1_["IA"] += DT1_["IA"];

    // compute norm and find maximum
    T1_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                t1a_norm_ += value * value;
            } else {
                t1b_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });

    T1norm_ = std::sqrt(t1a_norm_ + t1b_norm_);
    t1a_norm_ = std::sqrt(t1a_norm_);
    t1b_norm_ = std::sqrt(t1b_norm_);
}

void MRDSRG::update_t2_noccvv() {
    T2max_ = 0.0, t2aa_norm_ = 0.0, t2ab_norm_ = 0.0, t2bb_norm_ = 0.0;

    // create a temporary tensor
    BlockedTensor R2 = ambit::BlockedTensor::build(tensor_type_, "R2", spin_cases({"hhpp"}));
    R2["ijab"] = T2_["ijab"];
    R2["iJaB"] = T2_["iJaB"];
    R2["IJAB"] = T2_["IJAB"];
    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempR2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp R2", spin_cases({"hhpp"}));
        tempR2["klab"] = U_["ki"] * U_["lj"] * R2["ijab"];
        tempR2["kLaB"] = U_["ki"] * U_["LJ"] * R2["iJaB"];
        tempR2["KLAB"] = U_["KI"] * U_["LJ"] * R2["IJAB"];
        R2["ijcd"] = tempR2["ijab"] * U_["db"] * U_["ca"];
        R2["iJcD"] = tempR2["iJaB"] * U_["DB"] * U_["ca"];
        R2["IJCD"] = tempR2["IJAB"] * U_["DB"] * U_["CA"];

        BlockedTensor tempH2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp Hbar2", spin_cases({"hhpp"}));
        tempH2["klab"] = U_["ki"] * U_["lj"] * Hbar2_["ijab"];
        tempH2["kLaB"] = U_["ki"] * U_["LJ"] * Hbar2_["iJaB"];
        tempH2["KLAB"] = U_["KI"] * U_["LJ"] * Hbar2_["IJAB"];
        Hbar2_["ijcd"] = tempH2["ijab"] * U_["db"] * U_["ca"];
        Hbar2_["iJcD"] = tempH2["iJaB"] * U_["DB"] * U_["ca"];
        Hbar2_["IJCD"] = tempH2["IJAB"] * U_["DB"] * U_["CA"];
    }
    R2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
            value *= Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]];
        } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
            value *= Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]];
        } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
            value *= Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]];
        }
    });
    R2["ijab"] += Hbar2_["ijab"];
    R2["iJaB"] += Hbar2_["iJaB"];
    R2["IJAB"] += Hbar2_["IJAB"];

    // block labels
    std::vector<std::string> cv_blocks{acore_label_ + acore_label_ + avirt_label_ + avirt_label_,
                                       acore_label_ + bcore_label_ + avirt_label_ + bvirt_label_,
                                       bcore_label_ + bcore_label_ + bvirt_label_ + bvirt_label_};
    std::vector<std::string> other_blocks(R2.block_labels());
    other_blocks.erase(std::remove_if(other_blocks.begin(), other_blocks.end(),
                                      [&](std::string i) {
                                          return std::find(cv_blocks.begin(), cv_blocks.end(), i) !=
                                                 cv_blocks.end();
                                      }),
                       other_blocks.end());

    // map spin with Fock matrices
    std::map<bool, const std::vector<double>> Fock_spin{{true, Fa_}, {false, Fb_}};

    // ccvv blocks
    for (const std::string& block : cv_blocks) {
        // spin
        bool spin0 = islower(block[0]);
        bool spin1 = islower(block[1]);

        // diagonal Fock matrix elements
        const std::vector<double>& F0 = Fock_spin[spin0];
        const std::vector<double>& F1 = Fock_spin[spin1];

        R2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value /= F0[i0] + F1[i1] - F0[i2] - F0[i3];
            if (spin0 && spin1) {
                t2aa_norm_ += value * value;
            } else if (spin0 && !spin1) {
                t2ab_norm_ += value * value;
            } else if (!spin0 && !spin1) {
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // other blocks
    for (const std::string& block : other_blocks) {
        // spin
        bool spin0 = islower(block[0]);
        bool spin1 = islower(block[1]);

        // diagonal Fock matrix elements
        const std::vector<double>& F0 = Fock_spin[spin0];
        const std::vector<double>& F1 = Fock_spin[spin1];

        R2.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            size_t i2 = label_to_spacemo_[block[2]][i[2]];
            size_t i3 = label_to_spacemo_[block[3]][i[3]];
            value *=
                dsrg_source_->compute_renormalized_denominator(F0[i0] + F1[i1] - F0[i2] - F0[i3]);

            if (spin0 && spin1) {
                t2aa_norm_ += value * value;
            } else if (spin0 && !spin1) {
                t2ab_norm_ += value * value;
            } else if (!spin0 && !spin1) {
                t2bb_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempR2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp R2", spin_cases({"hhpp"}));
        tempR2["klab"] = U_["ik"] * U_["jl"] * R2["ijab"];
        tempR2["kLaB"] = U_["ik"] * U_["JL"] * R2["iJaB"];
        tempR2["KLAB"] = U_["IK"] * U_["JL"] * R2["IJAB"];
        R2["ijcd"] = tempR2["ijab"] * U_["bd"] * U_["ac"];
        R2["iJcD"] = tempR2["iJaB"] * U_["BD"] * U_["ac"];
        R2["IJCD"] = tempR2["IJAB"] * U_["BD"] * U_["AC"];
    }

    // zero internal amplitudes
    R2.block("aaaa").iterate([&](const std::vector<size_t>& i, double& value) {
        t2aa_norm_ -= value * value;
        value = 0.0;
    });
    R2.block("aAaA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2ab_norm_ -= value * value;
        value = 0.0;
    });
    R2.block("AAAA").iterate([&](const std::vector<size_t>& i, double& value) {
        t2bb_norm_ -= value * value;
        value = 0.0;
    });

    // compute RMS
    DT2_["ijab"] = T2_["ijab"] - R2["ijab"];
    DT2_["iJaB"] = T2_["iJaB"] - R2["iJaB"];
    DT2_["IJAB"] = T2_["IJAB"] - R2["IJAB"];
    T2rms_ = DT2_.norm();

    // copy R2 to T2
    T2_["ijab"] = R2["ijab"];
    T2_["iJaB"] = R2["iJaB"];
    T2_["IJAB"] = R2["IJAB"];

    // norms
    T2norm_ = std::sqrt(t2aa_norm_ + t2bb_norm_ + 4 * t2ab_norm_);
    t2aa_norm_ = std::sqrt(t2aa_norm_);
    t2ab_norm_ = std::sqrt(t2ab_norm_);
    t2bb_norm_ = std::sqrt(t2bb_norm_);
}

void MRDSRG::update_t1_nocv() {
    T1max_ = 0.0, t1a_norm_ = 0.0, t1b_norm_ = 0.0;

    // create a temporary tensor
    BlockedTensor R1 = ambit::BlockedTensor::build(tensor_type_, "R1", spin_cases({"hp"}));
    R1["ia"] = T1_["ia"];
    R1["IA"] = T1_["IA"];
    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempR1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp R1", spin_cases({"hp"}));
        tempR1["jb"] = U_["ji"] * R1["ia"] * U_["ba"];
        tempR1["JB"] = U_["JI"] * R1["IA"] * U_["BA"];
        R1["ia"] = tempR1["ia"];
        R1["IA"] = tempR1["IA"];

        BlockedTensor tempH1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp Hbar1", spin_cases({"hp"}));
        tempH1["jb"] = U_["ji"] * Hbar1_["ia"] * U_["ba"];
        tempH1["JB"] = U_["JI"] * Hbar1_["IA"] * U_["BA"];
        Hbar1_["ia"] = tempH1["ia"];
        Hbar1_["IA"] = tempH1["IA"];
    }
    R1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value *= Fa_[i[0]] - Fa_[i[1]];
        } else {
            value *= Fb_[i[0]] - Fb_[i[1]];
        }
    });
    R1["ia"] += Hbar1_["ia"];
    R1["IA"] += Hbar1_["IA"];

    // block labels
    std::vector<std::string> cv_blocks{acore_label_ + avirt_label_, bcore_label_ + bvirt_label_};
    std::vector<std::string> other_blocks(R1.block_labels());
    other_blocks.erase(std::remove_if(other_blocks.begin(), other_blocks.end(),
                                      [&](std::string i) {
                                          return std::find(cv_blocks.begin(), cv_blocks.end(), i) !=
                                                 cv_blocks.end();
                                      }),
                       other_blocks.end());

    // map spin with Fock matrices
    std::map<bool, const std::vector<double>> Fock_spin{{true, Fa_}, {false, Fb_}};

    // cv blocks
    for (const std::string& block : cv_blocks) {
        bool spin0 = islower(block[0]);
        const std::vector<double>& F0 = Fock_spin[spin0];

        R1.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            value /= F0[i0] - F0[i1];
            if (spin0) {
                t1a_norm_ += value * value;
            } else if (!spin0) {
                t1b_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });
    }

    // other blocks
    for (const std::string& block : other_blocks) {
        bool spin0 = islower(block[0]);
        const std::vector<double>& F0 = Fock_spin[spin0];

        R1.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            value *= dsrg_source_->compute_renormalized_denominator(F0[i0] - F0[i1]);

            if (spin0) {
                t1a_norm_ += value * value;
            } else if (!spin0) {
                t1b_norm_ += value * value;
            }
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempR1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp R1", spin_cases({"hp"}));
        tempR1["jb"] = U_["ij"] * R1["ia"] * U_["ab"];
        tempR1["JB"] = U_["IJ"] * R1["IA"] * U_["AB"];
        R1["ia"] = tempR1["ia"];
        R1["IA"] = tempR1["IA"];
    }

    // zero internal amplitudes
    R1.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
        t1a_norm_ -= value * value;
        value = 0.0;
    });
    R1.block("AA").iterate([&](const std::vector<size_t>& i, double& value) {
        t1b_norm_ -= value * value;
        value = 0.0;
    });

    // compute RMS
    DT1_["ia"] = T1_["ia"] - R1["ia"];
    DT1_["IA"] = T1_["IA"] - R1["IA"];
    T1rms_ = DT1_.norm();

    // copy R1 to T1
    T1_["ia"] = R1["ia"];
    T1_["IA"] = R1["IA"];

    // norms
    T1norm_ = std::sqrt(t1a_norm_ + t1b_norm_);
    t1a_norm_ = std::sqrt(t1a_norm_);
    t1b_norm_ = std::sqrt(t1b_norm_);
}

void MRDSRG::analyze_amplitudes(std::string name, BlockedTensor& T1, BlockedTensor& T2) {
    if (!name.empty())
        name += " ";
    outfile->Printf("\n\n  ==> %sExcitation Amplitudes Summary <==\n", name.c_str());
    outfile->Printf("\n    Active Indices: ");
    int c = 0;
    for (const auto& idx : aactv_mos_) {
        outfile->Printf("%4zu ", idx);
        if (++c % 10 == 0)
            outfile->Printf("\n    %16c", ' ');
    }
    check_t1(T1);
    check_t2(T2);

    outfile->Printf("\n\n  ==> Possible Intruders <==\n");
    print_intruder("A", lt1a_);
    print_intruder("B", lt1b_);
    print_intruder("AA", lt2aa_);
    print_intruder("AB", lt2ab_);
    print_intruder("BB", lt2bb_);
}

// Binary function to achieve sorting a vector of pair<vector, double>
// according to the double value in decending order
template <class T1, class T2, class G3 = std::greater<T2>> struct rsort_pair_second {
    bool operator()(const std::pair<T1, T2>& left, const std::pair<T1, T2>& right) {
        G3 p;
        return p(std::fabs(left.second), std::fabs(right.second));
    }
};

void MRDSRG::check_t2(BlockedTensor& T2) {
    size_t nonzero_aa = 0, nonzero_ab = 0, nonzero_bb = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t2aa, t2ab, t2bb;

    // create all knids of spin maps; 0: aa, 1: ab, 2:bb
    std::map<int, size_t> spin_to_nonzero;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t2;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt2;

    for (const std::string& block : T2.block_labels()) {
        int spin = static_cast<bool>(isupper(block[0])) + static_cast<bool>(isupper(block[1]));

        // create a reference to simplify the syntax
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t2 = spin_to_t2[spin];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt2 = spin_to_lt2[spin];

        T2.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];
                size_t idx2 = label_to_spacemo_[block[2]][i[2]];
                size_t idx3 = label_to_spacemo_[block[3]][i[3]];

                ++spin_to_nonzero[spin];

                if ((idx0 <= idx1) && (idx2 <= idx3)) {
                    std::vector<size_t> indices = {idx0, idx1, idx2, idx3};
                    std::pair<std::vector<size_t>, double> idx_value =
                        std::make_pair(indices, value);

                    temp_t2.push_back(idx_value);
                    std::sort(temp_t2.begin(), temp_t2.end(),
                              rsort_pair_second<std::vector<size_t>, double>());
                    if (temp_t2.size() == ntamp_ + 1) {
                        temp_t2.pop_back();
                    }

                    if (std::fabs(value) > std::fabs(intruder_tamp_)) {
                        temp_lt2.push_back(idx_value);
                    }
                    std::sort(temp_lt2.begin(), temp_lt2.end(),
                              rsort_pair_second<std::vector<size_t>, double>());
                }
            }
        });
    }

    // update values
    nonzero_aa = spin_to_nonzero[0];
    nonzero_ab = spin_to_nonzero[1];
    nonzero_bb = spin_to_nonzero[2];

    t2aa = spin_to_t2[0];
    t2ab = spin_to_t2[1];
    t2bb = spin_to_t2[2];

    lt2aa_ = spin_to_lt2[0];
    lt2ab_ = spin_to_lt2[1];
    lt2bb_ = spin_to_lt2[2];

    // print summary
    print_amp_summary("AA", t2aa, t2aa_norm_, nonzero_aa);
    print_amp_summary("AB", t2ab, t2ab_norm_, nonzero_ab);
    print_amp_summary("BB", t2bb, t2bb_norm_, nonzero_bb);
}

void MRDSRG::check_t1(BlockedTensor& T1) {
    size_t nonzero_a = 0, nonzero_b = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t1a, t1b;

    // create all kinds of spin maps; true: a, false: b
    std::map<bool, size_t> spin_to_nonzero;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t1;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt1;

    for (const std::string& block : T1.block_labels()) {
        bool spin_alpha = islower(block[0]) ? true : false;

        // create a reference to simplify the syntax
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t1 = spin_to_t1[spin_alpha];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt1 = spin_to_lt1[spin_alpha];

        T1.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];

                std::vector<size_t> indices = {idx0, idx1};
                std::pair<std::vector<size_t>, double> idx_value = std::make_pair(indices, value);

                ++spin_to_nonzero[spin_alpha];

                temp_t1.push_back(idx_value);
                std::sort(temp_t1.begin(), temp_t1.end(),
                          rsort_pair_second<std::vector<size_t>, double>());
                if (temp_t1.size() == ntamp_ + 1) {
                    temp_t1.pop_back();
                }

                if (std::fabs(value) > std::fabs(intruder_tamp_)) {
                    temp_lt1.push_back(idx_value);
                }
                std::sort(temp_lt1.begin(), temp_lt1.end(),
                          rsort_pair_second<std::vector<size_t>, double>());
            }
        });
    }

    // update value
    nonzero_a = spin_to_nonzero[true];
    nonzero_b = spin_to_nonzero[false];

    t1a = spin_to_t1[true];
    t1b = spin_to_t1[false];

    lt1a_ = spin_to_lt1[true];
    lt1b_ = spin_to_lt1[false];

    // print summary
    print_amp_summary("A", t1a, t1a_norm_, nonzero_a);
    print_amp_summary("B", t1b, t1b_norm_, nonzero_b);
}

void MRDSRG::print_amp_summary(const std::string& name,
                               const std::vector<std::pair<std::vector<size_t>, double>>& list,
                               const double& norm, const size_t& number_nonzero) {
    int rank = name.size();
    std::map<char, std::string> spin_case;
    spin_case['A'] = " ";
    spin_case['B'] = "_";

    std::string indent(4, ' ');
    std::string title =
        indent + "Largest T" + std::to_string(rank) + " amplitudes for spin case " + name + ":";
    std::string spin_title;
    std::string mo_title;
    std::string line;
    std::string output;
    std::string summary;

    auto extendstr = [&](std::string s, int n) {
        std::string o(s);
        while ((--n) > 0)
            o += s;
        return o;
    };

    if (rank == 1) {
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] % ' ' %
                          spin_case[name[0]] % ' ' % ' ');
        if (spin_title.find_first_not_of(' ') != std::string::npos) {
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        } else {
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % ' ' % 'a' % ' ' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for (size_t n = 0; n != list.size(); ++n) {
            if (n % 3 == 0)
                output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3c %3d %3c]%9.6f ") % idx[0] % ' ' % idx[1] % ' ' %
                          datapair.second);
        }
    } else if (rank == 2) {
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] %
                          spin_case[name[1]] % spin_case[name[0]] % spin_case[name[1]] % ' ');
        if (spin_title.find_first_not_of(' ') != std::string::npos) {
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        } else {
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % 'j' % 'a' % 'b' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for (size_t n = 0; n != list.size(); ++n) {
            if (n % 3 == 0)
                output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3d %3d %3d]%9.6f ") % idx[0] % idx[1] % idx[2] %
                          idx[3] % datapair.second);
        }
    } else {
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if (output.size() != 0) {
        int linesize = mo_title.size() - 2;
        line = "\n" + indent + std::string(linesize - indent.size(), '-');
        summary = "\n" + indent + "Norm of T" + std::to_string(rank) + name +
                  " vector: (nonzero elements: " + std::to_string(number_nonzero) + ")";
        std::string strnorm = str(boost::format("%.15f.") % norm);
        std::string blank(linesize - summary.size() - strnorm.size() + 1, ' ');
        summary += blank + strnorm;

        output = title + spin_title + mo_title + line + output + line + summary + line;
    } else {
        output = title + " NULL";
    }
    outfile->Printf("\n%s", output.c_str());
}

void MRDSRG::print_intruder(const std::string& name,
                            const std::vector<std::pair<std::vector<size_t>, double>>& list) {
    int rank = name.size();
    std::map<char, std::vector<double>> spin_to_F;
    spin_to_F['A'] = Fa_;
    spin_to_F['B'] = Fb_;

    std::string indent(4, ' ');
    std::string title = indent + "T" + std::to_string(rank) + " amplitudes larger than " +
                        str(boost::format("%.4f") % intruder_tamp_) + " for spin case " + name +
                        ":";
    std::string col_title;
    std::string line;
    std::string output;

    if (rank == 1) {
        int x = 30 + 2 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     " + blank + "Denominator" +
                     std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');

        for (size_t n = 0; n != list.size(); ++n) {
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], a = idx[1];
            double fi = spin_to_F[name[0]][i], fa = spin_to_F[name[0]][a];
            double down = fi - fa;
            double v = datapair.second;

            output += "\n" + indent +
                      str(boost::format("[%3d %3c %3d %3c] %13.8f (%10.6f - %10.6f = %10.6f)") % i %
                          ' ' % a % ' ' % v % fi % fa % down);
        }
    } else if (rank == 2) {
        int x = 50 + 4 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     " + blank + "Denominator" +
                     std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');
        for (size_t n = 0; n != list.size(); ++n) {
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], j = idx[1], a = idx[2], b = idx[3];
            double fi = spin_to_F[name[0]][i], fj = spin_to_F[name[1]][j];
            double fa = spin_to_F[name[0]][a], fb = spin_to_F[name[1]][b];
            double down = fi + fj - fa - fb;
            double v = datapair.second;

            output += "\n" + indent + str(boost::format("[%3d %3d %3d %3d] %13.8f (%10.6f + "
                                                        "%10.6f - %10.6f - %10.6f = %10.6f)") %
                                          i % j % a % b % v % fi % fj % fa % fb % down);
        }
    } else {
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if (output.size() != 0) {
        output = title + col_title + line + output + line;
    } else {
        output = title + " NULL";
    }
    outfile->Printf("\n%s", output.c_str());
}
}
}
