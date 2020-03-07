/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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
#include <cctype>
#include <map>
#include <memory>
#include <vector>

#include "psi4/libpsi4util/process.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libqt/qt.h"

#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sa_mrpt2.h"

using namespace psi;

namespace forte {

SA_MRPT2::SA_MRPT2(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : SADSRG(rdms, scf_info, options, ints, mo_space_info) {

    print_method_banner({"MR-DSRG Second-Order Perturbation Theory"});
    startup();
    read_options();
    print_options();
}

void SA_MRPT2::startup() {
    // test semi-canonical
    if (!semi_canonical_) {
        outfile->Printf("\n    Orbital invariant formalism will be employed for MR-DSRG.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", {"gg"});
        Fdiag_ = diagonalize_Fock_diagblocks(U_);
    }

    // link F_ with Fock_ of SADSRG
    F_ = Fock_;

    // prepare integrals
    build_ints();

    // initialize tensors for amplitudes
    init_amps();
}

void SA_MRPT2::read_options() {
    internal_amp_ = foptions_->get_str("INTERNAL_AMP");
    internal_amp_select_ = foptions_->get_str("INTERNAL_AMP_SELECT");
}

void SA_MRPT2::print_options() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info_int{
        {"Number of amplitudes for printing", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Taylor expansion threshold", std::pow(10.0, -double(taylor_threshold_))},
        {"Intruder amplitudes threshold", intruder_tamp_}};

    if (ints_type_ == "CHOLESKY") {
        auto cholesky_threshold = foptions_->get_double("CHOLESKY_TOLERANCE");
        calculation_info_double.push_back({"Cholesky tolerance", cholesky_threshold});
    }

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Integral type", ints_type_},
        {"Source operator", source_},
        {"Core-Virtual source type", ccvv_source_},
        {"Reference relaxation", relax_ref_},
        {"Internal amplitudes", internal_amp_}};

    if (multi_state_) {
        calculation_info_string.push_back({"State type", "multiple state"});
        calculation_info_string.push_back({"Multi-state type", multi_state_algorithm_});
    } else {
        calculation_info_string.push_back({"State type", "state specific"});
    }

    if (internal_amp_ != "NONE") {
        calculation_info_string.push_back({"Internal amplitudes selection", internal_amp_select_});
    }

    // Print some information
    print_h2("Calculation Information");
    for (auto& str_dim : calculation_info_int) {
        outfile->Printf("\n    %-40s %15d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

void SA_MRPT2::build_ints() {
    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->oei_a(i[0], i[1]);
    });

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        if (ints_type_ != "DISKDF") {
            B_ = BTF_->build(tensor_type_, "B 3-idx", {"Lph"});
            fill_three_index_ints(B_);
        }
        V_ = BTF_->build(tensor_type_, "V", {"vvaa", "aacc", "avca", "avac", "vaaa", "aaca"});
    } else {
        V_ = BTF_->build(tensor_type_, "V", {"pphh"});

        for (const std::string& block : V_.block_labels()) {
            auto mo_to_index = BTF_->get_mo_to_index();
            std::vector<size_t> i0 = mo_to_index[block.substr(0, 1)];
            std::vector<size_t> i1 = mo_to_index[block.substr(1, 1)];
            std::vector<size_t> i2 = mo_to_index[block.substr(2, 1)];
            std::vector<size_t> i3 = mo_to_index[block.substr(3, 1)];
            auto Vblock = ints_->aptei_ab_block(i0, i1, i2, i3);
            V_.block(block).copy(Vblock);
        }
    }
}

void SA_MRPT2::init_amps() {
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});
    if (!eri_df_) {
        std::vector<std::string> blocks{"aavv", "ccaa", "caav", "acav", "aava", "caaa", "aaaa"};
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", blocks);
        S2_ = BTF_->build(tensor_type_, "T2 Amplitudes", blocks);
    } else {
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
        S2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
    }
}

double SA_MRPT2::compute_energy() {
    // build amplitudes

    // compute energy
    double Ecorr = 0.0;

    return Ecorr;
}

void SA_MRPT2::compute_t2_df_minimal() {
    // ONLY these blocks are stored: aavv, ccaa, caav, acav, aava, caaa, aaaa

    // initialize T2 with V
    T2_["ijab"] = V_["abij"];

    // transform to semi-canonical basis
    BlockedTensor tempT2;
    if (!semi_canonical_) {
        tempT2 = ambit::BlockedTensor::build(tensor_type_, "TempT2", T2_.block_labels());
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2_["ijab"];
        T2_["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
    }

    // build T2
    T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double denom = Fdiag_[i[0]] + Fdiag_[i[1]] - Fdiag_[i[2]] - Fdiag_[i[3]];
        value *= dsrg_source_->compute_renormalized_denominator(denom);
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2_["ijab"];
        T2_["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
    }

    // internal amplitudes
    if (internal_amp_.find("DOUBLES") != std::string::npos) {
        // TODO: to be filled
    } else {
        T2_.block("aaaa").zero();
    }

    // form S2 = 2 * J - K
    // aavv, ccaa, caav, acav, aava, caaa, aaaa
    S2_["ijab"] = 2.0 * T2_["ijab"] - T2_["ijba"];
    S2_["muve"] = 2.0 * T2_["muve"] - T2_["umve"];
    S2_["umve"] = 2.0 * T2_["umve"] - T2_["muve"];
    S2_["uvex"] = 2.0 * T2_["uvex"] - T2_["vuex"];
}

void SA_MRPT2::compute_t2() {
    if (eri_df_) {
        compute_t2_df_minimal();
        return;
    }

    // initialize T2 with V
    T2_["ijab"] = V_["abij"];

    // transform to semi-canonical basis
    BlockedTensor tempT2;
    if (!semi_canonical_) {
        tempT2 = ambit::BlockedTensor::build(tensor_type_, "TempT2", T2_.block_labels());
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2_["ijab"];
        T2_["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
    }

    // block labels for ccvv blocks and others
    std::vector<std::string> T2blocks(T2_.block_labels());
    if (ccvv_source_ == "ZERO") {
        T2blocks.erase(std::remove(T2blocks.begin(), T2blocks.end(), "ccvv"), T2blocks.end());
        T2_.block("ccvv").iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = core_mos_[i[0]];
            size_t i1 = core_mos_[i[1]];
            size_t i2 = virt_mos_[i[2]];
            size_t i3 = virt_mos_[i[3]];
            value /= Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];
        });
    }

    // build T2
    for (const std::string& block : T2blocks) {
        T2_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
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
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2_["ijab"];
        T2_["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
    }

    // internal amplitudes
    if (internal_amp_.find("DOUBLES") != std::string::npos) {
        // TODO: to be filled
    } else {
        T2_.block("aaaa").zero();
    }

    // form 2 * J - K
    S2_["ijab"] = 2.0 * T2_["ijab"] - T2_["ijba"];
}

void SA_MRPT2::compute_t1() {
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aa"});
    temp["xu"] = L1_["xu"];

    // transform to semi-canonical basis
    BlockedTensor tempX;
    if (!semi_canonical_) {
        tempX = ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", {"aa"});
        tempX["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        temp["uv"] = tempX["uv"];
    }

    // scale by delta
    temp.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= Fdiag_[i[0]] - Fdiag_[i[1]];
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        tempX["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        temp["uv"] = tempX["uv"];
    }

    T1_["ia"] = F_["ia"];
    T1_["ia"] += temp["xu"] * T2_["iuax"];
    T1_["ia"] -= 0.5 * temp["xu"] * T2_["iuxa"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        tempX = ambit::BlockedTensor::build(tensor_type_, "Temp T1", {"hp"});
        tempX["jb"] = U_["ji"] * T1_["ia"] * U_["ba"];
        T1_["ia"] = tempX["ia"];
    }

    // labels for cv blocks and the rest blocks
    std::vector<std::string> T1blocks(T1_.block_labels());
    if (ccvv_source_ == "ZERO") {
        T1blocks.erase(std::remove(T1blocks.begin(), T1blocks.end(), "cv"), T1blocks.end());
        T1_.block("cv").iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = core_mos_[i[0]];
            size_t i1 = virt_mos_[i[1]];
            value /= Fdiag_[i0] - Fdiag_[i1];
        });
    }

    // build T1
    for (const std::string& block : T1blocks) {
        T1_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            size_t i0 = label_to_spacemo_[block[0]][i[0]];
            size_t i1 = label_to_spacemo_[block[1]][i[1]];
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i0] - Fdiag_[i1]);
        });
    }

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        tempX["jb"] = U_["ij"] * T1_["ia"] * U_["ab"];
        T1_["ia"] = tempX["ia"];
    }

    // internal amplitudes
    if (internal_amp_.find("SINGLES") != std::string::npos) {
        // TODO: to be filled
    } else {
        T1_.block("aa").zero();
    }
}

void SA_MRPT2::compute_hbar() {}

} // namespace forte
