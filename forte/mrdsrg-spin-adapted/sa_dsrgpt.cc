/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/disk_io.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "sa_dsrgpt.h"

using namespace psi;

namespace forte {

SA_DSRGPT::SA_DSRGPT(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : SADSRG(rdms, scf_info, options, ints, mo_space_info) {
    init_fock();
}

void SA_DSRGPT::read_options() { form_Hbar_ = (relax_ref_ != "NONE" || multi_state_); }

void SA_DSRGPT::print_options() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info_int{
        {"Number of amplitudes for printing", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Taylor expansion threshold", std::pow(10.0, -double(taylor_threshold_))},
        {"Intruder amplitudes threshold", intruder_tamp_}};

    if (ints_->integral_type() == Cholesky) {
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
        calculation_info_string.push_back({"State type", "MULTIPLE STATES"});
        calculation_info_string.push_back({"Multi-state type", multi_state_algorithm_});
    } else {
        calculation_info_string.push_back({"State type", "SINGLE STATE"});
    }

    if (internal_amp_ != "NONE") {
        calculation_info_string.push_back({"Internal amplitudes levels", internal_amp_});
        calculation_info_string.push_back({"Internal amplitudes selection", internal_amp_select_});
    }

    // Print some information
    print_selected_options("Computation Information", calculation_info_string, {},
                           calculation_info_double, calculation_info_int);
}

void SA_DSRGPT::init_fock() {
    // link F_ with Fock_ of SADSRG
    F_ = Fock_;

    F0th_ = BTF_->build(tensor_type_, "Fock 0th", diag_one_labels());
    F0th_["pq"] = F_["pq"];

    F1st_ = BTF_->build(tensor_type_, "Fock 1st", od_one_labels());
    F1st_["pq"] = F_["pq"];
}

void SA_DSRGPT::compute_t2_full() {
    timer t2("Compute complete T2");
    print_contents("Computing T2 amplitudes");

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
    internal_amps_T2(T2_);

    // form 2 * J - K
    S2_["ijab"] = 2.0 * T2_["ijab"] - T2_["ijba"];

    print_done(t2.stop());
}

void SA_DSRGPT::compute_t1() {
    timer t1("Compute T1");
    print_contents("Computing T1 amplitudes");

    // initialize T1 with F + [H0, A]
    T1_["ia"] = F_["ai"];
    T1_["ia"] += 0.5 * S2_["ivaw"] * F0th_["wu"] * L1_["uv"];
    T1_["ia"] -= 0.5 * S2_["iwau"] * F0th_["vw"] * L1_["uv"];

    // need to consider the S2 blocks that are not stored
    if (!S2_.is_block("cava")) {
        T1_["me"] += 0.5 * S2_["vmwe"] * F0th_["wu"] * L1_["uv"];
        T1_["me"] -= 0.5 * S2_["wmue"] * F0th_["vw"] * L1_["uv"];
    }

    // transform to semi-canonical basis
    BlockedTensor tempX;
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
    internal_amps_T1(T1_);

    print_done(t1.stop());
}

void SA_DSRGPT::renormalize_integrals(bool add) {
    // add R = H + [F, A] contributions to H
    timer rV("Renormalize V");
    print_contents("Renormalizing 2-body integrals");

    // to semicanonical orbitals
    BlockedTensor tempX;
    if (!semi_canonical_) {
        tempX = ambit::BlockedTensor::build(tensor_type_, "TempV", V_.block_labels());
        tempX["abkl"] = U_["ki"] * U_["lj"] * V_["abij"];
        V_["cdij"] = tempX["abij"] * U_["db"] * U_["ca"];
    }

    std::vector<std::string> Vblocks(V_.block_labels());
    if (ccvv_source_ == "ZERO") {
        Vblocks.erase(std::remove(Vblocks.begin(), Vblocks.end(), "vvcc"), Vblocks.end());
    }

    if (add) {
        for (const std::string& block : Vblocks) {
            V_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
                size_t i0 = label_to_spacemo_[block[0]][i[0]];
                size_t i1 = label_to_spacemo_[block[1]][i[1]];
                size_t i2 = label_to_spacemo_[block[2]][i[2]];
                size_t i3 = label_to_spacemo_[block[3]][i[3]];
                double denom = Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];
                value *= 1.0 + dsrg_source_->compute_renormalized(denom);
            });
        }
    } else {
        for (const std::string& block : Vblocks) {
            V_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
                size_t i0 = label_to_spacemo_[block[0]][i[0]];
                size_t i1 = label_to_spacemo_[block[1]][i[1]];
                size_t i2 = label_to_spacemo_[block[2]][i[2]];
                size_t i3 = label_to_spacemo_[block[3]][i[3]];
                double denom = Fdiag_[i0] + Fdiag_[i1] - Fdiag_[i2] - Fdiag_[i3];
                value *= dsrg_source_->compute_renormalized(denom);
            });
        }
    }

    // transform back if necessary
    if (!semi_canonical_) {
        tempX["abkl"] = U_["ik"] * U_["jl"] * V_["abij"];
        V_["cdij"] = tempX["abij"] * U_["bd"] * U_["ac"];
    }

    print_done(rV.stop());

    timer rF("Renormalize F");
    print_contents("Renormalizing 1-body integrals");

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ph"});
    temp["ai"] = F_["ai"];
    temp["ai"] += 0.5 * S2_["ivaw"] * F0th_["wu"] * L1_["uv"];
    temp["ai"] -= 0.5 * S2_["iwau"] * F0th_["vw"] * L1_["uv"];

    // need to consider the S2 blocks that are not stored
    if (!S2_.is_block("cava")) {
        temp["em"] += 0.5 * S2_["vmwe"] * F0th_["wu"] * L1_["uv"];
        temp["em"] -= 0.5 * S2_["wmue"] * F0th_["vw"] * L1_["uv"];
    }

    // to semicanonical basis
    if (!semi_canonical_) {
        tempX = ambit::BlockedTensor::build(tensor_type_, "TempV", {"ph"});
        tempX["bj"] = U_["ba"] * temp["ai"] * U_["ji"];
        temp["ai"] = tempX["ai"];
    }

    // scale by exp(-s * D^2)
    temp.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double denom = Fdiag_[i[0]] - Fdiag_[i[1]];
        value *= dsrg_source_->compute_renormalized(denom);
    });

    // transform back if necessary
    if (!semi_canonical_) {
        tempX["bj"] = U_["ab"] * temp["ai"] * U_["ij"];
        temp["ai"] = tempX["ai"];
    }

    if (add) {
        F_["ai"] += temp["ai"];
    } else {
        F_["ai"] = temp["ai"];
    }

    print_done(rF.stop());
}
} // namespace forte
