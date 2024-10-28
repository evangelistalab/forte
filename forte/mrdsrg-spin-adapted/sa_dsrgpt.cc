/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER,
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

SA_DSRGPT::SA_DSRGPT(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : SADSRG(rdms, scf_info, options, ints, mo_space_info) {
    init_fock();
}

void SA_DSRGPT::read_options() { form_Hbar_ = (relax_ref_ != "NONE" || multi_state_); }

void SA_DSRGPT::print_options() {
    // Print a summary
    table_printer printer;
    printer.add_int_data({{"Number of amplitudes for printing", ntamp_}});

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Taylor expansion threshold", std::pow(10.0, -double(taylor_threshold_))},
        {"Intruder amplitudes threshold", intruder_tamp_}};

    if (ints_->integral_type() == Cholesky) {
        auto cholesky_threshold = foptions_->get_double("CHOLESKY_TOLERANCE");
        calculation_info_double.push_back({"Cholesky tolerance", cholesky_threshold});
    }
    printer.add_double_data(calculation_info_double);

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Integral type", ints_type_},
        {"Source operator", source_},
        {"Core-Virtual source type", ccvv_source_},
        {"Reference relaxation", relax_ref_},
        {"3RDM algorithm", L3_algorithm_},
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

    printer.add_string_data(calculation_info_string);
    std::string table = printer.get_table("Calculation Information");
    psi::outfile->Printf("%s", table.c_str());
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
        apply_denominator(T2_, {"ccvv"}, [&](double d) { return 1.0 / d; });
    }

    // build T2
    apply_denominator(T2_, T2blocks,
                      [&](double v) { return dsrg_source_->compute_renormalized_denominator(v); });

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
        apply_denominator(T1_, {"cv"}, [](double d) { return 1.0 / d; });
    }

    // build T1
    apply_denominator(T1_, T1blocks,
                      [&](double d) { return dsrg_source_->compute_renormalized_denominator(d); });

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
        apply_denominator(V_, Vblocks,
                          [&](double d) { return 1.0 + dsrg_source_->compute_renormalized(d); });
    } else {
        apply_denominator(V_, Vblocks,
                          [&](double d) { return dsrg_source_->compute_renormalized(d); });
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
    apply_denominator(temp, temp.block_labels(),
                      [&](double d) { return dsrg_source_->compute_renormalized(d); });

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
