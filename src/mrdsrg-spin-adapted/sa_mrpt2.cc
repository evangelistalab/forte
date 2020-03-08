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
        calculation_info_string.push_back({"State type", "MULTIPLE STATES"});
        calculation_info_string.push_back({"Multi-state type", multi_state_algorithm_});
    } else {
        calculation_info_string.push_back({"State type", "SINGLE STATE"});
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
    timer t("Initialize integrals");

    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->oei_a(i[0], i[1]);
    });

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        std::vector<std::string> blocks{"vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"};
        V_ = BTF_->build(tensor_type_, "V", blocks);
        if (ints_type_ != "DISKDF") {
            B_ = BTF_->build(tensor_type_, "B 3-idx", {"Lph"});
            fill_three_index_ints(B_);
            V_["abij"] = B_["gai"] * B_["gbj"];
        } else {
            build_minimal_V();
        }
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

    // prepare Hbar
    if (relax_ref_ != "NONE" || multi_state_) {
        Hbar1_ = BTF_->build(tensor_type_, "1-body Hbar", {"aa"});
        Hbar2_ = BTF_->build(tensor_type_, "2-body Hbar", {"aaaa"});
        Hbar1_["uv"] = F_["uv"];
        Hbar2_["uvxy"] = V_["uvxy"];
    }

    t.stop();
}

void SA_MRPT2::build_minimal_V() {
    timer t("Build minimal V");

    // TODO: add memory check
    std::vector<std::string> Bblocks{"Lva", "Lac", "Laa"};

    auto B = ambit::BlockedTensor::build(tensor_type_, "B 3-idx", Bblocks);
    fill_three_index_ints(B);
    V_["abij"] = B["gai"] * B["gbj"];

    // the only block left is avac of V
    if (std::find(Bblocks.begin(), Bblocks.end(), "Lvc") == Bblocks.end()) {
        auto nQ = aux_mos_.size();
        auto nc = core_mos_.size();
        auto na = actv_mos_.size();
        auto nv = virt_mos_.size();

        auto& Vavac = V_.block("avac").data();
        auto dim2 = na * nc;
        auto dim1 = nv * dim2;

        for (size_t c = 0; c < nc; ++c) {
            auto Bsub = ambit::Tensor::build(tensor_type_, "Bsub PT2", {nQ, nv});
            Bsub.data() = ints_->three_integral_block(aux_mos_, virt_mos_, {core_mos_[c]}).data();

            auto Vsub = ambit::Tensor::build(tensor_type_, "Vsub PT2", {na, nv, na});
            Vsub("uev") = B.block("Laa")("guv") * Bsub("ge");

            Vsub.citerate([&](const std::vector<size_t>& i, const double& value) {
                Vavac[i[0] * dim1 + i[1] * dim2 + i[2] * nc + c] = value;
            });
        }
    }

    t.stop();
}

void SA_MRPT2::init_amps() {
    timer t("Initialize T1 and T2");
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});
    if (eri_df_) {
        std::vector<std::string> blocks{"aavv", "ccaa", "caav", "acav", "aava", "caaa", "aaaa"};
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", blocks);
        S2_ = BTF_->build(tensor_type_, "T2 Amplitudes", blocks);
    } else {
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
        S2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
    }
    t.stop();
}

double SA_MRPT2::compute_energy() {
    // build amplitudes
    compute_t2();
    compute_t1();
    analyze_amplitudes("First-Order", T1_, T2_);

    // scale the integrals
    renormalize_integrals();

    // compute energy
    double Ecorr = 0.0;

    double E_FT1 = H1_T1_C0(F_, T1_, 1.0, Ecorr);
    double E_FT2 = H1_T2_C0(F_, T2_, 1.0, Ecorr);

    double E_VT1 = H2_T1_C0(V_, T1_, 1.0, Ecorr);

    std::vector<double> E_VT2_comp;
    if (!eri_df_) {
        E_VT2_comp = H2_T2_C0(V_, T2_, S2_, 1.0, Ecorr);
    } else {
        E_VT2_comp = H2_T2_C0_T2small(V_, T2_, S2_);

        auto Eccvv = E_V_T2_CCVV();
        auto Ecavv = E_V_T2_CAVV();
        auto Eccav = E_V_T2_CCAV();

        E_VT2_comp[0] += Eccvv + Ecavv + Eccav;
        Ecorr += E_VT2_comp[0] + E_VT2_comp[1] + E_VT2_comp[2];

        if (print_ > 1) {
            outfile->Printf("\n  DF-PT2 CCVV energy: %22.15f", Eccvv);
            outfile->Printf("\n  DF-PT2 CAVV energy: %22.15f", Ecavv);
            outfile->Printf("\n  DF-PT2 CCAV energy: %22.15f", Eccav);
        }
    }
    double E_VT2 = E_VT2_comp[0] + E_VT2_comp[1] + E_VT2_comp[2];

    double Etotal = Ecorr + Eref_;
    Hbar0_ = Ecorr;

    // printing
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"< Phi_0 | [Fr, T1] | Phi_0 >", E_FT1});
    energy.push_back({"< Phi_0 | [Fr, T2] | Phi_0 >", E_FT2});
    energy.push_back({"< Phi_0 | [Vr, T1] | Phi_0 >", E_VT1});
    energy.push_back({"< Phi_0 | [Vr, T2] | Phi_0 >", E_VT2});
    energy.push_back({"  - [Vr, T2] L1 contribution", E_VT2_comp[0]});
    energy.push_back({"  - [Vr, T2] L2 contribution", E_VT2_comp[1]});
    energy.push_back({"  - [Vr, T2] L3 contribution", E_VT2_comp[2]});
    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Etotal});

    print_h2("DSRG-MRPT2 Energy Summary");
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

    // reference relaxation
    if (relax_ref_ != "NONE" || multi_state_) {
        compute_hbar();
    }

    return Etotal;
}

void SA_MRPT2::compute_t2_df_minimal() {
    timer t2min("Compute minimal T2");
    // ONLY these T2 blocks are stored: aavv, ccaa, caav, acav, aava, caaa, aaaa

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
    S2_["muve"] -= T2_["umve"];
    S2_["umve"] -= T2_["muve"];
    S2_["uvex"] -= T2_["vuex"];

    t2min.stop();
}

void SA_MRPT2::compute_t2() {
    timer t2("Compute T2");

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

    t2.stop();
}

void SA_MRPT2::compute_t1() {
    timer t1("Compute T1");

    // initialize T1 with F + [H0, A]
    T1_["ia"] = F_["ia"];
    T1_["ia"] += 0.5 * S2_["ivaw"] * F_["wu"] * L1_["uv"];
    T1_["ia"] -= 0.5 * S2_["iwau"] * F_["vw"] * L1_["uv"];

    // need to consider the S2 blocks that are not stored
    if (eri_df_) {
        T1_["me"] += 0.5 * S2_["vmwe"] * F_["wu"] * L1_["uv"];
        T1_["me"] -= 0.5 * S2_["wmue"] * F_["vw"] * L1_["uv"];
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
    if (internal_amp_.find("SINGLES") != std::string::npos) {
        // TODO: to be filled
    } else {
        T1_.block("aa").zero();
    }

    t1.stop();
}

void SA_MRPT2::renormalize_integrals() {
    // add R = H + [F, A] contributions to H

    timer rV("Renormalize V");

    // to semicanonical orbitals
    BlockedTensor tempX;
    if (!semi_canonical_) {
        tempX = ambit::BlockedTensor::build(tensor_type_, "TempV", V_.block_labels());
        tempX["klab"] = U_["ki"] * U_["lj"] * V_["ijab"];
        V_["ijcd"] = tempX["ijab"] * U_["db"] * U_["ca"];
    }

    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double denom = Fdiag_[i[0]] + Fdiag_[i[1]] - Fdiag_[i[2]] - Fdiag_[i[3]];
        value *= 1.0 + dsrg_source_->compute_renormalized(denom);
    });

    // transform back if necessary
    if (!semi_canonical_) {
        tempX["klab"] = U_["ik"] * U_["jl"] * V_["ijab"];
        V_["ijcd"] = tempX["ijab"] * U_["bd"] * U_["ac"];
    }

    rV.stop();

    timer rF("Renormalize F");

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ph"});
    temp["ai"] = F_["ai"];
    temp["ai"] += 0.5 * S2_["ivaw"] * F_["wu"] * L1_["uv"];
    temp["ai"] -= 0.5 * S2_["iwau"] * F_["vw"] * L1_["uv"];

    // need to consider the S2 blocks that are not stored
    if (eri_df_) {
        temp["em"] += 0.5 * S2_["vmwe"] * F_["wu"] * L1_["uv"];
        temp["em"] -= 0.5 * S2_["wmue"] * F_["vw"] * L1_["uv"];
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

    F_["ai"] += temp["ai"];

    rF.stop();
}

std::vector<ambit::Tensor> SA_MRPT2::init_tensor_vecs(int number_of_tensors) {
    std::vector<ambit::Tensor> out;
    out.reserve(number_of_tensors);
    return out;
}

double SA_MRPT2::E_V_T2_CCVV() {
    /**
     * Compute <[V, T2]> (C_2)^4 ccvv term
     * E = (em|fn) * [ 2 * (me|nf) - (mf|ne)] * [1 - exp(-2 * s * D^2)] / D
     *
     * Batching: for a given m and n, form B(ef) = Bm(L|e) * Bn(L|f)
     */

    timer t_ccvv("Compute CCVV energy term");
    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto nc = core_mos_.size();

    // TODO: need to check for memeory for these tensors
    // TODO: if orbitals not canonical

    int n_threads = n_threads_;

    // some tensors used for threading
    std::vector<ambit::Tensor> Bm_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> Bn_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> J_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> JK_vec = init_tensor_vecs(n_threads);

    for (int i = 0; i < n_threads; i++) {
        std::string t = std::to_string(i);
        Bm_vec.push_back(ambit::Tensor::build(tensor_type_, "Bm_thread" + t, {nQ, nv}));
        Bn_vec.push_back(ambit::Tensor::build(tensor_type_, "Bn_thread" + t, {nQ, nv}));
        J_vec.push_back(ambit::Tensor::build(tensor_type_, "J_thread" + t, {nv, nv}));
        JK_vec.push_back(ambit::Tensor::build(tensor_type_, "(2J - K) thread" + t, {nv, nv}));
    }

    double E = 0.0;
    bool complete_ccvv = (ccvv_source_ == "ZERO");

#pragma omp parallel for num_threads(n_threads) reduction(+ : E)
    for (size_t m = 0; m < nc; ++m) {
        auto im = core_mos_[m];
        double Fm = Fdiag_[im];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
#pragma omp critical
        { Bm_vec[thread].data() = ints_->three_integral_block(aux_mos_, virt_mos_, {im}).data(); }

        for (size_t n = m; n < nc; ++n) {
            auto in = core_mos_[n];
            double Fn = Fdiag_[in];
            double factor = (m < n) ? 2.0 : 1.0;

#pragma omp critical
            {
                Bn_vec[thread].data() =
                    ints_->three_integral_block(aux_mos_, virt_mos_, {in}).data();
            }

            J_vec[thread]("ef") = Bm_vec[thread]("ge") * Bn_vec[thread]("gf");
            JK_vec[thread]("ef") = 2.0 * J_vec[thread]("ef") - J_vec[thread]("fe");

            J_vec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double D = Fm + Fn - Fdiag_[virt_mos_[i[0]]] - Fdiag_[virt_mos_[i[1]]];
                if (complete_ccvv)
                    value /= D;
                else {
                    double rV = 1.0 + dsrg_source_->compute_renormalized(D);
                    value *= dsrg_source_->compute_renormalized_denominator(D) * rV;
                }
            });

            E += factor * J_vec[thread]("ef") * JK_vec[thread]("ef");
        }
    }

    t_ccvv.stop();
    return E;
}

double SA_MRPT2::E_V_T2_CAVV() {
    timer t("Compute [V, T2] CAVV energy term");

    auto na = actv_mos_.size();
    auto C1 = ambit::Tensor::build(tensor_type_, "C1 VT2 CAVV", {na, na});

    compute_Hbar1V_diskDF(C1, true);

    double E = C1("vu") * L1_.block("aa")("uv");

    t.stop();
    return E;
}

void SA_MRPT2::compute_Hbar1V_diskDF(ambit::Tensor& Hbar1, bool Vr) {
    /**
     * Compute Hbar1["vu"] += V["efmu"] * S["mvef"]
     *
     * - if Vr is false: V["efmu"] = B(L|em) * B(L|fu)
     * - if Vr is true: V["efmu"] = B(L|em) * B(L|fu) * [1 + exp(-s * D1^2)]
     * - S["mvef"] = [2 * (me|vf) - (mf|ve)] * [1 - exp(-s * D2^2)] / D2
     *
     * where the two denominators are:
     *   D1 = F_m + F_u - F_e - F_f
     *   D2 = F_m + F_v - F_e - F_f
     *
     * Batching: for a given m, form V(efu) and S(efv)
     */

    timer t("Compute C1 virtual contraction");
    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto nc = core_mos_.size();
    auto na = actv_mos_.size();

    // TODO: need to check memory for these tensors
    // TODO: if orbitals not canonical

    int n_threads = n_threads_;

    // some tensors used for threading
    std::vector<ambit::Tensor> Bm_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> V_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> S_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> C_vec = init_tensor_vecs(n_threads);

    // TODO: test indices permutations for speed
    for (int i = 0; i < n_threads; i++) {
        std::string t = std::to_string(i);
        Bm_vec.push_back(ambit::Tensor::build(tensor_type_, "Bm_thread" + t, {nQ, nv}));
        V_vec.push_back(ambit::Tensor::build(tensor_type_, "V_thread" + t, {nv, nv, na}));
        S_vec.push_back(ambit::Tensor::build(tensor_type_, "S_thread" + t, {nv, nv, na}));
        C_vec.push_back(ambit::Tensor::build(tensor_type_, "C_thread" + t, {na, na}));
    }

    auto Bva = ints_->three_integral_block(aux_mos_, virt_mos_, actv_mos_);

#pragma omp parallel for num_threads(n_threads)
    for (size_t m = 0; m < nc; ++m) {
        auto im = core_mos_[m];
        double Fm = Fdiag_[im];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
#pragma omp critical
        { Bm_vec[thread].data() = ints_->three_integral_block(aux_mos_, virt_mos_, {im}).data(); }

        V_vec[thread]("efu") = Bm_vec[thread]("ge") * Bva("gfu");
        S_vec[thread]("efu") = 2.0 * V_vec[thread]("efu") - V_vec[thread]("feu");

        // scale V by 1 + exp(-s * D^2)
        if (Vr) {
            V_vec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = Fm + Fdiag_[actv_mos_[i[2]]] - Fdiag_[virt_mos_[i[0]]] -
                               Fdiag_[virt_mos_[i[1]]];
                value *= 1.0 + dsrg_source_->compute_renormalized(denom);
            });
        }

        // scale T by [1 - exp(-s * D^2)] / D
        S_vec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
            double denom =
                Fm + Fdiag_[actv_mos_[i[2]]] - Fdiag_[virt_mos_[i[0]]] - Fdiag_[virt_mos_[i[1]]];
            value *= dsrg_source_->compute_renormalized_denominator(denom);
        });

        C_vec[thread]("vu") += V_vec[thread]("efu") * S_vec[thread]("efv");
    }

    // finalize results
    for (int thread = 0; thread < n_threads; thread++) {
        Hbar1("vu") += C_vec[thread]("vu");
    }

    t.stop();
}

double SA_MRPT2::E_V_T2_CCAV() {
    timer t("Compute [V, T2] CCAV energy term");

    auto na = actv_mos_.size();
    auto C1 = ambit::Tensor::build(tensor_type_, "C1 VT2 CCAV", {na, na});

    compute_Hbar1C_diskDF(C1, true);

    double E = C1("vu") * Eta1_.block("aa")("uv");

    t.stop();
    return E;
}

void SA_MRPT2::compute_Hbar1C_diskDF(ambit::Tensor& Hbar1, bool Vr) {
    /**
     * Compute Hbar1["vu"] += V["vemn"] * S["mnue"]
     *
     * - if Vr is false: V["vemn"] = B(L|vm) * B(L|en)
     * - if Vr is true: V["vemn"] = B(L|vm) * B(L|en) * [1 + exp(-s * D1^2)]
     * - S["mnue"] = [2 * (mu|ne) - (mu|en)] * [1 - exp(-s * D2^2)] / D2
     *
     * where the two denominators are:
     *   D1 = F_m + F_n - F_e - F_v
     *   D2 = F_m + F_n - F_e - F_v
     *
     * Batching: for a given e, form V(vmn) and S(umn)
     */

    timer t("Compute C1 core contraction");
    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto nc = core_mos_.size();
    auto na = actv_mos_.size();

    // TODO: if orbitals not canonical

    // some tensors used for threading
    std::vector<ambit::Tensor> Be_vec = init_tensor_vecs(n_threads_);
    std::vector<ambit::Tensor> V_vec = init_tensor_vecs(n_threads_);
    std::vector<ambit::Tensor> S_vec = init_tensor_vecs(n_threads_);
    std::vector<ambit::Tensor> C_vec = init_tensor_vecs(n_threads_);

    // TODO: test indices permutations for speed
    for (int i = 0; i < n_threads_; i++) {
        std::string t = std::to_string(i);
        Be_vec.push_back(ambit::Tensor::build(tensor_type_, "Bm_thread" + t, {nQ, nc}));
        V_vec.push_back(ambit::Tensor::build(tensor_type_, "V_thread" + t, {na, nc, nc}));
        S_vec.push_back(ambit::Tensor::build(tensor_type_, "S_thread" + t, {na, nc, nc}));
        C_vec.push_back(ambit::Tensor::build(tensor_type_, "C_thread" + t, {na, na}));
    }

    auto Bac = ints_->three_integral_block(aux_mos_, actv_mos_, core_mos_);

#pragma omp parallel for num_threads(n_threads)
    for (size_t e = 0; e < nv; ++e) {
        auto ie = virt_mos_[e];
        double Fe = Fdiag_[ie];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
#pragma omp critical
        { Be_vec[thread].data() = ints_->three_integral_block(aux_mos_, core_mos_, {ie}).data(); }

        V_vec[thread]("vmn") = Bac("gvm") * Be_vec[thread]("gn");
        S_vec[thread]("vmn") = 2.0 * V_vec[thread]("vmn") - V_vec[thread]("vnm");

        // scale V by 1 + exp(-s * D^2)
        if (Vr) {
            V_vec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = Fdiag_[core_mos_[i[1]]] + Fdiag_[core_mos_[i[2]]] - Fe -
                               Fdiag_[actv_mos_[i[0]]];
                value *= 1.0 + dsrg_source_->compute_renormalized(denom);
            });
        }

        // scale T by [1 - exp(-s * D^2)] / D
        S_vec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
            double denom =
                Fdiag_[core_mos_[i[1]]] + Fdiag_[core_mos_[i[2]]] - Fe - Fdiag_[actv_mos_[i[0]]];
            value *= dsrg_source_->compute_renormalized_denominator(denom);
        });

        C_vec[thread]("vu") += V_vec[thread]("vmn") * S_vec[thread]("umn");
    }

    // finalize results
    for (int thread = 0; thread < n_threads_; thread++) {
        Hbar1("vu") += C_vec[thread]("vu");
    }

    t.stop();
}

void SA_MRPT2::compute_hbar() {}

} // namespace forte
