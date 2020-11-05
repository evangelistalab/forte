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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "forte-def.h"
#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sa_mrpt2.h"

using namespace psi;

namespace forte {

SA_MRPT2::SA_MRPT2(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : SA_DSRGPT(rdms, scf_info, options, ints, mo_space_info) {

    print_method_banner({"MR-DSRG Second-Order Perturbation Theory"});
    read_options();
    print_options();
    check_memory();
    startup();
}

void SA_MRPT2::startup() {
    // test semi-canonical
    if (!semi_canonical_) {
        outfile->Printf("\n  Orbital invariant formalism will be employed for DSRG-MRPT2.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", {"gg"});
        Fdiag_ = diagonalize_Fock_diagblocks(U_);
    }

    // prepare integrals
    build_ints();

    // initialize tensors for amplitudes
    init_amps();
}

void SA_MRPT2::build_ints() {
    timer t("Initialize integrals");

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        std::vector<std::string> blocks{"vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"};
        V_ = BTF_->build(tensor_type_, "V", blocks);
        if (ints_type_ != "DISKDF") {
            auto B = BTF_->build(tensor_type_, "B 3-idx", {"Lph"});
            fill_three_index_ints(B);
            V_["abij"] = B["gai"] * B["gbj"];
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
    if (form_Hbar_) {
        Hbar1_ = BTF_->build(tensor_type_, "1-body Hbar", {"aa"});
        Hbar2_ = BTF_->build(tensor_type_, "2-body Hbar", {"aaaa"});
        Hbar1_["uv"] = F_["uv"];
        Hbar2_["uvxy"] = V_["uvxy"];
    }

    t.stop();
}

void SA_MRPT2::build_minimal_V() {
    timer t("Build minimal V");

    std::vector<std::string> Bblocks{"Lva", "Lac", "Laa"};

    auto B = ambit::BlockedTensor::build(tensor_type_, "B 3-idx", Bblocks);
    fill_three_index_ints(B);
    V_["abij"] = B["gai"] * B["gbj"];

    // the only block left is avac of V
    auto nQ = aux_mos_.size();
    auto nc = core_mos_.size();
    auto na = actv_mos_.size();
    auto nv = virt_mos_.size();

    auto& Vavac = V_.block("avac").data();
    auto dim2 = na * nc;
    auto dim1 = nv * dim2;

    auto Bsub = ambit::Tensor::build(tensor_type_, "Bsub PT2", {nQ, nv});
    auto Vsub = ambit::Tensor::build(tensor_type_, "Vsub PT2", {na, nv, na});

    for (size_t c = 0; c < nc; ++c) {
        Bsub.data() = ints_->three_integral_block(aux_mos_, virt_mos_, {core_mos_[c]}).data();

        Vsub("uev") = B.block("Laa")("guv") * Bsub("ge");

        Vsub.citerate([&](const std::vector<size_t>& i, const double& value) {
            Vavac[i[0] * dim1 + i[1] * dim2 + i[2] * nc + c] = value;
        });
    }

    t.stop();
}

void SA_MRPT2::init_amps() {
    timer t("Initialize T1 and T2");

    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});

    if (eri_df_) {
        std::vector<std::string> blocks{"aavv", "ccaa", "caav", "acav", "aava", "caaa", "aaaa"};
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", blocks);
        S2_ = BTF_->build(tensor_type_, "S2 Amplitudes", blocks);
    } else {
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
        S2_ = BTF_->build(tensor_type_, "S2 Amplitudes", {"hhpp"});
    }

    t.stop();
}

void SA_MRPT2::check_memory() {
    // memory of ints and amps
    if (eri_df_) {
        std::vector<std::string> blocks{"vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"};
        auto mem_blocks = dsrg_mem_.compute_memory(blocks);
        dsrg_mem_.add_entry("2-electron (4-index) integrals", mem_blocks);
        dsrg_mem_.add_entry("T2 cluster amplitudes", 2 * mem_blocks);
        if (ints_type_ != "DISKDF") {
            dsrg_mem_.add_entry("3-index auxiliary integrals", {"Lph"});
        }
    } else {
        dsrg_mem_.add_entry("2-electron (4-index) integrals", {"pphh"});
        dsrg_mem_.add_entry("T2 cluster amplitudes", {"pphh"});
    }
    dsrg_mem_.add_entry("T1 cluster amplitudes", {"hp"});

    if (form_Hbar_) {
        dsrg_mem_.add_entry("1- and 2-body Hbar", {"aa", "aaaa"});
    }

    // local memory for computing minimal V
    if (ints_type_ == "DISKDF") {
        dsrg_mem_.add_entry("Local 3-index integrals", {"Lca", "Laa", "Lav"}, 1, false);
    }

    // compute energy
    if (eri_df_) {
        auto size_Lv = dsrg_mem_.compute_memory({"Lv"});
        auto size_acc = dsrg_mem_.compute_memory({"acc"});

        auto mem_ccvv = 2 * (size_Lv + dsrg_mem_.compute_memory({"vv"}));
        auto mem_cavv = size_Lv + dsrg_mem_.compute_memory({"vva", "vva", "aa", "Lva"});
        auto mem_ccav = 2 * size_acc + dsrg_mem_.compute_memory({"Lc", "aa", "Lac"});

        if (!semi_canonical_) {
            mem_ccvv += 2 * size_Lv;
            mem_cavv += size_Lv + dsrg_mem_.compute_memory({"Lva"});
            mem_ccav += dsrg_mem_.compute_memory({"Lc", "Lac"});
        }

        mem_batched_["ccvv"] = mem_ccvv;
        dsrg_mem_.add_entry("Local integrals for CCVV energy", mem_ccvv, false);

        mem_batched_["cavv"] = mem_cavv;
        dsrg_mem_.add_entry("Local integrals for CAVV energy", mem_cavv, false);

        mem_batched_["ccav"] = mem_ccav;
        dsrg_mem_.add_entry("Local integrals for CCAV energy", mem_ccav, false);
    } else {
        dsrg_mem_.add_entry("Local 1- and 2-body intermediates", {"aa", "aaaa"}, 1, false);
    }

    // compute Hbar
    if (form_Hbar_) {
        dsrg_mem_.add_entry("Local integrals for forming Hbar",
                            {"avac", "aaac", "avaa", "paaa", "aaaa"}, 1, false);
    }

    dsrg_mem_.print("DSRG-MRPT2");
}

double SA_MRPT2::compute_energy() {
    // build amplitudes
    compute_t2();
    compute_t1();
    analyze_amplitudes("First-Order", T1_, T2_);

    // scale the integrals
    renormalize_integrals(true);

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
    for (const auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

    // reference relaxation
    if (form_Hbar_) {
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
    internal_amps_T2(T2_);

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
    } else {
        compute_t2_full();
    }

    t2.stop();
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

    // check memory
    int n_threads = n_threads_;
    while (n_threads * mem_batched_["ccvv"] > dsrg_mem_.available()) {
        n_threads -= 1;
    }
    if (n_threads != n_threads_) {
        outfile->Printf("\n  Use %d threads to compute CCVV energy due to memory shortage.",
                        n_threads);
    }

    // some tensors used for threading
    std::vector<ambit::Tensor> Bm_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> Bn_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> J_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> JK_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> Xm_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> Xn_vec = init_tensor_vecs(n_threads);

    for (int i = 0; i < n_threads; i++) {
        std::string t = std::to_string(i);
        Bm_vec.push_back(ambit::Tensor::build(tensor_type_, "Bm_thread" + t, {nQ, nv}));
        Bn_vec.push_back(ambit::Tensor::build(tensor_type_, "Bn_thread" + t, {nQ, nv}));
        J_vec.push_back(ambit::Tensor::build(tensor_type_, "J_thread" + t, {nv, nv}));
        JK_vec.push_back(ambit::Tensor::build(tensor_type_, "(2J - K) thread" + t, {nv, nv}));
    }
    if (!semi_canonical_) {
        for (int i = 0; i < n_threads; i++) {
            std::string t = std::to_string(i);
            Xm_vec.push_back(ambit::Tensor::build(tensor_type_, "Xm_thread" + t, {nQ, nv}));
            Xn_vec.push_back(ambit::Tensor::build(tensor_type_, "Xn_thread" + t, {nQ, nv}));
        }
    }

    double E = 0.0;
    bool complete_ccvv = (ccvv_source_ == "ZERO");

#pragma omp parallel for num_threads(n_threads) reduction(+ : E)
    for (size_t m = 0; m < nc; ++m) {
        auto im = core_mos_[m];
        double Fm = Fdiag_[im];

        int thread = omp_get_thread_num();
#pragma omp critical
        {
            Bm_vec[thread].data() = ints_->three_integral_block(aux_mos_, virt_mos_, {im}).data();
            if (!semi_canonical_) {
                Xm_vec[thread]("gf") = Bm_vec[thread]("ge") * U_.block("vv")("fe");
                Bm_vec[thread]("gf") = Xm_vec[thread]("gf");
            }
        }

        for (size_t n = m; n < nc; ++n) {
            auto in = core_mos_[n];
            double Fn = Fdiag_[in];
            double factor = (m < n) ? 2.0 : 1.0;

#pragma omp critical
            {
                Bn_vec[thread].data() =
                    ints_->three_integral_block(aux_mos_, virt_mos_, {in}).data();
                if (!semi_canonical_) {
                    Xn_vec[thread]("gf") = Bn_vec[thread]("ge") * U_.block("vv")("fe");
                    Bn_vec[thread]("gf") = Xn_vec[thread]("gf");
                }
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

    // check memory
    int n_threads = n_threads_;
    while (n_threads * mem_batched_["cavv"] > dsrg_mem_.available()) {
        n_threads -= 1;
    }
    if (n_threads != n_threads_) {
        outfile->Printf("\n  Use %d threads to compute CAVV energy due to memory shortage.",
                        n_threads);
    }

    // some tensors used for threading
    std::vector<ambit::Tensor> Bm_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> V_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> S_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> C_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> X_vec = init_tensor_vecs(n_threads);

    // TODO: test indices permutations for speed
    for (int i = 0; i < n_threads; i++) {
        std::string t = std::to_string(i);
        Bm_vec.push_back(ambit::Tensor::build(tensor_type_, "Bm_thread" + t, {nQ, nv}));
        V_vec.push_back(ambit::Tensor::build(tensor_type_, "V_thread" + t, {nv, nv, na}));
        S_vec.push_back(ambit::Tensor::build(tensor_type_, "S_thread" + t, {nv, nv, na}));
        C_vec.push_back(ambit::Tensor::build(tensor_type_, "C_thread" + t, {na, na}));
    }
    if (!semi_canonical_) {
        for (int i = 0; i < n_threads; i++) {
            std::string t = std::to_string(i);
            X_vec.push_back(ambit::Tensor::build(tensor_type_, "X_thread" + t, {nQ, nv}));
        }
    }

    auto Bva = ints_->three_integral_block(aux_mos_, virt_mos_, actv_mos_);
    if (!semi_canonical_) {
        auto X = ambit::Tensor::build(tensor_type_, "tempCAVV", {nQ, nv, na});
        X("gev") = Bva("geu") * U_.block("aa")("vu");
        Bva("gfv") = X("gev") * U_.block("vv")("fe");
    }

#pragma omp parallel for num_threads(n_threads)
    for (size_t m = 0; m < nc; ++m) {
        auto im = core_mos_[m];
        double Fm = Fdiag_[im];

        int thread = omp_get_thread_num();
#pragma omp critical
        {
            Bm_vec[thread].data() = ints_->three_integral_block(aux_mos_, virt_mos_, {im}).data();
            if (!semi_canonical_) {
                X_vec[thread]("gf") = Bm_vec[thread]("ge") * U_.block("vv")("fe");
                Bm_vec[thread]("gf") = X_vec[thread]("gf");
            }
        }

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

        C_vec[thread]("vu") += S_vec[thread]("efv") * V_vec[thread]("efu");
    }

    // finalize results
    auto C = ambit::Tensor::build(tensor_type_, "C1total_CAVV", {na, na});
    for (int thread = 0; thread < n_threads; thread++) {
        C("vu") += C_vec[thread]("vu");
    }

    // rotate back to original orbital basis
    if (!semi_canonical_) {
        auto X = ambit::Tensor::build(tensor_type_, "tempCAVV", {na, na});
        X("xv") = C("uv") * U_.block("aa")("ux");
        C("xy") = X("xv") * U_.block("aa")("vy");
    }

    Hbar1("uv") += C("uv");

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

    // check memory
    int n_threads = n_threads_;
    while (n_threads * mem_batched_["ccav"] > dsrg_mem_.available()) {
        n_threads -= 1;
    }
    if (n_threads != n_threads_) {
        outfile->Printf("\n  Use %d threads to compute CCAV energy due to memory shortage.",
                        n_threads);
    }

    // some tensors used for threading
    std::vector<ambit::Tensor> Be_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> V_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> S_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> C_vec = init_tensor_vecs(n_threads);
    std::vector<ambit::Tensor> X_vec = init_tensor_vecs(n_threads);

    // TODO: test indices permutations for speed
    for (int i = 0; i < n_threads; i++) {
        std::string t = std::to_string(i);
        Be_vec.push_back(ambit::Tensor::build(tensor_type_, "Bm_thread" + t, {nQ, nc}));
        V_vec.push_back(ambit::Tensor::build(tensor_type_, "V_thread" + t, {na, nc, nc}));
        S_vec.push_back(ambit::Tensor::build(tensor_type_, "S_thread" + t, {na, nc, nc}));
        C_vec.push_back(ambit::Tensor::build(tensor_type_, "C_thread" + t, {na, na}));
    }
    if (!semi_canonical_) {
        for (int i = 0; i < n_threads; i++) {
            std::string t = std::to_string(i);
            X_vec.push_back(ambit::Tensor::build(tensor_type_, "X_thread" + t, {nQ, nc}));
        }
    }

    auto Bac = ints_->three_integral_block(aux_mos_, actv_mos_, core_mos_);
    if (!semi_canonical_) {
        auto X = ambit::Tensor::build(tensor_type_, "tempCCAV", {nQ, na, nc});
        X("gun") = Bac("gum") * U_.block("cc")("nm");
        Bac("gvn") = X("gun") * U_.block("aa")("vu");
    }

#pragma omp parallel for num_threads(n_threads_)
    for (size_t e = 0; e < nv; ++e) {
        auto ie = virt_mos_[e];
        double Fe = Fdiag_[ie];

        int thread = omp_get_thread_num();
#pragma omp critical
        {
            Be_vec[thread].data() = ints_->three_integral_block(aux_mos_, core_mos_, {ie}).data();
            if (!semi_canonical_) {
                X_vec[thread]("gn") = Be_vec[thread]("gm") * U_.block("cc")("nm");
                Be_vec[thread]("gn") = X_vec[thread]("gn");
            }
        }

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
    auto C = ambit::Tensor::build(tensor_type_, "C1total_CCAV", {na, na});
    for (int thread = 0; thread < n_threads; thread++) {
        C("vu") += C_vec[thread]("vu");
    }

    // rotate back to original orbital basis
    if (!semi_canonical_) {
        auto X = ambit::Tensor::build(tensor_type_, "tempCCAV", {na, na});
        X("xv") = C("uv") * U_.block("aa")("ux");
        C("xy") = X("xv") * U_.block("aa")("vy");
    }

    Hbar1("uv") += C("uv");

    t.stop();
}

void SA_MRPT2::compute_hbar() {
    // Note F_ and V_ are renormalized integrals
    if (!eri_df_) {
        H_A_Ca(F_, V_, T1_, T2_, S2_, 0.5, Hbar1_, Hbar2_);
    } else {
        // set up G2["pqrs"] = 2 * H2["pqrs"] - H2["pqsr"]
        auto G2 = ambit::BlockedTensor::build(tensor_type_, "G2H", {"avac", "aaac", "avaa"});
        G2["uevm"] += 2.0 * V_["uevm"] - V_["uemv"];
        G2["uvwm"] += 2.0 * V_["vumw"] - V_["uvmw"];
        G2["uexy"] += 2.0 * V_["euyx"] - V_["euxy"];

        H_A_Ca_small(F_, V_, G2, T1_, T2_, S2_, 0.5, Hbar1_, Hbar2_);

        auto na = actv_mos_.size();
        auto temp = ambit::Tensor::build(ambit::CoreTensor, "PT2Hbar1temp", {na, na});
        compute_Hbar1V_diskDF(temp);
        Hbar1_.block("aa")("uv") += 0.5 * temp("uv");
        Hbar1_.block("aa")("vu") += 0.5 * temp("uv");

        temp.zero();
        compute_Hbar1C_diskDF(temp);
        Hbar1_.block("aa")("uv") -= 0.5 * temp("uv");
        Hbar1_.block("aa")("vu") -= 0.5 * temp("uv");
    }
}

} // namespace forte
