/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
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
#include "helpers/disk_io.h"
#include "helpers/helpers.h"
#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sa_mrpt2.h"

using namespace psi;

namespace forte {

SA_MRPT2::SA_MRPT2(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
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
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", {"cc", "aa", "vv"});
        Fdiag_ = diagonalize_Fock_diagblocks(U_);
        if (eri_df_) {
            std::unordered_set<std::string> blocks;
            if (!semi_checked_results_.at("RESTRICTED_DOCC")) {
                blocks.emplace("cv");
                blocks.emplace("ac");
            }
            if (!semi_checked_results_.at("RESTRICTED_UOCC")) {
                blocks.emplace("cv");
                blocks.emplace("va");
            }
            if (!semi_checked_results_.at("ACTIVE")) {
                blocks.emplace("ac");
                blocks.emplace("va");
            }
            canonicalize_B(blocks);
        }
    }

    print_h2("Prepare Integrals & Amplitudes");

    // prepare integrals
    build_ints();

    // initialize tensors for amplitudes
    init_amps();

    // build amplitudes
    compute_t2();
    compute_t1();
    analyze_amplitudes("First-Order", T1_, T2_);
}

void SA_MRPT2::build_ints() {
    timer t("Initialize integrals");
    print_contents("Initializing integrals");

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        std::vector<std::string> blocks{"vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"};
        V_ = BTF_->build(tensor_type_, "V", blocks);
        if (ints_->integral_type() != DiskDF) {
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

    print_done(t.stop());
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

    auto Baa = B.block("Laa");

    // to minimize the number of calls of ints_->three_integral_block,
    // we store as many (Q|em) as possible in memory.
    size_t max_num = dsrg_mem_.available() * 0.8 / (sizeof(double) * (nQ * nv + na * na * nv));

    // separate core indices into batches
    auto core_batches = split_indices_to_batches(core_mos_, max_num);
    size_t nbatch = core_batches.size();

    for (size_t batch = 0, offset = 0; batch < nbatch; ++batch) {
        auto size = core_batches[batch].size();
        auto Bsub = ints_->three_integral_block(aux_mos_, virt_mos_, core_batches[batch]);

        auto Vsub = ambit::Tensor::build(tensor_type_, "Vsub PT2", {na, na, nv, size});
        Vsub("uvem") = Baa("guv") * Bsub("gem");

        Vsub.citerate([&](const std::vector<size_t>& i, const double& value) {
            Vavac[i[0] * dim1 + i[2] * dim2 + i[1] * nc + i[3] + offset] = value;
        });

        offset += size;
    }

    t.stop();
}

std::vector<std::vector<size_t>>
SA_MRPT2::split_indices_to_batches(const std::vector<size_t>& indices, size_t max_size) {
    auto n_indices = indices.size();

    std::vector<std::vector<size_t>> batches;
    size_t quotient = n_indices / max_size;
    for (size_t batch = 0, offset = 0; batch < quotient; ++batch) {
        std::vector<size_t> batch_mos(max_size);
        for (size_t p = 0; p < max_size; ++p) {
            batch_mos[p] = indices[p + offset];
        }
        batches.push_back(batch_mos);
        offset += max_size;
    }

    size_t remainder = n_indices - quotient * max_size;
    if (remainder != 0) {
        std::vector<size_t> batch_mos(remainder);
        for (size_t p = 0; p < remainder; ++p) {
            batch_mos[p] = indices[p + quotient * max_size];
        }
        batches.push_back(batch_mos);
    }

    return batches;
}

void SA_MRPT2::init_amps() {
    timer t("Initialize T1 and T2");
    print_contents("Allocating amplitudes");

    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});

    if (eri_df_) {
        std::vector<std::string> blocks{"aavv", "ccaa", "caav", "acav", "aava", "caaa", "aaaa"};
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", blocks);
        S2_ = BTF_->build(tensor_type_, "S2 Amplitudes", blocks);
    } else {
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
        S2_ = BTF_->build(tensor_type_, "S2 Amplitudes", {"hhpp"});
    }

    print_done(t.stop());
}

void SA_MRPT2::check_memory() {
    // memory of ints and amps
    if (eri_df_) {
        std::vector<std::string> blocks{"vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"};
        auto mem_blocks = dsrg_mem_.compute_memory(blocks);
        dsrg_mem_.add_entry("2-electron (4-index) integrals", mem_blocks);
        dsrg_mem_.add_entry("T2 cluster amplitudes", 2 * mem_blocks);
        if (ints_->integral_type() != DiskDF) {
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
    if (ints_->integral_type() == DiskDF) {
        dsrg_mem_.add_entry("Local 3-index integrals", {"Lca", "Laa", "Lav"}, 1, false);
    }

    // compute energy
    if (eri_df_) {
        auto size_Lv = dsrg_mem_.compute_memory({"Lv"});
        auto size_av = dsrg_mem_.compute_memory({"av"});
        auto size_aa = dsrg_mem_.compute_memory({"aa"});
        auto size_acc = dsrg_mem_.compute_memory({"acc"});

        auto mem_ccvv = 2 * (size_Lv + n_threads_ * dsrg_mem_.compute_memory({"vv"}));
        auto mem_cavv =
            size_Lv + n_threads_ * (3 * size_av + size_aa) + dsrg_mem_.compute_memory({"Lva"});
        auto mem_ccav =
            n_threads_ * (2 * size_acc + size_aa) + dsrg_mem_.compute_memory({"Lc", "Lac"});

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
    build_1rdm_vv();

    // scale the integrals
    renormalize_integrals(true);

    // compute energy
    double Ecorr = 0.0;

    local_timer lt;
    print_contents("Computing <0|[Fr, T1]|0>");
    double E_FT1 = H1_T1_C0(F_, T1_, 1.0, Ecorr);
    print_done(lt.get());

    lt.reset();
    print_contents("Computing <0|[Fr, T2]|0>");
    double E_FT2 = H1_T2_C0(F_, T2_, 1.0, Ecorr);
    print_done(lt.get());

    lt.reset();
    print_contents("Computing <0|[Vr, T1]|0>");
    double E_VT1 = H2_T1_C0(V_, T1_, 1.0, Ecorr);
    print_done(lt.get());

    std::vector<double> E_VT2_comp;
    if (!eri_df_) {
        lt.reset();
        print_contents("Computing <0|[Vr, T2]|0>");
        E_VT2_comp = H2_T2_C0(V_, T2_, S2_, 1.0, Ecorr);
        print_done(lt.get());
    } else {
        lt.reset();
        print_contents("Computing <0|[Vr, T2]|0> minimal");
        E_VT2_comp = H2_T2_C0_T2small(V_, T2_, S2_);
        print_done(lt.get());

        auto Eccvv = E_V_T2_CCVV();
        auto Ecavv = E_V_T2_CAVV();
        auto Eccav = E_V_T2_CCAV();

        E_VT2_comp[0] += Eccvv + Ecavv + Eccav;
        Ecorr += E_VT2_comp[0] + E_VT2_comp[1] + E_VT2_comp[2];

        if (print_) {
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
    // ONLY these T2 blocks are stored: aavv, ccaa, caav, acav, aava, caaa, aaaa
    timer t2min("Compute minimal T2");
    print_contents("Computing T2 amplitudes minimal");

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

    print_done(t2min.stop());
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
     * Batching: for given m and n, form J(ef) = Bm(L|e) * Bn(L|f)
     *
     * To minimize the number of calls of ints_->three_integral_block,
     * we store as many B(L|em) and B(L|fn) as possible in memory.
     */
    timer t_ccvv("Compute CCVV energy term DF");
    print_contents("Computing DF <0|[Vr, T2]|0> CCVV");

    double Eout = 0.0;

    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto nc = core_mos_.size();
    auto nQv = nQ * nv;

    if (nc == 0 or nv == 0) {
        print_done(t_ccvv.stop(), "Skipped");
        return Eout;
    }

    // test memory
    int nthreads = std::min(n_threads_, int(nc * (nc + 1) / 2));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = 2 * nthreads * nv * nv;
    size_t batch_min_size = 2 * nQv;
    if ((memory_min + batch_min_size) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CCVV) energy.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + batch_min_size) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CCVV) energy!");
    }

    size_t max_occ = (memory_avai / sizeof(double) - memory_min) / batch_min_size;
    if (max_occ < nc) {
        outfile->Printf("\n -> DF-DSRG-PT2(CCVV) energy to be run in batches: max core size = %zu",
                        max_occ);
    } else {
        max_occ = nc;
    }

    // batches of occupied indices
    std::vector<std::vector<size_t>> batch_occ;
    batch_occ = split_vector(core_mos_, max_occ);
    auto nbatches = batch_occ.size();

    // use MP2 amplitudes instead of DSRG
    bool complete_ccvv = (ccvv_source_ == "ZERO");

    // temp tensors for each thread
    std::vector<ambit::Tensor> Jab(nthreads), JKab(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jab_thread" + t, {nv, nv});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKab_thread" + t, {nv, nv});
    }

    bool Bcv_file_exist =
        !semi_checked_results_["RESTRICTED_DOCC"] or !semi_checked_results_["RESTRICTED_UOCC"];

    for (size_t i_batch = 0, i_shift = 0; i_batch < nbatches; ++i_batch) {
        const auto& i_batch_occ_mos = batch_occ[i_batch];
        auto i_nocc = i_batch_occ_mos.size();
        ambit::Tensor Bi; // iaQ
        if (Bcv_file_exist)
            Bi = read_Bcanonical("cv", {i_shift, i_shift + i_nocc}, {0, nv}, pqQ);
        else
            Bi = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, virt_mos_, pqQ);
        auto& Bi_data = Bi.data();

        for (size_t j_batch = i_batch, j_shift = i_shift; j_batch < nbatches; ++j_batch) {
            const auto& j_batch_occ_mos = batch_occ[j_batch];
            auto j_nocc = j_batch_occ_mos.size();
            ambit::Tensor Bj; // jbQ
            if (j_batch == i_batch) {
                Bj = Bi;
            } else {
                if (Bcv_file_exist)
                    Bj = read_Bcanonical("cv", {j_shift, j_shift + j_nocc}, {0, nv}, pqQ);
                else
                    Bj = ints_->three_integral_block(aux_mos_, j_batch_occ_mos, virt_mos_, pqQ);
            }
            auto& Bj_data = Bj.data();

            // index pairs of i and j
            std::vector<std::pair<size_t, size_t>> ij_pairs;
            if (i_batch == j_batch) {
                for (size_t i = 0; i < i_nocc; ++i) {
                    for (size_t j = i; j < j_nocc; ++j) {
                        ij_pairs.emplace_back(i, j);
                    }
                }
            } else {
                for (size_t i = 0; i < i_nocc; ++i) {
                    for (size_t j = 0; j < j_nocc; ++j) {
                        ij_pairs.emplace_back(i, j);
                    }
                }
            }
            size_t ij_pairs_size = ij_pairs.size();

#pragma omp parallel for num_threads(nthreads) reduction(+ : Eout)
            for (size_t p = 0; p < ij_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                size_t i = ij_pairs[p].first;
                size_t j = ij_pairs[p].second;

                auto fock_i = Fdiag_[i_batch_occ_mos[i]];
                auto fock_j = Fdiag_[j_batch_occ_mos[j]];

                double* Bia_ptr = &Bi_data[i * nQv];
                double* Bjb_ptr = &Bj_data[j * nQv];

                // compute (ia|jb) for given indices i and j
                psi::C_DGEMM('N', 'T', nv, nv, nQ, 1.0, Bia_ptr, nQ, Bjb_ptr, nQ, 0.0,
                             Jab[thread].data().data(), nv);

                JKab[thread]("pq") = 2.0 * Jab[thread]("pq") - Jab[thread]("qp");

                if (complete_ccvv) {
                    Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                        value /=
                            fock_i + fock_j - Fdiag_[virt_mos_[i[0]]] - Fdiag_[virt_mos_[i[1]]];
                    });
                } else {
                    Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                        auto D =
                            fock_i + fock_j - Fdiag_[virt_mos_[i[0]]] - Fdiag_[virt_mos_[i[1]]];
                        value *= (1.0 + dsrg_source_->compute_renormalized(D)) *
                                 dsrg_source_->compute_renormalized_denominator(D);
                    });
                }

                auto factor = (i_batch_occ_mos[i] == j_batch_occ_mos[j]) ? 1.0 : 2.0;
                Eout += factor * Jab[thread]("ef") * JKab[thread]("ef");
            }
            j_shift += j_nocc;
        }
        i_shift += i_nocc;
    }

    print_done(t_ccvv.stop());
    return Eout;
}

double SA_MRPT2::E_V_T2_CAVV() {
    timer t("Compute [V, T2] CAVV energy term");

    auto na = actv_mos_.size();
    auto C1 = ambit::Tensor::build(tensor_type_, "C1 VT2 CAVV", {na, na});
    compute_Hbar1V_DF(C1, true);

    if (form_Hbar_)
        C1_VT2_CAVV_ = C1;

    double Eout = C1("vu") * L1_.block("aa")("uv");

    t.stop();
    return Eout;
}

void SA_MRPT2::compute_Hbar1V_DF(ambit::Tensor& Hbar1, bool Vr) {
    /**
     * Compute Hbar1["vu"] += V["efmu"] * S["mvef"]
     *
     * - if Vr is false: V["efmu"] = B(L|em) * B(L|fu)
     * - if Vr is true:  V["efmu"] = B(L|em) * B(L|fu) * [1 + exp(-s * D1^2)]
     * - S["mvef"] = [2 * (me|vf) - (mf|ve)] * [1 - exp(-s * D2^2)] / D2
     *
     * where the two denominators are:
     *   D1 = F_m + F_u - F_e - F_f
     *   D2 = F_m + F_v - F_e - F_f
     *
     * Batching: for given m and f, form V(eu) and S(ev)
     *
     * To minimize the number of calls of ints_->three_integral_block,
     * we store as many B(L|em) as possible in memory.
     */
    timer t("Compute C1 virtual contraction DF");
    print_contents("Computing DF Hbar1 CAVV");

    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto nc = core_mos_.size();
    auto na = actv_mos_.size();
    auto nQv = nQ * nv;
    auto nQa = nQ * na;

    if (nc == 0 or nv == 0 or na == 0) {
        print_done(t.stop(), "Skipped");
        return;
    }

    // test memory
    int nthreads = std::min(n_threads_, int(nv * nc));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * na * (3 * nv + na) + nQa * nv;
    if ((memory_min + nQv) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CAVV) Hbar1V.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQv) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CAVV) Hbar1V!");
    }

    size_t max_occ = (memory_avai / sizeof(double) - memory_min) / nQv;
    if (max_occ < nc) {
        outfile->Printf("\n -> DF-DSRG-PT2(CAVV) Hbar1V to be run in batches: max core size = %zu",
                        max_occ);
    } else {
        max_occ = nc;
    }

    // batches of occupied indices
    std::vector<std::vector<size_t>> batch_occ;
    batch_occ = split_vector(core_mos_, max_occ);
    auto nbatches = batch_occ.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> C1(nthreads);
    std::vector<ambit::Tensor> J1(nthreads), J2(nthreads), JK(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        C1[i] = ambit::Tensor::build(CoreTensor, "C1uv_thread" + t, {na, na});
        J1[i] = ambit::Tensor::build(CoreTensor, "J1av_thread" + t, {nv, na});
        J2[i] = ambit::Tensor::build(CoreTensor, "J2av_thread" + t, {nv, na});
        JK[i] = ambit::Tensor::build(CoreTensor, "JKav_thread" + t, {nv, na});
    }

    // if canonicalized B files are available on disk
    bool semi_c = semi_checked_results_.at("RESTRICTED_DOCC");
    bool semi_v = semi_checked_results_.at("RESTRICTED_UOCC");
    bool semi_a = semi_checked_results_.at("ACTIVE");

    // 3-index integrals (P|eu)
    ambit::Tensor Bu; // euQ
    if (!semi_v or !semi_a)
        Bu = read_Bcanonical("va", {0, nv}, {0, na}, pqQ);
    else
        Bu = ints_->three_integral_block(aux_mos_, virt_mos_, actv_mos_, pqQ);
    auto& Bu_data = Bu.data();

    for (size_t i_batch = 0, i_shift = 0; i_batch < nbatches; ++i_batch) {
        const auto& i_batch_occ_mos = batch_occ[i_batch];
        auto i_nocc = i_batch_occ_mos.size();
        ambit::Tensor Bi; // iaQ
        if (!semi_c or !semi_v)
            Bi = read_Bcanonical("cv", {i_shift, i_shift + i_nocc}, {0, nv}, pqQ);
        else
            Bi = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, virt_mos_, pqQ);
        auto& Bi_data = Bi.data();

        // index pairs of i and c
        std::vector<std::pair<size_t, size_t>> ic_pairs;
        for (size_t i = 0; i < i_nocc; ++i) {
            for (size_t c = 0; c < nv; ++c) {
                ic_pairs.emplace_back(i, c);
            }
        }
        size_t ic_pairs_size = ic_pairs.size();

#pragma omp parallel for num_threads(nthreads)
        for (size_t p = 0; p < ic_pairs_size; ++p) {
            int thread = omp_get_thread_num();

            size_t i = ic_pairs[p].first;
            size_t c = ic_pairs[p].second;

            auto fock_i = Fdiag_[i_batch_occ_mos[i]];
            auto fock_c = Fdiag_[virt_mos_[c]];

            // compute (ia|uc) for given indices i and c
            double* Bia_ptr = &Bi_data[i * nQv];
            double* Bcu_ptr = &Bu_data[c * nQa];

            psi::C_DGEMM('N', 'T', nv, na, nQ, 1.0, Bia_ptr, nQ, Bcu_ptr, nQ, 0.0,
                         J1[thread].data().data(), na);

            JK[thread]("pq") = 2.0 * J1[thread]("pq");

            // compute (ic|ua) for given indices i and c
            double* Bic_ptr = &Bi_data[i * nQv + c * nQ];

            psi::C_DGEMV('N', nv * na, nQ, 1.0, Bu_data.data(), nQ, Bic_ptr, 1, 0.0,
                         J2[thread].data().data(), 1);

            JK[thread]("pq") -= J2[thread]("pq");

            JK[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                auto D = fock_i + Fdiag_[actv_mos_[i[1]]] - fock_c - Fdiag_[virt_mos_[i[0]]];
                value *= dsrg_source_->compute_renormalized_denominator(D);
            });

            if (Vr) {
                J1[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    auto D = fock_i + Fdiag_[actv_mos_[i[1]]] - fock_c - Fdiag_[virt_mos_[i[0]]];
                    value *= 1.0 + dsrg_source_->compute_renormalized(D);
                });
            }

            C1[thread]("vu") += JK[thread]("ev") * J1[thread]("eu");
        }
        i_shift += i_nocc;
    }

    // finalize results
    auto C = ambit::Tensor::build(tensor_type_, "C1_CAVV", {na, na});
    for (int thread = 0; thread < nthreads; ++thread) {
        C("vu") += C1[thread]("vu");
    }

    // rotate back to original orbital basis
    if (!semi_canonical_) {
        auto X = ambit::Tensor::build(tensor_type_, "tempCAVV", {na, na});
        X("xv") = C("uv") * U_.block("aa")("ux");
        C("xy") = X("xv") * U_.block("aa")("vy");
    }

    Hbar1("uv") += C("uv");

    print_done(t.stop());
}

double SA_MRPT2::E_V_T2_CCAV() {
    timer t("Compute [V, T2] CCAV energy term");

    auto na = actv_mos_.size();
    auto C1 = ambit::Tensor::build(tensor_type_, "C1 VT2 CCAV", {na, na});
    compute_Hbar1C_DF(C1, true);

    if (form_Hbar_)
        C1_VT2_CCAV_ = C1;

    double Eout = C1("vu") * Eta1_.block("aa")("uv");

    t.stop();
    return Eout;
}

void SA_MRPT2::compute_Hbar1C_DF(ambit::Tensor& Hbar1, bool Vr) {
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
     *
     * To minimize the number of calls of ints_->three_integral_block,
     * we store as many B(L|en) as possible in memory.
     */
    timer t("Compute C1 core contraction DF");
    print_contents("Computing DF Hbar1 CCAV");

    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto nc = core_mos_.size();
    auto na = actv_mos_.size();
    auto nQc = nQ * nc;
    auto nac = na * nc;

    if (nc == 0 or nv == 0 or na == 0) {
        print_done(t.stop(), "Skipped");
        return;
    }

    // test memory
    int nthreads = std::min(n_threads_, int(nv));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * na * (2 * nc * nc + na) + nQc * na;
    if ((memory_min + nQc) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CCAV) Hbar1C.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQc) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CCAV) Hbar1C!");
    }

    size_t max_vir = (memory_avai / sizeof(double) - memory_min) / nQc;
    if (max_vir < nv) {
        outfile->Printf("\n -> DF-DSRG-PT2(CCAV) Hbar1C to be run in batches: max virt size = %zu",
                        max_vir);
    } else {
        max_vir = nv;
    }

    // batches of virtual indices
    auto batch_vir = split_vector(virt_mos_, max_vir);
    auto nbatches = batch_vir.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> C1(nthreads);
    std::vector<ambit::Tensor> J(nthreads), JK(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        J[i] = ambit::Tensor::build(CoreTensor, "Jumn_thread" + t, {na, nc, nc});
        JK[i] = ambit::Tensor::build(CoreTensor, "JKumn_thread" + t, {na, nc, nc});
        C1[i] = ambit::Tensor::build(CoreTensor, "C1uv_thread" + t, {na, na});
    }

    // if canonicalized B files are available on disk
    bool semi_c = semi_checked_results_.at("RESTRICTED_DOCC");
    bool semi_v = semi_checked_results_.at("RESTRICTED_UOCC");
    bool semi_a = semi_checked_results_.at("ACTIVE");

    // 3-index integrals (P|mv)
    ambit::Tensor Bu; // umQ
    if (!semi_c or !semi_a)
        Bu = read_Bcanonical("ac", {0, na}, {0, nc}, pqQ);
    else
        Bu = ints_->three_integral_block(aux_mos_, actv_mos_, core_mos_, pqQ);
    auto& Bu_vec = Bu.data();

    for (size_t c_batch = 0, c_shift = 0; c_batch < nbatches; ++c_batch) {
        const auto& c_batch_vir_mos = batch_vir[c_batch];
        auto c_nvir = c_batch_vir_mos.size();
        ambit::Tensor Bc; // aiQ
        if (!semi_c or !semi_v)
            Bc = read_Bcanonical("vc", {c_shift, c_shift + c_nvir}, {0, nc}, pqQ);
        else
            Bc = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, core_mos_, pqQ);
        auto& Bc_vec = Bc.data();

#pragma omp parallel for num_threads(nthreads)
        for (size_t c = 0; c < c_nvir; ++c) {
            int thread = omp_get_thread_num();

            auto fock_c = Fdiag_[c_batch_vir_mos[c]];
            double* Bci_ptr = &Bc_vec[c * nQc];

            // form (iv|kc) for given c
            psi::C_DGEMM('N', 'T', nac, nc, nQ, 1.0, Bu_vec.data(), nQ, Bci_ptr, nQ, 0.0,
                         J[thread].data().data(), nc);

            JK[thread]("umn") = 2.0 * J[thread]("umn") - J[thread]("unm");

            if (Vr) {
                J[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom = Fdiag_[core_mos_[i[1]]] + Fdiag_[core_mos_[i[2]]] - fock_c -
                                   Fdiag_[actv_mos_[i[0]]];
                    value *= 1.0 + dsrg_source_->compute_renormalized(denom);
                });
            }

            JK[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = Fdiag_[core_mos_[i[1]]] + Fdiag_[core_mos_[i[2]]] - fock_c -
                               Fdiag_[actv_mos_[i[0]]];
                value *= dsrg_source_->compute_renormalized_denominator(denom);
            });

            C1[thread]("vu") += J[thread]("vmn") * JK[thread]("umn");
        }
        c_shift += c_nvir;
    }

    // finalize results
    auto C = ambit::Tensor::build(tensor_type_, "C1_CCAV", {na, na});
    for (int thread = 0; thread < nthreads; ++thread) {
        C("vu") += C1[thread]("vu");
    }

    // rotate back to original orbital basis
    if (!semi_canonical_) {
        auto X = ambit::Tensor::build(tensor_type_, "tempCCAV", {na, na});
        X("xv") = C("uv") * U_.block("aa")("ux");
        C("xy") = X("xv") * U_.block("aa")("vy");
    }

    Hbar1("uv") += C("uv");

    print_done(t.stop());
}

void SA_MRPT2::compute_hbar() {
    // Note F_ and V_ are renormalized integrals
    if (!eri_df_) {
        local_timer lt;
        print_contents("Computing MRPT2 Hbar AAAA");
        H_A_Ca(F_, V_, T1_, T2_, S2_, 0.5, Hbar1_, Hbar2_);
        print_done(lt.get());
    } else {
        // set up G2["pqrs"] = 2 * H2["pqrs"] - H2["pqsr"]
        auto G2 = ambit::BlockedTensor::build(tensor_type_, "G2H", {"avac", "aaac", "avaa"});
        G2["uevm"] += 2.0 * V_["uevm"] - V_["uemv"];
        G2["uvwm"] += 2.0 * V_["vumw"] - V_["uvmw"];
        G2["uexy"] += 2.0 * V_["euyx"] - V_["euxy"];

        local_timer lt;
        print_contents("Computing MRPT2 Hbar AAAA minimal");
        H_A_Ca_small(F_, V_, G2, T1_, T2_, S2_, 0.5, Hbar1_, Hbar2_);
        print_done(lt.get());

        Hbar1_.block("aa")("uv") += 0.5 * C1_VT2_CAVV_("uv");
        Hbar1_.block("aa")("vu") += 0.5 * C1_VT2_CAVV_("uv");

        Hbar1_.block("aa")("uv") -= 0.5 * C1_VT2_CCAV_("uv");
        Hbar1_.block("aa")("vu") -= 0.5 * C1_VT2_CCAV_("uv");
    }
}
} // namespace forte
