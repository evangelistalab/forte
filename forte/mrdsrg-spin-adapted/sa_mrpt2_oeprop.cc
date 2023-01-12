/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER,
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
#include "helpers/helpers.h"
#include "helpers/disk_io.h"
#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sa_mrpt2.h"

using namespace psi;

namespace forte {

void SA_MRPT2::transform_one_body(const std::vector<ambit::BlockedTensor>& oetens,
                                  const std::vector<int>& max_levels) {
    /**
     * Define the unrelaxed 1-RDM as the energy derivative w.r.t. Fock matrix.
     * The unrelaxed dipole moment is then given by
     *
     * Mbar = Mref + [M^{od}, A] + 0.5 * [[M^{d}, A], A]
     *
     * where the superscript d and od refer to the diagonal and off-diagonal components.
     *
     * In general, dipole moment computed using unrelaxed 1-RDM is considered bad
     * and response terms must be included to improve the results.
     * However, it might be useful to use Mbar for transition dipole moment.
     *
     * @param oetens: a vector of bare multipole integrals
     * @param max_body: the max body kept in recursive linear commutator approximation
     *
     * If max_body = 1,
     *     Mbar_{0,1} = Mref + [M^{od}, A]_{0,1} + 0.5 * [[M^{d}, A]_{1}, A]_{0,1}
     *
     * If max_body = 2,
     *     Mbar_{0,1,2} = Mref + [M^{od}, A]_{0,1,2} + 0.5 * [[M^{d}, A]_{1,2}, A]_{0,1,2}
     */
    print_h2("Transform One-Electron Operators");

    int n_tensors = oetens.size();
    Mbar0_ = std::vector<double>(n_tensors, 0.0);

    // compute Mref and add to Mbar0_
    for (int i = 0; i < n_tensors; ++i) {
        auto& M1c = oetens[i].block("cc").data();
        for (size_t m = 0, ncore = core_mos_.size(); m < ncore; ++m) {
            Mbar0_[i] += 2.0 * M1c[m * ncore + m];
        }
        Mbar0_[i] += oetens[i]["uv"] * L1_["vu"];
    }

    Mbar1_.resize(n_tensors);
    Mbar2_.resize(n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        Mbar1_[i] = BTF_->build(tensor_type_, oetens[i].name() + "1", {"aa"});
        Mbar1_[i]["uv"] += oetens[i]["uv"]; // add bare one-electron integrals
        if (max_levels[i] > 1)
            Mbar2_[i] = BTF_->build(tensor_type_, oetens[i].name() + "2", {"aaaa"});
    }

    auto max_body = *std::max_element(max_levels.begin(), max_levels.end());

    // temporary tensors
    ambit::BlockedTensor O1, C2, O2, G2, temp1, temp2;
    O1 = BTF_->build(tensor_type_, "O1", od_one_labels_ph());
    temp1 = BTF_->build(tensor_type_, "temp1M", {"aa"});
    if (max_body > 1) {
        temp2 = BTF_->build(tensor_type_, "temp2M", {"aaaa"});
        G2 = BTF_->build(tensor_type_, "C2", {"avac", "aaac", "avaa"}, true);
        if (!eri_df_) {
            O2 = BTF_->build(tensor_type_, "O2", {"pphh"}, true);
        } else {
            O2 = BTF_->build(tensor_type_, "O2",
                             {"vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"}, true);
        }
    }

    // special treatment for large T2 amplitudes when doing DF
    auto D1 = BTF_->build(tensor_type_, "D1", {"gg"});
    if (eri_df_) {
        compute_1rdm_cc_CCVV_DF(D1);
        compute_1rdm_vv_CCVV_DF(D1);

        std::vector<ambit::BlockedTensor> mp_ints;
        if (max_body > 1 and eri_df_)
            mp_ints = oetens;

        compute_1rdm_cc_CCAV_DF(D1, mp_ints);
        compute_1rdm_aa_vv_CCAV_DF(D1, mp_ints);

        compute_1rdm_cc_aa_CAVV_DF(D1, mp_ints);
        compute_1rdm_vv_CAVV_DF(D1, mp_ints);
    }

    // transform each tensor
    for (int i = 0; i < n_tensors; ++i) {
        local_timer t_local;
        const auto& M = oetens[i];
        print_contents("Transforming " + M.name());

        // separate M to diagonal and off-diagonal components
        auto Md = BTF_->build(tensor_type_, M.name() + " D", {"cc", "aa", "vv"});
        auto Mod = BTF_->build(tensor_type_, M.name() + " OD", od_one_labels_ph());
        Md["pq"] = M["pq"];
        Mod["pq"] = M["pq"];

        // initialize Mbar
        auto& Mbar0 = Mbar0_[i];
        auto& Mbar1 = Mbar1_[i];
        temp1.zero();

        // prepare O1 = M^{od} + 0.5 * [M^{d}, A]^{od}
        O1["pq"] = Mod["pq"];
        H1d_A1_C1ph(Md, T1_, 0.5, O1);
        H1d_A2_C1ph(Md, S2_, 0.5, O1);

        // compute Mbar_{0,1}
        H1_T1_C0(O1, T1_, 2.0, Mbar0);
        H1_T2_C0(O1, T2_, 2.0, Mbar0);
        H1_T_C1a_smallS(O1, T1_, S2_, temp1);

        // if need two-body active integrals
        if (max_levels[i] > 1) {
            auto& Mbar2 = Mbar2_[i];
            O2.zero();
            temp2.zero();

            if (!eri_df_) {
                // O2 = 0.5 * [M^{d}, A]^{od}
                H1d_A2_C2pphh(Md, T2_, 0.5, O2);

                // active part of [O2, T2]1 with large T2
                temp1["wz"] += O2["efzm"] * S2_["wmef"];
                temp1["wz"] -= O2["wemn"] * S2_["mnze"];
            } else {
                // O2 = 0.5 * [M^{d}, A]^{od}
                H1d_A2_C2pphh_small(Md, T2_, 0.5, O2);

                // scalar part of [O2, T2] with large T2
                Mbar0 += D1["pq"] * Md["pq"];

                // NOTE: active part of [O2, T2]1 with large T2 have considered
                // when computing unrelaxed 1-RDM previously
            }

            // scalar part of [O2, T1 + T2]
            H2_T1_C0(O2, T1_, 2.0, Mbar0);
            H2_T2_C0(O2, T2_, S2_, 2.0, Mbar0);

            // prepare G2["pqrs"] = 2 * O2["pqrs"] - O2["pqsr"]
            G2.block("avac")("uevm") = 2.0 * O2.block("avac")("uevm") - O2.block("avca")("uemv");
            G2.block("aaac")("uvwm") = 2.0 * O2.block("aaca")("vumw") - O2.block("aaca")("uvmw");
            G2.block("avaa")("uexy") = 2.0 * O2.block("vaaa")("euyx") - O2.block("vaaa")("euxy");

            // active part of [O1, T2]2 + [O2, T1 + T2]
            H2_T_C1a_smallS(O2, T2_, S2_, temp1);
            H2_T_C1a_smallG(G2, T1_, T2_, temp1);
            H_T_C2a_smallS(O1, O2, T1_, T2_, S2_, temp2);

            // add 2-body results
            Mbar2["uvxy"] += temp2["uvxy"];
            Mbar2["xyuv"] += temp2["uvxy"];

            // outfile->Printf(", Mbar2 norm = %20.10f", Mbar2.norm());
        }

        // add 1-body results
        Mbar1["uv"] += temp1["uv"];
        Mbar1["vu"] += temp1["uv"];

        // outfile->Printf("\n %-20s: Mbar0 = %20.10f, Mbar1 norm = %20.10f", M.name().c_str(), Mbar0,
        //                 Mbar1.norm());

        print_done(t_local.get());
    }
}

void SA_MRPT2::compute_1rdm_cc_CCVV_DF(ambit::BlockedTensor& D1) {
    /**
     * Compute the core-core part of the MP2-like unrelaxed spin-summed 1-RDM.
     *
     * D1["ij"] -= T2["ikab"] * S2["jkab"] + 1.0 * T2["jkab"] * S2["ikab"]
     *
     * where S2["jkab"] = 2.0 * T2["jkab"] - T2["jkba"]
     * for core indices i, j, k and virtual indices a, b.
     *
     * Amplitudes are built in batches for every ab pairs.
     */
    timer t_ccvv("Compute CCVV 1RDM CC term DF");
    print_contents("Computing DF CCVV 1RDM CC part");

    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto nc = core_mos_.size();
    auto nQc = nQ * nc;

    // test memory
    int nthreads = std::min(n_threads_, int(nv * (nv + 1) / 2));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = 3 * nthreads * nc * nc;
    if ((memory_min + nQc * 2) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CCVV) 1RDM CC.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQc * 2) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CCVV) 1RDM CC!");
    }

    size_t max_vir = (memory_avai / sizeof(double) - memory_min) / (2 * nQc);
    if (max_vir < nv) {
        outfile->Printf("\n -> DF-DSRG-PT2(CCVV) 1RDM CC to be run in batches: max virt size = %zu",
                        max_vir);
    } else {
        max_vir = nv;
    }

    // batches of virtual indices
    std::vector<std::vector<size_t>> batch_vir;
    batch_vir = split_vector(virt_mos_, max_vir);
    auto nbatches = batch_vir.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> Da(nthreads);
    std::vector<ambit::Tensor> Jmn(nthreads), JKmn(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Da[i] = ambit::Tensor::build(CoreTensor, "Dcc_thread" + t, {nc, nc});
        Jmn[i] = ambit::Tensor::build(CoreTensor, "Jcc_thread" + t, {nc, nc});
        JKmn[i] = ambit::Tensor::build(CoreTensor, "JKcc_thread" + t, {nc, nc});
    }

    for (size_t c_batch = 0; c_batch < nbatches; ++c_batch) {
        const auto& c_batch_vir_mos = batch_vir[c_batch];
        auto c_nvir = c_batch_vir_mos.size();
        auto Bc = ambit::Tensor::build(ambit::CoreTensor, "Bc", {c_nvir, nc, nQ});
        Bc("aig") = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, core_mos_)("gai");
        auto& Bc_vec = Bc.data();

        for (size_t d_batch = c_batch; d_batch < nbatches; ++d_batch) {
            const auto& d_batch_vir_mos = batch_vir[d_batch];
            auto d_nvir = d_batch_vir_mos.size();
            ambit::Tensor Bd;
            if (d_batch == c_batch) {
                Bd = Bc;
            } else {
                Bd = ambit::Tensor::build(ambit::CoreTensor, "Bd", {d_nvir, nc, nQ});
                Bd("aig") =
                    ints_->three_integral_block(aux_mos_, d_batch_vir_mos, core_mos_)("gai");
            }
            auto& Bd_vec = Bd.data();

            // index pairs for c and d
            std::vector<std::pair<size_t, size_t>> cd_pairs;
            if (d_batch == c_batch) {
                for (size_t c = 0; c < c_nvir; ++c) {
                    for (size_t d = c; d < d_nvir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            } else {
                for (size_t c = 0; c < c_nvir; ++c) {
                    for (size_t d = 0; d < d_nvir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            }
            size_t cd_pairs_size = cd_pairs.size();

#pragma omp parallel for num_threads(nthreads)
            for (size_t p = 0; p < cd_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                auto c = cd_pairs[p].first;
                auto d = cd_pairs[p].second;

                auto fock_c = Fdiag_[c_batch_vir_mos[c]];
                auto fock_d = Fdiag_[d_batch_vir_mos[d]];

                double* Bci_ptr = &Bc_vec[c * nQc];
                double* Bdj_ptr = &Bd_vec[d * nQc];

                // compute (ci|dj) for given indices c and d
                C_DGEMM('N', 'T', nc, nc, nQ, 1.0, Bci_ptr, nQ, Bdj_ptr, nQ, 0.0,
                        Jmn[thread].data().data(), nc);

                Jmn[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom =
                        Fdiag_[core_mos_[i[0]]] + Fdiag_[core_mos_[i[1]]] - fock_c - fock_d;
                    value *= dsrg_source_->compute_renormalized_denominator(denom);
                });
                JKmn[thread]("pq") = 2.0 * Jmn[thread]("pq") - Jmn[thread]("qp");

                auto factor = (c_batch_vir_mos[c] == d_batch_vir_mos[d]) ? 0.5 : 1.0;
                Da[thread]("ij") -= factor * Jmn[thread]("ik") * JKmn[thread]("jk");
                Da[thread]("ij") -= factor * Jmn[thread]("ki") * JKmn[thread]("kj");
            }
        }
    }

    // add Dcc contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("cc")("pq") += Da[i]("pq");
        D1.block("cc")("pq") += Da[i]("qp");
    }

    print_done(t_ccvv.stop());
}

void SA_MRPT2::compute_1rdm_vv_CCVV_DF(ambit::BlockedTensor& D1) {
    /**
     * Compute the virtual-virtual part of the MP2-like unrelaxed spin-summed 1-RDM.
     *
     * D1["ab"] += T2["ijac"] * S2["ijbc"] + T2["ijbc"] * S2["ijac"]
     *
     * where S2["ijbc"] = 2.0 * T2["ijbc"] - T2["jibc"]
     * for core indices i, j and virtual indices a, b, c.
     *
     * Amplitudes are built in batches for every ij pairs.
     */
    timer t_ccvv("Compute CCVV 1RDM VV term DF");
    print_contents("Computing DF CCVV 1RDM VV part");

    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto nc = core_mos_.size();
    auto nQv = nQ * nv;

    // test memory
    int nthreads = std::min(n_threads_, int(nc * (nc + 1) / 2));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = 3 * nthreads * nv * nv;
    if ((memory_min + 2 * nQv) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CCVV) 1RDM VV.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + 2 * nQv) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CCVV) 1RDM VV!");
    }

    size_t max_occ = (memory_avai / sizeof(double) - memory_min) / (2 * nQv);
    if (max_occ < nc) {
        outfile->Printf("\n -> DF-DSRG-PT2(CCVV) 1RDM VV to be run in batches: max core size = %zu",
                        max_occ);
    } else {
        max_occ = nc;
    }

    // batches of occupied indices
    std::vector<std::vector<size_t>> batch_occ;
    batch_occ = split_vector(core_mos_, max_occ);
    auto nbatches = batch_occ.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> Da(nthreads);
    std::vector<ambit::Tensor> Jab(nthreads), JKab(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Da[i] = ambit::Tensor::build(CoreTensor, "Dvv_thread" + t, {nv, nv});
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jvv_thread" + t, {nv, nv});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKvv_thread" + t, {nv, nv});
    }

    for (size_t i_batch = 0; i_batch < nbatches; ++i_batch) {
        const auto& i_batch_occ_mos = batch_occ[i_batch];
        auto i_nocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Bi", {i_nocc, nv, nQ});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, virt_mos_)("gia");
        auto& Bi_vec = Bi.data();

        for (size_t j_batch = i_batch; j_batch < nbatches; ++j_batch) {
            const auto& j_batch_occ_mos = batch_occ[j_batch];
            auto j_nocc = j_batch_occ_mos.size();
            ambit::Tensor Bj;
            if (j_batch == i_batch) {
                Bj = Bi;
            } else {
                Bj = ambit::Tensor::build(ambit::CoreTensor, "Bj", {j_nocc, nv, nQ});
                Bj("iag") =
                    ints_->three_integral_block(aux_mos_, j_batch_occ_mos, virt_mos_)("gia");
            }
            auto& Bj_vec = Bj.data();

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

#pragma omp parallel for num_threads(nthreads)
            for (size_t p = 0; p < ij_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                size_t i = ij_pairs[p].first;
                size_t j = ij_pairs[p].second;

                auto fock_i = Fdiag_[i_batch_occ_mos[i]];
                auto fock_j = Fdiag_[j_batch_occ_mos[j]];

                double* Bia_ptr = &Bi_vec[i * nQv];
                double* Bjb_ptr = &Bj_vec[j * nQv];

                // compute (ia|jb) for given indices i and j
                C_DGEMM('N', 'T', nv, nv, nQ, 1.0, Bia_ptr, nQ, Bjb_ptr, nQ, 0.0,
                        Jab[thread].data().data(), nv);

                Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom =
                        fock_i + fock_j - Fdiag_[virt_mos_[i[0]]] - Fdiag_[virt_mos_[i[1]]];
                    value *= dsrg_source_->compute_renormalized_denominator(denom);
                });
                JKab[thread]("pq") = 2.0 * Jab[thread]("pq") - Jab[thread]("qp");

                auto factor = (i_batch_occ_mos[i] == j_batch_occ_mos[j]) ? 0.5 : 1.0;
                Da[thread]("ab") += factor * Jab[thread]("ac") * JKab[thread]("bc");
                Da[thread]("ab") += factor * Jab[thread]("ca") * JKab[thread]("cb");
            }
        }
    }

    // add Dvv contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("vv")("pq") += Da[i]("pq");
        D1.block("vv")("pq") += Da[i]("qp");
    }

    print_done(t_ccvv.stop());
}

void SA_MRPT2::compute_1rdm_cc_CCAV_DF(ambit::BlockedTensor& D1,
                                       const std::vector<ambit::BlockedTensor>& oetens) {
    /**
     * Compute the core-core part of the unrelaxed spin-summed 1-RDM from T2 ccav block.
     *
     * D1["ij"] -= 0.5 * (T2["ikva"] * S2["jkua"] + T2["kiva"] * S2["kjua"]) * Eta1["uv"]
     *           + 0.5 * (T2["jkva"] * S2["ikua"] + T2["kjva"] * S2["kiua"]) * Eta1["uv"]
     *
     * where S2["jkua"] = 2.0 * T2["jkua"] - T2["kjua"]
     * for core indices i, j, k; active indices u, v; and virtual index a.
     *
     * Amplitudes are built in batches for every index a.
     *
     * If oetens is not empty, the active part of the transformed one-electron integrals
     * are also computed inside the batches.
     *
     * Mt["vu"] += 0.5 * M["ji"] * (T2["ikva"] * S2["jkua"] + T2["kiva"] * S2["kjua"])
     *           + 0.5 * M["ji"] * (T2["jkva"] * S2["ikua"] + T2["kjva"] * S2["kiua"])
     *           + 0.5 * M["ji"] * (T2["ikua"] * S2["jkva"] + T2["kiua"] * S2["kjva"])
     *           + 0.5 * M["ji"] * (T2["jkua"] * S2["ikva"] + T2["kjua"] * S2["kiva"])
     */
    timer t_ccav("Compute CCAV 1RDM CC term DF");
    print_contents("Computing DF CCAV 1RDM CC part");

    auto N = static_cast<int>(oetens.size());

    auto nQ = aux_mos_.size();
    auto nc = core_mos_.size();
    auto na = actv_mos_.size();
    auto nv = virt_mos_.size();
    auto nQc = nQ * nc;
    auto nac = na * nc;

    // test memory
    int nthreads = std::min(n_threads_, int(nv));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * nc * nc * (3 * na + 1) + nQ * nc * na + nthreads * na * na * N;
    if ((memory_min + nQc) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CCAV) 1RDM CC.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQc) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CCAV) 1RDM CC!");
    }

    size_t max_vir = (memory_avai / sizeof(double) - memory_min) / nQc;
    if (max_vir < nv) {
        outfile->Printf("\n -> DF-DSRG-PT2(CCAV) 1RDM CC to be run in batches: max virt size = %zu",
                        max_vir);
    } else {
        max_vir = nv;
    }

    // batches of virtual indices
    auto batch_vir = split_vector(virt_mos_, max_vir);
    auto nbatches = batch_vir.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> Dc(nthreads);
    std::vector<ambit::Tensor> T2umn(nthreads), S2umn(nthreads), X2umn(nthreads);
    std::vector<std::vector<ambit::Tensor>> Ma(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Dc[i] = ambit::Tensor::build(CoreTensor, "Dcc_thread" + t, {nc, nc});
        T2umn[i] = ambit::Tensor::build(CoreTensor, "T2acc_thread" + t, {na, nc, nc});
        S2umn[i] = ambit::Tensor::build(CoreTensor, "S2acc_thread" + t, {na, nc, nc});
        X2umn[i] = ambit::Tensor::build(CoreTensor, "X2acc_thread" + t, {na, nc, nc});

        for (int n = 0; n < N; ++n) {
            std::string name = "Ma" + std::to_string(n) + "_thread" + t;
            Ma[i].push_back(ambit::Tensor::build(CoreTensor, name, {na, na}));
        }
    }

    // 3-index integrals (P|mv)
    auto Bmv = ambit::Tensor::build(CoreTensor, "Bmv", {na, nc, nQ});
    Bmv("vmg") = ints_->three_integral_block(aux_mos_, actv_mos_, core_mos_)("gvm");
    auto& Bu_vec = Bmv.data();

    // pointer to 1-hole density
    const auto& E1 = Eta1_.block("aa");

    for (size_t c_batch = 0; c_batch < nbatches; ++c_batch) {
        const auto& c_batch_vir_mos = batch_vir[c_batch];
        auto c_nvir = c_batch_vir_mos.size();
        auto Bc = ambit::Tensor::build(ambit::CoreTensor, "Bi", {c_nvir, nc, nQ});
        Bc("aig") = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, core_mos_)("gai");
        auto& Bc_vec = Bc.data();

#pragma omp parallel for num_threads(nthreads)
        for (size_t c = 0; c < nv; ++c) {
            int thread = omp_get_thread_num();

            auto fock_c = Fdiag_[c_batch_vir_mos[c]];
            double* Bci_ptr = &Bc_vec[c * nQc];

            // form T2["ikvc"] for given c
            C_DGEMM('N', 'T', nac, nc, nQ, 1.0, Bu_vec.data(), nQ, Bci_ptr, nQ, 0.0,
                    T2umn[thread].data().data(), nc);
            T2umn[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = Fdiag_[core_mos_[i[1]]] + Fdiag_[core_mos_[i[2]]] - fock_c -
                               Fdiag_[actv_mos_[i[0]]];
                value *= dsrg_source_->compute_renormalized_denominator(denom);
            });
            S2umn[thread]("umn") = 2.0 * T2umn[thread]("umn") - T2umn[thread]("unm");

            X2umn[thread]("umn") = T2umn[thread]("vmn") * E1("uv");

            Dc[thread]("ij") -= 0.5 * X2umn[thread]("uik") * S2umn[thread]("ujk");
            Dc[thread]("ij") -= 0.5 * X2umn[thread]("uki") * S2umn[thread]("ukj");

            for (int n = 0; n < N; ++n) {
                const auto& M1 = oetens[n].block("cc");
                Ma[thread][n]("vu") -= 0.5 * M1("ji") * T2umn[thread]("vik") * S2umn[thread]("ujk");
                Ma[thread][n]("vu") -= 0.5 * M1("ji") * T2umn[thread]("vki") * S2umn[thread]("ukj");
            }
        }
    }

    // add Dcc contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("cc")("pq") += Dc[i]("pq");
        D1.block("cc")("pq") += Dc[i]("qp");
    }

    // add Maa contributions to Mbar1_
    for (int n = 0; n < N; ++n) {
        const auto& M1 = Mbar1_.at(n).block("aa");
        for (int i = 0; i < nthreads; ++i) {
            M1("pq") -= Ma[i][n]("pq");
            M1("qp") -= Ma[i][n]("pq");
        }
    }

    print_done(t_ccav.stop());
}

void SA_MRPT2::compute_1rdm_aa_vv_CCAV_DF(ambit::BlockedTensor& D1,
                                          const std::vector<ambit::BlockedTensor>& oetens) {
    /**
     * Compute the active-active and virtual-virtual parts of
     * the unrelaxed spin-summed 1-RDM from T2 ccav block.
     *
     * D1["vy"] += 0.5 * T2["ijya"] * S2["ijua"] * Eta1["uv"]
     *           + 0.5 * T2["ijva"] * S2["ijua"] * Eta1["uy"]
     *
     * D1["ba"] += 0.5 * (T2["ijva"] * S2["ijub"] + T2["ijvb"] * S2["ijua"]) * Eta1["uv"]
     *
     * where S2["ijua"] = 2.0 * T2["ijua"] - T2["jiua"]
     * for core indices i, j; active indices u, v, y; and virtual indices a, b.
     *
     * Amplitudes are built in batches for every ij pairs.
     *
     * If oetens is not empty, the active part of the transformed one-electron integrals
     * are also computed inside the batches.
     *
     * Mt["vu"] -= 0.5 * T2["ijya"] * (M["yv"] * S2["ijua"] + M["yu"] * S2["ijva"])
     *
     * Mt["vu"] -= 0.5 * M["ab"] * (T2["ijva"] * S2["ijub"] + T2["ijua"] * S2["ijvb"])
     */
    timer t_ccav("Compute CCAV 1RDM AA/VV term DF");
    print_contents("Computing DF CCAV 1RDM AA/VV part");

    auto N = static_cast<int>(oetens.size());

    auto nQ = aux_mos_.size();
    auto nc = core_mos_.size();
    auto na = actv_mos_.size();
    auto nv = virt_mos_.size();
    auto nQv = nQ * nv;
    auto nQa = nQ * na;

    // test memory
    int nthreads = std::min(n_threads_, int(nc * (nc + 1) / 2));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * (3 * na * nv + na * na + nv * nv + na * na * N) + nQ * nc * na;
    if ((memory_min + nQv * 2) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CCAV) 1RDM AA/VV.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQv * 2) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CCAV) 1RDM AA/VV!");
    }

    size_t max_occ = (memory_avai / sizeof(double) - memory_min) / (2 * nQv);
    if (max_occ < nc) {
        outfile->Printf(
            "\n -> DF-DSRG-PT2(CCAV) 1RDM AA/VV to be run in batches: max core size = %zu",
            max_occ);
    } else {
        max_occ = nc;
    }

    // batches of occupied indices
    std::vector<std::vector<size_t>> batch_occ;
    batch_occ = split_vector(core_mos_, max_occ);
    auto nbatches = batch_occ.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> Da(nthreads), Dv(nthreads);
    std::vector<ambit::Tensor> J1(nthreads), J2(nthreads), JK(nthreads);
    std::vector<std::vector<ambit::Tensor>> Ma(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Da[i] = ambit::Tensor::build(CoreTensor, "Daa_thread" + t, {na, na});
        Dv[i] = ambit::Tensor::build(CoreTensor, "Dvv_thread" + t, {nv, nv});
        J1[i] = ambit::Tensor::build(CoreTensor, "J1va_thread" + t, {nv, na});
        J2[i] = ambit::Tensor::build(CoreTensor, "J2va_thread" + t, {nv, na});
        JK[i] = ambit::Tensor::build(CoreTensor, "JKva_thread" + t, {nv, na});

        for (int n = 0; n < N; ++n) {
            std::string name = "Ma" + std::to_string(n) + "_thread" + t;
            Ma[i].push_back(ambit::Tensor::build(CoreTensor, name, {na, na}));
        }
    }

    // 3-index integrals (P|mv)
    auto Bmv = ambit::Tensor::build(CoreTensor, "Bmv", {nc, na, nQ});
    Bmv("mvg") = ints_->three_integral_block(aux_mos_, core_mos_, actv_mos_)("gmv");
    auto& Bu_vec = Bmv.data();

    // 1-hole density pointer
    auto E1 = Eta1_.block("aa");

    for (size_t i_batch = 0, i_shift = 0; i_batch < nbatches; ++i_batch) {
        const auto& i_batch_occ_mos = batch_occ[i_batch];
        auto i_nocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Ba", {i_nocc, nv, nQ});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, virt_mos_)("gia");
        auto& Bi_vec = Bi.data();

        for (size_t j_batch = i_batch, j_shift = i_shift; j_batch < nbatches; ++j_batch) {
            const auto& j_batch_occ_mos = batch_occ[j_batch];
            auto j_nocc = j_batch_occ_mos.size();
            ambit::Tensor Bj;
            if (j_batch == i_batch) {
                Bj = Bi;
            } else {
                Bj = ambit::Tensor::build(ambit::CoreTensor, "Bb", {j_nocc, nv, nQ});
                Bj("iag") =
                    ints_->three_integral_block(aux_mos_, j_batch_occ_mos, virt_mos_)("gia");
            }
            auto& Bj_vec = Bj.data();

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

#pragma omp parallel for num_threads(nthreads)
            for (size_t p = 0; p < ij_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                size_t i = ij_pairs[p].first;
                size_t j = ij_pairs[p].second;

                auto fock_i = Fdiag_[i_batch_occ_mos[i]];
                auto fock_j = Fdiag_[j_batch_occ_mos[j]];

                // compute (ia|jv) for given indices i and j
                double* Bia_ptr = &Bi_vec[i * nQv];
                double* Bjv_ptr = &Bu_vec[(j + j_shift) * nQa];

                C_DGEMM('N', 'T', nv, na, nQ, 1.0, Bia_ptr, nQ, Bjv_ptr, nQ, 0.0,
                        J1[thread].data().data(), na);

                J1[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom =
                        fock_i + fock_j - Fdiag_[virt_mos_[i[0]]] - Fdiag_[actv_mos_[i[1]]];
                    value *= dsrg_source_->compute_renormalized_denominator(denom);
                });

                // compute (ja|iv) for given indices i and j
                double* Bja_ptr = &Bj_vec[j * nQv];
                double* Biv_ptr = &Bu_vec[(i + i_shift) * nQa];

                C_DGEMM('N', 'T', nv, na, nQ, 1.0, Bja_ptr, nQ, Biv_ptr, nQ, 0.0,
                        J2[thread].data().data(), na);

                J2[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom =
                        fock_i + fock_j - Fdiag_[virt_mos_[i[0]]] - Fdiag_[actv_mos_[i[1]]];
                    value *= dsrg_source_->compute_renormalized_denominator(denom);
                });

                auto factor = 0.5 * ((i_batch_occ_mos[i] == j_batch_occ_mos[j]) ? 0.5 : 1.0);

                JK[thread]("pq") = 2.0 * J1[thread]("pq") - J2[thread]("pq");
                Da[thread]("vy") += factor * J1[thread]("ey") * JK[thread]("eu") * E1("uv");
                Dv[thread]("ea") += factor * J1[thread]("av") * JK[thread]("eu") * E1("uv");

                for (int n = 0; n < N; ++n) {
                    const auto& M1a = oetens[n].block("aa");
                    const auto& M1v = oetens[n].block("vv");
                    Ma[thread][n]("vu") += factor * M1a("yv") * J1[thread]("ey") * JK[thread]("eu");
                    Ma[thread][n]("vu") += factor * M1v("ae") * J1[thread]("av") * JK[thread]("eu");
                }

                JK[thread]("pq") = 2.0 * J2[thread]("pq") - J1[thread]("pq");
                Da[thread]("vy") += factor * J2[thread]("ey") * JK[thread]("eu") * E1("uv");
                Dv[thread]("ea") += factor * J2[thread]("av") * JK[thread]("eu") * E1("uv");

                for (int n = 0; n < N; ++n) {
                    const auto& M1a = oetens[n].block("aa");
                    const auto& M1v = oetens[n].block("vv");
                    Ma[thread][n]("vu") += factor * M1a("yv") * J2[thread]("ey") * JK[thread]("eu");
                    Ma[thread][n]("vu") += factor * M1v("ae") * J2[thread]("av") * JK[thread]("eu");
                }
            }
            j_shift += j_nocc;
        }
        i_shift += i_nocc;
    }

    // add Daa and Dvv contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("aa")("pq") += Da[i]("pq");
        D1.block("aa")("pq") += Da[i]("qp");
        D1.block("vv")("pq") += Dv[i]("pq");
        D1.block("vv")("pq") += Dv[i]("qp");
    }

    // add Maa contributions to Mbar1_
    for (int n = 0; n < N; ++n) {
        const auto& M1 = Mbar1_.at(n).block("aa");
        for (int i = 0; i < nthreads; ++i) {
            M1("pq") -= Ma[i][n]("pq");
            M1("qp") -= Ma[i][n]("pq");
        }
    }

    print_done(t_ccav.stop());
}

void SA_MRPT2::compute_1rdm_cc_aa_CAVV_DF(ambit::BlockedTensor& D1,
                                          const std::vector<ambit::BlockedTensor>& oetens) {
    /**
     * Compute the core-core and active-active parts of
     * the unrelaxed spin-summed 1-RDM from T2 cavv block.
     *
     * D1["ij"] -= 0.5 * (T2["iuab"] * S2["jvab"] + T2["juab"] * S2["ivab"]) * L1["uv"]
     *
     * D1["xu"] -= 0.5 * T2["ixab"] * S2["ivab"] * L1["uv"]
     *           + 0.5 * T2["iuab"] * S2["ivab"] * L1["xv"]
     *
     * where S2["jvab"] = 2.0 * T2["jvab"] - T2["jvba"]
     * for core indices i, j; active indices u, v, x; and virtual indices a, b.
     *
     * Amplitudes are built in batches for every ab pairs.
     *
     * If oetens is not empty, the active part of the transformed one-electron integrals
     * are also computed inside the batches.
     *
     * Mt["vu"] -= 0.5 * M["ji"] * (T2["iuab"] * S2["jvab"] + T2["ivab"] * S2["juab"])
     *
     * Mt["vu"] -= 0.5 * T2["ixab"] * (M["ux"] * S2["ivab"] + M["vx"] * S2["iuab"])
     */
    timer t_cavv("Compute CAVV 1RDM CC/AA term DF");
    print_contents("Computing DF CAVV 1RDM CC/AA part");

    auto N = static_cast<int>(oetens.size());

    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto na = actv_mos_.size();
    auto nc = core_mos_.size();
    auto nQc = nQ * nc;
    auto nQa = nQ * na;

    // test memory
    int nthreads = std::min(n_threads_, int(nv * (nv + 1) / 2));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * (nc * nc + na * na + 3 * na * nc + na * na * N) + nQa * nv;
    if ((memory_min + nQc * 2) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CAVV) 1RDM CC/AA.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQc * 2) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CAVV) 1RDM CC/AA!");
    }

    size_t max_vir = (memory_avai / sizeof(double) - memory_min) / (2 * nQc);
    if (max_vir < nv) {
        outfile->Printf(
            "\n -> DF-DSRG-PT2(CAVV) 1RDM CC/AA to be run in batches: max virt size = %zu",
            max_vir);
    } else {
        max_vir = nv;
    }

    // batches of virtual indices
    std::vector<std::vector<size_t>> batch_vir;
    batch_vir = split_vector(virt_mos_, max_vir);
    auto nbatches = batch_vir.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> Dc(nthreads), Da(nthreads);
    std::vector<ambit::Tensor> J1(nthreads), J2(nthreads), JK(nthreads);
    std::vector<std::vector<ambit::Tensor>> Ma(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Dc[i] = ambit::Tensor::build(CoreTensor, "Dcc_thread" + t, {nc, nc});
        Da[i] = ambit::Tensor::build(CoreTensor, "Daa_thread" + t, {na, na});
        J1[i] = ambit::Tensor::build(CoreTensor, "J1ca_thread" + t, {nc, na});
        J2[i] = ambit::Tensor::build(CoreTensor, "J2ca_thread" + t, {nc, na});
        JK[i] = ambit::Tensor::build(CoreTensor, "JKca_thread" + t, {nc, na});

        for (int n = 0; n < N; ++n) {
            std::string name = "Ma" + std::to_string(n) + "_thread" + t;
            Ma[i].push_back(ambit::Tensor::build(CoreTensor, name, {na, na}));
        }
    }

    // 3-index integrals (P|mv)
    auto Beu = ambit::Tensor::build(CoreTensor, "Beu", {nv, na, nQ});
    Beu("eug") = ints_->three_integral_block(aux_mos_, virt_mos_, actv_mos_)("geu");
    auto& Bu_vec = Beu.data();

    // pointer to 1-particle density
    auto L1 = L1_.block("aa");

    for (size_t c_batch = 0, c_shift = 0; c_batch < nbatches; ++c_batch) {
        const auto& c_batch_vir_mos = batch_vir[c_batch];
        auto c_nvir = c_batch_vir_mos.size();
        auto Bc = ambit::Tensor::build(ambit::CoreTensor, "Bc", {c_nvir, nc, nQ});
        Bc("aig") = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, core_mos_)("gai");
        auto& Bc_vec = Bc.data();

        for (size_t d_batch = c_batch, d_shift = c_shift; d_batch < nbatches; ++d_batch) {
            const auto& d_batch_vir_mos = batch_vir[d_batch];
            auto d_nvir = d_batch_vir_mos.size();
            ambit::Tensor Bd;
            if (d_batch == c_batch) {
                Bd = Bc;
            } else {
                Bd = ambit::Tensor::build(ambit::CoreTensor, "Bd", {d_nvir, nc, nQ});
                Bd("aig") =
                    ints_->three_integral_block(aux_mos_, d_batch_vir_mos, core_mos_)("gai");
            }
            auto& Bd_vec = Bd.data();

            // index pairs for c and d
            std::vector<std::pair<size_t, size_t>> cd_pairs;
            if (d_batch == c_batch) {
                for (size_t c = 0; c < c_nvir; ++c) {
                    for (size_t d = c; d < d_nvir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            } else {
                for (size_t c = 0; c < c_nvir; ++c) {
                    for (size_t d = 0; d < d_nvir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            }
            size_t cd_pairs_size = cd_pairs.size();

#pragma omp parallel for num_threads(nthreads)
            for (size_t p = 0; p < cd_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                auto c = cd_pairs[p].first;
                auto d = cd_pairs[p].second;

                auto fock_c = Fdiag_[c_batch_vir_mos[c]];
                auto fock_d = Fdiag_[d_batch_vir_mos[d]];

                // compute (ic|vd) for given indices c and d
                double* Bci_ptr = &Bc_vec[c * nQc];
                double* Bdv_ptr = &Bu_vec[(d + d_shift) * nQa];

                C_DGEMM('N', 'T', nc, na, nQ, 1.0, Bci_ptr, nQ, Bdv_ptr, nQ, 0.0,
                        J1[thread].data().data(), na);

                J1[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom =
                        Fdiag_[core_mos_[i[0]]] + Fdiag_[actv_mos_[i[1]]] - fock_c - fock_d;
                    value *= dsrg_source_->compute_renormalized_denominator(denom);
                });

                // compute (id|vc) for given indices c and d
                double* Bdi_ptr = &Bd_vec[d * nQc];
                double* Bcv_ptr = &Bu_vec[(c + c_shift) * nQa];

                C_DGEMM('N', 'T', nc, na, nQ, 1.0, Bdi_ptr, nQ, Bcv_ptr, nQ, 0.0,
                        J2[thread].data().data(), na);

                J2[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom =
                        Fdiag_[core_mos_[i[0]]] + Fdiag_[actv_mos_[i[1]]] - fock_c - fock_d;
                    value *= dsrg_source_->compute_renormalized_denominator(denom);
                });

                auto factor = 0.5 * ((c_batch_vir_mos[c] == d_batch_vir_mos[d]) ? 0.5 : 1.0);

                JK[thread]("pq") = 2.0 * J1[thread]("pq") - J2[thread]("pq");
                Dc[thread]("im") -= factor * J1[thread]("iu") * JK[thread]("mv") * L1("uv");
                Da[thread]("xu") -= factor * J1[thread]("mx") * JK[thread]("mv") * L1("uv");

                for (int n = 0; n < N; ++n) {
                    const auto& M1c = oetens[n].block("cc");
                    const auto& M1a = oetens[n].block("aa");
                    Ma[thread][n]("vu") -= factor * M1c("mi") * J1[thread]("iu") * JK[thread]("mv");
                    Ma[thread][n]("vu") -= factor * M1a("ux") * J1[thread]("mx") * JK[thread]("mv");
                }

                JK[thread]("pq") = 2.0 * J2[thread]("pq") - J1[thread]("pq");
                Dc[thread]("im") -= factor * J2[thread]("iu") * JK[thread]("mv") * L1("uv");
                Da[thread]("xu") -= factor * J2[thread]("mx") * JK[thread]("mv") * L1("uv");

                for (int n = 0; n < N; ++n) {
                    const auto& M1c = oetens[n].block("cc");
                    const auto& M1a = oetens[n].block("aa");
                    Ma[thread][n]("vu") -= factor * M1c("mi") * J2[thread]("iu") * JK[thread]("mv");
                    Ma[thread][n]("vu") -= factor * M1a("ux") * J2[thread]("mx") * JK[thread]("mv");
                }
            }
            d_shift += d_nvir;
        }
        c_shift += c_nvir;
    }

    // add Daa and Dcc contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("aa")("pq") += Da[i]("pq");
        D1.block("aa")("pq") += Da[i]("qp");
        D1.block("cc")("pq") += Dc[i]("pq");
        D1.block("cc")("pq") += Dc[i]("qp");
    }

    // add Maa contributions to Mbar1_
    for (int n = 0; n < N; ++n) {
        const auto& M1 = Mbar1_.at(n).block("aa");
        for (int i = 0; i < nthreads; ++i) {
            M1("pq") += Ma[i][n]("pq");
            M1("qp") += Ma[i][n]("pq");
        }
    }

    print_done(t_cavv.stop());
}

void SA_MRPT2::compute_1rdm_vv_CAVV_DF(ambit::BlockedTensor& D1,
                                       const std::vector<ambit::BlockedTensor>& oetens) {
    /**
     * Compute the virtual-virtual parts of the unrelaxed spin-summed 1-RDM from T2 cavv block.
     *
     * D1["ba"] += 0.5 * (T2["iuac"] * S2["ivbc"] + T2["iuca"] * S2["ivcb"]) * L1["uv"]
     *           + 0.5 * (T2["iubc"] * S2["ivac"] + T2["iucb"] * S2["ivca"]) * L1["uv"]
     *
     * where S2["ivbc"] = 2.0 * T2["ivbc"] - T2["ivcb"]
     * for core index i; active indices u, v; and virtual indices a, b, c.
     *
     * Amplitudes are built in batches for every ic pairs.
     *
     * If oetens is not empty, the active part of the transformed one-electron integrals
     * are also computed inside the batches.
     *
     * Mt["vu"] += 0.5 * M["ab"] * (T2["iuac"] * S2["ivbc"] + T2["iuca"] * S2["ivcb"])
     *           + 0.5 * M["ab"] * (T2["iubc"] * S2["ivac"] + T2["iucb"] * S2["ivca"])
     *           + 0.5 * M["ab"] * (T2["ivac"] * S2["iubc"] + T2["ivca"] * S2["iucb"])
     *           + 0.5 * M["ab"] * (T2["ivbc"] * S2["iuac"] + T2["ivcb"] * S2["iuca"])
     */
    timer t_cavv("Compute CAVV 1RDM VV term DF");
    print_contents("Computing DF CAVV 1RDM VV part");

    auto N = static_cast<int>(oetens.size());

    auto nQ = aux_mos_.size();
    auto nv = virt_mos_.size();
    auto na = actv_mos_.size();
    auto nc = core_mos_.size();
    auto nQv = nQ * nv;
    auto nQa = nQ * na;

    // test memory
    int nthreads = std::min(n_threads_, int(nv * nc));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * (nv * nv + 3 * na * nv + na * na * N) + nQa * nv;
    if ((memory_min + nQv) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CAVV) 1RDM VV.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQv) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CAVV) 1RDM VV!");
    }

    size_t max_occ = (memory_avai / sizeof(double) - memory_min) / nQv;
    if (max_occ < nc) {
        outfile->Printf("\n -> DF-DSRG-PT2(CAVV) 1RDM VV to be run in batches: max core size = %zu",
                        max_occ);
    } else {
        max_occ = nc;
    }

    // batches of occupied indices
    std::vector<std::vector<size_t>> batch_occ;
    batch_occ = split_vector(core_mos_, max_occ);
    auto nbatches = batch_occ.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> Dv(nthreads);
    std::vector<ambit::Tensor> J1(nthreads), J2(nthreads), JK(nthreads);
    std::vector<std::vector<ambit::Tensor>> Ma(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Dv[i] = ambit::Tensor::build(CoreTensor, "Dvv_thread" + t, {nv, nv});
        J1[i] = ambit::Tensor::build(CoreTensor, "J1av_thread" + t, {nv, na});
        J2[i] = ambit::Tensor::build(CoreTensor, "J2av_thread" + t, {nv, na});
        JK[i] = ambit::Tensor::build(CoreTensor, "JKav_thread" + t, {nv, na});

        for (int n = 0; n < N; ++n) {
            std::string name = "Ma" + std::to_string(n) + "_thread" + t;
            Ma[i].push_back(ambit::Tensor::build(CoreTensor, name, {na, na}));
        }
    }

    // 3-index integrals (P|eu)
    auto Beu = ambit::Tensor::build(CoreTensor, "Beu", {nv, na, nQ});
    Beu("evg") = ints_->three_integral_block(aux_mos_, virt_mos_, actv_mos_)("gev");
    auto& Bu_vec = Beu.data();

    // 1-particle density pointer
    auto L1 = L1_.block("aa");

    for (size_t i_batch = 0; i_batch < nbatches; ++i_batch) {
        const auto& i_batch_occ_mos = batch_occ[i_batch];
        auto i_nocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Ba", {i_nocc, nv, nQ});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, virt_mos_)("gia");
        auto& Bi_vec = Bi.data();

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
            double* Bia_ptr = &Bi_vec[i * nQv];
            double* Bcu_ptr = &Bu_vec[c * nQa];

            C_DGEMM('N', 'T', nv, na, nQ, 1.0, Bia_ptr, nQ, Bcu_ptr, nQ, 0.0,
                    J1[thread].data().data(), na);

            J1[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = fock_i + Fdiag_[actv_mos_[i[1]]] - fock_c - Fdiag_[virt_mos_[i[0]]];
                value *= dsrg_source_->compute_renormalized_denominator(denom);
            });

            // compute (ic|ua) for given indices i and c
            double* Bic_ptr = &Bi_vec[i * nQv + c * nQ];

            C_DGEMV('N', nv * na, nQ, 1.0, Bu_vec.data(), nQ, Bic_ptr, 1, 0.0,
                    J2[thread].data().data(), 1);

            J2[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = fock_i + Fdiag_[actv_mos_[i[1]]] - fock_c - Fdiag_[virt_mos_[i[0]]];
                value *= dsrg_source_->compute_renormalized_denominator(denom);
            });

            JK[thread]("pq") = 2.0 * J1[thread]("pq") - J2[thread]("pq");
            Dv[thread]("ba") += 0.5 * J1[thread]("au") * JK[thread]("bv") * L1("uv");

            for (int n = 0; n < N; ++n) {
                Ma[thread][n]("vu") +=
                    0.5 * oetens[n].block("vv")("ab") * J1[thread]("au") * JK[thread]("bv");
            }

            JK[thread]("pq") = 2.0 * J2[thread]("pq") - J1[thread]("pq");
            Dv[thread]("ba") += 0.5 * J2[thread]("au") * JK[thread]("bv") * L1("uv");

            for (int n = 0; n < N; ++n) {
                Ma[thread][n]("vu") +=
                    0.5 * oetens[n].block("vv")("ab") * J2[thread]("au") * JK[thread]("bv");
            }
        }
    }

    // add Dvv contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("vv")("pq") += Dv[i]("pq");
        D1.block("vv")("pq") += Dv[i]("qp");
    }

    // add Maa contributions to Mbar1_
    for (int n = 0; n < N; ++n) {
        const auto& M1 = Mbar1_.at(n).block("aa");
        for (int i = 0; i < nthreads; ++i) {
            M1("pq") += Ma[i][n]("pq");
            M1("qp") += Ma[i][n]("pq");
        }
    }

    print_done(t_cavv.stop());
}
} // namespace forte
