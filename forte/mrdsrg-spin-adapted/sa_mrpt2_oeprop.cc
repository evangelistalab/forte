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
        // outfile->Printf("\n  tensor %d: Mref = %20.15f", i, Mbar0_[i]);
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
    ambit::BlockedTensor O1, O2, G2, temp1, temp2;
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
    ambit::BlockedTensor D1;
    if (eri_df_ and max_body > 1) {
        D1 = BTF_->build(tensor_type_, "D1", {"gg"});
        compute_1rdm_cc_CCVV_DF(D1);
        compute_1rdm_vv_CCVV_DF(D1);

        compute_1rdm_cc_CCAV_DF(D1, oetens);
        compute_1rdm_aa_vv_CCAV_DF(D1, oetens);

        compute_1rdm_cc_aa_CAVV_DF(D1, oetens);
        compute_1rdm_vv_CAVV_DF(D1, oetens);
    }

    // transform each tensor
    auto Md = BTF_->build(tensor_type_, "M_D", {"cc", "aa", "vv"});
    auto Mod = BTF_->build(tensor_type_, "M_OD", od_one_labels_ph());
    for (int i = 0; i < n_tensors; ++i) {
        local_timer t_local;
        const auto& M = oetens[i];
        print_contents("Transforming " + M.name());

        // separate M to diagonal and off-diagonal components
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
        }

        // add 1-body results
        Mbar1["uv"] += temp1["uv"];
        Mbar1["vu"] += temp1["uv"];

        print_done(t_local.get());

        // outfile->Printf("\n  tensor %d: Mbar0_ = %20.15f", i, Mbar0_[i]);
        // outfile->Printf(" Mbar1 norm = %20.15f", Mbar1.norm());
        // outfile->Printf(" Mbar2 norm = %20.15f", max_levels[i] > 1 ? Mbar2_[i].norm() : 0.0);
    }
}

void SA_MRPT2::compute_1rdm_cc_CCVV_DF(ambit::BlockedTensor& D1) {
    /**
     * Compute the core-core part of the MP2-like unrelaxed spin-summed 1-RDM.
     *
     * D1["ij"] -= T2["ikab"] * S2["jkab"] + T2["jkab"] * S2["ikab"]
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

    if (nc == 0 or nv == 0) {
        print_done(t_ccvv.stop(), "Skipped");
        return;
    }

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

    // batches of virtual indices
    size_t max_vir = (memory_avai / sizeof(double) - memory_min) / (2 * nQc);
    if (max_vir < nv) {
        outfile->Printf("\n -> DF-DSRG-PT2(CCVV) 1RDM CC to be run in batches: max virt size = %zu",
                        max_vir);
    } else {
        max_vir = nv;
    }
    auto batch_vir = split_vector(virt_mos_, max_vir);
    auto nbatches = batch_vir.size();

    // use MP2 amplitudes instead of DSRG
    bool complete_ccvv = (ccvv_source_ == "ZERO");

    // temp tensors for each thread
    std::vector<ambit::Tensor> Dc(nthreads);
    std::vector<ambit::Tensor> Jmn(nthreads), JKmn(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Dc[i] = ambit::Tensor::build(CoreTensor, "Dcc_thread" + t, {nc, nc});
        Jmn[i] = ambit::Tensor::build(CoreTensor, "Jcc_thread" + t, {nc, nc});
        JKmn[i] = ambit::Tensor::build(CoreTensor, "JKcc_thread" + t, {nc, nc});
    }

    bool Bcv_file_exist =
        !semi_checked_results_["RESTRICTED_DOCC"] or !semi_checked_results_["RESTRICTED_UOCC"];

    for (size_t c_batch = 0, c_shift = 0; c_batch < nbatches; ++c_batch) {
        const auto& c_batch_vir_mos = batch_vir[c_batch];
        auto c_nvir = c_batch_vir_mos.size();
        ambit::Tensor Bc; // ckQ
        if (Bcv_file_exist)
            Bc = read_Bcanonical("vc", {c_shift, c_shift + c_nvir}, {0, nc}, pqQ);
        else
            Bc = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, core_mos_, pqQ);
        auto& Bc_data = Bc.data();

        for (size_t d_batch = c_batch, d_shift = c_shift; d_batch < nbatches; ++d_batch) {
            const auto& d_batch_vir_mos = batch_vir[d_batch];
            auto d_nvir = d_batch_vir_mos.size();
            ambit::Tensor Bd;
            if (d_batch == c_batch) {
                Bd = Bc;
            } else {
                ambit::Tensor Bd; // dlQ
                if (Bcv_file_exist)
                    Bd = read_Bcanonical("vc", {d_shift, d_shift + d_nvir}, {0, nc}, pqQ);
                else
                    Bd = ints_->three_integral_block(aux_mos_, d_batch_vir_mos, core_mos_, pqQ);
            }
            auto& Bd_data = Bd.data();

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

                double* Bci_ptr = &Bc_data[c * nQc];
                double* Bdj_ptr = &Bd_data[d * nQc];

                // compute (ci|dj) for given indices c and d
                psi::C_DGEMM('N', 'T', nc, nc, nQ, 1.0, Bci_ptr, nQ, Bdj_ptr, nQ, 0.0,
                             Jmn[thread].data().data(), nc);

                if (complete_ccvv) {
                    Jmn[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                        value /=
                            Fdiag_[core_mos_[i[0]]] + Fdiag_[core_mos_[i[1]]] - fock_c - fock_d;
                    });
                } else {
                    Jmn[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                        double denom =
                            Fdiag_[core_mos_[i[0]]] + Fdiag_[core_mos_[i[1]]] - fock_c - fock_d;
                        value *= dsrg_source_->compute_renormalized_denominator(denom);
                    });
                }
                JKmn[thread]("pq") = 2.0 * Jmn[thread]("pq") - Jmn[thread]("qp");

                auto factor = (c_batch_vir_mos[c] == d_batch_vir_mos[d]) ? 0.5 : 1.0;
                Dc[thread]("ij") -= factor * Jmn[thread]("ik") * JKmn[thread]("jk");
                Dc[thread]("ij") -= factor * Jmn[thread]("ki") * JKmn[thread]("kj");
            }
            d_shift += d_nvir;
        }
        c_shift += c_nvir;
    }

    // collect all results to the first thread
    for (int i = 1; i < nthreads; ++i) {
        Dc[0]("pq") += Dc[i]("pq");
    }

    // rotate back to original basis
    if (!semi_checked_results_["RESTRICTED_DOCC"]) {
        auto X = ambit::Tensor::build(CoreTensor, "Dcc_U", {nc, nc});
        const auto& Ucc = U_.block("cc");
        X("kj") = Dc[0]("kl") * Ucc("lj");
        Dc[0]("ij") = Ucc("ki") * X("kj");
    }

    // add Dcc contributions to D1
    D1.block("cc")("pq") += Dc[0]("pq");
    D1.block("cc")("pq") += Dc[0]("qp");

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

    if (nc == 0 or nv == 0) {
        print_done(t_ccvv.stop(), "Skipped");
        return;
    }

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

    // batches of occupied indices
    size_t max_occ = (memory_avai / sizeof(double) - memory_min) / (2 * nQv);
    if (max_occ < nc) {
        outfile->Printf("\n -> DF-DSRG-PT2(CCVV) 1RDM VV to be run in batches: max core size = %zu",
                        max_occ);
    } else {
        max_occ = nc;
    }
    auto batch_occ = split_vector(core_mos_, max_occ);
    auto nbatches = batch_occ.size();

    // use MP2 amplitudes instead of DSRG
    bool complete_ccvv = (ccvv_source_ == "ZERO");

    // temp tensors for each thread
    std::vector<ambit::Tensor> Dv(nthreads);
    std::vector<ambit::Tensor> Jab(nthreads), JKab(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Dv[i] = ambit::Tensor::build(CoreTensor, "Dvv_thread" + t, {nv, nv});
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jvv_thread" + t, {nv, nv});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKvv_thread" + t, {nv, nv});
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

#pragma omp parallel for num_threads(nthreads)
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

                if (complete_ccvv) {
                    Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                        value /=
                            fock_i + fock_j - Fdiag_[virt_mos_[i[0]]] - Fdiag_[virt_mos_[i[1]]];
                    });
                } else {
                    Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                        double denom =
                            fock_i + fock_j - Fdiag_[virt_mos_[i[0]]] - Fdiag_[virt_mos_[i[1]]];
                        value *= dsrg_source_->compute_renormalized_denominator(denom);
                    });
                }
                JKab[thread]("pq") = 2.0 * Jab[thread]("pq") - Jab[thread]("qp");

                auto factor = (i_batch_occ_mos[i] == j_batch_occ_mos[j]) ? 0.5 : 1.0;
                Dv[thread]("ab") += factor * Jab[thread]("ac") * JKab[thread]("bc");
                Dv[thread]("ab") += factor * Jab[thread]("ca") * JKab[thread]("cb");
            }
            j_shift += j_nocc;
        }
        i_shift += i_nocc;
    }

    // collect all results to the first thread
    for (int i = 1; i < nthreads; ++i) {
        Dv[0]("pq") += Dv[i]("pq");
    }

    // rotate back to original basis
    if (!semi_checked_results_["RESTRICTED_UOCC"]) {
        auto X = ambit::Tensor::build(CoreTensor, "Dvv_U", {nv, nv});
        const auto& Uvv = U_.block("vv");
        X("cb") = Dv[0]("cd") * Uvv("db");
        Dv[0]("ab") = Uvv("ca") * X("cb");
    }

    // add Dvv contributions to D1
    D1.block("vv")("pq") += Dv[0]("pq");
    D1.block("vv")("pq") += Dv[0]("qp");

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

    if (nc == 0 or nv == 0 or na == 0) {
        print_done(t_ccav.stop(), "Skipped");
        return;
    }

    // check semi-canonical orbitals
    bool semi_c = semi_checked_results_.at("RESTRICTED_DOCC");
    bool semi_v = semi_checked_results_.at("RESTRICTED_UOCC");
    bool semi_a = semi_checked_results_.at("ACTIVE");

    // test memory
    int nthreads = std::min(n_threads_, int(nv));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * (nc * (3 * nac + nc) + N * na * na) + nQc * na;
    memory_min += (semi_c ? 0 : N * nc * nc);
    if ((memory_min + nQc) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CCAV) 1RDM CC.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQc) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CCAV) 1RDM CC!");
    }

    // batches of virtual indices
    size_t max_vir = (memory_avai / sizeof(double) - memory_min) / nQc;
    if (max_vir < nv) {
        outfile->Printf("\n -> DF-DSRG-PT2(CCAV) 1RDM CC to be run in batches: max virt size = %zu",
                        max_vir);
    } else {
        max_vir = nv;
    }
    auto batch_vir = split_vector(virt_mos_, max_vir);
    auto nbatches = batch_vir.size();

    // rotate one-electron integrals and 1-hole density to semicanonical basis
    ambit::Tensor E1;
    std::vector<ambit::Tensor> oei_cc(N);
    if (semi_c) {
        for (int n = 0; n < N; ++n) {
            oei_cc[n] = oetens[n].block("cc");
        }
    } else {
        const auto& Ucc = U_.block("cc");
        auto X = ambit::Tensor::build(CoreTensor, "oei_temp", {nc, nc});
        for (int n = 0; n < N; ++n) {
            oei_cc[n] = ambit::Tensor::build(CoreTensor, oetens[n].block("cc").name(), {nc, nc});
            X("kj") = Ucc("ki") * oetens[n].block("cc")("ij");
            oei_cc[n]("kl") = X("kj") * Ucc("lj");
        }
    }
    if (semi_a) {
        E1 = Eta1_.block("aa");
    } else {
        const auto& Uaa = U_.block("aa");
        E1 = ambit::Tensor::build(CoreTensor, "E1can", {na, na});
        E1("xy") = Uaa("xu") * Eta1_.block("aa")("uv") * Uaa("yv");
    }

    // temp tensors for each thread
    std::vector<ambit::Tensor> Dc(nthreads);
    std::vector<ambit::Tensor> T2(nthreads), S2(nthreads), X2(nthreads);
    std::vector<std::vector<ambit::Tensor>> Ma(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        std::string t = std::to_string(i);
        Dc[i] = ambit::Tensor::build(CoreTensor, "Dcc_thread" + t, {nc, nc});
        T2[i] = ambit::Tensor::build(CoreTensor, "T2acc_thread" + t, {na, nc, nc});
        S2[i] = ambit::Tensor::build(CoreTensor, "S2acc_thread" + t, {na, nc, nc});
        X2[i] = ambit::Tensor::build(CoreTensor, "X2acc_thread" + t, {na, nc, nc});

        for (int n = 0; n < N; ++n) {
            std::string name = "Ma" + std::to_string(n) + "_thread" + t;
            Ma[i].push_back(ambit::Tensor::build(CoreTensor, name, {na, na}));
        }
    }

    // 3-index integrals (P|mv)
    ambit::Tensor Bmv; // vmQ
    if (!semi_c or !semi_a)
        Bmv = read_Bcanonical("ac", {0, na}, {0, nc}, pqQ);
    else
        Bmv = ints_->three_integral_block(aux_mos_, actv_mos_, core_mos_, pqQ);
    auto& Bu_data = Bmv.data();

    for (size_t c_batch = 0, c_shift = 0; c_batch < nbatches; ++c_batch) {
        const auto& c_batch_vir_mos = batch_vir[c_batch];
        auto c_nvir = c_batch_vir_mos.size();
        ambit::Tensor Bc; // aiQ
        if (!semi_c or !semi_v)
            Bc = read_Bcanonical("vc", {c_shift, c_shift + c_nvir}, {0, nc}, pqQ);
        else
            Bc = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, core_mos_, pqQ);
        auto& Bc_data = Bc.data();

#pragma omp parallel for num_threads(nthreads)
        for (size_t c = 0; c < c_nvir; ++c) {
            int thread = omp_get_thread_num();

            auto fock_c = Fdiag_[c_batch_vir_mos[c]];
            double* Bci_ptr = &Bc_data[c * nQc];

            // form T2["ikvc"] for given c
            psi::C_DGEMM('N', 'T', nac, nc, nQ, 1.0, Bu_data.data(), nQ, Bci_ptr, nQ, 0.0,
                         T2[thread].data().data(), nc);
            T2[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = Fdiag_[core_mos_[i[1]]] + Fdiag_[core_mos_[i[2]]] - fock_c -
                               Fdiag_[actv_mos_[i[0]]];
                value *= dsrg_source_->compute_renormalized_denominator(denom);
            });
            S2[thread]("umn") = 2.0 * T2[thread]("umn") - T2[thread]("unm");

            X2[thread]("umn") = T2[thread]("vmn") * E1("uv");

            Dc[thread]("ij") -= 0.5 * X2[thread]("uik") * S2[thread]("ujk");
            Dc[thread]("ij") -= 0.5 * X2[thread]("uki") * S2[thread]("ukj");

            for (int n = 0; n < N; ++n) {
                Ma[thread][n]("vu") -=
                    0.5 * oei_cc[n]("ji") * T2[thread]("vik") * S2[thread]("ujk");
                Ma[thread][n]("vu") -=
                    0.5 * oei_cc[n]("ji") * T2[thread]("vki") * S2[thread]("ukj");
            }
        }
        c_shift += c_nvir;
    }

    // collect results to the first thread
    for (int i = 1; i < nthreads; ++i) {
        Dc[0]("pq") += Dc[i]("pq");
        for (int n = 0; n < N; ++n) {
            Ma[0][n]("pq") += Ma[i][n]("pq");
        }
    }

    // rotate back to original basis
    if (!semi_c) {
        auto X = ambit::Tensor::build(CoreTensor, "Xcc_U", {nc, nc});
        const auto& Ucc = U_.block("cc");
        X("kj") = Dc[0]("kl") * Ucc("lj");
        Dc[0]("ij") = Ucc("ki") * X("kj");
    }
    if (!semi_a) {
        const auto& Uaa = U_.block("aa");
        auto X = ambit::Tensor::build(CoreTensor, "Xaa_U", {na, na});
        for (int n = 0; n < N; ++n) {
            X("xv") = Ma[0][n]("xy") * Uaa("yv");
            Ma[0][n]("uv") = Uaa("xu") * X("xv");
        }
    }

    // add Dcc contributions to D1
    D1.block("cc")("pq") += Dc[0]("pq");
    D1.block("cc")("pq") += Dc[0]("qp");

    // add Maa contributions to Mbar1_
    for (int n = 0; n < N; ++n) {
        const auto& M1 = Mbar1_.at(n).block("aa");
        M1("pq") -= Ma[0][n]("pq");
        M1("qp") -= Ma[0][n]("pq");
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

    if (nc == 0 or nv == 0 or na == 0) {
        print_done(t_ccav.stop(), "Skipped");
        return;
    }

    // check semi-canonical orbitals
    bool semi_c = semi_checked_results_.at("RESTRICTED_DOCC");
    bool semi_v = semi_checked_results_.at("RESTRICTED_UOCC");
    bool semi_a = semi_checked_results_.at("ACTIVE");

    // test memory
    int nthreads = std::min(n_threads_, int(nc * (nc + 1) / 2));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * (3 * na * nv + na * na + nv * nv + na * na * N) + nQa * nc;
    memory_min += (semi_v ? 0 : N * nv * nv) + (semi_a ? 0 : N * na * na);
    if ((memory_min + nQv * 2) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CCAV) 1RDM AA/VV.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQv * 2) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CCAV) 1RDM AA/VV!");
    }

    // batches of occupied indices
    size_t max_occ = (memory_avai / sizeof(double) - memory_min) / (2 * nQv);
    if (max_occ < nc) {
        outfile->Printf(
            "\n -> DF-DSRG-PT2(CCAV) 1RDM AA/VV to be run in batches: max core size = %zu",
            max_occ);
    } else {
        max_occ = nc;
    }
    auto batch_occ = split_vector(core_mos_, max_occ);
    auto nbatches = batch_occ.size();

    // rotate one-electron integrals and 1-hole density to semi-canonical basis
    ambit::Tensor E1;
    std::vector<ambit::Tensor> oei_vv(N), oei_aa(N);
    if (semi_v) {
        for (int n = 0; n < N; ++n) {
            oei_vv[n] = oetens[n].block("vv");
        }
    } else {
        const auto& Uvv = U_.block("vv");
        auto X = ambit::Tensor::build(CoreTensor, "oei_temp", {nv, nv});
        for (int n = 0; n < N; ++n) {
            oei_vv[n] = ambit::Tensor::build(CoreTensor, oetens[n].block("vv").name(), {nv, nv});
            X("cb") = Uvv("ca") * oetens[n].block("vv")("ab");
            oei_vv[n]("cd") = X("cb") * Uvv("db");
        }
    }
    if (semi_a) {
        E1 = Eta1_.block("aa");
        for (int n = 0; n < N; ++n) {
            oei_aa[n] = oetens[n].block("aa");
        }
    } else {
        const auto& Uaa = U_.block("aa");
        auto X = ambit::Tensor::build(CoreTensor, "oei_temp", {na, na});
        for (int n = 0; n < N; ++n) {
            oei_aa[n] = ambit::Tensor::build(CoreTensor, oetens[n].block("aa").name(), {na, na});
            X("xv") = Uaa("xu") * oetens[n].block("aa")("uv");
            oei_aa[n]("xy") = X("xv") * Uaa("yv");
        }

        E1 = ambit::Tensor::build(CoreTensor, "E1can", {na, na});
        X("xv") = Uaa("xu") * Eta1_.block("aa")("uv");
        E1("xy") = X("xv") * Uaa("yv");
    }

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
    ambit::Tensor Bmv; // mvQ
    if (!semi_c or !semi_a)
        Bmv = read_Bcanonical("ca", {0, nc}, {0, na}, pqQ);
    else
        Bmv = ints_->three_integral_block(aux_mos_, core_mos_, actv_mos_, pqQ);
    auto& Bu_data = Bmv.data();

    for (size_t i_batch = 0, i_shift = 0; i_batch < nbatches; ++i_batch) {
        const auto& i_batch_occ_mos = batch_occ[i_batch];
        auto i_nocc = i_batch_occ_mos.size();
        ambit::Tensor Bi; // iaQ
        if (!semi_c or !semi_v)
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
                if (!semi_c or !semi_v)
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

#pragma omp parallel for num_threads(nthreads)
            for (size_t p = 0; p < ij_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                size_t i = ij_pairs[p].first;
                size_t j = ij_pairs[p].second;

                auto fock_i = Fdiag_[i_batch_occ_mos[i]];
                auto fock_j = Fdiag_[j_batch_occ_mos[j]];

                // compute (ia|jv) for given indices i and j
                double* Bia_ptr = &Bi_data[i * nQv];
                double* Bjv_ptr = &Bu_data[(j + j_shift) * nQa];

                psi::C_DGEMM('N', 'T', nv, na, nQ, 1.0, Bia_ptr, nQ, Bjv_ptr, nQ, 0.0,
                             J1[thread].data().data(), na);

                J1[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom =
                        fock_i + fock_j - Fdiag_[virt_mos_[i[0]]] - Fdiag_[actv_mos_[i[1]]];
                    value *= dsrg_source_->compute_renormalized_denominator(denom);
                });

                // compute (ja|iv) for given indices i and j
                double* Bja_ptr = &Bj_data[j * nQv];
                double* Biv_ptr = &Bu_data[(i + i_shift) * nQa];

                psi::C_DGEMM('N', 'T', nv, na, nQ, 1.0, Bja_ptr, nQ, Biv_ptr, nQ, 0.0,
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
                    Ma[thread][n]("vu") +=
                        factor * oei_aa[n]("yv") * J1[thread]("ey") * JK[thread]("eu");
                    Ma[thread][n]("vu") +=
                        factor * oei_vv[n]("ae") * J1[thread]("av") * JK[thread]("eu");
                }

                JK[thread]("pq") = 2.0 * J2[thread]("pq") - J1[thread]("pq");
                Da[thread]("vy") += factor * J2[thread]("ey") * JK[thread]("eu") * E1("uv");
                Dv[thread]("ea") += factor * J2[thread]("av") * JK[thread]("eu") * E1("uv");

                for (int n = 0; n < N; ++n) {
                    Ma[thread][n]("vu") +=
                        factor * oei_aa[n]("yv") * J2[thread]("ey") * JK[thread]("eu");
                    Ma[thread][n]("vu") +=
                        factor * oei_vv[n]("ae") * J2[thread]("av") * JK[thread]("eu");
                }
            }
            j_shift += j_nocc;
        }
        i_shift += i_nocc;
    }

    // collect results to the first thread
    for (int i = 1; i < nthreads; ++i) {
        Da[0]("pq") += Da[i]("pq");
        Dv[0]("pq") += Dv[i]("pq");
        for (int n = 0; n < N; ++n) {
            Ma[0][n]("pq") += Ma[i][n]("pq");
        }
    }

    // rotate back to original basis
    if (!semi_v) {
        auto X = ambit::Tensor::build(CoreTensor, "Xvv_U", {nv, nv});
        const auto& Uvv = U_.block("vv");
        X("cb") = Dv[0]("cd") * Uvv("db");
        Dv[0]("ab") = Uvv("ca") * X("cb");
    }
    if (!semi_a) {
        auto X = ambit::Tensor::build(CoreTensor, "Xaa_U", {na, na});
        const auto& Uaa = U_.block("aa");
        X("xv") = Da[0]("xy") * Uaa("yv");
        Da[0]("uv") = Uaa("xu") * X("xv");
        for (int n = 0; n < N; ++n) {
            X("xv") = Ma[0][n]("xy") * Uaa("yv");
            Ma[0][n]("uv") = Uaa("xu") * X("xv");
        }
    }

    // add Daa and Dvv contributions to D1
    D1.block("aa")("pq") += Da[0]("pq");
    D1.block("aa")("pq") += Da[0]("qp");
    D1.block("vv")("pq") += Dv[0]("pq");
    D1.block("vv")("pq") += Dv[0]("qp");

    // add Maa contributions to Mbar1_
    for (int n = 0; n < N; ++n) {
        const auto& M1 = Mbar1_.at(n).block("aa");
        M1("pq") -= Ma[0][n]("pq");
        M1("qp") -= Ma[0][n]("pq");
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

    if (nc == 0 or nv == 0 or na == 0) {
        print_done(t_cavv.stop(), "Skipped");
        return;
    }

    // check semi-canonical orbitals
    bool semi_c = semi_checked_results_.at("RESTRICTED_DOCC");
    bool semi_v = semi_checked_results_.at("RESTRICTED_UOCC");
    bool semi_a = semi_checked_results_.at("ACTIVE");

    // test memory
    int nthreads = std::min(n_threads_, int(nv * (nv + 1) / 2));
    size_t memory_avai = dsrg_mem_.available();
    size_t memory_min = nthreads * (nc * nc + na * na + 3 * na * nc + na * na * N) + nQa * nv;
    memory_min += (semi_c ? 0 : N * nc * nc) + (semi_a ? 0 : N * na * na);
    if ((memory_min + nQc * 2) * sizeof(double) > memory_avai) {
        outfile->Printf("\n  Error: Not enough memory for DF-DSRG-PT2(CAVV) 1RDM CC/AA.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + nQc * 2) * sizeof(double) - memory_avai);
        throw std::runtime_error("Not enough memory to run DF-DSRG-PT2(CAVV) 1RDM CC/AA!");
    }

    // batches of virtual indices
    size_t max_vir = (memory_avai / sizeof(double) - memory_min) / (2 * nQc);
    if (max_vir < nv) {
        outfile->Printf(
            "\n -> DF-DSRG-PT2(CAVV) 1RDM CC/AA to be run in batches: max virt size = %zu",
            max_vir);
    } else {
        max_vir = nv;
    }
    auto batch_vir = split_vector(virt_mos_, max_vir);
    auto nbatches = batch_vir.size();

    // rotate one-electron integrals and 1-RDM to semi-canonical basis
    ambit::Tensor L1;
    std::vector<ambit::Tensor> oei_cc(N), oei_aa(N);
    if (semi_c) {
        for (int n = 0; n < N; ++n) {
            oei_cc[n] = oetens[n].block("cc");
        }
    } else {
        const auto& Ucc = U_.block("cc");
        auto X = ambit::Tensor::build(CoreTensor, "oei_temp", {nc, nc});
        for (int n = 0; n < N; ++n) {
            oei_cc[n] = ambit::Tensor::build(CoreTensor, oetens[n].block("cc").name(), {nc, nc});
            X("kj") = Ucc("ki") * oetens[n].block("cc")("ij");
            oei_cc[n]("kl") = X("kj") * Ucc("lj");
        }
    }
    if (semi_a) {
        L1 = L1_.block("aa");
        for (int n = 0; n < N; ++n) {
            oei_aa[n] = oetens[n].block("aa");
        }
    } else {
        const auto& Uaa = U_.block("aa");
        auto X = ambit::Tensor::build(CoreTensor, "oei_temp", {na, na});
        for (int n = 0; n < N; ++n) {
            oei_aa[n] = ambit::Tensor::build(CoreTensor, oetens[n].block("aa").name(), {na, na});
            X("xv") = Uaa("xu") * oetens[n].block("aa")("uv");
            oei_aa[n]("xy") = X("xv") * Uaa("yv");
        }
        L1 = ambit::Tensor::build(CoreTensor, "L1can", {na, na});
        X("xv") = Uaa("xu") * L1_.block("aa")("uv");
        L1("xy") = X("xv") * Uaa("yv");
    }

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
    ambit::Tensor Beu; // euQ
    if (!semi_a or !semi_v)
        Beu = read_Bcanonical("va", {0, nv}, {0, na}, pqQ);
    else
        Beu = ints_->three_integral_block(aux_mos_, virt_mos_, actv_mos_, pqQ);
    auto& Bu_data = Beu.data();

    for (size_t c_batch = 0, c_shift = 0; c_batch < nbatches; ++c_batch) {
        const auto& c_batch_vir_mos = batch_vir[c_batch];
        auto c_nvir = c_batch_vir_mos.size();
        ambit::Tensor Bc; // ckQ
        if (!semi_c or !semi_v)
            Bc = read_Bcanonical("vc", {c_shift, c_shift + c_nvir}, {0, nc}, pqQ);
        else
            Bc = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, core_mos_, pqQ);
        auto& Bc_data = Bc.data();

        for (size_t d_batch = c_batch, d_shift = c_shift; d_batch < nbatches; ++d_batch) {
            const auto& d_batch_vir_mos = batch_vir[d_batch];
            auto d_nvir = d_batch_vir_mos.size();
            ambit::Tensor Bd; // dlQ
            if (d_batch == c_batch) {
                Bd = Bc;
            } else {
                if (!semi_c or !semi_v)
                    Bd = read_Bcanonical("vc", {d_shift, d_shift + d_nvir}, {0, nc}, pqQ);
                else
                    Bd = ints_->three_integral_block(aux_mos_, d_batch_vir_mos, core_mos_, pqQ);
            }
            auto& Bd_data = Bd.data();

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
                double* Bci_ptr = &Bc_data[c * nQc];
                double* Bdv_ptr = &Bu_data[(d + d_shift) * nQa];

                psi::C_DGEMM('N', 'T', nc, na, nQ, 1.0, Bci_ptr, nQ, Bdv_ptr, nQ, 0.0,
                             J1[thread].data().data(), na);

                J1[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double denom =
                        Fdiag_[core_mos_[i[0]]] + Fdiag_[actv_mos_[i[1]]] - fock_c - fock_d;
                    value *= dsrg_source_->compute_renormalized_denominator(denom);
                });

                // compute (id|vc) for given indices c and d
                double* Bdi_ptr = &Bd_data[d * nQc];
                double* Bcv_ptr = &Bu_data[(c + c_shift) * nQa];

                psi::C_DGEMM('N', 'T', nc, na, nQ, 1.0, Bdi_ptr, nQ, Bcv_ptr, nQ, 0.0,
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
                    Ma[thread][n]("vu") -=
                        factor * oei_cc[n]("mi") * J1[thread]("iu") * JK[thread]("mv");
                    Ma[thread][n]("vu") -=
                        factor * oei_aa[n]("ux") * J1[thread]("mx") * JK[thread]("mv");
                }

                JK[thread]("pq") = 2.0 * J2[thread]("pq") - J1[thread]("pq");
                Dc[thread]("im") -= factor * J2[thread]("iu") * JK[thread]("mv") * L1("uv");
                Da[thread]("xu") -= factor * J2[thread]("mx") * JK[thread]("mv") * L1("uv");

                for (int n = 0; n < N; ++n) {
                    Ma[thread][n]("vu") -=
                        factor * oei_cc[n]("mi") * J2[thread]("iu") * JK[thread]("mv");
                    Ma[thread][n]("vu") -=
                        factor * oei_aa[n]("ux") * J2[thread]("mx") * JK[thread]("mv");
                }
            }
            d_shift += d_nvir;
        }
        c_shift += c_nvir;
    }

    // collect results to the first thread
    for (int i = 1; i < nthreads; ++i) {
        Dc[0]("pq") += Dc[i]("pq");
        Da[0]("pq") += Da[i]("pq");
        for (int n = 0; n < N; ++n) {
            Ma[0][n]("pq") += Ma[i][n]("pq");
        }
    }

    // rotate back to original basis
    if (!semi_c) {
        auto X = ambit::Tensor::build(CoreTensor, "Xcc_U", {nc, nc});
        const auto& Ucc = U_.block("cc");
        X("kj") = Dc[0]("kl") * Ucc("lj");
        Dc[0]("ij") = Ucc("ki") * X("kj");
    }
    if (!semi_a) {
        const auto& Uaa = U_.block("aa");
        auto X = ambit::Tensor::build(CoreTensor, "Xaa_U", {na, na});
        X("xv") = Da[0]("xy") * Uaa("yv");
        Da[0]("uv") = Uaa("xu") * X("xv");
        for (int n = 0; n < N; ++n) {
            X("xv") = Ma[0][n]("xy") * Uaa("yv");
            Ma[0][n]("uv") = Uaa("xu") * X("xv");
        }
    }

    // add Daa and Dcc contributions to D1
    D1.block("aa")("pq") += Da[0]("pq");
    D1.block("aa")("pq") += Da[0]("qp");
    D1.block("cc")("pq") += Dc[0]("pq");
    D1.block("cc")("pq") += Dc[0]("qp");

    // add Maa contributions to Mbar1_
    for (int n = 0; n < N; ++n) {
        const auto& M1 = Mbar1_.at(n).block("aa");
        M1("pq") += Ma[0][n]("pq");
        M1("qp") += Ma[0][n]("pq");
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

    if (nc == 0 or nv == 0 or na == 0) {
        print_done(t_cavv.stop(), "Skipped");
        return;
    }

    // check semi-canonical orbitals
    bool semi_c = semi_checked_results_.at("RESTRICTED_DOCC");
    bool semi_v = semi_checked_results_.at("RESTRICTED_UOCC");
    bool semi_a = semi_checked_results_.at("ACTIVE");

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

    // batches of occupied indices
    size_t max_occ = (memory_avai / sizeof(double) - memory_min) / nQv;
    if (max_occ < nc) {
        outfile->Printf("\n -> DF-DSRG-PT2(CAVV) 1RDM VV to be run in batches: max core size = %zu",
                        max_occ);
    } else {
        max_occ = nc;
    }
    auto batch_occ = split_vector(core_mos_, max_occ);
    auto nbatches = batch_occ.size();

    // rotate one-electron integrals and 1-RDM to semi-canonical basis
    ambit::Tensor L1;
    std::vector<ambit::Tensor> oei_vv(N);
    if (semi_v) {
        for (int n = 0; n < N; ++n) {
            oei_vv[n] = oetens[n].block("vv");
        }
    } else {
        const auto& Uvv = U_.block("vv");
        auto X = ambit::Tensor::build(CoreTensor, "oei_temp", {nv, nv});
        for (int n = 0; n < N; ++n) {
            oei_vv[n] = ambit::Tensor::build(CoreTensor, oetens[n].block("vv").name(), {nv, nv});
            X("cb") = Uvv("ca") * oetens[n].block("vv")("ab");
            oei_vv[n]("cd") = X("cb") * Uvv("db");
        }
    }
    if (semi_a) {
        L1 = L1_.block("aa");
    } else {
        const auto& Uaa = U_.block("aa");
        L1 = ambit::Tensor::build(CoreTensor, "L1can", {na, na});
        L1("xy") = Uaa("xu") * L1_.block("aa")("uv") * Uaa("yv");
    }

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
    ambit::Tensor Beu; // euQ
    if (!semi_a or !semi_v)
        Beu = read_Bcanonical("va", {0, nv}, {0, na}, pqQ);
    else
        Beu = ints_->three_integral_block(aux_mos_, virt_mos_, actv_mos_, pqQ);
    auto& Bu_data = Beu.data();

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

            J1[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = fock_i + Fdiag_[actv_mos_[i[1]]] - fock_c - Fdiag_[virt_mos_[i[0]]];
                value *= dsrg_source_->compute_renormalized_denominator(denom);
            });

            // compute (ic|ua) for given indices i and c
            double* Bic_ptr = &Bi_data[i * nQv + c * nQ];

            psi::C_DGEMV('N', nv * na, nQ, 1.0, Bu_data.data(), nQ, Bic_ptr, 1, 0.0,
                         J2[thread].data().data(), 1);

            J2[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double denom = fock_i + Fdiag_[actv_mos_[i[1]]] - fock_c - Fdiag_[virt_mos_[i[0]]];
                value *= dsrg_source_->compute_renormalized_denominator(denom);
            });

            JK[thread]("pq") = 2.0 * J1[thread]("pq") - J2[thread]("pq");
            Dv[thread]("ba") += 0.5 * J1[thread]("au") * JK[thread]("bv") * L1("uv");

            for (int n = 0; n < N; ++n) {
                Ma[thread][n]("vu") += 0.5 * oei_vv[n]("ab") * J1[thread]("au") * JK[thread]("bv");
            }

            JK[thread]("pq") = 2.0 * J2[thread]("pq") - J1[thread]("pq");
            Dv[thread]("ba") += 0.5 * J2[thread]("au") * JK[thread]("bv") * L1("uv");

            for (int n = 0; n < N; ++n) {
                Ma[thread][n]("vu") += 0.5 * oei_vv[n]("ab") * J2[thread]("au") * JK[thread]("bv");
            }
        }
        i_shift += i_nocc;
    }

    // collect results to the first thread
    for (int i = 1; i < nthreads; ++i) {
        Dv[0]("pq") += Dv[i]("pq");
        for (int n = 0; n < N; ++n) {
            Ma[0][n]("pq") += Ma[i][n]("pq");
        }
    }

    // rotate back to original basis
    if (!semi_v) {
        const auto& Uvv = U_.block("vv");
        auto X = ambit::Tensor::build(CoreTensor, "Xvv_U", {nv, nv});
        X("cb") = Dv[0]("cd") * Uvv("db");
        Dv[0]("ab") = Uvv("ca") * X("cb");
    }
    if (!semi_a) {
        const auto& Uaa = U_.block("aa");
        auto X = ambit::Tensor::build(CoreTensor, "Xaa_U", {na, na});
        for (int n = 0; n < N; ++n) {
            X("xv") = Ma[0][n]("xy") * Uaa("yv");
            Ma[0][n]("uv") = Uaa("xu") * X("xv");
        }
    }

    // add Dvv contributions to D1
    D1.block("vv")("pq") += Dv[0]("pq");
    D1.block("vv")("pq") += Dv[0]("qp");

    // add Maa contributions to Mbar1_
    for (int n = 0; n < N; ++n) {
        const auto& M1 = Mbar1_.at(n).block("aa");
        M1("pq") += Ma[0][n]("pq");
        M1("qp") += Ma[0][n]("pq");
    }

    print_done(t_cavv.stop());
}

std::shared_ptr<psi::Matrix> SA_MRPT2::build_1rdm_cc() {
    timer tcc("1RDM-CC");

    auto D1c = D1_.block("cc");

    if (eri_df_) {
        compute_1rdm_cc_CCVV_DF(D1_);
        compute_1rdm_cc_CCAV_DF(D1_, {});
        compute_1rdm_cc_aa_CAVV_DF(D1_, {});
    } else {
        local_timer t;
        print_contents("Computing CCVV 1RDM CC part");
        D1c("ij") -= 2.0 * T2_.block("ccvv")("ikab") * S2_.block("ccvv")("jkab");
        print_done(t.get());

        t.reset();
        print_contents("Computing CCAV 1RDM CC part");
        D1c("ij") -=
            T2_.block("ccav")("ikva") * S2_.block("ccav")("jkua") * Eta1_.block("aa")("uv");
        D1c("ij") -=
            T2_.block("ccav")("kiva") * S2_.block("ccav")("kjua") * Eta1_.block("aa")("uv");
        print_done(t.get());

        t.reset();
        print_contents("Computing CAVV 1RDM CC part");
        D1c("ij") -= T2_.block("cavv")("iuab") * S2_.block("cavv")("jvab") * L1_.block("aa")("uv");
        print_done(t.get());
    }

    local_timer t;
    print_contents("Computing T1 contr. to 1RDM CC part");

    D1_["mn"] -= 2.0 * T1_["ne"] * T1_["me"];
    D1_["mn"] -= T1_["nv"] * T1_["mu"] * Eta1_["uv"];

    D1_["mn"] += T1_["nx"] * T2_["myuv"] * L2_["uvxy"];
    D1_["mn"] += T1_["mu"] * T2_["nvxy"] * L2_["uvxy"];
    print_done(t.get());

    t.reset();
    print_contents("Computing T2 contr. to 1RDM CC part");

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_D1cc", {"aaaa"});
    temp["uvxy"] = L2_["uvxy"];
    temp["uvxy"] += Eta1_["ux"] * Eta1_["vy"];
    temp["uvxy"] -= 0.5 * Eta1_["vx"] * Eta1_["uy"];
    D1c("ij") -= T2_.block("ccaa")("ikuv") * T2_.block("ccaa")("jkxy") * temp.block("aaaa")("uvxy");

    D1_["mn"] -= 0.5 * T2_["mzuv"] * T2_["nwxy"] * L1_["wz"] * L2_["uvxy"];

    D1_["mn"] -= 0.5 * T2_["ynve"] * S2_["xmue"] * L1_["yx"] * Eta1_["uv"];
    D1_["mn"] -= 0.5 * T2_["nyve"] * S2_["mxue"] * L1_["yx"] * Eta1_["uv"];

    D1_["mn"] -= 0.25 * T2_["nuyz"] * S2_["mvxw"] * L1_["uv"] * Eta1_["xy"] * Eta1_["wz"];

    D1_["mn"] -= T2_["vnye"] * S2_["xmue"] * L2_["uvxy"];
    D1_["mn"] += T2_["nvye"] * T2_["xmue"] * L2_["uvxy"];
    D1_["mn"] += T2_["nvye"] * T2_["mxue"] * L2_["uvyx"];

    D1_["mn"] -= 0.5 * T2_["nvzy"] * S2_["mxwu"] * Eta1_["wz"] * L2_["uvxy"];
    D1_["mn"] += 0.5 * T2_["nvyz"] * T2_["mxwu"] * Eta1_["wz"] * L2_["uvxy"];
    D1_["mn"] += 0.5 * T2_["nvyz"] * T2_["mxuw"] * Eta1_["wz"] * L2_["uvyx"];

    if (do_cu3_) {
        if (store_cu3_) {
            D1c("mn") += T2_.block("caaa")("nwxy") * T2_.block("caaa")("mzuv") * L3_("xyzuwv");
        } else {
            throw std::runtime_error("Direct algorithm for D1 CC not available!");
        }
    }

    print_done(t.get());

    return tensor_to_matrix(D1_.block("cc"), mo_space_info_->dimension("RESTRICTED_DOCC"));
}

std::shared_ptr<psi::Matrix> SA_MRPT2::build_1rdm_vv() {
    timer tvv("1RDM-VV");

    auto D1v = D1_.block("vv");

    if (eri_df_) {
        compute_1rdm_vv_CCVV_DF(D1_);
        compute_1rdm_vv_CAVV_DF(D1_, {});
        compute_1rdm_aa_vv_CCAV_DF(D1_, {});
    } else {
        local_timer t;
        print_contents("Computing CCVV 1RDM VV part");
        D1v("ab") += 2.0 * T2_.block("ccvv")("ijac") * S2_.block("ccvv")("ijbc");
        print_done(t.get());

        t.reset();
        print_contents("Computing CAVV 1RDM VV part");
        D1v("ab") += T2_.block("cavv")("iuac") * S2_.block("cavv")("ivbc") * L1_.block("aa")("uv");
        D1v("ab") += T2_.block("cavv")("iuca") * S2_.block("cavv")("ivcb") * L1_.block("aa")("uv");
        print_done(t.get());

        t.reset();
        print_contents("Computing CCAV 1RDM VV part");
        D1v("ab") +=
            T2_.block("ccav")("ijva") * S2_.block("ccav")("ijub") * Eta1_.block("aa")("uv");
        print_done(t.get());
    }

    local_timer t;
    print_contents("Computing T1 contr. to 1RDM VV part");

    D1_["ef"] += 2.0 * T1_["me"] * T1_["mf"];
    D1_["ef"] += T1_["ue"] * T1_["vf"] * L1_["uv"];

    D1_["ef"] += T1_["ue"] * T2_["xyfv"] * L2_["uvxy"];
    D1_["ef"] += T1_["xf"] * T2_["uvey"] * L2_["uvxy"];
    print_done(t.get());

    t.reset();
    print_contents("Computing T2 contr. to 1RDM VV part");

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_D1vv", {"aaaa"});
    temp["uvxy"] = L2_["uvxy"];
    temp["uvxy"] += L1_["ux"] * L1_["vy"];
    temp["uvxy"] -= 0.5 * L1_["vx"] * L1_["uy"];
    D1v("ab") += T2_.block("aavv")("uvac") * T2_.block("aavv")("xybc") * temp.block("aaaa")("uvxy");

    D1_["ef"] += 0.5 * T2_["uvez"] * T2_["xyfw"] * Eta1_["wz"] * L2_["uvxy"];

    D1_["ef"] += 0.5 * T2_["umxe"] * S2_["vmyf"] * L1_["uv"] * Eta1_["yx"];
    D1_["ef"] += 0.5 * T2_["muxe"] * S2_["mvyf"] * L1_["uv"] * Eta1_["yx"];

    D1_["ef"] += 0.25 * T2_["uxez"] * S2_["vyfw"] * L1_["uv"] * L1_["xy"] * Eta1_["wz"];

    D1_["ef"] += T2_["vmye"] * S2_["xmuf"] * L2_["uvxy"];
    D1_["ef"] -= T2_["mvye"] * T2_["xmuf"] * L2_["uvxy"];
    D1_["ef"] -= T2_["mvye"] * T2_["mxuf"] * L2_["uvyx"];

    D1_["ef"] += 0.5 * T2_["wvey"] * S2_["zxfu"] * L1_["wz"] * L2_["uvxy"];
    D1_["ef"] -= 0.5 * T2_["vwey"] * T2_["zxfu"] * L1_["wz"] * L2_["uvxy"];
    D1_["ef"] -= 0.5 * T2_["vwey"] * T2_["xzfu"] * L1_["wz"] * L2_["uvyx"];

    if (do_cu3_) {
        if (store_cu3_) {
            D1v("ef") += T2_.block("aava")("uvez") * T2_.block("aava")("xyfw") * L3_("xyzuwv");
        } else {
            throw std::runtime_error("Direct algorithm for D1 VV not available!");
        }
    }

    print_done(t.get());

    return tensor_to_matrix(D1_.block("vv"), mo_space_info_->dimension("RESTRICTED_UOCC"));
}

void SA_MRPT2::build_1rdm_unrelaxed(std::shared_ptr<psi::Matrix>& D1c,
                                    std::shared_ptr<psi::Matrix>& D1v) {
    print_h2("Build Spin-Summed Unrelaxed 1-RDM (CC and VV)");

    D1_ = BTF_->build(tensor_type_, "D1u", {"cc", "aa", "vv"});
    D1_.block("cc").iterate([&](const std::vector<size_t> i, double& value) {
        if (i[0] == i[1])
            value = 2.0;
    });
    D1_["uv"] = L1_["uv"];

    D1c = build_1rdm_cc();
    D1v = build_1rdm_vv();
}

} // namespace forte
