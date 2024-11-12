/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "sadsrg.h"

using namespace psi;

namespace forte {

double SADSRG::H2_T2_C0_cu3_direct(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2) {
    // direct algorithm for 3RDM: Alex's trick JCTC 16, 6343–6357 (2020)
    // t_{uvez} v_{ewxy} D_{xyzuwv} = - t_{uvez} v_{ezxy} D_{uvxy}
    //                                + t_{uvez} v_{ewxy} < x^+ y^+ w z^+ v u >

    // - need to transform the integrals to the same orbital basis as ActiveSpaceSolver
    // - TODO: maybe we (York) should make the CI vectors consistent at the first place
    ambit::Tensor Tbra, Tket;
    ambit::Tensor Ua = Uactv_.block("aa");

    timer timer_v("DSRG [H2, T2] D3V direct");
    Tbra = H2.block("vaaa").clone();
    Tbra("ewuv") = H2.block("vaaa")("ezxy") * Ua("wz") * Ua("ux") * Ua("vy");
    Tket = ambit::Tensor::build(tensor_type_, "Tket", Tbra.dims());
    Tket("ewuv") = T2.block("aava")("xyez") * Ua("wz") * Ua("ux") * Ua("vy");
    auto E3v_map = as_solver_->compute_complementary_H2caa_overlap(
        Tket, Tbra, mo_space_info_->symmetry("RESTRICTED_UOCC"), "v", load_mps_);
    timer_v.stop();

    timer timer_c("DSRG [H2, T2] D3C direct");
    Tket = T2.block("caaa").clone();
    Tket("mwuv") = T2.block("caaa")("mzxy") * Ua("wz") * Ua("ux") * Ua("vy");
    Tbra = ambit::Tensor::build(tensor_type_, "Tbra", Tket.dims());
    Tbra("mwuv") = H2.block("aaca")("xymz") * Ua("wz") * Ua("ux") * Ua("vy");
    auto E3c_map = as_solver_->compute_complementary_H2caa_overlap(
        Tket, Tbra, mo_space_info_->symmetry("RESTRICTED_DOCC"), "c", load_mps_);
    timer_c.stop();

    // - 2-RDM contributions
    auto G2 = ambit::BlockedTensor::build(ambit::CoreTensor, "G2", {"aaaa"});
    G2.block("aaaa")("pqrs") = rdms_->SF_G2()("pqrs");

    double E3v = -H2["ezxy"] * T2["uvez"] * G2["xyuv"];
    double E3c = T2["mzxy"] * H2["uvmz"] * G2["xyuv"];

    // - add together
    for (const auto& state_weights : state_to_weights_) {
        const auto& state = state_weights.first;
        const auto& weights = state_weights.second;
        for (size_t i = 0, nroots = weights.size(); i < nroots; ++i) {
            if (weights[i] < 1.0e-15)
                continue;
            E3v += weights[i] * E3v_map[state][i];
            E3c -= weights[i] * E3c_map[state][i];
        }
    }

    // => spin-free 1- and 2-cumulant contributions <=

    // - virtual contraction
    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_va", {"va"});
    temp["ex"] = H2["ewxy"] * L1_["yw"];
    temp["ex"] -= 0.5 * H2["ewyx"] * L1_["yw"];
    E3v -= temp["ex"] * T2["uvez"] * G2["xzuv"];

    temp["eu"] = 0.5 * S2["uvez"] * L1_["zv"];
    E3v -= H2["ewxy"] * temp["eu"] * L2_["xyuw"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp_vaaa", {"vaaa"});
    temp["ewuy"] = H2["ewxy"] * L1_["xu"];
    E3v -= 0.5 * temp["ewuy"] * S2["uvez"] * L2_["yzwv"];

    temp["ewxu"] = H2["ewxy"] * L1_["yu"];
    E3v += 0.5 * temp["ewxu"] * T2["uvez"] * L2_["xzwv"];
    E3v += 0.5 * temp["ewxv"] * T2["uvez"] * L2_["xzuw"];

    temp["ezxy"] = H2["ewxy"] * L1_["zw"];
    E3v += 0.5 * temp["ezxy"] * T2["uvez"] * G2["xyuv"];

    // - core contraction
    temp = ambit::BlockedTensor::build(tensor_type_, "temp_ac", {"ac"});
    temp["um"] = H2["uvmz"] * L1_["zv"];
    temp["um"] -= 0.5 * H2["vumz"] * L1_["zv"];
    E3c += temp["um"] * T2["mwxy"] * L2_["xyuw"];

    temp["xm"] = S2["mwxy"] * L1_["yw"];
    E3c += 0.5 * H2["uvmz"] * temp["xm"] * G2["xzuv"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp_caaa", {"caaa"});
    temp["mzxv"] = H2["uvmz"] * L1_["xu"];
    E3c += 0.5 * temp["mzxv"] * S2["mwxy"] * L2_["yzwv"];

    temp["mzuy"] = H2["uvmz"] * L1_["yv"];
    E3c -= 0.5 * temp["mzuy"] * T2["mwxy"] * L2_["xzuw"];
    E3c -= 0.5 * temp["mzux"] * T2["mwxy"] * L2_["yzwu"];

    temp["mwuv"] = H2["uvmz"] * L1_["zw"];
    E3c -= 0.5 * temp["mwuv"] * T2["mwxy"] * G2["xyuv"];

    return E3c + E3v;
}

double SADSRG::H2_T2_C0_cu3_approx(BlockedTensor& H2, BlockedTensor& T2) {
    // Francesco's 5-index cu3 approx. JCP 159, 114106 (2023)
    timer ftimer("DSRG [H2,T2]0 approx");

    auto na = actv_mos_.size();
    auto na2 = na * na;
    auto na3 = na * na2;
    auto na4 = na * na3;
    auto nv = virt_mos_.size();
    auto nva = nv * na;
    auto nva2 = nv * na2;
    auto nc = core_mos_.size();
    auto nca = nc * na;
    auto nca2 = nc * na2;

    // pair of active indices (off-diagonal)
    std::vector<std::pair<size_t, size_t>> na_pairs;
    na_pairs.reserve(na2 - na);
    for (size_t u = 0; u < na; ++u) {
        for (size_t v = 0; v < na; ++v) {
            if (v != u)
                na_pairs.emplace_back(u, v);
        }
    }
    size_t psize = na2 - na;
    size_t psize2 = psize * psize;

    // temp to be contracted with L3d
    auto temp1 = ambit::Tensor::build(tensor_type_, "temp1d", {na, na, na, na, na});
    auto temp2 = ambit::Tensor::build(tensor_type_, "temp2d", {na, na, na, na, na});
    auto& temp2_data = temp2.data();

    // virtual contraction
    timer timer1("Contraction virtual");
    auto T2v = T2.block("aava").clone();
    auto& T2v_data = T2v.data();
    auto H2v = H2.block("vaaa").clone();
    auto& H2v_data = H2v.data();

    // z = v
    timer timer1t("T2 diagonal");
    auto T2v_1 = ambit::Tensor::build(tensor_type_, "T2 uzez -> uze", {na, na, nv});
#pragma omp parallel for
    for (size_t i = 0; i < na2; ++i) {
        size_t u = i / na, z = i % na;
        double* x_ptr = &(T2v_data[u * nva2 + z * nva + z]);
        double* y_ptr = &(T2v_1.data()[u * nva + z * nv]);
        psi::C_DCOPY(nv, x_ptr, na, y_ptr, 1);
    }
    temp1("xyzuw") += H2v("ewxy") * T2v_1("uze");

    // z != v and z = u
    auto T2v_2 = ambit::Tensor::build(tensor_type_, "T2 zvez -> zve (z != v)", {na, na, nv});
#pragma omp parallel for
    for (size_t i = 0; i < na2 - na; ++i) {
        auto [z, v] = na_pairs[i];
        double* x_ptr = &(T2v_data[z * nva2 + v * nva + z]);
        double* y_ptr = &(T2v_2.data()[z * nva + v * nv]);
        psi::C_DCOPY(nv, x_ptr, na, y_ptr, 1);
    }
    temp2("yxzwv") += H2v("ewxy") * T2v_2("zve");

    // z != (u, v)
#pragma omp parallel for collapse(2)
    for (size_t e = 0; e < nv; ++e) {
        for (size_t i = 0; i < na2; ++i) {
            size_t z = i / na, v = i % na;
            T2v_data[v * nva2 + z * nva + e * na + z] = 0.0;
            T2v_data[z * nva2 + v * nva + e * na + z] = 0.0;
        }
    }
    timer1t.stop();

    // z != (u, v) and w = y
    timer timer1h("H2 diagonal");
    auto H2v_1 = ambit::Tensor::build(tensor_type_, "H2 ewxw -> ewx", {nv, na, na});
#pragma omp parallel for
    for (size_t i = 0; i < nva; ++i) {
        size_t e = i / na, w = i % na;
        double* x_ptr = &(H2v_data[e * na3 + w * na2 + w]);
        double* y_ptr = &(H2v_1.data()[e * na2 + w * na]);
        psi::C_DCOPY(na, x_ptr, na, y_ptr, 1);
    }
    temp1("xzwuv") += H2v_1("ewx") * T2v("uvez");

    // z != (u, v) and w != y and w = x
    auto H2v_2 = ambit::Tensor::build(tensor_type_, "H2 ewwy -> ewy (w != y)", {nv, na, na});
#pragma omp parallel for
    for (size_t i = 0; i < nva; ++i) {
        size_t e = i / na, w = i % na;
        double* x_ptr = &(H2v_data[e * na3 + w * na2 + w * na]);
        double* y_ptr = &(H2v_2.data()[e * na2 + w * na]);
        psi::C_DCOPY(na, x_ptr, 1, y_ptr, 1);
        H2v_2.data()[e * na2 + w * na + w] = 0.0;
    }
    temp2("zywvu") += H2v_2("ewy") * T2v("uvez");

    // w != (x, y)
#pragma omp parallel for collapse(2)
    for (size_t e = 0; e < nv; ++e) {
        for (size_t i = 0; i < na2; ++i) {
            size_t w = i / na, x = i % na;
            H2v_data[e * na3 + w * na2 + x * na + w] = 0.0;
            H2v_data[e * na3 + w * na2 + w * na + x] = 0.0;
        }
    }
    timer1h.stop();

    // w != (x, y) and z != (u, v) and x = u
    timer timer1m("H2T2 mixed diagonal");
    temp1("yzxwv") += H2v("ewxy") * T2v("xvez");

    // w != (x, y) and z != (u, v) and x != u and w = z
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < na2 - na; ++i) {
        auto [x, u] = na_pairs[i];
        for (size_t w = 0; w < na; ++w) {
            for (size_t y = 0; y < na; ++y) {
                for (size_t v = 0; v < na; ++v) {
                    double value = 0.0;
                    for (size_t e = 0; e < nv; ++e) {
                        value += H2v_data[e * na3 + w * na2 + x * na + y] *
                                 T2v_data[u * na2 * nv + v * na * nv + e * na + w];
                    }
                    temp2_data[x * na4 + y * na3 + w * na2 + u * na + v] += value;
                }
            }
        }
    }

    // w != (x, y) and z != (u, v) and x != u and w != z and x = v
#pragma omp parallel for
    for (size_t i = 0; i < psize2; ++i) {
        size_t xu = i / psize, zw = i % psize;
        auto [x, u] = na_pairs[xu];
        auto [z, w] = na_pairs[zw];
        for (size_t y = 0; y < na; ++y) {
            double value = 0.0;
            for (size_t e = 0; e < nv; ++e) {
                value += H2v_data[e * na3 + w * na2 + x * na + y] *
                         T2v_data[u * na2 * nv + x * na * nv + e * na + z];
            }
            temp2_data[y * na4 + z * na3 + x * na2 + w * na + u] += value;
        }
    }

    // w != (x, y) and z != (u, v) and x != u and w != z and x != v and y = v
#pragma omp parallel for
    for (size_t i = 0; i < psize2; ++i) {
        size_t xu = i / psize, zw = i % psize;
        auto [x, u] = na_pairs[xu];
        auto [z, w] = na_pairs[zw];
        for (size_t v = 0; v < na; ++v) {
            double factor = (v == x ? 0.0 : 1.0); // x != v
            double value = 0.0;
            for (size_t e = 0; e < nv; ++e) {
                value += H2v_data[e * na3 + w * na2 + x * na + v] *
                         T2v_data[u * na2 * nv + v * na * nv + e * na + z];
            }
            temp2_data[x * na4 + z * na3 + v * na2 + u * na + w] += value * factor;
        }
    }

    // w != (x, y) and z != (u, v) and x != u and w != z and x != v and y != v and y = u
#pragma omp parallel for
    for (size_t i = 0; i < psize2; ++i) {
        size_t xu = i / psize, zw = i % psize;
        auto [x, u] = na_pairs[xu];
        auto [z, w] = na_pairs[zw];
        for (size_t j = 0; j < psize; ++j) {
            auto [v, y] = na_pairs[j];
            double factor = ((v != x && y == u) ? 1.0 : 0.0);
            double value = 0.0;
            for (size_t e = 0; e < nv; ++e) {
                value += H2v_data[e * na3 + w * na2 + x * na + y] *
                         T2v_data[y * na2 * nv + v * na * nv + e * na + z];
            }
            temp2_data[z * na4 + x * na3 + y * na2 + v * na + w] += value * factor;
        }
    }
    timer1m.stop();
    timer1.stop();

    // core contraction
    timer timer2("Contraction core");
    auto T2c = T2.block("caaa").clone();
    auto& T2c_data = T2c.data();
    auto H2c = H2.block("aaca").clone();
    auto& H2c_data = H2c.data();

    // w = y
    timer timer2t("T2 diagonal");
    auto T2c_1 = ambit::Tensor::build(tensor_type_, "T2 mwxw -> mwx", {nc, na, na});
#pragma omp parallel for
    for (size_t i = 0; i < nca; ++i) {
        size_t m = i / na, w = i % na;
        double* x_ptr = &(T2c_data[m * na3 + w * na2 + w]);
        double* y_ptr = &(T2c_1.data()[m * na2 + w * na]);
        psi::C_DCOPY(na, x_ptr, na, y_ptr, 1);
    }
    temp1("xzwuv") += H2c("uvmz") * T2c_1("mwx");

    // w != y and w = x
    auto T2c_2 = ambit::Tensor::build(tensor_type_, "T2 mwwy -> mwy (w != y)", {nc, na, na});
#pragma omp parallel for
    for (size_t i = 0; i < nca; ++i) {
        size_t m = i / na, w = i % na;
        double* x_ptr = &(T2c_data[m * na3 + w * na2 + w * na]);
        double* y_ptr = &(T2c_2.data()[m * na2 + w * na]);
        psi::C_DCOPY(na, x_ptr, 1, y_ptr, 1);
        T2c_2.data()[m * na2 + w * na + w] = 0.0;
    }
    temp2("zywvu") += H2c("uvmz") * T2c_2("mwy");

    // w != (x, y)
#pragma omp parallel for collapse(2)
    for (size_t m = 0; m < nc; ++m) {
        for (size_t i = 0; i < na2; ++i) {
            size_t w = i / na, x = i % na;
            T2c_data[m * na3 + w * na2 + x * na + w] = 0.0;
            T2c_data[m * na3 + w * na2 + w * na + x] = 0.0;
        }
    }
    timer2t.stop();

    // w != (x, y) and z = v
    timer timer2h("H2 diagonal");
    auto H2c_1 = ambit::Tensor::build(tensor_type_, "H2 uzmz -> uzm", {na, na, nc});
#pragma omp parallel for
    for (size_t i = 0; i < na2; ++i) {
        size_t u = i / na, z = i % na;
        double* x_ptr = &(H2c_data[u * nca2 + z * nca + z]);
        double* y_ptr = &(H2c_1.data()[u * nca + z * nc]);
        psi::C_DCOPY(nc, x_ptr, na, y_ptr, 1);
    }
    temp1("xyzuw") += H2c_1("uzm") * T2c("mwxy");

    // w != (x, y) and z != v and z = u
    auto H2c_2 = ambit::Tensor::build(tensor_type_, "H2 zvmz -> zvm (z != v)", {na, na, nc});
#pragma omp parallel for
    for (size_t i = 0; i < na2 - na; ++i) {
        auto [z, v] = na_pairs[i];
        double* x_ptr = &(H2c_data[z * nca2 + v * nca + z]);
        double* y_ptr = &(H2c_2.data()[z * nca + v * nc]);
        psi::C_DCOPY(nc, x_ptr, na, y_ptr, 1);
    }
    temp2("yxzwv") += H2c_2("zvm") * T2c("mwxy");

    // z != (u, v)
#pragma omp parallel for collapse(2)
    for (size_t m = 0; m < nc; ++m) {
        for (size_t i = 0; i < na2; ++i) {
            size_t z = i / na, v = i % na;
            H2c_data[v * nca2 + z * nca + m * na + z] = 0.0;
            H2c_data[z * nca2 + v * nca + m * na + z] = 0.0;
        }
    }
    timer2h.stop();

    // w != (x, y) and z != (u, v) and x = u
    timer timer2m("H2T2 mixed diagonal");
    temp1("yzuwv") += H2c("uvmz") * T2c("mwuy");

    // w != (x, y) and z != (u, v) and x != u and z = w
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < na2 - na; ++i) {
        auto [x, u] = na_pairs[i];
        for (size_t z = 0; z < na; ++z) {
            for (size_t y = 0; y < na; ++y) {
                for (size_t v = 0; v < na; ++v) {
                    double value = 0.0;
                    for (size_t m = 0; m < nc; ++m) {
                        value += H2c_data[u * nc * na2 + v * nc * na + m * na + z] *
                                 T2c_data[m * na3 + z * na2 + x * na + y];
                    }
                    temp2_data[x * na4 + y * na3 + z * na2 + u * na + v] += value;
                }
            }
        }
    }

    // w != (x, y) and z != (u, v) and x != u and z != w and x = v
#pragma omp parallel for
    for (size_t i = 0; i < psize2; ++i) {
        size_t xu = i / psize, zw = i % psize;
        auto [x, u] = na_pairs[xu];
        auto [z, w] = na_pairs[zw];
        for (size_t y = 0; y < na; ++y) {
            double value = 0.0;
            for (size_t m = 0; m < nc; ++m) {
                value += H2c_data[u * nc * na2 + x * nc * na + m * na + z] *
                         T2c_data[m * na3 + w * na2 + x * na + y];
            }
            temp2_data[y * na4 + z * na3 + x * na2 + w * na + u] += value;
        }
    }

    // w != (x, y) and z != (u, v) and x != u and z != w and x != v and y = v
#pragma omp parallel for
    for (size_t i = 0; i < psize2; ++i) {
        size_t xu = i / psize, zw = i % psize;
        auto [x, u] = na_pairs[xu];
        auto [z, w] = na_pairs[zw];
        for (size_t v = 0; v < na; ++v) {
            double factor = (v == x ? 0.0 : 1.0); // x != v
            double value = 0.0;
            for (size_t m = 0; m < nc; ++m) {
                value += H2c_data[u * nc * na2 + v * nc * na + m * na + z] *
                         T2c_data[m * na3 + w * na2 + x * na + v];
            }
            temp2_data[x * na4 + z * na3 + v * na2 + u * na + w] += value * factor;
        }
    }

    // w != (x, y) and z != (u, v) and x != u and z != w and x != v and y != v and y = u
#pragma omp parallel for
    for (size_t i = 0; i < psize2; ++i) {
        size_t xu = i / psize, zw = i % psize;
        auto [x, u] = na_pairs[xu];
        auto [z, w] = na_pairs[zw];
        for (size_t j = 0; j < psize; ++j) {
            auto [v, y] = na_pairs[j];
            double factor = ((v != x && y == u) ? 1.0 : 0.0);
            double value = 0.0;
            for (size_t m = 0; m < nc; ++m) {
                value += H2c_data[u * nc * na2 + v * nc * na + m * na + z] *
                         T2c_data[m * na3 + w * na2 + x * na + u];
            }
            temp2_data[z * na4 + x * na3 + u * na2 + v * na + w] += value * factor;
        }
    }
    timer2m.stop();
    timer2.stop();

    double E3 = 0.0;
    E3 += temp1("xyzuv") * L3d1_("xyzuv");
    E3 += temp2("xyzuv") * L3d2_("xyzuv");
    return E3;
}
} // namespace forte