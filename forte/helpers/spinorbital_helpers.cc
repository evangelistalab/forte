/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "helpers/helpers.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"
#include "ambit/tensor.h"

#include "spinorbital_helpers.h"

namespace forte {

ambit::Tensor spinorbital_oei(const std::shared_ptr<ForteIntegrals> ints,
                              const std::vector<size_t>& p, const std::vector<size_t>& q) {
    ambit::Tensor t = ambit::Tensor::build(ambit::CoreTensor, "oei", {2 * p.size(), 2 * q.size()});
    t.iterate([&](const std::vector<size_t>& i, double& value) {
        if ((i[0] % 2 == 0) and (i[1] % 2 == 0)) {
            value = ints->oei_a(p[i[0] / 2], q[i[1] / 2]);
        }
        if ((i[0] % 2 == 1) and (i[1] % 2 == 1)) {
            value = ints->oei_b(p[i[0] / 2], q[i[1] / 2]);
        }
    });
    return t;
}

ambit::Tensor spinorbital_tei(const std::shared_ptr<ForteIntegrals> ints,
                              const std::vector<size_t>& p, const std::vector<size_t>& q,
                              const std::vector<size_t>& r, const std::vector<size_t>& s) {
    ambit::Tensor t = ambit::Tensor::build(
        ambit::CoreTensor, "tei", {2 * p.size(), 2 * q.size(), 2 * r.size(), 2 * s.size()});

    auto tei_aa = ints->aptei_aa_block(p, q, r, s);
    auto tei_ab = ints->aptei_ab_block(p, q, r, s);
    auto tei_ab2 = ints->aptei_ab_block(p, q, s, r);
    auto tei_ba = ints->aptei_ab_block(q, p, s, r);
    auto tei_ba2 = ints->aptei_ab_block(q, p, r, s);
    auto tei_bb = ints->aptei_bb_block(p, q, r, s);

    // fill in the tensor
    t.iterate([&](const std::vector<size_t>& i, double& value) {
        const size_t a = i[0] / 2;
        const size_t b = i[1] / 2;
        const size_t c = i[2] / 2;
        const size_t d = i[3] / 2;
        const bool sa = i[0] % 2;
        const bool sb = i[1] % 2;
        const bool sc = i[2] % 2;
        const bool sd = i[3] % 2;
        value = 0.0;
        if ((sa == sb) and (sb == sc) and (sc == sd)) {
            if (!sa)
                value = tei_aa.at({a, b, c, d});
            else
                value = tei_bb.at({a, b, c, d});
        }
        if ((sa == sc) and (sb == sd) and (sa != sb)) {
            if (!sa)
                value = tei_ab.at({a, b, c, d});
            else
                value = tei_ba.at({b, a, d, c});
        }
        if ((sa == sd) and (sb == sc) and (sa != sb)) {
            if (!sa)
                value = -tei_ab2.at({a, b, d, c});
            else
                value = -tei_ba2.at({b, a, c, d});
        }
    });
    return t;
}

ambit::Tensor spinorbital_fock(const std::shared_ptr<ForteIntegrals> ints,
                               const std::vector<size_t>& p, const std::vector<size_t>& q,
                               const std::vector<size_t>& docc) {
    ambit::Tensor t = ambit::Tensor::build(ambit::CoreTensor, "tei", {2 * p.size(), 2 * q.size()});

    auto h = spinorbital_oei(ints, p, q);
    auto v = spinorbital_tei(ints, p, docc, q, docc);
    for (size_t i = 0, maxi = 2 * p.size(); i < maxi; i++) {
        for (size_t j = 0, maxj = 2 * q.size(); j < maxj; j++) {
            double f = 0.0;
            for (auto k : docc) {
                f += v.at({i, 2 * k, j, 2 * k});
                f += v.at({i, 2 * k + 1, j, 2 * k + 1});
            }
            h.at({i, j}) += f;
        }
    }
    return h;
}

std::vector<ambit::Tensor> spinorbital_rdms(std::shared_ptr<RDMs> rdms) {
    auto max_rdm_level = rdms->max_rdm_level();

    std::vector<ambit::Tensor> sordms;
    if (max_rdm_level < 1)
        return sordms;

    ambit::Tensor g1a = rdms->g1a();
    ambit::Tensor g1b = rdms->g1b();
    size_t nso_actv = 2 * g1a.dim(0);

    auto g1 = ambit::Tensor::build(ambit::CoreTensor, "g1", {nso_actv, nso_actv});

    g1a.iterate([&](const std::vector<size_t>& i, double& value) {
        g1.at({2 * i[0], 2 * i[1]}) = value;
    });
    g1b.iterate([&](const std::vector<size_t>& i, double& value) {
        g1.at({2 * i[0] + 1, 2 * i[1] + 1}) = value;
    });
    sordms.push_back(g1);

    if (max_rdm_level < 2)
        return sordms;

    ambit::Tensor g2aa = rdms->g2aa();
    ambit::Tensor g2ab = rdms->g2ab();
    ambit::Tensor g2bb = rdms->g2bb();

    std::vector<size_t> dim4(4, nso_actv);
    auto g2 = ambit::Tensor::build(ambit::CoreTensor, "g2", dim4);

    g2aa.iterate([&](const std::vector<size_t>& i, double& value) {
        g2.at({2 * i[0], 2 * i[1], 2 * i[2], 2 * i[3]}) = value;
    });
    g2bb.iterate([&](const std::vector<size_t>& i, double& value) {
        g2.at({2 * i[0] + 1, 2 * i[1] + 1, 2 * i[2] + 1, 2 * i[3] + 1}) = value;
    });
    g2ab.iterate([&](const std::vector<size_t>& i, double& value) {
        const auto a = 2 * i[0];
        const auto B = 2 * i[1] + 1;
        const auto r = 2 * i[2];
        const auto S = 2 * i[3] + 1;
        g2.at({a, B, r, S}) = value;
        g2.at({a, B, S, r}) = -value;
        g2.at({B, a, r, S}) = -value;
        g2.at({B, a, S, r}) = value;
    });
    sordms.push_back(g2);

    if (max_rdm_level < 3)
        return sordms;

    ambit::Tensor g3aaa = rdms->g3aaa();
    ambit::Tensor g3aab = rdms->g3aab();
    ambit::Tensor g3abb = rdms->g3abb();
    ambit::Tensor g3bbb = rdms->g3bbb();

    std::vector<size_t> dim6(6, nso_actv);
    auto g3 = ambit::Tensor::build(ambit::CoreTensor, "g3", dim6);

    g3aaa.iterate([&](const std::vector<size_t>& i, double& value) {
        g3.at({2 * i[0], 2 * i[1], 2 * i[2], 2 * i[3], 2 * i[4], 2 * i[5]}) = value;
    });
    g3bbb.iterate([&](const std::vector<size_t>& i, double& value) {
        g3.at({2 * i[0] + 1, 2 * i[1] + 1, 2 * i[2] + 1, 2 * i[3] + 1, 2 * i[4] + 1,
               2 * i[5] + 1}) = value;
    });
    g3aab.iterate([&](const std::vector<size_t>& i, double& value) {
        const auto a = 2 * i[0];
        const auto b = 2 * i[1];
        const auto C = 2 * i[2] + 1;
        const auto r = 2 * i[3];
        const auto s = 2 * i[4];
        const auto T = 2 * i[5] + 1;
        g3.at({a, b, C, r, s, T}) = +value;
        g3.at({a, C, b, r, s, T}) = -value;
        g3.at({C, b, a, r, s, T}) = -value;
        g3.at({a, b, C, r, T, s}) = -value;
        g3.at({a, C, b, r, T, s}) = +value;
        g3.at({C, b, a, r, T, s}) = +value;
        g3.at({a, b, C, T, s, r}) = -value;
        g3.at({a, C, b, T, s, r}) = +value;
        g3.at({C, b, a, T, s, r}) = +value;
    });
    g3abb.iterate([&](const std::vector<size_t>& i, double& value) {
        const auto a = 2 * i[0];
        const auto B = 2 * i[1] + 1;
        const auto C = 2 * i[2] + 1;
        const auto r = 2 * i[3];
        const auto S = 2 * i[4] + 1;
        const auto T = 2 * i[5] + 1;
        g3.at({a, B, C, r, S, T}) = +value;
        g3.at({B, a, C, r, S, T}) = -value;
        g3.at({C, B, a, r, S, T}) = -value;
        g3.at({a, B, C, S, r, T}) = -value;
        g3.at({B, a, C, S, r, T}) = +value;
        g3.at({C, B, a, S, r, T}) = +value;
        g3.at({a, B, C, T, S, r}) = -value;
        g3.at({B, a, C, T, S, r}) = +value;
        g3.at({C, B, a, T, S, r}) = +value;
    });
    sordms.push_back(g3);

    return sordms;
}

std::vector<ambit::Tensor> spinorbital_cumulants(std::shared_ptr<RDMs> rdms) {
    auto max_rdm_level = rdms->max_rdm_level();

    std::vector<ambit::Tensor> sordms;
    if (max_rdm_level < 1)
        return sordms;

    ambit::Tensor l1a = rdms->g1a();
    ambit::Tensor l1b = rdms->g1b();
    size_t nso_actv = 2 * l1a.dim(0);

    auto l1 = ambit::Tensor::build(ambit::CoreTensor, "l1", {nso_actv, nso_actv});

    l1a.iterate([&](const std::vector<size_t>& i, double& value) {
        l1.at({2 * i[0], 2 * i[1]}) = value;
    });
    l1b.iterate([&](const std::vector<size_t>& i, double& value) {
        l1.at({2 * i[0] + 1, 2 * i[1] + 1}) = value;
    });
    sordms.push_back(l1);

    if (max_rdm_level < 2)
        return sordms;

    ambit::Tensor l2aa = rdms->L2aa();
    ambit::Tensor l2ab = rdms->L2ab();
    ambit::Tensor l2bb = rdms->L2bb();

    std::vector<size_t> dim4(4, nso_actv);
    auto l2 = ambit::Tensor::build(ambit::CoreTensor, "l2", dim4);

    l2aa.iterate([&](const std::vector<size_t>& i, double& value) {
        l2.at({2 * i[0], 2 * i[1], 2 * i[2], 2 * i[3]}) = value;
    });
    l2bb.iterate([&](const std::vector<size_t>& i, double& value) {
        l2.at({2 * i[0] + 1, 2 * i[1] + 1, 2 * i[2] + 1, 2 * i[3] + 1}) = value;
    });
    l2ab.iterate([&](const std::vector<size_t>& i, double& value) {
        const auto a = 2 * i[0];
        const auto B = 2 * i[1] + 1;
        const auto r = 2 * i[2];
        const auto S = 2 * i[3] + 1;
        l2.at({a, B, r, S}) = value;
        l2.at({a, B, S, r}) = -value;
        l2.at({B, a, r, S}) = -value;
        l2.at({B, a, S, r}) = value;
    });
    sordms.push_back(l2);

    if (max_rdm_level < 3)
        return sordms;

    ambit::Tensor l3aaa = rdms->L3aaa();
    ambit::Tensor l3aab = rdms->L3aab();
    ambit::Tensor l3abb = rdms->L3abb();
    ambit::Tensor l3bbb = rdms->L3bbb();

    std::vector<size_t> dim6(6, nso_actv);
    auto l3 = ambit::Tensor::build(ambit::CoreTensor, "l3", dim6);

    l3aaa.iterate([&](const std::vector<size_t>& i, double& value) {
        l3.at({2 * i[0], 2 * i[1], 2 * i[2], 2 * i[3], 2 * i[4], 2 * i[5]}) = value;
    });
    l3bbb.iterate([&](const std::vector<size_t>& i, double& value) {
        l3.at({2 * i[0] + 1, 2 * i[1] + 1, 2 * i[2] + 1, 2 * i[3] + 1, 2 * i[4] + 1,
               2 * i[5] + 1}) = value;
    });
    l3aab.iterate([&](const std::vector<size_t>& i, double& value) {
        const auto a = 2 * i[0];
        const auto b = 2 * i[1];
        const auto C = 2 * i[2] + 1;
        const auto r = 2 * i[3];
        const auto s = 2 * i[4];
        const auto T = 2 * i[5] + 1;
        l3.at({a, b, C, r, s, T}) = +value;
        l3.at({a, C, b, r, s, T}) = -value;
        l3.at({C, b, a, r, s, T}) = -value;
        l3.at({a, b, C, r, T, s}) = -value;
        l3.at({a, C, b, r, T, s}) = +value;
        l3.at({C, b, a, r, T, s}) = +value;
        l3.at({a, b, C, T, s, r}) = -value;
        l3.at({a, C, b, T, s, r}) = +value;
        l3.at({C, b, a, T, s, r}) = +value;
    });
    l3abb.iterate([&](const std::vector<size_t>& i, double& value) {
        const auto a = 2 * i[0];
        const auto B = 2 * i[1] + 1;
        const auto C = 2 * i[2] + 1;
        const auto r = 2 * i[3];
        const auto S = 2 * i[4] + 1;
        const auto T = 2 * i[5] + 1;
        l3.at({a, B, C, r, S, T}) = +value;
        l3.at({B, a, C, r, S, T}) = -value;
        l3.at({C, B, a, r, S, T}) = -value;
        l3.at({a, B, C, S, r, T}) = -value;
        l3.at({B, a, C, S, r, T}) = +value;
        l3.at({C, B, a, S, r, T}) = +value;
        l3.at({a, B, C, T, S, r}) = -value;
        l3.at({B, a, C, T, S, r}) = +value;
        l3.at({C, B, a, T, S, r}) = +value;
    });
    sordms.push_back(l3);

    return sordms;
}

} // namespace forte
