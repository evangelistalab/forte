/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "reference.h"

namespace psi {
namespace forte {

Reference::Reference() {}

Reference::~Reference() {}

double Reference::compute_Eref(std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info, double Enuc) {
    // similar to MASTER_DSRG::compute_reference_energy_from_ints (use Fock and cumulants)
    // here I form two density and directly use bare Hamiltonian

    double Efrzc = ints->frozen_core_energy();
    double E = Enuc + Efrzc;

    std::vector<size_t> core_mos = mo_space_info->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> actv_mos = mo_space_info->get_corr_abs_mo("ACTIVE");
    size_t ncore = core_mos.size();
    size_t nactv = actv_mos.size();

    // core 1-body: \sum_{m}^{C} h^{m}_{m}
    for (size_t m : core_mos) {
        E += ints->oei_a(m, m);
        E += ints->oei_b(m, m);
    }

    // active 1-body: \sum_{uv}^{A} h^{u}_{v} * G1^{v}_{u}
    ambit::Tensor Ha = ambit::Tensor::build(ambit::CoreTensor, "Ha", {nactv, nactv});
    ambit::Tensor Hb = ambit::Tensor::build(ambit::CoreTensor, "Hb", {nactv, nactv});
    for (size_t u = 0; u < nactv; ++u) {
        size_t nu = actv_mos[u];
        for (size_t v = 0; v < nactv; ++v) {
            size_t nv = actv_mos[v];
            Ha.data()[u * nactv + v] = ints->oei_a(nu, nv);
            Hb.data()[u * nactv + v] = ints->oei_b(nu, nv);
        }
    }
    E += Ha("uv") * L1a_("vu");
    E += Hb("uv") * L1b_("vu");

    // core-core 2-body: 0.5 * \sum_{mn}^{C} v^{mn}_{mn} in mini-batches
    // 4-index tensor of core-core-core-core could be large (> 600 electrons)

    ambit::Tensor I, Vtemp;
    I = ambit::Tensor::build(ambit::CoreTensor, "I", {1, ncore, 1, ncore});
    I.iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[1] == i[3]) {
            value = 1.0;
        }
    });

    for (size_t m = 0; m < ncore; ++m) {
        size_t nm = core_mos[m];

        Vtemp = ints->aptei_aa_block({nm}, core_mos, {nm}, core_mos);
        E += 0.5 * Vtemp("pqrs") * I("pqrs");

        Vtemp = ints->aptei_ab_block({nm}, core_mos, {nm}, core_mos);
        E += 0.5 * Vtemp("pqrs") * I("pqrs");

        Vtemp = ints->aptei_ab_block(core_mos, {nm}, core_mos, {nm});
        E += 0.5 * Vtemp("pqrs") * I("qpsr");

        Vtemp = ints->aptei_bb_block({nm}, core_mos, {nm}, core_mos);
        E += 0.5 * Vtemp("pqrs") * I("pqrs");
    }

    // core-active 2-body: \sum_{m}^{C} \sum_{uv}^{A} v^{mu}_{mv} * G1^{v}_{u}

    I = ambit::Tensor::build(ambit::CoreTensor, "I", {ncore, ncore});
    I.iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });

    Vtemp = ints->aptei_aa_block(core_mos, actv_mos, core_mos, actv_mos);
    E += Vtemp("munv") * I("mn") * L1a_("vu");

    Vtemp = ints->aptei_ab_block(core_mos, actv_mos, core_mos, actv_mos);
    E += Vtemp("munv") * I("mn") * L1b_("vu");

    Vtemp = ints->aptei_ab_block(actv_mos, core_mos, actv_mos, core_mos);
    E += Vtemp("umvn") * I("mn") * L1a_("vu");

    Vtemp = ints->aptei_bb_block(core_mos, actv_mos, core_mos, actv_mos);
    E += Vtemp("munv") * I("mn") * L1b_("vu");

    // active-active 2-body: 0.25 * \sum_{uvxy}^{A} * v^{uv}_{xy} * G2^{xy}_{uv}
    ambit::Tensor G2 = L2aa_.clone();
    G2("pqrs") += L1a_("pr") * L1a_("qs");
    G2("pqrs") -= L1a_("ps") * L1a_("qr");
    Vtemp = ints->aptei_aa_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += 0.25 * Vtemp("uvxy") * G2("uvxy");

    G2 = L2ab_.clone();
    G2("pqrs") += L1a_("pr") * L1b_("qs");
    Vtemp = ints->aptei_ab_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += Vtemp("uVxY") * G2("uVxY");

    G2 = L2bb_.clone();
    G2("pqrs") += L1b_("pr") * L1b_("qs");
    G2("pqrs") -= L1b_("ps") * L1b_("qr");
    Vtemp = ints->aptei_bb_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += 0.25 * Vtemp("UVXY") * G2("UVXY");

    return E;
}

//void Reference::set_G1(std::vector<double>& a, std::vector<double>& b, size_t nactv, bool move) {
//    if (a.size() != nactv * nactv || b.size() != nactv * nactv) {
//        throw PSIEXCEPTION("Inconsistent/unexpected vector size.");
//    }

//    L1a_ = ambit::Tensor::build(ambit::CoreTensor, "L1a", {nactv, nactv});
//    L1b_ = ambit::Tensor::build(ambit::CoreTensor, "L1b", {nactv, nactv});

//    if (move) {
//        L1a_.data() = std::move(a);
//        L1b_.data() = std::move(b);
//    } else {
//        L1a_.data() = a;
//        L1b_.data() = b;
//    }
//}

//void Reference::set_G2(std::vector<double>& aa, std::vector<double>& ab, std::vector<double>& bb,
//                       size_t nactv, bool cumulant, bool move) {
//    size_t na4 = nactv * nactv * nactv * nactv;
//    if (aa.size() != na4 || ab.size() != na4 || bb.size() != na4) {
//        throw PSIEXCEPTION("Inconsistent/unexpected vector size.");
//    }
//    if (cumulant) {
//        if (L1a_.numel() != nactv * nactv || L1b_.numel() != nactv * nactv) {
//            throw PSIEXCEPTION("Cannot compute 2-cumulants. Incorrect L1a/L1b size.");
//        }
//    }

//    g2aa_ = ambit::Tensor::build(ambit::CoreTensor, "G2aa", {nactv, nactv, nactv, nactv});
//    g2ab_ = ambit::Tensor::build(ambit::CoreTensor, "G2ab", {nactv, nactv, nactv, nactv});
//    g2bb_ = ambit::Tensor::build(ambit::CoreTensor, "G2bb", {nactv, nactv, nactv, nactv});

//    if (move) {
//        g2aa_.data() = std::move(aa);
//        g2ab_.data() = std::move(ab);
//        g2bb_.data() = std::move(bb);
//    } else {
//        g2aa_.data() = aa;
//        g2ab_.data() = ab;
//        g2bb_.data() = bb;
//    }

//    if (cumulant) {
//        L2aa_ = g2aa_.clone();
//        L2ab_ = g2ab_.clone();
//        L2bb_ = g2bb_.clone();

//        L2aa_("pqrs") -= L1a_("pr") * L1a_("qs");
//        L2aa_("pqrs") += L1a_("ps") * L1a_("qr");

//        L2bb_("pqrs") -= L1b_("pr") * L1b_("qs");
//        L2bb_("pqrs") += L1b_("ps") * L1b_("qr");

//        L2ab_("pqrs") -= L1a_("pr") * L1b_("qs");
//    }
//}

//void Reference::set_G3(std::vector<double>& aaa, std::vector<double>& aab, std::vector<double>& abb,
//                       std::vector<double>& bbb, size_t nactv, bool cumulant) {
//    size_t na2 = nactv * nactv;
//    size_t na4 = na2 * na2;
//    size_t na6 = na2 * na4;
//    if (aaa.size() != na6 || aab.size() != na6 || abb.size() != na6 || bbb.size() != na6) {
//        throw PSIEXCEPTION("Inconsistent/unexpected vector size.");
//    }
//    if (cumulant) {
//        if (L1a_.numel() != na2 || L1b_.numel() != na2) {
//            throw PSIEXCEPTION("Cannot compute 3-cumulants. Incorrect L1a/L1b size.");
//        }
//        if (L2aa_.numel() != na4 || L2ab_.numel() != na4 || L2bb_.numel() != na4) {
//            throw PSIEXCEPTION("Cannot compute 3-cumulants. Incorrect L2aa/L2ab/L2bb size.");
//        }
//    }

//    if (cumulant) {
//        // aaa
//        L3aaa_ = ambit::Tensor::build(ambit::CoreTensor, "L3aaa",
//                                      {nactv, nactv, nactv, nactv, nactv, nactv});
//        L3aaa_.data() = std::move(aaa);

//        L3aaa_("pqrstu") -= L1a_("ps") * L2aa_("qrtu");
//        L3aaa_("pqrstu") += L1a_("pt") * L2aa_("qrsu");
//        L3aaa_("pqrstu") += L1a_("pu") * L2aa_("qrts");

//        L3aaa_("pqrstu") -= L1a_("qt") * L2aa_("prsu");
//        L3aaa_("pqrstu") += L1a_("qs") * L2aa_("prtu");
//        L3aaa_("pqrstu") += L1a_("qu") * L2aa_("prst");

//        L3aaa_("pqrstu") -= L1a_("ru") * L2aa_("pqst");
//        L3aaa_("pqrstu") += L1a_("rs") * L2aa_("pqut");
//        L3aaa_("pqrstu") += L1a_("rt") * L2aa_("pqsu");

//        L3aaa_("pqrstu") -= L1a_("ps") * L1a_("qt") * L1a_("ru");
//        L3aaa_("pqrstu") -= L1a_("pt") * L1a_("qu") * L1a_("rs");
//        L3aaa_("pqrstu") -= L1a_("pu") * L1a_("qs") * L1a_("rt");

//        L3aaa_("pqrstu") += L1a_("ps") * L1a_("qu") * L1a_("rt");
//        L3aaa_("pqrstu") += L1a_("pu") * L1a_("qt") * L1a_("rs");
//        L3aaa_("pqrstu") += L1a_("pt") * L1a_("qs") * L1a_("ru");

//        // aab
//        L3aab_ = ambit::Tensor::build(ambit::CoreTensor, "L3aab",
//                                      {nactv, nactv, nactv, nactv, nactv, nactv});
//        L3aab_.data() = std::move(aab);

//        L3aab_("pqRstU") -= L1a_("ps") * L2ab_("qRtU");
//        L3aab_("pqRstU") += L1a_("pt") * L2ab_("qRsU");

//        L3aab_("pqRstU") -= L1a_("qt") * L2ab_("pRsU");
//        L3aab_("pqRstU") += L1a_("qs") * L2ab_("pRtU");

//        L3aab_("pqRstU") -= L1b_("RU") * L2aa_("pqst");

//        L3aab_("pqRstU") -= L1a_("ps") * L1a_("qt") * L1b_("RU");
//        L3aab_("pqRstU") += L1a_("pt") * L1a_("qs") * L1b_("RU");

//        // abb
//        L3abb_ = ambit::Tensor::build(ambit::CoreTensor, "L3abb",
//                                      {nactv, nactv, nactv, nactv, nactv, nactv});
//        L3abb_.data() = std::move(abb);

//        L3abb_("pQRsTU") -= L1a_("ps") * L2bb_("QRTU");

//        L3abb_("pQRsTU") -= L1b_("QT") * L2ab_("pRsU");
//        L3abb_("pQRsTU") += L1b_("QU") * L2ab_("pRsT");

//        L3abb_("pQRsTU") -= L1b_("RU") * L2ab_("pQsT");
//        L3abb_("pQRsTU") += L1b_("RT") * L2ab_("pQsU");

//        L3abb_("pQRsTU") -= L1a_("ps") * L1b_("QT") * L1b_("RU");
//        L3abb_("pQRsTU") += L1a_("ps") * L1b_("QU") * L1b_("RT");

//        // bbb
//        L3bbb_ = ambit::Tensor::build(ambit::CoreTensor, "L3bbb",
//                                      {nactv, nactv, nactv, nactv, nactv, nactv});
//        L3bbb_.data() = std::move(bbb);

//        L3bbb_("pqrstu") -= L1b_("ps") * L2bb_("qrtu");
//        L3bbb_("pqrstu") += L1b_("pt") * L2bb_("qrsu");
//        L3bbb_("pqrstu") += L1b_("pu") * L2bb_("qrts");

//        L3bbb_("pqrstu") -= L1b_("qt") * L2bb_("prsu");
//        L3bbb_("pqrstu") += L1b_("qs") * L2bb_("prtu");
//        L3bbb_("pqrstu") += L1b_("qu") * L2bb_("prst");

//        L3bbb_("pqrstu") -= L1b_("ru") * L2bb_("pqst");
//        L3bbb_("pqrstu") += L1b_("rs") * L2bb_("pqut");
//        L3bbb_("pqrstu") += L1b_("rt") * L2bb_("pqsu");

//        L3bbb_("pqrstu") -= L1b_("ps") * L1b_("qt") * L1b_("ru");
//        L3bbb_("pqrstu") -= L1b_("pt") * L1b_("qu") * L1b_("rs");
//        L3bbb_("pqrstu") -= L1b_("pu") * L1b_("qs") * L1b_("rt");

//        L3bbb_("pqrstu") += L1b_("ps") * L1b_("qu") * L1b_("rt");
//        L3bbb_("pqrstu") += L1b_("pu") * L1b_("qt") * L1b_("rs");
//        L3bbb_("pqrstu") += L1b_("pt") * L1b_("qs") * L1b_("ru");
//    } else {
//        g3aaa_ = ambit::Tensor::build(ambit::CoreTensor, "G3aaa",
//                                      {nactv, nactv, nactv, nactv, nactv, nactv});
//        g3aaa_.data() = std::move(aaa);

//        g3aab_ = ambit::Tensor::build(ambit::CoreTensor, "G3aab",
//                                      {nactv, nactv, nactv, nactv, nactv, nactv});
//        g3aab_.data() = std::move(aab);

//        g3abb_ = ambit::Tensor::build(ambit::CoreTensor, "G3abb",
//                                      {nactv, nactv, nactv, nactv, nactv, nactv});
//        g3abb_.data() = std::move(abb);

//        g3bbb_ = ambit::Tensor::build(ambit::CoreTensor, "G3bbb",
//                                      {nactv, nactv, nactv, nactv, nactv, nactv});
//        g3bbb_.data() = std::move(bbb);
//    }
//}
}
}
