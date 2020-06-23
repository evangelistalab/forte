/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "helpers/timer.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "base_classes/mo_space_info.h"

namespace forte {

RDMs::RDMs() : max_rdm_(0) {}

RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b)
    : max_rdm_(1), have_g1b_(true), g1a_(g1a), g1b_(g1b) {}

RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
           ambit::Tensor g2bb)
    : max_rdm_(2), have_g1b_(true), have_g2aa_(true), have_g2bb_(true), g1a_(g1a), g1b_(g1b),
      g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb) {}

RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
           ambit::Tensor g2bb, ambit::Tensor g3aaa, ambit::Tensor g3aab, ambit::Tensor g3abb,
           ambit::Tensor g3bbb)
    : max_rdm_(3), have_g1b_(true), have_g2aa_(true), have_g2bb_(true), have_g3aaa_(true),
      have_g3abb_(true), have_g3bbb_(true), g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab),
      g2bb_(g2bb), g3aaa_(g3aaa), g3aab_(g3aab), g3abb_(g3abb), g3bbb_(g3bbb) {}

RDMs::RDMs(bool ms_avg, ambit::Tensor g1a) : ms_avg_(ms_avg), max_rdm_(1), g1a_(g1a) {}

RDMs::RDMs(bool ms_avg, ambit::Tensor g1a, ambit::Tensor g2ab)
    : ms_avg_(ms_avg), max_rdm_(2), g1a_(g1a), g2ab_(g2ab) {}

RDMs::RDMs(bool ms_avg, ambit::Tensor g1a, ambit::Tensor g2ab, ambit::Tensor g3aab)
    : ms_avg_(ms_avg), max_rdm_(3), g1a_(g1a), g2ab_(g2ab), g3aab_(g3aab) {}

ambit::Tensor RDMs::g1b() {
    if (ms_avg_ and (not have_g1b_)) {
        g1b_ = g1a_.clone();
        have_g1b_ = true;
    }
    return g1b_;
}

ambit::Tensor RDMs::g2aa() {
    if (ms_avg_ and (not have_g2aa_)) {
        g2aa_ = make_g2_high_spin_case(g2ab_);
        have_g2aa_ = true;
    }
    return g2aa_;
}

ambit::Tensor RDMs::g2bb() {
    if (ms_avg_ and (not have_g2bb_)) {
        g2bb_ = make_g2_high_spin_case(g2ab_);
        have_g2bb_ = true;
    }
    return g2bb_;
}

ambit::Tensor RDMs::g3aaa() {
    if (ms_avg_ and (not have_g3aaa_)) {
        g3aaa_ = make_g3_high_spin_case(g3aab_);
        have_g3aaa_ = true;
    }
    return g3aaa_;
}

ambit::Tensor RDMs::g3abb() {
    if (ms_avg_ and (not have_g3abb_)) {
        g3abb_ = g3aab_.clone();
        g3abb_("rpqust") = g3aab_("pqrstu");
        have_g3abb_ = true;
    }
    return g3abb_;
}

ambit::Tensor RDMs::g3bbb() {
    if (ms_avg_ and (not have_g3bbb_)) {
        g3bbb_ = make_g3_high_spin_case(g3aab_);
        have_g3bbb_ = true;
    }
    return g3bbb_;
}

ambit::Tensor RDMs::L2aa() {
    if (not have_L2aa_) {
        L2aa_ = g2aa().clone();
        make_cumulant_L2aa_in_place(g1a_, L2aa_);
        have_L2aa_ = true;
    }
    return L2aa_;
}

ambit::Tensor RDMs::L2ab() {
    if (not have_L2ab_) {
        L2ab_ = g2ab_.clone();
        make_cumulant_L2ab_in_place(g1a_, g1b(), L2ab_);
        have_L2ab_ = true;
    }
    return L2ab_;
}

ambit::Tensor RDMs::L2bb() {
    if (not have_L2bb_) {
        L2bb_ = g2bb().clone();
        make_cumulant_L2bb_in_place(g1b(), L2bb_);
        have_L2bb_ = true;
    }
    return L2bb_;
}

ambit::Tensor RDMs::L3aaa() {
    if (not have_L3aaa_) {
        L3aaa_ = g3aaa().clone();
        make_cumulant_L3aaa_in_place(g1a_, L2aa(), L3aaa_);
        have_L3aaa_ = true;
    }
    return L3aaa_;
}

ambit::Tensor RDMs::L3aab() {
    if (not have_L3aab_) {
        L3aab_ = g3aab_.clone();
        make_cumulant_L3aab_in_place(g1a_, g1b(), L2aa(), L2ab_, L3aab_);
        have_L3aab_ = true;
    }
    return L3aab_;
}

ambit::Tensor RDMs::L3abb() {
    if (not have_L3abb_) {
        L3abb_ = g3abb().clone();
        make_cumulant_L3abb_in_place(g1a_, g1b(), L2ab_, L2bb(), L3abb_);
        have_L3abb_ = true;
    }
    return L3abb_;
}

ambit::Tensor RDMs::L3bbb() {
    if (not have_L3bbb_) {
        L3bbb_ = g3bbb().clone();
        make_cumulant_L3bbb_in_place(g1b(), L2bb(), L3bbb_);
        have_L3bbb_ = true;
    }
    return L3bbb_;
}

ambit::Tensor RDMs::SFg2() {
    SF_g2_ = g2ab_.clone();
    if (ms_avg_) {
        SF_g2_.scale(4.0);
        SF_g2_("pqrs") -= 2.0 * g2ab_("pqsr");
    } else {
        SF_g2_("pqrs") += g2ab_("qpsr");
        SF_g2_("pqrs") += g2aa()("pqrs") + g2bb()("pqrs");
    }
    return SF_g2_;
}

ambit::Tensor RDMs::SF_L1() {
    if (not ms_avg_) {
        throw std::runtime_error("Must turn on spin averaging.");
    }
    if (not have_SF_L1_) {
        SF_L1_ = g1a_.clone();
        SF_L1_.scale(2.0);
    }
    return SF_L1_;
}

ambit::Tensor RDMs::SF_L2() {
    if (not ms_avg_) {
        throw std::runtime_error("Must turn on spin averaging.");
    }
    if (not have_SF_L2_) {
        SF_L2_ = L2ab().clone();
        SF_L2_.scale(4.0);
        SF_L2_("pqrs") -= 2.0 * L2ab()("pqsr");
    }
    return SF_L2_;
}

ambit::Tensor RDMs::SF_L3() {
    if (not ms_avg_) {
        throw std::runtime_error("Must turn on spin averaging.");
    }
    if (not have_SF_L3_) {
        SF_L3_ = make_g3_high_spin_case(L3aab());
        SF_L3_("pqrstu") += L3aab()("pqrstu");
        SF_L3_("pqrstu") += L3aab()("prqsut");
        SF_L3_("pqrstu") += L3aab()("qrptus");
        SF_L3_.scale(2.0);
    }
    return SF_L3_;
}

ambit::Tensor make_g2_high_spin_case(const ambit::Tensor& g2ab) {
    auto g2hs = g2ab.clone();
    g2hs("pqrs") -= g2ab("pqsr");
    return g2hs;
}

ambit::Tensor make_g3_high_spin_case(const ambit::Tensor& g3aab) {
    auto g3hs = g3aab.clone();
    g3hs("pqrstu") -= g3aab("pqrsut");
    g3hs("pqrstu") += g3aab("pqrtus");
    return g3hs;
}

void make_cumulant_L2aa_in_place(const ambit::Tensor& g1a, ambit::Tensor& L2aa) {
    timer t("make_cumulant_L2aa_in_place");

    L2aa("pqrs") -= g1a("pr") * g1a("qs");
    L2aa("pqrs") += g1a("ps") * g1a("qr");
}

void make_cumulant_L2ab_in_place(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                 ambit::Tensor& L2ab) {
    timer t("make_cumulant_L2ab_in_place");

    L2ab("pqrs") -= g1a("pr") * g1b("qs");
}

void make_cumulant_L2bb_in_place(const ambit::Tensor& g1b, ambit::Tensor& L2bb) {
    timer t("make_cumulant_L2bb_in_place");

    L2bb("pqrs") -= g1b("pr") * g1b("qs");
    L2bb("pqrs") += g1b("ps") * g1b("qr");
}

void make_cumulant_L3aaa_in_place(const ambit::Tensor& g1a, const ambit::Tensor& L2aa,
                                  ambit::Tensor& L3aaa) {
    timer t("make_cumulant_L3aaa_in_place");

    L3aaa("pqrstu") -= g1a("ps") * L2aa("qrtu");
    L3aaa("pqrstu") += g1a("pt") * L2aa("qrsu");
    L3aaa("pqrstu") += g1a("pu") * L2aa("qrts");

    L3aaa("pqrstu") -= g1a("qt") * L2aa("prsu");
    L3aaa("pqrstu") += g1a("qs") * L2aa("prtu");
    L3aaa("pqrstu") += g1a("qu") * L2aa("prst");

    L3aaa("pqrstu") -= g1a("ru") * L2aa("pqst");
    L3aaa("pqrstu") += g1a("rs") * L2aa("pqut");
    L3aaa("pqrstu") += g1a("rt") * L2aa("pqsu");

    L3aaa("pqrstu") -= g1a("ps") * g1a("qt") * g1a("ru");
    L3aaa("pqrstu") -= g1a("pt") * g1a("qu") * g1a("rs");
    L3aaa("pqrstu") -= g1a("pu") * g1a("qs") * g1a("rt");

    L3aaa("pqrstu") += g1a("ps") * g1a("qu") * g1a("rt");
    L3aaa("pqrstu") += g1a("pu") * g1a("qt") * g1a("rs");
    L3aaa("pqrstu") += g1a("pt") * g1a("qs") * g1a("ru");
}

void make_cumulant_L3aab_in_place(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                  const ambit::Tensor& L2aa, const ambit::Tensor& L2ab,
                                  ambit::Tensor& L3aab) {
    timer t("make_cumulant_L3aab_in_place");

    L3aab("pqRstU") -= g1a("ps") * L2ab("qRtU");
    L3aab("pqRstU") += g1a("pt") * L2ab("qRsU");

    L3aab("pqRstU") -= g1a("qt") * L2ab("pRsU");
    L3aab("pqRstU") += g1a("qs") * L2ab("pRtU");

    L3aab("pqRstU") -= g1b("RU") * L2aa("pqst");

    L3aab("pqRstU") -= g1a("ps") * g1a("qt") * g1b("RU");
    L3aab("pqRstU") += g1a("pt") * g1a("qs") * g1b("RU");
}

void make_cumulant_L3abb_in_place(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                  const ambit::Tensor& L2ab, const ambit::Tensor& L2bb,
                                  ambit::Tensor& L3abb) {
    timer t("make_cumulant_L3abb_in_place");

    L3abb("pQRsTU") -= g1a("ps") * L2bb("QRTU");

    L3abb("pQRsTU") -= g1b("QT") * L2ab("pRsU");
    L3abb("pQRsTU") += g1b("QU") * L2ab("pRsT");

    L3abb("pQRsTU") -= g1b("RU") * L2ab("pQsT");
    L3abb("pQRsTU") += g1b("RT") * L2ab("pQsU");

    L3abb("pQRsTU") -= g1a("ps") * g1b("QT") * g1b("RU");
    L3abb("pQRsTU") += g1a("ps") * g1b("QU") * g1b("RT");
}

void make_cumulant_L3bbb_in_place(const ambit::Tensor& g1b, const ambit::Tensor& L2bb,
                                  ambit::Tensor& L3bbb) {
    timer t("make_cumulant_L3bbb_in_place");

    L3bbb("pqrstu") -= g1b("ps") * L2bb("qrtu");
    L3bbb("pqrstu") += g1b("pt") * L2bb("qrsu");
    L3bbb("pqrstu") += g1b("pu") * L2bb("qrts");

    L3bbb("pqrstu") -= g1b("qt") * L2bb("prsu");
    L3bbb("pqrstu") += g1b("qs") * L2bb("prtu");
    L3bbb("pqrstu") += g1b("qu") * L2bb("prst");

    L3bbb("pqrstu") -= g1b("ru") * L2bb("pqst");
    L3bbb("pqrstu") += g1b("rs") * L2bb("pqut");
    L3bbb("pqrstu") += g1b("rt") * L2bb("pqsu");

    L3bbb("pqrstu") -= g1b("ps") * g1b("qt") * g1b("ru");
    L3bbb("pqrstu") -= g1b("pt") * g1b("qu") * g1b("rs");
    L3bbb("pqrstu") -= g1b("pu") * g1b("qs") * g1b("rt");

    L3bbb("pqrstu") += g1b("ps") * g1b("qu") * g1b("rt");
    L3bbb("pqrstu") += g1b("pu") * g1b("qt") * g1b("rs");
    L3bbb("pqrstu") += g1b("pt") * g1b("qs") * g1b("ru");
}

double compute_Eref_from_rdms(RDMs& ref, std::shared_ptr<ForteIntegrals> ints,
                              std::shared_ptr<MOSpaceInfo> mo_space_info) {
    // similar to MASTER_DSRG::compute_reference_energy_from_ints (use Fock and cumulants)
    // here I form two density and directly use bare Hamiltonian
    double E = ints->nuclear_repulsion_energy() + ints->frozen_core_energy();

    std::vector<size_t> core_mos = mo_space_info->corr_absolute_mo("RESTRICTED_DOCC");
    std::vector<size_t> actv_mos = mo_space_info->corr_absolute_mo("ACTIVE");
    size_t ncore = core_mos.size();
    size_t nactv = actv_mos.size();

    ambit::Tensor g1a = ref.g1a();
    ambit::Tensor g1b = ref.g1b();
    ambit::Tensor g2aa = ref.g2aa();
    ambit::Tensor g2ab = ref.g2ab();
    ambit::Tensor g2bb = ref.g2bb();

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
    E += Ha("uv") * ref.g1a()("vu");
    E += Hb("uv") * ref.g1b()("vu");

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
    E += Vtemp("munv") * I("mn") * g1a("vu");

    Vtemp = ints->aptei_ab_block(core_mos, actv_mos, core_mos, actv_mos);
    E += Vtemp("munv") * I("mn") * g1b("vu");

    Vtemp = ints->aptei_ab_block(actv_mos, core_mos, actv_mos, core_mos);
    E += Vtemp("umvn") * I("mn") * g1a("vu");

    Vtemp = ints->aptei_bb_block(core_mos, actv_mos, core_mos, actv_mos);
    E += Vtemp("munv") * I("mn") * g1b("vu");

    Vtemp = ints->aptei_aa_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += 0.25 * Vtemp("uvxy") * g2aa("uvxy");

    Vtemp = ints->aptei_ab_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += Vtemp("uVxY") * g2ab("uVxY");

    Vtemp = ints->aptei_bb_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += 0.25 * Vtemp("UVXY") * g2bb("UVXY");

    return E;
}
} // namespace forte
