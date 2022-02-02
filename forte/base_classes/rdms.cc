/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/helpers.h"
#include "helpers/timer.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "base_classes/mo_space_info.h"

namespace forte {

RDMs::RDMs() : max_rdm_(0) {}

// RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b)
//     : max_rdm_(1), have_g1b_(true), g1a_(g1a), g1b_(g1b) {}
//
// RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
//            ambit::Tensor g2bb)
//     : max_rdm_(2), have_g1b_(true), have_g2aa_(true), have_g2bb_(true), g1a_(g1a), g1b_(g1b),
//       g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb) {}
//
// RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
//            ambit::Tensor g2bb, ambit::Tensor g3aaa, ambit::Tensor g3aab, ambit::Tensor g3abb,
//            ambit::Tensor g3bbb)
//     : max_rdm_(3), have_g1b_(true), have_g2aa_(true), have_g2bb_(true), have_g3aaa_(true),
//       have_g3abb_(true), have_g3bbb_(true), g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab),
//       g2bb_(g2bb), g3aaa_(g3aaa), g3aab_(g3aab), g3abb_(g3abb), g3bbb_(g3bbb) {}

RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b)
    : max_rdm_(1), have_g1a_(true), have_g1b_(true), g1a_(g1a), g1b_(g1b) {}

RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
           ambit::Tensor g2bb)
    : max_rdm_(2), have_g1a_(true), have_g1b_(true), have_g2aa_(true), have_g2ab_(true),
      have_g2bb_(true), g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb) {}

RDMs::RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
           ambit::Tensor g2bb, ambit::Tensor g3aaa, ambit::Tensor g3aab, ambit::Tensor g3abb,
           ambit::Tensor g3bbb)
    : max_rdm_(3), have_g1a_(true), have_g1b_(true), have_g2aa_(true), have_g2ab_(true),
      have_g2bb_(true), have_g3aaa_(true), have_g3aab_(true), have_g3abb_(true), have_g3bbb_(true),
      g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb), g3aaa_(g3aaa), g3aab_(g3aab),
      g3abb_(g3abb), g3bbb_(g3bbb) {}

void RDMs::make_rdms_spin_free() {
    if (max_rdm_ > 0) {
        g1a_("pq") += g1b_("pq");
        SF_G1_ = g1a_;
        SF_G1_.set_name("SF_G1");
        have_SF_G1_ = true;
        have_g1a_ = false;
        have_g1b_ = false;
        //        g1a_ = ambit::Tensor();
        g1b_.reset();
    }
    if (max_rdm_ > 1) {
        g2aa_("pqrs") += g2bb_("pqrs");
        g2aa_("pqrs") += g2ab_("pqrs");
        g2aa_("pqrs") += g2ab_("qpsr");
        SF_G2_ = g2aa_;
        SF_G2_.set_name("SF_G2");
        have_SF_G2_ = true;
        have_g2aa_ = false;
        have_g2ab_ = false;
        have_g2bb_ = false;
        //        g2aa_ = ambit::Tensor();
        g2ab_.reset();
        g2bb_.reset();
    }
    if (max_rdm_ > 2) {
        g3aaa_("pqrstu") += g3bbb_("pqrstu");
        g3bbb_.reset();

        g3aaa_("pqrstu") += g3aab_("pqrstu");
        g3aaa_("pqrstu") += g3aab_("prqsut");
        g3aaa_("pqrstu") += g3aab_("qrptus");
        g3aab_.reset();

        g3aaa_("pqrstu") += g3abb_("pqrstu");
        g3aaa_("pqrstu") += g3abb_("qprtsu");
        g3aaa_("pqrstu") += g3abb_("rpqust");
        g3abb_.reset();

        SF_G3_ = g3aaa_;
        SF_G3_.set_name("SF_G3");
        //        g3aaa_ = ambit::Tensor();
        have_SF_G3_ = true;

        have_g3aaa_ = false;
        have_g3aab_ = false;
        have_g3abb_ = false;
        have_g3bbb_ = false;
    }
    spin_free_ = true;
}

// RDMs::RDMs(bool ms_avg, ambit::Tensor g1a) : ms_avg_(ms_avg), max_rdm_(1), g1a_(g1a) {}
//
// RDMs::RDMs(bool ms_avg, ambit::Tensor g1a, ambit::Tensor g2ab)
//     : ms_avg_(ms_avg), max_rdm_(2), g1a_(g1a), g2ab_(g2ab) {}
//
// RDMs::RDMs(bool ms_avg, ambit::Tensor g1a, ambit::Tensor g2ab, ambit::Tensor g3aab)
//     : ms_avg_(ms_avg), max_rdm_(3), g1a_(g1a), g2ab_(g2ab), g3aab_(g3aab) {}

void RDMs::validate(const size_t& level, const std::string& name, const bool& must_ms_avg) const {
    if (level > max_rdm_) {
        std::string msg = "Impossible to build " + name;
        msg += ": max RDM level is " + std::to_string(max_rdm_);
        throw std::runtime_error(msg);
    }
    if (must_ms_avg and (not ms_avg_)) {
        throw std::runtime_error("Must turn on spin averaging.");
    }
}

ambit::Tensor RDMs::g1a() {
    validate(1, "g1a");
    if (spin_free_) {
        auto g1a = SF_G1_.clone();
        g1a.scale(0.5);
        g1a.set_name("g1a");
        return g1a;
    }
    return g1a_;
}

ambit::Tensor RDMs::g1b() {
    validate(1, "g1b");
    if (spin_free_) {
        auto g1b = SF_G1_.clone();
        g1b.scale(0.5);
        g1b.set_name("g1b");
        return g1b;
    }
    return g1b_;
}

// ambit::Tensor RDMs::g1b() {
//     validate(1, "g1b");
//     if (ms_avg_ and (not have_g1b_)) {
//         g1b_ = g1a_.clone();
//         have_g1b_ = true;
//     }
//     return g1b_;
// }

ambit::Tensor RDMs::g2ab() {
    validate(2, "g2ab");
    if (spin_free_) {
        auto g2ab = SF_G2_.clone();
        g2ab.scale(1.0 / 3.0);
        g2ab("pqrs") += (1.0 / 6.0) * SF_G2_("pqsr");
        g2ab.set_name("g2ab");
        return g2ab;
    }
    return g2ab_;
}

ambit::Tensor RDMs::g2aa() {
    validate(2, "g2aa");
    if (spin_free_) {
        auto g2aa = SF_G2_.clone();
        g2aa("pqrs") -= SF_G2_("pqsr");
        g2aa.scale(1.0 / 6.0);
        g2aa.set_name("g2aa");
        return g2aa;
    }
    return g2aa_;
}

ambit::Tensor RDMs::g2bb() {
    validate(2, "g2bb");
    if (spin_free_) {
        auto g2bb = g2aa();
        g2bb.set_name("g2bb");
        return g2bb;
    }
    return g2bb_;
}

// ambit::Tensor RDMs::g2aa() {
//     validate(2, "g2aa");
//     if (ms_avg_ and (not have_g2aa_)) {
//         g2aa_ = make_g2_high_spin_case(g2ab_);
//         have_g2aa_ = true;
//     }
//     return g2aa_;
// }
//
// ambit::Tensor RDMs::g2bb() {
//     validate(2, "g2bb");
//     if (ms_avg_ and (not have_g2bb_)) {
//         g2bb_ = make_g2_high_spin_case(g2ab_);
//         have_g2bb_ = true;
//     }
//     return g2bb_;
// }

ambit::Tensor RDMs::g3aab() {
    validate(3, "g3aab");
    if (spin_free_) {
        auto g3aab = SF_G3_.clone();
        g3aab("pqrstu") -= SF_G3_("pqrtus") + SF_G3_("pqrust") + 2.0 * SF_G3_("pqrtsu");
        g3aab.scale(1.0 / 12.0);
        g3aab.set_name("g3aab");
        return g3aab;
    }
    return g3aab_;
}

ambit::Tensor RDMs::g3abb() {
    validate(3, "g3abb");
    if (spin_free_) {
        auto g3abb = SF_G3_.clone();
        g3abb("pqrstu") -= SF_G3_("pqrtus") + SF_G3_("pqrust") + 2.0 * SF_G3_("pqrsut");
        g3abb.scale(1.0 / 12.0);
        g3abb.set_name("g3abb");
        return g3abb;
    }
    return g3abb_;
}

ambit::Tensor RDMs::g3aaa() {
    validate(3, "g3aaa");
    if (spin_free_) {
        auto g3aaa = SF_G3_.clone();
        g3aaa("pqrstu") += SF_G3_("pqrtus") + SF_G3_("pqrust");
        g3aaa.scale(1.0 / 12.0);
        g3aaa.set_name("g3aaa");
        return g3aaa;
    }
    return g3aaa_;
}

ambit::Tensor RDMs::g3bbb() {
    validate(3, "g3bbb");
    if (spin_free_) {
        auto g3bbb = g3aaa();
        g3bbb.set_name("g3bbb");
        return g3bbb;
    }
    return g3bbb_;
}

// ambit::Tensor RDMs::g3aaa() {
//     validate(3, "g3aaa");
//     if (ms_avg_ and (not have_g3aaa_)) {
//         g3aaa_ = make_g3_high_spin_case(g3aab_);
//         have_g3aaa_ = true;
//     }
//     return g3aaa_;
// }
//
// ambit::Tensor RDMs::g3abb() {
//     validate(3, "g3abb");
//     if (ms_avg_ and (not have_g3abb_)) {
//         g3abb_ = g3aab_.clone();
//         g3abb_("rpqust") = g3aab_("pqrstu");
//         have_g3abb_ = true;
//     }
//     return g3abb_;
// }
//
// ambit::Tensor RDMs::g3bbb() {
//     validate(3, "g3bbb");
//     if (ms_avg_ and (not have_g3bbb_)) {
//         g3bbb_ = make_g3_high_spin_case(g3aab_);
//         have_g3bbb_ = true;
//     }
//     return g3bbb_;
// }

ambit::Tensor RDMs::L2aa() {
    validate(2, "L2aa");
    if (not have_L2aa_) {
        L2aa_ = g2aa().clone();
        make_cumulant_L2aa_in_place(g1a_, L2aa_);
        have_L2aa_ = true;
    }
    return L2aa_;
}

ambit::Tensor RDMs::L2ab() {
    validate(2, "L2ab");
    if (not have_L2ab_) {
        L2ab_ = g2ab_.clone();
        make_cumulant_L2ab_in_place(g1a_, g1b(), L2ab_);
        have_L2ab_ = true;
    }
    return L2ab_;
}

ambit::Tensor RDMs::L2bb() {
    validate(2, "L2ab");
    if (not have_L2bb_) {
        L2bb_ = g2bb().clone();
        make_cumulant_L2bb_in_place(g1b(), L2bb_);
        have_L2bb_ = true;
    }
    return L2bb_;
}

ambit::Tensor RDMs::L3aaa() {
    validate(3, "L3aaa");
    if (not have_L3aaa_) {
        L3aaa_ = g3aaa().clone();
        make_cumulant_L3aaa_in_place(g1a_, L2aa(), L3aaa_);
        have_L3aaa_ = true;
    }
    return L3aaa_;
}

ambit::Tensor RDMs::L3aab() {
    validate(3, "L3aab");
    if (not have_L3aab_) {
        L3aab_ = g3aab_.clone();
        make_cumulant_L3aab_in_place(g1a_, g1b(), L2aa(), L2ab_, L3aab_);
        have_L3aab_ = true;
    }
    return L3aab_;
}

ambit::Tensor RDMs::L3abb() {
    validate(3, "L3abb");
    if (not have_L3abb_) {
        L3abb_ = g3abb().clone();
        make_cumulant_L3abb_in_place(g1a_, g1b(), L2ab_, L2bb(), L3abb_);
        have_L3abb_ = true;
    }
    return L3abb_;
}

ambit::Tensor RDMs::L3bbb() {
    validate(3, "L3bbb");
    if (not have_L3bbb_) {
        L3bbb_ = g3bbb().clone();
        make_cumulant_L3bbb_in_place(g1b(), L2bb(), L3bbb_);
        have_L3bbb_ = true;
    }
    return L3bbb_;
}

ambit::Tensor RDMs::SF_G1() {
    validate(1, "SF_G1");
    if (not have_SF_G1_) {
        SF_G1_ = g1a_.clone();
        if (ms_avg_) {
            SF_G1_.scale(2.0);
        } else {
            SF_G1_("pq") += g1b()("pq");
        }
        have_SF_G1_ = true;
    }
    return SF_G1_;
}

std::shared_ptr<psi::Matrix> RDMs::SF_G1mat() {
    SF_G1();
    auto M = tensor_to_matrix(SF_G1_);
    M->set_name("1-RDM NoSym");
    return M;
}

std::shared_ptr<psi::Matrix> RDMs::SF_G1mat(const psi::Dimension& dim) {
    SF_G1();
    auto M = tensor_to_matrix(SF_G1_, dim);
    M->set_name("1-RDM");
    return M;
}

ambit::Tensor RDMs::SF_G2() {
    validate(2, "SF_G2");
    if (not have_SF_G2_) {
        SF_G2_ = g2ab_.clone();
        if (ms_avg_) {
            SF_G2_.scale(4.0);
            SF_G2_("pqrs") -= 2.0 * g2ab_("pqsr");
        } else {
            SF_G2_("pqrs") += g2ab_("qpsr");
            SF_G2_("pqrs") += g2aa()("pqrs") + g2bb()("pqrs");
        }
        have_SF_G2_ = true;
    }
    return SF_G2_;
}

ambit::Tensor RDMs::SF_L1() { return SF_G1_; }

ambit::Tensor RDMs::SF_L2() {
    validate(2, "SF_L2", false);
    if (not have_SF_L2_) {
        SF_L2_ = SF_G2_.clone();
        SF_L2_("pqrs") -= SF_G1_("pr") * SF_G1_("qs");
        SF_L2_("pqrs") += 0.5 * SF_G1_("ps") * SF_G1_("qr");
        have_SF_L2_ = true;
    }
    return SF_L2_;
}

ambit::Tensor RDMs::SF_L3() {
    validate(3, "SF_L3", false);
    if (not have_SF_L3_) {
        SF_L3_ = SF_G3_.clone();

        SF_L3_("pqrstu") -= SF_G1_("ps") * SF_G2_("qrtu");
        SF_L3_("pqrstu") -= SF_G1_("qt") * SF_G2_("prsu");
        SF_L3_("pqrstu") -= SF_G1_("ru") * SF_G2_("pqst");

        SF_L3_("pqrstu") += 0.5 * SF_G1_("pt") * SF_G2_("qrsu");
        SF_L3_("pqrstu") += 0.5 * SF_G1_("pu") * SF_G2_("qrts");

        SF_L3_("pqrstu") += 0.5 * SF_G1_("qs") * SF_G2_("prtu");
        SF_L3_("pqrstu") += 0.5 * SF_G1_("qu") * SF_G2_("prst");

        SF_L3_("pqrstu") += 0.5 * SF_G1_("rs") * SF_G2_("pqut");
        SF_L3_("pqrstu") += 0.5 * SF_G1_("rt") * SF_G2_("pqsu");

        SF_L3_("pqrstu") += 2.0 * SF_G1_("ps") * SF_G1_("qt") * SF_G1_("ru");

        SF_L3_("pqrstu") -= SF_G1_("ps") * SF_G1_("qu") * SF_G1_("rt");
        SF_L3_("pqrstu") -= SF_G1_("pu") * SF_G1_("qt") * SF_G1_("rs");
        SF_L3_("pqrstu") -= SF_G1_("pt") * SF_G1_("qs") * SF_G1_("ru");

        SF_L3_("pqrstu") += 0.5 * SF_G1_("pt") * SF_G1_("qu") * SF_G1_("rs");
        SF_L3_("pqrstu") += 0.5 * SF_G1_("pu") * SF_G1_("qs") * SF_G1_("rt");

        have_SF_L3_ = true;
    }
    return SF_L3_;
}

// ambit::Tensor RDMs::SF_L2() {
//     validate(2, "SF_L2", true);
//     if (not have_SF_L2_) {
//         SF_L2_ = L2ab().clone();
//         SF_L2_.scale(4.0);
//         SF_L2_("pqrs") -= 2.0 * L2ab()("pqsr");
//         have_SF_L2_ = true;
//     }
//     return SF_L2_;
// }
//
// ambit::Tensor RDMs::SF_L3() {
//     validate(3, "SF_L3", true);
//     if (not have_SF_L3_) {
//         SF_L3_ = make_g3_high_spin_case(L3aab());
//         SF_L3_("pqrstu") += L3aab()("pqrstu");
//         SF_L3_("pqrstu") += L3aab()("prqsut");
//         SF_L3_("pqrstu") += L3aab()("qrptus");
//         SF_L3_.scale(2.0);
//         have_SF_L3_ = true;
//     }
//     return SF_L3_;
// }

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

void RDMs::rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) {
    //    if (ms_avg_) {
    if (spin_free_) {
        rotate_restricted(Ua);
    } else {
        rotate_unrestricted(Ua, Ub);
    }
}

void RDMs::rotate_restricted(const ambit::Tensor& Ua) {
    if (max_rdm_ < 1) {
        return;
    }

    psi::outfile->Printf("\n  Rotating RDMs using spin restricted formalism ...");
    reset_built_flags();

    // transform 1-RDM
    auto g1T = ambit::Tensor::build(ambit::CoreTensor, "g1aT", SF_G1_.dims());
    g1T("pq") = Ua("ap") * SF_G1_("ab") * Ua("bq");
    SF_G1_("pq") = g1T("pq");
    have_SF_G1_ = true;
    //    auto g1T = ambit::Tensor::build(ambit::CoreTensor, "g1aT", g1a_.dims());
    //    g1T("pq") = Ua("ap") * g1a_("ab") * Ua("bq");
    //    g1a_("pq") = g1T("pq");
    g1T.reset();
    psi::outfile->Printf("\n    Transformed 1 RDM.");
    if (max_rdm_ == 1)
        return;

    // transform 2-RDM
    auto g2T = ambit::Tensor::build(ambit::CoreTensor, "g2abT", SF_G2_.dims());
    g2T("pQrS") = Ua("ap") * Ua("BQ") * SF_G2_("aBcD") * Ua("cr") * Ua("DS");
    SF_G2_("pqrs") = g2T("pqrs");
    have_SF_G2_ = true;
    //    auto g2T = ambit::Tensor::build(ambit::CoreTensor, "g2abT", g2ab_.dims());
    //    g2T("pQrS") = Ua("ap") * Ua("BQ") * g2ab_("aBcD") * Ua("cr") * Ua("DS");
    //    g2ab_("pqrs") = g2T("pqrs");
    g2T.reset();
    psi::outfile->Printf("\n    Transformed 2 RDM.");
    if (max_rdm_ == 2)
        return;

    // transform 3-RDM
    auto g3T = ambit::Tensor::build(ambit::CoreTensor, "g3aabT", SF_G3_.dims());
    g3T("pqRstU") =
        Ua("ap") * Ua("bq") * Ua("CR") * SF_G3_("abCijK") * Ua("is") * Ua("jt") * Ua("KU");
    SF_G3_("abcijk") = g3T("abcijk");
    have_SF_G3_ = true;
    //    auto g3T = ambit::Tensor::build(ambit::CoreTensor, "g3aabT", g3aab_.dims());
    //    g3T("pqRstU") =
    //        Ua("ap") * Ua("bq") * Ua("CR") * g3aab_("abCijK") * Ua("is") * Ua("jt") * Ua("KU");
    //    g3aab_("abcijk") = g3T("abcijk");
    g3T.reset();
    psi::outfile->Printf("\n    Transformed 3 RDM.");
}

void RDMs::rotate_unrestricted(const ambit::Tensor& Ua, const ambit::Tensor& Ub) {
    if (max_rdm_ < 1)
        return;

    psi::outfile->Printf("\n  Rotating RDMs using spin unrestricted formalism ...");
    reset_built_flags();

    // Transform the 1-rdms
    ambit::Tensor g1aT = ambit::Tensor::build(ambit::CoreTensor, "g1aT", g1a_.dims());
    ambit::Tensor g1bT = ambit::Tensor::build(ambit::CoreTensor, "g1bT", g1b_.dims());

    g1aT("pq") = Ua("ap") * g1a_("ab") * Ua("bq");
    g1bT("PQ") = Ub("AP") * g1b_("AB") * Ub("BQ");

    g1a_("pq") = g1aT("pq");
    g1b_("pq") = g1bT("pq");

    psi::outfile->Printf("\n    Transformed 1 RDMs.");

    if (max_rdm_ == 1)
        return;

    // Transform the 2-rdms
    auto g2Taa = ambit::Tensor::build(ambit::CoreTensor, "g2aaT", g2aa_.dims());
    auto g2Tab = ambit::Tensor::build(ambit::CoreTensor, "g2abT", g2ab_.dims());
    auto g2Tbb = ambit::Tensor::build(ambit::CoreTensor, "g2bbT", g2bb_.dims());

    g2Taa("pqrs") = Ua("ap") * Ua("bq") * g2aa_("abcd") * Ua("cr") * Ua("ds");
    g2Tab("pQrS") = Ua("ap") * Ub("BQ") * g2ab_("aBcD") * Ua("cr") * Ub("DS");
    g2Tbb("PQRS") = Ub("AP") * Ub("BQ") * g2bb_("ABCD") * Ub("CR") * Ub("DS");

    g2aa_("pqrs") = g2Taa("pqrs");
    g2ab_("pqrs") = g2Tab("pqrs");
    g2bb_("pqrs") = g2Tbb("pqrs");

    psi::outfile->Printf("\n    Transformed 2 RDMs.");

    if (max_rdm_ == 2)
        return;

    // Transform the 3-rdms
    auto g3T = ambit::Tensor::build(ambit::CoreTensor, "g3T", g3aaa_.dims());
    g3T("pqrstu") =
        Ua("ap") * Ua("bq") * Ua("cr") * g3aaa_("abcijk") * Ua("is") * Ua("jt") * Ua("ku");
    g3aaa_("pqrstu") = g3T("pqrstu");

    g3T("pqRstU") =
        Ua("ap") * Ua("bq") * Ub("CR") * g3aab_("abCijK") * Ua("is") * Ua("jt") * Ub("KU");
    g3aab_("pqrstu") = g3T("pqrstu");

    g3T("pQRsTU") =
        Ua("ap") * Ub("BQ") * Ub("CR") * g3abb_("aBCiJK") * Ua("is") * Ub("JT") * Ub("KU");
    g3abb_("pqrstu") = g3T("pqrstu");

    g3T("PQRSTU") =
        Ub("AP") * Ub("BQ") * Ub("CR") * g3bbb_("ABCIJK") * Ub("IS") * Ub("JT") * Ub("KU");
    g3bbb_("pqrstu") = g3T("pqrstu");

    psi::outfile->Printf("\n    Transformed 3 RDMs.");
}

void RDMs::reset_built_flags() {
    if (max_rdm_ >= 1) {
        have_g1b_ = false;
        have_SF_G1_ = false;
    }
    if (max_rdm_ >= 2) {
        have_g2aa_ = false;
        have_g2bb_ = false;
        have_SF_G2_ = false;
        have_L2aa_ = false;
        have_L2ab_ = false;
        have_L2bb_ = false;
        have_SF_L2_ = false;
    }
    if (max_rdm_ >= 3) {
        have_g3aaa_ = false;
        have_g3abb_ = false;
        have_g3bbb_ = false;
        have_L3aaa_ = false;
        have_L3aab_ = false;
        have_L3abb_ = false;
        have_L3bbb_ = false;
        have_SF_L3_ = false;
    }
}

std::shared_ptr<RDMs_NEW> RDMs_NEW::build(size_t max_rdm_level, size_t n_orbs, RDMsType type) {
    std::vector<size_t> dims1(2, n_orbs);
    std::vector<size_t> dims2(4, n_orbs);
    std::vector<size_t> dims3(6, n_orbs);

    std::shared_ptr<RDMs_NEW> rdms;

    if (type == spin_dependent) {
        ambit::Tensor g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb;
        if (max_rdm_level > 0) {
            g1a = ambit::Tensor::build(ambit::CoreTensor, "g1a", dims1);
            g1b = ambit::Tensor::build(ambit::CoreTensor, "g1b", dims1);
        }
        if (max_rdm_level > 1) {
            g2aa = ambit::Tensor::build(ambit::CoreTensor, "g2aa", dims2);
            g2ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab", dims2);
            g2bb = ambit::Tensor::build(ambit::CoreTensor, "g2bb", dims2);
        }
        if (max_rdm_level > 2) {
            g3aaa = ambit::Tensor::build(ambit::CoreTensor, "g3aaa", dims3);
            g3aab = ambit::Tensor::build(ambit::CoreTensor, "g3aab", dims3);
            g3abb = ambit::Tensor::build(ambit::CoreTensor, "g3abb", dims3);
            g3bbb = ambit::Tensor::build(ambit::CoreTensor, "g3bbb", dims3);
        }

        if (max_rdm_level < 1) {
            rdms = std::make_shared<SD_RDMs>();
        } else if (max_rdm_level == 1) {
            rdms = std::make_shared<SD_RDMs>(g1a, g1b);
        } else if (max_rdm_level == 2) {
            rdms = std::make_shared<SD_RDMs>(g1a, g1b, g2aa, g2ab, g2bb);
        } else {
            rdms =
                std::make_shared<SD_RDMs>(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb);
        }
    } else {
        ambit::Tensor g1, g2, g3;
        if (max_rdm_level > 0) {
            g1 = ambit::Tensor::build(ambit::CoreTensor, "SF_G1", dims1);
        }
        if (max_rdm_level > 1) {
            g2 = ambit::Tensor::build(ambit::CoreTensor, "SF_G2", dims2);
        }
        if (max_rdm_level > 2) {
            g3 = ambit::Tensor::build(ambit::CoreTensor, "SF_G3", dims3);
        }

        if (max_rdm_level < 1) {
            rdms = std::make_shared<SF_RDMs>();
        } else if (max_rdm_level == 1) {
            rdms = std::make_shared<SF_RDMs>(g1);
        } else if (max_rdm_level == 2) {
            rdms = std::make_shared<SF_RDMs>(g1, g2);
        } else {
            rdms = std::make_shared<SF_RDMs>(g1, g2, g3);
        }
    }
    return rdms;
}

std::shared_ptr<psi::Matrix> RDMs_NEW::SF_G1mat() {
    auto G1 = SF_G1();
    auto M = tensor_to_matrix(G1);
    M->set_name("1-RDM NoSym");
    return M;
}

std::shared_ptr<psi::Matrix> RDMs_NEW::SF_G1mat(const psi::Dimension& dim) {
    auto G1 = SF_G1();
    auto M = tensor_to_matrix(G1, dim);
    M->set_name("1-RDM");
    return M;
}

ambit::Tensor RDMs_NEW::SF_L1() const {
    auto L1 = SF_G1();
    L1.set_name("SF_L1");
    return L1;
}

ambit::Tensor RDMs_NEW::SF_L2() const {
    _test_rdm_level(2, "L2");
    timer t("make_cumulant_L2");
    auto G1 = SF_G1();
    auto L2 = SF_G2().clone();
    L2("pqrs") -= G1("pr") * G1("qs");
    L2("pqrs") += 0.5 * G1("ps") * G1("qr");
    L2.set_name("SF_L2");
    return L2;
}

ambit::Tensor RDMs_NEW::SF_L3() const {
    _test_rdm_level(3, "SF_L3");
    timer t("make_cumulant_L3");

    auto G1 = SF_G1();
    auto G2 = SF_G2();
    auto L3 = SF_G3().clone();

    L3("pqrstu") -= G1("ps") * G2("qrtu");
    L3("pqrstu") -= G1("qt") * G2("prsu");
    L3("pqrstu") -= G1("ru") * G2("pqst");

    L3("pqrstu") += 0.5 * G1("pt") * G2("qrsu");
    L3("pqrstu") += 0.5 * G1("pu") * G2("qrts");

    L3("pqrstu") += 0.5 * G1("qs") * G2("prtu");
    L3("pqrstu") += 0.5 * G1("qu") * G2("prst");

    L3("pqrstu") += 0.5 * G1("rs") * G2("pqut");
    L3("pqrstu") += 0.5 * G1("rt") * G2("pqsu");

    L3("pqrstu") += 2.0 * G1("ps") * G1("qt") * G1("ru");

    L3("pqrstu") -= G1("ps") * G1("qu") * G1("rt");
    L3("pqrstu") -= G1("pu") * G1("qt") * G1("rs");
    L3("pqrstu") -= G1("pt") * G1("qs") * G1("ru");

    L3("pqrstu") += 0.5 * G1("pt") * G1("qu") * G1("rs");
    L3("pqrstu") += 0.5 * G1("pu") * G1("qs") * G1("rt");

    L3.set_name("SF_L3");
    return L3;
}

ambit::Tensor RDMs_NEW::make_cumulant_L2aa(const ambit::Tensor& g1a, const ambit::Tensor& g2aa) {
    timer t("make_cumulant_L2aa");
    auto L2aa = g2aa.clone();
    L2aa("pqrs") -= g1a("pr") * g1a("qs");
    L2aa("pqrs") += g1a("ps") * g1a("qr");
    return L2aa;
}

ambit::Tensor RDMs_NEW::make_cumulant_L2ab(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                           const ambit::Tensor& g2ab) {
    timer t("make_cumulant_L2ab");
    auto L2ab = g2ab.clone();
    L2ab("pqrs") -= g1a("pr") * g1b("qs");
    return L2ab;
}

ambit::Tensor RDMs_NEW::make_cumulant_L3aaa(const ambit::Tensor& g1a, const ambit::Tensor& g2aa,
                                            const ambit::Tensor& g3aaa) {
    timer t("make_cumulant_L3aaa");

    auto L3aaa = g3aaa.clone();

    L3aaa("pqrstu") -= g1a("ps") * g2aa("qrtu");
    L3aaa("pqrstu") += g1a("pt") * g2aa("qrsu");
    L3aaa("pqrstu") += g1a("pu") * g2aa("qrts");

    L3aaa("pqrstu") -= g1a("qt") * g2aa("prsu");
    L3aaa("pqrstu") += g1a("qs") * g2aa("prtu");
    L3aaa("pqrstu") += g1a("qu") * g2aa("prst");

    L3aaa("pqrstu") -= g1a("ru") * g2aa("pqst");
    L3aaa("pqrstu") += g1a("rs") * g2aa("pqut");
    L3aaa("pqrstu") += g1a("rt") * g2aa("pqsu");

    L3aaa("pqrstu") += 2.0 * g1a("ps") * g1a("qt") * g1a("ru");
    L3aaa("pqrstu") += 2.0 * g1a("pt") * g1a("qu") * g1a("rs");
    L3aaa("pqrstu") += 2.0 * g1a("pu") * g1a("qs") * g1a("rt");

    L3aaa("pqrstu") -= 2.0 * g1a("ps") * g1a("qu") * g1a("rt");
    L3aaa("pqrstu") -= 2.0 * g1a("pu") * g1a("qt") * g1a("rs");
    L3aaa("pqrstu") -= 2.0 * g1a("pt") * g1a("qs") * g1a("ru");

    return L3aaa;
}

ambit::Tensor RDMs_NEW::make_cumulant_L3aab(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                            const ambit::Tensor& g2aa, const ambit::Tensor& g2ab,
                                            const ambit::Tensor& g3aab) {
    timer t("make_cumulant_L3aab");

    auto L3aab = g3aab.clone();
    L3aab("pqRstU") -= g1b("RU") * g2aa("pqst");

    L3aab("pqRstU") -= g1a("ps") * g2ab("qRtU");
    L3aab("pqRstU") += g1a("pt") * g2ab("qRsU");

    L3aab("pqRstU") -= g1a("qt") * g2ab("pRsU");
    L3aab("pqRstU") += g1a("qs") * g2ab("pRtU");

    L3aab("pqRstU") += 2.0 * g1a("ps") * g1a("qt") * g1b("RU");
    L3aab("pqRstU") -= 2.0 * g1a("pt") * g1a("qs") * g1b("RU");

    return L3aab;
}

ambit::Tensor RDMs_NEW::make_cumulant_L3abb(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                            const ambit::Tensor& g2ab, const ambit::Tensor& g2bb,
                                            const ambit::Tensor& g3abb) {
    timer t("make_cumulant_L3abb");

    auto L3abb = g3abb.clone();
    L3abb("pQRsTU") -= g1a("ps") * g2bb("QRTU");

    L3abb("pQRsTU") -= g1b("QT") * g2ab("pRsU");
    L3abb("pQRsTU") += g1b("QU") * g2ab("pRsT");

    L3abb("pQRsTU") -= g1b("RU") * g2ab("pQsT");
    L3abb("pQRsTU") += g1b("RT") * g2ab("pQsU");

    L3abb("pQRsTU") += 2.0 * g1a("ps") * g1b("QT") * g1b("RU");
    L3abb("pQRsTU") -= 2.0 * g1a("ps") * g1b("QU") * g1b("RT");

    return L3abb;
}

ambit::Tensor RDMs_NEW::sf1_to_sd1(const ambit::Tensor& G1) {
    auto g1 = G1.clone();
    g1.scale(0.5);
    return g1;
}

ambit::Tensor RDMs_NEW::sf2_to_sd2aa(const ambit::Tensor& G2) {
    auto g2aa = G2.clone();
    g2aa("pqrs") -= G2("pqsr");
    g2aa.scale(1.0 / 6.0);
    return g2aa;
}

ambit::Tensor RDMs_NEW::sf2_to_sd2ab(const ambit::Tensor& G2) {
    auto g2ab = G2.clone();
    g2ab.scale(1.0 / 3.0);
    g2ab("pqrs") += (1.0 / 6.0) * G2("pqsr");
    return g2ab;
}

ambit::Tensor RDMs_NEW::sf3_to_sd3aaa(const ambit::Tensor& G3) {
    auto g3aaa = G3.clone();
    g3aaa("pqrstu") += G3("pqrtus") + G3("pqrust");
    g3aaa.scale(1.0 / 12.0);
    return g3aaa;
}

ambit::Tensor RDMs_NEW::sf3_to_sd3aab(const ambit::Tensor& G3) {
    auto g3aab = G3.clone();
    g3aab("pqrstu") -= G3("pqrtus") + G3("pqrust") + 2.0 * G3("pqrtsu");
    g3aab.scale(1.0 / 12.0);
    return g3aab;
}

ambit::Tensor RDMs_NEW::sf3_to_sd3abb(const ambit::Tensor& G3) {
    auto g3abb = G3.clone();
    g3abb("pqrstu") -= G3("pqrtus") + G3("pqrust") + 2.0 * G3("pqrsut");
    g3abb.scale(1.0 / 12.0);
    return g3abb;
}

void RDMs_NEW::_test_rdm_level(const size_t& level, const std::string& name) const {
    if (level > max_rdm_) {
        std::string msg = "Impossible to build " + name;
        msg += ": max RDM level is " + std::to_string(max_rdm_);
        throw std::runtime_error(msg);
    }
}

void RDMs_NEW::_test_rdm_dims(const ambit::Tensor& T, const std::string& name) const {
    const auto& dims = T.dims();
    if (dims.size() < 2) {
        throw std::runtime_error("Invalid dimension (too small) for " + name);
    }
    if (std::find_if(dims.begin(), dims.end(), [&](size_t i) { return i != n_orbs_; }) !=
        dims.end()) {
        std::stringstream ss;
        ss << "Invalid dimensions for " << name << ": " << dims[0];
        for (size_t i = 0, size = dims.size() - 1; i < size; ++i) {
            ss << " x " << dims[i + 1];
        }
        ss << "; Expect: " << n_orbs_;
        for (size_t i = 0, size = dims.size() - 1; i < size; ++i) {
            ss << " x " << n_orbs_;
        }
        throw std::runtime_error(ss.str());
    }
}

SD_RDMs::SD_RDMs() {
    max_rdm_ = 0;
    type_ = spin_dependent;
    n_orbs_ = 0;
}

SD_RDMs::SD_RDMs(ambit::Tensor g1a, ambit::Tensor g1b) : g1a_(g1a), g1b_(g1b) {
    max_rdm_ = 1;
    type_ = spin_dependent;
    n_orbs_ = g1a.dim(0);
    _test_rdm_dims(g1a, "g1a");
    _test_rdm_dims(g1b, "g1b");
}

SD_RDMs::SD_RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
                 ambit::Tensor g2bb)
    : g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb) {
    max_rdm_ = 2;
    type_ = spin_dependent;
    n_orbs_ = g1a.dim(0);
    _test_rdm_dims(g1a, "g1a");
    _test_rdm_dims(g1b, "g1b");
    _test_rdm_dims(g2aa, "g2aa");
    _test_rdm_dims(g2ab, "g2ab");
    _test_rdm_dims(g2bb, "g2bb");
}

SD_RDMs::SD_RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
                 ambit::Tensor g2bb, ambit::Tensor g3aaa, ambit::Tensor g3aab, ambit::Tensor g3abb,
                 ambit::Tensor g3bbb)
    : g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb), g3aaa_(g3aaa), g3aab_(g3aab),
      g3abb_(g3abb), g3bbb_(g3bbb) {
    max_rdm_ = 3;
    type_ = spin_dependent;
    n_orbs_ = g1a.dim(0);
    _test_rdm_dims(g1a, "g1a");
    _test_rdm_dims(g1b, "g1b");
    _test_rdm_dims(g2aa, "g2aa");
    _test_rdm_dims(g2ab, "g2ab");
    _test_rdm_dims(g2bb, "g2bb");
    _test_rdm_dims(g3aaa, "g3aaa");
    _test_rdm_dims(g3aab, "g3aab");
    _test_rdm_dims(g3abb, "g3abb");
    _test_rdm_dims(g3bbb, "g3bbb");
}

ambit::Tensor SD_RDMs::g1a() const {
    _test_rdm_level(1, "g1a");
    return g1a_;
}
ambit::Tensor SD_RDMs::g1b() const {
    _test_rdm_level(1, "g1b");
    return g1b_;
}
ambit::Tensor SD_RDMs::g2aa() const {
    _test_rdm_level(2, "g2aa");
    return g2aa_;
}
ambit::Tensor SD_RDMs::g2ab() const {
    _test_rdm_level(2, "g2ab");
    return g2ab_;
}
ambit::Tensor SD_RDMs::g2bb() const {
    _test_rdm_level(2, "g2bb");
    return g2bb_;
}
ambit::Tensor SD_RDMs::g3aaa() const {
    _test_rdm_level(3, "g3aaa");
    return g3aaa_;
}
ambit::Tensor SD_RDMs::g3aab() const {
    _test_rdm_level(3, "g3aab");
    return g3aab_;
}
ambit::Tensor SD_RDMs::g3abb() const {
    _test_rdm_level(3, "g3abb");
    return g3abb_;
}
ambit::Tensor SD_RDMs::g3bbb() const {
    _test_rdm_level(3, "g3bbb");
    return g3bbb_;
}
ambit::Tensor SD_RDMs::SF_G1() const {
    _test_rdm_level(1, "SF_G1");
    auto G1 = g1a_.clone();
    G1("pq") += g1b_("pq");
    G1.set_name("SF_G1");
    return G1;
}
ambit::Tensor SD_RDMs::SF_G2() const {
    _test_rdm_level(3, "SF_G2");
    auto G2 = g2aa_.clone();
    G2("pqrs") += g2bb_("pqrs");
    G2("pqrs") += g2ab_("pqrs");
    G2("pqrs") += g2ab_("qpsr");
    G2.set_name("SF_G2");
    return G2;
}
ambit::Tensor SD_RDMs::SF_G3() const {
    _test_rdm_level(3, "SF_G3");
    auto G3 = g3aaa_.clone();
    G3("pqrstu") += g3bbb_("pqrstu");

    G3("pqrstu") += g3aab_("pqrstu");
    G3("pqrstu") += g3aab_("prqsut");
    G3("pqrstu") += g3aab_("qrptus");

    G3("pqrstu") += g3abb_("pqrstu");
    G3("pqrstu") += g3abb_("qprtsu");
    G3("pqrstu") += g3abb_("rpqust");

    G3.set_name("SF_G3");
    return G3;
}
ambit::Tensor SD_RDMs::L1a() const {
    auto L1a = g1a();
    L1a.set_name("L1a");
    return L1a;
}
ambit::Tensor SD_RDMs::L1b() const {
    auto L1b = g1b();
    L1b.set_name("L1b");
    return L1b;
}
ambit::Tensor SD_RDMs::L2aa() const {
    _test_rdm_level(2, "L2aa");
    auto L2aa = make_cumulant_L2aa(g1a_, g2aa_);
    L2aa.set_name("L2aa");
    return L2aa;
}
ambit::Tensor SD_RDMs::L2ab() const {
    _test_rdm_level(2, "L2ab");
    auto L2ab = make_cumulant_L2ab(g1a_, g1b_, g2ab_);
    L2ab.set_name("L2ab");
    return L2ab;
}
ambit::Tensor SD_RDMs::L2bb() const {
    _test_rdm_level(2, "L2bb");
    auto L2bb = make_cumulant_L2aa(g1b_, g2bb_);
    L2bb.set_name("L2bb");
    return L2bb;
}
ambit::Tensor SD_RDMs::L3aaa() const {
    _test_rdm_level(3, "L3aaa");
    auto L3aaa = make_cumulant_L3aaa(g1a_, g2aa_, g3aaa_);
    L3aaa.set_name("L3aaa");
    return L3aaa;
}
ambit::Tensor SD_RDMs::L3aab() const {
    _test_rdm_level(3, "L3aab");
    auto L3aab = make_cumulant_L3aab(g1a_, g1b_, g2aa_, g2ab_, g3aab_);
    L3aab.set_name("L3aab");
    return L3aab;
}
ambit::Tensor SD_RDMs::L3abb() const {
    _test_rdm_level(3, "L3abb");
    auto L3abb = make_cumulant_L3abb(g1a_, g1b_, g2ab_, g2bb_, g3abb_);
    L3abb.set_name("L3abb");
    return L3abb;
}
ambit::Tensor SD_RDMs::L3bbb() const {
    _test_rdm_level(3, "L3bbb");
    auto L3bbb = make_cumulant_L3aaa(g1b_, g2bb_, g3bbb_);
    L3bbb.set_name("L3bbb");
    return L3bbb;
}

std::shared_ptr<RDMs_NEW> SD_RDMs::clone() {
    ambit::Tensor g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb;
    if (max_rdm_ > 0) {
        g1a = g1a_.clone();
        g1b = g1b_.clone();
    }
    if (max_rdm_ > 1) {
        g2aa = g2aa_.clone();
        g2ab = g2ab_.clone();
        g2bb = g2bb_.clone();
    }
    if (max_rdm_ > 2) {
        g3aaa = g3aaa_.clone();
        g3aab = g3aab_.clone();
        g3abb = g3abb_.clone();
        g3bbb = g3bbb_.clone();
    }

    std::shared_ptr<RDMs_NEW> rdms;

    if (max_rdm_ < 1)
        rdms = std::make_shared<SD_RDMs>();
    else if (max_rdm_ == 1)
        rdms = std::make_shared<SD_RDMs>(g1a, g1b);
    else if (max_rdm_ == 2)
        rdms = std::make_shared<SD_RDMs>(g1a, g1b, g2aa, g2ab, g2bb);
    else
        rdms = std::make_shared<SD_RDMs>(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb);

    return rdms;
}

void SD_RDMs::scale(double factor) {
    if (max_rdm_ > 0) {
        g1a_.scale(factor);
        g1b_.scale(factor);
    }
    if (max_rdm_ > 1) {
        g2aa_.scale(factor);
        g2ab_.scale(factor);
        g2bb_.scale(factor);
    }
    if (max_rdm_ > 2) {
        g3aaa_.scale(factor);
        g3aab_.scale(factor);
        g3abb_.scale(factor);
        g3bbb_.scale(factor);
    }
}

void SD_RDMs::axpy(std::shared_ptr<RDMs_NEW> rhs, double a) {
    if (max_rdm_ != rhs->max_rdm_level())
        throw std::runtime_error("RDMs AXPY Error: Inconsistent RDMs levels!");
    if (type_ != rhs->rdm_type())
        throw std::runtime_error("RDMs AXPY Error: Inconsistent RDMs types!");
    if (n_orbs_ != rhs->dim())
        throw std::runtime_error("RDMs AXPY Error: Inconsistent number of orbitals!");

    if (max_rdm_ > 0) {
        g1a_("pq") += a * rhs->g1a()("pq");
        g1b_("pq") += a * rhs->g1b()("pq");
    }
    if (max_rdm_ > 1) {
        g2aa_("pqrs") += a * rhs->g2aa()("pqrs");
        g2ab_("pqrs") += a * rhs->g2ab()("pqrs");
        g2bb_("pqrs") += a * rhs->g2bb()("pqrs");
    }
    if (max_rdm_ > 2) {
        g3aaa_("pqrstu") += a * rhs->g3aaa()("pqrstu");
        g3aab_("pqrstu") += a * rhs->g3aab()("pqrstu");
        g3abb_("pqrstu") += a * rhs->g3abb()("pqrstu");
        g3bbb_("pqrstu") += a * rhs->g3bbb()("pqrstu");
    }
}

void SD_RDMs::rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) {
    if (max_rdm_ < 1)
        return;

    psi::outfile->Printf("\n  Orbital rotations on spin-dependent RDMs ...");
    timer t("Rotate RDMs");

    // Transform the 1-rdms
    ambit::Tensor g1aT = ambit::Tensor::build(ambit::CoreTensor, "g1aT", g1a_.dims());
    ambit::Tensor g1bT = ambit::Tensor::build(ambit::CoreTensor, "g1bT", g1b_.dims());

    g1aT("pq") = Ua("ap") * g1a_("ab") * Ua("bq");
    g1bT("PQ") = Ub("AP") * g1b_("AB") * Ub("BQ");

    g1a_("pq") = g1aT("pq");
    g1b_("pq") = g1bT("pq");

    psi::outfile->Printf("\n    Transformed 1 RDMs.");

    if (max_rdm_ == 1)
        return;

    // Transform the 2-rdms
    auto g2Taa = ambit::Tensor::build(ambit::CoreTensor, "g2aaT", g2aa_.dims());
    auto g2Tab = ambit::Tensor::build(ambit::CoreTensor, "g2abT", g2ab_.dims());
    auto g2Tbb = ambit::Tensor::build(ambit::CoreTensor, "g2bbT", g2bb_.dims());

    g2Taa("pqrs") = Ua("ap") * Ua("bq") * g2aa_("abcd") * Ua("cr") * Ua("ds");
    g2Tab("pQrS") = Ua("ap") * Ub("BQ") * g2ab_("aBcD") * Ua("cr") * Ub("DS");
    g2Tbb("PQRS") = Ub("AP") * Ub("BQ") * g2bb_("ABCD") * Ub("CR") * Ub("DS");

    g2aa_("pqrs") = g2Taa("pqrs");
    g2ab_("pqrs") = g2Tab("pqrs");
    g2bb_("pqrs") = g2Tbb("pqrs");

    psi::outfile->Printf("\n    Transformed 2 RDMs.");

    if (max_rdm_ == 2)
        return;

    // Transform the 3-rdms
    auto g3T = ambit::Tensor::build(ambit::CoreTensor, "g3T", g3aaa_.dims());
    g3T("pqrstu") =
        Ua("ap") * Ua("bq") * Ua("cr") * g3aaa_("abcijk") * Ua("is") * Ua("jt") * Ua("ku");
    g3aaa_("pqrstu") = g3T("pqrstu");

    g3T("pqRstU") =
        Ua("ap") * Ua("bq") * Ub("CR") * g3aab_("abCijK") * Ua("is") * Ua("jt") * Ub("KU");
    g3aab_("pqrstu") = g3T("pqrstu");

    g3T("pQRsTU") =
        Ua("ap") * Ub("BQ") * Ub("CR") * g3abb_("aBCiJK") * Ua("is") * Ub("JT") * Ub("KU");
    g3abb_("pqrstu") = g3T("pqrstu");

    g3T("PQRSTU") =
        Ub("AP") * Ub("BQ") * Ub("CR") * g3bbb_("ABCIJK") * Ub("IS") * Ub("JT") * Ub("KU");
    g3bbb_("pqrstu") = g3T("pqrstu");

    psi::outfile->Printf("\n    Transformed 3 RDMs.");
}

SF_RDMs::SF_RDMs() {
    max_rdm_ = 0;
    type_ = spin_free;
    n_orbs_ = 0;
}

SF_RDMs::SF_RDMs(ambit::Tensor G1) : SF_G1_(G1) {
    max_rdm_ = 1;
    type_ = spin_free;
    n_orbs_ = G1.dim(0);
    _test_rdm_dims(G1, "G1");
}

SF_RDMs::SF_RDMs(ambit::Tensor G1, ambit::Tensor G2) : SF_G1_(G1), SF_G2_(G2) {
    max_rdm_ = 2;
    type_ = spin_free;
    n_orbs_ = G1.dim(0);
    _test_rdm_dims(G1, "G1");
    _test_rdm_dims(G2, "G2");
}

SF_RDMs::SF_RDMs(ambit::Tensor G1, ambit::Tensor G2, ambit::Tensor G3)
    : SF_G1_(G1), SF_G2_(G2), SF_G3_(G3) {
    max_rdm_ = 3;
    type_ = spin_free;
    n_orbs_ = G1.dim(0);
    _test_rdm_dims(G1, "G1");
    _test_rdm_dims(G2, "G2");
    _test_rdm_dims(G3, "G3");
}

ambit::Tensor SF_RDMs::SF_G1() const {
    _test_rdm_level(1, "SF_G1");
    return SF_G1_;
}
ambit::Tensor SF_RDMs::SF_G2() const {
    _test_rdm_level(2, "SF_G2");
    return SF_G2_;
}
ambit::Tensor SF_RDMs::SF_G3() const {
    _test_rdm_level(3, "SF_G3");
    return SF_G3_;
}
ambit::Tensor SF_RDMs::g1a() const {
    _test_rdm_level(1, "g1a");
    auto g1a = sf1_to_sd1(SF_G1_);
    g1a.set_name("g1a");
    return g1a;
}
ambit::Tensor SF_RDMs::g1b() const {
    _test_rdm_level(1, "g1b");
    auto g1b = sf1_to_sd1(SF_G1_);
    g1b.set_name("g1b");
    return g1b;
}
ambit::Tensor SF_RDMs::g2aa() const {
    _test_rdm_level(2, "g2aa");
    auto g2aa = sf2_to_sd2aa(SF_G2_);
    g2aa.set_name("g2aa");
    return g2aa;
}
ambit::Tensor SF_RDMs::g2ab() const {
    _test_rdm_level(2, "g2ab");
    auto g2ab = sf2_to_sd2ab(SF_G2_);
    g2ab.set_name("g2ab");
    return g2ab;
}
ambit::Tensor SF_RDMs::g2bb() const {
    _test_rdm_level(2, "g2bb");
    auto g2bb = sf2_to_sd2aa(SF_G2_);
    g2bb.set_name("g2bb");
    return g2bb;
}
ambit::Tensor SF_RDMs::g3aaa() const {
    _test_rdm_level(3, "g3aaa");
    auto g3aaa = sf3_to_sd3aaa(SF_G3_);
    g3aaa.set_name("g3aaa");
    return g3aaa;
}
ambit::Tensor SF_RDMs::g3aab() const {
    _test_rdm_level(3, "g3aab");
    auto g3aab = sf3_to_sd3aab(SF_G3_);
    g3aab.set_name("g3aab");
    return g3aab;
}
ambit::Tensor SF_RDMs::g3abb() const {
    _test_rdm_level(3, "g3abb");
    auto g3abb = sf3_to_sd3abb(SF_G3_);
    g3abb.set_name("g3abb");
    return g3abb;
}
ambit::Tensor SF_RDMs::g3bbb() const {
    _test_rdm_level(3, "g3bbb");
    auto g3bbb = sf3_to_sd3aaa(SF_G3_);
    g3bbb.set_name("g3bbb");
    return g3bbb;
}
ambit::Tensor SF_RDMs::L1a() const {
    _test_rdm_level(1, "L1a");
    auto L1a = sf1_to_sd1(SF_G1_);
    L1a.set_name("L1a");
    return L1a;
}
ambit::Tensor SF_RDMs::L1b() const {
    _test_rdm_level(1, "L1b");
    auto L1b = sf1_to_sd1(SF_G1_);
    L1b.set_name("L1b");
    return L1b;
}
ambit::Tensor SF_RDMs::L2aa() const {
    _test_rdm_level(2, "L2aa");
    auto L2aa = sf2_to_sd2aa(SF_L2());
    L2aa.set_name("L2aa");
    return L2aa;
}
ambit::Tensor SF_RDMs::L2ab() const {
    _test_rdm_level(2, "L2ab");
    auto L2ab = sf2_to_sd2ab(SF_L2());
    L2ab.set_name("L2ab");
    return L2ab;
}
ambit::Tensor SF_RDMs::L2bb() const {
    _test_rdm_level(2, "L2bb");
    auto L2bb = sf2_to_sd2aa(SF_L2());
    L2bb.set_name("L2bb");
    return L2bb;
}
ambit::Tensor SF_RDMs::L3aaa() const {
    _test_rdm_level(3, "L3aaa");
    auto L3aaa = sf3_to_sd3aaa(SF_L3());
    L3aaa.set_name("L3aaa");
    return L3aaa;
}
ambit::Tensor SF_RDMs::L3aab() const {
    _test_rdm_level(3, "L3aab");
    auto L3aab = sf3_to_sd3aab(SF_L3());
    L3aab.set_name("L3aab");
    return L3aab;
}
ambit::Tensor SF_RDMs::L3abb() const {
    _test_rdm_level(3, "L3abb");
    auto L3abb = sf3_to_sd3abb(SF_L3());
    L3abb.set_name("L3abb");
    return L3abb;
}
ambit::Tensor SF_RDMs::L3bbb() const {
    _test_rdm_level(3, "L3bbb");
    auto L3bbb = sf3_to_sd3aaa(SF_L3());
    L3bbb.set_name("L3bbb");
    return L3bbb;
}

std::shared_ptr<RDMs_NEW> SF_RDMs::clone() {
    ambit::Tensor g1, g2, g3;
    if (max_rdm_ > 0)
        g1 = SF_G1_.clone();
    if (max_rdm_ > 1)
        g2 = SF_G2_.clone();
    if (max_rdm_ > 2)
        g3 = SF_G3_.clone();

    std::shared_ptr<RDMs_NEW> rdms;

    if (max_rdm_ < 1)
        rdms = std::make_shared<SF_RDMs>();
    else if (max_rdm_ == 1)
        rdms = std::make_shared<SF_RDMs>(g1);
    else if (max_rdm_ == 2)
        rdms = std::make_shared<SF_RDMs>(g1, g2);
    else
        rdms = std::make_shared<SF_RDMs>(g1, g2, g3);

    return rdms;
}

void SF_RDMs::scale(double factor) {
    if (max_rdm_ > 0)
        SF_G1_.scale(factor);
    if (max_rdm_ > 1)
        SF_G2_.scale(factor);
    if (max_rdm_ > 2)
        SF_G3_.scale(factor);
}

void SF_RDMs::axpy(std::shared_ptr<RDMs_NEW> rhs, double a) {
    if (max_rdm_ != rhs->max_rdm_level())
        throw std::runtime_error("RDMs AXPY Error: Inconsistent RDMs levels!");
    if (type_ != rhs->rdm_type())
        throw std::runtime_error("RDMs AXPY Error: Inconsistent RDMs types!");
    if (n_orbs_ != rhs->dim())
        throw std::runtime_error("RDMs AXPY Error: Inconsistent number of orbitals!");

    if (max_rdm_ > 0)
        SF_G1_("pq") += a * rhs->SF_G1()("pq");
    if (max_rdm_ > 1)
        SF_G2_("pqrs") += a * rhs->SF_G2()("pqrs");
    if (max_rdm_ > 2)
        SF_G3_("pqrstu") += a * rhs->SF_G3()("pqrstu");
}

void SF_RDMs::rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) {
    if (max_rdm_ < 1)
        return;

    psi::outfile->Printf("\n  Orbital rotations on spin-free RDMs ...");

    // Test if Ua and Ub are the same
    auto dU = Ub.clone();
    dU("pq") -= Ua("pq");
    if (dU.norm(0) > 1.0e-12)
        throw std::runtime_error("Ua != Ub, spin-free RDMs assume restricted formalism!");

    timer t("Rotate RDMs");

    // Transform the 1-rdms
    ambit::Tensor g1T = ambit::Tensor::build(ambit::CoreTensor, "g1T", SF_G1_.dims());
    g1T("pq") = Ua("ap") * SF_G1_("ab") * Ua("bq");
    SF_G1_("pq") = g1T("pq");
    psi::outfile->Printf("\n    Transformed 1 RDMs.");
    if (max_rdm_ == 1)
        return;

    // Transform the 2-rdms
    auto g2T = ambit::Tensor::build(ambit::CoreTensor, "g2T", SF_G2_.dims());
    g2T("pqrs") = Ua("ap") * Ua("bq") * SF_G2_("abcd") * Ua("cr") * Ua("ds");
    SF_G2_("pqrs") = g2T("pqrs");
    psi::outfile->Printf("\n    Transformed 2 RDMs.");
    if (max_rdm_ == 2)
        return;

    // Transform the 3-rdms
    auto g3T = ambit::Tensor::build(ambit::CoreTensor, "g3T", SF_G3_.dims());
    g3T("pqrstu") =
        Ua("ap") * Ua("bq") * Ua("cr") * SF_G3_("abcijk") * Ua("is") * Ua("jt") * Ua("ku");
    SF_G3_("pqrstu") = g3T("pqrstu");
    psi::outfile->Printf("\n    Transformed 3 RDMs.");
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
