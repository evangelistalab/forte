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

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/helpers.h"
#include "helpers/timer.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "base_classes/mo_space_info.h"

namespace forte {

std::shared_ptr<RDMs> RDMs::build(size_t max_rdm_level, size_t n_orbs, RDMsType type) {
    std::vector<size_t> dims1(2, n_orbs);
    std::vector<size_t> dims2(4, n_orbs);
    std::vector<size_t> dims3(6, n_orbs);
    std::vector<size_t> dims4(8, n_orbs);

    std::shared_ptr<RDMs> rdms;

    if (type == RDMsType::spin_dependent) {
        ambit::Tensor g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb, g4aaaa, g4aaab,
            g4aabb, g4abbb, g4bbbb;
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
        if (max_rdm_level > 3) {
            g4aaaa = ambit::Tensor::build(ambit::CoreTensor, "g4aaaa", dims4);
            g4aaab = ambit::Tensor::build(ambit::CoreTensor, "g4aaab", dims4);
            g4aabb = ambit::Tensor::build(ambit::CoreTensor, "g4aabb", dims4);
            g4abbb = ambit::Tensor::build(ambit::CoreTensor, "g4abbb", dims4);
            g4bbbb = ambit::Tensor::build(ambit::CoreTensor, "g4bbbb", dims4);
        }

        if (max_rdm_level < 1) {
            rdms = std::make_shared<RDMsSpinDependent>();
        } else if (max_rdm_level == 1) {
            rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b);
        } else if (max_rdm_level == 2) {
            rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb);
        } else if (max_rdm_level == 3) {
            rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab,
                                                       g3abb, g3bbb);
        } else {
            rdms =
                std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb,
                                                    g3bbb, g4aaaa, g4aaab, g4aabb, g4abbb, g4bbbb);
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
            rdms = std::make_shared<RDMsSpinFree>();
        } else if (max_rdm_level == 1) {
            rdms = std::make_shared<RDMsSpinFree>(g1);
        } else if (max_rdm_level == 2) {
            rdms = std::make_shared<RDMsSpinFree>(g1, g2);
        } else if (max_rdm_level == 3) {
            rdms = std::make_shared<RDMsSpinFree>(g1, g2, g3);
        } else {
            throw std::runtime_error(
                "RDMs::build: max_rdm_level > 3 not implemented for spin-free RDMs.");
        }
    }
    return rdms;
}

std::shared_ptr<RDMs> RDMs::build_from_disk(size_t max_rdm_level, RDMsType type,
                                            const std::string& filename_prefix) {
    std::shared_ptr<RDMs> rdms;

    std::string prefix = (filename_prefix.empty() ? "" : filename_prefix + ".");

    if (type == RDMsType::spin_dependent) {
        ambit::Tensor g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb;
        if (max_rdm_level > 0) {
            g1a = ambit::load_tensor(prefix + "g1a.bin");
            g1b = ambit::load_tensor(prefix + "g1b.bin");
        }
        if (max_rdm_level > 1) {
            g2aa = ambit::load_tensor(prefix + "g2aa.bin");
            g2ab = ambit::load_tensor(prefix + "g2ab.bin");
            g2bb = ambit::load_tensor(prefix + "g2bb.bin");
        }
        if (max_rdm_level > 2) {
            g3aaa = ambit::load_tensor(prefix + "g3aaa.bin");
            g3aab = ambit::load_tensor(prefix + "g3aab.bin");
            g3abb = ambit::load_tensor(prefix + "g3abb.bin");
            g3bbb = ambit::load_tensor(prefix + "g3bbb.bin");
        }

        if (max_rdm_level < 1) {
            rdms = std::make_shared<RDMsSpinDependent>();
        } else if (max_rdm_level == 1) {
            rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b);
        } else if (max_rdm_level == 2) {
            rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb);
        } else {
            rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab,
                                                       g3abb, g3bbb);
        }
    } else {
        ambit::Tensor g1, g2, g3;
        if (max_rdm_level > 0) {
            g1 = ambit::load_tensor(prefix + "g1.bin");
        }
        if (max_rdm_level > 1) {
            g2 = ambit::load_tensor(prefix + "g2.bin");
        }
        if (max_rdm_level > 2) {
            g3 = ambit::load_tensor(prefix + "g3.bin");
        }

        if (max_rdm_level < 1) {
            rdms = std::make_shared<RDMsSpinFree>();
        } else if (max_rdm_level == 1) {
            rdms = std::make_shared<RDMsSpinFree>(g1);
        } else if (max_rdm_level == 2) {
            rdms = std::make_shared<RDMsSpinFree>(g1, g2);
        } else {
            rdms = std::make_shared<RDMsSpinFree>(g1, g2, g3);
        }
    }

    return rdms;
}

void RDMs::save_SF_G1(const std::string& filename) {
    auto d1 = SF_G1mat();
    d1->save(filename, false, false, true);
}

std::shared_ptr<psi::Matrix> RDMs::SF_G1mat() {
    auto G1 = SF_G1();
    auto M = tensor_to_matrix(G1);
    M->set_name("1-RDM NoSym");
    return M;
}

std::shared_ptr<psi::Matrix> RDMs::SF_G1mat(const psi::Dimension& dim) {
    auto G1 = SF_G1();
    auto M = tensor_to_matrix(G1, dim);
    M->set_name("1-RDM");
    return M;
}

ambit::Tensor RDMs::SF_L1() const {
    auto L1 = SF_G1().clone();
    L1.set_name("SF_L1");
    return L1;
}

ambit::Tensor RDMs::SF_L2() const {
    _test_rdm_level(2, "L2");
    timer t("make_cumulant_L2");
    auto G1 = SF_G1();
    auto L2 = SF_G2().clone();
    L2("pqrs") -= G1("pr") * G1("qs");
    L2("pqrs") += 0.5 * G1("ps") * G1("qr");
    L2.set_name("SF_L2");
    return L2;
}

ambit::Tensor RDMs::SF_L3() const {
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

ambit::Tensor RDMs::make_cumulant_L2aa(const ambit::Tensor& g1a, const ambit::Tensor& g2aa) {
    timer t("make_cumulant_L2aa");
    auto L2aa = g2aa.clone();
    L2aa("pqrs") -= g1a("pr") * g1a("qs");
    L2aa("pqrs") += g1a("ps") * g1a("qr");
    return L2aa;
}

ambit::Tensor RDMs::make_cumulant_L2ab(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                       const ambit::Tensor& g2ab) {
    timer t("make_cumulant_L2ab");
    auto L2ab = g2ab.clone();
    L2ab("pqrs") -= g1a("pr") * g1b("qs");
    return L2ab;
}

ambit::Tensor RDMs::make_cumulant_L3aaa(const ambit::Tensor& g1a, const ambit::Tensor& g2aa,
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

ambit::Tensor RDMs::make_cumulant_L3aab(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
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

ambit::Tensor RDMs::make_cumulant_L3abb(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
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

ambit::Tensor RDMs::make_cumulant_L4aaaa(const ambit::Tensor& L1a, const ambit::Tensor& L2aa,
                                         const ambit::Tensor& L3aaa, const ambit::Tensor& g4aaaa) {
    timer t("make_cumulant_L4aaaa");

    auto L4aaaa = g4aaaa.clone();
    /// 1111 contributions (24 terms)
    L4aaaa("pqrstuvw") -= L1a("pw") * L1a("qv") * L1a("ru") * L1a("st");
    L4aaaa("pqrstuvw") += L1a("pw") * L1a("qv") * L1a("rt") * L1a("su");
    L4aaaa("pqrstuvw") += L1a("pw") * L1a("qu") * L1a("rv") * L1a("st");
    L4aaaa("pqrstuvw") -= L1a("pw") * L1a("qu") * L1a("rt") * L1a("sv");
    L4aaaa("pqrstuvw") -= L1a("pw") * L1a("qt") * L1a("rv") * L1a("su");
    L4aaaa("pqrstuvw") += L1a("pw") * L1a("qt") * L1a("ru") * L1a("sv");
    L4aaaa("pqrstuvw") += L1a("pv") * L1a("qw") * L1a("ru") * L1a("st");
    L4aaaa("pqrstuvw") -= L1a("pv") * L1a("qw") * L1a("rt") * L1a("su");
    L4aaaa("pqrstuvw") -= L1a("pv") * L1a("qu") * L1a("rw") * L1a("st");
    L4aaaa("pqrstuvw") += L1a("pv") * L1a("qu") * L1a("rt") * L1a("sw");
    L4aaaa("pqrstuvw") += L1a("pv") * L1a("qt") * L1a("rw") * L1a("su");
    L4aaaa("pqrstuvw") -= L1a("pv") * L1a("qt") * L1a("ru") * L1a("sw");
    L4aaaa("pqrstuvw") -= L1a("pu") * L1a("qw") * L1a("rv") * L1a("st");
    L4aaaa("pqrstuvw") += L1a("pu") * L1a("qw") * L1a("rt") * L1a("sv");
    L4aaaa("pqrstuvw") += L1a("pu") * L1a("qv") * L1a("rw") * L1a("st");
    L4aaaa("pqrstuvw") -= L1a("pu") * L1a("qv") * L1a("rt") * L1a("sw");
    L4aaaa("pqrstuvw") -= L1a("pu") * L1a("qt") * L1a("rw") * L1a("sv");
    L4aaaa("pqrstuvw") += L1a("pu") * L1a("qt") * L1a("rv") * L1a("sw");
    L4aaaa("pqrstuvw") += L1a("pt") * L1a("qw") * L1a("rv") * L1a("su");
    L4aaaa("pqrstuvw") -= L1a("pt") * L1a("qw") * L1a("ru") * L1a("sv");
    L4aaaa("pqrstuvw") -= L1a("pt") * L1a("qv") * L1a("rw") * L1a("su");
    L4aaaa("pqrstuvw") += L1a("pt") * L1a("qv") * L1a("ru") * L1a("sw");
    L4aaaa("pqrstuvw") += L1a("pt") * L1a("qu") * L1a("rw") * L1a("sv");
    L4aaaa("pqrstuvw") -= L1a("pt") * L1a("qu") * L1a("rv") * L1a("sw");

    /// 211 contributions (72 terms)
    L4aaaa("pqrstuvw") += L2aa("pqvw") * L1a("ru") * L1a("st");
    L4aaaa("pqrstuvw") -= L2aa("pqvw") * L1a("rt") * L1a("su");
    L4aaaa("pqrstuvw") -= L2aa("pquw") * L1a("rv") * L1a("st");
    L4aaaa("pqrstuvw") += L2aa("pquw") * L1a("rt") * L1a("sv");
    L4aaaa("pqrstuvw") += L2aa("pqtw") * L1a("rv") * L1a("su");
    L4aaaa("pqrstuvw") -= L2aa("pqtw") * L1a("ru") * L1a("sv");
    L4aaaa("pqrstuvw") += L2aa("pquv") * L1a("rw") * L1a("st");
    L4aaaa("pqrstuvw") -= L2aa("pquv") * L1a("rt") * L1a("sw");
    L4aaaa("pqrstuvw") -= L2aa("pqtv") * L1a("rw") * L1a("su");
    L4aaaa("pqrstuvw") += L2aa("pqtv") * L1a("ru") * L1a("sw");
    L4aaaa("pqrstuvw") += L2aa("pqtu") * L1a("rw") * L1a("sv");
    L4aaaa("pqrstuvw") -= L2aa("pqtu") * L1a("rv") * L1a("sw");
    L4aaaa("pqrstuvw") -= L2aa("prvw") * L1a("qu") * L1a("st");
    L4aaaa("pqrstuvw") += L2aa("prvw") * L1a("qt") * L1a("su");
    L4aaaa("pqrstuvw") += L2aa("pruw") * L1a("qv") * L1a("st");
    L4aaaa("pqrstuvw") -= L2aa("pruw") * L1a("qt") * L1a("sv");
    L4aaaa("pqrstuvw") -= L2aa("prtw") * L1a("qv") * L1a("su");
    L4aaaa("pqrstuvw") += L2aa("prtw") * L1a("qu") * L1a("sv");
    L4aaaa("pqrstuvw") -= L2aa("pruv") * L1a("qw") * L1a("st");
    L4aaaa("pqrstuvw") += L2aa("pruv") * L1a("qt") * L1a("sw");
    L4aaaa("pqrstuvw") += L2aa("prtv") * L1a("qw") * L1a("su");
    L4aaaa("pqrstuvw") -= L2aa("prtv") * L1a("qu") * L1a("sw");
    L4aaaa("pqrstuvw") -= L2aa("prtu") * L1a("qw") * L1a("sv");
    L4aaaa("pqrstuvw") += L2aa("prtu") * L1a("qv") * L1a("sw");
    L4aaaa("pqrstuvw") += L2aa("psvw") * L1a("qu") * L1a("rt");
    L4aaaa("pqrstuvw") -= L2aa("psvw") * L1a("qt") * L1a("ru");
    L4aaaa("pqrstuvw") -= L2aa("psuw") * L1a("qv") * L1a("rt");
    L4aaaa("pqrstuvw") += L2aa("psuw") * L1a("qt") * L1a("rv");
    L4aaaa("pqrstuvw") += L2aa("pstw") * L1a("qv") * L1a("ru");
    L4aaaa("pqrstuvw") -= L2aa("pstw") * L1a("qu") * L1a("rv");
    L4aaaa("pqrstuvw") += L2aa("psuv") * L1a("qw") * L1a("rt");
    L4aaaa("pqrstuvw") -= L2aa("psuv") * L1a("qt") * L1a("rw");
    L4aaaa("pqrstuvw") -= L2aa("pstv") * L1a("qw") * L1a("ru");
    L4aaaa("pqrstuvw") += L2aa("pstv") * L1a("qu") * L1a("rw");
    L4aaaa("pqrstuvw") += L2aa("pstu") * L1a("qw") * L1a("rv");
    L4aaaa("pqrstuvw") -= L2aa("pstu") * L1a("qv") * L1a("rw");
    L4aaaa("pqrstuvw") += L2aa("qrvw") * L1a("pu") * L1a("st");
    L4aaaa("pqrstuvw") -= L2aa("qrvw") * L1a("pt") * L1a("su");
    L4aaaa("pqrstuvw") -= L2aa("qruw") * L1a("pv") * L1a("st");
    L4aaaa("pqrstuvw") += L2aa("qruw") * L1a("pt") * L1a("sv");
    L4aaaa("pqrstuvw") += L2aa("qrtw") * L1a("pv") * L1a("su");
    L4aaaa("pqrstuvw") -= L2aa("qrtw") * L1a("pu") * L1a("sv");
    L4aaaa("pqrstuvw") += L2aa("qruv") * L1a("pw") * L1a("st");
    L4aaaa("pqrstuvw") -= L2aa("qruv") * L1a("pt") * L1a("sw");
    L4aaaa("pqrstuvw") -= L2aa("qrtv") * L1a("pw") * L1a("su");
    L4aaaa("pqrstuvw") += L2aa("qrtv") * L1a("pu") * L1a("sw");
    L4aaaa("pqrstuvw") += L2aa("qrtu") * L1a("pw") * L1a("sv");
    L4aaaa("pqrstuvw") -= L2aa("qrtu") * L1a("pv") * L1a("sw");
    L4aaaa("pqrstuvw") -= L2aa("qsvw") * L1a("pu") * L1a("rt");
    L4aaaa("pqrstuvw") += L2aa("qsvw") * L1a("pt") * L1a("ru");
    L4aaaa("pqrstuvw") += L2aa("qsuw") * L1a("pv") * L1a("rt");
    L4aaaa("pqrstuvw") -= L2aa("qsuw") * L1a("pt") * L1a("rv");
    L4aaaa("pqrstuvw") -= L2aa("qstw") * L1a("pv") * L1a("ru");
    L4aaaa("pqrstuvw") += L2aa("qstw") * L1a("pu") * L1a("rv");
    L4aaaa("pqrstuvw") -= L2aa("qsuv") * L1a("pw") * L1a("rt");
    L4aaaa("pqrstuvw") += L2aa("qsuv") * L1a("pt") * L1a("rw");
    L4aaaa("pqrstuvw") += L2aa("qstv") * L1a("pw") * L1a("ru");
    L4aaaa("pqrstuvw") -= L2aa("qstv") * L1a("pu") * L1a("rw");
    L4aaaa("pqrstuvw") -= L2aa("qstu") * L1a("pw") * L1a("rv");
    L4aaaa("pqrstuvw") += L2aa("qstu") * L1a("pv") * L1a("rw");
    L4aaaa("pqrstuvw") += L2aa("rsvw") * L1a("pu") * L1a("qt");
    L4aaaa("pqrstuvw") -= L2aa("rsvw") * L1a("pt") * L1a("qu");
    L4aaaa("pqrstuvw") -= L2aa("rsuw") * L1a("pv") * L1a("qt");
    L4aaaa("pqrstuvw") += L2aa("rsuw") * L1a("pt") * L1a("qv");
    L4aaaa("pqrstuvw") += L2aa("rstw") * L1a("pv") * L1a("qu");
    L4aaaa("pqrstuvw") -= L2aa("rstw") * L1a("pu") * L1a("qv");
    L4aaaa("pqrstuvw") += L2aa("rsuv") * L1a("pw") * L1a("qt");
    L4aaaa("pqrstuvw") -= L2aa("rsuv") * L1a("pt") * L1a("qw");
    L4aaaa("pqrstuvw") -= L2aa("rstv") * L1a("pw") * L1a("qu");
    L4aaaa("pqrstuvw") += L2aa("rstv") * L1a("pu") * L1a("qw");
    L4aaaa("pqrstuvw") += L2aa("rstu") * L1a("pw") * L1a("qv");
    L4aaaa("pqrstuvw") -= L2aa("rstu") * L1a("pv") * L1a("qw");

    /// 22 contributions (18 terms)
    L4aaaa("pqrstuvw") -= L2aa("pqvw") * L2aa("rstu");
    L4aaaa("pqrstuvw") += L2aa("pquw") * L2aa("rstv");
    L4aaaa("pqrstuvw") -= L2aa("pqtw") * L2aa("rsuv");
    L4aaaa("pqrstuvw") -= L2aa("pquv") * L2aa("rstw");
    L4aaaa("pqrstuvw") += L2aa("pqtv") * L2aa("rsuw");
    L4aaaa("pqrstuvw") -= L2aa("pqtu") * L2aa("rsvw");
    L4aaaa("pqrstuvw") += L2aa("prvw") * L2aa("qstu");
    L4aaaa("pqrstuvw") -= L2aa("pruw") * L2aa("qstv");
    L4aaaa("pqrstuvw") += L2aa("prtw") * L2aa("qsuv");
    L4aaaa("pqrstuvw") += L2aa("pruv") * L2aa("qstw");
    L4aaaa("pqrstuvw") -= L2aa("prtv") * L2aa("qsuw");
    L4aaaa("pqrstuvw") += L2aa("prtu") * L2aa("qsvw");
    L4aaaa("pqrstuvw") -= L2aa("psvw") * L2aa("qrtu");
    L4aaaa("pqrstuvw") += L2aa("psuw") * L2aa("qrtv");
    L4aaaa("pqrstuvw") -= L2aa("pstw") * L2aa("qruv");
    L4aaaa("pqrstuvw") -= L2aa("psuv") * L2aa("qrtw");
    L4aaaa("pqrstuvw") += L2aa("pstv") * L2aa("qruw");
    L4aaaa("pqrstuvw") -= L2aa("pstu") * L2aa("qrvw");

    /// 31 contributions (16 terms)
    L4aaaa("pqrstuvw") += L3aaa("pqruvw") * L1a("st");
    L4aaaa("pqrstuvw") -= L3aaa("pqrtvw") * L1a("su");
    L4aaaa("pqrstuvw") += L3aaa("pqrtuw") * L1a("sv");
    L4aaaa("pqrstuvw") -= L3aaa("pqrtuv") * L1a("sw");
    L4aaaa("pqrstuvw") -= L3aaa("pqsuvw") * L1a("rt");
    L4aaaa("pqrstuvw") += L3aaa("pqstvw") * L1a("ru");
    L4aaaa("pqrstuvw") -= L3aaa("pqstuw") * L1a("rv");
    L4aaaa("pqrstuvw") += L3aaa("pqstuv") * L1a("rw");
    L4aaaa("pqrstuvw") += L3aaa("prsuvw") * L1a("qt");
    L4aaaa("pqrstuvw") -= L3aaa("prstvw") * L1a("qu");
    L4aaaa("pqrstuvw") += L3aaa("prstuw") * L1a("qv");
    L4aaaa("pqrstuvw") -= L3aaa("prstuv") * L1a("qw");
    L4aaaa("pqrstuvw") -= L3aaa("qrsuvw") * L1a("pt");
    L4aaaa("pqrstuvw") += L3aaa("qrstvw") * L1a("pu");
    L4aaaa("pqrstuvw") -= L3aaa("qrstuw") * L1a("pv");
    L4aaaa("pqrstuvw") += L3aaa("qrstuv") * L1a("pw");

    return L4aaaa;
}

ambit::Tensor RDMs::make_cumulant_L4aaab(const ambit::Tensor& L1a, const ambit::Tensor& L1b,
                                         const ambit::Tensor& L2aa, const ambit::Tensor& L2ab,
                                         const ambit::Tensor& L3aaa, const ambit::Tensor& L3aab,
                                         const ambit::Tensor& g4aaab) {
    timer t("make_cumulant_L4aaab");

    auto L4aaab = g4aaab.clone();
    /// 1111 contributions
    L4aaab("pqrStuvW") += L1a("pv") * L1a("qu") * L1a("rt") * L1b("SW");
    L4aaab("pqrStuvW") -= L1a("pv") * L1a("qt") * L1a("ru") * L1b("SW");
    L4aaab("pqrStuvW") -= L1a("pu") * L1a("qv") * L1a("rt") * L1b("SW");
    L4aaab("pqrStuvW") += L1a("pu") * L1a("qt") * L1a("rv") * L1b("SW");
    L4aaab("pqrStuvW") += L1a("pt") * L1a("qv") * L1a("ru") * L1b("SW");
    L4aaab("pqrStuvW") -= L1a("pt") * L1a("qu") * L1a("rv") * L1b("SW");

    /// 211 contributions
    L4aaab("pqrStuvW") -= L2aa("pquv") * L1a("rt") * L1b("SW");
    L4aaab("pqrStuvW") += L2aa("pqtv") * L1a("ru") * L1b("SW");
    L4aaab("pqrStuvW") -= L2aa("pqtu") * L1a("rv") * L1b("SW");
    L4aaab("pqrStuvW") += L2aa("pruv") * L1a("qt") * L1b("SW");
    L4aaab("pqrStuvW") -= L2aa("prtv") * L1a("qu") * L1b("SW");
    L4aaab("pqrStuvW") += L2aa("prtu") * L1a("qv") * L1b("SW");
    L4aaab("pqrStuvW") += L2ab("pSvW") * L1a("qu") * L1a("rt");
    L4aaab("pqrStuvW") -= L2ab("pSvW") * L1a("qt") * L1a("ru");
    L4aaab("pqrStuvW") -= L2ab("pSuW") * L1a("qv") * L1a("rt");
    L4aaab("pqrStuvW") += L2ab("pSuW") * L1a("qt") * L1a("rv");
    L4aaab("pqrStuvW") += L2ab("pStW") * L1a("qv") * L1a("ru");
    L4aaab("pqrStuvW") -= L2ab("pStW") * L1a("qu") * L1a("rv");
    L4aaab("pqrStuvW") -= L2aa("qruv") * L1a("pt") * L1b("SW");
    L4aaab("pqrStuvW") += L2aa("qrtv") * L1a("pu") * L1b("SW");
    L4aaab("pqrStuvW") -= L2aa("qrtu") * L1a("pv") * L1b("SW");
    L4aaab("pqrStuvW") -= L2ab("qSvW") * L1a("pu") * L1a("rt");
    L4aaab("pqrStuvW") += L2ab("qSvW") * L1a("pt") * L1a("ru");
    L4aaab("pqrStuvW") += L2ab("qSuW") * L1a("pv") * L1a("rt");
    L4aaab("pqrStuvW") -= L2ab("qSuW") * L1a("pt") * L1a("rv");
    L4aaab("pqrStuvW") -= L2ab("qStW") * L1a("pv") * L1a("ru");
    L4aaab("pqrStuvW") += L2ab("qStW") * L1a("pu") * L1a("rv");
    L4aaab("pqrStuvW") += L2ab("rSvW") * L1a("pu") * L1a("qt");
    L4aaab("pqrStuvW") -= L2ab("rSvW") * L1a("pt") * L1a("qu");
    L4aaab("pqrStuvW") -= L2ab("rSuW") * L1a("pv") * L1a("qt");
    L4aaab("pqrStuvW") += L2ab("rSuW") * L1a("pt") * L1a("qv");
    L4aaab("pqrStuvW") += L2ab("rStW") * L1a("pv") * L1a("qu");
    L4aaab("pqrStuvW") -= L2ab("rStW") * L1a("pu") * L1a("qv");

    /// 22 contributions
    L4aaab("pqrStuvW") -= L2aa("pquv") * L2ab("rStW");
    L4aaab("pqrStuvW") += L2aa("pqtv") * L2ab("rSuW");
    L4aaab("pqrStuvW") -= L2aa("pqtu") * L2ab("rSvW");
    L4aaab("pqrStuvW") += L2aa("pruv") * L2ab("qStW");
    L4aaab("pqrStuvW") -= L2aa("prtv") * L2ab("qSuW");
    L4aaab("pqrStuvW") += L2aa("prtu") * L2ab("qSvW");
    L4aaab("pqrStuvW") -= L2ab("pSvW") * L2aa("qrtu");
    L4aaab("pqrStuvW") += L2ab("pSuW") * L2aa("qrtv");
    L4aaab("pqrStuvW") -= L2ab("pStW") * L2aa("qruv");

    /// 31 contributions (16 terms)
    L4aaab("pqrStuvW") -= L3aaa("pqrtuv") * L1b("SW");
    L4aaab("pqrStuvW") -= L3aab("pqSuvW") * L1a("rt");
    L4aaab("pqrStuvW") += L3aab("pqStvW") * L1a("ru");
    L4aaab("pqrStuvW") -= L3aab("pqStuW") * L1a("rv");
    L4aaab("pqrStuvW") += L3aab("prSuvW") * L1a("qt");
    L4aaab("pqrStuvW") -= L3aab("prStvW") * L1a("qu");
    L4aaab("pqrStuvW") += L3aab("prStuW") * L1a("qv");
    L4aaab("pqrStuvW") -= L3aab("qrSuvW") * L1a("pt");
    L4aaab("pqrStuvW") += L3aab("qrStvW") * L1a("pu");
    L4aaab("pqrStuvW") -= L3aab("qrStuW") * L1a("pv");

    return L4aaab;
}

ambit::Tensor RDMs::make_cumulant_L4aabb(const ambit::Tensor& L1a, const ambit::Tensor& L1b,
                                         const ambit::Tensor& L2aa, const ambit::Tensor& L2ab,
                                         const ambit::Tensor& L2bb, const ambit::Tensor& L3aab,
                                         const ambit::Tensor& L3abb, const ambit::Tensor& g4aabb) {
    timer t("make_cumulant_L4aabb");

    auto L4aabb = g4aabb.clone();
    /// 1111 contributions
    L4aabb("pqRStuVW") -= L1a("pu") * L1a("qt") * L1b("RW") * L1b("SV");
    L4aabb("pqRStuVW") += L1a("pu") * L1a("qt") * L1b("RV") * L1b("SW");
    L4aabb("pqRStuVW") += L1a("pt") * L1a("qu") * L1b("RW") * L1b("SV");
    L4aabb("pqRStuVW") -= L1a("pt") * L1a("qu") * L1b("RV") * L1b("SW");

    /// 211 contributions
    L4aabb("pqRStuVW") += L2aa("pqtu") * L1b("RW") * L1b("SV");
    L4aabb("pqRStuVW") -= L2aa("pqtu") * L1b("RV") * L1b("SW");
    L4aabb("pqRStuVW") -= L2ab("pRuW") * L1a("qt") * L1b("SV");
    L4aabb("pqRStuVW") += L2ab("pRtW") * L1a("qu") * L1b("SV");
    L4aabb("pqRStuVW") += L2ab("pRuV") * L1a("qt") * L1b("SW");
    L4aabb("pqRStuVW") -= L2ab("pRtV") * L1a("qu") * L1b("SW");
    L4aabb("pqRStuVW") += L2ab("pSuW") * L1a("qt") * L1b("RV");
    L4aabb("pqRStuVW") -= L2ab("pStW") * L1a("qu") * L1b("RV");
    L4aabb("pqRStuVW") -= L2ab("pSuV") * L1a("qt") * L1b("RW");
    L4aabb("pqRStuVW") += L2ab("pStV") * L1a("qu") * L1b("RW");
    L4aabb("pqRStuVW") += L2ab("qRuW") * L1a("pt") * L1b("SV");
    L4aabb("pqRStuVW") -= L2ab("qRtW") * L1a("pu") * L1b("SV");
    L4aabb("pqRStuVW") -= L2ab("qRuV") * L1a("pt") * L1b("SW");
    L4aabb("pqRStuVW") += L2ab("qRtV") * L1a("pu") * L1b("SW");
    L4aabb("pqRStuVW") -= L2ab("qSuW") * L1a("pt") * L1b("RV");
    L4aabb("pqRStuVW") += L2ab("qStW") * L1a("pu") * L1b("RV");
    L4aabb("pqRStuVW") += L2ab("qSuV") * L1a("pt") * L1b("RW");
    L4aabb("pqRStuVW") -= L2ab("qStV") * L1a("pu") * L1b("RW");
    L4aabb("pqRStuVW") += L2bb("RSVW") * L1a("pu") * L1a("qt");
    L4aabb("pqRStuVW") -= L2bb("RSVW") * L1a("pt") * L1a("qu");

    /// 22 contributions
    L4aabb("pqRStuVW") -= L2aa("pqtu") * L2bb("RSVW");
    L4aabb("pqRStuVW") -= L2ab("pRuW") * L2ab("qStV");
    L4aabb("pqRStuVW") += L2ab("pRtW") * L2ab("qSuV");
    L4aabb("pqRStuVW") += L2ab("pRuV") * L2ab("qStW");
    L4aabb("pqRStuVW") -= L2ab("pRtV") * L2ab("qSuW");
    L4aabb("pqRStuVW") += L2ab("pSuW") * L2ab("qRtV");
    L4aabb("pqRStuVW") -= L2ab("pStW") * L2ab("qRuV");
    L4aabb("pqRStuVW") -= L2ab("pSuV") * L2ab("qRtW");
    L4aabb("pqRStuVW") += L2ab("pStV") * L2ab("qRuW");

    /// 31 contributions
    L4aabb("pqRStuVW") += L3aab("pqRtuW") * L1b("SV");
    L4aabb("pqRStuVW") -= L3aab("pqRtuV") * L1b("SW");
    L4aabb("pqRStuVW") -= L3aab("pqStuW") * L1b("RV");
    L4aabb("pqRStuVW") += L3aab("pqStuV") * L1b("RW");
    L4aabb("pqRStuVW") += L3abb("pRSuVW") * L1a("qt");
    L4aabb("pqRStuVW") -= L3abb("pRStVW") * L1a("qu");
    L4aabb("pqRStuVW") -= L3abb("qRSuVW") * L1a("pt");
    L4aabb("pqRStuVW") += L3abb("qRStVW") * L1a("pu");

    return L4aabb;
}

ambit::Tensor RDMs::make_cumulant_L4abbb(const ambit::Tensor& L1a, const ambit::Tensor& L1b,
                                         const ambit::Tensor& L2ab, const ambit::Tensor& L2bb,
                                         const ambit::Tensor& L3abb, const ambit::Tensor& L3bbb,
                                         const ambit::Tensor& g4abbb) {
    timer t("make_cumulant_L4abbb");

    auto L4abbb = g4abbb.clone();
    /// 1111 contributions (24 terms)
    L4abbb("pQRStUVW") += L1a("pt") * L1b("QW") * L1b("RV") * L1b("SU");
    L4abbb("pQRStUVW") -= L1a("pt") * L1b("QW") * L1b("RU") * L1b("SV");
    L4abbb("pQRStUVW") -= L1a("pt") * L1b("QV") * L1b("RW") * L1b("SU");
    L4abbb("pQRStUVW") += L1a("pt") * L1b("QV") * L1b("RU") * L1b("SW");
    L4abbb("pQRStUVW") += L1a("pt") * L1b("QU") * L1b("RW") * L1b("SV");
    L4abbb("pQRStUVW") -= L1a("pt") * L1b("QU") * L1b("RV") * L1b("SW");

    /// 211 contributions (72 terms)
    L4abbb("pQRStUVW") += L2ab("pQtW") * L1b("RV") * L1b("SU");
    L4abbb("pQRStUVW") -= L2ab("pQtW") * L1a("RU") * L1b("SV");
    L4abbb("pQRStUVW") -= L2ab("pQtV") * L1b("RW") * L1b("SU");
    L4abbb("pQRStUVW") += L2ab("pQtV") * L1b("RU") * L1b("SW");
    L4abbb("pQRStUVW") += L2ab("pQtU") * L1b("RW") * L1b("SV");
    L4abbb("pQRStUVW") -= L2ab("pQtU") * L1b("RV") * L1b("SW");
    L4abbb("pQRStUVW") -= L2ab("pRtW") * L1b("QV") * L1b("SU");
    L4abbb("pQRStUVW") += L2ab("pRtW") * L1b("QU") * L1b("SV");
    L4abbb("pQRStUVW") += L2ab("pRtV") * L1b("QW") * L1b("SU");
    L4abbb("pQRStUVW") -= L2ab("pRtV") * L1b("QU") * L1b("SW");
    L4abbb("pQRStUVW") -= L2ab("pRtU") * L1b("QW") * L1b("SV");
    L4abbb("pQRStUVW") += L2ab("pRtU") * L1b("QV") * L1b("SW");
    L4abbb("pQRStUVW") += L2ab("pStW") * L1b("QV") * L1b("RU");
    L4abbb("pQRStUVW") -= L2ab("pStW") * L1b("QU") * L1b("RV");
    L4abbb("pQRStUVW") -= L2ab("pStV") * L1b("QW") * L1b("RU");
    L4abbb("pQRStUVW") += L2ab("pStV") * L1b("QU") * L1b("RW");
    L4abbb("pQRStUVW") += L2ab("pStU") * L1b("QW") * L1b("RV");
    L4abbb("pQRStUVW") -= L2ab("pStU") * L1b("QV") * L1b("RW");

    L4abbb("pQRStUVW") -= L2bb("QRVW") * L1a("pt") * L1b("SU");
    L4abbb("pQRStUVW") += L2bb("QRUW") * L1a("pt") * L1b("SV");
    L4abbb("pQRStUVW") -= L2bb("QRUV") * L1a("pt") * L1b("SW");
    L4abbb("pQRStUVW") += L2bb("QSVW") * L1a("pt") * L1b("RU");
    L4abbb("pQRStUVW") -= L2bb("QSUW") * L1a("pt") * L1b("RV");
    L4abbb("pQRStUVW") += L2bb("QSUV") * L1a("pt") * L1b("RW");
    L4abbb("pQRStUVW") -= L2bb("RSVW") * L1a("pt") * L1b("QU");
    L4abbb("pQRStUVW") += L2bb("RSUW") * L1a("pt") * L1b("QV");
    L4abbb("pQRStUVW") -= L2bb("RSUV") * L1a("pt") * L1b("QW");

    /// 22 contributions (18 terms)
    L4abbb("pQRStUVW") -= L2ab("pQtW") * L2bb("RSUV");
    L4abbb("pQRStUVW") += L2ab("pQtV") * L2bb("RSUW");
    L4abbb("pQRStUVW") -= L2ab("pQtU") * L2bb("RSVW");
    L4abbb("pQRStUVW") += L2ab("pRtW") * L2bb("QSUV");
    L4abbb("pQRStUVW") -= L2ab("pRtV") * L2bb("QSUW");
    L4abbb("pQRStUVW") += L2ab("pRtU") * L2bb("QSVW");
    L4abbb("pQRStUVW") -= L2ab("pStW") * L2bb("QRUV");
    L4abbb("pQRStUVW") += L2ab("pStV") * L2bb("QRUW");
    L4abbb("pQRStUVW") -= L2ab("pStU") * L2bb("QRVW");

    /// 31 contributions (16 terms)
    L4abbb("pQRStUVW") -= L3abb("pQRtVW") * L1b("SU");
    L4abbb("pQRStUVW") += L3abb("pQRtUW") * L1b("SV");
    L4abbb("pQRStUVW") -= L3abb("pQRtUV") * L1b("SW");
    L4abbb("pQRStUVW") += L3abb("pQStVW") * L1b("RU");
    L4abbb("pQRStUVW") -= L3abb("pQStUW") * L1b("RV");
    L4abbb("pQRStUVW") += L3abb("pQStUV") * L1b("RW");
    L4abbb("pQRStUVW") -= L3abb("pRStVW") * L1b("QU");
    L4abbb("pQRStUVW") += L3abb("pRStUW") * L1b("QV");
    L4abbb("pQRStUVW") -= L3abb("pRStUV") * L1b("QW");

    L4abbb("pQRStUVW") -= L3bbb("QRSUVW") * L1a("pt");

    return L4abbb;
}

ambit::Tensor RDMs::sf1_to_sd1(const ambit::Tensor& G1) {
    auto g1 = G1.clone();
    g1.scale(0.5);
    return g1;
}

ambit::Tensor RDMs::sf2_to_sd2aa(const ambit::Tensor& G2) {
    auto g2aa = G2.clone();
    g2aa("pqrs") -= G2("pqsr");
    g2aa.scale(1.0 / 6.0);
    return g2aa;
}

ambit::Tensor RDMs::sf2_to_sd2ab(const ambit::Tensor& G2) {
    auto g2ab = G2.clone();
    g2ab.scale(1.0 / 3.0);
    g2ab("pqrs") += (1.0 / 6.0) * G2("pqsr");
    return g2ab;
}

ambit::Tensor RDMs::sf3_to_sd3aaa(const ambit::Tensor& G3) {
    auto g3aaa = G3.clone();
    g3aaa("pqrstu") += G3("pqrtus") + G3("pqrust");
    g3aaa.scale(1.0 / 12.0);
    return g3aaa;
}

ambit::Tensor RDMs::sf3_to_sd3aab(const ambit::Tensor& G3) {
    auto g3aab = G3.clone();
    g3aab("pqrstu") -= G3("pqrtus") + G3("pqrust") + 2.0 * G3("pqrtsu");
    g3aab.scale(1.0 / 12.0);
    return g3aab;
}

ambit::Tensor RDMs::sf3_to_sd3abb(const ambit::Tensor& G3) {
    auto g3abb = G3.clone();
    g3abb("pqrstu") -= G3("pqrtus") + G3("pqrust") + 2.0 * G3("pqrsut");
    g3abb.scale(1.0 / 12.0);
    return g3abb;
}

void RDMs::_test_rdm_level(const size_t& level, const std::string& name) const {
    if (level > max_rdm_) {
        std::string msg = "Impossible to build " + name;
        msg += ": max RDM level is " + std::to_string(max_rdm_);
        throw std::runtime_error(msg);
    }
}

void RDMs::_test_rdm_dims(const ambit::Tensor& T, const std::string& name,
                          size_t desired_dim_size) const {
    const auto& dims = T.dims();
    if (dims.size() != desired_dim_size) {
        throw std::runtime_error("Invalid dimension for " + name);
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

bool RDMs::_bypass_rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub,
                          const double& zero_threshold) const {
    if (max_rdm_ < 1)
        return true;

    auto nactv = Ua.dim(0);
    const auto& Ua_data = Ua.data();
    const auto& Ub_data = Ub.data();

    for (size_t i = 0; i < nactv; ++i) {
        auto va_ii = std::fabs(1.0 - std::fabs(Ua_data[i * nactv + i]));
        auto vb_ii = std::fabs(1.0 - std::fabs(Ub_data[i * nactv + i]));
        if (va_ii > zero_threshold or vb_ii > zero_threshold)
            return false;

        for (size_t j = i + 1; j < nactv; ++j) {
            auto va_ij = std::fabs(Ua_data[i * nactv + j]);
            auto vb_ij = std::fabs(Ub_data[i * nactv + j]);
            if (va_ij > zero_threshold or vb_ij > zero_threshold) {
                return false;
            }
        }
    }

    return true;
}

RDMsSpinDependent::RDMsSpinDependent() {
    max_rdm_ = 0;
    type_ = RDMsType::spin_dependent;
    n_orbs_ = 0;
}

RDMsSpinDependent::RDMsSpinDependent(ambit::Tensor g1a, ambit::Tensor g1b) : g1a_(g1a), g1b_(g1b) {
    max_rdm_ = 1;
    type_ = RDMsType::spin_dependent;
    n_orbs_ = g1a.dim(0);
    _test_rdm_dims(g1a, "g1a", 2);
    _test_rdm_dims(g1b, "g1b", 2);
}

RDMsSpinDependent::RDMsSpinDependent(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa,
                                     ambit::Tensor g2ab, ambit::Tensor g2bb)
    : g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb) {
    max_rdm_ = 2;
    type_ = RDMsType::spin_dependent;
    n_orbs_ = g1a.dim(0);
    _test_rdm_dims(g1a, "g1a", 2);
    _test_rdm_dims(g1b, "g1b", 2);
    _test_rdm_dims(g2aa, "g2aa", 4);
    _test_rdm_dims(g2ab, "g2ab", 4);
    _test_rdm_dims(g2bb, "g2bb", 4);
}

RDMsSpinDependent::RDMsSpinDependent(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa,
                                     ambit::Tensor g2ab, ambit::Tensor g2bb, ambit::Tensor g3aaa,
                                     ambit::Tensor g3aab, ambit::Tensor g3abb, ambit::Tensor g3bbb)
    : g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb), g3aaa_(g3aaa), g3aab_(g3aab),
      g3abb_(g3abb), g3bbb_(g3bbb) {
    max_rdm_ = 3;
    type_ = RDMsType::spin_dependent;
    n_orbs_ = g1a.dim(0);
    _test_rdm_dims(g1a, "g1a", 2);
    _test_rdm_dims(g1b, "g1b", 2);
    _test_rdm_dims(g2aa, "g2aa", 4);
    _test_rdm_dims(g2ab, "g2ab", 4);
    _test_rdm_dims(g2bb, "g2bb", 4);
    _test_rdm_dims(g3aaa, "g3aaa", 6);
    _test_rdm_dims(g3aab, "g3aab", 6);
    _test_rdm_dims(g3abb, "g3abb", 6);
    _test_rdm_dims(g3bbb, "g3bbb", 6);
}

RDMsSpinDependent::RDMsSpinDependent(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa,
                                     ambit::Tensor g2ab, ambit::Tensor g2bb, ambit::Tensor g3aaa,
                                     ambit::Tensor g3aab, ambit::Tensor g3abb, ambit::Tensor g3bbb,
                                     ambit::Tensor g4aaaa, ambit::Tensor g4aaab,
                                     ambit::Tensor g4aabb, ambit::Tensor g4abbb,
                                     ambit::Tensor g4bbbb)
    : g1a_(g1a), g1b_(g1b), g2aa_(g2aa), g2ab_(g2ab), g2bb_(g2bb), g3aaa_(g3aaa), g3aab_(g3aab),
      g3abb_(g3abb), g3bbb_(g3bbb), g4aaaa_(g4aaaa), g4aaab_(g4aaab), g4aabb_(g4aabb),
      g4abbb_(g4abbb), g4bbbb_(g4bbbb) {
    max_rdm_ = 4;
    type_ = RDMsType::spin_dependent;
    n_orbs_ = g1a.dim(0);
    _test_rdm_dims(g1a, "g1a", 2);
    _test_rdm_dims(g1b, "g1b", 2);
    _test_rdm_dims(g2aa, "g2aa", 4);
    _test_rdm_dims(g2ab, "g2ab", 4);
    _test_rdm_dims(g2bb, "g2bb", 4);
    _test_rdm_dims(g3aaa, "g3aaa", 6);
    _test_rdm_dims(g3aab, "g3aab", 6);
    _test_rdm_dims(g3abb, "g3abb", 6);
    _test_rdm_dims(g3bbb, "g3bbb", 6);
    _test_rdm_dims(g4aaaa, "g4aaaa", 8);
    _test_rdm_dims(g4aaab, "g4aaab", 8);
    _test_rdm_dims(g4aabb, "g4aabb", 8);
    _test_rdm_dims(g4abbb, "g4abbb", 8);
    _test_rdm_dims(g4bbbb, "g4bbbb", 8);
}

ambit::Tensor RDMsSpinDependent::g1a() const {
    _test_rdm_level(1, "g1a");
    return g1a_;
}
ambit::Tensor RDMsSpinDependent::g1b() const {
    _test_rdm_level(1, "g1b");
    return g1b_;
}
ambit::Tensor RDMsSpinDependent::g2aa() const {
    _test_rdm_level(2, "g2aa");
    return g2aa_;
}
ambit::Tensor RDMsSpinDependent::g2ab() const {
    _test_rdm_level(2, "g2ab");
    return g2ab_;
}
ambit::Tensor RDMsSpinDependent::g2bb() const {
    _test_rdm_level(2, "g2bb");
    return g2bb_;
}
ambit::Tensor RDMsSpinDependent::g3aaa() const {
    _test_rdm_level(3, "g3aaa");
    return g3aaa_;
}
ambit::Tensor RDMsSpinDependent::g3aab() const {
    _test_rdm_level(3, "g3aab");
    return g3aab_;
}
ambit::Tensor RDMsSpinDependent::g3abb() const {
    _test_rdm_level(3, "g3abb");
    return g3abb_;
}
ambit::Tensor RDMsSpinDependent::g3bbb() const {
    _test_rdm_level(3, "g3bbb");
    return g3bbb_;
}
ambit::Tensor RDMsSpinDependent::g4aaaa() const {
    _test_rdm_level(4, "g4aaaa");
    return g4aaaa_;
}
ambit::Tensor RDMsSpinDependent::g4aaab() const {
    _test_rdm_level(4, "g4aaab");
    return g4aaab_;
}
ambit::Tensor RDMsSpinDependent::g4aabb() const {
    _test_rdm_level(4, "g4aabb");
    return g4aabb_;
}
ambit::Tensor RDMsSpinDependent::g4abbb() const {
    _test_rdm_level(4, "g4abbb");
    return g4abbb_;
}
ambit::Tensor RDMsSpinDependent::g4bbbb() const {
    _test_rdm_level(4, "g4bbbb");
    return g4bbbb_;
}
ambit::Tensor RDMsSpinDependent::SF_G1() const {
    _test_rdm_level(1, "SF_G1");
    auto G1 = g1a_.clone();
    G1("pq") += g1b_("pq");
    G1.set_name("SF_G1");
    return G1;
}
ambit::Tensor RDMsSpinDependent::SF_G2() const {
    _test_rdm_level(2, "SF_G2");
    auto G2 = g2aa_.clone();
    G2("pqrs") += g2bb_("pqrs");
    G2("pqrs") += g2ab_("pqrs");
    G2("pqrs") += g2ab_("qpsr");
    G2.set_name("SF_G2");
    return G2;
}
ambit::Tensor RDMsSpinDependent::SF_G3() const {
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
ambit::Tensor RDMsSpinDependent::L1a() const {
    auto L1a = g1a();
    L1a.set_name("L1a");
    return L1a;
}
ambit::Tensor RDMsSpinDependent::L1b() const {
    auto L1b = g1b();
    L1b.set_name("L1b");
    return L1b;
}
ambit::Tensor RDMsSpinDependent::L2aa() const {
    _test_rdm_level(2, "L2aa");
    auto L2aa = make_cumulant_L2aa(g1a_, g2aa_);
    L2aa.set_name("L2aa");
    return L2aa;
}
ambit::Tensor RDMsSpinDependent::L2ab() const {
    _test_rdm_level(2, "L2ab");
    auto L2ab = make_cumulant_L2ab(g1a_, g1b_, g2ab_);
    L2ab.set_name("L2ab");
    return L2ab;
}
ambit::Tensor RDMsSpinDependent::L2bb() const {
    _test_rdm_level(2, "L2bb");
    auto L2bb = make_cumulant_L2aa(g1b_, g2bb_);
    L2bb.set_name("L2bb");
    return L2bb;
}
ambit::Tensor RDMsSpinDependent::L3aaa() const {
    _test_rdm_level(3, "L3aaa");
    auto L3aaa = make_cumulant_L3aaa(g1a_, g2aa_, g3aaa_);
    L3aaa.set_name("L3aaa");
    return L3aaa;
}
ambit::Tensor RDMsSpinDependent::L3aab() const {
    _test_rdm_level(3, "L3aab");
    auto L3aab = make_cumulant_L3aab(g1a_, g1b_, g2aa_, g2ab_, g3aab_);
    L3aab.set_name("L3aab");
    return L3aab;
}
ambit::Tensor RDMsSpinDependent::L3abb() const {
    _test_rdm_level(3, "L3abb");
    auto L3abb = make_cumulant_L3abb(g1a_, g1b_, g2ab_, g2bb_, g3abb_);
    L3abb.set_name("L3abb");
    return L3abb;
}
ambit::Tensor RDMsSpinDependent::L3bbb() const {
    _test_rdm_level(3, "L3bbb");
    auto L3bbb = make_cumulant_L3aaa(g1b_, g2bb_, g3bbb_);
    L3bbb.set_name("L3bbb");
    return L3bbb;
}

ambit::Tensor RDMsSpinDependent::L4aaaa() const {
    _test_rdm_level(4, "L4aaaa");
    auto _L1a = L1a();
    auto _L2aa = L2aa();
    auto _L3aaa = L3aaa();
    auto L4aaaa = make_cumulant_L4aaaa(_L1a, _L2aa, _L3aaa, g4aaaa_);
    L4aaaa.set_name("L4aaaa");
    return L4aaaa;
}

ambit::Tensor RDMsSpinDependent::L4aaab() const {
    _test_rdm_level(4, "L4aaab");
    auto _L1a = L1a();
    auto _L1b = L1b();
    auto _L2aa = L2aa();
    auto _L2ab = L2ab();
    auto _L3aaa = L3aaa();
    auto _L3aab = L3aab();
    auto L4aaab = make_cumulant_L4aaab(_L1a, _L1b, _L2aa, _L2ab, _L3aaa, _L3aab, g4aaab_);
    L4aaab.set_name("L4aaab");
    return L4aaab;
}

ambit::Tensor RDMsSpinDependent::L4aabb() const {
    _test_rdm_level(4, "L4aabb");
    auto _L1a = L1a();
    auto _L1b = L1b();
    auto _L2aa = L2aa();
    auto _L2ab = L2ab();
    auto _L2bb = L2bb();
    auto _L3aab = L3aab();
    auto _L3abb = L3abb();
    auto L4aabb = make_cumulant_L4aabb(_L1a, _L1b, _L2aa, _L2ab, _L2bb, _L3aab, _L3abb, g4aabb_);
    L4aabb.set_name("L4aabb");
    return L4aabb;
}

ambit::Tensor RDMsSpinDependent::L4abbb() const {
    _test_rdm_level(4, "L4abbb");
    auto _L1a = L1a();
    auto _L1b = L1b();
    auto _L2ab = L2ab();
    auto _L2bb = L2bb();
    auto _L3abb = L3abb();
    auto _L3bbb = L3bbb();
    auto L4abbb = make_cumulant_L4abbb(_L1a, _L1b, _L2ab, _L2bb, _L3abb, _L3bbb, g4abbb_);
    L4abbb.set_name("L4abbb");
    return L4abbb;
}

ambit::Tensor RDMsSpinDependent::L4bbbb() const {
    _test_rdm_level(4, "L4bbbb");
    auto _L1b = L1b();
    auto _L2bb = L2bb();
    auto _L3bbb = L3bbb();
    auto L4bbbb = make_cumulant_L4aaaa(_L1b, _L2bb, _L3bbb, g4bbbb_);
    L4bbbb.set_name("L4bbbb");
    return L4bbbb;
}

std::shared_ptr<RDMs> RDMsSpinDependent::clone() {
    ambit::Tensor g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb, g4aaaa, g4aaab, g4aabb,
        g4abbb, g4bbbb;
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
    if (max_rdm_ > 3) {
        g4aaaa = g4aaaa_.clone();
        g4aaab = g4aaab_.clone();
        g4aabb = g4aabb_.clone();
        g4abbb = g4abbb_.clone();
        g4bbbb = g4bbbb_.clone();
    }

    std::shared_ptr<RDMs> rdms;

    if (max_rdm_ < 1)
        rdms = std::make_shared<RDMsSpinDependent>();
    else if (max_rdm_ == 1)
        rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b);
    else if (max_rdm_ == 2)
        rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb);
    else if (max_rdm_ == 3)
        rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb,
                                                   g3bbb);
    else
        rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb,
                                                   g3bbb, g4aaaa, g4aaab, g4aabb, g4abbb, g4bbbb);

    return rdms;
}

void RDMsSpinDependent::scale(double factor) {
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
    if (max_rdm_ > 3) {
        g4aaaa_.scale(factor);
        g4aaab_.scale(factor);
        g4aabb_.scale(factor);
        g4abbb_.scale(factor);
        g4bbbb_.scale(factor);
    }
}

void RDMsSpinDependent::axpy(std::shared_ptr<RDMs> rhs, double a) {
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
    if (max_rdm_ > 3) {
        g4aaaa_("pqrstuvw") += a * rhs->g4aaaa()("pqrstuvw");
        g4aaab_("pqrstuvw") += a * rhs->g4aaab()("pqrstuvw");
        g4aabb_("pqrstuvw") += a * rhs->g4aabb()("pqrstuvw");
        g4abbb_("pqrstuvw") += a * rhs->g4abbb()("pqrstuvw");
        g4bbbb_("pqrstuvw") += a * rhs->g4bbbb()("pqrstuvw");
    }
}

void RDMsSpinDependent::rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) {
    if (_bypass_rotate(Ua, Ub))
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

    if (max_rdm_ == 3)
        return;

    if (max_rdm_ == 4)
        throw std::runtime_error("RDMs rotation not implemented for 4-RDMs.");
}

void RDMsSpinDependent::dump_to_disk(const std::string& filename_prefix) const {
    std::string prefix = filename_prefix + (filename_prefix.empty() ? "" : ".");

    if (max_rdm_ > 0) {
        ambit::save(g1a_, prefix + "g1a.bin");
        ambit::save(g1b_, prefix + "g1b.bin");
    }
    if (max_rdm_ > 1) {
        ambit::save(g2aa_, prefix + "g2aa.bin");
        ambit::save(g2ab_, prefix + "g2ab.bin");
        ambit::save(g2bb_, prefix + "g2bb.bin");
    }
    if (max_rdm_ > 2) {
        ambit::save(g3aaa_, prefix + "g3aaa.bin");
        ambit::save(g3aab_, prefix + "g3aab.bin");
        ambit::save(g3abb_, prefix + "g3abb.bin");
        ambit::save(g3bbb_, prefix + "g3bbb.bin");
    }
    if (max_rdm_ > 3) {
        ambit::save(g4aaaa_, prefix + "g4aaaa.bin");
        ambit::save(g4aaab_, prefix + "g4aaab.bin");
        ambit::save(g4aabb_, prefix + "g4aabb.bin");
        ambit::save(g4abbb_, prefix + "g4abbb.bin");
        ambit::save(g4bbbb_, prefix + "g4bbbb.bin");
    }
}

RDMsSpinFree::RDMsSpinFree() {
    max_rdm_ = 0;
    type_ = RDMsType::spin_free;
    n_orbs_ = 0;
}

RDMsSpinFree::RDMsSpinFree(ambit::Tensor G1) : SF_G1_(G1) {
    max_rdm_ = 1;
    type_ = RDMsType::spin_free;
    n_orbs_ = G1.dim(0);
    _test_rdm_dims(G1, "G1", 2);
}

RDMsSpinFree::RDMsSpinFree(ambit::Tensor G1, ambit::Tensor G2) : SF_G1_(G1), SF_G2_(G2) {
    max_rdm_ = 2;
    type_ = RDMsType::spin_free;
    n_orbs_ = G1.dim(0);
    _test_rdm_dims(G1, "G1", 2);
    _test_rdm_dims(G2, "G2", 4);
}

RDMsSpinFree::RDMsSpinFree(ambit::Tensor G1, ambit::Tensor G2, ambit::Tensor G3)
    : SF_G1_(G1), SF_G2_(G2), SF_G3_(G3) {
    max_rdm_ = 3;
    type_ = RDMsType::spin_free;
    n_orbs_ = G1.dim(0);
    _test_rdm_dims(G1, "G1", 2);
    _test_rdm_dims(G2, "G2", 4);
    _test_rdm_dims(G3, "G3", 6);
}

ambit::Tensor RDMsSpinFree::SF_G1() const {
    _test_rdm_level(1, "SF_G1");
    return SF_G1_;
}
ambit::Tensor RDMsSpinFree::SF_G2() const {
    _test_rdm_level(2, "SF_G2");
    return SF_G2_;
}
ambit::Tensor RDMsSpinFree::SF_G3() const {
    _test_rdm_level(3, "SF_G3");
    return SF_G3_;
}
ambit::Tensor RDMsSpinFree::g1a() const {
    _test_rdm_level(1, "g1a");
    auto g1a = sf1_to_sd1(SF_G1_);
    g1a.set_name("g1a");
    return g1a;
}
ambit::Tensor RDMsSpinFree::g1b() const {
    _test_rdm_level(1, "g1b");
    auto g1b = sf1_to_sd1(SF_G1_);
    g1b.set_name("g1b");
    return g1b;
}
ambit::Tensor RDMsSpinFree::g2aa() const {
    _test_rdm_level(2, "g2aa");
    auto g2aa = sf2_to_sd2aa(SF_G2_);
    g2aa.set_name("g2aa");
    return g2aa;
}
ambit::Tensor RDMsSpinFree::g2ab() const {
    _test_rdm_level(2, "g2ab");
    auto g2ab = sf2_to_sd2ab(SF_G2_);
    g2ab.set_name("g2ab");
    return g2ab;
}
ambit::Tensor RDMsSpinFree::g2bb() const {
    _test_rdm_level(2, "g2bb");
    auto g2bb = sf2_to_sd2aa(SF_G2_);
    g2bb.set_name("g2bb");
    return g2bb;
}
ambit::Tensor RDMsSpinFree::g3aaa() const {
    _test_rdm_level(3, "g3aaa");
    auto g3aaa = sf3_to_sd3aaa(SF_G3_);
    g3aaa.set_name("g3aaa");
    return g3aaa;
}
ambit::Tensor RDMsSpinFree::g3aab() const {
    _test_rdm_level(3, "g3aab");
    auto g3aab = sf3_to_sd3aab(SF_G3_);
    g3aab.set_name("g3aab");
    return g3aab;
}
ambit::Tensor RDMsSpinFree::g3abb() const {
    _test_rdm_level(3, "g3abb");
    auto g3abb = sf3_to_sd3abb(SF_G3_);
    g3abb.set_name("g3abb");
    return g3abb;
}
ambit::Tensor RDMsSpinFree::g3bbb() const {
    _test_rdm_level(3, "g3bbb");
    auto g3bbb = sf3_to_sd3aaa(SF_G3_);
    g3bbb.set_name("g3bbb");
    return g3bbb;
}
ambit::Tensor RDMsSpinFree::g4aaaa() const {
    throw std::runtime_error("RDMsSpinFree::g4aaaa not implemented.");
    ambit::Tensor g4aaaa;
    return g4aaaa;
}
ambit::Tensor RDMsSpinFree::g4aaab() const {
    throw std::runtime_error("RDMsSpinFree::g4aaab not implemented.");
    ambit::Tensor g4aaab;
    return g4aaab;
}
ambit::Tensor RDMsSpinFree::g4aabb() const {
    throw std::runtime_error("RDMsSpinFree::g4aabb not implemented.");
    ambit::Tensor g4aabb;
    return g4aabb;
}
ambit::Tensor RDMsSpinFree::g4abbb() const {
    throw std::runtime_error("RDMsSpinFree::g4abbb not implemented.");
    ambit::Tensor g4abbb;
    return g4abbb;
}
ambit::Tensor RDMsSpinFree::g4bbbb() const {
    throw std::runtime_error("RDMsSpinFree::g4bbbb not implemented.");
    ambit::Tensor g4bbbb;
    return g4bbbb;
}
ambit::Tensor RDMsSpinFree::L1a() const {
    _test_rdm_level(1, "L1a");
    auto L1a = sf1_to_sd1(SF_G1_);
    L1a.set_name("L1a");
    return L1a;
}
ambit::Tensor RDMsSpinFree::L1b() const {
    _test_rdm_level(1, "L1b");
    auto L1b = sf1_to_sd1(SF_G1_);
    L1b.set_name("L1b");
    return L1b;
}
ambit::Tensor RDMsSpinFree::L2aa() const {
    _test_rdm_level(2, "L2aa");
    auto L2aa = sf2_to_sd2aa(SF_L2());
    L2aa.set_name("L2aa");
    return L2aa;
}
ambit::Tensor RDMsSpinFree::L2ab() const {
    _test_rdm_level(2, "L2ab");
    auto L2ab = sf2_to_sd2ab(SF_L2());
    L2ab.set_name("L2ab");
    return L2ab;
}
ambit::Tensor RDMsSpinFree::L2bb() const {
    _test_rdm_level(2, "L2bb");
    auto L2bb = sf2_to_sd2aa(SF_L2());
    L2bb.set_name("L2bb");
    return L2bb;
}
ambit::Tensor RDMsSpinFree::L3aaa() const {
    _test_rdm_level(3, "L3aaa");
    auto L3aaa = sf3_to_sd3aaa(SF_L3());
    L3aaa.set_name("L3aaa");
    return L3aaa;
}
ambit::Tensor RDMsSpinFree::L3aab() const {
    _test_rdm_level(3, "L3aab");
    auto L3aab = sf3_to_sd3aab(SF_L3());
    L3aab.set_name("L3aab");
    return L3aab;
}
ambit::Tensor RDMsSpinFree::L3abb() const {
    _test_rdm_level(3, "L3abb");
    auto L3abb = sf3_to_sd3abb(SF_L3());
    L3abb.set_name("L3abb");
    return L3abb;
}
ambit::Tensor RDMsSpinFree::L3bbb() const {
    _test_rdm_level(3, "L3bbb");
    auto L3bbb = sf3_to_sd3aaa(SF_L3());
    L3bbb.set_name("L3bbb");
    return L3bbb;
}

ambit::Tensor RDMsSpinFree::L4aaaa() const {
    throw std::runtime_error("RDMsSpinFree::L4aaaa not implemented.");
    ambit::Tensor L4aaaa;
    return L4aaaa;
}

ambit::Tensor RDMsSpinFree::L4aaab() const {
    throw std::runtime_error("RDMsSpinFree::L4aaab not implemented.");
    ambit::Tensor L4aaab;
    return L4aaab;
}

ambit::Tensor RDMsSpinFree::L4aabb() const {
    throw std::runtime_error("RDMsSpinFree::L4aabb not implemented.");
    ambit::Tensor L4aabb;
    return L4aabb;
}

ambit::Tensor RDMsSpinFree::L4abbb() const {
    throw std::runtime_error("RDMsSpinFree::L4abbb not implemented.");
    ambit::Tensor L4abbb;
    return L4abbb;
}

ambit::Tensor RDMsSpinFree::L4bbbb() const {
    throw std::runtime_error("RDMsSpinFree::L4bbbb not implemented.");
    ambit::Tensor L4bbbb;
    return L4bbbb;
}

std::shared_ptr<RDMs> RDMsSpinFree::clone() {
    ambit::Tensor g1, g2, g3;
    if (max_rdm_ > 0)
        g1 = SF_G1_.clone();
    if (max_rdm_ > 1)
        g2 = SF_G2_.clone();
    if (max_rdm_ > 2)
        g3 = SF_G3_.clone();

    std::shared_ptr<RDMs> rdms;

    if (max_rdm_ < 1)
        rdms = std::make_shared<RDMsSpinFree>();
    else if (max_rdm_ == 1)
        rdms = std::make_shared<RDMsSpinFree>(g1);
    else if (max_rdm_ == 2)
        rdms = std::make_shared<RDMsSpinFree>(g1, g2);
    else
        rdms = std::make_shared<RDMsSpinFree>(g1, g2, g3);

    return rdms;
}

void RDMsSpinFree::scale(double factor) {
    if (max_rdm_ > 0)
        SF_G1_.scale(factor);
    if (max_rdm_ > 1)
        SF_G2_.scale(factor);
    if (max_rdm_ > 2)
        SF_G3_.scale(factor);
}

void RDMsSpinFree::axpy(std::shared_ptr<RDMs> rhs, double a) {
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

void RDMsSpinFree::rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) {
    if (_bypass_rotate(Ua, Ub))
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

void RDMsSpinFree::dump_to_disk(const std::string& filename_prefix) const {
    std::string prefix = (filename_prefix.empty() ? "" : filename_prefix + ".");

    if (max_rdm_ > 0) {
        ambit::save(SF_G1_, prefix + "g1.bin");
    }
    if (max_rdm_ > 1) {
        ambit::save(SF_G2_, prefix + "g2.bin");
    }
    if (max_rdm_ > 2) {
        ambit::save(SF_G3_, prefix + "g3.bin");
    }
}
} // namespace forte
