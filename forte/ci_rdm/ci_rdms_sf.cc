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

#include <cmath>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"

#include "helpers/timer.h"
#include "sparse_ci/determinant_substitution_lists.h"

#include "ci_rdms.h"

using namespace psi;

namespace forte {

void CI_RDMS::compute_1rdm_sf(std::vector<double>& opdm) {
    timer one("Build 1 Substitution Lists");
    get_one_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 1-map:   %1.6f", one.stop());

    timer build("Build SF 1-RDM");
    opdm.assign(no2_, 0.0);
    for (size_t J = 0; J < dim_space_; ++J) {
        for (auto& aJ_mo_sign : a_ann_list_[J]) {
            const auto aJ_add = aJ_mo_sign.first;
            const auto p = std::abs(aJ_mo_sign.second) - 1;
            const auto sign_p = aJ_mo_sign.second > 0 ? 1 : -1;
            for (auto& aaJ_mo_sign : a_cre_list_[aJ_add]) {
                const auto q = std::abs(aaJ_mo_sign.second) - 1;
                const auto sign_q = aaJ_mo_sign.second > 0 ? 1 : -1;
                const auto I = aaJ_mo_sign.first;
                opdm[q * no_ + p] +=
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_p * sign_q;
            }
        }
        for (auto& bJ_mo_sign : b_ann_list_[J]) {
            const auto bJ_add = bJ_mo_sign.first;
            const auto p = std::abs(bJ_mo_sign.second) - 1;
            const auto sign_p = bJ_mo_sign.second > 0 ? 1 : -1;
            for (auto& bbJ_mo_sign : b_cre_list_[bJ_add]) {
                const auto q = std::abs(bbJ_mo_sign.second) - 1;
                const auto sign_q = bbJ_mo_sign.second > 0 ? 1 : -1;
                const auto I = bbJ_mo_sign.first;
                opdm[q * no_ + p] +=
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_p * sign_q;
            }
        }
    }
    if (print_)
        outfile->Printf("\n  Time spent building 1-rdm:   %1.6f", build.stop());
}

void CI_RDMS::compute_1rdm_sf_op(std::vector<double>& opdm) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->op_s_lists(wfn_);

    timer build("Build SF 1-RDM");
    opdm.assign(no2_, 0.0);
    opdm.assign(no2_, 0.0);

    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t J = 0; J < dim_space_; ++J) {
        double cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
        for (int pp : dets[J].get_alfa_occ(no_)) {
            opdm[pp * no_ + pp] += cJ_sq;
        }
        for (int pp : dets[J].get_beta_occ(no_)) {
            opdm[pp * no_ + pp] += cJ_sq;
        }
    }

    _add_1rdm_op_IJ(opdm, op->a_list_);
    _add_1rdm_op_IJ(opdm, op->b_list_);
//    for (const auto& coupled_dets : op->a_list_) {
//        for (size_t a = 0, coupled_dets_size = coupled_dets.size(); a < coupled_dets_size; ++a) {
//            const auto [I, _p] = coupled_dets[a];
//            auto p = std::abs(_p) - 1;
//            auto sign_p = _p < 0;
//
//            auto vI1 = evecs_->get(I, root1_);
//            auto vI2 = evecs_->get(I, root2_);
//
//            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
//                const auto [J, _q] = coupled_dets[b];
//                auto q = std::abs(_q) - 1;
//                auto sign = ((_q < 0) == sign_p) ? 1 : -1;
//
//                opdm[p * no_ + q] += vI1 * evecs_->get(J, root2_) * sign;
//                opdm[q * no_ + p] += evecs_->get(J, root1_) * vI2 * sign;
//            }
//        }
//    }
//    for (const auto& coupled_dets : op->b_list_) {
//        for (size_t a = 0, coupled_dets_size = coupled_dets.size(); a < coupled_dets_size; ++a) {
//            const auto [I, _p] = coupled_dets[a];
//            auto p = std::abs(_p) - 1;
//            auto sign_p = _p < 0;
//
//            auto vI1 = evecs_->get(I, root1_);
//            auto vI2 = evecs_->get(I, root2_);
//
//            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
//                const auto [J, _q] = coupled_dets[b];
//                auto q = std::abs(_q) - 1;
//                auto sign = ((_q < 0) == sign_p) ? 1 : -1;
//
//                opdm[p * no_ + q] += vI1 * evecs_->get(J, root2_) * sign;
//                opdm[q * no_ + p] += evecs_->get(J, root1_) * vI2 * sign;
//            }
//        }
//    }

    if (print_) {
        outfile->Printf("\n  Time spent building 1-rdm: %.3e seconds", build.stop());
    }
}

void CI_RDMS::compute_2rdm_sf(std::vector<double>& tpdm) {
    tpdm.assign(no4_, 0.0);

    timer two("Build 2 Substitution Lists");
    get_two_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 2-map:   %1.6f", two.stop());

    timer build("Build SF 2-RDM");

    for (size_t J = 0; J < dim_space_; ++J) {
        auto vJ = evecs_->get(J, root1_);

        // aaaa
        for (const auto& aaJ_mo_sign : aa_ann_list_[J]) {
            const auto [aaJ_add, _p, q] = aaJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            for (const auto& aaaaJ_mo_sign : aa_cre_list_[aaJ_add]) {
                const auto [I, _r, s] = aaaaJ_mo_sign;
                auto r = std::abs(_r) - 1;

                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;
                auto value = evecs_->get(I, root2_) * vJ * sign;

                tpdm[p * no3_ + q * no2_ + r * no_ + s] += value;
                tpdm[p * no3_ + q * no2_ + s * no_ + r] -= value;
                tpdm[q * no3_ + p * no2_ + r * no_ + s] -= value;
                tpdm[q * no3_ + p * no2_ + s * no_ + r] += value;
            }
        }

        // bbbb
        for (auto& bbJ_mo_sign : bb_ann_list_[J]) {
            const auto [bbJ_add, _p, q] = bbJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            for (auto& bbbbJ_mo_sign : bb_cre_list_[bbJ_add]) {
                const auto [I, _r, s] = bbbbJ_mo_sign;
                auto r = std::abs(_r) - 1;

                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;
                auto value = evecs_->get(I, root2_) * vJ * sign;

                tpdm[p * no3_ + q * no2_ + r * no_ + s] += value;
                tpdm[p * no3_ + q * no2_ + s * no_ + r] -= value;
                tpdm[q * no3_ + p * no2_ + r * no_ + s] -= value;
                tpdm[q * no3_ + p * no2_ + s * no_ + r] += value;
            }
        }

        // aabb
        for (auto& abJ_mo_sign : ab_ann_list_[J]) {
            const auto [abJ_add, _p, q] = abJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            for (auto& aabbJ_mo_sign : ab_cre_list_[abJ_add]) {
                const auto [I, _r, s] = aabbJ_mo_sign;
                auto r = std::abs(_r) - 1;

                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;
                auto value = evecs_->get(I, root2_) * vJ * sign;

                tpdm[p * no3_ + q * no2_ + r * no_ + s] += value;
                tpdm[q * no3_ + p * no2_ + s * no_ + r] += value;
            }
        }
    }

    if (print_)
        outfile->Printf("\n  Time spent building 2-rdm:   %1.6f", build.stop());
}

void CI_RDMS::compute_2rdm_sf_op(std::vector<double>& tpdm) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->tp_s_lists(wfn_);

    tpdm.assign(no4_, 0.0);
    timer build("Build SF 2-RDM");

    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t J = 0; J < dim_space_; ++J) {
        auto cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
        auto aocc = dets[J].get_alfa_occ(no_);
        auto bocc = dets[J].get_beta_occ(no_);

        auto naocc = aocc.size();
        auto nbocc = bocc.size();

        for (size_t p = 0; p < naocc; ++p) {
            auto pp = aocc[p];
            for (size_t q = p + 1; q < naocc; ++q) {
                auto qq = aocc[q];

                tpdm[pp * no3_ + qq * no2_ + pp * no_ + qq] += cJ_sq;
                tpdm[qq * no3_ + pp * no2_ + pp * no_ + qq] -= cJ_sq;

                tpdm[qq * no3_ + pp * no2_ + qq * no_ + pp] += cJ_sq;
                tpdm[pp * no3_ + qq * no2_ + qq * no_ + pp] -= cJ_sq;
            }
        }

        for (size_t p = 0; p < nbocc; ++p) {
            auto pp = bocc[p];
            for (size_t q = p + 1; q < nbocc; ++q) {
                auto qq = bocc[q];

                tpdm[pp * no3_ + qq * no2_ + pp * no_ + qq] += cJ_sq;
                tpdm[qq * no3_ + pp * no2_ + pp * no_ + qq] -= cJ_sq;

                tpdm[qq * no3_ + pp * no2_ + qq * no_ + pp] += cJ_sq;
                tpdm[pp * no3_ + qq * no2_ + qq * no_ + pp] -= cJ_sq;
            }
        }

        for (size_t p = 0; p < naocc; ++p) {
            auto pp = aocc[p];
            for (size_t q = 0; q < nbocc; ++q) {
                auto qq = bocc[q];

                tpdm[pp * no3_ + qq * no2_ + pp * no_ + qq] += cJ_sq;
                tpdm[qq * no3_ + pp * no2_ + qq * no_ + pp] += cJ_sq;
            }
        }
    }

    // aaaa
    for (const auto& coupled_dets : op->aa_list_) {
        for (size_t a = 0, coupled_dets_size = coupled_dets.size(); a < coupled_dets_size; ++a) {
            const auto [J, _p, q] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            auto vJ1 = evecs_->get(J, root1_);
            auto vJ2 = evecs_->get(J, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [I, _r, s] = coupled_dets[b];
                auto r = std::abs(_r) - 1;
                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;

                auto value = vJ1 * evecs_->get(I, root2_) * sign;
                tpdm[p * no3_ + q * no2_ + r * no_ + s] += value;
                tpdm[p * no3_ + q * no2_ + s * no_ + r] -= value;
                tpdm[q * no3_ + p * no2_ + r * no_ + s] -= value;
                tpdm[q * no3_ + p * no2_ + s * no_ + r] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tpdm[r * no3_ + s * no2_ + p * no_ + q] += value;
                tpdm[s * no3_ + r * no2_ + p * no_ + q] -= value;
                tpdm[r * no3_ + s * no2_ + q * no_ + p] -= value;
                tpdm[s * no3_ + r * no2_ + q * no_ + p] += value;
            }
        }
    }

    // bbbb
    for (const auto& coupled_dets : op->bb_list_) {
        for (size_t a = 0, coupled_dets_size = coupled_dets.size(); a < coupled_dets_size; ++a) {
            const auto [J, _p, q] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            auto vJ1 = evecs_->get(J, root1_);
            auto vJ2 = evecs_->get(J, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [I, _r, s] = coupled_dets[b];
                auto r = std::abs(_r) - 1;
                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;

                auto value = vJ1 * evecs_->get(I, root2_) * sign;
                tpdm[p * no3_ + q * no2_ + r * no_ + s] += value;
                tpdm[p * no3_ + q * no2_ + s * no_ + r] -= value;
                tpdm[q * no3_ + p * no2_ + r * no_ + s] -= value;
                tpdm[q * no3_ + p * no2_ + s * no_ + r] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tpdm[r * no3_ + s * no2_ + p * no_ + q] += value;
                tpdm[s * no3_ + r * no2_ + p * no_ + q] -= value;
                tpdm[r * no3_ + s * no2_ + q * no_ + p] -= value;
                tpdm[s * no3_ + r * no2_ + q * no_ + p] += value;
            }
        }
    }

    // aabb
    for (const auto& coupled_dets : op->ab_list_) {
        for (size_t a = 0, coupled_dets_size = coupled_dets.size(); a < coupled_dets_size; ++a) {
            const auto [J, _p, q] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            auto vJ1 = evecs_->get(J, root1_);
            auto vJ2 = evecs_->get(J, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [I, _r, s] = coupled_dets[b];
                auto r = std::abs(_r) - 1;
                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;

                double value = vJ1 * evecs_->get(I, root2_) * sign;
                tpdm[p * no3_ + q * no2_ + r * no_ + s] += value;
                tpdm[q * no3_ + p * no2_ + s * no_ + r] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tpdm[r * no3_ + s * no2_ + p * no_ + q] += value;
                tpdm[s * no3_ + r * no2_ + q * no_ + p] += value;
            }
        }
    }

    if (print_) {
        outfile->Printf("\n  Time spent building 2-rdm: %.3e seconds", build.stop());
    }
}

void CI_RDMS::compute_3rdm_sf(std::vector<double>& tpdm3) {
    tpdm3.assign(no6_, 0.0);

    timer three("Build 3 Substitution Lists");
    get_three_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 3-map:   %1.6f", three.stop());

    timer build("Build SF 3-RDM");

    for (size_t J = 0; J < dim_space_; ++J) {
        auto vJ = evecs_->get(J, root1_);

        // aaa aaa
        for (const auto& aaaJ_mo_sign : aaa_ann_list_[J]) {
            const auto [aaaJ_add, _p, q, r] = aaaJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            for (const auto& a6J : aaa_cre_list_[aaaJ_add]) {
                const auto [I, _s, t, u] = a6J;
                auto s = std::abs(_s) - 1;

                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;
                auto value = evecs_->get(I, root2_) * vJ * sign;

                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] += value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ + t] -= value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + u * no2_ + t * no_ + s] -= value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + u * no2_ + s * no_ + t] += value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + s * no_ + u] -= value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + u * no_ + s] += value;

                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -= value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ + t] += value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + u * no2_ + t * no_ + s] += value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + u * no2_ + s * no_ + t] -= value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + s * no_ + u] += value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + u * no_ + s] -= value;

                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -= value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + u * no_ + t] += value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + t * no_ + s] += value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + s * no_ + t] -= value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + s * no_ + u] += value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + u * no_ + s] -= value;

                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + t * no_ + u] += value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ + t] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t * no_ + s] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + s * no_ + t] += value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + s * no_ + u] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + u * no_ + s] += value;

                tpdm3[r * no5_ + p * no4_ + q * no3_ + s * no2_ + t * no_ + u] += value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + s * no2_ + u * no_ + t] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t * no_ + s] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + s * no_ + t] += value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + s * no_ + u] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + u * no_ + s] += value;

                tpdm3[r * no5_ + q * no4_ + p * no3_ + s * no2_ + t * no_ + u] -= value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + s * no2_ + u * no_ + t] += value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t * no_ + s] += value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + s * no_ + t] -= value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + s * no_ + u] += value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + u * no_ + s] -= value;
            }
        }

        // aab aab
        for (const auto& aabJ_mo_sign : aab_ann_list_[J]) {
            const auto [aabJ_add, _p, q, r] = aabJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            for (const auto& aabJ : aab_cre_list_[aabJ_add]) {
                const auto [I, _s, t, u] = aabJ;
                auto s = std::abs(_s) - 1;

                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;
                auto value = evecs_->get(I, root2_) * vJ * sign;

                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] += value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + s * no_ + u] -= value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -= value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + s * no_ + u] += value;

                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t * no_ + s] += value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + s * no_ + t] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t * no_ + s] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + s * no_ + t] += value;

                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ + t] += value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + u * no_ + s] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ + t] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + u * no_ + s] += value;
            }
        }

        // abb abb
        for (const auto& abbJ_mo_sign : abb_ann_list_[J]) {
            const auto [abbJ_add, _p, q, r] = abbJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            for (const auto& abbJ : abb_cre_list_[abbJ_add]) {
                const auto [I, _s, t, u] = abbJ;
                auto s = std::abs(_s) - 1;

                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;
                double value = evecs_->get(I, root2_) * vJ * sign;

                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] += value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ + t] -= value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -= value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ + t] += value;

                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + s * no_ + u] += value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + s * no_ + t] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + s * no_ + u] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + s * no_ + t] += value;

                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t * no_ + s] += value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + u * no_ + s] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t * no_ + s] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + u * no_ + s] += value;
            }
        }

        // bbb bbb
        for (const auto& bbbJ_mo_sign : bbb_ann_list_[J]) {
            const auto [bbbJ_add, _p, q, r] = bbbJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            for (const auto& b6J : bbb_cre_list_[bbbJ_add]) {
                const auto [I, _s, t, u] = b6J;
                auto s = std::abs(_s) - 1;

                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;
                auto value = evecs_->get(I, root2_) * vJ * sign;

                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] += value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ + t] -= value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + u * no2_ + t * no_ + s] -= value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + u * no2_ + s * no_ + t] += value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + s * no_ + u] -= value;
                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + u * no_ + s] += value;

                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -= value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ + t] += value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + u * no2_ + t * no_ + s] += value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + u * no2_ + s * no_ + t] -= value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + s * no_ + u] += value;
                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + u * no_ + s] -= value;

                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -= value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + u * no_ + t] += value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + t * no_ + s] += value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + s * no_ + t] -= value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + s * no_ + u] += value;
                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + u * no_ + s] -= value;

                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + t * no_ + u] += value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ + t] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t * no_ + s] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + s * no_ + t] += value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + s * no_ + u] -= value;
                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + u * no_ + s] += value;

                tpdm3[r * no5_ + p * no4_ + q * no3_ + s * no2_ + t * no_ + u] += value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + s * no2_ + u * no_ + t] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t * no_ + s] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + s * no_ + t] += value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + s * no_ + u] -= value;
                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + u * no_ + s] += value;

                tpdm3[r * no5_ + q * no4_ + p * no3_ + s * no2_ + t * no_ + u] -= value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + s * no2_ + u * no_ + t] += value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t * no_ + s] += value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + s * no_ + t] -= value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + s * no_ + u] += value;
                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + u * no_ + s] -= value;
            }
        }
    }

    if (print_)
        outfile->Printf("\n  Time spent building 3-rdm:   %1.6f", build.stop());
}

void CI_RDMS::compute_3rdm_sf_op(std::vector<double>& tpdm3) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->three_s_lists(wfn_);

    timer build("Build SF 3-RDM");

    tpdm3.assign(no6_, 0.0);

    _add_3rdm_op_II([&](const std::vector<size_t>& i,
                        const double& value) { _add_3rdm_aaa(tpdm3, i, value); },
                    [&](const std::vector<size_t>& i, const double& value) {
                        _add_3rdm_aab(tpdm3, i, value, true);
                    },
                    [&](const std::vector<size_t>& i, const double& value) {
                        _add_3rdm_abb(tpdm3, i, value, true);
                    },
                    [&](const std::vector<size_t>& i, const double& value) {
                        _add_3rdm_aaa(tpdm3, i, value);
                    });

    _add_3rdm_op_IJ(op->aaa_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aaa(tpdm3, i, value);
    });
    _add_3rdm_op_IJ(op->aab_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aab(tpdm3, i, value, true);
    });
    _add_3rdm_op_IJ(op->abb_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_abb(tpdm3, i, value, true);
    });
    _add_3rdm_op_IJ(op->bbb_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aaa(tpdm3, i, value);
    });

//    // Build the diagonal part
//    const det_hashvec& dets = wfn_.wfn_hash();
//    for (size_t I = 0; I < dim_space_; ++I) {
//        double cI_sq = evecs_->get(I, root1_) * evecs_->get(I, root2_);
//
//        auto aocc = dets[I].get_alfa_occ(no_);
//        auto bocc = dets[I].get_beta_occ(no_);
//        auto na = aocc.size();
//        auto nb = bocc.size();
//
//        for (size_t p = 0; p < na; ++p) {
//            auto pp = aocc[p];
//            for (size_t q = p + 1; q < na; ++q) {
//                auto qq = aocc[q];
//                for (size_t r = q + 1; r < na; ++r) {
//                    auto rr = aocc[r];
//
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + qq * no_ + rr] += cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + rr * no_ + qq] -= cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + rr * no2_ + pp * no_ + qq] += cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + rr * no2_ + qq * no_ + pp] -= cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + rr * no_ + pp] += cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + pp * no_ + rr] -= cI_sq;
//
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + qq * no_ + rr] -= cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + rr * no_ + qq] += cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + rr * no2_ + pp * no_ + qq] -= cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + rr * no2_ + qq * no_ + pp] += cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + rr * no_ + pp] -= cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + pp * no_ + rr] += cI_sq;
//
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + pp * no2_ + qq * no_ + rr] += cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + pp * no2_ + rr * no_ + qq] -= cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + pp * no_ + qq] += cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + qq * no_ + pp] -= cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + rr * no_ + pp] += cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + pp * no_ + rr] -= cI_sq;
//
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + pp * no2_ + qq * no_ + rr] -= cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + pp * no2_ + rr * no_ + qq] += cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + pp * no_ + qq] -= cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + qq * no_ + pp] += cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + rr * no_ + pp] -= cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + pp * no_ + rr] += cI_sq;
//
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + qq * no_ + rr] += cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + rr * no_ + qq] -= cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + pp * no_ + qq] += cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + qq * no_ + pp] -= cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + rr * no_ + pp] += cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + pp * no_ + rr] -= cI_sq;
//
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + qq * no_ + rr] -= cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + rr * no_ + qq] += cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + pp * no_ + qq] -= cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + qq * no_ + pp] += cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + rr * no_ + pp] -= cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + pp * no_ + rr] += cI_sq;
//                }
//            }
//        }
//
//        for (size_t p = 0; p < na; ++p) {
//            auto pp = aocc[p];
//            for (size_t q = p + 1; q < na; ++q) {
//                auto qq = aocc[q];
//                for (size_t r = 0; r < nb; ++r) {
//                    auto rr = bocc[r];
//
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + qq * no_ + rr] += cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + pp * no_ + rr] -= cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + qq * no_ + rr] -= cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + pp * no_ + rr] += cI_sq;
//
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + qq * no_ + pp] += cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + pp * no_ + qq] -= cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + qq * no_ + pp] -= cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + pp * no_ + qq] += cI_sq;
//
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + rr * no_ + qq] += cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + rr * no_ + pp] -= cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + rr * no_ + qq] -= cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + rr * no_ + pp] += cI_sq;
//                }
//            }
//        }
//
//        for (size_t p = 0; p < na; ++p) {
//            auto pp = aocc[p];
//            for (size_t q = 0; q < nb; ++q) {
//                auto qq = bocc[q];
//                for (size_t r = q + 1; r < nb; ++r) {
//                    auto rr = bocc[r];
//
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + qq * no_ + rr] += cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + rr * no_ + qq] -= cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + qq * no_ + rr] -= cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + rr * no_ + qq] += cI_sq;
//
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + pp * no_ + rr] += cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + pp * no_ + qq] -= cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + pp * no_ + rr] -= cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + pp * no_ + qq] += cI_sq;
//
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + qq * no_ + pp] += cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + rr * no_ + pp] -= cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + qq * no_ + pp] -= cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + rr * no_ + pp] += cI_sq;
//                }
//            }
//        }
//
//        for (size_t p = 0; p < nb; ++p) {
//            auto pp = bocc[p];
//            for (size_t q = p + 1; q < nb; ++q) {
//                auto qq = bocc[q];
//                for (size_t r = q + 1; r < nb; ++r) {
//                    auto rr = bocc[r];
//
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + qq * no_ + rr] += cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + rr * no_ + qq] -= cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + rr * no2_ + pp * no_ + qq] += cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + rr * no2_ + qq * no_ + pp] -= cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + rr * no_ + pp] += cI_sq;
//                    tpdm3[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + pp * no_ + rr] -= cI_sq;
//
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + qq * no_ + rr] -= cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + rr * no_ + qq] += cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + rr * no2_ + pp * no_ + qq] -= cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + rr * no2_ + qq * no_ + pp] += cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + rr * no_ + pp] -= cI_sq;
//                    tpdm3[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + pp * no_ + rr] += cI_sq;
//
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + pp * no2_ + qq * no_ + rr] += cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + pp * no2_ + rr * no_ + qq] -= cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + pp * no_ + qq] += cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + qq * no_ + pp] -= cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + rr * no_ + pp] += cI_sq;
//                    tpdm3[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + pp * no_ + rr] -= cI_sq;
//
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + pp * no2_ + qq * no_ + rr] -= cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + pp * no2_ + rr * no_ + qq] += cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + pp * no_ + qq] -= cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + qq * no_ + pp] += cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + rr * no_ + pp] -= cI_sq;
//                    tpdm3[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + pp * no_ + rr] += cI_sq;
//
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + qq * no_ + rr] += cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + rr * no_ + qq] -= cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + pp * no_ + qq] += cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + qq * no_ + pp] -= cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + rr * no_ + pp] += cI_sq;
//                    tpdm3[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + pp * no_ + rr] -= cI_sq;
//
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + qq * no_ + rr] -= cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + rr * no_ + qq] += cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + pp * no_ + qq] -= cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + qq * no_ + pp] += cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + rr * no_ + pp] -= cI_sq;
//                    tpdm3[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + pp * no_ + rr] += cI_sq;
//                }
//            }
//        }
//    }
//
//    // Build the off-diagonal part
//
//    // aaa aaa
//    for (const auto& coupled_dets : op->aaa_list_) {
//        auto coupled_dets_size = coupled_dets.size();
//
//        for (size_t a = 0; a < coupled_dets_size; ++a) {
//            const auto [J, _p, q, r] = coupled_dets[a];
//            auto p = std::abs(_p) - 1;
//            auto sign_pqr = _p < 0;
//
//            auto vJ1 = evecs_->get(J, root1_);
//            auto vJ2 = evecs_->get(J, root2_);
//
//            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
//                const auto [I, _s, t, u] = coupled_dets[b];
//                auto s = std::abs(_s) - 1;
//                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;
//
//                auto value = vJ1 * evecs_->get(I, root2_) * sign;
//
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] += value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ + t] -= value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + u * no2_ + t * no_ + s] -= value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + u * no2_ + s * no_ + t] += value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + s * no_ + u] -= value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + u * no_ + s] += value;
//
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -= value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ + t] += value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + u * no2_ + t * no_ + s] += value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + u * no2_ + s * no_ + t] -= value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + s * no_ + u] += value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + u * no_ + s] -= value;
//
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -= value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + u * no_ + t] += value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + t * no_ + s] += value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + s * no_ + t] -= value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + s * no_ + u] += value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + u * no_ + s] -= value;
//
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + t * no_ + u] += value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ + t] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t * no_ + s] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + s * no_ + t] += value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + s * no_ + u] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + u * no_ + s] += value;
//
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + s * no2_ + t * no_ + u] += value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + s * no2_ + u * no_ + t] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t * no_ + s] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + s * no_ + t] += value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + s * no_ + u] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + u * no_ + s] += value;
//
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + s * no2_ + t * no_ + u] -= value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + s * no2_ + u * no_ + t] += value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t * no_ + s] += value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + s * no_ + t] -= value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + s * no_ + u] += value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + u * no_ + s] -= value;
//
//                value = evecs_->get(I, root1_) * vJ2 * sign;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + p * no2_ + q * no_ + r] += value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + p * no2_ + q * no_ + r] -= value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + p * no2_ + q * no_ + r] -= value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + p * no2_ + q * no_ + r] += value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + p * no2_ + q * no_ + r] -= value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + p * no2_ + q * no_ + r] += value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + p * no2_ + r * no_ + q] -= value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + p * no2_ + r * no_ + q] += value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + p * no2_ + r * no_ + q] += value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + p * no2_ + r * no_ + q] -= value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + p * no2_ + r * no_ + q] += value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + p * no2_ + r * no_ + q] -= value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + q * no2_ + p * no_ + r] -= value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + q * no2_ + p * no_ + r] += value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + q * no2_ + p * no_ + r] += value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + q * no2_ + p * no_ + r] -= value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + q * no2_ + p * no_ + r] += value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + q * no2_ + p * no_ + r] -= value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + q * no2_ + r * no_ + p] += value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + q * no2_ + r * no_ + p] -= value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + q * no2_ + r * no_ + p] -= value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + q * no2_ + r * no_ + p] += value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + q * no2_ + r * no_ + p] -= value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + q * no2_ + r * no_ + p] += value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + r * no2_ + p * no_ + q] += value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + r * no2_ + p * no_ + q] -= value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + r * no2_ + p * no_ + q] -= value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + r * no2_ + p * no_ + q] += value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + r * no2_ + p * no_ + q] -= value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + r * no2_ + p * no_ + q] += value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + r * no2_ + q * no_ + p] -= value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + r * no2_ + q * no_ + p] += value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + r * no2_ + q * no_ + p] += value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + r * no2_ + q * no_ + p] -= value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + r * no2_ + q * no_ + p] += value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + r * no2_ + q * no_ + p] -= value;
//            }
//        }
//    }
//
//    // aab aab
//    for (const auto& coupled_dets : op->aab_list_) {
//        auto coupled_dets_size = coupled_dets.size();
//
//        for (size_t a = 0; a < coupled_dets_size; ++a) {
//            const auto [J, _p, q, r] = coupled_dets[a];
//            auto p = std::abs(_p) - 1;
//            auto sign_pqr = _p < 0;
//
//            auto vJ1 = evecs_->get(J, root1_);
//            auto vJ2 = evecs_->get(J, root2_);
//
//            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
//                const auto [I, _s, t, u] = coupled_dets[b];
//                auto s = std::abs(_s) - 1;
//                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;
//
//                auto value = vJ1 * evecs_->get(I, root2_) * sign;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] += value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + s * no_ + u] -= value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -= value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + s * no_ + u] += value;
//
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t * no_ + s] += value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + s * no_ + t] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t * no_ + s] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + s * no_ + t] += value;
//
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ + t] += value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + u * no_ + s] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ + t] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + u * no_ + s] += value;
//
//                value = evecs_->get(I, root1_) * vJ2 * sign;
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + p * no2_ + q * no_ + r] += value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + p * no2_ + q * no_ + r] -= value;
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + q * no2_ + p * no_ + r] -= value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + q * no2_ + p * no_ + r] += value;
//
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + r * no2_ + q * no_ + p] += value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + r * no2_ + q * no_ + p] -= value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + r * no2_ + p * no_ + q] -= value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + r * no2_ + p * no_ + q] += value;
//
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + p * no2_ + r * no_ + q] += value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + p * no2_ + r * no_ + q] -= value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + q * no2_ + r * no_ + p] -= value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + q * no2_ + r * no_ + p] += value;
//            }
//        }
//    }
//
//    // abb abb
//    for (const auto& coupled_dets : op->abb_list_) {
//        auto coupled_dets_size = coupled_dets.size();
//
//        for (size_t a = 0; a < coupled_dets_size; ++a) {
//            const auto [J, _p, q, r] = coupled_dets[a];
//            auto p = std::abs(_p) - 1;
//            auto sign_pqr = _p < 0;
//
//            auto vJ1 = evecs_->get(J, root1_);
//            auto vJ2 = evecs_->get(J, root2_);
//
//            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
//                const auto [I, _s, t, u] = coupled_dets[b];
//                auto s = std::abs(_s) - 1;
//                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;
//
//                auto value = vJ1 * evecs_->get(I, root2_) * sign;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] += value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ + t] -= value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -= value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ + t] += value;
//
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + s * no_ + u] += value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + s * no_ + t] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + s * no_ + u] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + s * no_ + t] += value;
//
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t * no_ + s] += value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + u * no_ + s] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t * no_ + s] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + u * no_ + s] += value;
//
//                value = evecs_->get(I, root1_) * vJ2 * sign;
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + p * no2_ + q * no_ + r] += value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + p * no2_ + q * no_ + r] -= value;
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + p * no2_ + r * no_ + q] -= value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + p * no2_ + r * no_ + q] += value;
//
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + q * no2_ + p * no_ + r] += value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + q * no2_ + p * no_ + r] -= value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + r * no2_ + p * no_ + q] -= value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + r * no2_ + p * no_ + q] += value;
//
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + r * no2_ + q * no_ + p] += value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + r * no2_ + q * no_ + p] -= value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + q * no2_ + r * no_ + p] -= value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + q * no2_ + r * no_ + p] += value;
//            }
//        }
//    }
//
//    // bbb bbb
//    for (const auto& coupled_dets : op->bbb_list_) {
//        auto coupled_dets_size = coupled_dets.size();
//
//        for (size_t a = 0; a < coupled_dets_size; ++a) {
//            const auto [J, _p, q, r] = coupled_dets[a];
//            auto p = std::abs(_p) - 1;
//            auto sign_pqr = _p < 0;
//
//            auto vJ1 = evecs_->get(J, root1_);
//            auto vJ2 = evecs_->get(J, root2_);
//
//            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
//                const auto [I, _s, t, u] = coupled_dets[b];
//                auto s = std::abs(_s) - 1;
//                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;
//
//                auto value = vJ1 * evecs_->get(I, root2_) * sign;
//
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] += value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ + t] -= value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + u * no2_ + t * no_ + s] -= value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + u * no2_ + s * no_ + t] += value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + s * no_ + u] -= value;
//                tpdm3[p * no5_ + q * no4_ + r * no3_ + t * no2_ + u * no_ + s] += value;
//
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -= value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ + t] += value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + u * no2_ + t * no_ + s] += value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + u * no2_ + s * no_ + t] -= value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + s * no_ + u] += value;
//                tpdm3[p * no5_ + r * no4_ + q * no3_ + t * no2_ + u * no_ + s] -= value;
//
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -= value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + s * no2_ + u * no_ + t] += value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + t * no_ + s] += value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + u * no2_ + s * no_ + t] -= value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + s * no_ + u] += value;
//                tpdm3[q * no5_ + p * no4_ + r * no3_ + t * no2_ + u * no_ + s] -= value;
//
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + t * no_ + u] += value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ + t] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t * no_ + s] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + u * no2_ + s * no_ + t] += value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + s * no_ + u] -= value;
//                tpdm3[q * no5_ + r * no4_ + p * no3_ + t * no2_ + u * no_ + s] += value;
//
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + s * no2_ + t * no_ + u] += value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + s * no2_ + u * no_ + t] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t * no_ + s] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + u * no2_ + s * no_ + t] += value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + s * no_ + u] -= value;
//                tpdm3[r * no5_ + p * no4_ + q * no3_ + t * no2_ + u * no_ + s] += value;
//
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + s * no2_ + t * no_ + u] -= value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + s * no2_ + u * no_ + t] += value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t * no_ + s] += value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + u * no2_ + s * no_ + t] -= value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + s * no_ + u] += value;
//                tpdm3[r * no5_ + q * no4_ + p * no3_ + t * no2_ + u * no_ + s] -= value;
//
//                value = evecs_->get(I, root1_) * vJ2 * sign;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + p * no2_ + q * no_ + r] += value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + p * no2_ + q * no_ + r] -= value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + p * no2_ + q * no_ + r] -= value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + p * no2_ + q * no_ + r] += value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + p * no2_ + q * no_ + r] -= value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + p * no2_ + q * no_ + r] += value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + p * no2_ + r * no_ + q] -= value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + p * no2_ + r * no_ + q] += value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + p * no2_ + r * no_ + q] += value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + p * no2_ + r * no_ + q] -= value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + p * no2_ + r * no_ + q] += value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + p * no2_ + r * no_ + q] -= value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + q * no2_ + p * no_ + r] -= value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + q * no2_ + p * no_ + r] += value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + q * no2_ + p * no_ + r] += value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + q * no2_ + p * no_ + r] -= value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + q * no2_ + p * no_ + r] += value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + q * no2_ + p * no_ + r] -= value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + q * no2_ + r * no_ + p] += value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + q * no2_ + r * no_ + p] -= value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + q * no2_ + r * no_ + p] -= value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + q * no2_ + r * no_ + p] += value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + q * no2_ + r * no_ + p] -= value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + q * no2_ + r * no_ + p] += value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + r * no2_ + p * no_ + q] += value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + r * no2_ + p * no_ + q] -= value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + r * no2_ + p * no_ + q] -= value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + r * no2_ + p * no_ + q] += value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + r * no2_ + p * no_ + q] -= value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + r * no2_ + p * no_ + q] += value;
//
//                tpdm3[s * no5_ + t * no4_ + u * no3_ + r * no2_ + q * no_ + p] -= value;
//                tpdm3[s * no5_ + u * no4_ + t * no3_ + r * no2_ + q * no_ + p] += value;
//                tpdm3[u * no5_ + t * no4_ + s * no3_ + r * no2_ + q * no_ + p] += value;
//                tpdm3[u * no5_ + s * no4_ + t * no3_ + r * no2_ + q * no_ + p] -= value;
//                tpdm3[t * no5_ + s * no4_ + u * no3_ + r * no2_ + q * no_ + p] += value;
//                tpdm3[t * no5_ + u * no4_ + s * no3_ + r * no2_ + q * no_ + p] -= value;
//            }
//        }
//    }

    if (print_) {
        outfile->Printf("\n  Time spent building 3-rdm: %.3e seconds", build.stop());
    }
}

} // namespace forte
