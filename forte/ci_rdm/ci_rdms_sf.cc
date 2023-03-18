/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
    auto t_one = one.stop();
    if (print_)
        outfile->Printf("\n  Time spent forming 1-map:   %1.6f", t_one);

    opdm.assign(norb2_, 0.0);
    timer build("Build SF 1-RDM");

    for (size_t J = 0; J < dim_space_; ++J) {
        auto vJ = evecs_->get(J, root2_);
        for (auto& aJ_mo_sign : a_ann_list_[J]) {
            const auto [aJ_add, _p] = aJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_p = _p < 0;
            for (auto& aaJ_mo_sign : a_cre_list_[aJ_add]) {
                const auto [I, _q] = aaJ_mo_sign;
                auto q = std::abs(_q) - 1;
                auto sign = ((_q < 0) == sign_p) ? 1 : -1;
                opdm[q * norb_ + p] += evecs_->get(I, root1_) * vJ * sign;
            }
        }
        for (auto& bJ_mo_sign : b_ann_list_[J]) {
            const auto [bJ_add, _p] = bJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_p = _p < 0;
            for (auto& bbJ_mo_sign : b_cre_list_[bJ_add]) {
                const auto [I, _q] = bbJ_mo_sign;
                auto q = std::abs(_q) - 1;
                auto sign = ((_q < 0) == sign_p) ? 1 : -1;
                opdm[q * norb_ + p] += evecs_->get(I, root1_) * vJ * sign;
            }
        }
    }

    auto t_build = build.stop();
    if (print_)
        outfile->Printf("\n  Time spent building 1-rdm:   %1.6f", t_build);
}

void CI_RDMS::compute_1rdm_sf_op(std::vector<double>& opdm) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->op_s_lists(wfn_);
    compute_1rdm_sf_op(opdm, op);
}

void CI_RDMS::compute_1rdm_sf_op(std::vector<double>& opdm,
                                 std::shared_ptr<DeterminantSubstitutionLists> op) {
    opdm.assign(norb2_, 0.0);
    timer build("Build SF 1-RDM");

    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t J = 0; J < dim_space_; ++J) {
        double cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
        for (int pp : dets[J].get_alfa_occ(norb_)) {
            opdm[pp * norb_ + pp] += cJ_sq;
        }
        for (int pp : dets[J].get_beta_occ(norb_)) {
            opdm[pp * norb_ + pp] += cJ_sq;
        }
    }

    for (const auto& coupled_dets : op->a_list_) {
        for (size_t a = 0, coupled_dets_size = coupled_dets.size(); a < coupled_dets_size; ++a) {
            const auto [I, _p] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_p = _p < 0;

            auto vI1 = evecs_->get(I, root1_);
            auto vI2 = evecs_->get(I, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [J, _q] = coupled_dets[b];
                auto q = std::abs(_q) - 1;
                auto sign = ((_q < 0) == sign_p) ? 1 : -1;

                opdm[p * norb_ + q] += vI1 * evecs_->get(J, root2_) * sign;
                opdm[q * norb_ + p] += evecs_->get(J, root1_) * vI2 * sign;
            }
        }
    }
    for (const auto& coupled_dets : op->b_list_) {
        for (size_t a = 0, coupled_dets_size = coupled_dets.size(); a < coupled_dets_size; ++a) {
            const auto [I, _p] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_p = _p < 0;

            auto vI1 = evecs_->get(I, root1_);
            auto vI2 = evecs_->get(I, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [J, _q] = coupled_dets[b];
                auto q = std::abs(_q) - 1;
                auto sign = ((_q < 0) == sign_p) ? 1 : -1;

                opdm[p * norb_ + q] += vI1 * evecs_->get(J, root2_) * sign;
                opdm[q * norb_ + p] += evecs_->get(J, root1_) * vI2 * sign;
            }
        }
    }

    auto t_build = build.stop();
    if (print_) {
        outfile->Printf("\n  Time spent building 1-rdm: %.3e seconds", t_build);
    }
}

void CI_RDMS::compute_2rdm_sf(std::vector<double>& tpdm) {
    timer two("Build 2 Substitution Lists");
    get_two_map();
    auto t_two = two.stop();
    if (print_) {
        outfile->Printf("\n  Time spent forming 2-map:   %1.6f", t_two);
    }

    tpdm.assign(norb4_, 0.0);
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

                tpdm[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                tpdm[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                tpdm[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                tpdm[q * norb3_ + p * norb2_ + s * norb_ + r] += value;
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

                tpdm[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                tpdm[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                tpdm[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                tpdm[q * norb3_ + p * norb2_ + s * norb_ + r] += value;
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

                tpdm[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                tpdm[q * norb3_ + p * norb2_ + s * norb_ + r] += value;
            }
        }
    }

    auto t_build = build.stop();
    if (print_)
        outfile->Printf("\n  Time spent building 2-rdm:   %1.6f", t_build);
}

void CI_RDMS::compute_2rdm_sf_op(std::vector<double>& tpdm) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->tp_s_lists(wfn_);
    compute_2rdm_sf_op(tpdm, op);
}

void CI_RDMS::compute_2rdm_sf_op(std::vector<double>& tpdm,
                                 std::shared_ptr<DeterminantSubstitutionLists> op) {
    tpdm.assign(norb4_, 0.0);
    timer build("Build SF 2-RDM");

    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t J = 0; J < dim_space_; ++J) {
        auto cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
        auto aocc = dets[J].get_alfa_occ(norb_);
        auto bocc = dets[J].get_beta_occ(norb_);

        auto naocc = aocc.size();
        auto nbocc = bocc.size();

        for (size_t p = 0; p < naocc; ++p) {
            auto pp = aocc[p];
            for (size_t q = p + 1; q < naocc; ++q) {
                auto qq = aocc[q];

                tpdm[pp * norb3_ + qq * norb2_ + pp * norb_ + qq] += cJ_sq;
                tpdm[qq * norb3_ + pp * norb2_ + pp * norb_ + qq] -= cJ_sq;

                tpdm[qq * norb3_ + pp * norb2_ + qq * norb_ + pp] += cJ_sq;
                tpdm[pp * norb3_ + qq * norb2_ + qq * norb_ + pp] -= cJ_sq;
            }
        }

        for (size_t p = 0; p < nbocc; ++p) {
            auto pp = bocc[p];
            for (size_t q = p + 1; q < nbocc; ++q) {
                auto qq = bocc[q];

                tpdm[pp * norb3_ + qq * norb2_ + pp * norb_ + qq] += cJ_sq;
                tpdm[qq * norb3_ + pp * norb2_ + pp * norb_ + qq] -= cJ_sq;

                tpdm[qq * norb3_ + pp * norb2_ + qq * norb_ + pp] += cJ_sq;
                tpdm[pp * norb3_ + qq * norb2_ + qq * norb_ + pp] -= cJ_sq;
            }
        }

        for (size_t p = 0; p < naocc; ++p) {
            auto pp = aocc[p];
            for (size_t q = 0; q < nbocc; ++q) {
                auto qq = bocc[q];

                tpdm[pp * norb3_ + qq * norb2_ + pp * norb_ + qq] += cJ_sq;
                tpdm[qq * norb3_ + pp * norb2_ + qq * norb_ + pp] += cJ_sq;
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
                tpdm[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                tpdm[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                tpdm[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                tpdm[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tpdm[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                tpdm[s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm[r * norb3_ + s * norb2_ + q * norb_ + p] -= value;
                tpdm[s * norb3_ + r * norb2_ + q * norb_ + p] += value;
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
                tpdm[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                tpdm[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                tpdm[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                tpdm[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tpdm[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                tpdm[s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm[r * norb3_ + s * norb2_ + q * norb_ + p] -= value;
                tpdm[s * norb3_ + r * norb2_ + q * norb_ + p] += value;
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
                tpdm[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                tpdm[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tpdm[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                tpdm[s * norb3_ + r * norb2_ + q * norb_ + p] += value;
            }
        }
    }

    auto t_build = build.stop();
    if (print_) {
        outfile->Printf("\n  Time spent building 2-rdm: %.3e seconds", t_build);
    }
}

void CI_RDMS::compute_3rdm_sf(std::vector<double>& tpdm3) {
    timer three("Build 3 Substitution Lists");
    get_three_map();
    auto t_three_lists = three.stop();
    if (print_) {
        outfile->Printf("\n  Time spent forming 3-map:   %1.6f", t_three_lists);
    }

    timer build("Build SF 3-RDM");
    tpdm3.assign(norb6_, 0.0);

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

                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -= value;
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

                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] += value;

                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] += value;

                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] += value;
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

                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] += value;

                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] += value;

                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] += value;
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

                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -= value;
            }
        }
    }

    auto t_build = build.stop();
    if (print_)
        outfile->Printf("\n  Time spent building 3-rdm:   %1.6f", t_build);
}

void CI_RDMS::compute_3rdm_sf_op(std::vector<double>& tpdm3) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->three_s_lists(wfn_);

    tpdm3.assign(norb6_, 0.0);
    timer build("Build SF 3-RDM");

    // Build the diagonal part
    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t I = 0; I < dim_space_; ++I) {
        double cI_sq = evecs_->get(I, root1_) * evecs_->get(I, root2_);

        auto aocc = dets[I].get_alfa_occ(norb_);
        auto bocc = dets[I].get_beta_occ(norb_);
        auto na = aocc.size();
        auto nb = bocc.size();

        for (size_t p = 0; p < na; ++p) {
            auto pp = aocc[p];
            for (size_t q = p + 1; q < na; ++q) {
                auto qq = aocc[q];
                for (size_t r = q + 1; r < na; ++r) {
                    auto rr = aocc[r];

                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] += cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] -= cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] += cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] -= cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] += cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] -= cI_sq;

                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] -= cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] += cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] -= cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] += cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] -= cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] += cI_sq;

                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] += cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] -= cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] += cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] -= cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] += cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] -= cI_sq;

                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] -= cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] += cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] -= cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] += cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] -= cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] += cI_sq;

                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] += cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] -= cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] += cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] -= cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] += cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] -= cI_sq;

                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] -= cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] += cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] -= cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] += cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] -= cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] += cI_sq;
                }
            }
        }

        for (size_t p = 0; p < na; ++p) {
            auto pp = aocc[p];
            for (size_t q = p + 1; q < na; ++q) {
                auto qq = aocc[q];
                for (size_t r = 0; r < nb; ++r) {
                    auto rr = bocc[r];

                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] += cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] -= cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] -= cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] += cI_sq;

                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] += cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] -= cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] -= cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] += cI_sq;

                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] += cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] -= cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] -= cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] += cI_sq;
                }
            }
        }

        for (size_t p = 0; p < na; ++p) {
            auto pp = aocc[p];
            for (size_t q = 0; q < nb; ++q) {
                auto qq = bocc[q];
                for (size_t r = q + 1; r < nb; ++r) {
                    auto rr = bocc[r];

                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] += cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] -= cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] -= cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] += cI_sq;

                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] += cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] -= cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] -= cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] += cI_sq;

                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] += cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] -= cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] -= cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] += cI_sq;
                }
            }
        }

        for (size_t p = 0; p < nb; ++p) {
            auto pp = bocc[p];
            for (size_t q = p + 1; q < nb; ++q) {
                auto qq = bocc[q];
                for (size_t r = q + 1; r < nb; ++r) {
                    auto rr = bocc[r];

                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] += cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] -= cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] += cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] -= cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] += cI_sq;
                    tpdm3[pp * norb5_ + qq * norb4_ + rr * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] -= cI_sq;

                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] -= cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] += cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] -= cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] += cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] -= cI_sq;
                    tpdm3[pp * norb5_ + rr * norb4_ + qq * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] += cI_sq;

                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] += cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] -= cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] += cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] -= cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] += cI_sq;
                    tpdm3[rr * norb5_ + pp * norb4_ + qq * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] -= cI_sq;

                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] -= cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] += cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] -= cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] += cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] -= cI_sq;
                    tpdm3[rr * norb5_ + qq * norb4_ + pp * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] += cI_sq;

                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] += cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] -= cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] += cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] -= cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] += cI_sq;
                    tpdm3[qq * norb5_ + rr * norb4_ + pp * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] -= cI_sq;

                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + pp * norb2_ + qq * norb_ +
                          rr] -= cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + pp * norb2_ + rr * norb_ +
                          qq] += cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + rr * norb2_ + pp * norb_ +
                          qq] -= cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + rr * norb2_ + qq * norb_ +
                          pp] += cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + qq * norb2_ + rr * norb_ +
                          pp] -= cI_sq;
                    tpdm3[qq * norb5_ + pp * norb4_ + rr * norb3_ + qq * norb2_ + pp * norb_ +
                          rr] += cI_sq;
                }
            }
        }
    }

    // Build the off-diagonal part

    // aaa aaa
    for (const auto& coupled_dets : op->aaa_list_) {
        auto coupled_dets_size = coupled_dets.size();

        for (size_t a = 0; a < coupled_dets_size; ++a) {
            const auto [J, _p, q, r] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            auto vJ1 = evecs_->get(J, root1_);
            auto vJ2 = evecs_->get(J, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [I, _s, t, u] = coupled_dets[b];
                auto s = std::abs(_s) - 1;
                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;

                auto value = vJ1 * evecs_->get(I, root2_) * sign;

                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                value = evecs_->get(I, root1_) * vJ2 * sign;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] += value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] -= value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + p * norb2_ + q * norb_ + r] -= value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] += value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] -= value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + q * norb_ + r] += value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] -= value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] += value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] += value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] -= value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] += value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] -= value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] -= value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] += value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + p * norb_ + r] += value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] -= value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] += value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + p * norb_ + r] -= value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + r * norb_ + p] += value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] -= value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] -= value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] += value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + r * norb_ + p] -= value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] += value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] += value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] += value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] += value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + r * norb2_ + q * norb_ + p] -= value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] += value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] += value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] -= value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + q * norb_ + p] += value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] -= value;
            }
        }
    }

    // aab aab
    for (const auto& coupled_dets : op->aab_list_) {
        auto coupled_dets_size = coupled_dets.size();

        for (size_t a = 0; a < coupled_dets_size; ++a) {
            const auto [J, _p, q, r] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            auto vJ1 = evecs_->get(J, root1_);
            auto vJ2 = evecs_->get(J, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [I, _s, t, u] = coupled_dets[b];
                auto s = std::abs(_s) - 1;
                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;

                auto value = vJ1 * evecs_->get(I, root2_) * sign;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] += value;

                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] += value;

                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] += value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] -= value;
                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] -= value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] += value;

                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] += value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] -= value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] += value;

                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] += value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] -= value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] -= value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] += value;
            }
        }
    }

    // abb abb
    for (const auto& coupled_dets : op->abb_list_) {
        auto coupled_dets_size = coupled_dets.size();

        for (size_t a = 0; a < coupled_dets_size; ++a) {
            const auto [J, _p, q, r] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            auto vJ1 = evecs_->get(J, root1_);
            auto vJ2 = evecs_->get(J, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [I, _s, t, u] = coupled_dets[b];
                auto s = std::abs(_s) - 1;
                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;

                auto value = vJ1 * evecs_->get(I, root2_) * sign;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] += value;

                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] += value;

                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] += value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] -= value;
                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] -= value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] += value;

                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] += value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] -= value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] += value;

                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] += value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] -= value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] -= value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] += value;
            }
        }
    }

    // bbb bbb
    for (const auto& coupled_dets : op->bbb_list_) {
        auto coupled_dets_size = coupled_dets.size();

        for (size_t a = 0; a < coupled_dets_size; ++a) {
            const auto [J, _p, q, r] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            auto vJ1 = evecs_->get(J, root1_);
            auto vJ2 = evecs_->get(J, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [I, _s, t, u] = coupled_dets[b];
                auto s = std::abs(_s) - 1;
                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;

                auto value = vJ1 * evecs_->get(I, root2_) * sign;

                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] += value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] += value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -= value;
                tpdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] += value;

                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] -= value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -= value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] += value;
                tpdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -= value;

                value = evecs_->get(I, root1_) * vJ2 * sign;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] += value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] -= value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + p * norb2_ + q * norb_ + r] -= value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] += value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] -= value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + q * norb_ + r] += value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] -= value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] += value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] += value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] -= value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] += value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] -= value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] -= value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] += value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + p * norb_ + r] += value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] -= value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] += value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + p * norb_ + r] -= value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + r * norb_ + p] += value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] -= value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] -= value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] += value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + r * norb_ + p] -= value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] += value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] += value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] += value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] += value;

                tpdm3[s * norb5_ + t * norb4_ + u * norb3_ + r * norb2_ + q * norb_ + p] -= value;
                tpdm3[s * norb5_ + u * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] += value;
                tpdm3[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] += value;
                tpdm3[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] -= value;
                tpdm3[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + q * norb_ + p] += value;
                tpdm3[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] -= value;
            }
        }
    }

    auto t_build = build.stop();
    if (print_) {
        outfile->Printf("\n  Time spent building 3-rdm: %.3e seconds", t_build);
    }
}

void CI_RDMS::compute_rdms_dynamic_sf(std::vector<double>& rdm1, std::vector<double>& rdm2) {
    rdm1.assign(norb2_, 0.0);
    rdm2.assign(norb4_, 0.0);

    SortedStringList a_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Alpha);
    SortedStringList b_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Beta);
    const std::vector<String>& sorted_bstr = b_sorted_string_list_.sorted_half_dets();
    size_t num_bstr = sorted_bstr.size();
    const auto& sorted_b_dets = b_sorted_string_list_.sorted_dets();
    const auto& sorted_a_dets = a_sorted_string_list_.sorted_dets();
    local_timer diag;
    //*-  Diagonal Contributions  -*//
    for (size_t I = 0; I < dim_space_; ++I) {
        size_t Ia = b_sorted_string_list_.add(I);
        double CIa = evecs_->get(Ia, root1_) * evecs_->get(Ia, root2_);
        String det_a = sorted_b_dets[I].get_alfa_bits();
        String det_b = sorted_b_dets[I].get_beta_bits();

        for (size_t nda = 0; nda < na_; ++nda) {
            size_t p = det_a.find_first_one();
            rdm1[p * norb_ + p] += CIa;

            String det_ac(det_a);
            det_a.clear_first_one();
            for (size_t ndaa = nda; ndaa < na_; ++ndaa) {
                size_t q = det_ac.find_first_one();
                // aa 2-rdm
                rdm2[p * norb3_ + q * norb2_ + p * norb_ + q] += CIa;
                rdm2[q * norb3_ + p * norb2_ + q * norb_ + p] += CIa;
                rdm2[p * norb3_ + q * norb2_ + q * norb_ + p] -= CIa;
                rdm2[q * norb3_ + p * norb2_ + p * norb_ + q] -= CIa;
                det_ac.clear_first_one();
            }

            String det_bc(det_b);
            for (size_t n = 0; n < nb_; ++n) {
                size_t q = det_bc.find_first_one();
                rdm2[p * norb3_ + q * norb2_ + p * norb_ + q] += CIa;
                rdm2[q * norb3_ + p * norb2_ + q * norb_ + p] += CIa;
                det_bc.clear_first_one();
            }
        }
        det_a = sorted_b_dets[I].get_alfa_bits();
        det_b = sorted_b_dets[I].get_beta_bits();
        size_t Ib = a_sorted_string_list_.add(I);
        double CIb = evecs_->get(Ib, root1_) * evecs_->get(Ib, root2_);
        for (size_t ndb = 0; ndb < nb_; ++ndb) {
            size_t p = det_b.find_first_one();

            // b -1rdm
            rdm1[p * norb_ + p] += CIb;
            String det_bc(det_b);
            for (size_t ndbb = ndb; ndbb < nb_; ++ndbb) {
                size_t q = det_bc.find_first_one();
                // bb-2rdm
                rdm2[p * norb3_ + q * norb2_ + p * norb_ + q] += CIb;
                rdm2[q * norb3_ + p * norb2_ + q * norb_ + p] += CIb;
                rdm2[p * norb3_ + q * norb2_ + q * norb_ + p] -= CIb;
                rdm2[q * norb3_ + p * norb2_ + p * norb_ + q] -= CIb;
                det_bc.clear_first_one();
            }
            det_b.clear_first_one();
        }
    }
    outfile->Printf("\n  Diag takes %1.6f", diag.get());

    local_timer aaa;
    //-* All Alpha RDMs *-//

    // loop through all beta strings
    for (size_t bstr = 0; bstr < num_bstr; ++bstr) {
        const String& Ib = sorted_bstr[bstr];
        const auto& range_I = b_sorted_string_list_.range(Ib);

        String Ia;
        String Ja;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Double loop through determinants with same beta string
        for (size_t I = first_I; I < last_I; ++I) {
            Ia = sorted_b_dets[I].get_alfa_bits();
            double CI = evecs_->get(b_sorted_string_list_.add(I), root1_);
            for (size_t J = I + 1; J < last_I; ++J) {
                Ja = sorted_b_dets[J].get_alfa_bits();
                String IJa = Ia ^ Ja;

                int ndiff = IJa.count();

                if (ndiff == 2) {
                    // 1-rdm
                    String Ia_sub = Ia & IJa;
                    u_int64_t p = Ia_sub.find_first_one();
                    String Ja_sub = Ja & IJa;
                    u_int64_t q = Ja_sub.find_first_one();

                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ia.slater_sign(p, q);
                    rdm1[p * norb_ + q] += value;
                    rdm1[q * norb_ + p] += value;

                    // 2-rdm
                    auto Iac = Ia;
                    Iac ^= Ia_sub;
                    for (size_t nbit_a = 1; nbit_a < na_; nbit_a++) {
                        uint64_t m = Iac.find_first_one();
                        rdm2[p * norb3_ + m * norb2_ + q * norb_ + m] += value;
                        rdm2[m * norb3_ + p * norb2_ + q * norb_ + m] -= value;
                        rdm2[m * norb3_ + p * norb2_ + m * norb_ + q] += value;
                        rdm2[p * norb3_ + m * norb2_ + m * norb_ + q] -= value;

                        rdm2[q * norb3_ + m * norb2_ + p * norb_ + m] += value;
                        rdm2[m * norb3_ + q * norb2_ + p * norb_ + m] -= value;
                        rdm2[m * norb3_ + q * norb2_ + m * norb_ + p] += value;
                        rdm2[q * norb3_ + m * norb2_ + m * norb_ + p] -= value;
                        Iac.clear_first_one();
                    }
                    auto Ibc = Ib;
                    for (size_t nidx = 0; nidx < nb_; ++nidx) {
                        uint64_t n = Ibc.find_first_one();
                        rdm2[p * norb3_ + n * norb2_ + q * norb_ + n] += value;
                        rdm2[q * norb3_ + n * norb2_ + p * norb_ + n] += value;
                        rdm2[n * norb3_ + p * norb2_ + n * norb_ + q] += value;
                        rdm2[n * norb3_ + q * norb2_ + n * norb_ + p] += value;
                        Ibc.clear_first_one();
                    }
                } else if (ndiff == 4) {
                    // 2-rdm
                    auto Ia_sub = Ia & IJa;
                    uint64_t p = Ia_sub.find_first_one();
                    Ia_sub.clear_first_one();
                    uint64_t q = Ia_sub.find_first_one();

                    auto Ja_sub = Ja & IJa;
                    uint64_t r = Ja_sub.find_first_one();
                    Ja_sub.clear_first_one();
                    uint64_t s = Ja_sub.find_first_one();

                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ia.slater_sign(p, q) * Ja.slater_sign(r, s);

                    rdm2[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                    rdm2[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                    rdm2[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                    rdm2[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                    rdm2[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                    rdm2[s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                    rdm2[r * norb3_ + s * norb2_ + q * norb_ + p] -= value;
                    rdm2[s * norb3_ + r * norb2_ + q * norb_ + p] += value;
                }
            }
        }
    }
    outfile->Printf("\n all alpha takes %1.6f", aaa.get());

    //- All beta RDMs -//
    local_timer bbb;
    // loop through all alpha strings
    const std::vector<String>& sorted_astr = a_sorted_string_list_.sorted_half_dets();
    size_t num_astr = sorted_astr.size();
    for (size_t astr = 0; astr < num_astr; ++astr) {
        const String& Ia = sorted_astr[astr];
        const auto& range_I = a_sorted_string_list_.range(Ia);

        String Ib;
        String Jb;
        String IJb;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Double loop through determinants with same alpha string
        for (size_t I = first_I; I < last_I; ++I) {
            Ib = sorted_a_dets[I].get_beta_bits();
            double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
            for (size_t J = I + 1; J < last_I; ++J) {
                Jb = sorted_a_dets[J].get_beta_bits();
                IJb = Ib ^ Jb;
                int ndiff = IJb.count();

                if (ndiff == 2) {
                    auto Ib_sub = Ib & IJb;
                    uint64_t p = Ib_sub.find_first_one();
                    auto Jb_sub = Jb & IJb;
                    uint64_t q = Jb_sub.find_first_one();
                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);

                    double value = Csq * Ib.slater_sign(p, q);
                    rdm1[p * norb_ + q] += value;
                    rdm1[q * norb_ + p] += value;
                    auto Ibc = Ib;
                    Ibc ^= Ib_sub;
                    for (size_t ndb = 1; ndb < nb_; ++ndb) {
                        uint64_t m = Ibc.find_first_one();
                        rdm2[p * norb3_ + m * norb2_ + q * norb_ + m] += value;
                        rdm2[m * norb3_ + p * norb2_ + q * norb_ + m] -= value;
                        rdm2[m * norb3_ + p * norb2_ + m * norb_ + q] += value;
                        rdm2[p * norb3_ + m * norb2_ + m * norb_ + q] -= value;

                        rdm2[q * norb3_ + m * norb2_ + p * norb_ + m] += value;
                        rdm2[m * norb3_ + q * norb2_ + p * norb_ + m] -= value;
                        rdm2[m * norb3_ + q * norb2_ + m * norb_ + p] += value;
                        rdm2[q * norb3_ + m * norb2_ + m * norb_ + p] -= value;
                        Ibc.clear_first_one();
                    }
                    auto Iac = Ia;
                    for (size_t nidx = 0; nidx < na_; ++nidx) {
                        uint64_t n = Iac.find_first_one();
                        rdm2[n * norb3_ + p * norb2_ + n * norb_ + q] += value;
                        rdm2[n * norb3_ + q * norb2_ + n * norb_ + p] += value;
                        rdm2[p * norb3_ + n * norb2_ + q * norb_ + n] += value;
                        rdm2[q * norb3_ + n * norb2_ + p * norb_ + n] += value;
                        Iac.clear_first_one();
                    }
                } else if (ndiff == 4) {
                    auto Ib_sub = Ib & IJb;
                    uint64_t p = Ib_sub.find_first_one();
                    Ib_sub.clear_first_one();
                    uint64_t q = Ib_sub.find_first_one();

                    auto Jb_sub = Jb & IJb;
                    uint64_t r = Jb_sub.find_first_one();
                    Jb_sub.clear_first_one();
                    uint64_t s = Jb_sub.find_first_one();

                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ib.slater_sign(p, q) * Jb.slater_sign(r, s);
                    rdm2[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                    rdm2[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                    rdm2[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                    rdm2[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                    rdm2[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                    rdm2[s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                    rdm2[r * norb3_ + s * norb2_ + q * norb_ + p] -= value;
                    rdm2[s * norb3_ + r * norb2_ + q * norb_ + p] += value;
                }
            }
        }
    }
    outfile->Printf("\n all beta takes %1.6f", bbb.get());

    //- Alpha-Beta RDMs -//
    local_timer mix;
    double d2 = 0.0;
    for (auto& detIa : sorted_astr) {
        const auto& range_I = a_sorted_string_list_.range(detIa);
        String detIJa_common;
        String Ib;
        String Jb;
        String IJb;
        for (auto& detJa : sorted_astr) {
            detIJa_common = detIa ^ detJa;
            int ndiff = detIJa_common.count();
            if (ndiff == 2) {
                local_timer t2;
                auto Ia_d = detIa & detIJa_common;
                uint64_t p = Ia_d.find_first_one();
                auto Ja_d = detJa & detIJa_common;
                uint64_t s = Ja_d.find_first_one();

                const auto& range_J = a_sorted_string_list_.range(detJa);
                size_t first_I = range_I.first;
                size_t last_I = range_I.second;
                size_t first_J = range_J.first;
                size_t last_J = range_J.second;
                double sign_Ips = detIa.slater_sign(p, s);
                double sign_IJ = detIa.slater_sign(p) * detJa.slater_sign(s);
                for (size_t I = first_I; I < last_I; ++I) {
                    Ib = sorted_a_dets[I].get_beta_bits();
                    double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
                    for (size_t J = first_J; J < last_J; ++J) {
                        Jb = sorted_a_dets[J].get_beta_bits();
                        IJb = Ib ^ Jb;
                        int nbdiff = IJb.count();
                        if (nbdiff == 2) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            auto Ib_sub = Ib & IJb;
                            uint64_t q = Ib_sub.find_first_one();
                            auto Jb_sub = Jb & IJb;
                            uint64_t r = Jb_sub.find_first_one();

                            double value =
                                Csq * sign_Ips * Ib.slater_sign(q, r); // * ui64_slater_sign(Jb,r);
                            rdm2[p * norb3_ + q * norb2_ + s * norb_ + r] += value;
                            rdm2[q * norb3_ + p * norb2_ + r * norb_ + s] += value;
                        }
                    }
                }
                d2 += t2.get();
            }
        }
    }
    outfile->Printf("\n  2dif: %1.6f", d2);
    outfile->Printf("\n all alpha/beta takes %1.6f", mix.get());
}

void CI_RDMS::compute_rdms_dynamic_sf(std::vector<double>& rdm1, std::vector<double>& rdm2,
                                      std::vector<double>& rdm3) {
    rdm1.assign(norb2_, 0.0);
    rdm2.assign(norb4_, 0.0);
    rdm3.assign(norb6_, 0.0);

    SortedStringList a_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Alpha);
    SortedStringList b_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Beta);
    const std::vector<String>& sorted_bstr = b_sorted_string_list_.sorted_half_dets();
    size_t num_bstr = sorted_bstr.size();
    const auto& sorted_b_dets = b_sorted_string_list_.sorted_dets();
    const auto& sorted_a_dets = a_sorted_string_list_.sorted_dets();
    local_timer diag;
    //*-  Diagonal Contributions  -*//
    for (size_t I = 0; I < dim_space_; ++I) {
        size_t Ia = b_sorted_string_list_.add(I);
        double CIa = evecs_->get(Ia, root1_) * evecs_->get(Ia, root2_);
        String det_a = sorted_b_dets[I].get_alfa_bits();
        String det_b = sorted_b_dets[I].get_beta_bits();

        for (size_t nda = 0; nda < na_; ++nda) {
            size_t p = det_a.find_first_one();
            rdm1[p * norb_ + p] += CIa;

            String det_ac(det_a);
            det_a.clear_first_one();
            for (size_t ndaa = nda; ndaa < na_; ++ndaa) {
                size_t q = det_ac.find_first_one();
                // aa 2-rdm
                rdm2[p * norb3_ + q * norb2_ + p * norb_ + q] += CIa;
                rdm2[q * norb3_ + p * norb2_ + q * norb_ + p] += CIa;
                rdm2[p * norb3_ + q * norb2_ + q * norb_ + p] -= CIa;
                rdm2[q * norb3_ + p * norb2_ + p * norb_ + q] -= CIa;
                det_ac.clear_first_one();

                // aaa 3rdm
                String det_acc(det_ac);
                for (size_t ndaaa = ndaa + 1; ndaaa < na_; ++ndaaa) {
                    size_t r = det_acc.find_first_one();
                    fill_3rdm(rdm3, CIa, p, q, r, p, q, r, true);
                    det_acc.clear_first_one();
                }

                // aab 3rdm
                String det_bc(det_b);
                for (size_t n = 0; n < nb_; ++n) {
                    size_t r = det_bc.find_first_one();

                    rdm3[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] += CIa;
                    rdm3[p * norb5_ + q * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] -= CIa;
                    rdm3[q * norb5_ + p * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] -= CIa;
                    rdm3[q * norb5_ + p * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] += CIa;

                    rdm3[r * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + q * norb_ + p] += CIa;
                    rdm3[r * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + p * norb_ + q] -= CIa;
                    rdm3[r * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + q * norb_ + p] -= CIa;
                    rdm3[r * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + p * norb_ + q] += CIa;

                    rdm3[p * norb5_ + r * norb4_ + q * norb3_ + p * norb2_ + r * norb_ + q] += CIa;
                    rdm3[p * norb5_ + r * norb4_ + q * norb3_ + q * norb2_ + r * norb_ + p] -= CIa;
                    rdm3[q * norb5_ + r * norb4_ + p * norb3_ + p * norb2_ + r * norb_ + q] -= CIa;
                    rdm3[q * norb5_ + r * norb4_ + p * norb3_ + q * norb2_ + r * norb_ + p] += CIa;

                    det_bc.clear_first_one();
                }
            }

            String det_bc(det_b);
            for (size_t n = 0; n < nb_; ++n) {
                size_t q = det_bc.find_first_one();
                rdm2[p * norb3_ + q * norb2_ + p * norb_ + q] += CIa;
                rdm2[q * norb3_ + p * norb2_ + q * norb_ + p] += CIa;
                det_bc.clear_first_one();
            }
        }
        det_a = sorted_b_dets[I].get_alfa_bits();
        det_b = sorted_b_dets[I].get_beta_bits();
        size_t Ib = a_sorted_string_list_.add(I);
        double CIb = evecs_->get(Ib, root1_) * evecs_->get(Ib, root2_);
        for (size_t ndb = 0; ndb < nb_; ++ndb) {
            size_t p = det_b.find_first_one();

            // b -1rdm
            rdm1[p * norb_ + p] += CIb;
            String det_bc(det_b);
            for (size_t ndbb = ndb; ndbb < nb_; ++ndbb) {
                size_t q = det_bc.find_first_one();
                // bb-2rdm
                rdm2[p * norb3_ + q * norb2_ + p * norb_ + q] += CIb;
                rdm2[q * norb3_ + p * norb2_ + q * norb_ + p] += CIb;
                rdm2[p * norb3_ + q * norb2_ + q * norb_ + p] -= CIb;
                rdm2[q * norb3_ + p * norb2_ + p * norb_ + q] -= CIb;
                det_bc.clear_first_one();

                // bbb-3rdm
                String det_bcc(det_bc);
                for (size_t ndbbb = ndbb + 1; ndbbb < nb_; ++ndbbb) {
                    size_t r = det_bcc.find_first_one();
                    fill_3rdm(rdm3, CIa, p, q, r, p, q, r, true);
                    det_bcc.clear_first_one();
                }

                // abb - 3rdm
                String det_ac(det_a);
                for (size_t n = 0; n < na_; ++n) {
                    size_t r = det_ac.find_first_one();

                    rdm3[r * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + p * norb_ + q] += CIb;
                    rdm3[r * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + q * norb_ + p] -= CIb;
                    rdm3[r * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + p * norb_ + q] -= CIb;
                    rdm3[r * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + q * norb_ + p] += CIb;

                    rdm3[p * norb5_ + r * norb4_ + q * norb3_ + p * norb2_ + r * norb_ + q] += CIb;
                    rdm3[p * norb5_ + r * norb4_ + q * norb3_ + q * norb2_ + r * norb_ + p] -= CIb;
                    rdm3[q * norb5_ + r * norb4_ + p * norb3_ + p * norb2_ + r * norb_ + q] -= CIb;
                    rdm3[q * norb5_ + r * norb4_ + p * norb3_ + q * norb2_ + r * norb_ + p] += CIb;

                    rdm3[q * norb5_ + p * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] += CIb;
                    rdm3[q * norb5_ + p * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] -= CIb;
                    rdm3[p * norb5_ + q * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] -= CIb;
                    rdm3[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] += CIb;

                    det_ac.clear_first_one();
                }
            }
            det_b.clear_first_one();
        }
    }
    outfile->Printf("\n  Diag takes %1.6f", diag.get());

    local_timer aaa;
    //-* All Alpha RDMs *-//

    // loop through all beta strings
    for (size_t bstr = 0; bstr < num_bstr; ++bstr) {
        const String& Ib = sorted_bstr[bstr];
        const auto& range_I = b_sorted_string_list_.range(Ib);

        String Ia;
        String Ja;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Double loop through determinants with same beta string
        for (size_t I = first_I; I < last_I; ++I) {
            Ia = sorted_b_dets[I].get_alfa_bits();
            double CI = evecs_->get(b_sorted_string_list_.add(I), root1_);
            for (size_t J = I + 1; J < last_I; ++J) {
                Ja = sorted_b_dets[J].get_alfa_bits();
                String IJa = Ia ^ Ja;

                int ndiff = IJa.count();

                if (ndiff == 2) {
                    // 1-rdm
                    String Ia_sub = Ia & IJa;
                    u_int64_t p = Ia_sub.find_first_one();
                    String Ja_sub = Ja & IJa;
                    u_int64_t q = Ja_sub.find_first_one();

                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ia.slater_sign(p, q);
                    rdm1[p * norb_ + q] += value;
                    rdm1[q * norb_ + p] += value;

                    // 2-rdm
                    auto Iac = Ia;
                    Iac ^= Ia_sub;
                    for (size_t nbit_a = 1; nbit_a < na_; nbit_a++) {
                        uint64_t m = Iac.find_first_one();
                        rdm2[p * norb3_ + m * norb2_ + q * norb_ + m] += value;
                        rdm2[m * norb3_ + p * norb2_ + q * norb_ + m] -= value;
                        rdm2[m * norb3_ + p * norb2_ + m * norb_ + q] += value;
                        rdm2[p * norb3_ + m * norb2_ + m * norb_ + q] -= value;

                        rdm2[q * norb3_ + m * norb2_ + p * norb_ + m] += value;
                        rdm2[m * norb3_ + q * norb2_ + p * norb_ + m] -= value;
                        rdm2[m * norb3_ + q * norb2_ + m * norb_ + p] += value;
                        rdm2[q * norb3_ + m * norb2_ + m * norb_ + p] -= value;
                        Iac.clear_first_one();

                        auto Ibc = Ib;
                        for (size_t idx = 0; idx < nb_; ++idx) {
                            uint64_t n = Ibc.find_first_one();
                            rdm3[p * norb5_ + m * norb4_ + n * norb3_ + q * norb2_ + m * norb_ +
                                 n] += value;
                            rdm3[p * norb5_ + m * norb4_ + n * norb3_ + m * norb2_ + q * norb_ +
                                 n] -= value;
                            rdm3[m * norb5_ + p * norb4_ + n * norb3_ + m * norb2_ + q * norb_ +
                                 n] += value;
                            rdm3[m * norb5_ + p * norb4_ + n * norb3_ + q * norb2_ + m * norb_ +
                                 n] -= value;

                            rdm3[n * norb5_ + m * norb4_ + p * norb3_ + n * norb2_ + m * norb_ +
                                 q] += value;
                            rdm3[n * norb5_ + m * norb4_ + p * norb3_ + n * norb2_ + q * norb_ +
                                 m] -= value;
                            rdm3[n * norb5_ + p * norb4_ + m * norb3_ + n * norb2_ + q * norb_ +
                                 m] += value;
                            rdm3[n * norb5_ + p * norb4_ + m * norb3_ + n * norb2_ + m * norb_ +
                                 q] -= value;

                            rdm3[p * norb5_ + n * norb4_ + m * norb3_ + q * norb2_ + n * norb_ +
                                 m] += value;
                            rdm3[p * norb5_ + n * norb4_ + m * norb3_ + m * norb2_ + n * norb_ +
                                 q] -= value;
                            rdm3[m * norb5_ + n * norb4_ + p * norb3_ + m * norb2_ + n * norb_ +
                                 q] += value;
                            rdm3[m * norb5_ + n * norb4_ + p * norb3_ + q * norb2_ + n * norb_ +
                                 m] -= value;

                            rdm3[q * norb5_ + m * norb4_ + n * norb3_ + p * norb2_ + m * norb_ +
                                 n] += value;
                            rdm3[q * norb5_ + m * norb4_ + n * norb3_ + m * norb2_ + p * norb_ +
                                 n] -= value;
                            rdm3[m * norb5_ + q * norb4_ + n * norb3_ + m * norb2_ + p * norb_ +
                                 n] += value;
                            rdm3[m * norb5_ + q * norb4_ + n * norb3_ + p * norb2_ + m * norb_ +
                                 n] -= value;

                            rdm3[n * norb5_ + m * norb4_ + q * norb3_ + n * norb2_ + m * norb_ +
                                 p] += value;
                            rdm3[n * norb5_ + m * norb4_ + q * norb3_ + n * norb2_ + p * norb_ +
                                 m] -= value;
                            rdm3[n * norb5_ + q * norb4_ + m * norb3_ + n * norb2_ + p * norb_ +
                                 m] += value;
                            rdm3[n * norb5_ + q * norb4_ + m * norb3_ + n * norb2_ + m * norb_ +
                                 p] -= value;

                            rdm3[q * norb5_ + n * norb4_ + m * norb3_ + p * norb2_ + n * norb_ +
                                 m] += value;
                            rdm3[q * norb5_ + n * norb4_ + m * norb3_ + m * norb2_ + n * norb_ +
                                 p] -= value;
                            rdm3[m * norb5_ + n * norb4_ + q * norb3_ + m * norb2_ + n * norb_ +
                                 p] += value;
                            rdm3[m * norb5_ + n * norb4_ + q * norb3_ + p * norb2_ + n * norb_ +
                                 m] -= value;
                            Ibc.clear_first_one();
                        }
                    }
                    auto Ibc = Ib;
                    for (size_t nidx = 0; nidx < nb_; ++nidx) {
                        uint64_t n = Ibc.find_first_one();
                        rdm2[p * norb3_ + n * norb2_ + q * norb_ + n] += value;
                        rdm2[q * norb3_ + n * norb2_ + p * norb_ + n] += value;
                        rdm2[n * norb3_ + p * norb2_ + n * norb_ + q] += value;
                        rdm2[n * norb3_ + q * norb2_ + n * norb_ + p] += value;
                        Ibc.clear_first_one();

                        String Ibcc = Ibc;
                        for (size_t idx = nidx + 1; idx < nb_; ++idx) {
                            uint64_t m = Ibcc.find_first_one();
                            rdm3[p * norb5_ + m * norb4_ + n * norb3_ + q * norb2_ + m * norb_ +
                                 n] += value;
                            rdm3[p * norb5_ + m * norb4_ + n * norb3_ + q * norb2_ + n * norb_ +
                                 m] -= value;
                            rdm3[p * norb5_ + n * norb4_ + m * norb3_ + q * norb2_ + n * norb_ +
                                 m] += value;
                            rdm3[p * norb5_ + n * norb4_ + m * norb3_ + q * norb2_ + m * norb_ +
                                 n] -= value;

                            rdm3[m * norb5_ + p * norb4_ + n * norb3_ + m * norb2_ + q * norb_ +
                                 n] += value;
                            rdm3[m * norb5_ + p * norb4_ + n * norb3_ + n * norb2_ + q * norb_ +
                                 m] -= value;
                            rdm3[n * norb5_ + p * norb4_ + m * norb3_ + n * norb2_ + q * norb_ +
                                 m] += value;
                            rdm3[n * norb5_ + p * norb4_ + m * norb3_ + m * norb2_ + q * norb_ +
                                 n] -= value;

                            rdm3[n * norb5_ + m * norb4_ + p * norb3_ + n * norb2_ + m * norb_ +
                                 q] += value;
                            rdm3[n * norb5_ + m * norb4_ + p * norb3_ + m * norb2_ + n * norb_ +
                                 q] -= value;
                            rdm3[m * norb5_ + n * norb4_ + p * norb3_ + m * norb2_ + n * norb_ +
                                 q] += value;
                            rdm3[m * norb5_ + n * norb4_ + p * norb3_ + n * norb2_ + m * norb_ +
                                 q] -= value;

                            rdm3[q * norb5_ + m * norb4_ + n * norb3_ + p * norb2_ + m * norb_ +
                                 n] += value;
                            rdm3[q * norb5_ + m * norb4_ + n * norb3_ + p * norb2_ + n * norb_ +
                                 m] -= value;
                            rdm3[q * norb5_ + n * norb4_ + m * norb3_ + p * norb2_ + n * norb_ +
                                 m] += value;
                            rdm3[q * norb5_ + n * norb4_ + m * norb3_ + p * norb2_ + m * norb_ +
                                 n] -= value;

                            rdm3[m * norb5_ + q * norb4_ + n * norb3_ + m * norb2_ + p * norb_ +
                                 n] += value;
                            rdm3[m * norb5_ + q * norb4_ + n * norb3_ + n * norb2_ + p * norb_ +
                                 m] -= value;
                            rdm3[n * norb5_ + q * norb4_ + m * norb3_ + n * norb2_ + p * norb_ +
                                 m] += value;
                            rdm3[n * norb5_ + q * norb4_ + m * norb3_ + m * norb2_ + p * norb_ +
                                 n] -= value;

                            rdm3[n * norb5_ + m * norb4_ + q * norb3_ + n * norb2_ + m * norb_ +
                                 p] += value;
                            rdm3[n * norb5_ + m * norb4_ + q * norb3_ + m * norb2_ + n * norb_ +
                                 p] -= value;
                            rdm3[m * norb5_ + n * norb4_ + q * norb3_ + m * norb2_ + n * norb_ +
                                 p] += value;
                            rdm3[m * norb5_ + n * norb4_ + q * norb3_ + n * norb2_ + m * norb_ +
                                 p] -= value;
                            Ibcc.clear_first_one();
                        }
                    }
                    // 3-rdm
                    String Iacc = Ia ^ Ia_sub;
                    for (size_t id = 1; id < na_; ++id) {
                        uint64_t n = Iacc.find_first_one();
                        String I_n(Iacc);
                        I_n.clear_first_one(); // TODO: not clear what is going on here (Francesco)
                        for (size_t idd = id + 1; idd < na_; ++idd) {
                            // while( I_n > 0 ){
                            uint64_t m = I_n.find_first_one();
                            fill_3rdm(rdm3, value, p, n, m, q, n, m, false);
                            I_n.clear_first_one();
                        }
                        Iacc.clear_first_one();
                    }

                } else if (ndiff == 4) {
                    // 2-rdm
                    auto Ia_sub = Ia & IJa;
                    uint64_t p = Ia_sub.find_first_one();
                    Ia_sub.clear_first_one();
                    uint64_t q = Ia_sub.find_first_one();

                    auto Ja_sub = Ja & IJa;
                    uint64_t r = Ja_sub.find_first_one();
                    Ja_sub.clear_first_one();
                    uint64_t s = Ja_sub.find_first_one();

                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ia.slater_sign(p, q) * Ja.slater_sign(r, s);

                    rdm2[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                    rdm2[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                    rdm2[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                    rdm2[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                    rdm2[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                    rdm2[s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                    rdm2[r * norb3_ + s * norb2_ + q * norb_ + p] -= value;
                    rdm2[s * norb3_ + r * norb2_ + q * norb_ + p] += value;

                    // 3-rdm
                    String Iac(Ia);
                    Iac ^= Ia_sub;
                    for (size_t nda = 1; nda < na_; ++nda) {
                        uint64_t n = Iac.find_first_one();
                        fill_3rdm(rdm3, value, p, q, n, r, s, n, false);
                        Iac.clear_first_one();
                    }

                    String Ibc = Ib;
                    for (size_t ndb = 0; ndb < nb_; ++ndb) {
                        uint64_t n = Ibc.find_first_one();
                        rdm3[p * norb5_ + q * norb4_ + n * norb3_ + r * norb2_ + s * norb_ + n] +=
                            value;
                        rdm3[p * norb5_ + q * norb4_ + n * norb3_ + s * norb2_ + r * norb_ + n] -=
                            value;
                        rdm3[q * norb5_ + p * norb4_ + n * norb3_ + s * norb2_ + r * norb_ + n] +=
                            value;
                        rdm3[q * norb5_ + p * norb4_ + n * norb3_ + r * norb2_ + s * norb_ + n] -=
                            value;

                        rdm3[n * norb5_ + q * norb4_ + p * norb3_ + n * norb2_ + s * norb_ + r] +=
                            value;
                        rdm3[n * norb5_ + q * norb4_ + p * norb3_ + n * norb2_ + r * norb_ + s] -=
                            value;
                        rdm3[n * norb5_ + p * norb4_ + q * norb3_ + n * norb2_ + r * norb_ + s] +=
                            value;
                        rdm3[n * norb5_ + p * norb4_ + q * norb3_ + n * norb2_ + s * norb_ + r] -=
                            value;

                        rdm3[p * norb5_ + n * norb4_ + q * norb3_ + r * norb2_ + n * norb_ + s] +=
                            value;
                        rdm3[p * norb5_ + n * norb4_ + q * norb3_ + s * norb2_ + n * norb_ + r] -=
                            value;
                        rdm3[q * norb5_ + n * norb4_ + p * norb3_ + s * norb2_ + n * norb_ + r] +=
                            value;
                        rdm3[q * norb5_ + n * norb4_ + p * norb3_ + r * norb2_ + n * norb_ + s] -=
                            value;

                        rdm3[r * norb5_ + s * norb4_ + n * norb3_ + p * norb2_ + q * norb_ + n] +=
                            value;
                        rdm3[s * norb5_ + r * norb4_ + n * norb3_ + p * norb2_ + q * norb_ + n] -=
                            value;
                        rdm3[s * norb5_ + r * norb4_ + n * norb3_ + q * norb2_ + p * norb_ + n] +=
                            value;
                        rdm3[r * norb5_ + s * norb4_ + n * norb3_ + q * norb2_ + p * norb_ + n] -=
                            value;

                        rdm3[n * norb5_ + s * norb4_ + r * norb3_ + n * norb2_ + q * norb_ + p] +=
                            value;
                        rdm3[n * norb5_ + r * norb4_ + s * norb3_ + n * norb2_ + q * norb_ + p] -=
                            value;
                        rdm3[n * norb5_ + r * norb4_ + s * norb3_ + n * norb2_ + p * norb_ + q] +=
                            value;
                        rdm3[n * norb5_ + s * norb4_ + r * norb3_ + n * norb2_ + p * norb_ + q] -=
                            value;

                        rdm3[r * norb5_ + n * norb4_ + s * norb3_ + p * norb2_ + n * norb_ + q] +=
                            value;
                        rdm3[s * norb5_ + n * norb4_ + r * norb3_ + p * norb2_ + n * norb_ + q] -=
                            value;
                        rdm3[s * norb5_ + n * norb4_ + r * norb3_ + q * norb2_ + n * norb_ + p] +=
                            value;
                        rdm3[r * norb5_ + n * norb4_ + s * norb3_ + q * norb2_ + n * norb_ + p] -=
                            value;
                        Ibc.clear_first_one();
                    }

                } else if (ndiff == 6) {
                    auto Ia_sub = Ia & IJa;
                    uint64_t p = Ia_sub.find_first_one();
                    Ia_sub.clear_first_one();
                    uint64_t q = Ia_sub.find_first_one();
                    Ia_sub.clear_first_one();
                    uint64_t r = Ia_sub.find_first_one();

                    auto Ja_sub = Ja & IJa;
                    uint64_t s = Ja_sub.find_first_one();
                    Ja_sub.clear_first_one();
                    uint64_t t = Ja_sub.find_first_one();
                    Ja_sub.clear_first_one();
                    uint64_t u = Ja_sub.find_first_one();
                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double el = Csq * Ia.slater_sign(p, q) * Ia.slater_sign(r) *
                                Ja.slater_sign(s, t) * Ja.slater_sign(u);
                    fill_3rdm(rdm3, el, p, q, r, s, t, u, false);
                }
            }
        }
    }
    outfile->Printf("\n all alpha takes %1.6f", aaa.get());

    //- All beta RDMs -//
    local_timer bbb;
    // loop through all alpha strings
    const std::vector<String>& sorted_astr = a_sorted_string_list_.sorted_half_dets();
    size_t num_astr = sorted_astr.size();
    for (size_t astr = 0; astr < num_astr; ++astr) {
        const String& Ia = sorted_astr[astr];
        const auto& range_I = a_sorted_string_list_.range(Ia);

        String Ib;
        String Jb;
        String IJb;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Double loop through determinants with same alpha string
        for (size_t I = first_I; I < last_I; ++I) {
            Ib = sorted_a_dets[I].get_beta_bits();
            double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
            for (size_t J = I + 1; J < last_I; ++J) {
                Jb = sorted_a_dets[J].get_beta_bits();
                IJb = Ib ^ Jb;
                int ndiff = IJb.count();

                if (ndiff == 2) {
                    auto Ib_sub = Ib & IJb;
                    uint64_t p = Ib_sub.find_first_one();
                    auto Jb_sub = Jb & IJb;
                    uint64_t q = Jb_sub.find_first_one();
                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);

                    double value = Csq * Ib.slater_sign(p, q);
                    rdm1[p * norb_ + q] += value;
                    rdm1[q * norb_ + p] += value;
                    auto Ibc = Ib;
                    Ibc ^= Ib_sub;
                    for (size_t ndb = 1; ndb < nb_; ++ndb) {
                        uint64_t m = Ibc.find_first_one();
                        rdm2[p * norb3_ + m * norb2_ + q * norb_ + m] += value;
                        rdm2[m * norb3_ + p * norb2_ + q * norb_ + m] -= value;
                        rdm2[m * norb3_ + p * norb2_ + m * norb_ + q] += value;
                        rdm2[p * norb3_ + m * norb2_ + m * norb_ + q] -= value;

                        rdm2[q * norb3_ + m * norb2_ + p * norb_ + m] += value;
                        rdm2[m * norb3_ + q * norb2_ + p * norb_ + m] -= value;
                        rdm2[m * norb3_ + q * norb2_ + m * norb_ + p] += value;
                        rdm2[q * norb3_ + m * norb2_ + m * norb_ + p] -= value;
                        Ibc.clear_first_one();

                        String Iac = Ia;
                        for (size_t idx = 0; idx < na_; ++idx) {
                            uint64_t n = Iac.find_first_one();
                            rdm3[n * norb5_ + p * norb4_ + m * norb3_ + n * norb2_ + q * norb_ +
                                 m] += value;
                            rdm3[n * norb5_ + p * norb4_ + m * norb3_ + n * norb2_ + m * norb_ +
                                 q] -= value;
                            rdm3[n * norb5_ + m * norb4_ + p * norb3_ + n * norb2_ + m * norb_ +
                                 q] += value;
                            rdm3[n * norb5_ + m * norb4_ + p * norb3_ + n * norb2_ + q * norb_ +
                                 m] -= value;

                            rdm3[p * norb5_ + n * norb4_ + m * norb3_ + q * norb2_ + n * norb_ +
                                 m] += value;
                            rdm3[p * norb5_ + n * norb4_ + m * norb3_ + m * norb2_ + n * norb_ +
                                 q] -= value;
                            rdm3[m * norb5_ + n * norb4_ + p * norb3_ + m * norb2_ + n * norb_ +
                                 q] += value;
                            rdm3[m * norb5_ + n * norb4_ + p * norb3_ + q * norb2_ + n * norb_ +
                                 m] -= value;

                            rdm3[m * norb5_ + p * norb4_ + n * norb3_ + m * norb2_ + q * norb_ +
                                 n] += value;
                            rdm3[m * norb5_ + p * norb4_ + n * norb3_ + q * norb2_ + m * norb_ +
                                 n] -= value;
                            rdm3[p * norb5_ + m * norb4_ + n * norb3_ + q * norb2_ + m * norb_ +
                                 n] += value;
                            rdm3[p * norb5_ + m * norb4_ + n * norb3_ + m * norb2_ + q * norb_ +
                                 n] -= value;

                            rdm3[n * norb5_ + q * norb4_ + m * norb3_ + n * norb2_ + p * norb_ +
                                 m] += value;
                            rdm3[n * norb5_ + q * norb4_ + m * norb3_ + n * norb2_ + m * norb_ +
                                 p] -= value;
                            rdm3[n * norb5_ + m * norb4_ + q * norb3_ + n * norb2_ + m * norb_ +
                                 p] += value;
                            rdm3[n * norb5_ + m * norb4_ + q * norb3_ + n * norb2_ + p * norb_ +
                                 m] -= value;

                            rdm3[q * norb5_ + n * norb4_ + m * norb3_ + p * norb2_ + n * norb_ +
                                 m] += value;
                            rdm3[q * norb5_ + n * norb4_ + m * norb3_ + m * norb2_ + n * norb_ +
                                 p] -= value;
                            rdm3[m * norb5_ + n * norb4_ + q * norb3_ + m * norb2_ + n * norb_ +
                                 p] += value;
                            rdm3[m * norb5_ + n * norb4_ + q * norb3_ + p * norb2_ + n * norb_ +
                                 m] -= value;

                            rdm3[m * norb5_ + q * norb4_ + n * norb3_ + m * norb2_ + p * norb_ +
                                 n] += value;
                            rdm3[m * norb5_ + q * norb4_ + n * norb3_ + p * norb2_ + m * norb_ +
                                 n] -= value;
                            rdm3[q * norb5_ + m * norb4_ + n * norb3_ + p * norb2_ + m * norb_ +
                                 n] += value;
                            rdm3[q * norb5_ + m * norb4_ + n * norb3_ + m * norb2_ + p * norb_ +
                                 n] -= value;
                            Iac.clear_first_one();
                        }
                    }
                    auto Iac = Ia;
                    for (size_t nidx = 0; nidx < na_; ++nidx) {
                        uint64_t n = Iac.find_first_one();
                        rdm2[n * norb3_ + p * norb2_ + n * norb_ + q] += value;
                        rdm2[n * norb3_ + q * norb2_ + n * norb_ + p] += value;
                        rdm2[p * norb3_ + n * norb2_ + q * norb_ + n] += value;
                        rdm2[q * norb3_ + n * norb2_ + p * norb_ + n] += value;
                        Iac.clear_first_one();

                        auto Iacc = Iac;
                        for (size_t midx = nidx + 1; midx < na_; ++midx) {
                            uint64_t m = Iacc.find_first_one();
                            rdm3[n * norb5_ + m * norb4_ + p * norb3_ + n * norb2_ + m * norb_ +
                                 q] += value;
                            rdm3[n * norb5_ + m * norb4_ + p * norb3_ + m * norb2_ + n * norb_ +
                                 q] -= value;
                            rdm3[m * norb5_ + n * norb4_ + p * norb3_ + m * norb2_ + n * norb_ +
                                 q] += value;
                            rdm3[m * norb5_ + n * norb4_ + p * norb3_ + n * norb2_ + m * norb_ +
                                 q] -= value;

                            rdm3[p * norb5_ + m * norb4_ + n * norb3_ + q * norb2_ + m * norb_ +
                                 n] += value;
                            rdm3[p * norb5_ + m * norb4_ + n * norb3_ + q * norb2_ + n * norb_ +
                                 m] -= value;
                            rdm3[p * norb5_ + n * norb4_ + m * norb3_ + q * norb2_ + n * norb_ +
                                 m] += value;
                            rdm3[p * norb5_ + n * norb4_ + m * norb3_ + q * norb2_ + m * norb_ +
                                 n] -= value;

                            rdm3[n * norb5_ + p * norb4_ + m * norb3_ + n * norb2_ + q * norb_ +
                                 m] += value;
                            rdm3[n * norb5_ + p * norb4_ + m * norb3_ + m * norb2_ + q * norb_ +
                                 n] -= value;
                            rdm3[m * norb5_ + p * norb4_ + n * norb3_ + m * norb2_ + q * norb_ +
                                 n] += value;
                            rdm3[m * norb5_ + p * norb4_ + n * norb3_ + n * norb2_ + q * norb_ +
                                 m] -= value;

                            rdm3[n * norb5_ + m * norb4_ + q * norb3_ + n * norb2_ + m * norb_ +
                                 p] += value;
                            rdm3[n * norb5_ + m * norb4_ + q * norb3_ + m * norb2_ + n * norb_ +
                                 p] -= value;
                            rdm3[m * norb5_ + n * norb4_ + q * norb3_ + m * norb2_ + n * norb_ +
                                 p] += value;
                            rdm3[m * norb5_ + n * norb4_ + q * norb3_ + n * norb2_ + m * norb_ +
                                 p] -= value;

                            rdm3[q * norb5_ + m * norb4_ + n * norb3_ + p * norb2_ + m * norb_ +
                                 n] += value;
                            rdm3[q * norb5_ + m * norb4_ + n * norb3_ + p * norb2_ + n * norb_ +
                                 m] -= value;
                            rdm3[q * norb5_ + n * norb4_ + m * norb3_ + p * norb2_ + n * norb_ +
                                 m] += value;
                            rdm3[q * norb5_ + n * norb4_ + m * norb3_ + p * norb2_ + m * norb_ +
                                 n] -= value;

                            rdm3[n * norb5_ + q * norb4_ + m * norb3_ + n * norb2_ + p * norb_ +
                                 m] += value;
                            rdm3[n * norb5_ + q * norb4_ + m * norb3_ + m * norb2_ + p * norb_ +
                                 n] -= value;
                            rdm3[m * norb5_ + q * norb4_ + n * norb3_ + m * norb2_ + p * norb_ +
                                 n] += value;
                            rdm3[m * norb5_ + q * norb4_ + n * norb3_ + n * norb2_ + p * norb_ +
                                 m] -= value;

                            Iacc.clear_first_one();
                        }
                    }
                    // 3-rdm
                    String Ibcc(Ib);
                    Ibcc ^= Ib_sub;
                    for (size_t ndb = 1; ndb < nb_; ++ndb) {
                        // while(Ibcc >0){
                        uint64_t n = Ibcc.find_first_one();
                        Ibcc.clear_first_one();
                        String I_n = Ibcc;
                        for (size_t ndbb = ndb + 1; ndbb < nb_; ++ndbb) {
                            // while( I_n > 0){
                            uint64_t m = I_n.find_first_one();
                            fill_3rdm(rdm3, value, p, m, n, q, m, n, false);
                            I_n.clear_first_one();
                        }
                    }
                } else if (ndiff == 4) {
                    auto Ib_sub = Ib & IJb;
                    uint64_t p = Ib_sub.find_first_one();
                    Ib_sub.clear_first_one();
                    uint64_t q = Ib_sub.find_first_one();

                    auto Jb_sub = Jb & IJb;
                    uint64_t r = Jb_sub.find_first_one();
                    Jb_sub.clear_first_one();
                    uint64_t s = Jb_sub.find_first_one();

                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ib.slater_sign(p, q) * Jb.slater_sign(r, s);
                    rdm2[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                    rdm2[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                    rdm2[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                    rdm2[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                    rdm2[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                    rdm2[s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                    rdm2[r * norb3_ + s * norb2_ + q * norb_ + p] -= value;
                    rdm2[s * norb3_ + r * norb2_ + q * norb_ + p] += value;

                    // 3-rdm
                    auto Ibc = Ib;
                    Ibc ^= Ib_sub;
                    for (size_t ndb = 1; ndb < nb_; ++ndb) {
                        uint64_t n = Ibc.find_first_one();
                        fill_3rdm(rdm3, value, p, q, n, r, s, n, false);
                        Ibc.clear_first_one();
                    }
                    auto Iac = Ia;
                    for (size_t nda = 0; nda < na_; ++nda) {
                        uint64_t n = Iac.find_first_one();
                        rdm3[n * norb5_ + p * norb4_ + q * norb3_ + n * norb2_ + r * norb_ + s] +=
                            value;
                        rdm3[n * norb5_ + p * norb4_ + q * norb3_ + n * norb2_ + s * norb_ + r] -=
                            value;
                        rdm3[n * norb5_ + q * norb4_ + p * norb3_ + n * norb2_ + s * norb_ + r] +=
                            value;
                        rdm3[n * norb5_ + q * norb4_ + p * norb3_ + n * norb2_ + r * norb_ + s] -=
                            value;

                        rdm3[p * norb5_ + n * norb4_ + q * norb3_ + r * norb2_ + n * norb_ + s] +=
                            value;
                        rdm3[p * norb5_ + n * norb4_ + q * norb3_ + s * norb2_ + n * norb_ + r] -=
                            value;
                        rdm3[q * norb5_ + n * norb4_ + p * norb3_ + s * norb2_ + n * norb_ + r] +=
                            value;
                        rdm3[q * norb5_ + n * norb4_ + p * norb3_ + r * norb2_ + n * norb_ + s] -=
                            value;

                        rdm3[q * norb5_ + p * norb4_ + n * norb3_ + s * norb2_ + r * norb_ + n] +=
                            value;
                        rdm3[q * norb5_ + p * norb4_ + n * norb3_ + r * norb2_ + s * norb_ + n] -=
                            value;
                        rdm3[p * norb5_ + q * norb4_ + n * norb3_ + r * norb2_ + s * norb_ + n] +=
                            value;
                        rdm3[p * norb5_ + q * norb4_ + n * norb3_ + s * norb2_ + r * norb_ + n] -=
                            value;

                        rdm3[n * norb5_ + r * norb4_ + s * norb3_ + n * norb2_ + p * norb_ + q] +=
                            value;
                        rdm3[n * norb5_ + r * norb4_ + s * norb3_ + n * norb2_ + q * norb_ + p] -=
                            value;
                        rdm3[n * norb5_ + s * norb4_ + r * norb3_ + n * norb2_ + q * norb_ + p] +=
                            value;
                        rdm3[n * norb5_ + s * norb4_ + r * norb3_ + n * norb2_ + p * norb_ + q] -=
                            value;

                        rdm3[r * norb5_ + n * norb4_ + s * norb3_ + p * norb2_ + n * norb_ + q] +=
                            value;
                        rdm3[r * norb5_ + n * norb4_ + s * norb3_ + q * norb2_ + n * norb_ + p] -=
                            value;
                        rdm3[s * norb5_ + n * norb4_ + r * norb3_ + q * norb2_ + n * norb_ + p] +=
                            value;
                        rdm3[s * norb5_ + n * norb4_ + r * norb3_ + p * norb2_ + n * norb_ + q] -=
                            value;

                        rdm3[s * norb5_ + r * norb4_ + n * norb3_ + q * norb2_ + p * norb_ + n] +=
                            value;
                        rdm3[s * norb5_ + r * norb4_ + n * norb3_ + p * norb2_ + q * norb_ + n] -=
                            value;
                        rdm3[r * norb5_ + s * norb4_ + n * norb3_ + p * norb2_ + q * norb_ + n] +=
                            value;
                        rdm3[r * norb5_ + s * norb4_ + n * norb3_ + q * norb2_ + p * norb_ + n] -=
                            value;
                        Iac.clear_first_one();
                    }
                } else if (ndiff == 6) {
                    auto Ib_sub = Ib & IJb;
                    uint64_t p = Ib_sub.find_first_one();
                    Ib_sub.clear_first_one();
                    uint64_t q = Ib_sub.find_first_one();
                    Ib_sub.clear_first_one();
                    uint64_t r = Ib_sub.find_first_one();

                    auto Jb_sub = Jb & IJb;
                    uint64_t s = Jb_sub.find_first_one();
                    Jb_sub.clear_first_one();
                    uint64_t t = Jb_sub.find_first_one();
                    Jb_sub.clear_first_one();
                    uint64_t u = Jb_sub.find_first_one();
                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                    double el = Csq * Ib.slater_sign(p, q) * Ib.slater_sign(r) *
                                Jb.slater_sign(s, t) * Jb.slater_sign(u);
                    fill_3rdm(rdm3, el, p, q, r, s, t, u, false);
                }
            }
        }
    }
    outfile->Printf("\n all beta takes %1.6f", bbb.get());

    //- Alpha-Beta RDMs -//
    local_timer mix;
    double d2 = 0.0;
    double d4 = 0.0;
    for (auto& detIa : sorted_astr) {
        const auto& range_I = a_sorted_string_list_.range(detIa);
        String detIJa_common;
        String Ib;
        String Jb;
        String IJb;
        for (auto& detJa : sorted_astr) {
            detIJa_common = detIa ^ detJa;
            int ndiff = detIJa_common.count();
            if (ndiff == 2) {
                local_timer t2;
                auto Ia_d = detIa & detIJa_common;
                uint64_t p = Ia_d.find_first_one();
                auto Ja_d = detJa & detIJa_common;
                uint64_t s = Ja_d.find_first_one();

                const auto& range_J = a_sorted_string_list_.range(detJa);
                size_t first_I = range_I.first;
                size_t last_I = range_I.second;
                size_t first_J = range_J.first;
                size_t last_J = range_J.second;
                double sign_Ips = detIa.slater_sign(p, s);
                double sign_IJ = detIa.slater_sign(p) * detJa.slater_sign(s);
                for (size_t I = first_I; I < last_I; ++I) {
                    Ib = sorted_a_dets[I].get_beta_bits();
                    double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
                    for (size_t J = first_J; J < last_J; ++J) {
                        Jb = sorted_a_dets[J].get_beta_bits();
                        IJb = Ib ^ Jb;
                        int nbdiff = IJb.count();
                        if (nbdiff == 2) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            auto Ib_sub = Ib & IJb;
                            uint64_t q = Ib_sub.find_first_one();
                            auto Jb_sub = Jb & IJb;
                            uint64_t r = Jb_sub.find_first_one();

                            double value =
                                Csq * sign_Ips * Ib.slater_sign(q, r); // * ui64_slater_sign(Jb,r);
                            rdm2[p * norb3_ + q * norb2_ + s * norb_ + r] += value;
                            rdm2[q * norb3_ + p * norb2_ + r * norb_ + s] += value;

                            auto Iac(detIa);
                            Iac ^= Ia_d;
                            for (size_t d = 1; d < na_; ++d) {
                                uint64_t n = Iac.find_first_one();
                                rdm3[p * norb5_ + n * norb4_ + q * norb3_ + s * norb2_ + n * norb_ +
                                     r] += value;
                                rdm3[n * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + n * norb_ +
                                     r] -= value;
                                rdm3[n * norb5_ + p * norb4_ + q * norb3_ + n * norb2_ + s * norb_ +
                                     r] += value;
                                rdm3[p * norb5_ + n * norb4_ + q * norb3_ + n * norb2_ + s * norb_ +
                                     r] -= value;

                                rdm3[q * norb5_ + n * norb4_ + p * norb3_ + r * norb2_ + n * norb_ +
                                     s] += value;
                                rdm3[q * norb5_ + p * norb4_ + n * norb3_ + r * norb2_ + n * norb_ +
                                     s] -= value;
                                rdm3[q * norb5_ + p * norb4_ + n * norb3_ + r * norb2_ + s * norb_ +
                                     n] += value;
                                rdm3[q * norb5_ + n * norb4_ + p * norb3_ + r * norb2_ + s * norb_ +
                                     n] -= value;

                                rdm3[p * norb5_ + q * norb4_ + n * norb3_ + s * norb2_ + r * norb_ +
                                     n] += value;
                                rdm3[n * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + r * norb_ +
                                     n] -= value;
                                rdm3[n * norb5_ + q * norb4_ + p * norb3_ + n * norb2_ + r * norb_ +
                                     s] += value;
                                rdm3[p * norb5_ + q * norb4_ + n * norb3_ + n * norb2_ + r * norb_ +
                                     s] -= value;
                                Iac.clear_first_one();
                            }
                            auto Ibc(Ib);
                            Ibc ^= Ib_sub;
                            for (size_t d = 1; d < nb_; ++d) {
                                uint64_t n = Ibc.find_first_one();
                                rdm3[p * norb5_ + q * norb4_ + n * norb3_ + s * norb2_ + r * norb_ +
                                     n] += value;
                                rdm3[p * norb5_ + q * norb4_ + n * norb3_ + s * norb2_ + n * norb_ +
                                     r] -= value;
                                rdm3[p * norb5_ + n * norb4_ + q * norb3_ + s * norb2_ + n * norb_ +
                                     r] += value;
                                rdm3[p * norb5_ + n * norb4_ + q * norb3_ + s * norb2_ + r * norb_ +
                                     n] -= value;

                                rdm3[q * norb5_ + p * norb4_ + n * norb3_ + r * norb2_ + s * norb_ +
                                     n] += value;
                                rdm3[q * norb5_ + p * norb4_ + n * norb3_ + n * norb2_ + s * norb_ +
                                     r] -= value;
                                rdm3[n * norb5_ + p * norb4_ + q * norb3_ + n * norb2_ + s * norb_ +
                                     r] += value;
                                rdm3[n * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + s * norb_ +
                                     n] -= value;

                                rdm3[n * norb5_ + q * norb4_ + p * norb3_ + n * norb2_ + r * norb_ +
                                     s] += value;
                                rdm3[n * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + n * norb_ +
                                     s] -= value;
                                rdm3[q * norb5_ + n * norb4_ + p * norb3_ + r * norb2_ + n * norb_ +
                                     s] += value;
                                rdm3[q * norb5_ + n * norb4_ + p * norb3_ + n * norb2_ + r * norb_ +
                                     s] -= value;
                                Ibc.clear_first_one();
                            }
                        } else if (nbdiff == 4) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            auto Ib_sub = Ib & IJb;
                            uint64_t q = Ib_sub.find_first_one();
                            Ib_sub.clear_first_one();
                            uint64_t r = Ib_sub.find_first_one();

                            auto Jb_sub = Jb & IJb;
                            uint64_t t = Jb_sub.find_first_one();
                            Jb_sub.clear_first_one();
                            uint64_t u = Jb_sub.find_first_one();

                            double value = Csq * sign_IJ *
                                           Ib.slater_sign(q, r) * // ui64_slater_sign(Ib,r) *
                                           Jb.slater_sign(t, u);  // * ui64_slater_sign(Jb,u);
                            rdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ +
                                 u] += value;
                            rdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ +
                                 t] -= value;
                            rdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ +
                                 t] += value;
                            rdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ +
                                 u] -= value;

                            rdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ +
                                 u] += value;
                            rdm3[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ +
                                 t] -= value;
                            rdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ +
                                 t] += value;
                            rdm3[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ +
                                 u] -= value;

                            rdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ +
                                 s] += value;
                            rdm3[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ +
                                 s] -= value;
                            rdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ +
                                 s] += value;
                            rdm3[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ +
                                 s] -= value;
                        }
                    }
                }
                d2 += t2.get();
            } else if (ndiff == 4) {
                local_timer t4;
                // Get aa-aa part of aab 3rdm
                auto Ia_sub = detIa & detIJa_common;
                uint64_t p = Ia_sub.find_first_one();
                Ia_sub.clear_first_one();
                uint64_t q = Ia_sub.find_first_one();

                auto Ja_sub = detJa & detIJa_common;
                uint64_t s = Ja_sub.find_first_one();
                Ja_sub.clear_first_one();
                uint64_t t = Ja_sub.find_first_one();

                const auto& range_J = a_sorted_string_list_.range(detJa);
                size_t first_I = range_I.first;
                size_t last_I = range_I.second;
                size_t first_J = range_J.first;
                size_t last_J = range_J.second;

                // double sign = ui64_slater_sign(detIa,p,q) * ui64_slater_sign(detJa,s,t);
                double sign = detIa.slater_sign(p, q) * // ui64_slater_sign(detIa,q) *
                              detJa.slater_sign(s, t);  // ui64_slater_sign(detJa,t);

                // Now the b-b part
                for (size_t I = first_I; I < last_I; ++I) {
                    Ib = sorted_a_dets[I].get_beta_bits();
                    double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
                    for (size_t J = first_J; J < last_J; ++J) {
                        Jb = sorted_a_dets[J].get_beta_bits();
                        IJb = Ib ^ Jb;
                        int nbdiff = IJb.count();
                        if (nbdiff == 2) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            auto Ib_sub = Ib & IJb;
                            uint64_t r = Ib_sub.find_first_one();
                            auto Jb_sub = Jb & IJb;
                            uint64_t u = Jb_sub.find_first_one();
                            double el = Csq * sign * Ib.slater_sign(r) * Jb.slater_sign(u);

                            rdm3[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ +
                                 u] += el;
                            rdm3[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ +
                                 u] -= el;
                            rdm3[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ +
                                 u] -= el;
                            rdm3[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ +
                                 u] += el;

                            rdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ +
                                 s] += el;
                            rdm3[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ +
                                 t] -= el;
                            rdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ +
                                 s] -= el;
                            rdm3[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ +
                                 t] += el;

                            rdm3[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ +
                                 t] += el;
                            rdm3[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ +
                                 s] -= el;
                            rdm3[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ +
                                 t] -= el;
                            rdm3[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ +
                                 s] += el;
                        }
                    }
                }
                d4 += t4.get();
            }
        }
    }
    outfile->Printf("\n  2dif: %1.6f  \n  4dif: %1.6f", d2, d4);
    outfile->Printf("\n all alpha/beta takes %1.6f", mix.get());
}
} // namespace forte
