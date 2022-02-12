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

// A class that takes the determinants and expansion
// coefficients and computes reduced density matrices.

CI_RDMS::CI_RDMS(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                 const std::vector<Determinant>& det_space, psi::SharedMatrix evecs, int root1,
                 int root2)
    : fci_ints_(fci_ints), det_space_(det_space), evecs_(evecs), root1_(root1), root2_(root2) {
    startup();
}

CI_RDMS::CI_RDMS(DeterminantHashVec& wfn, std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                 psi::SharedMatrix evecs, int root1, int root2)
    : wfn_(wfn), fci_ints_(fci_ints), evecs_(evecs), root1_(root1), root2_(root2) {
    no_ = fci_ints_->nmo();
    no2_ = no_ * no_;
    no3_ = no2_ * no_;
    no4_ = no3_ * no_;
    no5_ = no4_ * no_;
    no6_ = no5_ * no_;

    print_ = false;
    dim_space_ = wfn.size();

    Determinant det(wfn_.get_det(0));
    na_ = det.count_alfa();
    nb_ = det.count_beta();
}

CI_RDMS::~CI_RDMS() {}

void CI_RDMS::startup() {
    /* Get all of the required info from MOSpaceInfo to initialize the
     * StringList*/

    // The number of correlated molecular orbitals
    no_ = fci_ints_->nmo();
    no2_ = no_ * no_;
    no3_ = no2_ * no_;
    no4_ = no3_ * no_;
    no5_ = no4_ * no_;
    no6_ = no5_ * no_;

    na_ = det_space_[0].count_alfa();
    nb_ = det_space_[0].count_beta();

    // psi::Dimension of the determinant space
    dim_space_ = det_space_.size();

    print_ = false;

    one_map_done_ = false;

    if (print_) {
        outfile->Printf("\n  Computing RDMS");
        outfile->Printf("\n  Number of active alpha electrons: %zu", na_);
        outfile->Printf("\n  Number of active beta electrons: %zu", nb_);
        outfile->Printf("\n  Number of correlated orbitals: %zu", no_);
    }
}

void CI_RDMS::set_max_rdm(int rdm) { max_rdm_ = rdm; }

double CI_RDMS::get_energy(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b,
                           std::vector<double>& tprdm_aa, std::vector<double>& tprdm_bb,
                           std::vector<double>& tprdm_ab) {
    double nuc_rep = fci_ints_->ints()->nuclear_repulsion_energy();
    double scalar_energy = fci_ints_->frozen_core_energy() + fci_ints_->scalar_energy();
    double energy_1rdm = 0.0;
    double energy_2rdm = 0.0;

    for (size_t p = 0; p < no_; ++p) {
        for (size_t q = 0; q < no_; ++q) {
            energy_1rdm += oprdm_a[no_ * p + q] * fci_ints_->oei_a(p, q);
            energy_1rdm += oprdm_b[no_ * p + q] * fci_ints_->oei_b(p, q);
        }
    }

    for (size_t p = 0; p < no_; ++p) {
        for (size_t q = 0; q < no_; ++q) {
            for (size_t r = 0; r < no_; ++r) {
                for (size_t s = 0; s < no_; ++s) {
                    if (na_ >= 2)
                        energy_2rdm += 0.25 * tprdm_aa[p * no3_ + q * no2_ + r * no_ + s] *
                                       fci_ints_->tei_aa(p, q, r, s);
                    if ((na_ >= 1) and (nb_ >= 1))
                        energy_2rdm += tprdm_ab[p * no3_ + q * no2_ + r * no_ + s] *
                                       fci_ints_->tei_ab(p, q, r, s);
                    if (nb_ >= 2)
                        energy_2rdm += 0.25 * tprdm_bb[p * no3_ + q * no2_ + r * no_ + s] *
                                       fci_ints_->tei_bb(p, q, r, s);
                }
            }
        }
    }
    double total_energy = nuc_rep + scalar_energy + energy_1rdm + energy_2rdm;

    if (print_) {
        outfile->Printf("\n  Total Energy: %25.15f\n", total_energy);
        outfile->Printf("\n  Scalar Energy = %8.8f", scalar_energy);
        outfile->Printf("\n  energy_1rdm = %8.8f", energy_1rdm);
        outfile->Printf("\n  energy_2rdm = %8.8f", energy_2rdm);
        outfile->Printf("\n  nuclear_repulsion_energy = %8.8f", nuc_rep);
    }

    return total_energy;
}

void CI_RDMS::compute_1rdm(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b) {
    local_timer one;
    get_one_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 1-map:   %1.6f", one.get());

    local_timer build;
    oprdm_a.assign(no2_, 0.0);
    oprdm_b.assign(no2_, 0.0);

    _add_1rdm(oprdm_a, a_ann_list_, a_cre_list_);
    _add_1rdm(oprdm_b, b_ann_list_, b_cre_list_);

    //    for (size_t J = 0; J < dim_space_; ++J) {
    //        for (auto& aJ_mo_sign : a_ann_list_[J]) {
    //            const size_t aJ_add = aJ_mo_sign.first;
    //            size_t p = std::abs(aJ_mo_sign.second) - 1;
    //            const double sign_p = aJ_mo_sign.second > 0 ? 1.0 : -1.0;
    //            for (auto& aaJ_mo_sign : a_cre_list_[aJ_add]) {
    //                size_t q = std::abs(aaJ_mo_sign.second) - 1;
    //                const double sign_q = aaJ_mo_sign.second > 0 ? 1.0 : -1.0;
    //                const size_t I = aaJ_mo_sign.first;
    //                oprdm_a[q * no_ + p] +=
    //                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_p * sign_q;
    //            }
    //        }
    //        for (auto& bJ_mo_sign : b_ann_list_[J]) {
    //            const size_t bJ_add = bJ_mo_sign.first;
    //            const size_t p = std::abs(bJ_mo_sign.second) - 1;
    //            const double sign_p = bJ_mo_sign.second > 0 ? 1.0 : -1.0;
    //            for (auto& bbJ_mo_sign : b_cre_list_[bJ_add]) {
    //                const size_t q = std::abs(bbJ_mo_sign.second) - 1;
    //                const double sign_q = bbJ_mo_sign.second > 0 ? 1.0 : -1.0;
    //                const size_t I = bbJ_mo_sign.first;
    //                oprdm_b[q * no_ + p] +=
    //                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_p * sign_q;
    //            }
    //        }
    //    }
    if (print_)
        outfile->Printf("\n  Time spent building 1-rdm:   %1.6f", build.get());
}

void CI_RDMS::_add_1rdm(std::vector<double>& opdm,
                        const std::vector<std::vector<std::pair<size_t, short>>>& ann_list,
                        const std::vector<std::vector<std::pair<size_t, short>>>& cre_list) {
    for (size_t J = 0; J < dim_space_; ++J) {
        auto vJ = evecs_->get(J, root2_);
        for (const auto& aJ_mo_sign : ann_list[J]) {
            const auto [aJ_add, _p] = aJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_p = _p < 0;
            for (auto& aaJ_mo_sign : cre_list[aJ_add]) {
                const auto [I, _q] = aaJ_mo_sign;
                auto q = std::abs(_q) - 1;
                auto sign = ((_q < 0) == sign_p) ? 1 : -1;
                opdm[q * no_ + p] += evecs_->get(I, root1_) * vJ * sign;
            }
        }
    }
}

void CI_RDMS::compute_1rdm_op(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b) {

    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->op_s_lists(wfn_);

    local_timer build;
    oprdm_a.assign(no2_, 0.0);
    oprdm_b.assign(no2_, 0.0);

    // Do something about diagonal
    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t J = 0; J < dim_space_; ++J) {
        double cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
        for (int pp : dets[J].get_alfa_occ(no_)) {
            oprdm_a[pp * no_ + pp] += cJ_sq;
        }
        for (int pp : dets[J].get_beta_occ(no_)) {
            oprdm_b[pp * no_ + pp] += cJ_sq;
        }
    }

    _add_1rdm_op_IJ(oprdm_a, op->a_list_);
    _add_1rdm_op_IJ(oprdm_b, op->b_list_);

    if (print_) {
        outfile->Printf("\n  Time spent building 1-rdm: %.3e seconds", build.get());
    }
}

void CI_RDMS::_add_1rdm_op_IJ(std::vector<double>& opdm,
                              const std::vector<std::vector<std::pair<size_t, short>>>& list) {
    for (const auto& coupled_dets : list) {
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

                opdm[p * no_ + q] += vI1 * evecs_->get(J, root2_) * sign;
                opdm[q * no_ + p] += evecs_->get(J, root1_) * vI2 * sign;
            }
        }
    }
}

void CI_RDMS::compute_2rdm(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                           std::vector<double>& tprdm_bb) {
    tprdm_aa.assign(no4_, 0.0);
    tprdm_ab.assign(no4_, 0.0);
    tprdm_bb.assign(no4_, 0.0);

    local_timer two;
    get_two_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 2-map:   %1.6f", two.get());

    local_timer build;

    _add_2rdm(aa_ann_list_, aa_cre_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_2rdm_aa(tprdm_aa, i, value);
    });
    _add_2rdm(ab_ann_list_, ab_cre_list_, [&](const std::vector<size_t>& i, const double& value) {
        tprdm_ab[i[0] * no3_ + i[1] * no2_ + i[2] * no_ + i[3]] += value;
    });
    _add_2rdm(bb_ann_list_, bb_cre_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_2rdm_aa(tprdm_bb, i, value);
    });

    //    for (size_t J = 0; J < dim_space_; ++J) {
    //        // aaaa
    //        for (auto& aaJ_mo_sign : aa_ann_list_[J]) {
    //            const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
    //
    //            const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
    //            const size_t q = std::get<2>(aaJ_mo_sign);
    //            const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //
    //            for (auto& aaaaJ_mo_sign : aa_cre_list_[aaJ_add]) {
    //                const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
    //                const size_t s = std::get<2>(aaaaJ_mo_sign);
    //                const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //                const size_t I = std::get<0>(aaaaJ_mo_sign);
    //                double rdm_element =
    //                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;
    //
    //                tprdm_aa[p * no3_ + q * no2_ + r * no_ + s] += rdm_element;
    //                tprdm_aa[p * no3_ + q * no2_ + s * no_ + r] -= rdm_element;
    //                tprdm_aa[q * no3_ + p * no2_ + r * no_ + s] -= rdm_element;
    //                tprdm_aa[q * no3_ + p * no2_ + s * no_ + r] += rdm_element;
    //            }
    //        }
    //
    //        // bbbb
    //        for (auto& bbJ_mo_sign : bb_ann_list_[J]) {
    //            const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
    //
    //            const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
    //            const size_t q = std::get<2>(bbJ_mo_sign);
    //            const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //
    //            for (auto& bbbbJ_mo_sign : bb_cre_list_[bbJ_add]) {
    //                const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
    //                const size_t s = std::get<2>(bbbbJ_mo_sign);
    //                const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //                const size_t I = std::get<0>(bbbbJ_mo_sign);
    //                double rdm_element =
    //                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;
    //
    //                tprdm_bb[p * no3_ + q * no2_ + r * no_ + s] += rdm_element;
    //                tprdm_bb[p * no3_ + q * no2_ + s * no_ + r] -= rdm_element;
    //                tprdm_bb[q * no3_ + p * no2_ + r * no_ + s] -= rdm_element;
    //                tprdm_bb[q * no3_ + p * no2_ + s * no_ + r] += rdm_element;
    //            }
    //        }
    //        // aabb
    //        for (auto& abJ_mo_sign : ab_ann_list_[J]) {
    //            const size_t abJ_add = std::get<0>(abJ_mo_sign);
    //
    //            const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
    //            const size_t q = std::get<2>(abJ_mo_sign);
    //            const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //
    //            for (auto& aabbJ_mo_sign : ab_cre_list_[abJ_add]) {
    //                const size_t r = std::abs(std::get<1>(aabbJ_mo_sign)) - 1;
    //                const size_t s = std::get<2>(aabbJ_mo_sign);
    //                const double sign_rs = std::get<1>(aabbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //                const size_t I = std::get<0>(aabbJ_mo_sign);
    //                double rdm_element =
    //                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;
    //
    //                tprdm_ab[p * no3_ + q * no2_ + r * no_ + s] += rdm_element;
    //            }
    //        }
    //    }
    if (print_)
        outfile->Printf("\n  Time spent building 2-rdm:   %1.6f", build.get());
}

void CI_RDMS::_add_2rdm(
    const std::vector<std::vector<std::tuple<size_t, short, short>>>& ann_list,
    const std::vector<std::vector<std::tuple<size_t, short, short>>>& cre_list,
    const std::function<void(const std::vector<size_t>&, const double&)>& func) {
    for (size_t J = 0; J < dim_space_; ++J) {
        auto vJ = evecs_->get(J, root2_);

        for (auto& aaJ_mo_sign : ann_list[J]) {
            const auto [aaJ_add, _p, q] = aaJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            for (auto& aaaaJ_mo_sign : cre_list[aaJ_add]) {
                const auto [I, _r, s] = aaaaJ_mo_sign;
                auto r = std::abs(_r) - 1;

                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;
                auto value = evecs_->get(I, root1_) * vJ * sign;

                func({r, s, p, q}, value);
            }
        }
    }
}

void CI_RDMS::_add_2rdm_aa(std::vector<double>& tpdm_aa, const std::vector<size_t>& i,
                           const double& value) {
    tpdm_aa[i[0] * no3_ + i[1] * no2_ + i[2] * no_ + i[3]] += value;
    tpdm_aa[i[0] * no3_ + i[1] * no2_ + i[3] * no_ + i[2]] -= value;
    tpdm_aa[i[1] * no3_ + i[0] * no2_ + i[2] * no_ + i[3]] -= value;
    tpdm_aa[i[1] * no3_ + i[0] * no2_ + i[3] * no_ + i[2]] += value;
}

void CI_RDMS::_add_2rdm_ab(std::vector<double>& tpdm_ab, const std::vector<size_t>& i,
                           const double& value, const bool& spin_free) {
    tpdm_ab[i[0] * no3_ + i[1] * no2_ + i[2] * no_ + i[3]] += value;
    if (spin_free)
        tpdm_ab[i[1] * no3_ + i[0] * no2_ + i[3] * no_ + i[2]] += value;
}

void CI_RDMS::compute_2rdm_op(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                              std::vector<double>& tprdm_bb) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->tp_s_lists(wfn_);

    local_timer build;

    tprdm_aa.assign(no4_, 0.0);
    tprdm_ab.assign(no4_, 0.0);
    tprdm_bb.assign(no4_, 0.0);

    _add_2rdm_op_II([&](const std::vector<size_t>& i,
                        const double& value) { _add_2rdm_aa(tprdm_aa, i, value); },
                    [&](const std::vector<size_t>& i, const double& value) {
                        tprdm_ab[i[0] * no3_ + i[1] * no2_ + i[0] * no_ + i[1]] += value;
                    },
                    [&](const std::vector<size_t>& i, const double& value) {
                        _add_2rdm_aa(tprdm_bb, i, value);
                    });
    //    const det_hashvec& dets = wfn_.wfn_hash();
    //    for (size_t J = 0; J < dim_space_; ++J) {
    //        double cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
    //        std::vector<int> aocc = dets[J].get_alfa_occ(no_);
    //        std::vector<int> bocc = dets[J].get_beta_occ(no_);
    //
    //        int naocc = aocc.size();
    //        int nbocc = bocc.size();
    //
    //        for (int p = 0; p < naocc; ++p) {
    //            int pp = aocc[p];
    //            for (int q = p + 1; q < naocc; ++q) {
    //                int qq = aocc[q];
    //                tprdm_aa[pp * no3_ + qq * no2_ + pp * no_ + qq] += cJ_sq;
    //                tprdm_aa[qq * no3_ + pp * no2_ + pp * no_ + qq] -= cJ_sq;
    //
    //                tprdm_aa[qq * no3_ + pp * no2_ + qq * no_ + pp] += cJ_sq;
    //                tprdm_aa[pp * no3_ + qq * no2_ + qq * no_ + pp] -= cJ_sq;
    //            }
    //        }
    //
    //        for (int p = 0; p < nbocc; ++p) {
    //            int pp = bocc[p];
    //            for (int q = p + 1; q < nbocc; ++q) {
    //                int qq = bocc[q];
    //                tprdm_bb[pp * no3_ + qq * no2_ + pp * no_ + qq] += cJ_sq;
    //                tprdm_bb[qq * no3_ + pp * no2_ + pp * no_ + qq] -= cJ_sq;
    //
    //                tprdm_bb[qq * no3_ + pp * no2_ + qq * no_ + pp] += cJ_sq;
    //                tprdm_bb[pp * no3_ + qq * no2_ + qq * no_ + pp] -= cJ_sq;
    //            }
    //        }
    //
    //        for (int p = 0; p < naocc; ++p) {
    //            int pp = aocc[p];
    //            for (int q = 0; q < nbocc; ++q) {
    //                int qq = bocc[q];
    //                tprdm_ab[pp * no3_ + qq * no2_ + pp * no_ + qq] += cJ_sq;
    //            }
    //        }
    //    }

    _add_2rdm_op_IJ(op->aa_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_2rdm_aa(tprdm_aa, i, value);
    });
    _add_2rdm_op_IJ(op->ab_list_, [&](const std::vector<size_t>& i, const double& value) {
        tprdm_ab[i[0] * no3_ + i[1] * no2_ + i[2] * no_ + i[3]] += value;
    });
    _add_2rdm_op_IJ(op->bb_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_2rdm_aa(tprdm_bb, i, value);
    });

    //    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_list = op->aa_list_;
    //    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_list = op->ab_list_;
    //    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_list = op->bb_list_;
    //
    //    // aaaa
    //    for (size_t K = 0, max_K = aa_list.size(); K < max_K; ++K) {
    //        std::vector<std::tuple<size_t, short, short>>& coupled_dets = aa_list[K];
    //        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
    //
    //            auto& detJ = coupled_dets[a];
    //
    //            const size_t& J = std::get<0>(detJ);
    //            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
    //            const size_t& q = std::get<2>(detJ);
    //            const double& sign_pq = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
    //
    //            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
    //
    //                auto& detI = coupled_dets[b];
    //
    //                const size_t& r = std::abs(std::get<1>(detI)) - 1;
    //                const size_t& s = std::get<2>(detI);
    //                const double& sign_rs = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
    //                const size_t& I = std::get<0>(detI);
    //                double rdm_element =
    //                    evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_pq * sign_rs;
    //
    //                tprdm_aa[p * no3_ + q * no2_ + r * no_ + s] += rdm_element;
    //                tprdm_aa[p * no3_ + q * no2_ + s * no_ + r] -= rdm_element;
    //                tprdm_aa[q * no3_ + p * no2_ + r * no_ + s] -= rdm_element;
    //                tprdm_aa[q * no3_ + p * no2_ + s * no_ + r] += rdm_element;
    //
    //                rdm_element = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq *
    //                sign_rs;
    //
    //                tprdm_aa[r * no3_ + s * no2_ + p * no_ + q] += rdm_element;
    //                tprdm_aa[s * no3_ + r * no2_ + p * no_ + q] -= rdm_element;
    //                tprdm_aa[r * no3_ + s * no2_ + q * no_ + p] -= rdm_element;
    //                tprdm_aa[s * no3_ + r * no2_ + q * no_ + p] += rdm_element;
    //            }
    //        }
    //    }
    //
    //    // bbbb
    //    for (size_t K = 0, max_K = bb_list.size(); K < max_K; ++K) {
    //        std::vector<std::tuple<size_t, short, short>>& coupled_dets = bb_list[K];
    //        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
    //
    //            auto& detJ = coupled_dets[a];
    //
    //            const size_t& J = std::get<0>(detJ);
    //            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
    //            const size_t& q = std::get<2>(detJ);
    //            const double& sign_pq = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
    //
    //            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
    //
    //                auto& detI = coupled_dets[b];
    //
    //                const size_t& r = std::abs(std::get<1>(detI)) - 1;
    //                const size_t& s = std::get<2>(detI);
    //                const double& sign_rs = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
    //                const size_t& I = std::get<0>(detI);
    //                double rdm_element =
    //                    evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_pq * sign_rs;
    //
    //                tprdm_bb[p * no3_ + q * no2_ + r * no_ + s] += rdm_element;
    //                tprdm_bb[p * no3_ + q * no2_ + s * no_ + r] -= rdm_element;
    //                tprdm_bb[q * no3_ + p * no2_ + r * no_ + s] -= rdm_element;
    //                tprdm_bb[q * no3_ + p * no2_ + s * no_ + r] += rdm_element;
    //
    //                rdm_element = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq *
    //                sign_rs;
    //
    //                tprdm_bb[r * no3_ + s * no2_ + p * no_ + q] += rdm_element;
    //                tprdm_bb[s * no3_ + r * no2_ + p * no_ + q] -= rdm_element;
    //                tprdm_bb[r * no3_ + s * no2_ + q * no_ + p] -= rdm_element;
    //                tprdm_bb[s * no3_ + r * no2_ + q * no_ + p] += rdm_element;
    //            }
    //        }
    //    }
    //    // aabb
    //    for (size_t K = 0, max_K = ab_list.size(); K < max_K; ++K) {
    //        std::vector<std::tuple<size_t, short, short>>& coupled_dets = ab_list[K];
    //        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
    //
    //            auto& detJ = coupled_dets[a];
    //
    //            const size_t& J = std::get<0>(detJ);
    //            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
    //            const size_t& q = std::get<2>(detJ);
    //            const double& sign_pq = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
    //
    //            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
    //
    //                auto& detI = coupled_dets[b];
    //
    //                const size_t& r = std::abs(std::get<1>(detI)) - 1;
    //                const size_t& s = std::get<2>(detI);
    //                const double& sign_rs = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
    //                const size_t& I = std::get<0>(detI);
    //
    //                double rdm_element =
    //                    evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_pq * sign_rs;
    //                tprdm_ab[p * no3_ + q * no2_ + r * no_ + s] += rdm_element;
    //
    //                rdm_element = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq *
    //                sign_rs; tprdm_ab[r * no3_ + s * no2_ + p * no_ + q] += rdm_element;
    //            }
    //        }
    //    }

    if (print_) {
        outfile->Printf("\n  Time spent building 2-rdm: %.3e seconds", build.get());
    }
}

void CI_RDMS::_add_2rdm_op_II(
    const std::function<void(const std::vector<size_t>&, const double&)>& func_aa,
    const std::function<void(const std::vector<size_t>&, const double&)>& func_ab,
    const std::function<void(const std::vector<size_t>&, const double&)>& func_bb) {
    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t J = 0; J < dim_space_; ++J) {
        double cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);

        std::vector<int> aocc = dets[J].get_alfa_occ(no_);
        std::vector<int> bocc = dets[J].get_beta_occ(no_);
        auto naocc = aocc.size();
        auto nbocc = bocc.size();

        for (size_t p = 0; p < naocc; ++p) {
            size_t pp = aocc[p];
            for (size_t q = p + 1; q < naocc; ++q) {
                size_t qq = aocc[q];
                func_aa({pp, qq, pp, qq}, cJ_sq);
            }
        }
        for (size_t p = 0; p < nbocc; ++p) {
            size_t pp = bocc[p];
            for (size_t q = p + 1; q < nbocc; ++q) {
                size_t qq = bocc[q];
                func_bb({pp, qq, pp, qq}, cJ_sq);
            }
        }
        for (size_t p = 0; p < naocc; ++p) {
            size_t pp = aocc[p];
            for (size_t q = 0; q < nbocc; ++q) {
                size_t qq = bocc[q];
                func_ab({pp, qq, pp, qq}, cJ_sq);
            }
        }
    }
}

void CI_RDMS::_add_2rdm_op_IJ(
    const std::vector<std::vector<std::tuple<size_t, short, short>>>& list,
    const std::function<void(const std::vector<size_t>&, const double&)>& func) {
    for (const auto& coupled_dets : list) {

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

                auto rdmJI = vJ1 * evecs_->get(I, root2_) * sign;
                func({p, q, r, s}, rdmJI);

                auto rdmIJ = evecs_->get(I, root1_) * vJ2 * sign;
                func({r, s, p, q}, rdmIJ);
            }
        }
    }
}

void CI_RDMS::compute_3rdm(std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab,
                           std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb) {
    tprdm_aaa.assign(no6_, 0.0);
    tprdm_aab.assign(no6_, 0.0);
    tprdm_abb.assign(no6_, 0.0);
    tprdm_bbb.assign(no6_, 0.0);

    local_timer three;
    get_three_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 3-map:   %1.6f", three.get());

    local_timer build;

    _add_3rdm(aaa_ann_list_, aaa_cre_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aaa(tprdm_aaa, i, value);
    });
    _add_3rdm(aab_ann_list_, aab_cre_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aab(tprdm_aab, i, value, false);
    });
    _add_3rdm(abb_ann_list_, abb_cre_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_abb(tprdm_abb, i, value, false);
    });
    _add_3rdm(bbb_ann_list_, bbb_cre_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aaa(tprdm_bbb, i, value);
    });

    //    for (size_t J = 0; J < dim_space_; ++J) {
    //        // aaa aaa
    //        for (auto& aaaJ_mo_sign : aaa_ann_list_[J]) {
    //            const size_t aaaJ_add = std::get<0>(aaaJ_mo_sign);
    //
    //            const size_t p = std::abs(std::get<1>(aaaJ_mo_sign)) - 1;
    //            const size_t q = std::get<2>(aaaJ_mo_sign);
    //            const size_t r = std::get<3>(aaaJ_mo_sign);
    //            const double sign_pqr = std::get<1>(aaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //
    //            for (auto& a6J : aaa_cre_list_[aaaJ_add]) {
    //                const size_t s = std::abs(std::get<1>(a6J)) - 1;
    //                const size_t t = std::get<2>(a6J);
    //                const size_t u = std::get<3>(a6J);
    //                const double sign_stu = std::get<1>(a6J) > 0.0 ? 1.0 : -1.0;
    //                const size_t I = std::get<0>(a6J);
    //
    //                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr *
    //                sign_stu;
    //
    //                tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //            }
    //        }
    //        // aab aab
    //        for (auto& aabJ_mo_sign : aab_ann_list_[J]) {
    //            const size_t aabJ_add = std::get<0>(aabJ_mo_sign);
    //
    //            const size_t p = std::abs(std::get<1>(aabJ_mo_sign)) - 1;
    //            const size_t q = std::get<2>(aabJ_mo_sign);
    //            const size_t r = std::get<3>(aabJ_mo_sign);
    //            const double sign_pqr = std::get<1>(aabJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //
    //            for (auto& aabJ : aab_cre_list_[aabJ_add]) {
    //                const size_t s = std::abs(std::get<1>(aabJ)) - 1;
    //                const size_t t = std::get<2>(aabJ);
    //                const size_t u = std::get<3>(aabJ);
    //                const double sign_stu = std::get<1>(aabJ) > 0.0 ? 1.0 : -1.0;
    //                const size_t I = std::get<0>(aabJ);
    //
    //                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr *
    //                sign_stu;
    //
    //                tprdm_aab[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_aab[p * no5_ + q * no4_ + r * no3_ + t * no2_ + s * no_ +
    //                u] -= el; tprdm_aab[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t *
    //                no_ + u] -= el; tprdm_aab[q * no5_ + p * no4_ + r * no3_ + t * no2_ +
    //                s * no_ + u] += el;
    //            }
    //        }
    //        // abb abb
    //        for (auto& abbJ_mo_sign : abb_ann_list_[J]) {
    //            const size_t abbJ_add = std::get<0>(abbJ_mo_sign);
    //
    //            const size_t p = std::abs(std::get<1>(abbJ_mo_sign)) - 1;
    //            const size_t q = std::get<2>(abbJ_mo_sign);
    //            const size_t r = std::get<3>(abbJ_mo_sign);
    //            const double sign_pqr = std::get<1>(abbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //
    //            for (auto& abbJ : abb_cre_list_[abbJ_add]) {
    //                const size_t s = std::abs(std::get<1>(abbJ)) - 1;
    //                const size_t t = std::get<2>(abbJ);
    //                const size_t u = std::get<3>(abbJ);
    //                const double sign_stu = std::get<1>(abbJ) > 0.0 ? 1.0 : -1.0;
    //                const size_t I = std::get<0>(abbJ);
    //
    //                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr *
    //                sign_stu;
    //
    //                tprdm_abb[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_abb[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_abb[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t *
    //                no_ + u] -= el; tprdm_abb[p * no5_ + r * no4_ + q * no3_ + s * no2_ +
    //                u * no_ + t] += el;
    //            }
    //        }
    //        // bbb bbb
    //        for (auto& bbbJ_mo_sign : bbb_ann_list_[J]) {
    //            const size_t bbbJ_add = std::get<0>(bbbJ_mo_sign);
    //
    //            const size_t p = std::abs(std::get<1>(bbbJ_mo_sign)) - 1;
    //            const size_t q = std::get<2>(bbbJ_mo_sign);
    //            const size_t r = std::get<3>(bbbJ_mo_sign);
    //            const double sign_pqr = std::get<1>(bbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
    //
    //            for (auto& b6J : bbb_cre_list_[bbbJ_add]) {
    //                const size_t s = std::abs(std::get<1>(b6J)) - 1;
    //                const size_t t = std::get<2>(b6J);
    //                const size_t u = std::get<3>(b6J);
    //                const double sign_stu = std::get<1>(b6J) > 0.0 ? 1.0 : -1.0;
    //                const size_t I = std::get<0>(b6J);
    //
    //                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr *
    //                sign_stu;
    //
    //                tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //            }
    //        }
    //    }

    if (print_)
        outfile->Printf("\n  Time spent building 3-rdm:   %1.6f", build.get());
}

void CI_RDMS::_add_3rdm(
    const std::vector<std::vector<std::tuple<size_t, short, short, short>>>& ann_list,
    const std::vector<std::vector<std::tuple<size_t, short, short, short>>>& cre_list,
    const std::function<void(const std::vector<size_t>&, const double&)>& func) {
    for (size_t J = 0; J < dim_space_; ++J) {
        auto vJ = evecs_->get(J, root2_);

        for (const auto& aaaJ_mo_sign : ann_list[J]) {
            const auto [aaaJ_add, _p, q, r] = aaaJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            for (const auto& a6J : cre_list[aaaJ_add]) {
                const auto [I, _s, t, u] = a6J;
                auto s = std::abs(_s) - 1;
                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;

                auto value = evecs_->get(I, root1_) * vJ * sign;
                func({s, t, u, p, q, r}, value);
            }
        }
    }
}

void CI_RDMS::_add_3rdm_aaa(std::vector<double>& tpdm_aaa, const std::vector<size_t>& i,
                            const double& value) {
    tpdm_aaa[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] += value;
    tpdm_aaa[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] -= value;
    tpdm_aaa[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] -= value;
    tpdm_aaa[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] += value;
    tpdm_aaa[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] -= value;
    tpdm_aaa[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] += value;

    tpdm_aaa[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] -= value;
    tpdm_aaa[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] += value;
    tpdm_aaa[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] += value;
    tpdm_aaa[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] -= value;
    tpdm_aaa[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] += value;
    tpdm_aaa[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] -= value;

    tpdm_aaa[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] -= value;
    tpdm_aaa[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] += value;
    tpdm_aaa[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] += value;
    tpdm_aaa[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] -= value;
    tpdm_aaa[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] += value;
    tpdm_aaa[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] -= value;

    tpdm_aaa[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] += value;
    tpdm_aaa[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] -= value;
    tpdm_aaa[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] -= value;
    tpdm_aaa[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] += value;
    tpdm_aaa[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] -= value;
    tpdm_aaa[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] += value;

    tpdm_aaa[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] += value;
    tpdm_aaa[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] -= value;
    tpdm_aaa[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] -= value;
    tpdm_aaa[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] += value;
    tpdm_aaa[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] -= value;
    tpdm_aaa[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] += value;

    tpdm_aaa[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] -= value;
    tpdm_aaa[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] += value;
    tpdm_aaa[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] += value;
    tpdm_aaa[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] -= value;
    tpdm_aaa[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] += value;
    tpdm_aaa[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] -= value;
}

void CI_RDMS::_add_3rdm_aab(std::vector<double>& aab, const std::vector<size_t>& i,
                            const double& value, const bool& spin_free) {
    aab[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] += value;
    aab[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] -= value;
    aab[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] -= value;
    aab[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] += value;

    if (spin_free) {
        aab[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] += value;
        aab[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] -= value;
        aab[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] -= value;
        aab[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] += value;

        aab[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] += value;
        aab[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] -= value;
        aab[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] -= value;
        aab[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] += value;
    }
}

void CI_RDMS::_add_3rdm_abb(std::vector<double>& abb, const std::vector<size_t>& i,
                            const double& value, const bool& spin_free) {
    abb[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] += value;
    abb[i[0] * no5_ + i[1] * no4_ + i[2] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] -= value;
    abb[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[3] * no2_ + i[4] * no_ + i[5]] -= value;
    abb[i[0] * no5_ + i[2] * no4_ + i[1] * no3_ + i[3] * no2_ + i[5] * no_ + i[4]] += value;

    if (spin_free) {
        abb[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] += value;
        abb[i[1] * no5_ + i[0] * no4_ + i[2] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] -= value;
        abb[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[4] * no2_ + i[3] * no_ + i[5]] -= value;
        abb[i[2] * no5_ + i[0] * no4_ + i[1] * no3_ + i[5] * no2_ + i[3] * no_ + i[4]] += value;

        abb[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] += value;
        abb[i[2] * no5_ + i[1] * no4_ + i[0] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] -= value;
        abb[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[5] * no2_ + i[4] * no_ + i[3]] -= value;
        abb[i[1] * no5_ + i[2] * no4_ + i[0] * no3_ + i[4] * no2_ + i[5] * no_ + i[3]] += value;
    }
}

void CI_RDMS::compute_3rdm_op(std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab,
                              std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb) {

    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->three_s_lists(wfn_);

    local_timer build;

    tprdm_aaa.assign(no6_, 0.0);
    tprdm_aab.assign(no6_, 0.0);
    tprdm_abb.assign(no6_, 0.0);
    tprdm_bbb.assign(no6_, 0.0);

    // Build the diagonal part
    _add_3rdm_op_II([&](const std::vector<size_t>& i,
                        const double& value) { _add_3rdm_aaa(tprdm_aaa, i, value); },
                    [&](const std::vector<size_t>& i, const double& value) {
                        _add_3rdm_aab(tprdm_aab, i, value, false);
                    },
                    [&](const std::vector<size_t>& i, const double& value) {
                        _add_3rdm_abb(tprdm_abb, i, value, false);
                    },
                    [&](const std::vector<size_t>& i, const double& value) {
                        _add_3rdm_aaa(tprdm_bbb, i, value);
                    });

    //    const det_hashvec& dets = wfn_.wfn_hash();
    //    for (size_t I = 0; I < dim_space_; ++I) {
    //        double cI_sq = evecs_->get(I, root1_) * evecs_->get(I, root2_);
    //        Determinant detI(dets[I]);
    //
    //        std::vector<int> aocc = detI.get_alfa_occ(no_);
    //        std::vector<int> bocc = detI.get_beta_occ(no_);
    //        int na = aocc.size();
    //        int nb = bocc.size();
    //
    //        for (int p = 0; p < na; ++p) {
    //            int pp = aocc[p];
    //            for (int q = p + 1; q < na; ++q) {
    //                int qq = aocc[q];
    //                for (int r = q + 1; r < na; ++r) {
    //                    int rr = aocc[r];
    //
    //                    tprdm_aaa[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] += cI_sq;
    //                    tprdm_aaa[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_aaa[pp * no5_ + qq * no4_ + rr * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_aaa[pp * no5_ + qq * no4_ + rr * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_aaa[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_aaa[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] -= cI_sq;
    //
    //                    tprdm_aaa[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_aaa[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_aaa[pp * no5_ + rr * no4_ + qq * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_aaa[pp * no5_ + rr * no4_ + qq * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_aaa[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_aaa[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] += cI_sq;
    //
    //                    tprdm_aaa[rr * no5_ + pp * no4_ + qq * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] += cI_sq;
    //                    tprdm_aaa[rr * no5_ + pp * no4_ + qq * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_aaa[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_aaa[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_aaa[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_aaa[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] -= cI_sq;
    //
    //                    tprdm_aaa[rr * no5_ + qq * no4_ + pp * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_aaa[rr * no5_ + qq * no4_ + pp * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_aaa[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_aaa[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_aaa[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_aaa[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] += cI_sq;
    //
    //                    tprdm_aaa[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] += cI_sq;
    //                    tprdm_aaa[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_aaa[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_aaa[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_aaa[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_aaa[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] -= cI_sq;
    //
    //                    tprdm_aaa[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_aaa[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_aaa[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_aaa[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_aaa[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_aaa[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] += cI_sq;
    //                }
    //            }
    //        }
    //
    //        for (int p = 0; p < na; ++p) {
    //            int pp = aocc[p];
    //            for (int q = p + 1; q < na; ++q) {
    //                int qq = aocc[q];
    //                for (int r = 0; r < nb; ++r) {
    //                    int rr = bocc[r];
    //
    //                    tprdm_aab[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] += cI_sq;
    //                    tprdm_aab[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_aab[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_aab[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] += cI_sq;
    //                }
    //            }
    //        }
    //
    //        for (int p = 0; p < na; ++p) {
    //            int pp = aocc[p];
    //            for (int q = 0; q < nb; ++q) {
    //                int qq = bocc[q];
    //                for (int r = q + 1; r < nb; ++r) {
    //                    int rr = bocc[r];
    //
    //                    tprdm_abb[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] += cI_sq;
    //                    tprdm_abb[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_abb[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_abb[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] += cI_sq;
    //                }
    //            }
    //        }
    //
    //        for (int p = 0; p < nb; ++p) {
    //            int pp = bocc[p];
    //            for (int q = p + 1; q < nb; ++q) {
    //                int qq = bocc[q];
    //                for (int r = q + 1; r < nb; ++r) {
    //                    int rr = bocc[r];
    //
    //                    tprdm_bbb[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] += cI_sq;
    //                    tprdm_bbb[pp * no5_ + qq * no4_ + rr * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_bbb[pp * no5_ + qq * no4_ + rr * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_bbb[pp * no5_ + qq * no4_ + rr * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_bbb[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_bbb[pp * no5_ + qq * no4_ + rr * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] -= cI_sq;
    //
    //                    tprdm_bbb[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_bbb[pp * no5_ + rr * no4_ + qq * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_bbb[pp * no5_ + rr * no4_ + qq * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_bbb[pp * no5_ + rr * no4_ + qq * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_bbb[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_bbb[pp * no5_ + rr * no4_ + qq * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] += cI_sq;
    //
    //                    tprdm_bbb[rr * no5_ + pp * no4_ + qq * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] += cI_sq;
    //                    tprdm_bbb[rr * no5_ + pp * no4_ + qq * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_bbb[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_bbb[rr * no5_ + pp * no4_ + qq * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_bbb[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_bbb[rr * no5_ + pp * no4_ + qq * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] -= cI_sq;
    //
    //                    tprdm_bbb[rr * no5_ + qq * no4_ + pp * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_bbb[rr * no5_ + qq * no4_ + pp * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_bbb[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_bbb[rr * no5_ + qq * no4_ + pp * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_bbb[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_bbb[rr * no5_ + qq * no4_ + pp * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] += cI_sq;
    //
    //                    tprdm_bbb[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] += cI_sq;
    //                    tprdm_bbb[qq * no5_ + rr * no4_ + pp * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_bbb[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_bbb[qq * no5_ + rr * no4_ + pp * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_bbb[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_bbb[qq * no5_ + rr * no4_ + pp * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] -= cI_sq;
    //
    //                    tprdm_bbb[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + qq *
    //                    no_ +
    //                              rr] -= cI_sq;
    //                    tprdm_bbb[qq * no5_ + pp * no4_ + rr * no3_ + pp * no2_ + rr *
    //                    no_ +
    //                              qq] += cI_sq;
    //                    tprdm_bbb[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + pp *
    //                    no_ +
    //                              qq] -= cI_sq;
    //                    tprdm_bbb[qq * no5_ + pp * no4_ + rr * no3_ + rr * no2_ + qq *
    //                    no_ +
    //                              pp] += cI_sq;
    //                    tprdm_bbb[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + rr *
    //                    no_ +
    //                              pp] -= cI_sq;
    //                    tprdm_bbb[qq * no5_ + pp * no4_ + rr * no3_ + qq * no2_ + pp *
    //                    no_ +
    //                              rr] += cI_sq;
    //                }
    //            }
    //        }
    //    }

    _add_3rdm_op_IJ(op->aaa_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aaa(tprdm_aaa, i, value);
    });
    _add_3rdm_op_IJ(op->aab_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aab(tprdm_aab, i, value, false);
    });
    _add_3rdm_op_IJ(op->abb_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_abb(tprdm_abb, i, value, false);
    });
    _add_3rdm_op_IJ(op->bbb_list_, [&](const std::vector<size_t>& i, const double& value) {
        _add_3rdm_aaa(tprdm_bbb, i, value);
    });

    //    for (size_t K = 0, max_K = aaa_list.size(); K < max_K; ++K) {
    //        // aaa aaa
    //        std::vector<std::tuple<size_t, short, short, short>>& coupled_dets = aaa_list[K];
    //        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
    //
    //            auto& detJ = coupled_dets[a];
    //
    //            const size_t& J = std::get<0>(detJ);
    //            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
    //            const size_t& q = std::get<2>(detJ);
    //            const size_t& r = std::get<3>(detJ);
    //            const double& sign_pqr = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
    //
    //            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
    //
    //                auto& detI = coupled_dets[b];
    //
    //                const size_t& s = std::abs(std::get<1>(detI)) - 1;
    //                const size_t& t = std::get<2>(detI);
    //                const size_t& u = std::get<3>(detI);
    //                const double& sign_stu = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
    //                const size_t& I = std::get<0>(detI);
    //
    //                double el = evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_pqr *
    //                sign_stu;
    //
    //                tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_aaa[p * no5_ + q * no4_ + r * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_aaa[p * no5_ + r * no4_ + q * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_aaa[q * no5_ + p * no4_ + r * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_aaa[q * no5_ + r * no4_ + p * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_aaa[r * no5_ + p * no4_ + q * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_aaa[r * no5_ + q * no4_ + p * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;
    //
    //                tprdm_aaa[s * no5_ + t * no4_ + u * no3_ + p * no2_ + q * no_ + r] +=
    //                el; tprdm_aaa[s * no5_ + u * no4_ + t * no3_ + p * no2_ + q * no_ +
    //                r] -= el; tprdm_aaa[u * no5_ + t * no4_ + s * no3_ + p * no2_ + q *
    //                no_ + r] -= el; tprdm_aaa[u * no5_ + s * no4_ + t * no3_ + p * no2_ +
    //                q * no_ + r] += el; tprdm_aaa[t * no5_ + s * no4_ + u * no3_ + p *
    //                no2_ + q * no_ + r] -= el; tprdm_aaa[t * no5_ + u * no4_ + s * no3_ +
    //                p * no2_ + q * no_ + r] += el;
    //
    //                tprdm_aaa[s * no5_ + t * no4_ + u * no3_ + p * no2_ + r * no_ + q] -=
    //                el; tprdm_aaa[s * no5_ + u * no4_ + t * no3_ + p * no2_ + r * no_ +
    //                q] += el; tprdm_aaa[u * no5_ + t * no4_ + s * no3_ + p * no2_ + r *
    //                no_ + q] += el; tprdm_aaa[u * no5_ + s * no4_ + t * no3_ + p * no2_ +
    //                r * no_ + q] -= el; tprdm_aaa[t * no5_ + s * no4_ + u * no3_ + p *
    //                no2_ + r * no_ + q] += el; tprdm_aaa[t * no5_ + u * no4_ + s * no3_ +
    //                p * no2_ + r * no_ + q] -= el;
    //
    //                tprdm_aaa[s * no5_ + t * no4_ + u * no3_ + q * no2_ + p * no_ + r] -=
    //                el; tprdm_aaa[s * no5_ + u * no4_ + t * no3_ + q * no2_ + p * no_ +
    //                r] += el; tprdm_aaa[u * no5_ + t * no4_ + s * no3_ + q * no2_ + p *
    //                no_ + r] += el; tprdm_aaa[u * no5_ + s * no4_ + t * no3_ + q * no2_ +
    //                p * no_ + r] -= el; tprdm_aaa[t * no5_ + s * no4_ + u * no3_ + q *
    //                no2_ + p * no_ + r] += el; tprdm_aaa[t * no5_ + u * no4_ + s * no3_ +
    //                q * no2_ + p * no_ + r] -= el;
    //
    //                tprdm_aaa[s * no5_ + t * no4_ + u * no3_ + q * no2_ + r * no_ + p] +=
    //                el; tprdm_aaa[s * no5_ + u * no4_ + t * no3_ + q * no2_ + r * no_ +
    //                p] -= el; tprdm_aaa[u * no5_ + t * no4_ + s * no3_ + q * no2_ + r *
    //                no_ + p] -= el; tprdm_aaa[u * no5_ + s * no4_ + t * no3_ + q * no2_ +
    //                r * no_ + p] += el; tprdm_aaa[t * no5_ + s * no4_ + u * no3_ + q *
    //                no2_ + r * no_ + p] -= el; tprdm_aaa[t * no5_ + u * no4_ + s * no3_ +
    //                q * no2_ + r * no_ + p] += el;
    //
    //                tprdm_aaa[s * no5_ + t * no4_ + u * no3_ + r * no2_ + p * no_ + q] +=
    //                el; tprdm_aaa[s * no5_ + u * no4_ + t * no3_ + r * no2_ + p * no_ +
    //                q] -= el; tprdm_aaa[u * no5_ + t * no4_ + s * no3_ + r * no2_ + p *
    //                no_ + q] -= el; tprdm_aaa[u * no5_ + s * no4_ + t * no3_ + r * no2_ +
    //                p * no_ + q] += el; tprdm_aaa[t * no5_ + s * no4_ + u * no3_ + r *
    //                no2_ + p * no_ + q] -= el; tprdm_aaa[t * no5_ + u * no4_ + s * no3_ +
    //                r * no2_ + p * no_ + q] += el;
    //
    //                tprdm_aaa[s * no5_ + t * no4_ + u * no3_ + r * no2_ + q * no_ + p] -=
    //                el; tprdm_aaa[s * no5_ + u * no4_ + t * no3_ + r * no2_ + q * no_ +
    //                p] += el; tprdm_aaa[u * no5_ + t * no4_ + s * no3_ + r * no2_ + q *
    //                no_ + p] += el; tprdm_aaa[u * no5_ + s * no4_ + t * no3_ + r * no2_ +
    //                q * no_ + p] -= el; tprdm_aaa[t * no5_ + s * no4_ + u * no3_ + r *
    //                no2_ + q * no_ + p] += el; tprdm_aaa[t * no5_ + u * no4_ + s * no3_ +
    //                r * no2_ + q * no_ + p] -= el;
    //            }
    //        }
    //    }
    //
    //    for (size_t K = 0, max_K = aab_list.size(); K < max_K; ++K) {
    //        // aab aab
    //        std::vector<std::tuple<size_t, short, short, short>>& coupled_dets = aab_list[K];
    //        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
    //
    //            auto& detJ = coupled_dets[a];
    //
    //            const size_t& J = std::get<0>(detJ);
    //            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
    //            const size_t& q = std::get<2>(detJ);
    //            const size_t& r = std::get<3>(detJ);
    //            const double& sign_pqr = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
    //
    //            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
    //
    //                auto& detI = coupled_dets[b];
    //
    //                const size_t& s = std::abs(std::get<1>(detI)) - 1;
    //                const size_t& t = std::get<2>(detI);
    //                const size_t& u = std::get<3>(detI);
    //                const double& sign_stu = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
    //                const size_t& I = std::get<0>(detI);
    //
    //                double el = evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_pqr *
    //                sign_stu; tprdm_aab[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t *
    //                no_ + u] += el; tprdm_aab[p * no5_ + q * no4_ + r * no3_ + t * no2_ +
    //                s * no_ + u] -= el; tprdm_aab[q * no5_ + p * no4_ + r * no3_ + s *
    //                no2_ + t * no_ + u] -= el; tprdm_aab[q * no5_ + p * no4_ + r * no3_ +
    //                t * no2_ + s * no_ + u] += el;
    //
    //                el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;
    //                tprdm_aab[s * no5_ + t * no4_ + u * no3_ + p * no2_ + q * no_ + r] +=
    //                el; tprdm_aab[t * no5_ + s * no4_ + u * no3_ + p * no2_ + q * no_ +
    //                r] -= el; tprdm_aab[s * no5_ + t * no4_ + u * no3_ + q * no2_ + p *
    //                no_ + r] -= el; tprdm_aab[t * no5_ + s * no4_ + u * no3_ + q * no2_ +
    //                p * no_ + r] += el;
    //            }
    //        }
    //    }
    //
    //    // abb abb
    //    for (size_t K = 0, max_K = abb_list.size(); K < max_K; ++K) {
    //        std::vector<std::tuple<size_t, short, short, short>>& coupled_dets = abb_list[K];
    //        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
    //
    //            auto& detJ = coupled_dets[a];
    //
    //            const size_t& J = std::get<0>(detJ);
    //            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
    //            const size_t& q = std::get<2>(detJ);
    //            const size_t& r = std::get<3>(detJ);
    //            const double& sign_pqr = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
    //
    //            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
    //
    //                auto& detI = coupled_dets[b];
    //
    //                const size_t& s = std::abs(std::get<1>(detI)) - 1;
    //                const size_t& t = std::get<2>(detI);
    //                const size_t& u = std::get<3>(detI);
    //                const double& sign_stu = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
    //                const size_t& I = std::get<0>(detI);
    //
    //                double el = evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_pqr *
    //                sign_stu; tprdm_abb[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t *
    //                no_ + u] += el; tprdm_abb[p * no5_ + q * no4_ + r * no3_ + s * no2_ +
    //                u * no_ + t] -= el; tprdm_abb[p * no5_ + r * no4_ + q * no3_ + s *
    //                no2_ + t * no_ + u] -= el; tprdm_abb[p * no5_ + r * no4_ + q * no3_ +
    //                s * no2_ + u * no_ + t] += el;
    //
    //                el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;
    //                tprdm_abb[s * no5_ + t * no4_ + u * no3_ + p * no2_ + q * no_ + r] +=
    //                el; tprdm_abb[s * no5_ + u * no4_ + t * no3_ + p * no2_ + q * no_ +
    //                r] -= el; tprdm_abb[s * no5_ + t * no4_ + u * no3_ + p * no2_ + r *
    //                no_ + q] -= el; tprdm_abb[s * no5_ + u * no4_ + t * no3_ + p * no2_ +
    //                r * no_ + q] += el;
    //            }
    //        }
    //    }
    //
    //    for (size_t K = 0, max_K = bbb_list.size(); K < max_K; ++K) {
    //        // bbb bbb
    //        std::vector<std::tuple<size_t, short, short, short>>& coupled_dets = bbb_list[K];
    //        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
    //
    //            auto& detJ = coupled_dets[a];
    //
    //            const size_t& J = std::get<0>(detJ);
    //            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
    //            const size_t& q = std::get<2>(detJ);
    //            const size_t& r = std::get<3>(detJ);
    //            const double& sign_pqr = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
    //
    //            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
    //
    //                auto& detI = coupled_dets[b];
    //
    //                const size_t& s = std::abs(std::get<1>(detI)) - 1;
    //                const size_t& t = std::get<2>(detI);
    //                const size_t& u = std::get<3>(detI);
    //                const double& sign_stu = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
    //                const size_t& I = std::get<0>(detI);
    //
    //                double el = evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_pqr *
    //                sign_stu;
    //
    //                tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_bbb[p * no5_ + q * no4_ + r * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_bbb[p * no5_ + r * no4_ + q * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_bbb[q * no5_ + p * no4_ + r * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_bbb[q * no5_ + r * no4_ + p * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + s * no2_ + t * no_ + u] +=
    //                el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + s * no2_ + u * no_ +
    //                t] -= el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + u * no2_ + t *
    //                no_ + s] -= el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + u * no2_ +
    //                s * no_ + t] += el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ + t *
    //                no2_ + s * no_ + u] -= el; tprdm_bbb[r * no5_ + p * no4_ + q * no3_ +
    //                t * no2_ + u * no_ + s] += el;
    //
    //                tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + s * no2_ + t * no_ + u] -=
    //                el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + s * no2_ + u * no_ +
    //                t] += el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + u * no2_ + t *
    //                no_ + s] += el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + u * no2_ +
    //                s * no_ + t] -= el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ + t *
    //                no2_ + s * no_ + u] += el; tprdm_bbb[r * no5_ + q * no4_ + p * no3_ +
    //                t * no2_ + u * no_ + s] -= el;
    //
    //                el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;
    //
    //                tprdm_bbb[s * no5_ + t * no4_ + u * no3_ + p * no2_ + q * no_ + r] +=
    //                el; tprdm_bbb[s * no5_ + u * no4_ + t * no3_ + p * no2_ + q * no_ +
    //                r] -= el; tprdm_bbb[u * no5_ + t * no4_ + s * no3_ + p * no2_ + q *
    //                no_ + r] -= el; tprdm_bbb[u * no5_ + s * no4_ + t * no3_ + p * no2_ +
    //                q * no_ + r] += el; tprdm_bbb[t * no5_ + s * no4_ + u * no3_ + p *
    //                no2_ + q * no_ + r] -= el; tprdm_bbb[t * no5_ + u * no4_ + s * no3_ +
    //                p * no2_ + q * no_ + r] += el;
    //
    //                tprdm_bbb[s * no5_ + t * no4_ + u * no3_ + p * no2_ + r * no_ + q] -=
    //                el; tprdm_bbb[s * no5_ + u * no4_ + t * no3_ + p * no2_ + r * no_ +
    //                q] += el; tprdm_bbb[u * no5_ + t * no4_ + s * no3_ + p * no2_ + r *
    //                no_ + q] += el; tprdm_bbb[u * no5_ + s * no4_ + t * no3_ + p * no2_ +
    //                r * no_ + q] -= el; tprdm_bbb[t * no5_ + s * no4_ + u * no3_ + p *
    //                no2_ + r * no_ + q] += el; tprdm_bbb[t * no5_ + u * no4_ + s * no3_ +
    //                p * no2_ + r * no_ + q] -= el;
    //
    //                tprdm_bbb[s * no5_ + t * no4_ + u * no3_ + q * no2_ + p * no_ + r] -=
    //                el; tprdm_bbb[s * no5_ + u * no4_ + t * no3_ + q * no2_ + p * no_ +
    //                r] += el; tprdm_bbb[u * no5_ + t * no4_ + s * no3_ + q * no2_ + p *
    //                no_ + r] += el; tprdm_bbb[u * no5_ + s * no4_ + t * no3_ + q * no2_ +
    //                p * no_ + r] -= el; tprdm_bbb[t * no5_ + s * no4_ + u * no3_ + q *
    //                no2_ + p * no_ + r] += el; tprdm_bbb[t * no5_ + u * no4_ + s * no3_ +
    //                q * no2_ + p * no_ + r] -= el;
    //
    //                tprdm_bbb[s * no5_ + t * no4_ + u * no3_ + q * no2_ + r * no_ + p] +=
    //                el; tprdm_bbb[s * no5_ + u * no4_ + t * no3_ + q * no2_ + r * no_ +
    //                p] -= el; tprdm_bbb[u * no5_ + t * no4_ + s * no3_ + q * no2_ + r *
    //                no_ + p] -= el; tprdm_bbb[u * no5_ + s * no4_ + t * no3_ + q * no2_ +
    //                r * no_ + p] += el; tprdm_bbb[t * no5_ + s * no4_ + u * no3_ + q *
    //                no2_ + r * no_ + p] -= el; tprdm_bbb[t * no5_ + u * no4_ + s * no3_ +
    //                q * no2_ + r * no_ + p] += el;
    //
    //                tprdm_bbb[s * no5_ + t * no4_ + u * no3_ + r * no2_ + p * no_ + q] +=
    //                el; tprdm_bbb[s * no5_ + u * no4_ + t * no3_ + r * no2_ + p * no_ +
    //                q] -= el; tprdm_bbb[u * no5_ + t * no4_ + s * no3_ + r * no2_ + p *
    //                no_ + q] -= el; tprdm_bbb[u * no5_ + s * no4_ + t * no3_ + r * no2_ +
    //                p * no_ + q] += el; tprdm_bbb[t * no5_ + s * no4_ + u * no3_ + r *
    //                no2_ + p * no_ + q] -= el; tprdm_bbb[t * no5_ + u * no4_ + s * no3_ +
    //                r * no2_ + p * no_ + q] += el;
    //
    //                tprdm_bbb[s * no5_ + t * no4_ + u * no3_ + r * no2_ + q * no_ + p] -=
    //                el; tprdm_bbb[s * no5_ + u * no4_ + t * no3_ + r * no2_ + q * no_ +
    //                p] += el; tprdm_bbb[u * no5_ + t * no4_ + s * no3_ + r * no2_ + q *
    //                no_ + p] += el; tprdm_bbb[u * no5_ + s * no4_ + t * no3_ + r * no2_ +
    //                q * no_ + p] -= el; tprdm_bbb[t * no5_ + s * no4_ + u * no3_ + r *
    //                no2_ + q * no_ + p] += el; tprdm_bbb[t * no5_ + u * no4_ + s * no3_ +
    //                r * no2_ + q * no_ + p] -= el;
    //            }
    //        }
    //    }

    if (print_) {
        outfile->Printf("\n  Time spent building 3-rdm: %.3e seconds", build.get());
    }
}

void CI_RDMS::_add_3rdm_op_II(
    const std::function<void(const std::vector<size_t>&, const double&)>& func_aaa,
    const std::function<void(const std::vector<size_t>&, const double&)>& func_aab,
    const std::function<void(const std::vector<size_t>&, const double&)>& func_abb,
    const std::function<void(const std::vector<size_t>&, const double&)>& func_bbb) {
    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t I = 0; I < dim_space_; ++I) {
        double cI_sq = evecs_->get(I, root1_) * evecs_->get(I, root2_);

        std::vector<int> aocc = dets[I].get_alfa_occ(no_);
        std::vector<int> bocc = dets[I].get_beta_occ(no_);
        auto na = aocc.size();
        auto nb = bocc.size();

        for (size_t p = 0; p < na; ++p) {
            size_t pp = aocc[p];
            for (size_t q = p + 1; q < na; ++q) {
                size_t qq = aocc[q];
                for (size_t r = q + 1; r < na; ++r) {
                    size_t rr = aocc[r];

                    func_aaa({pp, qq, rr, pp, qq, rr}, cI_sq);
                }
            }
        }
        for (size_t p = 0; p < na; ++p) {
            size_t pp = aocc[p];
            for (size_t q = p + 1; q < na; ++q) {
                size_t qq = aocc[q];
                for (size_t r = 0; r < nb; ++r) {
                    size_t rr = bocc[r];

                    func_aab({pp, qq, rr, pp, qq, rr}, cI_sq);
                }
            }
        }
        for (size_t p = 0; p < na; ++p) {
            size_t pp = aocc[p];
            for (size_t q = 0; q < nb; ++q) {
                size_t qq = bocc[q];
                for (size_t r = q + 1; r < nb; ++r) {
                    size_t rr = bocc[r];

                    func_abb({pp, qq, rr, pp, qq, rr}, cI_sq);
                }
            }
        }
        for (size_t p = 0; p < nb; ++p) {
            size_t pp = bocc[p];
            for (size_t q = p + 1; q < nb; ++q) {
                size_t qq = bocc[q];
                for (size_t r = q + 1; r < nb; ++r) {
                    size_t rr = bocc[r];

                    func_bbb({pp, qq, rr, pp, qq, rr}, cI_sq);
                }
            }
        }
    }
}

void CI_RDMS::_add_3rdm_op_IJ(
    const std::vector<std::vector<std::tuple<size_t, short, short, short>>>& list,
    const std::function<void(const std::vector<size_t>&, const double&)>& func) {
    for (const auto& coupled_dets : list) {
        for (size_t a = 0, coupled_dets_size = coupled_dets.size(); a < coupled_dets_size; ++a) {
            const auto [J, _p, q, r] = coupled_dets[a];
            auto p = std::abs(_p) - 1;
            auto sign_pqr = _p < 0;

            auto vJ1 = evecs_->get(J, root1_);
            auto vJ2 = evecs_->get(J, root2_);

            for (size_t b = a + 1; b < coupled_dets_size; ++b) {
                const auto [I, _s, t, u] = coupled_dets[b];
                auto s = std::abs(_s) - 1;
                auto sign = ((_s < 0) == sign_pqr) ? 1 : -1;

                auto eJI = vJ1 * evecs_->get(I, root2_) * sign;
                func({p, q, r, s, t, u}, eJI);

                auto eIJ = evecs_->get(I, root1_) * vJ2 * sign;
                func({s, t, u, p, q, r}, eJI);
            }
        }
    }
}

void CI_RDMS::get_one_map() {
    // The alpha and beta annihilation lists
    a_ann_list_.resize(dim_space_);
    b_ann_list_.resize(dim_space_);

    // The N-1 maps
    det_hash a_ann_map;
    det_hash b_ann_map;

    // Number of annihilations on alfa and beta strings
    size_t na_ann = 0;
    size_t nb_ann = 0;

    if (print_)
        outfile->Printf("\n\n  Generating one-particle maps.");

    for (size_t I = 0; I < dim_space_; ++I) {
        Determinant detI(det_space_[I]);

        // Alpha and beta occupation vectors
        std::vector<int> aocc = detI.get_alfa_occ(no_);
        std::vector<int> bocc = detI.get_beta_occ(no_);

        int noalfa = aocc.size();
        int nobeta = bocc.size();

        std::vector<std::pair<size_t, short>> a_ann(noalfa);
        std::vector<std::pair<size_t, short>> b_ann(nobeta);

        // Form alpha annihilation lists
        for (int i = 0; i < noalfa; ++i) {
            int ii = aocc[i];
            Determinant detJ(detI);

            // Annihilate bit ii, get the sign
            detJ.set_alfa_bit(ii, false);
            double sign = detI.slater_sign_a(ii);

            det_hash_it hash_it = a_ann_map.find(detJ);
            size_t detJ_add;
            if (hash_it == a_ann_map.end()) {
                detJ_add = na_ann;
                a_ann_map[detJ] = na_ann;
                na_ann++;
            } else {
                detJ_add = hash_it->second;
            }
            a_ann[i] = std::make_pair(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1));
        }
        a_ann_list_[I] = a_ann;

        // Form beta annihilation lists
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            Determinant detJ(detI);

            // Annihilate bit ii, get the sign
            detJ.set_beta_bit(ii, false);
            double sign = detI.slater_sign_b(ii);

            det_hash_it hash_it = b_ann_map.find(detJ);
            size_t detJ_add;
            if (hash_it == b_ann_map.end()) {
                detJ_add = nb_ann;
                b_ann_map[detJ] = nb_ann;
                nb_ann++;
            } else {
                detJ_add = hash_it->second;
            }
            b_ann[i] = std::make_pair(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1));
        }
        b_ann_list_[I] = b_ann;
    } // Done with annihilation lists

    // Generate alpha and beta creation lists
    a_cre_list_.resize(a_ann_map.size());
    b_cre_list_.resize(b_ann_map.size());

    for (size_t I = 0; I < dim_space_; ++I) {
        const std::vector<std::pair<size_t, short>>& a_ann = a_ann_list_[I];
        for (const std::pair<size_t, short>& Jsign : a_ann) {
            size_t J = Jsign.first;
            short sign = Jsign.second;
            a_cre_list_[J].push_back(std::make_pair(I, sign));
        }
        const std::vector<std::pair<size_t, short>>& b_ann = b_ann_list_[I];
        for (const std::pair<size_t, short>& Jsign : b_ann) {
            size_t J = Jsign.first;
            short sign = Jsign.second;
            b_cre_list_[J].push_back(std::make_pair(I, sign));
        }
    }
    one_map_done_ = true;
}

void CI_RDMS::get_two_map() {
    aa_ann_list_.resize(dim_space_);
    ab_ann_list_.resize(dim_space_);
    bb_ann_list_.resize(dim_space_);

    det_hash aa_ann_map;
    det_hash ab_ann_map;
    det_hash bb_ann_map;

    size_t naa_ann = 0;
    size_t nab_ann = 0;
    size_t nbb_ann = 0;

    if (print_)
        outfile->Printf("\n  Generating two-particle maps.");

    for (size_t I = 0; I < dim_space_; ++I) {
        Determinant detI(det_space_[I]);

        std::vector<int> aocc = detI.get_alfa_occ(no_);
        std::vector<int> bocc = detI.get_beta_occ(no_);

        int noalfa = aocc.size();
        int nobeta = bocc.size();

        std::vector<std::tuple<size_t, short, short>> aa_ann(noalfa * (noalfa - 1) / 2);
        std::vector<std::tuple<size_t, short, short>> ab_ann(noalfa * nobeta);
        std::vector<std::tuple<size_t, short, short>> bb_ann(nobeta * (nobeta - 1) / 2);

        // alpha-alpha annihilations
        for (int i = 0, ij = 0; i < noalfa; ++i) {
            for (int j = i + 1; j < noalfa; ++j, ++ij) {
                int ii = aocc[i];
                int jj = aocc[j];

                Determinant detJ(detI);
                detJ.set_alfa_bit(ii, false);
                detJ.set_alfa_bit(jj, false);

                double sign = detI.slater_sign_a(ii) * detI.slater_sign_a(jj);

                det_hash_it hash_it = aa_ann_map.find(detJ);
                size_t detJ_add;
                if (hash_it == aa_ann_map.end()) {
                    detJ_add = naa_ann;
                    aa_ann_map[detJ] = naa_ann;
                    naa_ann++;
                } else {
                    detJ_add = hash_it->second;
                }
                aa_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
            }
        }
        aa_ann_list_[I] = aa_ann;

        // beta-beta  annihilations
        for (int i = 0, ij = 0; i < nobeta; ++i) {
            for (int j = i + 1; j < nobeta; ++j, ++ij) {
                int ii = bocc[i];
                int jj = bocc[j];

                Determinant detJ(detI);
                detJ.set_beta_bit(ii, false);
                detJ.set_beta_bit(jj, false);

                double sign = detI.slater_sign_b(ii) * detI.slater_sign_b(jj);

                det_hash_it hash_it = bb_ann_map.find(detJ);
                size_t detJ_add;
                if (hash_it == bb_ann_map.end()) {
                    detJ_add = nbb_ann;
                    bb_ann_map[detJ] = nbb_ann;
                    nbb_ann++;
                } else {
                    detJ_add = hash_it->second;
                }
                bb_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
            }
        }
        bb_ann_list_[I] = bb_ann;

        // alpha-beta  annihilations
        for (int i = 0, ij = 0; i < noalfa; ++i) {
            for (int j = 0; j < nobeta; ++j, ++ij) {
                int ii = aocc[i];
                int jj = bocc[j];

                Determinant detJ(detI);
                detJ.set_alfa_bit(ii, false);
                detJ.set_beta_bit(jj, false);

                double sign = detI.slater_sign_a(ii) * detI.slater_sign_b(jj);

                det_hash_it hash_it = ab_ann_map.find(detJ);
                size_t detJ_add;
                if (hash_it == ab_ann_map.end()) {
                    detJ_add = nab_ann;
                    ab_ann_map[detJ] = nab_ann;
                    nab_ann++;
                } else {
                    detJ_add = hash_it->second;
                }
                ab_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
            }
        }
        ab_ann_list_[I] = ab_ann;
    } // Done building 2-hole lists

    aa_cre_list_.resize(aa_ann_map.size());
    ab_cre_list_.resize(ab_ann_map.size());
    bb_cre_list_.resize(bb_ann_map.size());

    for (size_t I = 0; I < dim_space_; ++I) {
        // alpha-alpha
        const std::vector<std::tuple<size_t, short, short>>& aa_ann = aa_ann_list_[I];
        for (const std::tuple<size_t, short, short>& Jsign : aa_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            aa_cre_list_[J].push_back(std::make_tuple(I, i, j));
        }
        // beta-beta
        const std::vector<std::tuple<size_t, short, short>>& bb_ann = bb_ann_list_[I];
        for (const std::tuple<size_t, short, short>& Jsign : bb_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            bb_cre_list_[J].push_back(std::make_tuple(I, i, j));
        }
        // alpha-alpha
        const std::vector<std::tuple<size_t, short, short>>& ab_ann = ab_ann_list_[I];
        for (const std::tuple<size_t, short, short>& Jsign : ab_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            ab_cre_list_[J].push_back(std::make_tuple(I, i, j));
        }
    }
}

void CI_RDMS::get_three_map() {
    aaa_ann_list_.resize(dim_space_);
    aab_ann_list_.resize(dim_space_);
    abb_ann_list_.resize(dim_space_);
    bbb_ann_list_.resize(dim_space_);

    det_hash aaa_ann_map;
    det_hash aab_ann_map;
    det_hash abb_ann_map;
    det_hash bbb_ann_map;

    size_t naaa_ann = 0;
    size_t naab_ann = 0;
    size_t nabb_ann = 0;
    size_t nbbb_ann = 0;

    if (print_)
        outfile->Printf("\n  Generating three-particle maps.");

    for (size_t I = 0; I < dim_space_; ++I) {
        Determinant detI(det_space_[I]);

        const std::vector<int>& aocc = detI.get_alfa_occ(no_);
        const std::vector<int>& bocc = detI.get_beta_occ(no_);

        int noalfa = aocc.size();
        int nobeta = bocc.size();

        std::vector<std::tuple<size_t, short, short, short>> aaa_ann(noalfa * (noalfa - 1) *
                                                                     (noalfa - 2) / 6);
        std::vector<std::tuple<size_t, short, short, short>> aab_ann(noalfa * (noalfa - 1) *
                                                                     nobeta / 2);
        std::vector<std::tuple<size_t, short, short, short>> abb_ann(noalfa * nobeta *
                                                                     (nobeta - 1) / 2);
        std::vector<std::tuple<size_t, short, short, short>> bbb_ann(nobeta * (nobeta - 1) *
                                                                     (nobeta - 2) / 6);

        // aaa
        for (int i = 0, ijk = 0; i < noalfa; ++i) {
            for (int j = i + 1; j < noalfa; ++j) {
                for (int k = j + 1; k < noalfa; ++k, ++ijk) {

                    int ii = aocc[i];
                    int jj = aocc[j];
                    int kk = aocc[k];

                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_alfa_bit(jj, false);
                    detJ.set_alfa_bit(kk, false);

                    double sign =
                        detI.slater_sign_a(ii) * detI.slater_sign_a(jj) * detI.slater_sign_a(kk);

                    det_hash_it hash_it = aaa_ann_map.find(detJ);
                    size_t detJ_add;

                    if (hash_it == aaa_ann_map.end()) {
                        detJ_add = naaa_ann;
                        aaa_ann_map[detJ] = naaa_ann;
                        naaa_ann++;
                    } else {
                        detJ_add = hash_it->second;
                    }
                    aaa_ann[ijk] =
                        std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk);
                }
            }
        }
        aaa_ann_list_[I] = aaa_ann;

        // aab
        for (int i = 0, ijk = 0; i < noalfa; ++i) {
            for (int j = i + 1; j < noalfa; ++j) {
                for (int k = 0; k < nobeta; ++k, ++ijk) {

                    int ii = aocc[i];
                    int jj = aocc[j];
                    int kk = bocc[k];

                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_alfa_bit(jj, false);
                    detJ.set_beta_bit(kk, false);

                    double sign =
                        detI.slater_sign_a(ii) * detI.slater_sign_a(jj) * detI.slater_sign_b(kk);

                    det_hash_it hash_it = aab_ann_map.find(detJ);
                    size_t detJ_add;

                    if (hash_it == aab_ann_map.end()) {
                        detJ_add = naab_ann;
                        aab_ann_map[detJ] = naab_ann;
                        naab_ann++;
                    } else {
                        detJ_add = hash_it->second;
                    }
                    aab_ann[ijk] =
                        std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk);
                }
            }
        }
        aab_ann_list_[I] = aab_ann;

        // abb
        for (int i = 0, ijk = 0; i < noalfa; ++i) {
            for (int j = 0; j < nobeta; ++j) {
                for (int k = j + 1; k < nobeta; ++k, ++ijk) {

                    int ii = aocc[i];
                    int jj = bocc[j];
                    int kk = bocc[k];

                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_beta_bit(jj, false);
                    detJ.set_beta_bit(kk, false);

                    double sign =
                        detI.slater_sign_a(ii) * detI.slater_sign_b(jj) * detI.slater_sign_b(kk);

                    det_hash_it hash_it = abb_ann_map.find(detJ);
                    size_t detJ_add;

                    if (hash_it == abb_ann_map.end()) {
                        detJ_add = nabb_ann;
                        abb_ann_map[detJ] = nabb_ann;
                        nabb_ann++;
                    } else {
                        detJ_add = hash_it->second;
                    }
                    abb_ann[ijk] =
                        std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk);
                }
            }
        }
        abb_ann_list_[I] = abb_ann;

        // bbb
        for (int i = 0, ijk = 0; i < nobeta; ++i) {
            for (int j = i + 1; j < nobeta; ++j) {
                for (int k = j + 1; k < nobeta; ++k, ++ijk) {

                    int ii = bocc[i];
                    int jj = bocc[j];
                    int kk = bocc[k];

                    Determinant detJ(detI);
                    detJ.set_beta_bit(ii, false);
                    detJ.set_beta_bit(jj, false);
                    detJ.set_beta_bit(kk, false);

                    double sign =
                        detI.slater_sign_b(ii) * detI.slater_sign_b(jj) * detI.slater_sign_b(kk);

                    det_hash_it hash_it = bbb_ann_map.find(detJ);
                    size_t detJ_add;

                    if (hash_it == bbb_ann_map.end()) {
                        detJ_add = nbbb_ann;
                        bbb_ann_map[detJ] = nbbb_ann;
                        nbbb_ann++;
                    } else {
                        detJ_add = hash_it->second;
                    }
                    bbb_ann[ijk] =
                        std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk);
                }
            }
        }
        bbb_ann_list_[I] = bbb_ann;
    } // End loop over determinants

    aaa_cre_list_.resize(aaa_ann_map.size());
    aab_cre_list_.resize(aab_ann_map.size());
    abb_cre_list_.resize(abb_ann_map.size());
    bbb_cre_list_.resize(bbb_ann_map.size());

    for (size_t I = 0; I < dim_space_; ++I) {
        // aaa
        const std::vector<std::tuple<size_t, short, short, short>>& aaa_ann = aaa_ann_list_[I];
        for (const std::tuple<size_t, short, short, short>& Jsign : aaa_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            short k = std::get<3>(Jsign);
            aaa_cre_list_[J].push_back(std::make_tuple(I, i, j, k));
        }
        // aab
        const std::vector<std::tuple<size_t, short, short, short>>& aab_ann = aab_ann_list_[I];
        for (const std::tuple<size_t, short, short, short>& Jsign : aab_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            short k = std::get<3>(Jsign);
            aab_cre_list_[J].push_back(std::make_tuple(I, i, j, k));
        }
        // abb
        const std::vector<std::tuple<size_t, short, short, short>>& abb_ann = abb_ann_list_[I];
        for (const std::tuple<size_t, short, short, short>& Jsign : abb_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            short k = std::get<3>(Jsign);
            abb_cre_list_[J].push_back(std::make_tuple(I, i, j, k));
        }
        // bbb
        const std::vector<std::tuple<size_t, short, short, short>>& bbb_ann = bbb_ann_list_[I];
        for (const std::tuple<size_t, short, short, short>& Jsign : bbb_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            short k = std::get<3>(Jsign);
            bbb_cre_list_[J].push_back(std::make_tuple(I, i, j, k));
        }
    }
}

void CI_RDMS::rdm_test(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b,
                       std::vector<double>& tprdm_aa, std::vector<double>& tprdm_bb,
                       std::vector<double>& tprdm_ab, std::vector<double>& tprdm_aaa,
                       std::vector<double>& tprdm_aab, std::vector<double>& tprdm_abb,
                       std::vector<double>& tprdm_bbb) {

    const det_hashvec& det_space = wfn_.wfn_hash();
    double error_1rdm_a = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        for (size_t q = 0; q < no_; ++q) {
            double rdm = 0.0;
            for (size_t i = 0; i < dim_space_; ++i) {
                Determinant I(det_space[i]);
                double sign = 1.0;
                sign *= I.destroy_alfa_bit(q);
                sign *= I.create_alfa_bit(p);
                for (size_t j = 0; j < dim_space_; ++j) {
                    if (I == det_space[j]) {
                        rdm += sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                    }
                }
            }
            if (std::fabs(rdm) > 1.0e-12) {
                error_1rdm_a += std::fabs(rdm - oprdm_a[q * no_ + p]);
                //     outfile->Printf("\n  D1(a)[%3lu][%3lu] = %18.12lf
                //     (%18.12lf,%18.12lf)", p,q,
                //     rdm-oprdm_a[p*no_+q],rdm,oprdm_a[p*no_+q]);
            }
        }
    }
    outfile->Printf("\n    A 1-RDM Error :   %2.15f", error_1rdm_a);
    double error_1rdm_b = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        for (size_t q = 0; q < no_; ++q) {
            double rdm = 0.0;
            for (size_t i = 0; i < dim_space_; ++i) {
                Determinant I(det_space[i]);
                double sign = 1.0;
                sign *= I.destroy_beta_bit(q);
                sign *= I.create_beta_bit(p);
                for (size_t j = 0; j < dim_space_; ++j) {
                    if (I == det_space[j]) {
                        rdm += sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                    }
                }
            }
            if (std::fabs(rdm) > 1.0e-12) {
                error_1rdm_b += std::fabs(rdm - oprdm_b[p * no_ + q]);
                // outfile->Printf("\n  D1(b)[%3lu][%3lu] = %18.12lf
                // (%18.12lf,%18.12lf)", p,q,
                // rdm-oprdm_b[p*no_+q],rdm,oprdm_b[p*no_+q]);
            }
        }
    }
    outfile->Printf("\n    B 1-RDM Error :   %2.15f", error_1rdm_b);

    double error_2rdm_aa = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        for (size_t q = 0; q < no_; ++q) {
            for (size_t r = 0; r < no_; ++r) {
                for (size_t s = 0; s < no_; ++s) {
                    double rdm = 0.0;
                    for (size_t i = 0; i < dim_space_; ++i) {
                        Determinant I(det_space[i]);
                        double sign = 1.0;
                        sign *= I.destroy_alfa_bit(r);
                        sign *= I.destroy_alfa_bit(s);
                        sign *= I.create_alfa_bit(q);
                        sign *= I.create_alfa_bit(p);
                        for (size_t j = 0; j < dim_space_; ++j) {
                            if (I == det_space[j]) {
                                rdm += sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                            }
                        }
                    }
                    if (std::fabs(rdm) > 1.0e-12) {
                        error_2rdm_aa +=
                            std::fabs(rdm - tprdm_aa[p * no3_ + q * no2_ + r * no_ + s]);
                        if (std::fabs(rdm - tprdm_aa[p * no3_ + q * no2_ + r * no_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(aaaa)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_aa[p * no3_ + q * no2_ + r * no_ + s], rdm,
                                            tprdm_aa[p * no3_ + q * no2_ + r * no_ + s]);
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n    AAAA 2-RDM Error :   %2.15f", error_2rdm_aa);
    double error_2rdm_bb = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        for (size_t q = 0; q < no_; ++q) {
            for (size_t r = 0; r < no_; ++r) {
                for (size_t s = 0; s < no_; ++s) {
                    double rdm = 0.0;
                    for (size_t i = 0; i < dim_space_; ++i) {
                        Determinant I(det_space[i]);
                        double sign = 1.0;
                        sign *= I.destroy_beta_bit(r);
                        sign *= I.destroy_beta_bit(s);
                        sign *= I.create_beta_bit(q);
                        sign *= I.create_beta_bit(p);
                        for (size_t j = 0; j < dim_space_; ++j) {
                            if (I == det_space[j]) {
                                rdm += sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                            }
                        }
                    }
                    if (std::fabs(rdm) > 1.0e-12) {
                        error_2rdm_bb +=
                            std::fabs(rdm - tprdm_bb[p * no3_ + q * no2_ + r * no_ + s]);
                        if (std::fabs(rdm - tprdm_bb[p * no3_ + q * no2_ + r * no_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(bbbb)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_bb[p * no3_ + q * no2_ + r * no_ + s], rdm,
                                            tprdm_bb[p * no3_ + q * no2_ + r * no_ + s]);
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n    BBBB 2-RDM Error :   %2.15f", error_2rdm_bb);
    double error_2rdm_ab = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        for (size_t q = 0; q < no_; ++q) {
            for (size_t r = 0; r < no_; ++r) {
                for (size_t s = 0; s < no_; ++s) {
                    double rdm = 0.0;
                    for (size_t i = 0; i < dim_space_; ++i) {
                        Determinant I(det_space[i]);
                        double sign = 1.0;
                        sign *= I.destroy_alfa_bit(r);
                        sign *= I.destroy_beta_bit(s);
                        sign *= I.create_beta_bit(q);
                        sign *= I.create_alfa_bit(p);
                        for (size_t j = 0; j < dim_space_; ++j) {
                            if (I == det_space[j]) {
                                rdm += sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                            }
                        }
                    }
                    if (std::fabs(rdm) > 1.0e-12) {
                        error_2rdm_ab +=
                            std::fabs(rdm - tprdm_ab[p * no3_ + q * no2_ + r * no_ + s]);
                        if (std::fabs(rdm - tprdm_ab[p * no3_ + q * no2_ + r * no_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(abab)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_ab[p * no3_ + q * no2_ + r * no_ + s], rdm,
                                            tprdm_ab[p * no3_ + q * no2_ + r * no_ + s]);
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n    ABAB 2-RDM Error :   %2.15f", error_2rdm_ab);
    // aaa aaa
    // psi::SharedMatrix three_rdm(new psi::Matrix("three", dim_space_, dim_space_));
    // three_rdm->zero();
    double error_3rdm_aaa = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < no_; ++q) {
            for (size_t r = 0; r < no_; ++r) {
                for (size_t s = 0; s < no_; ++s) {
                    for (size_t t = 0; t < no_; ++t) {
                        for (size_t a = 0; a < no_; ++a) {
                            double rdm = 0.0;
                            for (size_t i = 0; i < dim_space_; ++i) {
                                Determinant I(det_space[i]);
                                double sign = 1.0;
                                sign *= I.destroy_alfa_bit(s);
                                sign *= I.destroy_alfa_bit(t);
                                sign *= I.destroy_alfa_bit(a);
                                sign *= I.create_alfa_bit(r);
                                sign *= I.create_alfa_bit(q);
                                sign *= I.create_alfa_bit(p);
                                for (size_t j = 0; j < dim_space_; ++j) {
                                    if (I == det_space[j]) {
                                        rdm +=
                                            sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                                        // three_rdm->set(i,j,three_rdm->get(i,j)
                                        // + 1);
                                    }
                                }
                            }
                            if (std::fabs(rdm) > 1.0e-12) {
                                double rdm_comp = tprdm_aaa[p * no4_ * no_ + q * no4_ + r * no3_ +
                                                            s * no2_ + t * no_ + a];
                                if (rdm - rdm_comp > 1.0e-12) {
                                    outfile->Printf("\nD3(aaaaaa)[%3lu][%3lu][%3lu][%3lu][%3lu][%"
                                                    "3lu] = %18.12lf    (%18.12lf,%18.12lf)",
                                                    p, q, r, s, t, a, rdm - rdm_comp, rdm,
                                                    rdm_comp);
                                    error_3rdm_aaa += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    psi::Process::environment.globals["AAAAAA 3-RDM ERROR"] = error_3rdm_aaa;
    outfile->Printf("\n    AAAAAA 3-RDM Error : %2.15f", error_3rdm_aaa);
    // aab aab
    double error_3rdm_aab = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < no_; ++q) {
            for (size_t r = 0; r < no_; ++r) {
                for (size_t s = 0; s < no_; ++s) {
                    for (size_t t = 0; t < no_; ++t) {
                        for (size_t a = 0; a < no_; ++a) {
                            double rdm = 0.0;
                            for (size_t i = 0; i < dim_space_; ++i) {
                                Determinant I(det_space[i]);
                                double sign = 1.0;
                                sign *= I.destroy_alfa_bit(s);
                                sign *= I.destroy_alfa_bit(t);
                                sign *= I.destroy_beta_bit(a);
                                sign *= I.create_beta_bit(r);
                                sign *= I.create_alfa_bit(q);
                                sign *= I.create_alfa_bit(p);
                                for (size_t j = 0; j < dim_space_; ++j) {
                                    if (I == det_space[j]) {
                                        rdm +=
                                            sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                                        // three_rdm->set(i,j,three_rdm->get(i,j)
                                        // + 1);
                                    }
                                }
                            }
                            if (std::fabs(rdm) > 1.0e-12) {
                                double rdm_comp = tprdm_aab[p * no4_ * no_ + q * no4_ + r * no3_ +
                                                            s * no2_ + t * no_ + a];
                                if (rdm - rdm_comp > 1.0e-12) {
                                    outfile->Printf(
                                        "\n D3(aabaab)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = "
                                        "%18.12lf (%18.12lf,%18.12lf)",
                                        p, q, r, s, t, a, rdm - rdm_comp, rdm, rdm_comp);
                                }
                                error_3rdm_aab += std::fabs(rdm - rdm_comp);
                            }
                        }
                    }
                }
            }
        }
    }
    psi::Process::environment.globals["AABAAB 3-RDM ERROR"] = error_3rdm_aab;
    outfile->Printf("\n    AABAAB 3-RDM Error : %+e", error_3rdm_aab);

    // abb abb
    double error_3rdm_abb = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < no_; ++q) {
            for (size_t r = 0; r < no_; ++r) {
                for (size_t s = 0; s < no_; ++s) {
                    for (size_t t = 0; t < no_; ++t) {
                        for (size_t a = 0; a < no_; ++a) {
                            double rdm = 0.0;
                            for (size_t i = 0; i < dim_space_; ++i) {
                                Determinant I(det_space[i]);
                                double sign = 1.0;
                                sign *= I.destroy_alfa_bit(s);
                                sign *= I.destroy_beta_bit(t);
                                sign *= I.destroy_beta_bit(a);
                                sign *= I.create_beta_bit(r);
                                sign *= I.create_beta_bit(q);
                                sign *= I.create_alfa_bit(p);
                                for (size_t j = 0; j < dim_space_; ++j) {
                                    if (I == det_space[j]) {
                                        rdm +=
                                            sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                                        // three_rdm->set(i,j,three_rdm->get(i,j)
                                        // + 1);
                                    }
                                }
                            }
                            if (std::fabs(rdm) > 1.0e-12) {
                                double rdm_comp = tprdm_abb[p * no4_ * no_ + q * no4_ + r * no3_ +
                                                            s * no2_ + t * no_ + a];
                                if (rdm - rdm_comp > 1.0e-12) {
                                    outfile->Printf("\nD3(abbabb)[%3lu][%3lu][%3lu][%3lu][%3lu][%"
                                                    "3lu] = %18.12lf (%18.12lf,%18.12lf)",
                                                    p, q, r, s, t, a, rdm - rdm_comp, rdm,
                                                    rdm_comp);
                                }
                                error_3rdm_abb += std::fabs(rdm - rdm_comp);
                            }
                        }
                    }
                }
            }
        }
    }
    psi::Process::environment.globals["ABBABB 3-RDM ERROR"] = error_3rdm_abb;
    outfile->Printf("\n    ABBABB 3-RDM Error : %+e", error_3rdm_abb);

    // bbb bbb
    double error_3rdm_bbb = 0.0;
    for (size_t p = 0; p < no_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < no_; ++q) {
            for (size_t r = 0; r < no_; ++r) {
                for (size_t s = 0; s < no_; ++s) {
                    for (size_t t = 0; t < no_; ++t) {
                        for (size_t a = 0; a < no_; ++a) {
                            double rdm = 0.0;
                            for (size_t i = 0; i < dim_space_; ++i) {
                                Determinant I(det_space[i]);
                                double sign = 1.0;
                                sign *= I.destroy_beta_bit(s);
                                sign *= I.destroy_beta_bit(t);
                                sign *= I.destroy_beta_bit(a);
                                sign *= I.create_beta_bit(r);
                                sign *= I.create_beta_bit(q);
                                sign *= I.create_beta_bit(p);
                                for (size_t j = 0; j < dim_space_; ++j) {
                                    if (I == det_space[j]) {
                                        rdm +=
                                            sign * evecs_->get(i, root1_) * evecs_->get(j, root2_);
                                        // three_rdm->set(i,j,three_rdm->get(i,j)
                                        // + 1);
                                    }
                                }
                            }
                            if (std::fabs(rdm) > 1.0e-12) {
                                double rdm_comp = tprdm_bbb[p * no4_ * no_ + q * no4_ + r * no3_ +
                                                            s * no2_ + t * no_ + a];
                                if (rdm - rdm_comp > 1.0e-12) {
                                    outfile->Printf(
                                        "\n D3(bbbbbb)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = "
                                        "%18.12lf (%18.12lf,%18.12lf)",
                                        p, q, r, s, t, a, rdm - rdm_comp, rdm, rdm_comp);
                                    error_3rdm_bbb += std::fabs(rdm - rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    psi::Process::environment.globals["BBBBBB 3-RDM ERROR"] = error_3rdm_bbb;
    outfile->Printf("\n    BBBBBB 3-RDM Error : %2.15f", error_3rdm_bbb);
}
} // namespace forte
