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
    norb_ = fci_ints_->nmo();
    norb2_ = norb_ * norb_;
    norb3_ = norb2_ * norb_;
    norb4_ = norb3_ * norb_;
    norb5_ = norb4_ * norb_;
    norb6_ = norb5_ * norb_;

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
    norb_ = fci_ints_->nmo();
    norb2_ = norb_ * norb_;
    norb3_ = norb2_ * norb_;
    norb4_ = norb3_ * norb_;
    norb5_ = norb4_ * norb_;
    norb6_ = norb5_ * norb_;

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
        outfile->Printf("\n  Number of correlated orbitals: %zu", norb_);
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

    for (size_t p = 0; p < norb_; ++p) {
        for (size_t q = 0; q < norb_; ++q) {
            energy_1rdm += oprdm_a[norb_ * p + q] * fci_ints_->oei_a(p, q);
            energy_1rdm += oprdm_b[norb_ * p + q] * fci_ints_->oei_b(p, q);
        }
    }

    for (size_t p = 0; p < norb_; ++p) {
        for (size_t q = 0; q < norb_; ++q) {
            for (size_t r = 0; r < norb_; ++r) {
                for (size_t s = 0; s < norb_; ++s) {
                    if (na_ >= 2)
                        energy_2rdm += 0.25 * tprdm_aa[p * norb3_ + q * norb2_ + r * norb_ + s] *
                                       fci_ints_->tei_aa(p, q, r, s);
                    if ((na_ >= 1) and (nb_ >= 1))
                        energy_2rdm += tprdm_ab[p * norb3_ + q * norb2_ + r * norb_ + s] *
                                       fci_ints_->tei_ab(p, q, r, s);
                    if (nb_ >= 2)
                        energy_2rdm += 0.25 * tprdm_bb[p * norb3_ + q * norb2_ + r * norb_ + s] *
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
    oprdm_a.assign(norb2_, 0.0);
    oprdm_b.assign(norb2_, 0.0);

    for (size_t J = 0; J < dim_space_; ++J) {
        auto vJ = evecs_->get(J, root2_);
        for (const auto& aJ_mo_sign : a_ann_list_[J]) {
            const auto [aJ_add, _p] = aJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_p = _p < 0;
            for (const auto& aaJ_mo_sign : a_cre_list_[aJ_add]) {
                const auto [I, _q] = aaJ_mo_sign;
                auto q = std::abs(_q) - 1;
                auto sign = ((_q < 0) == sign_p) ? 1 : -1;
                oprdm_a[q * norb_ + p] += evecs_->get(I, root1_) * vJ * sign;
            }
        }
        for (const auto& bJ_mo_sign : b_ann_list_[J]) {
            const auto [bJ_add, _p] = bJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_p = _p < 0;
            for (const auto& bbJ_mo_sign : b_cre_list_[bJ_add]) {
                const auto [I, _q] = bbJ_mo_sign;
                auto q = std::abs(_q) - 1;
                auto sign = ((_q < 0) == sign_p) ? 1 : -1;
                oprdm_b[q * norb_ + p] += evecs_->get(I, root1_) * vJ * sign;
            }
        }
    }

    if (print_)
        outfile->Printf("\n  Time spent building 1-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_1rdm_op(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->op_s_lists(wfn_);
    compute_1rdm_op(oprdm_a, oprdm_b, op);
}

void CI_RDMS::compute_1rdm_op(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b,
                              std::shared_ptr<DeterminantSubstitutionLists> op) {
    local_timer build;
    oprdm_a.assign(norb2_, 0.0);
    oprdm_b.assign(norb2_, 0.0);

    // Build the diagonal part
    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t J = 0; J < dim_space_; ++J) {
        double cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
        for (int pp : dets[J].get_alfa_occ(norb_)) {
            oprdm_a[pp * norb_ + pp] += cJ_sq;
        }
        for (int pp : dets[J].get_beta_occ(norb_)) {
            oprdm_b[pp * norb_ + pp] += cJ_sq;
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

                oprdm_a[p * norb_ + q] += vI1 * evecs_->get(J, root2_) * sign;
                oprdm_a[q * norb_ + p] += evecs_->get(J, root1_) * vI2 * sign;
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

                oprdm_b[p * norb_ + q] += vI1 * evecs_->get(J, root2_) * sign;
                oprdm_b[q * norb_ + p] += evecs_->get(J, root1_) * vI2 * sign;
            }
        }
    }

    if (print_) {
        outfile->Printf("\n  Time spent building 1-rdm: %.3e seconds", build.get());
    }
}

void CI_RDMS::compute_2rdm(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                           std::vector<double>& tprdm_bb) {
    tprdm_aa.assign(norb4_, 0.0);
    tprdm_ab.assign(norb4_, 0.0);
    tprdm_bb.assign(norb4_, 0.0);

    local_timer two;
    get_two_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 2-map:   %1.6f", two.get());

    local_timer build;

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
                double rdm_element = evecs_->get(I, root2_) * vJ * sign;

                tprdm_aa[p * norb3_ + q * norb2_ + r * norb_ + s] += rdm_element;
                tprdm_aa[p * norb3_ + q * norb2_ + s * norb_ + r] -= rdm_element;
                tprdm_aa[q * norb3_ + p * norb2_ + r * norb_ + s] -= rdm_element;
                tprdm_aa[q * norb3_ + p * norb2_ + s * norb_ + r] += rdm_element;
            }
        }

        // bbbb
        for (const auto& bbJ_mo_sign : bb_ann_list_[J]) {
            const auto [bbJ_add, _p, q] = bbJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            for (const auto& bbbbJ_mo_sign : bb_cre_list_[bbJ_add]) {
                const auto [I, _r, s] = bbbbJ_mo_sign;
                auto r = std::abs(_r) - 1;

                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;
                double rdm_element = evecs_->get(I, root2_) * vJ * sign;

                tprdm_bb[p * norb3_ + q * norb2_ + r * norb_ + s] += rdm_element;
                tprdm_bb[p * norb3_ + q * norb2_ + s * norb_ + r] -= rdm_element;
                tprdm_bb[q * norb3_ + p * norb2_ + r * norb_ + s] -= rdm_element;
                tprdm_bb[q * norb3_ + p * norb2_ + s * norb_ + r] += rdm_element;
            }
        }

        // aabb
        for (const auto& abJ_mo_sign : ab_ann_list_[J]) {
            const auto [abJ_add, _p, q] = abJ_mo_sign;
            auto p = std::abs(_p) - 1;
            auto sign_pq = _p < 0;

            for (const auto& aabbJ_mo_sign : ab_cre_list_[abJ_add]) {
                const auto [I, _r, s] = aabbJ_mo_sign;
                auto r = std::abs(_r) - 1;

                auto sign = ((_r < 0) == sign_pq) ? 1 : -1;
                double rdm_element = evecs_->get(I, root2_) * vJ * sign;

                tprdm_ab[p * norb3_ + q * norb2_ + r * norb_ + s] += rdm_element;
            }
        }
    }

    if (print_)
        outfile->Printf("\n  Time spent building 2-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_2rdm_op(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                              std::vector<double>& tprdm_bb) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->tp_s_lists(wfn_);
    compute_2rdm_op(tprdm_aa, tprdm_ab, tprdm_bb, op);
}

void CI_RDMS::compute_2rdm_op(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                              std::vector<double>& tprdm_bb,
                              std::shared_ptr<DeterminantSubstitutionLists> op) {
    local_timer build;

    tprdm_aa.assign(norb4_, 0.0);
    tprdm_ab.assign(norb4_, 0.0);
    tprdm_bb.assign(norb4_, 0.0);

    // Build the diagonal part
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

                tprdm_aa[pp * norb3_ + qq * norb2_ + pp * norb_ + qq] += cJ_sq;
                tprdm_aa[qq * norb3_ + pp * norb2_ + pp * norb_ + qq] -= cJ_sq;

                tprdm_aa[qq * norb3_ + pp * norb2_ + qq * norb_ + pp] += cJ_sq;
                tprdm_aa[pp * norb3_ + qq * norb2_ + qq * norb_ + pp] -= cJ_sq;
            }
        }

        for (size_t p = 0; p < nbocc; ++p) {
            auto pp = bocc[p];
            for (size_t q = p + 1; q < nbocc; ++q) {
                auto qq = bocc[q];

                tprdm_bb[pp * norb3_ + qq * norb2_ + pp * norb_ + qq] += cJ_sq;
                tprdm_bb[qq * norb3_ + pp * norb2_ + pp * norb_ + qq] -= cJ_sq;

                tprdm_bb[qq * norb3_ + pp * norb2_ + qq * norb_ + pp] += cJ_sq;
                tprdm_bb[pp * norb3_ + qq * norb2_ + qq * norb_ + pp] -= cJ_sq;
            }
        }

        for (size_t p = 0; p < naocc; ++p) {
            auto pp = aocc[p];
            for (size_t q = 0; q < nbocc; ++q) {
                auto qq = bocc[q];

                tprdm_ab[pp * norb3_ + qq * norb2_ + pp * norb_ + qq] += cJ_sq;
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
                tprdm_aa[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                tprdm_aa[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                tprdm_aa[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                tprdm_aa[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tprdm_aa[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                tprdm_aa[s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tprdm_aa[r * norb3_ + s * norb2_ + q * norb_ + p] -= value;
                tprdm_aa[s * norb3_ + r * norb2_ + q * norb_ + p] += value;
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
                tprdm_bb[p * norb3_ + q * norb2_ + r * norb_ + s] += value;
                tprdm_bb[p * norb3_ + q * norb2_ + s * norb_ + r] -= value;
                tprdm_bb[q * norb3_ + p * norb2_ + r * norb_ + s] -= value;
                tprdm_bb[q * norb3_ + p * norb2_ + s * norb_ + r] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tprdm_bb[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
                tprdm_bb[s * norb3_ + r * norb2_ + p * norb_ + q] -= value;
                tprdm_bb[r * norb3_ + s * norb2_ + q * norb_ + p] -= value;
                tprdm_bb[s * norb3_ + r * norb2_ + q * norb_ + p] += value;
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
                tprdm_ab[p * norb3_ + q * norb2_ + r * norb_ + s] += value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tprdm_ab[r * norb3_ + s * norb2_ + p * norb_ + q] += value;
            }
        }
    }

    if (print_) {
        outfile->Printf("\n  Time spent building 2-rdm: %.3e seconds", build.get());
    }
}

void CI_RDMS::compute_3rdm(std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab,
                           std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb) {
    tprdm_aaa.assign(norb6_, 0.0);
    tprdm_aab.assign(norb6_, 0.0);
    tprdm_abb.assign(norb6_, 0.0);
    tprdm_bbb.assign(norb6_, 0.0);

    local_timer three;
    get_three_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 3-map:   %1.6f", three.get());

    local_timer build;

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

                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;
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

                tprdm_aab[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_aab[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_aab[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_aab[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
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
                auto value = evecs_->get(I, root2_) * vJ * sign;

                tprdm_abb[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_abb[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_abb[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_abb[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
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

                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;
            }
        }
    }

    if (print_)
        outfile->Printf("\n  Time spent building 3-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_3rdm_op(std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab,
                              std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb) {
    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->set_quiet_mode(not print_);
    op->build_strings(wfn_);
    op->three_s_lists(wfn_);

    local_timer build;

    tprdm_aaa.assign(norb6_, 0.0);
    tprdm_aab.assign(norb6_, 0.0);
    tprdm_abb.assign(norb6_, 0.0);
    tprdm_bbb.assign(norb6_, 0.0);

    // Build the diagonal part
    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t I = 0; I < dim_space_; ++I) {
        double cI_sq = evecs_->get(I, root1_) * evecs_->get(I, root2_);

        auto aocc = dets[I].get_alfa_occ(norb_);
        auto bocc = dets[I].get_beta_occ(norb_);
        auto na = aocc.size();
        auto nb = bocc.size();

        for (size_t _p = 0; _p < na; ++_p) {
            auto p = aocc[_p];
            for (size_t _q = _p + 1; _q < na; ++_q) {
                auto q = aocc[_q];
                for (size_t _r = _q + 1; _r < na; ++_r) {
                    auto r = aocc[_r];

                    tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] +=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + r * norb_ + q] -=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + r * norb2_ + p * norb_ + q] +=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + r * norb2_ + q * norb_ + p] -=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + q * norb2_ + r * norb_ + p] +=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] -=
                        cI_sq;

                    tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + p * norb2_ + q * norb_ + r] -=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + p * norb2_ + r * norb_ + q] +=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + r * norb2_ + p * norb_ + q] -=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + r * norb2_ + q * norb_ + p] +=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + q * norb2_ + r * norb_ + p] -=
                        cI_sq;
                    tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + q * norb2_ + p * norb_ + r] +=
                        cI_sq;

                    tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + p * norb2_ + q * norb_ + r] +=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + p * norb2_ + r * norb_ + q] -=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + p * norb_ + q] +=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + q * norb_ + p] -=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + q * norb2_ + r * norb_ + p] +=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + q * norb2_ + p * norb_ + r] -=
                        cI_sq;

                    tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + p * norb2_ + q * norb_ + r] -=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + p * norb2_ + r * norb_ + q] +=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + p * norb_ + q] -=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + q * norb_ + p] +=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + q * norb2_ + r * norb_ + p] -=
                        cI_sq;
                    tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + q * norb2_ + p * norb_ + r] +=
                        cI_sq;

                    tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + p * norb2_ + q * norb_ + r] +=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + p * norb2_ + r * norb_ + q] -=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + r * norb2_ + p * norb_ + q] +=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + r * norb2_ + q * norb_ + p] -=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + q * norb2_ + r * norb_ + p] +=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + q * norb2_ + p * norb_ + r] -=
                        cI_sq;

                    tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] -=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + p * norb2_ + r * norb_ + q] +=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + r * norb2_ + p * norb_ + q] -=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + r * norb2_ + q * norb_ + p] +=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + q * norb2_ + r * norb_ + p] -=
                        cI_sq;
                    tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] +=
                        cI_sq;
                }
            }
        }

        for (size_t _p = 0; _p < na; ++_p) {
            auto p = aocc[_p];
            for (size_t _q = _p + 1; _q < na; ++_q) {
                auto q = aocc[_q];
                for (size_t _r = 0; _r < nb; ++_r) {
                    auto r = bocc[_r];

                    tprdm_aab[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] +=
                        cI_sq;
                    tprdm_aab[p * norb5_ + q * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] -=
                        cI_sq;
                    tprdm_aab[q * norb5_ + p * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] -=
                        cI_sq;
                    tprdm_aab[q * norb5_ + p * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] +=
                        cI_sq;
                }
            }
        }

        for (size_t _p = 0; _p < na; ++_p) {
            auto p = aocc[_p];
            for (size_t _q = 0; _q < nb; ++_q) {
                auto q = bocc[_q];
                for (size_t _r = _q + 1; _r < nb; ++_r) {
                    auto r = bocc[_r];

                    tprdm_abb[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] +=
                        cI_sq;
                    tprdm_abb[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + r * norb_ + q] -=
                        cI_sq;
                    tprdm_abb[p * norb5_ + r * norb4_ + q * norb3_ + p * norb2_ + q * norb_ + r] -=
                        cI_sq;
                    tprdm_abb[p * norb5_ + r * norb4_ + q * norb3_ + p * norb2_ + r * norb_ + q] +=
                        cI_sq;
                }
            }
        }

        for (size_t _p = 0; _p < nb; ++_p) {
            auto p = bocc[_p];
            for (size_t _q = _p + 1; _q < nb; ++_q) {
                auto q = bocc[_q];
                for (size_t _r = _q + 1; _r < nb; ++_r) {
                    auto r = bocc[_r];

                    tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] +=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + p * norb2_ + r * norb_ + q] -=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + r * norb2_ + p * norb_ + q] +=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + r * norb2_ + q * norb_ + p] -=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + q * norb2_ + r * norb_ + p] +=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] -=
                        cI_sq;

                    tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + p * norb2_ + q * norb_ + r] -=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + p * norb2_ + r * norb_ + q] +=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + r * norb2_ + p * norb_ + q] -=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + r * norb2_ + q * norb_ + p] +=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + q * norb2_ + r * norb_ + p] -=
                        cI_sq;
                    tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + q * norb2_ + p * norb_ + r] +=
                        cI_sq;

                    tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + p * norb2_ + q * norb_ + r] +=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + p * norb2_ + r * norb_ + q] -=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + p * norb_ + q] +=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + r * norb2_ + q * norb_ + p] -=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + q * norb2_ + r * norb_ + p] +=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + q * norb2_ + p * norb_ + r] -=
                        cI_sq;

                    tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + p * norb2_ + q * norb_ + r] -=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + p * norb2_ + r * norb_ + q] +=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + p * norb_ + q] -=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + r * norb2_ + q * norb_ + p] +=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + q * norb2_ + r * norb_ + p] -=
                        cI_sq;
                    tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + q * norb2_ + p * norb_ + r] +=
                        cI_sq;

                    tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + p * norb2_ + q * norb_ + r] +=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + p * norb2_ + r * norb_ + q] -=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + r * norb2_ + p * norb_ + q] +=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + r * norb2_ + q * norb_ + p] -=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + q * norb2_ + r * norb_ + p] +=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + q * norb2_ + p * norb_ + r] -=
                        cI_sq;

                    tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + p * norb2_ + q * norb_ + r] -=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + p * norb2_ + r * norb_ + q] +=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + r * norb2_ + p * norb_ + q] -=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + r * norb2_ + q * norb_ + p] +=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + q * norb2_ + r * norb_ + p] -=
                        cI_sq;
                    tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + q * norb2_ + p * norb_ + r] +=
                        cI_sq;
                }
            }
        }
    }

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

                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_aaa[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_aaa[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_aaa[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_aaa[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_aaa[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_aaa[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                value = evecs_->get(I, root1_) * vJ2 * sign;

                tprdm_aaa[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] +=
                    value;
                tprdm_aaa[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] -=
                    value;
                tprdm_aaa[u * norb5_ + t * norb4_ + s * norb3_ + p * norb2_ + q * norb_ + r] -=
                    value;
                tprdm_aaa[u * norb5_ + s * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] +=
                    value;
                tprdm_aaa[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] -=
                    value;
                tprdm_aaa[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + q * norb_ + r] +=
                    value;

                tprdm_aaa[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] -=
                    value;
                tprdm_aaa[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] +=
                    value;
                tprdm_aaa[u * norb5_ + t * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] +=
                    value;
                tprdm_aaa[u * norb5_ + s * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] -=
                    value;
                tprdm_aaa[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] +=
                    value;
                tprdm_aaa[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] -=
                    value;

                tprdm_aaa[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] -=
                    value;
                tprdm_aaa[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] +=
                    value;
                tprdm_aaa[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + p * norb_ + r] +=
                    value;
                tprdm_aaa[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] -=
                    value;
                tprdm_aaa[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] +=
                    value;
                tprdm_aaa[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + p * norb_ + r] -=
                    value;

                tprdm_aaa[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + r * norb_ + p] +=
                    value;
                tprdm_aaa[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] -=
                    value;
                tprdm_aaa[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] -=
                    value;
                tprdm_aaa[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] +=
                    value;
                tprdm_aaa[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + r * norb_ + p] -=
                    value;
                tprdm_aaa[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] +=
                    value;

                tprdm_aaa[s * norb5_ + t * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] +=
                    value;
                tprdm_aaa[s * norb5_ + u * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] -=
                    value;
                tprdm_aaa[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] -=
                    value;
                tprdm_aaa[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] +=
                    value;
                tprdm_aaa[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] -=
                    value;
                tprdm_aaa[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] +=
                    value;

                tprdm_aaa[s * norb5_ + t * norb4_ + u * norb3_ + r * norb2_ + q * norb_ + p] -=
                    value;
                tprdm_aaa[s * norb5_ + u * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] +=
                    value;
                tprdm_aaa[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] +=
                    value;
                tprdm_aaa[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] -=
                    value;
                tprdm_aaa[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + q * norb_ + p] +=
                    value;
                tprdm_aaa[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] -=
                    value;
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
                tprdm_aab[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_aab[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_aab[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_aab[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tprdm_aab[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] +=
                    value;
                tprdm_aab[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] -=
                    value;
                tprdm_aab[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] -=
                    value;
                tprdm_aab[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] +=
                    value;
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
                tprdm_abb[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_abb[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_abb[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_abb[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;

                value = evecs_->get(I, root1_) * vJ2 * sign;
                tprdm_abb[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] +=
                    value;
                tprdm_abb[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] -=
                    value;
                tprdm_abb[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] -=
                    value;
                tprdm_abb[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] +=
                    value;
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

                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_bbb[p * norb5_ + q * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_bbb[p * norb5_ + r * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_bbb[q * norb5_ + p * norb4_ + r * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_bbb[q * norb5_ + r * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + t * norb_ + u] +=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + s * norb2_ + u * norb_ + t] -=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + t * norb_ + s] -=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + u * norb2_ + s * norb_ + t] +=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + s * norb_ + u] -=
                    value;
                tprdm_bbb[r * norb5_ + p * norb4_ + q * norb3_ + t * norb2_ + u * norb_ + s] +=
                    value;

                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + t * norb_ + u] -=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + s * norb2_ + u * norb_ + t] +=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + t * norb_ + s] +=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + u * norb2_ + s * norb_ + t] -=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + s * norb_ + u] +=
                    value;
                tprdm_bbb[r * norb5_ + q * norb4_ + p * norb3_ + t * norb2_ + u * norb_ + s] -=
                    value;

                value = evecs_->get(I, root1_) * vJ2 * sign;

                tprdm_bbb[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] +=
                    value;
                tprdm_bbb[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] -=
                    value;
                tprdm_bbb[u * norb5_ + t * norb4_ + s * norb3_ + p * norb2_ + q * norb_ + r] -=
                    value;
                tprdm_bbb[u * norb5_ + s * norb4_ + t * norb3_ + p * norb2_ + q * norb_ + r] +=
                    value;
                tprdm_bbb[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + q * norb_ + r] -=
                    value;
                tprdm_bbb[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + q * norb_ + r] +=
                    value;

                tprdm_bbb[s * norb5_ + t * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] -=
                    value;
                tprdm_bbb[s * norb5_ + u * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] +=
                    value;
                tprdm_bbb[u * norb5_ + t * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] +=
                    value;
                tprdm_bbb[u * norb5_ + s * norb4_ + t * norb3_ + p * norb2_ + r * norb_ + q] -=
                    value;
                tprdm_bbb[t * norb5_ + s * norb4_ + u * norb3_ + p * norb2_ + r * norb_ + q] +=
                    value;
                tprdm_bbb[t * norb5_ + u * norb4_ + s * norb3_ + p * norb2_ + r * norb_ + q] -=
                    value;

                tprdm_bbb[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] -=
                    value;
                tprdm_bbb[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] +=
                    value;
                tprdm_bbb[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + p * norb_ + r] +=
                    value;
                tprdm_bbb[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + p * norb_ + r] -=
                    value;
                tprdm_bbb[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + p * norb_ + r] +=
                    value;
                tprdm_bbb[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + p * norb_ + r] -=
                    value;

                tprdm_bbb[s * norb5_ + t * norb4_ + u * norb3_ + q * norb2_ + r * norb_ + p] +=
                    value;
                tprdm_bbb[s * norb5_ + u * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] -=
                    value;
                tprdm_bbb[u * norb5_ + t * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] -=
                    value;
                tprdm_bbb[u * norb5_ + s * norb4_ + t * norb3_ + q * norb2_ + r * norb_ + p] +=
                    value;
                tprdm_bbb[t * norb5_ + s * norb4_ + u * norb3_ + q * norb2_ + r * norb_ + p] -=
                    value;
                tprdm_bbb[t * norb5_ + u * norb4_ + s * norb3_ + q * norb2_ + r * norb_ + p] +=
                    value;

                tprdm_bbb[s * norb5_ + t * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] +=
                    value;
                tprdm_bbb[s * norb5_ + u * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] -=
                    value;
                tprdm_bbb[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] -=
                    value;
                tprdm_bbb[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + p * norb_ + q] +=
                    value;
                tprdm_bbb[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + p * norb_ + q] -=
                    value;
                tprdm_bbb[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + p * norb_ + q] +=
                    value;

                tprdm_bbb[s * norb5_ + t * norb4_ + u * norb3_ + r * norb2_ + q * norb_ + p] -=
                    value;
                tprdm_bbb[s * norb5_ + u * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] +=
                    value;
                tprdm_bbb[u * norb5_ + t * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] +=
                    value;
                tprdm_bbb[u * norb5_ + s * norb4_ + t * norb3_ + r * norb2_ + q * norb_ + p] -=
                    value;
                tprdm_bbb[t * norb5_ + s * norb4_ + u * norb3_ + r * norb2_ + q * norb_ + p] +=
                    value;
                tprdm_bbb[t * norb5_ + u * norb4_ + s * norb3_ + r * norb2_ + q * norb_ + p] -=
                    value;
            }
        }
    }

    if (print_) {
        outfile->Printf("\n  Time spent building 3-rdm: %.3e seconds", build.get());
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
        std::vector<int> aocc = detI.get_alfa_occ(norb_);
        std::vector<int> bocc = detI.get_beta_occ(norb_);

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

        std::vector<int> aocc = detI.get_alfa_occ(norb_);
        std::vector<int> bocc = detI.get_beta_occ(norb_);

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

        const std::vector<int>& aocc = detI.get_alfa_occ(norb_);
        const std::vector<int>& bocc = detI.get_beta_occ(norb_);

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
    for (size_t p = 0; p < norb_; ++p) {
        for (size_t q = 0; q < norb_; ++q) {
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
                error_1rdm_a += std::fabs(rdm - oprdm_a[q * norb_ + p]);
                //     outfile->Printf("\n  D1(a)[%3lu][%3lu] = %18.12lf
                //     (%18.12lf,%18.12lf)", p,q,
                //     rdm-oprdm_a[p*norb_+q],rdm,oprdm_a[p*norb_+q]);
            }
        }
    }
    outfile->Printf("\n    A 1-RDM Error :   %2.15f", error_1rdm_a);
    double error_1rdm_b = 0.0;
    for (size_t p = 0; p < norb_; ++p) {
        for (size_t q = 0; q < norb_; ++q) {
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
                error_1rdm_b += std::fabs(rdm - oprdm_b[p * norb_ + q]);
                // outfile->Printf("\n  D1(b)[%3lu][%3lu] = %18.12lf
                // (%18.12lf,%18.12lf)", p,q,
                // rdm-oprdm_b[p*norb_+q],rdm,oprdm_b[p*norb_+q]);
            }
        }
    }
    outfile->Printf("\n    B 1-RDM Error :   %2.15f", error_1rdm_b);

    double error_2rdm_aa = 0.0;
    for (size_t p = 0; p < norb_; ++p) {
        for (size_t q = 0; q < norb_; ++q) {
            for (size_t r = 0; r < norb_; ++r) {
                for (size_t s = 0; s < norb_; ++s) {
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
                            std::fabs(rdm - tprdm_aa[p * norb3_ + q * norb2_ + r * norb_ + s]);
                        if (std::fabs(rdm - tprdm_aa[p * norb3_ + q * norb2_ + r * norb_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(aaaa)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_aa[p * norb3_ + q * norb2_ + r * norb_ + s],
                                            rdm, tprdm_aa[p * norb3_ + q * norb2_ + r * norb_ + s]);
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n    AAAA 2-RDM Error :   %2.15f", error_2rdm_aa);
    double error_2rdm_bb = 0.0;
    for (size_t p = 0; p < norb_; ++p) {
        for (size_t q = 0; q < norb_; ++q) {
            for (size_t r = 0; r < norb_; ++r) {
                for (size_t s = 0; s < norb_; ++s) {
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
                            std::fabs(rdm - tprdm_bb[p * norb3_ + q * norb2_ + r * norb_ + s]);
                        if (std::fabs(rdm - tprdm_bb[p * norb3_ + q * norb2_ + r * norb_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(bbbb)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_bb[p * norb3_ + q * norb2_ + r * norb_ + s],
                                            rdm, tprdm_bb[p * norb3_ + q * norb2_ + r * norb_ + s]);
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n    BBBB 2-RDM Error :   %2.15f", error_2rdm_bb);
    double error_2rdm_ab = 0.0;
    for (size_t p = 0; p < norb_; ++p) {
        for (size_t q = 0; q < norb_; ++q) {
            for (size_t r = 0; r < norb_; ++r) {
                for (size_t s = 0; s < norb_; ++s) {
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
                            std::fabs(rdm - tprdm_ab[p * norb3_ + q * norb2_ + r * norb_ + s]);
                        if (std::fabs(rdm - tprdm_ab[p * norb3_ + q * norb2_ + r * norb_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(abab)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_ab[p * norb3_ + q * norb2_ + r * norb_ + s],
                                            rdm, tprdm_ab[p * norb3_ + q * norb2_ + r * norb_ + s]);
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
    for (size_t p = 0; p < norb_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < norb_; ++q) {
            for (size_t r = 0; r < norb_; ++r) {
                for (size_t s = 0; s < norb_; ++s) {
                    for (size_t t = 0; t < norb_; ++t) {
                        for (size_t a = 0; a < norb_; ++a) {
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
                                        // three_rdm->set(i, j, three_rdm->get(i,j) + 1);
                                    }
                                }
                            }
                            if (std::fabs(rdm) > 1.0e-12) {
                                double rdm_comp =
                                    tprdm_aaa[p * norb4_ * norb_ + q * norb4_ + r * norb3_ +
                                              s * norb2_ + t * norb_ + a];
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
    for (size_t p = 0; p < norb_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < norb_; ++q) {
            for (size_t r = 0; r < norb_; ++r) {
                for (size_t s = 0; s < norb_; ++s) {
                    for (size_t t = 0; t < norb_; ++t) {
                        for (size_t a = 0; a < norb_; ++a) {
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
                                        // three_rdm->set(i, j, three_rdm->get(i,j) + 1);
                                    }
                                }
                            }
                            if (std::fabs(rdm) > 1.0e-12) {
                                double rdm_comp =
                                    tprdm_aab[p * norb4_ * norb_ + q * norb4_ + r * norb3_ +
                                              s * norb2_ + t * norb_ + a];
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
    for (size_t p = 0; p < norb_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < norb_; ++q) {
            for (size_t r = 0; r < norb_; ++r) {
                for (size_t s = 0; s < norb_; ++s) {
                    for (size_t t = 0; t < norb_; ++t) {
                        for (size_t a = 0; a < norb_; ++a) {
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
                                        // three_rdm->set(i, j, three_rdm->get(i,j) + 1);
                                    }
                                }
                            }
                            if (std::fabs(rdm) > 1.0e-12) {
                                double rdm_comp =
                                    tprdm_abb[p * norb4_ * norb_ + q * norb4_ + r * norb3_ +
                                              s * norb2_ + t * norb_ + a];
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
    for (size_t p = 0; p < norb_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < norb_; ++q) {
            for (size_t r = 0; r < norb_; ++r) {
                for (size_t s = 0; s < norb_; ++s) {
                    for (size_t t = 0; t < norb_; ++t) {
                        for (size_t a = 0; a < norb_; ++a) {
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
                                        // three_rdm->set(i, j, three_rdm->get(i,j) + 1);
                                    }
                                }
                            }
                            if (std::fabs(rdm) > 1.0e-12) {
                                double rdm_comp =
                                    tprdm_bbb[p * norb4_ * norb_ + q * norb4_ + r * norb3_ +
                                              s * norb2_ + t * norb_ + a];
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
