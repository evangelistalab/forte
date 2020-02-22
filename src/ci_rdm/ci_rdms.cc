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

#include <cmath>

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

    Determinant det(wfn_.get_det(0));
    ncmo_ = fci_ints_->nmo();
    ncmo2_ = ncmo_ * ncmo_;
    ncmo3_ = ncmo2_ * ncmo_;
    ncmo4_ = ncmo3_ * ncmo_;
    ncmo5_ = ncmo3_ * ncmo2_;
    print_ = false;
    dim_space_ = wfn.size();

    na_ = det.count_alfa();
    nb_ = det.count_beta();
}

CI_RDMS::~CI_RDMS() {}

void CI_RDMS::startup() {
    /* Get all of the required info from MOSpaceInfo to initialize the
     * StringList*/

    // The number of correlated molecular orbitals
    ncmo_ = fci_ints_->nmo();
    ncmo2_ = ncmo_ * ncmo_;
    ncmo3_ = ncmo2_ * ncmo_;
    ncmo4_ = ncmo3_ * ncmo_;
    ncmo5_ = ncmo3_ * ncmo2_;

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
        outfile->Printf("\n  Number of correlated orbitals: %zu", ncmo_);
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

    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            energy_1rdm += oprdm_a[ncmo_ * p + q] * fci_ints_->oei_a(p, q);
            energy_1rdm += oprdm_b[ncmo_ * p + q] * fci_ints_->oei_b(p, q);
        }
    }

    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    if (na_ >= 2)
                        energy_2rdm += 0.25 * tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] *
                                       fci_ints_->tei_aa(p, q, r, s);
                    if ((na_ >= 1) and (nb_ >= 1))
                        energy_2rdm += tprdm_ab[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] *
                                       fci_ints_->tei_ab(p, q, r, s);
                    if (nb_ >= 2)
                        energy_2rdm += 0.25 * tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] *
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
    oprdm_a.assign(ncmo2_, 0.0);
    oprdm_b.assign(ncmo2_, 0.0);
    for (size_t J = 0; J < dim_space_; ++J) {
        for (auto& aJ_mo_sign : a_ann_list_[J]) {
            const size_t aJ_add = aJ_mo_sign.first;
            size_t p = std::abs(aJ_mo_sign.second) - 1;
            const double sign_p = aJ_mo_sign.second > 0 ? 1.0 : -1.0;
            for (auto& aaJ_mo_sign : a_cre_list_[aJ_add]) {
                size_t q = std::abs(aaJ_mo_sign.second) - 1;
                const double sign_q = aaJ_mo_sign.second > 0 ? 1.0 : -1.0;
                const size_t I = aaJ_mo_sign.first;
                oprdm_a[q * ncmo_ + p] +=
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_p * sign_q;
            }
        }
        for (auto& bJ_mo_sign : b_ann_list_[J]) {
            const size_t bJ_add = bJ_mo_sign.first;
            const size_t p = std::abs(bJ_mo_sign.second) - 1;
            const double sign_p = bJ_mo_sign.second > 0 ? 1.0 : -1.0;
            for (auto& bbJ_mo_sign : b_cre_list_[bJ_add]) {
                const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                const double sign_q = bbJ_mo_sign.second > 0 ? 1.0 : -1.0;
                const size_t I = bbJ_mo_sign.first;
                oprdm_b[q * ncmo_ + p] +=
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_p * sign_q;
            }
        }
    }
    if (print_)
        outfile->Printf("\n  Time spent building 1-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_1rdm_op(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b) {

    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->build_strings(wfn_);
    op->op_s_lists(wfn_);

    // Get the references to the coupling lists
    std::vector<std::vector<std::pair<size_t, short>>>& a_list = op->a_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_list = op->b_list_;

    local_timer build;
    oprdm_a.assign(ncmo2_, 0.0);
    oprdm_b.assign(ncmo2_, 0.0);

    //// Do something about diagonal
    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t J = 0; J < dim_space_; ++J) {
        double cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
        std::vector<int> aocc = dets[J].get_alfa_occ(ncmo_);
        std::vector<int> bocc = dets[J].get_beta_occ(ncmo_);
        std::vector<int> avir = dets[J].get_alfa_vir(ncmo_);
        std::vector<int> bvir = dets[J].get_beta_vir(ncmo_);

        for (int p = 0, max_p = aocc.size(); p < max_p; ++p) {
            int pp = aocc[p];
            oprdm_a[pp * ncmo_ + pp] += cJ_sq;
        }

        for (int p = 0, max_p = bocc.size(); p < max_p; ++p) {
            int pp = bocc[p];
            oprdm_b[pp * ncmo_ + pp] += cJ_sq;
        }
    }
    for (size_t K = 0, max_K = a_list.size(); K < max_K; ++K) {
        std::vector<std::pair<size_t, short>>& coupled_dets = a_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
            auto& detI = coupled_dets[a];
            const size_t& I = detI.first;
            const size_t& p = std::abs(detI.second) - 1;
            const double& sign_p = detI.second > 0 ? 1.0 : -1.0;
            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
                auto& detJ = coupled_dets[b];
                const size_t& q = std::abs(detJ.second) - 1;
                const double& sign_q = detJ.second > 0 ? 1.0 : -1.0;
                const size_t& J = detJ.first;
                oprdm_a[p * ncmo_ + q] +=
                    evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_p * sign_q;
                oprdm_a[q * ncmo_ + p] +=
                    evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_p * sign_q;
            }
        }
    }
    for (size_t K = 0, max_K = b_list.size(); K < max_K; ++K) {
        std::vector<std::pair<size_t, short>>& coupled_dets = b_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {
            auto& detI = coupled_dets[a];
            const size_t& I = detI.first;
            const size_t& p = std::abs(detI.second) - 1;
            const double& sign_p = detI.second > 0 ? 1.0 : -1.0;
            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {
                auto& detJ = coupled_dets[b];
                const size_t& q = std::abs(detJ.second) - 1;
                const double& sign_q = detJ.second > 0 ? 1.0 : -1.0;
                const size_t& J = detJ.first;
                oprdm_b[p * ncmo_ + q] +=
                    evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_p * sign_q;
                oprdm_b[q * ncmo_ + p] +=
                    evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_p * sign_q;
            }
        }
    }

    if (print_) {
        outfile->Printf("\n  Time spent building 1-rdm:   %1.6f", build.get());
    }
}

void CI_RDMS::compute_2rdm(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                           std::vector<double>& tprdm_bb) {
    tprdm_aa.assign(ncmo4_, 0.0);
    tprdm_ab.assign(ncmo4_, 0.0);
    tprdm_bb.assign(ncmo4_, 0.0);

    local_timer two;
    get_two_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 2-map:   %1.6f", two.get());

    local_timer build;
    for (size_t J = 0; J < dim_space_; ++J) {
        // aaaa
        for (auto& aaJ_mo_sign : aa_ann_list_[J]) {
            const size_t aaJ_add = std::get<0>(aaJ_mo_sign);

            const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aaJ_mo_sign);
            const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;

            for (auto& aaaaJ_mo_sign : aa_cre_list_[aaJ_add]) {
                const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                const size_t s = std::get<2>(aaaaJ_mo_sign);
                const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t I = std::get<0>(aaaaJ_mo_sign);
                double rdm_element =
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;

                tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += rdm_element;
                tprdm_aa[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] -= rdm_element;
                tprdm_aa[q * ncmo3_ + p * ncmo2_ + r * ncmo_ + s] -= rdm_element;
                tprdm_aa[q * ncmo3_ + p * ncmo2_ + s * ncmo_ + r] += rdm_element;
            }
        }

        // bbbb
        for (auto& bbJ_mo_sign : bb_ann_list_[J]) {
            const size_t bbJ_add = std::get<0>(bbJ_mo_sign);

            const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(bbJ_mo_sign);
            const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;

            for (auto& bbbbJ_mo_sign : bb_cre_list_[bbJ_add]) {
                const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                const size_t s = std::get<2>(bbbbJ_mo_sign);
                const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t I = std::get<0>(bbbbJ_mo_sign);
                double rdm_element =
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;

                tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += rdm_element;
                tprdm_bb[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] -= rdm_element;
                tprdm_bb[q * ncmo3_ + p * ncmo2_ + r * ncmo_ + s] -= rdm_element;
                tprdm_bb[q * ncmo3_ + p * ncmo2_ + s * ncmo_ + r] += rdm_element;
            }
        }
        // aabb
        for (auto& abJ_mo_sign : ab_ann_list_[J]) {
            const size_t abJ_add = std::get<0>(abJ_mo_sign);

            const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
            const size_t q = std::get<2>(abJ_mo_sign);
            const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;

            for (auto& aabbJ_mo_sign : ab_cre_list_[abJ_add]) {
                const size_t r = std::abs(std::get<1>(aabbJ_mo_sign)) - 1;
                const size_t s = std::get<2>(aabbJ_mo_sign);
                const double sign_rs = std::get<1>(aabbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t I = std::get<0>(aabbJ_mo_sign);
                double rdm_element =
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;

                tprdm_ab[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += rdm_element;
            }
        }
    }
    if (print_)
        outfile->Printf("\n  Time spent building 2-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_2rdm_op(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                              std::vector<double>& tprdm_bb) {
    local_timer build;

    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->build_strings(wfn_);
    op->tp_s_lists(wfn_);

    const det_hashvec& dets = wfn_.wfn_hash();

    tprdm_aa.assign(ncmo4_, 0.0);
    tprdm_ab.assign(ncmo4_, 0.0);
    tprdm_bb.assign(ncmo4_, 0.0);

    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_list = op->aa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_list = op->ab_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_list = op->bb_list_;

    for (size_t J = 0; J < dim_space_; ++J) {
        double cJ_sq = evecs_->get(J, root1_) * evecs_->get(J, root2_);
        std::vector<int> aocc = dets[J].get_alfa_occ(ncmo_);
        std::vector<int> bocc = dets[J].get_beta_occ(ncmo_);

        int naocc = aocc.size();
        int nbocc = bocc.size();

        for (int p = 0; p < naocc; ++p) {
            int pp = aocc[p];
            for (int q = p + 1; q < naocc; ++q) {
                int qq = aocc[q];
                tprdm_aa[pp * ncmo3_ + qq * ncmo2_ + pp * ncmo_ + qq] += cJ_sq;
                tprdm_aa[qq * ncmo3_ + pp * ncmo2_ + pp * ncmo_ + qq] -= cJ_sq;

                tprdm_aa[qq * ncmo3_ + pp * ncmo2_ + qq * ncmo_ + pp] += cJ_sq;
                tprdm_aa[pp * ncmo3_ + qq * ncmo2_ + qq * ncmo_ + pp] -= cJ_sq;
            }
        }

        for (int p = 0; p < nbocc; ++p) {
            int pp = bocc[p];
            for (int q = p + 1; q < nbocc; ++q) {
                int qq = bocc[q];
                tprdm_bb[pp * ncmo3_ + qq * ncmo2_ + pp * ncmo_ + qq] += cJ_sq;
                tprdm_bb[qq * ncmo3_ + pp * ncmo2_ + pp * ncmo_ + qq] -= cJ_sq;

                tprdm_bb[qq * ncmo3_ + pp * ncmo2_ + qq * ncmo_ + pp] += cJ_sq;
                tprdm_bb[pp * ncmo3_ + qq * ncmo2_ + qq * ncmo_ + pp] -= cJ_sq;
            }
        }

        for (int p = 0; p < naocc; ++p) {
            int pp = aocc[p];
            for (int q = 0; q < nbocc; ++q) {
                int qq = bocc[q];
                tprdm_ab[pp * ncmo3_ + qq * ncmo2_ + pp * ncmo_ + qq] += cJ_sq;
            }
        }
    }

    // aaaa
    for (size_t K = 0, max_K = aa_list.size(); K < max_K; ++K) {
        std::vector<std::tuple<size_t, short, short>>& coupled_dets = aa_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {

            auto& detJ = coupled_dets[a];

            const size_t& J = std::get<0>(detJ);
            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
            const size_t& q = std::get<2>(detJ);
            const double& sign_pq = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;

            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {

                auto& detI = coupled_dets[b];

                const size_t& r = std::abs(std::get<1>(detI)) - 1;
                const size_t& s = std::get<2>(detI);
                const double& sign_rs = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                const size_t& I = std::get<0>(detI);
                double rdm_element =
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;

                tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += rdm_element;
                tprdm_aa[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] -= rdm_element;
                tprdm_aa[q * ncmo3_ + p * ncmo2_ + r * ncmo_ + s] -= rdm_element;
                tprdm_aa[q * ncmo3_ + p * ncmo2_ + s * ncmo_ + r] += rdm_element;

                tprdm_aa[r * ncmo3_ + s * ncmo2_ + p * ncmo_ + q] += rdm_element;
                tprdm_aa[s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= rdm_element;
                tprdm_aa[r * ncmo3_ + s * ncmo2_ + q * ncmo_ + p] -= rdm_element;
                tprdm_aa[s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += rdm_element;
            }
        }
    }

    // bbbb
    for (size_t K = 0, max_K = bb_list.size(); K < max_K; ++K) {
        std::vector<std::tuple<size_t, short, short>>& coupled_dets = bb_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {

            auto& detJ = coupled_dets[a];

            const size_t& J = std::get<0>(detJ);
            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
            const size_t& q = std::get<2>(detJ);
            const double& sign_pq = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;

            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {

                auto& detI = coupled_dets[b];

                const size_t& r = std::abs(std::get<1>(detI)) - 1;
                const size_t& s = std::get<2>(detI);
                const double& sign_rs = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                const size_t& I = std::get<0>(detI);
                double rdm_element =
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;

                tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += rdm_element;
                tprdm_bb[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] -= rdm_element;
                tprdm_bb[q * ncmo3_ + p * ncmo2_ + r * ncmo_ + s] -= rdm_element;
                tprdm_bb[q * ncmo3_ + p * ncmo2_ + s * ncmo_ + r] += rdm_element;

                tprdm_bb[r * ncmo3_ + s * ncmo2_ + p * ncmo_ + q] += rdm_element;
                tprdm_bb[s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= rdm_element;
                tprdm_bb[r * ncmo3_ + s * ncmo2_ + q * ncmo_ + p] -= rdm_element;
                tprdm_bb[s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += rdm_element;
            }
        }
    }
    // aabb
    for (size_t K = 0, max_K = ab_list.size(); K < max_K; ++K) {
        std::vector<std::tuple<size_t, short, short>>& coupled_dets = ab_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {

            auto& detJ = coupled_dets[a];

            const size_t& J = std::get<0>(detJ);
            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
            const size_t& q = std::get<2>(detJ);
            const double& sign_pq = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;

            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {

                auto& detI = coupled_dets[b];

                const size_t& r = std::abs(std::get<1>(detI)) - 1;
                const size_t& s = std::get<2>(detI);
                const double& sign_rs = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                const size_t& I = std::get<0>(detI);
                double rdm_element =
                    evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;

                tprdm_ab[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += rdm_element;
                tprdm_ab[r * ncmo3_ + s * ncmo2_ + p * ncmo_ + q] += rdm_element;
            }
        }
    }

    if (print_)
        outfile->Printf("\n  Time spent building 2-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_3rdm(std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab,
                           std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb) {
    size_t ncmo5 = ncmo4_ * ncmo_;
    size_t ncmo6 = ncmo3_ * ncmo3_;

    tprdm_aaa.assign(ncmo6, 0.0);
    tprdm_aab.assign(ncmo6, 0.0);
    tprdm_abb.assign(ncmo6, 0.0);
    tprdm_bbb.assign(ncmo6, 0.0);

    local_timer three;
    get_three_map();
    if (print_)
        outfile->Printf("\n  Time spent forming 3-map:   %1.6f", three.get());

    local_timer build;
    for (size_t J = 0; J < dim_space_; ++J) {
        // aaa aaa
        for (auto& aaaJ_mo_sign : aaa_ann_list_[J]) {
            const size_t aaaJ_add = std::get<0>(aaaJ_mo_sign);

            const size_t p = std::abs(std::get<1>(aaaJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aaaJ_mo_sign);
            const size_t r = std::get<3>(aaaJ_mo_sign);
            const double sign_pqr = std::get<1>(aaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;

            for (auto& a6J : aaa_cre_list_[aaaJ_add]) {
                const size_t s = std::abs(std::get<1>(a6J)) - 1;
                const size_t t = std::get<2>(a6J);
                const size_t u = std::get<3>(a6J);
                const double sign_stu = std::get<1>(a6J) > 0.0 ? 1.0 : -1.0;
                const size_t I = std::get<0>(a6J);

                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;

                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;
            }
        }
        // aab aab
        for (auto& aabJ_mo_sign : aab_ann_list_[J]) {
            const size_t aabJ_add = std::get<0>(aabJ_mo_sign);

            const size_t p = std::abs(std::get<1>(aabJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aabJ_mo_sign);
            const size_t r = std::get<3>(aabJ_mo_sign);
            const double sign_pqr = std::get<1>(aabJ_mo_sign) > 0.0 ? 1.0 : -1.0;

            for (auto& aabJ : aab_cre_list_[aabJ_add]) {
                const size_t s = std::abs(std::get<1>(aabJ)) - 1;
                const size_t t = std::get<2>(aabJ);
                const size_t u = std::get<3>(aabJ);
                const double sign_stu = std::get<1>(aabJ) > 0.0 ? 1.0 : -1.0;
                const size_t I = std::get<0>(aabJ);

                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;

                tprdm_aab[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_aab[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_aab[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_aab[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
            }
        }
        // abb abb
        for (auto& abbJ_mo_sign : abb_ann_list_[J]) {
            const size_t abbJ_add = std::get<0>(abbJ_mo_sign);

            const size_t p = std::abs(std::get<1>(abbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(abbJ_mo_sign);
            const size_t r = std::get<3>(abbJ_mo_sign);
            const double sign_pqr = std::get<1>(abbJ_mo_sign) > 0.0 ? 1.0 : -1.0;

            for (auto& abbJ : abb_cre_list_[abbJ_add]) {
                const size_t s = std::abs(std::get<1>(abbJ)) - 1;
                const size_t t = std::get<2>(abbJ);
                const size_t u = std::get<3>(abbJ);
                const double sign_stu = std::get<1>(abbJ) > 0.0 ? 1.0 : -1.0;
                const size_t I = std::get<0>(abbJ);

                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;

                tprdm_abb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_abb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_abb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_abb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
            }
        }
        // bbb bbb
        for (auto& bbbJ_mo_sign : bbb_ann_list_[J]) {
            const size_t bbbJ_add = std::get<0>(bbbJ_mo_sign);

            const size_t p = std::abs(std::get<1>(bbbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(bbbJ_mo_sign);
            const size_t r = std::get<3>(bbbJ_mo_sign);
            const double sign_pqr = std::get<1>(bbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;

            for (auto& b6J : bbb_cre_list_[bbbJ_add]) {
                const size_t s = std::abs(std::get<1>(b6J)) - 1;
                const size_t t = std::get<2>(b6J);
                const size_t u = std::get<3>(b6J);
                const double sign_stu = std::get<1>(b6J) > 0.0 ? 1.0 : -1.0;
                const size_t I = std::get<0>(b6J);

                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;

                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;
            }
        }
    }

    if (print_)
        outfile->Printf("\n  Time spent building 3-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_3rdm_op(std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab,
                              std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb) {

    auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    op->build_strings(wfn_);
    op->three_s_lists(wfn_);

    size_t ncmo5 = ncmo4_ * ncmo_;
    size_t ncmo6 = ncmo3_ * ncmo3_;

    tprdm_aaa.assign(ncmo6, 0.0);
    tprdm_aab.assign(ncmo6, 0.0);
    tprdm_abb.assign(ncmo6, 0.0);
    tprdm_bbb.assign(ncmo6, 0.0);

    std::vector<std::vector<std::tuple<size_t, short, short, short>>>& aaa_list = op->aaa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>>& aab_list = op->aab_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>>& abb_list = op->abb_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>>& bbb_list = op->bbb_list_;

    // Build the diagonal part
    const det_hashvec& dets = wfn_.wfn_hash();
    for (size_t I = 0; I < dim_space_; ++I) {
        double cI_sq = evecs_->get(I, root1_) * evecs_->get(I, root2_);
        Determinant detI(dets[I]);

        std::vector<int> aocc = detI.get_alfa_occ(ncmo_);
        std::vector<int> bocc = detI.get_beta_occ(ncmo_);
        int na = aocc.size();
        int nb = bocc.size();

        for (int p = 0; p < na; ++p) {
            int pp = aocc[p];
            for (int q = p + 1; q < na; ++q) {
                int qq = aocc[q];
                for (int r = q + 1; r < na; ++r) {
                    int rr = aocc[r];

                    tprdm_aaa[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] += cI_sq;
                    tprdm_aaa[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_aaa[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] += cI_sq;
                    tprdm_aaa[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_aaa[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] += cI_sq;
                    tprdm_aaa[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] -= cI_sq;

                    tprdm_aaa[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_aaa[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] += cI_sq;
                    tprdm_aaa[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_aaa[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] += cI_sq;
                    tprdm_aaa[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_aaa[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] += cI_sq;

                    tprdm_aaa[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] += cI_sq;
                    tprdm_aaa[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_aaa[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] += cI_sq;
                    tprdm_aaa[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_aaa[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] += cI_sq;
                    tprdm_aaa[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] -= cI_sq;

                    tprdm_aaa[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_aaa[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] += cI_sq;
                    tprdm_aaa[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_aaa[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] += cI_sq;
                    tprdm_aaa[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_aaa[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] += cI_sq;

                    tprdm_aaa[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] += cI_sq;
                    tprdm_aaa[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_aaa[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] += cI_sq;
                    tprdm_aaa[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_aaa[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] += cI_sq;
                    tprdm_aaa[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] -= cI_sq;

                    tprdm_aaa[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_aaa[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] += cI_sq;
                    tprdm_aaa[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_aaa[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] += cI_sq;
                    tprdm_aaa[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_aaa[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] += cI_sq;
                }
            }
        }

        for (int p = 0; p < na; ++p) {
            int pp = aocc[p];
            for (int q = p + 1; q < na; ++q) {
                int qq = aocc[q];
                for (int r = 0; r < nb; ++r) {
                    int rr = bocc[r];

                    tprdm_aab[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] += cI_sq;
                    tprdm_aab[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_aab[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_aab[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] += cI_sq;
                }
            }
        }

        for (int p = 0; p < na; ++p) {
            int pp = aocc[p];
            for (int q = 0; q < nb; ++q) {
                int qq = bocc[q];
                for (int r = q + 1; r < nb; ++r) {
                    int rr = bocc[r];

                    tprdm_abb[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] += cI_sq;
                    tprdm_abb[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_abb[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_abb[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] += cI_sq;
                }
            }
        }

        for (int p = 0; p < nb; ++p) {
            int pp = bocc[p];
            for (int q = p + 1; q < nb; ++q) {
                int qq = bocc[q];
                for (int r = q + 1; r < nb; ++r) {
                    int rr = bocc[r];

                    tprdm_bbb[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] += cI_sq;
                    tprdm_bbb[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_bbb[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] += cI_sq;
                    tprdm_bbb[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_bbb[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] += cI_sq;
                    tprdm_bbb[pp * ncmo5 + qq * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] -= cI_sq;

                    tprdm_bbb[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_bbb[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] += cI_sq;
                    tprdm_bbb[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_bbb[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] += cI_sq;
                    tprdm_bbb[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_bbb[pp * ncmo5 + rr * ncmo4_ + qq * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] += cI_sq;

                    tprdm_bbb[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] += cI_sq;
                    tprdm_bbb[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_bbb[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] += cI_sq;
                    tprdm_bbb[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_bbb[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] += cI_sq;
                    tprdm_bbb[rr * ncmo5 + pp * ncmo4_ + qq * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] -= cI_sq;

                    tprdm_bbb[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_bbb[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] += cI_sq;
                    tprdm_bbb[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_bbb[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] += cI_sq;
                    tprdm_bbb[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_bbb[rr * ncmo5 + qq * ncmo4_ + pp * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] += cI_sq;

                    tprdm_bbb[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] += cI_sq;
                    tprdm_bbb[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_bbb[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] += cI_sq;
                    tprdm_bbb[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_bbb[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] += cI_sq;
                    tprdm_bbb[qq * ncmo5 + rr * ncmo4_ + pp * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] -= cI_sq;

                    tprdm_bbb[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + qq * ncmo_ +
                              rr] -= cI_sq;
                    tprdm_bbb[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + pp * ncmo2_ + rr * ncmo_ +
                              qq] += cI_sq;
                    tprdm_bbb[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + rr * ncmo2_ + pp * ncmo_ +
                              qq] -= cI_sq;
                    tprdm_bbb[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + rr * ncmo2_ + qq * ncmo_ +
                              pp] += cI_sq;
                    tprdm_bbb[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + rr * ncmo_ +
                              pp] -= cI_sq;
                    tprdm_bbb[qq * ncmo5 + pp * ncmo4_ + rr * ncmo3_ + qq * ncmo2_ + pp * ncmo_ +
                              rr] += cI_sq;
                }
            }
        }
    }

    for (size_t K = 0, max_K = aaa_list.size(); K < max_K; ++K) {
        // aaa aaa
        std::vector<std::tuple<size_t, short, short, short>>& coupled_dets = aaa_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {

            auto& detJ = coupled_dets[a];

            const size_t& J = std::get<0>(detJ);
            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
            const size_t& q = std::get<2>(detJ);
            const size_t& r = std::get<3>(detJ);
            const double& sign_pqr = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;

            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {

                auto& detI = coupled_dets[b];

                const size_t& s = std::abs(std::get<1>(detI)) - 1;
                const size_t& t = std::get<2>(detI);
                const size_t& u = std::get<3>(detI);
                const double& sign_stu = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                const size_t& I = std::get<0>(detI);

                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;

                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_aaa[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_aaa[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_aaa[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_aaa[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_aaa[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_aaa[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_aaa[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;
                tprdm_aaa[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
                tprdm_aaa[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
                tprdm_aaa[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;
                tprdm_aaa[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
                tprdm_aaa[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;

                tprdm_aaa[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;
                tprdm_aaa[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
                tprdm_aaa[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
                tprdm_aaa[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;
                tprdm_aaa[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
                tprdm_aaa[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;

                tprdm_aaa[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;
                tprdm_aaa[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
                tprdm_aaa[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
                tprdm_aaa[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;
                tprdm_aaa[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
                tprdm_aaa[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;

                tprdm_aaa[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;
                tprdm_aaa[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
                tprdm_aaa[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
                tprdm_aaa[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;
                tprdm_aaa[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
                tprdm_aaa[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;

                tprdm_aaa[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;
                tprdm_aaa[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
                tprdm_aaa[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
                tprdm_aaa[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;
                tprdm_aaa[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
                tprdm_aaa[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;

                tprdm_aaa[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
                tprdm_aaa[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
                tprdm_aaa[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
                tprdm_aaa[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
                tprdm_aaa[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
                tprdm_aaa[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
            }
        }
    }

    for (size_t K = 0, max_K = aab_list.size(); K < max_K; ++K) {
        // aab aab
        std::vector<std::tuple<size_t, short, short, short>>& coupled_dets = aab_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {

            auto& detJ = coupled_dets[a];

            const size_t& J = std::get<0>(detJ);
            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
            const size_t& q = std::get<2>(detJ);
            const size_t& r = std::get<3>(detJ);
            const double& sign_pqr = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;

            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {

                auto& detI = coupled_dets[b];

                const size_t& s = std::abs(std::get<1>(detI)) - 1;
                const size_t& t = std::get<2>(detI);
                const size_t& u = std::get<3>(detI);
                const double& sign_stu = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                const size_t& I = std::get<0>(detI);

                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;

                tprdm_aab[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_aab[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_aab[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_aab[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;

                tprdm_aab[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;
                tprdm_aab[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
                tprdm_aab[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;
                tprdm_aab[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
            }
        }
    }

    // abb abb
    for (size_t K = 0, max_K = abb_list.size(); K < max_K; ++K) {
        std::vector<std::tuple<size_t, short, short, short>>& coupled_dets = abb_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {

            auto& detJ = coupled_dets[a];

            const size_t& J = std::get<0>(detJ);
            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
            const size_t& q = std::get<2>(detJ);
            const size_t& r = std::get<3>(detJ);
            const double& sign_pqr = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;

            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {

                auto& detI = coupled_dets[b];

                const size_t& s = std::abs(std::get<1>(detI)) - 1;
                const size_t& t = std::get<2>(detI);
                const size_t& u = std::get<3>(detI);
                const double& sign_stu = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                const size_t& I = std::get<0>(detI);

                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;

                tprdm_abb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_abb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_abb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_abb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;

                tprdm_abb[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;
                tprdm_abb[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
                tprdm_abb[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;
                tprdm_abb[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
            }
        }
    }

    for (size_t K = 0, max_K = bbb_list.size(); K < max_K; ++K) {
        // bbb bbb
        std::vector<std::tuple<size_t, short, short, short>>& coupled_dets = bbb_list[K];
        for (size_t a = 0, max_a = coupled_dets.size(); a < max_a; ++a) {

            auto& detJ = coupled_dets[a];

            const size_t& J = std::get<0>(detJ);
            const size_t& p = std::abs(std::get<1>(detJ)) - 1;
            const size_t& q = std::get<2>(detJ);
            const size_t& r = std::get<3>(detJ);
            const double& sign_pqr = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;

            for (size_t b = a + 1, max_b = coupled_dets.size(); b < max_b; ++b) {

                auto& detI = coupled_dets[b];

                const size_t& s = std::abs(std::get<1>(detI)) - 1;
                const size_t& t = std::get<2>(detI);
                const size_t& u = std::get<3>(detI);
                const double& sign_stu = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                const size_t& I = std::get<0>(detI);

                double el = evecs_->get(I, root1_) * evecs_->get(J, root2_) * sign_pqr * sign_stu;

                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_bbb[p * ncmo5 + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_bbb[p * ncmo5 + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_bbb[q * ncmo5 + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_bbb[q * ncmo5 + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
                tprdm_bbb[r * ncmo5 + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
                tprdm_bbb[r * ncmo5 + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

                tprdm_bbb[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;
                tprdm_bbb[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
                tprdm_bbb[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
                tprdm_bbb[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;
                tprdm_bbb[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
                tprdm_bbb[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;

                tprdm_bbb[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;
                tprdm_bbb[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
                tprdm_bbb[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
                tprdm_bbb[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;
                tprdm_bbb[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
                tprdm_bbb[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;

                tprdm_bbb[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;
                tprdm_bbb[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
                tprdm_bbb[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
                tprdm_bbb[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;
                tprdm_bbb[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
                tprdm_bbb[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;

                tprdm_bbb[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;
                tprdm_bbb[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
                tprdm_bbb[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
                tprdm_bbb[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;
                tprdm_bbb[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
                tprdm_bbb[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;

                tprdm_bbb[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;
                tprdm_bbb[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
                tprdm_bbb[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
                tprdm_bbb[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;
                tprdm_bbb[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
                tprdm_bbb[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;

                tprdm_bbb[s * ncmo5 + t * ncmo4_ + u * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
                tprdm_bbb[s * ncmo5 + u * ncmo4_ + t * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
                tprdm_bbb[u * ncmo5 + t * ncmo4_ + s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
                tprdm_bbb[u * ncmo5 + s * ncmo4_ + t * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
                tprdm_bbb[t * ncmo5 + s * ncmo4_ + u * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
                tprdm_bbb[t * ncmo5 + u * ncmo4_ + s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
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
        std::vector<int> aocc = detI.get_alfa_occ(ncmo_);
        std::vector<int> bocc = detI.get_beta_occ(ncmo_);

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

        std::vector<int> aocc = detI.get_alfa_occ(ncmo_);
        std::vector<int> bocc = detI.get_beta_occ(ncmo_);

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

        const std::vector<int>& aocc = detI.get_alfa_occ(ncmo_);
        const std::vector<int>& bocc = detI.get_beta_occ(ncmo_);

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
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
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
                error_1rdm_a += std::fabs(rdm - oprdm_a[q * ncmo_ + p]);
                //     outfile->Printf("\n  D1(a)[%3lu][%3lu] = %18.12lf
                //     (%18.12lf,%18.12lf)", p,q,
                //     rdm-oprdm_a[p*ncmo_+q],rdm,oprdm_a[p*ncmo_+q]);
            }
        }
    }
    outfile->Printf("\n    A 1-RDM Error :   %2.15f", error_1rdm_a);
    double error_1rdm_b = 0.0;
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
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
                error_1rdm_b += std::fabs(rdm - oprdm_b[p * ncmo_ + q]);
                // outfile->Printf("\n  D1(b)[%3lu][%3lu] = %18.12lf
                // (%18.12lf,%18.12lf)", p,q,
                // rdm-oprdm_b[p*ncmo_+q],rdm,oprdm_b[p*ncmo_+q]);
            }
        }
    }
    outfile->Printf("\n    B 1-RDM Error :   %2.15f", error_1rdm_b);

    double error_2rdm_aa = 0.0;
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
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
                            std::fabs(rdm - tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]);
                        if (std::fabs(rdm - tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(aaaa)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s],
                                            rdm, tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]);
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n    AAAA 2-RDM Error :   %2.15f", error_2rdm_aa);
    double error_2rdm_bb = 0.0;
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
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
                            std::fabs(rdm - tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]);
                        if (std::fabs(rdm - tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(bbbb)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s],
                                            rdm, tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]);
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n    BBBB 2-RDM Error :   %2.15f", error_2rdm_bb);
    double error_2rdm_ab = 0.0;
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
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
                            std::fabs(rdm - tprdm_ab[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]);
                        if (std::fabs(rdm - tprdm_ab[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]) >
                            1.0e-12) {
                            outfile->Printf("\n  D2(abab)[%3lu][%3lu][%3lu][%3lu] = %18.12lf "
                                            "(%18.12lf,%18.12lf)",
                                            p, q, r, s,
                                            rdm - tprdm_ab[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s],
                                            rdm, tprdm_ab[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s]);
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
    for (size_t p = 0; p < ncmo_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    for (size_t t = 0; t < ncmo_; ++t) {
                        for (size_t a = 0; a < ncmo_; ++a) {
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
                                double rdm_comp =
                                    tprdm_aaa[p * ncmo4_ * ncmo_ + q * ncmo4_ + r * ncmo3_ +
                                              s * ncmo2_ + t * ncmo_ + a];
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
    for (size_t p = 0; p < ncmo_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    for (size_t t = 0; t < ncmo_; ++t) {
                        for (size_t a = 0; a < ncmo_; ++a) {
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
                                double rdm_comp =
                                    tprdm_aab[p * ncmo4_ * ncmo_ + q * ncmo4_ + r * ncmo3_ +
                                              s * ncmo2_ + t * ncmo_ + a];
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
    for (size_t p = 0; p < ncmo_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    for (size_t t = 0; t < ncmo_; ++t) {
                        for (size_t a = 0; a < ncmo_; ++a) {
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
                                double rdm_comp =
                                    tprdm_abb[p * ncmo4_ * ncmo_ + q * ncmo4_ + r * ncmo3_ +
                                              s * ncmo2_ + t * ncmo_ + a];
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
    for (size_t p = 0; p < ncmo_; ++p) {
        // for (size_t p = 0; p < 1; ++p){
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    for (size_t t = 0; t < ncmo_; ++t) {
                        for (size_t a = 0; a < ncmo_; ++a) {
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
                                double rdm_comp =
                                    tprdm_bbb[p * ncmo4_ * ncmo_ + q * ncmo4_ + r * ncmo3_ +
                                              s * ncmo2_ + t * ncmo_ + a];
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
