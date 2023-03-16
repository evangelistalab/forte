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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"

#include "sigma_vector_sparse_list.h"
#include "sparse_ci/determinant_substitution_lists.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

using namespace psi;

namespace forte {

SigmaVectorSparseList::SigmaVectorSparseList(const DeterminantHashVec& space,
                                             std::shared_ptr<ActiveSpaceIntegrals> fci_ints)
    : SigmaVector(space, fci_ints, SigmaVectorType::SparseList, "SigmaVectorSparseList") {

    op_ = std::make_shared<DeterminantSubstitutionLists>(fci_ints_);
    /// Build the coupling lists for 1- and 2-particle operators
    op_->build_strings(space_);
    op_->op_s_lists(space_);
    op_->tp_s_lists(space_);
    //    op_->set_quiet_mode(quiet_mode_);

    const det_hashvec& detmap = space_.wfn_hash();
    diag_.resize(space_.size());
    for (size_t I = 0, max_I = detmap.size(); I < max_I; ++I) {
        diag_[I] = fci_ints_->energy(detmap[I]);
    }
}

void SigmaVectorSparseList::add_bad_roots(
    std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SigmaVectorSparseList::get_diagonal(psi::Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

std::shared_ptr<DeterminantSubstitutionLists> SigmaVectorSparseList::substitution_lists() {
    return op_;
}

void SigmaVectorSparseList::compute_sigma(psi::SharedVector sigma, psi::SharedVector b) {
    auto a_list_ = op_->a_list_;
    auto b_list_ = op_->b_list_;
    auto aa_list_ = op_->aa_list_;
    auto ab_list_ = op_->ab_list_;
    auto bb_list_ = op_->bb_list_;

    sigma->zero();

    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();

    // Compute the overlap with each root
    int nbad = bad_states_.size();
    std::vector<double> overlap(nbad);
    if (nbad != 0) {
        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            double dprd = 0.0;
            for (size_t det = 0, ndet = bad_state.size(); det < ndet; ++det) {
                dprd += bad_state[det].second * b_p[bad_state[det].first];
            }
            overlap[n] = dprd;
        }
        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            size_t ndet = bad_state.size();

#pragma omp parallel for
            for (size_t det = 0; det < ndet; ++det) {
                b_p[bad_state[det].first] -= bad_state[det].second * overlap[n];
            }
        }
    }

    auto& dets = space_.wfn_hash();

#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();

        // Each thread gets local copy of sigma
        std::vector<double> sigma_t(size_);

        size_t bin_size = size_ / num_thread;
        bin_size += (tid < (size_ % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (size_ % num_thread))
                ? tid * bin_size
                : (size_ % num_thread) * (bin_size + 1) + (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

#pragma omp critical
        for (size_t J = start_idx; J < end_idx; ++J) {
            sigma_p[J] += diag_[J] * b_p[J]; // Make DDOT
        }

        // a singles
        size_t end_a_idx = a_list_.size();
        size_t start_a_idx = 0;
        for (size_t K = start_a_idx, max_K = end_a_idx; K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::pair<size_t, short>>& c_dets = a_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    const size_t J = detJ.first;
                    const size_t p = std::abs(detJ.second) - 1;
                    double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        const size_t q = std::abs(detI.second) - 1;
                        if (p != q) {
                            const size_t I = detI.first;
                            double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                fci_ints_->slater_rules_single_alpha_abs(dets[J], p, q) * sign_p *
                                sign_q;
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }

        // b singles
        size_t end_b_idx = b_list_.size();
        size_t start_b_idx = 0;
        for (size_t K = start_b_idx, max_K = end_b_idx; K < max_K; ++K) {
            // aa singles
            if ((K % num_thread) == tid) {
                const std::vector<std::pair<size_t, short>>& c_dets = b_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    const size_t J = detJ.first;
                    const size_t p = std::abs(detJ.second) - 1;
                    double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        const size_t q = std::abs(detI.second) - 1;
                        if (p != q) {
                            const size_t I = detI.first;
                            double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                fci_ints_->slater_rules_single_beta_abs(dets[J], p, q) * sign_p *
                                sign_q;
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }

        // AA doubles
        size_t aa_size = aa_list_.size();
        //      size_t bin_aa_size = aa_size / num_thread;
        //      bin_aa_size += (tid < (aa_size % num_thread)) ? 1 : 0;
        //      size_t start_aa_idx = (tid < (aa_size % num_thread))
        //                             ? tid * bin_aa_size
        //                             : (aa_size % num_thread) * (bin_aa_size + 1) +
        //                                   (tid - (aa_size % num_thread)) * bin_aa_size;
        //      size_t end_aa_idx = start_aa_idx + bin_aa_size;
        for (size_t K = 0, max_K = aa_size; K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = aa_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s) and (p != s) and (q != r)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_aa(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }

        // BB doubles
        for (size_t K = 0, max_K = bb_list_.size(); K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = bb_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s) and (p != s) and (q != r)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_bb(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }
        for (size_t K = 0, max_K = ab_list_.size(); K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = ab_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_ab(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }

#pragma omp critical
        for (size_t I = 0; I < size_; ++I) {
            // #pragma omp atomic update
            sigma_p[I] += sigma_t[I];
        }
    }
}

double SigmaVectorSparseList::compute_spin(const std::vector<double>& c) {
    auto ab_list_ = op_->ab_list_;

    double S2 = 0.0;
    const det_hashvec& wfn_map = space_.wfn_hash();

    for (size_t i = 0, max_i = wfn_map.size(); i < max_i; ++i) {
        // Compute diagonal
        // PhiI = PhiJ
        const Determinant& PhiI = wfn_map[i];
        double CI = c[i];
        int npair = PhiI.npair();
        int na = PhiI.count_alfa();
        int nb = PhiI.count_beta();
        double ms = 0.5 * static_cast<double>(na - nb);
        S2 += (ms * ms + ms + static_cast<double>(nb) - static_cast<double>(npair)) * CI * CI;
    }

    // Loop directly through all determinants with
    // spin-coupled electrons, i.e:
    // |PhiI> = a+(qa) a+(pb) a-(qb) a-(pa) |PhiJ>

    for (size_t K = 0, max_K = ab_list_.size(); K < max_K; ++K) {
        const std::vector<std::tuple<size_t, short, short>>& c_dets = ab_list_[K];
        for (auto& detI : c_dets) {
            const size_t I = std::get<0>(detI);
            double sign_pq = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
            short p = std::fabs(std::get<1>(detI)) - 1;
            short q = std::get<2>(detI);
            if (p == q)
                continue;
            for (auto& detJ : c_dets) {
                const size_t J = std::get<0>(detJ);
                if (I == J)
                    continue;
                double sign_rs = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                short r = std::fabs(std::get<1>(detJ)) - 1;
                short s = std::get<2>(detJ);
                if ((r != s) and (p == s) and (q == r)) {
                    sign_pq *= sign_rs;
                    S2 -= sign_pq * c[I] * c[J];
                }
            }
        }
    }
    return S2;
}

} // namespace forte
