/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "helpers/timer.h"
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

void SigmaVectorSparseList::compute_sigma(psi::SharedVector sigma, psi::SharedVector b) {
    timer timer_sigma("Build sigma");

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
        // outfile->Printf("\n Overlap: %1.6f", overlap[0]);

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

        //        #pragma omp critical
        //        {
        for (size_t I = 0; I < size_; ++I) {
#pragma omp atomic update
            sigma_p[I] += sigma_t[I];
        }
        //        }
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

void SigmaVectorSparseList::add_generalized_sigma_1a(const std::vector<double>& h1a,
                                                     psi::SharedVector b, double factor,
                                                     std::vector<double>& sigma) {
    timer timer_sigma("Build generalized sigma 1a");

    auto a_list = op_->a_list_; // substitution list already available

    auto nactv = fci_ints_->nmo();
    if (h1a.size() != nactv * nactv) {
        throw std::runtime_error("Incorrect dimension for 1e integrals.");
    }

    auto b_ptr = b->pointer();
    if (static_cast<size_t>(b->dim()) != size_ or sigma.size() != size_) {
        throw std::runtime_error("Incorrect dimension for determinants space.");
    }

    // copied from https://stackoverflow.com/questions/43168661/openmp-and-reduction-on-stdvector
#pragma omp declare reduction(vector_plus : std::vector<double> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp parallel for reduction(vector_plus : sigma)
    // diagonal: sigma_I <- c_I * \sum_{p} <I| p^+ p |I> * h^{p}_{p}
    for (size_t I = 0; I < size_; ++I) {
        auto det = space_[I];
        auto Ia = det.get_alfa_bits();
        auto naocc = Ia.count();

        double value = 0.0;
        for (int A = 0; A < naocc; ++A) {
            auto p = Ia.find_and_clear_first_one();
            value += h1a[p * nactv + p];
        }

        sigma[I] += factor * value * b_ptr[I];
    }

    // off-diagonal: sigma_I <- \sum_{J} c_J \sum_{pq} <I| p^+ q |J> * h^{p}_{q}
    std::vector<double> sigma_threads;
#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t Ksize = a_list.size();

#pragma omp single
        sigma_threads.resize(size_ * nthreads);

#pragma omp for schedule(static)
        for (size_t K = 0; K < Ksize; ++K) {
            const auto& cre_dets = a_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = detJ.first;
                const auto p = std::abs(detJ.second) - 1;
                const auto sign_p = detJ.second > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = detI.first;
                    const auto q = std::abs(detI.second) - 1;
                    const auto sign_q = detI.second > 0 ? 1 : -1;

                    const double HIJ = h1a[p * nactv + q] * sign_p * sign_q;
                    sigma_threads[I + tid * size_] += HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += HIJ * b_ptr[I];
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t I = 0; I < size_; ++I) {
            for (int t = 0; t < nthreads; ++t) {
                sigma[I] += factor * sigma_threads[size_ * t + I];
            }
        }
    };
}

void SigmaVectorSparseList::add_generalized_sigma_1b(const std::vector<double>& h1b,
                                                     psi::SharedVector b, double factor,
                                                     std::vector<double>& sigma) {
    timer timer_sigma("Build generalized sigma 1b");
    auto b_list = op_->b_list_;

    auto nactv = fci_ints_->nmo();
    if (h1b.size() != nactv * nactv) {
        throw std::runtime_error("Incorrect dimension for 1e integrals.");
    }

    auto b_ptr = b->pointer();
    if (static_cast<size_t>(b->dim()) != size_ or sigma.size() != size_) {
        throw std::runtime_error("Incorrect dimension for determinants space.");
    }

#pragma omp declare reduction(vector_plus : std::vector<double> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp parallel for reduction(vector_plus : sigma)
    // diagonal: sigma_I <- c_I * \sum_{p} <I| p^+ p |I> * h^{p}_{p}
    for (size_t I = 0; I < size_; ++I) {
        auto det = space_[I];
        auto Ib = det.get_beta_bits();
        auto nbocc = Ib.count();

        double value = 0.0;
        for (int B = 0; B < nbocc; ++B) {
            auto p = Ib.find_and_clear_first_one();
            value += h1b[p * nactv + p];
        }

        sigma[I] += factor * value * b_ptr[I];
    }

    // off-diagonal: sigma_I <- \sum_{J} c_J \sum_{pq} <I| p^+ q |J> h^{p}_{q}
    std::vector<double> sigma_threads;
#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t Ksize = b_list.size();

#pragma omp single
        sigma_threads.resize(size_ * nthreads);

#pragma omp for schedule(static)
        for (size_t K = 0; K < Ksize; ++K) {
            const auto& cre_dets = b_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = detJ.first;
                const auto p = std::abs(detJ.second) - 1;
                const auto sign_p = detJ.second > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = detI.first;
                    const auto q = std::abs(detI.second) - 1;
                    const auto sign_q = detI.second > 0 ? 1 : -1;

                    const double HIJ = h1b[p * nactv + q] * sign_p * sign_q;
                    sigma_threads[I + tid * size_] += HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += HIJ * b_ptr[I];
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t I = 0; I < size_; ++I) {
            for (int t = 0; t < nthreads; ++t) {
                sigma[I] += factor * sigma_threads[size_ * t + I];
            }
        }
    };
}

void SigmaVectorSparseList::add_generalized_sigma_2aa(const std::vector<double>& h2aa,
                                                      psi::SharedVector b, double factor,
                                                      std::vector<double>& sigma) {
    timer timer_sigma("Build generalized sigma 2aa");
    auto a_list = op_->a_list_;
    auto aa_list = op_->aa_list_;

    auto nactv = fci_ints_->nmo();
    auto nactv2 = nactv * nactv;
    auto nactv3 = nactv * nactv2;
    if (h2aa.size() != nactv * nactv3) {
        throw std::runtime_error("Incorrect dimension for 2e integrals.");
    }

    auto b_ptr = b->pointer();
    if (static_cast<size_t>(b->dim()) != size_ or sigma.size() != size_) {
        throw std::runtime_error("Incorrect dimension for determinants space.");
    }

#pragma omp declare reduction(vector_plus : std::vector<double> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp parallel for reduction(vector_plus : sigma)
    // diagonal: sigma_I <- c_I * \sum_{pq} <I| p^+ q^+ q p |I> v^{pq}_{pq}
    for (size_t I = 0; I < size_; ++I) {
        auto det = space_[I];
        auto Ia = det.get_alfa_bits();
        auto naocc = Ia.count();

        double value = 0.0;
        for (int A1 = 0; A1 < naocc; ++A1) {
            auto p = Ia.find_and_clear_first_one();

            auto Ia_p = Ia;
            for (int A2 = A1 + 1; A2 < naocc; ++A2) {
                auto q = Ia_p.find_and_clear_first_one();
                value += h2aa[p * nactv3 + q * nactv2 + p * nactv + q];
            }
        }

        sigma[I] += factor * value * b_ptr[I];
    }

    // off-diagonal: sigma_I <- \sum_{J} c_J \sum_{pqrs} <I| p^+ q^+ s r |J> * v^{pq}_{rs}
    std::vector<double> sigma_threads;
#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t Ksize1 = a_list.size();
        const size_t Ksize2 = aa_list.size();

#pragma omp single
        sigma_threads.resize(size_ * nthreads);

#pragma omp for schedule(static)
        // effectively singles
        for (size_t K = 0; K < Ksize1; ++K) {
            const auto& cre_dets = a_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = detJ.first;
                const auto p = std::abs(detJ.second) - 1;
                const auto sign_p = detJ.second > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = detI.first;
                    const auto q = std::abs(detI.second) - 1;
                    const auto sign_q = detI.second > 0 ? 1 : -1;

                    double HIJ = 0.0;
                    for (size_t u = 0; u < nactv; ++u) {
                        HIJ += h2aa[p * nactv3 + u * nactv2 + q * nactv + u] *
                               space_[J].get_alfa_bit(u);
                    }

                    HIJ *= sign_p * sign_q;
                    sigma_threads[I + tid * size_] += HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += HIJ * b_ptr[I];
                }
            }
        }

#pragma omp for schedule(static)
        // doubles
        for (size_t K = 0; K < Ksize2; ++K) {
            const auto& cre_dets = aa_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = std::get<0>(detJ);
                const auto p = std::abs(std::get<1>(detJ)) - 1;
                const auto q = std::get<2>(detJ);
                const auto sign_pq = std::get<1>(detJ) > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = std::get<0>(detI);
                    const auto r = std::abs(std::get<1>(detI)) - 1;
                    const auto s = std::get<2>(detI);
                    const auto sign_rs = std::get<1>(detI) > 0 ? 1 : -1;

                    int valid = ((p != r) and (q != s) and (p != s) and (q != r)) ? 1 : 0;
                    auto HIJ = h2aa[p * nactv3 + q * nactv2 + r * nactv + s] * sign_pq * sign_rs;
                    sigma_threads[I + tid * size_] += valid * HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += valid * HIJ * b_ptr[I];
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t I = 0; I < size_; ++I) {
            for (int t = 0; t < nthreads; ++t) {
                sigma[I] += factor * sigma_threads[size_ * t + I];
            }
        }
    };
}

void SigmaVectorSparseList::add_generalized_sigma_2ab(const std::vector<double>& h2ab,
                                                      psi::SharedVector b, double factor,
                                                      std::vector<double>& sigma) {
    timer timer_sigma("Build generalized sigma 2ab");
    auto a_list = op_->a_list_;
    auto b_list = op_->b_list_;
    auto ab_list = op_->ab_list_;

    auto nactv = fci_ints_->nmo();
    auto nactv2 = nactv * nactv;
    auto nactv3 = nactv * nactv2;
    if (h2ab.size() != nactv * nactv3) {
        throw std::runtime_error("Incorrect dimension for 2e integrals.");
    }

    auto b_ptr = b->pointer();
    if (static_cast<size_t>(b->dim()) != size_ or sigma.size() != size_) {
        throw std::runtime_error("Incorrect dimension for determinants space.");
    }

#pragma omp declare reduction(vector_plus : std::vector<double> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp parallel for reduction(vector_plus : sigma)
    // diagonal: sigma_I <- c_I * \sum_{pq} <I| p^+ q^+ q p |I> v^{pq}_{pq}
    for (size_t I = 0; I < size_; ++I) {
        auto det = space_[I];
        auto Ia = det.get_alfa_bits();
        auto Ib = det.get_beta_bits();
        auto naocc = Ia.count();
        auto nbocc = Ib.count();

        double value = 0.0;
        for (int A = 0; A < naocc; ++A) {
            auto p = Ia.find_and_clear_first_one();

            auto Ib_copy = Ib;
            for (int B = 0; B < nbocc; ++B) {
                auto q = Ib_copy.find_and_clear_first_one();
                value += h2ab[p * nactv3 + q * nactv2 + p * nactv + q];
            }
        }

        sigma[I] += factor * value * b_ptr[I];
    }

    // off-diagonal: sigma_I <- \sum_{J} c_J \sum_{pqrs} <I| p^+ q^+ s r |J> * v^{pq}_{rs}
    std::vector<double> sigma_threads;
#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();

#pragma omp single
        sigma_threads.resize(size_ * nthreads);

        const size_t Ksize1a = a_list.size();
#pragma omp for schedule(static)
        // effectively singles alpha
        for (size_t K = 0; K < Ksize1a; ++K) {
            const auto& cre_dets = a_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = detJ.first;
                const auto p = std::abs(detJ.second) - 1;
                const auto sign_p = detJ.second > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = detI.first;
                    const auto q = std::abs(detI.second) - 1;
                    const auto sign_q = detI.second > 0 ? 1 : -1;

                    double HIJ = 0.0;
                    for (size_t u = 0; u < nactv; ++u) {
                        HIJ += h2ab[p * nactv3 + u * nactv2 + q * nactv + u] *
                               space_[J].get_beta_bit(u);
                    }

                    HIJ *= sign_p * sign_q;
                    sigma_threads[I + tid * size_] += HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += HIJ * b_ptr[I];
                }
            }
        }

        const size_t Ksize1b = b_list.size();
#pragma omp for schedule(static)
        // effectively singles beta
        for (size_t K = 0; K < Ksize1b; ++K) {
            const auto& cre_dets = b_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = detJ.first;
                const auto p = std::abs(detJ.second) - 1;
                const auto sign_p = detJ.second > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = detI.first;
                    const auto q = std::abs(detI.second) - 1;
                    const auto sign_q = detI.second > 0 ? 1 : -1;

                    double HIJ = 0.0;
                    for (size_t u = 0; u < nactv; ++u) {
                        HIJ += h2ab[u * nactv3 + p * nactv2 + u * nactv + q] *
                               space_[J].get_alfa_bit(u);
                    }

                    HIJ *= sign_p * sign_q;
                    sigma_threads[I + tid * size_] += HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += HIJ * b_ptr[I];
                }
            }
        }

        const size_t Ksize2 = ab_list.size();
#pragma omp for schedule(static)
        // doubles
        for (size_t K = 0; K < Ksize2; ++K) {
            const auto& cre_dets = ab_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = std::get<0>(detJ);
                const auto p = std::abs(std::get<1>(detJ)) - 1;
                const auto q = std::get<2>(detJ);
                const auto sign_pq = std::get<1>(detJ) > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = std::get<0>(detI);
                    const auto r = std::abs(std::get<1>(detI)) - 1;
                    const auto s = std::get<2>(detI);
                    const auto sign_rs = std::get<1>(detI) > 0 ? 1 : -1;

                    int valid = ((p != r) and (q != s)) ? 1 : 0;
                    auto HIJ = h2ab[p * nactv3 + q * nactv2 + r * nactv + s] * sign_pq * sign_rs;
                    sigma_threads[I + tid * size_] += valid * HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += valid * HIJ * b_ptr[I];
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t I = 0; I < size_; ++I) {
            for (int t = 0; t < nthreads; ++t) {
                sigma[I] += factor * sigma_threads[size_ * t + I];
            }
        }
    };
}

void SigmaVectorSparseList::add_generalized_sigma_2bb(const std::vector<double>& h2bb,
                                                      psi::SharedVector b, double factor,
                                                      std::vector<double>& sigma) {
    timer timer_sigma("Build generalized sigma 2bb");
    auto b_list = op_->b_list_;
    auto bb_list = op_->bb_list_;

    auto nactv = fci_ints_->nmo();
    auto nactv2 = nactv * nactv;
    auto nactv3 = nactv * nactv2;
    if (h2bb.size() != nactv * nactv3) {
        throw std::runtime_error("Incorrect dimension for 2e integrals.");
    }

    auto b_ptr = b->pointer();
    if (static_cast<size_t>(b->dim()) != size_ or sigma.size() != size_) {
        throw std::runtime_error("Incorrect dimension for determinants space.");
    }

#pragma omp declare reduction(vector_plus : std::vector<double> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp parallel for reduction(vector_plus : sigma)
    // diagonal: sigma_I <- c_I * \sum_{pq} <I| p^+ q^+ q p |I> v^{pq}_{pq}
    for (size_t I = 0; I < size_; ++I) {
        auto det = space_[I];
        auto Ib = det.get_beta_bits();
        auto nbocc = Ib.count();

        double value = 0.0;
        for (int B1 = 0; B1 < nbocc; ++B1) {
            auto p = Ib.find_and_clear_first_one();

            auto Ib_p = Ib;
            for (int B2 = B1 + 1; B2 < nbocc; ++B2) {
                auto q = Ib_p.find_and_clear_first_one();
                value += h2bb[p * nactv3 + q * nactv2 + p * nactv + q];
            }
        }

        sigma[I] += factor * value * b_ptr[I];
    }

    // off-diagonal: sigma_I <- \sum_{J} c_J \sum_{pqrs} <I| p^+ q^+ s r |J> * v^{pq}_{rs}
    std::vector<double> sigma_threads;
#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t Ksize1 = b_list.size();
        const size_t Ksize2 = bb_list.size();

#pragma omp single
        sigma_threads.resize(size_ * nthreads);

#pragma omp for schedule(static)
        // effectively singles
        for (size_t K = 0; K < Ksize1; ++K) {
            const auto& cre_dets = b_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = detJ.first;
                const auto p = std::abs(detJ.second) - 1;
                const auto sign_p = detJ.second > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = detI.first;
                    const auto q = std::abs(detI.second) - 1;
                    const auto sign_q = detI.second > 0 ? 1 : -1;

                    double HIJ = 0.0;
                    for (size_t u = 0; u < nactv; ++u) {
                        HIJ += h2bb[p * nactv3 + u * nactv2 + q * nactv + u] *
                               space_[J].get_beta_bit(u);
                    }

                    HIJ *= sign_p * sign_q;
                    sigma_threads[I + tid * size_] += HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += HIJ * b_ptr[I];
                }
            }
        }

#pragma omp for schedule(static)
        // doubles
        for (size_t K = 0; K < Ksize2; ++K) {
            const auto& cre_dets = bb_list[K];
            const auto cre_dets_size = cre_dets.size();

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                const auto& detJ = cre_dets[det1];
                const auto J = std::get<0>(detJ);
                const auto p = std::abs(std::get<1>(detJ)) - 1;
                const auto q = std::get<2>(detJ);
                const auto sign_pq = std::get<1>(detJ) > 0 ? 1 : -1;

                for (size_t det2 = det1 + 1; det2 < cre_dets_size; ++det2) {
                    const auto& detI = cre_dets[det2];
                    const auto I = std::get<0>(detI);
                    const auto r = std::abs(std::get<1>(detI)) - 1;
                    const auto s = std::get<2>(detI);
                    const auto sign_rs = std::get<1>(detI) > 0 ? 1 : -1;

                    int valid = ((p != r) and (q != s) and (p != s) and (q != r)) ? 1 : 0;
                    auto HIJ = h2bb[p * nactv3 + q * nactv2 + r * nactv + s] * sign_pq * sign_rs;
                    sigma_threads[I + tid * size_] += valid * HIJ * b_ptr[J];
                    sigma_threads[J + tid * size_] += valid * HIJ * b_ptr[I];
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t I = 0; I < size_; ++I) {
            for (int t = 0; t < nthreads; ++t) {
                sigma[I] += factor * sigma_threads[size_ * t + I];
            }
        }
    };
}

} // namespace forte
