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

    op_ = std::make_shared<DeterminantSubstitutionLists>(fci_ints_->active_mo_symmetry());
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

void SigmaVectorSparseList::compute_sigma(std::shared_ptr<psi::Vector> sigma,
                                          std::shared_ptr<psi::Vector> b) {
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

void SigmaVectorSparseList::add_generalized_sigma_1(const std::vector<double>& h1,
                                                    std::shared_ptr<psi::Vector> b, double factor,
                                                    std::vector<double>& sigma,
                                                    const std::string& spin) {
    timer timer_sigma("Build generalized sigma 1" + spin);

    // test arguments
    auto nactv = fci_ints_->nmo();
    if (h1.size() != nactv * nactv) {
        throw std::runtime_error("Incorrect dimension for 1e integrals.");
    }
    if (static_cast<size_t>(b->dim()) != size_ or sigma.size() != size_) {
        throw std::runtime_error("Incorrect dimension for determinants space.");
    }

    // compute sigma 1
    if (spin == "a") {
        add_generalized_sigma1_impl(h1, b, factor, sigma, op_->a_list_);
    } else if (spin == "b") {
        add_generalized_sigma1_impl(h1, b, factor, sigma, op_->b_list_);
    } else {
        std::stringstream ss;
        ss << "Invalid spin label: " << spin << "! Expect a or b";
        throw std::runtime_error(ss.str());
    }
}

void SigmaVectorSparseList::add_generalized_sigma1_impl(
    const std::vector<double>& h1, std::shared_ptr<psi::Vector> b, double factor,
    std::vector<double>& sigma,
    const std::vector<std::vector<std::pair<size_t, short>>>& sub_lists) {
    auto nactv = fci_ints_->nmo();
    auto b_ptr = b->pointer();

    // sigma_I <- \sum_{J} c_J \sum_{pq} <I| p^+ q |J> * h^{p}_{q}
    std::vector<double> sigma_threads;
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        auto Ksize = sub_lists.size();

#pragma omp single
        sigma_threads.resize(size_ * nthreads);

#pragma omp for schedule(static)
        for (size_t K = 0; K < Ksize; ++K) {
            const auto& cre_dets = sub_lists[K];
            auto cre_dets_size = cre_dets.size();

            size_t I, J;
            int p, q;

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                std::tie(J, p) = cre_dets[det1];
                auto sign_p = p > 0 ? 1 : -1;
                p = std::abs(p) - 1;

                for (size_t det2 = det1; det2 < cre_dets_size; ++det2) {
                    std::tie(I, q) = cre_dets[det2];
                    auto sign_q = q > 0 ? 1 : -1;
                    q = std::abs(q) - 1;

                    double HIJ = h1[p * nactv + q] * sign_p * sign_q * (det1 == det2 ? 0.5 : 1.0);
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

void SigmaVectorSparseList::add_generalized_sigma_2(const std::vector<double>& h2,
                                                    std::shared_ptr<psi::Vector> b, double factor,
                                                    std::vector<double>& sigma,
                                                    const std::string& spin) {
    timer timer_sigma("Build generalized sigma 2" + spin);

    // test arguments
    auto na = fci_ints_->nmo();
    if (h2.size() != na * na * na * na) {
        throw std::runtime_error("Incorrect dimension for 2e integrals.");
    }
    if (static_cast<size_t>(b->dim()) != size_ or sigma.size() != size_) {
        throw std::runtime_error("Incorrect dimension for determinants space.");
    }

    // compute sigma 2
    if (spin == "aa") {
        if (!is_h2hs_antisymmetric(h2))
            throw std::runtime_error("h2 is not antisymmetric!");
        add_generalized_sigma2_impl(h2, b, factor, sigma, op_->aa_list_);
    } else if (spin == "ab") {
        add_generalized_sigma2_impl(h2, b, factor, sigma, op_->ab_list_);
    } else if (spin == "bb") {
        if (!is_h2hs_antisymmetric(h2))
            throw std::runtime_error("h2 is not antisymmetric!");
        add_generalized_sigma2_impl(h2, b, factor, sigma, op_->bb_list_);
    } else {
        std::stringstream ss;
        ss << "Invalid spin label: " << spin << "! Expect aa, ab, or bb";
        throw std::runtime_error(ss.str());
    }
}

bool SigmaVectorSparseList::is_h2hs_antisymmetric(const std::vector<double>& h2) {
    auto na = fci_ints_->nmo();
    if (na < 2)
        return true;

    auto na2 = na * na;
    auto na3 = na * na2;
    auto na4 = na * na3;

    size_t pass = 0;
    double zero = 1.0e-12;
    size_t nthreads = omp_get_num_threads();
    nthreads = nthreads > na4 ? na4 : nthreads;

#pragma omp parallel for default(none) shared(h2, na, na2, na3, na4, zero) num_threads(nthreads) reduction(+ : pass)
    for (size_t pqrs = 0; pqrs < na4; ++pqrs) {
        size_t p = pqrs / na3;
        size_t qrs = pqrs % na3;
        size_t q = qrs / na2;

        size_t rs = qrs % na2;
        size_t r = rs / na;
        size_t s = rs % na;

        int valid = (q > p) and (s > r);

        auto v_pqrs = h2[p * na3 + q * na2 + r * na + s];
        auto v_pqsr = h2[p * na3 + q * na2 + s * na + r];
        auto v_qprs = h2[q * na3 + p * na2 + r * na + s];
        auto v_qpsr = h2[q * na3 + p * na2 + s * na + r];

        int value = (std::fabs(v_pqrs + v_pqsr) < zero) + (std::fabs(v_pqrs + v_qprs) < zero) +
                    (std::fabs(v_pqrs - v_qpsr) < zero) + 1;
        pass += valid * value;
    }

    size_t target = na * (na - 1) * na * (na - 1);
    return pass == target;
}

void SigmaVectorSparseList::add_generalized_sigma2_impl(
    const std::vector<double>& h2, std::shared_ptr<psi::Vector> b, double factor,
    std::vector<double>& sigma,
    const std::vector<std::vector<std::tuple<size_t, short, short>>>& sub_lists) {
    auto b_ptr = b->pointer();
    auto na = fci_ints_->nmo();
    auto na2 = na * na;
    auto na3 = na * na2;

    // sigma_I <- \sum_{J} c_J \sum_{pqrs} <I| p^+ q^+ s r |J> * v^{pq}_{rs}
    std::vector<double> sigma_threads;
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        auto Ksize = sub_lists.size();

#pragma omp single
        sigma_threads.resize(size_ * nthreads);

#pragma omp for schedule(static)
        for (size_t K = 0; K < Ksize; ++K) {
            const auto& cre_dets = sub_lists[K];
            auto cre_dets_size = cre_dets.size();

            size_t I, J;
            int p, r;
            short q, s;

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                std::tie(J, p, q) = cre_dets[det1];
                auto sign_pq = p > 0 ? 1 : -1;
                p = std::abs(p) - 1;

                for (size_t det2 = det1; det2 < cre_dets_size; ++det2) {
                    std::tie(I, r, s) = cre_dets[det2];
                    auto sign_rs = r > 0 ? 1 : -1;
                    r = std::abs(r) - 1;

                    auto idx = p * na3 + q * na2 + r * na + s;
                    auto HIJ = h2[idx] * sign_pq * sign_rs * (det1 == det2 ? 0.5 : 1.0);
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

void SigmaVectorSparseList::add_generalized_sigma_3(const std::vector<double>& h3,
                                                    std::shared_ptr<psi::Vector> b, double factor,
                                                    std::vector<double>& sigma,
                                                    const std::string& spin) {
    timer timer_sigma("Build generalized sigma 3" + spin);

    // test arguments
    auto na = fci_ints_->nmo();
    if (h3.size() != na * na * na * na * na * na) {
        throw std::runtime_error("Incorrect dimension for 3e integrals.");
    }
    if (static_cast<size_t>(b->dim()) != size_ or sigma.size() != size_) {
        throw std::runtime_error("Incorrect dimension for determinants space.");
    }

    // compute sigma 3
    if (spin == "aaa") {
        if (!is_h3hs_antisymmetric(h3))
            throw std::runtime_error("h3aaa is not antisymmetric!");
        op_->lists_3aaa(space_);
        add_generalized_sigma3_impl(h3, b, factor, sigma, op_->aaa_list_);
    } else if (spin == "aab") {
        if (!is_h3ls_antisymmetric(h3, true))
            throw std::runtime_error("h3aab is not antisymmetric!");
        op_->lists_3aab(space_);
        add_generalized_sigma3_impl(h3, b, factor, sigma, op_->aab_list_);
    } else if (spin == "abb") {
        if (!is_h3ls_antisymmetric(h3, false))
            throw std::runtime_error("h3abb is not antisymmetric!");
        op_->lists_3abb(space_);
        add_generalized_sigma3_impl(h3, b, factor, sigma, op_->abb_list_);
    } else if (spin == "bbb") {
        if (!is_h3hs_antisymmetric(h3))
            throw std::runtime_error("h3bbb is not antisymmetric!");
        op_->lists_3bbb(space_);
        add_generalized_sigma3_impl(h3, b, factor, sigma, op_->bbb_list_);
    } else {
        std::stringstream ss;
        ss << "Invalid spin label: " << spin << "! Expect aaa, aab, abb, or bbb";
        throw std::runtime_error(ss.str());
    }

    // clear 3 substitution lists in op_
    op_->clear_3p_s_lists();
}

bool SigmaVectorSparseList::is_h3hs_antisymmetric(const std::vector<double>& h3) {
    timer timer_h3hs("Test antisymmetry of high-spin H3");
    auto na = fci_ints_->nmo();
    if (na < 3)
        return true;

    auto na2 = na * na;
    auto na3 = na * na2;
    auto na4 = na * na3;
    auto na5 = na * na4;
    auto na6 = na * na5;

    size_t pass = 0;
    double zero = 1.0e-12;
    size_t nthreads = omp_get_num_threads();
    nthreads = nthreads > na6 ? na6 : nthreads;

#pragma omp parallel for default(none) num_threads(nthreads) shared(h3, na, na2, na3, na4, na5, na6, zero) reduction(+ : pass)
    for (size_t pqrstu = 0; pqrstu < na6; ++pqrstu) {
        size_t p = pqrstu / na5;
        size_t qrstu = pqrstu % na5;

        size_t q = qrstu / na4;
        size_t rstu = qrstu % na4;

        size_t r = rstu / na3;
        size_t stu = rstu % na3;

        size_t s = stu / na2;
        size_t tu = stu % na2;

        size_t t = tu / na;
        size_t u = tu % na;

        int valid = (q > p) and (r > q) and (t > s) and (u > t);
        auto v_pqrstu = h3[p * na5 + q * na4 + r * na3 + s * na2 + t * na + u];

        auto v_pqrsut = h3[p * na5 + q * na4 + r * na3 + s * na2 + u * na + t];
        auto v_pqrtsu = h3[p * na5 + q * na4 + r * na3 + t * na2 + s * na + u];
        auto v_pqrtus = h3[p * na5 + q * na4 + r * na3 + t * na2 + u * na + s];
        auto v_pqrust = h3[p * na5 + q * na4 + r * na3 + u * na2 + s * na + t];
        auto v_pqruts = h3[p * na5 + q * na4 + r * na3 + u * na2 + t * na + s];

        int value = 1;
        value += (std::fabs(v_pqrstu + v_pqrsut) < zero) + (std::fabs(v_pqrstu + v_pqrtsu) < zero) +
                 (std::fabs(v_pqrstu - v_pqrtus) < zero) + (std::fabs(v_pqrstu - v_pqrust) < zero) +
                 (std::fabs(v_pqrstu + v_pqruts) < zero);

        auto v_prqstu = h3[p * na5 + r * na4 + q * na3 + s * na2 + t * na + u];
        auto v_prqsut = h3[p * na5 + r * na4 + q * na3 + s * na2 + u * na + t];
        auto v_prqtsu = h3[p * na5 + r * na4 + q * na3 + t * na2 + s * na + u];
        auto v_prqtus = h3[p * na5 + r * na4 + q * na3 + t * na2 + u * na + s];
        auto v_prqust = h3[p * na5 + r * na4 + q * na3 + u * na2 + s * na + t];
        auto v_prquts = h3[p * na5 + r * na4 + q * na3 + u * na2 + t * na + s];
        value += (std::fabs(v_pqrstu + v_prqstu) < zero) + (std::fabs(v_pqrstu - v_prqsut) < zero) +
                 (std::fabs(v_pqrstu - v_prqtsu) < zero) + (std::fabs(v_pqrstu + v_prqtus) < zero) +
                 (std::fabs(v_pqrstu + v_prqust) < zero) + (std::fabs(v_pqrstu - v_prquts) < zero);

        auto v_qprstu = h3[q * na5 + p * na4 + r * na3 + s * na2 + t * na + u];
        auto v_qprsut = h3[q * na5 + p * na4 + r * na3 + s * na2 + u * na + t];
        auto v_qprtsu = h3[q * na5 + p * na4 + r * na3 + t * na2 + s * na + u];
        auto v_qprtus = h3[q * na5 + p * na4 + r * na3 + t * na2 + u * na + s];
        auto v_qprust = h3[q * na5 + p * na4 + r * na3 + u * na2 + s * na + t];
        auto v_qpruts = h3[q * na5 + p * na4 + r * na3 + u * na2 + t * na + s];
        value += (std::fabs(v_pqrstu + v_qprstu) < zero) + (std::fabs(v_pqrstu - v_qprsut) < zero) +
                 (std::fabs(v_pqrstu - v_qprtsu) < zero) + (std::fabs(v_pqrstu + v_qprtus) < zero) +
                 (std::fabs(v_pqrstu + v_qprust) < zero) + (std::fabs(v_pqrstu - v_qpruts) < zero);

        auto v_qrpstu = h3[q * na5 + r * na4 + p * na3 + s * na2 + t * na + u];
        auto v_qrpsut = h3[q * na5 + r * na4 + p * na3 + s * na2 + u * na + t];
        auto v_qrptsu = h3[q * na5 + r * na4 + p * na3 + t * na2 + s * na + u];
        auto v_qrptus = h3[q * na5 + r * na4 + p * na3 + t * na2 + u * na + s];
        auto v_qrpust = h3[q * na5 + r * na4 + p * na3 + u * na2 + s * na + t];
        auto v_qrputs = h3[q * na5 + r * na4 + p * na3 + u * na2 + t * na + s];
        value += (std::fabs(v_pqrstu - v_qrpstu) < zero) + (std::fabs(v_pqrstu + v_qrpsut) < zero) +
                 (std::fabs(v_pqrstu + v_qrptsu) < zero) + (std::fabs(v_pqrstu - v_qrptus) < zero) +
                 (std::fabs(v_pqrstu - v_qrpust) < zero) + (std::fabs(v_pqrstu + v_qrputs) < zero);

        auto v_rqpstu = h3[r * na5 + q * na4 + p * na3 + s * na2 + t * na + u];
        auto v_rqpsut = h3[r * na5 + q * na4 + p * na3 + s * na2 + u * na + t];
        auto v_rqptsu = h3[r * na5 + q * na4 + p * na3 + t * na2 + s * na + u];
        auto v_rqptus = h3[r * na5 + q * na4 + p * na3 + t * na2 + u * na + s];
        auto v_rqpust = h3[r * na5 + q * na4 + p * na3 + u * na2 + s * na + t];
        auto v_rqputs = h3[r * na5 + q * na4 + p * na3 + u * na2 + t * na + s];
        value += (std::fabs(v_pqrstu + v_rqpstu) < zero) + (std::fabs(v_pqrstu - v_rqpsut) < zero) +
                 (std::fabs(v_pqrstu - v_rqptsu) < zero) + (std::fabs(v_pqrstu + v_rqptus) < zero) +
                 (std::fabs(v_pqrstu + v_rqpust) < zero) + (std::fabs(v_pqrstu - v_rqputs) < zero);

        auto v_rpqstu = h3[r * na5 + p * na4 + q * na3 + s * na2 + t * na + u];
        auto v_rpqsut = h3[r * na5 + p * na4 + q * na3 + s * na2 + u * na + t];
        auto v_rpqtsu = h3[r * na5 + p * na4 + q * na3 + t * na2 + s * na + u];
        auto v_rpqtus = h3[r * na5 + p * na4 + q * na3 + t * na2 + u * na + s];
        auto v_rpqust = h3[r * na5 + p * na4 + q * na3 + u * na2 + s * na + t];
        auto v_rpquts = h3[r * na5 + p * na4 + q * na3 + u * na2 + t * na + s];
        value += (std::fabs(v_pqrstu - v_rpqstu) < zero) + (std::fabs(v_pqrstu + v_rpqsut) < zero) +
                 (std::fabs(v_pqrstu + v_rpqtsu) < zero) + (std::fabs(v_pqrstu - v_rpqtus) < zero) +
                 (std::fabs(v_pqrstu - v_rpqust) < zero) + (std::fabs(v_pqrstu + v_rpquts) < zero);

        pass += valid * value;
    }

    size_t target = na * (na - 1) * (na - 2) * na * (na - 1) * (na - 2);
    return pass == target;
}

bool SigmaVectorSparseList::is_h3ls_antisymmetric(const std::vector<double>& h3, bool alpha) {
    timer timer_h3ls("Test antisymmetry of low-spin H3");

    auto na = fci_ints_->nmo();
    if (na < 3)
        return true;

    auto na2 = na * na;
    auto na3 = na * na2;
    auto na4 = na * na3;
    auto na5 = na * na4;
    auto na6 = na * na5;

    std::vector<size_t> actv;
    if (alpha)
        actv = {na5, na4, na3, na2, na, 1};
    else
        actv = {na3, na4, na5, 1, na, na2};

    size_t pass = 0;
    double zero = 1.0e-12;
    size_t nthreads = omp_get_num_threads();
    nthreads = nthreads > na6 ? na6 : nthreads;

#pragma omp parallel for default(none) num_threads(nthreads) shared(h3, na, na2, na3, na4, na5, na6, zero, actv) reduction(+ : pass)
    for (size_t pqrstu = 0; pqrstu < na6; ++pqrstu) {
        size_t p = pqrstu / na5;
        size_t qrstu = pqrstu % na5;

        size_t q = qrstu / na4;
        size_t rstu = qrstu % na4;

        size_t r = rstu / na3;
        size_t stu = rstu % na3;

        size_t s = stu / na2;
        size_t tu = stu % na2;

        size_t t = tu / na;
        size_t u = tu % na;

        int valid = (q > p) and (t > s);
        auto v_pqrstu =
            h3[p * actv[0] + q * actv[1] + r * actv[2] + s * actv[3] + t * actv[4] + u * actv[5]];
        auto v_qprstu =
            h3[q * actv[0] + p * actv[1] + r * actv[2] + s * actv[3] + t * actv[4] + u * actv[5]];
        auto v_pqrtsu =
            h3[p * actv[0] + q * actv[1] + r * actv[2] + t * actv[3] + s * actv[4] + u * actv[5]];
        auto v_qprtsu =
            h3[q * actv[0] + p * actv[1] + r * actv[2] + t * actv[3] + s * actv[4] + u * actv[5]];

        int value = (std::fabs(v_pqrstu + v_qprstu) < zero) +
                    (std::fabs(v_pqrstu + v_pqrtsu) < zero) +
                    (std::fabs(v_pqrstu - v_qprtsu) < zero) + 1;
        pass += valid * value;
    }

    size_t target = na * (na - 1) * na * (na - 1) * na * na;
    return pass == target;
}

void SigmaVectorSparseList::add_generalized_sigma3_impl(
    const std::vector<double>& h3, std::shared_ptr<psi::Vector> b, double factor,
    std::vector<double>& sigma,
    const std::vector<std::vector<std::tuple<size_t, short, short, short>>>& sub_lists) {
    auto b_ptr = b->pointer();
    auto na = fci_ints_->nmo();
    auto na2 = na * na;
    auto na3 = na * na2;
    auto na4 = na * na3;
    auto na5 = na * na4;

    // sigma_I <- \sum_{J} c_J \sum_{pqrstu} <I| s^+ t^+ u^+ r q p |J> * h^{pqr}_{stu}
    std::vector<double> sigma_threads;
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t Ksize = sub_lists.size();

#pragma omp single
        sigma_threads.resize(size_ * nthreads);

#pragma omp for schedule(static)
        for (size_t K = 0; K < Ksize; ++K) {
            const auto& cre_dets = sub_lists[K];
            auto cre_dets_size = cre_dets.size();

            size_t I, J;
            int p, s;
            short q, r, t, u;

            for (size_t det1 = 0; det1 < cre_dets_size; ++det1) {
                std::tie(J, p, q, r) = cre_dets[det1];
                auto sign_pqr = p > 0 ? 1 : -1;
                p = std::abs(p) - 1;

                for (size_t det2 = det1; det2 < cre_dets_size; ++det2) {
                    std::tie(I, s, t, u) = cre_dets[det2];
                    auto sign_stu = s > 0 ? 1 : -1;
                    s = std::abs(s) - 1;

                    auto idx = p * na5 + q * na4 + r * na3 + s * na2 + t * na + u;
                    auto HIJ = h3[idx] * sign_pqr * sign_stu * (det1 == det2 ? 0.5 : 1.0);
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

} // namespace forte
