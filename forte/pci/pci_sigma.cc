/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include <cfloat>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "psi4/libmints/vector.h"

#include "integrals/active_space_integrals.h"
#include "pci_sigma.h"

namespace forte {
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

void add(const det_hashvec& A, std::vector<double>& Ca, double beta, const det_hashvec& B,
         const std::vector<double> Cb);

double dot(const det_hashvec& A, const std::vector<double> Ca, const det_hashvec& B,
           const std::vector<double> Cb);

std::vector<double> to_std_vector(psi::SharedVector c) {
    const size_t c_size = c->dim();
    std::vector<double> c_vec(c_size);
    std::memcpy(c_vec.data(), c->pointer(), c_size * sizeof(double));
    return c_vec;
}

void set_psi_Vector(psi::SharedVector c_psi, const std::vector<double>& c_vec) {
    std::memcpy(c_psi->pointer(), c_vec.data(), c_vec.size() * sizeof(double));
}

PCISigmaVector::PCISigmaVector(
    det_hashvec& dets_hashvec, std::vector<double>& ref_C, double spawning_threshold,
    std::shared_ptr<ActiveSpaceIntegrals> as_ints,
    std::function<bool(double, double, double)> prescreen_H_CI,
    std::function<bool(double, double, double, double)> important_H_CI_CJ,
    const std::vector<std::tuple<int, double, std::vector<std::tuple<int, double>>>>& a_couplings,
    const std::vector<std::tuple<int, double, std::vector<std::tuple<int, double>>>>& b_couplings,
    const std::vector<std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>>>&
        aa_couplings,
    const std::vector<std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>>>&
        ab_couplings,
    const std::vector<std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>>>&
        bb_couplings,
    std::unordered_map<Determinant, std::pair<double, double>, Determinant::Hash>&
        dets_max_couplings,
    double dets_single_max_coupling, double dets_double_max_coupling,
    const std::vector<std::pair<det_hashvec, std::vector<double>>>& bad_roots)
    : SigmaVector(dets_hashvec, as_ints, SigmaVectorType::Full, "PCISigmaVector"),
      dets_(dets_hashvec), spawning_threshold_(spawning_threshold), as_ints_(as_ints),
      prescreen_H_CI_(prescreen_H_CI), important_H_CI_CJ_(important_H_CI_CJ),
      dets_max_couplings_(dets_max_couplings), dets_single_max_coupling_(dets_single_max_coupling),
      a_couplings_(a_couplings), b_couplings_(b_couplings), a_couplings_size_(a_couplings.size()),
      b_couplings_size_(b_couplings.size()), dets_double_max_coupling_(dets_double_max_coupling),
      aa_couplings_(aa_couplings), ab_couplings_(ab_couplings), bb_couplings_(bb_couplings),
      aa_couplings_size_(aa_couplings.size()), ab_couplings_size_(ab_couplings.size()),
      bb_couplings_size_(bb_couplings.size()), bad_roots_(bad_roots),
      num_threads_(omp_get_max_threads()) {
    reset(ref_C);
}

void PCISigmaVector::reset(std::vector<double>& ref_C) {
    sigma_build_count_ = 0;
    ref_size_ = ref_C.size();
    orthogonalize(dets_, ref_C, bad_roots_);
    apply_tau_H_symm(spawning_threshold_, dets_, ref_C, first_sigma_vec_, ref_size_);
    ref_C_ = ref_C;
    size_ = dets_.size();

#pragma omp parallel for
    for (size_t I = ref_size_; I < size_; ++I) {
        diag_[I] = as_ints_->energy(dets_[I]);
    }
}

void PCISigmaVector::compute_sigma(psi::SharedVector sigma, psi::SharedVector b) {
    if ((not first_sigma_vec_.empty()) and
        0 == std::memcmp(ref_C_.data(), b->pointer(), ref_size_) and
        std::all_of(b->pointer() + ref_size_, b->pointer() + b->dim(),
                    [](double x) { return x == 0; })) {
        set_psi_Vector(sigma, first_sigma_vec_);
        first_sigma_vec_.clear();
    } else {
        std::vector<double> b_vec = to_std_vector(b);
        orthogonalize(dets_, b_vec, bad_roots_);
        set_psi_Vector(b, b_vec);
        std::vector<double> sigma_vec;
        apply_tau_H_ref_C_symm(spawning_threshold_, dets_, ref_C_, b_vec, sigma_vec, ref_size_);
        set_psi_Vector(sigma, sigma_vec);
    }
    ++sigma_build_count_;
}

void PCISigmaVector::get_diagonal(psi::Vector& diag) {
    std::memcpy(diag.pointer(), diag_.data(), size_);
}

void PCISigmaVector::add_bad_roots(
    std::vector<std::vector<std::pair<size_t, double>>>& bad_states) {}

void PCISigmaVector::compute_sigma_with_diag(psi::SharedVector sigma, psi::SharedVector b) {
    compute_sigma(sigma, b);

#pragma omp parallel for
    for (size_t I = 0; I < size_; ++I) {
        sigma->set(I, sigma->get(I) + diag_[I] * b->get(I));
    }
}

size_t PCISigmaVector::get_num_off_diag() { return num_off_diag_elem_; }

size_t PCISigmaVector::get_sigma_build_count() { return sigma_build_count_; }

void PCISigmaVector::orthogonalize(
    const det_hashvec& space, std::vector<double>& C,
    const std::vector<std::pair<det_hashvec, std::vector<double>>>& solutions) {
    for (size_t n = 0, solution_size = solutions.size(); n < solution_size; ++n) {
        double dot_prod = dot(space, C, solutions[n].first, solutions[n].second);
        add(space, C, -dot_prod, solutions[n].first, solutions[n].second);
    }
}

void PCISigmaVector::apply_tau_H_symm(double spawning_threshold, det_hashvec& ref_dets,
                                      std::vector<double>& ref_C, std::vector<double>& result_C,
                                      size_t& overlap_size) {

    size_t ref_size = ref_dets.size();
    //    result_dets.clear();
    result_C.clear();
    //    det_hashvec result_dets(ref_dets);
    det_hashvec extra_dets;
    std::vector<double> extra_C;
    result_C.resize(ref_size, DBL_MIN);

    std::vector<std::vector<std::pair<Determinant, double>>> thread_det_C_vecs(num_threads_);
    num_off_diag_elem_ = 0;

#pragma omp parallel for
    for (size_t I = 0; I < ref_size; ++I) {
        std::pair<double, double> max_coupling;
        size_t current_rank = omp_get_thread_num();
#pragma omp critical(dets_coupling)
        { max_coupling = dets_max_couplings_[ref_dets[I]]; }
        if (max_coupling.first == 0.0 or max_coupling.second == 0.0) {
            thread_det_C_vecs[current_rank].clear();
            apply_tau_H_symm_det_dynamic_HBCI_2(spawning_threshold, ref_dets, ref_C, I, ref_C[I],
                                                result_C, thread_det_C_vecs[current_rank],
                                                max_coupling);
#pragma omp critical(merge_extra)
            {
                merge(extra_dets, extra_C, thread_det_C_vecs[current_rank],
                      std::function<double(double, double)>(std::plus<double>()), 0.0, false);
            }
#pragma omp critical(dets_coupling)
            { dets_max_couplings_[ref_dets[I]] = max_coupling; }
        } else {
            thread_det_C_vecs[current_rank].clear();
            apply_tau_H_symm_det_dynamic_HBCI_2(spawning_threshold, ref_dets, ref_C, I, ref_C[I],
                                                result_C, thread_det_C_vecs[current_rank],
                                                max_coupling);
#pragma omp critical(merge_extra)
            {
                merge(extra_dets, extra_C, thread_det_C_vecs[current_rank],
                      std::function<double(double, double)>(std::plus<double>()), 0.0, false);
            }
        }
    }

    std::vector<size_t> removing_indices;
    for (size_t I = 0; I < ref_size; ++I) {
        if (result_C[I] == DBL_MIN) {
            removing_indices.push_back(I);
        }
    }
    ref_dets.erase_by_index(removing_indices);
    std::reverse(removing_indices.begin(), removing_indices.end());
    for (size_t I : removing_indices) {
        result_C.erase(result_C.begin() + I);
        ref_C.erase(ref_C.begin() + I);
    }
    overlap_size = ref_dets.size();
    ref_dets.merge(extra_dets);
    result_C.insert(result_C.end(), extra_C.begin(), extra_C.end());

    diag_.resize(ref_dets.size());
#pragma omp parallel for
    for (size_t I = 0; I < overlap_size; ++I) {
        diag_[I] = as_ints_->energy(ref_dets[I]);
        result_C[I] += diag_[I] * ref_C[I];
    }
}

void PCISigmaVector::apply_tau_H_symm_det_dynamic_HBCI_2(
    double spawning_threshold, const det_hashvec& dets_hashvec, const std::vector<double>& pre_C,
    size_t I, double CI, std::vector<double>& result_C,
    std::vector<std::pair<Determinant, double>>& new_det_C_vec,
    std::pair<double, double>& max_coupling) {

    const Determinant& detI = dets_hashvec[I];
    size_t pre_C_size = pre_C.size();

    bool do_singles_1 = max_coupling.first == 0.0 and
                        std::fabs(dets_single_max_coupling_ * CI) >= spawning_threshold;
    bool do_singles = std::fabs(max_coupling.first * CI) >= spawning_threshold;
    bool do_doubles_1 = max_coupling.second == 0.0 and
                        std::fabs(dets_double_max_coupling_ * CI) >= spawning_threshold;
    bool do_doubles = std::fabs(max_coupling.second * CI) >= spawning_threshold;

    // Diagonal contributions
    // parallel_timer_on("PCI:diagonal", omp_get_thread_num());
    bool diagonal_flag = false;
    double diagonal_contribution = 0.0;
    // parallel_timer_off("PCI:diagonal", omp_get_thread_num());

    Determinant detJ(detI);
    if (do_singles) {
        // parallel_timer_on("PCI:singles", omp_get_thread_num());
        // Generate alpha excitations
        for (size_t x = 0; x < a_couplings_size_; ++x) {
            double HJI_max = std::get<1>(a_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(a_couplings_[x]);
            if (detI.get_alfa_bit(i)) {
                const std::vector<std::tuple<int, double>>& sub_couplings =
                    std::get<2>(a_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI_bound, CI, spawning_threshold)) {
                        break;
                    }
                    if (!detI.get_alfa_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_alpha_abs(detJ, i, a);

                        if (prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                            HJI *= detJ.single_excitation_a(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index > I) {
                                if (index >= pre_C_size) {
                                    if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                        new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                        diagonal_flag = true;
#pragma omp atomic
                                        num_off_diag_elem_ += 2;
                                    }
                                } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                              spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_flag = true;
                                    diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            }

                            detJ.set_alfa_bit(i, true);
                            detJ.set_alfa_bit(a, false);
                        }
                    }
                }
            }
        }
        // Generate beta excitations
        for (size_t x = 0; x < b_couplings_size_; ++x) {
            double HJI_max = std::get<1>(b_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(b_couplings_[x]);
            if (detI.get_beta_bit(i)) {
                const std::vector<std::tuple<int, double>>& sub_couplings =
                    std::get<2>(b_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI_bound, CI, spawning_threshold)) {
                        break;
                    }
                    if (!detI.get_beta_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_beta_abs(detJ, i, a);
                        if (prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                            HJI *= detJ.single_excitation_b(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index > I) {
                                if (index >= pre_C_size) {
                                    if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                        new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                        diagonal_flag = true;
#pragma omp atomic
                                        num_off_diag_elem_ += 2;
                                    }
                                } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                              spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_flag = true;
                                    diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            }

                            detJ.set_beta_bit(i, true);
                            detJ.set_beta_bit(a, false);
                        }
                    }
                }
            }
        }
        // parallel_timer_off("PCI:singles", omp_get_thread_num());
    } else if (do_singles_1) {
        // parallel_timer_on("PCI:singles", omp_get_thread_num());
        // Generate alpha excitations
        for (size_t x = 0; x < a_couplings_size_; ++x) {
            double HJI_max = std::get<1>(a_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(a_couplings_[x]);
            if (detI.get_alfa_bit(i)) {
                const std::vector<std::tuple<int, double>>& sub_couplings =
                    std::get<2>(a_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI_bound, CI, spawning_threshold)) {
                        break;
                    }
                    if (!detI.get_alfa_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_alpha_abs(detJ, i, a);

                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                            HJI *= detJ.single_excitation_a(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index > I) {
                                if (index >= pre_C_size) {
                                    if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                        new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                        diagonal_flag = true;
#pragma omp atomic
                                        num_off_diag_elem_ += 2;
                                    }
                                } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                              spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_flag = true;
                                    diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            }

                            detJ.set_alfa_bit(i, true);
                            detJ.set_alfa_bit(a, false);
                        }
                    }
                }
            }
        }
        // Generate beta excitations
        for (size_t x = 0; x < b_couplings_size_; ++x) {
            double HJI_max = std::get<1>(b_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(b_couplings_[x]);
            if (detI.get_beta_bit(i)) {
                const std::vector<std::tuple<int, double>>& sub_couplings =
                    std::get<2>(b_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI_bound, CI, spawning_threshold)) {
                        break;
                    }
                    if (!detI.get_beta_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_beta_abs(detJ, i, a);

                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                            HJI *= detJ.single_excitation_b(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index > I) {
                                if (index >= pre_C_size) {
                                    if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                        new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                        diagonal_flag = true;
#pragma omp atomic
                                        num_off_diag_elem_ += 2;
                                    }
                                } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                              spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_flag = true;
                                    diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            }

                            detJ.set_beta_bit(i, true);
                            detJ.set_beta_bit(a, false);
                        }
                    }
                }
            }
        }
        // parallel_timer_off("PCI:singles", omp_get_thread_num());
    }

    if (do_doubles) {
        // parallel_timer_on("PCI:doubles", omp_get_thread_num());
        // Generate alpha-alpha excitations
        for (size_t x = 0; x < aa_couplings_size_; ++x) {
            double HJI_max = std::get<2>(aa_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(aa_couplings_[x]);
            int j = std::get<1>(aa_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_alfa_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(aa_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_alfa_bit(b))) {
                        //                        Determinant detJ(detI);
                        HJI *= detJ.double_excitation_aa(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index >= pre_C_size) {
                                if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                    new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                    diagonal_flag = true;
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                          spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_flag = true;
                                diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            }
                        }

                        detJ.set_alfa_bit(i, true);
                        detJ.set_alfa_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_alfa_bit(b, false);
                    }
                }
            }
        }
        // Generate alpha-beta excitations
        for (size_t x = 0; x < ab_couplings_size_; ++x) {
            double HJI_max = std::get<2>(ab_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(ab_couplings_[x]);
            int j = std::get<1>(ab_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_beta_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(ab_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_beta_bit(b))) {
                        //                        Determinant detJ(detI);
                        HJI *= detJ.double_excitation_ab(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index >= pre_C_size) {
                                if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                    new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                    diagonal_flag = true;
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                          spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_flag = true;
                                diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            }
                        }

                        detJ.set_alfa_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // Generate beta-beta excitations
        for (size_t x = 0; x < bb_couplings_size_; ++x) {
            double HJI_max = std::get<2>(bb_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(bb_couplings_[x]);
            int j = std::get<1>(bb_couplings_[x]);
            if (detI.get_beta_bit(i) and detI.get_beta_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(bb_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_beta_bit(a) or detI.get_beta_bit(b))) {
                        //                        Determinant detJ(detI);
                        HJI *= detJ.double_excitation_bb(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index >= pre_C_size) {
                                if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                    new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                    diagonal_flag = true;
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                          spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_flag = true;
                                diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            }
                        }

                        detJ.set_beta_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_beta_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // parallel_timer_off("PCI:doubles", omp_get_thread_num());
    } else if (do_doubles_1) {
        // parallel_timer_on("PCI:doubles", omp_get_thread_num());
        // Generate alpha-alpha excitations
        for (size_t x = 0; x < aa_couplings_size_; ++x) {
            double HJI_max = std::get<2>(aa_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(aa_couplings_[x]);
            int j = std::get<1>(aa_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_alfa_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(aa_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_alfa_bit(b))) {
                        max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                        //                        Determinant detJ(detI);
                        HJI *= detJ.double_excitation_aa(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index >= pre_C_size) {
                                if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                    new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                    diagonal_flag = true;
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                          spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_flag = true;
                                diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            }
                        }

                        detJ.set_alfa_bit(i, true);
                        detJ.set_alfa_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_alfa_bit(b, false);
                    }
                }
            }
        }
        // Generate alpha-beta excitations
        for (size_t x = 0; x < ab_couplings_size_; ++x) {
            double HJI_max = std::get<2>(ab_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(ab_couplings_[x]);
            int j = std::get<1>(ab_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_beta_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(ab_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_beta_bit(b))) {
                        max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                        //                        Determinant detJ(detI);
                        HJI *= detJ.double_excitation_ab(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index >= pre_C_size) {
                                if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                    new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                    diagonal_flag = true;
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                          spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_flag = true;
                                diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            }
                        }

                        detJ.set_alfa_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // Generate beta-beta excitations
        for (size_t x = 0; x < bb_couplings_size_; ++x) {
            double HJI_max = std::get<2>(bb_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(bb_couplings_[x]);
            int j = std::get<1>(bb_couplings_[x]);
            if (detI.get_beta_bit(i) and detI.get_beta_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(bb_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_beta_bit(a) or detI.get_beta_bit(b))) {
                        max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                        //                        Determinant detJ(detI);
                        HJI *= detJ.double_excitation_bb(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index >= pre_C_size) {
                                if (important_H_CI_CJ_(HJI, CI, 0.0, spawning_threshold)) {
                                    new_det_C_vec.push_back(std::make_pair(detJ, HJI * CI));
                                    diagonal_flag = true;
#pragma omp atomic
                                    num_off_diag_elem_ += 2;
                                }
                            } else if (important_H_CI_CJ_(HJI, CI, pre_C[index],
                                                          spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_flag = true;
                                diagonal_contribution += HJI * pre_C[index];
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            }
                        }

                        detJ.set_beta_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_beta_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // parallel_timer_off("PCI:doubles", omp_get_thread_num());
    }
    if (diagonal_flag) {
        if (std::fabs(diagonal_contribution) > DBL_MIN) {
#pragma omp atomic
            result_C[I] += diagonal_contribution;
        } else {
#pragma omp atomic
            result_C[I] -= DBL_MIN;
        }
    }
}

void PCISigmaVector::apply_tau_H_ref_C_symm(
    double spawning_threshold, const det_hashvec& result_dets, const std::vector<double>& ref_C,
    const std::vector<double>& pre_C, std::vector<double>& result_C, const size_t overlap_size) {

    const size_t result_size = result_dets.size();
    result_C.clear();
    result_C.resize(result_size, 0.0);

#pragma omp parallel for
    for (size_t I = 0; I < overlap_size; ++I) {
        std::pair<double, double> max_coupling;
        max_coupling = dets_max_couplings_[result_dets[I]];
        apply_tau_H_ref_C_symm_det_dynamic_HBCI_2(spawning_threshold, result_dets, pre_C, ref_C, I,
                                                  pre_C[I], ref_C[I], overlap_size, result_C,
                                                  max_coupling);
    }

#pragma omp parallel for
    for (size_t I = 0; I < result_size; ++I) {
        result_C[I] += diag_[I] * pre_C[I];
    }
}

void PCISigmaVector::apply_tau_H_ref_C_symm_det_dynamic_HBCI_2(
    double spawning_threshold, const det_hashvec& dets_hashvec, const std::vector<double>& pre_C,
    const std::vector<double>& ref_C, size_t I, double CI, double ref_CI, const size_t overlap_size,
    std::vector<double>& result_C, const std::pair<double, double>& max_coupling) {

    const Determinant& detI = dets_hashvec[I];

    bool do_singles = std::fabs(max_coupling.first * ref_CI) >= spawning_threshold;
    bool do_doubles = std::fabs(max_coupling.second * ref_CI) >= spawning_threshold;

    // Diagonal contributions
    // parallel_timer_on("PCI:diagonal", omp_get_thread_num());
    double diagonal_contribution = 0.0;
    // parallel_timer_off("PCI:diagonal", omp_get_thread_num());

    Determinant detJ(detI);
    if (do_singles) {
        // parallel_timer_on("PCI:singles", omp_get_thread_num());
        // Generate alpha excitations
        for (size_t x = 0; x < a_couplings_size_; ++x) {
            double HJI_max = std::get<1>(a_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(a_couplings_[x]);
            if (detI.get_alfa_bit(i)) {
                const std::vector<std::tuple<int, double>>& sub_couplings =
                    std::get<2>(a_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI_bound, ref_CI, spawning_threshold)) {
                        break;
                    }
                    if (!detI.get_alfa_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_alpha_abs(detJ, i, a);

                        if (prescreen_H_CI_(HJI, ref_CI, spawning_threshold)) {
                            HJI *= detJ.single_excitation_a(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index > I) {
                                if (index < overlap_size) {
                                    if (important_H_CI_CJ_(HJI, ref_CI, ref_C[index],
                                                           spawning_threshold)) {
#pragma omp atomic
                                        result_C[index] += HJI * CI;
                                        diagonal_contribution += HJI * pre_C[index];
                                    }
                                } else if (important_H_CI_CJ_(HJI, ref_CI, 0.0,
                                                              spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_contribution += HJI * pre_C[index];
                                }
                            }
                            detJ.set_alfa_bit(i, true);
                            detJ.set_alfa_bit(a, false);
                        }
                    }
                }
            }
        }
        // Generate beta excitations
        for (size_t x = 0; x < b_couplings_size_; ++x) {
            double HJI_max = std::get<1>(b_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(b_couplings_[x]);
            if (detI.get_beta_bit(i)) {
                const std::vector<std::tuple<int, double>>& sub_couplings =
                    std::get<2>(b_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI_bound, ref_CI, spawning_threshold)) {
                        break;
                    }
                    if (!detI.get_beta_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_beta_abs(detJ, i, a);

                        if (prescreen_H_CI_(HJI, ref_CI, spawning_threshold)) {
                            HJI *= detJ.single_excitation_b(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index > I) {
                                if (index < overlap_size) {
                                    if (important_H_CI_CJ_(HJI, ref_CI, ref_C[index],
                                                           spawning_threshold)) {
#pragma omp atomic
                                        result_C[index] += HJI * CI;
                                        diagonal_contribution += HJI * pre_C[index];
                                    }
                                } else if (important_H_CI_CJ_(HJI, ref_CI, 0.0,
                                                              spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_contribution += HJI * pre_C[index];
                                }
                            }
                            detJ.set_beta_bit(i, true);
                            detJ.set_beta_bit(a, false);
                        }
                    }
                }
            }
        }
        // parallel_timer_off("PCI:singles", omp_get_thread_num());
    }

    if (do_doubles) {
        // parallel_timer_on("PCI:doubles", omp_get_thread_num());
        // Generate alpha-alpha excitations
        for (size_t x = 0; x < aa_couplings_size_; ++x) {
            double HJI_max = std::get<2>(aa_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(aa_couplings_[x]);
            int j = std::get<1>(aa_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_alfa_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(aa_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, ref_CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_alfa_bit(b))) {
                        HJI *= detJ.double_excitation_aa(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index < overlap_size) {
                                if (important_H_CI_CJ_(HJI, ref_CI, ref_C[index],
                                                       spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_contribution += HJI * pre_C[index];
                                }
                            } else if (important_H_CI_CJ_(HJI, ref_CI, 0.0, spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_contribution += HJI * pre_C[index];
                            }
                        }
                        detJ.set_alfa_bit(i, true);
                        detJ.set_alfa_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_alfa_bit(b, false);
                    }
                }
            }
        }
        // Generate alpha-beta excitations
        for (size_t x = 0; x < ab_couplings_size_; ++x) {
            double HJI_max = std::get<2>(ab_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(ab_couplings_[x]);
            int j = std::get<1>(ab_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_beta_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(ab_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, ref_CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_beta_bit(b))) {
                        HJI *= detJ.double_excitation_ab(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index < overlap_size) {
                                if (important_H_CI_CJ_(HJI, ref_CI, ref_C[index],
                                                       spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_contribution += HJI * pre_C[index];
                                }
                            } else if (important_H_CI_CJ_(HJI, ref_CI, 0.0, spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_contribution += HJI * pre_C[index];
                            }
                        }
                        detJ.set_alfa_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // Generate beta-beta excitations
        for (size_t x = 0; x < bb_couplings_size_; ++x) {
            double HJI_max = std::get<2>(bb_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(bb_couplings_[x]);
            int j = std::get<1>(bb_couplings_[x]);
            if (detI.get_beta_bit(i) and detI.get_beta_bit(j)) {
                const std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(bb_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (!prescreen_H_CI_(HJI, ref_CI, spawning_threshold)) {
                        break;
                    }
                    if (!(detI.get_beta_bit(a) or detI.get_beta_bit(b))) {
                        HJI *= detJ.double_excitation_bb(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index > I) {
                            if (index < overlap_size) {
                                if (important_H_CI_CJ_(HJI, ref_CI, ref_C[index],
                                                       spawning_threshold)) {
#pragma omp atomic
                                    result_C[index] += HJI * CI;
                                    diagonal_contribution += HJI * pre_C[index];
                                }
                            } else if (important_H_CI_CJ_(HJI, ref_CI, 0.0, spawning_threshold)) {
#pragma omp atomic
                                result_C[index] += HJI * CI;
                                diagonal_contribution += HJI * pre_C[index];
                            }
                        }
                        detJ.set_beta_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_beta_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // parallel_timer_off("PCI:doubles", omp_get_thread_num());
    }
#pragma omp atomic
    result_C[I] += diagonal_contribution;
}
} // namespace forte
