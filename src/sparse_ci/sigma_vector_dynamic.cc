/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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

#include <cmath>
#include <thread>
#include <future>

#include "psi4/libciomr/libciomr.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "../forte-def.h"
#include "../iterative_solvers.h"
#include "sigma_vector_dynamic.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif


namespace psi {
namespace forte {

#define SIGMA_VEC_DEBUG 0

#if SIGMA_VEC_DEBUG
size_t count_aa_total = 0;
size_t count_bb_total = 0;
size_t count_aaaa_total = 0;
size_t count_abab_total = 0;
size_t count_bbbb_total = 0;

size_t count_aa = 0;
size_t count_bb = 0;
size_t count_aaaa = 0;
size_t count_abab = 0;
size_t count_bbbb = 0;
#endif

void print_SigmaVectorDynamic_stats();

SigmaVectorDynamic::SigmaVectorDynamic(const DeterminantHashVec& space,
                                       std::shared_ptr<FCIIntegrals> fci_ints, size_t max_memory)
    : SigmaVector(space.size()), space_(space), fci_ints_(fci_ints),
      a_sorted_string_list_(space, fci_ints, DetSpinType::Alpha),
      b_sorted_string_list_(space, fci_ints, DetSpinType::Beta) {

    timer this_timer("creator");

    nmo_ = fci_ints_->nmo();

    for (size_t I = 0; I < size_; ++I) {
        const Determinant& detI = space.get_det(I);
        double EI = fci_ints_->energy(detI);
        diag_.push_back(EI);
    }
    temp_sigma_.resize(size_);
    temp_b_.resize(size_);

#pragma omp parallel 
{
    num_threads_ = omp_get_max_threads();
}

    total_space_ = max_memory;
    size_t space_per_thread = total_space_ / num_threads_;
    for (int t = 0; t < num_threads_; ++t) {
        H_IJ_list_thread_limit_.push_back((t + 1) * space_per_thread);

        H_IJ_aa_list_thread_start_.push_back(t * space_per_thread);
        H_IJ_aa_list_thread_end_.push_back(t * space_per_thread);

        H_IJ_bb_list_thread_start_.push_back(t * space_per_thread);
        H_IJ_bb_list_thread_end_.push_back(t * space_per_thread);

        H_IJ_abab_list_thread_start_.push_back(t * space_per_thread);
        H_IJ_abab_list_thread_end_.push_back(t * space_per_thread);

        first_aa_onthefly_group_.push_back(t);
        first_bb_onthefly_group_.push_back(t);
        first_abab_onthefly_group_.push_back(t);
    }
    H_IJ_list_.resize(total_space_);
    outfile->Printf("\n\n SigmaVectorDynamic:");
    outfile->Printf("\n Maximum memory   : %zu double", total_space_);
    outfile->Printf("\n Number of threads: %d\n", num_threads_);
}

SigmaVectorDynamic::~SigmaVectorDynamic() { print_SigmaVectorDynamic_stats(); }

void SigmaVectorDynamic::compute_sigma(SharedVector sigma, SharedVector b) {
    sigma->zero();
    compute_sigma_scalar(sigma, b);
    compute_sigma_aa(sigma, b);
    compute_sigma_bb(sigma, b);
    compute_sigma_abab(sigma, b);

    if (num_builds_ == 0) {
        print_thread_stats();
    }
    num_builds_ += 1;
}

void print_SigmaVectorDynamic_stats() {
#if SIGMA_VEC_DEBUG
    outfile->Printf("\n  Summary of SigmaVectorDynamic:");
    outfile->Printf("\n    aa   : %12zu / %12zu = %f", count_aa, count_aa_total,
                    double(count_aa) / double(count_aa_total));
    outfile->Printf("\n    bb   : %12zu / %12zu = %f", count_bb, count_bb_total,
                    double(count_bb) / double(count_bb_total));
    outfile->Printf("\n    aaaa : %12zu / %12zu = %f", count_aaaa, count_aa_total,
                    double(count_aaaa) / double(count_aa_total));
    outfile->Printf("\n    abab : %12zu / %12zu = %f", count_abab, count_abab_total,
                    double(count_abab) / double(count_abab_total));
    outfile->Printf("\n    bbbb : %12zu / %12zu = %f", count_bbbb, count_bb_total,
                    double(count_bbbb) / double(count_bb_total));
    outfile->Printf("\n");
#endif
}

void SigmaVectorDynamic::print_thread_stats() {
#if SIGMA_VEC_DEBUG
    outfile->Printf("\n  SigmaVectorDynamic Threads statistics:");
    outfile->Printf("\n  b-b coupling:");
    outfile->Printf("\n Thread     start          end        limit         size        first");
    for (int t = 0; t < num_threads_; ++t) {
        outfile->Printf("\n %3d %12zu %12zu %12zu %12zu %12zu", t, H_IJ_aa_list_thread_start_[t],
                        H_IJ_aa_list_thread_end_[t], H_IJ_list_thread_limit_[t],
                        H_IJ_aa_list_thread_end_[t] - H_IJ_aa_list_thread_start_[t],
                        first_aa_onthefly_group_[t]);
    }
    for (int t = 0; t < num_threads_; ++t) {
        outfile->Printf("\n %3d %12zu %12zu %12zu %12zu %12zu", t, H_IJ_bb_list_thread_start_[t],
                        H_IJ_bb_list_thread_end_[t], H_IJ_list_thread_limit_[t],
                        H_IJ_bb_list_thread_end_[t] - H_IJ_bb_list_thread_start_[t],
                        first_bb_onthefly_group_[t]);
    }
    for (int t = 0; t < num_threads_; ++t) {
        outfile->Printf("\n %3d %12zu %12zu %12zu %12zu %12zu", t, H_IJ_abab_list_thread_start_[t],
                        H_IJ_abab_list_thread_end_[t], H_IJ_list_thread_limit_[t],
                        H_IJ_abab_list_thread_end_[t] - H_IJ_abab_list_thread_start_[t],
                        first_abab_onthefly_group_[t]);
    }
#endif
}

void SigmaVectorDynamic::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
}

void SigmaVectorDynamic::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorDynamic::compute_sigma_scalar(SharedVector sigma, SharedVector b) {
    timer energy_timer("scalar");

    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    for (size_t I = 0; I < size_; ++I) {
        double b_I = b_p[I];
        sigma_p[I] += diag_[I] * b_I;
    }
}

void SigmaVectorDynamic::compute_sigma_aa(SharedVector sigma, SharedVector b) {
    timer energy_timer("sigma_aa");
    std::fill(temp_sigma_.begin(), temp_sigma_.end(), 0.0);
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = b_sorted_string_list_.add(I);
        temp_b_[I] = b->get(addI);
    }
    // launch asynchronous tasks
    std::vector<std::future<void>> tasks;
    for (int task_id = 0; task_id < num_threads_; ++task_id) {
        if ((mode_ == SigmaVectorMode::Dynamic) and ((num_builds_ == 0))) {
            // If running in dynamic mode, store Hamiltonian on first build
            tasks.push_back(std::async(std::launch::async, &SigmaVectorDynamic::sigma_aa_store_task,
                                       this, task_id, num_threads_));
        } else {
            tasks.push_back(std::async(std::launch::async,
                                       &SigmaVectorDynamic::sigma_aa_dynamic_task, this, task_id,
                                       num_threads_));
        }
    }
    // collect results
    for (auto& task : tasks) {
        task.get();
    }
    // Add sigma using the determinant address used in the DeterminantHashVector object
    double* sigma_p = sigma->pointer();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = b_sorted_string_list_.add(I);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDynamic::sigma_aa_store_task(size_t task_id, size_t num_tasks) {
    // loop over all determinants
    const auto& sorted_half_dets = b_sorted_string_list_.sorted_half_dets();
    size_t num_half_dets = sorted_half_dets.size();
    // a temp array that can fit one row of the Hamiltonian
    bool store = true;
    for (size_t group = task_id; group < num_half_dets; group += num_tasks) {
        const auto& Ib = sorted_half_dets[group];
        if (store) {
            store = compute_aa_coupling_and_store(Ib, temp_b_, task_id);
            if (store) {
                first_aa_onthefly_group_[task_id] = group + num_tasks;
            }
        } else {
            compute_aa_coupling(Ib, temp_b_);
        }
    }
    // update thread limits for next group of excitations
    H_IJ_bb_list_thread_start_[task_id] = H_IJ_aa_list_thread_end_[task_id];
    H_IJ_bb_list_thread_end_[task_id] = H_IJ_aa_list_thread_end_[task_id];
}

void SigmaVectorDynamic::sigma_aa_dynamic_task(size_t task_id, size_t num_tasks) {
    // loop over all determinants
    const auto& sorted_half_dets = b_sorted_string_list_.sorted_half_dets();
    size_t num_half_dets = sorted_half_dets.size();

    // compute contributions from elements stored in memory
    double H_IJ;
    size_t posI, posJ;
    size_t begin_el = H_IJ_aa_list_thread_start_[task_id];
    size_t end_el = H_IJ_aa_list_thread_end_[task_id];
    for (size_t el = begin_el; el < end_el; ++el) {
        std::tie(H_IJ, posI, posJ) = H_IJ_list_[el];
        temp_sigma_[posI] += H_IJ * temp_b_[posJ];
        temp_sigma_[posJ] += H_IJ * temp_b_[posI];
    }

    // compute contributions on-the-fly
    size_t first_group = first_aa_onthefly_group_[task_id];
    for (size_t group = first_group; group < num_half_dets; group += num_tasks) {
        const auto& Ib = sorted_half_dets[group];
        compute_aa_coupling(Ib, temp_b_);
    }
}

void SigmaVectorDynamic::compute_sigma_bb(SharedVector sigma, SharedVector b) {
    timer energy_timer("sigma_bb");
    std::fill(temp_sigma_.begin(), temp_sigma_.end(), 0.0);
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = a_sorted_string_list_.add(I);
        temp_b_[I] = b->get(addI);
    }
    // launch asynchronous tasks
    std::vector<std::future<void>> tasks;
    for (int task_id = 0; task_id < num_threads_; ++task_id) {
        if ((mode_ == SigmaVectorMode::Dynamic) and ((num_builds_ == 0))) {
            // If running in dynamic mode, store Hamiltonian on first build
            tasks.push_back(std::async(std::launch::async, &SigmaVectorDynamic::sigma_bb_store_task,
                                       this, task_id, num_threads_));
        } else {
            tasks.push_back(std::async(std::launch::async,
                                       &SigmaVectorDynamic::sigma_bb_dynamic_task, this, task_id,
                                       num_threads_));
        }
    }
    // collect results
    for (auto& task : tasks) {
        task.get();
    }
    // Add sigma using the determinant address used in the DeterminantHashVector object
    double* sigma_p = sigma->pointer();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = a_sorted_string_list_.add(I);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDynamic::sigma_bb_store_task(size_t task_id, size_t num_tasks) {
    // loop over all determinants
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    size_t num_half_dets = sorted_half_dets.size();
    bool store = true;
    for (size_t group = task_id; group < num_half_dets; group += num_tasks) {
        const auto& Ia = sorted_half_dets[group];
        if (store) {
            store = compute_bb_coupling_and_store(Ia, temp_b_, task_id);
            if (store) {
                first_bb_onthefly_group_[task_id] = group + num_tasks;
            }
        } else {
            compute_bb_coupling(Ia, temp_b_);
        }
    }
    // update thread limits for next group of excitations
    H_IJ_abab_list_thread_start_[task_id] = H_IJ_bb_list_thread_end_[task_id];
    H_IJ_abab_list_thread_end_[task_id] = H_IJ_bb_list_thread_end_[task_id];
}

void SigmaVectorDynamic::sigma_bb_dynamic_task(size_t task_id, size_t num_tasks) {
    // loop over all determinants
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    size_t num_half_dets = sorted_half_dets.size();

    // compute contributions from elements stored in memory
    double H_IJ;
    size_t posI, posJ;
    size_t begin_el = H_IJ_bb_list_thread_start_[task_id];
    size_t end_el = H_IJ_bb_list_thread_end_[task_id];
    for (size_t el = begin_el; el < end_el; ++el) {
        std::tie(H_IJ, posI, posJ) = H_IJ_list_[el];
        temp_sigma_[posI] += H_IJ * temp_b_[posJ];
        temp_sigma_[posJ] += H_IJ * temp_b_[posI];
    }

    // compute contributions on-the-fly
    size_t first_group = first_bb_onthefly_group_[task_id];
    for (size_t group = first_group; group < num_half_dets; group += num_tasks) {
        const auto& Ia = sorted_half_dets[group];
        compute_bb_coupling(Ia, temp_b_);
    }
}

void SigmaVectorDynamic::compute_sigma_abab(SharedVector sigma, SharedVector b) {
    timer energy_timer("sigma_abab");
    std::fill(temp_sigma_.begin(), temp_sigma_.end(), 0.0);
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = a_sorted_string_list_.add(I);
        temp_b_[I] = b->get(addI);
    }
    // launch asynchronous tasks
    std::vector<std::future<void>> tasks;
    for (int task_id = 0; task_id < num_threads_; ++task_id) {
        if ((mode_ == SigmaVectorMode::Dynamic) and ((num_builds_ == 0))) {
            // If running in dynamic mode, store Hamiltonian on first build
            tasks.push_back(std::async(std::launch::async,
                                       &SigmaVectorDynamic::sigma_abab_store_task, this, task_id,
                                       num_threads_));
        } else {
            tasks.push_back(std::async(std::launch::async,
                                       &SigmaVectorDynamic::sigma_abab_dynamic_task, this, task_id,
                                       num_threads_));
        }
    }
    // collect results
    for (auto& task : tasks) {
        task.get();
    }
    // Add sigma using the determinant address used in the DeterminantHashVector object
    double* sigma_p = sigma->pointer();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = a_sorted_string_list_.add(I);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDynamic::sigma_abab_store_task(size_t task_id, size_t num_tasks) {
    // loop over all determinants
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    size_t num_half_dets = sorted_half_dets.size();
    bool store = true;
    for (size_t group = task_id; group < num_half_dets; group += num_tasks) {
        const auto& detIa = sorted_half_dets[group];
        if (store) {
            store = compute_abab_coupling_and_store(detIa, temp_b_, task_id);
            if (store) {
                first_abab_onthefly_group_[task_id] = group + num_tasks;
            }
        } else {
            compute_abab_coupling(detIa, temp_b_, task_id);
        }
    }
}

void SigmaVectorDynamic::sigma_abab_dynamic_task(size_t task_id, size_t num_tasks) {
    // compute contributions from elements stored in memory
    double H_IJ;
    size_t posI, posJ;
    size_t begin_el = H_IJ_abab_list_thread_start_[task_id];
    size_t end_el = H_IJ_abab_list_thread_end_[task_id];
    for (size_t el = begin_el; el < end_el; ++el) {
        std::tie(H_IJ, posI, posJ) = H_IJ_list_[el];
        temp_sigma_[posI] += H_IJ * temp_b_[posJ];
    }

    // compute contributions on-the-fly
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    size_t num_half_dets = sorted_half_dets.size();
    size_t first_group = first_abab_onthefly_group_[task_id];
    for (size_t group = first_group; group < num_half_dets; group += num_tasks) {
        const auto& detIa = sorted_half_dets[group];
        compute_abab_coupling(detIa, temp_b_, task_id);
    }
}

bool SigmaVectorDynamic::compute_aa_coupling_and_store(const UI64Determinant::bit_t& Ib,
                                                       const std::vector<double>& b,
                                                       size_t task_id) {
    bool stored = true;
    size_t end = H_IJ_aa_list_thread_end_[task_id];
    size_t limit = H_IJ_list_thread_limit_[task_id];

    const auto& sorted_dets = b_sorted_string_list_.sorted_dets();
    const auto& range_I = b_sorted_string_list_.range(Ib);
    UI64Determinant::bit_t Ia;
    UI64Determinant::bit_t Ja;
    UI64Determinant::bit_t IJa;
    size_t first_I = range_I.first;
    size_t last_I = range_I.second;
    double sigma_I = 0.0;
    size_t num_elements = 0;
    for (size_t posI = first_I; posI < last_I; ++posI) {
        sigma_I = 0.0;
        Ia = sorted_dets[posI].get_alfa_bits();
        for (size_t posJ = posI + 1; posJ < last_I; ++posJ) {
            Ja = sorted_dets[posJ].get_alfa_bits();
#if SIGMA_VEC_DEBUG
            count_aa_total++;
#endif
            // find common bits
            IJa = Ja ^ Ia;
            int ndiff = ui64_bit_count(IJa);
            if (ndiff == 2) {
                double H_IJ = slater_rules_single_alpha(Ib, Ia, Ja, fci_ints_);
                sigma_I += H_IJ * b[posJ];
                temp_sigma_[posJ] += H_IJ * b[posI];
                // Add this to the Hamiltonian
                if (end + num_elements < limit) {
                    if (std::fabs(H_IJ) > H_threshold_) {
                        H_IJ_list_[end + num_elements] = std::tie(H_IJ, posI, posJ);
                        num_elements++;
                    }
                } else {
                    stored = false;
                }
#if SIGMA_VEC_DEBUG
                count_aa++;
#endif
            } else if (ndiff == 4) {
                double H_IJ = slater_rules_double_alpha_alpha(Ia, Ja, fci_ints_);
                sigma_I += H_IJ * b[posJ];
                temp_sigma_[posJ] += H_IJ * b[posI];
                // Add this to the Hamiltonian
                if (end + num_elements < limit) {
                    if (std::fabs(H_IJ) > H_threshold_) {
                        H_IJ_list_[end + num_elements] = std::tie(H_IJ, posI, posJ);
                        num_elements++;
                    }
                } else {
                    stored = false;
                }
#if SIGMA_VEC_DEBUG
                count_aaaa++;
#endif
            }
        }
        temp_sigma_[posI] += sigma_I;
    }
    if (stored) {
        H_IJ_aa_list_thread_end_[task_id] += num_elements;
    }
    return stored;
}

void SigmaVectorDynamic::compute_aa_coupling(const UI64Determinant::bit_t& Ib,
                                             const std::vector<double>& b) {
    const auto& sorted_dets = b_sorted_string_list_.sorted_dets();
    const auto& range_I = b_sorted_string_list_.range(Ib);
    UI64Determinant::bit_t Ia;
    UI64Determinant::bit_t Ja;
    UI64Determinant::bit_t IJa;
    size_t first_I = range_I.first;
    size_t last_I = range_I.second;
    double sigma_I = 0.0;
    for (size_t posI = first_I; posI < last_I; ++posI) {
        sigma_I = 0.0;
        Ia = sorted_dets[posI].get_alfa_bits();
        for (size_t posJ = posI + 1; posJ < last_I; ++posJ) {
            Ja = sorted_dets[posJ].get_alfa_bits();
#if SIGMA_VEC_DEBUG
            count_aa_total++;
#endif
            // find common bits
            IJa = Ja ^ Ia;
            int ndiff = ui64_bit_count(IJa);
            if (ndiff == 2) {
                double H_IJ = slater_rules_single_alpha(Ib, Ia, Ja, fci_ints_);
                sigma_I += H_IJ * b[posJ];
                temp_sigma_[posJ] += H_IJ * b[posI];
#if SIGMA_VEC_DEBUG
                count_aa++;
#endif
            } else if (ndiff == 4) {
                double H_IJ = slater_rules_double_alpha_alpha(Ia, Ja, fci_ints_);
                sigma_I += H_IJ * b[posJ];
                temp_sigma_[posJ] += H_IJ * b[posI];
#if SIGMA_VEC_DEBUG
                count_aaaa++;
#endif
            }
        }
        temp_sigma_[posI] += sigma_I;
    }
}

bool SigmaVectorDynamic::compute_bb_coupling_and_store(const UI64Determinant::bit_t& Ia,
                                                       const std::vector<double>& b,
                                                       size_t task_id) {
    bool stored = true;
    size_t end = H_IJ_bb_list_thread_end_[task_id];
    size_t limit = H_IJ_list_thread_limit_[task_id];

    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(Ia);
    UI64Determinant::bit_t Ib;
    UI64Determinant::bit_t Jb;
    UI64Determinant::bit_t IJb;
    size_t first_I = range_I.first;
    size_t last_I = range_I.second;
    double sigma_I = 0.0;
    size_t num_elements = 0;
    for (size_t posI = first_I; posI < last_I; ++posI) {
        sigma_I = 0.0;
        Ib = sorted_dets[posI].get_beta_bits();
        for (size_t posJ = posI + 1; posJ < last_I; ++posJ) {
            Jb = sorted_dets[posJ].get_beta_bits();
#if SIGMA_VEC_DEBUG
            count_bb_total++;
#endif
            // find common bits
            IJb = Jb ^ Ib;
            int ndiff = ui64_bit_count(IJb);
            if (ndiff == 2) {
                double H_IJ = slater_rules_single_beta(Ia, Ib, Jb, fci_ints_);
                sigma_I += H_IJ * b[posJ];
                temp_sigma_[posJ] += H_IJ * b[posI];
                // Add this to the Hamiltonian
                if (end + num_elements < limit) {
                    if (std::fabs(H_IJ) > H_threshold_) {
                        H_IJ_list_[end + num_elements] = std::tie(H_IJ, posI, posJ);
                        num_elements++;
                    }
                } else {
                    stored = false;
                }
#if SIGMA_VEC_DEBUG
                count_bb++;
#endif
            } else if (ndiff == 4) {
                double H_IJ = slater_rules_double_beta_beta(Ib, Jb, fci_ints_);
                sigma_I += H_IJ * b[posJ];
                temp_sigma_[posJ] += H_IJ * b[posI];
                // Add this to the Hamiltonian
                if (end + num_elements < limit) {
                    if (std::fabs(H_IJ) > H_threshold_) {
                        H_IJ_list_[end + num_elements] = std::tie(H_IJ, posI, posJ);
                        num_elements++;
                    }
                } else {
                    stored = false;
                }
#if SIGMA_VEC_DEBUG
                count_bbbb++;
#endif
            }
        }
        temp_sigma_[posI] += sigma_I;
    }
    if (stored) {
        H_IJ_bb_list_thread_end_[task_id] += num_elements;
    }
    return stored;
}

void SigmaVectorDynamic::compute_bb_coupling(const UI64Determinant::bit_t& Ia,
                                             const std::vector<double>& b) {
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(Ia);
    UI64Determinant::bit_t Ib;
    UI64Determinant::bit_t Jb;
    UI64Determinant::bit_t IJb;
    size_t first_I = range_I.first;
    size_t last_I = range_I.second;
    double sigma_I = 0.0;
    for (size_t posI = first_I; posI < last_I; ++posI) {
        sigma_I = 0.0;
        Ib = sorted_dets[posI].get_beta_bits();
        for (size_t posJ = posI + 1; posJ < last_I; ++posJ) {
            Jb = sorted_dets[posJ].get_beta_bits();
#if SIGMA_VEC_DEBUG
            count_bb_total++;
#endif
            // find common bits
            IJb = Jb ^ Ib;
            int ndiff = ui64_bit_count(IJb);
            if (ndiff == 2) {
                double H_IJ = slater_rules_single_beta(Ia, Ib, Jb, fci_ints_);
                sigma_I += H_IJ * b[posJ];
                temp_sigma_[posJ] += H_IJ * b[posI];
#if SIGMA_VEC_DEBUG
                count_bb++;
#endif
            } else if (ndiff == 4) {
                double H_IJ = slater_rules_double_beta_beta(Ib, Jb, fci_ints_);
                sigma_I += H_IJ * b[posJ];
                temp_sigma_[posJ] += H_IJ * b[posI];
#if SIGMA_VEC_DEBUG
                count_bbbb++;
#endif
            }
        }
        temp_sigma_[posI] += sigma_I;
    }
}

bool SigmaVectorDynamic::compute_abab_coupling_and_store(const UI64Determinant::bit_t& detIa,
                                                         const std::vector<double>& b,
                                                         size_t task_id) {
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(detIa);
    bool stored = true;
    size_t end = H_IJ_abab_list_thread_end_[task_id];
    size_t limit = H_IJ_list_thread_limit_[task_id];
    UI64Determinant::bit_t detIJa_common;
    UI64Determinant::bit_t Ib;
    UI64Determinant::bit_t Jb;
    UI64Determinant::bit_t IJb;

    size_t group_num_elements = 0;
    for (const auto& detJa : sorted_half_dets) {
        detIJa_common = detIa ^ detJa;
        int ndiff = ui64_bit_count(detIJa_common);
        if (ndiff == 2) {
            int i, a;
            for (int p = 0; p < nmo_; ++p) {
                const bool la_p = ui64_get_bit(detIa, p);
                const bool ra_p = ui64_get_bit(detJa, p);
                if (la_p ^ ra_p) {
                    i = la_p ? p : i;
                    a = ra_p ? p : a;
                }
            }
            double sign_ia = ui64_slater_sign(detIa, i, a);
            const auto& range_J = a_sorted_string_list_.range(detJa);

            size_t first_I = range_I.first;
            size_t last_I = range_I.second;
            size_t first_J = range_J.first;
            size_t last_J = range_J.second;
            double sigma_I = 0.0;
            //    size_t num_elements = 0;
            for (size_t posI = first_I; posI < last_I; ++posI) {
                sigma_I = 0.0;
                Ib = sorted_dets[posI].get_beta_bits();
                for (size_t posJ = first_J; posJ < last_J; ++posJ) {
                    Jb = sorted_dets[posJ].get_beta_bits();
#if SIGMA_VEC_DEBUG
                    count_abab_total++;
#endif
                    // find common bits
                    IJb = Jb ^ Ib;
                    int ndiff = ui64_bit_count(IJb);
                    if (ndiff == 2) {
                        double H_IJ =
                            sign_ia * slater_rules_double_alpha_beta_pre(i, a, Ib, Jb, fci_ints_);
                        sigma_I += H_IJ * b[posJ];
                        // Add this to the Hamiltonian
                        if (end + group_num_elements < limit) {
                            if (std::fabs(H_IJ) > H_threshold_) {
                                H_IJ_list_[end + group_num_elements] = std::tie(H_IJ, posI, posJ);
                                group_num_elements++;
                            }
                        } else {
                            stored = false;
                        }
#if SIGMA_VEC_DEBUG
                        count_abab++;
#endif
                    }
                }
                temp_sigma_[posI] += sigma_I;
            }
        }
    }
    if (stored) {
        H_IJ_abab_list_thread_end_[task_id] += group_num_elements;
    }
    return stored;
}

void SigmaVectorDynamic::compute_abab_coupling(const UI64Determinant::bit_t& detIa,
                                               const std::vector<double>& b, size_t task_id) {
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(detIa);
    UI64Determinant::bit_t detIJa_common;
    UI64Determinant::bit_t Ib;
    UI64Determinant::bit_t Jb;
    UI64Determinant::bit_t IJb;

    for (const auto& detJa : sorted_half_dets) {
        detIJa_common = detIa ^ detJa;
        int ndiff = ui64_bit_count(detIJa_common);
        if (ndiff == 2) {
            int i, a;
            for (int p = 0; p < nmo_; ++p) {
                const bool la_p = ui64_get_bit(detIa, p);
                const bool ra_p = ui64_get_bit(detJa, p);
                if (la_p ^ ra_p) {
                    i = la_p ? p : i;
                    a = ra_p ? p : a;
                }
            }
            double sign_ia = ui64_slater_sign(detIa, i, a);
            const auto& range_J = a_sorted_string_list_.range(detJa);
            size_t first_I = range_I.first;
            size_t last_I = range_I.second;
            size_t first_J = range_J.first;
            size_t last_J = range_J.second;
            double sigma_I = 0.0;
            //    size_t num_elements = 0;
            for (size_t posI = first_I; posI < last_I; ++posI) {
                sigma_I = 0.0;
                Ib = sorted_dets[posI].get_beta_bits();
                for (size_t posJ = first_J; posJ < last_J; ++posJ) {
                    Jb = sorted_dets[posJ].get_beta_bits();
#if SIGMA_VEC_DEBUG
                    count_abab_total++;
#endif
                    // find common bits
                    IJb = Jb ^ Ib;
                    int ndiff = ui64_bit_count(IJb);
                    if (ndiff == 2) {
                        double H_IJ =
                            sign_ia * slater_rules_double_alpha_beta_pre(i, a, Ib, Jb, fci_ints_);
                        sigma_I += H_IJ * b[posJ];
#if SIGMA_VEC_DEBUG
                        count_abab++;
#endif
                    }
                }
                temp_sigma_[posI] += sigma_I;
            }
        }
    }
}
}
}
