/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "forte-def.h"
#include "helpers/timer.h"
#include "helpers/iterative_solvers.h"
#include "sigma_vector_dynamic.h"
#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant_functions.hpp"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

using namespace psi;

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
double saa_time = 0.0;
double sbb_time = 0.0;
double sabab_time = 0.0;

void print_SigmaVectorDynamic_stats();

SigmaVectorDynamic::SigmaVectorDynamic(const DeterminantHashVec& space,
                                       std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                       size_t max_memory)
    : SigmaVector(space, fci_ints, SigmaVectorType::Dynamic, "SigmaVectorDynamic"),
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
    { num_threads_ = omp_get_max_threads(); }

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
    outfile->Printf("\n\n  SigmaVectorDynamic:");
    outfile->Printf("\n  Maximum memory   : %zu double", total_space_);
    outfile->Printf("\n  Number of threads: %d\n", num_threads_);
}

SigmaVectorDynamic::~SigmaVectorDynamic() { print_SigmaVectorDynamic_stats(); }

void SigmaVectorDynamic::reset() {
    num_builds_ = 0;
    H_IJ_list_.clear();
    H_IJ_list_.resize(total_space_);

    size_t space_per_thread = total_space_ / num_threads_;
    for (int t = 0; t < num_threads_; ++t) {
        H_IJ_list_thread_limit_[t] = (t + 1) * space_per_thread;

        H_IJ_aa_list_thread_start_[t] = t * space_per_thread;
        H_IJ_aa_list_thread_end_[t] = t * space_per_thread;

        H_IJ_bb_list_thread_start_[t] = t * space_per_thread;
        H_IJ_bb_list_thread_end_[t] = t * space_per_thread;

        H_IJ_abab_list_thread_start_[t] = t * space_per_thread;
        H_IJ_abab_list_thread_end_[t] = t * space_per_thread;

        first_aa_onthefly_group_[t] = t;
        first_bb_onthefly_group_[t] = t;
        first_abab_onthefly_group_[t] = t;
    }
}

void SigmaVectorDynamic::compute_sigma(psi::SharedVector sigma, psi::SharedVector b) {
    sigma->zero();

    compute_sigma_scalar(sigma, b);
    {
        local_timer t;
        compute_sigma_aa(sigma, b);
        saa_time += t.get();
    }
    {
        local_timer t;
        compute_sigma_bb(sigma, b);
        sbb_time += t.get();
    }
    {
        local_timer t;
        compute_sigma_abab(sigma, b);
        sabab_time += t.get();
    }

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
    outfile->Printf("\n\n SigmaVectorDynamic:");
    outfile->Printf("\n saa_time   : %e", saa_time);
    outfile->Printf("\n sbb_time   : %e", sbb_time);
    outfile->Printf("\n sabab_time : %e", sabab_time);
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
    bad_states_ = roots;
}

void SigmaVectorDynamic::get_diagonal(psi::Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorDynamic::compute_sigma_scalar(psi::SharedVector sigma, psi::SharedVector b) {
    timer energy_timer("scalar");

    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    for (size_t I = 0; I < size_; ++I) {
        double b_I = b_p[I];
        sigma_p[I] += diag_[I] * b_I;
    }
}

void SigmaVectorDynamic::compute_sigma_aa(psi::SharedVector sigma, psi::SharedVector b) {
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

void SigmaVectorDynamic::compute_sigma_bb(psi::SharedVector sigma, psi::SharedVector b) {
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

void SigmaVectorDynamic::compute_sigma_abab(psi::SharedVector sigma, psi::SharedVector b) {
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
            compute_abab_coupling(detIa, temp_b_);
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
        compute_abab_coupling(detIa, temp_b_);
    }
}

bool SigmaVectorDynamic::compute_aa_coupling_and_store(const String& Ib,
                                                       const std::vector<double>& b,
                                                       size_t task_id) {
    bool stored = true;
    size_t end = H_IJ_aa_list_thread_end_[task_id];
    size_t limit = H_IJ_list_thread_limit_[task_id];

    const auto& sorted_dets = b_sorted_string_list_.sorted_dets();
    const auto& range_I = b_sorted_string_list_.range(Ib);
    String Ia;
    String Ja;
    String IJa;
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
            int ndiff = IJa.count();
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

void SigmaVectorDynamic::compute_aa_coupling(const String& Ib, const std::vector<double>& b) {
    const auto& sorted_dets = b_sorted_string_list_.sorted_dets();
    const auto& range_I = b_sorted_string_list_.range(Ib);
    String Ia;
    String Ja;
    String IJa;
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
            int ndiff = IJa.count();
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

bool SigmaVectorDynamic::compute_bb_coupling_and_store(const String& Ia,
                                                       const std::vector<double>& b,
                                                       size_t task_id) {
    bool stored = true;
    size_t end = H_IJ_bb_list_thread_end_[task_id];
    size_t limit = H_IJ_list_thread_limit_[task_id];

    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(Ia);
    String Ib;
    String Jb;
    String IJb;
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
            int ndiff = IJb.count();
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

void SigmaVectorDynamic::compute_bb_coupling(const String& Ia, const std::vector<double>& b) {
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(Ia);
    String Ib;
    String Jb;
    String IJb;
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
            int ndiff = IJb.count();
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

bool SigmaVectorDynamic::compute_abab_coupling_and_store(const String& detIa,
                                                         const std::vector<double>& b,
                                                         size_t task_id) {
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(detIa);
    bool stored = true;
    size_t end = H_IJ_abab_list_thread_end_[task_id];
    size_t limit = H_IJ_list_thread_limit_[task_id];
    String Ib;
    String Jb;

    size_t group_num_elements = 0;
    for (const auto& detJa : sorted_half_dets) {
        if (detIa.fast_a_xor_b_count(detJa) == 2) {
            int i, a;
            for (size_t p = 0; p < nmo_; ++p) {
                const bool la_p = detIa.get_bit(p);
                const bool ra_p = detJa.get_bit(p);
                if (la_p ^ ra_p) {
                    i = la_p ? p : i;
                    a = ra_p ? p : a;
                }
            }
            double sign_ia = detIa.slater_sign(i, a);
            const auto& range_J = a_sorted_string_list_.range(detJa);

            size_t first_I = range_I.first;
            size_t last_I = range_I.second;
            size_t first_J = range_J.first;
            size_t last_J = range_J.second;
            double sigma_I = 0.0;
            //    size_t num_elements = 0;
            for (size_t posI = first_I; posI < last_I; ++posI) {
                sigma_I = 0.0;
                sorted_dets[posI].copy_beta_bits(Ib);
                for (size_t posJ = first_J; posJ < last_J; ++posJ) {
                    sorted_dets[posJ].copy_beta_bits(Jb);
#if SIGMA_VEC_DEBUG
                    count_abab_total++;
#endif
                    // find common bits
                    if (Ib.fast_a_xor_b_count(Jb) == 2) {
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

void SigmaVectorDynamic::compute_abab_coupling(const String& detIa, const std::vector<double>& b) {
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(detIa);
    String Ib;
    String Jb;
    String IJb;
    for (const auto& detJa : sorted_half_dets) {
        if (detIa.fast_a_xor_b_count(detJa) == 2) {
            int i, a;
            for (size_t p = 0; p < nmo_; ++p) {
                const bool la_p = detIa.get_bit(p);
                const bool ra_p = detJa.get_bit(p);
                if (la_p ^ ra_p) {
                    i = la_p ? p : i;
                    a = ra_p ? p : a;
                }
            }
            double sign_ia = detIa.slater_sign(i, a);
            const auto& range_J = a_sorted_string_list_.range(detJa);
            size_t first_I = range_I.first;
            size_t last_I = range_I.second;
            size_t first_J = range_J.first;
            size_t last_J = range_J.second;
            double sigma_I = 0.0;
            for (size_t posI = first_I; posI < last_I; ++posI) {
                sigma_I = 0.0;
                sorted_dets[posI].copy_beta_bits(Ib);
                for (size_t posJ = first_J; posJ < last_J; ++posJ) {
                    sorted_dets[posJ].copy_beta_bits(Jb);
#if SIGMA_VEC_DEBUG
                    count_abab_total++;
#endif
                    // find common bits
                    if (Ib.fast_a_xor_b_count(Jb) == 2) {
                        IJb = Ib ^ Jb;
                        uint64_t j = IJb.find_and_clear_first_one();
                        uint64_t bb = IJb.find_first_one();
                        const double H_IJ = Ib.slater_sign(j, bb) * fci_ints_->tei_ab(i, j, a, bb);
                        sigma_I += H_IJ * b[posJ];
#if SIGMA_VEC_DEBUG
                        count_abab++;
#endif
                    }
                }
                temp_sigma_[posI] += sign_ia * sigma_I;
            }
        }
    }
} // namespace forte

double SigmaVectorDynamic::compute_spin(const std::vector<double>& c) {
    double S2 = 0.0;
    const det_hashvec& wfn_map = space_.wfn_hash();

    for (size_t i = 0, max_i = wfn_map.size(); i < max_i; ++i) {
        // Compute the diagonal contribution
        // PhiI = PhiJ
        const Determinant& PhiI = wfn_map[i];
        double CI = c[i];
        int npair = PhiI.npair();
        int na = PhiI.count_alfa();
        int nb = PhiI.count_beta();
        double ms = 0.5 * static_cast<double>(na - nb);
        S2 += (ms * ms - ms + static_cast<double>(na) - static_cast<double>(npair)) * CI * CI;
    }

    // abab contribution
    //    SortedStringList a_sorted_string_list(space_, fci_ints_, DetSpinType::Alpha);
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    String detIJa_common;
    String Ib;
    String Jb;
    String IJb;

    // Loop over all the sorted I alpha strings
    for (const auto& detIa : sorted_half_dets) {
        const auto& range_I = a_sorted_string_list_.range(detIa);
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Loop over all the sorted J alpha strings
        for (const auto& detJa : sorted_half_dets) {
            detIJa_common = detIa ^ detJa;
            int ndiff = detIJa_common.count();
            if (ndiff == 2) {
                size_t i, a;
                for (size_t p = 0; p < nmo_; ++p) {
                    const bool la_p = detIa.get_bit(p);
                    const bool ra_p = detJa.get_bit(p);
                    if (la_p ^ ra_p) {
                        i = la_p ? p : i;
                        a = ra_p ? p : a;
                    }
                }
                double sign_ia = detIa.slater_sign(i, a);
                const auto& range_J = a_sorted_string_list_.range(detJa);
                size_t first_J = range_J.first;
                size_t last_J = range_J.second;
                for (size_t posI = first_I; posI < last_I; ++posI) {
                    Ib = sorted_dets[posI].get_beta_bits();
                    double CI = c[a_sorted_string_list_.add(posI)];
                    for (size_t posJ = first_J; posJ < last_J; ++posJ) {
                        Jb = sorted_dets[posJ].get_beta_bits();
                        IJb = Jb ^ Ib;
                        int ndiff = IJb.count();
                        if (ndiff == 2) {
                            auto Ib_sub = Ib & IJb;
                            auto j = Ib_sub.find_first_one();
                            auto Jb_sub = Jb & IJb;
                            auto b = Jb_sub.find_first_one();
                            if ((i != j) and (a != b) and (i == b) and (j == a)) {
                                double sign = sign_ia * Ib.slater_sign(j, b);
                                S2 -= sign * CI * c[a_sorted_string_list_.add(posJ)];
                            }
                        }
                    }
                }
            }
        }
    }
    return S2;
}
} // namespace forte
