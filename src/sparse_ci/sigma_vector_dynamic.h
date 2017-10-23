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

#ifndef _sigma_vector_dynamic_h_
#define _sigma_vector_dynamic_h_

#include "sigma_vector.h"
#include "sorted_string_list.h"

namespace psi {
namespace forte {

/**
 * @brief The SigmaVectorDynamic class
 * Computes the sigma vector with a dynamic approach
 */
class SigmaVectorDynamic : public SigmaVector {
  public:
    SigmaVectorDynamic(const DeterminantHashVec& space, std::shared_ptr<FCIIntegrals> fci_ints,
                       size_t max_memory);
    ~SigmaVectorDynamic();
    void compute_sigma(SharedVector sigma, SharedVector b);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);
    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    /// The number of threads
    int num_threads_ = 1;
    int nmo_ = 0;
    /// Number of sigma builds
    int num_builds_ = 0;
    /// Diagonal elements of the Hamiltonian
    std::vector<double> diag_;
    /// A temporary sigma vector of size N_det
    std::vector<double> temp_b_;
    /// A temporary sigma vector of size N_det
    std::vector<double> temp_sigma_;
    const DeterminantHashVec& space_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
    SortedStringList_UI64 a_sorted_string_list_ui64_;
    SortedStringList_UI64 b_sorted_string_list_ui64_;

    /// The Hamiltonian stored as a list of pairs (H_IJ, I, J)
    std::vector<std::tuple<double, std::uint32_t, std::uint32_t>> H_IJ_list_;
    std::vector<size_t> H_IJ_list_thread_limit_;

    std::vector<size_t> H_IJ_aa_list_thread_start_;
    std::vector<size_t> H_IJ_aa_list_thread_end_;

    std::vector<size_t> H_IJ_bb_list_thread_start_;
    std::vector<size_t> H_IJ_bb_list_thread_end_;

    std::vector<size_t> H_IJ_abab_list_thread_start_;
    std::vector<size_t> H_IJ_abab_list_thread_end_;

    std::vector<size_t> first_aa_onthefly_group_;
    std::vector<size_t> first_bb_onthefly_group_;
    std::vector<size_t> first_abab_onthefly_group_;
    /// Stores the position of the first elements of a row of H
    /// outer vector is for threads (index I, begin, end)
    //    std::vector<std::vector<std::tuple<size_t,size_t,size_t>>> H_I_first_;

    void print_thread_stats();
    void compute_sigma_scalar(SharedVector sigma, SharedVector b);
    void compute_sigma_aa_fast_search_group_ui64(SharedVector sigma, SharedVector b);
    void compute_sigma_bb_fast_search_group_ui64(SharedVector sigma, SharedVector b);
    void compute_sigma_abab_fast_search_group_ui64(SharedVector sigma, SharedVector b);

    void compute_sigma_aa_fast_search_group_ui64_parallel(SharedVector sigma, SharedVector b);
    void compute_sigma_bb_fast_search_group_ui64_parallel(SharedVector sigma, SharedVector b);
    void compute_sigma_abab_fast_search_group_ui64_parallel(SharedVector sigma, SharedVector b);

    void sigma_aa_task(size_t task_id, size_t num_tasks);
    void sigma_bb_task(size_t task_id, size_t num_tasks);
    void sigma_abab_task(size_t task_id, size_t num_tasks);

    void sigma_aa_store_task(size_t task_id, size_t num_tasks);
    void sigma_aa_dynamic_task(size_t task_id, size_t num_tasks);

    void sigma_bb_store_task(size_t task_id, size_t num_tasks);
    void sigma_bb_dynamic_task(size_t task_id, size_t num_tasks);

    void sigma_abab_store_task(size_t task_id, size_t num_tasks);
    void sigma_abab_dynamic_task(size_t task_id, size_t num_tasks);

    void compute_aa_coupling_compare_group_ui64(const UI64Determinant::bit_t& detIb,
                                                const std::vector<double>& b);
    void compute_bb_coupling_compare_group_ui64(const UI64Determinant::bit_t& detIa,
                                                const std::vector<double>& b);
    void compute_bb_coupling_compare_singles_group_ui64(const UI64Determinant::bit_t& detIa,
                                                        const UI64Determinant::bit_t& detJa,
                                                        double sign, int i, int a,
                                                        const std::vector<double>& b);
    void compute_bb_coupling_singles_parallel(const UI64Determinant::bit_t& detIa,
                                              const std::vector<double>& b, size_t task_id);

    bool compute_aa_coupling_and_store(const UI64Determinant::bit_t& Ib,
                                       const std::vector<double>& b, size_t task_id);
    bool compute_bb_coupling_and_store(const UI64Determinant::bit_t& Ia,
                                       const std::vector<double>& b, size_t task_id);
    bool compute_bb_coupling_singles_and_store(const UI64Determinant::bit_t& detIa,
                                               const std::vector<double>& b, size_t task_id);
};
}
}

#endif // _sigma_vector_dynamic_h_
