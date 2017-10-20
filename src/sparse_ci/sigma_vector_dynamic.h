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
    SigmaVectorDynamic(const DeterminantHashVec& space, std::shared_ptr<FCIIntegrals> fci_ints);
    ~SigmaVectorDynamic();
    void compute_sigma(SharedVector sigma, SharedVector b);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);
    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
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

    void compute_aa_coupling_compare_group_ui64(const UI64Determinant::bit_t& detIb,
                                                const std::vector<double>& b);
    void compute_bb_coupling_compare_group_ui64(const UI64Determinant::bit_t& detIa,
                                                const std::vector<double>& b);
    void compute_bb_coupling_compare_singles_group_ui64(const UI64Determinant::bit_t& detIa,
                                                        const UI64Determinant::bit_t& detJa,
                                                        double sign, int i, int a,
                                                        const std::vector<double>& b);
};
}
}

#endif // _sigma_vector_dynamic_h_
