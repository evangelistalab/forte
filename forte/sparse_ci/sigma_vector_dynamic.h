/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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
class Vector;
}

namespace forte {

enum class SigmaVectorMode { Dynamic, OnTheFly };

/**
 * @brief The SigmaVectorDynamic class
 * Computes the sigma vector with a dynamic approach
 */
class SigmaVectorDynamic : public SigmaVector {
  public:
    SigmaVectorDynamic(const DeterminantHashVec& space,
                       std::shared_ptr<ActiveSpaceIntegrals> fci_ints, size_t max_memory);
    ~SigmaVectorDynamic();
    void compute_sigma(std::shared_ptr<psi::Vector> sigma, std::shared_ptr<psi::Vector> b) override;
    void get_diagonal(psi::Vector& diag) override;
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states) override;
    double compute_spin(const std::vector<double>& c) override;

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    SigmaVectorMode mode_ = SigmaVectorMode::Dynamic;
    /// The number of threads
    int num_threads_ = 1;
    size_t total_space_ = 0;
    /// The number of molecular orbitals
    size_t nmo_ = 0;
    /// Number of sigma builds
    int num_builds_ = 0;
    double H_threshold_ = 1.0e-14;
    /// A temporary sigma vector of size N_det
    std::vector<double> temp_b_;
    /// A temporary sigma vector of size N_det
    std::vector<double> temp_sigma_;
    SortedStringList a_sorted_string_list_;
    SortedStringList b_sorted_string_list_;

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

    void print_thread_stats();
    /// Scalar contribution to sigma
    void compute_sigma_scalar(std::shared_ptr<psi::Vector> sigma, std::shared_ptr<psi::Vector> b);
    /// Alpha-alpha single and double excitation contributions to sigma
    void compute_sigma_aa(std::shared_ptr<psi::Vector> sigma, std::shared_ptr<psi::Vector> b);
    /// Beta-beta single and double excitation contributions to sigma
    void compute_sigma_bb(std::shared_ptr<psi::Vector> sigma, std::shared_ptr<psi::Vector> b);
    /// Alpha-beta double excitation contributions to sigma
    void compute_sigma_abab(std::shared_ptr<psi::Vector> sigma, std::shared_ptr<psi::Vector> b);

    /// Task to compute sigma_aa. Computes sigma and stores part of the Hamiltonian
    void sigma_aa_store_task(size_t task_id, size_t num_tasks);
    /// Task to compute sigma_aa. Computes sigma using a dynamic approach
    void sigma_aa_dynamic_task(size_t task_id, size_t num_tasks);

    /// Task to compute sigma_bb. Computes sigma and stores part of the Hamiltonian
    void sigma_bb_store_task(size_t task_id, size_t num_tasks);
    /// Task to compute sigma_bb. Computes sigma using a dynamic approach
    void sigma_bb_dynamic_task(size_t task_id, size_t num_tasks);

    /// Task to compute sigma_abab. Computes sigma and stores part of the Hamiltonian
    void sigma_abab_store_task(size_t task_id, size_t num_tasks);
    /// Task to compute sigma_abab. Computes sigma using a dynamic approach
    void sigma_abab_dynamic_task(size_t task_id, size_t num_tasks);

    bool compute_aa_coupling_and_store(const String& Ib, const std::vector<double>& b,
                                       size_t task_id);
    bool compute_bb_coupling_and_store(const String& Ia, const std::vector<double>& b,
                                       size_t task_id);
    bool compute_abab_coupling_and_store(const String& detIa, const std::vector<double>& b,
                                         size_t task_id);

    void compute_aa_coupling(const String& detIb, const std::vector<double>& b);
    void compute_bb_coupling(const String& detIa, const std::vector<double>& b);
    void compute_abab_coupling(const String& detIa, const std::vector<double>& b);
};
} // namespace forte

#endif // _sigma_vector_dynamic_h_
