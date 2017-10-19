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

#ifndef _sigma_vector_direct_h_
#define _sigma_vector_direct_h_

#include "sigma_vector.h"
#include "sorted_string_list.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace psi {
namespace forte {

// class SortedStringList;

/**
 * @brief The SigmaVectorDirect class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorDirect : public SigmaVector {
  public:
    SigmaVectorDirect(const DeterminantHashVec& space, std::shared_ptr<FCIIntegrals> fci_ints);
    ~SigmaVectorDirect();
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
    std::vector<double> temp_sigma_;
    const DeterminantHashVec& space_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
    SortedStringList a_sorted_string_list_;
    SortedStringList b_sorted_string_list_;
    SortedStringList_UI64 a_sorted_string_list_ui64_;
    SortedStringList_UI64 b_sorted_string_list_ui64_;

    void compute_sigma_scalar(SharedVector sigma, SharedVector b);
    void compute_sigma_aa(SharedVector sigma, SharedVector b);
    void compute_sigma_bb(SharedVector sigma, SharedVector b);
    void compute_sigma_aaaa(SharedVector sigma, SharedVector b);
    void compute_sigma_abab(SharedVector sigma, SharedVector b);
    void compute_sigma_bbbb(SharedVector sigma, SharedVector b);

    void compute_sigma_aa_fast_search(SharedVector sigma, SharedVector b);
    void compute_sigma_bb_fast_search(SharedVector sigma, SharedVector b);
    void compute_sigma_abab_fast_search(SharedVector sigma, SharedVector b);
    void compute_sigma_abab_fast_search_group(SharedVector sigma, SharedVector b);
    void compute_sigma_aa_fast_search_group_ui64(SharedVector sigma, SharedVector b);
    void compute_sigma_bb_fast_search_group_ui64(SharedVector sigma, SharedVector b);
    void compute_sigma_abab_fast_search_group_ui64(SharedVector sigma, SharedVector b);

    void compute_aa_coupling(const STLBitsetDeterminant& detI, const double b_I, double* sigma_p);
    void compute_bb_coupling(const STLBitsetDeterminant& detI, const double b_I);
    void compute_bb_coupling_compare(const STLBitsetDeterminant& detI, const double b_I);
    void compute_aa_coupling_compare(const STLBitsetDeterminant& detI, const double b_I);
    void compute_bb_coupling_compare_singles(const STLBitsetDeterminant& detI,
                                             const STLBitsetDeterminant& detI_ia, const double b_I,
                                             double sign, int i, int a);
    void compute_bb_coupling_compare_singles_group(const STLBitsetDeterminant& detIa,
                                                   const STLBitsetDeterminant& detJa, double sign,
                                                   int i, int a, const SharedVector& b);
    void compute_aa_coupling_compare_group_ui64(const UI64Determinant::bit_t& detIb,
                                                const SharedVector& b);
    void compute_bb_coupling_compare_group_ui64(const UI64Determinant::bit_t& detIa,
                                                const SharedVector& b);
    void compute_bb_coupling_compare_singles_group_ui64(const UI64Determinant::bit_t& detIa,
                                                        const UI64Determinant::bit_t& detJa,
                                                        double sign, int i, int a,
                                                        const SharedVector& b);
};
}
}

#endif // _sigma_vector_direct_h_
