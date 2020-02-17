/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _sigma_vector_sparse_list_h_
#define _sigma_vector_sparse_list_h_

#include "sigma_vector.h"

namespace forte {

/**
 * @brief The SigmaVectorSparseList class
 * Computes the sigma vector from a creation list sparse Hamiltonian.
 */
class SigmaVectorSparseList : public SigmaVector {
  public:
    SigmaVectorSparseList(const DeterminantHashVec& space,
                          std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                          std::shared_ptr<WFNOperator> op);

    void compute_sigma(psi::SharedVector sigma, psi::SharedVector b);
    // void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(psi::Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    bool print_;
    bool use_disk_ = false;

    std::vector<double> diag_;
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;
    std::vector<std::vector<std::pair<size_t, short>>>& a_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_list_;
};

} // namespace forte

#endif // _sigma_vector_sparse_list_h_
