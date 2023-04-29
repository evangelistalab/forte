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

#ifndef _sigma_vector_sparse_list_h_
#define _sigma_vector_sparse_list_h_

#include "sigma_vector.h"

namespace psi {
class Vector;
}

namespace forte {

/**
 * @brief The SigmaVectorSparseList class
 * Computes the sigma vector from a creation list sparse Hamiltonian.
 */
class SigmaVectorSparseList : public SigmaVector {
  public:
    SigmaVectorSparseList(const DeterminantHashVec& space,
                          std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

    void compute_sigma(std::shared_ptr<psi::Vector> sigma, std::shared_ptr<psi::Vector> b) override;
    void get_diagonal(psi::Vector& diag) override;
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_) override;
    double compute_spin(const std::vector<double>& c) override;

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

    std::shared_ptr<DeterminantSubstitutionLists> substitution_lists() override;

  protected:
    bool print_;
    bool use_disk_ = false;
    /// Substitutions lists
    std::shared_ptr<DeterminantSubstitutionLists> op_;
};

} // namespace forte

#endif // _sigma_vector_sparse_list_h_
