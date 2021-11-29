/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

    /// Compute the contribution to sigma due to alpha 1-body operator
    /// sigma_{I} <- factor * sum_{pq} h_{pq} sum_{J} b_{J} <I|p^+ q|J>
    /// h_{pq} = h1a[p * nactv + q]
    void add_generalized_sigma_1a(const std::vector<double>& h1a, psi::SharedVector b,
                                  double factor, std::vector<double>& sigma) override;
    /// Compute the contribution to sigma due to beta 1-body operator
    void add_generalized_sigma_1b(const std::vector<double>& h1b, psi::SharedVector b,
                                  double factor, std::vector<double>& sigma) override;
    /// Compute the contribution to sigma due to alpha-alpha 2-body operator
    /// sigma_{I} <- factor * sum_{pqrs} h_{pqrs} sum_{J} b_{J} <I|p^+ q^+ s r|J>
    /// h_{pqrs} = h2aa[p * nactv^3 + q * nactv^2 + r * nactv + s]
    void add_generalized_sigma_2aa(const std::vector<double>& h2aa, psi::SharedVector b,
                                   double factor, std::vector<double>& sigma) override;
    /// Compute the contribution to sigma due to alpha-beta 2-body operator
    void add_generalized_sigma_2ab(const std::vector<double>& h2ab, psi::SharedVector b,
                                   double factor, std::vector<double>& sigma) override;
    /// Compute the contribution to sigma due to beta-beta 2-body operator
    void add_generalized_sigma_2bb(const std::vector<double>& h2bb, psi::SharedVector b,
                                   double factor, std::vector<double>& sigma) override;
//    /// Compute the contribution to sigma due to alpha-alpha-alpha 3-body operator
//    /// sigma_{I} <- factor * sum_{pqrstu} h_{pqrstu} sum_{J} b_{J} <I|p^+ q^+ r^+ u t s|J>
//    /// h_{pqrstu} = h3aaa[p * nactv^5 + q * nactv^4 + r * nactv^3 + s * nactv^2 + t * nactv + u]
//    void add_generalized_sigma_3aaa(const std::vector<double>& h3aaa, psi::SharedVector b,
//                                    double factor, std::vector<double>& sigma) override;
//    /// Compute the contribution to sigma due to alpha-alpha-beta 3-body operator
//    void add_generalized_sigma_3aab(const std::vector<double>& h3aab, psi::SharedVector b,
//                                    double factor, std::vector<double>& sigma) override;
//    /// Compute the contribution to sigma due to alpha-beta-beta 3-body operator
//    void add_generalized_sigma_3abb(const std::vector<double>& h3abb, psi::SharedVector b,
//                                    double factor, std::vector<double>& sigma) override;
//    /// Compute the contribution to sigma due to beta-beta-beta 3-body operator
//    void add_generalized_sigma_3bbb(const std::vector<double>& h3bbb, psi::SharedVector b,
//                                    double factor, std::vector<double>& sigma) override;

  protected:
    bool print_;
    bool use_disk_ = false;
    /// Substitutions lists
    std::shared_ptr<DeterminantSubstitutionLists> op_;
};

} // namespace forte

#endif // _sigma_vector_sparse_list_h_
