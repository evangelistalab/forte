/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#pragma once

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

    /// Compute the contribution to sigma due to 1-body operator
    /// sigma_{I} <- factor * sum_{pq} h_{pq} sum_{J} b_{J} <I|p^+ q|J>
    /// h_{pq} = h1[p * nactv + q]
    void add_generalized_sigma_1(const std::vector<double>& h1, std::shared_ptr<psi::Vector> b,
                                 double factor, std::vector<double>& sigma,
                                 const std::string& spin) override;
    /// Compute the contribution to sigma due to 2-body operator
    /// sigma_{I} <- (1/4) * factor * sum_{pqrs} h_{pqrs} sum_{J} b_{J} <I|p^+ q^+ s r|J>
    /// sigma_{I} <- factor * sum_{pqrs} h_{pQrS} sum_{J} b_{J} <I|p^+ Q^+ S r|J>
    /// h_{pqrs} = h2[p * nactv^3 + q * nactv^2 + r * nactv + s]
    /// Integrals must be antisymmetric wrt index permutations!
    void add_generalized_sigma_2(const std::vector<double>& h2, std::shared_ptr<psi::Vector> b,
                                 double factor, std::vector<double>& sigma,
                                 const std::string& spin) override;
    /// Compute the contribution to sigma due to 3-body operator
    /// sigma_{I} <- (1/36) * factor * sum_{pqrstu} h_{pqrstu} sum_{J} b_{J} <I|p^+ q^+ r^+ u t s|J>
    /// sigma_{I} <- (1/4) * factor * sum_{pqRstU} h_{pqRstU} sum_{J} b_{J} <I|p^+ q^+ R^+ U t s|J>
    /// h_{pqrstu} = h3[p * nactv^5 + q * nactv^4 + r * nactv^3 + s * nactv^2 + t * nactv + u]
    /// Integrals must be antisymmetric wrt index permutations!
    void add_generalized_sigma_3(const std::vector<double>& h3, std::shared_ptr<psi::Vector> b,
                                 double factor, std::vector<double>& sigma,
                                 const std::string& spin) override;

  protected:
    bool print_;
    bool use_disk_ = false;
    /// Substitutions lists
    std::shared_ptr<DeterminantSubstitutionLists> op_;

    /// Compute the contribution to sigma due to 1-body operator
    /// sigma_{I} <- factor * sum_{pq} h_{pq} sum_{J} b_{J} <I|p^+ q|J>
    /// h_{pq} = h1[p * nactv + q]
    void add_generalized_sigma1_impl(
        const std::vector<double>& h1, std::shared_ptr<psi::Vector> b, double factor,
        std::vector<double>& sigma,
        const std::vector<std::vector<std::pair<size_t, short>>>& sub_lists);
    /// Compute the contribution to sigma due to 2-body operator
    /// sigma_{I} <- (1/4) * factor * sum_{pqrs} h_{pqrs} sum_{J} b_{J} <I|p^+ q^+ s r|J>
    /// sigma_{I} <- factor * sum_{pqrs} h_{pQrS} sum_{J} b_{J} <I|p^+ Q^+ S r|J>
    /// h_{pqrs} = h2[p * nactv^3 + q * nactv^2 + r * nactv + s]
    /// Integrals must be antisymmetric wrt index permutations!
    void add_generalized_sigma2_impl(
        const std::vector<double>& h2, std::shared_ptr<psi::Vector> b, double factor,
        std::vector<double>& sigma,
        const std::vector<std::vector<std::tuple<size_t, short, short>>>& sub_lists);
    /// Compute the contribution to sigma due to 3-body operator
    /// sigma_{I} <- (1/36) * factor * sum_{pqrstu} h_{pqrstu} sum_{J} b_{J} <I|p^+ q^+ r^+ u t s|J>
    /// sigma_{I} <- (1/4) * factor * sum_{pqRstU} h_{pqRstU} sum_{J} b_{J} <I|p^+ q^+ R^+ U t s|J>
    /// h_{pqrstu} = h3[p * nactv^5 + q * nactv^4 + r * nactv^3 + s * nactv^2 + t * nactv + u]
    /// Integrals must be antisymmetric wrt index permutations!
    void add_generalized_sigma3_impl(
        const std::vector<double>& h3, std::shared_ptr<psi::Vector> b, double factor,
        std::vector<double>& sigma,
        const std::vector<std::vector<std::tuple<size_t, short, short, short>>>& sub_lists);

    /// Test if h2aa or h2bb is antisymmetric
    bool is_h2hs_antisymmetric(const std::vector<double>& h2);
    /// Test if h3aaa or h3bbb is antisymmetric
    bool is_h3hs_antisymmetric(const std::vector<double>& h3);
    /// Test if h3aab or h3abb is antisymmetric
    bool is_h3ls_antisymmetric(const std::vector<double>& h3, bool alpha);
};

} // namespace forte
