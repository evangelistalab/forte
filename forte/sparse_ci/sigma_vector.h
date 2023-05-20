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

#ifndef _sigma_vector_h_
#define _sigma_vector_h_

#include <memory>
#include <string>

#include "sparse_ci/determinant_hashvector.h"

namespace psi {
class Vector;
}

namespace forte {

enum class SigmaVectorType { Dynamic, SparseList, Full };

class ActiveSpaceIntegrals;
class DeterminantSubstitutionLists;

/**
 * @brief The SigmaVector class
 *        Base class for a sigma vector object.
 */
class SigmaVector {
  public:
    SigmaVector(const DeterminantHashVec& space, std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                SigmaVectorType sigma_vector_type, std::string label)
        : space_(space), fci_ints_(fci_ints), size_(space.size()),
          sigma_vector_type_(sigma_vector_type), label_(label) {}

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~SigmaVector() = default;

    size_t size() { return size_; }

    std::shared_ptr<ActiveSpaceIntegrals> as_ints() { return fci_ints_; }

    SigmaVectorType sigma_vector_type() const { return sigma_vector_type_; }
    std::string label() const { return label_; }

    virtual void compute_sigma(std::shared_ptr<psi::Vector> sigma,
                               std::shared_ptr<psi::Vector> b) = 0;
    virtual void get_diagonal(psi::Vector& diag) = 0;
    virtual void
    add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& /*bad_states*/) {}
    virtual double compute_spin(const std::vector<double>& c) = 0;

    /// Compute the contribution to sigma due to 1-body operator
    /// sigma_{I} <- factor * sum_{pq} h_{pq} sum_{J} b_{J} <I|p^+ q|J>
    /// h_{pq} = h1a[p * nactv + q]
    virtual void add_generalized_sigma_1(const std::vector<double>& h1,
                                         std::shared_ptr<psi::Vector> b, double factor,
                                         std::vector<double>& sigma, const std::string& spin) {
        _throw_not_implemented_error("add_generalized_sigma_1");
    }

    /// Compute the contribution to sigma due to 2-body operator
    /// sigma_{I} <- (1/4) * factor * sum_{pqrs} h_{pqrs} sum_{J} b_{J} <I|p^+ q^+ s r|J>
    /// sigma_{I} <- factor * sum_{pqrs} h_{pQrS} sum_{J} b_{J} <I|p^+ Q^+ S r|J>
    /// h_{pqrs} = h2[p * nactv^3 + q * nactv^2 + r * nactv + s]
    /// Integrals must be antisymmetric wrt index permutations!
    virtual void add_generalized_sigma_2(const std::vector<double>& h2,
                                         std::shared_ptr<psi::Vector> b, double factor,
                                         std::vector<double>& sigma, const std::string& spin) {
        _throw_not_implemented_error("add_generalized_sigma_2");
    }

    /// Compute the contribution to sigma due to 3-body operator
    /// sigma_{I} <- (1/36) * factor * sum_{pqrstu} h_{pqrstu} sum_{J} b_{J} <I|p^+ q^+ r^+ u t s|J>
    /// sigma_{I} <- (1/4) * factor * sum_{pqRstU} h_{pqRstU} sum_{J} b_{J} <I|p^+ q^+ R^+ U t s|J>
    /// h_{pqrstu} = h3[p * nactv^5 + q * nactv^4 + r * nactv^3 + s * nactv^2 + t * nactv + u]
    /// Integrals must be antisymmetric wrt index permutations!
    virtual void add_generalized_sigma_3(const std::vector<double>& h3,
                                         std::shared_ptr<psi::Vector> b, double factor,
                                         std::vector<double>& sigma, const std::string& spin) {
        _throw_not_implemented_error("add_generalized_sigma_3");
    }

  protected:
    const DeterminantHashVec& space_;
    /// the active space integrals
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;
    /// Diagonal elements of the Hamiltonian
    std::vector<double> diag_;
    /// the length of the C/sigma vector (number of determinants)
    size_t size_;
    const SigmaVectorType sigma_vector_type_;
    /// the type of sigma vector algorithm
    const std::string label_;
    /// throw NotImplemented error
    void _throw_not_implemented_error(std::string msg) {
        throw std::runtime_error(msg + ": not implemented for this SigmaVector type! (" + label() +
                                 ")");
    }
};

SigmaVectorType string_to_sigma_vector_type(std::string type);

std::shared_ptr<SigmaVector> make_sigma_vector(DeterminantHashVec& space,
                                               std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                               size_t max_memory, SigmaVectorType sigma_type);

std::shared_ptr<SigmaVector> make_sigma_vector(const std::vector<Determinant>& space,
                                               std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                               size_t max_memory, SigmaVectorType sigma_type);

} // namespace forte

#endif // _sigma_vector_h_
