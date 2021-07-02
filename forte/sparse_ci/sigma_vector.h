/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
    virtual void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states) = 0;
    virtual double compute_spin(const std::vector<double>& c) = 0;

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
};

class SigmaVectorFull : public SigmaVector {
  public:
    SigmaVectorFull(const DeterminantHashVec& space,
                    std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

    void compute_sigma(std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Vector>) override;
    // void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(psi::Vector& diag) override;
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_) override;
    double compute_spin(const std::vector<double>&) override { return 0.0; }
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
