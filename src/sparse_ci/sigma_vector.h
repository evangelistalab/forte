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

#ifndef _sigma_vector_h_
#define _sigma_vector_h_

#include <string>

#include "sparse_ci/determinant_hashvector.h"

namespace forte {

enum class SigmaVectorType { Dynamic, SparseList, Full };

class ActiveSpaceIntegrals;
class WFNOperator;

/**
 * @brief The SigmaVector class
 *        Base class for a sigma vector object.
 */
class SigmaVector {
  public:
    SigmaVector(const DeterminantHashVec& space, std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                std::string sigma_type)
        : space_(space), fci_ints_(fci_ints), size_(space.size()), type_(sigma_type) {}

    size_t size() { return size_; }

    std::shared_ptr<ActiveSpaceIntegrals> as_ints() { return fci_ints_; }

    std::string type() { return type_; }

    virtual void compute_sigma(psi::SharedVector sigma, psi::SharedVector b) = 0;
    virtual void get_diagonal(psi::Vector& diag) = 0;
    virtual void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states) = 0;

  protected:
    const DeterminantHashVec& space_;
    /// the active space integrals
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;
    /// the length of the C/sigma vector (number of determinants)
    size_t size_;
    /// the type of sigma vector algorithm
    std::string type_;
};

std::shared_ptr<SigmaVector>
make_sigma_vector(DeterminantHashVec& space, std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                  size_t max_memory, SigmaVectorType sigma_type,
                  std::shared_ptr<WFNOperator> op = std::shared_ptr<WFNOperator>());

} // namespace forte

#endif // _sigma_vector_h_

///**
// * @brief The SigmaVectorSparse class
// * Computes the sigma vector from a sparse Hamiltonian.
// */
// class SigmaVectorSparse : public SigmaVector {
//  public:
//    SigmaVectorSparse(std::vector<std::pair<std::vector<size_t>, std::vector<double>>>& H,
//                      std::shared_ptr<ActiveSpaceIntegrals> fci_ints)
//        : SigmaVector(H.size()), H_(H), fci_ints_(fci_ints){};

//    void compute_sigma(psi::SharedVector sigma, psi::SharedVector b);
//    //   void compute_sigma(Matrix& sigma, Matrix& b, int nroot) {}
//    void get_diagonal(psi::Vector& diag);
//    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);

//    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

//  protected:
//    std::vector<std::pair<std::vector<size_t>, std::vector<double>>>& H_;
//    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;
//};
