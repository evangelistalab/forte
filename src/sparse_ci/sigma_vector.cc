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

#include "helpers/string_algorithms.h"
#include "integrals/active_space_integrals.h"
#include "sigma_vector_dynamic.h"
#include "sigma_vector_sparse_list.h"

namespace forte {

SigmaVectorType string_to_sigma_vector_type(std::string type) {
    //    to_upper_string(type);
    if (type == "FULL") {
        return SigmaVectorType::Full;
    } else if (type == "SPARSE") {
        return SigmaVectorType::SparseList;
    } else if (type == "DYNAMIC") {
        return SigmaVectorType::Dynamic;
    }
    throw std::runtime_error("string_to_sigma_vector_type() called with incorrect type: " + type);
    return SigmaVectorType::Dynamic;
}

std::shared_ptr<SigmaVector> make_sigma_vector(DeterminantHashVec& space,
                                               std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                               size_t max_memory, SigmaVectorType sigma_type) {
    std::shared_ptr<SigmaVector> sigma_vector;
    if (sigma_type == SigmaVectorType::Dynamic) {
        sigma_vector = std::make_shared<SigmaVectorDynamic>(space, fci_ints, max_memory);
    } else if (sigma_type == SigmaVectorType::SparseList) {
        sigma_vector = std::make_shared<SigmaVectorSparseList>(space, fci_ints);
    } else if (sigma_type == SigmaVectorType::Full) {
        sigma_vector = std::make_shared<SigmaVectorFull>(space, fci_ints);
    }
    return sigma_vector;
}

std::shared_ptr<SigmaVector> make_sigma_vector(const std::vector<Determinant>& space,
                                               std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                               size_t max_memory, SigmaVectorType sigma_type) {
    DeterminantHashVec dhv(space);
    return make_sigma_vector(dhv, fci_ints, max_memory, sigma_type);
}

SigmaVectorFull::SigmaVectorFull(const DeterminantHashVec& space,
                                 std::shared_ptr<ActiveSpaceIntegrals> fci_ints)
    : SigmaVector(space, fci_ints, SigmaVectorType::Full, "SigmaVectorFull") {}

void SigmaVectorFull::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {}

void SigmaVectorFull::get_diagonal(psi::Vector& diag) {}

void SigmaVectorFull::compute_sigma(std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Vector>) {}

} // namespace forte
