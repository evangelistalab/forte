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

#include "../determinant_hashvector.h"
#include "../fci/fci_integrals.h"
#include "../helpers.h"
#include "../operator.h"
#include "sigma_vector.h"
#include "stl_bitset_determinant.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace psi {
namespace forte {

/**
 * @brief The SigmaVectorDirect class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorDirect : public SigmaVector {
  public:
    SigmaVectorDirect(const std::vector<STLBitsetDeterminant>& space,
                      std::shared_ptr<FCIIntegrals> fci_ints);
    void compute_sigma(SharedVector sigma, SharedVector b);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    std::shared_ptr<FCIIntegrals> fci_ints_;
};
}
}

#endif // _sigma_vector_direct_h_
