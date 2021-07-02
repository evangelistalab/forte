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

#ifndef _sparse_hamiltonian_h_
#define _sparse_hamiltonian_h_

#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"

#include "sparse_ci/sparse_state_vector.h"
#include "sparse_ci/sparse_operator.h"
#include "sparse_ci/determinant_hashvector.h"

namespace forte {

class ActiveSpaceIntegrals;

/**
 * @brief The SparseHamiltonian class
 * This class implements an algorithm to apply the Hamiltonian to a StateVector object.
 */
class SparseHamiltonian {
  public:
    /// Constructor (requires the integrals)
    SparseHamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// @brief Compute the state H|state> using an algorithm that caches the elements of H
    /// This algorithm is useful when applying H repeatedly to the same state or in an
    /// iterative procedure
    /// This function applies only those elements of H that satisfy the condition:
    ///     |H_IJ C_J| > screen_thresh
    /// @param state the state to which the Hamiltonian will be applied
    /// @param screen_thresh a threshold to select which elements of H are applied to the state
    StateVector compute(const StateVector& state, double screen_thresh);

    /// @brief Compute the state H|state> using an on-the-fly algorithm that has no memory footprint
    /// This function applies only those elements of H that satisfy the condition:
    ///     |H_IJ C_J| > screen_thresh
    /// @param state the state to which the Hamiltonian will be applied
    /// @param screen_thresh a threshold to select which elements of H are applied to the state
    StateVector compute_on_the_fly(const StateVector& state, double screen_thresh);

    /// @return timings for this class    
    std::map<std::string, double> timings() const;

  private:
    /// Compute couplings for new determinants
    void compute_new_couplings(const std::vector<Determinant>& new_dets, double screen_thresh);
    /// Compute sigma using the couplings
    StateVector compute_sigma(const StateVector& state, double screen_thresh);

    /// The integral object
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;
    /// A map that holds the list of the determinants to which we apply H
    DeterminantHashVec state_hash_;
    /// A map that holds the list of the determinants obtained after applying H
    DeterminantHashVec sigma_hash_;
    /// A vector of determinant couplings
    std::map<Determinant,std::vector<std::pair<size_t, double>>> couplings_;
    // std::vector<std::tuple<size_t, size_t, double>> couplings_;
    /// A map that stores timing information
    std::map<std::string,double> timings_;
};

} // namespace forte

#endif // _sparse_hamiltonian_h_
