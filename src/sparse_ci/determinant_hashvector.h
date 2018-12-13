/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _determinant_hashvector_h_
#define _determinant_hashvector_h_

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"
#include "psi4/psi4-dec.h"
#include "sparse_ci/determinant.h"
#include "hash_vector.h"

namespace psi {
namespace forte {

/**
 * @brief A class to store sparse configuration interaction wave functions
 * Stores a hash of determinants.
 */

using det_hashvec = HashVector<Determinant, Determinant::Hash>;

class DeterminantHashVec {
  public:
    /// Default constructor
    DeterminantHashVec(std::vector<Determinant>& dets);

    /// Default constructor
    DeterminantHashVec(const std::vector<Determinant>& dets);

    /// Default constructor
    DeterminantHashVec(Determinant& det);

    /// Empty constructor
    DeterminantHashVec();

    /// Copy constructor
    DeterminantHashVec(const det_hashvec& wfn_);

    /// Move constructor
    DeterminantHashVec(det_hashvec&& wfn_);

    /// @return The hash
    const det_hashvec& wfn_hash() const;

    /// @return The hash
    det_hashvec& wfn_hash();

    /// Add a determinant
    void add(const Determinant& det);

    /// Return the number of determinants
    size_t size() const;

    // Clear hash
    void clear();

    // Return a specific determinant by reference
    const Determinant& get_det(const size_t value) const;

    // Return the index of a determinant
    size_t get_idx(const Determinant& det) const;

    // Return a vector of the determinants
    std::vector<Determinant> determinants() const;

    /// Return a vector of the determinants and their indices
    std::vector<std::pair<Determinant,size_t>> determinant_index_pairs() const;

    // Make this spin complete
    void make_spin_complete(int nmo);

    // Check if a determinant is in the wavefunction
    bool has_det(const Determinant& det) const;

    // Compute overlap between this and input wfn
    double overlap(std::vector<double>& det1_evecs, DeterminantHashVec& det2,
                   SharedMatrix det2_evecs, int root);

    // Compute overlap between this and input wfn
    double overlap(SharedMatrix det1_evecs, int root1, DeterminantHashVec& det2,
                   SharedMatrix det2_evecs, int root2);

    // Save most important subspace as this
    void subspace(DeterminantHashVec& dets, SharedMatrix evecs, std::vector<double>& new_evecs,
                  int dim, int root);

    // Merge a wavefunction into this
    void merge(DeterminantHashVec& dets);

    // Copy a wavefunctions
    void copy(DeterminantHashVec& dets);

    // Swap with another DeterminantHashVec object
    void swap(DeterminantHashVec& dets);

    // Swap with a det_hashvec object
    void swap(det_hashvec& dets);

  protected:
    /// A hashvector of determinants
    det_hashvec wfn_;
};
}
}

#endif // _determinant_hashvector_h_
