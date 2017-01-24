/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _determinant_map_h_
#define _determinant_map_h_

#include "stl_bitset_determinant.h"

namespace psi{ namespace forte{

/**
 * @brief A class to store sparse configuration interaction wave functions
 * Stores a hash of determinants.
 */

using detmap = det_hash<size_t>;

class DeterminantMap
{
public:

    /// Default constructor
    DeterminantMap( std::vector<STLBitsetDeterminant>& dets );

    /// Empty constructor
    DeterminantMap();

    /// Copy constructor
    DeterminantMap( detmap& wfn_ );

    /// @return The hash
    detmap& wfn_hash();

    /// Add a determinant
    void add( STLBitsetDeterminant& det );

    /// Return the number of determinants
    double size();

    // Clear hash
    void clear();

    // Return a specific determinant by value
    STLBitsetDeterminant get_det( size_t value );

    // Return the index of a determinant
    size_t get_idx( STLBitsetDeterminant& det );

    // Return a vector of the determinants
    std::vector<STLBitsetDeterminant> determinants();

    // Make this spin complete
    void make_spin_complete();

    // Check if a determinant is in the wavefunction
    bool has_det( STLBitsetDeterminant& det );

    // Compute overlap between this and input wfn
    double overlap( std::vector<double>& det1_evecs, DeterminantMap& det2, SharedMatrix det2_evecs, int root );

    // Compute overlap between this and input wfn
    double overlap( SharedMatrix det1_evecs, DeterminantMap& det2, SharedMatrix det2_evecs, int root );

    // Save most important subspace as this
    void subspace( DeterminantMap& dets, SharedMatrix evecs, std::vector<double>& new_evecs, int dim, int root);

protected:

    /// The dimension of the hash
    size_t wfn_size_;

    /// The number of roots
    int nroot_;

    /// The multiplicity
    int multiplicity_;    

    /// A hash of (determinants,coefficients)
    detmap wfn_;
};

}}

#endif // _determinant_map_h_
