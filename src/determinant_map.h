/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
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

    // Return a specific determinant by value
    STLBitsetDeterminant get_det( size_t& value );

    // Return a vector of the determinants
    std::vector<STLBitsetDeterminant> determinants();

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
