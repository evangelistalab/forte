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
    DeterminantMap( std::vector<STLBitsetDeterminant>& dets, std::vector<double>& cI, int nroot );

    /// Empty constructor
    DeterminantMap();

    /// Copy constructor
    DeterminantMap(detmap& wfn_, std::vector<double>& cI);

    /// @return The hash
    detmap& wfn();

    /// The coefficients
    std::vector<double> cI_; 

    /// Return the coefficients
    std::vector<double> coefficients();

    /// Return a coefficient
    double coefficient( size_t idx );

    /// Update the coefficients
    void update_coefficients( std::vector<double>& CI );

    /// Scale the wavefunction
    void scale( double value );

    /// Return the norm of the wavefunction
    double norm();

    /// Normalize the wavefunction
    void normalize();

    /// Add a determinant
    void add( STLBitsetDeterminant& det, double value );

    /// Merge two wavefunctions, overwriting the original in the case of conflicts
    void merge( DeterminantMap& wfn ); 

    /// Enlarge determinant space so that wfn is eigenfunction of total spin
    void enforce_spin_completeness();

    /// Print the most important determinants
    void print();

    /// Return the number of determinants
    double size();

    /// Return the number of roots
    int nroot();

protected:

    /// The dimension of the hash
    size_t wfn_size_;

    /// The number of roots
    int nroot_;

    /// A hash of (determinants,coefficients)
    detmap wfn_;
};

}}

#endif // _determinant_map_h_
