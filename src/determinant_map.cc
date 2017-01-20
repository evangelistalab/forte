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

#include "determinant_map.h"
#include <numeric>

namespace psi{ namespace forte{

DeterminantMap::DeterminantMap( std::vector<STLBitsetDeterminant>& dets ) 
{
    // The dimension of the wavefunction
    wfn_size_ = dets.size();

    // Take the determinants and coefficients and build the hash
    for( size_t I = 0; I < wfn_size_; ++I ){
        wfn_[dets[I]] = I;
    }        
}

DeterminantMap::DeterminantMap(){}

DeterminantMap::DeterminantMap(detmap& wfn) : wfn_(wfn) 
{
    wfn_size_ = wfn.size();
}

detmap& DeterminantMap::wfn_hash()
{
    return wfn_;
}


std::vector<STLBitsetDeterminant> DeterminantMap::determinants()
{
    std::vector<STLBitsetDeterminant> space;//( wfn_size_ );

    for( detmap::iterator it = wfn_.begin(); it != wfn_.end(); ++it ){
        space.push_back( it->first );
    } 
    return space;
}

double DeterminantMap::size()
{
    wfn_size_ = wfn_.size();
    return wfn_size_;
}


void DeterminantMap::add( STLBitsetDeterminant& det )
{
    wfn_[det] = wfn_size_;
    wfn_size_ = wfn_.size();
}

STLBitsetDeterminant DeterminantMap::get_det( size_t& value )
{
    // Iterate through map to find the right one
    // Possibly a faster way to do this?
    STLBitsetDeterminant det;
    
    for( detmap::iterator it = wfn_.begin(), endit = wfn_.end(); it != endit; ++it ){
        if( it->second == value ){
            det = it->first;
            break;
        }
    }
    return det;
}

}}
