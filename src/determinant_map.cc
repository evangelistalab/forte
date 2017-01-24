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

bool descending_pair( const std::pair<double,size_t> p1, const std::pair<double,size_t> p2 ){ return p1 > p2; }

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

void DeterminantMap::clear()
{
    wfn_.clear();
    wfn_size_ = wfn_.size();
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

STLBitsetDeterminant DeterminantMap::get_det( size_t value )
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

size_t DeterminantMap::get_idx( STLBitsetDeterminant& det )
{
    return wfn_[det];
}

void DeterminantMap::make_spin_complete()
{
   
}

bool DeterminantMap::has_det( STLBitsetDeterminant& det )
{
    bool has = false;

    if( wfn_.count(det) != 0 ){
        has = true;
    }
    return has;
}

double DeterminantMap::overlap( std::vector<double>& det1_evecs, DeterminantMap& det2, SharedMatrix det2_evecs, int root)
{
    double overlap = 0.0;
    for( detmap::iterator it = wfn_.begin, endit = wfn_.end(); it != endit; ++it ){
        if( det2.has_det( it.first ){
            size_t idx = det2.get_idx( it.first );
            overlap += det1_evecs[it.second] * det2_evecs->get( idx, root ); 
        }
    }
    return overlap;
}

double DeterminantMap::overlap( SharedMatrix det1_evecs, int root1, DeterminantMap& det2, SharedMatrix det2_evecs, int root2)
{
    double overlap = 0.0;
    for( detmap::iterator it = wfn_.begin, endit = wfn_.end(); it != endit; ++it ){
        if( det2.has_det( it.first ){
            size_t idx = det2.get_idx( it.first );
            overlap += det1_evecs->get(it.second, root1) * det2_evecs->get( idx, root2 ); 
        }
    }
    return overlap;
}

void DeterminantMap::subspace( DeterminantMap& dets, SharedMatrix evecs, std::vector<double>& new_evecs, int dim, int root )
{
    // Clear current wfn
    this.clear();
    new_evecs.reset(dim);

    std::vector<std::pair<double, size_t>> det_weights;
    for( size_t I = 0, maxI = dets.size(); I < maxI; ++I ){
        det_weights.push_back( evecs->get(I, root), I );
    }
    std::sort( det_weights.begin(), det_weights.end(), descending_pair);

    // Build this wfn with most important subset
    for( size_t I = 0; I < dim; ++I ){
        STLBitsetDeterminant& detI = dets.get_det((det_weights[I].second));
        this.add(detI);
        new_evecs[I] = evecs->get(det_weights[I].second, root); 
    } 
}

}}
