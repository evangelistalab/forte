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
