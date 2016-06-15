#include "determinant_map.h"
#include <numeric>

namespace psi{ namespace forte{

DeterminantMap::DeterminantMap( std::vector<STLBitsetDeterminant>& dets, std::vector<double>& cI ) 
{
    // The dimension of the wavefunction
    wfn_size_ = cI.size();
    cI_.resize( wfn_size_ );    

    // Take the determinants and coefficients and build the hash
    for( size_t I = 0; I < wfn_size_; ++I ){
        wfn_[dets[I]] = I;
        cI_[I] = cI[I];
    }        
}

DeterminantMap::DeterminantMap(detmap& wfn, std::vector<double>& cI) : wfn_(wfn) 
{
    wfn_size_ = wfn.size();
    for( auto& I : wfn ){
        cI_[I.second] = cI[I.second];
    }
}

detmap& DeterminantMap::wfn()
{
    return wfn_;
}

std::vector<double> DeterminantMap::coefficients()
{
    return cI_;
}

double DeterminantMap::get_size()
{
    wfn_size_ = cI_.size();
    return wfn_size_;
}

void DeterminantMap::scale( double value )
{
    //Update each element with the scaled value
    std::transform( cI_.begin(), cI_.end(), cI_.begin(), std::bind1st(std::multiplies<double>(), value ));
}

double DeterminantMap::wfn_norm()
{
    double norm = std::inner_product( cI_.begin(), cI_.end(), cI_.begin(), 0.0) ;

    // Take the square root
    norm = sqrt(norm);

    return norm;
}

void DeterminantMap::normalize()
{
    // Compute the norm
    double norm = this->wfn_norm();

    // Take the inverse;
    norm = 1.0/norm;

    // Scale the wavefunction
    this->scale( norm );
}

void DeterminantMap::add( STLBitsetDeterminant& det, double value )
{
    wfn_[det] = wfn_size_;
    cI_.resize(wfn_size_ + 1);
    cI_[wfn_size_] = value;
}

void DeterminantMap::merge( DeterminantMap& wfn )
{
    // Grab the new wavefunction hash
    detmap& nwfn = wfn.wfn();
    auto ncI = wfn.coefficients();

    for( auto& I : nwfn ){
        size_t idx = I.second + wfn_size_;
        wfn_[I.first] = idx;
        cI_[idx] = ncI[I.second];
    }
    wfn_size_ = cI_.size();
}

void DeterminantMap::print()
{
    // This will store the determinants and coefficients
    std::vector<std::pair<double, STLBitsetDeterminant>> det_weights;

    for( auto& I : wfn_ ){
        det_weights.push_back( std::make_pair( std::fabs(cI_[I.second]),I.first));
    }

    // Sort by coefficient in decreasing order
    std::sort( det_weights.begin(), det_weights.end());
    std::reverse( det_weights.begin(), det_weights.end());

    size_t max_dets = std::min( 10, static_cast<int>(wfn_size_) );
    for( size_t I = 0; I < max_dets; ++I ){
        outfile->Printf("\n  %3zu  %9.6f %.9f %s",
                I,
                cI_[ wfn_[det_weights[I].second] ],
                det_weights[I].first*det_weights[I].first,
                det_weights[I].second.str().c_str() );
    }
}

}}
