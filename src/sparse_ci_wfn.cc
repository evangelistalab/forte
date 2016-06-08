#include "sparse_ci_wfn.h"

namespace psi{ namespace forte{

SparseCIWavefunction::SparseCIWavefunction( std::vector<STLBitsetDeterminant>& dets, std::vector<double>& cI ) 
{
    // The dimension of the wavefunction
    wfn_size_ = cI.size();

    // Take the determinants and coefficients and build the hash
    for( size_t I = 0; I < wfn_size_; ++I ){
        wfn_[dets[I]] = cI[I];
    }        
}

SparseCIWavefunction::SparseCIWavefunction(wfn_hash& wfn) : wfn_(wfn) 
{
    wfn_size_ = wfn.size();
}

wfn_hash& SparseCIWavefunction::wfn()
{
    return wfn_;
}

void SparseCIWavefunction::scale( double value )
{
    //Update each element with the scaled value
    for( auto& I : wfn_ ){
        I.second *= value;
    }
}

double SparseCIWavefunction::wfn_norm()
{
    double norm = 0.0;

    // Sum the squares
    for( auto& I : wfn_ ){
        norm += (I.second * I.second);
    }
    
    // Take the square root
    norm = sqrt(norm);

    return norm;
}

void SparseCIWavefunction::normalize()
{
    // Compute the norm
    double norm = this->wfn_norm();

    // Take the inverse;
    norm = 1.0/norm;

    // Scale the wavefunction
    this->scale( norm );
}

void SparseCIWavefunction::add( STLBitsetDeterminant& det, double value )
{
    wfn_[det] = value;
}

void SparseCIWavefunction::merge( SparseCIWavefunction& wfn )
{
    // Grab the new wavefunction hash
    wfn_hash& nwfn = wfn.wfn();

    for( auto& I : nwfn ){
        wfn_[I.first] = I.second;
    }
}

void SparseCIWavefunction::print()
{
    // This will store the determinants and coefficients
    std::vector<std::pair<double, STLBitsetDeterminant>> det_weights;

    for( auto& I : wfn_ ){
        det_weights.push_back( std::make_pair( std::fabs(I.second), I.first ));
    }

    // Sort by coefficient in decreasing order
    std::sort( det_weights.begin(), det_weights.end());
    std::reverse( det_weights.begin(), det_weights.end());

    size_t max_dets = std::min( 10, static_cast<int>(wfn_size_) );
    for( size_t I = 0; I < max_dets; ++I ){
        outfile->Printf("\n  %3zu  %9.6f %.9f %s",
                I,
                wfn_[det_weights[I].second],
                det_weights[I].first*det_weights[I].first,
                det_weights[I].second.str().c_str() );
    }
}

}}
