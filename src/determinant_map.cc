#include "determinant_map.h"
#include <numeric>

namespace psi{ namespace forte{

DeterminantMap::DeterminantMap( std::vector<STLBitsetDeterminant>& dets, std::vector<double>& cI, int nroot ) 
{
    // The dimension of the wavefunction
    wfn_size_ = cI.size();
    cI_.resize( wfn_size_ );    

    // Take the determinants and coefficients and build the hash
    for( size_t I = 0; I < wfn_size_; ++I ){
        wfn_[dets[I]] = I;
        cI_[I] = cI[I];
    }        

    nroot_ = nroot;
}

DeterminantMap::DeterminantMap(){}

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

int DeterminantMap::nroot()
{
    return nroot_;
}

std::vector<double> DeterminantMap::coefficients()
{
    return cI_;
}

/*
std::vector<STLBitsetDeterminant> DeterminantMap::determinants()
{
    std::vector<STLBitsetDeterminant> space;//( wfn_size_ );

    for( detmap::iterator it = wfn_.begin; it != wfn_.end(); ++it ){
    } 
}
*/

double DeterminantMap::coefficient( size_t value )
{
    return cI_[value];
}

double DeterminantMap::size()
{
    wfn_size_ = cI_.size();
    return wfn_size_;
}

void DeterminantMap::scale( double value )
{
    //Update each element with the scaled value
    std::transform( cI_.begin(), cI_.end(), cI_.begin(), std::bind1st(std::multiplies<double>(), value ));
}

double DeterminantMap::norm()
{
    double norm = std::inner_product( cI_.begin(), cI_.end(), cI_.begin(), 0.0) ;

    // Take the square root
    norm = sqrt(norm);

    return norm;
}

void DeterminantMap::normalize()
{
    // Compute the norm
    double norm = this->norm();

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

void DeterminantMap::enforce_spin_completeness()
{
    int nmo = wfn_.begin()->first.nmo_;
    DeterminantMap new_dets;

    std::vector<size_t> closed(nmo, 0);
    std::vector<size_t> open(nmo, 0);
    std::vector<size_t> open_bits(nmo, 0);
    
    for( auto& det_pair : wfn_ ){
        const STLBitsetDeterminant& det = det_pair.first;
    
        for( int i = 0; i < nmo; ++i ){
            closed[i] = open[i] = 0;
            open_bits[i] = false;
        }

        int naopen = 0;
        int nbopen = 0;
        int nclosed = 0;

        for( int i = 0; i < nmo; ++i ){
            if( det.get_alfa_bit(i) and ( not det.get_beta_bit(i))){
                open[naopen + nbopen] = i;
                naopen++;
            } else if ((not det.get_alfa_bit(i)) and det.get_beta_bit(i)){
                open[naopen + nbopen] = i;
                nbopen ++;
            } else if ( det.get_alfa_bit(i) and det.get_beta_bit(i)){
                closed[nclosed] = i;
                nclosed++;
            }
        }

        if( naopen + nbopen == 0 ) continue;

        for( int i = 0; i < nbopen; ++i) open_bits[i] = false;
        for( int i = nbopen; i < naopen + nbopen; ++i ) open_bits[i] = true;
        
        do{
            STLBitsetDeterminant new_det;
            for( int c = 0; c < nclosed; ++c ){
                new_det.set_alfa_bit(closed[c],true);
                new_det.set_beta_bit(closed[c],true);
            }
            for( int o = 0; o < naopen + nbopen; ++o ){
                if( open_bits[o] ){
                    new_det.set_alfa_bit(open[o], true);
                }else{
                    new_det.set_beta_bit(open[o], true);
                }
            }
            new_dets.add( new_det, 0.0 );
        } while ( std::next_permutation(open_bits.begin(), open_bits.begin() + naopen + nbopen));
    }
    
    size_t old_size = wfn_size_;
    this->merge(new_dets);
    outfile->Printf("\n  Added %zu determinants to acheive spin-completeness.", wfn_size_ - old_size);

}

}}
