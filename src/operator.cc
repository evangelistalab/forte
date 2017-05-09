/*
 *@BEGIN LICENSE
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

#include "operator.h"
#include "forte-def.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

namespace psi {
namespace forte {

WFNOperator::WFNOperator(std::vector<int> symmetry)
{
    mo_symmetry_ = symmetry;
}

WFNOperator::WFNOperator() {}

void WFNOperator::set_quiet_mode( bool mode )
{
    quiet_ = mode;
}


void WFNOperator::initialize(std::vector<int>& symmetry) {
    mo_symmetry_ = symmetry;
}

std::vector<std::pair<std::vector<size_t>, std::vector<double>>> WFNOperator::build_H_sparse( const DeterminantMap& wfn )
{
Timer build;
    size_t size = wfn.size();
    std::vector<std::pair<std::vector<size_t>, std::vector<double>>> H_sparse(size);
    size_t n_nonzero = 0;

    const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();

    // Add diagonal
    for( size_t I = 0; I < size; ++I ){
        H_sparse[I].first.push_back(I);
        H_sparse[I].second.push_back(dets[I].energy());
        n_nonzero++;
    }

    for( size_t K = 0, max_K = a_list_.size(); K < max_K; ++K ){
            
        const std::vector<std::pair<size_t,short>>& c_dets = a_list_[K];
        for( auto& detI : c_dets ){
            size_t idx = detI.first;
            short p = std::abs(detI.second) - 1;
            double sign_p = detI.second > 0.0 ? 1.0 : -1.0;
            H_sparse[idx].first.reserve(c_dets.size());
            H_sparse[idx].second.reserve(c_dets.size());
            for( auto& detJ : c_dets ){
                short q = std::abs(detJ.second) - 1;
                if( p != q ){
                    size_t J = detJ.first;
                    double sign_q = detJ.second > 0.0 ? 1.0 : -1.0;
                    double HIJ = sign_p * sign_q * dets[J].slater_rules_single_alpha_abs(p,q);
                    H_sparse[idx].first.push_back(J);
                    H_sparse[idx].second.push_back(HIJ);
                    n_nonzero++;
                }
            }        
        }
    }
    for( size_t K = 0, max_K = b_list_.size(); K < max_K; ++K ){
            
        const std::vector<std::pair<size_t,short>>& c_dets = b_list_[K];
        for( auto& detI : c_dets ){
            size_t idx = detI.first;
            short p = std::abs(detI.second) - 1;
            double sign_p = detI.second > 0.0 ? 1.0 : -1.0;
            H_sparse[idx].first.reserve(c_dets.size());
            H_sparse[idx].second.reserve(c_dets.size());
            for( auto& detJ : c_dets ){
                short q = std::abs(detJ.second) - 1;
                if( p != q ){
                    size_t J = detJ.first;
                    double sign_q = detJ.second > 0.0 ? 1.0 : -1.0;
                    double HIJ = sign_p * sign_q * dets[J].slater_rules_single_beta_abs(p,q);
                    H_sparse[idx].first.push_back(J);
                    H_sparse[idx].second.push_back(HIJ);
                    n_nonzero++;
                }
            }        
        }
    }
    for( size_t K = 0, max_K = aa_list_.size(); K < max_K; ++K ){
            
        const std::vector<std::tuple<size_t,short,short>>& c_dets = aa_list_[K];
        for( auto& detI : c_dets ){
            size_t idx = std::get<0>(detI);
            short p = std::abs(std::get<1>(detI)) - 1;
            short q = std::get<2>(detI);
            double sign_p = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
            H_sparse[idx].first.reserve(c_dets.size());
            H_sparse[idx].second.reserve(c_dets.size());
            for( auto& detJ : c_dets ){
                short r = std::abs(std::get<1>(detJ)) - 1;
                short s = std::get<2>(detJ);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    size_t J = std::get<0>(detJ);
                    double sign_q = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    double HIJ = sign_p * sign_q * STLBitsetDeterminant::fci_ints_->tei_aa(p,q,r,s);
                    H_sparse[idx].first.push_back(J);
                    H_sparse[idx].second.push_back(HIJ);
                    n_nonzero++;
                }
            }        
        }
    }
    for( size_t K = 0, max_K = bb_list_.size(); K < max_K; ++K ){
            
        const std::vector<std::tuple<size_t,short,short>>& c_dets = bb_list_[K];
        for( auto& detI : c_dets ){
            size_t idx = std::get<0>(detI);
            short p = std::abs(std::get<1>(detI)) - 1;
            short q = std::get<2>(detI);
            double sign_p = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
            H_sparse[idx].first.reserve(c_dets.size());
            H_sparse[idx].second.reserve(c_dets.size());
            for( auto& detJ : c_dets ){
                short r = std::abs(std::get<1>(detJ)) - 1;
                short s = std::get<2>(detJ);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    size_t J = std::get<0>(detJ);
                    double sign_q = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    double HIJ = sign_p * sign_q * STLBitsetDeterminant::fci_ints_->tei_bb(p,q,r,s);
                    H_sparse[idx].first.push_back(J);
                    H_sparse[idx].second.push_back(HIJ);
                    n_nonzero++;
                }
            }        
        }
    }
    for( size_t K = 0, max_K = ab_list_.size(); K < max_K; ++K ){
            
        const std::vector<std::tuple<size_t,short,short>>& c_dets = ab_list_[K];
        for( auto& detI : c_dets ){
            size_t idx = std::get<0>(detI);
            short p = std::abs(std::get<1>(detI)) - 1;
            short q = std::get<2>(detI);
            double sign_p = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
            H_sparse[idx].first.reserve(c_dets.size());
            H_sparse[idx].second.reserve(c_dets.size());
            for( auto& detJ : c_dets ){
                short r = std::abs(std::get<1>(detJ)) - 1;
                short s = std::get<2>(detJ);
                if ( (p != r) and (q != s) ){
                    size_t J = std::get<0>(detJ);
                    double sign_q = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    double HIJ = sign_p * sign_q * STLBitsetDeterminant::fci_ints_->tei_ab(p,q,r,s);
                    H_sparse[idx].first.push_back(J);
                    H_sparse[idx].second.push_back(HIJ);
                    n_nonzero++;
                }
            }        
        }
    }

/*#pragma omp parallel reduction(+:n_nonzero)
{
        int num_thread = omp_get_max_threads();
        int tid = omp_get_thread_num();
        size_t bin_size = size / num_thread;
        bin_size += (tid < (size % num_thread)) ? 1 : 0;
        size_t start_idx = (tid < (size % num_thread))
                            ? tid * bin_size
                            : (size % num_thread) * (bin_size +1) +
                                (tid - (size % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;
        const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();

        for(size_t J = start_idx; J < end_idx; ++J ){
            size_t id = 0;
            std::vector<size_t> ids(1);           
            std::vector<double> H_vals(1);            

            //Diagonal term first
            H_vals[id] = dets[J].energy();
            ids[id] = J;
            id++;

            for( auto& aJ_mo_sign : a_ann_list_[J]){
                const size_t aJ_add = aJ_mo_sign.first;
                const size_t p = std::abs( aJ_mo_sign.second ) - 1;
                double sign_p = aJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                for( auto& aaJ_mo_sign : a_cre_list_[aJ_add] ){

                    const size_t q = std::abs( aaJ_mo_sign.second ) - 1;
                    if( p != q ){
                        const size_t I = aaJ_mo_sign.first;
                        double sign_q = aaJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                        
                        const double HIJ = dets[I].slater_rules_single_alpha_abs(p,q) *
                                            sign_p * sign_q;
                        
                        ids.resize( id+1 );
                        H_vals.resize( id+1 );
                        ids[id] = I;
                        H_vals[id] = HIJ; 
                        id++;
                    }
                }
            }
 
            for( auto& bJ_mo_sign : b_ann_list_[J]){
                const size_t bJ_add = bJ_mo_sign.first;
                const size_t p = std::abs( bJ_mo_sign.second ) - 1;
                double sign_p = bJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                for( auto& bbJ_mo_sign : b_cre_list_[bJ_add] ){

                    const size_t q = std::abs( bbJ_mo_sign.second ) - 1;
                    if( p != q ){
                        const size_t I = bbJ_mo_sign.first;
                        double sign_q = bbJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                        
                        const double HIJ = dets[I].slater_rules_single_beta_abs(p,q) *
                                            sign_p * sign_q;
                        
                        ids.resize( id+1 );
                        H_vals.resize( id+1 );
                        ids[id] = I;
                        H_vals[id] = HIJ; 
                        id++;
                    }
                }
            }
           for( auto& aaJ_mo_sign : aa_ann_list_[J] ){
                const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
                const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
                const size_t q = std::get<2>(aaJ_mo_sign);
                for( auto& aaaJ_mo : aa_cre_list_[aaJ_add] ){

                    const size_t r = std::abs(std::get<1>(aaaJ_mo)) - 1;
                    const size_t s = std::get<2>(aaaJ_mo);
                    if ((p != r) and (q != s) and (p != s) and (q != r)){
                        const size_t I = std::get<0>(aaaJ_mo);
                        const double sign_rs = std::get<1>(aaaJ_mo) > 0.0 ? 1.0 : -1.0;
                        const double HIJ = sign_pq * sign_rs * 
                            STLBitsetDeterminant::fci_ints_->tei_aa(p,q,r,s);

                        ids.resize( id+1 );
                        H_vals.resize( id+1 );
                        ids[id] = I;
                        H_vals[id] = HIJ; 
                        id++;
                    }
                }
            }

            for( auto& bbJ_mo_sign : bb_ann_list_[J] ){
                const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
                const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
                const size_t q = std::get<2>(bbJ_mo_sign);
                for( auto& bbbJ_mo : bb_cre_list_[bbJ_add] ){

                    const size_t r = std::abs(std::get<1>(bbbJ_mo)) - 1;
                    const size_t s = std::get<2>(bbbJ_mo);
                    if ((p != r) and (q != s) and (p != s) and (q != r)){
                        const size_t I = std::get<0>(bbbJ_mo);
                        const double sign_rs = std::get<1>(bbbJ_mo) > 0.0 ? 1.0 : -1.0;
                        const double HIJ = sign_pq * sign_rs * 
                            STLBitsetDeterminant::fci_ints_->tei_bb(p,q,r,s);

                        ids.resize( id+1 );
                        H_vals.resize( id+1 );
                        ids[id] = I;
                        H_vals[id] = HIJ; 
                        id++;
                    }
                }
            }
            for( auto& abJ_mo_sign : ab_ann_list_[J] ){
                const size_t abJ_add = std::get<0>(abJ_mo_sign);
                const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
                const size_t q = std::get<2>(abJ_mo_sign);
                for( auto& abbJ_mo : ab_cre_list_[abJ_add] ){

                    const size_t r = std::abs(std::get<1>(abbJ_mo)) - 1;
                    const size_t s = std::get<2>(abbJ_mo);
                    if ( (p != r) and (q != s) ){
                        const size_t I = std::get<0>(abbJ_mo);
                        const double sign_rs = std::get<1>(abbJ_mo) > 0.0 ? 1.0 : -1.0;
                        const double HIJ = sign_pq * sign_rs * 
                            STLBitsetDeterminant::fci_ints_->tei_ab(p,q,r,s);
                        ids.resize( id+1 );
                        H_vals.resize( id+1 );
                        ids[id] = I;
                        H_vals[id] = HIJ; 
                        id++;
                        
                    }
                }
            }
            ids.shrink_to_fit();
            H_vals.shrink_to_fit();
            n_nonzero += ids.size();
            H_sparse[J] = std::make_pair( ids, H_vals );
        }
    }
*/
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building H:   %1.6f s", build.get());
        outfile->Printf("\n  H contains %zu nonzero elements (%1.3f MB)", n_nonzero, (n_nonzero * 16.0)/(1024*1024) );
    }


//    for( size_t J = 0; J < size; ++J){
//   // for( size_t J = 0; J < 2; ++J){
//        auto& idx = H_sparse[J].first;
//        auto& val = H_sparse[J].second;
//        for(int i = 0; i < idx.size(); ++i ){
//            outfile->Printf("\n  (%zu, %zu) : %1.8f", J, idx[i], val[i]);
//        } 
//    }


    return H_sparse;
}


double WFNOperator::s2(DeterminantMap& wfn, SharedMatrix& evecs, int root) {
    double S2 = 0.0;
    const det_hash<size_t>& wfn_map = wfn.wfn_hash();

    for (auto& det : wfn_map) {
        // Compute diagonal
        // PhiI = PhiJ
        STLBitsetDeterminant PhiI = det.first;
        int npair = PhiI.npair();
        int na = PhiI.get_alfa_occ().size();
        int nb = PhiI.get_beta_occ().size();
        double ms = 0.5 * static_cast<double>(na - nb);
        S2 += (ms * ms + ms + static_cast<double>(nb) -
               static_cast<double>(npair)) *
              evecs->get(det.second, root) * evecs->get(det.second, root);
        if ((npair == nb) or (npair == na))
            continue;
    }

        // Loop directly through all determinants with
        // spin-coupled electrons, i.e:
        // |PhiI> = a+(qa) a+(pb) a-(qb) a-(pa) |PhiJ>

    for( size_t K = 0, max_K = ab_list_.size(); K < max_K; ++K ){ 
        const std::vector<std::tuple<size_t, short,short>>& c_dets = ab_list_[K];
        for (auto& detI : c_dets) {
            const size_t I = std::get<0>(detI);
            double sign_pq = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
            short p = std::fabs(std::get<1>(detI)) - 1;
            short q = std::get<2>(detI);
            if (p == q)
                continue;
            for (auto& detJ : c_dets) {
                const size_t J = std::get<0>(detJ);
                if( I == J ) continue;
                double sign_rs = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                short r = std::fabs(std::get<1>(detJ)) - 1;
                short s = std::get<2>(detJ);
                if ((r != s) and (p == s) and (q == r)) {
                    sign_pq *= sign_rs;
                    S2 -= sign_pq * evecs->get(J, root) *
                          evecs->get(J, root);
                }
            }
        }
    }

    S2 = std::fabs(S2);
    return S2;
}

void WFNOperator::add_singles(DeterminantMap& wfn) {
    det_hash<size_t>& wfn_map = wfn.wfn_hash();

    DeterminantMap external;
    // Loop through determinants, generate singles and add them to the wfn
    // Alpha excitations
    for (auto& I : wfn_map) {
        STLBitsetDeterminant det = I.first;
        std::vector<int> aocc = det.get_alfa_occ();
        std::vector<int> avir = det.get_alfa_vir();

        for (int i = 0, noalpha = aocc.size(); i < noalpha; ++i) {
            int ii = aocc[i];
            for (int a = 0, nvalpha = avir.size(); a < nvalpha; ++a) {
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    auto new_det = det;
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    external.add(det);
                }
            }
        }
        // }

        // Beta excitations
        // for( auto& I : wfn_map ){
        //     STLBitsetDeterminant det = I.first;
        std::vector<int> bocc = det.get_beta_occ();
        std::vector<int> bvir = det.get_beta_vir();

        for (int i = 0, nobeta = bocc.size(); i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0, nvbeta = bvir.size(); a < nvbeta; ++a) {
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    det.set_beta_bit(ii, false);
                    det.set_beta_bit(aa, true);
                    external.add(det);
                    det.set_beta_bit(ii, true);
                    det.set_beta_bit(aa, false);
                }
            }
        }
    }
    wfn.merge(external);
}

void WFNOperator::add_doubles(DeterminantMap& wfn) {
    const det_hash<size_t>& wfn_map = wfn.wfn_hash();

    DeterminantMap external;    

    for (auto& I : wfn_map) {
        STLBitsetDeterminant det = I.first;
        std::vector<int> aocc = det.get_alfa_occ();
        std::vector<int> bocc = det.get_beta_occ();
        std::vector<int> avir = det.get_alfa_vir();
        std::vector<int> bvir = det.get_beta_vir();

        int noalfa = aocc.size();
        int nvalfa = avir.size();
        int nobeta = bocc.size();
        int nvbeta = bvir.size();

        // alpha-alpha
        for (int i = 0; i < noalfa; ++i) {
            int ii = aocc[i];
            for (int j = i + 1; j < noalfa; ++j) {
                int jj = aocc[j];
                for (int a = 0; a < nvalfa; a++) {
                    int aa = avir[a];
                    for (int b = a + 1; b < nvalfa; ++b) {
                        int bb = avir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
                             mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                            det.set_alfa_bit(ii, false);
                            det.set_alfa_bit(jj, false);
                            det.set_alfa_bit(aa, true);
                            det.set_alfa_bit(bb, true);
                            external.add(det);
                            det.set_alfa_bit(aa, false);
                            det.set_alfa_bit(bb, false);
                            det.set_alfa_bit(ii, true);
                            det.set_alfa_bit(jj, true);
                        }
                    }
                }
            }
        }

        // beta-beta
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int j = i + 1; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < nvbeta; a++) {
                    int aa = bvir[a];
                    for (int b = a + 1; b < nvbeta; ++b) {
                        int bb = bvir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
                             mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                            det.set_beta_bit(ii, false);
                            det.set_beta_bit(jj, false);
                            det.set_beta_bit(aa, true);
                            det.set_beta_bit(bb, true);
                            external.add(det);
                            det.set_beta_bit(aa, false);
                            det.set_beta_bit(bb, false);
                            det.set_beta_bit(ii, true);
                            det.set_beta_bit(jj, true);
                        }
                    }
                }
            }
        }

        // alfa-beta
        for (int i = 0; i < noalfa; ++i) {
            int ii = aocc[i];
            for (int j = 0; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < nvalfa; a++) {
                    int aa = avir[a];
                    for (int b = 0; b < nvbeta; ++b) {
                        int bb = bvir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
                             mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                            det.set_alfa_bit(ii, false);
                            det.set_beta_bit(jj, false);
                            det.set_alfa_bit(aa, true);
                            det.set_beta_bit(bb, true);
                            external.add(det);
                            det.set_alfa_bit(aa, false);
                            det.set_beta_bit(bb, false);
                            det.set_alfa_bit(ii, true);
                            det.set_beta_bit(jj, true);
                        }
                    }
                }
            }
        }
    }
    wfn.merge(external);
}

void WFNOperator::build_strings(DeterminantMap& wfn)
{
    beta_strings_.clear();
    alpha_strings_.clear();
    alpha_a_strings_.clear();
    
    // First build a map from beta strings to determinants
    const det_hash<size_t>& wfn_map = wfn.wfn_hash();
    {
        det_hash<size_t> beta_str_hash;
        size_t nbeta = 0;
        for( auto& I : wfn_map ){
            // Grab mutable copy of determinant
            STLBitsetDeterminant detI = I.first;
            detI.zero_spin(0);
    
            det_hash<size_t>::iterator it = beta_str_hash.find(detI);
            size_t b_add;
            if( it == beta_str_hash.end() ){
                b_add = nbeta;
                beta_str_hash[detI] = b_add;
                nbeta++;
            }else{
                b_add = it->second;
            }
            beta_strings_.resize(nbeta);
            beta_strings_[b_add].push_back(I.second);
        }
    }
    
    {
        det_hash<size_t> alfa_str_hash;
        size_t nalfa = 0;
        for( auto& I : wfn_map ){
            // Grab mutable copy of determinant
            STLBitsetDeterminant detI = I.first;
            detI.zero_spin(1);
    
            det_hash<size_t>::iterator it = alfa_str_hash.find(detI);
            size_t a_add;
            if( it == alfa_str_hash.end() ){
                a_add = nalfa;
                alfa_str_hash[detI] = a_add;
                nalfa++;
            }else{
                a_add = it->second;
            }
            alpha_strings_.resize(nalfa);
            alpha_strings_[a_add].push_back(I.second);
        }
    }
    // Next build a map from annihilated alpha strings to determinants
    det_hash<size_t> alfa_str_hash;
    size_t naalpha = 0;
    for( auto& I : wfn_map ){
        // Grab mutable copy of determinant
        STLBitsetDeterminant detI = I.first;
        detI.zero_spin(1);
        const std::vector<int>& aocc = detI.get_alfa_occ();
        for( int i = 0, nalfa = aocc.size(); i < nalfa; ++i ){
            int ii = aocc[i];
            STLBitsetDeterminant ann_det(detI);
            ann_det.set_alfa_bit(ii,false);
            
            size_t a_add;
            det_hash<size_t>::iterator it = alfa_str_hash.find(ann_det);
            if( it == alfa_str_hash.end() ){
                a_add = naalpha;
                alfa_str_hash[ann_det] = a_add;
                naalpha++;
            } else {
                a_add = it->second;
            }
            alpha_a_strings_.resize(naalpha);
            alpha_a_strings_[a_add].push_back(std::make_pair(ii,I.second));
        }
    } 
}

void WFNOperator::op_s_lists(DeterminantMap& wfn) {

    // Get a reference to the determinants
    const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();
Timer ann;
    for( size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b ){
        size_t na_ann = 0; 
        std::vector<std::vector<std::pair<size_t,short>>> tmp;
        std::vector<size_t>& c_dets = beta_strings_[b];
        det_hash<int> map_a_ann;
        for( size_t I = 0, maxI=c_dets.size(); I < maxI; ++I ){
            size_t index = c_dets[I];
            const STLBitsetDeterminant& detI = dets[index];
            const std::vector<int>& aocc = detI.get_alfa_occ(); 
            int noalfa = aocc.size();
               
            for( int i = 0; i < noalfa; ++i ){
                int ii = aocc[i];
                STLBitsetDeterminant detJ(detI);
                detJ.set_alfa_bit(ii, false);
                double sign = detI.slater_sign_alpha(ii);
                size_t detJ_add;
                auto search = map_a_ann.find(detJ);
                if( search == map_a_ann.end() ){
                    detJ_add = na_ann;
                    map_a_ann[detJ] = na_ann;
                    na_ann++;
                    tmp.resize(na_ann);
                }else{
                    detJ_add = search->second;
                }
                tmp[detJ_add].push_back(std::make_pair( index, sign > 0.0 ? (ii+1) : (-ii-1) )) ;
            }
        }
        //size_t idx = 0;
        for( auto& vec : tmp ){
            if( vec.size() > 1 ){
                a_list_.push_back(vec);
              //  a_list_[idx] = vec;
              //  idx++;
            }
        } 
    }

    if( !quiet_ ){
        outfile->Printf("\n  Time spent building a_list   %1.6f s", ann.get());
    }
Timer bnn;
    for( size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a ){
        size_t nb_ann = 0; 
        std::vector<std::vector<std::pair<size_t,short>>> tmp;
        std::vector<size_t>& c_dets = alpha_strings_[a];
        det_hash<int> map_b_ann;
        for( size_t I = 0, maxI=c_dets.size(); I < maxI; ++I ){
            size_t index = c_dets[I];
            const STLBitsetDeterminant& detI = dets[index];
            const std::vector<int>& bocc = detI.get_beta_occ(); 
            int nobeta = bocc.size();
               
            for( int i = 0; i < nobeta; ++i ){
                int ii = bocc[i];
                STLBitsetDeterminant detJ(detI);
                detJ.set_beta_bit(ii, false);
                double sign = detI.slater_sign_beta(ii);
                size_t detJ_add;
                auto search = map_b_ann.find(detJ);
                if( search == map_b_ann.end() ){
                    detJ_add = nb_ann;
                    map_b_ann[detJ] = nb_ann;
                    nb_ann++;
                    tmp.resize(nb_ann);
                }else{
                    detJ_add = search->second;
                }
                tmp[detJ_add].push_back( std::make_pair( index, sign > 0.0 ? (ii+1) : (-ii-1) ) );
            }
        }
        for( auto& vec : tmp ){
            if( vec.size() > 1 ){
                b_list_.push_back(vec);
            }
        } 
    }
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building b_list   %1.6f s", bnn.get());
    }
}

void WFNOperator::op_lists(DeterminantMap& wfn) {
    size_t ndets = wfn.size();
    a_ann_list_.resize(ndets);
    b_ann_list_.resize(ndets);
    const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();
    // Generate alpha coupling list
Timer ann;
    {
        size_t na_ann = 0;
        for( size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b){
            std::vector<size_t>& c_dets = beta_strings_[b];
            size_t max_I = c_dets.size();
            det_hash<int> map_a_ann;
            for( size_t I = 0; I < max_I; ++I){
                const STLBitsetDeterminant& detI = dets[c_dets[I]];
                const std::vector<int>& aocc = detI.get_alfa_occ();
                int noalfa = aocc.size(); 
                
                std::vector<std::pair<size_t, short>> a_ann(noalfa);
                for( int i = 0; i < noalfa; ++i ){
                    int ii = aocc[i];
                    STLBitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii,false);
                    double sign = detI.slater_sign_alpha(ii);

                    det_hash<int>::iterator it = map_a_ann.find(detJ);
                    size_t detJ_add;
                    if (it == map_a_ann.end()) {
                        detJ_add = na_ann;
                        map_a_ann[detJ] = na_ann;
                        na_ann++;
                    } else {
                        detJ_add = it->second;
                    }
                    a_ann[i] = std::make_pair(detJ_add,(sign > 0.0) ? (ii + 1) : (-ii - 1));
                }
                a_ann_list_[c_dets[I]] = a_ann;
            }    
        } 
        a_ann_list_.shrink_to_fit();
        a_cre_list_.resize(na_ann);
    }
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building a_ann_list   %1.6f s", ann.get());
    }
    // Generate beta coupling list
Timer bnn;
    {
        size_t nb_ann = 0;
        for( size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a ){
            std::vector<size_t>& c_dets = alpha_strings_[a];
            size_t max_I = c_dets.size();
            det_hash<int> map_b_ann;
            for( size_t I = 0; I < max_I; ++I){
                int idx = c_dets[I];
                const STLBitsetDeterminant& detI = dets[idx];
                std::vector<int> bocc = detI.get_beta_occ();
                int nobeta = bocc.size(); 

                std::vector<std::pair<size_t, short>> b_ann(nobeta);

                for (int i = 0; i < nobeta; ++i) {
                    int ii = bocc[i];
                    STLBitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(ii, false);

                    double sign = detI.slater_sign_beta(ii);

                    det_hash<int>::iterator it = map_b_ann.find(detJ);
                    size_t detJ_add;
                    // Add detJ to map if it isn't there
                    if (it == map_b_ann.end()) {
                        detJ_add = nb_ann;
                        map_b_ann[detJ] = nb_ann;
                        nb_ann++;
                    } else {
                        detJ_add = it->second;
                    }
                    b_ann[i] = std::make_pair(detJ_add,(sign > 0.0) ? (ii + 1) : (-ii - 1));
                }
                b_ann_list_[idx] = b_ann;
            }
        }
        b_ann_list_.shrink_to_fit();
        b_cre_list_.resize(nb_ann);
    }
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building b_ann_list   %1.6f s", bnn.get());
    }

    for (size_t I = 0, max_I = a_ann_list_.size(); I < max_I; ++I) {
        const std::vector<std::pair<size_t, short>>& a_ann = a_ann_list_[I];
        for (const std::pair<size_t, short>& J_sign : a_ann) {
            size_t J = J_sign.first;
            short sign = J_sign.second;
            a_cre_list_[J].push_back(std::make_pair(I, sign));
        }
    }
    for (size_t I = 0, max_I = b_ann_list_.size(); I < max_I; ++I) {
        const std::vector<std::pair<size_t, short>>& b_ann = b_ann_list_[I];
        for (const std::pair<size_t, short>& J_sign : b_ann) {
            size_t J = J_sign.first;
            short sign = J_sign.second;
            b_cre_list_[J].push_back(std::make_pair(I, sign));
        }
    }
    a_cre_list_.shrink_to_fit();
    b_cre_list_.shrink_to_fit();

}

void WFNOperator::tp_s_lists(DeterminantMap& wfn) {
    
    const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();
    // Generate alpha-alpha coupling list
Timer aa;
    {
        for( size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b ){
            size_t naa_ann = 0;
            std::vector<std::vector<std::tuple<size_t,short,short>>> tmp;
            det_hash<int> map_aa_ann;
            std::vector<size_t> c_dets = beta_strings_[b];
            size_t max_I = c_dets.size();
            for( size_t I = 0; I < max_I; ++I ){
                size_t idx = c_dets[I];
                STLBitsetDeterminant detI = dets[idx];
                std::vector<int> aocc = detI.get_alfa_occ();
                int noalfa = aocc.size();

                for (int i = 0; i < noalfa; ++i) {
                    for (int j = i + 1; j < noalfa; ++j) {
                        int ii = aocc[i];
                        int jj = aocc[j];
                        STLBitsetDeterminant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(jj, false);

                        double sign =
                            detI.slater_sign_alpha(ii) * detI.slater_sign_alpha(jj);

                        det_hash<int>::iterator it = map_aa_ann.find(detJ);
                        size_t detJ_add;
                        // Add detJ to map if it isn't there
                        if (it == map_aa_ann.end()) {
                            detJ_add = naa_ann;
                            map_aa_ann[detJ] = naa_ann;
                            naa_ann++;
                            tmp.resize(naa_ann);
                        } else {
                            detJ_add = it->second;
                        }
                        tmp[detJ_add].push_back( std::make_tuple(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj) );
                    }
                }
            }
            for( auto& vec : tmp ){
                if( vec.size() > 1 ){
                    aa_list_.push_back(vec);
                }
            } 
        }
    }
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building aa_list  %1.6f s", aa.get());
    }
    // Generate beta-beta coupling list
Timer bb;
    {
        for( size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a ){
            size_t nbb_ann = 0;
            std::vector<std::vector<std::tuple<size_t,short,short>>> tmp;
            det_hash<int> map_bb_ann;
            std::vector<size_t>& c_dets = alpha_strings_[a];
            size_t max_I = c_dets.size();
            
            for( size_t I = 0; I < max_I; ++I ){    
                size_t idx = c_dets[I];                

                STLBitsetDeterminant detI = dets[idx];
                std::vector<int> bocc = detI.get_beta_occ();
                int nobeta = bocc.size();

                for (int i = 0, ij = 0; i < nobeta; ++i) {
                    for (int j = i + 1; j < nobeta; ++j, ++ij) {
                        int ii = bocc[i];
                        int jj = bocc[j];
                        STLBitsetDeterminant detJ(detI);
                        detJ.set_beta_bit(ii, false);
                        detJ.set_beta_bit(jj, false);

                        double sign =
                            detI.slater_sign_beta(ii) * detI.slater_sign_beta(jj);

                        det_hash<int>::iterator it = map_bb_ann.find(detJ);
                        size_t detJ_add;
                        // Add detJ to map if it isn't there
                        if (it == map_bb_ann.end()) {
                            detJ_add = nbb_ann;
                            map_bb_ann[detJ] = nbb_ann;
                            nbb_ann++;
                            tmp.resize(nbb_ann);
                        } else {
                            detJ_add = it->second;
                        }

                        tmp[detJ_add].push_back( std::make_tuple(
                            idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj));
                    }
                }
            }
            for( auto& vec : tmp ){
                if( vec.size() > 1 ){
                    bb_list_.push_back(vec);
                }
            } 
        }
    }
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building bb_list  %1.6f s", bb.get());
    }

Timer ab;
    // Generate alfa-beta coupling list
    {
        for( size_t a = 0, max_a = alpha_a_strings_.size(); a < max_a; ++a ){
            size_t nab_ann = 0;
            std::vector<std::vector<std::tuple<size_t,short,short>>> tmp;
            det_hash<int> map_ab_ann;
            std::vector<std::pair<int,size_t>>& c_dets = alpha_a_strings_[a];
            size_t max_I = c_dets.size();
            for( size_t I = 0; I < max_I; ++I ){        
                size_t idx = c_dets[I].second;
                int ii = c_dets[I].first;
                STLBitsetDeterminant detI = dets[idx];
                detI.set_alfa_bit(ii, false);
                std::vector<int> bocc = detI.get_beta_occ();
                size_t nobeta = bocc.size();

                for (size_t j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];

                    STLBitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(jj, false);

                    double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_beta(jj);
                    det_hash<int>::iterator it = map_ab_ann.find(detJ);
                    size_t detJ_add;

                    if (it == map_ab_ann.end()) {
                        detJ_add = nab_ann;
                        map_ab_ann[detJ] = nab_ann;
                        nab_ann++;
                        tmp.resize(nab_ann);
                    } else {
                        detJ_add = it->second;
                    }
                    tmp[detJ_add].push_back( std::make_tuple(
                        idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj));
                }
            }
            for( auto& vec : tmp ){
                if( vec.size() > 1 ){
                    ab_list_.push_back(vec);
                }
            } 
        }
        double map_size = (32. + sizeof(size_t)) * ab_list_.size();
        outfile->Printf("\n Memory for AB_ann: %1.3f MB", map_size / (1024. * 1024.));
    }
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building ab_list  %1.6f s", ab.get());
    }
}
 
void WFNOperator::tp_lists(DeterminantMap& wfn) {

    size_t ndets = wfn.size();
    aa_ann_list_.resize(ndets);
    ab_ann_list_.resize(ndets);
    bb_ann_list_.resize(ndets);
    const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();
    // Generate alpha-alpha coupling list
Timer aa;
    {
        size_t naa_ann = 0;
        for( size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b ){
            det_hash<int> map_aa_ann;
            std::vector<size_t> c_dets = beta_strings_[b];
            size_t max_I = c_dets.size();
            for( size_t I = 0; I < max_I; ++I ){
                size_t idx = c_dets[I];
                STLBitsetDeterminant detI = dets[idx];
                std::vector<int> aocc = detI.get_alfa_occ();
                int noalfa = aocc.size();

                std::vector<std::tuple<size_t, short, short>> aa_ann(noalfa * (noalfa-1) / 2);
                for (int i = 0, ij = 0; i < noalfa; ++i) {
                    for (int j = i + 1; j < noalfa; ++j, ++ij) {
                        int ii = aocc[i];
                        int jj = aocc[j];
                        STLBitsetDeterminant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(jj, false);

                        double sign =
                            detI.slater_sign_alpha(ii) * detI.slater_sign_alpha(jj);

                        det_hash<int>::iterator it = map_aa_ann.find(detJ);
                        size_t detJ_add;
                        // Add detJ to map if it isn't there
                        if (it == map_aa_ann.end()) {
                            detJ_add = naa_ann;
                            map_aa_ann[detJ] = naa_ann;
                            naa_ann++;
                        } else {
                            detJ_add = it->second;
                        }
                        aa_ann[ij] = std::make_tuple(detJ_add, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj);
                    }
                }
                aa_ann_list_[idx] = aa_ann;
            }
        }
        aa_cre_list_.resize(naa_ann);
    }
    aa_ann_list_.shrink_to_fit();
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building aa_ann_list  %1.6f s", aa.get());
    }
    // Generate beta-beta coupling list
Timer bb;
    {
        size_t nbb_ann = 0;
        for( size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a ){
            det_hash<int> map_bb_ann;
            std::vector<size_t>& c_dets = alpha_strings_[a];
            size_t max_I = c_dets.size();
            
            for( size_t I = 0; I < max_I; ++I ){    
                size_t idx = c_dets[I];                

                STLBitsetDeterminant detI = dets[idx];
                std::vector<int> bocc = detI.get_beta_occ();
                int nobeta = bocc.size();

                std::vector<std::tuple<size_t,short,short>> bb_ann(nobeta * (nobeta - 1) / 2);
                for (int i = 0, ij = 0; i < nobeta; ++i) {
                    for (int j = i + 1; j < nobeta; ++j, ++ij) {
                        int ii = bocc[i];
                        int jj = bocc[j];
                        STLBitsetDeterminant detJ(detI);
                        detJ.set_beta_bit(ii, false);
                        detJ.set_beta_bit(jj, false);

                        double sign =
                            detI.slater_sign_beta(ii) * detI.slater_sign_beta(jj);

                        det_hash<int>::iterator it = map_bb_ann.find(detJ);
                        size_t detJ_add;
                        // Add detJ to map if it isn't there
                        if (it == map_bb_ann.end()) {
                            detJ_add = nbb_ann;
                            map_bb_ann[detJ] = nbb_ann;
                            nbb_ann++;
                        } else {
                            detJ_add = it->second;
                        }

                        bb_ann[ij] = std::make_tuple(
                            detJ_add, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj);
                    }
                }
                bb_ann_list_[idx] = bb_ann;
            }
        }
        bb_cre_list_.resize(nbb_ann);
    }
    bb_ann_list_.shrink_to_fit();

    if( !quiet_ ){
        outfile->Printf("\n  Time spent building bb_ann_list  %1.6f s", bb.get());
    }
    
Timer ab;
    // Generate alfa-beta coupling list
    {
        size_t nab_ann = 0;
        for( size_t a = 0, max_a = alpha_a_strings_.size(); a < max_a; ++a ){
            det_hash<int> map_ab_ann;
            std::vector<std::pair<int,size_t>>& c_dets = alpha_a_strings_[a];
            size_t max_I = c_dets.size();
            for( size_t I = 0; I < max_I; ++I ){        
                size_t idx = c_dets[I].second;
                int ii = c_dets[I].first;
                STLBitsetDeterminant detI = dets[idx];
                detI.set_alfa_bit(ii, false);
                std::vector<int> bocc = detI.get_beta_occ();
                size_t nobeta = bocc.size();

                size_t offset = ab_ann_list_[idx].size();
                ab_ann_list_[idx].resize(offset + nobeta);
        
                for (size_t j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];

                    STLBitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(jj, false);

                    double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_beta(jj);
                    det_hash<int>::iterator it = map_ab_ann.find(detJ);
                    size_t detJ_add;

                    if (it == map_ab_ann.end()) {
                        detJ_add = nab_ann;
                        map_ab_ann[detJ] = nab_ann;
                        nab_ann++;
                    } else {
                        detJ_add = it->second;
                    }
                    ab_ann_list_[idx][offset+j] = std::make_tuple(
                        detJ_add, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj);
                }
            }
        }
        ab_cre_list_.resize(nab_ann);
        double map_size = (32. + sizeof(size_t)) * nab_ann;
        outfile->Printf("\n Memory for AB_ann: %1.3f MB", map_size / (1024. * 1024.));
    }
    ab_ann_list_.shrink_to_fit();
    if( !quiet_ ){
        outfile->Printf("\n  Time spent building ab_ann_list  %1.6f s", ab.get());
    }

    for (size_t I = 0, max_I = aa_ann_list_.size(); I < max_I; ++I) {
        const std::vector<std::tuple<size_t, short, short>>& aa_ann =
            aa_ann_list_[I];
        for (const std::tuple<size_t, short, short>& J_sign : aa_ann) {
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            aa_cre_list_[J].push_back(std::make_tuple(I, i, j));
        }
    }
    for (size_t I = 0, max_I = bb_ann_list_.size(); I < max_I; ++I) {
        const std::vector<std::tuple<size_t, short, short>>& bb_ann =
            bb_ann_list_[I];
        for (const std::tuple<size_t, short, short>& J_sign : bb_ann) {
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            bb_cre_list_[J].push_back(std::make_tuple(I, i, j));
        }
    }
    for (size_t I = 0, max_I = ab_ann_list_.size(); I < max_I; ++I) {
        const std::vector<std::tuple<size_t, short, short>>& ab_ann =
            ab_ann_list_[I];
        for (const std::tuple<size_t, short, short>& J_sign : ab_ann) {
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            ab_cre_list_[J].push_back(std::make_tuple(I, i, j));
        }
    }
    aa_cre_list_.shrink_to_fit();
    ab_cre_list_.shrink_to_fit();
    bb_cre_list_.shrink_to_fit();
}

void WFNOperator::clear_op_lists() {
    a_ann_list_.clear();
    b_ann_list_.clear();
    a_cre_list_.clear();
    b_cre_list_.clear();
}

void WFNOperator::clear_op_s_lists() {
    a_list_.clear();
    b_list_.clear();
}
void WFNOperator::clear_tp_lists() {
    aa_ann_list_.clear();
    bb_ann_list_.clear();
    aa_cre_list_.clear();
    bb_cre_list_.clear();
    ab_ann_list_.clear();
    ab_cre_list_.clear();
}

void WFNOperator::clear_tp_s_lists() {
    aa_list_.clear();
    bb_list_.clear();
    ab_list_.clear();
}

void WFNOperator::three_lists(DeterminantMap& wfn) {
    size_t ndets = wfn.size();
    const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();
    /// Compute aaa coupling
    {
        aaa_ann_list_.resize(ndets);
        size_t naaa_ann = 0;
        for( int b = 0, max_b = beta_strings_.size(); b < max_b; ++b ){ 
            std::vector<size_t>& c_dets = beta_strings_[b]; 
            size_t max_I = c_dets.size(); 
            det_hash<int> aaa_ann_map;

            for( size_t I = 0; I < max_I; ++I ){
                size_t idx = c_dets[I];
                STLBitsetDeterminant detI = dets[idx];

                std::vector<int> aocc = detI.get_alfa_occ();
                std::vector<int> bocc = detI.get_beta_occ();

                int noalfa = aocc.size();
                int nobeta = bocc.size();

                aaa_ann_list_[idx].resize( noalfa * (noalfa - 1) * (noalfa - 2) / 6);

                // aaa
                for (int i = 0, ijk = 0; i < noalfa; ++i) {
                    for (int j = i + 1; j < noalfa; ++j) {
                        for (int k = j + 1; k < noalfa; ++k, ++ijk) {

                            int ii = aocc[i];
                            int jj = aocc[j];
                            int kk = aocc[k];

                            STLBitsetDeterminant detJ(detI);
                            detJ.set_alfa_bit(ii, false);
                            detJ.set_alfa_bit(jj, false);
                            detJ.set_alfa_bit(kk, false);

                            double sign = detI.slater_sign_alpha(ii) *
                                          detI.slater_sign_alpha(jj) *
                                          detI.slater_sign_alpha(kk);

                            det_hash<int>::iterator hash_it =
                                aaa_ann_map.find(detJ);
                            size_t detJ_add;

                            if (hash_it == aaa_ann_map.end()) {
                                detJ_add = naaa_ann;
                                aaa_ann_map[detJ] = naaa_ann;
                                naaa_ann++;
                            } else {
                                detJ_add = hash_it->second;
                            }
                            aaa_ann_list_[idx][ijk] = std::make_tuple(
                                detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                                kk);
                        }
                    }
                }
            }
        }
        aaa_cre_list_.resize(naaa_ann);
    }
    aaa_ann_list_.shrink_to_fit();

    /// AAB coupling
    {
        // We need the beta-1 list:
        const det_hash<size_t>& wfn_map = wfn.wfn_hash();
        std::vector<std::vector<std::pair<int,size_t>>> beta_string;
        det_hash<size_t> beta_str_hash;
        size_t nabeta = 0;
        for( auto& I : wfn_map ){
            // Grab mutable copy of determinant
            STLBitsetDeterminant detI = I.first;
            detI.zero_spin(0);
            std::vector<int> bocc = detI.get_beta_occ();
            for( int i = 0, nbeta = bocc.size(); i < nbeta; ++i ){
                int ii = bocc[i];
                STLBitsetDeterminant ann_det(detI);
                ann_det.set_beta_bit(ii,false);
                
                size_t b_add;
                det_hash<size_t>::iterator it = beta_str_hash.find(ann_det);
                if( it == beta_str_hash.end() ){
                    b_add = nabeta;
                    beta_str_hash[ann_det] = b_add;
                    nabeta++;
                } else {
                    b_add = it->second;
                }
                beta_string.resize(nabeta);
                beta_string[b_add].push_back(std::make_pair(ii,I.second));
            }
        } 
        aab_ann_list_.resize(ndets);
        size_t naab_ann = 0;
        for( int b = 0, max_b = beta_string.size(); b < max_b; ++b ){
            det_hash<int> aab_ann_map;
            std::vector<std::pair<int, size_t>>& c_dets = beta_string[b];
            size_t max_I = c_dets.size();       
            for( int I = 0; I < max_I; ++I){
                size_t idx = c_dets[I].second; 

                STLBitsetDeterminant detI = dets[idx];
                int kk = c_dets[I].first;
                detI.set_beta_bit(kk, false);

                std::vector<int> aocc = detI.get_alfa_occ();

                int noalfa = aocc.size();

                // Dynamically allocate each sub-vector
                size_t offset = aab_ann_list_[idx].size(); 
                aab_ann_list_[idx].resize(offset +  noalfa * (noalfa - 1) / 2);

                for (int i = 0, jk = 0; i < noalfa; ++i) {
                    for (int j = i + 1; j < noalfa; ++j, ++jk) {

                        int ii = aocc[i];
                        int jj = aocc[j];

                        STLBitsetDeterminant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(jj, false);

                        double sign = detI.slater_sign_alpha(ii) *
                                      detI.slater_sign_alpha(jj) *
                                      detI.slater_sign_beta(kk);

                        det_hash<int>::iterator hash_it =
                            aab_ann_map.find(detJ);
                        size_t detJ_add;

                        if (hash_it == aab_ann_map.end()) {
                            detJ_add = naab_ann;
                            aab_ann_map[detJ] = naab_ann;
                            naab_ann++;
                        } else {
                            detJ_add = hash_it->second;
                        }
                        aab_ann_list_[idx][offset + jk] = std::make_tuple(
                            detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                            kk);
                        
                    }
                }
            }
        }
        aab_cre_list_.resize(naab_ann);
    }
    aab_ann_list_.shrink_to_fit();

    /// ABB coupling
    {
        abb_ann_list_.resize(ndets);
        size_t nabb_ann = 0;
        for( size_t a = 0, max_a = alpha_a_strings_.size(); a < max_a; ++a ){
            det_hash<int> abb_ann_map;
            std::vector<std::pair<int,size_t>>& c_dets = alpha_a_strings_[a];            
            size_t max_I = c_dets.size();

            for( int I = 0; I < max_I; ++I ){
                size_t idx = c_dets[I].second; 
     
                STLBitsetDeterminant detI = dets[idx];
                int ii = c_dets[I].first;
                detI.set_alfa_bit(ii, false);

                std::vector<int> bocc = detI.get_beta_occ();

                int nobeta = bocc.size();

                // Dynamically allocate each sub-vector
                size_t offset = abb_ann_list_[idx].size(); 
                abb_ann_list_[idx].resize(offset +  nobeta * (nobeta - 1) / 2);

                for (int j = 0, jk = 0; j < nobeta; ++j) {
                    for (int k = j + 1; k < nobeta; ++k, ++jk) {

                        int jj = bocc[j];
                        int kk = bocc[k];

                        STLBitsetDeterminant detJ(detI);
                        detJ.set_beta_bit(jj, false);
                        detJ.set_beta_bit(kk, false);

                        double sign = detI.slater_sign_alpha(ii) *
                                      detI.slater_sign_beta(jj) *
                                      detI.slater_sign_beta(kk);

                        det_hash<int>::iterator hash_it =
                            abb_ann_map.find(detJ);
                        size_t detJ_add;

                        if (hash_it == abb_ann_map.end()) {
                            detJ_add = nabb_ann;
                            abb_ann_map[detJ] = nabb_ann;
                            nabb_ann++;
                        } else {
                            detJ_add = hash_it->second;
                        }
                        abb_ann_list_[idx][offset + jk] = std::make_tuple(
                            detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                            kk);
                    }
                }
            }
        }
        abb_cre_list_.resize(nabb_ann);
    }
    abb_ann_list_.shrink_to_fit();

    /// BBB coupling
    {
        size_t nbbb_ann = 0;
        bbb_ann_list_.resize(ndets);
        for( size_t a = 0, max_a = alpha_a_strings_.size(); a < max_a; ++a ){
            det_hash<int> bbb_ann_map;
            std::vector<std::pair<int,size_t>>& c_dets = alpha_a_strings_[a];            
            size_t max_I = c_dets.size();
            
            for( int I = 0; I < max_I; ++I ){
                size_t idx = c_dets[I].second; 
                STLBitsetDeterminant detI = dets[idx];

                std::vector<int> aocc = detI.get_alfa_occ();
                std::vector<int> bocc = detI.get_beta_occ();

                int noalfa = aocc.size();
                int nobeta = bocc.size();

                bbb_ann_list_[idx].resize(nobeta * (nobeta - 1) * (nobeta - 2) / 6);
                // bbb
                for (int i = 0, ijk = 0; i < nobeta; ++i) {
                    for (int j = i + 1; j < nobeta; ++j) {
                        for (int k = j + 1; k < nobeta; ++k, ++ijk) {

                            int ii = bocc[i];
                            int jj = bocc[j];
                            int kk = bocc[k];

                            STLBitsetDeterminant detJ(detI);
                            detJ.set_beta_bit(ii, false);
                            detJ.set_beta_bit(jj, false);
                            detJ.set_beta_bit(kk, false);

                            double sign = detI.slater_sign_beta(ii) *
                                          detI.slater_sign_beta(jj) *
                                          detI.slater_sign_beta(kk);

                            det_hash<int>::iterator hash_it =
                                bbb_ann_map.find(detJ);
                            size_t detJ_add;

                            if (hash_it == bbb_ann_map.end()) {
                                detJ_add = nbbb_ann;
                                bbb_ann_map[detJ] = nbbb_ann;
                                nbbb_ann++;
                            } else {
                                detJ_add = hash_it->second;
                            }
                            bbb_ann_list_[idx][ijk] = std::make_tuple(
                                detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                                kk);
                        }
                    }
                }
            }
        }
        bbb_cre_list_.resize(nbbb_ann);
    }
    bbb_ann_list_.shrink_to_fit();

    // Make the creation lists

    // aaa
    for (size_t I = 0, max_size = aaa_ann_list_.size(); I < max_size; ++I) {
        const std::vector<std::tuple<size_t, short, short, short>>& aaa_ann =
            aaa_ann_list_[I];
        for (const std::tuple<size_t, short, short, short>& Jsign : aaa_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            short k = std::get<3>(Jsign);
            aaa_cre_list_[J].push_back(std::make_tuple(I, i, j, k));
        }
    }
    // aab
    for (size_t I = 0, max_size = aab_ann_list_.size(); I < max_size; ++I) {
        const std::vector<std::tuple<size_t, short, short, short>>& aab_ann =
            aab_ann_list_[I];
        for (const std::tuple<size_t, short, short, short>& Jsign : aab_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            short k = std::get<3>(Jsign);
            aab_cre_list_[J].push_back(std::make_tuple(I, i, j, k));
        }
    }
    // abb
    for (size_t I = 0, max_size = abb_ann_list_.size(); I < max_size; ++I) {
        const std::vector<std::tuple<size_t, short, short, short>>& abb_ann =
            abb_ann_list_[I];
        for (const std::tuple<size_t, short, short, short>& Jsign : abb_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            short k = std::get<3>(Jsign);
            abb_cre_list_[J].push_back(std::make_tuple(I, i, j, k));
        }
    }
    // bbb
    for (size_t I = 0, max_size = bbb_ann_list_.size(); I < max_size; ++I) {
        const std::vector<std::tuple<size_t, short, short, short>>& bbb_ann =
            bbb_ann_list_[I];
        for (const std::tuple<size_t, short, short, short>& Jsign : bbb_ann) {
            size_t J = std::get<0>(Jsign);
            short i = std::get<1>(Jsign);
            short j = std::get<2>(Jsign);
            short k = std::get<3>(Jsign);
            bbb_cre_list_[J].push_back(std::make_tuple(I, i, j, k));
        }
    }
}

//std::vector<std::pair<std::vector<size_t>, std::vector<double>>> WFNOperator::build_H_sparse2( const DeterminantMap& wfn )
//{
//    /*- Build the Hamiltonian without the coupling lists, but with the string lists -*/    
//        
//    
//}

}}

