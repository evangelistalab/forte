#include "operator.h"

namespace psi{ namespace forte{

WFNOperator::WFNOperator( std::shared_ptr<MOSpaceInfo> mo_space_info ) : mo_space_info_(mo_space_info) 
{
    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");
}

double WFNOperator::s2( DeterminantMap& wfn )
{
    double S2 = 0.0;
    const det_hash<size_t>& wfn_map = wfn.wfn();    

    for (auto& det : wfn_map){
        // Compute diagonal
        // PhiI = PhiJ
        STLBitsetDeterminant PhiI = det.first;
        int npair = PhiI.npair();
        int na = PhiI.get_alfa_occ().size();
        int nb = PhiI.get_beta_occ().size();
        double ms = 0.5 * static_cast<double>(na - nb); 
        S2 += ( ms*ms + ms + static_cast<double>(nb) - static_cast<double>(npair)) * wfn.coefficient(det.second) * wfn.coefficient(det.second);
        if( (npair == nb) or (npair == na) ) continue;

        // Loop directly through all determinants with
        // spin-coupled electrons, i.e:
        // |PhiI> = a+(qa) a+(pb) a-(qb) a-(pa) |PhiJ>
        for (auto& abJ_mo_sign : ab_ann_list_[det.second]){
            const size_t abJ_add = std::get<0>(abJ_mo_sign);
            double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            short p = std::fabs(std::get<1>(abJ_mo_sign)) - 1;
            short q = std::get<2>(abJ_mo_sign); 
            if( p == q ) continue;
            for (auto& ababJ_mo_sign : ab_cre_list_[abJ_add]){
                double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                short r = std::fabs(std::get<1>(ababJ_mo_sign)) - 1;
                short s = std::get<2>(ababJ_mo_sign);
                if( (r!=s) and (p==s) and (q==r) ) {
                    sign_pq *= sign_rs;
                    S2 -= sign_pq * wfn.coefficient(det.second) * wfn.coefficient(std::get<0>(ababJ_mo_sign));
                }
            }
        }
    }

    S2  = std::fabs(S2);
    S2 /= wfn.norm();
    return S2;
}

void WFNOperator::add_singles( DeterminantMap& wfn )
{
    det_hash<size_t>& wfn_map = wfn.wfn();

    // Loop through determinants, generate singles and add them to the wfn
    //Alpha excitations
    for( auto& I : wfn_map ){
        STLBitsetDeterminant det = I.first;
        std::vector<int> aocc = det.get_alfa_occ();
        std::vector<int> avir = det.get_alfa_vir();
        
        for( int i = 0, noalpha = aocc.size(); i < noalpha; ++i){
            int ii = aocc[i];
            for( int a = 0, nvalpha = avir.size(); a < nvalpha; ++a){
                int aa = avir[a];
                if( (mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0){
                    det.set_alfa_bit(ii,false);
                    det.set_alfa_bit(aa,true);
                    wfn.add( det, 0.0 );
                    det.set_alfa_bit(ii,true);
                    det.set_alfa_bit(aa,false);
                }
            }
        }
   // }
    
    // Beta excitations
   // for( auto& I : wfn_map ){
   //     STLBitsetDeterminant det = I.first;
        std::vector<int> bocc = det.get_beta_occ();
        std::vector<int> bvir = det.get_beta_vir();
        
        for( int i = 0, nobeta = bocc.size(); i < nobeta; ++i){
            int ii = bocc[i];
            for( int a = 0, nvbeta = bvir.size(); a < nvbeta; ++a){
                int aa = bvir[a];
                if( (mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0){
                    det.set_beta_bit(ii,false);
                    det.set_beta_bit(aa,true);
                    wfn.add( det, 0.0 );
                    det.set_beta_bit(ii,true);
                    det.set_beta_bit(aa,false);
                }
            }
        }
    }

}

void WFNOperator::add_doubles( DeterminantMap& wfn )
{
    const det_hash<size_t>& wfn_map = wfn.wfn();

    for( auto& I : wfn_map ){
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
        for( int i = 0; i < noalfa; ++i ){
            int ii = aocc[i];
            for( int j = i+1; j < noalfa; ++j ){
                int jj = aocc[j];
                for( int a = 0; a < nvalfa; a++ ){
                    int aa = avir[a];
                    for( int b = a + 1; b < nvalfa; ++b ){
                        int bb = avir[b];
                        if( (mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb] ) == 0 ){
                            det.set_alfa_bit(ii, false);
                            det.set_alfa_bit(jj, false);
                            det.set_alfa_bit(aa, true);
                            det.set_alfa_bit(bb, true);
                            wfn.add( det, 0.0);
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
        for( int i = 0; i < nobeta; ++i ){
            int ii = bocc[i];
            for( int j = i+1; j < nobeta; ++j ){
                int jj = bocc[j];
                for( int a = 0; a < nvbeta; a++ ){
                    int aa = bvir[a];
                    for( int b = a + 1; b < nvbeta; ++b ){
                        int bb = bvir[b];
                        if( (mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb] ) == 0 ){
                            det.set_beta_bit(ii, false);
                            det.set_beta_bit(jj, false);
                            det.set_beta_bit(aa, true);
                            det.set_beta_bit(bb, true);
                            wfn.add( det, 0.0);
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
        for( int i = 0; i < noalfa; ++i ){
            int ii = aocc[i];
            for( int j = 0; j < nobeta; ++j ){
                int jj = bocc[j];
                for( int a = 0; a < nvalfa; a++ ){
                    int aa = avir[a];
                    for( int b = 0; b < nvbeta; ++b ){
                        int bb = bvir[b];
                        if( (mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb] ) == 0 ){
                            det.set_alfa_bit(ii, false);
                            det.set_beta_bit(jj, false);
                            det.set_alfa_bit(aa, true);
                            det.set_beta_bit(bb, true);
                            wfn.add( det, 0.0);
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
}

void WFNOperator::op_lists( DeterminantMap& wfn )
{
    size_t ndets = wfn.size();
    a_ann_list_.resize(ndets);
    b_ann_list_.resize(ndets);
    const det_hash<size_t>& wfn_map = wfn.wfn();
    // Generate alpha coupling list
    {
        size_t na_ann = 0;
        det_hash<int> map_a_ann;
        for( auto& I : wfn_map ){
            STLBitsetDeterminant detI = I.first;
            std::vector<int> aocc = detI.get_alfa_occ();
            int noalfa = aocc.size();    
            
            std::vector< std::pair<size_t,short> > a_ann(noalfa);

            for( int i = 0; i < noalfa; ++i ){
                int ii = aocc[i]; 
                STLBitsetDeterminant detJ(detI);
                detJ.set_alfa_bit(ii,false);

                double sign = detI.slater_sign_alpha(ii);

                det_hash<int>::iterator it = map_a_ann.find(detJ);
                size_t detJ_add;
                // Add detJ to map if it isn't there
                if( it == map_a_ann.end() ){
                    detJ_add = na_ann;
                    map_a_ann[detJ] = na_ann;
                    na_ann++;
                }else{
                    detJ_add = it->second;
                }

                a_ann[i] = std::make_pair(detJ_add, (sign>0.0) ? (ii+1) : (-ii-1));
            }
            a_ann.shrink_to_fit();
            a_ann_list_[I.second] = a_ann;
        }
        a_ann_list_.shrink_to_fit();
        a_cre_list_.resize(na_ann);
    } 

    // Generate beta coupling list
    {
        size_t nb_ann = 0;
        det_hash<int> map_b_ann;
        for( auto& I : wfn_map ){
            STLBitsetDeterminant detI = I.first;
            std::vector<int> bocc = detI.get_beta_occ();
            int nobeta = bocc.size();    
            
            std::vector< std::pair<size_t,short> > b_ann(nobeta);

            for( int i = 0; i < nobeta; ++i ){
                int ii = bocc[i]; 
                STLBitsetDeterminant detJ(detI);
                detJ.set_beta_bit(ii,false);

                double sign = detI.slater_sign_beta(ii);

                det_hash<int>::iterator it = map_b_ann.find(detJ);
                size_t detJ_add;
                // Add detJ to map if it isn't there
                if( it == map_b_ann.end() ){
                    detJ_add = nb_ann;
                    map_b_ann[detJ] = nb_ann;
                    nb_ann++;
                }else{
                    detJ_add = it->second;
                }

                b_ann[i] = std::make_pair(detJ_add, (sign>0.0) ? (ii+1) : (-ii-1));
            }
            b_ann.shrink_to_fit();
            b_ann_list_[I.second] = b_ann;
        }
        b_ann_list_.shrink_to_fit();
        b_cre_list_.resize(nb_ann);
    } 

    for( size_t I = 0, max_I = a_ann_list_.size(); I < max_I; ++I ){
        const std::vector<std::pair<size_t,short>>& a_ann = a_ann_list_[I];
        for( const std::pair<size_t,short>& J_sign : a_ann ){
            size_t J = J_sign.first;
            short sign = J_sign.second;
            a_cre_list_[J].push_back(std::make_pair(I,sign));
        }
    }
    for( size_t I = 0, max_I = b_ann_list_.size(); I < max_I; ++I ){
        const std::vector<std::pair<size_t,short>>& b_ann = b_ann_list_[I];
        for( const std::pair<size_t,short>& J_sign : b_ann ){
            size_t J = J_sign.first;
            short sign = J_sign.second;
            b_cre_list_[J].push_back(std::make_pair(I,sign));
        }
    }
}

void WFNOperator::tp_lists( DeterminantMap& wfn )
{
    size_t ndets = wfn.size();
    aa_ann_list_.resize(ndets);
    ab_ann_list_.resize(ndets);
    bb_ann_list_.resize(ndets);
    const det_hash<size_t>& wfn_map = wfn.wfn();
    // Generate alpha-alpha coupling list
    {
        size_t naa_ann = 0;
        det_hash<int> map_aa_ann;
        for( auto& I : wfn_map ){
            STLBitsetDeterminant detI = I.first;
            std::vector<int> aocc = detI.get_alfa_occ();
            int noalfa = aocc.size();    
            
            std::vector< std::tuple<size_t,short,short> > aa_ann(noalfa*(noalfa-1));

            for( int i = 0, ij = 0; i < noalfa; ++i ){
                for( int j = i + 1; j < noalfa; ++j, ++ij){
                    int ii = aocc[i]; 
                    int jj = aocc[j];
                    STLBitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii,false);
                    detJ.set_alfa_bit(jj,false);

                    double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_alpha(jj);

                    det_hash<int>::iterator it = map_aa_ann.find(detJ);
                    size_t detJ_add;
                    // Add detJ to map if it isn't there
                    if( it == map_aa_ann.end() ){
                        detJ_add = naa_ann;
                        map_aa_ann[detJ] = naa_ann;
                        naa_ann++;
                    }else{
                        detJ_add = it->second;
                    }

                    aa_ann[ij] = std::make_tuple(detJ_add, (sign>0.0) ? (ii+1) : (-ii-1), jj);
                }
            }
            aa_ann.shrink_to_fit();
            aa_ann_list_[I.second] = aa_ann;
        }
        aa_ann_list_.shrink_to_fit();
        aa_cre_list_.resize(naa_ann);
    } 

    // Generate beta-beta coupling list
    {
        size_t nbb_ann = 0;
        det_hash<int> map_bb_ann;
        for( auto& I : wfn_map ){
            STLBitsetDeterminant detI = I.first;
            std::vector<int> bocc = detI.get_beta_occ();
            int nobeta = bocc.size();    
            
            std::vector< std::tuple<size_t,short,short> > bb_ann(nobeta*(nobeta-1));

            for( int i = 0, ij = 0; i < nobeta; ++i ){
                for( int j = i + 1; j < nobeta; ++j, ++ij){ 
                    int ii = bocc[i]; 
                    int jj = bocc[j]; 
                    STLBitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(ii,false);
                    detJ.set_beta_bit(jj,false);

                    double sign = detI.slater_sign_beta(ii) * detI.slater_sign_beta(jj);

                    det_hash<int>::iterator it = map_bb_ann.find(detJ);
                    size_t detJ_add;
                    // Add detJ to map if it isn't there
                    if( it == map_bb_ann.end() ){
                        detJ_add = nbb_ann;
                        map_bb_ann[detJ] = nbb_ann;
                        nbb_ann++;
                    }else{
                        detJ_add = it->second;
                    }

                    bb_ann[ij] = std::make_tuple(detJ_add, (sign>0.0) ? (ii+1) : (-ii-1), jj);
                }
            }
            bb_ann.shrink_to_fit();
            bb_ann_list_[I.second] = bb_ann;
        }
        bb_ann_list_.shrink_to_fit();
        bb_cre_list_.resize(nbb_ann);

    } 
    
    // Generate alfa-beta coupling list
    {
        size_t nab_ann = 0;
        det_hash<int> map_ab_ann;
        for( auto& I : wfn_map ){
            STLBitsetDeterminant detI = I.first;
            std::vector<int> aocc = detI.get_alfa_occ();
            std::vector<int> bocc = detI.get_beta_occ();
            
            size_t noalfa = aocc.size();
            size_t nobeta = bocc.size();
            
            std::vector<std::tuple<size_t,short,short>> ab_ann(noalfa*nobeta);

            for( size_t i = 0, ij = 0; i < noalfa; ++i ){
                for( size_t j = 0; j < nobeta; ++j, ++ij ){
                    int ii = aocc[i];
                    int jj = bocc[j];
        
                    STLBitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii,false);
                    detJ.set_beta_bit(jj,false);

                    double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_beta(jj);
                    det_hash<int>::iterator it = map_ab_ann.find(detJ);
                    size_t detJ_add;

                    if( it == map_ab_ann.end() ){
                        detJ_add = nab_ann;
                        map_ab_ann[detJ] = nab_ann;
                        nab_ann++;               
                    }else{
                        detJ_add = it->second;
                    }
                    ab_ann[ij] = std::make_tuple(detJ_add,(sign>0.0) ? (ii + 1) : (-ii - 1), jj);
                }
            } 
            ab_ann.shrink_to_fit();
            ab_ann_list_[I.second] = ab_ann;
        }
        ab_cre_list_.resize(map_ab_ann.size());
    }
    ab_ann_list_.shrink_to_fit();

    for( size_t I = 0, max_I = aa_ann_list_.size(); I < max_I; ++I ){
        const std::vector<std::tuple<size_t,short,short>>& aa_ann = aa_ann_list_[I];
        for( const std::tuple<size_t,short,short>& J_sign : aa_ann ){
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            aa_cre_list_[J].push_back(std::make_tuple(I,i,j));
        }
    }
    for( size_t I = 0, max_I = bb_ann_list_.size(); I < max_I; ++I ){
        const std::vector<std::tuple<size_t,short,short>>& bb_ann = bb_ann_list_[I];
        for( const std::tuple<size_t,short,short>& J_sign : bb_ann ){
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            bb_cre_list_[J].push_back(std::make_tuple(I,i,j));
        }
    }    
    for( size_t I = 0, max_I = ab_ann_list_.size(); I < max_I; ++I ){
        const std::vector<std::tuple<size_t,short,short>>& ab_ann = ab_ann_list_[I];
        for( const std::tuple<size_t,short,short>& J_sign : ab_ann ){
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            ab_cre_list_[J].push_back(std::make_tuple(I,i,j));
        }
    }    
    aa_cre_list_.shrink_to_fit();
    ab_cre_list_.shrink_to_fit();
    bb_cre_list_.shrink_to_fit();
}

}}
