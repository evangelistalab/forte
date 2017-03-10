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

namespace psi {
namespace forte {

WFNOperator::WFNOperator(std::vector<int> symmetry)
{
    mo_symmetry_ = symmetry;
}

WFNOperator::WFNOperator() {}

void WFNOperator::initialize(std::vector<int>& symmetry) {
    mo_symmetry_ = symmetry;
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

        // Loop directly through all determinants with
        // spin-coupled electrons, i.e:
        // |PhiI> = a+(qa) a+(pb) a-(qb) a-(pa) |PhiJ>
        for (auto& abJ_mo_sign : ab_ann_list_[det.second]) {
            const size_t abJ_add = std::get<0>(abJ_mo_sign);
            double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            short p = std::fabs(std::get<1>(abJ_mo_sign)) - 1;
            short q = std::get<2>(abJ_mo_sign);
            if (p == q)
                continue;
            for (auto& ababJ_mo_sign : ab_cre_list_[abJ_add]) {
                double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                short r = std::fabs(std::get<1>(ababJ_mo_sign)) - 1;
                short s = std::get<2>(ababJ_mo_sign);
                if ((r != s) and (p == s) and (q == r)) {
                    sign_pq *= sign_rs;
                    S2 -= sign_pq * evecs->get(det.second, root) *
                          evecs->get(std::get<0>(ababJ_mo_sign), root);
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
    
    // First build a map from beta strings to determinants
    const det_hash<size_t>& wfn_map = wfn.wfn_hash();
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
    // Next build a map from annihilated alpha strings to determinants
    det_hash<size_t> alfa_str_hash;
    size_t naalpha = 0;
    for( auto& I : wfn_map ){
        // Grab mutable copy of determinant
        STLBitsetDeterminant detI = I.first;
        detI.zero_spin(1);
        std::vector<int> aocc = detI.get_alfa_occ();
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
            alpha_strings_.resize(naalpha);
            alpha_strings_[a_add].push_back(std::make_pair(ii,I.second));
        }
    } 
}

void WFNOperator::op_lists(DeterminantMap& wfn) {
    size_t ndets = wfn.size();
    a_ann_list_.resize(ndets);
    b_ann_list_.resize(ndets);
    const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();
    // Generate alpha coupling list
    {
        size_t na_ann = 0;
        for( size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b){
            std::vector<size_t>& c_dets = beta_strings_[b];
            size_t max_I = c_dets.size();
            det_hash<int> map_a_ann;
            for( size_t I = 0; I < max_I; ++I){
                const STLBitsetDeterminant& detI = dets[c_dets[I]];
                std::vector<int> aocc = detI.get_alfa_occ();
                int noalfa = aocc.size(); 
                
                a_ann_list_[c_dets[I]].resize(noalfa);
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

                    a_ann_list_[c_dets[I]][i] = std::make_pair(detJ_add,
                                              (sign > 0.0) ? (ii + 1) : (-ii - 1));
                }
            }    
        } 
        a_ann_list_.shrink_to_fit();
        a_cre_list_.resize(na_ann);
    }
    // Generate beta coupling list
    {
        size_t nb_ann = 0;
        for( size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a ){
            std::vector<std::pair<int,size_t>>& c_dets = alpha_strings_[a];
            size_t max_I = c_dets.size();
            det_hash<int> map_b_ann;
            for( size_t I = 0; I < max_I; ++I){
                int idx = c_dets[I].second;
                const STLBitsetDeterminant& detI = dets[idx];
                std::vector<int> bocc = detI.get_beta_occ();
                int nobeta = bocc.size(); 

                b_ann_list_[idx].resize(nobeta);

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

                    b_ann_list_[idx][i] = std::make_pair(detJ_add,
                                              (sign > 0.0) ? (ii + 1) : (-ii - 1));
                }
            }
        }
        b_ann_list_.shrink_to_fit();
        b_cre_list_.resize(nb_ann);
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

void WFNOperator::tp_lists(DeterminantMap& wfn) {
    size_t ndets = wfn.size();
    aa_ann_list_.resize(ndets);
    ab_ann_list_.resize(ndets);
    bb_ann_list_.resize(ndets);
    const std::vector<STLBitsetDeterminant>& dets = wfn.determinants();
    // Generate alpha-alpha coupling list
    {
        size_t naa_ann = 0;
        for( size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b ){
            det_hash<int> map_aa_ann;
            std::vector<size_t>& c_dets = beta_strings_[b];
            size_t max_I = c_dets.size();
            
            for( size_t I = 0; I < max_I; ++I ){
                size_t idx = c_dets[I];
                STLBitsetDeterminant detI = dets[idx];
                std::vector<int> aocc = detI.get_alfa_occ();
                int noalfa = aocc.size();

                aa_ann_list_[idx].resize(noalfa * (noalfa-1) / 2);
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

                        aa_ann_list_[idx][ij] = std::make_tuple(
                            detJ_add, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj);
                    }
                }
            }
        }
        aa_cre_list_.resize(naa_ann);
    }
    aa_ann_list_.shrink_to_fit();
    // Generate beta-beta coupling list
    {
        size_t nbb_ann = 0;
        for( size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a ){
            det_hash<int> map_bb_ann;
            std::vector<std::pair<int,size_t>>& c_dets = alpha_strings_[a];
            size_t max_I = c_dets.size();
            
            for( size_t I = 0; I < max_I; ++I ){    
                size_t idx = c_dets[I].second;                

                STLBitsetDeterminant detI = dets[idx];
                std::vector<int> bocc = detI.get_beta_occ();
                int nobeta = bocc.size();

                bb_ann_list_[idx].resize(nobeta * (nobeta - 1) / 2);

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

                        bb_ann_list_[idx][ij] = std::make_tuple(
                            detJ_add, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj);
                    }
                }
            }
        }
        bb_cre_list_.resize(nbb_ann);
    }
    bb_ann_list_.shrink_to_fit();


    // Generate alfa-beta coupling list
    {
        size_t nab_ann = 0;
        for( size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a ){
            det_hash<int> map_ab_ann;
            std::vector<std::pair<int,size_t>>& c_dets = alpha_strings_[a];
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
                    } else {
                        detJ_add = it->second;
                    }
                    ab_ann_list_[idx].push_back(std::make_tuple(
                        detJ_add, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj));
                }
            }
        }
        ab_cre_list_.resize(nab_ann);
    }
    ab_ann_list_.shrink_to_fit();
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

void WFNOperator::clear_tp_lists() {
    aa_ann_list_.clear();
    bb_ann_list_.clear();
    aa_cre_list_.clear();
    bb_cre_list_.clear();
    ab_ann_list_.clear();
    ab_cre_list_.clear();
}

void WFNOperator::three_lists(DeterminantMap& wfn) {
    size_t ndets = wfn.size();
    const det_hash<size_t>& wfn_map = wfn.wfn_hash();

    /// Compute aaa coupling
    {
        aaa_ann_list_.resize(ndets);
        size_t naaa_ann = 0;
        det_hash<int> aaa_ann_map;
        for (auto& I : wfn_map) {
            STLBitsetDeterminant detI = I.first;

            std::vector<int> aocc = detI.get_alfa_occ();
            std::vector<int> bocc = detI.get_beta_occ();

            int noalfa = aocc.size();
            int nobeta = bocc.size();

            std::vector<std::tuple<size_t, short, short, short>> aaa_ann(
                noalfa * (noalfa - 1) * (noalfa - 2) / 6);

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
                        aaa_ann[ijk] = std::make_tuple(
                            detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                            kk);
                    }
                }
            }
            aaa_ann_list_[I.second] = aaa_ann;
        }
        aaa_cre_list_.resize(aaa_ann_map.size());
    }
    aaa_ann_list_.shrink_to_fit();

    /// AAB coupling
    {
        aab_ann_list_.resize(ndets);
        size_t naab_ann = 0;
        det_hash<int> aab_ann_map;
        for (auto& I : wfn_map) {
            STLBitsetDeterminant detI = I.first;

            std::vector<int> aocc = detI.get_alfa_occ();
            std::vector<int> bocc = detI.get_beta_occ();

            int noalfa = aocc.size();
            int nobeta = bocc.size();

            std::vector<std::tuple<size_t, short, short, short>> aab_ann(
                noalfa * (noalfa - 1) * nobeta / 2);

            for (int i = 0, ijk = 0; i < noalfa; ++i) {
                for (int j = i + 1; j < noalfa; ++j) {
                    for (int k = 0; k < nobeta; ++k, ++ijk) {

                        int ii = aocc[i];
                        int jj = aocc[j];
                        int kk = bocc[k];

                        STLBitsetDeterminant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(jj, false);
                        detJ.set_beta_bit(kk, false);

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
                        aab_ann[ijk] = std::make_tuple(
                            detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                            kk);
                    }
                }
            }
            aab_ann_list_[I.second] = aab_ann;
        }
        aab_cre_list_.resize(aab_ann_map.size());
    }
    aab_ann_list_.shrink_to_fit();

    /// ABB coupling
    {
        det_hash<int> abb_ann_map;
        abb_ann_list_.resize(ndets);
        size_t nabb_ann = 0;
        for (auto& I : wfn_map) {
            STLBitsetDeterminant detI = I.first;

            std::vector<int> aocc = detI.get_alfa_occ();
            std::vector<int> bocc = detI.get_beta_occ();

            int noalfa = aocc.size();
            int nobeta = bocc.size();

            std::vector<std::tuple<size_t, short, short, short>> abb_ann(
                noalfa * nobeta * (nobeta - 1) / 2);

            for (int i = 0, ijk = 0; i < noalfa; ++i) {
                for (int j = 0; j < nobeta; ++j) {
                    for (int k = j + 1; k < nobeta; ++k, ++ijk) {

                        int ii = aocc[i];
                        int jj = bocc[j];
                        int kk = bocc[k];

                        STLBitsetDeterminant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
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
                        abb_ann[ijk] = std::make_tuple(
                            detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                            kk);
                    }
                }
            }
            abb_ann_list_[I.second] = abb_ann;
        }
        abb_cre_list_.resize(abb_ann_map.size());
    }
    abb_ann_list_.shrink_to_fit();

    /// BBB coupling
    {
        size_t nbbb_ann = 0;
        bbb_ann_list_.resize(ndets);
        det_hash<int> bbb_ann_map;
        for (auto& I : wfn_map) {
            STLBitsetDeterminant detI = I.first;

            std::vector<int> aocc = detI.get_alfa_occ();
            std::vector<int> bocc = detI.get_beta_occ();

            int noalfa = aocc.size();
            int nobeta = bocc.size();

            std::vector<std::tuple<size_t, short, short, short>> bbb_ann(
                nobeta * (nobeta - 1) * (nobeta - 2) / 6);
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
                        bbb_ann[ijk] = std::make_tuple(
                            detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                            kk);
                    }
                }
            }
            bbb_ann_list_[I.second] = bbb_ann;
        }
        bbb_cre_list_.resize(bbb_ann_map.size());
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
}
}
