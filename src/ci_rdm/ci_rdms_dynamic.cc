/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/process.h"

#include "ci_rdms.h"
#include "../helpers.h"
#include "../reference.h"
#include "../sparse_ci/determinant.h"

namespace psi {
namespace forte {


void CI_RDMS::compute_rdms_dynamic(std::vector<double>& oprdm_a, 
                                   std::vector<double>& oprdm_b,
                                   std::vector<double>& tprdm_aa,
                                   std::vector<double>& tprdm_ab,
                                   std::vector<double>& tprdm_bb,
                                   std::vector<double>& tprdm_aaa,
                                   std::vector<double>& tprdm_aab,
                                   std::vector<double>& tprdm_abb,
                                   std::vector<double>& tprdm_bbb){

    SortedStringList_UI64 a_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Alpha);
    SortedStringList_UI64 b_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Beta);




    oprdm_a.resize(ncmo2_,0.0);
    oprdm_b.resize(ncmo2_,0.0);
    const std::vector<UI64Determinant::bit_t>& sorted_bstr = b_sorted_string_list_.sorted_half_dets();
    size_t num_bstr = sorted_bstr.size();
    const auto& sorted_b_dets = b_sorted_string_list_.sorted_dets();
    const auto& sorted_a_dets = a_sorted_string_list_.sorted_dets();

    for( size_t I = 0; I < dim_space_; ++I ){
        size_t Ia = b_sorted_string_list_.add(I);
        double CIa = evecs_->get(Ia, root1_) * evecs_->get(Ia,root2_);
        UI64Determinant::bit_t det_a = sorted_b_dets[I].get_alfa_bits();
        while( det_a > 0 ){
            int p = lowest_one_idx(det_a);
            oprdm_a[p * ncmo_ + p] += CIa;
            det_a = clear_lowest_one(det_a);
        } 
        size_t Ib = a_sorted_string_list_.add(I);
        double CIb = evecs_->get(Ib, root1_) * evecs_->get(Ib,root2_);
        UI64Determinant::bit_t det_b = sorted_a_dets[I].get_beta_bits();
        while( det_b > 0 ){
            int p = lowest_one_idx(det_b);
            oprdm_b[p * ncmo_ + p] += CIb;
            det_b = clear_lowest_one(det_b);
        } 

    }

    // First alpha
    // loop through all beta strings
    for( size_t bstr = 0; bstr < num_bstr; ++bstr ){
        const UI64Determinant::bit_t& Ib = sorted_bstr[bstr];
        const auto& range_I = b_sorted_string_list_.range(Ib);    

        UI64Determinant::bit_t Ia;
        UI64Determinant::bit_t Ja;
        UI64Determinant::bit_t IJa;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;
        
        // Double loop through determinants with same beta string
        for( size_t I = first_I; I < last_I; ++I ){
            Ia = sorted_b_dets[I].get_alfa_bits();
            for( size_t J = I+1; J < last_I; ++J ){
                Ja = sorted_b_dets[J].get_alfa_bits();
                
                IJa = Ia ^ Ja;
                int ndiff = ui64_bit_count(IJa);

                if( ndiff == 2 ){
                    uint64_t p = lowest_one_idx(IJa);
                    IJa = clear_lowest_one(IJa);
                    uint64_t q = lowest_one_idx(IJa);

                    double value = evecs_->get(b_sorted_string_list_.add(I),root1_) * 
                                   evecs_->get(b_sorted_string_list_.add(J),root2_) * ui64_slater_sign(Ia,p,q);
                    oprdm_a[p * ncmo_ + q] += value;
                    oprdm_a[q * ncmo_ + p] += value;
                }
            }
        }
    }
    // Then beta
    // loop through all alpha strings
    const std::vector<UI64Determinant::bit_t>& sorted_astr = a_sorted_string_list_.sorted_half_dets();
    size_t num_astr = sorted_astr.size();
    for( size_t astr = 0; astr < num_astr; ++astr ){
        const UI64Determinant::bit_t& Ia = sorted_astr[astr];
        const auto& range_I = a_sorted_string_list_.range(Ia);    

        UI64Determinant::bit_t Ib;
        UI64Determinant::bit_t Jb;
        UI64Determinant::bit_t IJb;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;
        
        // Double loop through determinants with same alpha string
        for( size_t I = first_I; I < last_I; ++I ){
            Ib = sorted_a_dets[I].get_beta_bits();
            for( size_t J = I+1; J < last_I; ++J ){
                Jb = sorted_a_dets[J].get_beta_bits();
                
                IJb = Ib ^ Jb;
                int ndiff = ui64_bit_count(IJb);

                if( ndiff == 2 ){
                    uint64_t p = lowest_one_idx(IJb);
                    IJb = clear_lowest_one(IJb);
                    uint64_t q = lowest_one_idx(IJb);

                    double value = evecs_->get(a_sorted_string_list_.add(I),root1_) * 
                                   evecs_->get(a_sorted_string_list_.add(J),root2_) * ui64_slater_sign(Ib,p,q);
                    oprdm_b[p * ncmo_ + q] += value;
                    oprdm_b[q * ncmo_ + p] += value;
                }
            }
        }
    }

}

//void CI_RDMS::compute_2rdm_dynamic(std::vector<double>& tprdm_aa, 
//                                   std::vector<double>& tprdm_ab,
//                                   std::vector<double>& tprdm_bb){
//
//}
//void CI_RDMS::compute_3rdm_dynamic(std::vector<double>& tprdm_aaa, 
//                                   std::vector<double>& tprdm_aab,
//                                   std::vector<double>& tprdm_abb,
//                                   std::vector<double>& tprdm_bbb){
//
//}

}}
