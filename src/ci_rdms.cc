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


#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"

#include "reference.h"
#include "helpers.h"
#include "stl_bitset_determinant.h"
#include "ci_rdms.h"

namespace psi{ namespace forte {

// A class that takes the determinants and expansion
// coefficients and computes reduced density matrices.


CI_RDMS::CI_RDMS(Options &options,
                 std::shared_ptr<FCIIntegrals>& fci_ints,
                 std::vector<STLBitsetDeterminant>& det_space,
                 SharedMatrix evecs,
                 int root1,
                 int root2)
    : options_(options),
      fci_ints_(fci_ints),
      det_space_(det_space),
      evecs_(evecs),
      root1_(root1),
      root2_(root2)
{
    startup();
//    convert_to_string( det_space_ );
}

CI_RDMS::CI_RDMS( Options &options,
                  std::shared_ptr<FCIIntegrals> fci_ints,
                  std::vector<STLBitsetString> alfa_strings,
                  std::vector<STLBitsetString> beta_strings,
                  std::vector<std::vector<size_t>> a_to_b,
                  std::vector<std::vector<size_t>> b_to_a,
                  SharedMatrix evecs,
                  int root1,
                  int root2)
    : options_(options),
      fci_ints_(fci_ints),
      evecs_(evecs),
      alfa_strings_(alfa_strings),
      beta_strings_(beta_strings),
      a_to_b_(a_to_b),
      b_to_a_(b_to_a),
      root1_(root1),
      root2_(root2)
{
    startup();
}
                  

CI_RDMS::~CI_RDMS()
{
}

void CI_RDMS::startup()
{
	/* Get all of the required info from MOSpaceInfo to initialize the StringList*/	

	// The number of correlated molecular orbitals
//	ncmo_ = mo_space_info_->size("ACTIVE");
    ncmo_ = STLBitsetDeterminant::nmo_;
	ncmo2_ = ncmo_ * ncmo_;
	ncmo3_ = ncmo2_ * ncmo_;
	ncmo4_ = ncmo3_ * ncmo_;
	ncmo5_ = ncmo3_ * ncmo2_;

    na_ = det_space_[0].get_alfa_occ().size();
    nb_ = det_space_[0].get_beta_occ().size();

	// Dimension of the determinant space
	dim_space_ = det_space_.size();

	symmetry_ = 0;
	if (options_["ROOT_SYM"].has_changed()){
		symmetry_ = options_.get_int("ROOT_SYM");
	}
	
    print_ = false;

	one_map_done_ = false;

    cre_list_buffer_.resize(9);	

    if(print_){
	    outfile->Printf("\n  Computing RDMS");
	    outfile->Printf("\n  Number of active alpha electrons: %zu", na_);
	    outfile->Printf("\n  Number of active beta electrons: %zu", nb_);
	    outfile->Printf("\n  Number of correlated orbitals: %zu", ncmo_);
	}
}

void CI_RDMS::convert_to_string( std::vector<STLBitsetDeterminant>& space )
{
    Timer timer;
    size_t space_size = space.size();
    size_t nalfa_str = 0;
    size_t nbeta_str = 0;
    
    a_to_b_.clear();
    b_to_a_.clear();

    string_hash<size_t> alfa_map;
    string_hash<size_t> beta_map;

    int root = 0;
   
    if( options_["ROOT"].has_changed() ){     
        root = options_.get_int("ROOT");
    }

    SharedMatrix tmp(new Matrix("tmp",space_size,space_size));
    tmp->zero();

    for( size_t I = 0; I < space_size; ++I ){
        STLBitsetDeterminant det = space[I];
        STLBitsetString alfa;
        STLBitsetString beta;

        alfa.set_nmo( ncmo_ );
        beta.set_nmo( ncmo_ );
    
        for( size_t i = 0; i < ncmo_; ++i ){
            alfa.set_bit(i, det.get_alfa_bit(i)); 
            beta.set_bit(i, det.get_beta_bit(i)); 
        }

        size_t a_id;
        size_t b_id;
    
        // Once we find a new alfa string, add it to the list
        string_hash<size_t>::iterator a_it = alfa_map.find(alfa);
        if( a_it == alfa_map.end() ){
            a_id = nalfa_str;
            alfa_map[alfa] = a_id;
            nalfa_str++;
        }else{
            a_id = a_it->second;
        } 

        string_hash<size_t>::iterator b_it = beta_map.find(beta);
        if( b_it == beta_map.end() ){
            b_id = nbeta_str;
            beta_map[beta] = b_id;
            nbeta_str++;
        }else{
            b_id = b_it->second; 
        }

        a_to_b_.resize(nalfa_str);
        b_to_a_.resize(nbeta_str);

        alfa_strings_.resize(nalfa_str);
        beta_strings_.resize(nbeta_str);
 
        alfa_strings_[a_id] = alfa;
        beta_strings_[b_id] = beta;

        a_to_b_[a_id].push_back(b_id);
        b_to_a_[b_id].push_back(a_id);

        tmp->set(a_id, b_id, evecs_->get(I,root));

    }

    c_map_.assign(nalfa_str*nbeta_str,0.0);
    for( size_t a = 0; a < nalfa_str; ++a){
        for( size_t b = 0; b < nbeta_str; ++b){
            c_map_[a*nbeta_str + b] = tmp->get(a,b);
        }
    }

    outfile->Printf("\n  Time spent converting to strings: %f", timer.get());
}

void CI_RDMS::set_max_rdm( int rdm )
{
    max_rdm_ = rdm;
}

double CI_RDMS::get_energy( std::vector<double>& oprdm_a, 
							std::vector<double>& oprdm_b, 
							std::vector<double>& tprdm_aa,
							std::vector<double>& tprdm_bb,
							std::vector<double>& tprdm_ab)
{
	double nuc_rep = Process::environment.molecule()->nuclear_repulsion_energy();
    double scalar_energy = fci_ints_->frozen_core_energy() + fci_ints_->scalar_energy();
    double energy_1rdm = 0.0;
    double energy_2rdm = 0.0;

    for (size_t p = 0; p < ncmo_; ++p){
        for (size_t q = 0; q < ncmo_; ++q){
            energy_1rdm += oprdm_a[ncmo_ * p + q] * fci_ints_->oei_a(p,q);
            energy_1rdm += oprdm_b[ncmo_ * p + q] * fci_ints_->oei_b(p,q);
        }
    }

    for (size_t p = 0; p < ncmo_; ++p){
        for (size_t q = 0; q < ncmo_; ++q){
            for (size_t r = 0; r < ncmo_; ++r){
                for (size_t s = 0; s < ncmo_; ++s){
                    if (na_ >= 2)
                        energy_2rdm += 0.25 * tprdm_aa[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] * fci_ints_->tei_aa(p,q,r,s);
                    if ((na_ >= 1) and (nb_ >= 1))
                        energy_2rdm += tprdm_ab[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] * fci_ints_->tei_ab(p,q,r,s);
                    if (nb_ >= 2)
                        energy_2rdm += 0.25 * tprdm_bb[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] * fci_ints_->tei_bb(p,q,r,s);
                }
            }
        }
    }
    double total_energy = nuc_rep+ scalar_energy + energy_1rdm + energy_2rdm;
    
    if(print_){
        outfile->Printf("\n  Total Energy: %25.15f\n",total_energy);
        outfile->Printf("\n  Scalar Energy = %8.8f", scalar_energy);
        outfile->Printf("\n  energy_1rdm = %8.8f", energy_1rdm);
        outfile->Printf("\n  energy_2rdm = %8.8f", energy_2rdm);
        outfile->Printf("\n  nuclear_repulsion_energy = %8.8f", nuc_rep);
    }

	return total_energy;
}

void CI_RDMS::compute_1rdm_str( std::vector<double>& oprdm_a, std::vector<double>& oprdm_b)
{
    Timer one;
    get_one_map_str();
    if(print_) outfile->Printf("\n Time spent forming 1-map:  %1.6f", one.get());
    size_t nalfa = alfa_strings_.size();
    size_t nbeta = beta_strings_.size();

    oprdm_a.resize(ncmo2_, 0.0);
    oprdm_b.resize(ncmo2_, 0.0);

    for( size_t B = 0; B < nbeta; ++B ){
        auto avec = b_to_a_[B]; 
        for( size_t A = 0, maxa = avec.size(); A < maxa; ++A ){
            size_t AI = avec[A];
            for( size_t a = 0; a < na_; ++a ){
                std::pair<size_t, int> apair = a_ann_list_s_[AI*na_ + a];
                const size_t a_add = apair.first;
                const size_t p = std::abs(apair.second) - 1;
                const double sign_p = apair.second > 0.0 ? 1.0 : -1.0; 
                //if( a_add == 0 and p == 0 and J != 0 ) continue;
                for( size_t adet = cre_list_buffer_[0][a_add]; adet < cre_list_buffer_[0][a_add+1]; ++adet){
                    std::pair<size_t,int> aa_add = a_cre_list_s_[adet];
                    size_t q = std::abs(aa_add.second) - 1;
                    const double sign_q = aa_add.second > 0.0 ? 1.0 : -1.0;
                    const size_t astr = aa_add.first;
                    oprdm_a[q*ncmo_ + p] += c_map_[AI*nbeta + B] * c_map_[astr*nbeta + B] * sign_p * sign_q;
                }
            }
        }
    }

    for( size_t A = 0; A < nalfa; ++A ){
        auto bvec = a_to_b_[A]; 
        for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B ){
            size_t BI = bvec[B];
            for( size_t b = 0; b < nb_; ++b ){
                std::pair<size_t, int> bpair = b_ann_list_s_[BI*nb_ + b];
                const size_t b_add = bpair.first;
                const size_t p = std::abs(bpair.second) - 1;
                const double sign_p = bpair.second > 0.0 ? 1.0 : -1.0; 
                //if( b_add == 0 and p == 0 and J != 0 ) continue;
                for( size_t bdet = cre_list_buffer_[1][b_add]; bdet < cre_list_buffer_[1][b_add+1]; ++bdet){
                    std::pair<size_t,int> bb_add = b_cre_list_s_[bdet];
                    size_t q = std::abs(bb_add.second) - 1;
                    const double sign_q = bb_add.second > 0.0 ? 1.0 : -1.0;
                    const size_t bstr = bb_add.first;
                    oprdm_b[q*ncmo_ + p] += c_map_[A*nbeta + BI] * c_map_[A*nbeta + bstr] * sign_p * sign_q;
                }
            }
        }
    }
}


void CI_RDMS::compute_1rdm( std::vector<double>& oprdm_a, std::vector<double>& oprdm_b)
{
	Timer one;
	get_one_map();
	if( print_ ) outfile->Printf("\n  Time spent forming 1-map:   %1.6f", one.get());

	Timer build;
	oprdm_a.resize(ncmo2_, 0.0);
	oprdm_b.resize(ncmo2_, 0.0);
	for (size_t J = 0; J < dim_space_; ++J){
		for (auto& aJ_mo_sign : a_ann_list_[J]){
    	    const size_t aJ_add = aJ_mo_sign.first;
    	    size_t p = std::abs(aJ_mo_sign.second) - 1;
    	    const double sign_p = aJ_mo_sign.second > 0 ? 1.0 : -1.0;
    	    for (auto& aaJ_mo_sign : a_cre_list_[aJ_add]){
    	        size_t q = std::abs(aaJ_mo_sign.second) - 1;
    	        const double sign_q = aaJ_mo_sign.second > 0 ? 1.0 : -1.0;
    	        const size_t I = aaJ_mo_sign.first;
    	        oprdm_a[q*ncmo_ + p] += evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_p * sign_q;
    	    }
    	}
		for (auto& bJ_mo_sign : b_ann_list_[J]){
    	    const size_t bJ_add = bJ_mo_sign.first;
    	    const size_t p = std::abs(bJ_mo_sign.second) - 1;
    	    const double sign_p = bJ_mo_sign.second > 0 ? 1.0 : -1.0;
    	    for (auto& bbJ_mo_sign : b_cre_list_[bJ_add]){
    	        const size_t q = std::abs(bbJ_mo_sign.second) - 1;
    	        const double sign_q = bbJ_mo_sign.second > 0 ? 1.0 : -1.0;
    	        const size_t I = bbJ_mo_sign.first;
    	        oprdm_b[q*ncmo_ + p] += evecs_->get(J, root1_) * evecs_->get(I, root2_) * sign_p * sign_q;
    	    }
    	}
	}
	if( print_ ) outfile->Printf("\n  Time spent building 1-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_2rdm_str( std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab, std::vector<double>& tprdm_bb) 
{
    tprdm_aa.resize(ncmo4_, 0.0);
    tprdm_ab.resize(ncmo4_, 0.0);
    tprdm_bb.resize(ncmo4_, 0.0);

    size_t nalfa = alfa_strings_.size();
    size_t nbeta = beta_strings_.size();

    Timer two;
    get_two_map_str();
    if( print_) outfile->Printf("\n Time spent forming 2-map:   %1.6f", two.get());

    size_t nex = na_ * (na_ - 1)/2;

    Timer build;

    for( size_t B = 0; B < nbeta; ++B ){
        auto avec = b_to_a_[B];
        for( size_t A = 0, maxa = avec.size(); A < maxa; ++A ){
            size_t AI = avec[A];
            for( size_t a = 0; a < nex; ++a ){
                std::tuple<size_t,short,short> aaJ = aa_ann_list_s_[AI*nex + a];
                
                size_t aaJ_add = std::get<0>(aaJ);
                int p = std::abs( std::get<1>(aaJ) ) - 1;
                int q = std::get<2>(aaJ);
                double pq_sign = (std::get<1>(aaJ) > 0.0) ? 1.0 : -1.0;

                for( size_t adet = cre_list_buffer_[2][aaJ_add]; adet < cre_list_buffer_[2][aaJ_add+1]; ++adet){
                    std::tuple<size_t,short,short> aaaaJ = aa_cre_list_s_[adet];
                    size_t A_new = std::get<0>(aaaaJ);
                    int r = std::abs( std::get<1>(aaaaJ) ) - 1;
                    int s = std::get<2>(aaaaJ);
                    double rs_sign = (std::get<1>(aaaaJ) > 0.0) ? 1.0 : -1.0;

                    double rdm_element = pq_sign * rs_sign * c_map_[A_new*nbeta + B] * c_map_[AI*nbeta + B];

				    tprdm_aa[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] += rdm_element;
				    tprdm_aa[p*ncmo3_ + q*ncmo2_ + s*ncmo_ + r] -= rdm_element;
				    tprdm_aa[q*ncmo3_ + p*ncmo2_ + r*ncmo_ + s] -= rdm_element;
				    tprdm_aa[q*ncmo3_ + p*ncmo2_ + s*ncmo_ + r] += rdm_element;
                } 
            }
        }
    } 

    size_t nb_ex = nb_ * (nb_ - 1)/2;

    for( size_t A = 0; A < nalfa; ++A ){
        auto bvec = a_to_b_[A];
        for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B ){
            size_t BI = bvec[B];
            for( size_t b = 0; b < nb_ex; ++b ){
                std::tuple<size_t,short,short> bbJ = bb_ann_list_s_[BI*nb_ex + b];
                
                size_t bbJ_add = std::get<0>(bbJ);
                int p = std::abs( std::get<1>(bbJ) ) - 1;
                int q = std::get<2>(bbJ);
                double pq_sign = (std::get<1>(bbJ) > 0.0) ? 1.0 : -1.0;

                for( size_t bdet = cre_list_buffer_[3][bbJ_add]; bdet < cre_list_buffer_[3][bbJ_add+1]; ++bdet){
                    std::tuple<size_t,short,short> bbbbJ = bb_cre_list_s_[bdet];
                    size_t B_new = std::get<0>(bbbbJ);
                    int r = std::abs( std::get<1>(bbbbJ) ) - 1;
                    int s = std::get<2>(bbbbJ);
                    double rs_sign = (std::get<1>(bbbbJ) > 0.0) ? 1.0 : -1.0;

                    double rdm_element = pq_sign * rs_sign * c_map_[A*nbeta + BI] * c_map_[A*nbeta + B_new];

				    tprdm_bb[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] += rdm_element;
				    tprdm_bb[p*ncmo3_ + q*ncmo2_ + s*ncmo_ + r] -= rdm_element;
				    tprdm_bb[q*ncmo3_ + p*ncmo2_ + r*ncmo_ + s] -= rdm_element;
				    tprdm_bb[q*ncmo3_ + p*ncmo2_ + s*ncmo_ + r] += rdm_element;
                } 
            }
        }
    } 

    size_t nab_ex = na_ * nb_;

 //   for( int i = 0; i < c_map_.size(); ++i){
 //       outfile->Printf("\n  cmap[%d] = %f", i, c_map_[i]);
 //   }

    for( size_t A = 0, I = 0; A < nalfa; ++A ){
        auto bvec = a_to_b_[A];
        for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B, ++I ){
            size_t BI = bvec[B];
        //for( size_t BI = 0; BI < nbeta; ++BI ){
            for( size_t a = 0; a < na_; ++a ){
                for( size_t b = 0; b < nb_; ++b ){
                    std::tuple<size_t,short,short> abJ = ab_ann_list_s_[A*na_*nb_*nbeta + BI*na_*nb_ + a*nb_ +b];
                    
                    size_t ab_add = std::get<0>(abJ);
                    int p = std::abs( std::get<1>(abJ) ) - 1;
                    int q = std::abs( std::get<2>(abJ) ) - 1;
                    double pq_sign = (std::get<1>(abJ) > 0.0) ? 1.0 : -1.0;
                    pq_sign *= (std::get<2>(abJ) > 0.0) ? 1.0 : -1.0;

                    for( size_t det = cre_list_buffer_[4][ab_add]; det < cre_list_buffer_[4][ab_add+1]; ++det){
                       // std::tuple<size_t,size_t,short,short> ababJ = ab_cre_list_s_[det];
                        std::tuple<size_t,size_t,short,short> ababJ = ab_cre_list_s_[det ];
                        size_t A_new = std::get<0>(ababJ);
                        size_t B_new = std::get<1>(ababJ);
                        int r = std::abs( std::get<2>(ababJ) ) - 1;
                        int s = std::abs( std::get<3>(ababJ) ) - 1;
                        double rs_sign = (std::get<2>(ababJ) > 0.0) ? 1.0 : -1.0;
                        rs_sign *= (std::get<3>(ababJ) > 0.0) ? 1.0 : -1.0;

                        double rdm_element = pq_sign * rs_sign * c_map_[A*nbeta + BI] * c_map_[A_new*nbeta + B_new];
			    	    tprdm_ab[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] += rdm_element;
                    } 
                }
            }
        }
    } 

}

void CI_RDMS::compute_2rdm( std::vector<double>& tprdm_aa,std::vector<double>& tprdm_ab,std::vector<double>& tprdm_bb)
{
	tprdm_aa.resize(ncmo4_, 0.0);	
	tprdm_ab.resize(ncmo4_, 0.0);	
	tprdm_bb.resize(ncmo4_, 0.0);	

    Timer two;
	get_two_map();
	if( print_) outfile->Printf("\n  Time spent forming 2-map:   %1.6f", two.get());

	Timer build;
	for( size_t J = 0; J < dim_space_; ++J){
		// aaaa
		for( auto& aaJ_mo_sign : aa_ann_list_[J]){
			const size_t aaJ_add = std::get<0>(aaJ_mo_sign);

			const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
			const size_t q = std::get<2>(aaJ_mo_sign);
			const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;

			for( auto& aaaaJ_mo_sign : aa_cre_list_[aaJ_add] ){
				const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
				const size_t s = std::get<2>(aaaaJ_mo_sign);
				const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
				const size_t I = std::get<0>(aaaaJ_mo_sign);
				double rdm_element = evecs_->get(I,root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;	

				tprdm_aa[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] += rdm_element;
				tprdm_aa[p*ncmo3_ + q*ncmo2_ + s*ncmo_ + r] -= rdm_element;
				tprdm_aa[q*ncmo3_ + p*ncmo2_ + r*ncmo_ + s] -= rdm_element;
				tprdm_aa[q*ncmo3_ + p*ncmo2_ + s*ncmo_ + r] += rdm_element;
			}
		}

		// bbbb
		for( auto& bbJ_mo_sign : bb_ann_list_[J]){
			const size_t bbJ_add = std::get<0>(bbJ_mo_sign);

			const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
			const size_t q = std::get<2>(bbJ_mo_sign);
			const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;

			for( auto& bbbbJ_mo_sign : bb_cre_list_[bbJ_add] ){
				const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
				const size_t s = std::get<2>(bbbbJ_mo_sign);
				const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
				const size_t I = std::get<0>(bbbbJ_mo_sign);
				double rdm_element = evecs_->get(I,root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;	

				tprdm_bb[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] += rdm_element;
				tprdm_bb[p*ncmo3_ + q*ncmo2_ + s*ncmo_ + r] -= rdm_element;
				tprdm_bb[q*ncmo3_ + p*ncmo2_ + r*ncmo_ + s] -= rdm_element;
				tprdm_bb[q*ncmo3_ + p*ncmo2_ + s*ncmo_ + r] += rdm_element;
			}
		}
		// aabb
		for( auto& abJ_mo_sign : ab_ann_list_[J]){
			const size_t abJ_add = std::get<0>(abJ_mo_sign);

			const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
			const size_t q = std::get<2>(abJ_mo_sign);
			const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;

			for( auto& aabbJ_mo_sign : ab_cre_list_[abJ_add] ){
				const size_t r = std::abs(std::get<1>(aabbJ_mo_sign)) - 1;
				const size_t s = std::get<2>(aabbJ_mo_sign);
				const double sign_rs = std::get<1>(aabbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
				const size_t I = std::get<0>(aabbJ_mo_sign);
				double rdm_element = evecs_->get(I,root1_) * evecs_->get(J, root2_) * sign_pq * sign_rs;	

				tprdm_ab[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] += rdm_element;

			}
		}

	}	
	if(print_) outfile->Printf("\n  Time spent building 2-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_3rdm_str( std::vector<double>& tprdm_aaa,
							     std::vector<double>& tprdm_aab,
							     std::vector<double>& tprdm_abb, 
							     std::vector<double>& tprdm_bbb)
{
    size_t nalfa = alfa_strings_.size();
    size_t nbeta = beta_strings_.size();

	size_t ncmo5 = ncmo4_ * ncmo_;
	size_t ncmo6 = ncmo3_ * ncmo3_;

    size_t naa = na_ * (na_-1) / 2;
    size_t nbb = nb_ * (nb_-1) / 2;

    size_t naaa = na_ * (na_-1) * (na_-2)/6;
    size_t naab = na_ * (na_-1) *  nb_/2;
    size_t nabb = na_ * (nb_-1) *  nb_/2;
    size_t nbbb = nb_ * (nb_-1) * (nb_-2)/6;

	tprdm_aaa.resize(ncmo6, 0.0);
	tprdm_aab.resize(ncmo6, 0.0);
	tprdm_abb.resize(ncmo6, 0.0);
	tprdm_bbb.resize(ncmo6, 0.0);

	Timer three;
	get_three_map_str();
	if( print_ ) outfile->Printf("\n  Time spent forming 3-map:   %1.6f", three.get());
	
	Timer build;
    // AAA
    for( size_t B = 0; B < nbeta; ++B ){
        auto avec = b_to_a_[B];
        for( size_t A = 0, maxa = avec.size(); A < maxa; ++A ){
            size_t AI = avec[A];
            for( size_t a = 0; a < naaa; ++a ){
                std::tuple<size_t,short,short,short> aaJ = aaa_ann_list_s_[AI*naaa + a];
                
                size_t aaJ_add = std::get<0>(aaJ);
                int p = std::abs( std::get<1>(aaJ) ) - 1;
                int q = std::get<2>(aaJ);
                int r = std::get<3>(aaJ);
                double pqr_sign = (std::get<1>(aaJ) > 0.0) ? 1.0 : -1.0;

                for( size_t adet = cre_list_buffer_[5][aaJ_add]; adet < cre_list_buffer_[5][aaJ_add+1]; ++adet){
                    std::tuple<size_t,short,short,short> aaaaJ = aaa_cre_list_s_[adet];
                    size_t A_new = std::get<0>(aaaaJ);
                    int s = std::abs( std::get<1>(aaaaJ) ) - 1;
                    int t = std::get<2>(aaaaJ);
                    int u = std::get<3>(aaaaJ);
                    double stu_sign = (std::get<1>(aaaaJ) > 0.0) ? 1.0 : -1.0;

                    double rdm_element = pqr_sign * stu_sign * c_map_[A_new*nbeta + B] * c_map_[AI*nbeta + B];

				    tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				    tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				    tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				    tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				    tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				    tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				    tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				    tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				    tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				    tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				    tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				    tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

				    tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				    tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				    tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				    tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				    tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				    tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

				    tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				    tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				    tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				    tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				    tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				    tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				    tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				    tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				    tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				    tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				    tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				    tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				    tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				    tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				    tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				    tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				    tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				    tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 
                }
			}
		}
    }
    // BBB
    for( size_t A = 0; A < nalfa; ++A ){
        auto bvec = a_to_b_[A];
        for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B ){
            size_t BI = bvec[B];
            for( size_t b = 0; b < nbbb; ++b ){
                std::tuple<size_t,short,short,short> bbJ = bbb_ann_list_s_[BI*nbbb + b];
                
                size_t bbJ_add = std::get<0>(bbJ);
                int p = std::abs( std::get<1>(bbJ) ) - 1;
                int q =std::get<2>(bbJ);
                int r =std::get<3>(bbJ);
                double pqr_sign = (std::get<1>(bbJ) > 0.0) ? 1.0 : -1.0;

                for( size_t bdet = cre_list_buffer_[6][bbJ_add]; bdet < cre_list_buffer_[6][bbJ_add+1]; ++bdet){
                    std::tuple<size_t,short,short,short> bbbbJ = bbb_cre_list_s_[bdet];
                    size_t B_new = std::get<0>(bbbbJ);
                    int s = std::abs( std::get<1>(bbbbJ) ) - 1;
                    int t = std::get<2>(bbbbJ);
                    int u = std::get<3>(bbbbJ);
                    double stu_sign = (std::get<1>(bbbbJ) > 0.0) ? 1.0 : -1.0;

                    double rdm_element = pqr_sign * stu_sign * c_map_[A*nbeta + B_new] * c_map_[A*nbeta + BI];

				    tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				    tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				    tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				    tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				    tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				    tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				    tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				    tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				    tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				    tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				    tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				    tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

				    tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				    tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				    tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				    tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				    tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				    tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

				    tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				    tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				    tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				    tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				    tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				    tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				    tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				    tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				    tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				    tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				    tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				    tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				    tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				    tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				    tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				    tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				    tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				    tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 
                }
			}
		}
    }
    // AAB

    for( size_t A = 0; A < nalfa; ++A ){
        auto bvec = a_to_b_[A];
        for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B ){
            size_t BI = bvec[B];
            for( size_t a = 0; a < naa; ++a ){
                for( size_t b = 0; b < nb_; ++b ){
                    std::tuple<size_t,short,short,short> aabJ_ann = aab_ann_list_s_[A*naa*nb_*nbeta + BI*naa*nb_ + a*nb_ + b];

			        const size_t aabJ_add = std::get<0>(aabJ_ann);

			        const size_t p = std::abs(std::get<1>(aabJ_ann)) - 1;
			        const size_t q = std::get<2>(aabJ_ann);
			        const size_t r = std::abs(std::get<3>(aabJ_ann)) - 1;
			        double sign_pqr = std::get<1>(aabJ_ann) > 0.0 ? 1.0 : -1.0;			
			        sign_pqr *= std::get<3>(aabJ_ann) > 0.0 ? 1.0 : -1.0;			
			        
                    for( size_t aab_det = cre_list_buffer_[7][aabJ_add]; aab_det < cre_list_buffer_[7][aabJ_add+1]; ++aab_det ){
                        std::tuple<size_t,size_t,short,short,short> aabJ = aab_cre_list_s_[aab_det];

			        	const size_t s = std::abs(std::get<2>(aabJ)) - 1;
			        	const size_t t = std::get<3>(aabJ);
			        	const size_t u = std::abs(std::get<4>(aabJ)) - 1;

			        	double sign_stu = std::get<2>(aabJ) > 0.0 ? 1.0 : -1.0;
			        //	sign_stu *= std::get<3>(aabJ) > 0.0 ? 1.0 : -1.0;
			        	sign_stu *= std::get<4>(aabJ) > 0.0 ? 1.0 : -1.0;

			        	const size_t A_new = std::get<0>(aabJ);
			        	const size_t B_new = std::get<1>(aabJ);

                        double rdm_element = sign_pqr * sign_stu * c_map_[A*nbeta + BI] * c_map_[A_new*nbeta + B_new];

			        	tprdm_aab[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
			        	tprdm_aab[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
			        	tprdm_aab[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
			        	tprdm_aab[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 

			        }
                }
            }
		}
    }
    // ABB

    for( size_t A = 0; A < nalfa; ++A ){
        auto bvec = a_to_b_[A];
        for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B ){
            size_t BI = bvec[B];
            for( size_t b = 0; b < nbb; ++b ){
                for( size_t a = 0; a < na_; ++a ){
                    std::tuple<size_t,short,short,short> abbJ_ann = abb_ann_list_s_[A*nbb*na_*nbeta + BI*nbb*na_ + a*nbb + b];

			        const size_t abbJ_add = std::get<0>(abbJ_ann);

			        const size_t p = std::abs(std::get<1>(abbJ_ann)) - 1;
			        const size_t q = std::abs(std::get<2>(abbJ_ann)) - 1;
			        const size_t r = std::get<3>(abbJ_ann);
			        double sign_pqr = std::get<1>(abbJ_ann) > 0.0 ? 1.0 : -1.0;			
			        sign_pqr *= std::get<2>(abbJ_ann) > 0.0 ? 1.0 : -1.0;			
			        
                    for( size_t abb_det = cre_list_buffer_[8][abbJ_add]; abb_det < cre_list_buffer_[8][abbJ_add+1]; ++abb_det ){
                        std::tuple<size_t,size_t,short,short,short> abbJ = abb_cre_list_s_[abb_det];

			        	const size_t s = std::abs(std::get<2>(abbJ)) - 1;
			        	const size_t t = std::abs(std::get<3>(abbJ)) - 1;
			        	const size_t u = std::get<4>(abbJ);

			        	double sign_stu = std::get<2>(abbJ) > 0.0 ? 1.0 : -1.0;
			        	sign_stu *= std::get<3>(abbJ) > 0.0 ? 1.0 : -1.0;

			        	const size_t A_new = std::get<0>(abbJ);
			        	const size_t B_new = std::get<1>(abbJ);

                        double rdm_element = sign_pqr * sign_stu * c_map_[A*nbeta + BI] * c_map_[A_new*nbeta + B_new];

				        tprdm_abb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				        tprdm_abb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				        tprdm_abb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				        tprdm_abb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 

			        }
                }
            }
		}
    }
}

void CI_RDMS::compute_3rdm( std::vector<double>& tprdm_aaa,
							std::vector<double>& tprdm_aab,
							std::vector<double>& tprdm_abb, 
							std::vector<double>& tprdm_bbb)
{
	size_t ncmo5 = ncmo4_ * ncmo_;
	size_t ncmo6 = ncmo3_ * ncmo3_;

	tprdm_aaa.resize(ncmo6, 0.0);
	tprdm_aab.resize(ncmo6, 0.0);
	tprdm_abb.resize(ncmo6, 0.0);
	tprdm_bbb.resize(ncmo6, 0.0);

	Timer three;
	get_three_map();
	if( print_ ) outfile->Printf("\n  Time spent forming 3-map:   %1.6f", three.get());
	

	Timer build;
	for( size_t J = 0; J < dim_space_; ++J){
		// aaa aaa
		for( auto& aaaJ_mo_sign : aaa_ann_list_[J] ){
			const size_t aaaJ_add = std::get<0>(aaaJ_mo_sign);

			const size_t p = std::abs(std::get<1>(aaaJ_mo_sign)) - 1;
			const size_t q = std::get<2>(aaaJ_mo_sign);
			const size_t r = std::get<3>(aaaJ_mo_sign);
			const double sign_pqr = std::get<1>(aaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;			
			
			for( auto& a6J : aaa_cre_list_[aaaJ_add] ){
				const size_t s = std::abs(std::get<1>(a6J)) - 1;
				const size_t t = std::get<2>(a6J);
				const size_t u = std::get<3>(a6J);
				const double sign_stu = std::get<1>(a6J) > 0.0 ? 1.0 : -1.0;
				const size_t I = std::get<0>(a6J);

				double rdm_element = evecs_->get(I,root1_) * evecs_->get(J,root2_) * sign_pqr * sign_stu;

				tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				tprdm_aaa[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				tprdm_aaa[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

				tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				tprdm_aaa[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

				tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				tprdm_aaa[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				tprdm_aaa[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				tprdm_aaa[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

			}
		}
		// aab aab
		for( auto& aabJ_mo_sign : aab_ann_list_[J] ){
			const size_t aabJ_add = std::get<0>(aabJ_mo_sign);

			const size_t p = std::abs(std::get<1>(aabJ_mo_sign)) - 1;
			const size_t q = std::get<2>(aabJ_mo_sign);
			const size_t r = std::get<3>(aabJ_mo_sign);
			const double sign_pqr = std::get<1>(aabJ_mo_sign) > 0.0 ? 1.0 : -1.0;			
			
			for( auto& aabJ : aab_cre_list_[aabJ_add] ){
				const size_t s = std::abs(std::get<1>(aabJ)) - 1;
				const size_t t = std::get<2>(aabJ);
				const size_t u = std::get<3>(aabJ);
				const double sign_stu = std::get<1>(aabJ) > 0.0 ? 1.0 : -1.0;
				const size_t I = std::get<0>(aabJ);

				double rdm_element = evecs_->get(I,root1_) * evecs_->get(J,root2_) * sign_pqr * sign_stu;

				tprdm_aab[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				tprdm_aab[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				tprdm_aab[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				tprdm_aab[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 

			}
		}
		// abb abb
		for( auto& abbJ_mo_sign : abb_ann_list_[J] ){
			const size_t abbJ_add = std::get<0>(abbJ_mo_sign);

			const size_t p = std::abs(std::get<1>(abbJ_mo_sign)) - 1;
			const size_t q = std::get<2>(abbJ_mo_sign);
			const size_t r = std::get<3>(abbJ_mo_sign);
			const double sign_pqr = std::get<1>(abbJ_mo_sign) > 0.0 ? 1.0 : -1.0;			
			
			for( auto& abbJ : abb_cre_list_[abbJ_add] ){
				const size_t s = std::abs(std::get<1>(abbJ)) - 1;
				const size_t t = std::get<2>(abbJ);
				const size_t u = std::get<3>(abbJ);
				const double sign_stu = std::get<1>(abbJ) > 0.0 ? 1.0 : -1.0;
				const size_t I = std::get<0>(abbJ);

				double rdm_element = evecs_->get(I,root1_) * evecs_->get(J,root2_) * sign_pqr * sign_stu;

				tprdm_abb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				tprdm_abb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				tprdm_abb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				tprdm_abb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 

			}
		}
		// bbb bbb
		for( auto& bbbJ_mo_sign : bbb_ann_list_[J] ){
			const size_t bbbJ_add = std::get<0>(bbbJ_mo_sign);

			const size_t p = std::abs(std::get<1>(bbbJ_mo_sign)) - 1;
			const size_t q = std::get<2>(bbbJ_mo_sign);
			const size_t r = std::get<3>(bbbJ_mo_sign);
			const double sign_pqr = std::get<1>(bbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;			
			
			for( auto& b6J : bbb_cre_list_[bbbJ_add] ){
				const size_t s = std::abs(std::get<1>(b6J)) - 1;
				const size_t t = std::get<2>(b6J);
				const size_t u = std::get<3>(b6J);
				const double sign_stu = std::get<1>(b6J) > 0.0 ? 1.0 : -1.0;
				const size_t I = std::get<0>(b6J);

				double rdm_element = evecs_->get(I,root1_) * evecs_->get(J,root2_) * sign_pqr * sign_stu;

				tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				tprdm_bbb[ p*ncmo5 + q*ncmo4_ + r*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				tprdm_bbb[ p*ncmo5 + r*ncmo4_ + q*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

				tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				tprdm_bbb[ q*ncmo5 + p*ncmo4_ + r*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 

				tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				tprdm_bbb[ q*ncmo5 + r*ncmo4_ + p*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] += rdm_element; 
				tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] -= rdm_element; 
				tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] -= rdm_element; 
				tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] += rdm_element; 
				tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] -= rdm_element; 
				tprdm_bbb[ r*ncmo5 + p*ncmo4_ + q*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] += rdm_element; 

				tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + s*ncmo2_ + t*ncmo_ + u ] -= rdm_element; 
				tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + s*ncmo2_ + u*ncmo_ + t ] += rdm_element; 
				tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + u*ncmo2_ + t*ncmo_ + s ] += rdm_element; 
				tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + u*ncmo2_ + s*ncmo_ + t ] -= rdm_element; 
				tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + t*ncmo2_ + s*ncmo_ + u ] += rdm_element; 
				tprdm_bbb[ r*ncmo5 + q*ncmo4_ + p*ncmo3_ + t*ncmo2_ + u*ncmo_ + s ] -= rdm_element; 


			}
		}
	}

	if( print_ ) outfile->Printf("\n  Time spent building 3-rdm:   %1.6f", build.get());
}

void CI_RDMS::get_one_map_str()
{
    size_t nalfa = alfa_strings_.size();
    size_t nbeta = beta_strings_.size();

    // First the alpha
    {
        a_ann_list_s_.resize(nalfa * na_);
        size_t na_ann = 0;

        string_hash<size_t> map_a_ann;
        for( size_t A = 0; A < nalfa; ++A){
            STLBitsetString a_str = alfa_strings_[A];
            a_str.set_nmo(ncmo_);
            std::vector<int> aocc = a_str.get_occ();
    
            for( size_t i = 0; i < na_; ++i ){
                int ii = aocc[i];
                double sign = a_str.SlaterSign(ii); 
                a_str.set_bit(ii,false);
                size_t s_add;
                
    
                string_hash<size_t>::iterator a_it = map_a_ann.find(a_str);
                if( a_it == map_a_ann.end() ){
                    s_add = na_ann;
                    map_a_ann[a_str] = na_ann;
                    cre_list_buffer_[0].push_back(1);
                    na_ann++;
                }else{
                    s_add = a_it->second;
                    cre_list_buffer_[0][s_add]++;
                }
                a_ann_list_s_[A*na_ + i] = std::make_pair(s_add, (sign > 0.0 ) ? (ii+1) : (-ii-1) );
                a_str.set_bit(ii, true);
            }
        }
    
        size_t sum = 0;
        for( size_t i = 0; i < na_ann; ++i ){
            size_t current = cre_list_buffer_[0][i];
            cre_list_buffer_[0][i] = sum;
            sum += current;
        }
        cre_list_buffer_[0].push_back(a_ann_list_s_.size());
        std::vector<int> buffer(na_ann,0);
        
        //Build the creation list
        a_cre_list_s_.resize(nalfa*na_);

        for( size_t A = 0; A < nalfa; ++A){
            for( size_t a = 0; a < na_; ++a){
                auto apair = a_ann_list_s_[A*na_ + a];
                size_t adet = apair.first;
                a_cre_list_s_[cre_list_buffer_[0][adet] + buffer[adet]] = std::make_pair( A, apair.second);
                buffer[adet]++;
            }
        } 
    }
    // Then beta
    {
        b_ann_list_s_.resize(nbeta * nb_);
        size_t nb_ann = 0;

        string_hash<size_t> map_b_ann;
        for( size_t B = 0; B < nbeta; ++B ){
            STLBitsetString b_str = beta_strings_[B];
            b_str.set_nmo( ncmo_ );
            std::vector<int> bocc = b_str.get_occ();
    
            for( size_t i = 0; i < nb_; ++i ){
                int ii = bocc[i];
                double sign = b_str.SlaterSign(ii);
                b_str.set_bit(ii,false);
                size_t s_add;
                
                string_hash<size_t>::iterator b_it = map_b_ann.find(b_str);
                if( b_it == map_b_ann.end() ){
                    s_add = nb_ann;
                    map_b_ann[b_str] = nb_ann;
                    cre_list_buffer_[1].push_back(1);
                    nb_ann++;
                }else{
                    s_add = b_it->second;
                    cre_list_buffer_[1][s_add]++;
                }
                b_ann_list_s_[B*nb_ + i] = std::make_pair(s_add, (sign > 0.0) ? (ii+1) : (-ii-1) );
                b_str.set_bit(ii, true);
            }
        }
    
        size_t sum = 0;
        for( size_t i = 0; i < nb_ann; ++i ){
            size_t current = cre_list_buffer_[1][i];
            cre_list_buffer_[1][i] = sum;
            sum += current;
        }
        cre_list_buffer_[1][nb_ann] = b_ann_list_s_.size();
        std::vector<int> buffer(nb_ann,0);
        
        //Build the creation list
        b_cre_list_s_.resize(b_ann_list_s_.size(), std::make_pair(0,0) );
        for( size_t B = 0; B < nbeta; ++B){
            for( size_t b = 0; b < nb_; ++b){
                auto bpair = b_ann_list_s_[B*nb_ + b];
                size_t bdet = bpair.first;
                b_cre_list_s_[cre_list_buffer_[1][bdet] + buffer[bdet]] = std::make_pair( B, bpair.second);
                buffer[bdet]++;
            }
        } 
    }
}


void CI_RDMS::get_one_map()
{
	// The alpha and beta annihilation lists
	a_ann_list_.resize(dim_space_);
	b_ann_list_.resize(dim_space_);

	// The N-1 maps
	det_hash a_ann_map;
	det_hash b_ann_map;

	// Number of annihilations on alfa and beta strings
	size_t na_ann = 0;
	size_t nb_ann = 0;

	if( print_ )outfile->Printf("\n\n  Generating one-particle maps.");

	for( size_t I = 0; I < dim_space_; ++I){
		STLBitsetDeterminant detI = det_space_[I];
		
		// Alpha and beta occupation vectors
		std::vector<int> aocc = detI.get_alfa_occ();
		std::vector<int> bocc = detI.get_beta_occ();

		int noalfa = aocc.size();
		int nobeta = bocc.size();
	
		std::vector< std::pair< size_t, short > > a_ann(noalfa);		
		std::vector< std::pair< size_t, short > > b_ann(nobeta);

		// Form alpha annihilation lists
		for( int i = 0; i < noalfa; ++i){
			int ii = aocc[i];
			STLBitsetDeterminant detJ(detI);

			// Annihilate bit ii, get the sign
			detJ.set_alfa_bit(ii, false);
			double sign = detI.slater_sign_alpha(ii);

		    det_hash_it hash_it = a_ann_map.find(detJ);
			size_t detJ_add;
			if( hash_it == a_ann_map.end() ){
				detJ_add = na_ann;
				a_ann_map[detJ] = na_ann;
				na_ann++;
			}else{
				detJ_add = hash_it->second;
			}	
			a_ann[i] = std::make_pair(detJ_add, ( sign > 0.5 ) ? (ii + 1) : (-ii - 1)); 
		}		
		a_ann_list_[I] = a_ann;

		// Form beta annihilation lists
		for( int i = 0; i < nobeta; ++i){
			int ii = bocc[i];
			STLBitsetDeterminant detJ(detI);

			// Annihilate bit ii, get the sign
			detJ.set_beta_bit(ii, false);
			double sign = detI.slater_sign_beta(ii);

		    det_hash_it hash_it = b_ann_map.find(detJ);
			size_t detJ_add;
			if( hash_it == b_ann_map.end() ){
				detJ_add = nb_ann;
				b_ann_map[detJ] = nb_ann;
				nb_ann++;
			}else{
				detJ_add = hash_it->second;
			}	
			b_ann[i] = std::make_pair(detJ_add, ( sign > 0.5 ) ? (ii + 1) : (-ii - 1)); 
		}		
		b_ann_list_[I] = b_ann;
	} // Done with annihilation lists
	
	// Generate alpha and beta creation lists
	a_cre_list_.resize(a_ann_map.size());
	b_cre_list_.resize(b_ann_map.size());

	for( size_t I = 0; I < dim_space_; ++I){
		const std::vector< std::pair< size_t, short>>& a_ann = a_ann_list_[I];
		for(const std::pair<size_t, short>& Jsign : a_ann){
			size_t J = Jsign.first;
			short sign = Jsign.second;
			a_cre_list_[J].push_back(std::make_pair(I,sign));
		}
		const std::vector< std::pair< size_t, short>>& b_ann = b_ann_list_[I];
		for(const std::pair<size_t, short>& Jsign : b_ann){
			size_t J = Jsign.first;
			short sign = Jsign.second;
			b_cre_list_[J].push_back(std::make_pair(I,sign));
		}
	}
	one_map_done_ = true;
}

void CI_RDMS::get_two_map_str()
{
    size_t nalfa = alfa_strings_.size();
    size_t nbeta = beta_strings_.size();

    // Map alpha strings by two excitations
    {
        size_t naa_ann = 0;
        size_t nex = na_* (na_ - 1)/2;
        aa_ann_list_s_.resize(nalfa * nex);

        string_hash<size_t> map_aa_ann;
        for( size_t A = 0; A < nalfa; ++A){
            STLBitsetString astr = alfa_strings_[A];
            astr.set_nmo(ncmo_);
            std::vector<int> aocc = astr.get_occ();

            for( size_t i = 0, ij = 0; i < na_; ++i){
                for( size_t j = i + 1; j < na_; ++j, ++ij){
                    int ii = aocc[i];
                    int jj = aocc[j];
                    double sign = astr.SlaterSign(ii) * astr.SlaterSign(jj);
                    
                    astr.set_bit(ii,false); 
                    astr.set_bit(jj,false); 
      
                    string_hash<size_t>::iterator a_it = map_aa_ann.find(astr); 
                    size_t a_add;
                
                    if( a_it == map_aa_ann.end()) {
                        a_add = naa_ann;
                        map_aa_ann[astr] = naa_ann;
                        naa_ann++;
                        cre_list_buffer_[2].push_back(1);
                    }else{
                        a_add = a_it->second;    
                        cre_list_buffer_[2][a_add]++;
                    }

                    aa_ann_list_s_[ A*nex + ij ] = std::make_tuple( a_add, (sign > 0.0) ? (ii+1) : (-ii-1), jj);
                    astr.set_bit(ii, true); 
                    astr.set_bit(jj, true); 
                }
            }    
        }
        size_t sum = 0; 
        for( size_t i = 0; i < naa_ann; ++i){
            size_t current = cre_list_buffer_[2][i];
            cre_list_buffer_[2][i] = sum;
            sum += current;
        }
        cre_list_buffer_[2].push_back( aa_ann_list_s_.size() );

        aa_cre_list_s_.resize( aa_ann_list_s_.size() );

        std::vector<int> buffer(naa_ann, 0);
        for( size_t A = 0; A < nalfa; ++A){
            for( size_t a = 0; a < nex; ++ a ){
                std::tuple<size_t, short, short> Ja = aa_ann_list_s_[A*nex + a];
                size_t A_ann = std::get<0>(Ja);
                short i = std::get<1>(Ja);
                short j = std::get<2>(Ja);
                aa_cre_list_s_[ cre_list_buffer_[2][A_ann] + buffer[A_ann] ] = std::make_tuple(A, i, j);
                buffer[A_ann]++;
            }
        }
    }

    // Map beta strings by two beta excitations
    {
        size_t nbb_ann = 0;
        size_t nex = nb_* (nb_ - 1)/2;
        bb_ann_list_s_.resize(nbeta * nex);

        string_hash<size_t> map_bb_ann;
        for( size_t B = 0; B < nbeta; ++B){
            STLBitsetString bstr = beta_strings_[B];
            bstr.set_nmo(ncmo_);
            std::vector<int> bocc = bstr.get_occ();

            for( size_t i = 0, ij = 0; i < nb_; ++i){
                for( size_t j = i + 1; j < nb_; ++j, ++ij){
                    int ii = bocc[i];
                    int jj = bocc[j];
                    double sign = bstr.SlaterSign(ii) * bstr.SlaterSign(jj);
                    
                    bstr.set_bit(ii,false); 
                    bstr.set_bit(jj,false); 
      
                    string_hash<size_t>::iterator b_it = map_bb_ann.find(bstr); 
                    size_t b_add;
                
                    if( b_it == map_bb_ann.end()) {
                        b_add = nbb_ann;
                        map_bb_ann[bstr] = nbb_ann;
                        nbb_ann++;
                        cre_list_buffer_[3].push_back(1);
                    }else{
                        b_add = b_it->second;    
                        cre_list_buffer_[3][b_add]++;
                    }

                    bb_ann_list_s_[ B*nex + ij ] = std::make_tuple( b_add, (sign > 0.0) ? (ii+1) : (-ii-1), jj);
                    bstr.set_bit(ii, true); 
                    bstr.set_bit(jj, true); 
                }
            }    
        }
        size_t sum = 0; 
        for( size_t i = 0; i < nbb_ann; ++i){
            size_t current = cre_list_buffer_[3][i];
            cre_list_buffer_[3][i] = sum;
            sum += current;
        }
        cre_list_buffer_[3].push_back( bb_ann_list_s_.size() );

        bb_cre_list_s_.resize( bb_ann_list_s_.size() );

        std::vector<int> buffer(nbb_ann, 0);
        for( size_t B = 0; B < nbeta; ++B){
            for( size_t b = 0; b < nex; ++b ){
                std::tuple<size_t, short, short> Jb = bb_ann_list_s_[B*nex + b];
                size_t B_ann = std::get<0>(Jb);
                short i = std::get<1>(Jb);
                short j = std::get<2>(Jb);
                bb_cre_list_s_[ cre_list_buffer_[3][B_ann] + buffer[B_ann] ] = std::make_tuple(B, i, j);
                buffer[B_ann]++;
            }
        }
    }

    // Map alpha-beta excitations
    {
        size_t nab_ann = 0;

        ab_ann_list_s_.resize(nalfa*nbeta*na_*nb_);
        std::vector<int> counter(nalfa*nbeta*na_*nb_, -1);
    
        for( size_t A = 0, I = 0; A < nalfa; ++A){
            auto bvec = a_to_b_[A];
            for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B, ++I ){
                size_t detB = bvec[B];
                for( size_t a = 0, ab = 0; a < na_; ++a ){
                    std::pair<size_t, short> JA = a_ann_list_s_[A*na_ + a];
                    int iia = JA.second;
                    size_t n_JA = JA.first;
                    for( size_t b = 0; b < nb_; ++b, ++ab ){
                        std::pair<size_t, short> JB = b_ann_list_s_[detB*nb_ + b];
                        int iib = JB.second;
                        size_t n_JB = JB.first;

                        size_t ab_add;                        

                        if( counter[n_JA*b_ann_list_s_.size() + n_JB] == -1 ){
                            ab_add = nab_ann;
                            counter[n_JA*b_ann_list_s_.size() + n_JB] = nab_ann;    
                            nab_ann++;
                            cre_list_buffer_[4].push_back(1);
                        }else{
                            ab_add = counter[n_JA*b_ann_list_s_.size() + n_JB];
                            cre_list_buffer_[4][ab_add]++;
                        }

                        ab_ann_list_s_[A*na_*nb_*nbeta + detB*na_*nb_ + a*nb_ + b] = std::make_tuple(ab_add, iia, iib);
                       // if(std::abs(iia) == 1 and std::abs(iib) == 1) outfile->Printf("\n\n A: %zu, B: %zu, iia: %d, iib: %d", A, detB, std::abs(iia)-1,std::abs(iib)-1);
                       // if(std::abs(iia) == 2 and std::abs(iib) == 2) outfile->Printf("\n\n A: %zu, B: %zu, iia: %d, iib: %d", A, detB, std::abs(iia)-1,std::abs(iib)-1);
                    }
                }        
            }
        }


        size_t sum = 0; 
        for( size_t i = 0; i <= nab_ann; ++i){
            size_t current = cre_list_buffer_[4][i];
            cre_list_buffer_[4][i] = sum;
            sum += current;
        }
        //cre_list_buffer_[4].push_back( ab_ann_list_s_.size() );
        //cre_list_buffer_[4].push_back(nab_ann );

       // for( size_t i = 0; i < cre_list_buffer_[4].size(); ++i){
       //     outfile->Printf("\n  crelist[%zu] = %zu", i, cre_list_buffer_[4][i]);
       // }

        ab_cre_list_s_.resize( ab_ann_list_s_.size() );
        
        std::vector<int> buffer(nab_ann, 0);
        for( size_t A = 0, I = 0; A < nalfa; ++A){
            auto bvec = a_to_b_[A];
            for( size_t B = 0, maxB = bvec.size(); B < maxB; ++B, ++I){
           // for( size_t B = 0; B < nbeta; ++B){
                for( size_t a = 0; a < na_ ; ++a){
                    for( size_t b = 0; b < nb_ ; ++b){
                        std::tuple<size_t,short, short> Jab = ab_ann_list_s_[A*na_*nb_*nbeta + bvec[B]*na_*nb_ + a*nb_ + b];
                        size_t AB_ann = std::get<0>(Jab);
                        short i = std::get<1>(Jab);
                        short j = std::get<2>(Jab);
                        ab_cre_list_s_[ cre_list_buffer_[4][AB_ann] + buffer[AB_ann] ] = std::make_tuple(A,bvec[B], i, j);
                        buffer[AB_ann]++;
                    }
                }
            }
        }
    }
}

void CI_RDMS::get_two_map()
{
	aa_ann_list_.resize(dim_space_);
	ab_ann_list_.resize(dim_space_);
	bb_ann_list_.resize(dim_space_);

	det_hash aa_ann_map;
	det_hash ab_ann_map;
	det_hash bb_ann_map;

	size_t naa_ann = 0;
	size_t nab_ann = 0;
	size_t nbb_ann = 0;

	if( print_ ) outfile->Printf("\n  Generating two-particle maps.");

	for( size_t I = 0; I < dim_space_; ++I){
		STLBitsetDeterminant detI = det_space_[I];
		
		std::vector<int> aocc = detI.get_alfa_occ();
		std::vector<int> bocc = detI.get_beta_occ();

		int noalfa = aocc.size();
		int  nobeta = bocc.size();

		std::vector<std::tuple<size_t, short, short>> aa_ann(noalfa * (noalfa -1) / 2);	
		std::vector<std::tuple<size_t, short, short>> ab_ann(noalfa * nobeta);	
		std::vector<std::tuple<size_t, short, short>> bb_ann(nobeta * (nobeta -1) / 2);	

		// alpha-alpha annihilations
		for( int i = 0, ij = 0; i < noalfa; ++i){
			for( int j = i + 1; j < noalfa; ++j, ++ij){
				int ii = aocc[i];
				int jj = aocc[j];

				STLBitsetDeterminant detJ(detI);
				detJ.set_alfa_bit(ii,false);
				detJ.set_alfa_bit(jj,false);
				
				double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_alpha(jj);
				
				det_hash_it hash_it = aa_ann_map.find(detJ);
				size_t detJ_add;
				if( hash_it == aa_ann_map.end()){
					detJ_add = naa_ann;
					aa_ann_map[detJ] = naa_ann;
					naa_ann++;
				}else{
					detJ_add = hash_it->second;
				}
				aa_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
			}
		}
		aa_ann_list_[I] = aa_ann;

		// beta-beta  annihilations
		for( int i = 0, ij = 0; i < nobeta; ++i){
			for( int j = i + 1; j < nobeta; ++j, ++ij){
				int ii = bocc[i];
				int jj = bocc[j];

				STLBitsetDeterminant detJ(detI);
				detJ.set_beta_bit(ii,false);
				detJ.set_beta_bit(jj,false);
				
				double sign = detI.slater_sign_beta(ii) * detI.slater_sign_beta(jj);
				
				det_hash_it hash_it = bb_ann_map.find(detJ);
				size_t detJ_add;
				if( hash_it == bb_ann_map.end()){
					detJ_add = nbb_ann;
					bb_ann_map[detJ] = nbb_ann;
					nbb_ann++;
				}else{
					detJ_add = hash_it->second;
				}
				bb_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
			}
		}
		bb_ann_list_[I] = bb_ann;
		
		// alpha-beta  annihilations
		for( int i = 0, ij = 0; i < noalfa; ++i){
			for( int j = 0; j < nobeta; ++j, ++ij){
				int ii = aocc[i];
				int jj = bocc[j];

				STLBitsetDeterminant detJ(detI);
				detJ.set_alfa_bit(ii,false);
				detJ.set_beta_bit(jj,false);
				
				double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_beta(jj);
				
				det_hash_it hash_it = ab_ann_map.find(detJ);
				size_t detJ_add;
				if( hash_it == ab_ann_map.end()){
					detJ_add = nab_ann;
					ab_ann_map[detJ] = nab_ann;
					nab_ann++;
				}else{
					detJ_add = hash_it->second;
				}
				ab_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
			}
		}
		ab_ann_list_[I] = ab_ann;
	} // Done building 2-hole lists

	aa_cre_list_.resize(aa_ann_map.size());
	ab_cre_list_.resize(ab_ann_map.size());
	bb_cre_list_.resize(bb_ann_map.size());

	for( size_t I = 0; I < dim_space_; ++I){
		// alpha-alpha
		const std::vector<std::tuple<size_t, short, short>>& aa_ann = aa_ann_list_[I];
		for( const std::tuple<size_t, short, short>& Jsign : aa_ann){
			size_t J = std::get<0>(Jsign);
			short i  = std::get<1>(Jsign);
			short j  = std::get<2>(Jsign);
			aa_cre_list_[J].push_back(std::make_tuple(I,i,j));
		}
		// beta-beta
		const std::vector<std::tuple<size_t, short, short>>& bb_ann = bb_ann_list_[I];
		for( const std::tuple<size_t, short, short>& Jsign : bb_ann){
			size_t J = std::get<0>(Jsign);
			short i  = std::get<1>(Jsign);
			short j  = std::get<2>(Jsign);
			bb_cre_list_[J].push_back(std::make_tuple(I,i,j));
		}
		// alpha-alpha
		const std::vector<std::tuple<size_t, short, short>>& ab_ann = ab_ann_list_[I];
		for( const std::tuple<size_t, short, short>& Jsign : ab_ann){
			size_t J = std::get<0>(Jsign);
			short i  = std::get<1>(Jsign);
			short j  = std::get<2>(Jsign);
			ab_cre_list_[J].push_back(std::make_tuple(I,i,j));
		}
	}

}

void CI_RDMS::get_three_map_str()
{
    size_t nalfa = alfa_strings_.size();
    size_t nbeta = beta_strings_.size();

    size_t naaa = na_ * (na_-1) * (na_-2)/6;
    size_t naab = na_ * (na_-1) *  nb_/2;
    size_t nabb = na_ * (nb_-1) *  nb_/2;
    size_t nbbb = nb_ * (nb_-1) * (nb_-2)/6;


    // AAA
    {
        aaa_ann_list_s_.resize(nalfa * naaa);
        size_t naaa_ann = 0;

        string_hash<size_t> map_aaa_ann;
        for( size_t A = 0; A < nalfa; ++A){
            STLBitsetString astr = alfa_strings_[A];
            astr.set_nmo(ncmo_);
            std::vector<int> aocc = astr.get_occ();

            for( size_t i = 0, ijk = 0; i < na_; ++i){
                for( size_t j = i + 1; j < na_; ++j){
                    for( size_t k = j + 1; k < na_; ++k, ++ijk){
                        int ii = aocc[i];
                        int jj = aocc[j];
                        int kk = aocc[k];
                        double sign = astr.SlaterSign(ii) * astr.SlaterSign(jj) * astr.SlaterSign(jj);
                        
                        astr.set_bit(ii,false); 
                        astr.set_bit(jj,false); 
                        astr.set_bit(kk,false); 
      
                        string_hash<size_t>::iterator a_it = map_aaa_ann.find(astr); 
                        size_t a_add;
                    
                        if( a_it == map_aaa_ann.end()) {
                            a_add = naaa_ann;
                            map_aaa_ann[astr] = naaa_ann;
                            naaa_ann++;
                            cre_list_buffer_[5].push_back(1);
                        }else{
                            a_add = a_it->second;    
                            cre_list_buffer_[5][a_add]++;
                        }

                        aaa_ann_list_s_[ A*naaa + ijk ] = std::make_tuple( a_add, (sign > 0.0) ? (ii+1) : (-ii-1), jj, kk);
                        astr.set_bit(ii, true); 
                        astr.set_bit(jj, true); 
                        astr.set_bit(kk, true); 
                    }
                }
            }    
        }

        size_t sum = 0; 
        for( size_t i = 0; i < naaa_ann; ++i){
            size_t current = cre_list_buffer_[5][i];
            cre_list_buffer_[5][i] = sum;
            sum += current;
        }
        cre_list_buffer_[5].push_back( aaa_ann_list_s_.size() );

        aaa_cre_list_s_.resize( aaa_ann_list_s_.size() );

        std::vector<int> buffer(naaa_ann, 0);
        for( size_t A = 0; A < nalfa; ++A){
            for( size_t a = 0; a < naaa; ++ a ){
                std::tuple<size_t, short, short, short> Ja = aaa_ann_list_s_[A*naaa + a];
                size_t A_ann = std::get<0>(Ja);
                short i = std::get<1>(Ja);
                short j = std::get<2>(Ja);
                short k = std::get<3>(Ja);
                aaa_cre_list_s_[ cre_list_buffer_[5][A_ann] + buffer[A_ann] ] = std::make_tuple(A, i, j, k);
                buffer[A_ann]++;
            }
        }

    }

    // BBB
    {
        bbb_ann_list_s_.resize(nbeta * nbbb);
        size_t nbbb_ann = 0;

        string_hash<size_t> map_bbb_ann;
        for( size_t B = 0; B < nbeta; ++B){
            STLBitsetString bstr = beta_strings_[B];
            bstr.set_nmo(ncmo_);
            std::vector<int> bocc = bstr.get_occ();

            for( size_t i = 0, ijk = 0; i < nb_; ++i){
                for( size_t j = i + 1; j < nb_; ++j){
                    for( size_t k = j + 1; k < nb_; ++k, ++ijk){
                        int ii = bocc[i];
                        int jj = bocc[j];
                        int kk = bocc[k];
                        double sign = bstr.SlaterSign(ii) * bstr.SlaterSign(jj) * bstr.SlaterSign(jj);
                        
                        bstr.set_bit(ii,false); 
                        bstr.set_bit(jj,false); 
                        bstr.set_bit(kk,false); 
      
                        string_hash<size_t>::iterator b_it = map_bbb_ann.find(bstr); 
                        size_t b_add;
                    
                        if( b_it == map_bbb_ann.end()) {
                            b_add = nbbb_ann;
                            map_bbb_ann[bstr] = nbbb_ann;
                            nbbb_ann++;
                            cre_list_buffer_[6].push_back(1);
                        }else{
                            b_add = b_it->second;    
                            cre_list_buffer_[6][b_add]++;
                        }

                        bbb_ann_list_s_[ B*nbbb + ijk ] = std::make_tuple( b_add, (sign > 0.0) ? (ii+1) : (-ii-1), jj, kk);
                        bstr.set_bit(ii, true); 
                        bstr.set_bit(jj, true); 
                        bstr.set_bit(kk, true); 
                    }
                }
            }    
        }

        size_t sum = 0; 
        for( size_t i = 0; i < nbbb_ann; ++i){
            size_t current = cre_list_buffer_[6][i];
            cre_list_buffer_[6][i] = sum;
            sum += current;
        }
        cre_list_buffer_[6].push_back( bbb_ann_list_s_.size() );

        bbb_cre_list_s_.resize( bbb_ann_list_s_.size() );

        std::vector<int> buffer(nbbb_ann, 0);
        for( size_t B = 0; B < nbeta; ++B){
            for( size_t b = 0; b < nbbb; ++ b ){
                std::tuple<size_t, short, short, short> Jb = bbb_ann_list_s_[B*nbbb + b];
                size_t B_ann = std::get<0>(Jb);
                short i = std::get<1>(Jb);
                short j = std::get<2>(Jb);
                short k = std::get<3>(Jb);
                bbb_cre_list_s_[ cre_list_buffer_[6][B_ann] + buffer[B_ann] ] = std::make_tuple(B, i, j, k);
                buffer[B_ann]++;
            }
        }
    }

    // AAB
    {
        size_t naa_ex = na_*(na_-1)/2;
        size_t naab_ann = 0;

        aab_ann_list_s_.resize(nalfa*naa_ex * nbeta*nb_);
        std::vector<int> counter(nalfa*naa_ex * nbeta*nb_, -1);
    
        for( size_t A = 0; A < nalfa; ++A){
            auto bvec = a_to_b_[A];
            for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B ){
                size_t detB = bvec[B];
                for( size_t a = 0; a < naa_ex; ++a ){
                    std::tuple<size_t,short, short> JA = aa_ann_list_s_[A*naa_ex + a];
                    int iia = std::get<1>(JA);
                    int jja = std::get<2>(JA);
                    size_t n_JA = std::get<0>(JA);
                    for( size_t b = 0; b < nb_; ++b ){
                        std::pair<size_t, short> JB = b_ann_list_s_[detB*nb_ + b];
                        int iib = JB.second;
                        size_t n_JB = JB.first;

                        size_t aab_add;                        

                        if( counter[n_JA*b_ann_list_s_.size() + n_JB] == -1 ){
                            aab_add = naab_ann;
                            counter[n_JA*b_ann_list_s_.size() + n_JB] = naab_ann;    
                            naab_ann++;
                            cre_list_buffer_[7].push_back(1);
                        }else{
                            aab_add = counter[n_JA*b_ann_list_s_.size() + n_JB];
                            cre_list_buffer_[7][aab_add]++;
                        }

                        aab_ann_list_s_[A*naa_ex*nb_*nbeta + detB*naa_ex*nb_ + b*naa_ex + a] = std::make_tuple(aab_add, iia, jja, iib);
                    }
                }        
            }
        }


        size_t sum = 0; 
        for( size_t i = 0; i <= naab_ann; ++i){
            size_t current = cre_list_buffer_[7][i];
            cre_list_buffer_[7][i] = sum;
            sum += current;
        }

        aab_cre_list_s_.resize( aab_ann_list_s_.size() );
        
        std::vector<int> buffer(naab_ann, 0);
        for( size_t A = 0; A < nalfa; ++A){
            auto bvec = a_to_b_[A];
            for( size_t B = 0, maxB = bvec.size(); B < maxB; ++B){
                for( size_t a = 0; a < naa_ex ; ++a){
                    for( size_t b = 0; b < nb_ ; ++b){
                        std::tuple<size_t,short, short, short> Jaab = aab_ann_list_s_[A*naa_ex*nb_*nbeta + bvec[B]*naa_ex*nb_ + b*naa_ex + a];
                        size_t AAB_ann = std::get<0>(Jaab);
                        short ia = std::get<1>(Jaab);
                        short ja = std::get<2>(Jaab);
                        short kb = std::get<3>(Jaab);
                        aab_cre_list_s_[ cre_list_buffer_[7][AAB_ann] + buffer[AAB_ann] ] = std::make_tuple(A,bvec[B], ia, ja,kb);
                        buffer[AAB_ann]++;
                    }
                }
            }
        }

    }

    // ABB

    {
        size_t nbb_ex = nb_*(nb_-1)/2;
        size_t nabb_ann = 0;

        abb_ann_list_s_.resize(nalfa*na_ * nbeta*nbb_ex);
        std::vector<int> counter(nalfa*na_ * nbeta*nbb_ex, -1);
    
        for( size_t A = 0; A < nalfa; ++A){
            auto bvec = a_to_b_[A];
            for( size_t B = 0, maxb = bvec.size(); B < maxb; ++B ){
                size_t detB = bvec[B];
                for( size_t b = 0; b < nbb_ex; ++b ){
                    std::tuple<size_t,short, short> JB = bb_ann_list_s_[detB*nbb_ex + b];
                    int iib = std::get<1>(JB);
                    int jjb = std::get<2>(JB);
                    size_t n_JB = std::get<0>(JB);
                    for( size_t a = 0; a < na_; ++a ){
                        std::pair<size_t, short> JA = a_ann_list_s_[A*na_ + a];
                        int iia = JA.second;
                        size_t n_JA = JA.first;

                        size_t abb_add;                        

                        if( counter[n_JA*bb_ann_list_s_.size() + n_JB] == -1 ){
                            abb_add = nabb_ann;
                            counter[n_JA*bb_ann_list_s_.size() + n_JB] = nabb_ann;    
                            nabb_ann++;
                            cre_list_buffer_[8].push_back(1);
                        }else{
                            abb_add = counter[n_JA*bb_ann_list_s_.size() + n_JB];
                            cre_list_buffer_[8][abb_add]++;
                        }

                        abb_ann_list_s_[A*na_*nbb_ex*nbeta + detB*na_*nbb_ex + a*nbb_ex + b] = std::make_tuple(abb_add, iia, iib, jjb);
                    }
                }        
            }
        }


        size_t sum = 0; 
        for( size_t i = 0; i <= nabb_ann; ++i){
            size_t current = cre_list_buffer_[8][i];
            cre_list_buffer_[8][i] = sum;
            sum += current;
        }

        abb_cre_list_s_.resize( abb_ann_list_s_.size() );
        
        std::vector<int> buffer(nabb_ann, 0);
        for( size_t A = 0; A < nalfa; ++A){
            auto bvec = a_to_b_[A];
            for( size_t B = 0, maxB = bvec.size(); B < maxB; ++B){
                for( size_t b = 0; b < nbb_ex ; ++b){
                    for( size_t a = 0; a < na_ ; ++a){
                        std::tuple<size_t,short, short, short> Jabb = abb_ann_list_s_[A*na_*nbb_ex*nbeta + bvec[B]*na_*nbb_ex + a*nbb_ex + b];
                        size_t ABB_ann = std::get<0>(Jabb);
                        short ia = std::get<1>(Jabb);
                        short jb = std::get<2>(Jabb);
                        short kb = std::get<3>(Jabb);
                        abb_cre_list_s_[ cre_list_buffer_[8][ABB_ann] + buffer[ABB_ann] ] = std::make_tuple(A,bvec[B], ia, jb,kb);
                        buffer[ABB_ann]++;
                    }
                }
            }
        }
    }

}

void CI_RDMS::get_three_map()
{
	aaa_ann_list_.resize(dim_space_);
	aab_ann_list_.resize(dim_space_);
	abb_ann_list_.resize(dim_space_);
	bbb_ann_list_.resize(dim_space_);
	
	det_hash aaa_ann_map;
	det_hash aab_ann_map;
	det_hash abb_ann_map;
	det_hash bbb_ann_map;

	size_t naaa_ann = 0;
	size_t naab_ann = 0;
	size_t nabb_ann = 0;
	size_t nbbb_ann = 0;

	if( print_ ) outfile->Printf("\n  Generating three-particle maps.");

	for( size_t I = 0; I < dim_space_; ++I){
		STLBitsetDeterminant detI = det_space_[I];

		std::vector<int> aocc = detI.get_alfa_occ();	
		std::vector<int> bocc = detI.get_beta_occ();	

		int noalfa = aocc.size();
		int nobeta = bocc.size();

		std::vector<std::tuple<size_t, short, short, short>> aaa_ann( noalfa * (noalfa - 1) * (noalfa - 2) / 6 );
		std::vector<std::tuple<size_t, short, short, short>> aab_ann( noalfa * (noalfa - 1) * nobeta / 2  );
		std::vector<std::tuple<size_t, short, short, short>> abb_ann( noalfa * nobeta * (nobeta - 1) / 2  );
		std::vector<std::tuple<size_t, short, short, short>> bbb_ann( nobeta * (nobeta - 1) * (nobeta- 2) / 6  );

		// aaa
		for( int i = 0, ijk = 0; i < noalfa; ++i){
			for( int j = i + 1; j < noalfa;  ++j){
				for( int k = j + 1; k < noalfa; ++k, ++ijk){
					
					int ii = aocc[i];
					int jj = aocc[j];
					int kk = aocc[k];
			
					STLBitsetDeterminant detJ(detI);
					detJ.set_alfa_bit(ii, false);
					detJ.set_alfa_bit(jj, false);
					detJ.set_alfa_bit(kk, false);

					double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_alpha(jj) * detI.slater_sign_alpha(kk);

					det_hash_it hash_it = aaa_ann_map.find(detJ);
					size_t detJ_add;
					
					if( hash_it == aaa_ann_map.end()) {
						detJ_add = naaa_ann;
						aaa_ann_map[detJ] = naaa_ann;
						naaa_ann++;
					}else{
						detJ_add = hash_it->second;
					}
					aaa_ann[ijk] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii+1) : (-ii - 1), jj, kk);
				}
			}	
		}
		aaa_ann_list_[I] = aaa_ann;

		// aab
		for( int i = 0, ijk = 0; i < noalfa; ++i){
			for( int j = i + 1; j < noalfa; ++j){
				for( int k = 0; k < nobeta; ++k, ++ijk){
					
					int ii = aocc[i];
					int jj = aocc[j];
					int kk = bocc[k];
			
					STLBitsetDeterminant detJ(detI);
					detJ.set_alfa_bit(ii, false);
					detJ.set_alfa_bit(jj, false);
					detJ.set_beta_bit(kk, false);

					double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_alpha(jj) * detI.slater_sign_beta(kk);

					det_hash_it hash_it = aab_ann_map.find(detJ);
					size_t detJ_add;
					
					if( hash_it == aab_ann_map.end()) {
						detJ_add = naab_ann;
						aab_ann_map[detJ] = naab_ann;
						naab_ann++;
					}else{
						detJ_add = hash_it->second;
					}
					aab_ann[ijk] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii+1) : (-ii - 1), jj, kk);
				}
			}	
		}
		aab_ann_list_[I] = aab_ann;

		// abb
		for( int i = 0, ijk = 0; i < noalfa; ++i){
			for( int j = 0; j < nobeta; ++j){
				for( int k = j + 1;k < nobeta; ++k, ++ijk){
					
					int ii = aocc[i];
					int jj = bocc[j];
					int kk = bocc[k];
			
					STLBitsetDeterminant detJ(detI);
					detJ.set_alfa_bit(ii, false);
					detJ.set_beta_bit(jj, false);
					detJ.set_beta_bit(kk, false);

					double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_beta(jj) * detI.slater_sign_beta(kk);

					det_hash_it hash_it = abb_ann_map.find(detJ);
					size_t detJ_add;
					
					if( hash_it == abb_ann_map.end()) {
						detJ_add = nabb_ann;
						abb_ann_map[detJ] = nabb_ann;
						nabb_ann++;
					}else{
						detJ_add = hash_it->second;
					}
					abb_ann[ijk] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii+1) : (-ii - 1), jj, kk);
				}
			}	
		}
		abb_ann_list_[I] = abb_ann;

		// bbb
		for( int i = 0, ijk = 0; i < nobeta; ++i){
			for( int j = i + 1; j < nobeta; ++j){
				for( int k = j + 1; k < nobeta; ++k, ++ijk){
					
					int ii = bocc[i];
					int jj = bocc[j];
					int kk = bocc[k];
			
					STLBitsetDeterminant detJ(detI);
					detJ.set_beta_bit(ii, false);
					detJ.set_beta_bit(jj, false);
					detJ.set_beta_bit(kk, false);

					double sign = detI.slater_sign_beta(ii) * detI.slater_sign_beta(jj) * detI.slater_sign_beta(kk);

					det_hash_it hash_it = bbb_ann_map.find(detJ);
					size_t detJ_add;
					
					if( hash_it == bbb_ann_map.end()) {
						detJ_add = nbbb_ann;
						bbb_ann_map[detJ] = nbbb_ann;
						nbbb_ann++;
					}else{
						detJ_add = hash_it->second;
					}
					bbb_ann[ijk] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii+1) : (-ii - 1), jj, kk);
				}
			}	
		}
		bbb_ann_list_[I] = bbb_ann;
	} // End loop over determinants

	aaa_cre_list_.resize(aaa_ann_map.size());
	aab_cre_list_.resize(aab_ann_map.size());
	abb_cre_list_.resize(abb_ann_map.size());
	bbb_cre_list_.resize(bbb_ann_map.size());

	for( size_t I = 0; I < dim_space_; ++I){
		// aaa
		const std::vector<std::tuple<size_t,short,short,short>>& aaa_ann = aaa_ann_list_[I];
		for( const std::tuple<size_t,short,short,short>& Jsign : aaa_ann){
			size_t J = std::get<0>(Jsign);
			short i = std::get<1>(Jsign);
			short j = std::get<2>(Jsign);
			short k = std::get<3>(Jsign);
			aaa_cre_list_[J].push_back(std::make_tuple(I, i,j,k));
		}
		// aab
		const std::vector<std::tuple<size_t,short,short,short>>& aab_ann = aab_ann_list_[I];
		for( const std::tuple<size_t,short,short,short>& Jsign : aab_ann){
			size_t J = std::get<0>(Jsign);
			short i = std::get<1>(Jsign);
			short j = std::get<2>(Jsign);
			short k = std::get<3>(Jsign);
			aab_cre_list_[J].push_back(std::make_tuple(I, i,j,k));
		}
		// abb
		const std::vector<std::tuple<size_t,short,short,short>>& abb_ann = abb_ann_list_[I];
		for( const std::tuple<size_t,short,short,short>& Jsign : abb_ann){
			size_t J = std::get<0>(Jsign);
			short i = std::get<1>(Jsign);
			short j = std::get<2>(Jsign);
			short k = std::get<3>(Jsign);
			abb_cre_list_[J].push_back(std::make_tuple(I, i,j,k));
		}
		// bbb
		const std::vector<std::tuple<size_t,short,short,short>>& bbb_ann = bbb_ann_list_[I];
		for( const std::tuple<size_t,short,short,short>& Jsign : bbb_ann){
			size_t J = std::get<0>(Jsign);
			short i = std::get<1>(Jsign);
			short j = std::get<2>(Jsign);
			short k = std::get<3>(Jsign);
			bbb_cre_list_[J].push_back(std::make_tuple(I, i,j,k));
		}
	}
}


Reference CI_RDMS::reference( std::vector<double>& oprdm_a,
							  std::vector<double>& oprdm_b,  
							  std::vector<double>& tprdm_aa, 
							  std::vector<double>& tprdm_ab, 
							  std::vector<double>& tprdm_bb, 
							  std::vector<double>& tprdm_aaa,
							  std::vector<double>& tprdm_aab,
							  std::vector<double>& tprdm_abb,
							  std::vector<double>& tprdm_bbb)

{
	Reference ci_ref;	

	ambit::Tensor L1a = ambit::Tensor::build(ambit::CoreTensor,"L1a", {ncmo_,ncmo_});
	ambit::Tensor L1b = ambit::Tensor::build(ambit::CoreTensor,"L1b", {ncmo_,ncmo_});

	ambit::Tensor L2aa = ambit::Tensor::build(ambit::CoreTensor,"L2aa",{ncmo_,ncmo_,ncmo_,ncmo_});
	ambit::Tensor L2ab = ambit::Tensor::build(ambit::CoreTensor,"L2ab",{ncmo_,ncmo_,ncmo_,ncmo_});
	ambit::Tensor L2bb = ambit::Tensor::build(ambit::CoreTensor,"L2bb",{ncmo_,ncmo_,ncmo_,ncmo_});

    if( max_rdm_ >= 1 ){
	    // Form OPCM from 1-RDM
	    
	    if( na_ >= 1 ){
	    	L1a.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = oprdm_a[ i[0] * ncmo_ + i[1]];});
	    }
	    if( nb_ >= 1 ){
	    	L1b.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = oprdm_b[ i[0] * ncmo_ + i[1]];});
	    }
    }
    
	// Form TPCMs from 2-RDMs
    if( max_rdm_ >= 2 ){

	    ambit::Tensor g2aa = ambit::Tensor::build(ambit::CoreTensor,"g2aa",{ncmo_,ncmo_,ncmo_,ncmo_});
	    ambit::Tensor g2ab = ambit::Tensor::build(ambit::CoreTensor,"g2ab",{ncmo_,ncmo_,ncmo_,ncmo_});
	    ambit::Tensor g2bb = ambit::Tensor::build(ambit::CoreTensor,"g2bb",{ncmo_,ncmo_,ncmo_,ncmo_});

	    // First copy the 2-RDMs
	    if( na_ >= 2 ){
	    	L2aa.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = tprdm_aa[i[0] * ncmo3_ + i[1] * ncmo2_ + i[2] * ncmo_ + i[3]];
	    	});
	    }	
	    if( (na_ >= 1) and (nb_ >=1 ) ){
	    	L2ab.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = tprdm_ab[i[0] * ncmo3_ + i[1] * ncmo2_ + i[2] * ncmo_ + i[3]];
	    	});
	    }	
	    if( nb_ >= 2 ){
	    	L2bb.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = tprdm_bb[i[0] * ncmo3_ + i[1] * ncmo2_ + i[2] * ncmo_ + i[3]];
	    	});
	    }	
	    
	    g2aa.copy(L2aa);
	    g2ab.copy(L2ab);
	    g2bb.copy(L2bb);

	    // Now build the cumulants
	    L2aa("pqrs") -= L1a("pr") * L1a("qs");
	    L2aa("pqrs") += L1a("ps") * L1a("qr");

	    L2ab("pqrs") -= L1a("pr") * L1b("qs");

	    L2bb("pqrs") -= L1b("pr") * L1b("qs");
	    L2bb("pqrs") += L1b("ps") * L1b("qr");

    	double energy = get_energy( oprdm_a, oprdm_b, tprdm_aa, tprdm_bb, tprdm_ab );
    
    	ci_ref.set_Eref(energy);
    	ci_ref.set_L1a(L1a);
    	ci_ref.set_L1b(L1b);
    
    	ci_ref.set_L2aa(L2aa);
    	ci_ref.set_L2ab(L2ab);
    	ci_ref.set_L2bb(L2bb);
    
    	ci_ref.set_g2aa(g2aa);
    	ci_ref.set_g2ab(g2ab);
    	ci_ref.set_g2bb(g2bb);

    }

    if( max_rdm_ >= 3 ){
	    // Form the 3-RCMs
	    ambit::Tensor L3aaa = ambit::Tensor::build(ambit::CoreTensor,"L3aaa",{ncmo_,ncmo_,ncmo_,ncmo_,ncmo_,ncmo_,});
	    ambit::Tensor L3aab = ambit::Tensor::build(ambit::CoreTensor,"L3aab",{ncmo_,ncmo_,ncmo_,ncmo_,ncmo_,ncmo_,});
	    ambit::Tensor L3abb = ambit::Tensor::build(ambit::CoreTensor,"L3abb",{ncmo_,ncmo_,ncmo_,ncmo_,ncmo_,ncmo_,});
	    ambit::Tensor L3bbb = ambit::Tensor::build(ambit::CoreTensor,"L3bbb",{ncmo_,ncmo_,ncmo_,ncmo_,ncmo_,ncmo_,});

	    // First copy the RDMs
	    if( na_ >= 3){
	    	L3aaa.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = tprdm_aaa[i[0]*ncmo5_ + i[1]*ncmo4_ + i[2]*ncmo3_ + i[3]*ncmo2_ + i[4]*ncmo_ + i[5] ];
	    	});
	    }
	    if( (na_ >= 2) and (nb_ >= 1) ){
	    	L3aab.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = tprdm_aab[i[0]*ncmo5_ + i[1]*ncmo4_ + i[2]*ncmo3_ + i[3]*ncmo2_ + i[4]*ncmo_ + i[5] ];
	    	});
	    }
	    if( (na_ >= 1) and (nb_ >= 2) ){
	    	L3abb.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = tprdm_abb[i[0]*ncmo5_ + i[1]*ncmo4_ + i[2]*ncmo3_ + i[3]*ncmo2_ + i[4]*ncmo_ + i[5] ];
	    	});
	    }
	    if( nb_ >= 3){
	    	L3bbb.iterate([&](const std::vector<size_t>& i, double& value){
	    		value = tprdm_bbb[i[0]*ncmo5_ + i[1]*ncmo4_ + i[2]*ncmo3_ + i[3]*ncmo2_ + i[4]*ncmo_ + i[5] ];
	    	});
	    }

	    // Now form the cumulants
	    L3aaa("pqrstu") -= L1a("ps") * L2aa("qrtu");
        L3aaa("pqrstu") += L1a("pt") * L2aa("qrsu");
        L3aaa("pqrstu") += L1a("pu") * L2aa("qrts");

        L3aaa("pqrstu") -= L1a("qt") * L2aa("prsu");
        L3aaa("pqrstu") += L1a("qs") * L2aa("prtu");
        L3aaa("pqrstu") += L1a("qu") * L2aa("prst");

        L3aaa("pqrstu") -= L1a("ru") * L2aa("pqst");
        L3aaa("pqrstu") += L1a("rs") * L2aa("pqut");
        L3aaa("pqrstu") += L1a("rt") * L2aa("pqsu");

        L3aaa("pqrstu") -= L1a("ps") * L1a("qt") * L1a("ru");
        L3aaa("pqrstu") -= L1a("pt") * L1a("qu") * L1a("rs");
        L3aaa("pqrstu") -= L1a("pu") * L1a("qs") * L1a("rt");

        L3aaa("pqrstu") += L1a("ps") * L1a("qu") * L1a("rt");
        L3aaa("pqrstu") += L1a("pu") * L1a("qt") * L1a("rs");
        L3aaa("pqrstu") += L1a("pt") * L1a("qs") * L1a("ru");


        L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
        L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

        L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
        L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

        L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

        L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
        L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");


        L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

        L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
        L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

        L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
        L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

        L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
        L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");


        L3bbb("pqrstu") -= L1b("ps") * L2bb("qrtu");
        L3bbb("pqrstu") += L1b("pt") * L2bb("qrsu");
        L3bbb("pqrstu") += L1b("pu") * L2bb("qrts");

        L3bbb("pqrstu") -= L1b("qt") * L2bb("prsu");
        L3bbb("pqrstu") += L1b("qs") * L2bb("prtu");
        L3bbb("pqrstu") += L1b("qu") * L2bb("prst");

        L3bbb("pqrstu") -= L1b("ru") * L2bb("pqst");
        L3bbb("pqrstu") += L1b("rs") * L2bb("pqut");
        L3bbb("pqrstu") += L1b("rt") * L2bb("pqsu");

        L3bbb("pqrstu") -= L1b("ps") * L1b("qt") * L1b("ru");
        L3bbb("pqrstu") -= L1b("pt") * L1b("qu") * L1b("rs");
        L3bbb("pqrstu") -= L1b("pu") * L1b("qs") * L1b("rt");

        L3bbb("pqrstu") += L1b("ps") * L1b("qu") * L1b("rt");
        L3bbb("pqrstu") += L1b("pu") * L1b("qt") * L1b("rs");
        L3bbb("pqrstu") += L1b("pt") * L1b("qs") * L1b("ru");

	    ci_ref.set_L3aaa(L3aaa);
	    ci_ref.set_L3aab(L3aab);
	    ci_ref.set_L3abb(L3abb);
	    ci_ref.set_L3bbb(L3bbb);
    }
	// Update the reference object

	return ci_ref;

}


void CI_RDMS::rdm_test(std::vector<double>& oprdm_a,
                       std::vector<double>& oprdm_b,
                       std::vector<double>& tprdm_aa,
                       std::vector<double>& tprdm_bb,
                       std::vector<double>& tprdm_ab,
					   std::vector<double>& tprdm_aaa,
					   std::vector<double>& tprdm_aab,
					   std::vector<double>& tprdm_abb,
					   std::vector<double>& tprdm_bbb)
{

		double error_1rdm_a = 0.0;
		for (size_t p = 0; p < ncmo_; ++p){
			for(size_t q = 0; q < ncmo_; ++q){
				double rdm = 0.0;
				for(size_t i = 0; i < dim_space_; ++i){
					STLBitsetDeterminant I(det_space_[i]);
					double sign = 1.0;
					sign *= I.destroy_alfa_bit(q);	
					sign *= I.create_alfa_bit(p);
					for(size_t j = 0; j < dim_space_; ++j){
						if( I == det_space_[j] ){
							rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
						}
					}	
				}	
				if (std::fabs(rdm) > 1.0e-12 ){
					error_1rdm_a += std::fabs(rdm - oprdm_a[q*ncmo_ + p]);
//outfile->Printf("\n  D1(a)[%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)", p,q, rdm-oprdm_a[p*ncmo_+q],rdm,oprdm_a[p*ncmo_+q]);
				}
			}
		}
		outfile->Printf("\n    A 1-RDM Error :   %2.15f", error_1rdm_a);

		double error_1rdm_b = 0.0;
		for (size_t p = 0; p < ncmo_; ++p){
			for(size_t q = 0; q < ncmo_; ++q){
				double rdm = 0.0;
				for(size_t i = 0; i < dim_space_; ++i){
					STLBitsetDeterminant I(det_space_[i]);
					double sign = 1.0;
					sign *= I.destroy_beta_bit(q);	
					sign *= I.create_beta_bit(p);
					for(size_t j = 0; j < dim_space_; ++j){
						if( I == det_space_[j] ){
							rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
						}
					}	
				}	
				if (std::fabs(rdm) > 1.0e-12 ){
					error_1rdm_b += std::fabs(rdm - oprdm_b[p*ncmo_ + q]);
//outfile->Printf("\n  D1(b)[%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)", p,q, rdm-oprdm_b[p*ncmo_+q],rdm,oprdm_b[p*ncmo_+q]);
				}
			}
		}
		outfile->Printf("\n    B 1-RDM Error :   %2.15f", error_1rdm_b);


        double error_2rdm_aa = 0.0;
        for (size_t p = 0; p < ncmo_; ++p){
            for (size_t q = 0; q < ncmo_; ++q){
                for (size_t r = 0; r < ncmo_; ++r){
                    for (size_t s = 0; s < ncmo_; ++s){
                        double rdm = 0.0;
                        for (size_t i = 0; i < dim_space_; ++i){
                            STLBitsetDeterminant I(det_space_[i]);
                            double sign = 1.0;
                            sign *= I.destroy_alfa_bit(r);
                            sign *= I.destroy_alfa_bit(s);
                            sign *= I.create_alfa_bit(q);
                            sign *= I.create_alfa_bit(p);
                            for(size_t j = 0; j < dim_space_; ++j){
                            if (I == det_space_[j]){
                                rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
                            }
                        }}
                        if (std::fabs(rdm) > 1.0e-12){
                            error_2rdm_aa += std::fabs(rdm - tprdm_aa[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s]);
//outfile->Printf("\n  D2(aaaa)[%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)", p,q,r,s,rdm-tprdm_aa[p*ncmo3_+q*ncmo2_+r*ncmo_+s],rdm,tprdm_aa[p*ncmo3_+q*ncmo2_+r*ncmo_+s]);
                        }
                    }
                }
            }
        }
        outfile->Printf("\n    AAAA 2-RDM Error :   %2.15f",error_2rdm_aa);

        double error_2rdm_bb = 0.0;
        for (size_t p = 0; p < ncmo_; ++p){
            for (size_t q = 0; q < ncmo_; ++q){
                for (size_t r = 0; r < ncmo_; ++r){
                    for (size_t s = 0; s < ncmo_; ++s){
                        double rdm = 0.0;
                        for (size_t i = 0; i < dim_space_; ++i){
                            STLBitsetDeterminant I(det_space_[i]);
                            double sign = 1.0;
                            sign *= I.destroy_beta_bit(r);
                            sign *= I.destroy_beta_bit(s);
                            sign *= I.create_beta_bit(q);
                            sign *= I.create_beta_bit(p);
                            for(size_t j = 0; j < dim_space_; ++j){
                            if (I == det_space_[j]){
                                rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
                            }
                        }}
                        if (std::fabs(rdm) > 1.0e-12){
                            error_2rdm_bb += std::fabs(rdm - tprdm_bb[p*ncmo3_+q*ncmo2_+r*ncmo_+s]);
//outfile->Printf("\n  D2(bbbb)[%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)", p,q,r,s,rdm-tprdm_bb[p*ncmo3_+q*ncmo2_+r*ncmo_+s],rdm,tprdm_bb[p*ncmo3_+q*ncmo2_+r*ncmo_+s]);
                        }
                    }
                }
            }
        }
        outfile->Printf("\n    BBBB 2-RDM Error :   %2.15f",error_2rdm_bb);

        double error_2rdm_ab = 0.0;
        for (size_t p = 0; p < ncmo_; ++p){
            for (size_t q = 0; q < ncmo_; ++q){
                for (size_t r = 0; r < ncmo_; ++r){
                    for (size_t s = 0; s < ncmo_; ++s){
                        double rdm = 0.0;
                        for (size_t i = 0; i < dim_space_; ++i){
                            STLBitsetDeterminant I(det_space_[i]);
                            double sign = 1.0;
                            sign *= I.destroy_alfa_bit(r);
                            sign *= I.destroy_beta_bit(s);
                            sign *= I.create_beta_bit(q);
                            sign *= I.create_alfa_bit(p);
                            for(size_t j = 0; j < dim_space_; ++j){
                            if (I == det_space_[j]){
                                rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
                            }
                        }}
                        if (std::fabs(rdm) > 1.0e-12){
                            error_2rdm_ab += std::fabs(rdm - tprdm_ab[p*ncmo3_+q*ncmo2_+r*ncmo_+s]);
//outfile->Printf("\n  D2(abab)[%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)", p,q,r,s,rdm-tprdm_ab[p*ncmo3_+q*ncmo2_+r*ncmo_+s],rdm,tprdm_ab[p*ncmo3_+q*ncmo2_+r*ncmo_+s]);
                        }
                    }
                }
            }
        }
        outfile->Printf("\n    ABAB 2-RDM Error :   %2.15f",error_2rdm_ab);

// aaa aaa
//SharedMatrix three_rdm(new Matrix("three", dim_space_, dim_space_));
//three_rdm->zero();    
        double error_3rdm_aaa = 0.0;
                for (size_t p = 0; p < ncmo_; ++p){
        //for (size_t p = 0; p < 1; ++p){
            for (size_t q =  0; q < ncmo_; ++q){
                for (size_t r = 0; r < ncmo_; ++r){
                    for (size_t s = 0; s < ncmo_; ++s){
                        for (size_t t = 0; t < ncmo_; ++t){
                            for (size_t a = 0; a < ncmo_; ++a){
                                double rdm = 0.0;
                                for (size_t i = 0; i < dim_space_; ++i){
                                    STLBitsetDeterminant I(det_space_[i]);
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_alfa_bit(t);
                                    sign *= I.destroy_alfa_bit(a);
                                    sign *= I.create_alfa_bit(r);
                                    sign *= I.create_alfa_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    for( size_t j = 0; j < dim_space_; ++j){
                                        if (I == det_space_[j]){
                                            rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
                                           // three_rdm->set(i,j,three_rdm->get(i,j) + 1);
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12){
                                    double rdm_comp = tprdm_aaa[p*ncmo4_*ncmo_ + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + a];
                                   // outfile->Printf("\n  D3(aaaaaa)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)",
                                   //                                                     p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
                                    error_3rdm_aaa += std::fabs(rdm-rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        Process::environment.globals["AAAAAA 3-RDM ERROR"] = error_3rdm_aaa;
        outfile->Printf("\n    AAAAAA 3-RDM Error : %2.15f",error_3rdm_aaa);

// aab aab
        double error_3rdm_aab = 0.0;
                for (size_t p = 0; p < ncmo_; ++p){
        //for (size_t p = 0; p < 1; ++p){
            for (size_t q = 0; q < ncmo_; ++q){
                for (size_t r = 0; r < ncmo_; ++r){
                    for (size_t s = 0; s < ncmo_; ++s){
                        for (size_t t = 0; t < ncmo_; ++t){
                            for (size_t a = 0; a < ncmo_; ++a){
                                double rdm = 0.0;
                                for (size_t i = 0; i < dim_space_; ++i){
                                    STLBitsetDeterminant I(det_space_[i]);
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_alfa_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_alfa_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    for( size_t j = 0; j < dim_space_; ++j){
                                        if (I == det_space_[j]){
                                            rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
                                           // three_rdm->set(i,j,three_rdm->get(i,j) + 1);
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12){
                                    double rdm_comp = tprdm_aab[p*ncmo4_*ncmo_ + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + a];
                                    //outfile->Printf("\n  D3(aabaab)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)",
                                    //                p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
                                    error_3rdm_aab += std::fabs(rdm-rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        Process::environment.globals["AABAAB 3-RDM ERROR"] = error_3rdm_aab;
        outfile->Printf("\n    AABAAB 3-RDM Error : %+e",error_3rdm_aab);

// abb abb
        double error_3rdm_abb = 0.0;
                for (size_t p = 0; p < ncmo_; ++p){
        //for (size_t p = 0; p < 1; ++p){
            for (size_t q = 0; q < ncmo_; ++q){
                for (size_t r = 0; r < ncmo_; ++r){
                    for (size_t s = 0; s < ncmo_; ++s){
                        for (size_t t = 0; t < ncmo_; ++t){
                            for (size_t a = 0; a < ncmo_; ++a){
                                double rdm = 0.0;
                                for (size_t i = 0; i < dim_space_; ++i){
                                    STLBitsetDeterminant I(det_space_[i]);
                                    double sign = 1.0;
                                    sign *= I.destroy_alfa_bit(s);
                                    sign *= I.destroy_beta_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_beta_bit(q);
                                    sign *= I.create_alfa_bit(p);
                                    for( size_t j = 0; j < dim_space_; ++j){
                                        if (I == det_space_[j]){
                                            rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
                                            //three_rdm->set(i,j,three_rdm->get(i,j) + 1);
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12){
                                    double rdm_comp = tprdm_abb[p*ncmo4_*ncmo_ + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + a];
//                                   outfile->Printf("\n  D3(abbabb)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)",
//                                                   p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
                                    error_3rdm_abb += std::fabs(rdm-rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        Process::environment.globals["ABBABB 3-RDM ERROR"] = error_3rdm_abb;
        outfile->Printf("\n    ABBABB 3-RDM Error : %+e",error_3rdm_abb);

// bbb bbb
        double error_3rdm_bbb = 0.0;
                for (size_t p = 0; p < ncmo_; ++p){
        //for (size_t p = 0; p < 1; ++p){
            for (size_t q = 0; q < ncmo_; ++q){
                for (size_t r = 0; r < ncmo_; ++r){
                    for (size_t s = 0; s < ncmo_; ++s){
                        for (size_t t = 0; t < ncmo_; ++t){
                            for (size_t a = 0; a < ncmo_; ++a){
                                double rdm = 0.0;
                                for (size_t i = 0; i < dim_space_; ++i){
                                    STLBitsetDeterminant I(det_space_[i]);
                                    double sign = 1.0;
                                    sign *= I.destroy_beta_bit(s);
                                    sign *= I.destroy_beta_bit(t);
                                    sign *= I.destroy_beta_bit(a);
                                    sign *= I.create_beta_bit(r);
                                    sign *= I.create_beta_bit(q);
                                    sign *= I.create_beta_bit(p);
                                    for( size_t j = 0; j < dim_space_; ++j){
                                        if (I == det_space_[j]){
                                            rdm += sign * evecs_->get(i,root1_) * evecs_->get(j,root2_);
                                           // three_rdm->set(i,j,three_rdm->get(i,j) + 1);
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12){
                                    double rdm_comp = tprdm_bbb[p*ncmo4_*ncmo_ + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + a];
                                 //                                       outfile->Printf("\n  D3(bbbbbb)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)",
                                 //                                                       p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
                                    error_3rdm_bbb += std::fabs(rdm-rdm_comp);
                                }
                            }
                        }
                    }
                }
            }
        }
        Process::environment.globals["BBBBBB 3-RDM ERROR"] = error_3rdm_bbb;
        outfile->Printf("\n    BBBBBB 3-RDM Error : %2.15f",error_3rdm_bbb);
//three_rdm->print_to_mathematica();
}


}} // End Namespaces
