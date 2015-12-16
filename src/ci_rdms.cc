#include <cmath>

#include <liboptions/liboptions.h>
#include <libmints/molecule.h>
#include <libmints/wavefunction.h>

#include "helpers.h"
#include "stl_bitset_determinant.h"
#include "ci_rdms.h"

namespace psi{ namespace forte {

// A class that takes the determinants and expansion
// coefficients and computes reduced density matrices.


CI_RDMS::CI_RDMS( Options &options, 
				  boost::shared_ptr<Wavefunction> wfn, 
				  std::shared_ptr<FCIIntegrals> fci_ints,
			      std::shared_ptr<MOSpaceInfo> mo_space_info, 
				  std::vector<STLBitsetDeterminant> det_space,
				  SharedMatrix evecs)
				: options_(options),
				  wfn_(wfn),
				  fci_ints_(fci_ints),
				  mo_space_info_(mo_space_info),
				  det_space_(det_space),
				  evecs_(evecs)
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
	ncmo_ = mo_space_info_->size("ACTIVE");
	ncmo2_ = ncmo_ * ncmo_;
	ncmo3_ = ncmo2_ * ncmo_;
	ncmo4_ = ncmo3_ * ncmo_;
	

	// The number of irreps
	nirrep_ = wfn_->nirrep(); 

	size_t nfdocc = mo_space_info_->size("FROZEN_DOCC");

	// The restricted MOs	
	std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"); 	

	// The number of electrons
    int nel = 0;	
	int natom = Process::environment.molecule()->natom();
    for(int i=0; i < natom;i++){
        nel += static_cast<int>(Process::environment.molecule()->Z(i));
    }
	nel -= options_.get_int("CHARGE");

	// The wavefunction multiplicity
	int multiplicity = options_.get_int("MULTIPLICITY");
	int ms = multiplicity - 1;

	// The number of active electrons
	int nactel = nel - 2*nfdocc - 2*rdocc.size();

	// The number of alpha electrons 
	na_ = (nactel + ms) / 2;
	
	// The number of beta electrons
	nb_ = nactel - na_;	
	

	// The correlated MOs per irrep (active)
	active_dim_ = mo_space_info_->get_dimension("ACTIVE");
	

	// The active MOs
	active_mo_ = mo_space_info_->get_corr_abs_mo("ACTIVE");

	// Dimension of the determinant space
	dim_space_ = det_space_.size();

	symmetry_ = 0;
	if (options_["ROOT_SYM"].has_changed()){
		symmetry_ = options_.get_int("ROOT_SYM");
	}
	
	one_map_done_ = false;
	
	outfile->Printf("\n  Computing RDMS");
	outfile->Printf("\n  Number of active alpha electrons: %zu", na_);
	outfile->Printf("\n  Number of active beta electrons: %zu", nb_);
	outfile->Printf("\n  Number of correlated orbitals: %zu", ncmo_);
	

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
    outfile->Printf("\n  Total Energy: %25.15f\n",total_energy);
    outfile->Printf("\n  Scalar Energy = %8.8f", scalar_energy);
    outfile->Printf("\n  energy_1rdm = %8.8f", energy_1rdm);
    outfile->Printf("\n  energy_2rdm = %8.8f", energy_2rdm);
    outfile->Printf("\n  nuclear_repulsion_energy = %8.8f", nuc_rep);

	return total_energy;
}


void CI_RDMS::compute_1rdm( std::vector<double>& oprdm_a, std::vector<double>& oprdm_b, int root )
{
	get_one_map();
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
    	        oprdm_a[q*ncmo_ + p] += evecs_->get(J, root) * evecs_->get(I, root) * sign_p * sign_q;
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
    	        oprdm_b[q*ncmo_ + p] += evecs_->get(J, root) * evecs_->get(I, root) * sign_p * sign_q;
    	    }
    	}
	}
}

void CI_RDMS::compute_2rdm( std::vector<double>& tprdm_aa,std::vector<double>& tprdm_ab,std::vector<double>& tprdm_bb, int root )
{
	tprdm_aa.resize(ncmo4_, 0.0);	
	tprdm_ab.resize(ncmo4_, 0.0);	
	tprdm_bb.resize(ncmo4_, 0.0);	
	get_two_map();

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
				double rdm_element = evecs_->get(I,root) * evecs_->get(J, root) * sign_pq * sign_rs;	

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
				double rdm_element = evecs_->get(I,root) * evecs_->get(J, root) * sign_pq * sign_rs;	

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
				double rdm_element = evecs_->get(I,root) * evecs_->get(J, root) * sign_pq * sign_rs;	

				tprdm_ab[p*ncmo3_ + q*ncmo2_ + r*ncmo_ + s] += rdm_element;

			}
		}

	}	
}


void CI_RDMS::compute_3rdm_aaa()
{
}


void CI_RDMS::compute_3rdm_aab()
{
}


void CI_RDMS::compute_3rdm_abb()
{
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

	outfile->Printf("\n Generating one-particle maps.\n");

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

	outfile->Printf("\n  Generating two-particle maps.");

	for( size_t I = 0; I < dim_space_; ++I){
		STLBitsetDeterminant detI = det_space_[I];
		
		std::vector<int> aocc = detI.get_alfa_occ();
		std::vector<int> bocc = detI.get_beta_occ();

		size_t noalfa = aocc.size();
		size_t nobeta = bocc.size();

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


Reference CI_RDMS::reference()
{
}


void CI_RDMS::rdm_test(std::vector<double>& oprdm_a,
                       std::vector<double>& oprdm_b,
                       std::vector<double>& tprdm_aa,
                       std::vector<double>& tprdm_bb,
                       std::vector<double>& tprdm_ab)
{

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
                                rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
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
                                rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
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
                                rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
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
}


}} // End Namespaces
