#include <cmath>

#include <liboptions/liboptions.h>
#include <libmints/molecule.h>
#include <libmints/wavefunction.h>

#include "reference.h"
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
	ncmo5_ = ncmo3_ * ncmo2_;
	

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
	Timer one;
	get_one_map();
	outfile->Printf("\n  Time spent forming 1-map:   %1.6f", one.get());

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
	outfile->Printf("\n  Time spent building 1-rdm:   %1.6f", build.get());
}

void CI_RDMS::compute_2rdm( std::vector<double>& tprdm_aa,std::vector<double>& tprdm_ab,std::vector<double>& tprdm_bb, int root )
{
	tprdm_aa.resize(ncmo4_, 0.0);	
	tprdm_ab.resize(ncmo4_, 0.0);	
	tprdm_bb.resize(ncmo4_, 0.0);	

    Timer two;
	get_two_map();
	outfile->Printf("\n  Time spent forming 2-map:   %1.6f", two.get());

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
	outfile->Printf("\n  Time spent building 2-rdm:   %1.6f", build.get());
}


void CI_RDMS::compute_3rdm( std::vector<double>& tprdm_aaa,
							std::vector<double>& tprdm_aab,
							std::vector<double>& tprdm_abb, 
							std::vector<double>& tprdm_bbb, int root)
{
	size_t ncmo5 = ncmo4_ * ncmo_;
	size_t ncmo6 = ncmo3_ * ncmo3_;

	tprdm_aaa.resize(ncmo6, 0.0);
	tprdm_aab.resize(ncmo6, 0.0);
	tprdm_abb.resize(ncmo6, 0.0);
	tprdm_bbb.resize(ncmo6, 0.0);

	Timer three;
	get_three_map();
	outfile->Printf("\n  Time spent forming 3-map:   %1.6f", three.get());
	

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

				double rdm_element = evecs_->get(I,root) * evecs_->get(J,root) * sign_pqr * sign_stu;

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

				double rdm_element = evecs_->get(I,root) * evecs_->get(J,root) * sign_pqr * sign_stu;

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

				double rdm_element = evecs_->get(I,root) * evecs_->get(J,root) * sign_pqr * sign_stu;

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

				double rdm_element = evecs_->get(I,root) * evecs_->get(J,root) * sign_pqr * sign_stu;

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

	outfile->Printf("\n  Time spent building 3-rdm:   %1.6f", build.get());
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

	outfile->Printf("\n\n  Generating one-particle maps.");

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

	outfile->Printf("\n  Generating three-particle maps.");

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

	// Form OPCM from 1-RDM
	ambit::Tensor L1a = ambit::Tensor::build(ambit::CoreTensor,"L1a", {ncmo_,ncmo_});
	ambit::Tensor L1b = ambit::Tensor::build(ambit::CoreTensor,"L1b", {ncmo_,ncmo_});
	
	if( na_ >= 1 ){
		L1a.iterate([&](const std::vector<size_t>& i, double& value){
			value = oprdm_a[ i[0] * ncmo_ + i[1]];});
	}
	if( nb_ >= 1 ){
		L1b.iterate([&](const std::vector<size_t>& i, double& value){
			value = oprdm_b[ i[0] * ncmo_ + i[1]];});
	}

	// Form TPCMs from 2-RDMs

	ambit::Tensor L2aa = ambit::Tensor::build(ambit::CoreTensor,"L2aa",{ncmo_,ncmo_,ncmo_,ncmo_});
	ambit::Tensor L2ab = ambit::Tensor::build(ambit::CoreTensor,"L2ab",{ncmo_,ncmo_,ncmo_,ncmo_});
	ambit::Tensor L2bb = ambit::Tensor::build(ambit::CoreTensor,"L2bb",{ncmo_,ncmo_,ncmo_,ncmo_});

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

	// Update the reference object

	double energy = get_energy( oprdm_a, oprdm_b, tprdm_aa, tprdm_bb, tprdm_ab );

	Reference ci_ref;	

	ci_ref.set_Eref(energy);
	ci_ref.set_L1a(L1a);
	ci_ref.set_L1b(L1b);

	ci_ref.set_L2aa(L2aa);
	ci_ref.set_L2ab(L2ab);
	ci_ref.set_L2bb(L2bb);

	ci_ref.set_L3aaa(L3aaa);
	ci_ref.set_L3aab(L3aab);
	ci_ref.set_L3abb(L3abb);
	ci_ref.set_L3bbb(L3bbb);

	ci_ref.set_g2aa(g2aa);
	ci_ref.set_g2ab(g2ab);
	ci_ref.set_g2bb(g2bb);

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
							rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
						}
					}	
				}	
				if (std::fabs(rdm) > 1.0e-12 ){
					error_1rdm_a += std::fabs(rdm - oprdm_a[q*ncmo_ + p]);
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
							rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
						}
					}	
				}	
				if (std::fabs(rdm) > 1.0e-12 ){
					error_1rdm_b += std::fabs(rdm - oprdm_b[p*ncmo_ + q]);
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

// aaa aaa
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
                                            rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12){
                                    double rdm_comp = tprdm_aaa[p*ncmo4_*ncmo_ + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + a];
                                   //                                     outfile->Printf("\n  D3(aaaaaa)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)",
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
        outfile->Printf("\n    AAAAAA 3-RDM Error : %+e",error_3rdm_aaa);

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
                                            rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12){
                                    double rdm_comp = tprdm_aab[p*ncmo4_*ncmo_ + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + a];
                                   //                                     outfile->Printf("\n  D3(aabaab)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)",
                                   //                                                     p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
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
                                            rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
                                        }
                                    }
                                }
                                if (std::fabs(rdm) > 1.0e-12){
                                    double rdm_comp = tprdm_abb[p*ncmo4_*ncmo_ + q*ncmo4_ + r*ncmo3_ + s*ncmo2_ + t*ncmo_ + a];
                                   //                                     outfile->Printf("\n  D3(abbabb)[%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.12lf (%18.12lf,%18.12lf)",
                                   //                                                     p,q,r,s,t,a,rdm-rdm_comp,rdm,rdm_comp);
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
                                            rdm += sign * evecs_->get(i,0) * evecs_->get(j,0);
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
        outfile->Printf("\n    BBBBBB 3-RDM Error : %+e",error_3rdm_bbb);

}


}} // End Namespaces
