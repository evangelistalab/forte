#ifndef _ci_rdms_h_
#define _ci_rdms_h_


#include <cmath>
#include <numeric>

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>
#include <boost/functional/hash.hpp>

#include "helpers.h"
#include "stl_bitset_determinant.h"
#include "reference.h"
#include "string_lists.h"

namespace psi{ namespace forte{

class CI_RDMS
{
public:

	using det_hash    = std::unordered_map<STLBitsetDeterminant, size_t, STLBitsetDeterminant::Hash>;	
	using det_hash_it = det_hash::iterator; 

	// Class constructor and destructor
	CI_RDMS(Options &options, boost::shared_ptr<Wavefunction> wfn, std::shared_ptr<FCIIntegrals> fci_ints, std::shared_ptr<MOSpaceInfo> mo_space_info, std::vector<STLBitsetDeterminant> det_space, SharedMatrix evecs);

	~CI_RDMS();

	// Return a reference object
	Reference reference(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b, std::vector<double>& tprdm_aa, std::vector<double>& tprdm_bb, std::vector<double>& tprdm_ab,
				  std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab, std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb); 

	// Compute rdms
	void compute_1rdm( std::vector<double>& oprdm_a, std::vector<double>& oprdm_b, int root);
	void compute_2rdm( std::vector<double>& tprdm_aa,std::vector<double>& tprdm_ab,std::vector<double>& tprdm_bb, int root);
	void compute_3rdm( std::vector<double>& tprdm_aaa,std::vector<double>& tprdm_aab,std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb, int root);
	
	double get_energy(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b, std::vector<double>& tprdm_aa, std::vector<double>& tprdm_bb, std::vector<double>& tprdm_ab); 
	
	void rdm_test(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b, std::vector<double>& tprdm_aa, std::vector<double>& tprdm_bb, std::vector<double>& tprdm_ab,
				  std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab, std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb); 
private:
   /* Class Variables*/  
	
	// The options object
	Options& options_;
	// The Wavefunction Object
	boost::shared_ptr<Wavefunction> wfn_;
	// The FCI integrals
	std::shared_ptr<FCIIntegrals> fci_ints_;
	// The MOSpaceInfo object
	std::shared_ptr<MOSpaceInfo> mo_space_info_;

	// The Determinant Space
	std::vector<STLBitsetDeterminant> det_space_; 

	// The CI coefficients
	SharedMatrix evecs_;

	// The wavefunction symmetry
	int symmetry_;
	// The number of irreps
	int nirrep_;

	// The number of alpha electrons
	int na_;

	// The number of beta electrons
	int nb_;
	
	// The number of correlated mos
	size_t ncmo_;
	size_t ncmo2_;
	size_t ncmo3_;
	size_t ncmo4_;
 
	// The correlated mos per irrep
	Dimension active_dim_;	

    std::vector<size_t> active_mo_;	

	// The dimension of the vector space
	size_t dim_space_;
	
	// Has the one-map been constructed?
	bool one_map_done_;

	// The list of a_p |N>
	std::vector<std::vector<std::pair<size_t,short>>> a_ann_list_;
	std::vector<std::vector<std::pair<size_t,short>>> b_ann_list_;

	// The list of a^(+)_q |N-1>
	std::vector<std::vector<std::pair<size_t,short>>> a_cre_list_;
	std::vector<std::vector<std::pair<size_t,short>>> b_cre_list_;

	// The list of a_q a_p|N>
	std::vector<std::vector<std::tuple<size_t,short,short>>> aa_ann_list_;
	std::vector<std::vector<std::tuple<size_t,short,short>>> ab_ann_list_;
	std::vector<std::vector<std::tuple<size_t,short,short>>> bb_ann_list_;
	
	// The list of a_q^(+) a_p^(+)|N-1>
	std::vector<std::vector<std::tuple<size_t,short,short>>> aa_cre_list_;
	std::vector<std::vector<std::tuple<size_t,short,short>>> ab_cre_list_;
	std::vector<std::vector<std::tuple<size_t,short,short>>> bb_cre_list_;

	// The list of a_r a_q a_p |N>
	std::vector<std::vector<std::tuple<size_t,short,short,short>>> aaa_ann_list_;
	std::vector<std::vector<std::tuple<size_t,short,short,short>>> aab_ann_list_;
	std::vector<std::vector<std::tuple<size_t,short,short,short>>> abb_ann_list_;
	std::vector<std::vector<std::tuple<size_t,short,short,short>>> bbb_ann_list_;
	
	// The list of a^(+)_r a^(+)_q a^(+)_p |N-1>
	std::vector<std::vector<std::tuple<size_t,short,short,short>>> aaa_cre_list_;
	std::vector<std::vector<std::tuple<size_t,short,short,short>>> aab_cre_list_;
	std::vector<std::vector<std::tuple<size_t,short,short,short>>> abb_cre_list_;
	std::vector<std::vector<std::tuple<size_t,short,short,short>>> bbb_cre_list_;



	/* Class functions*/ 

	// Startup function, mostly just gathering all variables
	void startup();

	// Generate one-particle map
	void get_one_map();	
	
	// Generate two-particle map
	void get_two_map();

	// Generate three-particle map
	void get_three_map();
};

}} // End namepaces

#endif // _ci_rdms_h_
