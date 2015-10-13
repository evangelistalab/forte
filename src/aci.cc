#include <cmath>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <numeric>

#include <boost/timer.hpp>
#include <boost/format.hpp>

#include <libciomr/libciomr.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include <libqt/qt.h>
#include <libmints/molecule.h>

#include "aci.h"
#include "sparse_ci_solver.h"
#include "dynamic_bitset_determinant.h"
#include "stl_bitset_determinant.h"
#include "fci_vector.h"

using namespace std;
using namespace psi;


namespace psi{ namespace forte{

/**
 * Template used to quickly access
 * vectors that store three related quantities
 **/

template <typename a, typename b, typename c>
using oVector = std::vector<std::pair<a, std::pair< b,c >> >;

/**
 * Template for vector of pairs
 **/

template < typename a, typename b>
using pVector = std::vector<std::pair< a,b> >;

inline double clamp(double x, double a, double b)

{
    return x < a ? a : (x > b ? b : x);
}

bool pairComp(const std::pair<double, STLBitsetDeterminant> E1, const std::pair<double, STLBitsetDeterminant> E2){
	return E1.first < E2.first;
}

// Hash for BSD 
//std::size_t BSD_hash_value(const STLBitsetDeterminant& input)
//{
//    return (input.alfa_bits_.to_ulong() % 100000 + input.beta_bits_.to_ulong() % 100000);
//}

/**
 * This is a smooth step function that is
 * 0.0 for x <= edge0
 * 1.0 for x >= edge1
 */
inline double smootherstep(double edge0, double edge1, double x)
{
    if (edge1 == edge0){
        return x <= edge0 ? 0.0 : 1.0;
    }
    // Scale, and clamp x to 0..1 range
    x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    // Evaluate polynomial
    return x * x * x *( x *( x * 6. - 15.) + 10.);
}

AdaptiveCI::AdaptiveCI(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options,_default_psio_lib_),
		options_(options), 
		ints_(ints), 
		mo_space_info_(mo_space_info)
{
    // Copy the wavefunction information
    copy(wfn);

    startup();
    print_info();
}

AdaptiveCI::~AdaptiveCI()
{
}

void AdaptiveCI::startup()
{
	fci_ints_ = std::make_shared<FCIIntegrals>(ints_, mo_space_info_);
    STLBitsetDeterminant::set_ints(fci_ints_);

    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();
	//Get wfn info
    wavefunction_symmetry_ = 0;
    if(options_["ROOT_SYM"].has_changed()){
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }
	wavefunction_multiplicity_ = 1;
	if(options_["MULTIPLICITY"].has_changed()){
		wavefunction_multiplicity_ = options_.get_int("MULTIPLICITY");
	}
	nact_ = mo_space_info_->size("ACTIVE");
	nactpi_ = mo_space_info_->get_dimension("ACTIVE");

	// Include frozen_docc and restricted_docc
	frzcpi_ = mo_space_info_->get_dimension("INACTIVE_DOCC");
	frzvpi_ = mo_space_info_->get_dimension("INACTIVE_UOCC");
	nfrzc_ = mo_space_info_->size("INACTIVE_DOCC");

	// "Correlated" includes restricted_docc
    ncmo_ = mo_space_info_->size("CORRELATED");
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
	rdoccpi_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");

	// Number of correlated electrons
	nactel_ = 0;
	nalpha_ = 0;
	nbeta_ = 0;
	int nel = 0;
	for(int h = 0; h < nirrep_; ++h){
        nel += 2 * doccpi_[h] + soccpi_[h];
	}
		
	int ms = wavefunction_multiplicity_ - 1;
	nactel_ = nel - 2 * nfrzc_;
	nalpha_ = (nactel_ + ms) / 2;
	nbeta_  = nactel_ - nalpha_; 

	outfile->Printf("\n  Total electrons: %zu", nel);
	outfile->Printf("\n  Number of active electrons: %d",nactel_);
	outfile->Printf("\n  nalpha: %d", nalpha_);
	outfile->Printf("\n  nbeta:  %d \n", nbeta_);


	mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");

	rdocc_ = mo_space_info_->size("RESTRICTED_DOCC");
	rvir_  = mo_space_info_->size("RESTRICTED_UOCC");	

	outfile->Printf("\n  There are %d frozen orbitals.", nfrzc_);
	outfile->Printf("\n  There are %zu active orbitals.", nact_);

    // Build the reference determinant and compute its energy
    reference_determinant_ = STLBitsetDeterminant(get_occupation());
    outfile->Printf("\n  The reference determinant is:\n");
    reference_determinant_.print();
	outfile->Printf("\n  The reference energy is %1.8f", reference_determinant_.energy() + nuclear_repulsion_energy_ + fci_ints_->scalar_energy());


    // Read options
    nroot_ = options_.get_int("NROOT");
    tau_p_ = options_.get_double("TAUP");
    tau_q_ = options_.get_double("TAUQ");

    add_aimed_degenerate_ = options_.get_bool("ACI_ADD_AIMED_DEGENERATE");
    project_out_spin_contaminants_ = options_.get_bool("PROJECT_OUT_SPIN_CONTAMINANTS");
    spin_complete_ = options_.get_bool("ENFORCE_SPIN_COMPLETE");

    do_smooth_ = options_.get_bool("SMOOTH");
    smooth_threshold_ = options_.get_double("SMOOTH_THRESHOLD");

	spin_tol_ = options_.get_double("SPIN_TOL");
	//set the initial S^@ guess as input multiplicity
	int S   = (wavefunction_multiplicity_ - 1.0)/2.0; 
	int S2 = wavefunction_multiplicity_ - 1.0;
	for(int n = 0; n < nroot_; ++n){
		root_spin_vec_.push_back(make_pair( S, S2 ));
	}
	

	//get options for algorithm
	perturb_select_ = options_.get_bool("PERTURB_SELECT");
    pq_function_ = options_.get_str("PQ_FUNCTION");
    q_rel_ = options_.get_bool("Q_REL");
    q_reference_ = options_.get_str("Q_REFERENCE");
    ex_alg_ = options_.get_str("EXCITED_ALGORITHM");
    post_root_ = max( nroot_, options_.get_int("POST_ROOT") );
    post_diagonalize_ = options_.get_bool("POST_DIAGONALIZE");
    form_1_RDM_ = options_.get_bool("1_RDM");
    do_guess_ = options_.get_bool("LAMBDA_GUESS");
    det_save_ = options_.get_bool("SAVE_DET_FILE");


    if (options_.get_str("DIAG_ALGORITHM") == "FULL"){
        diag_method_ = Full;
    } else if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
        diag_method_ = DavidsonLiuSparse;
    } else if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSONLIST"){
        diag_method_ = DavidsonLiuList;
    } else if (options_.get_str("DIAG_ALGORITHM") == "SOLVER"){
        diag_method_ = DLSolver;
    }

    aimed_selection_ = false;
    energy_selection_ = false;
    if (options_.get_str("SELECT_TYPE") == "AIMED_AMP"){
        aimed_selection_ = true;
        energy_selection_ = false;
    }else if (options_.get_str("SELECT_TYPE") == "AIMED_ENERGY"){
        aimed_selection_ = true;
        energy_selection_ = true;
    }else if(options_.get_str("SELECT_TYPE") == "ENERGY"){
        aimed_selection_ = false;
        energy_selection_ = true;
    }else if(options_.get_str("SELECT_TYPE") == "AMP"){
        aimed_selection_ = false;
        energy_selection_ = false;
    }
}

void AdaptiveCI::print_info()
{
    print_method_banner({"Adaptive Configuration Interaction","written by Francesco A. Evangelista"});

    // Print a summary
    std::vector<std::pair<std::string,int>> calculation_info{
        {"Multiplicity",wavefunction_multiplicity_},
        {"Symmetry",wavefunction_symmetry_},
        {"Number of roots",nroot_},
        {"Root used for properties",options_.get_int("ROOT")}};

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"P-threshold",tau_p_},
        {"Q-threshold",tau_q_},
        {"Convergence threshold",options_.get_double("E_CONVERGENCE")}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"Determinant selection criterion",energy_selection_ ? "Second-order Energy" : "First-order Coefficients"},
        {"Selection criterion",aimed_selection_ ? "Aimed selection" : "Threshold"},
        {"PQ Function", options_.get_str("PQ_FUNCTION")},
        {"Q Type", q_rel_ ? "Relative Energy" : "Absolute Energy"},
        {"PT2 Parameters", options_.get_bool("PERTURB_SELECT") ? "True" : "False"},
        {"Project out spin contaminants",project_out_spin_contaminants_ ? "True" : "False"},
        {"Enforce spin completeness of basis", spin_complete_ ? "True" : "False"},
        {"Enforce complete aimed selection", add_aimed_degenerate_ ? "True" : "False"}};

    // Print some information
    outfile->Printf("\n  ==> Calculation Information <==\n");
    outfile->Printf("\n  %s",string(65,'-').c_str());
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-40s %-5d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-40s %8.2e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-40s %s",str_dim.first.c_str(),str_dim.second.c_str());
    }
    outfile->Printf("\n  %s",string(65,'-').c_str());
    outfile->Flush();
}

std::vector<int> AdaptiveCI::get_occupation()
{
	
	std::vector<int> occupation(2 * nact_,0);

	//Get reference type
	std::string ref_type = options_.get_str("REFERENCE");
	outfile->Printf("\n  Using %s reference.\n", ref_type.c_str());

	//nyms denotes the number of electrons needed to assign symmetry and multiplicity
	int nsym = wavefunction_multiplicity_ - 1;
	int orb_sym = wavefunction_symmetry_;

	if(wavefunction_multiplicity_ == 1){
		nsym = 2;
	}

	// Grab an ordered list of orbital energies, sym labels, and idxs
	oVector<double,int,int> labeled_orb_en;
	oVector<double,int,int> labeled_orb_en_alfa;
	oVector<double,int,int> labeled_orb_en_beta;
	
	// For a restricted reference
	if(ref_type == "RHF" or ref_type == "RKS" or ref_type == "ROHF"){
		labeled_orb_en = sym_labeled_orbitals("RHF");

		// Build initial reference determinant from restricted reference
		for(int i = 0 ; i < nalpha_; ++i){
			occupation[labeled_orb_en[i].second.second] = 1;
		}
		for(int i = 0; i < nbeta_; ++i){
			occupation[nact_ + labeled_orb_en[i].second.second] = 1;
		}

		// Loop over as many outer-shell electrons as needed to get correct sym
		for(int k = 1; k <= nsym;){
		
			bool add = false;
			// Remove electron from highest energy docc
			occupation[labeled_orb_en[nalpha_ - k].second.second] = 0;
			outfile->Printf("\n  Electron removed from %d, out of %d", labeled_orb_en[nalpha_ - k].second.second, nactel_);
		
			// Determine proper symmetry for new occupation
			orb_sym = wavefunction_symmetry_; 
			
			if(wavefunction_multiplicity_ == 1){
				orb_sym = labeled_orb_en[nalpha_ - 1].second.first ^ orb_sym;
			}else{
				for(int i = 1; i <= nsym; ++i){
					orb_sym = labeled_orb_en[nalpha_ - i].second.first ^ orb_sym;
				}
				orb_sym  = labeled_orb_en[nalpha_ - k].second.first ^ orb_sym;
			}

			outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

			// Add electron to lowest-energy orbital of proper symmetry
			// Loop from current occupation to max MO until correct orbital is reached
			for(int i = nalpha_ - k, maxi = nactel_; i < maxi; ++i){
				if(orb_sym == labeled_orb_en[i].second.first and occupation[labeled_orb_en[i].second.second] != 1){
					occupation[labeled_orb_en[i].second.second] = 1;
					outfile->Printf("\n  Added electron to %d", labeled_orb_en[i].second.second);
					add = true;
					break;
				}else{
					continue;
				}
			}
			//If a new occupation could not be created, put electron back and remove a different one
			if(!add){
				occupation[labeled_orb_en[nalpha_ - k].second.second] = 1;
				outfile->Printf("\n  No orbital of symmetry %d available! Putting electron back...", orb_sym);
				++k;
			}else{
				break;
			}

		} // End loop over k

	}else{
		labeled_orb_en_alfa = sym_labeled_orbitals("ALFA");
		labeled_orb_en_beta = sym_labeled_orbitals("BETA");

		// For an unrestricted reference
		// Make the reference
		// For singlets, this will be closed-shell

		for(int i = 0; i < nalpha_; ++i){
			occupation[labeled_orb_en_alfa[i].second.second] = 1;
		}
		for(int i = 0; i < nbeta_; ++i){
			occupation[labeled_orb_en_beta[i].second.second + nact_] = 1;
		}

		if( nalpha_ >= nbeta_ ) {
			
			// Loop over k
			for(int k = 1; k < nsym;){
				
				bool add = false;
				// Remove highest energy alpha electron
				occupation[labeled_orb_en_alfa[nalpha_ - k].second.second] = 0;

				outfile->Printf("\n  Electron removed from %d, out of %d", labeled_orb_en_alfa[nalpha_ - k].second.second, nactel_);

				//Determine proper symmetry for new electron
				
				orb_sym = wavefunction_symmetry_;

				if(wavefunction_multiplicity_ == 1){
					orb_sym = labeled_orb_en_alfa[nalpha_ - 1].second.first ^ orb_sym;
				}else{
					for(int i = 1; i <= nsym; ++i){
						orb_sym = labeled_orb_en_alfa[nalpha_ - i].second.first ^ orb_sym;
					}
					orb_sym = labeled_orb_en_alfa[nalpha_ - k].second.first ^ orb_sym;
				}

				outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

				// Add electron to lowest-energy orbital of proper symmetry
				for(int i = nalpha_ - k; i < nactel_; ++i){
					if(orb_sym == labeled_orb_en_alfa[i].second.first and occupation[labeled_orb_en_alfa[i].second.second] != 1 ){
						occupation[labeled_orb_en_alfa[i].second.second] = 1;
						outfile->Printf("\n  Added electron to %d", labeled_orb_en_alfa[i].second.second);
						add = true;
						break;
					}else{
						continue;
					}
				}

				// If a new occupation could not be made,
				// add electron back and try a different one

				if(!add){
					occupation[labeled_orb_en_alfa[nalpha_ - k].second.second] = 1;
					outfile->Printf("\n  No orbital of symmetry %d available! Putting it back...", orb_sym);
					++k;
				}else{
					break;
				}

			}//	End loop over k
		}else{    // End if(nalpha_ >= nbeta_ )
			
			for(int k = 1; k < nsym;){
				
				bool add = false;

				// Remove highest-energy beta electron
				occupation[labeled_orb_en_beta[nbeta_ - k].second.second] = 0;
				outfile->Printf("\n  Electron removed from %d, out of %d", labeled_orb_en_beta[nbeta_ - k].second.second, nactel_);

				//Determine proper symetry for new occupation
				orb_sym = wavefunction_symmetry_;

				if(wavefunction_multiplicity_ == 1){
					orb_sym = labeled_orb_en_beta[nbeta_ - 1].second.first ^ orb_sym;
				}else{
					for(int i = 1; i <= nsym; ++i){
						orb_sym = labeled_orb_en_beta[nbeta_ - i].second.first ^ orb_sym;
					}
					orb_sym = labeled_orb_en_beta[nbeta_ - k].second.first ^  orb_sym;
				}

				outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

				// Add electron to lowest-energy beta orbital
				
				for(int i = nbeta_ - k; i < nactel_; ++i){
					if( orb_sym == labeled_orb_en_beta[i].second.first and occupation[labeled_orb_en_beta[i].second.second] != 1){
						occupation[labeled_orb_en_beta[i].second.second] = 1;
						outfile->Printf("\n Added electron to %d", labeled_orb_en_beta[i].second.second);
						add = true;
						break;
					}
				}

				// If a new occupation could not be made,
				// replace the electron and try again

				if(!add){
					occupation[labeled_orb_en_beta[nbeta_ - k].second.second] = 1;
					outfile->Printf("\n  No orbital of symmetry %d available! Putting electron back...", orb_sym);
					++k;
				}else{
					break;
				}

			} //End loop over k
		} // End if nalpha_ < nbeta_
	}
	return occupation;	
}

double AdaptiveCI::compute_energy()
{
    boost::timer t_iamrcisd;
    outfile->Printf("\n\n  Iterative Adaptive CI");

    SharedMatrix H;
    SharedMatrix P_evecs;
    SharedMatrix PQ_evecs;
    SharedVector P_evals;
    SharedVector PQ_evals;

    // Use the reference determinant as a starting point
    std::vector<bool> alfa_bits = reference_determinant_.get_alfa_bits_vector_bool();
    std::vector<bool> beta_bits = reference_determinant_.get_beta_bits_vector_bool();
    STLBitsetDeterminant bs_det(alfa_bits,beta_bits);
	P_space_.push_back(bs_det);
    P_space_map_[bs_det] = 1;
	
	det_history_[bs_det].push_back(std::make_pair(0, "I"));
	
    outfile->Printf("\n  The model space contains %zu determinants",P_space_.size());
    outfile->Flush();

    std::vector<std::vector<double> > energy_history;
    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("MAXITER_DAVIDSON"));
    sparse_solver.set_spin_project(project_out_spin_contaminants_);

	int spin_projection = options_.get_int("SPIN_PROJECTION");

	if( det_save_) det_list_.open("det_list.txt");
    int maxcycle = 20;
    for (int cycle = 0; cycle < maxcycle; ++cycle){

        // Step 1. Diagonalize the Hamiltonian in the P space
        int num_ref_roots = std::min(nroot_,int(P_space_.size()));
		cycle_ = cycle;
		std::string cycle_h = "Cycle " + std::to_string(cycle_); 
		print_h2(cycle_h);
        outfile->Printf("\n  Initial P space dimension: %zu", P_space_.size());

		// Check that the initial space is spin-complete
		if(spin_complete_){
            STLBitsetDeterminant::enforce_spin_completeness(P_space_);
			outfile->Printf("\n  %s: %zu determinants","Spin-complete dimension of the P space",P_space_.size());
        }else{
			outfile->Printf("\n Not checking for spin-completeness.");
		}
        // Diagonalize H in the P space
        sparse_solver.diagonalize_hamiltonian(P_space_,P_evals,P_evecs,num_ref_roots,wavefunction_multiplicity_,diag_method_);
		if(det_save_) save_dets_to_file( P_space_, P_evecs );

		// Save the dimention of the previous PQ space
		//size_t PQ_space_prev = PQ_space_.size();

		// Use spin projection to ensure the P space is spin pure
		if( spin_projection == 1 or spin_projection == 3){
			double spin_contamination = compute_spin_contamination(P_space_, P_evecs, num_ref_roots);
			if(spin_contamination >= spin_tol_){
				outfile->Printf("\n  Average spin contamination per root is %1.5f", spin_contamination);
				full_spin_transform(P_space_, P_evecs, num_ref_roots);
				P_evecs->zero();
				P_evecs = PQ_spin_evecs_->clone();
                sparse_solver.compute_H_expectation_val(P_space_,P_evals,P_evecs,num_ref_roots,diag_method_);
			}else{
				outfile->Printf("\n  Average spin contamination (%1.5f) is less than tolerance (%1.5f)", spin_contamination, spin_tol_);
				outfile->Printf("\n  No need to perform spin projection.");
			}
		}else{
			outfile->Printf("\n  Not performing spin projection.");
		}

        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots; ++i){
            double abs_energy = P_evals->get(i) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
            double exc_energy = pc_hartree2ev * (P_evals->get(i) - P_evals->get(0));
            outfile->Printf("\n    P-space  CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,abs_energy,exc_energy);
        }
        outfile->Printf("\n");
        outfile->Flush();


        // Step 2. Find determinants in the Q space        
        find_q_space(num_ref_roots,P_evals,P_evecs);

		// Check if P+Q space is spin complete
		if(spin_complete_){
            STLBitsetDeterminant::enforce_spin_completeness(PQ_space_);
			outfile->Printf("\n  Spin-complete dimension of the PQ space: %zu", PQ_space_.size());
		}

        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        sparse_solver.diagonalize_hamiltonian(PQ_space_,PQ_evals,PQ_evecs,num_ref_roots,wavefunction_multiplicity_,diag_method_);
		if(det_save_) save_dets_to_file( PQ_space_, PQ_evecs );

		// Ensure the solutions are spin-pure
		if( spin_projection == 1 or spin_projection == 3){
			double spin_contamination = compute_spin_contamination(PQ_space_, PQ_evecs, num_ref_roots);
			if(spin_contamination >= spin_tol_){
				outfile->Printf("\n  Average spin contamination per root is %1.5f", spin_contamination);
				full_spin_transform(PQ_space_, PQ_evecs, num_ref_roots);
				PQ_evecs->zero();
				PQ_evecs = PQ_spin_evecs_->clone();
                sparse_solver.compute_H_expectation_val(PQ_space_,PQ_evals,PQ_evecs,num_ref_roots,diag_method_);
			}else{
				outfile->Printf("\n Average spin contamination (%1.5f) is less than tolerance (%1.5f)", spin_contamination, spin_tol_);
				outfile->Printf("\n No need to perform spin projection.");
			}
		}else{
			outfile->Printf("\n  Not performing spin projection.");
		}

        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots; ++ i){
            double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
            double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
            outfile->Printf("\n    PQ-space CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,abs_energy,exc_energy);
            outfile->Printf("\n    PQ-space CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV",i + 1,abs_energy + multistate_pt2_energy_correction_[i],
                            exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
        }
        outfile->Printf("\n");
        outfile->Flush();


        // Step 4. Check convergence and break if needed
        bool converged = check_convergence(energy_history,PQ_evals);
        if (converged) break;

        // Step 5. Prune the P + Q space to get an updated P space
        prune_q_space(PQ_space_,P_space_,P_space_map_,PQ_evecs,num_ref_roots);        

        // Print information about the wave function
        print_wfn(PQ_space_,PQ_evecs,num_ref_roots);
    }// end iterations

	if( det_save_ ) det_list_.close();
    outfile->Printf("\n\n  ==> Post-Iterations <==\n");

	//STLBitsetDeterminant::check_uniqueness(P_space_);

	// Ensure the solutions are spin-pure
	if( spin_projection == 2 or spin_projection == 3){
		double spin_contamination = compute_spin_contamination(PQ_space_, PQ_evecs, nroot_);
		if(spin_contamination >= spin_tol_){
			outfile->Printf("\n  Average spin contamination per root is %1.5f", spin_contamination);
			full_spin_transform(PQ_space_, PQ_evecs, nroot_);
			outfile->Printf("\n  ...done");
			PQ_evecs->print();
		//	P_evecs->zero();
			//P_evecs = PQ_spin_evecs_->clone();
            sparse_solver.compute_H_expectation_val(PQ_space_,PQ_evals,PQ_evecs,nroot_,diag_method_);
		}else{
			outfile->Printf("\n  Average spin contamination (%1.5f) is less than tolerance (%1.5f)", spin_contamination, spin_tol_);
			outfile->Printf("\n  No need to perform spin projection.");
		}
	}else{
		outfile->Printf("\n  Not performing spin projection.");
	}

    // Do Hamiltonian smoothing
    if (do_smooth_){
        smooth_hamiltonian(P_space_,P_evals,P_evecs,nroot_);
    }

	outfile->Printf("\n  Printing Wavefunction Information");
	print_wfn(PQ_space_, PQ_evecs, nroot_);
	outfile->Printf("\n\n     Order		 # of Dets        Total |c^2|   ");
	outfile->Printf(  "\n  __________ 	____________   ________________ ");
    wfn_analyzer(PQ_space_, PQ_evecs, nroot_);	

    for (int i = 0; i < nroot_; ++ i){
        double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
        double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
        outfile->Printf("\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,abs_energy,exc_energy);
        outfile->Printf("\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV",i + 1,abs_energy + multistate_pt2_energy_correction_[i],
                exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
    }


	if(options_.get_bool("DETERMINANT_HISTORY")){
		outfile->Printf("\n Det history (number,cycle,origin)");
		size_t counter = 0;
		for( auto &I : PQ_space_ ){
			outfile->Printf("\n Det number : %zu", counter);
			for( auto &n : det_history_[I]){
				outfile->Printf("\n %zu	   %s", n.first, n.second.c_str());		
			}
			++counter;
		}
	}
	if( form_1_RDM_ ){
	//Compute and print 1-RDM
		SharedMatrix Dalpha(new Matrix("Dalpha", nmo_, nmo_));
		SharedMatrix Dbeta(new Matrix("Dbeta", nmo_, nmo_));

		compute_1rdm(Dalpha,Dbeta,PQ_space_,PQ_evecs,nroot_);
		diagonalize_order nMatz = evals_only_descending;

		SharedVector no_occnum(new Vector("NO occ num", nmo_));
		SharedMatrix NO_evecs(new Matrix("NO Evecs", nmo_,nmo_));
		D1_->diagonalize(NO_evecs, no_occnum, nMatz);
		no_occnum->print();

		std::vector<int> active_space(nirrep_, 0);
		for(size_t p = 0, maxp = D1_->nrow(); p < maxp; ++p){
			if( (D1_->get(p,p) >= 0.02) and (D1_->get(p,p) <= 1.98) ){
				active_space[mo_symmetry_[p]] += 1;
			}else{
				continue;
			}
		}
		outfile->Printf("\n Suggested active space from ACI:  \n  [");
		for(int i = 0; i < nirrep_; ++i) outfile->Printf(" %d", active_space[i]);
		outfile->Printf(" ]");
	}

    outfile->Printf("\n\n  %s: %f s","Adaptive-CI (bitset) ran in ",t_iamrcisd.elapsed());
    outfile->Printf("\n\n  %s: %d","Saving information for root",options_.get_int("ROOT") + 1);
    outfile->Flush();

    double root_energy = PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
    double root_energy_pt2 = root_energy + multistate_pt2_energy_correction_[options_.get_int("ROOT")];
    Process::environment.globals["CURRENT ENERGY"] = root_energy;
    Process::environment.globals["ACI ENERGY"] = root_energy;
    Process::environment.globals["ACI+PT2 ENERGY"] = root_energy_pt2;

    return PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_ + fci_ints_->scalar_energy();
}


void AdaptiveCI::find_q_space(int nroot,SharedVector evals,SharedMatrix evecs)
{
    boost::timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    det_hash<std::vector<double> > V_hash;

    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I){
        STLBitsetDeterminant& det = P_space_[I];
        generate_excited_determinants(nroot,I,evecs,det,V_hash);
    }
    outfile->Printf("\n  %s: %zu determinants","Dimension of the SD space",V_hash.size());
    outfile->Printf("\n  %s: %f s\n","Time spent building the model space",t_ms_build.elapsed());
    outfile->Flush();

    // This will contain all the determinants
    PQ_space_.clear();

    // Add the P-space determinants and zero the hash
    for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J){
	//	outfile->Printf("\n  det: %s", P_space_[J].str().c_str());
        PQ_space_.push_back(P_space_[J]);
        V_hash.erase(P_space_[J]);
    }

    boost::timer t_ms_screen;

    std::vector<double> C1(nroot_,0.0);
    std::vector<double> E2(nroot_,0.0);
	std::vector<double> e2(nroot_,0.0);
    std::vector<double> ept2(nroot_,0.0);
	double criteria;
    std::vector<std::pair<double,STLBitsetDeterminant>> sorted_dets;

	// Define coupling out of loop, assume perturb_select_ = false	
	std::function<double (double A, double B, double C)> C1_eq = [](double A, double B, double C)->double 
		{return 0.5 * ( (B - C) - sqrt( (B - C)*(B - C) + 4.0 * A*A) ) / A;};
  
	std::function< double (double A, double B, double C)> E2_eq = [](double A, double B, double C)->double
        {return 0.5 * ( (B - C) - sqrt( (B - C)*(B - C) + 4.0 * A*A) );};

	if(perturb_select_){
		C1_eq = [](double A, double B, double C)->double {return -A / (B - C); }; 
		E2_eq = [](double A, double B, double C)->double {return -A * A/ (B - C);}; 
	}else{
		outfile->Printf("\n  Using non-perturbative energy estimates");
	}

    // Check the coupling between the reference and the SD space
    for (const auto& it : V_hash){
        double EI = it.first.energy();
        for (int n = 0; n < nroot; ++n){
            double V = it.second[n];
            double C1_I = C1_eq(V, EI, evals->get(n) );
            double E2_I = E2_eq(V, EI, evals->get(n) );

            C1[n] = std::fabs(C1_I);
            E2[n] = std::fabs(E2_I);

			e2[n] = E2_I;
        }

		if(ex_alg_ == "STATE_AVERAGE" and nroot_ != 1){
			criteria = average_q_values(nroot, C1, E2);
		}else{
			criteria = root_select(nroot, C1, E2);
		}

        if (aimed_selection_){
            sorted_dets.push_back(std::make_pair(criteria,it.first));
		}else{
            if (std::fabs(criteria) > tau_q_){
                PQ_space_.push_back(it.first);
            }else{
                for (int n = 0; n < nroot; ++n){
                    ept2[n] += e2[n];
                }
            }
        }
    } // end loop over determinants 
    if (aimed_selection_){
        // Sort the CI coefficients in ascending order
        std::sort(sorted_dets.begin(),sorted_dets.end(),pairComp);

        double sum = 0.0;
        size_t last_excluded = 0;
        for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I){
            const STLBitsetDeterminant& det = sorted_dets[I].second;
            if (sum + sorted_dets[I].first < tau_q_){
                sum += sorted_dets[I].first;
                double EI = det.energy();
                const std::vector<double>& V_vec = V_hash[det];
                for (int n = 0; n < nroot; ++n){
                    double V = V_vec[n];
                    double E2_I = E2_eq( V, EI, evals->get(n) );

                    ept2[n] += E2_I;
                }
                last_excluded = I;
            }else{
                PQ_space_.push_back(sorted_dets[I].second);
                det_history_[sorted_dets[I].second].push_back(std::make_pair(cycle_, "Q"));
            }
        }

        // add missing determinants that have the same weight as the last one included
        if (add_aimed_degenerate_){
            size_t num_extra = 0;
            for (size_t I = 0, max_I = last_excluded; I < max_I; ++I){
                size_t J = last_excluded - I;
                if (std::fabs(sorted_dets[last_excluded + 1].first - sorted_dets[J].first) < 1.0e-9){
                    PQ_space_.push_back(sorted_dets[J].second);
                    det_history_[sorted_dets[J].second].push_back(std::make_pair(cycle_, "Q"));
                    num_extra++;
                }else{
                    break;
                }
            }
            if (num_extra > 0){
                outfile->Printf("\n  Added %zu missing determinants in aimed selection.",num_extra);
            }
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    outfile->Printf("\n  %s: %zu determinants","Dimension of the P + Q space",PQ_space_.size());
    outfile->Printf("\n  %s: %f s","Time spent screening the model space",t_ms_screen.elapsed());
    outfile->Flush();
}

double AdaptiveCI::average_q_values( int nroot,std::vector<double> C1, std::vector<double>E2)
{
	// f_E2 and f_C1 will store the selected function of the chosen q criteria
	// This functions should only be called when nroot_ > 1
	
	double f_C1 = 0.0;
	double f_E2 = 0.0;

	std::vector<double> dE2(nroot, 0.0);

	q_rel_ = options_.get_bool("Q_REL");

	if(q_rel_ == true and nroot > 1){
		if( q_reference_ == "ADJACENT" ){
			for( int n = 1; n < nroot; ++n){
				dE2[n] = std::fabs(E2[n-1] - E2[n]);
			}
		}else{ //Default to "GS"
			for( int n = 1; n < nroot; ++n){
				dE2[n] = std::fabs(E2[n] - E2[0]);
			}
		}
	}else if(q_rel_ == true and nroot == 1){
		q_rel_ = false;
	}

	//Choose the function of the couplings for each root
	//If nroot = 1, choose the max
	
	if( pq_function_ == "MAX" or nroot == 1 ){
		f_C1 = *std::max_element(C1.begin(), C1.end());
		f_E2 = (q_rel_ and (nroot!=1)) ? *std::max_element(dE2.begin(), dE2.end()) : 
										 *std::max_element(E2.begin(), E2.end() );
	}else if( pq_function_ == "AVERAGE" ){
		double C1_average = 0.0;
		double E2_average = 0.0;
		double dE2_average = 0.0;
		double dim_inv = 1.0 / nroot;
		for(int n = 0; n < nroot; ++n){
			C1_average += C1[n] * dim_inv;
			E2_average += E2[n] * dim_inv;
		}
		if(q_rel_){
			double inv = 1.0 / (nroot - 1.0);
			for(int n = 1; n < nroot; ++n){
				dE2_average += dE2[n] * inv;
			}
		}
		f_C1 = C1_average;
		f_E2 = q_rel_ ? dE2_average : E2_average;
	}

	double select_value = 0.0;
	if(aimed_selection_){
		select_value = energy_selection_ ? f_E2 : (f_C1*f_C1);
	}else{
		select_value = energy_selection_ ? f_E2 : f_C1;
	}

	return select_value;
}

double AdaptiveCI::root_select( int nroot, std::vector<double> C1, std::vector<double> E2)
{
	double select_value;
	ref_root_ = options_.get_int("ROOT");

	if(ref_root_ + 1 > nroot_){
		throw PSIEXCEPTION("\n  Your selection is not valid. Check ROOT in options.");
	}

	if(nroot == 1){
		ref_root_ = 0;
	}

	if(aimed_selection_){
		select_value = energy_selection_ ? E2[ref_root_] : (C1[ref_root_]*C1[ref_root_]);
	}else{
		select_value = energy_selection_ ? E2[ref_root_] : C1[ref_root_];
	}

	return select_value;
}

void AdaptiveCI::generate_excited_determinants(int nroot,int I,SharedMatrix evecs,STLBitsetDeterminant& det,det_hash<std::vector<double>>& V_hash)
{
    std::vector<int> aocc = det.get_alfa_occ();
    std::vector<int> bocc = det.get_beta_occ();
    std::vector<int> avir = det.get_alfa_vir();
    std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a){
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0){
                STLBitsetDeterminant new_det(det);
                new_det.set_alfa_bit(ii,false);
                new_det.set_alfa_bit(aa,true);
                if(P_space_map_.find(new_det) == P_space_map_.end()){
                    double HIJ = det.slater_rules(new_det);
                    if (V_hash.count(new_det) == 0){
                        V_hash[new_det] = std::vector<double>(nroot);
                    }
                    for (int n = 0; n < nroot; ++n){
                        V_hash[new_det][n] += HIJ * evecs->get(I,n);
                    }
                }
            }
        }
    }

    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a){
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0){
                STLBitsetDeterminant new_det(det);
                new_det.set_beta_bit(ii,false);
                new_det.set_beta_bit(aa,true);
                if(P_space_map_.find(new_det) == P_space_map_.end()){
                    double HIJ = det.slater_rules(new_det);
                    if (V_hash.count(new_det) == 0){
                        V_hash[new_det] = std::vector<double>(nroot);
                    }
                    for (int n = 0; n < nroot; ++n){
                        V_hash[new_det][n] += HIJ * evecs->get(I,n);
                    }
                }
            }
        }
    }

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j){
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b){
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0){
                        STLBitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii,false);
                        new_det.set_alfa_bit(jj,false);
                        new_det.set_alfa_bit(aa,true);
                        new_det.set_alfa_bit(bb,true);
                        if(P_space_map_.find(new_det) == P_space_map_.end()){
							double HIJ = fci_ints_->tei_aa(ii,jj,aa,bb);

							HIJ *= det.slater_sign_alpha(ii) * det.slater_sign_alpha(jj) * new_det.slater_sign_alpha(aa) * new_det.slater_sign_alpha(bb);

                            if (V_hash.count(new_det) == 0){
                                V_hash[new_det] = std::vector<double>(nroot);
                            }
                            for (int n = 0; n < nroot; ++n){
                                V_hash[new_det][n] += HIJ * evecs->get(I,n);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j){
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b){
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0){
                        STLBitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii,false);
                        new_det.set_beta_bit(jj,false);
                        new_det.set_alfa_bit(aa,true);
                        new_det.set_beta_bit(bb,true);
                        if(P_space_map_.find(new_det) == P_space_map_.end()){
							double HIJ = fci_ints_->tei_ab(ii,jj,aa,bb);

							HIJ *= det.slater_sign_alpha(ii) * det.slater_sign_beta(jj) * new_det.slater_sign_alpha(aa) * new_det.slater_sign_beta(bb);

                            if (V_hash.count(new_det) == 0){
                                V_hash[new_det] = std::vector<double>(nroot);
                            }
                            for (int n = 0; n < nroot; ++n){
                                V_hash[new_det][n] += HIJ * evecs->get(I,n);
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j){
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a){
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b){
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0){
                        STLBitsetDeterminant new_det(det);
                        new_det.set_beta_bit(ii,false);
                        new_det.set_beta_bit(jj,false);
                        new_det.set_beta_bit(aa,true);
                        new_det.set_beta_bit(bb,true);
                        if(P_space_map_.find(new_det) == P_space_map_.end()){
                            double HIJ = fci_ints_->tei_bb(ii,jj,aa,bb);

							HIJ *= det.slater_sign_beta(ii) * det.slater_sign_beta(jj) * new_det.slater_sign_beta(aa) * new_det.slater_sign_beta(bb);
                            if (V_hash.count(new_det) == 0){
                                V_hash[new_det] = std::vector<double>(nroot);
                            }
                            for (int n = 0; n < nroot; ++n){
                                V_hash[new_det][n] += HIJ * evecs->get(I,n);
                            }
                        }
                    }
                }
            }
        }
    }
}

void AdaptiveCI::generate_pair_excited_determinants(int nroot,int I,SharedMatrix evecs,STLBitsetDeterminant& det,det_hash<std::vector<double>>& V_hash)
{
    std::vector<int> aocc = det.get_alfa_occ();
    std::vector<int> bocc = det.get_beta_occ();
    std::vector<int> avir = det.get_alfa_vir();
    std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a){
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0){
                STLBitsetDeterminant new_det(det);
                new_det.set_alfa_bit(ii,false);
                new_det.set_alfa_bit(aa,true);
                if(P_space_map_.find(new_det) == P_space_map_.end()){
                    double HIJ = det.slater_rules(new_det);
                    if (V_hash.count(new_det) == 0){
                        V_hash[new_det] = std::vector<double>(nroot);
                    }
                    for (int n = 0; n < nroot; ++n){
                        V_hash[new_det][n] += HIJ * evecs->get(I,n);
                    }
                }
            }
        }
    }

    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a){
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == 0){
                STLBitsetDeterminant new_det(det);
                new_det.set_beta_bit(ii,false);
                new_det.set_beta_bit(aa,true);
                if(P_space_map_.find(new_det) == P_space_map_.end()){
                    double HIJ = det.slater_rules(new_det);
                    if (V_hash.count(new_det) == 0){
                        V_hash[new_det] = std::vector<double>(nroot);
                    }
                    for (int n = 0; n < nroot; ++n){
                        V_hash[new_det][n] += HIJ * evecs->get(I,n);
                    }
                }
            }
        }
    }

    // Generate a/b pair excitations

    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        if (det.get_beta_bit(ii)){
            int jj = ii;
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                if (not det.get_beta_bit(aa)){
                    int bb = aa;
                    STLBitsetDeterminant new_det(det);
                    new_det.set_alfa_bit(ii,false);
                    new_det.set_beta_bit(jj,false);
                    new_det.set_alfa_bit(aa,true);
                    new_det.set_beta_bit(bb,true);
                    if(P_space_map_.find(new_det) == P_space_map_.end()){
                        double HIJ = det.slater_rules(new_det);
                        if (V_hash.count(new_det) == 0){
                            V_hash[new_det] = std::vector<double>(nroot);
                        }
                        for (int n = 0; n < nroot; ++n){
                            V_hash[new_det][n] += HIJ * evecs->get(I,n);
                        }
                    }
                }
            }
        }
    }
}

bool AdaptiveCI::check_convergence(std::vector<std::vector<double>>& energy_history,SharedVector evals)
{
    int nroot = evals->dim();

    if (energy_history.size() == 0){
        std::vector<double> new_energies;
        for (int n = 0; n < nroot; ++ n){
            double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
            new_energies.push_back(state_n_energy);
        }
        energy_history.push_back(new_energies);
        return false;
    }

    double old_avg_energy = 0.0;
    double new_avg_energy = 0.0;

    std::vector<double> new_energies;
    std::vector<double> old_energies = energy_history[energy_history.size() - 1];
    for (int n = 0; n < nroot; ++ n){
        double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
        new_energies.push_back(state_n_energy);
        new_avg_energy += state_n_energy;
        old_avg_energy += old_energies[n];
    }
    old_avg_energy /= static_cast<double>(nroot);
    new_avg_energy /= static_cast<double>(nroot);

    energy_history.push_back(new_energies);

    // Check for convergence
    return (std::fabs(new_avg_energy - old_avg_energy) < options_.get_double("E_CONVERGENCE"));
    //        // Check the history of energies to avoid cycling in a loop
    //        if(cycle > 3){
    //            bool stuck = true;
    //            for(int cycle_test = cycle - 2; cycle_test < cycle; ++cycle_test){
    //                for (int n = 0; n < nroot_; ++n){
    //                    if(std::fabs(energy_history[cycle_test][n] - energies[n]) < 1.0e-12){
    //                        stuck = true;
    //                    }
    //                }
    //            }
    //            if(stuck) break; // exit the cycle
    //        }
}

void AdaptiveCI::prune_q_space(std::vector<STLBitsetDeterminant>& large_space,std::vector<STLBitsetDeterminant>& pruned_space,
                               det_hash<int>& pruned_space_map,SharedMatrix evecs,int nroot)
{
    // Select the new reference space using the sorted CI coefficients
    pruned_space.clear();
    pruned_space_map.clear();

    // Create a vector that stores the absolute value of the CI coefficients
    std::vector<std::pair<double,size_t> > dm_det_list;
    for (size_t I = 0, max = large_space.size(); I < max; ++I){
        double criteria = 0.0;
        for (int n = 0; n < nroot; ++n){
            if(pq_function_ == "MAX"){
				criteria = std::max(criteria, std::fabs(evecs->get(I,n)));
			}else if(pq_function_ == "AVERAGE"){
				criteria += std::fabs(evecs->get(I,n));
			}
		}
		criteria /= static_cast<double>(nroot);
        dm_det_list.push_back(std::make_pair(criteria,I));
    }

    // Decide which determinants will go in pruned_space
    // Include all determinants such that
    // sum_I |C_I|^2 < tau_p, where the sum runs over all the excluded determinants
    if (aimed_selection_){
        // Sort the CI coefficients in ascending order
        std::sort(dm_det_list.begin(),dm_det_list.end());

        double sum = 0.0;
        size_t last_excluded = 0;
        for (size_t I = 0; I < large_space.size(); ++I){
            double dsum = std::pow(dm_det_list[I].first,2.0);
            if (sum + dsum < tau_p_){ // exclude small contributions that sum to less than tau_p
                sum += dsum;
                last_excluded = I;
            }else{
                pruned_space.push_back(large_space[dm_det_list[I].second]);
                pruned_space_map[large_space[dm_det_list[I].second]] = 1;
            }
        }

        // add missing determinants that have the same weight as the last one included
        if (add_aimed_degenerate_){
            size_t num_extra = 0;
            for (size_t I = 0, max_I = last_excluded; I < max_I; ++I){
                size_t J = last_excluded - I;
                if (std::fabs(dm_det_list[last_excluded + 1].first - dm_det_list[J].first) < 1.0e-9){
                    pruned_space.push_back(large_space[dm_det_list[J].second]);
                    pruned_space_map[large_space[dm_det_list[J].second]] = 1;
                    num_extra += 1;
                }else{
                    break;
                }
            }
            if (num_extra > 0){
                outfile->Printf("\n  Added %zu missing determinants in aimed selection.",num_extra);
            }
        }
    }
    // Include all determinants such that |C_I| > tau_p
    else{
        for (size_t I = 0; I < large_space.size(); ++I){
            if (dm_det_list[I].first > tau_p_){
                pruned_space.push_back(large_space[dm_det_list[I].second]);
                pruned_space_map[large_space[dm_det_list[I].second]] = 1;
            }
        }
    }
}


bool AdaptiveCI::check_stuck(std::vector<std::vector<double>>& energy_history, SharedVector evals)
{
	int nroot = evals->dim();
	if(cycle_ < 3){
		return false;
	}else{
		std::vector<double> av_energies;
		for(int i = 0; i < cycle_; ++i){
			double energy = 0.0;
			for(int n = 0; n < nroot; ++n){
				energy += energy_history[i][n];
			}
			energy /= static_cast<double>(nroot);
			av_energies.push_back(energy);
		}

		if( std::fabs( av_energies[cycle_ - 1] - av_energies[ cycle_ - 3] ) < options_.get_double("E_CONVERGENCE") and
			std::fabs( av_energies[cycle_] - av_energies[cycle_ - 2] ) < options_.get_double("E_CONVERGENCE") ){
			return true;
		}else{
			return false;
		}
	}
}

pVector<std::pair<double, double>, std::pair<size_t,double>> AdaptiveCI::compute_spin(std::vector<STLBitsetDeterminant> space,
																					  SharedMatrix evecs,
																					  int nroot)
{
	double norm;
	double S2;
	double S;
	pVector<std::pair<double,double>, std::pair<size_t, double> > spin_vec;

	for(int n = 0; n < nroot; ++n){
		//Compute the expectation value of the spin
		size_t max_sample = 1000;
		size_t max_I = 0;
		double sum_weight = 0.0;
		pVector<double, size_t> det_weight;

		for(size_t I = 0, max = space.size(); I < max; ++I){
			det_weight.push_back(make_pair(evecs->get(I,n), I));
		}

		//Don't require the determinats to be pre-ordered

		std::sort(det_weight.begin(), det_weight.end());
		std::reverse(det_weight.begin(), det_weight.end());

		const double wfn_threshold = (space.size() < 10) ? 1.00: 0.95;
		for(size_t I = 0, max = space.size(); I < max; ++I){
			if( (sum_weight < wfn_threshold) and (I < max_sample)){
				sum_weight += det_weight[I].first * det_weight[I].first;
				max_I++;
			}else if(std::fabs(det_weight[I].first - det_weight[I-1].first) < 1.0e-6){
				//Special case, if there are several equivalent determinants
				sum_weight += det_weight[I].first * det_weight[I].first;
				max_I++;
			}else{
				break;
			}
		}
		
		S2 = 0.0;
		norm = 0.0;
		for(size_t sI = 0; sI < max_I; ++sI){
			size_t I = det_weight[sI].second;
			for(size_t sJ = 0; sJ < max_I; ++sJ){
				size_t J = det_weight[sJ].second;
				if(std::fabs(evecs->get(I,n) * evecs->get(J,n)) > 1.0e-12){
					const double S2IJ = space[I].spin2(space[J]);
					S2 += evecs->get(I,n) * evecs->get(J,n) * S2IJ;
				}
			}
			norm += evecs->get(I,n) * evecs->get(I,n);
		}

		S2 /= norm;
		S2  = std::fabs(S2);
		S   = std::fabs( 0.5*(std::sqrt(1.0 + 4.0*S2)-1.0) );
		spin_vec.push_back( make_pair(make_pair(S,S2), make_pair(max_I, sum_weight)));
	}
	return spin_vec;
}

void AdaptiveCI::wfn_analyzer(std::vector<STLBitsetDeterminant> det_space, SharedMatrix evecs, int nroot)
{

    std::vector<bool> occ(2*nact_,0);
    oVector<double,int,int> labeled_orb_en = sym_labeled_orbitals("RHF");
    for(int i = 0 ; i < nalpha_; ++i){
        occ[labeled_orb_en[i].second.second] = 1;
    }
    for(int i = 0; i < nbeta_; ++i){
        occ[nact_ + labeled_orb_en[i].second.second] = 1;
    } 
    
    STLBitsetDeterminant rdet(occ);
	auto ref_bits = rdet.bits();
	for(int n = 0; n < nroot; ++n){
		pVector<size_t,double> excitation_counter( 1 + (1 + cycle_) * 2 );
		pVector<double,size_t> det_weight;
		for( size_t I = 0, max = det_space.size(); I < max; ++I){
			det_weight.push_back(std::make_pair(std::fabs(evecs->get(I,n)),I));
		}

		std::sort(det_weight.begin(), det_weight.end());
		std::reverse(det_weight.begin(), det_weight.end());


		for(size_t I = 0, max = det_space.size(); I < max; ++I){
			int ndiff = 0;
			auto ex_bits = det_space[det_weight[I].second].bits();

			//Compute number of differences in both alpha and beta strings wrt ref
			for(size_t a = 0; a < nact_ *2; ++a){
				if(ref_bits[a] != ex_bits[a]){
					++ndiff;
				}
			}
			ndiff /= 2;
			excitation_counter[ndiff] = std::make_pair(excitation_counter[ndiff].first + 1,
													   excitation_counter[ndiff].second + det_weight[I].first * det_weight[I].first);
		}
		int order = 0;
		size_t det = 0;
		for(auto& i : excitation_counter){
			outfile->Printf("\n     %2d			%8zu		%.11f", order, i.first, i.second);
			det += i.first;
			if(det == det_space.size()) break;
			++order;
		}
		outfile->Printf("\n\n Highest-order excitation searched:     %zu  \n", excitation_counter.size() - 1);
	}
}

oVector<double, int, int> AdaptiveCI::sym_labeled_orbitals(std::string type)
{
	oVector<double, int, int> labeled_orb;

	if(type == "RHF" or type == "ROHF" or type == "ALFA"){
		
		// Create a vector of orbital energy and index pairs
		pVector<double, int> orb_e;
		int cumidx = 0;
		for(int h = 0; h < nirrep_; ++h){
			for(int a = 0; a < nactpi_[h]; ++a ){
                orb_e.push_back(make_pair(epsilon_a_->get(h,frzcpi_[h] + a), a + cumidx));
			}
			cumidx += nactpi_[h];
		}

		// Create a vector that stores the orbital energy, symmetry, and idx
		for( size_t a = 0; a < nact_; ++a){
			labeled_orb.push_back( make_pair(orb_e[a].first, make_pair(mo_symmetry_[a], orb_e[a].second) ) );
		}
		//Order by energy, low to high
		std::sort(labeled_orb.begin(), labeled_orb.end());
	}
	if(type == "BETA"){
		//Create a vector of orbital energies and index pairs
		pVector<double, int> orb_e;
		int cumidx = 0;
		for(int h  = 0; h < nirrep_; ++h){
			for(size_t a = 0, max = nactpi_[h]; a < max; ++a){
                orb_e.push_back(make_pair(epsilon_b_->get(h,frzcpi_[h] + a), a+cumidx));
			}
			cumidx += nactpi_[h];
		}

		//Create a vector that stores the orbital energy, sym, and idx
		for(size_t a = 0; a < nact_; ++a){
			labeled_orb.push_back(make_pair(orb_e[a].first, make_pair(mo_symmetry_[a], orb_e[a].second) ));
		}
		std::sort(labeled_orb.begin(), labeled_orb.end());
	}

//	for(int i = 0; i < nact_; ++i){
//		outfile->Printf("\n %1.5f    %d    %d", labeled_orb[i].first, labeled_orb[i].second.first, labeled_orb[i].second.second);
//	}

	return labeled_orb;
	
}

void AdaptiveCI::compute_1rdm( SharedMatrix A, SharedMatrix B, std::vector<STLBitsetDeterminant> det_space, SharedMatrix evecs, int nroot)
{
	// Make a vector of indices for core and active orbitals
	
	size_t ncmopi = 0;
	std::vector<size_t> idx_a;
	std::vector<size_t> idx_c;

	for(int h = 0; h < nirrep_; ++h){
		for( int i = 0; i < ncmopi_[h]; ++i){
			int idx = i + ncmopi;
			if( i < frzcpi_[h] ){
				idx_c.push_back(idx);
			}else if( i >= frzcpi_[h] and i < (frzcpi_[h] + ncmopi_[h])){
				idx_a.push_back(idx);
			}
		}
		ncmopi += nmopi_[h];
	}

	//Occupy frozen core with 1.0
	for(int p = 0; p < nfrzc_; ++p){
		size_t np = idx_c[p];
		A->set(np,np,1.0);
		B->set(np,np,1.0);
	}
	
	double trace = 0.0;
	for(int p = 0; p < ncmo_; ++p){
		//size_t np = idx_a[p];
		for(int q = p; q < nmo_; ++q){
			D1_->add(p,q, B->get(p,q));
			if(p ==q) trace += D1_->get(p,p);
		}
	}
	outfile->Printf("\n\n Trace of 1-RDM: %6.3f\n", trace); 
}

double AdaptiveCI::OneOP(const STLBitsetDeterminant &J, STLBitsetDeterminant &Jnew, const bool sp, const size_t &p, const size_t &q)
{
	STLBitsetDeterminant tmp = J;

	double sign = 1.0;

	if( sp == false ){
		if( tmp.get_alfa_bit(q)) {
			sign *= CheckSign(tmp.get_alfa_occ(),q);
			tmp.set_alfa_bit(q,0);
		}else{
			return 0.0;
		}

		if( !tmp.get_alfa_bit(p) ){
			sign *= CheckSign(tmp.get_alfa_occ(),p);
			tmp.set_alfa_bit(p,1);
			Jnew = tmp;
			return sign;
		}else{
			return 0.0;
		}
	}else{
		if( tmp.get_beta_bit(q) ){
			sign *= CheckSign(tmp.get_beta_occ(),q);
			tmp.set_beta_bit(q,0);
		}else{
			return 0.0;
		}
 
		if( !tmp.get_beta_bit(p) ){
			sign *= CheckSign(tmp.get_beta_occ(),p);
			tmp.set_beta_bit(p,1);
			Jnew = tmp;
			return sign;
		 }else{
			return 0.0;
		}
	}
}

double AdaptiveCI::CheckSign( std::vector<int> I, const int &n )
{
	size_t count = 0;
	for( int i = 0; i < n; ++i){
		
		if( I[i] ) count++;
	
	}

	return std::pow( -1.0, count % 2 );

}

void AdaptiveCI::print_wfn(std::vector<STLBitsetDeterminant> space,SharedMatrix evecs,int nroot)
{
	std::string state_label;
	std::vector<string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet", "sextet","septet","octet","nonet", "decatet"});

    for (int n = 0; n < nroot; ++n){
        outfile->Printf("\n\n  Most important contributions to root %3d:",n);

        std::vector<std::pair<double,size_t> > det_weight;
        for (size_t I = 0; I < space.size(); ++I){
            det_weight.push_back(std::make_pair(std::fabs(evecs->get(I,n)),I));
        }
        std::sort(det_weight.begin(),det_weight.end());
        std::reverse(det_weight.begin(),det_weight.end());
        size_t max_dets = std::min(10,evecs->nrow());
        for (size_t I = 0; I < max_dets; ++I){
            outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s",
                    I,
                    evecs->get(det_weight[I].second,n),
                    det_weight[I].first * det_weight[I].first,
                    det_weight[I].second,
                    space[det_weight[I].second].str().c_str());
        }

		auto spins = compute_spin(space,evecs,nroot);
		state_label = s2_labels[std::round(spins[n].first.first * 2.0)];
		root_spin_vec_.clear();
		root_spin_vec_[n] = make_pair(spins[n].first.first, spins[n].first.second);
        outfile->Printf("\n\n  Spin state for root %zu: S^2 = %5.3f, S = %5.3f, %s (from %zu determinants, %3.2f%)",
			n,
			spins[n].first.second,
			spins[n].first.first,
			state_label.c_str(),
			spins[n].second.first,
			100.0 * spins[n].second.second);
    }
	outfile->Flush();
}

void AdaptiveCI::full_spin_transform( std::vector< STLBitsetDeterminant > det_space, SharedMatrix cI, int nroot )
{
	Timer timer;
	outfile->Printf("\n  Performing spin projection...");

	// Build the S^2 Matrix
	size_t det_size = det_space.size();
	SharedMatrix S2(new Matrix("S^2", det_size, det_size));

	for(size_t I = 0; I < det_size; ++I ){
		for(size_t J = 0; J <= I; ++J){
			S2->set(I,J, det_space[I].spin2(det_space[J]) );
			S2->set(J,I, S2->get(I,J) );
		}
	}

	//Diagonalize S^2, evals will be in ascending order
	SharedMatrix T(new Matrix("T", det_size, det_size)); 
	SharedVector evals(new Vector("evals", det_size));
	S2->diagonalize(T, evals);

	//evals->print();	

	// Count the number of CSFs with correct spin
	// and get their indices wrt columns in T
	size_t csf_num = 0;
	size_t csf_idx = 0;
	double criteria = (0.25 * (wavefunction_multiplicity_ * wavefunction_multiplicity_ - 1.0));
	//double criteria = static_cast<double>(wavefunction_multiplicity_) - 1.0;
	for(size_t l = 0; l < det_size; ++l){
		if( std::fabs(evals->get(l) - criteria) <= 0.01 ){
			csf_num++;
		}else if( csf_num == 0 ){
			csf_idx++;
		}else{
			continue;
		}
	}
	outfile->Printf("\n  Number of CSFs: %zu", csf_num);

	// Perform the transformation wrt csf eigenvectors
	// CHECK FOR TRIPLET (SHOULD INCLUDE CSF_IDX
	SharedMatrix C_trans(new Matrix("C_trans", det_size, nroot));
	SharedMatrix C(new Matrix("C", det_size, nroot));
	C->gemm('t','n',csf_num,nroot,det_size,1.0,T,det_size,cI,nroot,0.0,nroot);
	C_trans->gemm('n','n',det_size,nroot, csf_num, 1.0,T,det_size,C,nroot,0.0,nroot);

	//Normalize transformed vectors
	for( int n = 0; n < nroot; ++n ){
		double denom = 0.0;
		for( size_t I = 0; I < det_size; ++I){
			denom += C_trans->get(I,n) * C_trans->get(I,n);
		}
		denom = std::sqrt( 1.0/denom );
		C_trans->scale_column( 0, n, denom );
	}
	PQ_spin_evecs_.reset(new Matrix("PQ SPIN EVECS", det_size, nroot));
	PQ_spin_evecs_ = C_trans->clone();

	outfile->Printf("\n  Time spent performing spin transformation: %6.6f", timer.get());
	outfile->Flush();
}

double AdaptiveCI::compute_spin_contamination( std::vector<STLBitsetDeterminant> space, SharedMatrix evecs, int nroot)
{
	auto spins = compute_spin(space, evecs, nroot);
	double spin_contam = 0.0;
	for(int n = 0; n < nroot; ++n){
		spin_contam += spins[n].first.second;
	}
	spin_contam /= static_cast<double>(nroot);
	spin_contam -= (0.25 * (wavefunction_multiplicity_ * wavefunction_multiplicity_ - 1.0));

	return spin_contam;
}

void AdaptiveCI::save_dets_to_file( std::vector<STLBitsetDeterminant> space, SharedMatrix evecs )
{
	//Use for single-root calculations only
	for(size_t I = 0, max = space.size(); I < max; ++I){
		det_list_ << fabs(evecs->get(I,0)) << " ";	
	}
	det_list_ << "\n";
}

}} // EndNamespaces


