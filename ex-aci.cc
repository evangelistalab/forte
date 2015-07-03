#include "lambda-ci.h"

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

#include "ex-aci.h"
#include "cartographer.h"
#include "sparse_ci_solver.h"
#include "string_determinant.h"
#include "bitset_determinant.h"

using namespace std;
using namespace psi;


namespace psi{ namespace libadaptive{

/**
 * Template to store 3-index quantity of any type
 * I use it to store c_I, symmetry label, and ordered labels
 * for determinants
 **/
template <typename a, typename b, typename c>
using oVector = std::vector<std::pair< a,std::pair< b,c >> >;


/**
 * Used for initializing a vector of pairs of any type
 */
template <typename a, typename b>
using pVector = std::vector<std::pair< a,b > >;


inline double clamp(double x, double a, double b)

{
    return x < a ? a : (x > b ? b : x);
}

/**
 * @brief smootherstep
 * @param edge0
 * @param edge1
 * @param x
 * @return
 *
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

EX_ACI::EX_ACI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), options_(options), ints_(ints)
{
    // Copy the wavefunction information
    copy(wfn);

    startup();
    print_info();
}


void EX_ACI::startup()
{

    // Connect the integrals to the determinant class
    StringDeterminant::set_ints(ints_);
    BitsetDeterminant::set_ints(ints_);

    // The number of correlated molecular orbitals
    ncmo_ = ints_->ncmo();
    ncmopi_ = ints_->ncmopi();

    // Number of correlated electrons
    ncel_ = 0;
    for(int h = 0; h < nirrep_; ++h){
        ncel_ += 2*doccpi_[h] + soccpi_[h];
    }
    outfile->Printf("\n  Number of electrons: %d", ncel_);

    // Overwrite the frozen orbitals arrays
    frzcpi_ = ints_->frzcpi();
    frzvpi_ = ints_->frzvpi();

    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    // Create the array with mo symmetry and compute the number of frozen orbitals
    nfrzc_ = 0;
    for (int h = 0; h < nirrep_; ++h){
        nfrzc_ += frzcpi_[h];
        for (int p = 0; p < ncmopi_[h]; ++p){
            mo_symmetry_.push_back(h);
        }
    }

    outfile->Printf("\n  There are %d frozen orbitals.", nfrzc_);

    //Collect information about the reference wavefunction
    wavefunction_multiplicity_ = 1;
    if(options_["MULTIPLICITY"].has_changed()){
        wavefunction_multiplicity_ = options_.get_int("MULTIPLICITY");
    }
    wavefunction_symmetry_ = 0;
    if(options_["ROOT_SYM"].has_changed()){
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }

    //Build the reference determinant with correct symmetry
    reference_determinant_ = StringDeterminant(get_occupation());


    outfile->Printf("\n  The reference determinant is:\n");
    reference_determinant_.print();

    // Read options
    nroot_ = options_.get_int("NROOT");

    tau_p_ = options_.get_double("TAUP");
    tau_q_ = options_.get_double("TAUQ");

    do_smooth_ = options_.get_bool("SMOOTH");
    smooth_threshold_ = options_.get_double("SMOOTH_THRESHOLD");

    spin_tol_ = options_.get_double("SPIN_TOL");
    //set the initial S^2 guess as input multiplicity
    for(int n = 0; n < nroot_; ++n){
        root_spin_vec_.push_back(make_pair((wavefunction_multiplicity_ - 1.0)/2.0, wavefunction_multiplicity_-1.0) );
    }

    perturb_select_ = options_.get_bool("PERTURB_SELECT");
    pq_function_ = options_.get_str("PQ_FUNCTION");
    q_rel_ = options_.get_bool("Q_REL");
    q_reference_ = options_.get_str("Q_REFERENCE");
    ex_alg_ = options_.get_str("EXCITED_ALGORITHM");
    post_root_ = max( nroot_, options_.get_int("POST_ROOT") );
    post_diagonalize_ = options_.get_bool("POST_DIAGONALIZE");



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

EX_ACI::~EX_ACI()
{
}

std::vector<int> EX_ACI::get_occupation()
{

    std::vector<int> occupation(2 * ncmo_,0);

    // Get reference type
    std::string ref_type = options_.get_str("REFERENCE");
    outfile->Printf("\n  Using %s reference.\n", ref_type.c_str());


    //nsym denotes the number of electrons needed to assign symmetry and multiplicity
    int nsym = wavefunction_multiplicity_ - 1;
    int orb_sym = wavefunction_symmetry_;

    if(wavefunction_multiplicity_ == 1){
        nsym = 2;
    }

    // Grab an ordered list of orbital energies, symmetry labels, and Pitzer-indices
    oVector<double,int,int> labeled_orb_en;
    oVector<double,int,int> labeled_orb_en_alfa;
    oVector<double,int,int> labeled_orb_en_beta;

    if(ref_type == "RHF" or ref_type == "RKS" or ref_type == "ROHF"){
       labeled_orb_en = sym_labeled_orbitals("RHF");
    }
    else if(ref_type == "UHF" or ref_type == "UKS"){
        labeled_orb_en_alfa = sym_labeled_orbitals("ALFA");
        labeled_orb_en_beta = sym_labeled_orbitals("BETA");
    }

    //For a restricted reference
    if (ref_type ==  "RHF" or ref_type == "RKS" or ref_type == "ROHF"){

        // Build initial reference determinant from restricted reference
        for(size_t i = 0;  i < nalpha() - nfrzc_; ++i){
            occupation[labeled_orb_en[i].second.second] = 1;
        }
        for(size_t i = 0;  i < nbeta() - nfrzc_; ++i){
            occupation[ncmo_ + labeled_orb_en[i].second.second] = 1;
        }

        //Loop over as many outer-shell electrons needed to get correct symmetry
        for(int k = 1; k <= nsym;){

            bool add = false;

            //remove electron from highest-energy docc
            occupation[labeled_orb_en[nalpha()-k- nfrzc_].second.second] = 0;
            outfile->Printf("\n  Electron removed from %d, out of %d",labeled_orb_en[nalpha() - k- nfrzc_].second.second, ncel_ );

            // Determine proper symmetry for new occupation
            orb_sym = wavefunction_symmetry_;

            if(wavefunction_multiplicity_ == 1){
                orb_sym = direct_sym_product(labeled_orb_en[nalpha() - 1 - nfrzc_].second.first, orb_sym);
            }else{
                for(int i = 1; i <= nsym; ++i){
                    orb_sym = direct_sym_product(labeled_orb_en[nalpha() - i - nfrzc_].second.first, orb_sym);
                }
                orb_sym = direct_sym_product(labeled_orb_en[nalpha() - k - nfrzc_].second.first, orb_sym);
            }


            outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

            //add electron to lowest-energy orbital of proper symmetry
            //Loop from current occupation to max MO until correct orbital is reached
            for(int i = nalpha() - k - nfrzc_; i < ncmo_ - nfrzc_; ++i){
                if(orb_sym == labeled_orb_en[i].second.first and occupation[labeled_orb_en[i].second.second] !=1 ){
                    occupation[labeled_orb_en[i].second.second] = 1;
                    outfile->Printf("\n  Added electron to %d",labeled_orb_en[i].second.second);
                    add = true;
                    break;
                }
            }

            // If a new occupation could not be created, add the electron back and remove a different one
            if(!add){
                occupation[labeled_orb_en[nalpha() - k- nfrzc_].second.second] = 1;
                outfile->Printf("\n  No orbital of %d symmetry available! Putting electron back. \n", orb_sym);
                ++k;
            }
            else{
                break;
            }

        }


    }
    // For an unrestricted reference
    else if(ref_type == "UHF" or ref_type == "UKS"){

        //Make the reference
        //For singlets, this will be "ground-state", closed-shell

        for(size_t i = 0;  i < nalpha()- nfrzc_; ++i){
            occupation[labeled_orb_en_alfa[i].second.second] = 1;
        }
        for(size_t i = 0;  i < nbeta()- nfrzc_; ++i){
            occupation[ncmo_ + labeled_orb_en_beta[i].second.second] = 1;
        }
        if( nalpha() >= nbeta() ){

            for(int k = 1; k < nsym;){

                bool add = false;
                //remove electron from highest-energy docc
                occupation[labeled_orb_en_alfa[nalpha()-k- nfrzc_].second.second] = 0;
                outfile->Printf("\n  Electron removed from %d, out of %d",labeled_orb_en_alfa[nalpha() - k- nfrzc_].second.second, ncel_ );

                // Determine proper symmetry for new occupation
                orb_sym = wavefunction_symmetry_;

                if(wavefunction_multiplicity_ == 1){
                    orb_sym = direct_sym_product(labeled_orb_en_alfa[nalpha() - 1- nfrzc_].second.first, orb_sym);
                }else{
                    for(int i = 1; i <= nsym; ++i){
                        orb_sym = direct_sym_product(labeled_orb_en_alfa[nalpha() - i- nfrzc_].second.first, orb_sym);
                    }
                    orb_sym = direct_sym_product(labeled_orb_en_alfa[nalpha() - k- nfrzc_].second.first, orb_sym);
                }


                outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

                //add electron to lowest-energy orbital of proper symmetry
                for(int i = nalpha() - k- nfrzc_; i < ncmo_; ++i){
                    if(orb_sym == labeled_orb_en_alfa[i].second.first and occupation[labeled_orb_en_alfa[i].second.second] !=1 ){
                        occupation[labeled_orb_en_alfa[i].second.second] = 1;
                        outfile->Printf("\n  Added electron to %d",labeled_orb_en_alfa[i].second.second);
                        add = true;
                        break;
                    }
                }

                // If a new occupation could not be created, add the electron back and remove a different one
                if(!add){
                    occupation[labeled_orb_en_alfa[nalpha() - k- nfrzc_].second.second] = 1;
                    outfile->Printf("\n  No orbital of %d symmetry available! Putting electron back. \n", orb_sym);
                    ++k;
                }
                else{
                    break;
                }

            }
        }

        if( nalpha() < nbeta() ){

            for(int k = 1; k < nsym;){

                bool add = false;
                //remove electron from highest-energy docc
                occupation[labeled_orb_en_beta[nbeta()-k- nfrzc_].second.second] = 0;
                outfile->Printf("\n  Electron removed from %d, out of %d",labeled_orb_en_beta[nbeta() - k- nfrzc_].second.second, ncel_ );

                // Determine proper symmetry for new occupation
                orb_sym = wavefunction_symmetry_;

                if(wavefunction_multiplicity_ == 1){
                    orb_sym = direct_sym_product(labeled_orb_en_beta[nbeta() - 1- nfrzc_].second.first, orb_sym);
                }else{
                    for(int i = 1; i <= nsym; ++i){
                        orb_sym = direct_sym_product(labeled_orb_en_beta[nbeta() - i- nfrzc_].second.first, orb_sym);
                    }
                    orb_sym = direct_sym_product(labeled_orb_en_beta[nbeta() - k- nfrzc_].second.first, orb_sym);
                }


                outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

                //add electron to lowest-energy orbital of proper symmetry
                for(int i = nbeta() - k- nfrzc_; i < ncmo_; ++i){
                    if(orb_sym == labeled_orb_en_beta[i].second.first and occupation[labeled_orb_en_beta[i].second.second] !=1 ){
                        occupation[labeled_orb_en_beta[i].second.second] = 1;
                        outfile->Printf("\n  Added electron to %d",labeled_orb_en_beta[i].second.second);
                        add = true;
                        break;
                    }
                }

                // If a new occupation could not be created, add the electron back and remove a different one
                if(!add){
                    occupation[labeled_orb_en_beta[nbeta() - k- nfrzc_].second.second] = 1;
                    outfile->Printf("\n  No orbital of %d symmetry available! Putting electron back. \n", orb_sym);
                    ++k;
                }
                else{
                    break;
                }

            }
        }

    }

    return occupation;
}

void EX_ACI::print_info()
{
    // Print a summary
    std::vector<std::pair<std::string,int>> calculation_info{
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
        {"Parameter type", perturb_select_ ? "PT" : "Non-PT"},
        {"PQ Function",options_.get_str("PQ_FUNCTION")},
        {"Q Type", q_rel_ ? "Relative Energy" : "Absolute Energy"}};
//    {"Number of electrons",nel},
//    {"Number of correlated alpha electrons",nalpha_},
//    {"Number of correlated beta electrons",nbeta_},
//    {"Number of restricted docc electrons",rdoccpi_.sum()},
//    {"Charge",charge},
//    {"Multiplicity",multiplicity},

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    outfile->Printf("\n  %s",string(52,'-').c_str());
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-40s   %5d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-39s %8.2e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-39s %s",str_dim.first.c_str(),str_dim.second.c_str());
    }
    outfile->Printf("\n  %s",string(52,'-').c_str());
    outfile->Flush();
}


double EX_ACI::compute_energy()
{
    boost::timer t_iamrcisd;
    outfile->Printf("\n\n  Iterative Adaptive CI (v 2.0)");

    SharedMatrix H;
    SharedMatrix P_evecs;
    SharedMatrix PQ_evecs;
    SharedVector P_evals;
    SharedVector PQ_evals;

    // Use the reference determinant as a starting point
    std::vector<bool> alfa_bits = reference_determinant_.get_alfa_bits_vector_bool();
    std::vector<bool> beta_bits = reference_determinant_.get_beta_bits_vector_bool();
    BitsetDeterminant bs_det(alfa_bits,beta_bits);
    P_space_.push_back(bs_det);

    if(alfa_bits != beta_bits){
        BitsetDeterminant ref2 = bs_det;
        ref2.spin_flip();
        P_space_.push_back(ref2);
        outfile->Printf("\n  The initial reference space is: ");
        ref2.print();
        bs_det.print();
    }

    P_space_map_[bs_det] = 1;


    outfile->Printf("\n  The model space contains %zu determinants",P_space_.size());
    outfile->Flush();

    double old_avg_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_avg_energy = 0.0;

    std::vector<std::vector<double> > energy_history;
    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);


    int root;
    int maxcycle = 20;
    for (cycle_ = 0; cycle_ < maxcycle; ++cycle_){
        // Step 1. Diagonalize the Hamiltonian in the P space

        //Set the roots as the lowest possible as initial guess
        int num_ref_roots = std::min(nroot_,int(P_space_.size()));

        outfile->Printf("\n\n  Cycle %3d",cycle_);
        outfile->Printf("\n  %s: %zu determinants","Dimension of the P space",P_space_.size());
        outfile->Flush();

        //save the dimention of the previous iteration
        int PQ_space_init = PQ_space_.size();

        if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSONLIST"){
            sparse_solver.diagonalize_hamiltonian(P_space_,P_evals,P_evecs,nroot_,DavidsonLiuList);
        }else{
            sparse_solver.diagonalize_hamiltonian(P_space_,P_evals,P_evecs,nroot_,DavidsonLiuSparse);
        }

        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots; ++i){
            double abs_energy = P_evals->get(i) + nuclear_repulsion_energy_;
            double exc_energy = pc_hartree2ev * (P_evals->get(i) - P_evals->get(0));
            outfile->Printf("\n    P-space  CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,abs_energy,exc_energy);
        }
        outfile->Printf("\n");
        outfile->Flush();

        for(int n = 0; n < num_ref_roots; ++n){
            outfile->Printf("\n\n  root_spin[%d] : %6.6f", n, root_spin_vec_[n].second);
        }

        // Step 2. Find determinants in the Q space
        find_q_space(num_ref_roots,P_evals,P_evecs);

        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSONLIST"){
            sparse_solver.diagonalize_hamiltonian(PQ_space_,PQ_evals,PQ_evecs,nroot_,DavidsonLiuList);
        }else{
            sparse_solver.diagonalize_hamiltonian(PQ_space_,PQ_evals,PQ_evecs,nroot_,DavidsonLiuSparse);
        }


        // Print the energy
        outfile->Printf("\n");
        for ( int i = 0; i < num_ref_roots; ++i){
            double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_;
            double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
            outfile->Printf("\n    PQ-space CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,abs_energy,exc_energy);
            outfile->Printf("\n    PQ-space CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV",i + 1,abs_energy + multistate_pt2_energy_correction_[i],
                            exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
        }
        outfile->Printf("\n");
        outfile->Flush();

        //get final dimention of P space
        int PQ_space_final = PQ_space_.size();
        outfile->Printf("\n PQ space dimention difference (current - previous) : %d \n", PQ_space_final - PQ_space_init);

        // Step 4. Check convergence and break if needed 
        if(check_convergence(energy_history,PQ_evals)){
            break;
        }

        //Step 5. Check if the procedure is stuck
        bool stuck = check_stuck(energy_history,PQ_evals);
        if(stuck){
            outfile->Printf("\n  The procedure is stuck! Printing final energy (but be careful).");
            break;
        }

        // Step 6. Prune the P + Q space to get an updated P space
        prune_q_space(PQ_space_,P_space_,P_space_map_,PQ_evecs,nroot_);

        // Print information about the wave function
        print_wfn(PQ_space_,PQ_evecs,nroot_);

    }//end cycle

    // Do Hamiltonian smoothing
    if (do_smooth_){
        smooth_hamiltonian(P_space_,P_evals,P_evecs,nroot_);
    }


    //Re-diagonalize H, solving for more roots
    if(post_diagonalize_){
        root = nroot_;
        sparse_solver.diagonalize_hamiltonian(PQ_space_,PQ_evals,PQ_evecs,post_root_, DavidsonLiuSparse);
        outfile->Printf(" \n  Re-diagonalizing the Hamiltonian with %zu roots.\n", post_root_);
        outfile->Printf(" \n  WARNING: EPT2 is meaningless for roots %zu and higher. I'm not even printing them.", root+1);
        nroot_ = post_root_;
    }

    outfile->Printf("\n\n  ==> Post-Iterations <==\n");

    outfile->Printf("\n  Printing Wavefunction Information:");
    print_wfn(PQ_space_,PQ_evecs,nroot_);
    outfile->Printf("\n\n     Order       # of Dets        Total |c^2|  ");
    outfile->Printf("\n   ---------   -------------   -----------------  ");

    wfn_analyzer(PQ_space_, PQ_evecs, nroot_);
    for (int i = 0; i < nroot_; ++i){
        double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_;
        double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
        outfile->Printf("\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,abs_energy,exc_energy);
        if(post_diagonalize_ == false){
            outfile->Printf("\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV",i + 1,abs_energy + multistate_pt2_energy_correction_[i],
                exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
        }else if(post_diagonalize_ and i < root){
            outfile->Printf("\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV",i + 1,abs_energy + multistate_pt2_energy_correction_[i],
                exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
        }
    }


    outfile->Printf("\n\n  %s: %f s","Adaptive-CI (bitset) ran in ",t_iamrcisd.elapsed());
    outfile->Printf("\n\n  %s: %d","Saving information for root",options_.get_int("ROOT") + 1);
    outfile->Flush();

    double root_energy = PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_;
    double root_energy_pt2 = root_energy + multistate_pt2_energy_correction_[options_.get_int("ROOT")];
    Process::environment.globals["CURRENT ENERGY"] = root_energy;
    Process::environment.globals["EX-ACI ENERGY"] = root_energy;
    Process::environment.globals["EX-ACI+PT2 ENERGY"] = root_energy_pt2;

    return PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_;

}


void EX_ACI::find_q_space(int nroot, SharedVector evals,SharedMatrix evecs)
{
    // Find the SD space out of the reference
    std::vector<BitsetDeterminant> sd_dets_vec;
    std::map<BitsetDeterminant,int> new_dets_map;

    boost::timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    std::map<BitsetDeterminant,std::vector<double> > V_hash;

    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I){
        auto& det = P_space_[I];
        generate_excited_determinants(nroot,I,evecs,det,V_hash);
    }
    outfile->Printf("\n  %s: %zu determinants","Dimension of the SD space",V_hash.size());
    outfile->Printf("\n  %s: %f s\n","Time spent building the model space",t_ms_build.elapsed());
    outfile->Flush();

    // This will contain all the determinants
    PQ_space_.clear();

    // Add the P-space determinants and zero the hash
    for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J){
        PQ_space_.push_back(P_space_[J]);
        V_hash.erase(P_space_[J]);
    }

    boost::timer t_ms_screen;

    using bsmap_it = std::map<BitsetDeterminant,std::vector<double> >::const_iterator;
    pVector<double,double> C1(nroot_,make_pair(0.0,0.0));
    pVector<double,double> E2(nroot_,make_pair(0.0,0.0));
    std::vector<double> V(nroot_, 0.0);
    BitsetDeterminant det;
    pVector<double,BitsetDeterminant> sorted_dets;
    std::vector<double> ept2(nroot_,0.0);
    double criteria;
    print_warning_ = false;

    // Check the coupling between the reference and the SD space
    for(const auto& it : V_hash){
        double EI = it.first.energy();
        //Loop over roots
        //The tau_q parameter type is chosen here ( keyword bool "perturb_select" )
        for (int n = 0; n < nroot; ++n){
            det = it.first;
            V[n] = it.second[n];

            double C1_I = perturb_select_ ? -V[n] / (EI - evals->get(n)) :
                                            ( ((EI - evals->get(n))/2.0) - sqrt( std::pow(((EI - evals->get(n))/2.0),2.0) + std::pow(V[n],2.0)) ) / V[n];
            double E2_I = perturb_select_ ? -V[n] * V[n] / (EI - evals->get(n)) :
                                            ((EI - evals->get(n))/2.0) - sqrt( std::pow(((EI - evals->get(n))/2.0),2.0) + std::pow(V[n],2.0) );

            C1[n] = make_pair(std::fabs(C1_I),C1_I);
            E2[n] = make_pair(std::fabs(E2_I),E2_I);

        }
        //make q space in a number of ways with C1 and E1 as input, produces PQ_space
        if(ex_alg_ == "STATE_AVERAGE" and nroot_ != 1){
            criteria = average_q_values(nroot, C1, E2);
        }
        else if(ex_alg_ == "ROOT_SELECT"){
            criteria = root_select(nroot, C1, E2);
        }
        else if(nroot_ == 1){
            criteria = root_select(nroot, C1, E2);
        }

        if(aimed_selection_){
            sorted_dets.push_back(std::make_pair(criteria,it.first));
        }else{
            if(std::fabs(criteria) > tau_q_){
                PQ_space_.push_back(it.first);
            }else{
                for (int n = 0; n < nroot; ++n){
                    ept2[n] += E2[n].second;
                }
            }
        }
    }// end loop over determinants

    if(ex_alg_ == "STATE_AVERAGE" and print_warning_){
        outfile->Printf("\n  WARNING: There are not enough roots with the correct S^2 to compute dE2! You should increase nroot.");
        outfile->Printf("\n  Setting q_rel = false for this iteration.\n");
    }


    if (aimed_selection_){
        std::sort(sorted_dets.begin(),sorted_dets.end());
        double sum = 0.0;
        double E2_I = 0.0;
        for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I){
            const BitsetDeterminant& det = sorted_dets[I].second;
            if (sum + sorted_dets[I].first < tau_q_){
                sum += sorted_dets[I].first;
                double EI = det.energy();
                const auto& V_vec = V_hash[det];
                for (int n = 0; n < nroot; ++n){
                    double V = V_vec[n];
                    E2_I = perturb_select_ ? -V * V / (EI - evals->get(n)) :
                                                    ((EI - evals->get(n))/2.0) - sqrt( std::pow(((EI - evals->get(n))/2.0),2.0) + std::pow(V,2.0) );
                    ept2[n] += E2_I;
                }
            }else{
                PQ_space_.push_back(sorted_dets[I].second);
            }
        }
    }


    multistate_pt2_energy_correction_ = ept2;

    outfile->Printf("\n  %s: %zu determinants","Dimension of the P + Q space",PQ_space_.size());
    outfile->Printf("\n  %s: %f s","Time spent screening the model space",t_ms_screen.elapsed());
    outfile->Flush();
}

double EX_ACI::average_q_values(int nroot, pVector<double,double> C1, pVector<double,double> E2)
{
    pVector<double,double> C_s;
    pVector<double,double> E_s;

    C_s.clear();
    E_s.clear();
    //If the spin is correct, use the C1 and E2 values
    for(int n = 0; n < nroot; ++n){
        if(std::fabs(root_spin_vec_[n].second - wavefunction_multiplicity_ + 1.0) < spin_tol_ ){
            C_s.push_back(make_pair(C1[n].first,C1[n].second));
            E_s.push_back(make_pair(E2[n].first,E2[n].second));
        }
    }
    int dim = C_s.size();

    if(dim == 0){
        throw PSIEXCEPTION(" There are no roots with correct S^2 value! ");
    }


    //f_E2 and f_C1 will store the selected function of the chosen q-criteria
    std::pair<double,double> f_C1;
    std::pair<double,double> f_E2;

    //Make vector of pairs for ∆e_n,0
    pVector<double,double> dE2(dim,make_pair(0.0,0.0));

    //Compute a determinant's effect on ground state or adjacent state transition
    q_rel_ = options_.get_bool("Q_REL");
    print_warning_ = false;

    if(q_rel_ == true and dim > 1){
        if( q_reference_ == "GS"){
            for(int n = 0; n < dim; ++n){
                dE2[n] = make_pair(std::fabs(E_s[n].first - E_s[0].first),E_s[n].second - E_s[0].second );
            }
        }
        if( q_reference_ == "ADJACENT"){
            for(int n = 1; n < dim; ++n){
                dE2[n] = make_pair(std::fabs(E_s[n].first - E_s[n-1].first),E_s[n].second - E_s[n-1].second );
            }
        }
    }else if(q_rel_ == true and dim == 1){
        print_warning_ = true;
        q_rel_ = false;
    }

    //Choose the function of couplings for each root.
    //If nroot = 1, choose the max
    if(pq_function_ == "MAX" or dim == 1){
        f_C1 = *std::max_element(C_s.begin(),C_s.end());
        f_E2 = q_rel_ and (dim!=1) ? *std::max_element(dE2.begin(),dE2.end()) :
                                       *std::max_element(E_s.begin(),E_s.end());
    }
    else if(pq_function_ == "AVERAGE"){
        double C1_average = 0.0;
        double E2_average = 0.0;
        double dE2_average = 0.0;
        double dim_inv = 1.0 / dim;
        for(int n = 0; n < dim; ++n){
            C1_average += C_s[n].first * dim_inv;
            E2_average += E_s[n].first * dim_inv;
        }
        if(q_rel_){
            double inv_d = 1.0 / (dim - 1.0);
            for(int n = 1; n < dim; ++n){
                dE2_average += dE2[n].first * inv_d;
            }
        }
        f_C1 = make_pair(C1_average, 0);
        f_E2 = q_rel_ ? make_pair(dE2_average,0) : make_pair(E2_average, 0);
    }
    else{
        throw PSIEXCEPTION(options_.get_str("PQ_FUNCTION") + " is not a valid option");
    }

    double select_value = 0.0;
    if (aimed_selection_){
        select_value = energy_selection_ ? f_E2.first : std::pow(f_C1.first,2.0);
    }else{
        select_value = energy_selection_ ? f_E2.first : f_C1.first;
    }
    return select_value;

}

double EX_ACI::root_select(int nroot,pVector<double,double> C1, pVector<double,double> E2)
{
    double select_value;
    ref_root_ = options_.get_int("REF_ROOT");

    if(ref_root_ +1 > nroot_){
        throw PSIEXCEPTION("Your selection is not a valid reference option. Check REF_ROOT in options.");
    }

    if(nroot == 1){
        ref_root_ = 0;
    }

    if(aimed_selection_){
        select_value = energy_selection_ ? E2[ref_root_].first : std::pow(C1[ref_root_].first,2.0);
    }else{
        select_value = energy_selection_ ? E2[ref_root_].first : C1[ref_root_].first;
    }

    return select_value;
}


void EX_ACI::find_q_space_single_root(int nroot,SharedVector evals,SharedMatrix evecs)
{
    // Find the SD space out of the reference
    std::vector<BitsetDeterminant> sd_dets_vec;
    std::map<BitsetDeterminant,int> new_dets_map;

    boost::timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    std::map<BitsetDeterminant,double> V_map;

    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I){
        BitsetDeterminant& det = P_space_[I];
        generate_excited_determinants_single_root(nroot,I,evecs,det,V_map);
    }
    outfile->Printf("\n  %s: %zu determinants","Dimension of the SD space",V_map.size());
    outfile->Printf("\n  %s: %f s\n","Time spent building the model space",t_ms_build.elapsed());
    outfile->Flush();

    // This will contain all the determinants
    PQ_space_.clear();

    // Add the P-space determinants to PQ and remove them from the hash
    for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J){
        PQ_space_.push_back(P_space_[J]);
        V_map.erase(P_space_[J]);
    }

    boost::timer t_ms_screen;

    using bsmap_it =  std::map<BitsetDeterminant,std::vector<double> >::const_iterator;
    std::vector<std::pair<double,double> > C1(nroot_,make_pair(0.0,0.0));
    std::vector<std::pair<double,double> > E2(nroot_,make_pair(0.0,0.0));
    std::vector<double> ept2(nroot_,0.0);

    std::vector<std::pair<double,BitsetDeterminant>> sorted_dets;

    // Check the coupling between the reference and the SD space
    for (const auto& det_V : V_map){
        double EI = det_V.first.energy();
        for (int n = 0; n < nroot; ++n){
            double V = det_V.second;
            double C1_I = -V / (EI - evals->get(n));
            double E2_I = -V * V / (EI - evals->get(n));

            C1[n] = make_pair(std::fabs(C1_I),C1_I);
            E2[n] = make_pair(std::fabs(E2_I),E2_I);
        }

        std::pair<double,double> max_C1 = *std::max_element(C1.begin(),C1.end());
        std::pair<double,double> max_E2 = *std::max_element(E2.begin(),E2.end());

        if (aimed_selection_){
            double aimed_value = energy_selection_ ? max_E2.first : std::pow(max_C1.first,2.0);
            sorted_dets.push_back(std::make_pair(aimed_value,det_V.first));
        }else{
            double select_value = energy_selection_ ? max_E2.first : max_C1.first;
            if (std::fabs(select_value) > tau_q_){
                PQ_space_.push_back(det_V.first);
            }else{
                for (int n = 0; n < nroot; ++n){
                    ept2[n] += E2[n].second;
                }
            }
        }
    }

    if (aimed_selection_){
        // Sort the CI coefficients in ascending order
        std::sort(sorted_dets.begin(),sorted_dets.end());

        double sum = 0.0;
        for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I){
            const BitsetDeterminant& det = sorted_dets[I].second;
            if (sum + sorted_dets[I].first < tau_q_){
                sum += sorted_dets[I].first;
                double EI = det.energy();
                const double V = V_map[det];
                for (int n = 0; n < nroot; ++n){
                    double E2_I = -V * V / (EI - evals->get(n));
                    ept2[n] += E2_I;
                }
            }else{
                PQ_space_.push_back(sorted_dets[I].second);
            }
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    outfile->Printf("\n  %s: %zu determinants","Dimension of the P + Q space",PQ_space_.size());
    outfile->Printf("\n  %s: %f s","Time spent screening the model space",t_ms_screen.elapsed());
    outfile->Flush();
}


void EX_ACI::generate_excited_determinants_single_root(int nroot,int I,SharedMatrix evecs,BitsetDeterminant& det,std::map<BitsetDeterminant,double>& V_hash)
{
    std::vector<int> aocc = det.get_alfa_occ();
    std::vector<int> bocc = det.get_beta_occ();
    std::vector<int> avir = det.get_alfa_vir();
    std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    int n = 0;
    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a){
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0){
                BitsetDeterminant new_det(det);
                new_det.set_alfa_bit(ii,false);
                new_det.set_alfa_bit(aa,true);
                double HIJ = det.slater_rules(new_det);
                V_hash[new_det] += HIJ * evecs->get(I,n);
            }
        }
    }

    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a){
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == 0){
                BitsetDeterminant new_det(det);
                new_det.set_beta_bit(ii,false);
                new_det.set_beta_bit(aa,true);
                double HIJ = det.slater_rules(new_det);
                V_hash[new_det] += HIJ * evecs->get(I,n);
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
                        BitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii,false);
                        new_det.set_alfa_bit(jj,false);
                        new_det.set_alfa_bit(aa,true);
                        new_det.set_alfa_bit(bb,true);

                        double HIJ = ints_->aptei_aa(ii,jj,aa,bb);

                        // grap the alpha bits of both determinants
                        const boost::dynamic_bitset<>& Ia = det.alfa_bits();
                        const boost::dynamic_bitset<>& Ja = new_det.alfa_bits();

                        // compute the sign of the matrix element
                        HIJ *= BitsetDeterminant::SlaterSign(Ia,ii) * BitsetDeterminant::SlaterSign(Ia,jj) * BitsetDeterminant::SlaterSign(Ja,aa) * BitsetDeterminant::SlaterSign(Ja,bb);

                        V_hash[new_det] += HIJ * evecs->get(I,n);
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
                        BitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii,false);
                        new_det.set_beta_bit(jj,false);
                        new_det.set_alfa_bit(aa,true);
                        new_det.set_beta_bit(bb,true);

                        double HIJ = ints_->aptei_ab(ii,jj,aa,bb);

                        // grap the alpha bits of both determinants
                        const boost::dynamic_bitset<>& Ia = det.alfa_bits();
                        const boost::dynamic_bitset<>& Ib = det.beta_bits();
                        const boost::dynamic_bitset<>& Ja = new_det.alfa_bits();
                        const boost::dynamic_bitset<>& Jb = new_det.beta_bits();

                        // compute the sign of the matrix element
                        HIJ *= BitsetDeterminant::SlaterSign(Ia,ii) * BitsetDeterminant::SlaterSign(Ib,jj) * BitsetDeterminant::SlaterSign(Ja,aa) * BitsetDeterminant::SlaterSign(Jb,bb);

                        V_hash[new_det] += HIJ * evecs->get(I,n);
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
                        BitsetDeterminant new_det(det);
                        new_det.set_beta_bit(ii,false);
                        new_det.set_beta_bit(jj,false);
                        new_det.set_beta_bit(aa,true);
                        new_det.set_beta_bit(bb,true);

                        double HIJ = ints_->aptei_bb(ii,jj,aa,bb);

                        // grap the alpha bits of both determinants
                        const boost::dynamic_bitset<>& Ib = det.beta_bits();
                        const boost::dynamic_bitset<>& Jb = new_det.beta_bits();

                        // compute the sign of the matrix element
                        HIJ *= BitsetDeterminant::SlaterSign(Ib,ii) * BitsetDeterminant::SlaterSign(Ib,jj) * BitsetDeterminant::SlaterSign(Jb,aa) * BitsetDeterminant::SlaterSign(Jb,bb);

                        V_hash[new_det] += HIJ * evecs->get(I,n);
                    }
                }
            }
        }
    }
}

void EX_ACI::generate_excited_determinants(int nroot,int I,SharedMatrix evecs,BitsetDeterminant& det,std::map<BitsetDeterminant,std::vector<double>>& V_hash)
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
                BitsetDeterminant new_det(det);
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
                BitsetDeterminant new_det(det);
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
                        BitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii,false);
                        new_det.set_alfa_bit(jj,false);
                        new_det.set_alfa_bit(aa,true);
                        new_det.set_alfa_bit(bb,true);
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

    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j){
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b){
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0){
                        BitsetDeterminant new_det(det);
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
    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j){
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a){
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b){
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0){
                        BitsetDeterminant new_det(det);
                        new_det.set_beta_bit(ii,false);
                        new_det.set_beta_bit(jj,false);
                        new_det.set_beta_bit(aa,true);
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
}

void EX_ACI::generate_pair_excited_determinants(int nroot,int I,SharedMatrix evecs,BitsetDeterminant& det,std::map<BitsetDeterminant,std::vector<double>>& V_hash)
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
                BitsetDeterminant new_det(det);
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
                BitsetDeterminant new_det(det);
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
                    BitsetDeterminant new_det(det);
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

bool EX_ACI::check_convergence(std::vector<std::vector<double>>& energy_history,SharedVector evals)
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

    //Only average over roots with correct S^2
    int denom = 0;
    for (int n = 0; n < nroot; ++ n){
        double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
        new_energies.push_back(state_n_energy);
        if(std::fabs(root_spin_vec_[n].second - wavefunction_multiplicity_ + 1.0) < spin_tol_ ){
            new_avg_energy += state_n_energy;
            old_avg_energy += old_energies[n];
            ++denom;
        }
    }

    if(denom == 0){
        for(int n = 0; n < nroot; ++n){
            outfile->Printf("\n  S^2, S for root %d : %6.12f  %6.12f", n, root_spin_vec_[n].second,root_spin_vec_[n].first);
        }
        outfile->Printf("\n");
        throw PSIEXCEPTION("  No roots have the correct S^2! ");
    }

    old_avg_energy /= static_cast<double>(denom);
    new_avg_energy /= static_cast<double>(denom);

    energy_history.push_back(new_energies);

    // Check for convergence
    return (std::fabs(new_avg_energy - old_avg_energy) < options_.get_double("E_CONVERGENCE"));
}

bool EX_ACI::check_stuck(std::vector<std::vector<double>>& energy_history, SharedVector evals)
{
    int nroot = evals->dim();
    if(cycle_ < 3){
        return false;
    }
    else{
        std::vector<double> av_energies;
        av_energies.clear();

        for(int i = 0; i < cycle_; ++i){
            double energy = 0.0;
            for(int n = 0; n < nroot; ++n){
                energy += energy_history[i][n] / static_cast<double>(nroot);
            }
            av_energies.push_back(energy);
        }

        if( std::fabs(av_energies[cycle_-1]- av_energies[cycle_ - 3]) < options_.get_double("E_CONVERGENCE")
                and std::fabs(av_energies[cycle_ - 2]- av_energies[cycle_ - 4]) < options_.get_double("E_CONVERGENCE") ){
            return true;
        }else{ return false;}
    }
}

void EX_ACI::prune_q_space(std::vector<BitsetDeterminant>& large_space,std::vector<BitsetDeterminant>& pruned_space,
                               std::map<BitsetDeterminant,int>& pruned_space_map,SharedMatrix evecs,int nroot)
{
    // Select the new reference space using the sorted CI coefficients
    pruned_space.clear();
    pruned_space_map.clear();

    // Create a vector that stores the absolute value of the CI coefficients
    // Use a function of the CI coefficients for each root as the criteria
    // This function will be the same one used when selecting the PQ space
    pVector<double,size_t> dm_det_list;
    for (size_t I = 0; I < large_space.size(); ++I){
        double criteria = 0.0;
        int dim = 0;
        for (int n = 0; n < nroot; ++n){
            if(pq_function_ == "MAX" and std::fabs(root_spin_vec_[n].second - wavefunction_multiplicity_ + 1.0) < spin_tol_){
                criteria = std::max(criteria, std::fabs(evecs->get(I,n)));
                dim = 1;
            }
            else if(pq_function_ == "AVERAGE" and std::fabs(root_spin_vec_[n].second - wavefunction_multiplicity_ + 1.0) < spin_tol_){
                criteria += std::fabs(evecs->get(I,n));
                dim++;
            }

        }
        criteria /= static_cast<double>(dim);
        dm_det_list.push_back(make_pair(criteria,I));
    }

    // Decide which determinants will go in pruned_space
    // Include all determinants such that
    // sum_I |C_I|^2 < tau_p, where the sum runs over all the excluded determinants
    if (aimed_selection_){
        // Sort the CI coefficients in ascending order
        outfile->Printf("  AIMED SELECTION \n");
        std::sort(dm_det_list.begin(),dm_det_list.end());

        double sum = 0.0;
        for (size_t I = 0; I < large_space.size(); ++I){
            double dsum = std::pow(dm_det_list[I].first,2.0);
            if (sum + dsum < tau_p_){
                sum += dsum;
            }else{
                pruned_space.push_back(large_space[dm_det_list[I].second]);
                pruned_space_map[large_space[dm_det_list[I].second]] = 1;
            }
        }
    }
    // Include all determinants such that |C_I| > tau_p
    else{

        std::sort(dm_det_list.begin(),dm_det_list.end());
        for (size_t I = 0 ; I < large_space.size(); ++I){
            if (dm_det_list[I].first > tau_p_){
                pruned_space.push_back(large_space[dm_det_list[I].second]);
                pruned_space_map[large_space[dm_det_list[I].second]] = 1;
            }
        }
    }
}


void EX_ACI::smooth_hamiltonian(std::vector<BitsetDeterminant>& space,SharedVector evals,SharedMatrix evecs,int nroot)
{
    size_t ndets = space.size();

    SharedMatrix H(new Matrix("H-smooth",ndets,ndets));

    SharedMatrix F(new Matrix("F-smooth",ndets,ndets));

    // Build the smoothed Hamiltonian
    for (int I = 0; I < ndets; ++I){
        for (int J = 0; J < ndets; ++J){
            double CI = evecs->get(I,0);
            double CJ = evecs->get(J,0);
            double HIJ = space[I].slater_rules(space[J]);
            double factorI = smootherstep(tau_p_ * tau_p_,smooth_threshold_,CI * CI);
            double factorJ = smootherstep(tau_p_ * tau_p_,smooth_threshold_,CJ * CJ);
            if (I != J){
                HIJ *= factorI * factorJ;
                F->set(I,J,factorI * factorJ);
            }
            H->set(I,J,HIJ);
        }
    }

    evecs->print();
    H->print();
    F->print();

    SharedMatrix evecs_s(new Matrix("C-smooth",ndets,ndets));
    SharedVector evals_s(new Vector("lambda-smooth",ndets));

    H->diagonalize(evecs_s,evals_s);

    outfile->Printf("\n  * sAdaptive-CI Energy Root %3d        = %.12f Eh",1,evals_s->get(0) + nuclear_repulsion_energy_);
}

pVector<std::pair<double,double>,std::pair<size_t,double> > EX_ACI::compute_spin(std::vector<BitsetDeterminant> space,
                                                                                 SharedMatrix evecs,
                                                                                 int nroot,
                                                                                 pVector<double,size_t> det_weight)
{
    double norm;
    double S2;
    double S;
    pVector<std::pair<double,double>, std::pair<size_t,double> > spin_vec;

    for(int n = 0; n < nroot; ++n){
        // Compute the expectation value of the spin
        size_t max_sample = 1000;
        size_t max_I = 0;
        double sum_weight = 0.0;

        //Don't require the determinants to be pre-sorted
        std::sort(det_weight.begin(),det_weight.end());
        std::reverse(det_weight.begin(),det_weight.end());

        const double wfn_threshold = 0.95;
        for (size_t I = 0; I < space.size(); ++I){
            if ((sum_weight < wfn_threshold) and (I < max_sample)) {
                sum_weight += std::pow(det_weight[I].first,2.0);
                max_I++;
            }else if (std::fabs(det_weight[I].first - det_weight[I-1].first) < 1.0e-6){
                // Special case, if there are several equivalent determinants
                sum_weight += std::pow(det_weight[I].first,2.0);
                max_I++;
            }else{
                break;
            }
        }

        S2 = 0.0;
        norm = 0.0;
        for (int sI = 0; sI < max_I; ++sI){
            size_t I = det_weight[sI].second;
            for (int sJ = 0; sJ < max_I; ++sJ){
                size_t J = det_weight[sJ].second;
                if (std::fabs(evecs->get(I,n) * evecs->get(J,n)) > 1.0e-12){
                    const double S2IJ = space[I].spin2(space[J]);
                    S2 += evecs->get(I,n) * evecs->get(J,n) * S2IJ;
                }
            }
            norm += std::pow(evecs->get(I,n),2.0);
        }

        S2 /= norm;
        S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        spin_vec.push_back( make_pair(make_pair(S,S2),make_pair(max_I,sum_weight)) );

    }
   return spin_vec;
}

void EX_ACI::print_wfn(std::vector<BitsetDeterminant> space,SharedMatrix evecs,int nroot)
{
    pVector<double,size_t> det_weight;
    pVector<std::pair<double,double>, std::pair<size_t,double> > spins;
    double sum_weight;
    double S2;
    double S;
    size_t max_I;

    std::vector<string> s2_labels({"singlet","doublet","triplet","quartet","quintet","sextet","septet","octet","nonet","decaet"});
    string state_label;

    for (int n = 0; n < nroot; ++n){
        det_weight.clear();
        outfile->Printf("\n\n  Most important contributions to root %3d:",n);

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

        spins = compute_spin(space,evecs,nroot,det_weight);
        S = spins[n].first.first;
        S2 = spins[n].first.second;
        max_I = spins[n].second.first;
        sum_weight = spins[n].second.second;

        state_label = s2_labels[std::round(S * 2.0)];
        outfile->Printf("\n\n  Spin State for root %zu: S^2 = %5.3f, S = %5.3f, %s (from %zu determinants,%.2f\%)",n,S2,S,state_label.c_str(),max_I,100.0 * sum_weight);
        root_spin_vec_.clear();
        root_spin_vec_[n] = make_pair(S, S2);
    }
    outfile->Flush();
}

int EX_ACI::direct_sym_product(int sym1, int sym2)
{

    /*Create a matrix for direct products of Abelian symmetry groups
    *
    * This matrix is 8x8, but it works for molecules of both D2H and
    * C2V symmetry
    *
    * Due to properties of groups, direct_sym_product(a,b) solves both
    *
    * a (x) b = ?
    * and
    * a (x) ? = b
    */

    boost::shared_ptr<Matrix> dp(new Matrix("dp",8,8));

    for(int p = 0; p < 2; ++p){
        for(int q = 0; q < 2; ++q){
            if(p != q){
                dp->set(p,q, 1);
            }
        }
    }

    for(int p = 2; p < 4; ++p){
        for(int q = 0; q < 2; ++q){
            dp->set(p,q, dp->get(p-2,q) + 2);
            dp->set(q,p, dp->get(p-2,q) + 2);
        }
    }

    for(int p = 2; p < 4; ++p){
        for(int q = 2; q < 4; ++q){
            dp->set(p,q, dp->get(p-2,q-2));
        }
    }

    for(int p = 4; p < 8; ++p){
        for(int q = 0; q < 4; ++q){
            dp->set(p,q, dp->get(p-4,q)+4);
            dp->set(q,p, dp->get(p-4,q)+4);
        }
    }

    for(int p = 4; p < 8; ++p){
        for(int q = 4; q < 8; ++q){
            dp->set(p,q, dp->get(p-4,q-4));
        }
    }

    return dp->get(sym1, sym2);
}

void EX_ACI::wfn_analyzer(std::vector<BitsetDeterminant> det_space, SharedMatrix evecs,int nroot)
{
    for(int n = 0; n < nroot; ++n){
        pVector<size_t,double> excitation_counter( 1 + (1 + cycle_)*2 );
        pVector<double,size_t> det_weight;
        for (size_t I = 0; I < det_space.size(); ++I){
            det_weight.push_back(std::make_pair(std::fabs(evecs->get(I,n)),I));
        }

        std::sort(det_weight.begin(),det_weight.end());
        std::reverse(det_weight.begin(),det_weight.end());

        BitsetDeterminant ref;
        ref.copy( det_space[det_weight[0].second] );

        auto alfa_bits = ref.alfa_bits();
        auto beta_bits = ref.beta_bits();

        for(size_t I = 0; I < det_space.size(); ++I ){
            int ndiff = 0;
            auto ex_alfa_bits = det_space[det_weight[I].second].alfa_bits();
            auto ex_beta_bits = det_space[det_weight[I].second].beta_bits();

            //Compute number of differences in both alpha and beta strings wrt ref
            for(int a = 0; a < alfa_bits.size(); ++a){
                if(alfa_bits[a] != ex_alfa_bits[a]){
                    ++ndiff;
                }
                if(beta_bits[a] != ex_beta_bits[a]){
                    ++ndiff;
                }
            }
            ndiff /= 2;
            excitation_counter[ndiff] = std::make_pair(excitation_counter[ndiff].first + 1,
                                                       excitation_counter[ndiff].second + std::pow(det_weight[I].first,2.0));

        }
        int order = 0;
        size_t det = 0;
        for(auto i: excitation_counter){
            outfile->Printf("\n      %2d          %4zu           %.11f", order, i.first, i.second);
            det += i.first;
            if(det == det_space.size()) break;
            ++order;
        }
        outfile->Printf("\n\n  Highest-order exitation searched:     %zu  \n", excitation_counter.size()-1);

    }

}

oVector<double,int,int> EX_ACI::sym_labeled_orbitals(std::string type)
{
    oVector<double,int,int> labeled_orb;

    if(type == "RHF" or type == "ROHF" or type == "ALFA"){

        //Create a vector of orbital energy and index pairs (Pitzer ordered)
        pVector<double,int> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; h++) {
            for (int a = 0; a < ncmopi_[h]; a++){
                orb_e.push_back(make_pair(epsilon_a_->get(h,a+frzcpi_[h]), a+cumidx));
            }
            cumidx += ncmopi_[h];
        }

        //Create a vector that stores the orbital energy, symmetry, and Pitzer-ordered index
        for (int a = 0; a < ncmo_; ++a){
            labeled_orb.push_back( make_pair(orb_e[a].first, make_pair(mo_symmetry_[a], orb_e[a].second) ) );
        }

        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }

    if(type == "BETA"){
        //Create a vector of orbital energy and index pairs (Pitzer ordered)
        pVector<double,int> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; h++) {
            for (int a = 0; a < ncmopi_[h]- frzcpi_[h]; a++){
                orb_e.push_back(make_pair(epsilon_b_->get(h,a+frzcpi_[h]), a+cumidx));
            }
            cumidx += (ncmopi_[h]);
        }


        //Create a vector that stores the orbital energy, symmetry, and Pitzer-ordered index
        for (int a = 0; a < ncmo_ ; ++a){
            labeled_orb.push_back( make_pair(orb_e[a].first, make_pair(mo_symmetry_[a], orb_e[a].second) ) );
        }

        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());

    }

//    for(int i = 0; i < ncmo_; ++i){
//        outfile->Printf("\n %f    %d    %d", labeled_orb[i].first, labeled_orb[i].second.first, labeled_orb[i].second.second);
//    }


    return labeled_orb;
}

void print_timing_info()
{

}

}} // EndNamespaces


