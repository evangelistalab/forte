#include "lambda-ci.h"

//#include <cstdio>
//#include <cstdlib>
//#include <cstring>
#include <cmath>
#include <functional>
#include <algorithm>

#include <boost/unordered_map.hpp>
#include <boost/timer.hpp>
#include <boost/format.hpp>

#include <libciomr/libciomr.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include <libqt/qt.h>
#include <libmints/molecule.h>

#include "adaptive-ci.h"
#include "cartographer.h"
#include "sparse_ci_solver.h"
#include "string_determinant.h"
#include "bitset_determinant.h"

#define BIGNUM 1E100
#define MAXIT 500

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

AdaptiveCI::AdaptiveCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), options_(options), ints_(ints)
{
    // Copy the wavefunction information
    copy(wfn);

    startup();
    print_info();
}


void AdaptiveCI::startup()
{
    // Connect the integrals to the determinant class
    StringDeterminant::set_ints(ints_);
    BitsetDeterminant::set_ints(ints_);

    // The number of correlated molecular orbitals
    ncmo_ = ints_->ncmo();
    ncmopi_ = ints_->ncmopi();

    // Overwrite the frozen orbitals arrays
    frzcpi_ = ints_->frzcpi();
    frzvpi_ = ints_->frzvpi();

    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    // Create the array with mo symmetry
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < ncmopi_[h]; ++p){
            mo_symmetry_.push_back(h);
        }
    }

    wavefunction_symmetry_ = 0;
    if(options_["ROOT_SYM"].has_changed()){
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }

    // Build the reference determinant and compute its energy
    std::vector<int> occupation(2 * ncmo_,0);
    int cumidx = 0;
    for (int h = 0; h < nirrep_; ++h){
        for (int i = 0; i < doccpi_[h] - frzcpi_[h]; ++i){
            occupation[i + cumidx] = 1;
            occupation[ncmo_ + i + cumidx] = 1;
        }
        for (int i = 0; i < soccpi_[h]; ++i){
            occupation[i + cumidx] = 1;
        }
        cumidx += ncmopi_[h];
    }
    reference_determinant_ = StringDeterminant(occupation);

    outfile->Printf("\n  The reference determinant is:\n");
    reference_determinant_.print();

    // Read options
    nroot_ = options_.get_int("NROOT");

    tau_p_ = options_.get_double("TAUP");
    tau_q_ = options_.get_double("TAUQ");

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

AdaptiveCI::~AdaptiveCI()
{
}

void AdaptiveCI::print_info()
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
    outfile->Printf("\n  %s",string(52,'-').c_str());
    outfile->Flush();
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
    BitsetDeterminant bs_det(alfa_bits,beta_bits);
    P_space_.push_back(bs_det);
    P_space_map_[bs_det] = 1;


    outfile->Printf("\n  The model space contains %zu determinants",P_space_.size());
    outfile->Flush();

    double old_avg_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_avg_energy = 0.0;

    std::vector<std::vector<double> > energy_history;
    SparseCISolver sparse_solver;
    int maxcycle = 20;
    for (int cycle = 0; cycle < maxcycle; ++cycle){
        // Step 1. Diagonalize the Hamiltonian in the P space
        int num_ref_roots = std::min(nroot_,int(P_space_.size()));

        outfile->Printf("\n\n  Cycle %3d",cycle);
        outfile->Printf("\n  %s: %zu determinants","Dimension of the P space",P_space_.size());
        outfile->Flush();


        sparse_solver.diagonalize_hamiltonian(P_space_,P_evals,P_evecs,nroot_,DavidsonLiuSparse);

        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots; ++i){
            double abs_energy = P_evals->get(i) + nuclear_repulsion_energy_;
            double exc_energy = pc_hartree2ev * (P_evals->get(i) - P_evals->get(0));
            outfile->Printf("\n    P-space  CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,abs_energy,exc_energy);
        }
        outfile->Printf("\n");
        outfile->Flush();


        // Step 2. Find determinants in the Q space        
        find_q_space(num_ref_roots,P_evals,P_evecs);


        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        sparse_solver.diagonalize_hamiltonian(PQ_space_,PQ_evals,PQ_evecs,nroot_,DavidsonLiuSparse);

        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < nroot_; ++ i){
            double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_;
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


        // Step 5. Prune the P + Q space to get an update P space
        prune_q_space(PQ_space_,P_space_,P_space_map_,PQ_evecs,nroot_);        

        // Print information about the wave function
        print_wfn(PQ_space_,PQ_evecs,nroot_);
    }

    outfile->Printf("\n\n  ==> Post-Iterations <==\n");
    for (int i = 0; i < nroot_; ++ i){
        double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_;
        double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
        outfile->Printf("\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,abs_energy,exc_energy);
        outfile->Printf("\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV",i + 1,abs_energy + multistate_pt2_energy_correction_[i],
                exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
    }
    outfile->Printf("\n\n  %s: %f s","Adaptive-CI (bitset) ran in ",t_iamrcisd.elapsed());
    outfile->Printf("\n\n  %s: %d","Saving information for root",options_.get_int("ROOT") + 1);
    outfile->Flush();

    return PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_;
}


void AdaptiveCI::diagonalize_hamiltonian(const std::vector<BitsetDeterminant>& space,SharedVector& evals,SharedMatrix& evecs,int nroot)
{
    size_t dim_space = space.size();
    SharedMatrix H(new Matrix("Hamiltonian Matrix",dim_space,dim_space));

    boost::timer t_h_build;
#pragma omp parallel for schedule(dynamic)
    for (size_t I = 0; I < dim_space; ++I){
        const BitsetDeterminant& detI = space[I];
        for (size_t J = I; J < dim_space; ++J){
            const BitsetDeterminant& detJ = space[J];
            double HIJ = detI.slater_rules(detJ);
            H->set(I,J,HIJ);
            H->set(J,I,HIJ);
        }
    }
    double t_h_build_d = t_h_build.elapsed();

    // Be careful, we might not have as many reference dets as roots (just in the first cycle)
    int num_ref_roots = std::min(nroot_,int(dim_space));

    // Diagonalize the Hamiltonian
    evecs.reset(new Matrix("U",dim_space,num_ref_roots));
    evals.reset(new Vector("e",num_ref_roots));
//    outfile->Printf("\n  Using the Davidson-Liu algorithm.");
    boost::timer t_hdiag_large;
    davidson_liu(H,evals,evecs,num_ref_roots);
    double t_hdiag_large_d = t_hdiag_large.elapsed();
    outfile->Printf("\n  %s: %f s (%f s + %f s)","Time to build and diagonalize H",t_h_build_d + t_hdiag_large_d,t_h_build_d,t_hdiag_large_d);
    outfile->Flush();

}

void AdaptiveCI::diagonalize_hamiltonian2(const std::vector<BitsetDeterminant>& space,SharedVector& evals,SharedMatrix& evecs,int nroot)
{
    size_t dim_space = space.size();
    SharedMatrix H(new Matrix("Hamiltonian Matrix",dim_space,dim_space));
    evecs.reset(new Matrix("U",dim_space,nroot));
    evals.reset(new Vector("e",nroot));
    // Full algorithm
    if (options_.get_str("ENERGY_TYPE") == "ACI"){
        H.reset(new Matrix("Hamiltonian Matrix",dim_space,dim_space));

        boost::timer t_h_build2;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < dim_space; ++I){
            const BitsetDeterminant& detI = space[I];
            for (size_t J = I; J < dim_space; ++J){
                const BitsetDeterminant& detJ = space[J];
                double HIJ = detI.slater_rules(detJ);
                H->set(I,J,HIJ);
                H->set(J,I,HIJ);
            }
        }
        outfile->Printf("\n  Time spent building H               = %f s",t_h_build2.elapsed());
        outfile->Flush();

        // 4) Diagonalize the Hamiltonian
        boost::timer t_hdiag_large2;
        if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H,evals,evecs,nroot);
        }else if (options_.get_str("DIAG_ALGORITHM") == "FULL"){
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs,evals);
        }

        outfile->Printf("\n  Time spent diagonalizing H          = %f s",t_hdiag_large2.elapsed());
        outfile->Flush();
    }
    // Sparse algorithm
    else{
        boost::timer t_h_build2;
        std::vector<std::vector<std::pair<int,double> > > H_sparse;

        size_t num_nonzero = 0;
        // Form the Hamiltonian matrix
        for (size_t I = 0; I < dim_space; ++I){
            std::vector<std::pair<int,double> > H_row;
            const BitsetDeterminant& detI = space[I];
            double HII = detI.slater_rules(detI);
            H_row.push_back(make_pair(int(I),HII));
            for (size_t J = 0; J < dim_space; ++J){
                if (I != J){
                    const BitsetDeterminant& detJ = space[J];
                    double HIJ = detI.slater_rules(detJ);
                    if (std::fabs(HIJ) >= 1.0e-12){
                        H_row.push_back(make_pair(int(J),HIJ));
                        num_nonzero += 1;
                    }
                }
            }
            H_sparse.push_back(H_row);
        }

//        outfile->Printf("\n  %s: %zu nonzero elements out of %zu (%e)",num_nonzero,dim_space * dim_space,double(num_nonzero)/double(dim_space * dim_space));
        outfile->Printf("\n  %s: %f s","Time spent building H",t_h_build2.elapsed());
        outfile->Flush();

        // 4) Diagonalize the Hamiltonian
        boost::timer t_hdiag_large2;
        outfile->Printf("\n  Using the sparse Davidson-Liu algorithm.");
        davidson_liu_sparse(H_sparse,evals,evecs, nroot);
        outfile->Printf("\n  %s: %f s","Time spent diagonalizing H",t_hdiag_large2.elapsed());
        outfile->Flush();
    }
}



void AdaptiveCI::find_q_space(int nroot,SharedVector evals,SharedMatrix evecs)
{

    int nmo = reference_determinant_.nmo();

    std::vector<int> aocc(nalpha_);
    std::vector<int> bocc(nbeta_);
    std::vector<int> avir(ncmo_ - nalpha_);
    std::vector<int> bvir(ncmo_ - nbeta_);

    int noalpha = nalpha_;
    int nobeta  = nbeta_;
    int nvalpha = ncmo_ - nalpha_;
    int nvbeta  = ncmo_ - nbeta_;

    // Find the SD space out of the reference
    std::vector<BitsetDeterminant> sd_dets_vec;
    std::map<BitsetDeterminant,int> new_dets_map;

    boost::timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    std::map<BitsetDeterminant,std::vector<double> > V_hash;

    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I){
        const BitsetDeterminant& det = P_space_[I];
        for (int p = 0, i = 0, a = 0; p < ncmo_; ++p){
            if (det.get_alfa_bit(p)){
                aocc[i] = p;
                i++;
            }else{
                avir[a] = p;
                a++;
            }
        }
        for (int p = 0, i = 0, a = 0; p < ncmo_; ++p){
            if (det.get_beta_bit(p)){
                bocc[i] = p;
                i++;
            }else{
                bvir[a] = p;
                a++;
            }
        }

        // Generate aa excitations
        for (int i = 0; i < noalpha; ++i){
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
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
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
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
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
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
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
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
                        if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == wavefunction_symmetry_){
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
    outfile->Printf("\n  %s: %zu determinants","Dimension of the SD space",V_hash.size());
    outfile->Printf("\n  %s: %f s\n","Time spent building the model space",t_ms_build.elapsed());
    outfile->Flush();

    // This will contain all the determinants
    PQ_space_.clear();

    // Add  the P-space determinants
    for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J){
        PQ_space_.push_back(P_space_[J]);
    }

    boost::timer t_ms_screen;

    typedef std::map<BitsetDeterminant,std::vector<double> >::iterator bsmap_it;
    std::vector<std::pair<double,double> > C1(nroot_,make_pair(0.0,0.0));
    std::vector<std::pair<double,double> > E2(nroot_,make_pair(0.0,0.0));
    std::vector<double> ept2(nroot_,0.0);

    // Check the coupling between the reference and the SD space
    for (bsmap_it it = V_hash.begin(), endit = V_hash.end(); it != endit; ++it){
        double EI = it->first.energy();
        for (int n = 0; n < nroot; ++n){
            double V = it->second[n];
            double C1_I = -V / (EI - evals->get(n));
            double E2_I = -V * V / (EI - evals->get(n));

            C1[n] = make_pair(std::fabs(C1_I),C1_I);
            E2[n] = make_pair(std::fabs(E2_I),E2_I);
        }

        std::pair<double,double> max_C1 = *std::max_element(C1.begin(),C1.end());
        std::pair<double,double> max_E2 = *std::max_element(E2.begin(),E2.end());

        double select_value = energy_selection_ ? max_E2.first : max_C1.first;

        if (std::fabs(select_value) > tau_q_){
            PQ_space_.push_back(it->first);
        }else{
            for (int n = 0; n < nroot; ++n){
                ept2[n] += E2[n].second;
            }
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    outfile->Printf("\n  %s: %zu determinants","Dimension of the P + Q space",PQ_space_.size());
    outfile->Printf("\n  %s: %f s","Time spent screening the model space",t_ms_screen.elapsed());
    outfile->Flush();
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

void AdaptiveCI::prune_q_space(std::vector<BitsetDeterminant>& large_space,std::vector<BitsetDeterminant>& pruned_space,
                               std::map<BitsetDeterminant,int>& pruned_space_map,SharedMatrix evecs,int nroot)
{
    // Create a vector that stores the absolute value of the CI coefficients
    std::vector<std::pair<double,size_t> > dm_det_list;

    for (size_t I = 0; I < large_space.size(); ++I){
        double max_dm = 0.0;
        for (int n = 0; n < nroot; ++n){
            max_dm = std::max(max_dm,std::fabs(evecs->get(I,n)));
        }
        dm_det_list.push_back(std::make_pair(max_dm,I));
    }

    // Sort the CI coefficients
    std::sort(dm_det_list.begin(),dm_det_list.end());
    std::reverse(dm_det_list.begin(),dm_det_list.end());

    // Select the new reference space using the sorted CI coefficients
    pruned_space.clear();
    pruned_space_map.clear();

    // Decide which will go in pruned_space
    for (size_t I = 0; I < large_space.size(); ++I){
        if (dm_det_list[I].first > tau_p_){
            pruned_space.push_back(large_space[dm_det_list[I].second]);
            pruned_space_map[large_space[dm_det_list[I].second]] = 1;
        }
    }
}

void AdaptiveCI::print_wfn(std::vector<BitsetDeterminant> space,SharedMatrix evecs,int nroot)
{
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
    }
}

int AdaptiveCI::david2(double **A, int N, int M, double *eps, double **v,
           double cutoff, int print)
{
    int i, j, k, L, I;
    double minimum;
    int min_pos, numf, iter, *conv, converged, maxdim, skip_check;
    int *small2big, init_dim;
    int smart_guess = 1;
    double *Adiag, **b, **bnew, **sigma, **G;
    double *lambda, **alpha, **f, *lambda_old;
    double norm, denom, diff;

    maxdim = 8 * M;

    b = block_matrix(maxdim, N);  /* current set of guess vectors, stored by row */
    bnew = block_matrix(M, N); /* guess vectors formed from old vectors, stored by row*/
    sigma = block_matrix(N, maxdim); /* sigma vectors, stored by column */
    G = block_matrix(maxdim, maxdim); /* Davidson mini-Hamitonian */
    f = block_matrix(maxdim, N); /* residual eigenvectors, stored by row */
    alpha = block_matrix(maxdim, maxdim); /* eigenvectors of G */
    lambda = init_array(maxdim); /* eigenvalues of G */
    lambda_old = init_array(maxdim); /* approximate roots from previous iteration */

    if(smart_guess) { /* Use eigenvectors of a sub-matrix as initial guesses */

        if(N > 7*M) init_dim = 7*M;
        else init_dim = M;
        Adiag = init_array(N);
        small2big = init_int_array(7*M);
        for(i=0; i < N; i++) { Adiag[i] = A[i][i]; }
        for(i=0; i < init_dim; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for(j=1; j < N; j++)
                if(Adiag[j] < minimum) {
                    minimum = Adiag[j];
                    min_pos = j;
                    small2big[i] = j;
                }

            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        for(i=0; i < init_dim; i++) {
            for(j=0; j < init_dim; j++)
                G[i][j] = A[small2big[i]][small2big[j]];
        }

        sq_rsp(init_dim, init_dim, G, lambda, 1, alpha, 1e-12);

        for(i=0; i < init_dim; i++) {
            for(j=0; j < init_dim; j++)
                b[i][small2big[j]] = alpha[j][i];
        }

        free(Adiag);
        free(small2big);
    }
    else { /* Use unit vectors as initial guesses */
        Adiag = init_array(N);
        for(i=0; i < N; i++) { Adiag[i] = A[i][i]; }
        for(i=0; i < M; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for(j=1; j < N; j++)
                if(Adiag[j] < minimum) { minimum = Adiag[j]; min_pos = j; }

            b[i][min_pos] = 1.0;
            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        free(Adiag);
    }

    L = init_dim;
    iter =0;
    converged = 0;
    conv = init_int_array(M); /* boolean array for convergence of each
                   root */
    while(converged < M && iter < MAXIT) {

        skip_check = 0;
        if(print) printf("\niter = %d\n", iter);

        /* form mini-matrix */
        C_DGEMM('n','t', N, L, N, 1.0, &(A[0][0]), N, &(b[0][0]), N,
                0.0, &(sigma[0][0]), maxdim);
        C_DGEMM('n','n', L, L, N, 1.0, &(b[0][0]), N,
                &(sigma[0][0]), maxdim, 0.0, &(G[0][0]), maxdim);

        /* diagonalize mini-matrix */
        sq_rsp(L, L, G, lambda, 1, alpha, 1e-12);

        /* form preconditioned residue vectors */
        for(k=0; k < M; k++)
            for(I=0; I < N; I++) {
                f[k][I] = 0.0;
                for(i=0; i < L; i++) {
                    f[k][I] += alpha[i][k] * (sigma[I][i] - lambda[k] * b[i][I]);
                }
                denom = lambda[k] - A[I][I];
                if(std::fabs(denom) > 1e-6) f[k][I] /= denom;
                else f[k][I] = 0.0;
            }

        /* normalize each residual */
        for(k=0; k < M; k++) {
            norm = 0.0;
            for(I=0; I < N; I++) {
                norm += f[k][I] * f[k][I];
            }
            norm = std::sqrt(norm);
            for(I=0; I < N; I++) {
                if(norm > 1e-6) f[k][I] /= norm;
                else f[k][I] = 0.0;
            }
        }

        /* schmidt orthogonalize the f[k] against the set of b[i] and add
       new vectors */
        for(k=0,numf=0; k < M; k++)
            if(schmidt_add(b, L, N, f[k])) { L++; numf++; }

        /* If L is close to maxdim, collapse to one guess per root */
        if(maxdim - L < M) {
            if(print) {
                printf("Subspace too large: maxdim = %d, L = %d\n", maxdim, L);
                printf("Collapsing eigenvectors.\n");
            }
            for(i=0; i < M; i++) {
                memset((void *) bnew[i], 0, N*sizeof(double));
                for(j=0; j < L; j++) {
                    for(k=0; k < N; k++) {
                        bnew[i][k] += alpha[j][i] * b[j][k];
                    }
                }
            }
            /* orthonormalize the new vectors */
            /* copy new vectors into place */
            for(i=0; i < M; i++){
                norm = 0.0;
                // Project out the orthonormal vectors
                for (j = 0; j < i; ++j){
                    double proj = 0.0;
                    for(k=0; k < N; k++){
                        proj += b[j][k] * bnew[i][k];
                    }
                    for(k=0; k < N; k++){
                        bnew[i][k] -= proj * b[j][k];
                    }
                }
                for(k=0; k < N; k++){
                    norm += bnew[i][k] * bnew[i][k];
                }
                norm = std::sqrt(norm);
                for(k=0; k < N; k++){
                    b[i][k] = bnew[i][k] / norm;
                }
            }

            skip_check = 1;

            L = M;
        }

        /* check convergence on all roots */
        if(!skip_check) {
            converged = 0;
            zero_int_array(conv, M);
            if(print) {
                printf("Root      Eigenvalue       Delta  Converged?\n");
                printf("---- -------------------- ------- ----------\n");
            }
            for(k=0; k < M; k++) {
                diff = std::fabs(lambda[k] - lambda_old[k]);
                if(diff < cutoff) {
                    conv[k] = 1;
                    converged++;
                }
                lambda_old[k] = lambda[k];

                norm = 0.0;
                if(print) {
                    printf("%3d  %20.14f %4.3e    %1s\n", k, lambda[k], diff,
                           conv[k] == 1 ? "Y" : "N");
                }
            }
        }

        iter++;
    }

    /* generate final eigenvalues and eigenvectors */
    //if(converged == M) {
    for(i=0; i < M; i++) {
        eps[i] = lambda[i];
        for(j=0; j < L; j++) {
            for(I=0; I < N; I++) {
                v[I][i] += alpha[j][i] * b[j][I];
            }
        }
        // Normalize v
        norm = 0.0;
        for(I=0; I < N; I++) {
            norm += v[I][i] * v[I][i];
        }
        norm = std::sqrt(norm);
        for(I=0; I < N; I++) {
            v[I][i] /= norm;
        }
    }
    if(print) printf("Davidson algorithm converged in %d iterations.\n", iter);
    //    }

    free(conv);
    free_block(b);
    free_block(bnew);
    free_block(sigma);
    free_block(G);
    free_block(f);
    free_block(alpha);
    free(lambda);
    free(lambda_old);

    return converged;
}


void AdaptiveCI::davidson_liu(SharedMatrix H,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroot_s)
{
    david2(H->pointer(),H->nrow(),nroot_s,Eigenvalues->pointer(),Eigenvectors->pointer(),1.0e-10,0);
}

bool AdaptiveCI::davidson_liu_sparse(std::vector<std::vector<std::pair<int,double> > > H_sparse,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroot_s)
{
//    david2(H->pointer(),H->nrow(),nroot_s,Eigenvalues->pointer(),Eigenvectors->pointer(),1.0e-10,0);

    int N = static_cast<int>(H_sparse.size());
    int M = nroot_s;
    double* eps = Eigenvalues->pointer();
    double** v = Eigenvectors->pointer();
    double cutoff = 1.0e-10;
    int print = 0;

    int i, j, k, L, I;
    double minimum;
    int min_pos, numf, iter, *conv, converged, maxdim, skip_check;
    int *small2big, init_dim;
    int smart_guess = 1;
    double *Adiag, **b, **bnew, **sigma, **G;
    double *lambda, **alpha, **f, *lambda_old;
    double norm, denom, diff;

    maxdim = 8 * M; // Set it back to the original value (8)

    b = block_matrix(maxdim, N);  /* current set of guess vectors,
                       stored by row */
    bnew = block_matrix(M, N); /* guess vectors formed from old vectors,
                    stored by row*/
    sigma = block_matrix(N, maxdim); /* sigma vectors, stored by column */
    G = block_matrix(maxdim, maxdim); /* Davidson mini-Hamitonian */
    f = block_matrix(maxdim, N); /* residual eigenvectors, stored by row */
    alpha = block_matrix(maxdim, maxdim); /* eigenvectors of G */
    lambda = init_array(maxdim); /* eigenvalues of G */
    lambda_old = init_array(maxdim); /* approximate roots from previous
                          iteration */

    if(smart_guess) { /* Use eigenvectors of a sub-matrix as initial guesses */

        if(N > 7 * M) init_dim = 7 * M;
        else init_dim = M;
        Adiag = init_array(N);
        small2big = init_int_array(7 * M);
        for(i=0; i < N; i++) { Adiag[i] = H_sparse[i][0].second; }
        for(i=0; i < init_dim; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for(j=1; j < N; j++)
                if(Adiag[j] < minimum) {
                    minimum = Adiag[j];
                    min_pos = j;
                    small2big[i] = j;
                }

            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        for(i=0; i < init_dim; i++) {
            for(j=0; j < init_dim; j++){
                std::vector<std::pair<int,double> >& H_row = H_sparse[small2big[i]];
                size_t maxc = H_row.size();
                for (size_t c = 0; c < maxc; ++c){
                    if (H_row[c].first == small2big[j]){
                        G[i][j] = H_row[c].second;
                        break;
                    }
                }
                //G[i][j] = A[small2big[i]][small2big[j]];
            }
        }

        sq_rsp(init_dim, init_dim, G, lambda, 1, alpha, 1e-12);

        for(i=0; i < init_dim; i++) {
            for(j=0; j < init_dim; j++)
                b[i][small2big[j]] = alpha[j][i];
        }

        free(Adiag);
        free(small2big);
    }
    else { /* Use unit vectors as initial guesses */
        Adiag = init_array(N);
//        for(i=0; i < N; i++) { Adiag[i] = A[i][i]; }
        for(i=0; i < N; i++) { Adiag[i] = H_sparse[i][0].second; }
        for(i=0; i < M; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for(j=1; j < N; j++)
                if(Adiag[j] < minimum) { minimum = Adiag[j]; min_pos = j; }

            b[i][min_pos] = 1.0;
            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        free(Adiag);
    }

    L = init_dim;
    iter =0;
    converged = 0;
    conv = init_int_array(M); /* boolean array for convergence of each
                       root */
    while(converged < M && iter < MAXIT) {

        skip_check = 0;
        if(print) printf("\niter = %d\n", iter);

        /* form mini-matrix */
        for (int J = 0; J < N; ++J){
            for (int r = 0; r < maxdim; ++r){
                sigma[J][r] = 0.0;
            }
            std::vector<std::pair<int,double> >& H_row = H_sparse[J];
            size_t maxc = H_row.size();
            for (int c = 0; c < maxc; ++c){
                int K = H_row[c].first;
                double HJK = H_row[c].second;
                for (int r = 0; r < L; ++r){
                    sigma[J][r] +=  HJK * b[r][K];
                }
            }
        }
//        C_DGEMM('n','t', N, L, N, 1.0, &(A[0][0]), N, &(b[0][0]), N,
//                0.0, &(sigma[0][0]), maxdim);
        C_DGEMM('n','n', L, L, N, 1.0, &(b[0][0]), N,
                &(sigma[0][0]), maxdim, 0.0, &(G[0][0]), maxdim);

        /* diagonalize mini-matrix */
        sq_rsp(L, L, G, lambda, 1, alpha, 1e-12);

        /* form preconditioned residue vectors */
        for(k=0; k < M; k++)
            for(I=0; I < N; I++) {
                f[k][I] = 0.0;
                for(i=0; i < L; i++) {
                    f[k][I] += alpha[i][k] * (sigma[I][i] - lambda[k] * b[i][I]);
                }
                denom = lambda[k] - H_sparse[I][0].second;//A[I][I];
                if(fabs(denom) > 1e-6) f[k][I] /= denom;
                else f[k][I] = 0.0;
            }

        /* normalize each residual */
        for(k=0; k < M; k++) {
            norm = 0.0;
            for(I=0; I < N; I++) {
                norm += f[k][I] * f[k][I];
            }
            norm = sqrt(norm);
            for(I=0; I < N; I++) {
                f[k][I] /= norm;
                if(norm > 1e-6) f[k][I] /= norm;
                else f[k][I] = 0.0;
            }
        }

        /* schmidt orthogonalize the f[k] against the set of b[i] and add
           new vectors */
        for(k=0,numf=0; k < M; k++)
            if(schmidt_add(b, L, N, f[k])) { L++; numf++; }

        /* If L is close to maxdim, collapse to one guess per root */
        if(maxdim - L < M) {
            if(print) {
                printf("Subspace too large: maxdim = %d, L = %d\n", maxdim, L);
                printf("Collapsing eigenvectors.\n");
            }
            for(i=0; i < M; i++) {
                memset((void *) bnew[i], 0, N*sizeof(double));
                for(j=0; j < L; j++) {
                    for(k=0; k < N; k++) {
                        bnew[i][k] += alpha[j][i] * b[j][k];
                    }
                }
            }

            /* orthonormalize the new vectors */
            /* copy new vectors into place */
            for(i=0; i < M; i++){
                norm = 0.0;
                // Project out the orthonormal vectors
                for (j = 0; j < i; ++j){
                    double proj = 0.0;
                    for(k=0; k < N; k++){
                        proj += b[j][k] * bnew[i][k];
                    }
                    for(k=0; k < N; k++){
                        bnew[i][k] -= proj * b[j][k];
                    }
                }
                for(k=0; k < N; k++){
                    norm += bnew[i][k] * bnew[i][k];
                }
                norm = std::sqrt(norm);
                for(k=0; k < N; k++){
                    b[i][k] = bnew[i][k] / norm;
                }
            }

            skip_check = 1;

            L = M;
        }

        /* check convergence on all roots */
        if(!skip_check) {
            converged = 0;
            zero_int_array(conv, M);
            if(print) {
                printf("Root      Eigenvalue       Delta  Converged?\n");
                printf("---- -------------------- ------- ----------\n");
            }
            for(k=0; k < M; k++) {
                diff = fabs(lambda[k] - lambda_old[k]);
                if(diff < cutoff) {
                    conv[k] = 1;
                    converged++;
                }
                lambda_old[k] = lambda[k];
                if(print) {
                    printf("%3d  %20.14f %4.3e    %1s\n", k, lambda[k], diff,
                           conv[k] == 1 ? "Y" : "N");
                }
            }
        }

        iter++;
    }

    /* generate final eigenvalues and eigenvectors */
    //if(converged == M) {
    for(i=0; i < M; i++) {
        eps[i] = lambda[i];
        for(I=0; I < N; I++){
            v[I][i] = 0.0;
        }
        for(j=0; j < L; j++) {
            for(I=0; I < N; I++) {
                v[I][i] += alpha[j][i] * b[j][I];
            }
        }
        // Normalize v
        norm = 0.0;
        for(I=0; I < N; I++) {
            norm += v[I][i] * v[I][i];
        }
        norm = std::sqrt(norm);
        for(I=0; I < N; I++) {
            v[I][i] /= norm;
        }
    }
    if(print) printf("Davidson algorithm converged in %d iterations.\n", iter);
    //    }

    free(conv);
    free_block(b);
    free_block(bnew);
    free_block(sigma);
    free_block(G);
    free_block(f);
    free_block(alpha);
    free(lambda);
    free(lambda_old);

    return converged;
}


/*

double AdaptiveCI::compute_energy2()
{
    boost::timer t_iamrcisd;

    outfile->Printf("\n\n  Iterative Adaptive MRCISD");

    int nroot_ = options_.get_int("nroot_");

    double tau_p = options_.get_double("TAUP");
    double tau_q = options_.get_double("TAUQ");

    outfile->Printf("\n\n  TAU_P = %f Eh",tau_p);
    outfile->Printf("\n  TAU_Q = %.12f Eh\n",tau_q);

    bool aimed_selection_ = false;
    bool energy_selection_ = false;
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

    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    std::vector<StringDeterminant> P_space_;
    std::map<StringDeterminant,int> P_space_map_;
    P_space_.push_back(reference_determinant_);
    P_space_map_[reference_determinant_] = 1;


    double old_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_energy = 0.0;

    int maxcycle = 20;
    for (int cycle = 0; cycle < maxcycle; ++cycle){
        // Build the Hamiltonian in the P space

        size_t dim_P_space_ = P_space_.size();

        outfile->Printf("\n\n  Cycle %3d. The model space contains %zu determinants",cycle,dim_P_space_);
        outfile->Flush();

        int num_ref_roots = (cycle == 0 ? std::max(nroot_,4) : nroot_);

        H.reset(new Matrix("Hamiltonian Matrix",dim_P_space_,dim_P_space_));
        evecs.reset(new Matrix("U",dim_P_space_,num_ref_roots));
        evals.reset(new Vector("e",num_ref_roots));

        boost::timer t_h_build;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < dim_P_space_; ++I){
            const StringDeterminant& detI = P_space_[I];
            for (size_t J = I; J < dim_P_space_; ++J){
                const StringDeterminant& detJ = P_space_[J];
                double HIJ = detI.slater_rules(detJ);
                H->set(I,J,HIJ);
                H->set(J,I,HIJ);
            }
        }
        outfile->Printf("\n  Time spent building H               = %f s",t_h_build.elapsed());
        outfile->Flush();

        // Diagonalize the Hamiltonian
        boost::timer t_hdiag_large;
        if (cycle == 0){
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs,evals);
        }else{
            if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
                outfile->Printf("\n  Using the Davidson-Liu algorithm.");
                davidson_liu(H,evals,evecs,num_ref_roots);
            }else if (options_.get_str("DIAG_ALGORITHM") == "FULL"){
                outfile->Printf("\n  Performing full diagonalization.");
                H->diagonalize(evecs,evals);
            }
        }

        outfile->Printf("\n  Time spent diagonalizing H          = %f s",t_hdiag_large.elapsed());
        outfile->Flush();

        // Print the energy
        for (int i = 0; i < num_ref_roots; ++ i){
            outfile->Printf("\n  P-space CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,pc_hartree2ev * (evals->get(i) - evals->get(0)));
        }
        outfile->Flush();


        int nmo = reference_determinant_.nmo();
//        size_t nfrzc = frzc_.size();
//        size_t nfrzv = frzv_.size();

        std::vector<int> aocc(nalpha_);
        std::vector<int> bocc(nbeta_);
        std::vector<int> avir(ncmo_ - nalpha_);
        std::vector<int> bvir(ncmo_ - nbeta_);

        int noalpha = nalpha_;
        int nobeta  = nbeta_;
        int nvalpha = ncmo_ - nalpha_;
        int nvbeta  = ncmo_ - nbeta_;

        // Find the SD space out of the reference
        std::vector<StringDeterminant> sd_dets_vec;
        std::map<StringDeterminant,int> new_dets_map;

        boost::timer t_ms_build;

        for (size_t I = 0, max_I = P_space_map_.size(); I < max_I; ++I){
            const StringDeterminant& det = P_space_[I];
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p){
                if (det.get_alfa_bit(p)){
//                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                        aocc[i] = p;
                        i++;
//                    }
                }else{
//                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                        avir[a] = p;
                        a++;
//                    }
                }
            }
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p){
                if (det.get_beta_bit(p)){
//                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                        bocc[i] = p;
                        i++;
//                    }
                }else{
//                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                        bvir[a] = p;
                        a++;
//                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i){
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a){
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
                        StringDeterminant new_det(det);
                        new_det.set_alfa_bit(ii,false);
                        new_det.set_alfa_bit(aa,true);
                        if(P_space_map_.find(new_det) == P_space_map_.end()){
                            sd_dets_vec.push_back(new_det);
                        }
                    }
                }
            }

            for (int i = 0; i < nobeta; ++i){
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a){
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
                        StringDeterminant new_det(det);
                        new_det.set_beta_bit(ii,false);
                        new_det.set_beta_bit(aa,true);
                        if(P_space_map_.find(new_det) == P_space_map_.end()){
                            sd_dets_vec.push_back(new_det);
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
                            if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == wavefunction_symmetry_){
                                StringDeterminant new_det(det);
                                new_det.set_alfa_bit(ii,false);
                                new_det.set_alfa_bit(jj,false);
                                new_det.set_alfa_bit(aa,true);
                                new_det.set_alfa_bit(bb,true);
                                if(P_space_map_.find(new_det) == P_space_map_.end()){
                                    sd_dets_vec.push_back(new_det);
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
                            if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == wavefunction_symmetry_){
                                StringDeterminant new_det(det);
                                new_det.set_alfa_bit(ii,false);
                                new_det.set_beta_bit(jj,false);
                                new_det.set_alfa_bit(aa,true);
                                new_det.set_beta_bit(bb,true);
                                if(P_space_map_.find(new_det) == P_space_map_.end()){
                                    sd_dets_vec.push_back(new_det);
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
                            if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == wavefunction_symmetry_){
                                StringDeterminant new_det(det);
                                new_det.set_beta_bit(ii,false);
                                new_det.set_beta_bit(jj,false);
                                new_det.set_beta_bit(aa,true);
                                new_det.set_beta_bit(bb,true);
                                if(P_space_map_.find(new_det) == P_space_map_.end()){
                                    sd_dets_vec.push_back(new_det);
                                }
                            }
                        }
                    }
                }
            }
        }

        outfile->Printf("\n  The SD excitation space has dimension: %zu",sd_dets_vec.size());
        outfile->Printf("\n  Time spent building the model space = %f s",t_ms_build.elapsed());
        outfile->Flush();


        // Remove the duplicate determinants
        boost::timer t_ms_unique;
        sort( sd_dets_vec.begin(), sd_dets_vec.end() );
        sd_dets_vec.erase( unique( sd_dets_vec.begin(), sd_dets_vec.end() ), sd_dets_vec.end() );
        outfile->Printf("\n  The SD excitation space has dimension: %zu (unique)",sd_dets_vec.size());
        outfile->Printf("\n  Time spent to eliminate duplicate   = %f s",t_ms_unique.elapsed());


        boost::timer t_ms_screen;
        // This will contain all the determinants
        std::vector<StringDeterminant> PQ_space_;

        // Add  the P-space determinants
        for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J){
            PQ_space_.push_back(P_space_[J]);
        }

        // Check the coupling between the reference and the SD space
        std::vector<std::pair<double,size_t> > new_dets_importance_vec;

        std::vector<double> V(nroot_,0.0);
        std::vector<std::pair<double,double> > C1(nroot_,make_pair(0.0,0.0));
        std::vector<std::pair<double,double> > E2(nroot_,make_pair(0.0,0.0));
        std::vector<double> ept2(nroot_,0.0);

        double aimed_selection__sum = 0.0;

        for (size_t I = 0, max_I = sd_dets_vec.size(); I < max_I; ++I){
            double EI = sd_dets_vec[I].energy();
            for (int n = 0; n < nroot_; ++n){
                V[n] = 0;
            }
//#pragma omp parallel for schedule(dynamic)
            for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J){
                double HIJ = sd_dets_vec[I].slater_rules(P_space_[J]);
//#pragma omp critical
                for (int n = 0; n < nroot_; ++n){
                    V[n] += evecs->get(J,n) * HIJ;
                }
            }
            for (int n = 0; n < nroot_; ++n){
                double C1_I = -V[n] / (EI - evals->get(n));
                double E2_I = -V[n] * V[n] / (EI - evals->get(n));
                C1[n] = make_pair(std::fabs(C1_I),C1_I);
                E2[n] = make_pair(std::fabs(E2_I),E2_I);
            }

            std::pair<double,double> max_C1 = *std::max_element(C1.begin(),C1.end());
            std::pair<double,double> max_E2 = *std::max_element(E2.begin(),E2.end());

            double select_value = energy_selection_ ? max_E2.first : max_C1.first;

            // Do not select now, just store the determinant index and the selection criterion
            if(aimed_selection_){
                if (energy_selection_){
                    new_dets_importance_vec.push_back(std::make_pair(select_value,I));
                    aimed_selection__sum += select_value;
                }else{
                    new_dets_importance_vec.push_back(std::make_pair(select_value * select_value,I));
                    aimed_selection__sum += select_value * select_value;
                }
            }else{
                if (std::fabs(select_value) > tau_q){
                    new_dets_importance_vec.push_back(std::make_pair(select_value,I));
                }else{
                    for (int n = 0; n < nroot_; ++n) ept2[n] += E2[n].second;
                }
            }
        }

        if(aimed_selection_){
            std::sort(new_dets_importance_vec.begin(),new_dets_importance_vec.end());
            std::reverse(new_dets_importance_vec.begin(),new_dets_importance_vec.end());
            size_t maxI = new_dets_importance_vec.size();
            outfile->Printf("\n  The SD space will be generated using the aimed scheme (%s)",energy_selection_ ? "energy" : "amplitude");
            outfile->Printf("\n  Initial value of sigma in the aimed selection = %24.14f",aimed_selection__sum);
            for (size_t I = 0; I < maxI; ++I){
                if (aimed_selection__sum > t2_threshold_){
                    PQ_space_.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
                    aimed_selection__sum -= new_dets_importance_vec[I].first;
                }else{
                    break;
                }
            }
            outfile->Printf("\n  Final value of sigma in the aimed selection   = %24.14f",aimed_selection__sum);
            outfile->Printf("\n  Selected %zu determinants",PQ_space_.size()-P_space_.size());
        }else{
            outfile->Printf("\n  The SD space will be generated by screening (%s)",energy_selection_ ? "energy" : "amplitude");
            size_t maxI = new_dets_importance_vec.size();
            for (size_t I = 0; I < maxI; ++I){
                PQ_space_.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
            }
        }

        multistate_pt2_energy_correction_ = ept2;

        size_t PQ_space_.size() = PQ_space_.size();

        outfile->Printf("\n  After screening the ia-MRCISD space contains %zu determinants",PQ_space_.size());
        outfile->Printf("\n  Time spent screening the model space = %f s",t_ms_screen.elapsed());
        outfile->Flush();


        evecs.reset(new Matrix("U",PQ_space_.size(),nroot_));
        evals.reset(new Vector("e",nroot_));
        // Full algorithm
        if (options_.get_str("ENERGY_TYPE") == "ACI"){
            H.reset(new Matrix("Hamiltonian Matrix",PQ_space_.size(),PQ_space_.size()));

            boost::timer t_h_build2;
#pragma omp parallel for schedule(dynamic)
            for (size_t I = 0; I < PQ_space_.size(); ++I){
                const StringDeterminant& detI = PQ_space_[I];
                for (size_t J = I; J < PQ_space_.size(); ++J){
                    const StringDeterminant& detJ = PQ_space_[J];
                    double HIJ = detI.slater_rules(detJ);
                    H->set(I,J,HIJ);
                    H->set(J,I,HIJ);
                }
            }
            outfile->Printf("\n  Time spent building H               = %f s",t_h_build2.elapsed());
            outfile->Flush();

            // 4) Diagonalize the Hamiltonian
            boost::timer t_hdiag_large2;
            if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
                outfile->Printf("\n  Using the Davidson-Liu algorithm.");
                davidson_liu(H,evals,evecs,nroot_);
            }else if (options_.get_str("DIAG_ALGORITHM") == "FULL"){
                outfile->Printf("\n  Performing full diagonalization.");
                H->diagonalize(evecs,evals);
            }

            outfile->Printf("\n  Time spent diagonalizing H          = %f s",t_hdiag_large2.elapsed());
            outfile->Flush();
        }
        // Sparse algorithm
        else{
            boost::timer t_h_build2;
            std::vector<std::vector<std::pair<int,double> > > H_sparse;

            size_t num_nonzero = 0;
            // Form the Hamiltonian matrix
            for (size_t I = 0; I < PQ_space_.size(); ++I){
                std::vector<std::pair<int,double> > H_row;
                const StringDeterminant& detI = PQ_space_[I];
                double HII = detI.slater_rules(detI);
                H_row.push_back(make_pair(int(I),HII));
                for (size_t J = 0; J < PQ_space_.size(); ++J){
                    if (I != J){
                        const StringDeterminant& detJ = PQ_space_[J];
                        double HIJ = detI.slater_rules(detJ);
                        if (std::fabs(HIJ) >= 1.0e-12){
                            H_row.push_back(make_pair(int(J),HIJ));
                            num_nonzero += 1;
                        }
                    }
                }
                H_sparse.push_back(H_row);
            }

            outfile->Printf("\n  %ld nonzero elements out of %ld (%e)",num_nonzero,size_t(PQ_space_.size() * PQ_space_.size()),double(num_nonzero)/double(PQ_space_.size() * PQ_space_.size()));
            outfile->Printf("\n  Time spent building H               = %f s",t_h_build2.elapsed());
            outfile->Flush();

            // 4) Diagonalize the Hamiltonian
            boost::timer t_hdiag_large2;
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu_sparse(H_sparse,evals,evecs,nroot_);
            outfile->Printf("\n  Time spent diagonalizing H          = %f s",t_hdiag_large2.elapsed());
            outfile->Flush();
        }

        //
        for (int i = 0; i < nroot_; ++ i){
            outfile->Printf("\n  Adaptive CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,pc_hartree2ev * (evals->get(i) - evals->get(0)));
            outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_ + multistate_pt2_energy_correction_[i],
                    pc_hartree2ev * (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
        }
        outfile->Flush();


        // Select the new reference space
        P_space_.clear();
        P_space_map_.clear();

        new_energy = evals->get(0) + nuclear_repulsion_energy_;

        outfile->Flush();
        if (std::fabs(new_energy - old_energy) < ia_mrcisd_threshold){
            break;
        }
        old_energy = new_energy;

        std::vector<std::pair<double,size_t> > dm_det_list;

        for (size_t I = 0; I < PQ_space_.size(); ++I){
            double max_dm = 0.0;
            for (int n = 0; n < nroot_; ++n){
                max_dm = std::max(max_dm,std::fabs(evecs->get(I,n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm,I));
        }

        std::sort(dm_det_list.begin(),dm_det_list.end());
        std::reverse(dm_det_list.begin(),dm_det_list.end());

        // Decide which will go in P_space_
        for (size_t I = 0; I < PQ_space_.size(); ++I){
            if (dm_det_list[I].first > tau_p){
                P_space_.push_back(PQ_space_[dm_det_list[I].second]);
                P_space_map_[PQ_space_[dm_det_list[I].second]] = 1;
            }
        }
        //        unordered_map<std::vector<bool>,int> a_str_hash;
        //        unordered_map<std::vector<bool>,int> b_str_hash;

        //        boost::timer t_stringify;
        //        for (size_t I = 0; I < PQ_space_.size(); ++I){
        //            const StringDeterminant& detI = PQ_space_[I];
        //            const std::vector<bool> a_str = detI.get_alfa_bits_vector_bool();
        //            const std::vector<bool> b_str = detI.get_beta_bits_vector_bool();
        //            a_str_hash[a_str] = 1;
        //            b_str_hash[b_str] = 1;
        //        }
        //        outfile->Printf("\n  Size of the @MRCISD space: %zu",PQ_space_.size());
        //        outfile->Printf("\n  Size of the alpha strings: %zu",a_str_hash.size());
        //        outfile->Printf("\n  Size of the beta  strings: %zu",b_str_hash.size());

        //        outfile->Printf("\n\n  Time to stringify: %f s",t_stringify.elapsed());


    }

    for (int i = 0; i < nroot_; ++ i){
        outfile->Printf("\n  * IA-MRCISD total energy (%3d)        = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,pc_hartree2ev * (evals->get(i) - evals->get(0)));
        outfile->Printf("\n  * IA-MRCISD total energy (%3d) + EPT2 = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_ + multistate_pt2_energy_correction_[i],
                pc_hartree2ev * (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
    }

    outfile->Printf("\n\n  iterative_adaptive_mrcisd        ran in %f s",t_iamrcisd.elapsed());
    outfile->Flush();


}

*/
}} // EndNamespaces


