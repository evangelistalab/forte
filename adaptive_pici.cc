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

#include "cartographer.h"
#include "adaptive_pici.h"
#include "sparse_ci_solver.h"

#ifdef _OPENMP
   #include <omp.h>
   bool have_omp = true;
#else
   #define omp_get_max_threads() 1
   #define omp_get_thread_num() 0
   bool have_omp = false;
#endif

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

typedef std::map<BitsetDeterminant,double> bsmap;
typedef std::map<BitsetDeterminant,double>::iterator bsmap_it;

void combine_maps(std::vector<bsmap>& thread_det_C_map, bsmap& dets_C_map);
void copy_map_to_vec(bsmap& dets_C_map,std::vector<BitsetDeterminant>& dets,std::vector<double>& C);
void normalize(std::vector<double>& C);

AdaptivePathIntegralCI::AdaptivePathIntegralCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_),
      options_(options),
      ints_(ints),
      variational_estimate_(false),
      prescreening_tollerance_factor_(1.5)
{
    // Copy the wavefunction information
    copy(wfn);

    startup();
}


void AdaptivePathIntegralCI::startup()
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
    maxiter_ = options_.get_int("MAXITER");

    spawning_threshold_ = options_.get_double("SPAWNING_THRESHOLD");
    initial_guess_spawning_threshold_ = options_.get_double("GUESS_SPAWNING_THRESHOLD");
    time_step_ = options_.get_double("TAU");
    variational_estimate_ = options_.get_bool("VAR_ESTIMATE");
    energy_estimate_freq_ = options_.get_int("ENERGY_ESTIMATE_FREQ");
    adaptive_beta_ = options_.get_bool("ADAPTIVE_BETA");
    e_convergence_ = options_.get_double("E_CONVERGENCE");
    do_shift_ = options_.get_bool("USE_SHIFT");
    do_prescreen_spawning_ = options_.get_bool("PRESCREEN_SPAWNING");
    do_dynamic_prescreening_ = options_.get_bool("DYNAMIC_PRESCREENING");

    if (options_.get_str("PROPAGATOR") == "LINEAR"){
        propagator_ = LinearPropagator;
        propagator_description_ = "Linear";
    }else if (options_.get_str("PROPAGATOR") == "QUADRATIC"){
        propagator_ = QuadraticPropagator;
        propagator_description_ = "Quadratic";
    }
    num_threads_ = omp_get_max_threads();
}

AdaptivePathIntegralCI::~AdaptivePathIntegralCI()
{
}

void AdaptivePathIntegralCI::print_info()
{
    // Print a summary
    std::vector<std::pair<std::string,int>> calculation_info{
        {"Symmetry",wavefunction_symmetry_},
        {"Number of roots",nroot_},
        {"Root used for properties",options_.get_int("ROOT")},
        {"Maximum number of iterations",maxiter_},
        {"Energy estimation frequency",energy_estimate_freq_},
        {"Number of threads",num_threads_}};

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"Time step (beta)",time_step_},
        {"Spawning threshold",spawning_threshold_},
        {"Initial guess spawning threshold",initial_guess_spawning_threshold_},
        {"Convergence threshold",e_convergence_},
        {"Prescreening tollerance factor",prescreening_tollerance_factor_}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"Propagator type",propagator_description_},
        {"Compute variational estimate",variational_estimate_ ? "YES" : "NO"},
        {"Adaptive time step",adaptive_beta_ ? "YES" : "NO"},
        {"Shift the energy",do_shift_ ? "YES" : "NO"},
        {"Prescreen singles spawning",do_prescreen_spawning_ ? "YES" : "NO"},
        {"Dynamic prescreening",do_dynamic_prescreening_ ? "YES" : "NO"},
        {"Using OpenMP", have_omp ? "YES" : "NO"},
    };
//    {"Number of electrons",nel},
//    {"Number of correlated alpha electrons",nalpha_},
//    {"Number of correlated beta electrons",nbeta_},
//    {"Number of restricted docc electrons",rdoccpi_.sum()},
//    {"Charge",charge},
//    {"Multiplicity",multiplicity},

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
//    outfile->Printf("\n  %s",string(52,'-').c_str());
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-40s   %5d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-39s %8.2e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-39s %8s",str_dim.first.c_str(),str_dim.second.c_str());
    }
//    outfile->Printf("\n  %s",string(52,'-').c_str());
    outfile->Flush();
}


double AdaptivePathIntegralCI::compute_energy()
{
    timer_on("PIFCI:Energy");
    boost::timer t_apici;
    outfile->Printf("\n\n\t  ---------------------------------------------------------");
    outfile->Printf("\n\t      Adaptive Path-Integral Full Configuration Interaction");
    outfile->Printf("\n\t                   by Francesco A. Evangelista");
    outfile->Printf("\n\t                    %4d thread(s) %s",num_threads_,have_omp ? "(OMP)" : "");
    outfile->Printf("\n\t  ---------------------------------------------------------");

    // Print a summary of the options
    print_info();

    /// A vector of determinants in the P space
    std::vector<BitsetDeterminant> dets;
    std::vector<double> C;

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);

    // Initial guess
    outfile->Printf("\n\n  ==> Initial Guess <==");
    double apici_energy = initial_guess(dets,C);

    old_max_one_HJI_ = 1e100;
    new_max_one_HJI_ = 1e100;
    old_max_two_HJI_ = 1e100;
    new_max_two_HJI_ = 1e100;

    print_wfn(dets,C);
    std::map<BitsetDeterminant,double> old_space_map;
    for (int I = 0; I < dets.size(); ++I){
        old_space_map[dets[I]] = C[I];
    }

    outfile->Printf("\n\n  ------------------------------------------------------------------------------------------");
    outfile->Printf("\n    Cycle      Ndets     Beta         Proj. Energy          Var. Energy        DE    |dE/dt|");
    outfile->Printf("\n  ------------------------------------------------------------------------------------------");

    double projective_energy_estimator = 0.0;
    double variational_energy_estimator = 0.0;

    int maxcycle = maxiter_;
    double shift = do_shift_ ? apici_energy : 0.0;
    double initial_gradient_norm = 0.0;
    double old_apici_energy = 0.0;
    double beta = 0.0;

    for (int cycle = 0; cycle < maxcycle; ++cycle){
        // Compute |n+1> = exp(-tau H)|n>
        timer_on("PIFCI:Step");
        propagate(propagator_,dets,C,time_step_,spawning_threshold_,shift);
        timer_off("PIFCI:Step");

        // Compute the energy and check for convergence
        if (cycle % energy_estimate_freq_ == 0){
            old_apici_energy = apici_energy;

            timer_on("PIFCI:<E>");
            std::map<std::string,double> results = estimate_energy(dets,C);
            timer_off("PIFCI:<E>");

            std::string energy_type = "VARIATIONAL ENERGY";
            apici_energy = results[energy_type];
            double energy_gradient = std::fabs((apici_energy - old_apici_energy) / (time_step_ * energy_estimate_freq_));

            outfile->Printf("\n%9d %10zu %8.4f %20.12f %20.12f %20.12f %.3e %.3e %10zu %10zu",cycle,beta,C.size(),
                            results["PROJECTIVE ENERGY"],results["VARIATIONAL ENERGY"],
                            results["VARIATIONAL ENERGY 2"],
                            std::fabs(apici_energy - old_apici_energy),energy_gradient,ndet_visited_,ndet_accepted_);

            if (std::fabs(energy_gradient) < e_convergence_){
                break;
            }
        }
        beta += time_step_;
        outfile->Flush();
    }

    outfile->Printf("\n  ------------------------------------------------------------------------------------------");
    outfile->Printf("\n\n  Calculation converged.\n");
    print_wfn(dets,C);

    Process::environment.globals["APICI ENERGY"] = apici_energy;

    outfile->Printf("\n\n  ==> Post-Iterations <==\n");
    outfile->Printf("\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV",1,apici_energy);

    outfile->Printf("\n  * Size of CI space                   = %zu",C.size());
    outfile->Printf("\n  * Determinants spawned/cycle         = %zu",nspawned_);
    outfile->Printf("\n  * Determinants that do not spawn     = %zu",nzerospawn_);

    outfile->Printf("\n\n  %s: %f s","Adaptive Path-Integral CI (bitset) ran in ",t_apici.elapsed());
    outfile->Flush();

    timer_off("PIFCI:Energy");
    return apici_energy;
}


double AdaptivePathIntegralCI::initial_guess(std::vector<BitsetDeterminant>& dets,std::vector<double>& C)
{
    // Use the reference determinant as a starting point
    std::vector<bool> alfa_bits = reference_determinant_.get_alfa_bits_vector_bool();
    std::vector<bool> beta_bits = reference_determinant_.get_beta_bits_vector_bool();
    std::map<BitsetDeterminant,double> dets_C;


    // Do one time step starting from the reference determinant
    BitsetDeterminant bs_det(alfa_bits,beta_bits);
    time_step_optimized(spawning_threshold_ * 10.0,bs_det,1.0,dets_C,0.0);

    // Save the list of determinants
    copy_map_to_vec(dets_C,dets,C);

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);

    size_t wfn_size = dets.size();
    SharedMatrix evecs(new Matrix("Eigenvectors",wfn_size,1));
    SharedVector evals(new Vector("Eigenvalues",wfn_size));
    sparse_solver.diagonalize_hamiltonian(dets,evals,evecs,1);
    double var_energy = evals->get(0) + nuclear_repulsion_energy_;
    outfile->Printf("\n\n  Initial guess energy (variational) = %20.12f Eh",var_energy);

    // Copy the ground state eigenvector
    for (int I = 0; I < wfn_size; ++I){
        C[I] = evecs->get(I,0);
    }
    return var_energy;
}

void AdaptivePathIntegralCI::propagate(PropagatorType propagator, std::vector<BitsetDeterminant>& dets, std::vector<double>& C, double tau, double spawning_threshold, double S)
{
    // Reset statistics
    ndet_visited_ = 0;
    ndet_accepted_ = 0;

    // Reset prescreening boundary
    if (do_prescreen_spawning_){
        new_max_one_HJI_ = 0.0;
        new_max_two_HJI_ = 0.0;
    }

    // Evaluate (1-beta H) |C>
    if (propagator == LinearPropagator){
        propagate_first_order(dets,C,tau,spawning_threshold,S);
    }else if (propagator == QuadraticPropagator){
        propagate_second_order(dets,C,tau,spawning_threshold,S);
    }

    // Update prescreening boundary
    if (do_prescreen_spawning_){
        old_max_one_HJI_ = new_max_one_HJI_;
        old_max_two_HJI_ = new_max_two_HJI_;
    }
    normalize(C);
}


void AdaptivePathIntegralCI::propagate_first_order(std::vector<BitsetDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S)
{
    // A map that contains the pair (determinant,coefficient)
    std::map<BitsetDeterminant,double> dets_C_map;
    // A vector of maps that hold (determinant,coefficient)
    std::vector<std::map<BitsetDeterminant,double> > thread_det_C_map(num_threads_);

    // Propagate the wave function for one time step using |n+1> = (1 - tau (H-S))|n>
    if(do_dynamic_prescreening_){
        // Term 1. |n>
        std::map<size_t,size_t> stats;
        size_t max_I = dets.size();
        for (size_t I = 0; I < max_I; ++I){
            dets_C_map[dets[I]] = C[I] * (1.0 - tau * (dets[I].energy() - S));
        }

        // Term 2. -tau (H-S)|n>
        nspawned_ = 0;
        nzerospawn_ = 0;
        std::pair<double,double> zero_pair;
#pragma omp parallel for
        for (size_t I = 0; I < max_I; ++I){
            int thread_id = omp_get_thread_num();
            size_t spawned = 0;
            // Update the list of couplings
            std::pair<double,double> max_coupling;
            #pragma omp critical
            {
                max_coupling = dets_max_couplings_[dets[I]];
            }
            if (max_coupling == zero_pair){
                spawned = apply_tau_H_spawning(-tau,spawning_threshold,dets[I],C[I],thread_det_C_map[thread_id],S,max_coupling);
                #pragma omp critical
                {
                    dets_max_couplings_[dets[I]] = max_coupling;
                }
            }else{
                spawned = apply_tau_H_spawning(-tau,spawning_threshold,dets[I],C[I],thread_det_C_map[thread_id],S,max_coupling);
            }
            #pragma omp critical
            {
                nspawned_ += spawned;
                ndet_accepted_ += spawned;
            }
        }
    }else{
        size_t max_I = dets.size();
        for (size_t I = 0; I < max_I; ++I){
            dets_C_map[dets[I]] = C[I];
        }
#pragma omp parallel for
        for (size_t I = 0; I < max_I; ++I){
            int thread_id = omp_get_thread_num();
            apply_tau_H(-tau,spawning_threshold,dets[I],C[I],thread_det_C_map[thread_id],S);
//            time_step_optimized(spawning_threshold,dets[I],C[I],thread_det_C_map[thread_id],S);
        }
    }

    // Cobine the results of all the threads
    combine_maps(thread_det_C_map,dets_C_map);

    // Overwrite the input vectors with the updated wave function
    copy_map_to_vec(dets_C_map,dets,C);
}


void AdaptivePathIntegralCI::propagate_second_order(std::vector<BitsetDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S)
{
    // A map that contains the pair (determinant,coefficient)
    std::map<BitsetDeterminant,double> dets_C_map;
    // A vector of maps that hold (determinant,coefficient)
    std::vector<std::map<BitsetDeterminant,double> > thread_det_C_map(num_threads_);

    // Propagate the wave function for one time step using |n+1> = (1 - tau (H-S) + tau^2 (H-S)^2 / 2)|n>
    size_t max_I = dets.size();

    // Term 1. |n>
    for (size_t I = 0; I < max_I; ++I){
        dets_C_map[dets[I]] = C[I];
    }

    // Term 2. -tau (H-S)|n>
#pragma omp parallel for
    for (size_t I = 0; I < max_I; ++I){
        int thread_id = omp_get_thread_num();
        apply_tau_H(-tau,spawning_threshold,dets[I],C[I],thread_det_C_map[thread_id],S);
    }

    // Cobine the results of all the threads
    combine_maps(thread_det_C_map,dets_C_map);
    // Reset the maps
    for (int t = 0; t < thread_det_C_map.size(); ++t) thread_det_C_map[t].clear();
    // Copy the wave function to a vector
    copy_map_to_vec(dets_C_map,dets,C);

    // Term 3. -tau/2 H(-tau H|old>) = tau^2/2 H^2 |old>
#pragma omp parallel for
    for (size_t I = 0; I < max_I; ++I){
        int thread_id = omp_get_thread_num();
        apply_tau_H(-0.5 * tau,spawning_threshold,dets[I],C[I],thread_det_C_map[thread_id],S);
    }

    // Cobine the results of all the threads
    combine_maps(thread_det_C_map,dets_C_map);
    // Reset the maps
    for (int t = 0; t < thread_det_C_map.size(); ++t) thread_det_C_map[t].clear();
    // Copy the wave function to a vector
    copy_map_to_vec(dets_C_map,dets,C);
}




double AdaptivePathIntegralCI::time_step_optimized(double spawning_threshold,BitsetDeterminant& detI, double CI, std::map<BitsetDeterminant,double>& new_space_C, double E0)
{
    std::vector<int> aocc = detI.get_alfa_occ();
    std::vector<int> bocc = detI.get_beta_occ();
    std::vector<int> avir = detI.get_alfa_vir();
    std::vector<int> bvir = detI.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    double gradient_norm = 0.0;

    // Contribution of this determinant
    new_space_C[detI] += (1.0 - time_step_ * (detI.energy() - E0)) * CI;

    double my_new_max_one_HJI_ = 0.0;
    size_t my_ndet_accepted = 0;
    size_t my_ndet_visited = 0;
    //timer_on("APICI: S");
    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a){
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
                if (std::fabs(prescreening_tollerance_factor_ * old_max_one_HJI_ * CI) >= spawning_threshold){
                    BitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii,false);
                    detJ.set_alfa_bit(aa,true);
                    double HJI = detJ.slater_rules(detI);
                    my_new_max_one_HJI_ = std::max(my_new_max_one_HJI_,std::fabs(HJI));
                    if (std::fabs(HJI * CI) >= spawning_threshold){
                        new_space_C[detJ] += -time_step_ * HJI * CI;
                        gradient_norm += std::fabs(-time_step_ * HJI * CI);
                        my_ndet_accepted++;
                    }
                    my_ndet_visited++;
                }
            }
        }
    }

    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a){
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
                if (std::fabs(prescreening_tollerance_factor_ * old_max_one_HJI_ * CI) >= spawning_threshold){
                    BitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(ii,false);
                    detJ.set_beta_bit(aa,true);
                    double HJI = detJ.slater_rules(detI);
                    my_new_max_one_HJI_ = std::max(my_new_max_one_HJI_,std::fabs(HJI));
                    if (std::fabs(HJI * CI) >= spawning_threshold){
                        new_space_C[detJ] += -time_step_ * HJI * CI;
                        gradient_norm += std::fabs(-time_step_ * HJI * CI);
                        my_ndet_accepted++;
                    }
                    my_ndet_visited++;
                }
            }
        }
    }

    //timer_off("APICI: S");

    //timer_on("APICI: D");
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
                        double HJI = ints_->aptei_aa(ii,jj,aa,bb);
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            BitsetDeterminant detJ(detI);
                            detJ.set_alfa_bit(ii,false);
                            detJ.set_alfa_bit(jj,false);
                            detJ.set_alfa_bit(aa,true);
                            detJ.set_alfa_bit(bb,true);

                            // grap the alpha bits of both determinants
                            const boost::dynamic_bitset<>& Ia = detI.alfa_bits();
                            const boost::dynamic_bitset<>& Ja = detJ.alfa_bits();

                            // compute the sign of the matrix element
                            HJI *= SlaterSign(Ia,ii) * SlaterSign(Ia,jj) * SlaterSign(Ja,aa) * SlaterSign(Ja,bb);

                            new_space_C[detJ] += -time_step_ * HJI * CI;

                            gradient_norm += std::fabs(time_step_ * HJI * CI);
                            my_ndet_accepted++;
                        }
                        my_ndet_visited++;
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
                        double HJI = ints_->aptei_ab(ii,jj,aa,bb);
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            BitsetDeterminant detJ(detI);
                            detJ.set_alfa_bit(ii,false);
                            detJ.set_beta_bit(jj,false);
                            detJ.set_alfa_bit(aa,true);
                            detJ.set_beta_bit(bb,true);

                            // grap the alpha bits of both determinants
                            const boost::dynamic_bitset<>& Ia = detI.alfa_bits();
                            const boost::dynamic_bitset<>& Ib = detI.beta_bits();
                            const boost::dynamic_bitset<>& Ja = detJ.alfa_bits();
                            const boost::dynamic_bitset<>& Jb = detJ.beta_bits();

                            // compute the sign of the matrix element
                            HJI *= SlaterSign(Ia,ii) * SlaterSign(Ib,jj) * SlaterSign(Ja,aa) * SlaterSign(Jb,bb);

                            new_space_C[detJ] += -time_step_ * HJI * CI;

                            gradient_norm += std::fabs(-time_step_ * HJI * CI);
                            my_ndet_accepted++;
                        }
                        my_ndet_visited++;
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
                        double HJI = ints_->aptei_bb(ii,jj,aa,bb);
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            BitsetDeterminant detJ(detI);
                            detJ.set_beta_bit(ii,false);
                            detJ.set_beta_bit(jj,false);
                            detJ.set_beta_bit(aa,true);
                            detJ.set_beta_bit(bb,true);

                            // grap the alpha bits of both determinants
                            const boost::dynamic_bitset<>& Ib = detI.beta_bits();
                            const boost::dynamic_bitset<>& Jb = detJ.beta_bits();

                            // compute the sign of the matrix element
                            HJI *= SlaterSign(Ib,ii) * SlaterSign(Ib,jj) * SlaterSign(Jb,aa) * SlaterSign(Jb,bb);

                            new_space_C[detJ] += -time_step_ * HJI * CI;

                            gradient_norm += std::fabs(time_step_ * HJI * CI);
                            my_ndet_accepted++;
                        }
                        my_ndet_visited++;
                    }
                }
            }
        }
    }
    //timer_off("APICI: D");

    // Reduce race condition
    new_max_one_HJI_ = std::max(my_new_max_one_HJI_,new_max_one_HJI_);
    ndet_accepted_ += my_ndet_accepted;
    ndet_visited_ += my_ndet_visited;

    return gradient_norm;
}


size_t AdaptivePathIntegralCI::apply_tau_H(double tau,double spawning_threshold,BitsetDeterminant& detI, double CI, std::map<BitsetDeterminant,double>& new_space_C, double E0)
{
    std::vector<int> aocc = detI.get_alfa_occ();
    std::vector<int> bocc = detI.get_beta_occ();
    std::vector<int> avir = detI.get_alfa_vir();
    std::vector<int> bvir = detI.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    double my_new_max_one_HJI = 0.0;
    double my_new_max_two_HJI = 0.0;

    // Diagonal contributions
    double det_energy = detI.energy();
//    if (std::fabs((det_energy - E0) * CI) >= spawning_threshold){
        new_space_C[detI] += tau * (det_energy - E0) * CI;
//    }

    size_t spawned = 0;

    if (std::fabs(prescreening_tollerance_factor_ * old_max_one_HJI_ * CI) >= spawning_threshold){
        // Generate aa excitations
        for (int i = 0; i < noalpha; ++i){
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
                    BitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii,false);
                    detJ.set_alfa_bit(aa,true);
                    double HJI = detJ.slater_rules(detI);
                    my_new_max_one_HJI = std::max(my_new_max_one_HJI,std::fabs(HJI));
                    if (std::fabs(HJI * CI) >= spawning_threshold){
                        new_space_C[detJ] += tau * HJI * CI;
                        spawned++;
                    }
                }
            }
        }

        for (int i = 0; i < nobeta; ++i){
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a){
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
                    BitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(ii,false);
                    detJ.set_beta_bit(aa,true);
                    double HJI = detJ.slater_rules(detI);
                    my_new_max_one_HJI = std::max(my_new_max_one_HJI,std::fabs(HJI));
                    if (std::fabs(HJI * CI) >= spawning_threshold){
                        new_space_C[detJ] += tau * HJI * CI;
                        spawned++;
                    }
                }
            }
        }
    }

    if (std::fabs(prescreening_tollerance_factor_ * old_max_two_HJI_ * CI) >= spawning_threshold){
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
                            double HJI = ints_->aptei_aa(ii,jj,aa,bb);
                            my_new_max_two_HJI = std::max(my_new_max_two_HJI,std::fabs(HJI));

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_alfa_bit(ii,false);
                                detJ.set_alfa_bit(jj,false);
                                detJ.set_alfa_bit(aa,true);
                                detJ.set_alfa_bit(bb,true);

                                // grap the alpha bits of both determinants
                                const boost::dynamic_bitset<>& Ia = detI.alfa_bits();
                                const boost::dynamic_bitset<>& Ja = detJ.alfa_bits();

                                // compute the sign of the matrix element
                                HJI *= SlaterSign(Ia,ii) * SlaterSign(Ia,jj) * SlaterSign(Ja,aa) * SlaterSign(Ja,bb);

                                new_space_C[detJ] += tau * HJI * CI;


                                spawned++;
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
                            double HJI = ints_->aptei_ab(ii,jj,aa,bb);
                            my_new_max_two_HJI = std::max(my_new_max_two_HJI,std::fabs(HJI));

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_alfa_bit(ii,false);
                                detJ.set_beta_bit(jj,false);
                                detJ.set_alfa_bit(aa,true);
                                detJ.set_beta_bit(bb,true);

                                // grap the alpha bits of both determinants
                                const boost::dynamic_bitset<>& Ia = detI.alfa_bits();
                                const boost::dynamic_bitset<>& Ib = detI.beta_bits();
                                const boost::dynamic_bitset<>& Ja = detJ.alfa_bits();
                                const boost::dynamic_bitset<>& Jb = detJ.beta_bits();

                                // compute the sign of the matrix element
                                HJI *= SlaterSign(Ia,ii) * SlaterSign(Ib,jj) * SlaterSign(Ja,aa) * SlaterSign(Jb,bb);

                                new_space_C[detJ] += tau * HJI * CI;


                                spawned++;
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
                            double HJI = ints_->aptei_bb(ii,jj,aa,bb);
                            my_new_max_two_HJI = std::max(my_new_max_two_HJI,std::fabs(HJI));

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_beta_bit(ii,false);
                                detJ.set_beta_bit(jj,false);
                                detJ.set_beta_bit(aa,true);
                                detJ.set_beta_bit(bb,true);

                                // grap the alpha bits of both determinants
                                const boost::dynamic_bitset<>& Ib = detI.beta_bits();
                                const boost::dynamic_bitset<>& Jb = detJ.beta_bits();

                                // compute the sign of the matrix element
                                HJI *= SlaterSign(Ib,ii) * SlaterSign(Ib,jj) * SlaterSign(Jb,aa) * SlaterSign(Jb,bb);

                                new_space_C[detJ] += tau * HJI * CI;


                                spawned++;
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce race condition
    new_max_one_HJI_ = std::max(my_new_max_one_HJI,new_max_one_HJI_);
    new_max_two_HJI_ = std::max(my_new_max_two_HJI,new_max_two_HJI_);

    return spawned;
}


size_t AdaptivePathIntegralCI::apply_tau_H_spawning(double tau,double spawning_threshold,BitsetDeterminant& detI, double CI, std::map<BitsetDeterminant,double>& new_space_C, double E0,std::pair<double,double>& max_coupling)
{
    std::vector<int> aocc = detI.get_alfa_occ();
    std::vector<int> bocc = detI.get_beta_occ();
    std::vector<int> avir = detI.get_alfa_vir();
    std::vector<int> bvir = detI.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    size_t spawned = 0;

    // No diagonal contributions

    if ((max_coupling.first == 0.0) or (std::fabs(max_coupling.first * CI) >= spawning_threshold)){
        // Generate aa excitations
        for (int i = 0; i < noalpha; ++i){
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
                    BitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii,false);
                    detJ.set_alfa_bit(aa,true);
                    double HJI = detJ.slater_rules(detI);
                    max_coupling.first = std::max(max_coupling.first,std::fabs(HJI));
                    if (std::fabs(HJI * CI) >= spawning_threshold){
                        new_space_C[detJ] += tau * HJI * CI;
                        spawned++;
                    }
                    ndet_visited_++;
                }
            }
        }

        for (int i = 0; i < nobeta; ++i){
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a){
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
                    BitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(ii,false);
                    detJ.set_beta_bit(aa,true);
                    double HJI = detJ.slater_rules(detI);
                    max_coupling.first = std::max(max_coupling.first,std::fabs(HJI));
                    if (std::fabs(HJI * CI) >= spawning_threshold){
                        new_space_C[detJ] += tau * HJI * CI;
                        spawned++;
                    }
                    ndet_visited_++;
                }
            }
        }
    }

    if ((max_coupling.second == 0.0) or (std::fabs(max_coupling.second  * CI) >= spawning_threshold)){
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
                            double HJI = ints_->aptei_aa(ii,jj,aa,bb);
                            max_coupling.second = std::max(max_coupling.second,std::fabs(HJI));

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_alfa_bit(ii,false);
                                detJ.set_alfa_bit(jj,false);
                                detJ.set_alfa_bit(aa,true);
                                detJ.set_alfa_bit(bb,true);

                                // grap the alpha bits of both determinants
                                const boost::dynamic_bitset<>& Ia = detI.alfa_bits();
                                const boost::dynamic_bitset<>& Ja = detJ.alfa_bits();

                                // compute the sign of the matrix element
                                HJI *= SlaterSign(Ia,ii) * SlaterSign(Ia,jj) * SlaterSign(Ja,aa) * SlaterSign(Ja,bb);

                                new_space_C[detJ] += tau * HJI * CI;


                                spawned++;
                            }
                            ndet_visited_++;
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
                            double HJI = ints_->aptei_ab(ii,jj,aa,bb);
                            max_coupling.second = std::max(max_coupling.second,std::fabs(HJI));

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_alfa_bit(ii,false);
                                detJ.set_beta_bit(jj,false);
                                detJ.set_alfa_bit(aa,true);
                                detJ.set_beta_bit(bb,true);

                                // grap the alpha bits of both determinants
                                const boost::dynamic_bitset<>& Ia = detI.alfa_bits();
                                const boost::dynamic_bitset<>& Ib = detI.beta_bits();
                                const boost::dynamic_bitset<>& Ja = detJ.alfa_bits();
                                const boost::dynamic_bitset<>& Jb = detJ.beta_bits();

                                // compute the sign of the matrix element
                                HJI *= SlaterSign(Ia,ii) * SlaterSign(Ib,jj) * SlaterSign(Ja,aa) * SlaterSign(Jb,bb);

                                new_space_C[detJ] += tau * HJI * CI;


                                spawned++;
                            }
                            ndet_visited_++;
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
                            double HJI = ints_->aptei_bb(ii,jj,aa,bb);
                            max_coupling.second = std::max(max_coupling.second,std::fabs(HJI));

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_beta_bit(ii,false);
                                detJ.set_beta_bit(jj,false);
                                detJ.set_beta_bit(aa,true);
                                detJ.set_beta_bit(bb,true);

                                // grap the alpha bits of both determinants
                                const boost::dynamic_bitset<>& Ib = detI.beta_bits();
                                const boost::dynamic_bitset<>& Jb = detJ.beta_bits();

                                // compute the sign of the matrix element
                                HJI *= SlaterSign(Ib,ii) * SlaterSign(Ib,jj) * SlaterSign(Jb,aa) * SlaterSign(Jb,bb);

                                new_space_C[detJ] += tau * HJI * CI;


                                spawned++;
                            }
                            ndet_visited_++;
                        }
                    }
                }
            }
        }
    }

    // Reduce race condition
    new_max_one_HJI_ = std::max(max_coupling.first,new_max_one_HJI_);
    new_max_two_HJI_ = std::max(max_coupling.second,new_max_two_HJI_);

    return spawned;
}

std::map<std::string,double> AdaptivePathIntegralCI::estimate_energy(std::vector<BitsetDeterminant>& dets,std::vector<double>& C)
{
    std::map<std::string,double> results;
    results["PROJECTIVE ENERGY"] = estimate_proj_energy(dets,C);

    timer_on("PIFCI:<Evar1>");
    results["VARIATIONAL ENERGY"] = estimate_var_energy(dets,C);
    timer_off("PIFCI:<Evar1>");

    timer_on("PIFCI:<Evar2>");
    results["VARIATIONAL ENERGY 2"] = estimate_var_energy2(dets,C);
    timer_off("PIFCI:<Evar2>");
    return results;
}

static bool abs_compare(double a, double b)
{
    return (std::abs(a) < std::abs(b));
}

double AdaptivePathIntegralCI::estimate_proj_energy(std::vector<BitsetDeterminant>& dets,std::vector<double>& C)
{
    // Find the determinant with the largest value of C
    auto result = std::max_element(C.begin(), C.end(), abs_compare);
    size_t J = std::distance(C.begin(), result);
    double CJ = C[J];

    // Compute the projective energy
    double projective_energy_estimator = 0.0;
    for (int I = 0, max_I = dets.size(); I < max_I; ++I){
        double HIJ = dets[I].slater_rules(dets[J]);
        projective_energy_estimator += HIJ * C[I] / CJ;
    }
    return projective_energy_estimator + nuclear_repulsion_energy_;
}


double AdaptivePathIntegralCI::estimate_var_energy(std::vector<BitsetDeterminant> &dets, std::vector<double> &C)
{
    // Compute a variational estimator of the energy
    size_t size = dets.size();
    double variational_energy_estimator = 0.0;
#pragma omp parallel for reduction(+:variational_energy_estimator)
    for (int I = 0; I < size; ++I){
        const BitsetDeterminant& detI = dets[I];
        variational_energy_estimator += C[I] * C[I] * detI.energy();
        for (int J = I + 1; J < size; ++J){
            if (std::fabs(C[I] * C[J]) > 1.0e-12){
                double HIJ = dets[I].slater_rules(dets[J]);
                variational_energy_estimator += 2.0 * C[I] * HIJ * C[J];
            }
        }
    }
    return variational_energy_estimator + nuclear_repulsion_energy_;
}


double AdaptivePathIntegralCI::estimate_var_energy2(std::vector<BitsetDeterminant> &dets, std::vector<double> &C)
{
    // A map that contains the pair (determinant,coefficient)
    std::map<BitsetDeterminant,double> dets_C_map;

    double tau = time_step_;
    double spawning_threshold2 = 1.0e-9;
    double variational_energy_estimator = 0.0;
    std::vector<double> energy(num_threads_,0.0);

    size_t max_I = dets.size();
    for (size_t I = 0; I < max_I; ++I){
        dets_C_map[dets[I]] = C[I];
    }

    std::pair<double,double> zero(0.0,0.0);
#pragma omp parallel for
    for (size_t I = 0; I < max_I; ++I){
        int thread_id = omp_get_thread_num();
        // Update the list of couplings
        std::pair<double,double> max_coupling;
#pragma omp critical
        {
            max_coupling = dets_max_couplings_[dets[I]];
        }
        if (max_coupling == zero){
            max_coupling = {1.0,1.0};
        }
        energy[thread_id] += form_H_C(1.0,spawning_threshold2,dets[I],C[I],dets_C_map,max_coupling);
    }

    for (size_t I = 0; I < max_I; ++I){
        variational_energy_estimator += C[I] * C[I] * dets[I].energy();
    }
    for (int t = 0; t < num_threads_; ++t){
        variational_energy_estimator += energy[t];
    }

    return variational_energy_estimator + nuclear_repulsion_energy_;
}

double AdaptivePathIntegralCI::form_H_C(double tau,double spawning_threshold,BitsetDeterminant& detI, double CI, std::map<BitsetDeterminant,double>& det_C,std::pair<double,double>& max_coupling)
{
    double result = 0.0;

    std::vector<int> aocc = detI.get_alfa_occ();
    std::vector<int> bocc = detI.get_beta_occ();
    std::vector<int> avir = detI.get_alfa_vir();
    std::vector<int> bvir = detI.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    size_t spawned = 0;

    // No diagonal contributions

    if ((std::fabs(max_coupling.first * CI) >= spawning_threshold)){
        // Generate aa excitations
        for (int i = 0; i < noalpha; ++i){
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
                    BitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii,false);
                    detJ.set_alfa_bit(aa,true);
                    bsmap_it it = det_C.find(detJ);
                    if (it != det_C.end()){
                        double HJI = detJ.slater_rules(detI);
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            result += tau * HJI * CI * it->second;
                            spawned++;
                        }
                    }
                    ndet_visited_++;
                }
            }
        }

        for (int i = 0; i < nobeta; ++i){
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a){
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
                    BitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(ii,false);
                    detJ.set_beta_bit(aa,true);
                    bsmap_it it = det_C.find(detJ);
                    if (it != det_C.end()){
                        double HJI = detJ.slater_rules(detI);
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            result += tau * HJI * CI * it->second;
                            spawned++;
                        }
                    }
                }
            }
        }
    }

    if ((max_coupling.second == 0.0) or (std::fabs(max_coupling.second  * CI) >= spawning_threshold)){
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
                            double HJI = ints_->aptei_aa(ii,jj,aa,bb);

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_alfa_bit(ii,false);
                                detJ.set_alfa_bit(jj,false);
                                detJ.set_alfa_bit(aa,true);
                                detJ.set_alfa_bit(bb,true);

                                bsmap_it it = det_C.find(detJ);
                                if (it != det_C.end()){
                                    // grap the alpha bits of both determinants
                                    const boost::dynamic_bitset<>& Ia = detI.alfa_bits();
                                    const boost::dynamic_bitset<>& Ja = detJ.alfa_bits();

                                    // compute the sign of the matrix element
                                    HJI *= SlaterSign(Ia,ii) * SlaterSign(Ia,jj) * SlaterSign(Ja,aa) * SlaterSign(Ja,bb);

                                    result += tau * HJI * CI * it->second;
                                    spawned++;
                                }
                            }
                            ndet_visited_++;
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
                            double HJI = ints_->aptei_ab(ii,jj,aa,bb);

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_alfa_bit(ii,false);
                                detJ.set_beta_bit(jj,false);
                                detJ.set_alfa_bit(aa,true);
                                detJ.set_beta_bit(bb,true);

                                bsmap_it it = det_C.find(detJ);
                                if (it != det_C.end()){
                                    // grap the alpha bits of both determinants
                                    const boost::dynamic_bitset<>& Ia = detI.alfa_bits();
                                    const boost::dynamic_bitset<>& Ib = detI.beta_bits();
                                    const boost::dynamic_bitset<>& Ja = detJ.alfa_bits();
                                    const boost::dynamic_bitset<>& Jb = detJ.beta_bits();

                                    // compute the sign of the matrix element
                                    HJI *= SlaterSign(Ia,ii) * SlaterSign(Ib,jj) * SlaterSign(Ja,aa) * SlaterSign(Jb,bb);

                                    result += tau * HJI * CI * it->second;
                                    spawned++;
                                }
                            }
                            ndet_visited_++;
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
                            double HJI = ints_->aptei_bb(ii,jj,aa,bb);

                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                BitsetDeterminant detJ(detI);
                                detJ.set_beta_bit(ii,false);
                                detJ.set_beta_bit(jj,false);
                                detJ.set_beta_bit(aa,true);
                                detJ.set_beta_bit(bb,true);

                                bsmap_it it = det_C.find(detJ);
                                if (it != det_C.end()){
                                    // grap the alpha bits of both determinants
                                    const boost::dynamic_bitset<>& Ib = detI.beta_bits();
                                    const boost::dynamic_bitset<>& Jb = detJ.beta_bits();

                                    // compute the sign of the matrix element
                                    HJI *= SlaterSign(Ib,ii) * SlaterSign(Ib,jj) * SlaterSign(Jb,aa) * SlaterSign(Jb,bb);

                                    result += tau * HJI * CI * it->second;
                                    spawned++;
                                }
                            }
                            ndet_visited_++;
                        }
                    }
                }
            }
        }
    }
    return result;
}

void AdaptivePathIntegralCI::print_wfn(std::vector<BitsetDeterminant> space,std::vector<double> C)
{
    outfile->Printf("\n\n  Most important contributions to the wave function:");

    std::vector<std::pair<double,size_t> > det_weight;
    for (size_t I = 0; I < space.size(); ++I){
        det_weight.push_back(std::make_pair(std::fabs(C[I]),I));
    }
    std::sort(det_weight.begin(),det_weight.end());
    std::reverse(det_weight.begin(),det_weight.end());
    size_t max_dets = std::min(10,int(C.size()));
    for (size_t I = 0; I < max_dets; ++I){
        outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s",
                        I,
                        C[det_weight[I].second],
                        det_weight[I].first * det_weight[I].first,
                        det_weight[I].second,
                        space[det_weight[I].second].str().c_str());
    }
}

void combine_maps(std::vector<bsmap>& thread_det_C_map,bsmap& dets_C_map)
{
    // Combine the content of varius wave functions stored as maps
    for (size_t t = 0; t < thread_det_C_map.size(); ++t){
        for (bsmap_it it = thread_det_C_map[t].begin(), endit = thread_det_C_map[t].end(); it != endit; ++it){
            dets_C_map[it->first] += it->second;
        }
    }
}

void copy_map_to_vec(bsmap& dets_C_map,std::vector<BitsetDeterminant>& dets,std::vector<double>& C)
{
    size_t size = dets_C_map.size();
    dets.resize(size);
    C.resize(size);

    size_t I = 0;
    for (bsmap_it it = dets_C_map.begin(), endit = dets_C_map.end(); it != endit; ++it){
        dets[I] = it->first;
        C[I] = it->second;
        I++;
    }
}

void normalize(std::vector<double>& C)
{
    size_t size = C.size();
    double norm = 0.0;
    for (size_t I = 0; I < size; ++I){
        norm += C[I] * C[I];
    }
    norm = std::sqrt(norm);
    for (size_t I = 0; I < size; ++I){
        C[I] /= norm;
    }
}

//    size_t wfn_size = det_C_map.size();
//    dets.resize(wfn_size);
//    C.resize(wfn_size);
////        std::vector<double> C_tilde(wfn_size);

//    timer_on("APICI: collect C");
//    double new_estimate = 0.0;
//    double normerone = 1.0;
//    size_t I = 0;
//    double norm_C = 0.0;
//    for (bsmap_it it = det_C_map.begin(), endit = det_C_map.end(); it != endit; ++it){
//        dets[I] = it->first;
//        C[I] = it->second;
//        norm_C += (it->second) * (it->second);
//        double CI_n = old_space_map[it->first];
////            normerone += CI_n * CI_n;
//        new_estimate += CI_n * C[I];
//        I++;
//    }
//    new_estimate = nuclear_repulsion_energy_ + (normerone - new_estimate) / time_step_;


//    timer_off("APICI: collect C");

//    // Normalize C and save coefficients in the map
//    old_space_map.clear();
//    norm_C = std::sqrt(norm_C);
//    for (int I = 0; I < wfn_size; ++I){
////            C_tilde[I] = C[I];
//        C[I] /= norm_C;
//        old_space_map[dets[I]] = C[I];
//    }

}} // EndNamespaces

/*
 *
double AdaptivePathIntegralCI::time_step(double spawning_threshold,BitsetDeterminant& detI, double CI, std::map<BitsetDeterminant,double>& new_space_C, double E0)
{
    std::vector<int> aocc = detI.get_alfa_occ();
    std::vector<int> bocc = detI.get_beta_occ();
    std::vector<int> avir = detI.get_alfa_vir();
    std::vector<int> bvir = detI.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    double gradient_norm = 0.0;

    // Contribution of this determinant
    new_space_C[detI] += (1.0 - time_step_ * (detI.energy() - E0)) * CI;

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a){
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
                BitsetDeterminant detJ(detI);
                detJ.set_alfa_bit(ii,false);
                detJ.set_alfa_bit(aa,true);
                double HJI = detJ.slater_rules(detI);
                if (std::fabs(HJI * CI) >= spawning_threshold){
                    new_space_C[detJ] += -time_step_ * HJI * CI;
                    gradient_norm += std::fabs(-time_step_ * HJI * CI);
                    ndet_accepted_++;
                }
                ndet_visited_++;
            }
        }
    }

    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a){
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
                BitsetDeterminant detJ(detI);
                detJ.set_beta_bit(ii,false);
                detJ.set_beta_bit(aa,true);
                double HJI = detJ.slater_rules(detI);
                if (std::fabs(HJI * CI) >= spawning_threshold){
                    new_space_C[detJ] += -time_step_ * HJI * CI;
                    gradient_norm += std::fabs(-time_step_ * HJI * CI);
                    ndet_accepted_++;
                }
                ndet_visited_++;
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
                        double HJI_abs = ints_->aptei_aa(ii,jj,aa,bb);
                        if (std::fabs(HJI_abs * CI) >= spawning_threshold){
                            BitsetDeterminant detJ(detI);
                            detJ.set_alfa_bit(ii,false);
                            detJ.set_alfa_bit(jj,false);
                            detJ.set_alfa_bit(aa,true);
                            detJ.set_alfa_bit(bb,true);
                            double HJI = detJ.slater_rules(detI);
                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                new_space_C[detJ] += -time_step_ * HJI * CI;
                                gradient_norm += std::fabs(-time_step_ * HJI * CI);
                                ndet_accepted_++;
                            }
                        }
                        ndet_visited_++;
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
                        double HJI_abs = ints_->aptei_ab(ii,jj,aa,bb);
                        if (std::fabs(HJI_abs * CI) >= spawning_threshold){
                            BitsetDeterminant detJ(detI);
                            detJ.set_alfa_bit(ii,false);
                            detJ.set_beta_bit(jj,false);
                            detJ.set_alfa_bit(aa,true);
                            detJ.set_beta_bit(bb,true);
                            double HJI = detJ.slater_rules(detI);
                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                new_space_C[detJ] += -time_step_ * HJI * CI;
                                gradient_norm += std::fabs(-time_step_ * HJI * CI);
                                ndet_accepted_++;
                            }
                        }
                        ndet_visited_++;
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
                        double HJI_abs = ints_->aptei_bb(ii,jj,aa,bb);
                        if (std::fabs(HJI_abs * CI) >= spawning_threshold){
                            BitsetDeterminant detJ(detI);
                            detJ.set_beta_bit(ii,false);
                            detJ.set_beta_bit(jj,false);
                            detJ.set_beta_bit(aa,true);
                            detJ.set_beta_bit(bb,true);
                            double HJI = detJ.slater_rules(detI);
                            if (std::fabs(HJI * CI) >= spawning_threshold){
                                new_space_C[detJ] += -time_step_ * HJI * CI;
                                gradient_norm += std::fabs(-time_step_ * HJI * CI);
                                ndet_accepted_++;
                            }
                        }
                        ndet_visited_++;
                    }
                }
            }
        }
    }
    return gradient_norm;
}
 */
