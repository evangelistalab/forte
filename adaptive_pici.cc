
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

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

typedef std::map<BitsetDeterminant,double>::iterator bsmap_it;

AdaptivePathIntegralCI::AdaptivePathIntegralCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), options_(options), ints_(ints), variational_estimate_(false)
{
    // Copy the wavefunction information
    copy(wfn);

    startup();
    print_info();
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

    tau_ = options_.get_double("TAU");
    beta_ = options_.get_double("BETA");
    variational_estimate_ = options_.get_bool("VAR_ESTIMATE");
    var_estimate_freq_ = options_.get_int("VAR_ESTIMATE_FREQ");
    adaptive_beta_ = options_.get_bool("ADAPTIVE_BETA");
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
        {"Maximum number of iterations",maxiter_}};

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"Time step (beta)",beta_},
        {"C-threshold (tau)",tau_},
        {"Convergence threshold",options_.get_double("E_CONVERGENCE")}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"Compute variational estimate",variational_estimate_ ? "YES" : "NO"},
        {"Adaptive time step",adaptive_beta_ ? "YES" : "NO"}
    };
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


double AdaptivePathIntegralCI::compute_energy()
{
    boost::timer t_apici;
    outfile->Printf("\n\n  Adaptive Path-Integral CI");

    /// A vector of determinants in the P space
    std::vector<BitsetDeterminant> old_space;
    std::vector<BitsetDeterminant> new_space;
    std::vector<double> old_C;
    std::vector<double> new_C;

    // Use the reference determinant as a starting point
    std::vector<bool> alfa_bits = reference_determinant_.get_alfa_bits_vector_bool();
    std::vector<bool> beta_bits = reference_determinant_.get_beta_bits_vector_bool();
    BitsetDeterminant bs_det(alfa_bits,beta_bits);

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);

    old_space.push_back(bs_det);
    old_C.push_back(1.0);
    print_wfn(old_space,old_C);

    int maxcycle = maxiter_;
    double shift = bs_det.energy();
    double initial_gradient_norm = 0.0;

    double apici_energy = bs_det.energy();

    for (int cycle = 0; cycle < maxcycle; ++cycle){
        // The number of determinants visited in this iteration
        size_t ndet_visited = 0;
        double gradient_norm = 0.0;

        std::map<BitsetDeterminant,double> new_space_map;

        // Evaluate (1-beta H) |old>
        size_t max_I = old_space.size();
        for (size_t I = 0; I < max_I; ++I){
            BitsetDeterminant& detI = old_space[I];
            double CI = old_C[I];
            gradient_norm += time_step(detI,CI,new_space_map,shift);
        }

        if (cycle == 0) initial_gradient_norm = gradient_norm;

        size_t wfn_size = new_space_map.size();
        old_space.resize(wfn_size);
        old_C.resize(wfn_size);

        // Update the wave function
        size_t I = 0;
        double norm_C = 0.0;
        for (bsmap_it it = new_space_map.begin(), endit = new_space_map.end(); it != endit; ++it){
            old_space[I] = it->first;
            old_C[I] = it->second;
            norm_C += (it->second) * (it->second);
            I++;
        }

        // Normalize C
        norm_C = std::sqrt(norm_C);
        for (int I = 0; I < wfn_size; ++I){
            old_C[I] /= norm_C;
        }

        outfile->Printf("\n  Cycle %3d: %10zu determinants accepted, %10zu visited",cycle,wfn_size,ndet_visited);
        outfile->Printf("\n  Gradient norm: %f (%f)",gradient_norm,initial_gradient_norm);
        outfile->Printf("\n  Beta: %f",beta_);

        if (variational_estimate_ and (cycle % var_estimate_freq_ == 0)){
            SharedMatrix evecs(new Matrix("Eigenvectors",wfn_size,1));
            SharedVector evals(new Vector("Eigenvalues",wfn_size));
            sparse_solver.diagonalize_hamiltonian(old_space,evals,evecs,1);
            double var_energy = evals->get(0) + nuclear_repulsion_energy_;
            outfile->Printf("\n  Variational energy = %20.12f",var_energy);
            Process::environment.globals["APICI-VAR ENERGY"] = var_energy;
        }

        // find the determinant with the largest value of C
        double maxC = 0.0;
        size_t maxI = 0;
        for (size_t I = 0; I < wfn_size; ++I){
            if (std::fabs(old_C[I]) > maxC){
                maxI = I;
                maxC = std::fabs(old_C[I]);
            }
        }

        double energy_estimator = 0.0;
        for (int I = 0; I < wfn_size; ++I){
            const BitsetDeterminant& detI = old_space[I];
            double HI0 = detI.slater_rules(old_space[maxI]);
            energy_estimator += HI0 * old_C[I] / old_C[maxI];
        }
        shift = energy_estimator;
        energy_estimator += nuclear_repulsion_energy_;
        apici_energy = energy_estimator;
        outfile->Printf("\n  Estimated energy   = %20.12f",energy_estimator);


        if (adaptive_beta_){
            if (initial_gradient_norm / gradient_norm > 1.0)
                beta_ *= initial_gradient_norm / gradient_norm;
        }
        print_wfn(old_space,old_C);
        outfile->Flush();
    }

    Process::environment.globals["APICI ENERGY"] = apici_energy;

    outfile->Printf("\n\n  ==> Post-Iterations <==\n");
    outfile->Printf("\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,apici_energy);

    outfile->Printf("\n\n  %s: %f s","Adaptive Path-Integral CI (bitset) ran in ",t_apici.elapsed());
    outfile->Flush();

    return apici_energy;
}


double AdaptivePathIntegralCI::time_step(BitsetDeterminant& detI, double CI, std::map<BitsetDeterminant,double>& new_space_C, double E0)
{
    std::vector<int> aocc = detI.get_alfa_occ();
    std::vector<int> bocc = detI.get_beta_occ();
    std::vector<int> avir = detI.get_alfa_vir();
    std::vector<int> bvir = detI.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    size_t ndet_visited = 0.0;
    double gradient_norm = 0.0;

    // Contribution of this determinant
    new_space_C[detI] += (1.0 - beta_ * (detI.energy() - E0)) * CI;

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
                if (std::fabs(HJI * CI) >= tau_){
                    new_space_C[detJ] += -beta_ * HJI * CI;
                    gradient_norm += std::fabs(-beta_ * HJI * CI);
                    ndet_visited++;
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
                if (std::fabs(HJI * CI) >= tau_){
                    new_space_C[detJ] += -beta_ * HJI * CI;
                    gradient_norm += std::fabs(-beta_ * HJI * CI);
                    ndet_visited++;
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
                        BitsetDeterminant detJ(detI);
                        detJ.set_alfa_bit(ii,false);
                        detJ.set_alfa_bit(jj,false);
                        detJ.set_alfa_bit(aa,true);
                        detJ.set_alfa_bit(bb,true);
                        double HJI = detJ.slater_rules(detI);
                        if (std::fabs(HJI * CI) >= tau_){
                            new_space_C[detJ] += -beta_ * HJI * CI;
                            gradient_norm += std::fabs(-beta_ * HJI * CI);
                            ndet_visited++;
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
                        BitsetDeterminant detJ(detI);
                        detJ.set_alfa_bit(ii,false);
                        detJ.set_beta_bit(jj,false);
                        detJ.set_alfa_bit(aa,true);
                        detJ.set_beta_bit(bb,true);
                        double HJI = detJ.slater_rules(detI);
                        if (std::fabs(HJI * CI) >= tau_){
                            new_space_C[detJ] += -beta_ * HJI * CI;
                            gradient_norm += std::fabs(-beta_ * HJI * CI);
                            ndet_visited++;
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
                        BitsetDeterminant detJ(detI);
                        detJ.set_beta_bit(ii,false);
                        detJ.set_beta_bit(jj,false);
                        detJ.set_beta_bit(aa,true);
                        detJ.set_beta_bit(bb,true);
                        double HJI = detJ.slater_rules(detI);
                        if (std::fabs(HJI * CI) >= tau_){
                            new_space_C[detJ] += -beta_ * HJI * CI;
                            gradient_norm += std::fabs(-beta_ * HJI * CI);
                            ndet_visited++;
                        }
                    }
                }
            }
        }
    }
    return gradient_norm;
}

void AdaptivePathIntegralCI::print_wfn(std::vector<BitsetDeterminant> space,std::vector<double> C)
{
    outfile->Printf("\n\n  Most important contributions to wave function");

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




//bool AdaptivePathIntegralCI::check_convergence(std::vector<std::vector<double>>& energy_history,SharedVector evals)
//{
//    int nroot = evals->dim();

//    if (energy_history.size() == 0){
//        std::vector<double> new_energies;
//        for (int n = 0; n < nroot; ++ n){
//            double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
//            new_energies.push_back(state_n_energy);
//        }
//        energy_history.push_back(new_energies);
//        return false;
//    }

//    double old_avg_energy = 0.0;
//    double new_avg_energy = 0.0;

//    std::vector<double> new_energies;
//    std::vector<double> old_energies = energy_history[energy_history.size() - 1];
//    for (int n = 0; n < nroot; ++ n){
//        double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
//        new_energies.push_back(state_n_energy);
//        new_avg_energy += state_n_energy;
//        old_avg_energy += old_energies[n];
//    }
//    old_avg_energy /= static_cast<double>(nroot);
//    new_avg_energy /= static_cast<double>(nroot);

//    energy_history.push_back(new_energies);

//    // Check for convergence
//    return (std::fabs(new_avg_energy - old_avg_energy) < options_.get_double("E_CONVERGENCE"));
//    //        // Check the history of energies to avoid cycling in a loop
//    //        if(cycle > 3){
//    //            bool stuck = true;
//    //            for(int cycle_test = cycle - 2; cycle_test < cycle; ++cycle_test){
//    //                for (int n = 0; n < nroot_; ++n){
//    //                    if(std::fabs(energy_history[cycle_test][n] - energies[n]) < 1.0e-12){
//    //                        stuck = true;
//    //                    }
//    //                }
//    //            }
//    //            if(stuck) break; // exit the cycle
//    //        }
//}

//void AdaptivePathIntegralCI::prune_q_space(std::vector<BitsetDeterminant>& large_space,std::vector<BitsetDeterminant>& pruned_space,
//                               std::map<BitsetDeterminant,int>& pruned_space_map,SharedMatrix evecs,int nroot)
//{
//    // Create a vector that stores the absolute value of the CI coefficients
//    std::vector<std::pair<double,size_t> > dm_det_list;

//    for (size_t I = 0; I < large_space.size(); ++I){
//        double max_dm = 0.0;
//        for (int n = 0; n < nroot; ++n){
//            max_dm = std::max(max_dm,std::fabs(evecs->get(I,n)));
//        }
//        dm_det_list.push_back(std::make_pair(max_dm,I));
//    }

//    // Sort the CI coefficients
//    std::sort(dm_det_list.begin(),dm_det_list.end());
//    std::reverse(dm_det_list.begin(),dm_det_list.end());

//    // Select the new reference space using the sorted CI coefficients
//    pruned_space.clear();
//    pruned_space_map.clear();

//    // Decide which will go in pruned_space
//    for (size_t I = 0; I < large_space.size(); ++I){
//        if (dm_det_list[I].first > tau_p_){
//            pruned_space.push_back(large_space[dm_det_list[I].second]);
//            pruned_space_map[large_space[dm_det_list[I].second]] = 1;
//        }
//    }
//}


//void AdaptivePathIntegralCI::find_q_space(int nroot,SharedVector evals,SharedMatrix evecs)
//{
//    // Find the SD space out of the reference
//    std::vector<BitsetDeterminant> sd_dets_vec;
//    std::map<BitsetDeterminant,int> new_dets_map;

//    boost::timer t_ms_build;

//    // This hash saves the determinant coupling to the model space eigenfunction
//    std::map<BitsetDeterminant,std::vector<double> > V_hash;

//    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I){
//        BitsetDeterminant& det = P_space_[I];
//        generate_excited_determinants(nroot,I,evecs,det,V_hash);
//    }
//    outfile->Printf("\n  %s: %zu determinants","Dimension of the SD space",V_hash.size());
//    outfile->Printf("\n  %s: %f s\n","Time spent building the model space",t_ms_build.elapsed());
//    outfile->Flush();

//    // This will contain all the determinants
//    PQ_space_.clear();

//    // Add  the P-space determinants
//    for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J){
//        PQ_space_.push_back(P_space_[J]);
//    }

//    boost::timer t_ms_screen;

//    typedef std::map<BitsetDeterminant,std::vector<double> >::iterator bsmap_it;
//    std::vector<std::pair<double,double> > C1(nroot_,make_pair(0.0,0.0));
//    std::vector<std::pair<double,double> > E2(nroot_,make_pair(0.0,0.0));
//    std::vector<double> ept2(nroot_,0.0);

//    // Check the coupling between the reference and the SD space
//    for (bsmap_it it = V_hash.begin(), endit = V_hash.end(); it != endit; ++it){
//        double EI = it->first.energy();
//        for (int n = 0; n < nroot; ++n){
//            double V = it->second[n];
//            double C1_I = -V / (EI - evals->get(n));
//            double E2_I = -V * V / (EI - evals->get(n));

//            C1[n] = make_pair(std::fabs(C1_I),C1_I);
//            E2[n] = make_pair(std::fabs(E2_I),E2_I);
//        }

//        std::pair<double,double> max_C1 = *std::max_element(C1.begin(),C1.end());
//        std::pair<double,double> max_E2 = *std::max_element(E2.begin(),E2.end());

//        double select_value = energy_selection_ ? max_E2.first : max_C1.first;

//        if (std::fabs(select_value) > tau_q_){
//            PQ_space_.push_back(it->first);
//        }else{
//            for (int n = 0; n < nroot; ++n){
//                ept2[n] += E2[n].second;
//            }
//        }
//    }

//    multistate_pt2_energy_correction_ = ept2;

//    outfile->Printf("\n  %s: %zu determinants","Dimension of the P + Q space",PQ_space_.size());
//    outfile->Printf("\n  %s: %f s","Time spent screening the model space",t_ms_screen.elapsed());
//    outfile->Flush();
//}

}} // EndNamespaces


