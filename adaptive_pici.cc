
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

    spawning_threshold_ = options_.get_double("SPAWNING_THRESHOLD");
    initial_guess_spawning_threshold_ = options_.get_double("GUESS_SPAWNING_THRESHOLD");
    time_step_ = options_.get_double("TAU");
    variational_estimate_ = options_.get_bool("VAR_ESTIMATE");
    energy_estimate_freq_ = options_.get_int("ENERGY_ESTIMATE_FREQ");
    adaptive_beta_ = options_.get_bool("ADAPTIVE_BETA");
    e_convergence_ = options_.get_double("E_CONVERGENCE");
    do_shift_ = options_.get_bool("USE_SHIFT");
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
        {"Energy estimation frequency",energy_estimate_freq_}};

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"Time step (beta)",time_step_},
        {"Spawning threshold",spawning_threshold_},
        {"Initial guess spawning threshold",initial_guess_spawning_threshold_},
        {"Convergence threshold",e_convergence_}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"Compute variational estimate",variational_estimate_ ? "YES" : "NO"},
        {"Adaptive time step",adaptive_beta_ ? "YES" : "NO"},
        {"Shift the energy",do_shift_ ? "YES" : "NO"}
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
        outfile->Printf("\n    %-39s %7s",str_dim.first.c_str(),str_dim.second.c_str());
    }
    outfile->Printf("\n  %s",string(52,'-').c_str());
    outfile->Flush();
}


double AdaptivePathIntegralCI::compute_energy()
{
    boost::timer t_apici;
    outfile->Printf("\n\n  Adaptive Path-Integral CI");

    /// A vector of determinants in the P space
    std::vector<BitsetDeterminant> dets;
    std::vector<double> C;

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);

    double apici_energy = initial_guess(dets,C);

    print_wfn(dets,C);

    outfile->Printf("\n\n  ------------------------------------------------------------------------------------------");
    outfile->Printf("\n    Cycle      Ndets         Proj. Energy          Var. Energy   Delta E     beta   |grad|");
    outfile->Printf("\n  ------------------------------------------------------------------------------------------");

    double projective_energy_estimator = 0.0;
    double variational_energy_estimator = 0.0;

    int maxcycle = maxiter_;
    double shift = do_shift_ ? apici_energy : 0.0;
    double initial_gradient_norm = 0.0;
    double old_apici_energy = 0.0;
    double beta = 0.0;
    for (int cycle = 0; cycle < maxcycle; ++cycle){
        // The number of determinants visited in this iteration
        size_t ndet_visited = 0;
        double gradient_norm = 0.0;

        std::map<BitsetDeterminant,double> new_space_map;

        // Evaluate (1-beta H) |old>
        size_t max_I = dets.size();
        for (size_t I = 0; I < max_I; ++I){
            BitsetDeterminant& detI = dets[I];
            double CI = C[I];
            gradient_norm += time_step(spawning_threshold_,detI,CI,new_space_map,shift);
        }

        if (cycle == 0) initial_gradient_norm = gradient_norm;

        size_t wfn_size = new_space_map.size();
        dets.resize(wfn_size);
        C.resize(wfn_size);

        // Update the wave function
        size_t I = 0;
        double norm_C = 0.0;
        for (bsmap_it it = new_space_map.begin(), endit = new_space_map.end(); it != endit; ++it){
            dets[I] = it->first;
            C[I] = it->second;
            norm_C += (it->second) * (it->second);
            I++;
        }

        // Normalize C
        norm_C = std::sqrt(norm_C);
        for (int I = 0; I < wfn_size; ++I){
            C[I] /= norm_C;
        }

        if (variational_estimate_ and (cycle % energy_estimate_freq_ == 0)){
            SharedMatrix evecs(new Matrix("Eigenvectors",wfn_size,1));
            SharedVector evals(new Vector("Eigenvalues",wfn_size));
            sparse_solver.diagonalize_hamiltonian(dets,evals,evecs,1);
            double var_energy = evals->get(0) + nuclear_repulsion_energy_;
            outfile->Printf("\n  Variational energy = %20.12f",var_energy);
            Process::environment.globals["APICI-VAR ENERGY"] = var_energy;
        }

        // find the determinant with the largest value of C
        double maxC = 0.0;
        size_t maxI = 0;
        for (size_t I = 0; I < wfn_size; ++I){
            if (std::fabs(C[I]) > maxC){
                maxI = I;
                maxC = std::fabs(C[I]);
            }
        }



        // Compute the energy
        if (cycle % energy_estimate_freq_ == 0){
            // Compute a perturbative estimator of the energy
            projective_energy_estimator = 0.0;
            for (int I = 0; I < wfn_size; ++I){
                const BitsetDeterminant& detI = dets[I];
                double HI0 = detI.slater_rules(dets[maxI]);
                projective_energy_estimator += HI0 * C[I] / C[maxI];
            }
            projective_energy_estimator += nuclear_repulsion_energy_;

            // Compute a variational estimator of the energy
            variational_energy_estimator = 0.0;
            for (int I = 0; I < wfn_size; ++I){
                const BitsetDeterminant& detI = dets[I];
                for (int J = I; J < wfn_size; ++J){
                    const BitsetDeterminant& detJ = dets[J];
                    double HIJ = detI.slater_rules(detJ);
                    variational_energy_estimator += (I == J) ? C[I] * HIJ * C[J] : 2.0 * C[I] * HIJ * C[J];
                }
            }
            variational_energy_estimator += nuclear_repulsion_energy_;

            // Update the shift
            if (do_shift_){
                shift = projective_energy_estimator;
            }

            outfile->Printf("\n%9d %10zu %20.12f %20.12f %.3e %.6f %.6f",cycle,wfn_size,projective_energy_estimator,
                            variational_energy_estimator,std::fabs(variational_energy_estimator - old_apici_energy),beta,gradient_norm);

            apici_energy = variational_energy_estimator;

            // Check for convergence
            if (std::fabs(apici_energy - old_apici_energy) < e_convergence_){
                break;
            }

            old_apici_energy = apici_energy;

            if (adaptive_beta_){
                if (initial_gradient_norm / gradient_norm > 1.0)
                    time_step_ *= initial_gradient_norm / gradient_norm;
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

    outfile->Printf("\n\n  %s: %f s","Adaptive Path-Integral CI (bitset) ran in ",t_apici.elapsed());
    outfile->Flush();

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
    time_step(spawning_threshold_ * 10.0,bs_det,1.0,dets_C,0.0);

    // Save the list of determinants
    size_t I = 0;
    for (bsmap_it it = dets_C.begin(), endit = dets_C.end(); it != endit; ++it){
        dets.push_back(it->first);
        I++;
    }

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);

    size_t wfn_size = dets.size();
    SharedMatrix evecs(new Matrix("Eigenvectors",wfn_size,1));
    SharedVector evals(new Vector("Eigenvalues",wfn_size));
    sparse_solver.diagonalize_hamiltonian(dets,evals,evecs,1);
    double var_energy = evals->get(0) + nuclear_repulsion_energy_;
    outfile->Printf("\n  Variational energy = %20.12f",var_energy);

    // Copy the eigenvector
    for (int I = 0; I < wfn_size; ++I){
        C.push_back(evecs->get(I,0));
    }
    return var_energy;
}

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

    size_t ndet_visited = 0.0;
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
                if (std::fabs(HJI * CI) >= spawning_threshold){
                    new_space_C[detJ] += -time_step_ * HJI * CI;
                    gradient_norm += std::fabs(-time_step_ * HJI * CI);
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
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            new_space_C[detJ] += -time_step_ * HJI * CI;
                            gradient_norm += std::fabs(-time_step_ * HJI * CI);
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
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            new_space_C[detJ] += -time_step_ * HJI * CI;
                            gradient_norm += std::fabs(-time_step_ * HJI * CI);
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
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            new_space_C[detJ] += -time_step_ * HJI * CI;
                            gradient_norm += std::fabs(-time_step_ * HJI * CI);
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

}} // EndNamespaces


