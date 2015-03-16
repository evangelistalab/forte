#include "lambda-ci.h"

#include <cmath>
#include <functional>
#include <algorithm>
#include <unordered_map>

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

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{


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

    do_smooth_ = options_.get_bool("SMOOTH");
    smooth_threshold_ = options_.get_double("SMOOTH_THRESHOLD");

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

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"Determinant selection criterion",energy_selection_ ? "Second-order Energy" : "First-order Coefficients"},
        {"Selection criterion",aimed_selection_ ? "Aimed selection" : "Threshold"}};
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
    sparse_solver.set_parallel(true);

    int maxcycle = 20;
    for (int cycle = 0; cycle < maxcycle; ++cycle){
        // Step 1. Diagonalize the Hamiltonian in the P space
        int num_ref_roots = std::min(nroot_,int(P_space_.size()));

        outfile->Printf("\n\n  Cycle %3d",cycle);
        outfile->Printf("\n  %s: %zu determinants","Dimension of the P space",P_space_.size());
        outfile->Flush();

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

        // Step 5. Prune the P + Q space to get an updated P space
        prune_q_space(PQ_space_,P_space_,P_space_map_,PQ_evecs,nroot_);        

        // Print information about the wave function
        print_wfn(PQ_space_,PQ_evecs,nroot_);
    }

    // Do Hamiltonian smoothing
    if (do_smooth_){
        smooth_hamiltonian(P_space_,P_evals,P_evecs,nroot_);
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

    double root_energy = PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_;
    double root_energy_pt2 = root_energy + multistate_pt2_energy_correction_[options_.get_int("ROOT")];
    Process::environment.globals["CURRENT ENERGY"] = root_energy;
    Process::environment.globals["ACI ENERGY"] = root_energy;
    Process::environment.globals["ACI+PT2 ENERGY"] = root_energy_pt2;

    return PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_;
}


void AdaptiveCI::find_q_space(int nroot,SharedVector evals,SharedMatrix evecs)
{
    // Find the SD space out of the reference
    std::vector<BitsetDeterminant> sd_dets_vec;
    std::map<BitsetDeterminant,int> new_dets_map;

    boost::timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    std::map<BitsetDeterminant,double> V_map;

    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I){
        BitsetDeterminant& det = P_space_[I];
        generate_excited_determinants(nroot,I,evecs,det,V_map);
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

    typedef std::map<BitsetDeterminant,std::vector<double> >::iterator bsmap_it;
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

/*
void AdaptiveCI::find_q_space(int nroot,SharedVector evals,SharedMatrix evecs)
{
    // Find the SD space out of the reference
    std::vector<BitsetDeterminant> sd_dets_vec;
    std::map<BitsetDeterminant,int> new_dets_map;

    boost::timer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    std::map<BitsetDeterminant,std::vector<double> > V_hash;

    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I){
        BitsetDeterminant& det = P_space_[I];
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

    typedef std::map<BitsetDeterminant,std::vector<double> >::iterator bsmap_it;
    std::vector<std::pair<double,double> > C1(nroot_,make_pair(0.0,0.0));
    std::vector<std::pair<double,double> > E2(nroot_,make_pair(0.0,0.0));
    std::vector<double> ept2(nroot_,0.0);

    std::vector<std::pair<double,BitsetDeterminant>> sorted_dets;

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

        if (aimed_selection_){
            double aimed_value = energy_selection_ ? max_E2.first : std::pow(max_C1.first,2.0);
            sorted_dets.push_back(std::make_pair(aimed_value,it->first));
        }else{
            double select_value = energy_selection_ ? max_E2.first : max_C1.first;
            if (std::fabs(select_value) > tau_q_){
                PQ_space_.push_back(it->first);
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
                const std::vector<double>& V_vec = V_hash[det];
                for (int n = 0; n < nroot; ++n){
                    double V = V_vec[n];
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
*/

void AdaptiveCI::generate_excited_determinants(int nroot,int I,SharedMatrix evecs,BitsetDeterminant& det,std::map<BitsetDeterminant,double>& V_hash)
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
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
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
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
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
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
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
                        HIJ *= SlaterSign(Ia,ii) * SlaterSign(Ia,jj) * SlaterSign(Ja,aa) * SlaterSign(Ja,bb);

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
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
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
                        HIJ *= SlaterSign(Ia,ii) * SlaterSign(Ib,jj) * SlaterSign(Ja,aa) * SlaterSign(Jb,bb);

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
                    if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == wavefunction_symmetry_){
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
                        HIJ *= SlaterSign(Ib,ii) * SlaterSign(Ib,jj) * SlaterSign(Jb,aa) * SlaterSign(Jb,bb);

                        V_hash[new_det] += HIJ * evecs->get(I,n);
                    }
                }
            }
        }
    }
}

void AdaptiveCI::generate_excited_determinants_original(int nroot,int I,SharedMatrix evecs,BitsetDeterminant& det,std::map<BitsetDeterminant,std::vector<double>>& V_hash)
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

void AdaptiveCI::generate_pair_excited_determinants(int nroot,int I,SharedMatrix evecs,BitsetDeterminant& det,std::map<BitsetDeterminant,std::vector<double>>& V_hash)
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
    // Select the new reference space using the sorted CI coefficients
    pruned_space.clear();
    pruned_space_map.clear();

    // Create a vector that stores the absolute value of the CI coefficients
    std::vector<std::pair<double,size_t> > dm_det_list;
    for (size_t I = 0; I < large_space.size(); ++I){
        double max_dm = 0.0;
        for (int n = 0; n < nroot; ++n){
            max_dm = std::max(max_dm,std::fabs(evecs->get(I,n)));
        }
        dm_det_list.push_back(std::make_pair(max_dm,I));
    }

    // Decide which determinants will go in pruned_space
    // Include all determinants such that
    // sum_I |C_I|^2 < tau_p, where the sum runs over all the excluded determinants
    if (aimed_selection_){
        // Sort the CI coefficients in ascending order
        outfile->Printf("AIMED SELECTION");
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
        for (size_t I = 0; I < large_space.size(); ++I){
            if (dm_det_list[I].first > tau_p_){
                pruned_space.push_back(large_space[dm_det_list[I].second]);
                pruned_space_map[large_space[dm_det_list[I].second]] = 1;
            }
        }
    }
}


void AdaptiveCI::smooth_hamiltonian(std::vector<BitsetDeterminant>& space,SharedVector evals,SharedMatrix evecs,int nroot)
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

}} // EndNamespaces


