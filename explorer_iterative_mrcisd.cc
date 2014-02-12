#include "explorer.h"

#include <cmath>
#include <functional>
#include <algorithm>
#include <unordered_map>

#include <boost/timer.hpp>
#include <boost/format.hpp>

#include <libqt/qt.h>


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <libciomr/libciomr.h>
//#include <libqt/qt.h>

#include "explorer.h"
#include "cartographer.h"
#include "string_determinant.h"
#include "bitset_determinant.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

/**
 * Diagonalize the
 */
void Explorer::iterative_adaptive_mrcisd(psi::Options& options)
{
    boost::timer t_iamrcisd;

    fprintf(outfile,"\n\n  Iterative Adaptive MRCISD");

    int nroot = options.get_int("NROOT");

    double tau_p = options.get_double("TAUP");
    double tau_q = options.get_double("TAUQ");

    fprintf(outfile,"\n\n  TAU_P = %f Eh",tau_p);
    fprintf(outfile,"\n  TAU_Q = %.12f Eh\n",tau_q);


    double ia_mrcisd_threshold = 1.0e-9;


    bool aimed_selection = false;
    bool energy_select = false;
    if (options.get_str("SELECT_TYPE") == "AIMED_AMP"){
        aimed_selection = true;
        energy_select = false;
    }else if (options.get_str("SELECT_TYPE") == "AIMED_ENERGY"){
        aimed_selection = true;
        energy_select = true;
    }else if(options.get_str("SELECT_TYPE") == "ENERGY"){
        aimed_selection = false;
        energy_select = true;
    }else if(options.get_str("SELECT_TYPE") == "AMP"){
        aimed_selection = false;
        energy_select = false;
    }

    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    std::vector<StringDeterminant> ref_space;
    std::map<StringDeterminant,int> ref_space_map;
    ref_space.push_back(reference_determinant_);
    ref_space_map[reference_determinant_] = 1;


    double old_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_energy = 0.0;

    int maxcycle = 20;
    for (int cycle = 0; cycle < maxcycle; ++cycle){
        // Build the Hamiltonian in the P space

        size_t dim_ref_space = ref_space.size();

        fprintf(outfile,"\n\n  Cycle %3d. The model space contains %zu determinants",cycle,dim_ref_space);
        fflush(outfile);

        H.reset(new Matrix("Hamiltonian Matrix",dim_ref_space,dim_ref_space));
        evecs.reset(new Matrix("U",dim_ref_space,nroot));
        evals.reset(new Vector("e",nroot));

        boost::timer t_h_build;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < dim_ref_space; ++I){
            const StringDeterminant& detI = ref_space[I];
            for (size_t J = I; J < dim_ref_space; ++J){
                const StringDeterminant& detJ = ref_space[J];
                double HIJ = detI.slater_rules(detJ);
                H->set(I,J,HIJ);
                H->set(J,I,HIJ);
            }
        }
        fprintf(outfile,"\n  Time spent building H               = %f s",t_h_build.elapsed());
        fflush(outfile);

        // Diagonalize the Hamiltonian
        boost::timer t_hdiag_large;
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
            fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H,evals,evecs,nroot);
        }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
            fprintf(outfile,"\n  Performing full diagonalization.");
            H->diagonalize(evecs,evals);
        }

        fprintf(outfile,"\n  Time spent diagonalizing H          = %f s",t_hdiag_large.elapsed());
        fflush(outfile);

        // Print the energy
        for (int i = 0; i < nroot; ++ i){
            fprintf(outfile,"\n  P-space CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
        }
        fflush(outfile);


        int nmo = reference_determinant_.nmo();
        size_t nfrzc = frzc_.size();
        size_t nfrzv = frzv_.size();

        std::vector<int> aocc(nalpha_ - nfrzc);
        std::vector<int> bocc(nbeta_ - nfrzc);
        std::vector<int> avir(nmo_ - nalpha_ - nfrzv);
        std::vector<int> bvir(nmo_ - nbeta_ - nfrzv);

        int noalpha = nalpha_ - nfrzc;
        int nobeta  = nbeta_ - nfrzc;
        int nvalpha = nmo_ - nalpha_;
        int nvbeta  = nmo_ - nbeta_;

        // Find the SD space out of the reference
        std::vector<StringDeterminant> sd_dets_vec;
        std::map<StringDeterminant,int> new_dets_map;
        boost::timer t_ms_build;

        for (size_t I = 0, max_I = ref_space_map.size(); I < max_I; ++I){
            const StringDeterminant& det = ref_space[I];
            for (int p = 0, i = 0, a = 0; p < nmo_; ++p){
                if (det.get_alfa_bit(p)){
                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                        aocc[i] = p;
                        i++;
                    }
                }else{
                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                        avir[a] = p;
                        a++;
                    }
                }
            }
            for (int p = 0, i = 0, a = 0; p < nmo_; ++p){
                if (det.get_beta_bit(p)){
                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                        bocc[i] = p;
                        i++;
                    }
                }else{
                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                        bvir[a] = p;
                        a++;
                    }
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
                        if(ref_space_map.find(new_det) == ref_space_map.end()){
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
                        if(ref_space_map.find(new_det) == ref_space_map.end()){
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                                StringDeterminant new_det(det);
                                new_det.set_alfa_bit(ii,false);
                                new_det.set_alfa_bit(jj,false);
                                new_det.set_alfa_bit(aa,true);
                                new_det.set_alfa_bit(bb,true);
                                if(ref_space_map.find(new_det) == ref_space_map.end()){
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                                StringDeterminant new_det(det);
                                new_det.set_alfa_bit(ii,false);
                                new_det.set_beta_bit(jj,false);
                                new_det.set_alfa_bit(aa,true);
                                new_det.set_beta_bit(bb,true);
                                if(ref_space_map.find(new_det) == ref_space_map.end()){
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                                StringDeterminant new_det(det);
                                new_det.set_beta_bit(ii,false);
                                new_det.set_beta_bit(jj,false);
                                new_det.set_beta_bit(aa,true);
                                new_det.set_beta_bit(bb,true);
                                if(ref_space_map.find(new_det) == ref_space_map.end()){
                                    sd_dets_vec.push_back(new_det);
                                }
                            }
                        }
                    }
                }
            }
        }

        fprintf(outfile,"\n  The SD excitation space has dimension: %zu",sd_dets_vec.size());

        boost::timer t_ms_screen;

        sort( sd_dets_vec.begin(), sd_dets_vec.end() );
        sd_dets_vec.erase( unique( sd_dets_vec.begin(), sd_dets_vec.end() ), sd_dets_vec.end() );

        fprintf(outfile,"\n  The SD excitation space has dimension: %zu (unique)",sd_dets_vec.size());
        fprintf(outfile,"\n  Time spent building the model space = %f s",t_ms_build.elapsed());
        fflush(outfile);

        // This will contain all the determinants
        std::vector<StringDeterminant> ref_sd_dets;

        // Add  the P-space determinants
        for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J){
            ref_sd_dets.push_back(ref_space[J]);
        }


        // Check the coupling between the reference and the SD space
        std::vector<std::pair<double,size_t> > new_dets_importance_vec;

        std::vector<double> V(nroot,0.0);
        std::vector<std::pair<double,double> > C1(nroot,make_pair(0.0,0.0));
        std::vector<std::pair<double,double> > E2(nroot,make_pair(0.0,0.0));
        std::vector<double> ept2(nroot,0.0);

        double aimed_selection_sum = 0.0;

        for (size_t I = 0, max_I = sd_dets_vec.size(); I < max_I; ++I){
            double EI = sd_dets_vec[I].energy();
            for (int n = 0; n < nroot; ++n){
                V[n] = 0;
            }
            for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J){
                double HIJ = sd_dets_vec[I].slater_rules(ref_space[J]);
                for (int n = 0; n < nroot; ++n){
                    V[n] += evecs->get(J,n) * HIJ;
                }
            }
            for (int n = 0; n < nroot; ++n){
                double C1_I = -V[n] / (EI - evals->get(n));
                double E2_I = -V[n] * V[n] / (EI - evals->get(n));
                C1[n] = make_pair(std::fabs(C1_I),C1_I);
                E2[n] = make_pair(std::fabs(E2_I),E2_I);
            }

            std::pair<double,double> max_C1 = *std::max_element(C1.begin(),C1.end());
            std::pair<double,double> max_E2 = *std::max_element(E2.begin(),E2.end());

            double select_value = energy_select ? max_E2.first : max_C1.first;

            // Do not select now, just store the determinant index and the selection criterion
            if(aimed_selection){
                if (energy_select){
                    new_dets_importance_vec.push_back(std::make_pair(select_value,I));
                    aimed_selection_sum += select_value;
                }else{
                    new_dets_importance_vec.push_back(std::make_pair(select_value * select_value,I));
                    aimed_selection_sum += select_value * select_value;
                }
            }else{
                if (std::fabs(select_value) > tau_q){
                    new_dets_importance_vec.push_back(std::make_pair(select_value,I));
                }else{
                    for (int n = 0; n < nroot; ++n) ept2[n] += E2[n].second;
                }
            }
        }

        if(aimed_selection){
            std::sort(new_dets_importance_vec.begin(),new_dets_importance_vec.end());
            std::reverse(new_dets_importance_vec.begin(),new_dets_importance_vec.end());
            size_t maxI = new_dets_importance_vec.size();
            fprintf(outfile,"\n  The SD space will be generated using the aimed scheme (%s)",energy_select ? "energy" : "amplitude");
            fprintf(outfile,"\n  Initial value of sigma in the aimed selection = %24.14f",aimed_selection_sum);
            for (size_t I = 0; I < maxI; ++I){
                if (aimed_selection_sum > t2_threshold_){
                    ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
                    aimed_selection_sum -= new_dets_importance_vec[I].first;
                }else{
                    break;
                }
            }
            fprintf(outfile,"\n  Final value of sigma in the aimed selection   = %24.14f",aimed_selection_sum);
            fprintf(outfile,"\n  Selected %zu determinants",ref_sd_dets.size()-ref_space.size());
        }else{
            fprintf(outfile,"\n  The SD space will be generated by screening (%s)",energy_select ? "energy" : "amplitude");
            size_t maxI = new_dets_importance_vec.size();
            for (size_t I = 0; I < maxI; ++I){
                ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
            }
        }

        multistate_pt2_energy_correction_ = ept2;

        size_t dim_ref_sd_dets = ref_sd_dets.size();

        fprintf(outfile,"\n  After screening the ia-MRCISD space contains %zu determinants",dim_ref_sd_dets);
        fprintf(outfile,"\n  Time spent screening the model space = %f s",t_ms_screen.elapsed());
        fflush(outfile);


        evecs.reset(new Matrix("U",dim_ref_sd_dets,nroot));
        evals.reset(new Vector("e",nroot));
        // Full algorithm
        if (options.get_str("ENERGY_TYPE") == "IMRCISD"){
            H.reset(new Matrix("Hamiltonian Matrix",dim_ref_sd_dets,dim_ref_sd_dets));

            boost::timer t_h_build2;
#pragma omp parallel for schedule(dynamic)
            for (size_t I = 0; I < dim_ref_sd_dets; ++I){
                const StringDeterminant& detI = ref_sd_dets[I];
                for (size_t J = I; J < dim_ref_sd_dets; ++J){
                    const StringDeterminant& detJ = ref_sd_dets[J];
                    double HIJ = detI.slater_rules(detJ);
                    H->set(I,J,HIJ);
                    H->set(J,I,HIJ);
                }
            }
            fprintf(outfile,"\n  Time spent building H               = %f s",t_h_build2.elapsed());
            fflush(outfile);

            // 4) Diagonalize the Hamiltonian
            boost::timer t_hdiag_large2;
            if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
                fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
                davidson_liu(H,evals,evecs,nroot);
            }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
                fprintf(outfile,"\n  Performing full diagonalization.");
                H->diagonalize(evecs,evals);
            }

            fprintf(outfile,"\n  Time spent diagonalizing H          = %f s",t_hdiag_large2.elapsed());
            fflush(outfile);
        }
        // Sparse algorithm
        else{
            boost::timer t_h_build2;
            std::vector<std::vector<std::pair<int,double> > > H_sparse;

            size_t num_nonzero = 0;
            // Form the Hamiltonian matrix
            for (size_t I = 0; I < dim_ref_sd_dets; ++I){
                std::vector<std::pair<int,double> > H_row;
                const StringDeterminant& detI = ref_sd_dets[I];
                double HII = detI.slater_rules(detI);
                H_row.push_back(make_pair(int(I),HII));
                for (size_t J = 0; J < dim_ref_sd_dets; ++J){
                    if (I != J){
                        const StringDeterminant& detJ = ref_sd_dets[J];
                        double HIJ = detI.slater_rules(detJ);
                        if (std::fabs(HIJ) >= 1.0e-12){
                            H_row.push_back(make_pair(int(J),HIJ));
                            num_nonzero += 1;
                        }
                    }
                }
                H_sparse.push_back(H_row);
            }

            fprintf(outfile,"\n  %ld nonzero elements out of %ld (%e)",num_nonzero,size_t(dim_ref_sd_dets * dim_ref_sd_dets),double(num_nonzero)/double(dim_ref_sd_dets * dim_ref_sd_dets));
            fprintf(outfile,"\n  Time spent building H               = %f s",t_h_build2.elapsed());
            fflush(outfile);

            // 4) Diagonalize the Hamiltonian
            boost::timer t_hdiag_large2;
            fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
            davidson_liu_sparse(H_sparse,evals,evecs,nroot);
            fprintf(outfile,"\n  Time spent diagonalizing H          = %f s",t_hdiag_large2.elapsed());
            fflush(outfile);
        }

        //
        for (int i = 0; i < nroot; ++ i){
            fprintf(outfile,"\n  Adaptive CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
            fprintf(outfile,"\n  Adaptive CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_ + multistate_pt2_energy_correction_[i],
                    27.211 * (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
        }
        fflush(outfile);


        // Select the new reference space
        ref_space.clear();
        ref_space_map.clear();

        new_energy = evals->get(0) + nuclear_repulsion_energy_;

        fflush(outfile);
        if (std::fabs(new_energy - old_energy) < ia_mrcisd_threshold){
            break;
        }
        old_energy = new_energy;

        std::vector<std::pair<double,size_t> > dm_det_list;

        for (size_t I = 0; I < dim_ref_sd_dets; ++I){
            double max_dm = 0.0;
            for (int n = 0; n < nroot; ++n){
                max_dm = std::max(max_dm,std::fabs(evecs->get(I,n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm,I));
        }

        std::sort(dm_det_list.begin(),dm_det_list.end());
        std::reverse(dm_det_list.begin(),dm_det_list.end());

        // Decide which will go in ref_space
        for (size_t I = 0; I < dim_ref_sd_dets; ++I){
            if (dm_det_list[I].first > tau_p){
                ref_space.push_back(ref_sd_dets[dm_det_list[I].second]);
                ref_space_map[ref_sd_dets[dm_det_list[I].second]] = 1;
            }
        }
        unordered_map<std::vector<bool>,int> a_str_hash;
        unordered_map<std::vector<bool>,int> b_str_hash;

        boost::timer t_stringify;
        for (size_t I = 0; I < dim_ref_sd_dets; ++I){
            const StringDeterminant& detI = ref_sd_dets[I];
            const std::vector<bool> a_str = detI.get_alfa_bits_vector_bool();
            const std::vector<bool> b_str = detI.get_beta_bits_vector_bool();
            a_str_hash[a_str] = 1;
            b_str_hash[b_str] = 1;
        }
        fprintf(outfile,"\n  Size of the @MRCISD space: %zu",dim_ref_sd_dets);
        fprintf(outfile,"\n  Size of the alpha strings: %zu",a_str_hash.size());
        fprintf(outfile,"\n  Size of the beta  strings: %zu",b_str_hash.size());

        fprintf(outfile,"\n\n  Time to stringify: %f s",t_stringify.elapsed());


    }

    for (int i = 0; i < nroot; ++ i){
        fprintf(outfile,"\n  * IA-MRCISD total energy (%3d)        = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
        fprintf(outfile,"\n  * IA-MRCISD total energy (%3d) + EPT2 = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_ + multistate_pt2_energy_correction_[i],
                27.211 * (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
    }

    fprintf(outfile,"\n\n  iterative_adaptive_mrcisd        ran in %f s",t_iamrcisd.elapsed());
    fflush(outfile);


}


/**
 * Diagonalize the
 */
void Explorer::iterative_adaptive_mrcisd_bitset(psi::Options& options)
{
    boost::timer t_iamrcisd;
    fprintf(outfile,"\n\n  Iterative Adaptive MRCISD");

    int nroot = options.get_int("NROOT");

    double tau_p = options.get_double("TAUP");
    double tau_q = options.get_double("TAUQ");

    fprintf(outfile,"\n\n  TAU_P = %f Eh",tau_p);
    fprintf(outfile,"\n  TAU_Q = %.12f Eh\n",tau_q);
    fflush(outfile);

    double ia_mrcisd_threshold = 1.0e-9;


    bool aimed_selection = false;
    bool energy_select = false;
    if (options.get_str("SELECT_TYPE") == "AIMED_AMP"){
        aimed_selection = true;
        energy_select = false;
    }else if (options.get_str("SELECT_TYPE") == "AIMED_ENERGY"){
        aimed_selection = true;
        energy_select = true;
    }else if(options.get_str("SELECT_TYPE") == "ENERGY"){
        aimed_selection = false;
        energy_select = true;
    }else if(options.get_str("SELECT_TYPE") == "AMP"){
        aimed_selection = false;
        energy_select = false;
    }

    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    std::vector<BitsetDeterminant> ref_space;
    std::map<BitsetDeterminant,int> ref_space_map;
    std::vector<bool> ref_abits = reference_determinant_.get_alfa_bits_vector_bool();
    std::vector<bool> ref_bbits = reference_determinant_.get_beta_bits_vector_bool();
    BitsetDeterminant bs_ref_(ref_abits,ref_bbits);
    ref_space.push_back(bs_ref_);
    ref_space_map[bs_ref_] = 1;


    double old_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_energy = 0.0;

    int maxcycle = 20;
    for (int cycle = 0; cycle < maxcycle; ++cycle){
        // Build the Hamiltonian in the P space

        size_t dim_ref_space = ref_space.size();

        fprintf(outfile,"\n\n  Cycle %3d. The model space contains %zu determinants",cycle,dim_ref_space);
        fflush(outfile);

        H.reset(new Matrix("Hamiltonian Matrix",dim_ref_space,dim_ref_space));
        evecs.reset(new Matrix("U",dim_ref_space,nroot));
        evals.reset(new Vector("e",nroot));

        boost::timer t_h_build;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < dim_ref_space; ++I){
            const BitsetDeterminant& detI = ref_space[I];
            for (size_t J = I; J < dim_ref_space; ++J){
                const BitsetDeterminant& detJ = ref_space[J];
                double HIJ = detI.slater_rules(detJ);
                H->set(I,J,HIJ);
                H->set(J,I,HIJ);
            }
        }
        fprintf(outfile,"\n  Time spent building H               = %f s",t_h_build.elapsed());
        fflush(outfile);

        // Diagonalize the Hamiltonian
        boost::timer t_hdiag_large;
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
            fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H,evals,evecs,nroot);
        }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
            fprintf(outfile,"\n  Performing full diagonalization.");
            H->diagonalize(evecs,evals);
        }

        fprintf(outfile,"\n  Time spent diagonalizing H          = %f s",t_hdiag_large.elapsed());
        fflush(outfile);

        // Print the energy
        for (int i = 0; i < nroot; ++ i){
            fprintf(outfile,"\n  P-space CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
        }
        fflush(outfile);


        int nmo = reference_determinant_.nmo();
        size_t nfrzc = frzc_.size();
        size_t nfrzv = frzv_.size();

        std::vector<int> aocc(nalpha_ - nfrzc);
        std::vector<int> bocc(nbeta_ - nfrzc);
        std::vector<int> avir(nmo_ - nalpha_ - nfrzv);
        std::vector<int> bvir(nmo_ - nbeta_ - nfrzv);

        int noalpha = nalpha_ - nfrzc;
        int nobeta  = nbeta_ - nfrzc;
        int nvalpha = nmo_ - nalpha_;
        int nvbeta  = nmo_ - nbeta_;

        // Find the SD space out of the reference
        std::vector<BitsetDeterminant> sd_dets_vec;
        std::map<BitsetDeterminant,int> new_dets_map;
        boost::timer t_ms_build;

        for (size_t I = 0, max_I = ref_space_map.size(); I < max_I; ++I){
            const BitsetDeterminant& det = ref_space[I];
            for (int p = 0, i = 0, a = 0; p < nmo_; ++p){
                if (det.get_alfa_bit(p)){
                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                        aocc[i] = p;
                        i++;
                    }
                }else{
                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                        avir[a] = p;
                        a++;
                    }
                }
            }
            for (int p = 0, i = 0, a = 0; p < nmo_; ++p){
                if (det.get_beta_bit(p)){
                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                        bocc[i] = p;
                        i++;
                    }
                }else{
                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                        bvir[a] = p;
                        a++;
                    }
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
                        if(ref_space_map.find(new_det) == ref_space_map.end()){
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
                        BitsetDeterminant new_det(det);
                        new_det.set_beta_bit(ii,false);
                        new_det.set_beta_bit(aa,true);
                        if(ref_space_map.find(new_det) == ref_space_map.end()){
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                                BitsetDeterminant new_det(det);
                                new_det.set_alfa_bit(ii,false);
                                new_det.set_alfa_bit(jj,false);
                                new_det.set_alfa_bit(aa,true);
                                new_det.set_alfa_bit(bb,true);
                                if(ref_space_map.find(new_det) == ref_space_map.end()){
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                                BitsetDeterminant new_det(det);
                                new_det.set_alfa_bit(ii,false);
                                new_det.set_beta_bit(jj,false);
                                new_det.set_alfa_bit(aa,true);
                                new_det.set_beta_bit(bb,true);
                                if(ref_space_map.find(new_det) == ref_space_map.end()){
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
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                                BitsetDeterminant new_det(det);
                                new_det.set_beta_bit(ii,false);
                                new_det.set_beta_bit(jj,false);
                                new_det.set_beta_bit(aa,true);
                                new_det.set_beta_bit(bb,true);
                                if(ref_space_map.find(new_det) == ref_space_map.end()){
                                    sd_dets_vec.push_back(new_det);
                                }
                            }
                        }
                    }
                }
            }
        }

        fprintf(outfile,"\n  The SD excitation space has dimension: %zu",sd_dets_vec.size());

        boost::timer t_ms_screen;

        sort( sd_dets_vec.begin(), sd_dets_vec.end() );
        sd_dets_vec.erase( unique( sd_dets_vec.begin(), sd_dets_vec.end() ), sd_dets_vec.end() );

        fprintf(outfile,"\n  The SD excitation space has dimension: %zu (unique)",sd_dets_vec.size());
        fprintf(outfile,"\n  Time spent building the model space = %f s",t_ms_build.elapsed());
        fflush(outfile);

        // This will contain all the determinants
        std::vector<BitsetDeterminant> ref_sd_dets;

        // Add  the P-space determinants
        for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J){
            ref_sd_dets.push_back(ref_space[J]);
        }


        // Check the coupling between the reference and the SD space
        std::vector<std::pair<double,size_t> > new_dets_importance_vec;

        std::vector<double> V(nroot,0.0);
        std::vector<std::pair<double,double> > C1(nroot,make_pair(0.0,0.0));
        std::vector<std::pair<double,double> > E2(nroot,make_pair(0.0,0.0));
        std::vector<double> ept2(nroot,0.0);

        double aimed_selection_sum = 0.0;

        for (size_t I = 0, max_I = sd_dets_vec.size(); I < max_I; ++I){
            double EI = sd_dets_vec[I].energy();
            for (int n = 0; n < nroot; ++n){
                V[n] = 0;
            }
            for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J){
                double HIJ = sd_dets_vec[I].slater_rules(ref_space[J]);
                for (int n = 0; n < nroot; ++n){
                    V[n] += evecs->get(J,n) * HIJ;
                }
            }
            for (int n = 0; n < nroot; ++n){
                double C1_I = -V[n] / (EI - evals->get(n));
                double E2_I = -V[n] * V[n] / (EI - evals->get(n));
                C1[n] = make_pair(std::fabs(C1_I),C1_I);
                E2[n] = make_pair(std::fabs(E2_I),E2_I);
            }

            std::pair<double,double> max_C1 = *std::max_element(C1.begin(),C1.end());
            std::pair<double,double> max_E2 = *std::max_element(E2.begin(),E2.end());

            double select_value = energy_select ? max_E2.first : max_C1.first;

            // Do not select now, just store the determinant index and the selection criterion
            if(aimed_selection){
                if (energy_select){
                    new_dets_importance_vec.push_back(std::make_pair(select_value,I));
                    aimed_selection_sum += select_value;
                }else{
                    new_dets_importance_vec.push_back(std::make_pair(select_value * select_value,I));
                    aimed_selection_sum += select_value * select_value;
                }
            }else{
                if (std::fabs(select_value) > tau_q){
                    new_dets_importance_vec.push_back(std::make_pair(select_value,I));
                }else{
                    for (int n = 0; n < nroot; ++n) ept2[n] += E2[n].second;
                }
            }
        }

        if(aimed_selection){
            std::sort(new_dets_importance_vec.begin(),new_dets_importance_vec.end());
            std::reverse(new_dets_importance_vec.begin(),new_dets_importance_vec.end());
            size_t maxI = new_dets_importance_vec.size();
            fprintf(outfile,"\n  The SD space will be generated using the aimed scheme (%s)",energy_select ? "energy" : "amplitude");
            fprintf(outfile,"\n  Initial value of sigma in the aimed selection = %24.14f",aimed_selection_sum);
            for (size_t I = 0; I < maxI; ++I){
                if (aimed_selection_sum > t2_threshold_){
                    ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
                    aimed_selection_sum -= new_dets_importance_vec[I].first;
                }else{
                    break;
                }
            }
            fprintf(outfile,"\n  Final value of sigma in the aimed selection   = %24.14f",aimed_selection_sum);
            fprintf(outfile,"\n  Selected %zu determinants",ref_sd_dets.size()-ref_space.size());
        }else{
            fprintf(outfile,"\n  The SD space will be generated by screening (%s)",energy_select ? "energy" : "amplitude");
            size_t maxI = new_dets_importance_vec.size();
            for (size_t I = 0; I < maxI; ++I){
                ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
            }
        }

        multistate_pt2_energy_correction_ = ept2;

        size_t dim_ref_sd_dets = ref_sd_dets.size();

        fprintf(outfile,"\n  After screening the ia-MRCISD space contains %zu determinants",dim_ref_sd_dets);
        fprintf(outfile,"\n  Time spent screening the model space = %f s",t_ms_screen.elapsed());
        fflush(outfile);


        evecs.reset(new Matrix("U",dim_ref_sd_dets,nroot));
        evals.reset(new Vector("e",nroot));
        // Full algorithm
        if (options.get_str("ENERGY_TYPE") == "IMRCISD"){
            H.reset(new Matrix("Hamiltonian Matrix",dim_ref_sd_dets,dim_ref_sd_dets));

            boost::timer t_h_build2;
#pragma omp parallel for schedule(dynamic)
            for (size_t I = 0; I < dim_ref_sd_dets; ++I){
                const BitsetDeterminant& detI = ref_sd_dets[I];
                for (size_t J = I; J < dim_ref_sd_dets; ++J){
                    const BitsetDeterminant& detJ = ref_sd_dets[J];
                    double HIJ = detI.slater_rules(detJ);
                    H->set(I,J,HIJ);
                    H->set(J,I,HIJ);
                }
            }
            fprintf(outfile,"\n  Time spent building H               = %f s",t_h_build2.elapsed());
            fflush(outfile);

            // 4) Diagonalize the Hamiltonian
            boost::timer t_hdiag_large2;
            if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
                fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
                davidson_liu(H,evals,evecs,nroot);
            }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
                fprintf(outfile,"\n  Performing full diagonalization.");
                H->diagonalize(evecs,evals);
            }

            fprintf(outfile,"\n  Time spent diagonalizing H          = %f s",t_hdiag_large2.elapsed());
            fflush(outfile);
        }
        // Sparse algorithm
        else{
            boost::timer t_h_build2;
            std::vector<std::vector<std::pair<int,double> > > H_sparse;

            size_t num_nonzero = 0;
            // Form the Hamiltonian matrix
            for (size_t I = 0; I < dim_ref_sd_dets; ++I){
                std::vector<std::pair<int,double> > H_row;
                const BitsetDeterminant& detI = ref_sd_dets[I];
                double HII = detI.slater_rules(detI);
                H_row.push_back(make_pair(int(I),HII));
                for (size_t J = 0; J < dim_ref_sd_dets; ++J){
                    if (I != J){
                        const BitsetDeterminant& detJ = ref_sd_dets[J];
                        double HIJ = detI.slater_rules(detJ);
                        if (std::fabs(HIJ) >= 1.0e-12){
                            H_row.push_back(make_pair(int(J),HIJ));
                            num_nonzero += 1;
                        }
                    }
                }
                H_sparse.push_back(H_row);
            }

            fprintf(outfile,"\n  %ld nonzero elements out of %ld (%e)",num_nonzero,size_t(dim_ref_sd_dets * dim_ref_sd_dets),double(num_nonzero)/double(dim_ref_sd_dets * dim_ref_sd_dets));
            fprintf(outfile,"\n  Time spent building H               = %f s",t_h_build2.elapsed());
            fflush(outfile);

            // 4) Diagonalize the Hamiltonian
            boost::timer t_hdiag_large2;
            fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
            davidson_liu_sparse(H_sparse,evals,evecs,nroot);
            fprintf(outfile,"\n  Time spent diagonalizing H          = %f s",t_hdiag_large2.elapsed());
            fflush(outfile);
        }

        //
        for (int i = 0; i < nroot; ++ i){
            fprintf(outfile,"\n  Adaptive CI Energy Root %3d        = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
            fprintf(outfile,"\n  Adaptive CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_ + multistate_pt2_energy_correction_[i],
                    27.211 * (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
        }
        fflush(outfile);


        // Select the new reference space
        ref_space.clear();
        ref_space_map.clear();

        new_energy = evals->get(0) + nuclear_repulsion_energy_;

        fflush(outfile);
        if (std::fabs(new_energy - old_energy) < ia_mrcisd_threshold){
            break;
        }
        old_energy = new_energy;

        std::vector<std::pair<double,size_t> > dm_det_list;

        for (size_t I = 0; I < dim_ref_sd_dets; ++I){
            double max_dm = 0.0;
            for (int n = 0; n < nroot; ++n){
                max_dm = std::max(max_dm,std::fabs(evecs->get(I,n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm,I));
        }

        std::sort(dm_det_list.begin(),dm_det_list.end());
        std::reverse(dm_det_list.begin(),dm_det_list.end());

        // Decide which will go in ref_space
        for (size_t I = 0; I < dim_ref_sd_dets; ++I){
            if (dm_det_list[I].first > tau_p){
                ref_space.push_back(ref_sd_dets[dm_det_list[I].second]);
                ref_space_map[ref_sd_dets[dm_det_list[I].second]] = 1;
            }
        }
    }

    for (int i = 0; i < nroot; ++ i){
        fprintf(outfile,"\n  * IA-MRCISD total energy (%3d)        = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
        fprintf(outfile,"\n  * IA-MRCISD total energy (%3d) + EPT2 = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_ + multistate_pt2_energy_correction_[i],
                27.211 * (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] - multistate_pt2_energy_correction_[0]));
    }
    fprintf(outfile,"\n\n  iterative_adaptive_mrcisd_bitset ran in %f s",t_iamrcisd.elapsed());
    fflush(outfile);
}


}} // EndNamespaces


