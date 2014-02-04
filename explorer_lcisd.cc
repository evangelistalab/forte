#include "explorer.h"

#include <cmath>
#include <functional>
#include <algorithm>

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

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

/**
 * Diagonalize the
 */
void Explorer::lambda_mrcisd(psi::Options& options)
{
    fprintf(outfile,"\n\n  Lambda-MRCISD");

    int nroot = options.get_int("NROOT");

    double selection_threshold = t2_threshold_;

    fprintf(outfile,"\n\n  Diagonalizing the Hamiltonian in the model space (Lambda = %.2f Eh)\n",space_m_threshold_);

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

    // 1) Build the Hamiltonian using the StringDeterminant representation
    std::vector<StringDeterminant> ref_space;
    std::map<StringDeterminant,int> ref_space_map;

    for (size_t I = 0, maxI = determinants_.size(); I < maxI; ++I){
        boost::tuple<double,int,int,int,int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
        StringDeterminant det(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>());
        ref_space.push_back(det);
        ref_space_map[det] = 1;
    }

    size_t dim_ref_space = ref_space.size();

    fprintf(outfile,"\n  The model space contains %zu determinants",dim_ref_space);
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

    // 4) Diagonalize the Hamiltonian
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

    // 5) Print the energy
    for (int i = 0; i < nroot; ++ i){
        fprintf(outfile,"\n  Ren. step CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
        //        fprintf(outfile,"\n  Ren. step CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f",i + 1,evals->get(i) + multistate_pt2_energy_correction_[i],
        //                evals->get(i),multistate_pt2_energy_correction_[i]);
    }
    fprintf(outfile,"\n  Finished building H");
    fflush(outfile);


    int nmo = reference_determinant_.nmo();
    std::vector<int> aocc(nalpha_);
    std::vector<int> bocc(nbeta_);
    std::vector<int> avir(nmo_ - nalpha_);
    std::vector<int> bvir(nmo_ - nbeta_);

    int nvalpha = nmo_ - nalpha_;
    int nvbeta = nmo_ - nbeta_;

    // Find the SD space out of the reference
    std::vector<StringDeterminant> sd_dets_vec;
    std::map<StringDeterminant,int> new_dets_map;
    boost::timer t_ms_build;

    for (size_t I = 0, max_I = ref_space_map.size(); I < max_I; ++I){
        const StringDeterminant& det = ref_space[I];
        for (int p = 0, i = 0, a = 0; p < nmo_; ++p){
            if (det.get_alfa_bit(p)){
                aocc[i] = p;
                i++;
            }else{
                avir[a] = p;
                a++;
            }
        }
        for (int p = 0, i = 0, a = 0; p < nmo_; ++p){
            if (det.get_beta_bit(p)){
                bocc[i] = p;
                i++;
            }else{
                bvir[a] = p;
                a++;
            }
        }

        // Generate aa excitations
        for (int i = 0; i < nalpha_; ++i){
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

        for (int i = 0; i < nbeta_; ++i){
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
        for (int i = 0; i < nalpha_; ++i){
            int ii = aocc[i];
            for (int j = i + 1; j < nalpha_; ++j){
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

        for (int i = 0; i < nalpha_; ++i){
            int ii = aocc[i];
            for (int j = 0; j < nbeta_; ++j){
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
        for (int i = 0; i < nbeta_; ++i){
            int ii = bocc[i];
            for (int j = i + 1; j < nbeta_; ++j){
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
    for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J){
        ref_sd_dets.push_back(ref_space[J]);
    }


    // Check the coupling between the reference and the SD space
    std::vector<std::pair<double,size_t> > new_dets_importance_vec;

    std::vector<double> V_q(nroot,0.0);
    std::vector<double> t_q(nroot,0.0);
    std::vector<std::pair<double,double> > kappa_q(nroot,make_pair(0.0,0.0));
    std::vector<std::pair<double,double> > chi_q(nroot,make_pair(0.0,0.0));
    std::vector<double> ept2(nroot,0.0);

    double aimed_selection_sum = 0.0;

    for (size_t I = 0, max_I = sd_dets_vec.size(); I < max_I; ++I){
        double V = 0.0;
        double EI = sd_dets_vec[I].energy();
        for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J){
            V += sd_dets_vec[I].slater_rules(ref_space[J]) * evecs->get(J,0);  // HJI * C_I
        }
        double C1 = std::fabs(V / (EI - evals->get(0)));
        double E2 = std::fabs(V * V / (EI - evals->get(0)));

        double select_value = (energy_select ? E2 : C1);

        // Do not select now, just store the determinant index and the selection criterion
        if(aimed_selection){
            new_dets_importance_vec.push_back(std::make_pair(select_value,I));
            aimed_selection_sum += select_value;
        }else{
            if (std::fabs(select_value) > t2_threshold_){
                new_dets_importance_vec.push_back(std::make_pair(select_value,I));
            }else{
                //                    for (int n = 0; n < nroot; ++n) ept2[n] += chi_q[n].second;
            }
        }
    }

    if(aimed_selection){
        std::sort(new_dets_importance_vec.begin(),new_dets_importance_vec.end());
        std::reverse(new_dets_importance_vec.begin(),new_dets_importance_vec.end());
        size_t maxI = new_dets_importance_vec.size();

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
        size_t maxI = new_dets_importance_vec.size();
        for (size_t I = 0; I < maxI; ++I){
            ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
        }
    }

    size_t dim_ref_sd_dets = ref_sd_dets.size();

    fprintf(outfile,"\n  The model space contains %zu determinants",dim_ref_sd_dets);
    fprintf(outfile,"\n  Time spent screening the model space = %f s",t_ms_screen.elapsed());
    fflush(outfile);


    evecs.reset(new Matrix("U",dim_ref_sd_dets,nroot));
    evals.reset(new Vector("e",nroot));
    // Full algorithm
    if (options.get_str("ENERGY_TYPE") == "LMRCISD"){
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

        // 4) Diagonalize the Hamiltonian
        boost::timer t_hdiag;
        fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
        davidson_liu_sparse(H_sparse,evals,evecs,nroot);
    }

    // 5) Print the energy
    for (int i = 0; i < nroot; ++ i){
        fprintf(outfile,"\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
        //        fprintf(outfile,"\n  Ren. step CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f",i + 1,evals->get(i) + multistate_pt2_energy_correction_[i],
        //                evals->get(i),multistate_pt2_energy_correction_[i]);
    }
    fprintf(outfile,"\n  Finished building H");
    fflush(outfile);

    fflush(outfile);
}


}} // EndNamespaces

