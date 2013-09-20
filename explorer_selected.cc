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
void Explorer::diagonalize_selected_space(psi::Options& options)
{
    fprintf(outfile,"\n\n  Diagonalizing the Hamiltonian in the model space (Lambda = %.2f Eh)\n",space_m_threshold_);

    // 1) Build the Hamiltonian
    boost::timer t_hbuild;
    SharedMatrix H_m = build_model_space_hamiltonian(options);
    fprintf(outfile,"\n  Time spent building H model       = %f s",t_hbuild.elapsed());
    fflush(outfile);

    // 2) Setup stuff necessary to diagonalize the Hamiltonian
    int ndets_m = H_m->nrow();
    int nroots = ndets_m;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
        nroots = std::min(options.get_int("NROOT"),ndets_m);
    }
    SharedMatrix evecs_m(new Matrix("U",ndets_m,nroots));
    SharedVector evals_m(new Vector("e",nroots));

    // 3) Diagonalize the model space Hamiltonian
    boost::timer t_hdiag;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
        fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H_m,evals_m,evecs_m,nroots);
    }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
        fprintf(outfile,"\n  Performing full diagonalization.");
        H_m->diagonalize(evecs_m,evals_m);
    }
    fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag.elapsed());
    fflush(outfile);

    // 4) Print the energy
    int nroots_print = std::min(nroots,25);
    for (int i = 0; i < nroots_print; ++ i){
        fprintf(outfile,"\n  Small CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals_m->get(i),27.211 * (evals_m->get(i) - evals_m->get(0)));
    }

    double significant_threshold = 0.001;
    double significant_wave_function = 0.95;
    for (int i = 0; i < nroots_print; ++ i){
        fprintf(outfile,"\n  The most important determinants (%.0f%% of the wave functions) for root %d:",100.0 * significant_wave_function,i + 1);
        // Identify all contributions with |C_J| > significant_threshold
        double** C_mat = evecs_m->pointer();
        std::vector<std::pair<double,int> > C_J_sorted;
        for (int J = 0; J < ndets_m; ++J){
            if (std::fabs(C_mat[J][i]) > significant_threshold){
                C_J_sorted.push_back(make_pair(std::fabs(C_mat[J][i]),J));
            }
        }
        // Sort them and print
        std::sort(C_J_sorted.begin(),C_J_sorted.end(),std::greater<std::pair<double,int> >());
        double cum_wfn = 0.0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I){
            int J = C_J_sorted[I].second;
            fprintf(outfile,"\n %3ld   %+9.6f   %9.6f   %.6f   %d",I,C_mat[J][i],C_mat[J][i] * C_mat[J][i],H_m->get(J,J),J);
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            if (cum_wfn > significant_wave_function) break;
        }
    }
    fflush(outfile);

    int root = options.get_int("ROOT");
    fprintf(outfile,"\n\n  Building a selected Hamiltonian using the criterium by Roth (kappa) for root %d",root + 1);
    SharedMatrix H = build_select_hamiltonian_roth(options,evals_m->get(root),evecs_m->get_column(0,root));


    // 3) Setup stuff necessary to diagonalize the Hamiltonian
    int ndets = H->nrow();
    SharedMatrix evecs(new Matrix("U",ndets,nroots));
    SharedVector evals(new Vector("e",nroots));

    // 4) Diagonalize the Hamiltonian
    boost::timer t_hdiag_large;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
        fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H,evals,evecs,nroots);
    }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
        fprintf(outfile,"\n  Performing full diagonalization.");
        H->diagonalize(evecs,evals);
    }
    fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag_large.elapsed());
    fflush(outfile);

    // 5) Print the energy
    for (int i = 0; i < nroots_print; ++ i){
        fprintf(outfile,"\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i),27.211 * (evals->get(i) - evals->get(0)));
        fprintf(outfile,"\n  Adaptive CI Energy + EPT2 Root %3d = %.12f",i + 1,evals->get(i) + pt2_energy_correction_);
    }

    // 6) Print the major contributions to the eigenvector
    for (int i = 0; i < nroots_print; ++ i){
        fprintf(outfile,"\n  The most important determinants (%.0f%% of the wave functions) for root %d:",100.0 * significant_wave_function,i + 1);
        // Identify all contributions with |C_J| > significant_threshold
        double** C_mat = evecs->pointer();
        std::vector<std::pair<double,int> > C_J_sorted;
        for (int J = 0; J < ndets; ++J){
            if (std::fabs(C_mat[J][i]) > significant_threshold){
                C_J_sorted.push_back(make_pair(std::fabs(C_mat[J][i]),J));
            }
        }
        // Sort them and print
        std::sort(C_J_sorted.begin(),C_J_sorted.end(),std::greater<std::pair<double,int> >());
        double cum_wfn = 0.0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I){
            int J = C_J_sorted[I].second;
            fprintf(outfile,"\n %3ld   %+9.6f   %9.6f   %.6f   %d",I,C_mat[J][i],C_mat[J][i] * C_mat[J][i],H->get(J,J),J);
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            if (cum_wfn > significant_wave_function) break;
        }
    }
    fflush(outfile);
}

/**
 * Build the Hamiltonian matrix for all determinants that fall in the model space.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix Explorer::build_model_space_hamiltonian(Options& options)
{
    int ntot_dets = static_cast<int>(determinants_.size());

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = 0;

    // Determine the size of the Hamiltonian matrix
    if (options.get_str("H_TYPE") == "FIXED_SIZE"){
        ndets = std::min(options.get_int("NDETS"),ntot_dets);
        fprintf(outfile,"\n  Building the model space Hamiltonian using the first %d determinants\n",ndets);
        fprintf(outfile,"\n  The energy range spanned is [%f,%f]\n",determinants_[0].get<0>(),determinants_[ndets-1].get<0>());
    }else if (options.get_str("H_TYPE") == "FIXED_ENERGY"){
        double E0 = determinants_[0].get<0>();
        for (int I = 0; I < ntot_dets; ++I){
            double EI = determinants_[I].get<0>();
            if (EI - E0 > space_m_threshold_){
                break;
            }
            ndets++;
        }
        fprintf(outfile,"\n\n  Building the model Hamiltonian using determinants with excitation energy less than %f Eh",space_m_threshold_);
        fprintf(outfile,"\n  This requires a total of %d determinants",ndets);
        int max_ndets_fixed_energy = options.get_int("MAX_NDETS");
        if (ndets > max_ndets_fixed_energy){
            fprintf(outfile,"\n\n  WARNING: the number of determinants required to build the Hamiltonian (%d)\n"
                    "  exceeds the maximum number allowed (%d).  Reducing the size of H.\n\n",ndets,max_ndets_fixed_energy);
            ndets = max_ndets_fixed_energy;
        }
    }

    SharedMatrix H(new Matrix("Hamiltonian Matrix",ndets,ndets));

    // Form the Hamiltonian matrix
    #pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < ndets; ++I){
        boost::tuple<double,int,int,int,int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
        for (int J = I + 1; J < ndets; ++J){
            boost::tuple<double,int,int,int,int>& determinantJ = determinants_[J];
            const int J_class_a = determinantJ.get<1>();  //std::get<1>(determinantI);
            const int Jsa = determinantJ.get<2>();        //std::get<1>(determinantI);
            const int J_class_b = determinantJ.get<3>(); //std::get<2>(determinantI);
            const int Jsb = determinantJ.get<4>();        //std::get<2>(determinantI);
            const double HIJ = StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>(),vec_astr_symm_[J_class_a][Jsa].get<2>(),vec_bstr_symm_[J_class_b][Jsb].get<2>());
            H->set(I,J,HIJ);
            H->set(J,I,HIJ);
        }
        H->set(I,I,determinantI.get<0>());
    }
    return H;
}

/**
 * Build the Hamiltonian matrix for all determinants that fall in the model space.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix Explorer::build_select_hamiltonian_roth(Options& options, double E, SharedVector evect)
{
    int ntot_dets = static_cast<int>(determinants_.size());

    // Find out which determinants will be included
    std::vector<int> selected_dets;
    int ndets_m = evect->dim();
    int ndets_i = 0;

    for (int J = 0; J < ndets_m; ++J) selected_dets.push_back(J);

    double ept2 = 0.0;
//    #pragma omp parallel for schedule(dynamic)
    for (int I = ndets_m; I < ntot_dets; ++I){
        boost::tuple<double,int,int,int,int>& determinantI = determinants_[I];
        const double EI = determinantI.get<0>();  //std::get<0>(determinantI);
        const int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
        double V = 0.0;
        for (int J = 0; J < ndets_m; ++J){
            boost::tuple<double,int,int,int,int>& determinantJ = determinants_[J];
            const double EJ = determinantJ.get<0>();  //std::get<0>(determinantJ);
            const int J_class_a = determinantJ.get<1>();  //std::get<1>(determinantJ);
            const int Jsa = determinantJ.get<2>();        //std::get<1>(determinantJ);
            const int J_class_b = determinantJ.get<3>(); //std::get<2>(determinantJ);
            const int Jsb = determinantJ.get<4>();        //std::get<2>(determinantJ);
            const double HIJ = StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>(),vec_astr_symm_[J_class_a][Jsa].get<2>(),vec_bstr_symm_[J_class_b][Jsb].get<2>());
            V += evect->get(J) * HIJ;
        }
        double kappa =  - V / (EI - E);
        double chi = - V * V / (EI - E);
        if (std::fabs(kappa) > t2_threshold_){
//            #pragma omp critical
            selected_dets.push_back(I);
            ndets_i += 1;
        }else{
            ept2 += chi;
        }
    }

    pt2_energy_correction_ = ept2;

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = ndets_i + ndets_m;
    fprintf(outfile,"\n\n  %d total states: %d (main) + %d (intermediate)",ntot_dets,ndets_m,ndets_i);
    fprintf(outfile,"\n  %d states were discarded because the coupling to the main space is less than %f muE_h",ntot_dets - ndets_m - ndets_i,t2_threshold_ * 1000000.0);
    fprintf(outfile,"\n  The estimated contribution from the excluded space is %.9f Eh",ept2);
    SharedMatrix H(new Matrix("Hamiltonian Matrix",ndets,ndets));
    // Form the Hamiltonian matrix
    #pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < ndets; ++I){
        boost::tuple<double,int,int,int,int>& determinantI = determinants_[selected_dets[I]];
        const int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
        for (int J = I + 1; J < ndets; ++J){
            boost::tuple<double,int,int,int,int>& determinantJ = determinants_[selected_dets[J]];
            const int J_class_a = determinantJ.get<1>();  //std::get<1>(determinantI);
            const int Jsa = determinantJ.get<2>();        //std::get<1>(determinantI);
            const int J_class_b = determinantJ.get<3>(); //std::get<2>(determinantI);
            const int Jsb = determinantJ.get<4>();        //std::get<2>(determinantI);
            const double HIJ = StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>(),vec_astr_symm_[J_class_a][Jsa].get<2>(),vec_bstr_symm_[J_class_b][Jsb].get<2>());
            H->set(I,J,HIJ);
            H->set(J,I,HIJ);
        }
        H->set(I,I,determinantI.get<0>());
    }
    return H;
}

}} // EndNamespaces
