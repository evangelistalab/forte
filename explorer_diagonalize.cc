#include "explorer.h"

#include <cmath>

#include <boost/timer.hpp>
#include <boost/format.hpp>

#include "explorer.h"
#include "cartographer.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

inline double clamp(double x, double a, double b)

{
    return x < a ? a : (x > b ? b : x);
}

inline double smootherstep(double edge0, double edge1, float x)
{
    // Scale, and clamp x to 0..1 range
    x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    // Evaluate polynomial
    return x * x * x *( x *( x * 6. - 15.) + 10.);
}

/**
 * Find all the Slater determinants with an energy lower than determinant_threshold_
 */
void Explorer::diagonalize(psi::Options& options)
{
    fprintf(outfile,"\n\n  Diagonalizing the Hamiltonian in a small space\n");

    SharedMatrix H = build_hamiltonian(options);

    smooth_hamiltonian(H);

    int ndets = H->nrow();
    SharedMatrix evecs(new Matrix("U",ndets,ndets));
    SharedVector evals(new Vector("e",ndets));
    H->diagonalize(evecs,evals);

    int ndets_print = std::min(ndets,10);
    for (int i = 0; i < ndets_print; ++ i){
        fprintf(outfile,"\n Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i),27.211 * (evals->get(i) - evals->get(0)));
    }



//    fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag.elapsed());
//    fprintf(outfile,"\n  Time spent building H             = %f s",t_hbuild.elapsed());

//    timer t_hdiag;

//    double unscreened_range = determinant_threshold - h_buffer;

//    if (std::fabs(h_buffer) > 0.0){
//        for (int I = 1; I < ndets; ++I){
//            double deltaE = H->get(I,I) - H->get(0,0);
//            if (deltaE > unscreened_range){
//                double excessE = deltaE - unscreened_range;
//                double factor = smootherstep(excessE,h_buffer,h_buffer);
//                fprintf(outfile,"\n  Det %d , De = %f, exE = %f, f = %f",I,deltaE,excessE,factor);

//                //double factor = 1.0 - excessE / h_buffer;
//                for (int J = 0; J <= I; ++J){
//                    double element = H->get(I,J);
//                    H->set(I,J,element * factor);
//                    H->set(J,I,element * factor);
//                }
//            }
//        }
//    }

//    for (int n = 1; n <= std::min(ndets,4000); n += 1){
//        timer t_hbuild;
//        for (int I = 0; I < n; ++I){
//            boost::tuple<double,int,int>& determinantI = determinants[I];
//            int Isa = std::get<1>(determinantI);
//            int Isb = std::get<2>(determinantI);
//            detI.set_bits(vec_astr[Isa].second,vec_bstr[Isb].second);
//            for (int J = I; J < n; ++J){
//                boost::tuple<double,int,int>& determinantJ = determinants[J];
//                int Jsa = std::get<1>(determinantJ);
//                int Jsb = std::get<2>(determinantJ);
//                detJ.set_bits(vec_astr[Jsa].second,vec_bstr[Jsb].second);
//                double HIJ = SlaterRules(detI,detJ,ints_);
//                H->set(I,J,HIJ);
//                H->set(J,I,HIJ);
//            }
//        }
//        fprintf(outfile,"\n  Time spent building H             = %f s",t_hbuild.elapsed());

//        timer t_hdiag;

////        int nroots = 5;
////        SharedMatrix evecs(new Matrix("U",maxdet,nroots));
////        SharedVector evals(new Vector("e",maxdet));
////        lanczos(H,evecs,evals,nroots);

//        SharedMatrix evecs(new Matrix("U",maxdet,maxdet));
//        SharedVector evals(new Vector("e",maxdet));
//        H->diagonalize(evecs,evals);

//        fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag.elapsed());
//        fprintf(outfile,"\n %6d",n);
//        for (int i = 0; i < 5; ++ i){
//            fprintf(outfile," %3d %15.9f %8.4f",i,evals->get(i)+ nuclear_energy,27.211*(evals->get(i) - evals->get(0)));
//        }
//    }
}

/**
 * Build the Hamiltonian matrix from the list of determinants.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix Explorer::build_hamiltonian(Options& options)
{
    int ntot_dets = static_cast<int>(determinants_.size());

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = 0;

    // Determine the size of the Hamiltonian matrix
    if (options.get_str("H_TYPE") == "FIXED_SIZE"){
        ndets = std::min(options.get_int("NDETS"),ntot_dets);
        fprintf(outfile,"\n  Building the Hamiltonian using the first %d determinants\n",ndets);
        fprintf(outfile,"\n  The energy range spanned is [%f,%f]\n",determinants_[0].get<0>(),determinants_[ndets-1].get<0>());
    }else if (options.get_str("H_TYPE") == "FIXED_ENERGY"){
        fprintf(outfile,"\n\n  Building the Hamiltonian using determinants with excitation energy less than %f Eh",determinant_threshold_);
        int max_ndets_fixed_energy = 3000;
        ndets = std::min(max_ndets_fixed_energy,ntot_dets);
        if (ndets == max_ndets_fixed_energy){
            fprintf(outfile,"\n\n  WARNING: the number of determinants used to build the Hamiltonian\n"
                            "  exceeds the maximum number allowed (%d).  Reducing the size of H.\n\n",max_ndets_fixed_energy);
        }
    }

    SharedMatrix H(new Matrix("Hamiltonian Matrix",ndets,ndets));

    // Form the Hamiltonian matrix
    StringDeterminant detI(reference_determinant_);
    StringDeterminant detJ(reference_determinant_);

    for (int I = 0; I < ndets; ++I){
        boost::tuple<double,int,int,int,int>& determinantI = determinants_[I];
        int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
        int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
        int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
        int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
        detI.set_bits(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>());
        for (int J = I + 1; J < ndets; ++J){
            boost::tuple<double,int,int,int,int>& determinantJ = determinants_[J];
            int J_class_a = determinantJ.get<1>();  //std::get<1>(determinantI);
            int Jsa = determinantJ.get<2>();        //std::get<1>(determinantI);
            int J_class_b = determinantJ.get<3>(); //std::get<2>(determinantI);
            int Jsb = determinantJ.get<4>();        //std::get<2>(determinantI);
            detJ.set_bits(vec_astr_symm_[J_class_a][Jsa].get<2>(),vec_bstr_symm_[J_class_b][Jsb].get<2>());
            double HIJ = detI.slater_rules(detJ);
            H->set(I,J,HIJ);
            H->set(J,I,HIJ);
        }
        H->set(I,I,determinantI.get<0>());
    }

    return H;
}

void Explorer::smooth_hamiltonian(SharedMatrix H)
{
    int ndets = H->nrow();
    double main_space_threshold = determinant_threshold_ - smoothing_threshold_;
    // Partition the Hamiltonian into main and intermediate model space
    int ndets_main = 0;
    for (int I = 0; I < ndets; ++I){
        if (H->get(I,I) - H->get(0,0) > main_space_threshold){
            ndets_main = I;
            break;
        }
    }
    fprintf(outfile,"\n\n  The model space of dimension %d will be split into %d (main) + %d (intermediate) states",ndets,ndets_main,ndets - ndets_main);
    for (int I = 0; I < ndets; ++I){
        for (int J = ndets_main; J < ndets; ++J){
            if (I != J){
                double HIJ = H->get(I,J);
                double EI = H->get(I,I);
                double EJ = H->get(J,J);
                double EJ0 = EJ - H->get(0,0);
                double factor = 1.0 - smootherstep(0.0,smoothing_threshold_,std::fabs(EJ0 - main_space_threshold));
                H->set(I,J,factor * HIJ);
                H->set(J,I,factor * HIJ);
            }
        }
    }
}

void Explorer::davidson_liu(SharedMatrix H,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroots)
{
    int n = H->nrow();
    int n_small = std::min(50,n);

    // Diagonalize a small matrix of dimension 50 x 50 or less
    SharedMatrix Hsmall(new Matrix("U",n_small,n_small));
    SharedMatrix evecs_small(new Matrix("U",n_small,n_small));
    SharedVector evals_small(new Vector("e",n_small));
    for (int I = 0; I < n_small; ++I){
        for (int J = 0; J < n_small; ++J){
            Hsmall->set(I,J,H->get(I,J));
        }
    }
    Hsmall->diagonalize(evecs_small,evals_small);
    continuer from here sir...
    if (n_small == n){
        Eigenvalues = evals_small;
        Eigenvectors = evecs_small;
        return;
    }


}

}} // EndNamespaces




