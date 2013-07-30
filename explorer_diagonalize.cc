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

/**
 * An ancillary function to compare the det_info data structures.  Used to sort determinants.
 * @param t1
 * @param t2
 * @return
 */
bool compare_det_info(const det_info& t1, const det_info& t2)
{
    if (t1.get<0>() != t2.get<0>()){
        return (t1.get<0>() < t2.get<0>());
    }
    else if (t1.get<1>() != t2.get<1>()){
        return (t1.get<1>() < t2.get<1>());
    }
    else if (t1.get<2>() != t2.get<2>()){
        return (t1.get<2>() < t2.get<2>());
    }
    else if (t1.get<3>() != t2.get<3>()){
        return (t1.get<3>() < t2.get<3>());
    }
    return (t1.get<4>() < t2.get<4>());
}

/**
 * Find all the Slater determinants with an energy lower than determinant_threshold_
 */
void Explorer::diagonalize(psi::Options& options)
{
    fprintf(outfile,"\n\n  Diagonalizing the Hamiltonian in a small space\n");

    // sort the determinants
    std::sort(determinants_.begin(),determinants_.end(),compare_det_info);

    int ntot_dets = static_cast<int>(determinants_.size());

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = std::min(200,ntot_dets);

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
    SharedMatrix evecs(new Matrix("U",ndets,ndets));
    SharedVector evals(new Vector("e",ndets));
    H->diagonalize(evecs,evals);

//    fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag.elapsed());
    for (int i = 0; i < 5; ++ i){
        fprintf(outfile,"\n Adaptive CI Energy Root %3d = %.12f %8.4f",i,evals->get(i),27.211*(evals->get(i) - evals->get(0)));
    }

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

}} // EndNamespaces




