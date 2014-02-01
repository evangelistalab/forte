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

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

/**
 * Diagonalize the
 */
void Explorer::renormalized_mrcisd(psi::Options& options)
{
    fprintf(outfile,"\n\n  Renormalized MRCISD");

    int nroot = options.get_int("NROOT");
    size_t ren_ndets = options.get_int("REN_MAX_NDETS");
    double selection_threshold = t2_threshold_;
    double rmrci_threshold = 1.0e-9;
    int maxcycle = 50;

    fprintf(outfile,"\n\n  Diagonalizing the Hamiltonian in the model space");
    fprintf(outfile,"\n  using a renormalization procedure keeping %zu determinants\n",ren_ndets);
    fprintf(outfile,"\n  and exciting those with a first-order coefficient greather than %f\n",selection_threshold);
    fflush(outfile);

    int nmo = reference_determinant_.nmo();
    std::vector<int> aocc(nalpha_);
    std::vector<int> bocc(nbeta_);
    std::vector<int> avir(nmo_ - nalpha_);
    std::vector<int> bvir(nmo_ - nbeta_);

    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    int nvalpha = nmo_ - nalpha_;
    int nvbeta = nmo_ - nbeta_;

    std::vector<StringDeterminant> old_dets_vec;
    std::vector<double> coefficient;
    old_dets_vec.push_back(reference_determinant_);
    coefficient.push_back(1.0);
    double old_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_energy = 0.0;

    for (int cycle = 0; cycle < maxcycle; ++cycle){

        fprintf(outfile,"\n  Cycle %3d: %zu determinants in the RMRCISD wave function",cycle,old_dets_vec.size());

        std::map<StringDeterminant,int> new_dets_map;
        boost::timer t_ms_build;
//        fprintf(outfile,"\n  Processing determinants:\n");
        for (size_t I = 0, max_I = old_dets_vec.size(); I < max_I; ++I){
            const StringDeterminant& det = old_dets_vec[I];

            double Em = det.energy();

//            fprintf(outfile,"+");
//            fflush(outfile);
            new_dets_map[det] = 1;
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
                        new_dets_map[new_det] = 1;
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
                        new_dets_map[new_det] = 1;
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

                                double Ex = new_det.energy();
                                double V = det.slater_rules(new_det);
                                double c1 = std::fabs(V / (Em - Ex) * coefficient[I]);
                                if (c1 > selection_threshold){
                                    new_dets_map[new_det] = 1;
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

                                double Ex = new_det.energy();
                                double V = det.slater_rules(new_det);
                                double c1 = std::fabs(V / (Em - Ex) * coefficient[I]);
                                if (c1 > selection_threshold){
                                    new_dets_map[new_det] = 1;
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

                                double Ex = new_det.energy();
                                double V = det.slater_rules(new_det);
                                double c1 = std::fabs(V / (Em - Ex) * coefficient[I]);
                                if (c1 > selection_threshold){
                                    new_dets_map[new_det] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }


        std::vector<StringDeterminant> new_dets_vec;
        for (auto it = new_dets_map.begin(), endit = new_dets_map.end(); it != endit; ++it){
            new_dets_vec.push_back(it->first);
        }

        size_t num_mrcisd_dets = new_dets_vec.size();


        fprintf(outfile,"\n  The model space contains %zu determinants",num_mrcisd_dets);
        fprintf(outfile,"\n  Time spent building the model space = %f s",t_ms_build.elapsed());
        fflush(outfile);

        H.reset(new Matrix("Hamiltonian Matrix",num_mrcisd_dets,num_mrcisd_dets));
        evecs.reset(new Matrix("U",num_mrcisd_dets,nroot));
        evals.reset(new Vector("e",nroot));

        boost::timer t_h_build;
        for (size_t I = 0; I < num_mrcisd_dets; ++I){
            for (size_t J = I; J < num_mrcisd_dets; ++J){
                double HIJ = new_dets_vec[I].slater_rules(new_dets_vec[J]);
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

        new_energy = evals->get(0) + nuclear_repulsion_energy_;

        std::vector<std::pair<double,size_t> > dm_det_list;

        for (size_t I = 0; I < num_mrcisd_dets; ++I){
            double max_dm = 0.0;
            for (int n = 0; n < nroot; ++n){
                max_dm = std::max(max_dm,std::fabs(evecs->get(I,n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm,I));
        }

        std::sort(dm_det_list.begin(),dm_det_list.end());
        std::reverse(dm_det_list.begin(),dm_det_list.end());

        old_dets_vec.clear();
        coefficient.clear();
        for (size_t I = 0, max_I = std::min(dm_det_list.size(),ren_ndets); I < max_I; ++I){
            old_dets_vec.push_back(new_dets_vec[dm_det_list[I].second]);
            coefficient.push_back(evecs->get(dm_det_list[I].second,0));
        }

        size_t size_small_ci = std::min(dm_det_list.size(),ren_ndets);
        H.reset(new Matrix("Hamiltonian Matrix",size_small_ci,size_small_ci));
        evecs.reset(new Matrix("U",size_small_ci,nroot));
        evals.reset(new Vector("e",nroot));

        boost::timer t_h_small_build;
        for (size_t I = 0; I < size_small_ci; ++I){
            for (size_t J = I; J < size_small_ci; ++J){
                double HIJ = old_dets_vec[I].slater_rules(old_dets_vec[J]);
                H->set(I,J,HIJ);
                H->set(J,I,HIJ);
            }
        }
        fprintf(outfile,"\n  Time spent building H               = %f s",t_h_small_build.elapsed());
        fflush(outfile);

        // 4) Diagonalize the Hamiltonian
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
            fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H,evals,evecs,nroot);
        }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
            fprintf(outfile,"\n  Performing full diagonalization.");
            H->diagonalize(evecs,evals);
        }


        // 5) Print the energy
        for (int i = 0; i < nroot; ++ i){
            fprintf(outfile,"\n  Ren. step (small) CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i) + nuclear_repulsion_energy_,27.211 * (evals->get(i) - evals->get(0)));
        }
        fflush(outfile);

        new_energy = evals->get(0) + nuclear_repulsion_energy_;
        fprintf(outfile,"\n ->  %2d  %24.16f",cycle,new_energy);

        fflush(outfile);
        if (std::fabs(new_energy - old_energy) < rmrci_threshold){
            break;
        }
        old_energy = new_energy;

        fprintf(outfile,"\n  After diagonalization there are %zu determinants",old_dets_vec.size());
        fflush(outfile);
    }


    fflush(outfile);
}



}} // EndNamespaces

