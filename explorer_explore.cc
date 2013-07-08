#include "explorer.h"

#include <cmath>

#include <boost/timer.hpp>
#include <boost/format.hpp>


#include "explorer.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

/**
 * Find all the Slater determinants with an energy lower than determinant_threshold_
 */
void Explorer::explore(psi::Options& options)
{
    fprintf(outfile,"\n\n  Exploring the space of Slater determinants\n");

    int nfrozen = 0;
    int naocc = nalpha_ - nfrozen;
    int nbocc = nbeta_ - nfrozen;
    int navir = nmo_ - nalpha_;
    int nbvir = nmo_ - nbeta_;

    // Calculate the maximum excitation level
    int maxnaex = std::min(naocc,navir);
    int maxnbex = std::min(nbocc,nbvir);
    int minnex = 0;
    int maxnex = maxnaex + maxnbex;

    // Allocate an array of bits for fast manipulation
    bool* Ia = new bool[2 * nmo_];
    bool* Ib = &Ia[nmo_];

    boost::timer t;
    double time_string = 0.0;
    double time_dets = 0.0;
    long num_dets_visited = 0;
    long num_dets_accepted = 0;
    unsigned long long num_total_dets = 0;
    unsigned long num_permutations = 0;

    // Count the total number of determinants
    for (int nex = minnex; nex <= maxnex; ++nex){
        for (int naex = std::max(0,nex-maxnbex); naex <= std::min(maxnaex,nex); ++naex){
            int nbex = nex - naex;
            // count the number of determinants
            unsigned long long num_dets_total_class = choose(naocc,naex) * choose(navir,naex) * choose(nbocc,nbex) * choose(nbvir,nbex);
            num_total_dets += num_dets_total_class;
        }
    }

    // Open the file that will contain the determinants and energies
    ofstream os;
    os.open ("det_energy.txt");

    // Generate all alpha and beta strings with energy < threshold
    // The strings are in QT format and are stored using the following structure:
    // [<string irrep>][<string index>](<string energy>,<string structure>)
    fprintf(outfile,"\n  Screening the alpha strings..."); fflush(outfile);
    boost::timer timer_astr;
    vec_astr_symm_ = compute_strings_screened(epsilon_a_qt_,naocc,maxnaex);
    fprintf(outfile," done.  Time required: %f s",timer_astr.elapsed());

    for (int ha = 0; ha < nirrep_; ++ha){
        string_list& vec_astr = vec_astr_symm_[ha];
        size_t nsa = vec_astr.size();
        fprintf(outfile,"\n  irrep %d: %ld strings",ha,nsa);
    }
    fflush(outfile);

    fprintf(outfile,"\n  Screening the beta strings..."); fflush(outfile);
    boost::timer timer_bstr;
    vec_bstr_symm_ = compute_strings_screened(epsilon_b_qt_,nbocc,maxnbex);
    fprintf(outfile," done.  Time required: %f s",timer_bstr.elapsed());

    for (int hb = 0; hb < nirrep_; ++hb){
        string_list& vec_bstr = vec_bstr_symm_[hb];
        size_t nsb = vec_bstr.size();
        fprintf(outfile,"\n  irrep %d: %ld strings",hb,nsb);
    }
    fflush(outfile);

    vector<bool> empty_det(2 * nmo_,false);
    StringDeterminant det(empty_det);
    boost::timer t_dets;
    // Loop over the irreps
    for (int ha = 0; ha < nirrep_; ++ha){
        int hb = wavefunction_symmetry_ ^ ha;
        string_list& vec_astr = vec_astr_symm_[ha];
        string_list& vec_bstr = vec_bstr_symm_[hb];
        size_t nsa = vec_astr.size();
        size_t nsb = vec_bstr.size();
        // Loop over alpha strings
        for (size_t sa = 0; sa < nsa; ++sa){
            double ea = vec_astr[sa].first;
            std::vector<bool>& str_sa = vec_astr[sa].second;
            // Copy the string and translate it to Pitzer ordering
            for (int p = 0; p < nmo_; ++p) Ia[qt_to_pitzer_[p]] = str_sa[p];

            // Loop over beta strings
            for (size_t sb = 0; sb < nsb; ++sb){
                double eb = vec_bstr[sb].first;
                if (ea + eb < denominator_threshold_){
                    std::vector<bool>& str_sb = vec_bstr[sb].second;
                    // Copy the string and translate it to Pitzer ordering
                    for (int p = 0; p < nmo_; ++p) Ib[qt_to_pitzer_[p]] = str_sb[p];

                    // set the alpha/beta strings and compute the energy of this determinant
                    det.set_bits(Ia,Ib);
                    double det_energy = det.energy() + nuclear_repulsion_energy_;

                    // check to see if the energy is below a given threshold
                    if (det_energy < min_energy_ + determinant_threshold_){
                        // TODO this step is actually a bit slow, perhaps it is best to accumulate
                        // this information in a string and then flush it out at the end
                        determinants_.push_back(std::make_tuple(det_energy,ha,sa,sb));
                        write_determinant_energy(os,Ia,Ib,det_energy,ea + eb);
                        if (det_energy < min_energy_){
                            reference_determinant_ = det;
                            min_energy_ = det_energy;
                        }
//                        det_den_energy.push_back(make_tuple(det_energy,ea + eb,reference_determinant_.excitation_level(Ia,Ib)));
                        num_dets_accepted++;
                    }
                    num_dets_visited++;
                }else{
                    break;  // since the strings are ordered by energy, if you are here we can just skip this loop
                }
            }
        }
    }
    time_dets += t_dets.elapsed();
    os.close();
    delete[] Ia;

    fprintf(outfile,"\n  The new reference determinant is:");
    reference_determinant_.print();
    fprintf(outfile,"\n  and its energy: %.12f Eh",min_energy_);

    fprintf(outfile,"\n\n  Number of full ci determinants    = %llu",num_total_dets);
    fprintf(outfile,"\n  Number of determinants visited    = %ld (%e)",num_dets_visited,double(num_dets_visited) / double(num_total_dets));
    fprintf(outfile,"\n  Number of determinants accepted   = %ld (%e)",num_dets_accepted,double(num_dets_accepted) / double(num_total_dets));
    fprintf(outfile,"\n  Number of permutations visited    = %ld",num_permutations);
    fprintf(outfile,"\n  Time spent on generating strings  = %f s",time_string);
    fprintf(outfile,"\n  Time spent on generating dets     = %f s",time_dets);
    fprintf(outfile,"\n  Precompute algorithm time elapsed = %f s",t.elapsed());
    fflush(outfile);
}

}} // EndNamespaces
