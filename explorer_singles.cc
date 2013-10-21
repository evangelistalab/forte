#include "explorer.h"

#include <cmath>
#include <set>

#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <boost/unordered_map.hpp>

#include "explorer.h"
#include "cartographer.h"
#include "string_determinant.h"
#include "excitation_determinant.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

/**
 * Find all the Slater determinants with an energy lower than determinant_threshold_
 * by performing single excitations at a time
 */
void Explorer::explore_singles(psi::Options& options)
{
    fprintf(outfile,"\n\n  Exploring the space of Slater determinants using the singles method\n");
    boost::timer t;

    // No explorer will succeed without a cartographer
    Cartographer cg(options,min_energy_,min_energy_ + determinant_threshold_);

    int nfrzc = frzcpi_.sum();
    int nfrzv = frzvpi_.sum();
    int naocc = nalpha_ - nfrzc;
    int nbocc = nbeta_ - nfrzc;
    int navir = nmo_ - naocc - nfrzc - nfrzv;
    int nbvir = nmo_ - nbocc - nfrzc - nfrzv;


    ExcitationDeterminant zero_ex;
    size_t nastr = 0;
    size_t nbstr = 0;
    boost::unordered_map<std::vector<short int>, size_t> alpha_strings_map;
    boost::unordered_map<std::vector<short int>, size_t> beta_strings_map;
    boost::unordered_map<std::pair<size_t,size_t>, size_t> good_determinants;
    std::pair<std::vector<double>,std::vector<double> > fock_diagonals;
    fock_diagonals.first = std::vector<double>(nmo_,0.0);
    fock_diagonals.second = std::vector<double>(nmo_,0.0);
    size_t failed_attepts = 0;

    // Store energy,irrep,and excitation operator
    std::vector<boost::tuple<double,int,ExcitationDeterminant> > selected_determinants;
    selected_determinants.push_back(boost::make_tuple(ref_energy_,wavefunction_symmetry_,zero_ex));

    std::vector<boost::tuple<double,int,ExcitationDeterminant> > to_be_processed_elements;
    to_be_processed_elements.push_back(boost::make_tuple(ref_energy_,wavefunction_symmetry_,zero_ex));

    size_t num_dets_visited = 0;
    boost::timer t_dets;
    bool iterate = true;
    int level = 0;
    while(iterate){
        int maxn = static_cast<int>(to_be_processed_elements.size());
        std::vector<boost::tuple<double,int,ExcitationDeterminant> > new_elements;
        size_t num_dets_visited_ex = 0;
        size_t num_dets_accepted_ex = 0;
        for (int n = 0; n < maxn; ++n){
            // Orbitals to consider when adding a single excitation to the element n
            double energy_ex = to_be_processed_elements[n].get<0>();
            int irrep_ex = to_be_processed_elements[n].get<1>();
            ExcitationDeterminant ex = to_be_processed_elements[n].get<2>();

            int minai = 0;
            int maxai = naocc;
            int minaa = naocc;
            int maxaa = naocc + navir;
            int minbi = 0;
            int maxbi = nbocc;
            int minba = nbocc;
            int maxba = nbocc + nbvir;
            if (ex.naex() > 0){
                maxai = ex.aann(ex.naex()-1);
                minaa = ex.acre(ex.naex()-1) + 1;
            }
            if (ex.nbex() > 0){
                maxbi = ex.bann(ex.nbex()-1);
                minba = ex.bcre(ex.nbex()-1) + 1;
            }

            // Form the reference determinant in the string representation
            ExcitationDeterminant ref_pitzer(ex);
            ref_pitzer.to_pitzer(qt_to_pitzer_);
            StringDeterminant str_ex(reference_determinant_,ref_pitzer);

            ints_->make_fock_diagonal(str_ex.get_alfa_bits(),str_ex.get_beta_bits(),fock_diagonals);
            std::vector<double>& fock_diagonal_alpha = fock_diagonals.first;
            std::vector<double>& fock_diagonal_beta = fock_diagonals.second;

            size_t num_dets_accepted_ex_el = 0;
            for (int i = maxai - 1; i >= minai; --i){
                for (int a = minaa; a < maxaa; ++a){
                    double excitation_energy = fock_diagonal_alpha[qt_to_pitzer_[a]] - fock_diagonal_alpha[qt_to_pitzer_[i]] - ints_->diag_ce_rtei(qt_to_pitzer_[a],qt_to_pitzer_[i]);
                    excitation_energy += +energy_ex - ref_energy_;
                    double absolute_energy = excitation_energy + ref_energy_;
                    double relative_energy = absolute_energy - min_energy_;

                    if (relative_energy < denominator_threshold_){
                        ExcitationDeterminant ia_ex(ex);
                        ia_ex.add_alpha_ex(i,a);
                        num_dets_visited_ex++;
                        num_dets_visited++;

                        int irrep_ia_ex = irrep_ex ^ (mo_symmetry_qt_[i] ^ mo_symmetry_qt_[a]);

                        // check if this string is in our database
                        std::pair<size_t,size_t> ab_str;
                        if (alpha_strings_map.find(ia_ex.alpha_ops()) == alpha_strings_map.end()){
                            alpha_strings_map[ia_ex.alpha_ops()] = nastr;
                            ab_str.first = nastr;
                            nastr += 1;
                        }else{
                            ab_str.first = alpha_strings_map[ia_ex.alpha_ops()];
                        }
                        if (beta_strings_map.find(ia_ex.beta_ops()) == beta_strings_map.end()){
                            beta_strings_map[ia_ex.beta_ops()] = nbstr;
                            ab_str.second = nbstr;
                            nbstr += 1;
                        }else{
                            ab_str.second = beta_strings_map[ia_ex.beta_ops()];
                        }

                        if (good_determinants.find(ab_str) == good_determinants.end()){
                            good_determinants[ab_str] = num_dets_visited;
                            new_elements.push_back(boost::make_tuple(absolute_energy,irrep_ia_ex,ia_ex));
                            if ((relative_energy < determinant_threshold_) and (irrep_ia_ex == wavefunction_symmetry_)){
                                num_dets_accepted_ex++;
                                num_dets_accepted_ex_el++;
                                selected_determinants.push_back(boost::make_tuple(absolute_energy,irrep_ia_ex,ia_ex));
                                min_energy_ = std::min(min_energy_,absolute_energy);
                                max_energy_ = std::max(max_energy_,absolute_energy);
                            }
                        }
                    }
                }
            }
            for (int i = maxbi - 1; i >= minbi; --i){
                for (int a = minba; a < maxba; ++a){
                    double excitation_energy = fock_diagonal_beta[qt_to_pitzer_[a]] - fock_diagonal_beta[qt_to_pitzer_[i]] - ints_->diag_ce_rtei(qt_to_pitzer_[a],qt_to_pitzer_[i]);
                    excitation_energy += +energy_ex - ref_energy_;
                    double absolute_energy = excitation_energy + ref_energy_;
                    double relative_energy = absolute_energy - min_energy_;

                    if (relative_energy < denominator_threshold_){
                        ExcitationDeterminant ia_ex(ex);
                        ia_ex.add_beta_ex(i,a);
                        num_dets_visited_ex++;
                        num_dets_visited++;

                        int irrep_ia_ex = irrep_ex ^ (mo_symmetry_qt_[i] ^ mo_symmetry_qt_[a]);

                        // check if this string is in our database
                        std::pair<size_t,size_t> ab_str;
                        if (alpha_strings_map.find(ia_ex.alpha_ops()) == alpha_strings_map.end()){
                            alpha_strings_map[ia_ex.alpha_ops()] = nastr;
                            ab_str.first = nastr;
                            nastr += 1;
                        }else{
                            ab_str.first = alpha_strings_map[ia_ex.alpha_ops()];
                        }
                        if (beta_strings_map.find(ia_ex.beta_ops()) == beta_strings_map.end()){
                            beta_strings_map[ia_ex.beta_ops()] = nbstr;
                            ab_str.second = nbstr;
                            nbstr += 1;
                        }else{
                            ab_str.second = beta_strings_map[ia_ex.beta_ops()];
                        }

                        if (good_determinants.find(ab_str) == good_determinants.end()){
                            good_determinants[ab_str] = num_dets_visited;
                            new_elements.push_back(boost::make_tuple(absolute_energy,irrep_ia_ex,ia_ex));
                            if ((relative_energy < determinant_threshold_) and (irrep_ia_ex == wavefunction_symmetry_)){
                                num_dets_accepted_ex++;
                                num_dets_accepted_ex_el++;
                                selected_determinants.push_back(boost::make_tuple(absolute_energy,irrep_ia_ex,ia_ex));
                                min_energy_ = std::min(min_energy_,absolute_energy);
                                max_energy_ = std::max(max_energy_,absolute_energy);
                            }
                        }
                    }
                }
            }
            if (num_dets_accepted_ex_el == 0){
                failed_attepts += 1;
            }
        }
        fprintf(outfile,"\n  %2d   %12ld   %12ld   %12ld",level + 1,num_dets_visited_ex,num_dets_accepted_ex,failed_attepts);

        to_be_processed_elements = new_elements;
        level += 1;
        if (level > 10)
            iterate = false;
        if (new_elements.size() == 0)
            iterate = false;
    }
    double time_dets = t_dets.elapsed();

    fprintf(outfile,"\n\n  The determinants visited fall in the range [%f,%f]",min_energy_,max_energy_);
//    fprintf(outfile,"\n\n  Number of full ci determinants    = %llu",num_total_dets);
    fprintf(outfile,"\n\n  Number of determinants visited    = %ld (%e)",num_dets_visited,0.0);
    fprintf(outfile,"\n  Number of determinants accepted   = %ld (%e)",selected_determinants.size(),0.0);
    fprintf(outfile,"\n  Time spent on generating dets     = %f s",time_dets);
    fprintf(outfile,"\n  Precompute algorithm time elapsed = %f s",t.elapsed());
    fflush(outfile);
}


}} // EndNamespaces
