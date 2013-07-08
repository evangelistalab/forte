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
 * Analyze the space of excited configurations
 */
void Explorer::analyze()
{
//    fprintf(outfile,"\n\n  Analyzing the space of Slater determinants\n");

//    // Open the file that will contain the determinants and energies
//    ofstream os;
//    os.open ("det_energy_plot.txt");

//    // Sort the determinants by energy
//    std::sort(determinants_.begin(),determinants_.end());

//    size_t ndets = determinants_.size();
//    double det_energy;
//    int ha,hb,sa,sb;
//    for (auto det : determinants_){
//        std::tie(det_energy,ha,sa,sb) = det;
//        hb = wavefunction_symmetry_ ^ ha;

//        if (write_occupation_){
//            std::vector<bool> Ia = vec_astr_symm_[ha][sa].second;
//            std::vector<bool> Ib = vec_astr_symm_[hb][sb].second;
//            for (int p = 0; p < nmo_; ++p){
//                os << Ia[qt_to_pitzer_[p]];
//            }
//            os << " ";
//            for (int p = 0; p < nmo_; ++p){
//                os << Ib[qt_to_pitzer_[p]];
//            }
//        }
//        if (write_det_energy_) os << boost::format(" %.12f") % det_energy;
//        if (write_den_energy_){
//        double compute_denominator(bool is_occ,bool *begin, bool *end,std::vector<double>& epsilon);
//            os << boost::format(" %.12f") % den_energy;
//        }
//        os << endl;
//    }
//    std::vector<std::tuple<double,double,int>> det_den_energy;

//    std::sort(det_den_energy.begin(),det_den_energy.end());
//    size_t npairs = det_den_energy.size();
//    double min_distance = 0.0005;
//    std::vector<std::pair<double,double>> reduced_pairs;
//    size_t range_check = 100;
//    size_t nreduced = 0;
//    for (size_t i = 0; i < range_check; ++i){
//        reduced_pairs.push_back(det_den_energy[i]);
//        nreduced++;
//    }

//    for (size_t i = range_check; i < npairs; ++i){
//        double det_energy = det_den_energy[i].first;
//        double den_energy = det_den_energy[i].second;
//        bool keep_pair = true;
//        for (int j = 0; j < range_check; ++j){
//            double det_test = reduced_pairs[nreduced-j-1].first;
//            double den_test = reduced_pairs[nreduced-j-1].second;
//            double distance = (det_energy - det_test) * (det_energy - det_test) + (den_energy - den_test) * (den_energy - den_test);
//            if (distance  < min_distance) keep_pair = false;
//        }
//        if (keep_pair){
//            reduced_pairs.push_back(det_den_energy[i]);
//            nreduced++;
//        }
//    }
//    fprintf(outfile,"\n  Number of points saved for plotting: %d (%d)",int(reduced_pairs.size()),int(npairs));


//    ofstream osplot;
//    osplot.open ("det_energy_plot.txt");
//    for (size_t i = 0, maxi = reduced_pairs.size(); i < maxi; ++i){
//        osplot << boost::format("%f %f\n") % reduced_pairs[i].first % reduced_pairs[i].second;
//    }
}

}} // EndNamespaces
