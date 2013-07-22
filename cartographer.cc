#include "explorer.h"

#include <cmath>

#include <boost/timer.hpp>
#include <boost/format.hpp>

#include "cartographer.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

Cartographer::Cartographer(Options &options,double min_energy,double max_energy)
    :
      min_energy_(min_energy),
      max_energy_(max_energy),
      ndod_bins_(2000),
      dod_percent_margin_(1),
      dod_type_(GaussianDOD),
      dod_gaussian_width_(0.02),
      dod_fname_("density_of_determinants.txt"),
      dettour_fname_("dettour.txt")
{
    dod_.assign(ndod_bins_,0.0);
    fprintf(outfile,"\n  Cartographer initialized with range [%f,%f]\n",min_energy_,max_energy_);

    write_file_ = options.get_bool("WRITE_FILE");
    write_occupation_ = options.get_bool("WRITE_OCCUPATION");
    write_det_energy_ = options.get_bool("WRITE_DET_ENERGY");
    write_den_energy_ = options.get_bool("WRITE_DEN_ENERGY");
    write_excitation_level_ = options.get_bool("WRITE_EXC_LEVEL");
    restrict_excitation_ = options.get_int("RESTRICT_EXCITATION");
    dettour_file_ = 0;
    if (write_file_){
        dettour_file_ = new ofstream(dettour_fname_.c_str());
    }
}

Cartographer::~Cartographer()
{
    write_dod();
    write_dettour();
}

void Cartographer::accumulate_data(int nmo,std::vector<bool>& Ia,std::vector<bool>& Ib,double det_energy,double a_den_energy,double b_den_energy,int naex,int nbex)
{
    accumulate_dod(det_energy);
    accumulate_dettour(nmo,Ia,Ib,det_energy,a_den_energy,b_den_energy,naex,nbex);
}

void Cartographer::accumulate_dod(double det_energy)
{
    // Density of determinants
    // Use 98% of the width to represent the range [min_energy,max_energy] and
    // add a margin on each side to represent 1% of the range
    int ndod_bin_margin = dod_percent_margin_ * ndod_bins_ / 100;
    int ndod_bin_center = ndod_bins_ - 2 * ndod_bin_margin;
    int position = ndod_bin_margin + int(double(ndod_bin_center) * (det_energy - min_energy_) / (max_energy_ - min_energy_));
    if (dod_type_ == GaussianDOD){
        int gaussian_half_width = int(2.5 * double(ndod_bin_center) * dod_gaussian_width_ / (max_energy_ - min_energy_));
        int mini = position - gaussian_half_width;
        int maxi = position + gaussian_half_width;
        for (int i = mini; i <= maxi; ++i){
            double de = (i - position) * (max_energy_ - min_energy_) / double(ndod_bin_center);
            double value = std::exp(-de * de / (2.0 * dod_gaussian_width_ * dod_gaussian_width_));
            if (i >= 0 and i < ndod_bins_){
                dod_[i] += value;
            }
        }
    }
}

/**
 * Write the occupation number representation of a determinant and its energy
 *
 * For occupied strings it returns the sum of the fock matrix elements corresponding
 * to the zeros, e.g. 111010 -> -fock[3][3] - fock[5][5]
 *
 * For virtual strings it returns the sum of the fock matrix elements corresponding
 * to the ones, e.g. 010010 -> +fock[1][1] + fock[4][4]
 *
 * @param os - an ofstream object that will receive the output
 * @param Ia - the alpha string
 * @param Ib - the beta string
 * @param energy - the energy of the determinant
 * @return
 */
void Cartographer::accumulate_dettour(int nmo,std::vector<bool>& Ia,std::vector<bool>& Ib,double det_energy,double a_den_energy,double b_den_energy,int naex,int nbex)
{
    if(write_file_){
        if (restrict_excitation_ != 0){
            if(naex + nbex != restrict_excitation_){
                return;
            }
        }

        if (write_occupation_){
            for (int p = 0; p < nmo; ++p){
                *dettour_file_ << Ia[p];
            }
            *dettour_file_ << " ";
            for (int p = 0; p < nmo; ++p){
                *dettour_file_ << Ib[p];
            }
        }
        if (write_det_energy_) *dettour_file_ << boost::format(" %.12f") % det_energy;
        if (write_den_energy_) *dettour_file_ << boost::format(" %.12f") % (a_den_energy + b_den_energy);
        if (write_excitation_level_){
            *dettour_file_ << boost::format(" %d") % (naex + nbex);
        }
        *dettour_file_ << endl;
    }
}

void Cartographer::write_dod()
{
    fprintf(outfile,"\n  Cartographer is writing the density of determinants to the file: %s\n",dod_fname_.c_str());
    dod_file_ = new ofstream(dod_fname_.c_str());
    int ndod_bin_margin = dod_percent_margin_ * ndod_bins_ / 100;
    int ndod_bin_center = ndod_bins_ - 2 * ndod_bin_margin;
    for (int i = 0; i < ndod_bins_; ++i){
        double energy = min_energy_ + (max_energy_ - min_energy_) * static_cast<double>(i) / double(ndod_bin_center);
        *dod_file_ << boost::format("%.9f %12e\n") % energy % dod_[i];
    }
    dod_file_->close();
    delete dod_file_;
}

void Cartographer::write_dettour()
{
    if (write_file_){
        fprintf(outfile,"\n  Cartographer is writing the dettour output to the file: %s\n",dettour_fname_.c_str());
        dettour_file_->close();
        delete dettour_file_;
    }
}
}} // EndNamespaces
