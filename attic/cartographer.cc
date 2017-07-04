/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <cmath>

#include "mini-boost/boost/format.hpp"
#include "mini-boost/boost/timer.hpp"

#include "cartographer.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

Cartographer::Cartographer(Options& options, double min_energy, double max_energy)
    : dod_percent_margin_(1), dod_type_(HistogramDOD), dod_bin_width_(0.05),
      dod_fname_("density_of_determinants.txt"), dod_file_(0), dettour_fname_("dettour.txt"),
      dettour_file_(0) {
    if (options.get_str("DOD_FORMAT") == "GAUSSIAN") {
        dod_type_ = GaussianDOD;
    } else if (options.get_str("DOD_FORMAT") == "HISTOGRAM") {
        dod_type_ = HistogramDOD;
    }
    dod_bin_width_ = options.get_double("DOD_BIN_WIDTH");

    min_energy_ = int(min_energy / dod_bin_width_) * dod_bin_width_;
    max_energy_ = int(max_energy / dod_bin_width_) * dod_bin_width_;

    ndod_bins_margin_ = 20;
    ndod_bins_center_ = 1 + (max_energy - min_energy) / dod_bin_width_;
    ndod_bins_total_ = ndod_bins_center_ + 2 * ndod_bins_margin_;

    dod_.assign(2 * ndod_bins_margin_ + ndod_bins_center_, 0.0);
    write_file_ = options.get_bool("DETTOUR_WRITE_FILE");
    write_occupation_ = options.get_bool("WRITE_OCCUPATION");
    write_det_energy_ = options.get_bool("WRITE_DET_ENERGY");
    write_den_energy_ = options.get_bool("WRITE_DEN_ENERGY");
    write_excitation_level_ = options.get_bool("WRITE_EXC_LEVEL");
    restrict_excitation_ = options.get_int("RESTRICT_EXCITATION");

    outfile->Printf("\n\n  Cartographer initialized with range [%f,%f]", min_energy_, max_energy_);
    outfile->Printf("\n  The %s bin width is %f Hartree\n",
                    (dod_type_ == GaussianDOD ? "Gaussian" : "histogram"), dod_bin_width_);
    outfile->Printf("\n  The density of determinants will be sampled at %d-points",
                    ndod_bins_center_ + 2 * ndod_bins_margin_);

    if (dod_type_ == GaussianDOD) {
        int gaussian_half_width =
            int(2.5 * double(ndod_bins_center_) * dod_bin_width_ / (max_energy_ - min_energy_));
        for (int i = -gaussian_half_width; i <= gaussian_half_width; ++i) {
            double de = i * (max_energy_ - min_energy_) / double(ndod_bins_center_);
            double value = std::exp(-de * de / (2.0 * dod_bin_width_ * dod_bin_width_)) /
                           (dod_bin_width_ * 2.50662827463);
            dod_contribution_.push_back(value);
        }
    }
    if (dod_type_ == HistogramDOD) {
        int histogram_width =
            int(0.5 * double(ndod_bins_center_) * dod_bin_width_ / (max_energy_ - min_energy_));
        for (int i = -histogram_width; i <= histogram_width; ++i) {
            dod_contribution_.push_back(1.0 / dod_bin_width_);
        }
    }

    // Create the Pitzer to QT mapping
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    SharedVector epsilon_a = wfn->epsilon_a();
    std::vector<std::pair<double, int>> e_mo_pair;
    for (int h = 0, q = 0; h < epsilon_a->nirrep(); ++h) {
        for (int p = 0; p < epsilon_a->dim(h); ++p) {
            e_mo_pair.push_back(std::make_pair(epsilon_a->get(h, p), q));
            q += 1;
        }
    }
    std::sort(e_mo_pair.begin(), e_mo_pair.end());
    for (int p = 0; p < e_mo_pair.size(); ++p) {
        pitzer_to_qt_.push_back(e_mo_pair[p].second);
    }
}

Cartographer::~Cartographer() {
    write_dod();
    write_dod_gnuplot_input();
    write_dettour();
}

void Cartographer::accumulate_data(int nmo, std::vector<bool>& Ia, std::vector<bool>& Ib,
                                   double det_energy, double a_den_energy, double b_den_energy,
                                   int naex, int nbex) {
    accumulate_dod(det_energy);
    accumulate_dettour(nmo, Ia, Ib, det_energy, a_den_energy, b_den_energy, naex, nbex);
}

void Cartographer::accumulate_dod(double det_energy) {
    // Density of determinants
    if (dod_type_ == HistogramDOD) {
        int position = ndod_bins_margin_ +
                       int((det_energy - (min_energy_ + 0.5 * dod_bin_width_)) / dod_bin_width_);
        if (position >= 0 and position < ndod_bins_total_) {
            dod_[position] += 1.0;
        } else {
            outfile->Printf(
                "\n  Cartographer Warning: The datapoint %f is out of the range [%f,%f]",
                det_energy, min_energy_, max_energy_);
        }
    }
}

/**
 * Write the occupation number representation of a determinant and its energy.
 * If the dettour output file pointer (dettour_file_) is not allocated it will
 * create a new output file.
 *
 * For occupied strings it returns the sum of the fock matrix elements corresponding
 * to the zeros, e.g. 111010 -> -fock[3][3] - fock[5][5]
 *
 * For virtual strings it returns the sum of the fock matrix elements corresponding
 * to the ones, e.g. 010010 -> +fock[1][1] + fock[4][4]
 *
 * @param os - an std::ofstream object that will receive the output
 * @param Ia - the alpha string
 * @param Ib - the beta string
 * @param energy - the energy of the determinant
 * @return
 */
void Cartographer::accumulate_dettour(int nmo, std::vector<bool>& Ia, std::vector<bool>& Ib,
                                      double det_energy, double a_den_energy, double b_den_energy,
                                      int naex, int nbex) {
    if (write_file_) {
        if (dettour_file_ == 0) {
            dettour_file_ = new std::ofstream(dettour_fname_.c_str());
        }
        if (restrict_excitation_ != 0) {
            if (naex + nbex != restrict_excitation_) {
                return;
            }
        }

        if (write_occupation_) {
            for (int p = 0; p < nmo; ++p) {
                *dettour_file_ << Ia[pitzer_to_qt_[p]];
            }
            *dettour_file_ << " ";
            for (int p = 0; p < nmo; ++p) {
                *dettour_file_ << Ib[pitzer_to_qt_[p]];
            }
        }
        if (write_det_energy_)
            *dettour_file_ << boost::format(" %.12f") % det_energy;
        if (write_den_energy_)
            *dettour_file_ << boost::format(" %.12f") % (a_den_energy + b_den_energy);
        if (write_excitation_level_) {
            *dettour_file_ << boost::format(" %d") % (naex + nbex);
        }
        *dettour_file_ << endl;
    }
}

void Cartographer::write_dod() {
    outfile->Printf("\n  Cartographer is writing the density of determinants to the file: %s\n",
                    dod_fname_.c_str());
    dod_file_ = new std::ofstream(dod_fname_.c_str());
    double margin_energy_offset = -static_cast<double>(ndod_bins_margin_) * dod_bin_width_;
    for (int i = 0; i < ndod_bins_total_; ++i) {
        double energy =
            margin_energy_offset + min_energy_ + dod_bin_width_ * static_cast<double>(i);
        *dod_file_ << boost::format("%.9f %12e\n") % energy % dod_[i];
    }
    dod_file_->close();
    delete dod_file_;
}

void Cartographer::write_dettour() {
    if (write_file_) {
        outfile->Printf("\n  Cartographer is writing the dettour output to the file: %s\n",
                        dettour_fname_.c_str());
        dettour_file_->close();
        delete dettour_file_;
    }
}

void Cartographer::write_dod_gnuplot_input() {
    double dod_min, dod_max;
    double margin_energy_offset = -static_cast<double>(ndod_bins_margin_) * dod_bin_width_;
    for (int i = 0; i < ndod_bins_total_; ++i) {
        if (dod_[i] != 0) {
            dod_min = margin_energy_offset + min_energy_ + dod_bin_width_ * static_cast<double>(i);
            break;
        }
    }
    for (int i = ndod_bins_total_ - 1; i >= 0; --i) {
        if (dod_[i] != 0) {
            dod_max = margin_energy_offset + min_energy_ + dod_bin_width_ * static_cast<double>(i);
            break;
        }
    }
    double maxsum = 0;
    for (int i = 0; i < ndod_bins_total_; ++i) {
        maxsum = std::max(maxsum, dod_[i]);
    }

    dod_min = int((dod_min - 0.3) * 10.) / 10.0;
    dod_max = int((dod_max + 0.3) * 10.) / 10.0;

    outfile->Printf("\n  Cartographer is preparing a gnuplot file for the range [%.3f,%.3f]",
                    dod_min, dod_max);

    std::string gnuplot_input;
    gnuplot_input += boost::str(boost::format("set xrange [%.3f:%.3f]\n") % dod_min % dod_max);
    gnuplot_input += boost::str(boost::format("set yrange [%.3f:%.3f]\n") % 0 % (maxsum + 2));

    gnuplot_input += "set xtics nomirror\n";
    gnuplot_input += "set ytics nomirror\n";

    gnuplot_input += "set ytics scale 0.75\n";
    gnuplot_input += "set xtics scale 0.75\n";

    double l = std::log10(maxsum + 1.0);
    int ytics = 2;
    if (l > 1.041) {
        ytics = 5 * std::pow(10.0, int(l - 1));
    }

    gnuplot_input +=
        boost::str(boost::format("set ytics %d,%d,%d\n") % ytics % ytics % int(maxsum + 2));

    gnuplot_input += "set border 3\n";
    gnuplot_input += "set xzeroaxis\n";

    gnuplot_input += "set format x \"%.0f\"\n";
    //

    gnuplot_input += "set terminal postscript portrait enhanced monochrome\n"
                     "set output 'dod_hist.eps'\n"
                     "set terminal postscript size 6,3.0\n"
                     "set style fill solid 0.25 noborder\n"
                     "set xlabel \"Energy (E_h)\"\n"
                     "plot 'density_of_determinants.txt' using 1:2 notitle with boxes lc rgb "
                     "\"black\" fs solid 1";

    std::ofstream ofs("dod_hist.plt");
    ofs << gnuplot_input;
    //    set xtics -109,1,-104''
}
}
} // EndNamespaces
