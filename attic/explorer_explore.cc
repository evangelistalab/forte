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

#include "lambda-ci.h"

#include <cmath>

#include "mini-boost/boost/format.hpp"
#include "mini-boost/boost/timer.hpp"

#include "cartographer.h"
#include "lambda-ci.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

/**
 * An ancillary function to compare the det_info data structures.  Used to sort determinants.
 * @param t1
 * @param t2
 * @return
 */
bool compare_det_info(const det_info& t1, const det_info& t2) {
    if (t1.get<0>() != t2.get<0>()) {
        return (t1.get<0>() < t2.get<0>());
    } else if (t1.get<1>() != t2.get<1>()) {
        return (t1.get<1>() < t2.get<1>());
    } else if (t1.get<2>() != t2.get<2>()) {
        return (t1.get<2>() < t2.get<2>());
    } else if (t1.get<3>() != t2.get<3>()) {
        return (t1.get<3>() < t2.get<3>());
    }
    return (t1.get<4>() < t2.get<4>());
}

/**
 * Find all the Slater determinants with an energy lower than determinant_threshold_
 */
void LambdaCI::explore_original(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Exploring the space of Slater determinants\n");

    // No explorer will succeed without a cartographer
    Cartographer cg(options, min_energy_, min_energy_ + determinant_threshold_);

    int nfrzc = frzcpi_.sum();
    int nfrzv = frzvpi_.sum();
    int naocc = nalpha_ - rdoccpi_.sum();
    int nbocc = nbeta_ - rdoccpi_.sum();
    int navir = ncmo_ - naocc - rdoccpi_.sum() - ruoccpi_.sum();
    int nbvir = ncmo_ - nbocc - rdoccpi_.sum() - ruoccpi_.sum();

    // Calculate the maximum excitation level
    maxnaex_ = std::min(naocc, navir);
    maxnbex_ = std::min(nbocc, nbvir);
    minnex_ = options.get_int("MIN_EXC_LEVEL");
    maxnex_ = maxnaex_ + maxnbex_;
    if (options["MAX_EXC_LEVEL"].has_changed()) {
        maxnex_ = options.get_int("MAX_EXC_LEVEL");
        maxnaex_ = std::min(maxnex_, maxnaex_);
        maxnbex_ = std::min(maxnex_, maxnbex_);
    }

    // Allocate an array of bits for fast manipulation
    bool* Ia = new bool[2 * ncmo_];
    bool* Ib = &Ia[ncmo_];

    ForteTimer t;
    double time_string = 0.0;
    double time_dets = 0.0;
    long num_dets_visited = 0;
    long num_dets_accepted = 0;
    unsigned long long num_total_dets = 0;
    unsigned long num_permutations = 0;

    // Count the total number of determinants
    for (int nex = minnex_; nex <= maxnex_; ++nex) {
        for (int naex = std::max(0, nex - maxnbex_); naex <= std::min(maxnaex_, nex); ++naex) {
            int nbex = nex - naex;
            // count the number of determinants
            unsigned long long num_dets_total_class = choose(naocc, naex) * choose(navir, naex) *
                                                      choose(nbocc, nbex) * choose(nbvir, nbex);
            num_total_dets += num_dets_total_class;
        }
    }

    // Generate all alpha and beta strings with energy < threshold
    // The strings are in QT format and are stored using the following structure:
    // [<string irrep>][<string index>](<string energy>,<string structure>)
    outfile->Printf("\n  +++ Screening the alpha strings +++\n");

    ForteTimer timer_astr;
    vec_astr_symm_ = compute_strings_screened(epsilon_a_qt_, naocc, navir, maxnaex_, true);
    outfile->Printf("\n  Time required: %f s", timer_astr.elapsed());

    outfile->Printf("\n\n  +++ Screening the beta strings +++\n");

    ForteTimer timer_bstr;
    vec_bstr_symm_ = compute_strings_screened(epsilon_b_qt_, nbocc, nbvir, maxnbex_, false);
    outfile->Printf("\n  Time required: %f s", timer_bstr.elapsed());

    outfile->Printf("\n  denominator_threshold_: %f s", denominator_threshold_);

    std::ofstream de("det_energy.txt");
    double ref_one_electron_energy = min_energy_determinant_.one_electron_energy();

    std::vector<bool> empty_det(2 * ncmo_, false);
    StringDeterminant det(empty_det);
    ForteTimer t_dets;
    // Loop over the excitation level
    for (int nex = minnex_; nex <= maxnex_; ++nex) {
        size_t num_dets_accepted_ex = 0;
        size_t num_dets_visited_ex = 0;
        for (int naex = std::max(0, nex - maxnbex_); naex <= std::min(maxnaex_, nex); ++naex) {
            int nbex = nex - naex;
            // Loop over the irreps
            for (int ha = 0; ha < nirrep_; ++ha) {
                int hb = wavefunction_symmetry_ ^ ha;
                int exc_class_a = excitation_class(naex, ha);
                int exc_class_b = excitation_class(nbex, hb);
                string_list& vec_astr = vec_astr_symm_[exc_class_a];
                string_list& vec_bstr = vec_bstr_symm_[exc_class_b];
                size_t nsa = vec_astr.size();
                size_t nsb = vec_bstr.size();
                size_t num_dets_accepted_ex_irrep = 0;
                size_t num_dets_visited_ex_irrep = 0;
                // Loop over alpha strings
                for (size_t sa = 0; sa < nsa; ++sa) {
                    double ea = vec_astr[sa].get<0>();
                    double da = vec_astr[sa].get<1>();
                    std::vector<bool>& str_sa = vec_astr[sa].get<2>();
                    for (int p = 0; p < ncmo_; ++p)
                        Ia[p] = str_sa[p];

                    // Loop over beta strings
                    for (size_t sb = 0; sb < nsb; ++sb) {
                        double eb = vec_bstr[sb].get<0>();
                        double db = vec_bstr[sb].get<1>();
                        if (ea + eb < denominator_threshold_) {
                            std::vector<bool>& str_sb = vec_bstr[sb].get<2>();
                            for (int p = 0; p < ncmo_; ++p)
                                Ib[p] = str_sb[p];

                            // set the alpha/beta strings and compute the energy of this determinant
                            det.set_bits(Ia, Ib);
                            //                            double det_energy = det.energy() +
                            //                            nuclear_repulsion_energy_;
                            //                            double det_energy =
                            //                            det.excitation_energy(reference_determinant_)
                            //                            + min_energy_;
                            double det_energy = det.excitation_ab_energy(reference_determinant_) +
                                                da + db + ref_energy_;

                            // check to see if the energy is below a given threshold
                            if (det_energy < min_energy_ + determinant_threshold_) {
                                cg.accumulate_data(ncmo_, str_sa, str_sb, det_energy, ea, eb, naex,
                                                   nbex);
                                determinants_.push_back(boost::make_tuple(det_energy, exc_class_a,
                                                                          sa, exc_class_b, sb));
                                double one_electron_energy =
                                    det.one_electron_energy() - ref_one_electron_energy;
                                de << boost::format("%.9f %.9f %.9f\n") % det_energy % (ea + eb) %
                                          one_electron_energy;
                                if (det_energy < min_energy_) {
                                    min_energy_determinant_ = det;
                                }
                                min_energy_ = std::min(min_energy_, det_energy);
                                max_energy_ = std::max(max_energy_, det_energy);
                                num_dets_accepted++;
                                num_dets_accepted_ex++;
                                num_dets_accepted_ex_irrep++;
                            }
                            num_dets_visited++;
                            num_dets_visited_ex++;
                            num_dets_visited_ex_irrep++;
                        } else {
                            break; // since the strings are ordered by energy, if you are here we
                                   // can just skip this loop
                        }
                    }
                }
            }
        }
        outfile->Printf("\n  %2d   %12ld   %12ld", nex, num_dets_visited_ex, num_dets_accepted_ex);

        if ((num_dets_accepted_ex == 0) and (nex > 4)) {
            outfile->Printf("\n\n  Excitation level %d produced no determinants, finishing search",
                            nex);

            break;
        }
    }
    time_dets += t_dets.elapsed();
    delete[] Ia;

    // sort the determinants
    outfile->Printf("\n\n  Sorting the determinants according to their energy.");
    std::sort(determinants_.begin(), determinants_.end(), compare_det_info);

    // delete determinants that lie above the determinant threshold
    size_t ndets = determinants_.size();
    double det_cutoff = min_energy_ + determinant_threshold_;
    size_t ndets_deleted = 0;
    while ((not determinants_.empty()) and (determinants_.back().get<0>() > det_cutoff)) {
        determinants_.pop_back();
        ndets_deleted++;
    }
    outfile->Printf("\n\n  %ld determinants were deleted because their energy is above the "
                    "determinat threshold.",
                    ndets_deleted);

    outfile->Printf("\n\n  The new reference determinant is:");
    reference_determinant_.print();
    outfile->Printf("\n  and its energy: %.12f Eh", min_energy_);

    outfile->Printf("\n  Occupation numbers of the minimum energy determinant:");
    outfile->Printf("|");
    for (int h = 0, p = 0; h < nirrep_; ++h) {
        int n = 0;
        for (int i = 0; i < ncmopi_[h]; ++i) {
            n += min_energy_determinant_.get_alfa_bit(p);
            p += 1;
        }
        outfile->Printf(" %d", n);
    }
    outfile->Printf(" > x ");
    outfile->Printf("|");
    for (int h = 0, p = 0; h < nirrep_; ++h) {
        int n = 0;
        for (int i = 0; i < ncmopi_[h]; ++i) {
            n += min_energy_determinant_.get_beta_bit(p);
            p += 1;
        }
        outfile->Printf(" %d", n);
    }
    outfile->Printf(" >");

    outfile->Printf("\n\n  The determinants visited fall in the range [%f,%f]", min_energy_,
                    max_energy_);

    outfile->Printf("\n\n  Number of full ci determinants    = %llu", num_total_dets);
    outfile->Printf("\n\n  Number of determinants visited    = %ld (%e)", num_dets_visited,
                    double(num_dets_visited) / double(num_total_dets));
    outfile->Printf("\n  Number of determinants accepted   = %ld (%e)", determinants_.size(),
                    double(determinants_.size()) / double(num_total_dets));
    outfile->Printf("\n  Time spent on generating strings  = %f s", time_string);
    outfile->Printf("\n  Time spent on generating dets     = %f s", time_dets);
    outfile->Printf("\n  Precompute algorithm time elapsed = %f s", t.elapsed());
}

/**
 * Find all the Slater determinants with an energy lower than determinant_threshold_
 */
void LambdaCI::explore(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Exploring the space of Slater determinants\n");

    // No explorer will succeed without a cartographer
    Cartographer cg(options, min_energy_, min_energy_ + determinant_threshold_);

    int nfrzc = ints_->frzcpi().sum();
    int nfrzv = ints_->frzvpi().sum();
    int naocc = nalpha_ - nfrzc;
    int nbocc = nbeta_ - nfrzc;
    int navir = ncmo_ - naocc - nfrzc - nfrzv;
    int nbvir = ncmo_ - nbocc - nfrzc - nfrzv;

    // Calculate the maximum excitation level
    maxnaex_ = std::min(naocc, navir);
    maxnbex_ = std::min(nbocc, nbvir);
    minnex_ = options.get_int("MIN_EXC_LEVEL");
    maxnex_ = maxnaex_ + maxnbex_;
    if (options["MAX_EXC_LEVEL"].has_changed()) {
        maxnex_ = options.get_int("MAX_EXC_LEVEL");
        maxnaex_ = std::min(maxnex_, maxnaex_);
        maxnbex_ = std::min(maxnex_, maxnbex_);
    }

    // Allocate an array of bits for fast manipulation
    bool* Ia = new bool[2 * ncmo_];
    bool* Ib = &Ia[ncmo_];

    ForteTimer t;
    double time_string = 0.0;
    double time_dets = 0.0;
    long num_dets_visited = 0;
    long num_dets_accepted = 0;
    unsigned long long num_total_dets = 0;
    unsigned long num_permutations = 0;

    // Count the total number of determinants
    for (int nex = minnex_; nex <= maxnex_; ++nex) {
        for (int naex = std::max(0, nex - maxnbex_); naex <= std::min(maxnaex_, nex); ++naex) {
            int nbex = nex - naex;
            // count the number of determinants
            unsigned long long num_dets_total_class = choose(naocc, naex) * choose(navir, naex) *
                                                      choose(nbocc, nbex) * choose(nbvir, nbex);
            num_total_dets += num_dets_total_class;
        }
    }

    // Generate all alpha and beta strings with energy < threshold
    // The strings are in QT format and are stored using the following structure:
    // [<string irrep>][<string index>](<string energy>,<string structure>)
    outfile->Printf("\n  +++ Screening the alpha strings +++\n");

    ForteTimer timer_astr;
    vec_astr_symm_ = compute_strings_screened(epsilon_a_qt_, naocc, navir, maxnaex_, true);
    outfile->Printf("\n  Time required: %f s", timer_astr.elapsed());

    outfile->Printf("\n\n  +++ Screening the beta strings +++\n");

    ForteTimer timer_bstr;
    vec_bstr_symm_ = compute_strings_screened(epsilon_b_qt_, nbocc, nbvir, maxnbex_, false);
    outfile->Printf("\n  Time required: %f s", timer_bstr.elapsed());

    std::vector<bool> empty_det(2 * ncmo_, false);
    StringDeterminant det(empty_det);
    ForteTimer t_dets;
    // Loop over the excitation level
    outfile->Printf("\n\n==================================");
    outfile->Printf("\n Exc. lvl    Tested       Accepted");
    outfile->Printf("\n==================================");
    for (int nex = minnex_; nex <= maxnex_; ++nex) {
        size_t num_dets_accepted_ex = 0;
        size_t num_dets_visited_ex = 0;
        for (int naex = std::max(0, nex - maxnbex_); naex <= std::min(maxnaex_, nex); ++naex) {
            int nbex = nex - naex;
            //            outfile->Printf("\n  (ex,naex,nbex) = (%d,%d,%d)",nex,naex,nbex);
            // Loop over the irreps
            for (int ha = 0; ha < nirrep_; ++ha) {
                int hb = wavefunction_symmetry_ ^ ha;
                int exc_class_a = excitation_class(naex, ha);
                int exc_class_b = excitation_class(nbex, hb);
                string_list& vec_astr = vec_astr_symm_[exc_class_a];
                string_list& vec_bstr = vec_bstr_symm_[exc_class_b];
                size_t nsa = vec_astr.size();
                size_t nsb = vec_bstr.size();
                size_t num_dets_accepted_ex_irrep = 0;
                size_t num_dets_visited_ex_irrep = 0;

                int nslices = 20;
                int empty_layers = 0;
                size_t layer_count[20][20];
                for (int i = 0; i < 20; ++i) {
                    for (int j = 0; j < 20; ++j) {
                        layer_count[i][j] = 0;
                    }
                }
                for (int l = 0; l < nslices; ++l) {
                    size_t num_dets_accepted_l = 0;
                    size_t num_dets_visited_l = 0;
                    for (int la = 0; la <= l; ++la) {
                        size_t num_dets_accepted_la_lb = 0;
                        int lb = l - la;
                        double minea =
                            double(la) * denominator_threshold_ / static_cast<double>(nslices);
                        double maxea =
                            double(la + 1) * denominator_threshold_ / static_cast<double>(nslices);
                        double mineb =
                            double(lb) * denominator_threshold_ / static_cast<double>(nslices);
                        double maxeb =
                            double(lb + 1) * denominator_threshold_ / static_cast<double>(nslices);

                        // find the range of sa that lies in [minea,maxea)
                        size_t minsa = -1, maxsa = -1;
                        for (size_t sa = 0; sa < nsa; ++sa) {
                            double ea = vec_astr[sa].get<0>();
                            //                            outfile->Printf("\n ea[%3ld] = %f
                            //                            (%ld,%ld)",sa,ea,minsa,maxsa);
                            if (ea >= minea and minsa == -1)
                                minsa = sa;
                            if (ea >= maxea and maxsa == -1)
                                maxsa = sa;
                        }
                        // find the range of sb that lies in [mineb,maxeb)
                        size_t minsb = -1, maxsb = -1;
                        for (size_t sb = 0; sb < nsb; ++sb) {
                            double eb = vec_bstr[sb].get<0>();
                            if (eb >= mineb and minsb == -1)
                                minsb = sb;
                            if (eb >= maxeb and maxsb == -1)
                                maxsb = sb;
                        }
                        if (maxsa == -1)
                            maxsa = nsa;
                        if (maxsb == -1)
                            maxsb = nsb;

                        //                        outfile->Printf("\n  Ranges (la,lb) =
                        //                        (%d,%d)",la,lb);
                        //                        outfile->Printf("\n  Ranges ea = [%f,%f]  eb =
                        //                        [%f,%f]",minea,maxea,mineb,maxeb);
                        //                        outfile->Printf("\n  Ranges sa = [%ld,%ld]  sb =
                        //                        [%ld,%ld]",minsa,maxsa,minsb,maxsb);
                        // Loop over alpha strings
                        for (size_t sa = minsa; sa < maxsa; ++sa) {
                            double ea = vec_astr[sa].get<0>();
                            double da = vec_astr[sa].get<1>();
                            std::vector<bool>& str_sa = vec_astr[sa].get<2>();
                            for (int p = 0; p < ncmo_; ++p)
                                Ia[p] = str_sa[p];

                            // Loop over beta strings
                            for (size_t sb = minsb; sb < maxsb; ++sb) {
                                double eb = vec_bstr[sb].get<0>();
                                double db = vec_bstr[sb].get<1>();
                                if (ea + eb < denominator_threshold_) {
                                    std::vector<bool>& str_sb = vec_bstr[sb].get<2>();
                                    for (int p = 0; p < ncmo_; ++p)
                                        Ib[p] = str_sb[p];

                                    // set the alpha/beta strings and compute the energy of this
                                    // determinant
                                    det.set_bits(Ia, Ib);
                                    //                            double det_energy = det.energy() +
                                    //                            nuclear_repulsion_energy_;
                                    //                            double det_energy =
                                    //                            det.excitation_energy(reference_determinant_)
                                    //                            + min_energy_;
                                    double det_energy =
                                        det.excitation_ab_energy(reference_determinant_) + da + db +
                                        ref_energy_;

                                    // check to see if the energy is below a given threshold
                                    if (det_energy < min_energy_ + determinant_threshold_) {
                                        cg.accumulate_data(ncmo_, str_sa, str_sb, det_energy, ea,
                                                           eb, naex, nbex);
                                        determinants_.push_back(boost::make_tuple(
                                            det_energy, exc_class_a, sa, exc_class_b, sb));
                                        if (det_energy < min_energy_) {
                                            min_energy_determinant_ = det;
                                        }
                                        min_energy_ = std::min(min_energy_, det_energy);
                                        max_energy_ = std::max(max_energy_, det_energy);
                                        num_dets_accepted++;
                                        num_dets_accepted_ex++;
                                        num_dets_accepted_ex_irrep++;
                                        num_dets_accepted_l++;
                                        num_dets_accepted_la_lb++;
                                    }
                                    num_dets_visited++;
                                    num_dets_visited_ex++;
                                    num_dets_visited_ex_irrep++;
                                    num_dets_visited_l++;
                                } else {
                                    break; // since the strings are ordered by energy, if you are
                                           // here we can just skip this loop
                                }
                            }
                        }
                        layer_count[la][lb] = num_dets_accepted_la_lb;
                    }
                    //                    outfile->Printf("\n-->  %2d   %12ld   %12ld
                    //                    %d",l,num_dets_visited_l,num_dets_accepted_l,empty_layers);
                    if ((num_dets_visited_l > 0) and (num_dets_accepted_l == 0)) {
                        empty_layers += 1;
                    }
                    if (empty_layers > 1) {
                        //                        outfile->Printf("\n  stopping here!");
                        break;
                    }
                }
            }
        }
        outfile->Printf("\n  %2d   %12ld   %12ld", nex, num_dets_visited_ex, num_dets_accepted_ex);

        if (num_dets_accepted_ex == 0) {
            outfile->Printf("\n\n  Excitation level %d produced no determinants, finishing search",
                            nex);

            break;
        }
    }
    outfile->Printf("\n==================================\n");
    time_dets += t_dets.elapsed();
    delete[] Ia;

    // sort the determinants
    outfile->Printf("\n\n  Sorting the determinants according to their energy.");
    std::sort(determinants_.begin(), determinants_.end(), compare_det_info);

    // delete determinants that lie above the determinant threshold
    size_t ndets = determinants_.size();
    double det_cutoff = min_energy_ + determinant_threshold_;
    size_t ndets_deleted = 0;

    outfile->Printf("\n  Number of determinants accepted   = %ld (%e)", determinants_.size(),
                    double(num_dets_accepted) / double(num_total_dets));

    while ((not determinants_.empty()) and (determinants_.back().get<0>() > det_cutoff)) {
        determinants_.pop_back();
        ndets_deleted++;
    }
    outfile->Printf("\n\n  %ld determinants were deleted because their energy is above the "
                    "determinat threshold.",
                    ndets_deleted);
    outfile->Printf("\n  Number of determinants accepted   = %ld (%e)", determinants_.size(),
                    double(num_dets_accepted) / double(num_total_dets));

    outfile->Printf("\n\n  The new reference determinant is:");
    reference_determinant_.print();
    outfile->Printf("\n  and its energy: %.12f Eh", min_energy_);

    outfile->Printf("\n  Occupation numbers of the minimum energy determinant:");
    outfile->Printf("|");
    for (int h = 0, p = 0; h < nirrep_; ++h) {
        int n = 0;
        for (int i = 0; i < ncmopi_[h]; ++i) {
            n += min_energy_determinant_.get_alfa_bit(p);
            p += 1;
        }
        outfile->Printf(" %d", n);
    }
    outfile->Printf(" > x ");
    outfile->Printf("|");
    for (int h = 0, p = 0; h < nirrep_; ++h) {
        int n = 0;
        for (int i = 0; i < ncmopi_[h]; ++i) {
            n += min_energy_determinant_.get_beta_bit(p);
            p += 1;
        }
        outfile->Printf(" %d", n);
    }
    outfile->Printf(" >");

    outfile->Printf("\n\n  The determinants visited fall in the range [%f,%f]", min_energy_,
                    max_energy_);

    outfile->Printf("\n\n  Number of full ci determinants    = %llu", num_total_dets);
    outfile->Printf("\n\n  Number of determinants visited    = %ld (%e)", num_dets_visited,
                    double(num_dets_visited) / double(num_total_dets));
    outfile->Printf("\n  Number of determinants accepted   = %ld (%e)", determinants_.size(),
                    double(num_dets_accepted) / double(num_total_dets));
    outfile->Printf("\n  Time spent on generating strings  = %f s", time_string);
    outfile->Printf("\n  Time spent on generating dets     = %f s", time_dets);
    outfile->Printf("\n  Precompute algorithm time elapsed = %f s", t.elapsed());
}
}
} // EndNamespaces
