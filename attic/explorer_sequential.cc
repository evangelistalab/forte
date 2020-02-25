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
#include <set>

#include "mini-boost/boost/format.hpp"
#include "mini-boost/boost/timer.hpp"
#include <boost/unordered_map.hpp>

#include "cartographer.h"
#include "excitation_determinant.h"
#include "lambda-ci.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

/**
 * Find all the Slater determinants with an energy lower than determinant_threshold_
 * by performing single excitations at a time
 */
void LambdaCI::explore_singles_sequential(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Exploring the space of Slater determinants using the singles method\n");
    ForteTimer t;

    // No explorer will succeed without a cartographer
    Cartographer cg(options, min_energy_, min_energy_ + determinant_threshold_);

    //    int nfrzc = frzcpi_.sum();
    //    int nfrzv = frzvpi_.sum();
    int naocc = nalpha_;
    int nbocc = nbeta_;
    int navir = ncmo_ - naocc;
    int nbvir = ncmo_ - nbocc;

    ExcitationDeterminant zero_ex;

    //    boost::unordered_map<std::vector<short int>, size_t> alpha_strings_map;
    //    boost::unordered_map<std::vector<short int>, size_t> beta_strings_map;

    boost::unordered_map<std::pair<size_t, size_t>, size_t> good_determinants;
    std::pair<std::vector<double>, std::vector<double>> fock_diagonals;
    std::vector<double> fock_diagonal_alpha(ncmo_, 0.0);
    std::vector<double> fock_diagonal_beta(ncmo_, 0.0);
    size_t failed_attepts = 0;

    // Store energy,irrep,and excitation operator
    std::vector<boost::tuple<double, int, ExcitationDeterminant>> selected_determinants;
    std::vector<boost::tuple<double, int, ExcitationDeterminant>> selected_determinants_alpha;
    selected_determinants.push_back(
        boost::make_tuple(ref_energy_, wavefunction_symmetry_, zero_ex));
    selected_determinants_alpha.push_back(
        boost::make_tuple(ref_energy_, wavefunction_symmetry_, zero_ex));

    std::vector<boost::tuple<double, int, ExcitationDeterminant>> to_be_processed_elements;
    to_be_processed_elements.push_back(
        boost::make_tuple(ref_energy_, wavefunction_symmetry_, zero_ex));

    size_t num_dets_visited = 0;
    ForteTimer t_dets;
    bool iterate = true;
    int level = 0;
    while (iterate) {
        int maxn = static_cast<int>(to_be_processed_elements.size());
        std::vector<boost::tuple<double, int, ExcitationDeterminant>> new_elements;
        size_t num_dets_visited_ex = 0;
        size_t num_dets_accepted_ex = 0;
        for (int n = 0; n < maxn; ++n) {
            // Orbitals to consider when adding a single excitation to the element n
            double energy_ex = to_be_processed_elements[n].get<0>();
            int irrep_ex = to_be_processed_elements[n].get<1>();
            ExcitationDeterminant ex = to_be_processed_elements[n].get<2>();

            int minai = 0;
            int maxai = naocc;
            int minaa = naocc;
            int maxaa = naocc + navir;
            if (ex.naex() > 0) {
                maxai = ex.aann(ex.naex() - 1);
                minaa = ex.acre(ex.naex() - 1) + 1;
            }

            // Form the reference determinant in the string representation
            ExcitationDeterminant ref_pitzer(ex);
            ref_pitzer.to_pitzer(qt_to_pitzer_);
            StringDeterminant str_ex(reference_determinant_, ref_pitzer);

            ints_->make_alpha_fock_diagonal(str_ex.get_alfa_bits(), str_ex.get_beta_bits(),
                                            fock_diagonal_alpha);

            size_t num_dets_accepted_ex_el = 0;
            for (int i = maxai - 1; i >= minai; --i) {
                for (int a = minaa; a < maxaa; ++a) {
                    //                    double excitation_energy =
                    //                    fock_diagonal_alpha[qt_to_pitzer_[a]] -
                    //                    fock_diagonal_alpha[qt_to_pitzer_[i]] -
                    //                    ints_->diag_ce_rtei(qt_to_pitzer_[a],qt_to_pitzer_[i]);
                    double excitation_energy =
                        fock_diagonal_alpha[qt_to_pitzer_[a]] -
                        fock_diagonal_alpha[qt_to_pitzer_[i]] -
                        ints_->diag_aptei_aa(qt_to_pitzer_[a], qt_to_pitzer_[i]);
                    excitation_energy += +energy_ex - ref_energy_;
                    double absolute_energy = excitation_energy + ref_energy_;
                    double relative_energy = absolute_energy - min_energy_;

                    if (relative_energy < denominator_threshold_) {
                        ExcitationDeterminant ia_ex(ex);
                        ia_ex.add_alpha_ex(i, a);
                        num_dets_visited_ex++;
                        num_dets_visited++;

                        int irrep_ia_ex = irrep_ex ^ (mo_symmetry_qt_[i] ^ mo_symmetry_qt_[a]);
                        num_dets_accepted_ex++;
                        num_dets_accepted_ex_el++;
                        selected_determinants_alpha.push_back(
                            boost::make_tuple(absolute_energy, irrep_ia_ex, ia_ex));

                        new_elements.push_back(
                            boost::make_tuple(absolute_energy, irrep_ia_ex, ia_ex));
                        if (relative_energy < determinant_threshold_) {
                            if (irrep_ia_ex == wavefunction_symmetry_) {
                                selected_determinants.push_back(
                                    boost::make_tuple(absolute_energy, irrep_ia_ex, ia_ex));
                                min_energy_ = std::min(min_energy_, absolute_energy);
                                max_energy_ = std::max(max_energy_, absolute_energy);
                            }
                        }
                    }
                }
            }
            if (num_dets_accepted_ex_el == 0) {
                failed_attepts += 1;
            }
        }
        outfile->Printf("\n  %2d   %12ld   %12ld   %12ld", level + 1, num_dets_visited_ex,
                        num_dets_accepted_ex, failed_attepts);

        to_be_processed_elements = new_elements;
        level += 1;
        if (level > 10)
            iterate = false;
        if (new_elements.size() == 0)
            iterate = false;
    }

    to_be_processed_elements = selected_determinants_alpha;

    iterate = true;
    level = 0;
    while (iterate) {
        int maxn = static_cast<int>(to_be_processed_elements.size());
        std::vector<boost::tuple<double, int, ExcitationDeterminant>> new_elements;
        size_t num_dets_visited_ex = 0;
        size_t num_dets_accepted_ex = 0;
        for (int n = 0; n < maxn; ++n) {
            // Orbitals to consider when adding a single excitation to the element n
            double energy_ex = to_be_processed_elements[n].get<0>();
            int irrep_ex = to_be_processed_elements[n].get<1>();
            ExcitationDeterminant ex = to_be_processed_elements[n].get<2>();

            int minbi = 0;
            int maxbi = nbocc;
            int minba = nbocc;
            int maxba = nbocc + nbvir;
            if (ex.nbex() > 0) {
                maxbi = ex.bann(ex.nbex() - 1);
                minba = ex.bcre(ex.nbex() - 1) + 1;
            }

            // Form the reference determinant in the string representation
            ExcitationDeterminant ref_pitzer(ex);
            ref_pitzer.to_pitzer(qt_to_pitzer_);
            StringDeterminant str_ex(reference_determinant_, ref_pitzer);

            ints_->make_beta_fock_diagonal(str_ex.get_alfa_bits(), str_ex.get_beta_bits(),
                                           fock_diagonal_beta);

            size_t num_dets_accepted_ex_el = 0;
            for (int i = maxbi - 1; i >= minbi; --i) {
                for (int a = minba; a < maxba; ++a) {
                    double excitation_energy =
                        fock_diagonal_beta[qt_to_pitzer_[a]] -
                        fock_diagonal_beta[qt_to_pitzer_[i]] -
                        ints_->diag_aptei_bb(qt_to_pitzer_[a], qt_to_pitzer_[i]);
                    excitation_energy += +energy_ex - ref_energy_;
                    double absolute_energy = excitation_energy + ref_energy_;
                    double relative_energy = absolute_energy - min_energy_;

                    if (relative_energy < denominator_threshold_) {
                        ExcitationDeterminant ia_ex(ex);
                        ia_ex.add_beta_ex(i, a);
                        num_dets_visited_ex++;
                        num_dets_visited++;

                        int irrep_ia_ex = irrep_ex ^ (mo_symmetry_qt_[i] ^ mo_symmetry_qt_[a]);

                        new_elements.push_back(
                            boost::make_tuple(absolute_energy, irrep_ia_ex, ia_ex));
                        if ((relative_energy < determinant_threshold_) and
                            (irrep_ia_ex == wavefunction_symmetry_)) {
                            num_dets_accepted_ex++;
                            num_dets_accepted_ex_el++;
                            selected_determinants.push_back(
                                boost::make_tuple(absolute_energy, irrep_ia_ex, ia_ex));
                            min_energy_ = std::min(min_energy_, absolute_energy);
                            max_energy_ = std::max(max_energy_, absolute_energy);
                        }
                    }
                }
            }
            if (num_dets_accepted_ex_el == 0) {
                failed_attepts += 1;
            }
        }
        outfile->Printf("\n  %2d   %12ld   %12ld   %12ld", level + 1, num_dets_visited_ex,
                        num_dets_accepted_ex, failed_attepts);

        to_be_processed_elements = new_elements;
        level += 1;
        if (level > 10)
            iterate = false;
        if (new_elements.size() == 0)
            iterate = false;
    }

    double time_dets = t_dets.elapsed();

    // Convert to a string representation

    //    std::vector<bool> bits(I,I + nmo_);
    //    int h = string_symmetry(I);

    //    // set the alpha/beta strings and compute the energy of this determinant
    //    // Copy the string and translate it to Pitzer ordering
    //    for (int p = 0; p < nmo_; ++p) Im[p] = I[p];
    //    det.set_bits(Ia,Ib);
    //    double exc_energy = det.excitation_energy(reference_determinant_);

    //    int exc_class = excitation_class(nex,h);

    //    if(mp_screening_){
    //        vec_str[exc_class].push_back(boost::make_tuple(eo + ev,exc_energy,bits));

    outfile->Printf("\n\n  The determinants visited fall in the range [%f,%f]", min_energy_,
                    max_energy_);
    //    outfile->Printf("\n\n  Number of full ci determinants    = %llu",num_total_dets);
    outfile->Printf("\n\n  Number of determinants visited    = %ld (%e)", num_dets_visited, 0.0);
    outfile->Printf("\n  Number of determinants accepted   = %ld (%e)",
                    selected_determinants.size(), 0.0);
    outfile->Printf("\n  Time spent on generating dets     = %f s", time_dets);
    outfile->Printf("\n  Precompute algorithm time elapsed = %f s", t.elapsed());
}
}
} // EndNamespaces
