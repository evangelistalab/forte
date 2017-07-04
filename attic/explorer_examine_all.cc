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

double compute_mp_energy(bool* begin, bool* end, const std::vector<double>& epsilon) {
    double sum = 0.0;
    int n = 0;
    for (bool* it = begin; it != end; ++it) {
        if (*it)
            sum += epsilon[n];
        ++n;
    }
    return sum;
}

/**
 * Examine all the Slater determinant in the FCI space
 */
void LambdaCI::examine_all(psi::Options& options) {
    outfile->Printf("\n\n  Exploring the space of Slater determinants\n");
    StringDeterminant det(reference_determinant_);

    // No explorer will succeed without a cartographer
    Cartographer cg(options, min_energy_, min_energy_ + determinant_threshold_);

    int nfrzc = ints_->frzcpi().sum();
    int nfrzv = ints_->frzcpi().sum();
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

    ForteTimer t;
    double time_string = 0.0;
    double time_dets = 0.0;
    long num_dets_visited = 0;
    long num_dets_accepted = 0;
    unsigned long long num_total_dets = 0;
    unsigned long num_permutations = 0;

    // Allocate an array of bits for fast manipulation
    bool* Ia = new bool[2 * ncmo_];
    bool* Ib = &Ia[ncmo_];

    // Create the alpha and beta strings |0000011111|
    for (int p = 0; p < ncmo_; ++p)
        Ia[p] = Ib[p] = false;
    for (int i = 0; i < naocc + nfrzc; ++i)
        Ia[ncmo_ - i - 1] = true;
    for (int i = 0; i < nbocc + nfrzc; ++i)
        Ib[ncmo_ - i - 1] = true;

    std::vector<double> epsilon;
    for (int p = 0; p < ncmo_; ++p) {
        epsilon.push_back(ref_eps_a_->get(p));
    }

    // Generate the alpha strings
    bool* Ia_ref = det.get_alfa_bits();
    double a_mp_energy_ref = compute_mp_energy(Ia_ref, Ia_ref + ncmo_, epsilon);
    std::vector<std::vector<bool>> astr_vec;
    std::vector<double> ea_vec;
    do {
        double ea = compute_mp_energy(Ia, Ia + ncmo_, epsilon) - a_mp_energy_ref;
        if (ea < denominator_threshold_) {
            std::vector<bool> bits(Ia, Ia + ncmo_);
            astr_vec.push_back(bits);
            ea_vec.push_back(ea);
        }
    } while (std::next_permutation(Ia, Ia + ncmo_));

    outfile->Printf("\n Number of alpha strings accepted: %d", int(ea_vec.size()));

    // Generate the beta strings
    bool* Ib_ref = det.get_beta_bits();
    double b_mp_energy_ref = compute_mp_energy(Ib_ref, Ib_ref + ncmo_, epsilon);
    std::vector<std::vector<bool>> bstr_vec;
    std::vector<double> eb_vec;
    do {
        double eb = compute_mp_energy(Ib, Ib + ncmo_, epsilon) - b_mp_energy_ref;
        if (eb < denominator_threshold_) {
            std::vector<bool> bits(Ib, Ib + ncmo_);
            bstr_vec.push_back(bits);
            eb_vec.push_back(eb);
        }
    } while (std::next_permutation(Ib, Ib + ncmo_));

    size_t nastr = astr_vec.size();
    size_t nbstr = bstr_vec.size();
    num_total_dets = nastr * nbstr;
    for (size_t nsa = 0; nsa < nastr; ++nsa) {
        double ea = ea_vec[nsa];
        std::vector<bool>& str_sa = astr_vec[nsa];
        for (int p = 0; p < ncmo_; ++p)
            Ia[p] = str_sa[p];
        int ha = string_symmetry(Ia);
        for (size_t nsb = 0; nsb < nbstr; ++nsb) {
            double eb = ea_vec[nsb];
            if (ea + eb < denominator_threshold_) {
                std::vector<bool>& str_sb = bstr_vec[nsb];
                for (int p = 0; p < ncmo_; ++p)
                    Ib[p] = str_sb[p];
                int hb = string_symmetry(Ib);
                if ((ha ^ hb) == wavefunction_symmetry_) {
                    det.set_bits(Ia, Ib);
                    double det_energy = det.energy() + nuclear_repulsion_energy_;
                    // check to see if the energy is below a given threshold
                    if (det_energy < min_energy_ + determinant_threshold_) {
                        cg.accumulate_data(ncmo_, str_sa, str_sb, det_energy, 0, 0, 0, 0);
                        if (det_energy < min_energy_) {
                            reference_determinant_ = det;
                            min_energy_ = det_energy;
                        }
                        num_dets_accepted++;
                    }
                    num_dets_visited++;
                }
            }
        }
    }

    delete[] Ia;

    outfile->Printf("\n\n  The new reference determinant is:");
    reference_determinant_.print();
    outfile->Printf("\n  and its energy: %.12f Eh", min_energy_);

    outfile->Printf("\n\n  Number of full ci determinants    = %llu", num_total_dets);
    outfile->Printf("\n\n  Number of determinants visited    = %ld (%e)", num_dets_visited,
                    double(num_dets_visited) / double(num_total_dets));
    outfile->Printf("\n  Number of determinants accepted   = %ld (%e)", num_dets_accepted,
                    double(num_dets_accepted) / double(num_total_dets));
}
}
} // EndNamespaces
