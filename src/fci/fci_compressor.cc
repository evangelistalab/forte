/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "boost/format.hpp"

#include "base_classes/reference.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"

#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"
#include "helpers/iterative_solvers.h"

#include "fci_solver.h"
#include "fci_vector.h"
#include "fci_compressor.h"
#include "string_lists.h"
#include "helpers/helpers.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "psi4/psi4-dec.h"

using namespace psi;

namespace forte {

class MOSpaceInfo;

FCICompressor::FCICompressor(std::shared_ptr<FCIVector> C_fci, FCIVector HC_fci,
                             std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                             double nuclear_repulsion_energy,
                             double fci_energy){
    C_fci_ = std::make_shared<FCIVector>( *C_fci);
    HC_fci_ = std::make_shared<FCIVector>( HC_fci);
    fci_ints_ = std::make_shared<ActiveSpaceIntegrals>( *as_ints);
    fci_energy_ = fci_energy;
    nuclear_repulsion_energy_ = nuclear_repulsion_energy;
    std::cout<<"Igethere3"<<std::endl;
}

// FCICompressor::FCICompressor(std::shared_ptr<FCIVector> C_fci, FCIVector HC_fci,
//                              std::shared_ptr<ActiveSpaceIntegrals> as_ints,
//                              double nuclear_repulsion_energy,
//                              double fci_energy)
//                              : C_fci_(std::make_shared<FCIVector>( *C_fci)),
//                              HC_fci_(std::make_shared<FCIVector>( HC_fci)),
//                              fci_ints_(as_ints)
//                               {
//     std::cout<<"Igethere"<<std::endl;
//     // C_fci_ = std::make_shared<FCIVector>( *C_fci);
//     // HC_fci_ = std::make_shared<FCIVector>( HC_fci);
//     // fci_ints_ = as_ints;
//     fci_energy_ = fci_energy;
//     nuclear_repulsion_energy_ = nuclear_repulsion_energy;
// }

void FCICompressor::set_options(std::shared_ptr<ForteOptions> options){
    set_do_rr(options->get_bool("DO_RANK_REDUCTION"));
    set_do_st(options->get_bool("DO_STRING_TRIM"));
    set_do_ds(options->get_bool("DO_DET_SCREEN"));
    set_do_mp(options->get_bool("DO_MPS_TRANSFORM"));
    set_tau(options->get_double("TAU_COMPRESSION"));
    set_delta_tao(options->get_double("DELTA_TAU_COMPRESSION"));
    set_num_compressions(options->get_int("NUM_COMPRESSIONS"));
}

void FCICompressor::compress_and_analyze() {
    for(size_t k = 0; k < num_compressions_; k++){

        if (do_rr_){ rank_reduce(); }
        // if (do_st_){ string_trim(); }
        // if (do_ds_){ det_screen(); }
        // if (do_mp_){ mps_ify(); }

        tau_ += delta_tau_;
    }
}

// double FCICompressor::two_rcm_diff(std::shared_ptr<FCIVector> C_compressed) {}
//
// double FCICompressor::C_diff(std::shared_ptr<FCIVector> C_compressed) {}
//
void FCICompressor::rank_reduce() {
    FCIVector RR(* C_fci_);
    FCIVector HC_RR(* HC_fci_);
    std::vector<std::shared_ptr<psi::Matrix>> C_RR = RR.coefficients_blocks();

    int nirrep = C_RR.size();
    int total_rank = 0;
    int total_red_rank = 0;
    int N_par = 0;

    double norm_C = 0.0;
    double sigma_norm = 0.0;
    double norm_cut = 1.0 - tau_;
    double sig_sum = 0.0;
    double C_error = 0.0;

    std::vector<int> reduced_rank_irrep(nirrep, 0);
    std::vector<int> full_rank_irrep(nirrep, 0);
    std::vector<double> c_diff_irrep(nirrep, 0.0);
    std::vector<std::pair<double, int>> sorted_sigma;
    std::vector<std::shared_ptr<psi::Matrix>> C_diff(nirrep);

    // copy C_fci as for a difference matrix
    for (int h = 0; h < nirrep; h++) { C_diff[h] = C_RR[h]->clone(); }

    // norm of the wave function
    for (auto C_h : C_RR) { norm_C += C_h->sum_of_squares(); }

    // for each irrep h add find singualr values
    for (int h = 0; h < nirrep; h++) {
        int rowdim = C_RR[h]->rowdim();
        int coldim = C_RR[h]->coldim();
        full_rank_irrep[h] = std::min(coldim,rowdim);

        auto U = std::make_shared<Matrix>("U", coldim, coldim);
        auto s = std::make_shared<Vector>("s", std::min(coldim, rowdim));
        auto V = std::make_shared<Matrix>("V", rowdim, rowdim);

        C_RR[h]->svd(U, s, V);

        for (int i = 0; i < std::min(coldim, rowdim); i++) {
            sorted_sigma.push_back(std::make_pair(s->get(i), h));
        }
    }

    // sort the singular values from smallest to largest
    std::sort(sorted_sigma.rbegin(), sorted_sigma.rend());

    // calculate the norm of the sigma vector
    for (auto sigma_h : sorted_sigma) {
         //outfile->Printf("\n   %20.12f      %d", sigma_h.first, sigma_h.second);
        sigma_norm += std::pow(sigma_h.first, 2.0);
    }

    // determine reduced rank in each irrep
    for (auto sigma_h : sorted_sigma) {
        reduced_rank_irrep[sigma_h.second] += 1;
        sig_sum += std::pow(sigma_h.first, 2.0);
        if (sig_sum > norm_cut * sigma_norm) { break; }
    }

    outfile->Printf("\n ||C|| = %20.12f \n", std::sqrt(norm_C));
    outfile->Printf("\n irrep red  full      ||C_h - C_RR_h||\n");

    for (int h = 0; h < nirrep; h++) {
        int rowdim = C_RR[h]->rowdim();
        int coldim = C_RR[h]->coldim();

        auto U = std::make_shared<Matrix>("U", coldim, coldim);
        auto s = std::make_shared<Vector>("s", std::min(coldim, rowdim));
        auto V = std::make_shared<Matrix>("V", rowdim, rowdim);

        C_RR[h]->svd(U, s, V);

        // Copy diagonal of s to a matrix
        auto S = std::make_shared<Matrix>("sig_mat", coldim, rowdim);
        for (int i = 0; i < reduced_rank_irrep[h]; i++) { S->set(i, i, s->get(i)); }

        // auto C_red_rank_h = Matrix::triplet(U, S, V, false, false, false);
        // C_red_rank.push_back(C_red_rank_h);

        // set C_RR[h] to the pointer to the reduced rank matrix
        C_RR[h] = Matrix::triplet(U, S, V, false, false, false);

        // auto C_diff = C[h]->clone();
        C_diff[h]->subtract(C_RR[h]);
        c_diff_irrep[h] = std::sqrt(C_diff[h]->sum_of_squares());

        N_par += reduced_rank_irrep[h] * coldim;
        N_par += reduced_rank_irrep[h] * rowdim;

        outfile->Printf(" %1d %6d %6d %20.12f \n", h, reduced_rank_irrep[h], std::min(coldim, rowdim), c_diff_irrep[h]);
    }

    // re-normalize C_RR
    // double norm_C_red_rank = 0.0;
    // for (auto C_red_rank_h : C_red_rank) {
    //     norm_C_red_rank += C_red_rank_h->sum_of_squares();
    // }
    // norm_C_red_rank = std::sqrt(norm_C_red_rank);
    // for (auto C_red_rank_h : C_red_rank) {
    //     C_red_rank_h->scale(1. / norm_C_red_rank);
    // }

    // re-normalize C_RR
    RR.normalize();
    std::cout<<"Here1"<<std::endl;
    // double norm_C_red_rank = 0.0;
    // for (auto C_red_rank_h : C_red_rank) {
    //     norm_C_red_rank += C_red_rank_h->sum_of_squares();
    //     //C_red_rank_h->print();
    // }
    // norm_C_red_rank = std::sqrt(norm_C_red_rank);
    // outfile->Printf("\n ||C|| = %20.12f", std::sqrt(norm_C_red_rank));

    //C_->set_coefficient_blocks(C_red_rank);
    // HC = H C

    // I AM HERE //
    // FCIVector HC_RR();
    //C_->Hamiltonian(HC, fci_ints, twoSubstituitionVVOO);

    RR.Hamiltonian(HC_RR, fci_ints_);

    // E = C^T HC
    double E_RR = HC_RR.dot(RR) + nuclear_repulsion_energy_;

    outfile->Printf("\n\n Here2");

    //double Npar_db = Npar;

    outfile->Printf("\n////////////////// Full SVD /////////////////\n");
    outfile->Printf("\n Tau               = %20.12f", tau_);
    outfile->Printf("\n E_fci             = %20.12f", fci_energy_);
    outfile->Printf("\n E_red_rank        = %20.12f", E_RR);
    outfile->Printf("\n Delta(E_red_rank) = %20.12f", E_RR-fci_energy_);
    outfile->Printf("\n Npar red          =     %6d", N_par);

}

// void FCICompressor::string_trim() {}
//
// void FCICompressor::det_screen() {}
//
// void FCICompressor::mps_ify() {}

} // namespace forte
