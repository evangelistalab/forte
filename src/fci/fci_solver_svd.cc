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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "../mini-boost/boost/format.hpp"

#include "../sparse_ci/determinant.h"
#include "../iterative_solvers.h"

#include "fci_solver.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "psi4/psi4-dec.h"

using namespace psi;

namespace psi {
namespace forte {

class MOSpaceInfo;

void FCISolver::fci_svd(FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy)
{
  double nuclear_repulsion_energy =
      Process::environment.molecule()->nuclear_repulsion_energy({0, 0, 0});

    std::vector<SharedMatrix> C = C_->coefficients_blocks();
    std::vector<SharedMatrix> C_red_rank;

    double norm_C = 0.0;
    for (auto C_h : C) {
        norm_C += C_h->sum_of_squares();
    }

    int nirrep = C.size();
    int total_rank = 0;
    int total_red_rank = 0;

    //// norm based cuttoff scheme BEGIN

    /// would like a way to generate the tao value based on the norm %
    // Best way to initalize a vector with correct size?

    //    int all_sig_dim = 0;

    //    for (int h = 0; h < nirrep; h++) {
    //      all_sig_dim += std::min(C[h]->rowdim(), C[h]->coldim());
    //    }

    //    std::vector<double> all_sig(all_sig_dim);
    //    int sig_counter = 0;

    //    std::vector<double> all_sig;
    std::vector<std::pair<double, int>> sorted_sigma; // [(sigma_i, irrep)]

    for (int h = 0; h < nirrep; h++) {
        int nrow = C[h]->rowdim();
        int ncol = C[h]->coldim();

        auto u_p = std::make_shared<Matrix>("u_p", nrow, nrow);
        auto s_p = std::make_shared<Vector>("s_p", std::min(ncol, nrow));
        auto v_p = std::make_shared<Matrix>("v_p", ncol, ncol);

        C[h]->svd(u_p, s_p, v_p);

        for (int i = 0; i < std::min(ncol, nrow); i++) {
            //            all_sig.push_back(s_p->get(i));
            sorted_sigma.push_back(std::make_pair(s_p->get(i), h));
            //        all_sig[sig_counter] = s_p->get(i);
            //        sig_counter++;
        }
    }

    std::sort(sorted_sigma.rbegin(), sorted_sigma.rend());
    double sigma_norm = 0.0;
    int k = 0;
    for (auto sigma_h : sorted_sigma) {
        outfile->Printf("\n element [%6d] of sorted_sigma is: %20.12f irrep : %d", k, sigma_h.first,
                        sigma_h.second);
        sigma_norm += std::pow(sigma_h.first, 2.0);
    }
    outfile->Printf("\n ");
    outfile->Printf("Norm of sigma: %20.12f \n", sigma_norm);

    double norm_cut = 1.0 - options_.get_double("FCI_SVD_TAU");
    double sig_sum = 0;
    double tao_norm;
    std::vector<int> rank_irrep(nirrep, 0);
    for (auto sigma_h : sorted_sigma) {
        rank_irrep[sigma_h.second] += 1;
        sig_sum += std::pow(sigma_h.first, 2.0);
        if (sig_sum > norm_cut * sigma_norm) {
            break;
        }
    }

    outfile->Printf("Rank for each irrep:");
    for (int r : rank_irrep) {
        outfile->Printf(" %d", r);
    }
    outfile->Printf("\n");

    double tao = options_.get_double("FCI_SVD_TAU");

    outfile->Printf("\n ||C|| = %20.12f", std::sqrt(norm_C));

    outfile->Printf("\n irrep red  full        ||C - C_rr||\n");

    for (int h = 0; h < nirrep; h++) {
        int nrow = C[h]->rowdim();
        int ncol = C[h]->coldim();

        auto u_p = std::make_shared<Matrix>("u_p", nrow, nrow);
        auto s_p = std::make_shared<Vector>("s_p", std::min(ncol, nrow));
        auto v_p = std::make_shared<Matrix>("v_p", ncol, ncol);

        C[h]->svd(u_p, s_p, v_p);

        int rank_ef = std::min(ncol, nrow);
        if (options_.get_str("FCI_SVD_TYPE") == "THRESHOLD") {
            for (int i = 0; i < std::min(ncol, nrow); i++) {
                if (s_p->get(i) < tao) {
                    rank_ef = i;
                    break;
                }
            }
        } else {
            rank_ef = rank_irrep[h];
        }

        total_rank += std::min(ncol, nrow);
        total_red_rank += rank_ef;

        // Copy diagonal of s_p to a matrix
        auto sig_mat = std::make_shared<Matrix>("sig_mat", nrow, ncol);
        for (int i = 0; i < rank_ef; i++) {
            sig_mat->set(i, i, s_p->get(i));
        }

        auto C_red_rank_h = Matrix::triplet(u_p, sig_mat, v_p, false, false, false);
        C_red_rank.push_back(C_red_rank_h);

        auto C_diff = C[h]->clone();
        C_diff->subtract(C_red_rank_h);
        double norm = std::sqrt(C_diff->sum_of_squares());

        outfile->Printf(" %1d %6d %6d %20.12f \n", h, rank_ef, std::min(ncol, nrow), norm);
    }
    outfile->Printf("   %6d %6d\n", total_red_rank, total_rank);

    double norm_C_red_rank = 0.0;
    for (auto C_red_rank_h : C_red_rank) {
        norm_C_red_rank += C_red_rank_h->sum_of_squares();
    }
    norm_C_red_rank = std::sqrt(norm_C_red_rank);
    for (auto C_red_rank_h : C_red_rank) {
        C_red_rank_h->scale(1. / norm_C_red_rank);
    }

    {
        double norm_C_red_rank = 0.0;
        for (auto C_red_rank_h : C_red_rank) {
            norm_C_red_rank += C_red_rank_h->sum_of_squares();
        }
        norm_C_red_rank = std::sqrt(norm_C_red_rank);
        outfile->Printf("\n ||C|| = %20.12f", std::sqrt(norm_C_red_rank));
    }

    // Compute the energy

    C_->set_coefficient_blocks(C_red_rank);
    // HC = H C
    C_->Hamiltonian(HC, fci_ints, twoSubstituitionVVOO);
    // E = C^T HC
    double E_red_rank = HC.dot(C_) + nuclear_repulsion_energy;

    outfile->Printf("\n E_fci             = %20.12f", fci_energy);
    outfile->Printf("\n E_red_rank        = %20.12f", E_red_rank);
    outfile->Printf("\n Delta(E_red_rank) = %20.12f", E_red_rank-fci_energy);

}

} // namespace forte
} // namespace psi
