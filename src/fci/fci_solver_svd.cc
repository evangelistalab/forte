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

void FCISolver::fci_svd_tiles(FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy, int dim, double OMEGA)
{
  double nuclear_repulsion_energy =
      Process::environment.molecule()->nuclear_repulsion_energy({0, 0, 0});

    std::vector<SharedMatrix> C = C_->coefficients_blocks();
    std::vector<SharedMatrix> C_tiled_rr;

    int nirrep = C.size();
    int total_rank = 0;
    int total_red_rank = 0;

    ///// PRE MATRIX TILER BEGIN /////

    std::vector<std::tuple<double, int, int, int> > sorted_sigma;
    int size_of_ssv = 0;
    std::vector<int> b_r(1);
    std::vector<int> e_r(1);
    std::vector<int> b_c(1);
    std::vector<int> e_c(1);

    std::vector<std::vector<std::vector<int> > > rank_tile_inirrep;
    rank_tile_inirrep.resize(nirrep);

    // now need to put singular values into said vector ...

    for (int h=0; h<nirrep; h++) {
        // loop over irreps
        int ncol = C[h]->rowdim();
        int nrow = C[h]->coldim();
        //outfile->Printf("\nncol in h: %6d\n", ncol);
        //outfile->Printf("\nnrow in h: %6d\n", nrow);
        int nt_cols = ncol/dim;
        int nt_rows = nrow/dim;
        //outfile->Printf("\nnt_col in h: %6d\n", nt_cols);
        //outfile->Printf("\nnt_row in h: %6d\n", nt_rows);

        int last_col_dim = ncol%dim;
        int last_row_dim = nrow%dim;
        //outfile->Printf("\nlast_col_dim in h: %6d\n", last_col_dim);
        //outfile->Printf("\nlast_row_dim in h: %6d\n", last_row_dim);

        int n_sing_vals_h = dim*nt_cols*nt_rows
                             + last_col_dim*nt_rows
                             + last_row_dim*nt_cols
                             + std::min(last_col_dim, last_row_dim);

        size_of_ssv += n_sing_vals_h;
        rank_tile_inirrep[h].resize(nt_rows+1);


        outfile->Printf("\n-------------------- NEXT IRREP --------------------\n");
        C[h]->print();

        for(int i=0; i<nt_rows+1; i++){
          // allocate memory for sorting vector
          rank_tile_inirrep[h][i].resize(nt_cols+1);
          for(int j=0; j<nt_cols+1; j++){
            // make dimension objects
            if(j == nt_cols && i == nt_rows){
              // make dimension objects for case of very last tile
              b_r[0] = i*dim;
              e_r[0] = i*dim + last_row_dim;
              b_c[0] = j*dim;
              e_c[0] = j*dim + last_col_dim;

              Dimension begin_row(b_r);
              Dimension end_row(e_r);
              Dimension begin_col(b_c);
              Dimension end_col(e_c);

              // make slice objects
              Slice row_slice(begin_row, end_row);
              Slice col_slice(begin_col, end_col);

              // get matrix block
              auto M = C[h]->get_block(row_slice, col_slice);

              int nrow_M = M->rowdim();
              //std::cout << M->rowdim() << std::endl;
              int ncol_M = M->coldim();

              auto u = std::make_shared<Matrix>("u", last_row_dim, last_col_dim);
              auto s = std::make_shared<Vector>("s", std::min(last_row_dim, last_col_dim));
              auto v = std::make_shared<Matrix>("v", last_col_dim, last_col_dim);

              M->svd(u, s, v);
              s->print();

              for (int k = 0; k < std::min(last_row_dim, last_col_dim); k++) {
                  sorted_sigma.push_back(std::make_tuple(s->get(k), h, i, j));
              }


            } else if(j == nt_cols){
              // make dimension objects for case of last tile colum
              b_r[0] = i*dim;
              e_r[0] = i*dim + dim;
              b_c[0] = j*dim;
              e_c[0] = j*dim + last_col_dim;

              Dimension begin_row(b_r);
              Dimension end_row(e_r);
              Dimension begin_col(b_c);
              Dimension end_col(e_c);

              // make slice objects
              Slice row_slice(begin_row, end_row);
              Slice col_slice(begin_col, end_col);

              // get matrix block
              auto M = C[h]->get_block(row_slice, col_slice);

              int nrow_M = M->rowdim();
              //std::cout << M->rowdim() << std::endl;
              int ncol_M = M->coldim();

              auto u = std::make_shared<Matrix>("u", dim, last_col_dim);
              auto s = std::make_shared<Vector>("s", std::min(dim, last_col_dim));
              auto v = std::make_shared<Matrix>("v", last_col_dim, last_col_dim);

              M->svd(u, s, v);
              s->print();

              for (int k = 0; k < std::min(dim, last_col_dim); k++) {
                  sorted_sigma.push_back(std::make_tuple(s->get(k), h, i, j));
              }

            } else if(i == nt_rows){

              // make dimension objects for case of last tile row
              b_r[0] = i*dim;
              e_r[0] = i*dim + last_row_dim;
              b_c[0] = j*dim;
              e_c[0] = j*dim + dim;

              Dimension begin_row(b_r);
              Dimension end_row(e_r);
              Dimension begin_col(b_c);
              Dimension end_col(e_c);

              // make slice objects
              Slice row_slice(begin_row, end_row);
              Slice col_slice(begin_col, end_col);

              // get matrix block
              auto M = C[h]->get_block(row_slice, col_slice);

              int nrow_M = M->rowdim();
              //std::cout << M->rowdim() << std::endl;
              int ncol_M = M->coldim();

              auto u = std::make_shared<Matrix>("u", last_row_dim, dim);
              auto s = std::make_shared<Vector>("s", std::min(last_row_dim, dim));
              auto v = std::make_shared<Matrix>("v", dim, dim);

              M->svd(u, s, v);
              s->print();

              for (int k = 0; k < std::min(last_row_dim, dim); k++) {
                  sorted_sigma.push_back(std::make_tuple(s->get(k), h, i, j));
              }


            } else {

            // make dimension objects

            b_r[0] = i*dim;
            e_r[0] = i*dim + dim;
            b_c[0] = j*dim;
            e_c[0] = j*dim + dim;

            Dimension begin_row(b_r);
            Dimension end_row(e_r);
            Dimension begin_col(b_c);
            Dimension end_col(e_c);

            // make slice objects
            Slice row_slice(begin_row, end_row);
            Slice col_slice(begin_col, end_col);

            // get matrix block
            auto M = C[h]->get_block(row_slice, col_slice);

            int nrow_M = M->rowdim();
            //std::cout << M->rowdim() << std::endl;
            int ncol_M = M->coldim();

            auto u = std::make_shared<Matrix>("u", dim, dim);
            auto s = std::make_shared<Vector>("s", dim);
            auto v = std::make_shared<Matrix>("v", dim, dim);

            M->svd(u, s, v);
            s->print();

            for (int k = 0; k < dim ; k++) {
                sorted_sigma.push_back(std::make_tuple(s->get(k), h, i, j));
            }
          }
          //outfile->Printf("\nh dim: %6d  i dim: %6d  j dim: %6d  ",rank_tile_inirrep.size(), rank_tile_inirrep[h].size(), rank_tile_inirrep[h][i].size());
          } // end j
        } // end i
    } // end h

    /////////////////////////////////// check status ////////////////////////////////////

    std::sort(sorted_sigma.rbegin(), sorted_sigma.rend());
    double sigma_norm = 0.0;

    outfile->Printf("\n    singular value      irrep");
    outfile->Printf("\n--------------------------------");

    for (auto sigma_h : sorted_sigma) {
        outfile->Printf("\n   %20.12f      %d", std::get<0>(sigma_h), std::get<1>(sigma_h));
        sigma_norm += std::pow(std::get<0>(sigma_h), 2.0);
    }

    outfile->Printf("\n");
    outfile->Printf("\n size of sorted sigma vec: %6d ", size_of_ssv);
    outfile->Printf("\n size of count: %6d ", sorted_sigma.size());
    outfile->Printf("\n Norm of sigma: %20.12f ", sigma_norm);

    // find cutoff OMEGA
    double norm_cut = 1.0 - OMEGA;
    double sig_sum = 0;
    double omega_norm;
    //std::vector<int> rank_irrep(nirrep, 0);


    //std::tuple<int, int, int, int> rank_tile_inirrep;


    for (auto sigma_h : sorted_sigma) {
        rank_tile_inirrep[std::get<1>(sigma_h)][std::get<2>(sigma_h)][std::get<3>(sigma_h)] += 1;
        sig_sum += std::pow(std::get<0>(sigma_h), 2.0);
        if (sig_sum > norm_cut * sigma_norm) {
            break;
        }
    }

    for(int h = 0; h<8; h++){
      for(int i = 0; i<2; i++){
        for(int j = 0; j<2; j++){
          outfile->Printf("\nfor irrep: %6d in tile: %6d,%6d the red rank is: %6d", h,i,j,rank_tile_inirrep[h][i][j]);
        }
      }
    }

    ///// PRE MATRIX TILER END /////

}

void FCISolver::fci_svd(FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy, double TAU)
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


    //outfile->Printf("\n    singular value      irrep");
    //outfile->Printf("\n--------------------------------");

    for (auto sigma_h : sorted_sigma) {
        //outfile->Printf("\n   %20.12f      %d", sigma_h.first, sigma_h.second);
        sigma_norm += std::pow(sigma_h.first, 2.0);
    }

    outfile->Printf("\n");
    outfile->Printf("\n Norm of sigma: %20.12f \n", sigma_norm);

    double norm_cut = 1.0 - TAU;
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

    outfile->Printf(" Rank for each irrep:");
    for (int r : rank_irrep) {
        outfile->Printf(" %d", r);
    }
    outfile->Printf("\n");

    double tao = options_.get_double("FCI_SVD_TAU");

    outfile->Printf("\n ||C|| = %20.12f", std::sqrt(norm_C));

    outfile->Printf("\n irrep red  full        ||C - C_rr||\n");

    int N_par = 0;

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

        N_par += rank_ef*std::min(ncol, nrow);

        outfile->Printf(" %1d %6d %6d %20.12f \n", h, rank_ef, std::min(ncol, nrow), norm);
    }
    outfile->Printf("   %6d %6d\n", total_red_rank, total_rank);

    int N_sto = N_par;

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

    outfile->Printf("\n Tau               = %20.12f", TAU);
    outfile->Printf("\n E_fci             = %20.12f", fci_energy);
    outfile->Printf("\n E_red_rank        = %20.12f", E_red_rank);
    outfile->Printf("\n Delta(E_red_rank) = %20.12f", E_red_rank-fci_energy);
    outfile->Printf("\n Npar              =     %6d", N_par);
    outfile->Printf("\n Nsto              =     %6d", N_sto);

}

} // namespace forte
} // namespace psi
