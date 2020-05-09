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
#include "fci_vector.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "psi4/psi4-dec.h"

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace psi;

namespace psi {
namespace forte {

class MOSpaceInfo;

void FCISolver::py_mat_print(SharedMatrix C_h, const std::string& input)
{
  std::ofstream myfile1;
  myfile1.open (input);
  for (int i=0; i<C_h->coldim(); i++){
    myfile1 << "\n";
    for (int j=0; j<C_h->rowdim(); j++){
        myfile1 << C_h->get(i,j) << " ";
    }
  }
  myfile1.close();
}

bool FCISolver::pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem)
{
    return firstElem.first > secondElem.first;
}

bool FCISolver::pairTupleCompare(const std::tuple<double, int, int>& firstElem, const std::tuple<double, int, int>& secondElem)
{
    return std::get<0>(firstElem) > std::get<0>(secondElem);
}

void FCISolver::basis_cluster(std::vector<SharedMatrix>& C, std::vector<std::pair<double, int> >& st_vec)
{
  // fill vectors of pairs for ij indicies using Calph^2, the sorted order will be the i'j' indicies
  for(auto C_h: C){
    // for alpha strings
    for(int i=0; i<C_h->coldim(); i++){
      st_vec[i].second = i;
      for(int j=0; j<C_h->rowdim(); j++){
        st_vec[i].first += std::pow(C_h->get(i,j), 2);
      }
    }
  }

  std::sort(st_vec.begin(), st_vec.end(), pairCompare);

  // allocate a new matrix C'
  std::vector<SharedMatrix> Cprime(nirrep_);
  for(int h=0; h<nirrep_; h++){
    // would be better if I could just allocate the 'empty' Sh.Mat.
    Cprime[h] = C[h]->clone();
  }

  // edit C to use new clustered basis order
  for(int h=0; h<nirrep_; h++){
    for(int i=0; i<C[h]->coldim(); i++){
      for(int j=0; j<C[h]->rowdim(); j++){
        Cprime[h]->set(i, j, C[h]->get(st_vec[i].second, st_vec[j].second) );
      }
    }
  }

  // py_mat_print(Cprime[0], "Cprime.mat");

  //set C as Cprime...
  for(int h=0; h<nirrep_; h++){
    C[h]->copy(Cprime[h]);
  }
}

void FCISolver::rev_basis_cluster(std::vector<SharedMatrix>& C, std::vector<std::pair<double, int> > st_vec)
{
  // allocate a new matrix C'
  std::vector<SharedMatrix> Cprime(nirrep_);
  for(int h=0; h<nirrep_; h++){
    // would be better if I could just allocate the 'empty' Sh.Mat.
    Cprime[h] = C[h]->clone();
  }

  // reset C to use old un-clustered basis order
  for(int h=0; h<nirrep_; h++){
    for(int i=0; i<C[h]->coldim(); i++){
      for(int j=0; j<C[h]->rowdim(); j++){
        Cprime[h]->set(st_vec[i].second, st_vec[j].second, C[h]->get(i, j) );
      }
    }
  }

  // py_mat_print(Cprime[0], "C_rebased.mat");

  //set C as Cprime...
  for(int h=0; h<nirrep_; h++){
    C[h]->copy(Cprime[h]);
  }
}

//will each time write a file (ment for use with single tau run..)
void FCISolver::twomulent_correlation(std::vector<double>& Tau_2RCM_cor_info, std::string Tau_method){

  C_->compute_rdms();

  size_t nact = active_mo_.size();
  //size_t nact = active_dim_.sum();
  size_t nact2 = nact * nact;
  size_t nact3 = nact2 * nact;
  size_t nact4 = nact3 * nact;
  size_t nact5 = nact4 * nact;

  //if (max_rdm_level_ >= 1) {

      // One-particle density matrices in the active space
      std::vector<double>& opdm_a = C_->opdm_a();
      //std::cout << opdm_a.size() << std::endl;

      std::vector<double>& opdm_b = C_->opdm_b();
      ambit::Tensor L1a = ambit::Tensor::build(ambit::CoreTensor, "L1a", {nact, nact});
      ambit::Tensor L1b = ambit::Tensor::build(ambit::CoreTensor, "L1b", {nact, nact});
      if (na_ >= 1) {
          L1a.iterate([&](const std::vector<size_t>& i, double& value) {
              value = opdm_a[i[0] * nact + i[1]];
          });
      }
      if (nb_ >= 1) {
          L1b.iterate([&](const std::vector<size_t>& i, double& value) {
              value = opdm_b[i[0] * nact + i[1]];
          });
      }

      //if (max_rdm_level_ >= 2) {

          //std::cout << "I get here 2" <<std::endl;
          // Two-particle density matrices in the active space
          ambit::Tensor L2aa =
              ambit::Tensor::build(ambit::CoreTensor, "L2aa", {nact, nact, nact, nact});
          ambit::Tensor L2ab =
              ambit::Tensor::build(ambit::CoreTensor, "L2ab", {nact, nact, nact, nact});
          ambit::Tensor L2bb =
              ambit::Tensor::build(ambit::CoreTensor, "L2bb", {nact, nact, nact, nact});
          ambit::Tensor g2aa =
              ambit::Tensor::build(ambit::CoreTensor, "L2aa", {nact, nact, nact, nact});
          ambit::Tensor g2ab =
              ambit::Tensor::build(ambit::CoreTensor, "L2ab", {nact, nact, nact, nact});
          ambit::Tensor g2bb =
              ambit::Tensor::build(ambit::CoreTensor, "L2bb", {nact, nact, nact, nact});


          if (na_ >= 2) {
              std::vector<double>& tpdm_aa = C_->tpdm_aa();
              L2aa.iterate([&](const std::vector<size_t>& i, double& value) {
                  value = tpdm_aa[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
              });
          }
          if ((na_ >= 1) and (nb_ >= 1)) {
              std::vector<double>& tpdm_ab = C_->tpdm_ab();
              L2ab.iterate([&](const std::vector<size_t>& i, double& value) {
                  value = tpdm_ab[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
              });
          }
          if (nb_ >= 2) {
              std::vector<double>& tpdm_bb = C_->tpdm_bb();
              L2bb.iterate([&](const std::vector<size_t>& i, double& value) {
                  value = tpdm_bb[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
              });
          }
          g2aa.copy(L2aa);
          g2ab.copy(L2ab);
          g2bb.copy(L2bb);

          // Convert the 2-RDMs to 2-RCMs
          L2aa("pqrs") -= L1a("pr") * L1a("qs");
          L2aa("pqrs") += L1a("ps") * L1a("qr");

          L2ab("pqrs") -= L1a("pr") * L1b("qs");

          L2bb("pqrs") -= L1b("pr") * L1b("qs");
          L2bb("pqrs") += L1b("ps") * L1b("qr");

          std::vector<double> twoRCMaa = L2aa.data();
          std::vector<double> twoRCMab = L2ab.data();
          std::vector<double> twoRCMbb = L2bb.data();

          double Cumu_Fnorm_sq = 0.0;
          for(int i = 0; i < nact4; i++){
            //double idx = i*nact3 + i*nact2 + i*nact + i;
            Cumu_Fnorm_sq += twoRCMaa[i] * twoRCMaa[i]
                           + 2.0 * twoRCMab[i] * twoRCMab[i]
                           + twoRCMbb[i] * twoRCMbb[i];
          }
          //std::cout << "I get here 3" <<std::endl;
          Tau_2RCM_cor_info.push_back(Cumu_Fnorm_sq);
          if (options_.get_bool("PRINT_ALL_2RCM")) {

            std::ofstream my_2RCM_file;
            my_2RCM_file.open (Tau_method + "2RCM.dat");

            for(int i = 0; i < nact4; i++){
              my_2RCM_file << twoRCMaa[i] << " " << twoRCMab[i] << " " << twoRCMab[i] << " " << twoRCMbb[i] << " ";
            }
            my_2RCM_file.close();
          }
      //}
   //}
}

//will add to a vector to allow a file similar to Compression.dat to be written
void FCISolver::entanglement_info_1orb(std::vector<std::vector<double> >& Tau_1oee_info){
  //size_t nact = active_mo_.size();

  C_->compute_rdms();
  size_t nact = active_dim_.sum();
  size_t nact2 = nact * nact;
  size_t nact3 = nact2 * nact;
  size_t nact4 = nact3 * nact;
  size_t nact5 = nact4 * nact;

  /////////////////////////////////////////////////////////////////////////////

  // One-particle density matrices in the active space
  std::vector<double>& opdm_a = C_->opdm_a();
  //std::cout << opdm_a.size() << std::endl;

  std::vector<double>& opdm_b = C_->opdm_b();
  ambit::Tensor L1a = ambit::Tensor::build(ambit::CoreTensor, "L1a", {nact, nact});
  ambit::Tensor L1b = ambit::Tensor::build(ambit::CoreTensor, "L1b", {nact, nact});
  if (na_ >= 1) {
      L1a.iterate([&](const std::vector<size_t>& i, double& value) {
          value = opdm_a[i[0] * nact + i[1]];
      });
  }
  if (nb_ >= 1) {
      L1b.iterate([&](const std::vector<size_t>& i, double& value) {
          value = opdm_b[i[0] * nact + i[1]];
      });
  }

  //if (max_rdm_level_ >= 2) {

      //std::cout << "I get here 2" <<std::endl;
      // Two-particle density matrices in the active space
      ambit::Tensor L2aa =
          ambit::Tensor::build(ambit::CoreTensor, "L2aa", {nact, nact, nact, nact});
      ambit::Tensor L2ab =
          ambit::Tensor::build(ambit::CoreTensor, "L2ab", {nact, nact, nact, nact});
      ambit::Tensor L2bb =
          ambit::Tensor::build(ambit::CoreTensor, "L2bb", {nact, nact, nact, nact});
      ambit::Tensor g2aa =
          ambit::Tensor::build(ambit::CoreTensor, "L2aa", {nact, nact, nact, nact});
      ambit::Tensor g2ab =
          ambit::Tensor::build(ambit::CoreTensor, "L2ab", {nact, nact, nact, nact});
      ambit::Tensor g2bb =
          ambit::Tensor::build(ambit::CoreTensor, "L2bb", {nact, nact, nact, nact});


      if (na_ >= 2) {
          std::vector<double>& tpdm_aa = C_->tpdm_aa();
          L2aa.iterate([&](const std::vector<size_t>& i, double& value) {
              value = tpdm_aa[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
          });
      }
      if ((na_ >= 1) and (nb_ >= 1)) {
          std::vector<double>& tpdm_ab = C_->tpdm_ab();
          L2ab.iterate([&](const std::vector<size_t>& i, double& value) {
              value = tpdm_ab[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
          });
      }
      if (nb_ >= 2) {
          std::vector<double>& tpdm_bb = C_->tpdm_bb();
          L2bb.iterate([&](const std::vector<size_t>& i, double& value) {
              value = tpdm_bb[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
          });
      }
      g2aa.copy(L2aa);
      g2ab.copy(L2ab);
      g2bb.copy(L2bb);

      // Convert the 2-RDMs to 2-RCMs
      L2aa("pqrs") -= L1a("pr") * L1a("qs");
      L2aa("pqrs") += L1a("ps") * L1a("qr");

      L2ab("pqrs") -= L1a("pr") * L1b("qs");

      L2bb("pqrs") -= L1b("pr") * L1b("qs");
      L2bb("pqrs") += L1b("ps") * L1b("qr");

  /////////////////////////////////////////////////////////////////////////////
  // std::vector<double>& opdm_a = C_->opdm_a();
  // std::vector<double>& opdm_b = C_->opdm_b();
  // std::vector<double>& tpdm_ab = C_->tpdm_ab();

  std::vector<double> oneRDM_a = L1a.data();
  std::vector<double> oneRDM_b = L1b.data();
  std::vector<double> twoRDM_ab = g2ab.data();

  //std::cout << "I GET HERE 2" << std::endl;
  //std::cout << "nact: "<< nact << std::endl;

  double eig_discard_cut = 1.0e-15;

  std::vector<double> one_orb_ee(nact);
  for(int i=0; i<nact; i++){
    //std::cout << "I GET HERE 3" << std::endl;
    double idx1 = i*nact + i;
    double idx2 = i*nact3 + i*nact2 + i*nact + i;

    // std::cout << "(" << i << ")" << "  1RDM_a_val:  " << oneRDM_a[i] << std::endl;
    // std::cout << " (" << i << ")" << "  1RDM_b_val:  " << oneRDM_b[i] << std::endl;
    // std::cout << "  (" << i << ")" << "  2RDM_ab_val:  " << twoRDM_ab[i] << std::endl;

    //TEST
    //std::cout << "OPDM_a("<< i <<"): " << opdm_a[idx1] << std::endl;
    //END TEST
    // double value = (1.0-opdm_a[idx1]-opdm_b[idx1]+tpdm_ab[idx2])*std::log(1.0-opdm_a[idx1]-opdm_b[idx1]+tpdm_ab[idx2])
    //              + (opdm_a[idx1] - tpdm_ab[idx2])*std::log(opdm_a[idx1] - tpdm_ab[idx2])
    //              + (opdm_b[idx1] - tpdm_ab[idx2])*std::log(opdm_b[idx1] - tpdm_ab[idx2])
    //              + (tpdm_ab[idx2])*std::log(tpdm_ab[idx2]);
    // value *= -1.0;

    double val1 = (1.0 - oneRDM_a[idx1] - oneRDM_b[idx1] + twoRDM_ab[idx2]);
    double val2 = (oneRDM_a[idx1] - twoRDM_ab[idx2]);
    double val3 = (oneRDM_b[idx1] - twoRDM_ab[idx2]);
    double val4 = (twoRDM_ab[idx2]);

    // std::cout << "(" << i << ")" << "  val1:  " << val1 << std::endl;
    // std::cout << " (" << i << ")" << "  val2:  " << val2 << std::endl;
    // std::cout << "  (" << i << ")" << "  val3:  " << val3 << std::endl;
    // std::cout << "   (" << i << ")" << "  val4:  " << val4 << std::endl;

    double value = (1.0-oneRDM_a[idx1]-oneRDM_b[idx1]+twoRDM_ab[idx2])*std::log(1.0-oneRDM_a[idx1]-oneRDM_b[idx1]+twoRDM_ab[idx2])
                 + (oneRDM_a[idx1] - twoRDM_ab[idx2])*std::log(oneRDM_a[idx1] - twoRDM_ab[idx2])
                 + (oneRDM_b[idx1] - twoRDM_ab[idx2])*std::log(oneRDM_b[idx1] - twoRDM_ab[idx2])
                 + (twoRDM_ab[idx2])*std::log(twoRDM_ab[idx2]);
    value *= -1.0;

    one_orb_ee[i] = value;
  }
  Tau_1oee_info.push_back(one_orb_ee);
}

void FCISolver::ap_sci(std::vector<SharedMatrix>& C, double ETA,
                             FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints,
                             double fci_energy, std::vector<double>& Tau_info)
{

    /// ** Pt. 1 **
    double nuclear_repulsion_energy =
        Process::environment.molecule()->nuclear_repulsion_energy({0, 0, 0});

    int nirrep = C.size();

    //find total number of parameters
    int Npar = 0;
    for( auto C_h: C){
      Npar += (C_h->rowdim()) * (C_h->coldim());
    }

    std::cout << "\n==> AP-SCI <==" << std::endl;

    /// A sorted vector of CI coefficients
    std::vector<std::tuple<double, int, int, int> > sorted_CI;

    // set values of sorted_CI to actual coefs^2
    double wfn_nrm = 0.0;
    for (int h=0; h<nirrep; h++) {
        for(int i=0; i<C[h]->coldim(); i++){
            for(int j=0; j<C[h]->rowdim(); j++){
                double val1 = std::pow(C[h]->get(i,j), 2);
                wfn_nrm += val1;

                double val2 = std::abs(C[h]->get(i,j));
                sorted_CI.push_back(std::make_tuple(val2, h, i, j));
            }
        }
    }
    std::cout << "/* Npar (FCI) =           " << Npar << std::endl;
    std::cout << "/* Size of sorted_CI =    " << sorted_CI.size() << std::endl;
    std::cout << "/* Wave function norm =   "<< std::setprecision (17) << wfn_nrm << std::endl;

    // sort them
    std::sort(sorted_CI.begin(), sorted_CI.end());

    // make sure the sort is corret, print largest and smallest 5 coeffs.
    std::cout << "\n/* Largest and smallest |Ci| values */" << std::endl;
    std::cout << "    big idx" << "    Ci" << "                big idx" << "    Ci" << '\n';
    for (int k=0; k<5; k++) {
        std::cout << "    " << Npar-1-k << "        " << std::setprecision (17) << std::get<0>(sorted_CI[Npar-1-k])
                  << "    " <<        k << "        " << std::setprecision (17) << std::get<0>(sorted_CI[k])
                  << std::endl;
    }

    // reconstruct C_ with only largest eigenvalues
    double sum_val = 0.0;
    int tk = 0; // the tau index
    while (sum_val < ETA) {
        // should skip entirely if ETA = 0.0
        int h_ = std::get<1>(sorted_CI[tk]);
        int i_ = std::get<2>(sorted_CI[tk]);
        int j_ = std::get<3>(sorted_CI[tk]);
        double Ci_ = C[h_]->get(i_, j_);

        // souce of a problem may be here, squaring small numbers may cause issues?
        sum_val += std::pow(Ci_, 2);

        C[h_]->set(i_, j_, 0.0);
        tk++;
    }

    int Nred = Npar - tk;

    // re-Normalize
    double trunk_norm = 0.0;
    for (auto C_h : C) {
        trunk_norm += C_h->sum_of_squares();
    }
    trunk_norm = std::sqrt(trunk_norm);
    for (auto C_h : C) {
        C_h->scale(1. / trunk_norm);
    }

    double Norm = 0.0;
    for(auto C_h: C){
      Norm += C_h->sum_of_squares();
    }

    std::cout << "\n/* Nred =           " << Nred << std::endl;
    std::cout << "/* Tau ap-SCI =     " << std::setprecision (17) << ETA << std::endl;
    std::cout << "/* Trunc. norm =    " << std::setprecision (17) << trunk_norm << std::endl;
    std::cout << "/* New norm =       " << std::setprecision (17) << Norm << std::endl;

    // do other stuff...
}


void FCISolver::tile_chopper(std::vector<SharedMatrix>& C, double ETA,
                             FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints,
                             double fci_energy, int dim, std::vector<double>& Tau_info)
{
  double nuclear_repulsion_energy =
      Process::environment.molecule()->nuclear_repulsion_energy({0, 0, 0});

    // allocate memorey for string vector, for now to handle C1 symmetry
    std::vector<std::pair<double, int> > st_vec(C[0]->coldim());

    // optional clustering of basis here
    if(options_.get_bool("FCI_CLUSTER_BASIS")){
      basis_cluster(C, st_vec);
    }

    int nirrep = C.size();

    std::vector<int> b_r(1);
    std::vector<int> e_r(1);
    std::vector<int> b_c(1);
    std::vector<int> e_c(1);

    std::vector<std::tuple<double, int, int, int> > sorted_tiles;

    //find total number of parameters
    int Npar = 0;
    for( auto C_h: C){
      Npar += (C_h->rowdim()) * (C_h->coldim());
    }

    //find and sort tiles by tile_factor
    for (int h=0; h<nirrep; h++) {
      // loop over irreps
      int ncol = C[h]->rowdim();
      int nrow = C[h]->coldim();

      int nt_cols = ncol/dim; //is actually 1 less than total # of colums UNLESS last_col_dim = 0
      int nt_rows = nrow/dim;

      int last_col_dim = ncol%dim;
      int last_row_dim = nrow%dim;

      // in cases mat can be tiled perfectly (/w out remainder)
      if(last_col_dim == 0 || last_row_dim == 0){
        for(int i=0; i<nt_rows; i++){
          for(int j=0; j<nt_cols; j++){
            add_to_tle_vect(C, b_r, e_r, b_c, e_c, dim, dim, dim, h, i, j, sorted_tiles);
          }
        }
      } else { // if not a perfect fit...

        //std::cout << "nt_cols:  "  << nt_cols <<std::endl;

        for(int i=0; i<nt_rows+1; i++){
          for(int j=0; j<nt_cols+1; j++){
            // make dimension objects
            //std::cout << "I get here i: " << i << "  j:  " << j <<std::endl;
            if(j == nt_cols && i == nt_rows){
              // make dimension objects for case of very last tile
              add_to_tle_vect(C, b_r, e_r, b_c, e_c, dim, last_row_dim, last_col_dim, h, i, j, sorted_tiles);

            } else if(j == nt_cols){
              //std::cout << "I get here i: " << i << "  j:  " << j <<std::endl;
              // case of last tile colum
              add_to_tle_vect(C, b_r, e_r, b_c, e_c, dim, dim, last_col_dim, h, i, j, sorted_tiles);

            } else if(i == nt_rows){
              // case of last tile row
              add_to_tle_vect(C, b_r, e_r, b_c, e_c, dim, last_row_dim, dim, h, i, j, sorted_tiles);

            } else {
              // most cases (main block of tiles of dim x dim)
              add_to_tle_vect(C, b_r, e_r, b_c, e_c, dim, dim, dim, h, i, j, sorted_tiles);
              //std::cout << "I get here i: " << i << "  j:  " << j <<std::endl;
              }
          //outfile->Printf("\nh dim: %6d  i dim: %6d  j dim: %6d  ",rank_tile_inirrep.size(), rank_tile_inirrep[h].size(), rank_tile_inirrep[h][i].size());
          }
        }
      } // end else
    } // end h

    //re-define tile_norm_cut
    double area_norm = 0.0;

    // for (auto T : sorted_tiles) {
    //     outfile->Printf("\n   %20.12f      %d      %d      %d", std::get<0>(T), std::get<1>(T), std::get<2>(T), std::get<3>(T));
    //     area_norm += std::get<0>(T);
    // }

    outfile->Printf("\n");

    std::sort(sorted_tiles.rbegin(), sorted_tiles.rend());

    // outfile->Printf("\n");
    // outfile->Printf("\n        t norm value    irrep      tile(i)     tile(j)");
    // outfile->Printf("\n------------------------------------------------------------");

    area_norm = 0.0;
    for (auto T : sorted_tiles) {
        //outfile->Printf("\n   %20.12f      %d      %d      %d", std::get<0>(T), std::get<1>(T), std::get<2>(T), std::get<3>(T));
        area_norm += std::get<0>(T);
    }

    // outfile->Printf("\n");
    // outfile->Printf("\n Norm of tiles: %20.12f ", area_norm);

    // find cutoff ETA
    double norm_cut = 1.0 - ETA;
    double tile_norm_sum = 0.0;
    int eta_counter = 0;
    double diff_val;

    //std::cout << "\n ==> NEW TAU : " << ETA << " <===  " << std::endl;

    for (auto T : sorted_tiles) {
        tile_norm_sum += std::get<0>(T);
        if (tile_norm_sum > norm_cut * area_norm) {
            break;
        }
        eta_counter++;
        //std::cout << "ETA Count :  " << eta_counter << std::endl;
        //std::cout << "tle norm sum :  " << tile_norm_sum << std::endl;
        //std::cout << "N tiles :  " << sorted_tiles.size() << std::endl;
        if(eta_counter < sorted_tiles.size()){
          diff_val = std::get<0>(sorted_tiles[eta_counter - 1]) - std::get<0>(sorted_tiles[eta_counter]);
          //std::cout << "==> vec[count-1] - vec[count] :  " << diff_val << std::endl;
        }
    }

    //std::cout << " I get here and the 13th - 14th: vec[12] - vec[13]  " << diff_val << std::endl;
    //std::cout << " I get here and the (c-1)th - (c)th: vec[c-2] - vec[c-1]  " << diff_val << std::endl;
    // std::cout << "ETA Count :  " << eta_counter << std::endl;
    // std::cout << "tle norm cut :  " << std::get<0>(sorted_tiles[eta_counter]) << std::endl;
    // std::cout << "N tiles  " << sorted_tiles.size() << std::endl;


    double tile_norm_cut;
    if(eta_counter == sorted_tiles.size() ) {
      tile_norm_cut = 0.0;
    } else if (eta_counter < sorted_tiles.size() ) {
      //std::cout << "eta_counter less than sorted_tiles.size" << std::endl;
      if(diff_val < 1e-16 /* OPTION */){ // in case s_t[eta_counter-1] is very close in value to s_t[eta_counter]

        if(eta_counter == sorted_tiles.size() - 1){ // again include all tiles if we make it to 2nd to last one
          tile_norm_cut = 0.0;
          //std::cout << "  ==> Included all tiles b/c last-1 was very close to last! : " << eta_counter << std::endl;
        } else { // inclue the almost same valed tiles
          double diff_val2 = std::get<0>(sorted_tiles[eta_counter]) - std::get<0>(sorted_tiles[eta_counter+1]);
          tile_norm_cut =  tile_norm_cut = std::get<0>(sorted_tiles[eta_counter]) - 0.5*diff_val2;
          //std::cout << "  ==> Close Tile inclueded! : " << eta_counter << std::endl;
        }

      } else { // buisness as usual
        tile_norm_cut = std::get<0>(sorted_tiles[eta_counter-1]) - 0.5*diff_val;
        //std::cout << "  ==> Buisness as usual : " << eta_counter << std::endl;
        //std::cout << "  ==> tnc : " << tile_norm_cut << std::endl;
        //std::cout << "  ==> last included : " << std::get<0>(sorted_tiles[eta_counter-1]) << std::endl;
        //std::cout << "  ==> first excluded : " << std::get<0>(sorted_tiles[eta_counter]) << std::endl;

      }
    } else {
      std::cout << "WOAH, should be a seg fault ..." << std::endl;
    }




    for (int h=0; h<nirrep; h++) {
      // loop over irreps
      int ncol = C[h]->rowdim();
      int nrow = C[h]->coldim();

      int nt_cols = ncol/dim;
      int nt_rows = nrow/dim;

      int last_col_dim = ncol%dim;
      int last_row_dim = nrow%dim;

      //std::cout << "nt_cols:  "  << nt_cols <<std::endl;
      // in cases mat can be tiled perfectly (/w out remainder)
      if(last_col_dim == 0 || last_row_dim == 0){
        for(int i=0; i<nt_rows; i++){
          for(int j=0; j<nt_cols; j++){
            zero_tile(C, b_r, e_r, b_c, e_c, tile_norm_cut, dim, dim, dim, h, i, j, Npar);
          }
        }
      } else {

        for(int i=0; i<nt_rows+1; i++){
          for(int j=0; j<nt_cols+1; j++){
            // make dimension objects
            //std::cout << "I get here i: " << i << "  j:  " << j <<std::endl;
            if(j == nt_cols && i == nt_rows){
              // make dimension objects for case of very last tile
              zero_tile(C, b_r, e_r, b_c, e_c, tile_norm_cut, dim, last_row_dim, last_col_dim, h, i, j, Npar);

            } else if(j == nt_cols){
              //std::cout << "I get here i: " << i << "  j:  " << j <<std::endl;
              // case of last tile colum
              zero_tile(C, b_r, e_r, b_c, e_c, tile_norm_cut, dim, dim, last_col_dim, h, i, j, Npar);

            } else if(i == nt_rows){
              // case of last tile row
              zero_tile(C, b_r, e_r, b_c, e_c, tile_norm_cut, dim, last_row_dim, dim, h, i, j, Npar);

            } else {
              // most cases (main block of tiles of dim x dim)
              zero_tile(C, b_r, e_r, b_c, e_c, tile_norm_cut, dim, dim, dim, h, i, j, Npar);
              //std::cout << "I get here i: " << i << "  j:  " << j <<std::endl;
              }
          //outfile->Printf("\nh dim: %6d  i dim: %6d  j dim: %6d  ",rank_tile_inirrep.size(), rank_tile_inirrep[h].size(), rank_tile_inirrep[h][i].size());
          }
        }
      } // end else
    }

    //re-nomralize Wfn
    double norm_C_chopped = 0.0;
    for (auto C_h : C) {
        norm_C_chopped += C_h->sum_of_squares();
    }
    norm_C_chopped = std::sqrt(norm_C_chopped);
    for (auto C_h : C) {
        C_h->scale(1. / norm_C_chopped);
    }

    double Norm = 0.0;
    for(auto C_h: C){
      Norm += C_h->sum_of_squares();
    }

    // Re-order basis from clustered form
    if(options_.get_bool("FCI_CLUSTER_BASIS")){
      rev_basis_cluster(C, st_vec);
    }

    //print Block Sparse C
    //py_mat_print(C[0], "C_tc.mat");

    //compute energy

    // Set HC = H C
    C_->Hamiltonian(HC, fci_ints, twoSubstituitionVVOO);
    // Calculate E = C^T HC
    double E_block_chop = HC.dot(C_) + nuclear_repulsion_energy;

    int Nsto = Npar;
    //double Npar_db = Npar;

    Tau_info.push_back(Npar);
    Tau_info.push_back(E_block_chop-fci_energy);

    outfile->Printf("\n////////////////// Tile Chopper /////////////////");
    outfile->Printf("\n");
    outfile->Printf("\n ETA               = %20.12f", ETA);
    outfile->Printf("\n tile cutoff       = %20.12f", tile_norm_cut);
    outfile->Printf("\n Norm              = %20.12f", Norm);
    outfile->Printf("\n E_fci             = %20.12f", fci_energy);
    outfile->Printf("\n E_tile_chop       = %20.12f", E_block_chop);
    outfile->Printf("\n Delta(E_tle_chp)  = %20.12f", E_block_chop-fci_energy);
    outfile->Printf("\n TC Dim            =     %6d", dim);
    outfile->Printf("\n Npar tle_chp      =     %6d", Npar);
    outfile->Printf("\n Nsto tle_chp      =     %6d", Nsto);
    outfile->Printf("\n");
}

void FCISolver::string_trimmer(std::vector<SharedMatrix>& C, double DELTA, FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy, std::vector<double>& Tau_info)
{
  double nuclear_repulsion_energy =
      Process::environment.molecule()->nuclear_repulsion_energy({0, 0, 0});

  int nirrep = C.size();
  //std::vector<std::pair<double, int> > sorted_strings;
  std::vector<std::tuple<double, int, int> > sorted_strings;
                      // Omega    h    i
  for(int h=0; h < nirrep; h++){
    // for alpha strings
    for(int i=0; i<C[h]->coldim(); i++){
      double temp = 0.0;
      for(int j=0; j<C[h]->rowdim(); j++){
        temp += std::pow(C[h]->get(i,j), 2);
      }
      sorted_strings.push_back(std::make_tuple(temp, h, i));
    }
  }



  // for(auto C_h: C){
  //   // for alpha strings
  //   for(int i=0; i<C_h->coldim(); i++){
  //     double temp = 0.0;
  //     for(int j=0; j<C_h->rowdim(); j++){
  //       temp += std::pow(C_h->get(i,j), 2);
  //     }
  //     sorted_strings.push_back(std::make_pair(temp, i));
  //   }
  // }

  // for (auto T : sorted_strings) {
  //     outfile->Printf("\n   %20.12f      %d      ", T.first, T.second);
  //     //st_norm += ST.first;
  // }

  outfile->Printf("\n");

  std::sort(sorted_strings.begin(), sorted_strings.end(), pairTupleCompare);

  // check 1

  double st_norm = 0.0;
  for (auto ST : sorted_strings) {
      //outfile->Printf("\n   %20.12f      %d      ", T.first, T.second);
      st_norm += std::get<0>(ST);
      //st_norm += T.first;
  }

  // outfile->Printf("\n");
  // outfile->Printf("\n Norm of tiles: %20.12f ", area_norm);

  // find cutoff ETA
  double norm_cut = 1.0 - DELTA;
  double string_norm_sum = 0;
  int delta_counter = 0;
  double diff_val;

  //std::cout << "\n ==> NEW TAU : " << DELTA << " <===  " << std::endl;

  for (auto ST : sorted_strings) { // shoud be for loop over all strings
      //string_norm_sum += ST.first;
      string_norm_sum += std::get<0>(ST);
      if (string_norm_sum > norm_cut * st_norm) {
          break;
      }
      delta_counter++;
      // std::cout << "ETA Count :  " << delta_counter << std::endl;
      // std::cout << "string norm sum :  " << string_norm_sum << std::endl;
      // std::cout << "N strings  " << sorted_strings.size() << std::endl;
      if(delta_counter < sorted_strings.size()){
        diff_val = std::get<0>(sorted_strings[delta_counter - 1]) - std::get<0>(sorted_strings[delta_counter]);
        //diff_val = sorted_strings[delta_counter - 1].first - sorted_strings[delta_counter].first;
        //std::cout << "vec[count-1] - vec[count] :  " << diff_val << std::endl;
      }
  }

  //check 2

  // std::cout << "Delta Count :  " << delta_counter << std::endl;
  // std::cout << "str norm cut :  " << sorted_strings[delta_counter].first << std::endl;
  // std::cout << "N Strings :  " << sorted_strings.size() << std::endl;

  //double sum_cut = sorted_strings[delta_counter-1].first;
  double sum_cut;
  if(delta_counter == sorted_strings.size() ) {
    sum_cut = 0.0;
  } else if (delta_counter < sorted_strings.size() ) {
    //sum_cut = sorted_strings[delta_counter-1].first;
      if(diff_val < 1e-16 /* OPTION */){ // in case s_s[delta_counter-1] is very close in value to s_s[delta_counter]

        if(delta_counter == sorted_strings.size() - 1){ // again include all strings if we make it to 2nd to last one
          sum_cut = 0.0;
        } else { // inclue the 'almost' same valed strings
          double diff_val2 = std::get<0>(sorted_strings[delta_counter]) - std::get<0>(sorted_strings[delta_counter+1]);
          sum_cut = std::get<0>(sorted_strings[delta_counter]) - 0.5*diff_val2;
          // double diff_val2 = sorted_strings[delta_counter].first - sorted_strings[delta_counter+1].first;
          // sum_cut = sorted_strings[delta_counter].first - 0.5*diff_val2;
          //std::cout << "==> Close string inclueded! : " << delta_counter << std::endl;
        }

      } else { // buisness as usual
        sum_cut = std::get<0>(sorted_strings[delta_counter-1]) - 0.5*diff_val;
        //sum_cut = sorted_strings[delta_counter-1].first - 0.5*diff_val;
        // std::cout << "  ==> Buisness as usual : " << delta_counter << std::endl;
        // std::cout << "  ==> tnc : " << sum_cut << std::endl;
        // std::cout << "  ==> last included : " << sorted_strings[delta_counter-1].first << std::endl;
        // std::cout << "  ==> first excluded : " << sorted_strings[delta_counter].first << std::endl;

      }
  } else {
    std::cout << "WOAH, should be a seg fault ..." << std::endl;
  }

  //check 3

  double Om_a;
  double Om_b;

  int N_par = 0;
  std::vector<std::vector<int> > Ia_bool(nirrep);
  std::vector<std::vector<int> > Ib_bool(nirrep);



  // std::vector<int> Ia_bool(C[0]->coldim(), 1);
  // std::vector<int> Ib_bool(C[0]->rowdim(), 1);

  // for(auto C_h: C){
  //   N_par += C_h->rowdim() * C_h->coldim();
  // }

  for(int h=0; h < nirrep; h++){
  // for(auto C: C){
    // for alpha strings
    for(int i=0; i<C[h]->coldim(); i++){
      Om_a = 0;
      for(int j=0; j<C[h]->rowdim(); j++){
        Om_a += std::pow(C[h]->get(i,j), 2);
      }
      if(Om_a < sum_cut){
        Ia_bool[h].push_back(0);
        // for(int j=0; j<C_h->rowdim(); j++){
        //   //C_h->set(i,j,0);
        //   N_par--;
        // }
      } else {
        Ia_bool[h].push_back(1);
      }
    }

    // for beta strings
    for(int j=0; j<C[h]->rowdim(); j++){
      Om_b = 0;
      for(int i=0; i<C[h]->coldim(); i++){
        Om_b += std::pow(C[h]->get(i,j), 2);
      }
      if(Om_b < sum_cut){
        Ib_bool[h].push_back(0);
        // for(int i=0; i<C_h->coldim(); i++){
        //   //C_h->set(i,j,0);
        //   N_par--;
      } else {
        Ib_bool[h].push_back(1);
      }

      ////WHERE WAS I? //////

      // for(int j=0; j<C_h->rowdim(); j++){
      //   Om_b = 0;
      //   for(int i=0; i<C_h->coldim(); i++){
      //     Om_b += std::pow(C_h->get(i,j), 2);
      //   }
      //   if(Om_b < sum_cut){
      //     Ib_bool[j]--;
      //     // for(int i=0; i<C_h->coldim(); i++){
      //     //   //C_h->set(i,j,0);
      //     //   N_par--;
      //   }
    }


    //set blocks
    for(int i=0; i<C[h]->coldim(); i++){
      for(int j=0; j<C[h]->rowdim(); j++){
        if(Ia_bool[h][i]*Ib_bool[h][j] > 0) N_par++;
        C[h]->set(i ,j , Ia_bool[h][i]*Ib_bool[h][j]*C[h]->get(i,j));
      }
    }
  }

  //std::cout << "I get here 2!" << std::endl;
  // Re-Normalize Wvfn

  double norm_C_trimmed = 0.0;
  for (int h=0; h<nirrep; h++) {
      norm_C_trimmed += C[h]->sum_of_squares();
  }

  // for (auto C_h : C) {
  //     norm_C_trimmed += C_h->sum_of_squares();
  // }
  norm_C_trimmed = std::sqrt(norm_C_trimmed);
  for (int h=0; h<nirrep; h++) {
      C[h]->scale(1. / norm_C_trimmed);
  }
  // for (auto C_h : C) {
  //     C_h->scale(1. / norm_C_trimmed);
  // }

  double Norm = 0.0;

  for(int h=0; h<nirrep; h++){
    Norm += C[h]->sum_of_squares();
  }

  //Print MATRIX
  //py_mat_print(C[0], "C_st.mat");

  //Compute energy

  // Set HC = H C
  C_->Hamiltonian(HC, fci_ints, twoSubstituitionVVOO);
  // Calculate E = C^T HC
  double E_string_trim = HC.dot(C_) + nuclear_repulsion_energy;

  int N_sto = N_par;
  //double Npar_db = Npar;

  Tau_info.push_back(N_par);
  Tau_info.push_back(E_string_trim-fci_energy);

  outfile->Printf("\n////////////////// String Trimmer /////////////////");
  outfile->Printf("\n");
  outfile->Printf("\n DELTA             = %20.12f", DELTA);
  outfile->Printf("\n string cutoff     = %20.12f", sum_cut);
  outfile->Printf("\n Norm              = %20.12f", Norm);
  outfile->Printf("\n E_fci             = %20.12f", fci_energy);
  outfile->Printf("\n E_red_rank        = %20.12f", E_string_trim);
  outfile->Printf("\n Delta(E_st)       = %20.12f", E_string_trim-fci_energy);
  outfile->Printf("\n Npar st           =     %6d", N_par);
  outfile->Printf("\n Nsto st           =     %6d", N_sto);
  outfile->Printf("\n");
}

void FCISolver::string_stats(std::vector<SharedMatrix> C)
{
  int n_a = 0;
  int n_b = 0;

  //may need to apend loop bounds to account for multiple irreps if using symmetry
  for(auto C_h: C){
    n_a += C_h->rowdim();
    n_b += C_h->coldim();
  }


  double Om_a;
  double Om_b;
  std::vector<double> C_Ia_mag(n_a, 0);
  std::vector<double> C_Ib_mag(n_b, 0);
  std::vector<int> C_Ia_mag_histo(11, 0);
  std::vector<int> C_Ib_mag_histo(11, 0);
  std::vector<int> C_Ia_nuses(n_a, 0); //range 0 to n_b
  std::vector<int> C_Ib_nuses(n_b, 0); //range 0 to n_a

  for(auto C_h: C){
    // alpha string statistics
    for(int i=0; i<C_h->coldim(); i++){
      Om_a = 0;
      for(int j=0; j<C_h->rowdim(); j++){
        //std::cout << std::abs(C_h->get(i,j)) << std::endl;
        Om_a += std::pow(C_h->get(i,j), 2);
        //std::cout << "Om_a: " << Om_a << std::endl;
        C_Ia_mag[i] += std::pow(C_h->get(i,j), 2);
        if(std::pow(C_h->get(i,j),2) > 1e-12){
          C_Ia_nuses[i]++;
        }
      }

      //std::cout << "  Om_a:  " << Om_a << std::endl;
      Om_a = -std::log10(Om_a);
      //std::cout << "i: " << i << "  Om_a:  " << Om_a << std::endl;
      if(Om_a >= 10){
        C_Ia_mag_histo[10]++;
      } else; C_Ia_mag_histo[std::floor(Om_a)]++;
    }

    // now for beta strings (should be the same for symmetric Cmat)
    for(int i=0; i<C_h->rowdim(); i++){
      Om_b = 0;
      for(int j=0; j<C_h->coldim(); j++){
        C_Ib_mag[i] += std::pow(C_h->get(j,i), 2);
        Om_b += std::pow(C_h->get(j,i), 2);
        if(std::pow(C_h->get(j,i),2) > 1e-12){
          C_Ib_nuses[i]++;
        }
      }
      Om_b = -std::log10(Om_b);
      if(Om_b >= 10){
        C_Ib_mag_histo[10]++;
      } else C_Ib_mag_histo[std::floor(Om_b)]++;
    }
  }

  std::sort(C_Ia_mag.rbegin(), C_Ia_mag.rend());
  std::sort(C_Ib_mag.rbegin(), C_Ib_mag.rend());

  std::ofstream myfile1;
  std::ofstream myfile2;

  myfile1.open ("Ia.dat");
  for(int j=0; j<C_Ia_mag.size(); j++){ myfile1 << " " << C_Ia_mag[j]; }
  myfile1 << "\n";
  for(int j=0; j<C_Ia_mag.size(); j++){ myfile1 << " " << C_Ia_nuses[j]; }
  myfile1.close();

  myfile2.open ("Ib.dat");
  for(int j=0; j<C_Ib_mag.size(); j++){ myfile2 << " " << C_Ib_mag[j]; }
  myfile2 << "\n";
  for(int j=0; j<C_Ib_mag.size(); j++){ myfile2 << " " << C_Ib_nuses[j]; }
  myfile2.close();

  outfile->Printf("\n alpha Strings in bin:");
  for(auto nC_Ia: C_Ia_mag_histo){
    outfile->Printf(" %1d,", nC_Ia);
  }

  outfile->Printf("\n beta  Strings in bin:");
  for(auto nC_Ib: C_Ib_mag_histo){
    outfile->Printf(" %1d,", nC_Ib);
  }

}

void FCISolver::zero_tile(std::vector<SharedMatrix>& C,
                                 std::vector<int> b_r,
                                 std::vector<int> e_r,
                                 std::vector<int> b_c,
                                 std::vector<int> e_c,
                                 double tile_norm_cut,
                                 int dim, int n, int d,
                                 int h, int i, int j, int& Npar )
{
  //prepare dimension objects
  b_r[0] = i*dim;
  e_r[0] = i*dim + n;
  b_c[0] = j*dim;
  e_c[0] = j*dim + d;

  Dimension begin_row(b_r);
  Dimension end_row(e_r);
  Dimension begin_col(b_c);
  Dimension end_col(e_c);


  // make slice objects
  Slice row_slice(begin_row, end_row);
  Slice col_slice(begin_col, end_col);

  // get matrix block
  auto M = C[h]->get_block(row_slice, col_slice);

  double area_factor = ((double)n*(double)d) / ((double)dim*(double)dim);
  //double tile_factor = M->sum_of_squares() / area_factor;
  double tile_factor = C[h]->get(i,j);
  tile_factor *= tile_factor;


  if(tile_factor < tile_norm_cut){
    M->set(0.0);
    C[h]->set_block(row_slice, col_slice, M);
    Npar -= n*d;
  }

}

void FCISolver::add_to_tle_vect(std::vector<SharedMatrix>& C,
                                 std::vector<int> b_r,
                                 std::vector<int> e_r,
                                 std::vector<int> b_c,
                                 std::vector<int> e_c,
                                 int dim, int n, int d,
                                 int h, int i, int j,
                                 std::vector<std::tuple<double, int, int, int> >& sorted_tiles)
{
  //prepare dimension objects
  b_r[0] = i*dim;
  e_r[0] = i*dim + n;
  b_c[0] = j*dim;
  e_c[0] = j*dim + d;

  Dimension begin_row(b_r);
  Dimension end_row(e_r);
  Dimension begin_col(b_c);
  Dimension end_col(e_c);

  // make slice objects
  Slice row_slice(begin_row, end_row);
  Slice col_slice(begin_col, end_col);

  // get matrix block
  auto M = C[h]->get_block(row_slice, col_slice);

  //if(n == 0 || d == 0){std::cout << "\nrought ro! i:  " << i << "  j:  " << std::endl; }

  double area_factor = ((double)n*(double)d) / ((double)dim*(double)dim);
  //double tile_factor = M->sum_of_squares() / area_factor;
  double tile_factor = C[h]->get(i,j);
  tile_factor *= tile_factor;

  if(area_factor < 0.001){std::cout << "\nrought ro! n:  " << n << "  d:  " << d << " val:  "<< (n*d) / (dim*dim)<< std::endl; }
  //if(M->sum_of_squares() > 0.5){std::cout << "\nrought ro! i:  " << i << "  j:  " << j << " val:  "<< M->sum_of_squares() << std::endl; }

  sorted_tiles.push_back(std::make_tuple(tile_factor, h, i, j));

}

void FCISolver::add_to_sig_vect(std::vector<std::tuple<double, int, int, int> >& sorted_sigma,
                                 std::vector<SharedMatrix> C,
                                 std::vector<int> b_r,
                                 std::vector<int> e_r,
                                 std::vector<int> b_c,
                                 std::vector<int> e_c,
                                 int dim, int n, int d,
                                 int h, int i, int j)
{
  //prepare dimension objects
  b_r[0] = i*dim;
  e_r[0] = i*dim + n;
  b_c[0] = j*dim;
  e_c[0] = j*dim + d;

  Dimension begin_row(b_r);
  Dimension end_row(e_r);
  Dimension begin_col(b_c);
  Dimension end_col(e_c);

  // make slice objects
  Slice row_slice(begin_row, end_row);
  Slice col_slice(begin_col, end_col);

  // get matrix block
  auto M = C[h]->get_block(row_slice, col_slice);

  auto u = std::make_shared<Matrix>("u", n, n);
  auto s = std::make_shared<Vector>("s", std::min(n, d));
  auto v = std::make_shared<Matrix>("v", d, d);

  M->svd(u, s, v);
  //s->print();

  // add to sigma vector
  for (int k = 0; k < std::min(n, d); k++) {
      sorted_sigma.push_back(std::make_tuple(s->get(k), h, i, j));
  }
  //outfile->Printf("\n size of sigma vect: %4d h: %4d i: %4d j: %4d",sorted_sigma.size(),h,i,j);
}


void FCISolver::patch_Cmat(std::vector<std::tuple<double, int, int, int> >& sorted_sigma,
                                 std::vector<SharedMatrix>& C,
                                 std::vector<std::vector<std::vector<int> > > rank_tile_inirrep,
                                 std::vector<int> b_r,
                                 std::vector<int> e_r,
                                 std::vector<int> b_c,
                                 std::vector<int> e_c,
                                 int dim, int n, int d,
                                 int h, int i, int j,
                                 int& N_par)
{
  //prepare dimension objects
  b_r[0] = i*dim;
  e_r[0] = i*dim + n;
  b_c[0] = j*dim;
  e_c[0] = j*dim + d;

  Dimension begin_row(b_r);
  Dimension end_row(e_r);
  Dimension begin_col(b_c);
  Dimension end_col(e_c);

  // make slice objects
  Slice row_slice(begin_row, end_row);
  Slice col_slice(begin_col, end_col);

  // get matrix block
  auto M = C[h]->get_block(row_slice, col_slice);

  auto u = std::make_shared<Matrix>("u", n, n);
  auto s = std::make_shared<Vector>("s", std::min(n, d));
  auto v = std::make_shared<Matrix>("v", d, d);

  M->svd_a(u, s, v);
  //s->print();

  //rebuild sigma as matrix
  auto sig_mat = std::make_shared<Matrix>("sig_mat", n, d);
  int rank_ef = rank_tile_inirrep[h][i][j];
  for (int l = 0; l < rank_ef; l++) {
      sig_mat->set(l, l, s->get(l));
  }

  //count paramaters
  N_par += rank_ef*(n + d);

  //std::cout << "i: " << i << "  j:  " << j << " n: " << n << "  d:  " << d <<"  rank eff:  " << rank_ef << std::endl;

  //rebuild M with reduced rank
  auto M_red_rank = Matrix::triplet(u, sig_mat, v, false, false, false);

  // auto M_clone = M->clone();
  // M_clone->subtract(M_red_rank);
  // std::cout << "\ntile diff: " << M_clone->sum_of_squares() << std::endl;

  //splice M_red_rank back into C
  C[h]->set_block(row_slice, col_slice, M_red_rank);



}


void FCISolver::fci_svd_tiles(FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy, int dim, double OMEGA, std::vector<double>& Tau_info)
{
  double nuclear_repulsion_energy =
      Process::environment.molecule()->nuclear_repulsion_energy({0, 0, 0});

    std::vector<SharedMatrix> C_link = C_->coefficients_blocks();
    std::vector<std::pair<double, int> > st_vec1(C_link[0]->coldim());

    if(options_.get_bool("FCI_CLUSTER_BASIS")){
      basis_cluster(C_link, st_vec1);
    }

    std::vector<SharedMatrix> C = C_->coefficients_blocks();
    std::vector<SharedMatrix> C_tiled_rr = C_->coefficients_blocks();

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

        int nt_cols = ncol/dim;
        int nt_rows = nrow/dim;

        int last_col_dim = ncol%dim;
        int last_row_dim = nrow%dim;

        int n_sing_vals_h = dim*nt_cols*nt_rows
                             + last_col_dim*nt_rows
                             + last_row_dim*nt_cols
                             + std::min(last_col_dim, last_row_dim);

        size_of_ssv += n_sing_vals_h;
        rank_tile_inirrep[h].resize(nt_rows+1);

        //outfile->Printf("\n-------------------- NEXT IRREP --------------------\n");
        //C[h]->print();

        // in cases mat can be tiled perfectly (/w out remainder)
        // if(last_col_dim == 0 || last_row_dim == 0){
        //   for(int i=0; i<nt_rows; i++){
        //     for(int j=0; j<nt_cols; j++){
        //       add_to_sig_vect(sorted_sigma, C, b_r, e_r, b_c, e_c, dim, dim, dim, h, i, j);
        //     }
        //   }
        // } else { // if not a perfect fit...

          for(int i=0; i<nt_rows+1; i++){
            // allocate memory for sorting vector
            rank_tile_inirrep[h][i].resize(nt_cols+1); //why I get seg fault...
            for(int j=0; j<nt_cols+1; j++){
              // make dimension objects
              if(j == nt_cols && i == nt_rows){
                // case of very last tile
                add_to_sig_vect(sorted_sigma, C, b_r, e_r, b_c, e_c, dim, last_row_dim, last_col_dim, h, i, j);

              } else if(j == nt_cols){
                // case of last tile colum
                add_to_sig_vect(sorted_sigma, C, b_r, e_r, b_c, e_c, dim, dim, last_col_dim, h, i, j);

              } else if(i == nt_rows){
                // case of last tile row
                add_to_sig_vect(sorted_sigma, C, b_r, e_r, b_c, e_c, dim, last_row_dim, dim, h, i, j);

              } else {
                // most cases (main block of tiles of dim x dim)
                add_to_sig_vect(sorted_sigma, C, b_r, e_r, b_c, e_c, dim, dim, dim, h, i, j);
              }
            //outfile->Printf("\nh dim: %6d  i dim: %6d  j dim: %6d  ",rank_tile_inirrep.size(), rank_tile_inirrep[h].size(), rank_tile_inirrep[h][i].size());
            }
          }
      //} // end else
    }

    /////////////////////////////////// check status ////////////////////////////////////

    std::sort(sorted_sigma.rbegin(), sorted_sigma.rend());
    double sigma_norm = 0.0;

    // outfile->Printf("\n");
    // outfile->Printf("\n        singular value    irrep      tile(i)     tile(j)");
    // outfile->Printf("\n------------------------------------------------------------");

    for (auto sigma_h : sorted_sigma) {
        //outfile->Printf("\n   %20.12f      %d      %d      %d", std::get<0>(sigma_h), std::get<1>(sigma_h), std::get<2>(sigma_h), std::get<3>(sigma_h));
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

    for (auto sigma_h : sorted_sigma) {
        rank_tile_inirrep[std::get<1>(sigma_h)][std::get<2>(sigma_h)][std::get<3>(sigma_h)] += 1;
        sig_sum += std::pow(std::get<0>(sigma_h), 2.0);
        if (sig_sum > norm_cut * sigma_norm) {
            break;
        }
    }

    // outfile->Printf("\n  irrep     tile(i)     tile(j)     red rank");
    // outfile->Printf("\n------------------------------------------------------------");
    //
    // outfile->Printf("\n");
    // for(int h = 0; h<nirrep; h++){
    //   for(int i = 0; i<rank_tile_inirrep[h].size(); i++){
    //     for(int j = 0; j<rank_tile_inirrep[h][i].size(); j++){
    //       outfile->Printf("\n  %6d     %6d     %6d     %6d", h,i,j,rank_tile_inirrep[h][i][j]);
    //     }
    //   }
    // }

    ///// PRE MATRIX TILER END /////

    //// Count how many tiles are of each rank in dimension
    std::vector<int> tile_rank_histo(dim+1,0);
    //std::cout << "\n test:  " << tile_rank_histo[5] << "\n";

    for(int h = 0; h<nirrep; h++){
      for(int i = 0; i<rank_tile_inirrep[h].size(); i++){
        for(int j = 0; j<rank_tile_inirrep[h][i].size(); j++){
          tile_rank_histo[rank_tile_inirrep[h][i][j]]++;
        }
      }
    }

    outfile->Printf("\n \n Tile rank histo: ");
    for(auto quantity_of_rank : tile_rank_histo){
      outfile->Printf("%d, ", quantity_of_rank);
    }


    ///// MAIN MATRIX TILER BEGIN /////

    int N_par = 0;

    // outfile->Printf("\n");
    // outfile->Printf("\n//////////////////// REBUILDING RED RANK TILES ////////////////////\n");
    // outfile->Printf("\n");

    //now reduce rank accordingly!
    for (int h=0; h<nirrep; h++) {
        // loop over irreps
        int ncol = C_tiled_rr[h]->rowdim();
        int nrow = C_tiled_rr[h]->coldim();

        int nt_cols = ncol/dim;
        int nt_rows = nrow/dim;

        int last_col_dim = ncol%dim;
        int last_row_dim = nrow%dim;

        int n_sing_vals_h = dim*nt_cols*nt_rows
                             + last_col_dim*nt_rows
                             + last_row_dim*nt_cols
                             + std::min(last_col_dim, last_row_dim);

        size_of_ssv += n_sing_vals_h;
        rank_tile_inirrep[h].resize(nt_rows+1);

        //outfile->Printf("\n-------------------- NEXT IRREP --------------------\n");
        // in cases mat can be tiled perfectly (/w out remainder)
        // if(last_col_dim == 0 || last_row_dim == 0){
        //   for(int i=0; i<nt_rows; i++){
        //     for(int j=0; j<nt_cols; j++){
        //       patch_Cmat(sorted_sigma, C_tiled_rr, rank_tile_inirrep, b_r, e_r, b_c, e_c, dim, dim, dim, h, i, j, N_par);
        //     }
        //   }
        // } else {

          for(int i=0; i<nt_rows+1; i++){
            // allocate memory for sorting vector
            rank_tile_inirrep[h][i].resize(nt_cols+1);
            for(int j=0; j<nt_cols+1; j++){
              // make dimension objects
              if(j == nt_cols && i == nt_rows){
                // make dimension objects for case of very last tile
                patch_Cmat(sorted_sigma, C_tiled_rr, rank_tile_inirrep, b_r, e_r, b_c, e_c, dim, last_row_dim, last_col_dim, h, i, j, N_par);
              } else if(j == nt_cols){
                // make dimension objects for case of last tile colum
                patch_Cmat(sorted_sigma, C_tiled_rr, rank_tile_inirrep, b_r, e_r, b_c, e_c, dim, dim, last_col_dim, h, i, j, N_par);
              } else if(i == nt_rows){
                // make dimension objects for case of last tile row
                patch_Cmat(sorted_sigma, C_tiled_rr, rank_tile_inirrep, b_r, e_r, b_c, e_c, dim, last_row_dim, dim, h, i, j, N_par);
              } else {
                // make dimension objects
                patch_Cmat(sorted_sigma, C_tiled_rr, rank_tile_inirrep, b_r, e_r, b_c, e_c, dim, dim, dim, h, i, j, N_par);
                }
            //outfile->Printf("\nh dim: %6d  i dim: %6d  j dim: %6d  ",rank_tile_inirrep.size(), rank_tile_inirrep[h].size(), rank_tile_inirrep[h][i].size());
            } // end j
          } // end i
        //} // end else
    } // end h

    //re-normalize

    double norm_C_tiled_rr = 0.0;
    for (auto C_tiled_rr_h : C_tiled_rr) {
        norm_C_tiled_rr += C_tiled_rr_h->sum_of_squares();
    }
    norm_C_tiled_rr = std::sqrt(norm_C_tiled_rr);
    for (auto C_tiled_rr_h : C_tiled_rr) {
        C_tiled_rr_h->scale(1. / norm_C_tiled_rr);
    }

/*
    outfile->Printf("\n irrep red        ||C - C_rr||\n");

    for(int h = 0; h<nirrep; h++){
      auto C_diff = C[h]->clone();
      C_diff->subtract(C_tiled_rr[h]);
      double norm = std::sqrt(C_diff->sum_of_squares());
      outfile->Printf(" %1d %20.12f \n", h, norm);
      //outfile->Printf("\n //// C ////\n");
      //C[h]->print();
      //outfile->Printf("\n //// C_tiled_rr ////\n");
      //C_tiled_rr[h]->print();
    }
*/
    // Compute the energy

    //Print MATRIX

    if(options_.get_bool("FCI_CLUSTER_BASIS")){
      rev_basis_cluster(C_tiled_rr, st_vec1);
    }

    //py_mat_print(C_tiled_rr[0],"C_svd_t.mat");

    // if(options_.get_bool("FCI_CLUSTER_BASIS")){
    //   rev_basis_cluster(C_tiled_rr,st_vec2);
    // }

    C_->set_coefficient_blocks(C_tiled_rr);


    // HC = H C
    C_->Hamiltonian(HC, fci_ints, twoSubstituitionVVOO);
    // E = C^T HC
    double E_red_rank = HC.dot(C_) + nuclear_repulsion_energy;

    //double Npar_db = Npar;

    Tau_info.push_back(N_par);
    Tau_info.push_back(E_red_rank-fci_energy);

    outfile->Printf("\n////////////////// Tile SVD /////////////////\n");
    outfile->Printf("\n OMEGA             = %20.12f", OMEGA);
    outfile->Printf("\n tile size         =     %6d", dim);
    outfile->Printf("\n E_fci             = %20.12f", fci_energy);
    outfile->Printf("\n E_tiled_rr        = %20.12f", E_red_rank);
    outfile->Printf("\n Delta(E_tiled_rr) = %20.12f", E_red_rank-fci_energy);
    outfile->Printf("\n Npar trr          =     %6d", N_par);
    outfile->Printf("\n Nsto trr          =     %6d", N_par);

    ///// MAIN MATRIX TILER END /////

}




void FCISolver::fci_svd(FCIVector& HC, std::shared_ptr<FCIIntegrals> fci_ints, double fci_energy, double TAU, std::vector<double>& Tau_info)
{
  double nuclear_repulsion_energy =
      Process::environment.molecule()->nuclear_repulsion_energy({0, 0, 0});

    std::vector<SharedMatrix> C = C_->coefficients_blocks();
    std::vector<SharedMatrix> C_red_rank;

    //outfile->Printf("\n");
    //outfile->Printf("\n PRINTING CLEAN C MATRICIES\n");

    double norm_C = 0.0;
    for (auto C_h : C) {
        //C_h->print();
        norm_C += C_h->sum_of_squares();
    }

    int nirrep = C.size();
    int total_rank = 0;
    int total_red_rank = 0;

    std::vector<std::pair<double, int>> sorted_sigma; // [(sigma_i, irrep)]

    for (int h = 0; h < nirrep; h++) {
        int nrow = C[h]->rowdim();
        int ncol = C[h]->coldim();

        auto u_p = std::make_shared<Matrix>("u_p", nrow, nrow);
        auto s_p = std::make_shared<Vector>("s_p", std::min(ncol, nrow));
        auto v_p = std::make_shared<Matrix>("v_p", ncol, ncol);

        C[h]->svd(u_p, s_p, v_p);

        std::string mat_file_U = "U_mat_ft_irrep_" + std::to_string(h);
        std::string mat_file_V = "V_mat_ft_irrep_" + std::to_string(h);
        //py_mat_print(u_p, mat_file_U);
        //py_mat_print(v_p, mat_file_V);

        for (int i = 0; i < std::min(ncol, nrow); i++) {
            sorted_sigma.push_back(std::make_pair(s_p->get(i), h));
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
    std::vector<int> rank_ef_inirrep(nirrep,0);

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

        rank_ef_inirrep[h] = rank_ef;

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

    // re-normalize C_red_rank
    double norm_C_red_rank = 0.0;
    for (auto C_red_rank_h : C_red_rank) {
        norm_C_red_rank += C_red_rank_h->sum_of_squares();
    }
    norm_C_red_rank = std::sqrt(norm_C_red_rank);
    for (auto C_red_rank_h : C_red_rank) {
        C_red_rank_h->scale(1. / norm_C_red_rank);
    }

    //outfile->Printf("\n");
    //outfile->Printf("\n PRINTING RR C MATRICIES\n");

    {
        double norm_C_red_rank = 0.0;
        for (auto C_red_rank_h : C_red_rank) {
            norm_C_red_rank += C_red_rank_h->sum_of_squares();
            //C_red_rank_h->print();
        }
        norm_C_red_rank = std::sqrt(norm_C_red_rank);
        outfile->Printf("\n ||C|| = %20.12f", std::sqrt(norm_C_red_rank));
    }

    // Compute the energy

    //Print Matrix
    //py_mat_print(C_red_rank[0],"C_svd_f.mat");

    C_->set_coefficient_blocks(C_red_rank);
    // HC = H C
    C_->Hamiltonian(HC, fci_ints, twoSubstituitionVVOO);
    // E = C^T HC
    double E_red_rank = HC.dot(C_) + nuclear_repulsion_energy;

    //double Npar_db = Npar;

    Tau_info.push_back(N_par*2);
    Tau_info.push_back(E_red_rank-fci_energy);

    outfile->Printf("\n////////////////// Full SVD /////////////////\n");
    outfile->Printf("\n Tau               = %20.12f", TAU);
    outfile->Printf("\n E_fci             = %20.12f", fci_energy);
    outfile->Printf("\n E_red_rank        = %20.12f", E_red_rank);
    outfile->Printf("\n Delta(E_red_rank) = %20.12f", E_red_rank-fci_energy);
    outfile->Printf("\n Npar frr          =     %6d", N_par*2); //this gives full number of parameters
    outfile->Printf("\n Nsto frr          =     %6d", N_sto*2);

    ////testing purturbation idea...////
    // std::vector<SharedMatrix> C_rr_p;
    // std::vector<SharedMatrix> C_sum;
    // //first SVD C_red_rank..
    // for(int h=0; h<nirrep; h++){
    //   int nrow = C_red_rank[h]->rowdim();
    //   int ncol = C_red_rank[h]->coldim();
    //
    //   auto u_p = std::make_shared<Matrix>("u_p", nrow, nrow);
    //   auto s_p = std::make_shared<Vector>("s_p", std::min(ncol, nrow));
    //   auto v_p = std::make_shared<Matrix>("v_p", ncol, ncol);
    //
    //   C_red_rank[h]->svd(u_p, s_p, v_p);
    //
    //   //now build the 'inverse' sigma matrix
    //   auto sig_mat = std::make_shared<Matrix>("sig_mat", nrow, ncol);
    //   std::cout << '\n' << "rank_ef " << rank_ef_inirrep[h] << std::endl;
    //   std::cout << '\n' << "total rank " << rank_irrep[h] << std::endl;
    //   for(int i = 60; i < 252; i++) {
    //     sig_mat->set(i, i, s_p->get(i));
    //   }
    //
    //   auto C_rr_p_h = Matrix::triplet(u_p, sig_mat, v_p, false, false, false);
    //   C_rr_p.push_back(C_rr_p_h);
    //
    //   //add prime matrix to C_rr
    //   auto C_sum_h = C_red_rank[h]->clone();
    //   C_sum_h->add(C_rr_p_h);
    //   C_sum.push_back(C_sum_h);
    //
    //   // auto C_diff = C[h]->clone();
    //   // C_diff->subtract(C_red_rank_h);
    //   // double norm = std::sqrt(C_diff->sum_of_squares());
    //
    // }
    // C_rr_p[0]->print();
    //
    // // re-normalize C_sum
    // std::cout << '\n' << "dim of C_sum " << C_sum.size() << std::endl;
    // double norm_C_sum = 0.0;
    // for (auto C_sum_h : C_sum) {
    //     norm_C_sum += C_sum_h->sum_of_squares();
    // }
    // norm_C_sum = std::sqrt(norm_C_sum);
    // for (auto C_sum_h : C_sum) {
    //     C_sum_h->scale(1. / norm_C_sum);
    // }
    //
    // //re-compute energy...
    // C_->set_coefficient_blocks(C_sum);
    // // HC = H C
    // C_->Hamiltonian(HC, fci_ints, twoSubstituitionVVOO);
    // // E = C^T HC
    // double E_pert = HC.dot(C_) + nuclear_repulsion_energy;
    //
    // outfile->Printf("\nI get here!");
    //
    // outfile->Printf("\n //////// HERE WE GO! ////////");
    // outfile->Printf("\n Tau               = %20.12f", TAU);
    // outfile->Printf("\n E_fci             = %20.12f", fci_energy);
    // outfile->Printf("\n E_pert            = %20.12f", E_pert);
    // outfile->Printf("\n Delta(E_red_rank) = %20.12f", E_pert-fci_energy);

}

} // namespace forte
} // namespace psi
