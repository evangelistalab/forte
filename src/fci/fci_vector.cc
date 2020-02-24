/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"

#include "base_classes/mo_space_info.h"
#include "fci_vector.h"
#include "string_lists.h"

using namespace psi;

namespace forte {

psi::SharedMatrix FCIVector::C1;
psi::SharedMatrix FCIVector::Y1;
size_t FCIVector::sizeC1 = 0;
// FCIVector* FCIVector::tmp_wfn1 = nullptr;
// FCIVector* FCIVector::tmp_wfn2 = nullptr;

double FCIVector::hdiag_timer = 0.0;
double FCIVector::h1_aa_timer = 0.0;
double FCIVector::h1_bb_timer = 0.0;
double FCIVector::h2_aaaa_timer = 0.0;
double FCIVector::h2_aabb_timer = 0.0;
double FCIVector::h2_bbbb_timer = 0.0;

void FCIVector::allocate_temp_space(std::shared_ptr<StringLists> lists_, int print_) {
    // TODO Avoid allocating and deallocating these temp

    size_t nirreps = lists_->nirrep();
    size_t maxC1 = 0;
    for (size_t Ia_sym = 0; Ia_sym < nirreps; ++Ia_sym) {
        maxC1 = std::max(maxC1, lists_->alfa_graph()->strpi(Ia_sym));
    }
    for (size_t Ib_sym = 0; Ib_sym < nirreps; ++Ib_sym) {
        maxC1 = std::max(maxC1, lists_->beta_graph()->strpi(Ib_sym));
    }

    // Allocate the temporary arrays C1 and Y1 with the largest sizes
    C1 = std::make_shared<psi::Matrix>("C1", maxC1, maxC1);
    Y1 = std::make_shared<psi::Matrix>("Y1", maxC1, maxC1);

    if (print_)
        outfile->Printf("\n  Allocating memory for the Hamiltonian algorithm. "
                        "Size: 2 x %zu x %zu.   Memory: %8.6f GB",
                        maxC1, maxC1, to_gb(2 * maxC1 * maxC1));

    sizeC1 = maxC1 * maxC1 * static_cast<size_t>(sizeof(double));
}

void FCIVector::release_temp_space() {}

FCIVector::FCIVector(std::shared_ptr<StringLists> lists, size_t symmetry)
    : symmetry_(symmetry), lists_(lists), alfa_graph_(lists_->alfa_graph()),
      beta_graph_(lists_->beta_graph()) {
    startup();
}

FCIVector::~FCIVector() { cleanup(); }

///**
// * Copy data from moinfo and allocate the memory
// */
void FCIVector::startup() {

    nirrep_ = lists_->nirrep();
    ncmo_ = lists_->ncmo();
    cmopi_ = lists_->cmopi();
    cmopi_offset_ = lists_->cmopi_offset();
    //    cmo_to_mo_ = lists_->cmo_to_mo();

    ndet_ = 0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t detpi = alfa_graph_->strpi(alfa_sym) * beta_graph_->strpi(beta_sym);
        ndet_ += detpi;
        detpi_.push_back(detpi);
    }

    // Allocate the symmetry blocks of the wave function
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        //    outfile->Printf("\n\n  Block %d: allocate %d *
        //    %d",alfa_sym,(int)alfa_graph_->strpi(alfa_sym),(int)beta_graph_->strpi(beta_sym));
        C_.push_back(psi::SharedMatrix(
            new psi::Matrix("C", alfa_graph_->strpi(alfa_sym), beta_graph_->strpi(beta_sym))));
    }
}

/**
 * Dellocate the memory
 */
void FCIVector::cleanup() {}

///**
// * Set the wave function to a single Slater determinant
// */
// void FCIVector::set_to(Determinant& det)
//{
//  zero();
//  DetAddress add = get_det_address(det);
//  coefficients[add.alfa_sym][add.alfa_string][add.beta_string] = 1.0;
//}

///**
// * Get the coefficient of a single Slater determinant
// */
// double FCIVector::get_coefficient(Determinant& det)
//{
//  DetAddress add = get_det_address(det);
//  return coefficients[add.alfa_sym][add.alfa_string][add.beta_string];
//}

/**
 * Set the wave function to another wave function
 */
void FCIVector::copy(FCIVector& wfn) {
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        C_[alfa_sym]->copy(wfn.C_[alfa_sym]);
    }
}

void FCIVector::copy(psi::SharedVector vec) {
    size_t I = 0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_graph_->strpi(alfa_sym);
        size_t maxIb = beta_graph_->strpi(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                C_ha[Ia][Ib] = vec->get(I);
                I += 1;
            }
        }
    }
}

void FCIVector::copy_to(psi::SharedVector vec) {
    size_t I = 0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_graph_->strpi(alfa_sym);
        size_t maxIb = beta_graph_->strpi(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                vec->set(I, C_ha[Ia][Ib]);
                I += 1;
            }
        }
    }
}

void FCIVector::set(std::vector<std::tuple<size_t, size_t, size_t, double>>& sparse_vec) {
    zero();
    double C;
    size_t h, Ia, Ib;
    for (auto& el : sparse_vec) {
        std::tie(h, Ia, Ib, C) = el;
        C_[h]->set(Ia, Ib, C);
    }
}

///**
// * Set the wave function to the nth determinant in the list
// */
// void FCIVector::set_to(int n)
//{
//  int k = 0;
//  for(int h = 0; h < nirrep_; ++h){
//    int beta_sym = h ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(h);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        if(k == n){
//          coefficients[h][Ia][Ib] = 1.0;
//        }else{
//          coefficients[h][Ia][Ib] = 0.0;
//        }
//        k++;
//      }
//    }
//  }
//}

///**
// * Get the coefficient of the nth determinant in the list
// */
// double FCIVector::get(int n)
//{
//  int k = 0;
//  double c = 0.0;
//  for(int h = 0; h < nirrep_; ++h){
//    int beta_sym = h ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(h);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        if(k == n){
//          c = coefficients[h][Ia][Ib];
//        }
//        k++;
//      }
//    }
//  }
//  return c;
//}

///**
// * Get a vector of the determinants with weight greather than alpha
// */
// std::vector<int> FCIVector::get_important(double alpha)
//{
//  int k = 0;
//  std::vector<int> list;
//  for(int h = 0; h < nirrep_; ++h){
//    int beta_sym = h ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(h);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        if(std::fabs(coefficients[h][Ia][Ib]) >= alpha){
//          list.push_back(k);
//        }
//        k++;
//      }
//    }
//  }
//  return list;
//}

/////**
//// * Get a vector of the determinants with weight greather than alpha
//// */
////vector<int> FCIVector::get_sorted_important()
////{
////  std::vector<pair<double,int> > list;
////  for(int h = 0; h < nirrep_; ++h){
////    int beta_sym = h ^ symmetry_;
////    size_t maxIa = alfa_graph_->strpi(h);
////    size_t maxIb = beta_graph_->strpi(beta_sym);
////    for(size_t Ia = 0; Ia < maxIa; ++Ia){
////      for(size_t Ib = 0; Ib < maxIb; ++Ib){
////        list.push_back(std::make_pair(std::fabs(coefficients[h][Ia][Ib]),k));
////        k++;
////      }
////    }
////  }
////  sort(list.begin(),list.end(),std::greater<pair<double,int> >());
////  return list;
////}

/**
 * Normalize the wave function without changing the phase
 */
void FCIVector::normalize() {
    double factor = norm(2.0);
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_graph_->strpi(alfa_sym);
        size_t maxIb = beta_graph_->strpi(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                C_ha[Ia][Ib] /= factor;
            }
        }
    }
}

///**
// * Normalize the wave function wrt to a single Slater determinant
// */
// void FCIVector::randomize()
//{
//  for(int h = 0; h < nirrep_; ++h){
//    int beta_sym = h ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(h);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        coefficients[h][Ia][Ib] += 0.001 * static_cast<double>(std::rand()) /
//        static_cast<double>(RAND_MAX);
//      }
//    }
//  }
//}

/**
 * Zero the wave function
 */
void FCIVector::zero() {
    for (psi::SharedMatrix C_h : C_) {
        C_h->zero();
    }
}

void FCIVector::print_natural_orbitals(std::shared_ptr<MOSpaceInfo> mo_space_info) {
    print_h2("NATURAL ORBITALS");
    psi::Dimension active_dim = mo_space_info->dimension("ACTIVE");

    size_t na = alfa_graph_->nones();
    size_t nb = beta_graph_->nones();

    auto opdm = std::make_shared<psi::Matrix>(new psi::Matrix("OPDM", active_dim, active_dim));

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < active_dim[h]; u++) {
            for (int v = 0; v < active_dim[h]; v++) {
                double gamma_uv = 0.0;
                if (na > 0) {
                    gamma_uv += opdm_a_[(u + offset) * ncmo_ + v + offset];
                }
                if (nb > 0) {
                    gamma_uv += opdm_b_[(u + offset) * ncmo_ + v + offset];
                }
                opdm->set(h, u, v, gamma_uv);
            }
        }
        offset += active_dim[h];
    }

    auto OCC = std::make_shared<psi::Vector>("Occupation numbers", active_dim);
    auto NO = std::make_shared<psi::Matrix>("MO -> NO transformation", active_dim, active_dim);

    opdm->diagonalize(NO, OCC, descending);
    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < active_dim[h]; u++) {
            auto irrep_occ = std::make_pair(OCC->get(h, u), std::make_pair(h, u + 1));
            vec_irrep_occupation.push_back(irrep_occ);
        }
    }
    CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
              std::greater<std::pair<double, std::pair<int, int>>>());

    size_t count = 0;
    outfile->Printf("\n    ");
    for (auto vec : vec_irrep_occupation) {
        outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second, ct.gamma(vec.second.first).symbol(),
                        vec.first);
        if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
            outfile->Printf("\n    ");
    }
    outfile->Printf("\n");
}

///**
// * Zero a symmetry block of the wave function
// * @param h symmetry of the alpha strings of the block to zero
// */
// void FCIVector::zero_block(int h)
//{
//  int beta_sym = h ^ symmetry_;
//  size_t size = alfa_graph_->strpi(h) * beta_graph_->strpi(beta_sym) *
//  static_cast<size_t> (sizeof(double));
//  if(size > 0)
//    memset(&(coefficients[h][0][0]), 0, size);
//}

///**
// * Transpose a block of the matrix (Works only for total symmetric wfns!)
// * @param h symmetry of the alpha strings of the block to zero
// */
// void FCIVector::transpose_block(int h)
//{
//  int beta_sym = h ^ symmetry_;
//  size_t maxIa = alfa_graph_->strpi(h);
//  size_t maxIb = beta_graph_->strpi(beta_sym);
//  size_t size = maxIa * maxIb * static_cast<size_t> (sizeof(double));
//  if(size > 0){
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = Ia + 1; Ib < maxIb; ++Ib){
//        double temp = coefficients[h][Ia][Ib];
//        coefficients[h][Ia][Ib] = coefficients[h][Ib][Ia];
//        coefficients[h][Ib][Ia] = temp;
//      }
//    }
//  }
//}

/**
 * Compute the 2-norm of the wave function
 */
double FCIVector::norm(double power) {
    double norm = 0.0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_graph_->strpi(alfa_sym);
        size_t maxIb = beta_graph_->strpi(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                norm += std::pow(std::fabs(C_ha[Ia][Ib]), power);
            }
        }
    }
    return std::pow(norm, 1.0 / power);
}

/**
 * Compute the dot product with another wave function
 */
double FCIVector::dot(FCIVector& wfn) {
    double dot = 0.0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        dot += C_[alfa_sym]->vector_dot(wfn.C_[alfa_sym]);
    }
    return (dot);

    //        int beta_sym = alfa_sym ^ symmetry_;
    //        size_t maxIa = alfa_graph_->strpi(alfa_sym);
    //        size_t maxIb = beta_graph_->strpi(beta_sym);

    //        double** Ca = C_[alfa_sym]->pointer();
    //        double** Cb = wfn.C_[alfa_sym]->pointer();
    //        for(size_t Ia = 0; Ia < maxIa; ++Ia){
    //            for(size_t Ib = 0; Ib < maxIb; ++Ib){
    //                dot += coefficients[alfa_sym][Ia][Ib] *
    //                wfn.coefficients[alfa_sym][Ia][Ib];
    //            }
    //        }
}
double FCIVector::dot(std::shared_ptr<FCIVector>& wfn) {
    double dot = 0.0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        dot += C_[alfa_sym]->vector_dot(wfn->C_[alfa_sym]);
    }
    return (dot);
}

///**
// * Find the largest element in the wave function
// */
// double FCIVector::max_element()
//{
//  double maxelement = 0.0;
//  for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(alfa_sym);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        if (std::abs(coefficients[alfa_sym][Ia][Ib]) > std::abs(maxelement)){
//          maxelement = coefficients[alfa_sym][Ia][Ib];
//        }
//      }
//    }
//  }
//  return maxelement;
//}

///**
// * Find the smallest element in the wave function
// */
// double FCIVector::min_element()
//{
//  double min_element = 0.0;
//  for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(alfa_sym);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        min_element = std::min(min_element,coefficients[alfa_sym][Ia][Ib]);
//      }
//    }
//  }
//  return min_element;
//}

///**
// * Implements the update method of Bendazzoli and Evangelisti modified
// */
// void FCIVector::two_update(double alpha,double E,FCIVector& H,FCIVector& R)
//{
//  for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(alfa_sym);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        double r = R.coefficients[alfa_sym][Ia][Ib];
//        double h = H.coefficients[alfa_sym][Ia][Ib];
//        double c =   coefficients[alfa_sym][Ia][Ib];
//        coefficients[alfa_sym][Ia][Ib] = - r / (h - E + alpha * c * c - 2.0 *
//        r * c);
//      }
//    }
//  }
//}

///**
// * Implements the update method of Bendazzoli and Evangelisti
// */
// void FCIVector::bendazzoli_update(double alpha,double E,FCIVector& H,FCIVector& R)
//{
//  for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(alfa_sym);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        double r = R.coefficients[alfa_sym][Ia][Ib];
//        double h = H.coefficients[alfa_sym][Ia][Ib];
//        double c =   coefficients[alfa_sym][Ia][Ib];
//        coefficients[alfa_sym][Ia][Ib] -= r / (h - E + alpha * c * c - 2.0 * r
//        * c);
//      }
//    }
//  }
//}

///**
// * Implements the update method of Davidson and Liu
// */
// void FCIVector::davidson_update(double E,FCIVector& H,FCIVector& R)
//{
//  for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(alfa_sym);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        double r = R.coefficients[alfa_sym][Ia][Ib];
//        double h = H.coefficients[alfa_sym][Ia][Ib];
//        if(std::fabs(h-E) > 1.0e-8){
//          coefficients[alfa_sym][Ia][Ib] = - r / (h - E);
//        }else {
//          coefficients[alfa_sym][Ia][Ib] = 0.0;
//          outfile->Printf("\n  WARNING: skipped determinant in davidson
//          update. h - E = %20.12f",h-E);
//        }
//      }
//    }
//  }
//}

///**
// * Add a scaled amount of another wave function
// */
// void FCIVector::plus_equal(double factor,FCIVector& wfn)
//{
//  for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(alfa_sym);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        coefficients[alfa_sym][Ia][Ib] += factor *
//        wfn.coefficients[alfa_sym][Ia][Ib];
//      }
//    }
//  }
//}

///**
// * Add a scaled amount of another wave function
// */
// void FCIVector::scale(double factor)
//{
//  for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    size_t maxIa = alfa_graph_->strpi(alfa_sym);
//    size_t maxIb = beta_graph_->strpi(beta_sym);
//    for(size_t Ia = 0; Ia < maxIa; ++Ia){
//      for(size_t Ib = 0; Ib < maxIb; ++Ib){
//        coefficients[alfa_sym][Ia][Ib] *= factor;
//      }
//    }
//  }
//}

/**
 * Print the non-zero contributions to the wave function
 */
void FCIVector::print() {
    // print the non-zero elements of the wave function
    size_t det = 0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < alfa_graph_->strpi(alfa_sym); ++Ia) {
            for (size_t Ib = 0; Ib < beta_graph_->strpi(beta_sym); ++Ib) {
                if (std::fabs(C_ha[Ia][Ib]) > 1.0e-9) {
                    outfile->Printf("\n  %15.9f [%1d][%2d][%2d] (%d)", C_ha[Ia][Ib], alfa_sym,
                                    static_cast<int>(Ia), static_cast<int>(Ib),
                                    static_cast<int>(det));
                }
                ++det;
            }
        }
    }
}

///**
// * Apply the same-spin two-particle Hamiltonian to the wave function
// * @param alfa flag for alfa or beta component, true = alfa, false = beta
// */
// void FCIVector::H2_aaaa2(FCIVector& result, bool alfa)
//{
//  for(int alfa_sym = 0; alfa_sym < nirreps; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    if(detpi[alfa_sym] > 0){
//      double** Y = alfa ? result.coefficients[alfa_sym] : Y1;
//      double** C = alfa ? coefficients[alfa_sym]        : C1;
//      if(!alfa){
//        memset(&(C[0][0]), 0, sizeC1);
//        memset(&(Y[0][0]), 0, sizeC1);
//        size_t maxIa = alfa_graph_->strpi(alfa_sym);
//        size_t maxIb = beta_graph_->strpi(beta_sym);
//        // Copy C transposed in C1
//        for(size_t Ia = 0; Ia < maxIa; ++Ia)
//          for(size_t Ib = 0; Ib < maxIb; ++Ib)
//            C[Ib][Ia] = coefficients[alfa_sym][Ia][Ib];
//      }

//      size_t maxL = alfa ? beta_graph_->strpi(beta_sym) :
//      alfa_graph_->strpi(alfa_sym);
//      // Loop over (p>q) == (p>q)
//      for(int pq_sym = 0; pq_sym < nirreps; ++pq_sym){
//        size_t max_pq = lists->get_pairpi(pq_sym);
//        for(size_t pq = 0; pq < max_pq; ++pq){
//          const Pair& pq_pair = lists->get_nn_list_pair(pq_sym,pq);
//          int p_abs = pq_pair.first;
//          int q_abs = pq_pair.second;
//          double integral = alfa ? tei_aaaa(p_abs,p_abs,q_abs,q_abs)
//                                 : tei_bbbb(p_abs,p_abs,q_abs,q_abs); // Grab
//                                 the integral
//          integral -= alfa ? tei_aaaa(p_abs,q_abs,q_abs,p_abs)
//                           : tei_bbbb(p_abs,q_abs,q_abs,p_abs); // Grab the
//                           integral

//          std::vector<StringSubstitution>& OO = alfa ?
//          lists->get_alfa_oo_list(pq_sym,pq,alfa_sym)
//                                                     :
//                                                     lists->get_beta_oo_list(pq_sym,pq,beta_sym);

//          size_t maxss = OO.size();
//          for(size_t ss = 0; ss < maxss; ++ss)
//            C_DAXPY(maxL,static_cast<double>(OO[ss].sign) * integral,
//            &C[OO[ss].I][0], 1, &Y[OO[ss].J][0], 1);
//        }
//      }
//      // Loop over (p>q) > (r>s)
//      for(int pq_sym = 0; pq_sym < nirreps; ++pq_sym){
//        size_t max_pq = lists->get_pairpi(pq_sym);
//        for(size_t pq = 0; pq < max_pq; ++pq){
//          const Pair& pq_pair = lists->get_nn_list_pair(pq_sym,pq);
//          int p_abs = pq_pair.first;
//          int q_abs = pq_pair.second;
//          for(size_t rs = 0; rs < pq; ++rs){
//              const Pair& rs_pair = lists->get_nn_list_pair(pq_sym,rs);
//              int r_abs = rs_pair.first;
//              int s_abs = rs_pair.second;
//              double integral = alfa ? tei_aaaa(p_abs,r_abs,q_abs,s_abs)
//                                     : tei_bbbb(p_abs,r_abs,q_abs,s_abs); //
//                                     Grab the integral
//              integral -= alfa ? tei_aaaa(p_abs,s_abs,q_abs,r_abs)
//                               : tei_bbbb(p_abs,s_abs,q_abs,r_abs); // Grab
//                               the integral

//              {
//                std::vector<StringSubstitution>& VVOO = alfa ?
//                lists->get_alfa_vvoo_list(p_abs,q_abs,r_abs,s_abs,alfa_sym)
//                                                             :
//                                                             lists->get_beta_vvoo_list(p_abs,q_abs,r_abs,s_abs,beta_sym);
//                // TODO loop in a differen way
//                size_t maxss = VVOO.size();
//                for(size_t ss = 0; ss < maxss; ++ss)
//                  C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral,
//                  &C[VVOO[ss].I][0], 1, &Y[VVOO[ss].J][0], 1);
//              }
//              {
//                std::vector<StringSubstitution>& VVOO = alfa ?
//                lists->get_alfa_vvoo_list(r_abs,s_abs,p_abs,q_abs,alfa_sym)
//                                                             :
//                                                             lists->get_beta_vvoo_list(r_abs,s_abs,p_abs,q_abs,beta_sym);
//                // TODO loop in a differen way
//                size_t maxss = VVOO.size();
//                for(size_t ss = 0; ss < maxss; ++ss)
//                  C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral,
//                  &C[VVOO[ss].I][0], 1, &Y[VVOO[ss].J][0], 1);
//              }
//          }
//        }
//      }
//      if(!alfa){
//        size_t maxIa = alfa_graph_->strpi(alfa_sym);
//        size_t maxIb = beta_graph_->strpi(beta_sym);
//        // Add Y1 transposed to Y
//        for(size_t Ia = 0; Ia < maxIa; ++Ia)
//          for(size_t Ib = 0; Ib < maxIb; ++Ib)
//            result.coefficients[alfa_sym][Ia][Ib] += Y[Ib][Ia];
//      }
//    }
//  } // End loop over h
//}
} // namespace forte
