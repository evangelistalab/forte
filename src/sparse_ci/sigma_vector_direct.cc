/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/libciomr/libciomr.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "../forte-def.h"
#include "../iterative_solvers.h"
#include "sigma_vector_direct.h"

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

SigmaVectorDirect::SigmaVectorDirect(const DeterminantHashVec& space,
                                     std::shared_ptr<FCIIntegrals> fci_ints)
    : SigmaVector(space.size()), space_(space), fci_ints_(fci_ints) {

    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space.get_det(I);
        double EI = fci_ints_->energy(detI);
        diag_.push_back(EI);
    }
}

void SigmaVectorDirect::compute_sigma(SharedVector sigma, SharedVector b) {

    sigma->zero();
    compute_sigma_scalar(sigma, b);
    compute_sigma_aa(sigma, b);
    compute_sigma_bb(sigma, b);
    compute_sigma_aaaa(sigma, b);
    compute_sigma_abab(sigma, b);
    compute_sigma_bbbb(sigma, b);
}

void SigmaVectorDirect::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {}

void SigmaVectorDirect::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorDirect::compute_sigma_scalar(SharedVector sigma, SharedVector b) {
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    STLBitsetDeterminant detJ;
    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space_.get_det(I);
        double b_I = b_p[I];
        sigma_p[I] += diag_[I] * b_I;
    }
}

void SigmaVectorDirect::compute_sigma_aa(SharedVector sigma, SharedVector b) {
    int nmo = fci_ints_->nmo();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    STLBitsetDeterminant detJ;
    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space_.get_det(I);
        double b_I = b_p[I];
        for (int i = 0; i < nmo; i++) {
            if (detI.get_alfa_bit(i)) {
                for (int a = 0; a < nmo; a++) {
                    if (not detI.get_alfa_bit(a)) {
                        // this is a valid i -> a excitation
                        detJ = detI;
                        detJ.set_alfa_bit(i,false);
                        detJ.set_alfa_bit(a,true);
                        size_t addJ = space_.get_idx(detJ);
                        if (addJ < size_) {
                            double h_ia = fci_ints_->slater_rules_single_alpha(detI, i, a);
                            sigma_p[addJ] += h_ia * b_I;
                        }
                    }
                }
            }
        }
    }
}

void SigmaVectorDirect::compute_sigma_bb(SharedVector sigma, SharedVector b) {
    int nmo = fci_ints_->nmo();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    STLBitsetDeterminant detJ;
    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space_.get_det(I);
        double b_I = b_p[I];
        for (int i = 0; i < nmo; i++) {
            if (detI.get_beta_bit(i)) {
                for (int a = 0; a < nmo; a++) {
                    if (not detI.get_beta_bit(a)) {
                        // this is a valid i -> a excitation
                        detJ = detI;
                        detJ.set_beta_bit(i,false);
                        detJ.set_beta_bit(a,true);
                        size_t addJ = space_.get_idx(detJ);
                        if (addJ < size_) {
                            double h_ia = fci_ints_->slater_rules_single_beta(detI, i, a);
                            sigma_p[addJ] += h_ia * b_I;
                        }
                    }
                }
            }
        }
    }
}

void SigmaVectorDirect::compute_sigma_aaaa(SharedVector sigma, SharedVector b) {
    int nmo = fci_ints_->nmo();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    STLBitsetDeterminant detJ;
    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space_.get_det(I);
        double b_I = b_p[I];
        for (int i = 0; i < nmo; i++) {
            if (detI.get_alfa_bit(i)) {
                for (int j = i + 1; j < nmo; j++) {
                    if (detI.get_alfa_bit(j)) {
                        for (int a = 0; a < nmo; a++) {
                            if (not detI.get_alfa_bit(a)) {
                                for (int b = a + 1; b < nmo; b++) {
                                    if (not detI.get_alfa_bit(b)) {
                                        // this is a valid ij -> ab excitation
                                        detJ = detI;
                                        detJ.set_alfa_bit(i,false);
                                        detJ.set_alfa_bit(j,false);
                                        detJ.set_alfa_bit(a,true);
                                        detJ.set_alfa_bit(b,true);
                                        size_t addJ = space_.get_idx(detJ);
                                        if (addJ < size_) {
                                            double h_ijab = fci_ints_->tei_aa(i, j, a, b) *
                                                            detI.slater_sign(i, j, a, b);
                                            sigma_p[addJ] += h_ijab * b_I;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void SigmaVectorDirect::compute_sigma_abab(SharedVector sigma, SharedVector b) {
    int nmo = fci_ints_->nmo();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    STLBitsetDeterminant detJ;
    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space_.get_det(I);
        double b_I = b_p[I];
        for (int i = 0; i < nmo; i++) {
            if (detI.get_alfa_bit(i)) {
                for (int j = 0; j < nmo; j++) {
                    if (detI.get_beta_bit(j)) {
                        for (int a = 0; a < nmo; a++) {
                            if (not detI.get_alfa_bit(a)) {
                                for (int b = 0; b < nmo; b++) {
                                    if (not detI.get_beta_bit(b)) {
                                        // this is a valid ij -> ab excitation
                                        detJ = detI;
                                        detJ.set_alfa_bit(i,false);
                                        detJ.set_beta_bit(j,false);
                                        detJ.set_alfa_bit(a,true);
                                        detJ.set_beta_bit(b,true);
                                        size_t addJ = space_.get_idx(detJ);
                                        if (addJ < size_) {
                                            double h_ijab =
                                                fci_ints_->tei_ab(i, j, a, b) *
                                                detI.slater_sign(i, j + nmo, a, b + nmo);
                                            sigma_p[addJ] += h_ijab * b_I;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void SigmaVectorDirect::compute_sigma_bbbb(SharedVector sigma, SharedVector b) {
    int nmo = fci_ints_->nmo();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    STLBitsetDeterminant detJ;
    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space_.get_det(I);
        double b_I = b_p[I];
        for (int i = 0; i < nmo; i++) {
            if (detI.get_beta_bit(i)) {
                for (int j = i + 1; j < nmo; j++) {
                    if (detI.get_beta_bit(j)) {
                        for (int a = 0; a < nmo; a++) {
                            if (not detI.get_beta_bit(a)) {
                                for (int b = a + 1; b < nmo; b++) {
                                    if (not detI.get_beta_bit(b)) {
                                        // this is a valid ij -> ab excitation
                                        detJ = detI;
                                        detJ.set_beta_bit(i,false);
                                        detJ.set_beta_bit(j,false);
                                        detJ.set_beta_bit(a,true);
                                        detJ.set_beta_bit(b,true);
                                        size_t addJ = space_.get_idx(detJ);
                                        if (addJ < size_) {
                                            double h_ijab = fci_ints_->tei_bb(i, j, a, b) *
                                                            detI.slater_sign(i + nmo, j + nmo,
                                                                             a + nmo, b + nmo);
                                            sigma_p[addJ] += h_ijab * b_I;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
}
}
