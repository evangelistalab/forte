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

size_t count_aa_total = 0;
size_t count_bb_total = 0;
size_t count_aaaa_total = 0;
size_t count_abab_total = 0;
size_t count_bbbb_total = 0;

size_t count_aa = 0;
size_t count_bb = 0;
size_t count_aaaa = 0;
size_t count_abab = 0;
size_t count_bbbb = 0;

void print_SigmaVectorDirect_stats();

#define SIGMA_VEC_DEBUG 1

SigmaVectorDirect::SigmaVectorDirect(const DeterminantHashVec& space,
                                     std::shared_ptr<FCIIntegrals> fci_ints)
    : SigmaVector(space.size()), space_(space), fci_ints_(fci_ints),
      a_sorted_string_list_(space, fci_ints,false), b_sorted_string_list_(space, fci_ints,true) {

    nmo_ = fci_ints_->nmo();

    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space.get_det(I);
        double EI = fci_ints_->energy(detI);
        diag_.push_back(EI);
    }
}

SigmaVectorDirect::~SigmaVectorDirect() { print_SigmaVectorDirect_stats(); }

void SigmaVectorDirect::compute_sigma(SharedVector sigma, SharedVector b) {

    sigma->zero();
    compute_sigma_scalar(sigma, b);
    compute_sigma_aa_fast_search(sigma, b);
    compute_sigma_bb(sigma, b);
    compute_sigma_aaaa(sigma, b);
    compute_sigma_abab(sigma, b);
    compute_sigma_bbbb(sigma, b);
}

void print_SigmaVectorDirect_stats() {
#if SIGMA_VEC_DEBUG
    outfile->Printf("\n");
    outfile->Printf("\n    aa   : %12zu / %12zu = %f", count_aa, count_aa_total,
                    double(count_aa) / double(count_aa_total));
    outfile->Printf("\n    bb   : %12zu / %12zu = %f", count_bb, count_bb_total,
                    double(count_bb) / double(count_bb_total));
    outfile->Printf("\n    aaaa : %12zu / %12zu = %f", count_aaaa, count_aaaa_total,
                    double(count_aaaa) / double(count_aaaa_total));
    outfile->Printf("\n    abab : %12zu / %12zu = %f", count_abab, count_abab_total,
                    double(count_abab) / double(count_abab_total));
    outfile->Printf("\n    bbbb : %12zu / %12zu = %f", count_bbbb, count_bbbb_total,
                    double(count_bbbb) / double(count_bbbb_total));
    outfile->Printf("\n");
#endif
}

void SigmaVectorDirect::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {}

void SigmaVectorDirect::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorDirect::compute_sigma_scalar(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDirect:scalar");

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
    timer energy_timer("SigmaVectorDirect:sigma_aa");

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
                        detJ.set_alfa_bit(i, false);
                        detJ.set_alfa_bit(a, true);
#if SIGMA_VEC_DEBUG
                        count_aa_total++;
#endif
                        size_t addJ = space_.get_idx(detJ);
                        if (addJ < size_) {
                            double h_ia = fci_ints_->slater_rules_single_alpha(detI, i, a);
                            sigma_p[addJ] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
                            count_aa++;
#endif
                        }
                    }
                }
            }
        }
    }
}

void SigmaVectorDirect::compute_sigma_bb(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDirect:sigma_bb");

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
                        detJ.set_beta_bit(i, false);
                        detJ.set_beta_bit(a, true);
#if SIGMA_VEC_DEBUG
                        count_bb_total++;
#endif
                        size_t addJ = space_.get_idx(detJ);
                        if (addJ < size_) {
                            double h_ia = fci_ints_->slater_rules_single_beta(detI, i, a);
                            sigma_p[addJ] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
                            count_bb++;
#endif
                        }
                    }
                }
            }
        }
    }
}

void SigmaVectorDirect::compute_sigma_aaaa(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDirect:sigma_aaaa");

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
                                        detJ.set_alfa_bit(i, false);
                                        detJ.set_alfa_bit(j, false);
                                        detJ.set_alfa_bit(a, true);
                                        detJ.set_alfa_bit(b, true);
#if SIGMA_VEC_DEBUG
                                        count_aaaa_total++;
#endif
                                        size_t addJ = space_.get_idx(detJ);
                                        if (addJ < size_) {
                                            double h_ijab = fci_ints_->tei_aa(i, j, a, b) *
                                                            detI.slater_sign(i, j, a, b);
                                            sigma_p[addJ] += h_ijab * b_I;
#if SIGMA_VEC_DEBUG
                                            count_aaaa++;
#endif
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
    timer energy_timer("SigmaVectorDirect:sigma_abab");

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
                                        detJ.set_alfa_bit(i, false);
                                        detJ.set_beta_bit(j, false);
                                        detJ.set_alfa_bit(a, true);
                                        detJ.set_beta_bit(b, true);
#if SIGMA_VEC_DEBUG
                                        count_abab_total++;
#endif
                                        size_t addJ = space_.get_idx(detJ);
                                        if (addJ < size_) {
                                            double h_ijab =
                                                fci_ints_->tei_ab(i, j, a, b) *
                                                detI.slater_sign(i, j + nmo, a, b + nmo);
                                            sigma_p[addJ] += h_ijab * b_I;
#if SIGMA_VEC_DEBUG
                                            count_abab++;
#endif
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
    timer energy_timer("SigmaVectorDirect:sigma_bbbb");

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
                                        detJ.set_beta_bit(i, false);
                                        detJ.set_beta_bit(j, false);
                                        detJ.set_beta_bit(a, true);
                                        detJ.set_beta_bit(b, true);
#if SIGMA_VEC_DEBUG
                                        count_bbbb_total++;
#endif
                                        size_t addJ = space_.get_idx(detJ);
                                        if (addJ < size_) {
                                            double h_ijab = fci_ints_->tei_bb(i, j, a, b) *
                                                            detI.slater_sign(i + nmo, j + nmo,
                                                                             a + nmo, b + nmo);
                                            sigma_p[addJ] += h_ijab * b_I;
#if SIGMA_VEC_DEBUG
                                            count_bbbb++;
#endif
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

void SigmaVectorDirect::compute_sigma_aa_fast_search(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDirect:sigma_aa");
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    for (size_t I = 0; I < size_; ++I) {
        STLBitsetDeterminant detI = space_.get_det(I);
        double b_I = b_p[I];
        compute_aa_coupling(detI, b_I, sigma_p);
    }
}

void SigmaVectorDirect::compute_aa_coupling(const STLBitsetDeterminant& detI, const double b_I,
                                            double* sigma_p) {
    STLBitsetDeterminant detJ;
    // Case I
    for (int i = nmo_ - 1; i >= 0; i--) {
        if (detI.get_alfa_bit(i)) {
            for (int a = 0; a < i; a++) {
                if (not detI.get_alfa_bit(a)) {
                    // this is a valid i -> a excitation
                    detJ = detI;
                    detJ.set_alfa_bit(i, false);
                    detJ.set_alfa_bit(a, true);
#if SIGMA_VEC_DEBUG
                    count_aa_total++;
#endif
                    size_t addJ = space_.get_idx(detJ);
                    if (addJ < size_) {
                        double h_ia = fci_ints_->slater_rules_single_alpha(detI, i, a);
                        sigma_p[addJ] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
                        count_aa++;
#endif
                    }
                }
            }
        }
    }

    // Case II
    for (int a = 0; a < nmo_; a++) {
        if (not detI.get_alfa_bit(a)) {
            for (int i = a - 1; i >= 0; i--) {
                if (detI.get_alfa_bit(i)) {
                    // this is a valid i -> a excitation
                    detJ = detI;
                    detJ.set_alfa_bit(i, false);
                    detJ.set_alfa_bit(a, true);
#if SIGMA_VEC_DEBUG
                    count_aa_total++;
#endif
                    size_t addJ = space_.get_idx(detJ);
                    if (addJ < size_) {
                        double h_ia = fci_ints_->slater_rules_single_alpha(detI, i, a);
                        sigma_p[addJ] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
                        count_aa++;
#endif
                    }
                }
            }
        }
    }
}
}
}
