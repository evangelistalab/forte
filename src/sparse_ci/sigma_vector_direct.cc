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
      a_sorted_string_list_(space, fci_ints, STLBitsetDeterminant::SpinType::AlphaSpin),
      b_sorted_string_list_(space, fci_ints, STLBitsetDeterminant::SpinType::BetaSpin),
      a_sorted_string_list_ui64_(space, fci_ints, STLBitsetDeterminant::SpinType::AlphaSpin) {

    nmo_ = fci_ints_->nmo();

    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space.get_det(I);
        double EI = fci_ints_->energy(detI);
        diag_.push_back(EI);
    }
    temp_sigma_.resize(size_);
}

SigmaVectorDirect::~SigmaVectorDirect() { print_SigmaVectorDirect_stats(); }

void SigmaVectorDirect::compute_sigma(SharedVector sigma, SharedVector b) {

    sigma->zero();
    compute_sigma_scalar(sigma, b);
    compute_sigma_aa_fast_search(sigma, b);
    compute_sigma_bb_fast_search(sigma, b);
//    compute_sigma_abab_fast_search_group(sigma, b);
    compute_sigma_abab_fast_search_group_ui64(sigma, b);
    //    compute_sigma_aaaa(sigma, b);
    //    compute_sigma_abab(sigma, b);
    //    compute_sigma_bbbb(sigma, b);
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
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
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
    STLBitsetDeterminant detJ = space_.get_det(0);
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
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
    STLBitsetDeterminant detJ = space_.get_det(0);
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
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
    STLBitsetDeterminant detJ = space_.get_det(0);
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
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
                                                            detI.slater_sign_aaaa(i, j, a, b);
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
    STLBitsetDeterminant detJ = space_.get_det(0);
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
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
                                            double h_ijab = fci_ints_->tei_ab(i, j, a, b) *
                                                            detI.slater_sign_aa(i, a) *
                                                            detI.slater_sign_bb(j, b);
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
    STLBitsetDeterminant detJ = space_.get_det(0);
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
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
                                                            detI.slater_sign_bbbb(i, j, a, b);
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

// void SigmaVectorDirect::compute_sigma_aa_fast_search(SharedVector sigma, SharedVector b) {
//    timer energy_timer("SigmaVectorDirect:sigma_aa");
//    double* sigma_p = sigma->pointer();
//    double* b_p = b->pointer();
//    // loop over all determinants
//    for (size_t I = 0; I < size_; ++I) {
//        const STLBitsetDeterminant& detI = space_.get_det(I);
//        double b_I = b_p[I];
//        compute_aa_coupling(detI, b_I, sigma_p);
//    }
//}

void SigmaVectorDirect::compute_sigma_aa_fast_search(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDirect:sigma_aa");
    for (size_t I = 0; I < size_; ++I)
        temp_sigma_[I] = 0.0;
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
        double b_I = b_p[I];
        compute_aa_coupling_compare(detI, b_I);
    }
    // Add sigma using the determinant address used in the DeterminantHashVector object
    const auto& sorted_dets = b_sorted_string_list_.sorted_dets();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = space_.get_idx(sorted_dets[I]);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDirect::compute_sigma_bb_fast_search(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDirect:sigma_bb");
    for (size_t I = 0; I < size_; ++I)
        temp_sigma_[I] = 0.0;
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
        double b_I = b_p[I];
        compute_bb_coupling_compare(detI, b_I);
    }
    // Add sigma using the determinant address used in the DeterminantHashVector object
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = space_.get_idx(sorted_dets[I]);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDirect::compute_sigma_abab_fast_search(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDirect:sigma_abab");
    for (size_t I = 0; I < size_; ++I)
        temp_sigma_[I] = 0.0;
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();

    STLBitsetDeterminant detJ = space_.get_det(0);
    // loop over all determinants
    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space_.get_det(I);
        double b_I = b_p[I];

        // find all singles compared to this determinant
        const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
        for (const auto& d : sorted_half_dets) {
            // find common bits
            detJ = d;
            detJ ^= detI;
            int ndiff = detJ.count_alfa();
            if (ndiff == 2) {
                int i, a;
                for (int p = 0; p < nmo_; ++p) {
                    const bool la_p = detI.get_alfa_bit(p);
                    const bool ra_p = d.get_alfa_bit(p);
                    if (la_p ^ ra_p) {
                        i = la_p ? p : i;
                        a = ra_p ? p : a;
                    }
                }
                double sign = detI.slater_sign_aa(i, a);
                compute_bb_coupling_compare_singles(detI, d, b_I, sign, i, a);
            }
        }
    }
    // Add sigma using the determinant address used in the DeterminantHashVector object
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = space_.get_idx(sorted_dets[I]);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDirect::compute_sigma_abab_fast_search_group(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDirect:sigma_abab");
    for (size_t I = 0; I < size_; ++I)
        temp_sigma_[I] = 0.0;
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();

    STLBitsetDeterminant detIJa_common = space_.get_det(0);
    // loop over all determinants
    const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    for (const auto& detIa : sorted_half_dets) {
        for (const auto& detJa : sorted_half_dets) {
            detIJa_common = detIa ^ detJa;
            int ndiff = detIJa_common.count_alfa();
            if (ndiff == 2) {
                int i, a;
                for (int p = 0; p < nmo_; ++p) {
                    const bool la_p = detIa.get_alfa_bit(p);
                    const bool ra_p = detJa.get_alfa_bit(p);
                    if (la_p ^ ra_p) {
                        i = la_p ? p : i;
                        a = ra_p ? p : a;
                    }
                }
                double sign = detIa.slater_sign_aa(i, a);
                compute_bb_coupling_compare_singles_group(detIa, detJa, sign, i, a, b);
            }
        }
    }

    //    for (size_t I = 0; I < size_; ++I) {
    //        const STLBitsetDeterminant& detI = space_.get_det(I);
    //        double b_I = b_p[I];

    //        // find all singles compared to this determinant
    //        const auto& sorted_half_dets = a_sorted_string_list_.sorted_half_dets();
    //        for (const auto& d : sorted_half_dets) {
    //            // find common bits
    //            detJ = d;
    //            detJ ^= detI;
    //            int ndiff = detJ.count_alfa();
    //            if (ndiff == 2) {
    //                int i, a;
    //                for (int p = 0; p < nmo_; ++p) {
    //                    const bool la_p = detI.get_alfa_bit(p);
    //                    const bool ra_p = d.get_alfa_bit(p);
    //                    if (la_p ^ ra_p) {
    //                        i = la_p ? p : i;
    //                        a = ra_p ? p : a;
    //                    }
    //                }
    //                double sign = detI.slater_sign_aa(i, a);
    //                compute_bb_coupling_compare_singles(detI, d, b_I, sign, i, a);
    //            }
    //        }
    //    }
    // Add sigma using the determinant address used in the DeterminantHashVector object
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = space_.get_idx(sorted_dets[I]);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDirect::compute_sigma_abab_fast_search_group_ui64(SharedVector sigma,
                                                                  SharedVector b) {
    timer energy_timer("SigmaVectorDirect:sigma_abab");
    for (size_t I = 0; I < size_; ++I)
        temp_sigma_[I] = 0.0;
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();

    UI64Determinant::bit_t detIJa_common;
    // loop over all determinants
    const auto& sorted_half_dets = a_sorted_string_list_ui64_.sorted_half_dets();
    for (const auto& detIa : sorted_half_dets) {
        for (const auto& detJa : sorted_half_dets) {
            detIJa_common = detIa ^ detJa;
            int ndiff = ui64_bit_count(detIJa_common);
            if (ndiff == 2) {
                int i, a;
                for (int p = 0; p < nmo_; ++p) {
                    const bool la_p = ui64_get_bit(detIa, p);
                    const bool ra_p = ui64_get_bit(detJa, p);
                    if (la_p ^ ra_p) {
                        i = la_p ? p : i;
                        a = ra_p ? p : a;
                    }
                }
                double sign = ui64_slater_sign(detIa, i, a);
                compute_bb_coupling_compare_singles_group_ui64(detIa, detJa, sign, i, a, b);
            }
        }
    }
    // Add sigma using the determinant address used in the DeterminantHashVector object
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = space_.get_idx(sorted_dets[I]);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDirect::compute_aa_coupling(const STLBitsetDeterminant& detI, const double b_I,
                                            double* sigma_p) {
    STLBitsetDeterminant detJ = space_.get_det(0);
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

void SigmaVectorDirect::compute_bb_coupling(const STLBitsetDeterminant& detI, const double b_I) {
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range = a_sorted_string_list_.range(detI);
    STLBitsetDeterminant detJ = space_.get_det(0);
    size_t first = range.first;
    size_t last = range.second;
    // Case I
    for (int i = nmo_ - 1; i >= 0; i--) {
        if (detI.get_beta_bit(i)) {
            for (int a = 0; a < i; a++) {
                if ((not detI.get_beta_bit(a)) and (first < last)) {
                    // this is a valid i -> a excitation
                    detJ = detI;
                    detJ.set_beta_bit(i, false);
                    detJ.set_beta_bit(a, true);
#if SIGMA_VEC_DEBUG
                    count_bb_total++;
#endif
                    size_t addJ = a_sorted_string_list_.find(detJ, first, last);

                    if (addJ < size_) {
                        double h_ia = fci_ints_->slater_rules_single_beta(detI, i, a);
                        temp_sigma_[addJ] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
                        count_bb++;
#endif
                    }
                }
            }
        }
    }
    // Case II
    for (int a = 0; a < nmo_; a++) {
        if (not detI.get_beta_bit(a)) {
            for (int i = a - 1; i >= 0; i--) {
                if ((detI.get_beta_bit(i)) and (first < last)) {
                    // this is a valid i -> a excitation
                    detJ = detI;
                    detJ.set_beta_bit(i, false);
                    detJ.set_beta_bit(a, true);
#if SIGMA_VEC_DEBUG
                    count_bb_total++;
#endif
                    size_t addJ = a_sorted_string_list_.find(detJ, first, last);

                    if (addJ < size_) {
                        double h_ia = fci_ints_->slater_rules_single_beta(detI, i, a);
                        temp_sigma_[addJ] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
                        count_bb++;
#endif
                    }
                }
            }
        }
    }
}

void SigmaVectorDirect::compute_aa_coupling_compare(const STLBitsetDeterminant& detI,
                                                    const double b_I) {
    const auto& sorted_dets = b_sorted_string_list_.sorted_dets();
    const auto& range = b_sorted_string_list_.range(detI);
    STLBitsetDeterminant detJ = space_.get_det(0);
    size_t first = range.first;
    size_t last = range.second;
    for (size_t pos = first; pos < last; ++pos) {
        detJ = sorted_dets[pos];

#if SIGMA_VEC_DEBUG
        count_aa_total++;
#endif
        // find common bits
        detJ ^= detI;
        int ndiff = detJ.count_alfa();
        if (ndiff == 2) {
            double h_ia = fci_ints_->slater_rules_single_alpha(detI, sorted_dets[pos]);
            temp_sigma_[pos] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
            count_aa++;
#endif
        }
        if (ndiff == 4) {
            double h_ia = fci_ints_->slater_rules_double_alpha_alpha(detI, sorted_dets[pos]);
            temp_sigma_[pos] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
            count_aaaa++;
#endif
        }
    }
}

void SigmaVectorDirect::compute_bb_coupling_compare(const STLBitsetDeterminant& detI,
                                                    const double b_I) {
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range = a_sorted_string_list_.range(detI);
    STLBitsetDeterminant detJ = space_.get_det(0);
    size_t first = range.first;
    size_t last = range.second;
    for (size_t pos = first; pos < last; ++pos) {
        detJ = sorted_dets[pos];

#if SIGMA_VEC_DEBUG
        count_bb_total++;
#endif
        // find common bits
        detJ ^= detI;
        int ndiff = detJ.count_beta();
        if (ndiff == 2) {
            double h_ia = fci_ints_->slater_rules_single_beta(detI, sorted_dets[pos]);
            temp_sigma_[pos] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
            count_bb++;
#endif
        }
        if (ndiff == 4) {
            double h_ia = fci_ints_->slater_rules_double_beta_beta(detI, sorted_dets[pos]);
            temp_sigma_[pos] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
            count_bbbb++;
#endif
        }
    }
}

void SigmaVectorDirect::compute_bb_coupling_compare_singles(const STLBitsetDeterminant& detI,
                                                            const STLBitsetDeterminant& detI_ia,
                                                            const double b_I, double sign, int i,
                                                            int a) {
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range = a_sorted_string_list_.range(detI_ia);
    STLBitsetDeterminant detJ = detI;
    size_t first = range.first;
    size_t last = range.second;
    for (size_t pos = first; pos < last; ++pos) {
        detJ = sorted_dets[pos];
#if SIGMA_VEC_DEBUG
        count_abab_total++;
#endif
        // find common bits
        detJ ^= detI;
        int ndiff = detJ.count_beta();
        if (ndiff == 2) {
            double h_ia =
                sign * fci_ints_->slater_rules_double_alpha_beta_pre(detI, sorted_dets[pos], i, a);
            temp_sigma_[pos] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
            count_abab++;
#endif
        }
    }
}

void SigmaVectorDirect::compute_bb_coupling_compare_singles_group(const STLBitsetDeterminant& detIa,
                                                                  const STLBitsetDeterminant& detJa,
                                                                  double sign, int i, int a,
                                                                  const SharedVector& b) {
    const auto& sorted_dets = a_sorted_string_list_.sorted_dets();
    const auto& range_I = a_sorted_string_list_.range(detIa);
    const auto& range_J = a_sorted_string_list_.range(detJa);
    STLBitsetDeterminant detI = detIa;
    STLBitsetDeterminant detJ = detJa;
    STLBitsetDeterminant detIJ = detJa;
    size_t first_I = range_I.first;
    size_t last_I = range_I.second;
    size_t first_J = range_J.first;
    size_t last_J = range_J.second;
    for (size_t posI = first_I; posI < last_I; ++posI) {
        detI = sorted_dets[posI];
        double b_I = b->get(space_.get_idx(detI));
        for (size_t posJ = first_J; posJ < last_J; ++posJ) {
            detJ = sorted_dets[posJ];
#if SIGMA_VEC_DEBUG
            count_abab_total++;
#endif
            // find common bits
            detIJ = detJ ^ detI;
            int ndiff = detIJ.count_beta();
            if (ndiff == 2) {
                double h_ia =
                    sign * fci_ints_->slater_rules_double_alpha_beta_pre(detI, detJ, i, a);
                temp_sigma_[posJ] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
                count_abab++;
#endif
            }
        }
    }
}

void SigmaVectorDirect::compute_bb_coupling_compare_singles_group_ui64(
    const UI64Determinant::bit_t& detIa, const UI64Determinant::bit_t& detJa, double sign, int i,
    int a, const SharedVector& b) {
    const auto& sorted_dets = a_sorted_string_list_ui64_.sorted_dets();
    const auto& range_I = a_sorted_string_list_ui64_.range(detIa);
    const auto& range_J = a_sorted_string_list_ui64_.range(detJa);
    UI64Determinant::bit_t detIb;
    UI64Determinant::bit_t detJb;
    UI64Determinant::bit_t detIJb;
    size_t first_I = range_I.first;
    size_t last_I = range_I.second;
    size_t first_J = range_J.first;
    size_t last_J = range_J.second;
    for (size_t posI = first_I; posI < last_I; ++posI) {
        detIb = sorted_dets[posI].get_beta_bits();
        size_t addI = a_sorted_string_list_ui64_.add(posI);
        double b_I = b->get(addI);
        for (size_t posJ = first_J; posJ < last_J; ++posJ) {
            detJb = sorted_dets[posJ].get_beta_bits();
#if SIGMA_VEC_DEBUG
            count_abab_total++;
#endif
            // find common bits
            detIJb = detJb ^ detIb;
            int ndiff = ui64_bit_count(detIJb);
            if (ndiff == 2) {
                auto sign_j_b = ui64_slater_sign_single(detIb, detJb);
                double sign_b = std::get<0>(sign_j_b);
                size_t j = std::get<1>(sign_j_b);
                size_t b = std::get<2>(sign_j_b);
                double h_ia = sign * sign_b * fci_ints_->tei_ab(i, j, a, b);
                temp_sigma_[posJ] += h_ia * b_I;
#if SIGMA_VEC_DEBUG
                count_abab++;
#endif
            }
        }
    }
}

// void find_next_bb_excitation()
// for (int i = nmo_ - 1; i >= 0; i--) {
//    if (detI.get_beta_bit(i)) {
//        for (int a = 0; a < i; a++) {
//            if (not detI.get_beta_bit(a)) {
//                // this is a valid i -> a excitation
//                detJ = detI;
//                detJ.set_beta_bit(i, false);
//                detJ.set_beta_bit(a, true);
//#if SIGMA_VEC_DEBUG
//                count_bb_total++;
//#endif
//                size_t addJ = space_.get_idx(detJ);
//                if (addJ < size_) {
//                    double h_ia = fci_ints_->slater_rules_single_beta(detI, i, a);
//                    temp_sigma_[addJ] += h_ia * b_I;
//#if SIGMA_VEC_DEBUG
//                    count_bb++;
//#endif
//                }
//            }
//        }
//    }
//}
}
}
