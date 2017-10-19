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
#include "sigma_vector_dynamic.h"

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

extern size_t count_aa_total;
extern size_t count_bb_total;
extern size_t count_aaaa_total;
extern size_t count_abab_total;
extern size_t count_bbbb_total;

extern size_t count_aa;
extern size_t count_bb;
extern size_t count_aaaa;
extern size_t count_abab;
extern size_t count_bbbb;

void print_SigmaVectorDynamic_stats();

#define SIGMA_VEC_DEBUG 1

SigmaVectorDynamic::SigmaVectorDynamic(const DeterminantHashVec& space,
                                       std::shared_ptr<FCIIntegrals> fci_ints)
    : SigmaVector(space.size()), space_(space), fci_ints_(fci_ints),
      a_sorted_string_list_ui64_(space, fci_ints, STLBitsetDeterminant::SpinType::AlphaSpin),
      b_sorted_string_list_ui64_(space, fci_ints, STLBitsetDeterminant::SpinType::BetaSpin) {

    nmo_ = fci_ints_->nmo();

    for (size_t I = 0; I < size_; ++I) {
        const STLBitsetDeterminant& detI = space.get_det(I);
        double EI = fci_ints_->energy(detI);
        diag_.push_back(EI);
    }
    temp_sigma_.resize(size_);
    temp_b_.resize(size_);
}

SigmaVectorDynamic::~SigmaVectorDynamic() { print_SigmaVectorDynamic_stats(); }

void SigmaVectorDynamic::compute_sigma(SharedVector sigma, SharedVector b) {
    sigma->zero();
    compute_sigma_scalar(sigma, b);
    compute_sigma_aa_fast_search_group_ui64(sigma, b);
    compute_sigma_bb_fast_search_group_ui64(sigma, b);
    compute_sigma_abab_fast_search_group_ui64(sigma, b);
}

void print_SigmaVectorDynamic_stats() {
#if SIGMA_VEC_DEBUG
    outfile->Printf("\n  Summary of SigmaVectorDynamic:");
    outfile->Printf("\n    aa   : %12zu / %12zu = %f", count_aa, count_aa_total,
                    double(count_aa) / double(count_aa_total));
    outfile->Printf("\n    bb   : %12zu / %12zu = %f", count_bb, count_bb_total,
                    double(count_bb) / double(count_bb_total));
    outfile->Printf("\n    aaaa : %12zu / %12zu = %f", count_aaaa, count_aa_total,
                    double(count_aaaa) / double(count_aa_total));
    outfile->Printf("\n    abab : %12zu / %12zu = %f", count_abab, count_abab_total,
                    double(count_abab) / double(count_abab_total));
    outfile->Printf("\n    bbbb : %12zu / %12zu = %f", count_bbbb, count_bb_total,
                    double(count_bbbb) / double(count_bb_total));
    outfile->Printf("\n");
#endif
}

void SigmaVectorDynamic::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
}

void SigmaVectorDynamic::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorDynamic::compute_sigma_scalar(SharedVector sigma, SharedVector b) {
    timer energy_timer("SigmaVectorDynamic:scalar");

    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    // loop over all determinants
    for (size_t I = 0; I < size_; ++I) {
        double b_I = b_p[I];
        sigma_p[I] += diag_[I] * b_I;
    }
}

void SigmaVectorDynamic::compute_sigma_aa_fast_search_group_ui64(SharedVector sigma,
                                                                 SharedVector b) {
    timer energy_timer("SigmaVectorDynamic:sigma_aa");
    for (size_t I = 0; I < size_; ++I)
        temp_sigma_[I] = 0.0;
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = b_sorted_string_list_ui64_.add(I);
        temp_b_[I] = b->get(addI);
    }

    // loop over all determinants
    const auto& sorted_half_dets = b_sorted_string_list_ui64_.sorted_half_dets();
    for (const auto& Ib : sorted_half_dets) {
        compute_aa_coupling_compare_group_ui64(Ib, temp_b_);
    }

    // Add sigma using the determinant address used in the DeterminantHashVector object
    //    const auto& sorted_dets = b_sorted_string_list_ui64_.sorted_dets();
    double* sigma_p = sigma->pointer();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = b_sorted_string_list_ui64_.add(I);
        //        size_t addI = space_.get_idx(sorted_dets[I]);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDynamic::compute_sigma_bb_fast_search_group_ui64(SharedVector sigma,
                                                                 SharedVector b) {
    timer energy_timer("SigmaVectorDynamic:sigma_bb");
    for (size_t I = 0; I < size_; ++I)
        temp_sigma_[I] = 0.0;
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = a_sorted_string_list_ui64_.add(I);
        temp_b_[I] = b->get(addI);
    }

    // loop over all determinants
    const auto& sorted_half_dets = a_sorted_string_list_ui64_.sorted_half_dets();
    for (const auto& Ia : sorted_half_dets) {
        compute_bb_coupling_compare_group_ui64(Ia, temp_b_);
    }

    // Add sigma using the determinant address used in the DeterminantHashVector object
    //    const auto& sorted_dets = a_sorted_string_list_ui64_.sorted_dets();
    double* sigma_p = sigma->pointer();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = a_sorted_string_list_ui64_.add(I);
        //        size_t addI = space_.get_idx(sorted_dets[I]);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDynamic::compute_sigma_abab_fast_search_group_ui64(SharedVector sigma,
                                                                   SharedVector b) {
    timer energy_timer("SigmaVectorDynamic:sigma_abab");
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
    //    const auto& sorted_dets = a_sorted_string_list_ui64_.sorted_dets();
    for (size_t I = 0; I < size_; ++I) {
        size_t addI = a_sorted_string_list_ui64_.add(I);
        //        size_t addI = space_.get_idx(sorted_dets[I]);
        sigma_p[addI] += temp_sigma_[I];
    }
}

void SigmaVectorDynamic::compute_aa_coupling_compare_group_ui64(const UI64Determinant::bit_t& Ib,
                                                                std::vector<double>& b) {
    const auto& sorted_dets = b_sorted_string_list_ui64_.sorted_dets();
    const auto& range_I = b_sorted_string_list_ui64_.range(Ib);
    UI64Determinant::bit_t Ia;
    UI64Determinant::bit_t Ja;
    UI64Determinant::bit_t IJa;
    size_t first_I = range_I.first;
    size_t last_I = range_I.second;
    for (size_t posI = first_I; posI < last_I; ++posI) {
        Ia = sorted_dets[posI].get_alfa_bits();
        for (size_t posJ = first_I; posJ < last_I; ++posJ) {
            Ja = sorted_dets[posJ].get_alfa_bits();
#if SIGMA_VEC_DEBUG
            count_aa_total++;
#endif
            // find common bits
            IJa = Ja ^ Ia;
            int ndiff = ui64_bit_count(IJa);
            if (ndiff == 2) {
                double H_IJ = slater_rules_single_alpha(Ib, Ia, Ja, fci_ints_);
                temp_sigma_[posI] += H_IJ * b[posJ];
#if SIGMA_VEC_DEBUG
                count_aa++;
#endif
            } else if (ndiff == 4) {
                double H_IJ = slater_rules_double_alpha_alpha(Ia, Ja, fci_ints_);
                temp_sigma_[posI] += H_IJ * b[posJ];
#if SIGMA_VEC_DEBUG
                count_aaaa++;
#endif
            }
        }
    }
}
void SigmaVectorDynamic::compute_bb_coupling_compare_group_ui64(const UI64Determinant::bit_t& Ia,
                                                                std::vector<double>& b) {
    const auto& sorted_dets = a_sorted_string_list_ui64_.sorted_dets();
    const auto& range_I = a_sorted_string_list_ui64_.range(Ia);
    UI64Determinant::bit_t Ib;
    UI64Determinant::bit_t Jb;
    UI64Determinant::bit_t IJb;
    size_t first_I = range_I.first;
    size_t last_I = range_I.second;
    for (size_t posI = first_I; posI < last_I; ++posI) {
        Ib = sorted_dets[posI].get_beta_bits();
        for (size_t posJ = first_I; posJ < last_I; ++posJ) {
            Jb = sorted_dets[posJ].get_beta_bits();
#if SIGMA_VEC_DEBUG
            count_bb_total++;
#endif
            // find common bits
            IJb = Jb ^ Ib;
            int ndiff = ui64_bit_count(IJb);
            if (ndiff == 2) {
                double H_IJ = slater_rules_single_beta(Ia, Ib, Jb, fci_ints_);
                temp_sigma_[posI] += H_IJ * b[posJ];
#if SIGMA_VEC_DEBUG
                count_bb++;
#endif
            } else if (ndiff == 4) {
                double H_IJ = slater_rules_double_beta_beta(Ib, Jb, fci_ints_);
                temp_sigma_[posI] += H_IJ * b[posJ];
#if SIGMA_VEC_DEBUG
                count_bbbb++;
#endif
            }
        }
    }
}

void SigmaVectorDynamic::compute_bb_coupling_compare_singles_group_ui64(
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
}
}
