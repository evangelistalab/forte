/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <cmath>

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libciomr/libciomr.h"
#include "psi4/libqt/qt.h"

#include "forte-def.h"
#include "iterative_solvers.h"
#include "sparse_ci_solver.h"
//#include "fci/fci_vector.h"

struct PairHash {
    size_t operator()(const std::pair<size_t, size_t>& p) const {
        return (p.first * 1000) + p.second;
    }
};

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

#ifdef HAVE_MPI
SigmaVectorMPI::SigmaVectorMPI( const DeterminantMap& space, WFNOperator& op)
    : SigmaVector(space.size()), space_(space) {

}

void SigmaVectorMPI::compute_sigma( SharedVector sigma, SharedVector b)
{
    
}

#endif


SigmaVectorWfn::SigmaVectorWfn(const DeterminantMap& space, WFNOperator& op)
    : SigmaVector(space.size()), space_(space), a_ann_list_(op.a_ann_list_),
      a_cre_list_(op.a_cre_list_), b_ann_list_(op.b_ann_list_),
      b_cre_list_(op.b_cre_list_), aa_ann_list_(op.aa_ann_list_),
      aa_cre_list_(op.aa_cre_list_), ab_ann_list_(op.ab_ann_list_),
      ab_cre_list_(op.ab_cre_list_), bb_ann_list_(op.bb_ann_list_),
      bb_cre_list_(op.bb_cre_list_) {

    det_hash<size_t> detmap = space_.wfn_hash();
    diag_.resize(space_.size());
    for (det_hash<size_t>::const_iterator it = detmap.begin(),
                                          endit = detmap.end();
         it != endit; ++it) {
        diag_[it->second] = it->first.energy();
    }
}

SigmaVectorString::SigmaVectorString(
    const std::vector<STLBitsetDeterminant>& space, bool print_details,
    bool disk)
    : SigmaVector(space.size()), space_(space) {
    //    for( auto& I : space) I.print();
    using det_hash = std::unordered_map<STLBitsetDeterminant, size_t,
                                        STLBitsetDeterminant::Hash>;
    using bstmap_it = det_hash::iterator;

    int ntd = omp_get_max_threads();
    outfile->Printf("\n  Using %d threads.", ntd);

    print_ = print_details;

    size_t max_I = space.size();

    use_disk_ = false;
    //    if( max_I > 2.3e6 ){
    //        use_disk_ = true;
    //        if( print_ ) outfile->Printf("\n  Determinant space exceeds 2.3 x
    //        10^6. Switching to disk algorithm");
    //    }
    noalfa_ = space[0].get_alfa_occ().size();
    nobeta_ = space[0].get_beta_occ().size();

    // Maps alpha and beta strings to counter
    det_hash alfa_map;
    det_hash beta_map;

    // Number of alpha and beta strings
    size_t n_alfa_strings = 0;
    size_t n_beta_strings = 0;

    // Vector that links alpha/beta strings back to original determinant
    std::vector<std::vector<size_t>> alfa_to_det;
    std::vector<std::vector<size_t>> beta_to_det;

    std::vector<std::vector<std::pair<size_t, short>>> a_str_list(max_I * noalfa_);
    // Fill maps and list
    det_hash a_ann;
    size_t nadd = 0;
    for (size_t I = 0; I < max_I; ++I) {
        STLBitsetDeterminant adet = space[I];
        STLBitsetDeterminant bdet = space[I];
        double EI = adet.energy();
        diag_.push_back(EI);

        adet.zero_spin(1);
        bdet.zero_spin(0);

        size_t a_add;
        size_t b_add;
        bstmap_it a_it = alfa_map.find(adet);
        if (a_it == alfa_map.end()) {
            a_add = n_alfa_strings;
            alfa_map[adet] = a_add;
            n_alfa_strings++;
        } else {
            a_add = a_it->second;
        }

        bstmap_it b_it = beta_map.find(bdet);
        if (b_it == beta_map.end()) {
            b_add = n_beta_strings;
            beta_map[bdet] = b_add;
            n_beta_strings++;
        } else {
            b_add = b_it->second;
        }

        alfa_to_det.resize(n_alfa_strings);
        beta_to_det.resize(n_beta_strings);

        alfa_to_det[a_add].push_back(I);
        beta_to_det[b_add].push_back(I);

        // Build a vector linking a_ann dets to dets
        std::vector<int> aocc = adet.get_alfa_occ();
        for (size_t i = 0; i < noalfa_; ++i) {
            size_t nstr;
            int ii = aocc[i];
            adet.set_alfa_bit(ii, false);
            bstmap_it it = a_ann.find(adet);
            if (it == a_ann.end()) {
                nstr = nadd;
                a_ann[adet] = nstr;
                nadd++;
            } else {
                nstr = it->second;
            }
            a_str_list[nstr].push_back(std::make_pair(I, i));
            adet.set_alfa_bit(ii, true);
        }
    }
    a_str_list.resize(nadd);
    a_str_list.shrink_to_fit();

    outfile->Printf("\n  Number of alpha strings:  %zu", n_alfa_strings);
    outfile->Printf("\n  Number of beta strings:   %zu", n_beta_strings);

    Timer single;
    // Build alpha annihilation list
    {
        a_ann_list_.resize(max_I);
        outfile->Printf("\n\n  Building alpha annihilation list");
        size_t na_ann = 0;

        for (size_t B = 0; B < n_beta_strings; ++B) {
            det_hash map_a_ann;
            std::vector<size_t> a_list = beta_to_det[B];
            size_t max_A = a_list.size();

            if (max_A < 2)
                continue;

            for (size_t A = 0; A < max_A; ++A) {
                size_t I = a_list[A];
                STLBitsetDeterminant detA(space_[I]);
                detA.zero_spin(1);
                std::vector<int> aocc = detA.get_alfa_occ();
                a_ann_list_[I].resize(noalfa_);
                for (size_t a = 0; a < noalfa_; ++a) {
                    int aa = aocc[a];
                    detA.set_alfa_bit(aa, false);
                    const double sign = detA.slater_sign_alpha(aa);

                    size_t detA_add;
                    bstmap_it it = map_a_ann.find(detA);
                    if (it == map_a_ann.end()) {
                        detA_add = na_ann;
                        map_a_ann[detA] = na_ann;
                        na_ann++;
                    } else {
                        detA_add = it->second;
                    }
                    a_ann_list_[I][a] = std::make_pair(
                        detA_add, (sign > 0.0) ? (aa + 1) : (-aa - 1));
                    detA.set_alfa_bit(aa, true);
                }
            }
        }

        a_ann_list_.shrink_to_fit();

        outfile->Printf("      ...done");
        outfile->Printf("\n  Building alpha creation list");
        a_cre_list_.resize(na_ann);
        for (size_t I = 0; I < max_I; ++I) {
            const std::vector<std::pair<size_t, short>>& a_ann = a_ann_list_[I];
            for (size_t a = 0, max_a = a_ann.size(); a < max_a; ++a) {
                const std::pair<size_t, short>& apair = a_ann[a];
                size_t J = apair.first;
                a_cre_list_[J].push_back(std::make_pair(I, apair.second));
            }
        }
        if (use_disk_) {
            write_single_to_disk(a_ann_list_, 0);
            a_ann_list_.clear();
            a_ann_list_.shrink_to_fit();
            write_single_to_disk(a_cre_list_, 2);
            a_cre_list_.clear();
            a_cre_list_.shrink_to_fit();
        }
        outfile->Printf("          ...done");
    }

    // Compute the beta annihilation lists
    {
        outfile->Printf("\n  Building beta annihilation list");
        b_ann_list_.resize(max_I);
        size_t nb_ann = 0;
        for (size_t A = 0; A < n_alfa_strings; ++A) {
            det_hash map_b_ann;
            std::vector<size_t> b_list = alfa_to_det[A];
            size_t max_B = b_list.size();

            if (max_B < 2)
                continue;

            for (size_t B = 0; B < max_B; ++B) {
                size_t I = b_list[B];
                STLBitsetDeterminant detB(space_[I]);
                detB.zero_spin(0);

                std::vector<int> bocc = detB.get_beta_occ();
                b_ann_list_[I].resize(nobeta_);
                for (size_t a = 0; a < nobeta_; ++a) {
                    int aa = bocc[a];
                    detB.set_beta_bit(aa, false);
                    const double sign = detB.slater_sign_beta(aa);
                    size_t detB_add;
                    bstmap_it it = map_b_ann.find(detB);
                    if (it == map_b_ann.end()) {
                        detB_add = nb_ann;
                        map_b_ann[detB] = nb_ann;
                        nb_ann++;
                    } else {
                        detB_add = it->second;
                    }
                    b_ann_list_[I][a] = std::make_pair(
                        detB_add, (sign > 0.0) ? (aa + 1) : (-aa - 1));
                    detB.set_beta_bit(aa, true);
                }
                //  b_ann_list_[I] = b_ann;
            }
        }

        b_ann_list_.shrink_to_fit();
        outfile->Printf("       ...done");
        outfile->Printf("\n  Building beta creation list");
        b_cre_list_.resize(nb_ann);
        for (size_t I = 0; I < max_I; ++I) {
            const std::vector<std::pair<size_t, short>>& b_ann = b_ann_list_[I];
            for (size_t b = 0, max_b = b_ann.size(); b < max_b; ++b) {
                const std::pair<size_t, short>& bpair = b_ann[b];
                size_t J = bpair.first;
                b_cre_list_[J].push_back(std::make_pair(I, bpair.second));
            }
        }

        if (use_disk_) {
            write_single_to_disk(b_ann_list_, 1);
            b_ann_list_.clear();
            b_ann_list_.shrink_to_fit();
            write_single_to_disk(b_cre_list_, 3);
            b_cre_list_.clear();
            b_cre_list_.shrink_to_fit();
        }
        outfile->Printf("           ...done");
    }
    outfile->Printf("\n  Time spent building single lists:    %7.6f s \n",
                    single.get());

    Timer doubles;
    // Compute alpha-alpha coupling list
    outfile->Printf("\n  Building alpha-alpha annihilation lists");
    {
        size_t naa_ann = 0;
        aa_ann_list_.resize(max_I);
        for (size_t B = 0; B < n_beta_strings; ++B) {
            det_hash map_aa_ann;
            std::vector<size_t> a_list = beta_to_det[B];
            size_t max_A = a_list.size();

            if (max_A < 2)
                continue;

            for (size_t A = 0; A < max_A; ++A) {
                const size_t I = a_list[A];
                STLBitsetDeterminant detA(space_[I]);
                detA.zero_spin(1);

                std::vector<int> aocc = detA.get_alfa_occ();

                std::vector<std::tuple<size_t, short, short>> aa_ann(
                    noalfa_ * (noalfa_ - 1) / 2);

                for (size_t i = 0, ij = 0; i < noalfa_; ++i) {
                    for (size_t j = i + 1; j < noalfa_; ++j, ++ij) {
                        int ii = aocc[i];
                        int jj = aocc[j];
                        double sign = detA.slater_sign_alpha(ii) *
                                      detA.slater_sign_alpha(jj);
                        detA.set_alfa_bit(jj, false);
                        detA.set_alfa_bit(ii, false);

                        bstmap_it it = map_aa_ann.find(detA);
                        size_t detA_add;
                        // detJ is not in the map, add it
                        if (it == map_aa_ann.end()) {
                            detA_add = naa_ann;
                            map_aa_ann[detA] = naa_ann;
                            naa_ann++;
                        } else {
                            detA_add = it->second;
                        }
                        aa_ann[ij] = std::make_tuple(
                            detA_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
                        detA.set_alfa_bit(jj, true);
                        detA.set_alfa_bit(ii, true);
                    }
                }
                aa_ann_list_[I] = aa_ann;
            }
        }

        outfile->Printf("     ...done");
        outfile->Printf("\n  Building alpha-alpha creation lists");
        aa_cre_list_.resize(naa_ann);
        for (size_t I = 0; I < max_I; ++I) {
            const std::vector<std::tuple<size_t, short, short>>& aa_ann =
                aa_ann_list_[I];
            for (size_t a = 0, max_a = aa_ann.size(); a < max_a; ++a) {
                auto& J_sign = aa_ann[a];
                size_t J = std::get<0>(J_sign);
                short i = std::get<1>(J_sign);
                short j = std::get<2>(J_sign);
                aa_cre_list_[J].push_back(std::make_tuple(I, i, j));
            }
        }
        aa_cre_list_.shrink_to_fit();
        if (use_disk_) {
            write_double_to_disk(aa_cre_list_, 7);
            aa_cre_list_.clear();
            aa_cre_list_.shrink_to_fit();
            write_double_to_disk(aa_ann_list_, 4);
            aa_ann_list_.clear();
            aa_ann_list_.shrink_to_fit();
        }
    }
    outfile->Printf("         ...done");

    // Compute beta-beta coupling list
    outfile->Printf("\n  Building beta-beta annihilation lists");
    {
        size_t nbb_ann = 0;
        bb_ann_list_.resize(max_I);
        for (size_t A = 0; A < n_alfa_strings; ++A) {
            det_hash map_bb_ann;
            std::vector<size_t> b_list = alfa_to_det[A];
            size_t max_B = b_list.size();

            if (max_B < 2)
                continue;

            for (size_t B = 0; B < max_B; ++B) {
                size_t I = b_list[B];

                STLBitsetDeterminant detB(space_[I]);
                detB.zero_spin(0);

                std::vector<int> bocc = detB.get_beta_occ();

                std::vector<std::tuple<size_t, short, short>> bb_ann(
                    nobeta_ * (nobeta_ - 1) / 2);

                for (size_t i = 0, ij = 0; i < nobeta_; ++i) {
                    for (size_t j = i + 1; j < nobeta_; ++j, ++ij) {
                        int ii = bocc[i];
                        int jj = bocc[j];
                        double sign = detB.slater_sign_beta(ii) *
                                      detB.slater_sign_beta(jj);
                        detB.set_beta_bit(jj, false);
                        detB.set_beta_bit(ii, false);

                        bstmap_it it = map_bb_ann.find(detB);
                        size_t detB_add;
                        // detJ is not in the map, add it
                        if (it == map_bb_ann.end()) {
                            detB_add = nbb_ann;
                            map_bb_ann[detB] = nbb_ann;
                            nbb_ann++;
                        } else {
                            detB_add = it->second;
                        }
                        bb_ann[ij] = std::make_tuple(
                            detB_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
                        detB.set_beta_bit(jj, true);
                        detB.set_beta_bit(ii, true);
                    }
                }
                bb_ann_list_[I] = bb_ann;
            }
        }
        outfile->Printf("       ...done");
        outfile->Printf("\n  Building beta-beta creation lists");

        bb_cre_list_.resize(nbb_ann);
        for (size_t I = 0; I < max_I; ++I) {
            const std::vector<std::tuple<size_t, short, short>>& bb_ann =
                bb_ann_list_[I];
            for (size_t a = 0, max_a = bb_ann.size(); a < max_a; ++a) {
                auto& J_sign = bb_ann[a];
                size_t J = std::get<0>(J_sign);
                short i = std::get<1>(J_sign);
                short j = std::get<2>(J_sign);
                bb_cre_list_[J].push_back(std::make_tuple(I, i, j));
            }
        }
        bb_cre_list_.shrink_to_fit();
        if (use_disk_) {
            write_double_to_disk(bb_cre_list_, 8);
            bb_cre_list_.clear();
            bb_cre_list_.shrink_to_fit();
            write_double_to_disk(bb_ann_list_, 5);
            bb_ann_list_.clear();
            bb_ann_list_.shrink_to_fit();
        }
    }
    outfile->Printf("           ...done");

    // Form alpha-beta annihilation list
    outfile->Printf("\n  Building alpha-beta annihilation lists");

    {
        size_t nab_ann = 0;
        ab_ann_list_.resize(max_I);

        // Loop through a_str_list to get all n-1(a) determinants
        size_t nastr = a_str_list.size();
        for (size_t adet = 0; adet < nastr; ++adet) {
            det_hash map_ab_ann;
            std::vector<std::pair<size_t, short>>& a_list = a_str_list[adet];

            for (size_t a = 0, maxa = a_list.size(); a < maxa; ++a) {
                size_t I = a_list[a].first;
                int i = a_list[a].second;
                STLBitsetDeterminant detA = space_[I];
                detA.zero_spin(1);
                std::vector<int> aocc = detA.get_alfa_occ();
                int ii = aocc[i];

                STLBitsetDeterminant detB = space[I];
                detB.zero_spin(0);
                std::vector<int> bocc = detB.get_beta_occ();

                std::vector<std::tuple<size_t, short, short>>
                    ab_ann; //(noalfa_*nobeta_);

                for (size_t j = 0; j < nobeta_; ++j) {
                    int jj = bocc[j];
                    double sign =
                        detA.slater_sign_alpha(ii) * detB.slater_sign_beta(jj);
                    detB.set_beta_bit(jj, false);
                    bstmap_it it = map_ab_ann.find(detB);
                    size_t detJ_add;
                    // detJ is not in the map, add it
                    if (it == map_ab_ann.end()) {
                        detJ_add = nab_ann;
                        map_ab_ann[detB] = nab_ann;
                        nab_ann++;
                    } else {
                        detJ_add = it->second;
                    }

                    ab_ann_list_[I].push_back(std::make_tuple(
                        detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj));
                    detB.set_beta_bit(jj, true);
                }
                //  ab_ann_list_[I] = ab_ann;
            }
        }
        outfile->Printf("      ...done");

        outfile->Printf("\n  Building alpha-beta creation lists");
        ab_cre_list_.resize(nab_ann);
        for (size_t I = 0; I < max_I; ++I) {
            const std::vector<std::tuple<size_t, short, short>>& ab_ann =
                ab_ann_list_[I];
            for (size_t a = 0, max_a = ab_ann.size(); a < max_a; ++a) {
                auto J_sign = ab_ann[a];
                size_t J = std::get<0>(J_sign);
                short i = std::get<1>(J_sign);
                short j = std::get<2>(J_sign);
                ab_cre_list_[J].push_back(std::make_tuple(I, i, j));
                //               outfile->Printf("\n  I, i, j : %zu, %d, %d", J,
                //               i, j);
            }
        }
        ab_cre_list_.shrink_to_fit();
        if (use_disk_) {
            write_double_to_disk(ab_ann_list_, 6);
            ab_ann_list_.clear();
            ab_ann_list_.shrink_to_fit();
            write_double_to_disk(ab_cre_list_, 9);
            ab_cre_list_.clear();
            ab_cre_list_.shrink_to_fit();
        }
        outfile->Printf("          ...done");
    }

    outfile->Printf("\n  Time spent building doubles lists:   %7.6f s \n",
                    doubles.get());
    outfile->Flush();
}

void SigmaVectorString::compute_sigma(Matrix&, Matrix&, int) {
    outfile->Printf("\n Implement me. Maybe.");
}

void SigmaVectorString::compute_sigma(SharedVector sigma, SharedVector b) {
    sigma->zero();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();

    // Compute the overlap with each root
    int nbad = bad_states_.size();
    std::vector<double> overlap(nbad);
    if (nbad != 0) {
        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            double dprd = 0.0;
            for (size_t det = 0, ndet = bad_state.size(); det < ndet; ++det) {
                dprd += bad_state[det].second * b_p[bad_state[det].first];
            }
            overlap[n] = dprd;
        }
        // outfile->Printf("\n Overlap: %1.6f", overlap[0]);

        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            size_t ndet = bad_state.size();

#pragma omp parallel for
            for (size_t det = 0; det < ndet; ++det) {
                b_p[bad_state[det].first] -= bad_state[det].second * overlap[n];
            }
        }
    }
// outfile->Printf("\n  size: %zu", size_);

#pragma omp parallel
    {
        int num_thread = omp_get_max_threads();
        int tid = omp_get_thread_num();

        size_t bin_size = size_ / num_thread;
        bin_size += (tid < (size_ % num_thread)) ? 1 : 0;

        size_t start_idx = (tid < (size_ % num_thread))
                               ? tid * bin_size
                               : (size_ % num_thread) * (bin_size + 1) +
                                     (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        for (size_t J = start_idx; J < end_idx; ++J) {
            // reference
            sigma_p[J] += diag_[J] * b_p[J];

            // aa singles
            for (auto& aJ_mo_sign : a_ann_list_[J]) {
                const size_t aJ_add = aJ_mo_sign.first;
                const size_t p = std::abs(aJ_mo_sign.second) - 1;
                const double sign_p = (aJ_mo_sign.second > 0.0) ? 1.0 : -1.0;
                for (auto& aaJ_mo : a_cre_list_[aJ_add]) {
                    const size_t q = std::abs(aaJ_mo.second) - 1;
                    if (p != q) {
                        const size_t I = aaJ_mo.first;
                        const double sign_q =
                            (aaJ_mo.second > 0.0) ? 1.0 : -1.0;
                        const double HIJ =
                            space_[I].slater_rules_single_alpha_abs(p, q) *
                            sign_p * sign_q;
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // bb singles
            for (auto& bpair : b_ann_list_[J]) {
                const size_t bJ_add = bpair.first;
                const size_t p = std::abs(bpair.second) - 1;
                const double sign_p = bpair.second > 0.0 ? 1.0 : -1.0;
                for (auto& bbJ_mo : b_cre_list_[bJ_add]) {
                    const size_t q = std::abs(bbJ_mo.second) - 1;
                    if (p != q) {
                        const double sign_q = bbJ_mo.second > 0.0 ? 1.0 : -1.0;
                        const size_t I = bbJ_mo.first;
                        const double HIJ =
                            space_[I].slater_rules_single_beta_abs(p, q) *
                            sign_p * sign_q;
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // aaaa doubles
            for (auto& aaJ_mo_sign : aa_ann_list_[J]) {
                const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
                double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
                const size_t q = std::get<2>(aaJ_mo_sign);
                for (auto& aaaaJ_mo_sign : aa_cre_list_[aaJ_add]) {
                    const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(aaaaJ_mo_sign);
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        const size_t I = std::get<0>(aaaaJ_mo_sign);
                        const double sign_rs =
                            std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_aa(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // bbbb singles
            for (auto& bbJ_mo_sign : bb_ann_list_[J]) {
                const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
                const double sign_pq =
                    std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
                const size_t q = std::get<2>(bbJ_mo_sign);
                for (auto& bbbbJ_mo_sign : bb_cre_list_[bbJ_add]) {
                    const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(bbbbJ_mo_sign);
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        const size_t I = std::get<0>(bbbbJ_mo_sign);
                        const double sign_rs =
                            std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_bb(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            for (auto& abJ_mo_sign : ab_ann_list_[J]) {
                const size_t abJ_add = std::get<0>(abJ_mo_sign);
                double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
                const size_t q = std::get<2>(abJ_mo_sign);
                for (auto& ababJ_mo_sign : ab_cre_list_[abJ_add]) {
                    const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(ababJ_mo_sign);
                    if ((p != r) and (q != s)) {
                        const size_t I = std::get<0>(ababJ_mo_sign);
                        double sign_rs =
                            std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_ab(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
        }
    }
}
void SigmaVectorString::add_bad_roots(
    std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SigmaVectorString::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

SigmaVectorList::SigmaVectorList(const std::vector<STLBitsetDeterminant>& space,
                                 bool print_details)
    : SigmaVector(space.size()), space_(space) {
    using det_hash = std::unordered_map<STLBitsetDeterminant, size_t,
                                        STLBitsetDeterminant::Hash>;
    using bstmap_it = det_hash::iterator;

    size_t max_I = space.size();

    //  for( auto& I : space) I.print();

    size_t naa_ann = 0;
    size_t nab_ann = 0;
    size_t nbb_ann = 0;

    // Make alpha and beta strings
    det_hash a_str_map;
    det_hash b_str_map;
    Timer single;
    if (print_details) {
        outfile->Printf("\n  Generating determinants with N-1 electrons.\n");
    }
    a_ann_list.resize(max_I);
    double s_map_size = 0.0;
    // Generate alpha annihilation
    {
        size_t na_ann = 0;
        det_hash map_a_ann;
        for (size_t I = 0; I < max_I; ++I) {
            STLBitsetDeterminant detI = space[I];
            double EI = detI.energy();
            diag_.push_back(EI);

            std::vector<int> aocc = detI.get_alfa_occ();
            int noalpha = aocc.size();

            std::vector<std::pair<size_t, short>> a_ann(noalpha);

            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                STLBitsetDeterminant detJ(detI);
                detJ.set_alfa_bit(ii, false);

                double sign = detI.slater_sign_alpha(ii);

                bstmap_it it = map_a_ann.find(detJ);
                size_t detJ_add;
                // detJ is not in the map, add it
                if (it == map_a_ann.end()) {
                    detJ_add = na_ann;
                    map_a_ann[detJ] = na_ann;
                    na_ann++;
                } else {
                    detJ_add = it->second;
                }
                a_ann[i] = std::make_pair(detJ_add,
                                          (sign > 0.5) ? (ii + 1) : (-ii - 1));
            }
            a_ann.shrink_to_fit();
            a_ann_list[I] = a_ann;
        }
        a_ann_list.shrink_to_fit();
        a_cre_list.resize(na_ann);
        s_map_size += (32. + sizeof(size_t)) * na_ann;
    }
    // Generate beta annihilation
    b_ann_list.resize(max_I);
    {
        size_t nb_ann = 0;
        det_hash map_b_ann;
        for (size_t I = 0; I < max_I; ++I) {
            STLBitsetDeterminant detI = space[I];

            std::vector<int> bocc = detI.get_beta_occ();
            int nobeta = bocc.size();

            std::vector<std::pair<size_t, short>> b_ann(nobeta);

            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                STLBitsetDeterminant detJ(detI);
                detJ.set_beta_bit(ii, false);

                double sign = detI.slater_sign_beta(ii);

                bstmap_it it = map_b_ann.find(detJ);
                size_t detJ_add;
                // detJ is not in the map, add it
                if (it == map_b_ann.end()) {
                    detJ_add = nb_ann;
                    map_b_ann[detJ] = nb_ann;
                    nb_ann++;
                } else {
                    detJ_add = it->second;
                }
                b_ann[i] = std::make_pair(detJ_add,
                                          (sign > 0.5) ? (ii + 1) : (-ii - 1));
            }
            b_ann.shrink_to_fit();
            b_ann_list[I] = b_ann;
        }
        b_ann_list.shrink_to_fit();
        b_cre_list.resize(map_b_ann.size());
        s_map_size += (32. + sizeof(size_t)) * map_b_ann.size();
    }

    for (size_t I = 0; I < max_I; ++I) {
        const std::vector<std::pair<size_t, short>>& a_ann = a_ann_list[I];
        for (const std::pair<size_t, short>& J_sign : a_ann) {
            size_t J = J_sign.first;
            short sign = J_sign.second;
            a_cre_list[J].push_back(std::make_pair(I, sign));
            //    num_tuples_sigles++;
        }
        const std::vector<std::pair<size_t, short>>& b_ann = b_ann_list[I];
        for (const std::pair<size_t, short>& J_sign : b_ann) {
            size_t J = J_sign.first;
            short sign = J_sign.second;
            b_cre_list[J].push_back(std::make_pair(I, sign));
            //    num_tuples_sigles++;
        }
    }
    size_t mem_tuple_singles =
        a_cre_list.capacity() * (sizeof(size_t) + sizeof(short));
    mem_tuple_singles +=
        b_cre_list.capacity() * (sizeof(size_t) + sizeof(short));

    //    outfile->Printf("\n  Size of lists:");
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",a_ann_list.size());
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",b_ann_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",a_cre_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",b_cre_list.size());
    if (print_details) {
        outfile->Printf("\n  Time spent building single lists: %f s",
                        single.get());
        outfile->Printf("\n  Memory for single-hole lists: %f MB",
                        double(mem_tuple_singles) /
                            (1024. * 1024.)); // Convert to MB
        outfile->Printf("\n  Memory for single-hole maps:  %f MB",
                        s_map_size / (1024. * 1024.)); // Convert to MB
        outfile->Printf("\n  Generating determinants with N-2 electrons.\n");
    }

    // Generate alpha-alpha annihilation
    double d_map_size = 0.0;
    aa_ann_list.resize(max_I);
    {
        det_hash map_aa_ann;
        for (size_t I = 0; I < max_I; ++I) {
            STLBitsetDeterminant detI = space[I];

            std::vector<int> aocc = detI.get_alfa_occ();
            size_t noalpha = aocc.size();

            std::vector<std::tuple<size_t, short, short>> aa_ann(
                noalpha * (noalpha - 1) / 2);

            for (size_t i = 0, ij = 0; i < noalpha; ++i) {
                for (size_t j = i + 1; j < noalpha; ++j, ++ij) {
                    int ii = aocc[i];
                    int jj = aocc[j];
                    STLBitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_alfa_bit(jj, false);

                    double sign =
                        detI.slater_sign_alpha(ii) * detI.slater_sign_alpha(jj);

                    bstmap_it it = map_aa_ann.find(detJ);
                    size_t detJ_add;
                    // detJ is not in the map, add it
                    if (it == map_aa_ann.end()) {
                        detJ_add = naa_ann;
                        map_aa_ann[detJ] = naa_ann;
                        naa_ann++;
                    } else {
                        detJ_add = it->second;
                    }
                    aa_ann[ij] = std::make_tuple(
                        detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
                }
            }
            aa_ann.shrink_to_fit();
            aa_ann_list[I] = aa_ann;
        }
        aa_cre_list.resize(map_aa_ann.size());
        d_map_size += (32. + sizeof(size_t)) * map_aa_ann.size();
    }
    // Generate beta-beta annihilation
    aa_ann_list.shrink_to_fit();
    bb_ann_list.resize(max_I);
    {
        det_hash map_bb_ann;
        for (size_t I = 0; I < max_I; ++I) {
            STLBitsetDeterminant detI = space[I];

            std::vector<int> bocc = detI.get_beta_occ();

            size_t nobeta = bocc.size();

            std::vector<std::tuple<size_t, short, short>> bb_ann(
                nobeta * (nobeta - 1) / 2);
            for (size_t i = 0, ij = 0; i < nobeta; ++i) {
                for (size_t j = i + 1; j < nobeta; ++j, ++ij) {
                    int ii = bocc[i];
                    int jj = bocc[j];
                    STLBitsetDeterminant detJ(detI);
                    detJ.set_beta_bit(ii, false);
                    detJ.set_beta_bit(jj, false);

                    double sign =
                        detI.slater_sign_beta(ii) * detI.slater_sign_beta(jj);
                    ;

                    bstmap_it it = map_bb_ann.find(detJ);
                    size_t detJ_add;
                    // detJ is not in the map, add it
                    if (it == map_bb_ann.end()) {
                        detJ_add = nbb_ann;
                        map_bb_ann[detJ] = nbb_ann;
                        nbb_ann++;
                    } else {
                        detJ_add = it->second;
                    }
                    bb_ann[ij] = std::make_tuple(
                        detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
                }
            }
            bb_ann.shrink_to_fit();
            bb_ann_list[I] = bb_ann;
        }
        bb_cre_list.resize(map_bb_ann.size());
        d_map_size += (32. + sizeof(size_t)) * map_bb_ann.size();
    }
    // Generate alpha-beta annihilation
    bb_ann_list.shrink_to_fit();
    ab_ann_list.resize(max_I);
    {
        det_hash map_ab_ann;
        for (size_t I = 0; I < max_I; ++I) {
            STLBitsetDeterminant detI = space[I];

            std::vector<int> aocc = detI.get_alfa_occ();
            std::vector<int> bocc = detI.get_beta_occ();

            size_t noalpha = aocc.size();
            size_t nobeta = bocc.size();

            std::vector<std::tuple<size_t, short, short>> ab_ann(noalpha *
                                                                 nobeta);
            for (size_t i = 0, ij = 0; i < noalpha; ++i) {
                for (size_t j = 0; j < nobeta; ++j, ++ij) {
                    int ii = aocc[i];
                    int jj = bocc[j];
                    STLBitsetDeterminant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_beta_bit(jj, false);

                    double sign =
                        detI.slater_sign_alpha(ii) * detI.slater_sign_beta(jj);

                    bstmap_it it = map_ab_ann.find(detJ);
                    size_t detJ_add;
                    // detJ is not in the map, add it
                    if (it == map_ab_ann.end()) {
                        detJ_add = nab_ann;
                        map_ab_ann[detJ] = nab_ann;
                        nab_ann++;
                    } else {
                        detJ_add = it->second;
                    }
                    ab_ann[ij] = std::make_tuple(
                        detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
                    //                    outfile->Printf("\n  %zu, %d, %d",
                    //                    detJ_add, (sign > 0.5) ? (ii + 1) :
                    //                    (-ii-1),jj);
                }
            }
            ab_ann.shrink_to_fit();
            ab_ann_list[I] = ab_ann;
        }
        ab_cre_list.resize(map_ab_ann.size());
        d_map_size += (32. + sizeof(size_t)) * map_ab_ann.size();
    }
    ab_ann_list.shrink_to_fit();

    for (size_t I = 0; I < max_I; ++I) {
        const std::vector<std::tuple<size_t, short, short>>& aa_ann =
            aa_ann_list[I];
        for (const std::tuple<size_t, short, short>& J_sign : aa_ann) {
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            aa_cre_list[J].push_back(std::make_tuple(I, i, j));
        }
        const std::vector<std::tuple<size_t, short, short>>& bb_ann =
            bb_ann_list[I];
        for (const std::tuple<size_t, short, short>& J_sign : bb_ann) {
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            bb_cre_list[J].push_back(std::make_tuple(I, i, j));
        }
        const std::vector<std::tuple<size_t, short, short>>& ab_ann =
            ab_ann_list[I];
        for (const std::tuple<size_t, short, short>& J_sign : ab_ann) {
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            ab_cre_list[J].push_back(std::make_tuple(I, i, j));
        }
    }
    aa_cre_list.shrink_to_fit();
    bb_cre_list.shrink_to_fit();
    ab_cre_list.shrink_to_fit();

    size_t mem_tuple_doubles =
        aa_cre_list.capacity() * (sizeof(size_t) + 2 * sizeof(short));
    mem_tuple_doubles +=
        bb_cre_list.capacity() * (sizeof(size_t) + 2 * sizeof(short));
    mem_tuple_doubles +=
        ab_cre_list.capacity() * (sizeof(size_t) + 2 * sizeof(short));

    //    outfile->Printf("\n  Size of lists:");
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",aa_ann_list.size());
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",ab_ann_list.size());
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",bb_ann_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",aa_cre_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",ab_cre_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",bb_cre_list.size());

    if (print_details) {
        outfile->Printf("\n  Memory for double-hole lists: %f MB",
                        double(mem_tuple_doubles) / (1024. * 1024.));
        outfile->Printf("\n  Memory for double-hole maps:  %f MB",
                        d_map_size / (1024. * 1024.));
    }
}

void SigmaVectorList::compute_sigma(Matrix& sigma, Matrix& b, int nroot) {
    sigma.zero();
    double** sigma_p = sigma.pointer();
    double** b_p = b.pointer();
    for (size_t J = 0; J < size_; ++J) {
        // reference
        for (int a = 0; a < nroot; ++a) {
            sigma_p[J][a] += diag_[J] * b_p[a][J];
        }

        // aa singles
        for (auto& aJ_mo_sign : a_ann_list[J]) {
            const size_t aJ_add = aJ_mo_sign.first;
            const size_t p = std::abs(aJ_mo_sign.second) - 1;
            for (auto& aaJ_mo_sign : a_cre_list[aJ_add]) {
                const size_t q = std::abs(aaJ_mo_sign.second) - 1;
                if (p != q) {
                    const double HIJ =
                        space_[aaJ_mo_sign.first].slater_rules_single_alpha(p,
                                                                            q);
                    const size_t I = aaJ_mo_sign.first;
                    for (int a = 0; a < nroot; ++a) {
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }

        // bb singles
        for (auto& bJ_mo_sign : b_ann_list[J]) {
            const size_t bJ_add = bJ_mo_sign.first;
            const size_t p = std::abs(bJ_mo_sign.second) - 1;
            for (auto& bbJ_mo_sign : b_cre_list[bJ_add]) {
                const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                if (p != q) {
                    const double HIJ =
                        space_[bbJ_mo_sign.first].slater_rules_single_beta(p,
                                                                           q);
                    const size_t I = bbJ_mo_sign.first;
                    for (int a = 0; a < nroot; ++a) {
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }

        // aaaa doubles
        for (auto& aaJ_mo_sign : aa_ann_list[J]) {
            const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
            const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aaJ_mo_sign);
            for (auto& aaaaJ_mo_sign : aa_cre_list[aaJ_add]) {
                const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                const size_t s = std::get<2>(aaaaJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)) {
                    const size_t aaaaJ_add = std::get<0>(aaaaJ_mo_sign);
                    const double sign_rs =
                        std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = aaaaJ_add;
                    const double HIJ =
                        sign_pq * sign_rs *
                        STLBitsetDeterminant::fci_ints_->tei_aa(p, q, r, s);
                    for (int a = 0; a < nroot; ++a) {
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }

        // aabb singles
        for (auto& abJ_mo_sign : ab_ann_list[J]) {
            const size_t abJ_add = std::get<0>(abJ_mo_sign);
            const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
            const size_t q = std::get<2>(abJ_mo_sign);
            for (auto& ababJ_mo_sign : ab_cre_list[abJ_add]) {
                const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                const size_t s = std::get<2>(ababJ_mo_sign);
                if ((p != r) and (q != s)) {
                    const size_t ababJ_add = std::get<0>(ababJ_mo_sign);
                    const double sign_rs =
                        std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = ababJ_add;
                    const double HIJ =
                        sign_pq * sign_rs *
                        STLBitsetDeterminant::fci_ints_->tei_ab(p, q, r, s);
                    for (int a = 0; a < nroot; ++a) {
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }

        // bbbb singles
        for (auto& bbJ_mo_sign : bb_ann_list[J]) {
            const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
            const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(bbJ_mo_sign);
            for (auto& bbbbJ_mo_sign : bb_cre_list[bbJ_add]) {
                const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                const size_t s = std::get<2>(bbbbJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)) {
                    const size_t bbbbJ_add = std::get<0>(bbbbJ_mo_sign);
                    const double sign_rs =
                        std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = bbbbJ_add;
                    const double HIJ =
                        sign_pq * sign_rs *
                        STLBitsetDeterminant::fci_ints_->tei_bb(p, q, r, s);
                    for (int a = 0; a < nroot; ++a) {
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }
    }
}

void SigmaVectorList::compute_sigma(SharedVector sigma, SharedVector b) {
    sigma->zero();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();

    // Compute the overlap with each root
    int nbad = bad_states_.size();
    std::vector<double> overlap(nbad);
    if (nbad != 0) {
        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            double dprd = 0.0;
            for (size_t det = 0, ndet = bad_state.size(); det < ndet; ++det) {
                dprd += bad_state[det].second * b_p[bad_state[det].first];
            }
            overlap[n] = dprd;
        }
        // outfile->Printf("\n Overlap: %1.6f", overlap[0]);

        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            size_t ndet = bad_state.size();

#pragma omp parallel for
            for (size_t det = 0; det < ndet; ++det) {
                b_p[bad_state[det].first] -= bad_state[det].second * overlap[n];
            }
        }
    }

#pragma omp parallel
    {
        int num_thread = omp_get_max_threads();
        int tid = omp_get_thread_num();

        size_t bin_size = size_ / num_thread;
        bin_size += (tid < (size_ % num_thread)) ? 1 : 0;

        size_t start_idx = (tid < (size_ % num_thread))
                               ? tid * bin_size
                               : (size_ % num_thread) * (bin_size + 1) +
                                     (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        for (size_t J = start_idx; J < end_idx; ++J) {
            // reference
            sigma_p[J] += diag_[J] * b_p[J];

            // aa singles
            for (auto& aJ_mo_sign : a_ann_list[J]) {
                const size_t aJ_add = aJ_mo_sign.first;
                const size_t p = std::abs(aJ_mo_sign.second) - 1;
                double sign_p = aJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                for (auto& aaJ_mo_sign : a_cre_list[aJ_add]) {
                    const size_t q = std::abs(aaJ_mo_sign.second) - 1;
                    if (p != q) {
                        const size_t I = aaJ_mo_sign.first;
                        double sign_q = aaJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                        const double HIJ =
                            space_[I].slater_rules_single_alpha_abs(p, q) *
                            sign_p * sign_q;
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // bb singles
            for (auto& bJ_mo_sign : b_ann_list[J]) {
                const size_t bJ_add = bJ_mo_sign.first;
                const size_t p = std::abs(bJ_mo_sign.second) - 1;
                double sign_p = bJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                for (auto& bbJ_mo_sign : b_cre_list[bJ_add]) {
                    const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                    if (p != q) {
                        const size_t I = bbJ_mo_sign.first;
                        double sign_q = bbJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                        const double HIJ =
                            space_[I].slater_rules_single_beta_abs(p, q) *
                            sign_p * sign_q;
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // aaaa doubles
            for (auto& aaJ_mo_sign : aa_ann_list[J]) {
                const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
                const double sign_pq =
                    std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
                const size_t q = std::get<2>(aaJ_mo_sign);
                for (auto& aaaaJ_mo_sign : aa_cre_list[aaJ_add]) {
                    const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(aaaaJ_mo_sign);
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        const size_t aaaaJ_add = std::get<0>(aaaaJ_mo_sign);
                        const double sign_rs =
                            std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const size_t I = aaaaJ_add;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_aa(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // aabb singles
            for (auto& abJ_mo_sign : ab_ann_list[J]) {
                const size_t abJ_add = std::get<0>(abJ_mo_sign);
                const double sign_pq =
                    std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
                const size_t q = std::get<2>(abJ_mo_sign);
                for (auto& ababJ_mo_sign : ab_cre_list[abJ_add]) {
                    const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(ababJ_mo_sign);
                    if ((p != r) and (q != s)) {
                        const size_t ababJ_add = std::get<0>(ababJ_mo_sign);
                        const double sign_rs =
                            std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const size_t I = ababJ_add;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_ab(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // bbbb singles
            for (auto& bbJ_mo_sign : bb_ann_list[J]) {
                const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
                const double sign_pq =
                    std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
                const size_t q = std::get<2>(bbJ_mo_sign);
                for (auto& bbbbJ_mo_sign : bb_cre_list[bbJ_add]) {
                    const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(bbbbJ_mo_sign);
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        const size_t bbbbJ_add = std::get<0>(bbbbJ_mo_sign);
                        const double sign_rs =
                            std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const size_t I = bbbbJ_add;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_bb(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
        }
    }
}

void SigmaVectorList::add_bad_roots(
    std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SigmaVectorList::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorWfn::add_bad_roots(
    std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SigmaVectorWfn::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorWfn::compute_sigma(Matrix& sigma, Matrix& b, int nroot) {}

void SigmaVectorWfn::compute_sigma(SharedVector sigma, SharedVector b) {
    sigma->zero();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();

    // Compute the overlap with each root
    int nbad = bad_states_.size();
    std::vector<double> overlap(nbad);
    if (nbad != 0) {
        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            double dprd = 0.0;
            for (size_t det = 0, ndet = bad_state.size(); det < ndet; ++det) {
                dprd += bad_state[det].second * b_p[bad_state[det].first];
            }
            overlap[n] = dprd;
        }
        // outfile->Printf("\n Overlap: %1.6f", overlap[0]);

        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            size_t ndet = bad_state.size();

#pragma omp parallel for
            for (size_t det = 0; det < ndet; ++det) {
                b_p[bad_state[det].first] -= bad_state[det].second * overlap[n];
            }
        }
    }
#pragma omp parallel
    {
        int num_thread = omp_get_max_threads();
        int tid = omp_get_thread_num();

        size_t bin_size = size_ / num_thread;
        bin_size += (tid < (size_ % num_thread)) ? 1 : 0;

        size_t start_idx = (tid < (size_ % num_thread))
                               ? tid * bin_size
                               : (size_ % num_thread) * (bin_size + 1) +
                                     (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;
        // Timer cycl;
        {
            const std::vector<STLBitsetDeterminant>& dets =
                space_.determinants();
            for (size_t J = start_idx; J < end_idx; ++J) {
                // reference
                sigma_p[J] += diag_[J] * b_p[J];
                // aa singles
                for (auto& aJ_mo_sign : a_ann_list_[J]) {
                    const size_t aJ_add = aJ_mo_sign.first;
                    const size_t p = std::abs(aJ_mo_sign.second) - 1;
                    double sign_p = aJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                    for (auto& aaJ_mo_sign : a_cre_list_[aJ_add]) {
                        const size_t q = std::abs(aaJ_mo_sign.second) - 1;
                        if (p != q) {
                            const size_t I = aaJ_mo_sign.first;
                            double sign_q =
                                aaJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                dets[I].slater_rules_single_alpha_abs(p, q) *
                                sign_p * sign_q;
                            sigma_p[J] += HIJ * b_p[I];
                        }
                    }
                }
                // bb singles
                for (auto& bJ_mo_sign : b_ann_list_[J]) {
                    const size_t bJ_add = bJ_mo_sign.first;
                    const size_t p = std::abs(bJ_mo_sign.second) - 1;
                    double sign_p = bJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                    for (auto& bbJ_mo_sign : b_cre_list_[bJ_add]) {
                        const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                        if (p != q) {
                            const size_t I = bbJ_mo_sign.first;
                            double sign_q =
                                bbJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                dets[I].slater_rules_single_beta_abs(p, q) *
                                sign_p * sign_q;
                            sigma_p[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }
        // outfile->Printf("\n  Time spent on singles: %1.5f", cycl.get());
        // Timer cycl2;
        for (size_t J = start_idx; J < end_idx; ++J) {
            // aaaa doubles
            for (auto& aaJ_mo_sign : aa_ann_list_[J]) {
                const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
                const double sign_pq =
                    std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
                const size_t q = std::get<2>(aaJ_mo_sign);
                for (auto& aaaaJ_mo_sign : aa_cre_list_[aaJ_add]) {
                    const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(aaaaJ_mo_sign);
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        const size_t I = std::get<0>(aaaaJ_mo_sign);
                        const double sign_rs =
                            std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_aa(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // aabb singles
            for (auto& abJ_mo_sign : ab_ann_list_[J]) {
                const size_t abJ_add = std::get<0>(abJ_mo_sign);
                const double sign_pq =
                    std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
                const size_t q = std::get<2>(abJ_mo_sign);
                for (auto& ababJ_mo_sign : ab_cre_list_[abJ_add]) {
                    const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(ababJ_mo_sign);
                    if ((p != r) and (q != s)) {
                        const size_t I = std::get<0>(ababJ_mo_sign);
                        const double sign_rs =
                            std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_ab(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // bbbb singles
            for (auto& bbJ_mo_sign : bb_ann_list_[J]) {
                const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
                const double sign_pq =
                    std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
                const size_t q = std::get<2>(bbJ_mo_sign);
                for (auto& bbbbJ_mo_sign : bb_cre_list_[bbJ_add]) {
                    const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(bbbbJ_mo_sign);
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        const size_t I = std::get<0>(bbbbJ_mo_sign);
                        const double sign_rs =
                            std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ =
                            sign_pq * sign_rs *
                            STLBitsetDeterminant::fci_ints_->tei_bb(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
        }
        // outfile->Printf("\n  Time spent on doubles: %1.5f", cycl2.get());
    }
}
void SparseCISolver::set_spin_project(bool value) { spin_project_ = value; }

void SparseCISolver::set_e_convergence(double value) { e_convergence_ = value; }

void SparseCISolver::set_maxiter_davidson(int value) {
    maxiter_davidson_ = value;
}

void SparseCISolver::set_force_diag(int value) { force_diag_method_ = value; }

void SparseCISolver::diagonalize_hamiltonian(
    const std::vector<STLBitsetDeterminant>& space, SharedVector& evals,
    SharedMatrix& evecs, int nroot, int multiplicity,
    DiagonalizationMethod diag_method) {
    if (space.size() <= 200 && !force_diag_method_) {
        diagonalize_full(space, evals, evecs, nroot, multiplicity);
    } else {
        if (diag_method == Full) {
            diagonalize_full(space, evals, evecs, nroot, multiplicity);
        } else if (diag_method == DLSolver) {
            diagonalize_davidson_liu_solver(space, evals, evecs, nroot,
                                            multiplicity);
        } else if (diag_method == DLString) {
            diagonalize_davidson_liu_string(space, evals, evecs, nroot,
                                            multiplicity, false);
        } else if (diag_method == DLDisk) {
            diagonalize_davidson_liu_string(space, evals, evecs, nroot,
                                            multiplicity, true);
        }
    }
}

void SparseCISolver::diagonalize_hamiltonian_map(
    const DeterminantMap& space, WFNOperator& op, SharedVector& evals,
    SharedMatrix& evecs, int nroot, int multiplicity,
    DiagonalizationMethod diag_method) {
    if ((space.size() <= 200 && !force_diag_method_) or diag_method == Full) {
        const std::vector<STLBitsetDeterminant> dets = space.determinants();
        diagonalize_full(dets, evals, evecs, nroot, multiplicity);
    } else if ( diag_method == MPI ){
        diagonalize_mpi(space, op, evals, evecs, nroot, multiplicity);
    } else {
        diagonalize_dl(space, op, evals, evecs, nroot, multiplicity);
    }
}
#ifdef HAVE_MPI
void SparseCISolver::diagonalize_mpi( 
    const DeterminantMap& space, WFNOperator& op, SharedVector& evals,
    SharedMatrix& evecs, int nroot, int multiplicity){

    if ( print_details_ ){
        outfile->Printf("\n\n  Distributed Davidson-Liu algorithm");
    }

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space, nroot));
    evals.reset(new Vector("e", nroot));

    SigmaVectorMPI sv(space, op);
    SigmaVector* sigma_vector = &sv;
    sigma_vector->add_bad_roots(bad_states_);
    davidson_liu_solver_map(space, sigma_vector, evals, evecs, nroot,
                            multiplicity);
}
#endif

void SparseCISolver::diagonalize_dl(const DeterminantMap& space,
                                    WFNOperator& op, SharedVector& evals,
                                    SharedMatrix& evecs, int nroot,
                                    int multiplicity) {
    if (print_details_) {
        outfile->Printf("\n\n Davidson-Liu solver algorithm");
    }
    size_t dim_space = space.size();
    evecs.reset(new Matrix("U", dim_space, nroot));
    evals.reset(new Vector("e", nroot));

    SigmaVectorWfn svw(space, op);
    SigmaVector* sigma_vector = &svw;
    sigma_vector->add_bad_roots(bad_states_);
    davidson_liu_solver_map(space, sigma_vector, evals, evecs, nroot,
                            multiplicity);
}

void SparseCISolver::diagonalize_full(
    const std::vector<STLBitsetDeterminant>& space, SharedVector& evals,
    SharedMatrix& evecs, int, int) {
    // Find all the eigenvalues and eigenvectors of the Hamiltonian
    SharedMatrix H = build_full_hamiltonian(space);

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U", dim_space, dim_space));
    evals.reset(new Vector("e", dim_space));

    // Diagonalize H
    H->diagonalize(evecs, evals);
}

void SparseCISolver::diagonalize_davidson_liu_solver(
    const std::vector<STLBitsetDeterminant>& space, SharedVector& evals,
    SharedMatrix& evecs, int nroot, int multiplicity) {
    if (print_details_) {
        outfile->Printf("\n\n  Davidson-liu solver algorithm");
        outfile->Flush();
    }

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U", dim_space, nroot));
    evals.reset(new Vector("e", nroot));

    // Diagonalize H
    SigmaVectorList svl(space, print_details_);
    SigmaVector* sigma_vector = &svl;
    sigma_vector->add_bad_roots(bad_states_);
    davidson_liu_solver(space, sigma_vector, evals, evecs, nroot, multiplicity);
}

    
void SparseCISolver::diagonalize_davidson_liu_string(
    const std::vector<STLBitsetDeterminant>& space, SharedVector& evals,
    SharedMatrix& evecs, int nroot, int multiplicity, bool disk) {
    if (print_details_ and !disk) {
        outfile->Printf("\n\n  Davidson-Liu String algorithm");
        outfile->Flush();
    } else if (print_details_ and disk) {
        outfile->Printf("\n\n  Disk-based Davidson-Liu String Algorithm");
    }

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U", dim_space, nroot));
    evals.reset(new Vector("e", nroot));

    // Diagonalize H
    SigmaVectorString svl(space, print_details_, disk);
    SigmaVector* sigma_vector = &svl;
    sigma_vector->add_bad_roots(bad_states_);
    davidson_liu_solver(space, sigma_vector, evals, evecs, nroot, multiplicity);
}

SharedMatrix SparseCISolver::build_full_hamiltonian(
    const std::vector<STLBitsetDeterminant>& space) {
    // Build the H matrix
    size_t dim_space = space.size();
    SharedMatrix H(new Matrix("H", dim_space, dim_space));
    // If you are using DiskDF, Kevin found that openmp does not like this!
    int threads = 0;
    if (STLBitsetDeterminant::fci_ints_->get_integral_type() == DiskDF) {
        threads = 1;
    } else {
        threads = omp_get_max_threads();
    }
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (size_t I = 0; I < dim_space; ++I) {
        const STLBitsetDeterminant& detI = space[I];
        for (size_t J = I; J < dim_space; ++J) {
            const STLBitsetDeterminant& detJ = space[J];
            double HIJ = detI.slater_rules(detJ);
            H->set(I, J, HIJ);
            H->set(J, I, HIJ);
        }
    }

    if (root_project_) {
        // Form the projection matrix
        for (int n = 0, max_n = bad_states_.size(); n < max_n; ++n) {
            SharedMatrix P(new Matrix("P", dim_space, dim_space));
            P->identity();
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            for (size_t det1 = 0, ndet = bad_state.size(); det1 < ndet;
                 ++det1) {
                for (size_t det2 = 0; det2 < ndet; ++det2) {
                    size_t& I = bad_state[det1].first;
                    size_t& J = bad_state[det2].first;
                    double& el1 = bad_state[det1].second;
                    double& el2 = bad_state[det2].second;
                    P->set(I, J, P->get(I, J) - el1 * el2);
                }
            }
            H->transform(P);
        }
    }

    return H;
}

std::vector<std::pair<std::vector<int>, std::vector<double>>>
SparseCISolver::build_sparse_hamiltonian(
    const std::vector<STLBitsetDeterminant>& space) {
    Timer t_h_build2;
    // Allocate as many elements as we need
    size_t dim_space = space.size();
    std::vector<std::pair<std::vector<int>, std::vector<double>>> H_sparse(
        dim_space);

    size_t num_nonzero = 0;

    outfile->Printf("\n  Building H using OpenMP-take2");
    outfile->Flush();

// Form the Hamiltonian matrix

#pragma omp parallel for schedule(dynamic)
    for (size_t I = 0; I < dim_space; ++I) {
        std::vector<double> H_row;
        std::vector<int> index_row;
        const STLBitsetDeterminant& detI = space[I];
        double HII = detI.slater_rules(detI);
        H_row.push_back(HII);
        index_row.push_back(I);
        for (size_t J = 0; J < dim_space; ++J) {
            if (I != J) {
                const STLBitsetDeterminant detJ = space[J];
                double HIJ = detI.slater_rules(detJ);
                if (std::fabs(HIJ) >= 1.0e-12) {
                    H_row.push_back(HIJ);
                    index_row.push_back(J);
                }
            }
        }

#pragma omp critical(save_h_row)
        {
            H_sparse[I] = make_pair(index_row, H_row);
            num_nonzero += index_row.size();
        }
    }
    outfile->Printf("\n  The sparse Hamiltonian matrix contains %zu nonzero "
                    "elements out of %zu (%f)",
                    num_nonzero, dim_space * dim_space,
                    double(num_nonzero) / double(dim_space * dim_space));
    outfile->Printf("\n  %s: %f s", "Time spent building H (openmp)",
                    t_h_build2.get());
    outfile->Flush();
    return H_sparse;
}

std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>>
SparseCISolver::initial_guess(const std::vector<STLBitsetDeterminant>& space,
                              int nroot, int multiplicity) {
    size_t ndets = space.size();
    size_t nguess = std::min(static_cast<size_t>(nroot) * dl_guess_, ndets);
    std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>>
        guess(nguess);

    // Find the ntrial lowest diagonals
    std::vector<std::pair<STLBitsetDeterminant, size_t>> guess_dets_pos;
    std::vector<std::pair<double, size_t>> smallest(ndets);
    for (size_t I = 0; I < ndets; ++I) {
        smallest[I] = std::make_pair(space[I].energy(), I);
    }
    std::sort(smallest.begin(), smallest.end());

    std::vector<STLBitsetDeterminant> guess_det;
    for (size_t i = 0; i < nguess; i++) {
        size_t I = smallest[i].second;
        guess_dets_pos.push_back(
            std::make_pair(space[I], I)); // store a det and its position
        guess_det.push_back(space[I]);
    }

    if (spin_project_) {
        STLBitsetDeterminant::enforce_spin_completeness(guess_det);
        if (guess_det.size() > nguess) {
            size_t nnew_dets = guess_det.size() - nguess;
            if (print_details_)
                outfile->Printf("\n  Initial guess space is incomplete.\n  "
                                "Trying to add %d determinant(s).",
                                nnew_dets);
            int nfound = 0;
            for (size_t i = 0; i < nnew_dets; ++i) {
                for (size_t j = nguess; j < ndets; ++j) {
                    size_t J = smallest[j].second;
                    if (space[J] == guess_det[nguess + i]) {
                        guess_dets_pos.push_back(std::make_pair(
                            space[J], J)); // store a det and its position
                        nfound++;
                        break;
                    }
                }
            }
            if (print_details_)
                outfile->Printf("  %d determinant(s) added.", nfound);
        }
        nguess = guess_dets_pos.size();
    }

    // Form the S^2 operator matrix and diagonalize it
    Matrix S2("S^2", nguess, nguess);
    for (size_t I = 0; I < nguess; I++) {
        for (size_t J = I; J < nguess; J++) {
            const STLBitsetDeterminant& detI = guess_dets_pos[I].first;
            const STLBitsetDeterminant& detJ = guess_dets_pos[J].first;
            double S2IJ = detI.spin2(detJ);
            S2.set(I, J, S2IJ);
            S2.set(J, I, S2IJ);
        }
    }
    Matrix S2evecs("S^2", nguess, nguess);
    Vector S2evals("S^2", nguess);
    S2.diagonalize(S2evecs, S2evals);

    // Form the Hamiltonian
    Matrix H("H", nguess, nguess);
    for (size_t I = 0; I < nguess; I++) {
        for (size_t J = I; J < nguess; J++) {
            const STLBitsetDeterminant& detI = guess_dets_pos[I].first;
            const STLBitsetDeterminant& detJ = guess_dets_pos[J].first;
            double HIJ = detI.slater_rules(detJ);
            H.set(I, J, HIJ);
            H.set(J, I, HIJ);
        }
    }
    // H.print();
    // Project H onto the spin-adapted subspace
    H.transform(S2evecs);

    // Find groups of solutions with same spin
    double Stollerance = 1.0e-6;
    std::map<int, std::vector<int>> mult_list;
    for (size_t i = 0; i < nguess; ++i) {
        double mult = std::sqrt(
            1.0 + 4.0 * S2evals.get(i)); // 2S + 1 = Sqrt(1 + 4 S (S + 1))
        int mult_int = std::round(mult);
        double error = mult - static_cast<double>(mult_int);
        if (std::fabs(error) < Stollerance) {
            mult_list[mult_int].push_back(i);
        } else if (print_details_) {
            outfile->Printf("\n  Found a guess vector with spin not close to "
                            "integer value (%f)",
                            mult);
        }
    }
    if (mult_list[multiplicity].size() < static_cast<size_t>(nroot)) {
        size_t nfound = mult_list[multiplicity].size();
        outfile->Printf("\n  Error: %d guess vectors with 2S+1 = %d but only "
                        "%d were found!",
                        nguess, multiplicity, nfound);
        if (nfound == 0) {
            exit(1);
        }
    }

    std::vector<int> mult_vals;
    for (auto kv : mult_list) {
        mult_vals.push_back(kv.first);
    }
    std::sort(mult_vals.begin(), mult_vals.end());

    for (int m : mult_vals) {
        std::vector<int>& mult_list_s = mult_list[m];
        int nspin_states = mult_list_s.size();
        if (print_details_)
            outfile->Printf(
                "\n  Initial guess found %d solutions with 2S+1 = %d %c",
                nspin_states, m, m == multiplicity ? '*' : ' ');
        // Extract the spin manifold
        Matrix HS2("HS2", nspin_states, nspin_states);
        Vector HS2evals("HS2", nspin_states);
        Matrix HS2evecs("HS2", nspin_states, nspin_states);
        for (int I = 0; I < nspin_states; I++) {
            for (int J = 0; J < nspin_states; J++) {
                HS2.set(I, J, H.get(mult_list_s[I], mult_list_s[J]));
            }
        }
        HS2.diagonalize(HS2evecs, HS2evals);

        // Project the spin-adapted solution onto the full manifold
        for (int r = 0; r < nspin_states; ++r) {
            std::vector<std::pair<size_t, double>> det_C;
            for (size_t I = 0; I < nguess; I++) {
                double CIr = 0.0;
                for (int J = 0; J < nspin_states; ++J) {
                    CIr += S2evecs.get(I, mult_list_s[J]) * HS2evecs(J, r);
                }
                det_C.push_back(std::make_pair(guess_dets_pos[I].second, CIr));
            }
            guess.push_back(std::make_pair(m, det_C));
        }
    }

    return guess;
}

std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>>
SparseCISolver::initial_guess_map(const DeterminantMap& space, int nroot,
                                  int multiplicity) {
    size_t ndets = space.size();
    size_t nguess = std::min(static_cast<size_t>(nroot) * dl_guess_, ndets);
    std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>>
        guess(nguess);

    // Find the ntrial lowest diagonals
    std::vector<std::pair<STLBitsetDeterminant, size_t>> guess_dets_pos;
    std::vector<std::pair<double, STLBitsetDeterminant>> smallest;
    const det_hash<size_t>& detmap = space.wfn_hash();

    for (det_hash<size_t>::const_iterator it = detmap.begin(),
                                          endit = detmap.end();
         it != endit; ++it) {
        smallest.push_back(std::make_pair(it->first.energy(), it->first));
    }
    std::sort(smallest.begin(), smallest.end());

    std::vector<STLBitsetDeterminant> guess_det;
    for (size_t i = 0; i < nguess; i++) {
        STLBitsetDeterminant detI = smallest[i].second;
        guess_dets_pos.push_back(std::make_pair(
            detI, space.get_idx(detI))); // store a det and its position
        guess_det.push_back(detI);
    }

    if (spin_project_) {
        STLBitsetDeterminant::enforce_spin_completeness(guess_det);
        if (guess_det.size() > nguess) {
            size_t nnew_dets = guess_det.size() - nguess;
            if (print_details_)
                outfile->Printf("\n  Initial guess space is incomplete.\n  "
                                "Trying to add %d determinant(s).",
                                nnew_dets);
            int nfound = 0;
            for (size_t i = 0; i < nnew_dets; ++i) {
                for (size_t j = nguess; j < ndets; ++j) {
                    STLBitsetDeterminant detJ = smallest[j].second;
                    if (detJ == guess_det[nguess + i]) {
                        guess_dets_pos.push_back(std::make_pair(
                            detJ, space.get_idx(
                                      detJ))); // store a det and its position
                        nfound++;
                        break;
                    }
                }
            }
            if (print_details_)
                outfile->Printf("  %d determinant(s) added.", nfound);
        }
        nguess = guess_dets_pos.size();
    }

    // Form the S^2 operator matrix and diagonalize it
    Matrix S2("S^2", nguess, nguess);
    for (size_t I = 0; I < nguess; I++) {
        for (size_t J = I; J < nguess; J++) {
            const STLBitsetDeterminant& detI = guess_dets_pos[I].first;
            const STLBitsetDeterminant& detJ = guess_dets_pos[J].first;
            double S2IJ = detI.spin2(detJ);
            S2.set(I, J, S2IJ);
            S2.set(J, I, S2IJ);
        }
    }
    Matrix S2evecs("S^2", nguess, nguess);
    Vector S2evals("S^2", nguess);
    S2.diagonalize(S2evecs, S2evals);

    // Form the Hamiltonian
    Matrix H("H", nguess, nguess);
    for (size_t I = 0; I < nguess; I++) {
        for (size_t J = I; J < nguess; J++) {
            const STLBitsetDeterminant& detI = guess_dets_pos[I].first;
            const STLBitsetDeterminant& detJ = guess_dets_pos[J].first;
            double HIJ = detI.slater_rules(detJ);
            H.set(I, J, HIJ);
            H.set(J, I, HIJ);
        }
    }
    // H.print();
    // Project H onto the spin-adapted subspace
    H.transform(S2evecs);

    // Find groups of solutions with same spin
    double Stollerance = 1.0e-6;
    std::map<int, std::vector<int>> mult_list;
    for (size_t i = 0; i < nguess; ++i) {
        double mult = std::sqrt(
            1.0 + 4.0 * S2evals.get(i)); // 2S + 1 = Sqrt(1 + 4 S (S + 1))
        int mult_int = std::round(mult);
        double error = mult - static_cast<double>(mult_int);
        if (std::fabs(error) < Stollerance) {
            mult_list[mult_int].push_back(i);
        } else if (print_details_) {
            outfile->Printf("\n  Found a guess vector with spin not close to "
                            "integer value (%f)",
                            mult);
        }
    }
    if (mult_list[multiplicity].size() < static_cast<size_t>(nroot)) {
        size_t nfound = mult_list[multiplicity].size();
        outfile->Printf("\n  Error: %d guess vectors with 2S+1 = %d but only "
                        "%d were found!",
                        nguess, multiplicity, nfound);
        if (nfound == 0) {
            exit(1);
        }
    }

    std::vector<int> mult_vals;
    for (auto kv : mult_list) {
        mult_vals.push_back(kv.first);
    }
    std::sort(mult_vals.begin(), mult_vals.end());

    for (int m : mult_vals) {
        std::vector<int>& mult_list_s = mult_list[m];
        int nspin_states = mult_list_s.size();
        if (print_details_)
            outfile->Printf(
                "\n  Initial guess found %d solutions with 2S+1 = %d %c",
                nspin_states, m, m == multiplicity ? '*' : ' ');
        // Extract the spin manifold
        Matrix HS2("HS2", nspin_states, nspin_states);
        Vector HS2evals("HS2", nspin_states);
        Matrix HS2evecs("HS2", nspin_states, nspin_states);
        for (int I = 0; I < nspin_states; I++) {
            for (int J = 0; J < nspin_states; J++) {
                HS2.set(I, J, H.get(mult_list_s[I], mult_list_s[J]));
            }
        }
        HS2.diagonalize(HS2evecs, HS2evals);

        // Project the spin-adapted solution onto the full manifold
        for (int r = 0; r < nspin_states; ++r) {
            std::vector<std::pair<size_t, double>> det_C;
            for (size_t I = 0; I < nguess; I++) {
                double CIr = 0.0;
                for (int J = 0; J < nspin_states; ++J) {
                    CIr += S2evecs.get(I, mult_list_s[J]) * HS2evecs(J, r);
                }
                det_C.push_back(std::make_pair(guess_dets_pos[I].second, CIr));
            }
            guess.push_back(std::make_pair(m, det_C));
        }
    }

    return guess;
}

void SparseCISolver::add_bad_states(
    std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SparseCISolver::set_root_project(bool value) { root_project_ = value; }

void SparseCISolver::manual_guess(bool value) { set_guess_ = value; }

void SparseCISolver::set_initial_guess(
    std::vector<std::pair<size_t, double>>& guess) {
    set_guess_ = true;
    guess_.clear();

    for (size_t I = 0, max_I = guess.size(); I < max_I; ++I) {
        guess_.push_back(guess[I]);
    }
}

void SparseCISolver::set_num_vecs(size_t value) { nvec_ = value; }

bool SparseCISolver::davidson_liu_solver(
    const std::vector<STLBitsetDeterminant>& space, SigmaVector* sigma_vector,
    SharedVector Eigenvalues, SharedMatrix Eigenvectors, int nroot,
    int multiplicity) {
    //    print_details_ = true;
    size_t fci_size = sigma_vector->size();
    DavidsonLiuSolver dls(fci_size, nroot);
    dls.set_e_convergence(e_convergence_);
    dls.set_print_level(0);

    // allocate vectors
    SharedVector b(new Vector("b", fci_size));
    SharedVector sigma(new Vector("sigma", fci_size));

    // get and pass diagonal
    sigma_vector->get_diagonal(*sigma);
    dls.startup(sigma);

    std::vector<std::vector<std::pair<size_t, double>>> bad_roots;
    size_t guess_size = std::min(nvec_, dls.collapse_size());

    auto guess = initial_guess(space, nroot, multiplicity);
    if (!set_guess_) {
        std::vector<int> guess_list;
        for (size_t g = 0; g < guess.size(); ++g) {
            if (guess[g].first == multiplicity)
                guess_list.push_back(g);
        }

        // number of guess to be used
        size_t nguess = std::min(guess_list.size(), guess_size);

        if (nguess == 0) {
            throw PSIEXCEPTION("\n\n  Found zero FCI guesses with the "
                               "requested multiplicity.\n\n");
        }

        for (size_t n = 0; n < nguess; ++n) {
            b->zero();
            for (auto& guess_vec_info : guess[guess_list[n]].second) {
                b->set(guess_vec_info.first, guess_vec_info.second);
            }
            if (print_details_)
                outfile->Printf("\n  Adding guess %d (multiplicity = %f)", n,
                                guess[guess_list[n]].first);

            dls.add_guess(b);
        }
    }

    // Prepare a list of bad roots to project out and pass them to the solver
    for (auto& g : guess) {
        if (g.first != multiplicity)
            bad_roots.push_back(g.second);
    }
    dls.set_project_out(bad_roots);

    if (set_guess_) {
        // Use previous solution as guess
        b->zero();
        for (size_t I = 0, max_I = guess_.size(); I < max_I; ++I) {
            b->set(guess_[I].first, guess_[I].second);
        }
        double norm = sqrt(1.0 / b->norm());
        b->scale(norm);
        dls.add_guess(b);
    }

    SolverStatus converged = SolverStatus::NotConverged;

    if (print_details_) {
        outfile->Printf("\n\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  ----------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E");
        outfile->Printf("\n  ----------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;

    //    maxiter_davidson_ = 2;
    //    b->print();
    for (int cycle = 0; cycle < maxiter_davidson_; ++cycle) {
        bool add_sigma = true;
        do {
            dls.get_b(b);
            sigma_vector->compute_sigma(sigma, b);

            add_sigma = dls.add_sigma(sigma);
        } while (add_sigma);

        converged = dls.update();

        if (converged != SolverStatus::Collapse) {
            double avg_energy = 0.0;
            for (int r = 0; r < nroot; ++r)
                avg_energy += dls.eigenvalues()->get(r);
            avg_energy /= static_cast<double>(nroot);
            if (print_details_) {
                outfile->Printf("\n    %3d  %20.12f  %+.3e", real_cycle,
                                avg_energy, avg_energy - old_avg_energy);
            }
            old_avg_energy = avg_energy;
            real_cycle++;
        }

        if (converged == SolverStatus::Converged)
            break;
    }

    if (print_details_) {
        outfile->Printf("\n  ----------------------------------------");
        if (converged == SolverStatus::Converged) {
            outfile->Printf(
                "\n  The Davidson-Liu algorithm converged in %d iterations.",
                real_cycle);
        }
    }

    if (converged == SolverStatus::NotConverged) {
        outfile->Printf("\n  FCI did not converge!");
        exit(1);
    }

    //    dls.get_results();
    SharedVector evals = dls.eigenvalues();
    SharedMatrix evecs = dls.eigenvectors();
    for (int r = 0; r < nroot; ++r) {
        Eigenvalues->set(r, evals->get(r));
        for (size_t I = 0; I < fci_size; ++I) {
            Eigenvectors->set(I, r, evecs->get(r, I));
        }
    }
    return true;
}

bool SparseCISolver::davidson_liu_solver_map(const DeterminantMap& space,
                                             SigmaVector* sigma_vector,
                                             SharedVector Eigenvalues,
                                             SharedMatrix Eigenvectors,
                                             int nroot, int multiplicity) {
    //    print_details_ = true;
    size_t fci_size = sigma_vector->size();
    DavidsonLiuSolver dls(fci_size, nroot);
    dls.set_e_convergence(e_convergence_);
    dls.set_print_level(0);

    // allocate vectors
    SharedVector b(new Vector("b", fci_size));
    SharedVector sigma(new Vector("sigma", fci_size));

    // get and pass diagonal
    sigma_vector->get_diagonal(*sigma);
    dls.startup(sigma);

    std::vector<std::vector<std::pair<size_t, double>>> bad_roots;
    size_t guess_size = std::min(nvec_, dls.collapse_size());

    auto guess = initial_guess_map(space, nroot, multiplicity);
    if (!set_guess_) {
        std::vector<int> guess_list;
        for (size_t g = 0; g < guess.size(); ++g) {
            if (guess[g].first == multiplicity)
                guess_list.push_back(g);
        }

        // number of guess to be used
        size_t nguess = std::min(guess_list.size(), guess_size);

        if (nguess == 0) {
            throw PSIEXCEPTION("\n\n  Found zero FCI guesses with the "
                               "requested multiplicity.\n\n");
        }

        for (size_t n = 0; n < nguess; ++n) {
            b->zero();
            for (auto& guess_vec_info : guess[guess_list[n]].second) {
                b->set(guess_vec_info.first, guess_vec_info.second);
            }
            if (print_details_)
                outfile->Printf("\n  Adding guess %d (multiplicity = %f)", n,
                                guess[guess_list[n]].first);

            dls.add_guess(b);
        }
    }

    // Prepare a list of bad roots to project out and pass them to the solver
    for (auto& g : guess) {
        if (g.first != multiplicity)
            bad_roots.push_back(g.second);
    }
    dls.set_project_out(bad_roots);

    if (set_guess_) {
        // Use previous solution as guess
        b->zero();
        for (size_t I = 0, max_I = guess_.size(); I < max_I; ++I) {
            b->set(guess_[I].first, guess_[I].second);
        }
        double norm = sqrt(1.0 / b->norm());
        b->scale(norm);
        dls.add_guess(b);
    }

    SolverStatus converged = SolverStatus::NotConverged;

    if (print_details_) {
        outfile->Printf("\n\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  ----------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E");
        outfile->Printf("\n  ----------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;

    //    maxiter_davidson_ = 2;
    //    b->print();
    for (int cycle = 0; cycle < maxiter_davidson_; ++cycle) {
        bool add_sigma = true;
        do {
            dls.get_b(b);
            sigma_vector->compute_sigma(sigma, b);

            add_sigma = dls.add_sigma(sigma);
        } while (add_sigma);

        converged = dls.update();

        if (converged != SolverStatus::Collapse) {
            double avg_energy = 0.0;
            for (int r = 0; r < nroot; ++r)
                avg_energy += dls.eigenvalues()->get(r);
            avg_energy /= static_cast<double>(nroot);
            if (print_details_) {
                outfile->Printf("\n    %3d  %20.12f  %+.3e", real_cycle,
                                avg_energy, avg_energy - old_avg_energy);
            }
            old_avg_energy = avg_energy;
            real_cycle++;
        }

        if (converged == SolverStatus::Converged)
            break;
    }

    if (print_details_) {
        outfile->Printf("\n  ----------------------------------------");
        if (converged == SolverStatus::Converged) {
            outfile->Printf(
                "\n  The Davidson-Liu algorithm converged in %d iterations.",
                real_cycle);
        }
    }

    if (converged == SolverStatus::NotConverged) {
        outfile->Printf("\n  FCI did not converge!");
        exit(1);
    }

    //    dls.get_results();
    SharedVector evals = dls.eigenvalues();
    SharedMatrix evecs = dls.eigenvectors();
    for (int r = 0; r < nroot; ++r) {
        Eigenvalues->set(r, evals->get(r));
        for (size_t I = 0; I < fci_size; ++I) {
            Eigenvectors->set(I, r, evecs->get(r, I));
        }
    }
    return true;
}
}
}
