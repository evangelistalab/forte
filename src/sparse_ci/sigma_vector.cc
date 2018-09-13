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

#include <cmath>

#include "psi4/libciomr/libciomr.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "../forte-def.h"
#include "../iterative_solvers.h"
#include "sigma_vector.h"

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
SigmaVectorMPI::SigmaVectorMPI(const DeterminantHashVec& space, WFNOperator& op)
    : SigmaVector(space.size()), space_(space) {}

void SigmaVectorMPI::compute_sigma(SharedVector sigma, SharedVector b) {}

#endif


inline double clamp(double x, double a, double b)

{
    return x < a ? a : (x > b ? b : x);
}

/**
 * @brief smootherstep
 * @param edge0
 * @param edge1
 * @param x
 * @return
 *
 * This is a smooth step function that is
 * 0.0 for x <= edge0
 * 1.0 for x >= edge1
 */
inline double smootherstep(double edge0, double edge1, double x) {
    if (edge1 == edge0) {
        return x <= edge0 ? 0.0 : 1.0;
    }
    // Scale, and clamp x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    // Evaluate polynomial
    return x * x * x * (x * (x * 6. - 15.) + 10.);
}

SigmaVectorList::SigmaVectorList(const std::vector<Determinant>& space, bool print_details,
                                 std::shared_ptr<FCIIntegrals> fci_ints)
    : SigmaVector(space.size()), space_(space), fci_ints_(fci_ints) {
    using det_hash = std::unordered_map<Determinant, size_t, Determinant::Hash>;
    using bstmap_it = det_hash::iterator;

    size_t max_I = space.size();
    size_t ncmo = fci_ints_->nmo();

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
            const Determinant& detI = space[I];
            double EI = fci_ints_->energy(detI);
            diag_.push_back(EI);

            std::vector<int> aocc = detI.get_alfa_occ(ncmo);
            int noalpha = aocc.size();

            std::vector<std::pair<size_t, short>> a_ann(noalpha);

            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                Determinant detJ(detI);
                detJ.set_alfa_bit(ii, false);

                double sign = detI.slater_sign_a(ii);

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
                a_ann[i] = std::make_pair(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1));
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
            const Determinant& detI = space[I];

            std::vector<int> bocc = detI.get_beta_occ(ncmo);
            int nobeta = bocc.size();

            std::vector<std::pair<size_t, short>> b_ann(nobeta);

            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                Determinant detJ(detI);
                detJ.set_beta_bit(ii, false);

                double sign = detI.slater_sign_b(ii);

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
                b_ann[i] = std::make_pair(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1));
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
    size_t mem_tuple_singles = a_cre_list.capacity() * (sizeof(size_t) + sizeof(short));
    mem_tuple_singles += b_cre_list.capacity() * (sizeof(size_t) + sizeof(short));

    //    outfile->Printf("\n  Size of lists:");
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",a_ann_list.size());
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",b_ann_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",a_cre_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",b_cre_list.size());
    if (print_details) {
        outfile->Printf("\n  Time spent building single lists: %f s", single.get());
        outfile->Printf("\n  Memory for single-hole lists: %f MB",
                        double(mem_tuple_singles) / (1024. * 1024.)); // Convert to MB
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
            const Determinant& detI = space[I];

            std::vector<int> aocc = detI.get_alfa_occ(ncmo);
            size_t noalpha = aocc.size();

            std::vector<std::tuple<size_t, short, short>> aa_ann(noalpha * (noalpha - 1) / 2);

            for (size_t i = 0, ij = 0; i < noalpha; ++i) {
                for (size_t j = i + 1; j < noalpha; ++j, ++ij) {
                    int ii = aocc[i];
                    int jj = aocc[j];
                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_alfa_bit(jj, false);

                    double sign = detI.slater_sign_a(ii) * detI.slater_sign_a(jj);

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
                    aa_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
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
            const Determinant& detI = space[I];

            std::vector<int> bocc = detI.get_beta_occ(ncmo);

            size_t nobeta = bocc.size();

            std::vector<std::tuple<size_t, short, short>> bb_ann(nobeta * (nobeta - 1) / 2);
            for (size_t i = 0, ij = 0; i < nobeta; ++i) {
                for (size_t j = i + 1; j < nobeta; ++j, ++ij) {
                    int ii = bocc[i];
                    int jj = bocc[j];
                    Determinant detJ(detI);
                    detJ.set_beta_bit(ii, false);
                    detJ.set_beta_bit(jj, false);

                    double sign = detI.slater_sign_b(ii) * detI.slater_sign_b(jj);

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
                    bb_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
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
            const Determinant& detI = space[I];

            std::vector<int> aocc = detI.get_alfa_occ(ncmo);
            std::vector<int> bocc = detI.get_beta_occ(ncmo);

            size_t noalpha = aocc.size();
            size_t nobeta = bocc.size();

            std::vector<std::tuple<size_t, short, short>> ab_ann(noalpha * nobeta);
            for (size_t i = 0, ij = 0; i < noalpha; ++i) {
                for (size_t j = 0; j < nobeta; ++j, ++ij) {
                    int ii = aocc[i];
                    int jj = bocc[j];
                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_beta_bit(jj, false);

                    double sign = detI.slater_sign_a(ii) * detI.slater_sign_b(jj);

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
                    ab_ann[ij] = std::make_tuple(detJ_add, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj);
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
        const std::vector<std::tuple<size_t, short, short>>& aa_ann = aa_ann_list[I];
        for (const std::tuple<size_t, short, short>& J_sign : aa_ann) {
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            aa_cre_list[J].push_back(std::make_tuple(I, i, j));
        }
        const std::vector<std::tuple<size_t, short, short>>& bb_ann = bb_ann_list[I];
        for (const std::tuple<size_t, short, short>& J_sign : bb_ann) {
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            bb_cre_list[J].push_back(std::make_tuple(I, i, j));
        }
        const std::vector<std::tuple<size_t, short, short>>& ab_ann = ab_ann_list[I];
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

    size_t mem_tuple_doubles = aa_cre_list.capacity() * (sizeof(size_t) + 2 * sizeof(short));
    mem_tuple_doubles += bb_cre_list.capacity() * (sizeof(size_t) + 2 * sizeof(short));
    mem_tuple_doubles += ab_cre_list.capacity() * (sizeof(size_t) + 2 * sizeof(short));

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
        outfile->Printf("\n  Memory for double-hole maps:  %f MB", d_map_size / (1024. * 1024.));
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

        size_t start_idx =
            (tid < (size_ % num_thread))
                ? tid * bin_size
                : (size_ % num_thread) * (bin_size + 1) + (tid - (size_ % num_thread)) * bin_size;
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
                            fci_ints_->slater_rules_single_alpha_abs(space_[I], p, q) * sign_p *
                            sign_q;
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
                            fci_ints_->slater_rules_single_beta_abs(space_[I], p, q) * sign_p *
                            sign_q;
                        sigma_p[J] += HIJ * b_p[I];
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
                        const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const size_t I = aaaaJ_add;
                        const double HIJ = sign_pq * sign_rs * fci_ints_->tei_aa(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
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
                        const double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const size_t I = ababJ_add;
                        const double HIJ = sign_pq * sign_rs * fci_ints_->tei_ab(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
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
                        const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const size_t I = bbbbJ_add;
                        const double HIJ = sign_pq * sign_rs * fci_ints_->tei_bb(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
        }
    }
}

void SigmaVectorList::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
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

SigmaVectorWfn1::SigmaVectorWfn1(const DeterminantHashVec& space, WFNOperator& op,
                                 std::shared_ptr<FCIIntegrals> fci_ints)
    : SigmaVector(space.size()), space_(space), a_ann_list_(op.a_ann_list_), fci_ints_(fci_ints),
      a_cre_list_(op.a_cre_list_), b_ann_list_(op.b_ann_list_), b_cre_list_(op.b_cre_list_),
      aa_ann_list_(op.aa_ann_list_), aa_cre_list_(op.aa_cre_list_), ab_ann_list_(op.ab_ann_list_),
      ab_cre_list_(op.ab_cre_list_), bb_ann_list_(op.bb_ann_list_), bb_cre_list_(op.bb_cre_list_) {

    const det_hashvec& detmap = space_.wfn_hash();
    diag_.resize(space_.size());
    for (size_t I = 0, max_I = detmap.size(); I < max_I; ++I) {
        diag_[I] = fci_ints_->energy(detmap[I]);
    }
}

void SigmaVectorWfn1::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SigmaVectorWfn1::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorWfn1::compute_sigma(SharedVector sigma, SharedVector b) {
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

        size_t start_idx =
            (tid < (size_ % num_thread))
                ? tid * bin_size
                : (size_ % num_thread) * (bin_size + 1) + (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;
        // Timer cycl;
        {
            const det_hashvec& dets = space_.wfn_hash();
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
                            double sign_q = aaJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                fci_ints_->slater_rules_single_alpha_abs(dets[I], p, q) * sign_p *
                                sign_q;
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
                            double sign_q = bbJ_mo_sign.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                fci_ints_->slater_rules_single_beta_abs(dets[I], p, q) * sign_p *
                                sign_q;
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
                const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
                const size_t q = std::get<2>(aaJ_mo_sign);
                for (auto& aaaaJ_mo_sign : aa_cre_list_[aaJ_add]) {
                    const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(aaaaJ_mo_sign);
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        const size_t I = std::get<0>(aaaaJ_mo_sign);
                        const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ = sign_pq * sign_rs * fci_ints_->tei_aa(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // aabb singles
            for (auto& abJ_mo_sign : ab_ann_list_[J]) {
                const size_t abJ_add = std::get<0>(abJ_mo_sign);
                const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
                const size_t q = std::get<2>(abJ_mo_sign);
                for (auto& ababJ_mo_sign : ab_cre_list_[abJ_add]) {
                    const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(ababJ_mo_sign);
                    if ((p != r) and (q != s)) {
                        const size_t I = std::get<0>(ababJ_mo_sign);
                        const double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ = sign_pq * sign_rs * fci_ints_->tei_ab(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
            // bbbb singles
            for (auto& bbJ_mo_sign : bb_ann_list_[J]) {
                const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
                const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
                const size_t q = std::get<2>(bbJ_mo_sign);
                for (auto& bbbbJ_mo_sign : bb_cre_list_[bbJ_add]) {
                    const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                    const size_t s = std::get<2>(bbbbJ_mo_sign);
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        const size_t I = std::get<0>(bbbbJ_mo_sign);
                        const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                        const double HIJ = sign_pq * sign_rs * fci_ints_->tei_bb(p, q, r, s);
                        sigma_p[J] += HIJ * b_p[I];
                    }
                }
            }
        }
        // outfile->Printf("\n  Time spent on doubles: %1.5f", cycl2.get());
    }
}

SigmaVectorWfn2::SigmaVectorWfn2(const DeterminantHashVec& space, WFNOperator& op,
                                 std::shared_ptr<FCIIntegrals> fci_ints)
    : SigmaVector(space.size()), space_(space), fci_ints_(fci_ints), a_list_(op.a_list_),
      b_list_(op.b_list_), aa_list_(op.aa_list_), ab_list_(op.ab_list_), bb_list_(op.bb_list_) {

    const det_hashvec& detmap = space_.wfn_hash();
    diag_.resize(space_.size());
    for (size_t I = 0, max_I = detmap.size(); I < max_I; ++I) {
        diag_[I] = fci_ints_->energy(detmap[I]);
    }
}

void SigmaVectorWfn1::set_smooth( int idx, std::vector<double>& smooth_en){}
void SigmaVectorWfn3::set_smooth( int idx, std::vector<double>& smooth_en){}
void SigmaVectorList::set_smooth( int idx, std::vector<double>& smooth_en){}
void SigmaVectorSparse::set_smooth( int idx, std::vector<double>& smooth_en){}

void SigmaVectorWfn2::set_smooth( int idx, std::vector<double>& smooth_en){ 
    smooth_idx_ = idx;
    smooth_en_ = smooth_en;
 }

void SigmaVectorWfn2::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SigmaVectorWfn2::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorWfn2::compute_sigma(SharedVector sigma, SharedVector b) {
    sigma->zero();

    size_t ncmo = fci_ints_->nmo();

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
    for (size_t J = 0; J < size_; ++J) {
        sigma_p[J] += diag_[J] * b_p[J]; // Make DDOT
    }

    auto& dets = space_.wfn_hash();
    std::vector<double> F;

    if( smooth_idx_ >= 0 ){
        int nsmooth = size_ - smooth_idx_;
        F.resize(nsmooth);

        // Get the edge energies
        double E0 = smooth_en_[0];
        double E1 = smooth_en_[nsmooth];
    
        for( int I = 0; I < nsmooth; ++I ){
            double EI = smooth_en_[I];
            F[I] = smootherstep(E1, E0, EI); 
        //    outfile->Printf("\n  Det %zu (%1.9f) scale by %1.8f", I, EI,F[I]);
            b_p[I + smooth_idx_] = b_p[I + smooth_idx_] * F[I];
        } 
    }


#pragma omp parallel
    {
        int num_thread = omp_get_max_threads();
        int tid = omp_get_thread_num();

        // Each thread gets local copy of sigma
        std::vector<double> sigma_t(size_);

        size_t bin_size = size_ / num_thread;
        bin_size += (tid < (size_ % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (size_ % num_thread))
                ? tid * bin_size
                : (size_ % num_thread) * (bin_size + 1) + (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

       // for (size_t J = start_idx; J < end_idx; ++J) {
       //     sigma_p[J] += diag_[J] * b_p[J]; // Make DDOT
       // }

        // a singles
        size_t end_a_idx = a_list_.size();
        size_t start_a_idx = 0;
        for (size_t K = start_a_idx, max_K = end_a_idx; K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::pair<size_t, short>>& c_dets = a_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    const size_t J = detJ.first;
                    const size_t p = std::abs(detJ.second) - 1;
                    double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det+1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        const size_t q = std::abs(detI.second) - 1;
                        if (p != q) {
                            const size_t I = detI.first;
                            double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                fci_ints_->slater_rules_single_alpha_abs(dets[J], p, q) * sign_p *
                                sign_q;
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                            
                        }
                    }
                }
            }
        }

        // b singles
        size_t end_b_idx = b_list_.size();
        size_t start_b_idx = 0;
        for (size_t K = start_b_idx, max_K = end_b_idx; K < max_K; ++K) {
            // aa singles
            if ((K % num_thread) == tid) {
                const std::vector<std::pair<size_t, short>>& c_dets = b_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    const size_t J = detJ.first;
                    const size_t p = std::abs(detJ.second) - 1;
                    double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det+1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        const size_t q = std::abs(detI.second) - 1;
                        if (p != q) {
                            const size_t I = detI.first;
                            double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                fci_ints_->slater_rules_single_beta_abs(dets[J], p, q) * sign_p *
                                sign_q;
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                            
                        }
                    }
                }
            }
        }

        // AA doubles
        size_t aa_size = aa_list_.size();
        //      size_t bin_aa_size = aa_size / num_thread;
        //      bin_aa_size += (tid < (aa_size % num_thread)) ? 1 : 0;
        //      size_t start_aa_idx = (tid < (aa_size % num_thread))
        //                             ? tid * bin_aa_size
        //                             : (aa_size % num_thread) * (bin_aa_size + 1) +
        //                                   (tid - (aa_size % num_thread)) * bin_aa_size;
        //      size_t end_aa_idx = start_aa_idx + bin_aa_size;
        for (size_t K = 0, max_K = aa_size; K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = aa_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det+1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s) and (p != s) and (q != r)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_aa(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }

        // BB doubles
        for (size_t K = 0, max_K = bb_list_.size(); K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = bb_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det+1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s) and (p != s) and (q != r)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_bb(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }
        for (size_t K = 0, max_K = ab_list_.size(); K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = ab_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det+1; det2 < max_det; ++det2) {
                            auto& detI = c_dets[det2];
                            short r = std::abs(std::get<1>(detI)) - 1;
                            short s = std::get<2>(detI);
                        if ((p != r) and (q != s)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_ab(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                            sigma_t[J] += HIJ * b_p[I];
                        }
                    }
                }
            }
        }

        //        #pragma omp critical
        //        {
        for (size_t I = 0, maxI= size_; I < maxI; ++I) {
            if( I < smooth_idx_ ){
#pragma omp atomic update
                sigma_p[I] += sigma_t[I];
            } else {
#pragma omp atomic update
                sigma_p[I] += sigma_t[I] * F[I-smooth_idx_];
            }
        }

    }

   // if( smooth_idx_ >= 0 ){ 
   //     int nsmooth = size_ - smooth_idx_;
   //     for( int I = 0; I < nsmooth; ++I ){
   //         sigma_p[I + smooth_idx_] *= F[I];
   //     }
   // }
}

SigmaVectorWfn3::SigmaVectorWfn3(const DeterminantHashVec& space, WFNOperator& op,
                                 std::shared_ptr<FCIIntegrals> fci_ints)
    : SigmaVector(space.size()), space_(space), fci_ints_(fci_ints), a_list_(op.a_list_),
      b_list_(op.b_list_), aa_list_(op.aa_list_), ab_list_(op.ab_list_), bb_list_(op.bb_list_) {

    const det_hashvec& detmap = space_.wfn_hash();
    diag_.resize(space_.size());
    for (size_t I = 0, max_I = detmap.size(); I < max_I; ++I) {
        diag_[I] = fci_ints_->energy(detmap[I]);
    }

    size_t nact = fci_ints_->nmo();
    size_t nact2 = nact * nact;
    aa_tei_ = SharedMatrix(new Matrix("aa", nact2, nact2));
    bb_tei_ = SharedMatrix(new Matrix("aa", nact2, nact2));
    ab_tei_ = SharedMatrix(new Matrix("aa", nact2, nact2));

    outfile->Printf("\n  Building integral matrices");
    Timer build;
    aa_tei_->zero();
    ab_tei_->zero();
    bb_tei_->zero();
    for (int p = 0; p < nact; ++p) {
        for (int q = 0; q < nact; ++q) {
            for (int r = 0; r < nact; ++r) {
                for (int s = 0; s < nact; ++s) {
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        aa_tei_->set(p * nact + q, r * nact + s, fci_ints_->tei_aa(p, q, r, s));
                    }
                    if ((p != r) and (q != s)) {
                        ab_tei_->set(p * nact + q, r * nact + s, fci_ints_->tei_ab(p, q, r, s));
                    }
                    if ((p != r) and (q != s) and (p != s) and (q != r)) {
                        bb_tei_->set(p * nact + q, r * nact + s, fci_ints_->tei_bb(p, q, r, s));
                    }
                }
            }
        }
    }
    outfile->Printf(" took %1.6f s", build.get());

    size_t nnon = 0;
    size_t maxK = ab_list_.size();
    for (size_t K = 0; K < maxK; ++K) {
        nnon += ab_list_[K].size();
    }

    outfile->Printf("\n  C(rs,K) contains %zu non-zero elements out of %zu (%1.3f %%)", nnon,
                    maxK * nact2, (static_cast<double>(nnon) / (maxK * nact2)) * 100);
}

void SigmaVectorWfn3::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SigmaVectorWfn3::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVectorWfn3::compute_sigma(SharedVector sigma, SharedVector b) {
    sigma->zero();

    size_t ncmo = fci_ints_->nmo();
    size_t ncmo2 = ncmo * ncmo;

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
        size_t start_idx =
            (tid < (size_ % num_thread))
                ? tid * bin_size
                : (size_ % num_thread) * (bin_size + 1) + (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        for (size_t J = start_idx; J < end_idx; ++J) {
            sigma_p[J] += diag_[J] * b_p[J]; // Make DDOT
        }
    }

    // Each thread gets local copy of sigma
    const det_hashvec& dets = space_.wfn_hash();

    // a singles
    size_t end_a_idx = a_list_.size();
    size_t start_a_idx = 0;
    for (size_t K = start_a_idx, max_K = end_a_idx; K < max_K; ++K) {
        for (auto& detJ : a_list_[K]) { // Each gives unique J
            const size_t J = detJ.first;
            const size_t p = std::abs(detJ.second) - 1;
            double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
            for (auto& detI : a_list_[K]) {
                const size_t q = std::abs(detI.second) - 1;
                if (p != q) {
                    const size_t I = detI.first;
                    double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                    const double HIJ =
                        fci_ints_->slater_rules_single_alpha_abs(dets[J], p, q) * sign_p * sign_q;
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }
    }

    // b singles
    size_t end_b_idx = b_list_.size();
    size_t start_b_idx = 0;
    //   size_t bin_b_size = b_size / num_thread;
    //   bin_b_size += (tid < (b_size % num_thread)) ? 1 : 0;
    //   size_t start_b_idx = (tid < (b_size % num_thread))
    //                          ? tid * bin_b_size
    //                          : (b_size % num_thread) * (bin_b_size + 1) +
    //                                (tid - (b_size % num_thread)) * bin_b_size;
    //   size_t end_b_idx = start_b_idx + bin_b_size;
    for (size_t K = start_b_idx, max_K = end_b_idx; K < max_K; ++K) {
        // aa singles
        for (auto& detJ : b_list_[K]) {
            const size_t J = detJ.first;
            const size_t p = std::abs(detJ.second) - 1;
            double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
            for (auto& detI : b_list_[K]) {
                const size_t q = std::abs(detI.second) - 1;
                if (p != q) {
                    const size_t I = detI.first;
                    double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                    const double HIJ =
                        fci_ints_->slater_rules_single_beta_abs(dets[J], p, q) * sign_p * sign_q;
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }
    }

    // AA doubles
    {
        size_t max_K = aa_list_.size();
        SharedMatrix B_pq = SharedMatrix(new Matrix("B_pq", ncmo2, max_K));
        SharedMatrix C_rs = SharedMatrix(new Matrix("C_rs", max_K, ncmo2));
        for (size_t K = 0; K < max_K; ++K) {
            const std::vector<std::tuple<size_t, short, short>>& c_dets = aa_list_[K];
            size_t maxI = c_dets.size();
            for (size_t det = 0; det < maxI; ++det) {
                auto& detJ = c_dets[det];
                const size_t J = std::get<0>(detJ);
                const double sign_rs = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                const size_t r = std::abs(std::get<1>(detJ)) - 1;
                const size_t s = std::get<2>(detJ);
                C_rs->set(K, r * ncmo + s, sign_rs * b_p[J]);
            }
        }
        // Timer mult;
        B_pq->gemm(false, true, 1.0, aa_tei_, C_rs, 0.0);
        // C_DGEMV('N',ncmo2,ncmo2,1.0, &(ab_tei_->pointer()[0][0]),ncmo2,
        // &(C_rs->pointer()[0]),1,0.0,&(B_pq->pointer()[0]),1);
        // outfile->Printf("\n  Time spent on GEMV: %1.6f", mult.get());
        for (size_t K = 0; K < max_K; ++K) {
            auto& c_dets = aa_list_[K];
            size_t maxI = c_dets.size();
            for (size_t det = 0; det < maxI; ++det) {
                auto& detI = c_dets[det];
                const size_t p = std::abs(std::get<1>(detI)) - 1;
                const size_t q = std::get<2>(detI);
                const size_t I = std::get<0>(detI);
                const double sign_pq = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                sigma_p[I] += sign_pq * B_pq->get(p * ncmo + q, K);
            }
        }
    }

    // BB doubles
    {
        size_t max_K = bb_list_.size();
        SharedMatrix B_pq = SharedMatrix(new Matrix("B_pq", ncmo2, max_K));
        SharedMatrix C_rs = SharedMatrix(new Matrix("C_rs", max_K, ncmo2));
        for (size_t K = 0; K < max_K; ++K) {
            const std::vector<std::tuple<size_t, short, short>>& c_dets = bb_list_[K];
            size_t maxI = c_dets.size();
            for (size_t det = 0; det < maxI; ++det) {
                auto& detJ = c_dets[det];
                const size_t J = std::get<0>(detJ);
                const double sign_rs = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                const size_t r = std::abs(std::get<1>(detJ)) - 1;
                const size_t s = std::get<2>(detJ);
                C_rs->set(K, r * ncmo + s, sign_rs * b_p[J]);
            }
        }
        // Timer mult;
        B_pq->gemm(false, true, 1.0, bb_tei_, C_rs, 0.0);
        // C_DGEMV('N',ncmo2,ncmo2,1.0, &(ab_tei_->pointer()[0][0]),ncmo2,
        // &(C_rs->pointer()[0]),1,0.0,&(B_pq->pointer()[0]),1);
        // outfile->Printf("\n  Time spent on GEMV: %1.6f", mult.get());
        for (size_t K = 0; K < max_K; ++K) {
            auto& c_dets = bb_list_[K];
            size_t maxI = c_dets.size();
            for (size_t det = 0; det < maxI; ++det) {
                auto& detI = c_dets[det];
                const size_t p = std::abs(std::get<1>(detI)) - 1;
                const size_t q = std::get<2>(detI);
                const size_t I = std::get<0>(detI);
                const double sign_pq = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                sigma_p[I] += sign_pq * B_pq->get(p * ncmo + q, K);
            }
        }
    }
    // AB doubles
    {
        size_t max_K = ab_list_.size();
        SharedMatrix B_pq = SharedMatrix(new Matrix("B_pq", ncmo2, max_K));
        SharedMatrix C_rs = SharedMatrix(new Matrix("C_rs", max_K, ncmo2));

        // Timer AB;
        B_pq->zero();
        C_rs->zero();
        for (size_t K = 0; K < max_K; ++K) {
            auto& c_dets = ab_list_[K];
            size_t maxI = c_dets.size();
            for (size_t det = 0; det < maxI; ++det) {
                auto& detJ = c_dets[det];
                const size_t J = std::get<0>(detJ);
                const double sign_rs = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                const size_t r = std::abs(std::get<1>(detJ)) - 1;
                const size_t s = std::get<2>(detJ);
                C_rs->set(K, r * ncmo + s, sign_rs * b_p[J]);
            }
        }
        // Timer mult;
        B_pq->gemm(false, true, 1.0, ab_tei_, C_rs, 0.0);
        // C_DGEMV('N',ncmo2,ncmo2,1.0, &(ab_tei_->pointer()[0][0]),ncmo2,
        // &(C_rs->pointer()[0]),1,0.0,&(B_pq->pointer()[0]),1);
        // outfile->Printf("\n  Time spent on GEMV: %1.6f", mult.get());
        for (size_t K = 0; K < max_K; ++K) {
            auto& c_dets = ab_list_[K];
            size_t maxI = c_dets.size();
            for (size_t det = 0; det < maxI; ++det) {
                auto& detI = c_dets[det];
                const size_t p = std::abs(std::get<1>(detI)) - 1;
                const size_t q = std::get<2>(detI);
                const size_t I = std::get<0>(detI);
                const double sign_pq = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                sigma_p[I] += sign_pq * B_pq->get(p * ncmo + q, K);
            }
        }
    }
    // outfile->Printf("\n  Time spent on AB: %1.6f", AB.get());
    // exit(1);
}

/*  Sigma Vector Sparse functions */

void SigmaVectorSparse::compute_sigma(SharedVector sigma, SharedVector b) {
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
        size_t start_idx =
            (tid < (size_ % num_thread))
                ? tid * bin_size
                : (size_ % num_thread) * (bin_size + 1) + (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        for (size_t J = start_idx; J < end_idx; ++J) {
            std::vector<double>& H_row = H_[J].second;
            std::vector<size_t>& index_row = H_[J].first;
            size_t maxc = index_row.size();
            for (size_t c = 0; c < maxc; ++c) {
                int K = index_row[c];
                double HJK = H_row[c];
                sigma_p[J] += HJK * b_p[K];
            }
        }
    }
}
void SigmaVectorSparse::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < size_; ++I) {
        diag.set(I, H_[I].second[0]);
    }
}

void SigmaVectorSparse::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}
}
}
