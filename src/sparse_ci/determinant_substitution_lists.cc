/*
 *@BEGIN LICENSE
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

#include <cmath>

#include "psi4/libmints/dimension.h"

#include "sparse_ci/determinant_substitution_lists.h"
#include "forte-def.h"
#include "helpers/timer.h"
#include "helpers/printing.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

using namespace psi;

namespace forte {

DeterminantSubstitutionLists::DeterminantSubstitutionLists(std::shared_ptr<ActiveSpaceIntegrals> fci_ints)
    : ncmo_(fci_ints->nmo()), mo_symmetry_(fci_ints->active_mo_symmetry()), fci_ints_(fci_ints) {}

void DeterminantSubstitutionLists::set_quiet_mode(bool mode) { quiet_ = mode; }

void DeterminantSubstitutionLists::build_strings(const DeterminantHashVec& wfn) {
    beta_strings_.clear();
    alpha_strings_.clear();
    alpha_a_strings_.clear();

    // First build a map from beta strings to determinants
    const det_hashvec& wfn_map = wfn.wfn_hash();
    {
        det_hash<size_t> beta_str_hash;
        size_t nbeta = 0;
        for (size_t I = 0, max_I = wfn_map.size(); I < max_I; ++I) {
            // Grab mutable copy of determinant
            Determinant detI(wfn_map[I]);
            detI.zero_spin(DetSpinType::Alpha);

            det_hash<size_t>::iterator it = beta_str_hash.find(detI);
            size_t b_add;
            if (it == beta_str_hash.end()) {
                b_add = nbeta;
                beta_str_hash[detI] = b_add;
                nbeta++;
            } else {
                b_add = it->second;
            }
            beta_strings_.resize(nbeta);
            beta_strings_[b_add].push_back(I);
        }
    }

    {
        det_hash<size_t> alfa_str_hash;
        size_t nalfa = 0;
        for (size_t I = 0, max_I = wfn_map.size(); I < max_I; ++I) {
            // Grab mutable copy of determinant
            Determinant detI(wfn_map[I]);
            detI.zero_spin(DetSpinType::Beta);

            det_hash<size_t>::iterator it = alfa_str_hash.find(detI);
            size_t a_add;
            if (it == alfa_str_hash.end()) {
                a_add = nalfa;
                alfa_str_hash[detI] = a_add;
                nalfa++;
            } else {
                a_add = it->second;
            }
            alpha_strings_.resize(nalfa);
            alpha_strings_[a_add].push_back(I);
        }
    }
    // Next build a map from annihilated alpha strings to determinants
    det_hash<size_t> alfa_str_hash;
    size_t naalpha = 0;
    for (size_t I = 0, max_I = wfn_map.size(); I < max_I; ++I) {
        // Grab mutable copy of determinant
        Determinant detI(wfn_map[I]);
        detI.zero_spin(DetSpinType::Beta);
        const std::vector<int>& aocc = detI.get_alfa_occ(ncmo_);
        for (int i = 0, nalfa = aocc.size(); i < nalfa; ++i) {
            int ii = aocc[i];
            Determinant ann_det(detI);
            ann_det.set_alfa_bit(ii, false);

            size_t a_add;
            det_hash<size_t>::iterator it = alfa_str_hash.find(ann_det);
            if (it == alfa_str_hash.end()) {
                a_add = naalpha;
                alfa_str_hash[ann_det] = a_add;
                naalpha++;
            } else {
                a_add = it->second;
            }
            alpha_a_strings_.resize(naalpha);
            alpha_a_strings_[a_add].push_back(std::make_pair(ii, I));
        }
    }
}

void DeterminantSubstitutionLists::op_s_lists(const DeterminantHashVec& wfn) {
    timer ops("Single sub. lists");

    if (!quiet_) {
        print_h2("Computing Coupling Lists");
        outfile->Printf("  --------------------------------");
    }

    // Get a reference to the determinants
    const det_hashvec& dets = wfn.wfn_hash();
    local_timer ann;
    for (size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b) {
        size_t na_ann = 0;
        std::vector<std::vector<std::pair<size_t, short>>> tmp;
        std::vector<size_t>& c_dets = beta_strings_[b];
        det_hash<int> map_a_ann;
        for (size_t I = 0, maxI = c_dets.size(); I < maxI; ++I) {
            size_t index = c_dets[I];
            const Determinant& detI = dets[index];

            const std::vector<int>& aocc = detI.get_alfa_occ(ncmo_);
            int noalfa = aocc.size();

            for (int i = 0; i < noalfa; ++i) {
                int ii = aocc[i];
                Determinant detJ(detI);
                detJ.set_alfa_bit(ii, false);
                double sign = detI.slater_sign_a(ii);
                size_t detJ_add;
                auto search = map_a_ann.find(detJ);
                if (search == map_a_ann.end()) {
                    detJ_add = na_ann;
                    map_a_ann[detJ] = na_ann;
                    na_ann++;
                    tmp.resize(na_ann);
                } else {
                    detJ_add = search->second;
                }
                tmp[detJ_add].push_back(std::make_pair(index, sign > 0.0 ? (ii + 1) : (-ii - 1)));
            }
        }
        // size_t idx = 0;
        for (auto& vec : tmp) {
            if (vec.size() > 1) {
                a_list_.push_back(vec);
                //  a_list_[idx] = vec;
                //  idx++;
            }
        }
    }

    if (!quiet_) {
        outfile->Printf("\n        α          %7.6f s", ann.get());
    }
    local_timer bnn;
    for (size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a) {
        size_t nb_ann = 0;
        std::vector<std::vector<std::pair<size_t, short>>> tmp;
        std::vector<size_t>& c_dets = alpha_strings_[a];
        det_hash<int> map_b_ann;
        for (size_t I = 0, maxI = c_dets.size(); I < maxI; ++I) {
            size_t index = c_dets[I];
            const Determinant& detI = dets[index];
            const std::vector<int>& bocc = detI.get_beta_occ(ncmo_);
            int nobeta = bocc.size();

            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                Determinant detJ(detI);
                detJ.set_beta_bit(ii, false);
                double sign = detI.slater_sign_b(ii);
                size_t detJ_add;
                auto search = map_b_ann.find(detJ);
                if (search == map_b_ann.end()) {
                    detJ_add = nb_ann;
                    map_b_ann[detJ] = nb_ann;
                    nb_ann++;
                    tmp.resize(nb_ann);
                } else {
                    detJ_add = search->second;
                }
                tmp[detJ_add].push_back(std::make_pair(index, sign > 0.0 ? (ii + 1) : (-ii - 1)));
            }
        }
        for (auto& vec : tmp) {
            if (vec.size() > 1) {
                b_list_.push_back(vec);
            }
        }
    }
    if (!quiet_) {
        outfile->Printf("\n        β          %7.6f s", bnn.get());
    }
}

void DeterminantSubstitutionLists::tp_s_lists(const DeterminantHashVec& wfn) {

    timer ops("Double sub. lists");
    const det_hashvec& dets = wfn.wfn_hash();
    // Generate alpha-alpha coupling list
    local_timer aa;
    {
        for (size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b) {
            size_t naa_ann = 0;
            std::vector<std::vector<std::tuple<size_t, short, short>>> tmp;
            det_hash<int> map_aa_ann;
            std::vector<size_t> c_dets = beta_strings_[b];
            size_t max_I = c_dets.size();
            for (size_t I = 0; I < max_I; ++I) {
                size_t idx = c_dets[I];
                Determinant detI = dets[idx];
                std::vector<int> aocc = detI.get_alfa_occ(ncmo_);
                int noalfa = aocc.size();

                for (int i = 0; i < noalfa; ++i) {
                    for (int j = i + 1; j < noalfa; ++j) {
                        int ii = aocc[i];
                        int jj = aocc[j];
                        Determinant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(jj, false);

                        double sign = detI.slater_sign_a(ii) * detI.slater_sign_a(jj);

                        det_hash<int>::iterator it = map_aa_ann.find(detJ);
                        size_t detJ_add;
                        // Add detJ to map if it isn't there
                        if (it == map_aa_ann.end()) {
                            detJ_add = naa_ann;
                            map_aa_ann[detJ] = naa_ann;
                            naa_ann++;
                            tmp.resize(naa_ann);
                        } else {
                            detJ_add = it->second;
                        }
                        tmp[detJ_add].push_back(
                            std::make_tuple(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj));
                    }
                }
            }
            for (auto& vec : tmp) {
                if (vec.size() > 1) {
                    aa_list_.push_back(vec);
                }
            }
        }
    }
    if (!quiet_) {
        outfile->Printf("\n        αα         %7.6f s", aa.get());
    }
    // Generate beta-beta coupling list
    local_timer bb;
    {
        for (size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a) {
            size_t nbb_ann = 0;
            std::vector<std::vector<std::tuple<size_t, short, short>>> tmp;
            det_hash<int> map_bb_ann;
            std::vector<size_t>& c_dets = alpha_strings_[a];
            size_t max_I = c_dets.size();

            for (size_t I = 0; I < max_I; ++I) {
                size_t idx = c_dets[I];

                Determinant detI(dets[idx]);
                std::vector<int> bocc = detI.get_beta_occ(ncmo_);
                int nobeta = bocc.size();

                for (int i = 0, ij = 0; i < nobeta; ++i) {
                    for (int j = i + 1; j < nobeta; ++j, ++ij) {
                        int ii = bocc[i];
                        int jj = bocc[j];
                        Determinant detJ(detI);
                        detJ.set_beta_bit(ii, false);
                        detJ.set_beta_bit(jj, false);

                        double sign = detI.slater_sign_b(ii) * detI.slater_sign_b(jj);

                        det_hash<int>::iterator it = map_bb_ann.find(detJ);
                        size_t detJ_add;
                        // Add detJ to map if it isn't there
                        if (it == map_bb_ann.end()) {
                            detJ_add = nbb_ann;
                            map_bb_ann[detJ] = nbb_ann;
                            nbb_ann++;
                            tmp.resize(nbb_ann);
                        } else {
                            detJ_add = it->second;
                        }

                        tmp[detJ_add].push_back(
                            std::make_tuple(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj));
                    }
                }
            }
            for (auto& vec : tmp) {
                if (vec.size() > 1) {
                    bb_list_.push_back(vec);
                }
            }
        }
    }
    if (!quiet_) {
        outfile->Printf("\n        ββ         %7.6f s", bb.get());
    }

    local_timer ab;
    // Generate alfa-beta coupling list
    {
        for (size_t a = 0, max_a = alpha_a_strings_.size(); a < max_a; ++a) {
            size_t nab_ann = 0;
            std::vector<std::vector<std::tuple<size_t, short, short>>> tmp;
            det_hash<int> map_ab_ann;
            std::vector<std::pair<int, size_t>>& c_dets = alpha_a_strings_[a];
            size_t max_I = c_dets.size();
            for (size_t I = 0; I < max_I; ++I) {
                size_t idx = c_dets[I].second;
                int ii = c_dets[I].first;
                Determinant detI(dets[idx]);
                detI.set_alfa_bit(ii, false);
                std::vector<int> bocc = detI.get_beta_occ(ncmo_);
                size_t nobeta = bocc.size();

                for (size_t j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];

                    Determinant detJ(detI);
                    detJ.set_beta_bit(jj, false);

                    double sign = detI.slater_sign_a(ii) * detI.slater_sign_b(jj);
                    det_hash<int>::iterator it = map_ab_ann.find(detJ);
                    size_t detJ_add;

                    if (it == map_ab_ann.end()) {
                        detJ_add = nab_ann;
                        map_ab_ann[detJ] = nab_ann;
                        nab_ann++;
                        tmp.resize(nab_ann);
                    } else {
                        detJ_add = it->second;
                    }
                    tmp[detJ_add].push_back(
                        std::make_tuple(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj));
                }
            }
            for (auto& vec : tmp) {
                if (vec.size() > 1) {
                    ab_list_.push_back(vec);
                }
            }
        }
    }
    if (!quiet_) {
        outfile->Printf("\n        αβ         %7.6f s", ab.get());
        outfile->Printf("\n  --------------------------------");
    }
}

void DeterminantSubstitutionLists::clear_op_s_lists() {
    a_list_.clear();
    b_list_.clear();
}

void DeterminantSubstitutionLists::clear_tp_s_lists() {
    aa_list_.clear();
    bb_list_.clear();
    ab_list_.clear();
}

void DeterminantSubstitutionLists::three_s_lists(const DeterminantHashVec& wfn) {

    timer ops("Triple sub. lists");
    const det_hashvec& dets = wfn.wfn_hash();
    //  Timer aaa;
    {
        for (size_t b = 0, max_b = beta_strings_.size(); b < max_b; ++b) {
            size_t naa_ann = 0;
            std::vector<std::vector<std::tuple<size_t, short, short, short>>> tmp;
            det_hash<int> map_aaa;
            std::vector<size_t> c_dets = beta_strings_[b];
            size_t max_I = c_dets.size();
            for (size_t I = 0; I < max_I; ++I) {
                size_t idx = c_dets[I];
                Determinant detI(dets[idx]);
                std::vector<int> aocc = detI.get_alfa_occ(ncmo_);
                int noalfa = aocc.size();

                for (int i = 0; i < noalfa; ++i) {
                    for (int j = i + 1; j < noalfa; ++j) {
                        for (int k = j + 1; k < noalfa; ++k) {
                            int ii = aocc[i];
                            int jj = aocc[j];
                            int kk = aocc[k];
                            Determinant detJ(detI);
                            detJ.set_alfa_bit(ii, false);
                            detJ.set_alfa_bit(jj, false);
                            detJ.set_alfa_bit(kk, false);

                            double sign = detI.slater_sign_a(ii) * detI.slater_sign_a(jj) *
                                          detI.slater_sign_a(kk);

                            det_hash<int>::iterator it = map_aaa.find(detJ);
                            size_t detJ_add;
                            // Add detJ to map if it isn't there
                            if (it == map_aaa.end()) {
                                detJ_add = naa_ann;
                                map_aaa[detJ] = naa_ann;
                                naa_ann++;
                                tmp.resize(naa_ann);
                            } else {
                                detJ_add = it->second;
                            }
                            tmp[detJ_add].push_back(
                                std::make_tuple(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj, kk));
                        }
                    }
                }
            }
            for (auto& vec : tmp) {
                if (vec.size() > 1) {
                    aaa_list_.push_back(vec);
                }
            }
        }
    }
    //  if (!quiet_) {
    //      outfile->Printf("\n  Time spent building aaa_list  %1.6f s", aaa.get());
    //  }

    /// AAB coupling
    {
        local_timer aab;
        // We need the beta-1 list:
        const det_hashvec& wfn_map = wfn.wfn_hash();
        std::vector<std::vector<std::pair<int, size_t>>> beta_string;
        det_hash<size_t> beta_str_hash;
        size_t nabeta = 0;
        for (size_t I = 0, max_I = wfn_map.size(); I < max_I; ++I) {
            // Grab mutable copy of determinant
            Determinant detI(wfn_map[I]);
            detI.zero_spin(DetSpinType::Alpha);
            std::vector<int> bocc = detI.get_beta_occ(ncmo_);
            for (int i = 0, nbeta = bocc.size(); i < nbeta; ++i) {
                int ii = bocc[i];
                Determinant ann_det(detI);
                ann_det.set_beta_bit(ii, false);

                size_t b_add;
                det_hash<size_t>::iterator it = beta_str_hash.find(ann_det);
                if (it == beta_str_hash.end()) {
                    b_add = nabeta;
                    beta_str_hash[ann_det] = b_add;
                    nabeta++;
                } else {
                    b_add = it->second;
                }
                beta_string.resize(nabeta);
                beta_string[b_add].push_back(std::make_pair(ii, I));
            }
        }
        for (size_t b = 0, max_b = beta_string.size(); b < max_b; ++b) {
            size_t naab_ann = 0;
            det_hash<int> aab_ann_map;
            std::vector<std::pair<int, size_t>>& c_dets = beta_string[b];
            std::vector<std::vector<std::tuple<size_t, short, short, short>>> tmp;
            size_t max_I = c_dets.size();
            for (size_t I = 0; I < max_I; ++I) {
                size_t idx = c_dets[I].second;

                Determinant detI(dets[idx]);
                size_t kk = c_dets[I].first;
                detI.set_beta_bit(kk, false);

                std::vector<int> aocc = detI.get_alfa_occ(ncmo_);

                size_t noalfa = aocc.size();

                for (size_t i = 0, jk = 0; i < noalfa; ++i) {
                    for (size_t j = i + 1; j < noalfa; ++j, ++jk) {

                        int ii = aocc[i];
                        int jj = aocc[j];

                        Determinant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(jj, false);

                        double sign = detI.slater_sign_a(ii) * detI.slater_sign_a(jj) *
                                      detI.slater_sign_b(kk);

                        det_hash<int>::iterator hash_it = aab_ann_map.find(detJ);
                        size_t detJ_add;

                        if (hash_it == aab_ann_map.end()) {
                            detJ_add = naab_ann;
                            aab_ann_map[detJ] = naab_ann;
                            naab_ann++;
                            tmp.resize(naab_ann);
                        } else {
                            detJ_add = hash_it->second;
                        }
                        tmp[detJ_add].push_back(
                            std::make_tuple(idx, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk));
                    }
                }
            }
            for (auto& vec : tmp) {
                if (vec.size() > 1) {
                    aab_list_.push_back(vec);
                }
            }
        }
        //      outfile->Printf("\n  Time spent building aab_list  %1.6f s", aab.get());
    }

    /// ABB coupling
    {
        //   Timer abb;
        for (size_t a = 0, max_a = alpha_a_strings_.size(); a < max_a; ++a) {
            size_t nabb_ann = 0;
            det_hash<int> abb_ann_map;
            std::vector<std::pair<int, size_t>>& c_dets = alpha_a_strings_[a];
            size_t max_I = c_dets.size();
            std::vector<std::vector<std::tuple<size_t, short, short, short>>> tmp;

            for (size_t I = 0; I < max_I; ++I) {
                size_t idx = c_dets[I].second;

                Determinant detI(dets[idx]);
                int ii = c_dets[I].first;
                detI.set_alfa_bit(ii, false);

                std::vector<int> bocc = detI.get_beta_occ(ncmo_);

                int nobeta = bocc.size();

                for (int j = 0, jk = 0; j < nobeta; ++j) {
                    for (int k = j + 1; k < nobeta; ++k, ++jk) {

                        int jj = bocc[j];
                        int kk = bocc[k];

                        Determinant detJ(detI);
                        detJ.set_beta_bit(jj, false);
                        detJ.set_beta_bit(kk, false);

                        double sign = detI.slater_sign_a(ii) * detI.slater_sign_b(jj) *
                                      detI.slater_sign_b(kk);

                        det_hash<int>::iterator hash_it = abb_ann_map.find(detJ);
                        size_t detJ_add;

                        if (hash_it == abb_ann_map.end()) {
                            detJ_add = nabb_ann;
                            abb_ann_map[detJ] = nabb_ann;
                            nabb_ann++;
                            tmp.resize(nabb_ann);
                        } else {
                            detJ_add = hash_it->second;
                        }
                        tmp[detJ_add].push_back(
                            std::make_tuple(idx, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk));
                    }
                }
            }
            for (auto& vec : tmp) {
                if (vec.size() > 1) {
                    abb_list_.push_back(vec);
                }
            }
        }
        //    outfile->Printf("\n  Time spent building abb_list  %1.6f s", abb.get());
    }

    /// BBB coupling
    {
        // Timer bbb;
        for (size_t a = 0, max_a = alpha_strings_.size(); a < max_a; ++a) {
            size_t nbbb_ann = 0;
            det_hash<int> bbb_ann_map;
            std::vector<size_t>& c_dets = alpha_strings_[a];
            size_t max_I = c_dets.size();
            std::vector<std::vector<std::tuple<size_t, short, short, short>>> tmp;

            for (size_t I = 0; I < max_I; ++I) {
                size_t idx = c_dets[I];
                Determinant detI(dets[idx]);

                std::vector<int> bocc = detI.get_beta_occ(ncmo_);

                int nobeta = bocc.size();

                // bbb
                for (int i = 0; i < nobeta; ++i) {
                    for (int j = i + 1; j < nobeta; ++j) {
                        for (int k = j + 1; k < nobeta; ++k) {

                            int ii = bocc[i];
                            int jj = bocc[j];
                            int kk = bocc[k];

                            Determinant detJ(detI);
                            detJ.set_beta_bit(ii, false);
                            detJ.set_beta_bit(jj, false);
                            detJ.set_beta_bit(kk, false);

                            double sign = detI.slater_sign_b(ii) * detI.slater_sign_b(jj) *
                                          detI.slater_sign_b(kk);

                            det_hash<int>::iterator hash_it = bbb_ann_map.find(detJ);
                            size_t detJ_add;

                            if (hash_it == bbb_ann_map.end()) {
                                detJ_add = nbbb_ann;
                                bbb_ann_map[detJ] = nbbb_ann;
                                nbbb_ann++;
                                tmp.resize(nbbb_ann);
                            } else {
                                detJ_add = hash_it->second;
                            }
                            tmp[detJ_add].push_back(
                                std::make_tuple(idx, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk));
                        }
                    }
                }
            }
            for (auto& vec : tmp) {
                if (vec.size() > 1) {
                    bbb_list_.push_back(vec);
                }
            }
        }
        //  outfile->Printf("\n  Time spent building bbb_list  %1.6f s", bbb.get());
    }
}

} // namespace forte
