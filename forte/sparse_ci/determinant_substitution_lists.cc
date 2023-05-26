/*
 *@BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <numeric>

#include "psi4/libpsi4util/PsiOutStream.h"
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

DeterminantSubstitutionLists::DeterminantSubstitutionLists(const std::vector<int>& mo_symmetry)
    : ncmo_(mo_symmetry.size()), mo_symmetry_(mo_symmetry) {}

void DeterminantSubstitutionLists::set_quiet_mode(bool mode) { quiet_ = mode; }

void DeterminantSubstitutionLists::build_strings(const DeterminantHashVec& wfn) {
    beta_strings_.clear();
    alpha_strings_.clear();
    alpha_a_strings_.clear();

    // Build a map from beta strings to determinants
    const det_hashvec& wfn_map = wfn.wfn_hash();
    {
        det_hash<size_t> beta_str_hash;
        size_t nbeta = 0;
        for (size_t I = 0, max_I = wfn_map.size(); I < max_I; ++I) {
            // Grab mutable copy of determinant
            Determinant detI(wfn_map[I]);
            detI.zero_alfa();

            auto it = beta_str_hash.find(detI);
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

    // Build a map from alpha strings to determinants
    {
        det_hash<size_t> alfa_str_hash;
        size_t nalfa = 0;
        for (size_t I = 0, max_I = wfn_map.size(); I < max_I; ++I) {
            // Grab mutable copy of determinant
            Determinant detI(wfn_map[I]);
            detI.zero_beta();

            auto it = alfa_str_hash.find(detI);
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

    // Build a map from annihilated alpha strings to determinants
    det_hash<size_t> alfa_str_hash;
    size_t naalpha = 0;
    for (size_t I = 0, max_I = wfn_map.size(); I < max_I; ++I) {
        // Grab mutable copy of determinant
        Determinant detI(wfn_map[I]);
        detI.zero_beta();
        const std::vector<int>& aocc = detI.get_alfa_occ(static_cast<int>(ncmo_));
        for (int ii : aocc) {
            Determinant ann_det(detI);
            ann_det.set_alfa_bit(ii, false);

            size_t a_add;
            auto it = alfa_str_hash.find(ann_det);
            if (it == alfa_str_hash.end()) {
                a_add = naalpha;
                alfa_str_hash[ann_det] = a_add;
                naalpha++;
            } else {
                a_add = it->second;
            }
            alpha_a_strings_.resize(naalpha);
            alpha_a_strings_[a_add].emplace_back(ii, I);
        }
    }
}

void DeterminantSubstitutionLists::op_s_lists(const DeterminantHashVec& wfn) {
    timer ops("Single sub. lists");
    if (!quiet_) {
        print_h2("Computing 1 Coupling Lists");
    }
    lists_1a(wfn);
    lists_1b(wfn);
}

void DeterminantSubstitutionLists::lists_1a(const DeterminantHashVec& wfn) {
    timer ann("A lists");

    const det_hashvec& dets = wfn.wfn_hash();
    std::map<size_t, size_t> vec_size;

    for (auto& c_dets : beta_strings_) {
        size_t na_ann = 0;
        std::vector<std::vector<std::pair<size_t, short>>> tmp;
        det_hash<int> map_a_ann;
        for (unsigned long index : c_dets) {
            const Determinant& detI = dets[index];

            const std::vector<int>& aocc = detI.get_alfa_occ(static_cast<int>(ncmo_));

            for (int ii : aocc) {
                Determinant detJ(detI);
                detJ.set_alfa_bit(ii, false);
                double sign = detI.slater_sign_a(ii);
                size_t detJ_add;
                auto search = map_a_ann.find(detJ);
                if (search == map_a_ann.end()) {
                    detJ_add = na_ann;
                    map_a_ann[detJ] = static_cast<int>(na_ann);
                    na_ann++;
                    tmp.resize(na_ann);
                } else {
                    detJ_add = search->second;
                }
                tmp[detJ_add].emplace_back(index, sign > 0.0 ? (ii + 1) : (-ii - 1));
            }
        }

        for (const auto& vec : tmp) {
            // TODO: if (vec.size() > 1) { ??
            if (!vec.empty()) {
                a_list_.push_back(vec);
            }
            auto s = vec.size();
            if (vec_size.find(s) == vec_size.end()) {
                vec_size[s] = 1;
            } else {
                vec_size[s] += 1;
            }
        }
    }
    outfile->Printf("\n  (N-1) a lists size counts");
    outfile->Printf("\n      Size      Count");
    for (const auto& p : vec_size) {
        outfile->Printf("\n  %8zu %10zu", p.first, p.second);
    }

    if (!quiet_) {
        outfile->Printf("\n        α          %.3e seconds", ann.stop());
    }
}

void DeterminantSubstitutionLists::lists_1b(const DeterminantHashVec& wfn) {
    timer bnn("B lists");

    const det_hashvec& dets = wfn.wfn_hash();

    for (auto& c_dets : alpha_strings_) {
        size_t nb_ann = 0;
        std::vector<std::vector<std::pair<size_t, short>>> tmp;
        det_hash<int> map_b_ann;
        for (unsigned long index : c_dets) {
            const Determinant& detI = dets[index];
            const std::vector<int>& bocc = detI.get_beta_occ(static_cast<int>(ncmo_));

            for (int ii : bocc) {
                Determinant detJ(detI);
                detJ.set_beta_bit(ii, false);
                double sign = detI.slater_sign_b(ii);
                size_t detJ_add;
                auto search = map_b_ann.find(detJ);
                if (search == map_b_ann.end()) {
                    detJ_add = nb_ann;
                    map_b_ann[detJ] = static_cast<int>(nb_ann);
                    nb_ann++;
                    tmp.resize(nb_ann);
                } else {
                    detJ_add = search->second;
                }
                tmp[detJ_add].emplace_back(index, sign > 0.0 ? (ii + 1) : (-ii - 1));
            }
        }

        for (auto& vec : tmp) {
            // TODO: if (vec.size() > 1) { ??
            if (!vec.empty()) {
                b_list_.push_back(vec);
            }
        }
    }

    if (!quiet_) {
        outfile->Printf("\n        β          %.3e seconds", bnn.stop());
    }
}

void DeterminantSubstitutionLists::tp_s_lists(const DeterminantHashVec& wfn) {
    timer ops("Double sub. lists");
    if (!quiet_) {
        print_h2("Computing 2 Coupling Lists");
    }
    lists_2aa(wfn);
    lists_2ab(wfn);
    lists_2bb(wfn);
}

void DeterminantSubstitutionLists::lists_2aa(const DeterminantHashVec& wfn) {
    timer timer_aa("AA lists");

    const det_hashvec& dets = wfn.wfn_hash();
    std::map<size_t, size_t> vec_size;

    for (const auto& c_dets : beta_strings_) {
        size_t naa_ann = 0;
        std::vector<std::vector<std::tuple<size_t, short, short>>> tmp;
        det_hash<int> map_aa_ann;
        size_t max_I = c_dets.size();
        for (size_t I = 0; I < max_I; ++I) {
            size_t idx = c_dets[I];
            Determinant detI = dets[idx];
            std::vector<int> aocc = detI.get_alfa_occ(static_cast<int>(ncmo_));

            for (int i = 0, noalfa = static_cast<int>(aocc.size()); i < noalfa; ++i) {
                for (int j = i + 1; j < noalfa; ++j) {
                    int ii = aocc[i];
                    int jj = aocc[j];
                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_alfa_bit(jj, false);

                    double sign = detI.slater_sign_a(ii) * detI.slater_sign_a(jj);

                    auto it = map_aa_ann.find(detJ);
                    size_t detJ_add;
                    // Add detJ to map if it isn't there
                    if (it == map_aa_ann.end()) {
                        detJ_add = naa_ann;
                        map_aa_ann[detJ] = static_cast<int>(naa_ann);
                        naa_ann++;
                        tmp.resize(naa_ann);
                    } else {
                        detJ_add = it->second;
                    }
                    tmp[detJ_add].emplace_back(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj);
                }
            }
        }

        for (auto& vec : tmp) {
            if (!vec.empty()) {
                aa_list_.push_back(vec);
            }
            auto s = vec.size();
            if (vec_size.find(s) == vec_size.end()) {
                vec_size[s] = 1;
            } else {
                vec_size[s] += 1;
            }
        }
    }
    outfile->Printf("\n  (N-2) aa lists size counts");
    outfile->Printf("\n      Size      Count");
    for (const auto& p : vec_size) {
        outfile->Printf("\n  %8zu %10zu", p.first, p.second);
    }

    if (!quiet_) {
        outfile->Printf("\n        αα         %.3e seconds", timer_aa.stop());
    }
}

void DeterminantSubstitutionLists::lists_2ab(const DeterminantHashVec& wfn) {
    timer timer_ab("AB lists");

    const det_hashvec& dets = wfn.wfn_hash();
    std::map<size_t, size_t> vec_size;

    for (auto& c_dets : alpha_a_strings_) {
        size_t nab_ann = 0;
        std::vector<std::vector<std::tuple<size_t, short, short>>> tmp;
        det_hash<int> map_ab_ann;
        size_t max_I = c_dets.size();
        for (size_t I = 0; I < max_I; ++I) {
            size_t idx = c_dets[I].second;
            int ii = c_dets[I].first;
            Determinant detI(dets[idx]);
            detI.set_alfa_bit(ii, false);
            std::vector<int> bocc = detI.get_beta_occ(static_cast<int>(ncmo_));

            for (int jj : bocc) {
                Determinant detJ(detI);
                detJ.set_beta_bit(jj, false);

                double sign = detI.slater_sign_a(ii) * detI.slater_sign_b(jj);
                auto it = map_ab_ann.find(detJ);
                size_t detJ_add;

                if (it == map_ab_ann.end()) {
                    detJ_add = nab_ann;
                    map_ab_ann[detJ] = static_cast<int>(nab_ann);
                    nab_ann++;
                    tmp.resize(nab_ann);
                } else {
                    detJ_add = it->second;
                }
                tmp[detJ_add].emplace_back(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj);
            }
        }

        for (auto& vec : tmp) {
            if (!vec.empty()) {
                ab_list_.push_back(vec);
            }
            auto s = vec.size();
            if (vec_size.find(s) == vec_size.end()) {
                vec_size[s] = 1;
            } else {
                vec_size[s] += 1;
            }
        }
    }
    outfile->Printf("\n  (N-2) ab lists size counts");
    outfile->Printf("\n      Size      Count");
    for (const auto& p : vec_size) {
        outfile->Printf("\n  %8zu %10zu", p.first, p.second);
    }

    if (!quiet_) {
        outfile->Printf("\n        αβ         %.3e seconds", timer_ab.stop());
    }
}

void DeterminantSubstitutionLists::lists_2bb(const DeterminantHashVec& wfn) {
    timer timer_bb("BB lists");

    const det_hashvec& dets = wfn.wfn_hash();

    for (auto& c_dets : alpha_strings_) {
        size_t nbb_ann = 0;
        std::vector<std::vector<std::tuple<size_t, short, short>>> tmp;
        det_hash<int> map_bb_ann;
        size_t max_I = c_dets.size();

        for (size_t I = 0; I < max_I; ++I) {
            size_t idx = c_dets[I];

            Determinant detI(dets[idx]);
            std::vector<int> bocc = detI.get_beta_occ(static_cast<int>(ncmo_));

            for (int i = 0, nobeta = static_cast<int>(bocc.size()); i < nobeta; ++i) {
                for (int j = i + 1; j < nobeta; ++j) {
                    int ii = bocc[i];
                    int jj = bocc[j];
                    Determinant detJ(detI);
                    detJ.set_beta_bit(ii, false);
                    detJ.set_beta_bit(jj, false);

                    double sign = detI.slater_sign_b(ii) * detI.slater_sign_b(jj);

                    auto it = map_bb_ann.find(detJ);
                    size_t detJ_add;
                    // Add detJ to map if it isn't there
                    if (it == map_bb_ann.end()) {
                        detJ_add = nbb_ann;
                        map_bb_ann[detJ] = static_cast<int>(nbb_ann);
                        nbb_ann++;
                        tmp.resize(nbb_ann);
                    } else {
                        detJ_add = it->second;
                    }

                    tmp[detJ_add].emplace_back(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj);
                }
            }
        }

        for (auto& vec : tmp) {
            if (!vec.empty()) {
                bb_list_.push_back(vec);
            }
        }
    }

    if (!quiet_) {
        outfile->Printf("\n        ββ         %.3e seconds", timer_bb.stop());
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

void DeterminantSubstitutionLists::clear_3p_s_lists() {
    aaa_list_.clear();
    aab_list_.clear();
    abb_list_.clear();
    bbb_list_.clear();
}

void DeterminantSubstitutionLists::three_s_lists(const DeterminantHashVec& wfn) {
    timer ops("Triple sub. lists");
    if (!quiet_) {
        print_h2("Computing 3 Coupling Lists");
    }
    lists_3aaa(wfn);
    lists_3aab(wfn);
    lists_3abb(wfn);
    lists_3bbb(wfn);
}

void DeterminantSubstitutionLists::lists_3aaa(const DeterminantHashVec& wfn) {
    timer aaa("AAA lists");

    const det_hashvec& dets = wfn.wfn_hash();
    std::map<size_t, size_t> vec_size;

    for (auto c_dets : beta_strings_) {
        size_t naa_ann = 0;
        std::vector<std::vector<std::tuple<size_t, short, short, short>>> tmp;
        det_hash<int> map_aaa;
        size_t max_I = c_dets.size();
        for (size_t I = 0; I < max_I; ++I) {
            size_t idx = c_dets[I];
            Determinant detI(dets[idx]);
            std::vector<int> aocc = detI.get_alfa_occ(static_cast<int>(ncmo_));

            for (int i = 0, noalfa = static_cast<int>(aocc.size()); i < noalfa; ++i) {
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

                        auto it = map_aaa.find(detJ);
                        size_t detJ_add;
                        // Add detJ to map if it isn't there
                        if (it == map_aaa.end()) {
                            detJ_add = naa_ann;
                            map_aaa[detJ] = static_cast<int>(naa_ann);
                            naa_ann++;
                            tmp.resize(naa_ann);
                        } else {
                            detJ_add = it->second;
                        }
                        tmp[detJ_add].emplace_back(idx, (sign > 0.0) ? (ii + 1) : (-ii - 1), jj,
                                                   kk);
                    }
                }
            }
        }
        for (auto& vec : tmp) {
            if (!vec.empty()) {
                aaa_list_.push_back(vec);
            }
            auto s = vec.size();
            if (vec_size.find(s) == vec_size.end()) {
                vec_size[s] = 1;
            } else {
                vec_size[s] += 1;
            }
        }
    }
    outfile->Printf("\n  (N-3) aaa lists size counts");
    outfile->Printf("\n      Size      Count");
    for (const auto& p : vec_size) {
        outfile->Printf("\n  %8zu %10zu", p.first, p.second);
    }

    if (!quiet_) {
        outfile->Printf("\n        ααα        %.3e seconds", aaa.stop());
    }
}

void DeterminantSubstitutionLists::lists_3aab(const DeterminantHashVec& wfn) {
    timer aab("AAB lists");

    const det_hashvec& dets = wfn.wfn_hash();
    std::map<size_t, size_t> vec_size;

    // We need the beta-1 list:
    const det_hashvec& wfn_map = wfn.wfn_hash();
    std::vector<std::vector<std::pair<int, size_t>>> beta_string;
    det_hash<size_t> beta_str_hash;
    size_t nabeta = 0;
    for (size_t I = 0, max_I = wfn_map.size(); I < max_I; ++I) {
        // Grab mutable copy of determinant
        Determinant detI(wfn_map[I]);
        detI.zero_alfa();
        std::vector<int> bocc = detI.get_beta_occ(static_cast<int>(ncmo_));
        for (int ii : bocc) {
            Determinant ann_det(detI);
            ann_det.set_beta_bit(ii, false);

            size_t b_add;
            auto it = beta_str_hash.find(ann_det);
            if (it == beta_str_hash.end()) {
                b_add = nabeta;
                beta_str_hash[ann_det] = b_add;
                nabeta++;
            } else {
                b_add = it->second;
            }
            beta_string.resize(nabeta);
            beta_string[b_add].emplace_back(ii, I);
        }
    }

    for (auto& c_dets : beta_string) {
        size_t naab_ann = 0;
        det_hash<int> aab_ann_map;
        std::vector<std::vector<std::tuple<size_t, short, short, short>>> tmp;
        size_t max_I = c_dets.size();
        for (size_t I = 0; I < max_I; ++I) {
            size_t idx = c_dets[I].second;

            Determinant detI(dets[idx]);
            size_t kk = c_dets[I].first;
            detI.set_beta_bit(kk, false);

            std::vector<int> aocc = detI.get_alfa_occ(static_cast<int>(ncmo_));

            for (size_t i = 0, noalfa = static_cast<int>(aocc.size()); i < noalfa; ++i) {
                for (size_t j = i + 1; j < noalfa; ++j) {

                    int ii = aocc[i];
                    int jj = aocc[j];

                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_alfa_bit(jj, false);

                    double sign =
                        detI.slater_sign_a(ii) * detI.slater_sign_a(jj) * detI.slater_sign_b(kk);

                    auto hash_it = aab_ann_map.find(detJ);
                    size_t detJ_add;

                    if (hash_it == aab_ann_map.end()) {
                        detJ_add = naab_ann;
                        aab_ann_map[detJ] = static_cast<int>(naab_ann);
                        naab_ann++;
                        tmp.resize(naab_ann);
                    } else {
                        detJ_add = hash_it->second;
                    }
                    tmp[detJ_add].emplace_back(idx, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk);
                }
            }
        }

        for (auto& vec : tmp) {
            if (!vec.empty()) {
                aab_list_.push_back(vec);
            }
            auto s = vec.size();
            if (vec_size.find(s) == vec_size.end()) {
                vec_size[s] = 1;
            } else {
                vec_size[s] += 1;
            }
        }
    }
    outfile->Printf("\n  (N-3) aab lists size counts");
    outfile->Printf("\n      Size      Count");
    for (const auto& p : vec_size) {
        outfile->Printf("\n  %8zu %10zu", p.first, p.second);
    }

    if (!quiet_)
        outfile->Printf("\n        ααβ        %.3e seconds", aab.stop());
}

void DeterminantSubstitutionLists::lists_3abb(const DeterminantHashVec& wfn) {
    timer abb("ABB lists");

    const det_hashvec& dets = wfn.wfn_hash();

    for (auto& c_dets : alpha_a_strings_) {
        size_t nabb_ann = 0;
        det_hash<int> abb_ann_map;
        size_t max_I = c_dets.size();
        std::vector<std::vector<std::tuple<size_t, short, short, short>>> tmp;

        for (size_t I = 0; I < max_I; ++I) {
            size_t idx = c_dets[I].second;

            Determinant detI(dets[idx]);
            int ii = c_dets[I].first;
            detI.set_alfa_bit(ii, false);

            std::vector<int> bocc = detI.get_beta_occ(static_cast<int>(ncmo_));

            for (int j = 0, nobeta = static_cast<int>(bocc.size()); j < nobeta; ++j) {
                for (int k = j + 1; k < nobeta; ++k) {

                    int jj = bocc[j];
                    int kk = bocc[k];

                    Determinant detJ(detI);
                    detJ.set_beta_bit(jj, false);
                    detJ.set_beta_bit(kk, false);

                    double sign =
                        detI.slater_sign_a(ii) * detI.slater_sign_b(jj) * detI.slater_sign_b(kk);

                    auto hash_it = abb_ann_map.find(detJ);
                    size_t detJ_add;

                    if (hash_it == abb_ann_map.end()) {
                        detJ_add = nabb_ann;
                        abb_ann_map[detJ] = static_cast<int>(nabb_ann);
                        nabb_ann++;
                        tmp.resize(nabb_ann);
                    } else {
                        detJ_add = hash_it->second;
                    }
                    tmp[detJ_add].emplace_back(idx, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj, kk);
                }
            }
        }

        for (auto& vec : tmp) {
            if (!vec.empty()) {
                abb_list_.push_back(vec);
            }
        }
    }

    if (!quiet_)
        outfile->Printf("\n        αββ        %.3e seconds", abb.stop());
}

void DeterminantSubstitutionLists::lists_3bbb(const DeterminantHashVec& wfn) {
    timer bbb("BBB lists");

    const det_hashvec& dets = wfn.wfn_hash();

    for (auto& c_dets : alpha_strings_) {
        size_t nbbb_ann = 0;
        det_hash<int> bbb_ann_map;
        size_t max_I = c_dets.size();
        std::vector<std::vector<std::tuple<size_t, short, short, short>>> tmp;

        for (size_t I = 0; I < max_I; ++I) {
            size_t idx = c_dets[I];
            Determinant detI(dets[idx]);

            std::vector<int> bocc = detI.get_beta_occ(static_cast<int>(ncmo_));

            for (int i = 0, nobeta = static_cast<int>(bocc.size()); i < nobeta; ++i) {
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

                        auto hash_it = bbb_ann_map.find(detJ);
                        size_t detJ_add;

                        if (hash_it == bbb_ann_map.end()) {
                            detJ_add = nbbb_ann;
                            bbb_ann_map[detJ] = static_cast<int>(nbbb_ann);
                            nbbb_ann++;
                            tmp.resize(nbbb_ann);
                        } else {
                            detJ_add = hash_it->second;
                        }
                        tmp[detJ_add].emplace_back(idx, (sign > 0.5) ? (ii + 1) : (-ii - 1), jj,
                                                   kk);
                    }
                }
            }
        }

        for (auto& vec : tmp) {
            if (!vec.empty()) {
                bbb_list_.push_back(vec);
            }
        }
    }

    if (not quiet_)
        outfile->Printf("\n        βββ        %.3e seconds", bbb.stop());
}
} // namespace forte
