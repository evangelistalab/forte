/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"

#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"

#include "helpers/threading.h"
#include "forte-def.h"
#include "sci/aci.h"

using namespace psi;

namespace forte {

void AdaptiveCI::get_gas_excited_determinants_sr(
    SharedMatrix evecs, std::shared_ptr<psi::Vector> evals, DeterminantHashVec& P_space,
    std::vector<std::pair<double, Determinant>>& F_space) {
    local_timer build;
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();

    det_hash<double> V_hash;
// Loop over reference determinants
#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        const auto [start_idx, end_idx] = thread_range(max_P, num_thread, tid);

        det_hash<double> V_hash_t;
        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double Cp = evecs->get(P, ref_root_);

            // Generate the occupied/virtual alpha/beta orbitals for different GAS
            std::vector<std::vector<int>> gas_occ_a;
            std::vector<std::vector<int>> gas_vir_a;
            std::vector<std::vector<int>> gas_occ_b;
            std::vector<std::vector<int>> gas_vir_b;
            for (size_t gas_count = 0; gas_count < gas_num_; gas_count++) {
                std::vector<int> occ_a;
                std::vector<int> occ_b;
                std::vector<int> vir_a;
                std::vector<int> vir_b;
                for (const auto& p : relative_gas_mo_[gas_count]) {
                    if (det.get_alfa_bit(p)) {
                        occ_a.push_back(p);
                    } else {
                        vir_a.push_back(p);
                    }
                    if (det.get_beta_bit(p)) {
                        occ_b.push_back(p);
                    } else {
                        vir_b.push_back(p);
                    }
                }
                gas_occ_a.push_back(occ_a);
                gas_vir_a.push_back(vir_a);
                gas_occ_b.push_back(occ_b);
                gas_vir_b.push_back(vir_b);
            }

            // Generate the number of electrons in each GAS
            std::vector<int> gas_configuration;
            for (size_t gas_count = 0; gas_count < 6; gas_count++) {
                if (gas_count < gas_num_) {
                    gas_configuration.push_back(gas_occ_a[gas_count].size());
                    gas_configuration.push_back(gas_occ_b[gas_count].size());
                } else {
                    gas_configuration.push_back(0);
                    gas_configuration.push_back(0);
                }
            }
            //            for (size_t ii = 0; ii < gas_configuration.size(); ii++) {
            //                outfile->Printf("  %d", gas_configuration[ii]);
            //            }
            //            outfile->Printf("\n");
            // Generate a excitations
            for (auto& gas_count : gas_single_criterion_.first[gas_configuration]) {
                size_t gas_count_1 = gas_count.first;
                size_t gas_count_2 = gas_count.second;
                //                outfile->Printf("\n Allowed a %d %d", gas_count_1, gas_count_2);
                auto& occ = gas_occ_a[gas_count_1];
                auto& vir = gas_vir_a[gas_count_2];
                Determinant new_det(det);
                for (size_t ii : occ) {
                    for (size_t aa : vir) {
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            double HIJ = as_ints_->slater_rules_single_alpha(det, ii, aa) * Cp;
                            if (std::abs(HIJ) >= screen_thresh_) {
                                new_det = det;
                                new_det.set_alfa_bit(ii, false);
                                new_det.set_alfa_bit(aa, true);
                                //                                outfile->Printf("\n a %d %d %f",
                                //                                ii, aa,
                                //                                                as_ints_->slater_rules_single_alpha(det,
                                //                                                ii, aa));
                                V_hash_t[new_det] += HIJ;
                            }
                        }
                    }
                }
            }

            // Generate b excitations
            for (auto& gas_count : gas_single_criterion_.second[gas_configuration]) {
                size_t gas_count_1 = gas_count.first;
                size_t gas_count_2 = gas_count.second;
                //                outfile->Printf("\n Allowed b %d %d", gas_count_1, gas_count_2);
                auto& occ = gas_occ_b[gas_count_1];
                auto& vir = gas_vir_b[gas_count_2];
                Determinant new_det(det);
                for (size_t ii : occ) {
                    for (size_t aa : vir) {
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            double HIJ = as_ints_->slater_rules_single_beta(det, ii, aa) * Cp;
                            if (std::abs(HIJ) >= screen_thresh_) {
                                new_det = det;
                                new_det.set_beta_bit(ii, false);
                                new_det.set_beta_bit(aa, true);
                                V_hash_t[new_det] += HIJ;
                            }
                        }
                    }
                }
            }

            // Generate aa excitations
            for (auto& gas_count : std::get<0>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_a[gas_count_1];
                auto& occ2 = gas_occ_a[gas_count_2];
                auto& vir1 = gas_vir_a[gas_count_3];
                auto& vir2 = gas_vir_a[gas_count_4];
                //                outfile->Printf("\n Allowed aa %d %d %d %d", gas_count_1,
                //                gas_count_2, gas_count_3,
                //                                gas_count_4);
                for (size_t i = 0, maxi = occ1.size(); i < maxi; ++i) {
                    size_t ii = occ1[i];
                    size_t jstart = (gas_count_1 == gas_count_2 ? i + 1 : 0);
                    for (size_t j = jstart, maxj = occ2.size(); j < maxj; ++j) {
                        size_t jj = occ2[j];
                        for (size_t a = 0, maxa = vir1.size(); a < maxa; ++a) {
                            size_t aa = vir1[a];
                            size_t bstart = (gas_count_3 == gas_count_4 ? a + 1 : 0);
                            for (size_t b = bstart, maxb = vir2.size(); b < maxb; ++b) {
                                size_t bb = vir2[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_aa(ii, jj, aa, bb) * Cp;
                                    if (std::abs(HIJ) >= screen_thresh_) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                        V_hash_t[new_det] += HIJ;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate ab excitations
            for (auto& gas_count : std::get<2>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_a[gas_count_1];
                auto& occ2 = gas_occ_b[gas_count_2];
                auto& vir1 = gas_vir_a[gas_count_3];
                auto& vir2 = gas_vir_b[gas_count_4];
                //                outfile->Printf("\n Allowed ab %d %d %d %d", gas_count_1,
                //                gas_count_2, gas_count_3,
                //                                gas_count_4);
                for (size_t ii : occ1) {
                    for (size_t jj : occ2) {
                        for (size_t aa : vir1) {
                            for (size_t bb : vir2) {
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_ab(ii, jj, aa, bb) * Cp;
                                    if (std::abs(HIJ) >= screen_thresh_) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                        V_hash_t[new_det] += HIJ;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Genearte bb excitations
            for (auto& gas_count : std::get<1>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_b[gas_count_1];
                auto& occ2 = gas_occ_b[gas_count_2];
                auto& vir1 = gas_vir_b[gas_count_3];
                auto& vir2 = gas_vir_b[gas_count_4];
                //                outfile->Printf("\n Allowed bb %d %d %d %d", gas_count_1,
                //                gas_count_2, gas_count_3,
                //                                gas_count_4);
                for (size_t i = 0, maxi = occ1.size(); i < maxi; ++i) {
                    size_t ii = occ1[i];
                    size_t jstart = (gas_count_1 == gas_count_2 ? i + 1 : 0);
                    for (size_t j = jstart, maxj = occ2.size(); j < maxj; ++j) {
                        size_t jj = occ2[j];
                        for (size_t a = 0, maxa = vir1.size(); a < maxa; ++a) {
                            size_t aa = vir1[a];
                            size_t bstart = (gas_count_3 == gas_count_4 ? a + 1 : 0);
                            for (size_t b = bstart, maxb = vir2.size(); b < maxb; ++b) {
                                size_t bb = vir2[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_bb(ii, jj, aa, bb) * Cp;
                                    if (std::abs(HIJ) >= screen_thresh_) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);
                                        V_hash_t[new_det] += HIJ;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (tid == 0)
            outfile->Printf("\n  Time spent forming F space: %20.6f", build.get());
        local_timer merge_t;
#pragma omp critical
        {
            for (auto& pair : V_hash_t) {
                const Determinant& det = pair.first;
                V_hash[det] += pair.second;
            }
        }
        if (tid == 0)
            outfile->Printf("\n  Time spent merging thread F spaces: %20.6f", merge_t.get());
    } // Close threads

    // Remove P space
    const det_hashvec& pdets = P_space.wfn_hash();
    for (det_hashvec::iterator it = pdets.begin(), endit = pdets.end(); it != endit; ++it) {
        V_hash.erase(*it);
    }

    // Loop through hash, compute criteria
    F_space.resize(V_hash.size());
    outfile->Printf("\n  Size of F space: %zu", F_space.size());

    local_timer convert;

#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        size_t N = 0;
        for (const auto& I : V_hash) {
            if (N % num_thread == tid) {
                double delta = as_ints_->energy(I.first) - evals->get(ref_root_);
                double V = I.second;
                double criteria = 0.5 * (delta - sqrt(delta * delta + V * V * 4.0));
                F_space[N] = std::make_pair(std::fabs(criteria), I.first);
            }
            N++;
        }
    }

    outfile->Printf("\n  Time spent building sorting list: %1.6f", convert.get());
}

void AdaptiveCI::get_gas_excited_determinants_avg(
    int nroot, SharedMatrix evecs, std::shared_ptr<psi::Vector> evals, DeterminantHashVec& P_space,
    std::vector<std::pair<double, Determinant>>& F_space) {
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();

    det_hash<std::vector<double>> V_hash;
// Loop over reference determinants
#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        const auto [start_idx, end_idx] = thread_range(max_P, num_thread, tid);

        if (omp_get_thread_num() == 0 and !quiet_mode_) {
            outfile->Printf("\n  Using %d threads.", num_thread);
        }
        // This will store the excited determinant info for each thread
        std::vector<std::pair<Determinant, std::vector<double>>>
            thread_ex_dets; //( noalpha * nvalpha  );

        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double evecs_P_row_norm = evecs->get_row(0, P)->norm();

            // Generate the occupied/virtual alpha/beta orbitals for different GAS
            std::vector<std::vector<int>> gas_occ_a;
            std::vector<std::vector<int>> gas_vir_a;
            std::vector<std::vector<int>> gas_occ_b;
            std::vector<std::vector<int>> gas_vir_b;
            for (size_t gas_count = 0; gas_count < gas_num_; gas_count++) {
                std::vector<int> occ_a;
                std::vector<int> occ_b;
                std::vector<int> vir_a;
                std::vector<int> vir_b;
                for (const auto& p : relative_gas_mo_[gas_count]) {
                    if (det.get_alfa_bit(p)) {
                        occ_a.push_back(p);
                    } else {
                        vir_a.push_back(p);
                    }
                    if (det.get_beta_bit(p)) {
                        occ_b.push_back(p);
                    } else {
                        vir_b.push_back(p);
                    }
                }
                gas_occ_a.push_back(occ_a);
                gas_vir_a.push_back(vir_a);
                gas_occ_b.push_back(occ_b);
                gas_vir_b.push_back(vir_b);
            }

            // Generate the number of electrons in each GAS
            std::vector<int> gas_configuration;
            for (size_t gas_count = 0; gas_count < 6; gas_count++) {
                if (gas_count < gas_num_) {
                    gas_configuration.push_back(gas_occ_a[gas_count].size());
                    gas_configuration.push_back(gas_occ_b[gas_count].size());
                } else {
                    gas_configuration.push_back(0);
                    gas_configuration.push_back(0);
                }
            }

            // Generate a excitations
            for (auto& gas_count : gas_single_criterion_.first[gas_configuration]) {
                size_t gas_count_1 = gas_count.first;
                size_t gas_count_2 = gas_count.second;
                auto& occ = gas_occ_a[gas_count_1];
                auto& vir = gas_vir_a[gas_count_2];
                Determinant new_det(det);
                for (size_t ii : occ) {
                    for (size_t aa : vir) {
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            double HIJ = as_ints_->slater_rules_single_alpha(det, ii, aa);
                            if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                new_det = det;
                                new_det.set_alfa_bit(ii, false);
                                new_det.set_alfa_bit(aa, true);
                                if (!(P_space.has_det(new_det))) {
                                    std::vector<double> coupling(nroot, 0.0);
                                    for (int n = 0; n < nroot; ++n) {
                                        coupling[n] += HIJ * evecs->get(P, n);
                                    }
                                    thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                }
                            }
                        }
                    }
                }
            }

            // Generate b excitations
            for (auto& gas_count : gas_single_criterion_.second[gas_configuration]) {
                size_t gas_count_1 = gas_count.first;
                size_t gas_count_2 = gas_count.second;
                auto& occ = gas_occ_b[gas_count_1];
                auto& vir = gas_vir_b[gas_count_2];
                Determinant new_det(det);
                for (size_t ii : occ) {
                    for (size_t aa : vir) {
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            double HIJ = as_ints_->slater_rules_single_beta(det, ii, aa);
                            if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                new_det = det;
                                new_det.set_beta_bit(ii, false);
                                new_det.set_beta_bit(aa, true);
                                if (!(P_space.has_det(new_det))) {
                                    std::vector<double> coupling(nroot, 0.0);
                                    for (int n = 0; n < nroot; ++n) {
                                        coupling[n] += HIJ * evecs->get(P, n);
                                    }
                                    thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                }
                            }
                        }
                    }
                }
            }

            // Generate aa excitations
            for (auto& gas_count : std::get<0>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_a[gas_count_1];
                auto& occ2 = gas_occ_a[gas_count_2];
                auto& vir1 = gas_vir_a[gas_count_3];
                auto& vir2 = gas_vir_a[gas_count_4];
                for (size_t i = 0, maxi = occ1.size(); i < maxi; ++i) {
                    size_t ii = occ1[i];
                    size_t jstart = (gas_count_1 == gas_count_2 ? i + 1 : 0);
                    for (size_t j = jstart, maxj = occ2.size(); j < maxj; ++j) {
                        size_t jj = occ2[j];
                        for (size_t a = 0, maxa = vir1.size(); a < maxa; ++a) {
                            size_t aa = vir1[a];
                            size_t bstart = (gas_count_3 == gas_count_4 ? a + 1 : 0);
                            for (size_t b = bstart, maxb = vir2.size(); b < maxb; ++b) {
                                size_t bb = vir2[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_aa(ii, jj, aa, bb);
                                    if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                        if (!(P_space.has_det(new_det))) {
                                            std::vector<double> coupling(nroot, 0.0);
                                            for (int n = 0; n < nroot; ++n) {
                                                coupling[n] += HIJ * evecs->get(P, n);
                                            }
                                            thread_ex_dets.push_back(
                                                std::make_pair(new_det, coupling));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate ab excitations
            for (auto& gas_count : std::get<2>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_a[gas_count_1];
                auto& occ2 = gas_occ_b[gas_count_2];
                auto& vir1 = gas_vir_a[gas_count_3];
                auto& vir2 = gas_vir_b[gas_count_4];
                for (size_t ii : occ1) {
                    for (size_t jj : occ2) {
                        for (size_t aa : vir1) {
                            for (size_t bb : vir2) {
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_ab(ii, jj, aa, bb);
                                    if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                        if (!(P_space.has_det(new_det))) {
                                            std::vector<double> coupling(nroot, 0.0);
                                            for (int n = 0; n < nroot; ++n) {
                                                coupling[n] += HIJ * evecs->get(P, n);
                                            }
                                            thread_ex_dets.push_back(
                                                std::make_pair(new_det, coupling));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Genearte bb excitations
            for (auto& gas_count : std::get<1>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_b[gas_count_1];
                auto& occ2 = gas_occ_b[gas_count_2];
                auto& vir1 = gas_vir_b[gas_count_3];
                auto& vir2 = gas_vir_b[gas_count_4];
                for (size_t i = 0, maxi = occ1.size(); i < maxi; ++i) {
                    size_t ii = occ1[i];
                    size_t jstart = (gas_count_1 == gas_count_2 ? i + 1 : 0);
                    for (size_t j = jstart, maxj = occ2.size(); j < maxj; ++j) {
                        size_t jj = occ2[j];
                        for (size_t a = 0, maxa = vir1.size(); a < maxa; ++a) {
                            size_t aa = vir1[a];
                            size_t bstart = (gas_count_3 == gas_count_4 ? a + 1 : 0);
                            for (size_t b = bstart, maxb = vir2.size(); b < maxb; ++b) {
                                size_t bb = vir2[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_bb(ii, jj, aa, bb);
                                    if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);
                                        if (!(P_space.has_det(new_det))) {
                                            std::vector<double> coupling(nroot, 0.0);
                                            for (int n = 0; n < nroot; ++n) {
                                                coupling[n] += HIJ * evecs->get(P, n);
                                            }
                                            thread_ex_dets.push_back(
                                                std::make_pair(new_det, coupling));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for (size_t I = 0, maxI = thread_ex_dets.size(); I < maxI; ++I) {
                std::vector<double>& coupling = thread_ex_dets[I].second;
                Determinant& det = thread_ex_dets[I].first;
                if (V_hash.count(det) != 0) {
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[det][n] += coupling[n];
                    }
                } else {
                    V_hash[det] = coupling;
                }
            }
        }
    } // Close threads

    F_space.resize(V_hash.size());
    outfile->Printf("\n  Size of F space: %zu", F_space.size());

    local_timer convert;

#pragma omp parallel
    {
        size_t tid = omp_get_thread_num();
        size_t ntd = omp_get_num_threads();
        size_t N = 0;
        for (const auto& detpair : V_hash) {
            if (N % ntd == tid) {
                double EI = as_ints_->energy(detpair.first);
                std::vector<double> criteria(nroot, 0.0);
                for (int n = 0; n < nroot; ++n) {
                    double V = detpair.second[n];
                    double delta = EI - evals->get(n);
                    double criterion = 0.5 * (delta - sqrt(delta * delta + V * V * 4.0));
                    criteria[n] = std::fabs(criterion);
                }
                double value = average_q_values(criteria);

                F_space[N] = std::make_pair(value, detpair.first);
            }
            N++;
        }
    }

    outfile->Printf("\n  Time spent building sorting list: %1.6f", convert.get());
}

void AdaptiveCI::get_gas_excited_determinants_core(
    SharedMatrix evecs, std::shared_ptr<psi::Vector> evals, DeterminantHashVec& P_space,
    std::vector<std::pair<double, Determinant>>& F_space) {
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();
    int nroot = 1;
    det_hash<std::vector<double>> V_hash;
// Loop over reference determinants
#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        const auto [start_idx, end_idx] = thread_range(max_P, num_thread, tid);

        if (omp_get_thread_num() == 0 and !quiet_mode_) {
            outfile->Printf("\n  Using %d threads.", num_thread);
        }
        // This will store the excited determinant info for each thread
        std::vector<std::pair<Determinant, std::vector<double>>>
            thread_ex_dets; //( noalpha * nvalpha  );

        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double evecs_P_row_norm = evecs->get_row(0, P)->norm();

            // Generate the occupied/virtual alpha/beta orbitals for different GAS
            std::vector<std::vector<int>> gas_occ_a;
            std::vector<std::vector<int>> gas_vir_a;
            std::vector<std::vector<int>> gas_occ_b;
            std::vector<std::vector<int>> gas_vir_b;
            for (size_t gas_count = 0; gas_count < gas_num_; gas_count++) {
                std::vector<int> occ_a;
                std::vector<int> occ_b;
                std::vector<int> vir_a;
                std::vector<int> vir_b;
                for (const auto& p : relative_gas_mo_[gas_count]) {
                    if (det.get_alfa_bit(p)) {
                        occ_a.push_back(p);
                    } else {
                        vir_a.push_back(p);
                    }
                    if (det.get_beta_bit(p)) {
                        occ_b.push_back(p);
                    } else {
                        vir_b.push_back(p);
                    }
                }
                gas_occ_a.push_back(occ_a);
                gas_vir_a.push_back(vir_a);
                gas_occ_b.push_back(occ_b);
                gas_vir_b.push_back(vir_b);
            }

            // Generate the number of electrons in each GAS
            std::vector<int> gas_configuration;
            for (size_t gas_count = 0; gas_count < 6; gas_count++) {
                if (gas_count < gas_num_) {
                    gas_configuration.push_back(gas_occ_a[gas_count].size());
                    gas_configuration.push_back(gas_occ_b[gas_count].size());
                } else {
                    gas_configuration.push_back(0);
                    gas_configuration.push_back(0);
                }
            }

            // Generate a excitations
            for (auto& gas_count : gas_single_criterion_.first[gas_configuration]) {
                size_t gas_count_1 = gas_count.first;
                size_t gas_count_2 = gas_count.second;
                auto& occ = gas_occ_a[gas_count_1];
                auto& vir = gas_vir_a[gas_count_2];
                Determinant new_det(det);
                for (size_t ii : occ) {
                    for (size_t aa : vir) {
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            double HIJ = as_ints_->slater_rules_single_alpha(det, ii, aa);
                            if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                new_det = det;
                                new_det.set_alfa_bit(ii, false);
                                new_det.set_alfa_bit(aa, true);
                                if (!(P_space.has_det(new_det))) {
                                    std::vector<double> coupling(nroot, 0.0);
                                    for (int n = 0; n < nroot; ++n) {
                                        coupling[n] += HIJ * evecs->get(P, n);
                                    }
                                    thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                }
                            }
                        }
                    }
                }
            }

            // Generate b excitations
            for (auto& gas_count : gas_single_criterion_.second[gas_configuration]) {
                size_t gas_count_1 = gas_count.first;
                size_t gas_count_2 = gas_count.second;
                auto& occ = gas_occ_b[gas_count_1];
                auto& vir = gas_vir_b[gas_count_2];
                Determinant new_det(det);
                for (size_t ii : occ) {
                    for (size_t aa : vir) {
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            double HIJ = as_ints_->slater_rules_single_beta(det, ii, aa);
                            if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                new_det = det;
                                new_det.set_beta_bit(ii, false);
                                new_det.set_beta_bit(aa, true);
                                if (!(P_space.has_det(new_det))) {
                                    std::vector<double> coupling(nroot, 0.0);
                                    for (int n = 0; n < nroot; ++n) {
                                        coupling[n] += HIJ * evecs->get(P, n);
                                    }
                                    thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                }
                            }
                        }
                    }
                }
            }

            // Generate aa excitations
            for (auto& gas_count : std::get<0>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_a[gas_count_1];
                auto& occ2 = gas_occ_a[gas_count_2];
                auto& vir1 = gas_vir_a[gas_count_3];
                auto& vir2 = gas_vir_a[gas_count_4];
                for (size_t i = 0, maxi = occ1.size(); i < maxi; ++i) {
                    size_t ii = occ1[i];
                    size_t jstart = (gas_count_1 == gas_count_2 ? i + 1 : 0);
                    for (size_t j = jstart, maxj = occ2.size(); j < maxj; ++j) {
                        size_t jj = occ2[j];
                        for (size_t a = 0, maxa = vir1.size(); a < maxa; ++a) {
                            size_t aa = vir1[a];
                            size_t bstart = (gas_count_3 == gas_count_4 ? a + 1 : 0);
                            for (size_t b = bstart, maxb = vir2.size(); b < maxb; ++b) {
                                size_t bb = vir2[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_aa(ii, jj, aa, bb);
                                    if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                        if (!(P_space.has_det(new_det))) {
                                            std::vector<double> coupling(nroot, 0.0);
                                            for (int n = 0; n < nroot; ++n) {
                                                coupling[n] += HIJ * evecs->get(P, n);
                                            }
                                            thread_ex_dets.push_back(
                                                std::make_pair(new_det, coupling));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate ab excitations
            for (auto& gas_count : std::get<2>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_a[gas_count_1];
                auto& occ2 = gas_occ_b[gas_count_2];
                auto& vir1 = gas_vir_a[gas_count_3];
                auto& vir2 = gas_vir_b[gas_count_4];
                for (size_t ii : occ1) {
                    for (size_t jj : occ2) {
                        for (size_t aa : vir1) {
                            for (size_t bb : vir2) {
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_ab(ii, jj, aa, bb);
                                    if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                        if (!(P_space.has_det(new_det))) {
                                            std::vector<double> coupling(nroot, 0.0);
                                            for (int n = 0; n < nroot; ++n) {
                                                coupling[n] += HIJ * evecs->get(P, n);
                                            }
                                            thread_ex_dets.push_back(
                                                std::make_pair(new_det, coupling));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Genearte bb excitations
            for (auto& gas_count : std::get<1>(gas_double_criterion_)[gas_configuration]) {
                Determinant new_det(det);
                size_t gas_count_1 = std::get<0>(gas_count);
                size_t gas_count_2 = std::get<1>(gas_count);
                size_t gas_count_3 = std::get<2>(gas_count);
                size_t gas_count_4 = std::get<3>(gas_count);
                auto& occ1 = gas_occ_b[gas_count_1];
                auto& occ2 = gas_occ_b[gas_count_2];
                auto& vir1 = gas_vir_b[gas_count_3];
                auto& vir2 = gas_vir_b[gas_count_4];
                for (size_t i = 0, maxi = occ1.size(); i < maxi; ++i) {
                    size_t ii = occ1[i];
                    size_t jstart = (gas_count_1 == gas_count_2 ? i + 1 : 0);
                    for (size_t j = jstart, maxj = occ2.size(); j < maxj; ++j) {
                        size_t jj = occ2[j];
                        for (size_t a = 0, maxa = vir1.size(); a < maxa; ++a) {
                            size_t aa = vir1[a];
                            size_t bstart = (gas_count_3 == gas_count_4 ? a + 1 : 0);
                            for (size_t b = bstart, maxb = vir2.size(); b < maxb; ++b) {
                                size_t bb = vir2[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    double HIJ = as_ints_->tei_bb(ii, jj, aa, bb);
                                    if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                        new_det = det;
                                        HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);
                                        if (!(P_space.has_det(new_det))) {
                                            std::vector<double> coupling(nroot, 0.0);
                                            for (int n = 0; n < nroot; ++n) {
                                                coupling[n] += HIJ * evecs->get(P, n);
                                            }
                                            thread_ex_dets.push_back(
                                                std::make_pair(new_det, coupling));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for (size_t I = 0, maxI = thread_ex_dets.size(); I < maxI; ++I) {
                std::vector<double>& coupling = thread_ex_dets[I].second;
                Determinant& det = thread_ex_dets[I].first;
                if (V_hash.count(det) != 0) {
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[det][n] += coupling[n];
                    }
                } else {
                    V_hash[det] = coupling;
                }
            }
        }
    } // Close threads

    F_space.resize(V_hash.size());
    outfile->Printf("\n  Size of F space: %zu", F_space.size());

    local_timer convert;

#pragma omp parallel
    {
        size_t tid = omp_get_thread_num();
        size_t ntd = omp_get_num_threads();
        size_t N = 0;
        for (const auto& detpair : V_hash) {
            if (N % ntd == tid) {
                double EI = as_ints_->energy(detpair.first);
                std::vector<double> criteria(nroot, 0.0);
                for (int n = 0; n < nroot; ++n) {
                    double V = detpair.second[n];
                    double delta = EI - evals->get(n);
                    double criterion = 0.5 * (delta - sqrt(delta * delta + V * V * 4.0));
                    criteria[n] = std::fabs(criterion);
                }
                double value = average_q_values(criteria);

                F_space[N] = std::make_pair(value, detpair.first);
            }
            N++;
        }
    }

    outfile->Printf("\n  Time spent building sorting list: %1.6f", convert.get());
}
} // namespace forte
