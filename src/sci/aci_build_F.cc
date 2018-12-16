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


#include "aci.h"

using namespace psi;

namespace forte {

bool pair_comp(const std::pair<double, Determinant> E1, const std::pair<double, Determinant> E2) {
    return E1.first < E2.first;
}
bool pair_comp2(const std::pair<Determinant, double> E1, const std::pair<Determinant, double> E2) {
    return E1.second < E2.second;
}
bool pair_compd(const std::pair<Determinant, double> E1, const std::pair<Determinant, double> E2) {
    return E1.first < E2.first;
}

void AdaptiveCI::get_excited_determinants_sr(psi::SharedMatrix evecs, DeterminantHashVec& P_space,
                                             det_hash<double>& V_hash) {
    local_timer build;
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();

// Loop over reference determinants
#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        size_t bin_size = max_P / num_thread;
        bin_size += (tid < (max_P % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (max_P % num_thread))
                ? tid * bin_size
                : (max_P % num_thread) * (bin_size + 1) + (tid - (max_P % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        det_hash<double> V_hash_t;
        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double Cp = evecs->get(P, 0);

            std::vector<int> aocc = det.get_alfa_occ(nact_); // TODO check size
            std::vector<int> bocc = det.get_beta_occ(nact_); // TODO check size
            std::vector<int> avir = det.get_alfa_vir(nact_); // TODO check size
            std::vector<int> bvir = det.get_beta_vir(nact_); // TODO check size

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            Determinant new_det(det);
            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa) * Cp;
                        if (std::abs(HIJ) >= screen_thresh_) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            V_hash_t[new_det] += HIJ;
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa) * Cp;
                        if (std::abs(HIJ) >= screen_thresh_) {
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            V_hash_t[new_det] += HIJ;
                        }
                    }
                }
            }
            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < noalpha; ++j) {
                    int jj = aocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = a + 1; b < nvalpha; ++b) {
                            int bb = avir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb) * Cp;
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
            // Generate ab excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = 0; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb) * Cp;
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

            // Generate bb excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        for (int b = a + 1; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb) * Cp;
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
}

void AdaptiveCI::get_excited_determinants_seq(int nroot, psi::SharedMatrix evecs,
                                              DeterminantHashVec& P_space,
                                              det_hash<std::vector<double>>& V_hash) {
    const size_t n_dets = P_space.size();

    int nmo = fci_ints_->nmo();
    double max_mem = options_.get_double("PT2_MAX_MEM");

    size_t guess_size = n_dets * nmo * nmo;
    double nbyte = (1073741824 * max_mem) / (sizeof(double));

    size_t nbin = std::ceil(guess_size / (nbyte));

#pragma omp parallel
    {
        size_t tid = omp_get_thread_num();
        size_t ntds = omp_get_num_threads();

        if ((ntds > nbin)) {
            nbin = ntds;
        }

        if (tid == 0) {
            outfile->Printf("\n  Number of bins for exitation space:  %d", nbin);
            outfile->Printf("\n  Number of threads: %d", ntds);
        }
        size_t bin_size = n_dets / ntds;
        bin_size += (tid < (n_dets % ntds)) ? 1 : 0;
        size_t start_idx = (tid < (n_dets % ntds)) ? tid * bin_size
                                                   : (n_dets % ntds) * (bin_size + 1) +
                                                         (tid - (n_dets % ntds)) * bin_size;
        size_t end_idx = start_idx + bin_size;
        for (size_t bin = 0; bin < nbin; ++bin) {

            det_hash<std::vector<double>> A_I;
            // std::vector<std::pair<Determinant, std::vector<double>>> A_I;

            const det_hashvec& dets = P_space.wfn_hash();
            for (size_t I = start_idx; I < end_idx; ++I) {
                double c_norm = evecs->get_row(0, I)->norm();
                const Determinant& det = dets[I];
                std::vector<int> aocc = det.get_alfa_occ(nmo);
                std::vector<int> bocc = det.get_beta_occ(nmo);
                std::vector<int> avir = det.get_alfa_vir(nmo);
                std::vector<int> bvir = det.get_beta_vir(nmo);

                int noalpha = aocc.size();
                int nobeta = bocc.size();
                int nvalpha = avir.size();
                int nvbeta = bvir.size();
                Determinant new_det(det);

                // Generate alpha excitations
                for (int i = 0; i < noalpha; ++i) {
                    int ii = aocc[i];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if (P_space.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = Determinant::Hash()(new_det);
                            if ((hash_val % nbin) == bin) {
                                double HIJ = fci_ints_->slater_rules_single_alpha(new_det, ii, aa);
                                if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                    std::vector<double> coupling(nroot_);
                                    for (int n = 0; n < nroot_; ++n) {
                                        coupling[n] = HIJ * evecs->get(I, n);
                                        if (A_I.find(new_det) != A_I.end()) {
                                            coupling[n] += A_I[new_det][n];
                                        }
                                        A_I[new_det] = coupling;
                                    }
                                }
                            }
                        }
                    }
                }
                // Generate beta excitations
                for (int i = 0; i < nobeta; ++i) {
                    int ii = bocc[i];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if (P_space.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = Determinant::Hash()(new_det);
                            if ((hash_val % nbin) == bin) {
                                double HIJ = fci_ints_->slater_rules_single_beta(new_det, ii, aa);
                                if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                    std::vector<double> coupling(nroot_);
                                    for (int n = 0; n < nroot_; ++n) {
                                        coupling[n] = HIJ * evecs->get(I, n);
                                        if (A_I.find(new_det) != A_I.end()) {
                                            coupling[n] += A_I[new_det][n];
                                        }
                                        A_I[new_det] = coupling;
                                    }
                                }
                            }
                        }
                    }
                }
                // Generate aa excitations
                for (int i = 0; i < noalpha; ++i) {
                    int ii = aocc[i];
                    for (int j = i + 1; j < noalpha; ++j) {
                        int jj = aocc[j];
                        for (int a = 0; a < nvalpha; ++a) {
                            int aa = avir[a];
                            for (int b = a + 1; b < nvalpha; ++b) {
                                int bb = avir[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_aa(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Generate bb excitations
                for (int i = 0; i < nobeta; ++i) {
                    int ii = bocc[i];
                    for (int j = i + 1; j < nobeta; ++j) {
                        int jj = bocc[j];
                        for (int a = 0; a < nvbeta; ++a) {
                            int aa = bvir[a];
                            for (int b = a + 1; b < nvbeta; ++b) {
                                int bb = bvir[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_bb(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Generate ab excitations
                for (int i = 0; i < noalpha; ++i) {
                    int ii = aocc[i];
                    for (int j = 0; j < nobeta; ++j) {
                        int jj = bocc[j];
                        for (int a = 0; a < nvalpha; ++a) {
                            int aa = avir[a];
                            for (int b = 0; b < nvbeta; ++b) {
                                int bb = bvir[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_ab(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
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
                for (auto& dpair : A_I) {
                    const std::vector<double>& coupling = dpair.second;
                    const Determinant& det = dpair.first;
                    if (V_hash.count(det) != 0) {
                        for (int n = 0; n < nroot; ++n) {
                            V_hash[det][n] += coupling[n];
                        }
                    } else {
                        V_hash[det] = coupling;
                    }
                }
            }
            // outfile->Printf("\n TD, bin, size of hash: %d %d %zu", tid, bin, A_I.size());
        }
    }
}

void AdaptiveCI::get_excited_determinants(int nroot, psi::SharedMatrix evecs,
                                          DeterminantHashVec& P_space,
                                          det_hash<std::vector<double>>& V_hash) {
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();

// Loop over reference determinants
#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        size_t bin_size = max_P / num_thread;
        bin_size += (tid < (max_P % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (max_P % num_thread))
                ? tid * bin_size
                : (max_P % num_thread) * (bin_size + 1) + (tid - (max_P % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        if (omp_get_thread_num() == 0 and !quiet_mode_) {
            outfile->Printf("\n  Using %d threads.", num_thread);
        }
        // This will store the excited determinant info for each thread
        std::vector<std::pair<Determinant, std::vector<double>>>
            thread_ex_dets; //( noalpha * nvalpha  );

        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double evecs_P_row_norm = evecs->get_row(0, P)->norm();

            std::vector<int> aocc = det.get_alfa_occ(nact_);
            std::vector<int> bocc = det.get_beta_occ(nact_);
            std::vector<int> avir = det.get_alfa_vir(nact_);
            std::vector<int> bvir = det.get_beta_vir(nact_);

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            Determinant new_det(det);

            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa);
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            //      if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * noalpha + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }

            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa);
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * nobeta + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < noalpha; ++j) {
                    int jj = aocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = a + 1; b < nvalpha; ++b) {
                            int bb = avir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i *
                                        // noalpha*noalpha*nvalpha +
                                        // j*nvalpha*noalpha +  a*nvalpha + b ]
                                        // = std::make_pair(new_det,coupling);
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate ab excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = 0; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i * nobeta * nvalpha
                                        // *nvbeta + j * bvalpha * nvbeta + a *
                                        // nvalpha]
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate bb excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        for (int b = a + 1; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^
                                 (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    // if( std::abs(HIJ * evecs->get(0, P)) >= screen_thresh_ ){
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);

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
}

void AdaptiveCI::get_core_excited_determinants(psi::SharedMatrix evecs, DeterminantHashVec& P_space,
                                               det_hash<std::vector<double>>& V_hash) {
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();
    int nroot = 1;

// Loop over reference determinants
#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        size_t bin_size = max_P / num_thread;
        bin_size += (tid < (max_P % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (max_P % num_thread))
                ? tid * bin_size
                : (max_P % num_thread) * (bin_size + 1) + (tid - (max_P % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        if (omp_get_thread_num() == 0 and !quiet_mode_) {
            outfile->Printf("\n  Using %d threads.", num_thread);
        }
        // This will store the excited determinant info for each thread
        std::vector<std::pair<Determinant, std::vector<double>>>
            thread_ex_dets; //( noalpha * nvalpha  );

        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double evecs_P_row_norm = evecs->get_row(0, P)->norm();

            std::vector<int> aocc = det.get_alfa_occ(nact_); // TODO check size
            std::vector<int> bocc = det.get_beta_occ(nact_); // TODO check size
            std::vector<int> avir = det.get_alfa_vir(nact_); // TODO check size
            std::vector<int> bvir = det.get_beta_vir(nact_); // TODO check size

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            Determinant new_det(det);

            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if (((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) and
                        ((aa != hole_) or (!det.get_beta_bit(aa)))) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa);
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            //      if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * noalpha + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if (((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) and
                        ((aa != hole_) or (!det.get_alfa_bit(aa)))) {
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa);
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * nobeta + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < noalpha; ++j) {
                    int jj = aocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = a + 1; b < nvalpha; ++b) {
                            int bb = avir[b];
                            if (((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                  mo_symmetry_[bb]) == 0) and
                                (aa != hole_ and bb != hole_)) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i *
                                        // noalpha*noalpha*nvalpha +
                                        // j*nvalpha*noalpha +  a*nvalpha + b ]
                                        // = std::make_pair(new_det,coupling);
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate ab excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = 0; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if (((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                  mo_symmetry_[bb]) == 0) and
                                (aa != hole_ and bb != hole_)) {
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i * nobeta * nvalpha
                                        // *nvbeta + j * bvalpha * nvbeta + a *
                                        // nvalpha]
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate bb excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        for (int b = a + 1; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if (((mo_symmetry_[ii] ^
                                  (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) ==
                                 0) and
                                (aa != hole_ and bb != hole_)) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    // if( std::abs(HIJ * evecs->get(0, P)) >= screen_thresh_ ){
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);

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
}

double AdaptiveCI::get_excited_determinants_batch_old(
    psi::SharedMatrix evecs, psi::SharedVector evals, DeterminantHashVec& P_space,
    std::vector<std::pair<double, Determinant>>& F_space) {
    const size_t n_dets = P_space.size();

    int nmo = fci_ints_->nmo();
    double max_mem = options_.get_double("ACI_MAX_MEM");
    double aci_scale = options_.get_double("ACI_SCALE_SIGMA");

    size_t guess_size = n_dets * nmo * nmo;
    double guess_mem = guess_size * 400.0e-7; // Est of map size in MB
    int nruns = static_cast<int>(std::ceil(guess_mem / max_mem));

    double total_excluded = 0.0;

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int n_threads = omp_get_max_threads();
        int nbin = nruns * n_threads;
        if (thread_id == 0) {
            outfile->Printf("\n  Setting nbin to %d based on estimated memory (%6.3f MB)", nbin,
                            guess_mem);
        }

        if (n_threads >= nbin) {
            nbin = n_threads;
            if (thread_id == 0) {
                outfile->Printf("\n  WARNING: Batching not required, use another algorithm");
            }
        }

        if (options_["ACI_NBATCH"].has_changed()) {
            nbin = options_.get_int("ACI_NBATCH");
            if (thread_id == 0) {
                outfile->Printf("\n  Overwriting nbin to %d based on user input", nbin);
            }
        }
        std::vector<std::pair<double, Determinant>> F_td;

        int bin_size = nbin / n_threads;
        bin_size += (thread_id < (nbin % n_threads)) ? 1 : 0;
        int start_idx =
            (thread_id < (nbin % n_threads))
                ? thread_id * bin_size
                : (nbin % n_threads) * (bin_size + 1) + (thread_id - (nbin % n_threads)) * bin_size;
        int end_idx = start_idx + bin_size;

        if (thread_id == 0) {
            outfile->Printf("\n  Number of bins for exitation space:  %d", nbin);
            outfile->Printf("\n  Number of threads: %d", n_threads);
        }

        // Loop over bins
        for (int bin = start_idx; bin < end_idx; ++bin) {

            local_timer sp;
            // 1. Build the full bin-subset
            det_hash<double> A_b = get_bin_F_space_old(bin, nbin, evecs, P_space);
            outfile->Printf("\n  F subspace (bin %d) takes %1.6f s", bin, sp.get());

            // 2. Put the dets/vals in a sortable list (F_tmp)
            std::vector<std::pair<double, Determinant>> F_tmp;
            for (auto& det_p : A_b) {
                auto& det = det_p.first;
                double& V = det_p.second;

                double delta = fci_ints_->energy(det) - evals->get(0);

                F_tmp.push_back(std::make_pair(
                    std::fabs(0.5 * (delta - sqrt(delta * delta + V * V * 4.0))), det));
            }
            A_b.clear();
            outfile->Printf("\n  Size of F subspace: %zu dets (bin %d)", F_tmp.size(), bin);

            // 3. Sort the list
            std::sort(F_tmp.begin(), F_tmp.end(), pair_comp);

            // 4. Screen subspaces
            double b_sigma = sigma_ * (aci_scale / nbin);
            //            outfile->Printf("\n  Using sigma = %1.5f", b_sigma);
            double excluded = 0.0;
            size_t ndet = 0;
            for (size_t I = 0, max_I = F_tmp.size(); I < max_I; ++I) {
                double en = F_tmp[I].first;
                Determinant& det = F_tmp[I].second;
                if (excluded + en < b_sigma) {
                    excluded += en;
                } else {
                    F_td.push_back(std::make_pair(en, det));
                    ndet++;
                }
            }
            outfile->Printf("\n  Added %zu dets from bin %d, (td %d)", ndet, bin, thread_id);
            total_excluded += excluded;

        } // End iteration over bins

// 6. Merge thread spaces
#pragma omp critical
        {
            for (auto& pair : F_td) {
                F_space.push_back(pair);
            }
        }
    } // close threads

    outfile->Printf("\n  Screened out %1.10f Eh of correlation", total_excluded);
    return total_excluded;
}

det_hash<double> AdaptiveCI::get_bin_F_space_old(int bin, int nbin, psi::SharedMatrix evecs,
                                                 DeterminantHashVec& P_space) {

    const size_t n_dets = P_space.size();
    const det_hashvec& dets = P_space.wfn_hash();
    std::vector<int> act_mo = mo_space_info_->get_dimension("ACTIVE").blocks();
    det_hash<double> A_b;

    // Loop over P space determinants
    for (size_t I = 0; I < n_dets; ++I) {
        double c_I = evecs->get(I, 0);
        const Determinant& det = dets[I];
        std::vector<std::vector<int>> noalpha = det.get_asym_occ(act_mo);
        std::vector<std::vector<int>> nobeta = det.get_bsym_occ(act_mo);
        std::vector<std::vector<int>> nvalpha = det.get_asym_vir(act_mo);
        std::vector<std::vector<int>> nvbeta = det.get_bsym_vir(act_mo);

        Determinant new_det(det);
        // Generate alpha excitations
        for (int h = 0; h < nirrep_; ++h) {

            // Precompute indices
            const auto& noalpha_h = noalpha[h];
            const auto& nvalpha_h = nvalpha[h];

            for (auto& ii : noalpha_h) {
                for (auto& aa : nvalpha_h) {
                    new_det = det;
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    size_t hash_val = Determinant::Hash()(new_det);
                    if ((hash_val % nbin) == bin) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa) * c_I;
                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                            A_b[new_det] += HIJ;
                        }
                    }
                }
            }
            // Generate beta excitations
            const auto& nobeta_h = nobeta[h];
            const auto& nvbeta_h = nvbeta[h];
            for (auto& ii : nobeta_h) {
                for (auto& aa : nvbeta_h) {
                    // Check if the determinant goes in this bin
                    new_det = det;
                    new_det.set_beta_bit(ii, false);
                    new_det.set_beta_bit(aa, true);
                    size_t hash_val = Determinant::Hash()(new_det);
                    if ((hash_val % nbin) == bin) {
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa) * c_I;
                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                            A_b[new_det] += HIJ;
                        }
                    }
                }
            }
        }
        for (int p = 0; p < nirrep_; ++p) {
            for (int q = p; q < nirrep_; ++q) {
                for (int r = 0; r < nirrep_; ++r) {
                    int sp = p ^ q ^ r;
                    if (sp < r)
                        continue;

                    // Precompute index lists
                    const auto& noalpha_p = noalpha[p];
                    const auto& noalpha_q = noalpha[q];
                    const auto& nvalpha_r = nvalpha[r];
                    const auto& nvalpha_s = nvalpha[sp];

                    int max_i = noalpha_p.size();
                    int max_j = noalpha_q.size();
                    int max_a = nvalpha_r.size();
                    int max_b = nvalpha_s.size();

                    // Generate aa excitations
                    for (int i = 0; i < max_i; ++i) {
                        int ii = noalpha_p[i];
                        for (int j = (p == q ? i + 1 : 0); j < max_j; ++j) {
                            int jj = noalpha_q[j];
                            for (int a = 0; a < max_a; ++a) {
                                int aa = nvalpha_r[a];
                                for (int b = (r == sp ? a + 1 : 0); b < max_b; ++b) {
                                    int bb = nvalpha_s[b];

                                    // Check if the determinant goes in this bin
                                    new_det = det;
                                    double sign = new_det.double_excitation_aa(ii, jj, aa, bb);
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb) * sign * c_I;
                                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                                            A_b[new_det] += HIJ;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Generate bb excitations
                    const auto& nobeta_p = nobeta[p];
                    const auto& nobeta_q = nobeta[q];
                    const auto& nvbeta_r = nvbeta[r];
                    const auto& nvbeta_s = nvbeta[sp];

                    max_i = nobeta_p.size();
                    max_j = nobeta_q.size();
                    max_a = nvbeta_r.size();
                    max_b = nvbeta_s.size();

                    for (int i = 0; i < max_i; ++i) {
                        int ii = nobeta_p[i];
                        for (int j = (p == q ? i + 1 : 0); j < max_j; ++j) {
                            int jj = nobeta_q[j];
                            for (int a = 0; a < max_a; ++a) {
                                int aa = nvbeta_r[a];
                                for (int b = (r == sp ? a + 1 : 0); b < max_b; ++b) {
                                    int bb = nvbeta_s[b];
                                    // Check if the determinant goes in this bin
                                    new_det = det;
                                    double sign = new_det.double_excitation_bb(ii, jj, aa, bb);
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb) * sign * c_I;
                                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                                            A_b[new_det] += HIJ;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int p = 0; p < nirrep_; ++p) {
            for (int q = 0; q < nirrep_; ++q) {
                for (int r = 0; r < nirrep_; ++r) {
                    int sp = p ^ q ^ r;
                    const auto& noalpha_p = noalpha[p];
                    const auto& nobeta_q = nobeta[q];
                    const auto& nvalpha_r = nvalpha[r];
                    const auto& nvbeta_s = nvbeta[sp];
                    // Generate ab excitations
                    for (auto& ii : noalpha_p) {
                        for (auto& jj : nobeta_q) {
                            for (auto& aa : nvalpha_r) {
                                for (auto& bb : nvbeta_s) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_ab(ii, jj, aa, bb);
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb) * sign * c_I;
                                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                                            A_b[new_det] += HIJ;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // end loop over reference

    // Remove duplicates
    for (det_hashvec::iterator it = dets.begin(), endit = dets.end(); it != endit; ++it) {
        A_b[*it] = 0.0;
    }

    return A_b;
}

// New threading strategy
double AdaptiveCI::get_excited_determinants_batch_vecsort(
    psi::SharedMatrix evecs, psi::SharedVector evals, DeterminantHashVec& P_space,
    std::vector<std::pair<double, Determinant>>& F_space) {
    const size_t n_dets = P_space.size();

    int nmo = fci_ints_->nmo();
    double max_mem = options_.get_double("ACI_MAX_MEM");
    double aci_scale = options_.get_double("ACI_SCALE_SIGMA");

    // Guess the total memory needed to store all singles and doubles out of all dets
    //    size_t nsingle_a = nalpha_ * (ncmo_ - nalpha_);
    //    size_t nsingle_b = nbeta_ * (ncmo_ - nbeta_);
    //    size_t ndouble_aa = nalpha_ * (nalpha_ - 1) * (ncmo_ - nalpha_) * (ncmo_ - nalpha_ - 1) /
    //    4;
    //    size_t ndouble_bb = nbeta_ * (nbeta_ - 1) * (ncmo_ - nbeta_) * (ncmo_ - nbeta_ - 1) / 4;
    //    size_t ndouble_ab = nsingle_a * nsingle_b;
    //    size_t nexcitations = nsingle_a + nsingle_b + ndouble_aa + ndouble_bb + ndouble_ab;
    //    size_t guess_size = n_dets * nexcitations;

    size_t nocc2 = nalpha_ * nalpha_;
    size_t nvir2 = (nmo - nalpha_) * (nmo - nalpha_);
    size_t guess_size = n_dets * nocc2 * nvir2;

    double guess_mem = guess_size * (4.0 + double(Determinant::num_det_bits)) * 1.25e-7 *
                       1.4; // Est of map size in MB
    int nruns = static_cast<int>(std::ceil(guess_mem / max_mem));

    double total_excluded = 0.0;
    int nbin = nruns;
    outfile->Printf("\n  Setting nbin to %d based on estimated memory (%6.3f MB)", nbin, guess_mem);

    if (options_["ACI_NBATCH"].has_changed()) {
        nbin = options_.get_int("ACI_NBATCH");
        outfile->Printf("\n  Overwriting nbin to %d based on user input", nbin);
    }

    // Loop over bins
    outfile->Printf("\n -----------------------------------------");
    for (int bin = 0; bin < nbin; ++bin) {
        outfile->Printf("\n                Bin %d", bin);
        local_timer sp;
        // 1. Build the full bin-subset // all threading in here
        auto A_b_t = get_bin_F_space_vecsort(bin, nbin, evecs, P_space);
        outfile->Printf("\n    Build F                %10.6f ", sp.get());

        // 2. Put the dets/vals in a sortable list (F_tmp)
        local_timer bint;
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& A_b = A_b_t.first[tid];
            double E0 = evals->get(0);
            for (size_t I = 0, max_I = A_b_t.second[tid]; I < max_I; I++) {
                auto& pair = A_b[I];
                double& V = pair.second;
                if (V != 0.0) {
                    auto& det = pair.first;
                    double delta = fci_ints_->energy(det) - E0;
                    V = std::fabs(0.5 * (delta - sqrt(delta * delta + V * V * 4.0)));
                }
            }
        }
        outfile->Printf("\n    Build criteria vector  %10.6f", bint.get());

        // 3. Sort the list
        //  local_timer sortt;
        //  #pragma omp parallel
        //  {
        //      int tid = omp_get_thread_num();
        //      std::sort(A_b_t.first[tid].begin(), A_b_t.first[tid].begin() + A_b_t.second[tid],
        //                [](const std::pair<Determinant, double>& a,
        //                   const std::pair<Determinant, double>& b) -> bool {
        //                    return a.second < b.second;
        //                });
        //  }
        //  outfile->Printf("\n    Sort vector            %10.6f", sortt.get());

        // 4. Screen subspaces
        // Can it be done without recombining/resorting?
        local_timer screener;
        // single-thread algorithm
        double b_sigma = sigma_ * (aci_scale / nbin);
        double excluded = 0.0;
        size_t total_size = 0;

        for (size_t& s : A_b_t.second) {
            total_size += s;
        }

        // Test threaded det generation
        std::vector<std::pair<Determinant, double>> master;
        master.reserve(total_size);
#pragma omp parallel
        {
            int tid = omp_get_thread_num();

            auto& A_b = A_b_t.first[tid];

#pragma omp critical
            {
                //        outfile->Printf("\n Ab(%d) size = %zu", tid, A_b.size());
                for (size_t I = 0, max_I = A_b_t.second[tid]; I < max_I; ++I) {
                    master.push_back(A_b[I]);
                }
            }
        }
        std::sort(
            master.begin(), master.end(),
            [](const std::pair<Determinant, double>& a,
               const std::pair<Determinant, double>& b) -> bool { return a.second < b.second; });

        for (auto& pair : master) {
            const double& en = pair.second;
            const auto& det = pair.first;
            if (excluded + en < b_sigma) {
                excluded += en;
            } else {
                F_space.push_back(std::make_pair(en, det));
            }
        }
        total_excluded += excluded;

        /*
                local_timer sc;

                int num_threads = A_b_t.second.size();
                std::vector<std::pair<std::pair<Determinant,double>,int>> criteria(num_threads);
                std::vector<size_t> indices(num_threads, 0);
                bool include = false;
                int count = 0;

                std::vector<int> good_threads;
                for( int i = 0; i < num_threads; ++i ){
                    if( A_b_t.first[i].size() > 0 ){
                        good_threads.push_back(i);
                    }
                }

                while( !include ){
                    // Get lowest energy determinant from each thread
                    for( int& i : good_threads ){
                        size_t index = indices[i];
                        const auto& A_b = A_b_t.first[i];
        //                if( count == 0 ){
        //                    outfile->Printf("\n Ab(%d) size = %zu, second = %zu", i, A_b.size(),
        A_b_t.second[i]);
        //                }
                        criteria[i] = std::make_pair(A_b[index], i);
                    }

                    // Sort the lowest criteria
                    std::sort(criteria.begin(), criteria.end(),
                        [](const std::pair<std::pair<Determinant, double>,int>& a,
                           const std::pair<std::pair<Determinant, double>,int>& b) -> bool {
                            return a.first.second < b.first.second;
                        });
        //            outfile->Printf("\n");
        //            for( int i = 0; i < num_threads; ++i ){
        //                outfile->Printf("\n   (%d, %d, %1.12f)", criteria[i].second, indices[i],
        criteria[i].first.second);
        //            }

                    // Add or remove lowest one
                    auto& pair = criteria[0].first;
                    double en = pair.second;
                    if (excluded + en < b_sigma) {
                        excluded += en;
                        size_t s_td = criteria[0].second;
                        indices[s_td]++;
        //                outfile->Printf("\n %d: (%zu)  %1.14f",s_td, count, en);
                    } else {
                        include = true;
                    }

                    good_threads.clear();
                    for( int i = 0; i < num_threads; ++i ){
                        size_t idx = indices[i];
                        size_t max = A_b_t.second[i];
        //                outfile->Printf("\n  td:%d idx: %d max:%d", i, idx, max);
                        if( (idx < max) and (max > 0) ){
                            good_threads.push_back(i);
        //                    outfile->Printf("\n keep thread %d", i);
                        } else {
                            double& val = criteria[i].first.second;
                            val = 1.0;
                        }
                    }
                    count++;

                }
                total_excluded += excluded;
                // Now add the remainder to the model space
                local_timer fill;
                for( int& i : good_threads ){
                    auto& A_b = A_b_t.first[i];
                    size_t start = indices[i] ;
                    F_space.reserve( F_space.size() + A_b_t.second[i] );
                    for( int I = start, end = A_b_t.second[i]; I < end; ++I ){
                        auto& pair = A_b[I];
                        const double& en = pair.second;
                        const auto& det = pair.first;
                        F_space.push_back(std::make_pair(en,det));
                    }
                }
                outfile->Printf("\n    Fill vector              %1.6f", fill.get());
        */

        outfile->Printf("\n    Screening              %10.6f", screener.get());
        outfile->Printf("\n    Added %zu dets of %zu from bin %d", F_space.size(), total_size, bin);
    } // End iteration over bins

    outfile->Printf("\n ------------------------------------");
    outfile->Printf("\n  Screened out %1.10f Eh of correlation", total_excluded);
    return total_excluded;
} // namespace forte

// New threading strategy
double
AdaptiveCI::get_excited_determinants_batch(psi::SharedMatrix evecs, psi::SharedVector evals,
                                           DeterminantHashVec& P_space,
                                           std::vector<std::pair<double, Determinant>>& F_space) {
    const size_t n_dets = P_space.size();

    int nmo = fci_ints_->nmo();
    double max_mem = options_.get_double("ACI_MAX_MEM");
    double aci_scale = options_.get_double("ACI_SCALE_SIGMA");

    size_t nocc2 = nalpha_ * nalpha_;
    size_t nvir2 = (nmo - nalpha_) * (nmo - nalpha_);
    size_t guess_size = n_dets * nocc2 * nvir2;
    // outfile->Printf("\n  guess_size: %zu o: %zu, v: %zu", guess_size, nocc2, nvir2);
    double guess_mem = guess_size * (4.0 + double(Determinant::num_det_bits)) * 1.25e-7 *
                       1.4; // Est of map size in MB
    int nruns = static_cast<int>(std::ceil(guess_mem / max_mem));

    double total_excluded = 0.0;
    int nbin = nruns;
    outfile->Printf("\n  Setting nbin to %d based on estimated memory (%6.3f MB)", nbin, guess_mem);

    if (options_["ACI_NBATCH"].has_changed()) {
        nbin = options_.get_int("ACI_NBATCH");
        outfile->Printf("\n  Overwriting nbin to %d based on user input", nbin);
    }

    // Loop over bins
    outfile->Printf("\n -----------------------------------------");
    for (int bin = 0; bin < nbin; ++bin) {
        outfile->Printf("\n                Bin %d", bin);
        local_timer sp;

        // 0. Get energy estimate from prescreening.
        //        total_excluded += prescreen_F(bin,nbin,evals->get(0), evecs,P_space);

        // 1. Build the full bin-subset // all threading in here
        auto A_b = get_bin_F_space(bin, nbin, evecs, P_space);
        outfile->Printf("\n    Build F                %10.6f ", sp.get());

        // 2. Put the dets/vals in a sortable list (F_tmp)
        local_timer bint;

        // get sizes
        size_t subspace_size = A_b.size();
        std::vector<std::pair<double, Determinant>> F_tmp(subspace_size);
#pragma omp parallel
        {
            int ntd = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int idx = 0;
            double E0 = evals->get(0);
            for (auto& pair : A_b) {
                if ((idx % ntd) == tid) {
                    auto& det = pair.first;
                    double& V = pair.second;

                    // double delta = fci_ints_->energy(det) - E0;
                    double delta = fci_ints_->energy(det) - E0;

                    F_tmp[idx] = std::make_pair(
                        std::fabs(0.5 * (delta - sqrt(delta * delta + V * V * 4.0))), det);
                }
                idx++;
            }
        }
        A_b.clear();

        outfile->Printf("\n    Build criteria vector  %10.6f", bint.get());

        // 3. Sort the list
        local_timer sortt;
        std::sort(F_tmp.begin(), F_tmp.end(), pair_comp);
        outfile->Printf("\n    Sort vector            %10.6f", sortt.get());

        // 4. Screen subspaces
        local_timer screener;
        double b_sigma = sigma_ * (aci_scale / nbin);
        double excluded = 0.0;
        for (size_t I = 0, max_I = F_tmp.size(); I < max_I; ++I) {
            double en = F_tmp[I].first;
            Determinant& det = F_tmp[I].second;
            if (excluded + en < b_sigma) {
                excluded += en;
            } else {
                F_space.push_back(std::make_pair(en, det));
            }
        }
        total_excluded += excluded;
        outfile->Printf("\n    Screening              %10.6f", screener.get());
        outfile->Printf("\n    Added %zu dets of %zu from bin %d", F_space.size(), subspace_size,
                        bin);
    } // End iteration over bins

    outfile->Printf("\n ------------------------------------");
    outfile->Printf("\n  Screened out %1.10f Eh of correlation", total_excluded);
    return total_excluded;
}

/*
double AdaptiveCI::prescreen_F(int bin, int nbin, double E0, psi::SharedMatrix evecs,DeterminantHashVec&
P_space) {

    det_hash<double> ex_f_space;
    local_timer build;

    const size_t n_dets = P_space.size();
    const det_hashvec& dets = P_space.wfn_hash();
    int nmo = fci_ints_->nmo();
    std::vector<int> act_mo = mo_space_info_->get_dimension("ACTIVE").blocks();

    std::vector<det_hash<double>> A_b_t;

        double ret_value = 0.0;
#pragma omp parallel reduction(+:ret_value)
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

#pragma omp critical
        { A_b_t.resize(n_threads); }

        det_hash<double>& A_b = A_b_t[thread_id];

        int bin_size = n_dets / n_threads;
        bin_size += (thread_id < (n_dets % n_threads)) ? 1 : 0;
        int start_idx = (thread_id < (n_dets % n_threads))
                            ? thread_id * bin_size
                            : (n_dets % n_threads) * (bin_size + 1) +
                                  (thread_id - (n_dets % n_threads)) * bin_size;
        int end_idx = start_idx + bin_size;

        // Loop over P space determinants
       // size_t guess_a = nalpha_ * (ncmo_ - nalpha_);
       // size_t guess_b = nbeta_ * (ncmo_ - nbeta_);
       // size_t guess_aa = guess_a * guess_a / 4;
       // size_t guess_bb = guess_b * guess_b / 4;
       // size_t guess_ab = guess_a * guess_b;

       // size_t guess = (n_dets / nbin) * (guess_a + guess_b + guess_aa + guess_bb + guess_ab);
       // outfile->Printf("\n Guessing %zu dets in bin %d", guess, bin);
        //        A_b.reserve(guess);
        for (size_t I = start_idx; I < end_idx; ++I) {
            double c_I = evecs->get(I, 0);
            const Determinant& det = dets[I];
            std::vector<std::vector<int>> noalpha = det.get_asym_occ(nmo, act_mo);
            std::vector<std::vector<int>> nobeta = det.get_bsym_occ(nmo, act_mo);
            std::vector<std::vector<int>> nvalpha = det.get_asym_vir(nmo, act_mo);
            std::vector<std::vector<int>> nvbeta = det.get_bsym_vir(nmo, act_mo);

            Determinant new_det(det);
            // Generate alpha excitations
            for (int h = 0; h < nirrep_; ++h) {
                // Precompute indices
                const auto& noalpha_h = noalpha[h];
                const auto& nvalpha_h = nvalpha[h];

                for (auto& ii : noalpha_h) {
                    new_det.set_alfa_bit(ii, false);
                    for (auto& aa : nvalpha_h) {
                        new_det.set_alfa_bit(aa, true);
                        size_t hash_val = Determinant::Hash()(new_det);
                        if ((hash_val % nbin) == bin) {
                            double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa) * c_I;
                            if ( (std::fabs(HIJ) < screen_thresh_) and (std::fabs(HIJ) >= 1e-12 ) )
{
                                A_b[new_det] += HIJ;
                            }
                        }
                        new_det.set_alfa_bit(aa, false);
                    }
                    new_det.set_alfa_bit(ii, true);
                }
                // Generate beta excitations
                const auto& nobeta_h = nobeta[h];
                const auto& nvbeta_h = nvbeta[h];
                for (auto& ii : nobeta_h) {
                    new_det.set_beta_bit(ii, false);
                    for (auto& aa : nvbeta_h) {
                        new_det.set_beta_bit(aa, true);
                        size_t hash_val = Determinant::Hash()(new_det);
                        if ((hash_val % nbin) == bin) {
                            double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa) * c_I;
                            if ((std::fabs(HIJ) < screen_thresh_) and (std::fabs(HIJ) >= 1e-12 )) {
                                A_b[new_det] += HIJ;
                            }
                        }
                        new_det.set_beta_bit(aa, false);
                    }
                    new_det.set_beta_bit(ii, true);
                }
            }
            for (int p = 0; p < nirrep_; ++p) {
                const auto& noalpha_p = noalpha[p];
                for (int q = p; q < nirrep_; ++q) {
                    const auto& noalpha_q = noalpha[q];
                    for (int r = 0; r < nirrep_; ++r) {
                        int sp = p ^ q ^ r;
                        if (sp < r)
                            continue;

                        // Precompute index lists
                        const auto& nvalpha_r = nvalpha[r];
                        const auto& nvalpha_s = nvalpha[sp];

                        int max_i = noalpha_p.size();
                        int max_j = noalpha_q.size();
                        int max_a = nvalpha_r.size();
                        int max_b = nvalpha_s.size();

                        // Generate aa excitations
                        for (int i = 0; i < max_i; ++i) {
                            int ii = noalpha_p[i];
                            new_det.set_alfa_bit(ii, false);
                            for (int j = (p == q ? i + 1 : 0); j < max_j; ++j) {
                                int jj = noalpha_q[j];
                                new_det.set_alfa_bit(jj, false);
                                for (int a = 0; a < max_a; ++a) {
                                    int aa = nvalpha_r[a];
                                    new_det.set_alfa_bit(aa, true);
                                    for (int b = (r == sp ? a + 1 : 0); b < max_b; ++b) {
                                        int bb = nvalpha_s[b];
                                        new_det.set_alfa_bit(bb, true);
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) < screen_thresh_)and (std::fabs(HIJ)
>= 1e-12 ) ) {
                                                A_b[new_det] +=
                                                    (HIJ * det.slater_sign_aaaa(ii, jj, aa, bb));
                                            }
                                        }
                                        new_det.set_alfa_bit(bb, false);
                                    }
                                    new_det.set_alfa_bit(aa, false);
                                }
                                new_det.set_alfa_bit(jj, true);
                            }
                            new_det.set_alfa_bit(ii, true);
                        }
                        // Generate bb excitations
                        const auto& nobeta_p = nobeta[p];
                        const auto& nobeta_q = nobeta[q];
                        const auto& nvbeta_r = nvbeta[r];
                        const auto& nvbeta_s = nvbeta[sp];

                        max_i = nobeta_p.size();
                        max_j = nobeta_q.size();
                        max_a = nvbeta_r.size();
                        max_b = nvbeta_s.size();

                        for (int i = 0; i < max_i; ++i) {
                            int ii = nobeta_p[i];
                            new_det.set_beta_bit(ii, false);
                            for (int j = (p == q ? i + 1 : 0); j < max_j; ++j) {
                                int jj = nobeta_q[j];
                                new_det.set_beta_bit(jj, false);
                                for (int a = 0; a < max_a; ++a) {
                                    int aa = nvbeta_r[a];
                                    new_det.set_beta_bit(aa, true);
                                    for (int b = (r == sp ? a + 1 : 0); b < max_b; ++b) {
                                        int bb = nvbeta_s[b];
                                        new_det.set_beta_bit(bb, true);
                                        // Check if the determinant goes in this bin
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) < screen_thresh_) and
(std::fabs(HIJ) >= 1e-12 ) ) {
                                                A_b[new_det] +=
                                                    (HIJ * det.slater_sign_bbbb(ii, jj, aa, bb));
                                            }
                                        }
                                        new_det.set_beta_bit(bb, false);
                                    }
                                    new_det.set_beta_bit(aa, false);
                                }
                                new_det.set_beta_bit(jj, true);
                            }
                            new_det.set_beta_bit(ii, true);
                        }
                    }
                }
            }
            for (int p = 0; p < nirrep_; ++p) {
                const auto& noalpha_p = noalpha[p];
                for (int q = 0; q < nirrep_; ++q) {
                    const auto& nobeta_q = nobeta[q];
                    for (int r = 0; r < nirrep_; ++r) {
                        int sp = p ^ q ^ r;
                        const auto& nvalpha_r = nvalpha[r];
                        const auto& nvbeta_s = nvbeta[sp];
                        // Generate ab excitations
                        for (auto& ii : noalpha_p) {
                            new_det.set_alfa_bit(ii, false);
                            for (auto& aa : nvalpha_r) {
                                new_det.set_alfa_bit(aa, true);
                                for (auto& jj : nobeta_q) {
                                    new_det.set_beta_bit(jj, false);
                                    for (auto& bb : nvbeta_s) {
                                        new_det.set_beta_bit(bb, true);
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) < screen_thresh_) and
(std::fabs(HIJ) >= 1e-12 ) ) {
                                                A_b[new_det] +=
                                                    (HIJ * new_det.slater_sign_aa(ii, aa) *
                                                     new_det.slater_sign_bb(jj, bb));
                                            }
                                        }
                                        new_det.set_beta_bit(bb, false);
                                    }
                                    new_det.set_beta_bit(jj, true);
                                }
                                new_det.set_alfa_bit(aa, false);
                            }
                            new_det.set_alfa_bit(ii, true);
                        }
                    }
                }
            }
        } // end loop over reference

        //outfile->Printf("\n  Added %zu dets", A_b.size());

        // Remove duplicates
        for (det_hashvec::iterator it = dets.begin(), endit = dets.end(); it != endit; ++it) {
            A_b_t[thread_id].erase(*it);
        }
        if (thread_id == 0)
            outfile->Printf("\n  Build: %1.6f", build.get());

        local_timer merge;
#pragma omp critical
        {
            for (auto& pair : A_b_t[thread_id]) {
                ex_f_space[pair.first] += pair.second;
            }
        }
#pragma omp barrier
        if (thread_id == 0)
            outfile->Printf("\n  Merge: %1.6f", merge.get());

        // Compute and accumulate energy estimates

        size_t idx = 0;
        for( auto& pair : ex_f_space ){
            if( (idx % n_threads) == thread_id ){
                auto& det = pair.first;
                double& V = pair.second;

                double delta = fci_ints_->energy(det) - E0;

                ret_value += std::fabs(0.5 * (delta - sqrt(delta * delta + V * V * 4.0)));
            }
            idx++;
        }
    } // close threads

    return ret_value;
}

*/
det_hash<double> AdaptiveCI::get_bin_F_space(int bin, int nbin, psi::SharedMatrix evecs,
                                             DeterminantHashVec& P_space) {

    det_hash<double> bin_f_space;
    local_timer build;

    const size_t n_dets = P_space.size();
    const det_hashvec& dets = P_space.wfn_hash();
    std::vector<int> act_mo = mo_space_info_->get_dimension("ACTIVE").blocks();

    std::vector<det_hash<double>> A_b_t;

#pragma omp parallel 
    {
        size_t n_threads = omp_get_num_threads();
        size_t thread_id = omp_get_thread_num();

#pragma omp critical
        {
            A_b_t.resize(n_threads);
            // E_b_t.resize(n_threads);
        }

        det_hash<double>& A_b = A_b_t[thread_id];

        size_t bin_size = n_dets / n_threads;
        bin_size += (thread_id < (n_dets % n_threads)) ? 1 : 0;
        size_t start_idx = (thread_id < (n_dets % n_threads))
                            ? thread_id * bin_size
                            : (n_dets % n_threads) * (bin_size + 1) +
                                  (thread_id - (n_dets % n_threads)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        // Loop over P space determinants
        // size_t guess_a = nalpha_ * (ncmo_ - nalpha_);
        // size_t guess_b = nbeta_ * (ncmo_ - nbeta_);
        // size_t guess_aa = guess_a * guess_a / 4;
        // size_t guess_bb = guess_b * guess_b / 4;
        // size_t guess_ab = guess_a * guess_b;

        // size_t guess = (n_dets / nbin) * (guess_a + guess_b + guess_aa + guess_bb + guess_ab);
        // outfile->Printf("\n Guessing %zu dets in bin %d", guess, bin);
        //        A_b.reserve(guess);
        for (size_t I = start_idx; I < end_idx; ++I) {
            double c_I = evecs->get(I, 0);
            const Determinant& det = dets[I];
            std::vector<std::vector<int>> noalpha = det.get_asym_occ(act_mo);
            std::vector<std::vector<int>> nobeta  = det.get_bsym_occ(act_mo);
            std::vector<std::vector<int>> nvalpha = det.get_asym_vir(act_mo);
            std::vector<std::vector<int>> nvbeta  = det.get_bsym_vir(act_mo);

            Determinant new_det(det);
            // Generate alpha excitations
            for (int h = 0; h < nirrep_; ++h) {
                // Precompute indices
                const auto& noalpha_h = noalpha[h];
                const auto& nvalpha_h = nvalpha[h];

                for (auto& ii : noalpha_h) {
                    new_det.set_alfa_bit(ii, false);
                    for (auto& aa : nvalpha_h) {
                        new_det.set_alfa_bit(aa, true);
                        size_t hash_val = Determinant::Hash()(new_det);
                        if ((hash_val % nbin) == bin) {
                            double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa) * c_I;
                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                A_b[new_det] += HIJ;
                                //} else if (std::fabs(HIJ) >= 1e-12) {
                                //    E_b[new_det] += HIJ;
                            }
                        }
                        new_det.set_alfa_bit(aa, false);
                    }
                    new_det.set_alfa_bit(ii, true);
                }
                // Generate beta excitations
                const auto& nobeta_h = nobeta[h];
                const auto& nvbeta_h = nvbeta[h];
                for (auto& ii : nobeta_h) {
                    new_det.set_beta_bit(ii, false);
                    for (auto& aa : nvbeta_h) {
                        new_det.set_beta_bit(aa, true);
                        size_t hash_val = Determinant::Hash()(new_det);
                        if ((hash_val % nbin) == bin) {
                            double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa) * c_I;
                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                A_b[new_det] += HIJ;
                                //} else if (std::fabs(HIJ) >= 1e-12) {
                                //    E_b[new_det] += HIJ;
                            }
                        }
                        new_det.set_beta_bit(aa, false);
                    }
                    new_det.set_beta_bit(ii, true);
                }
            }
            for (int p = 0; p < nirrep_; ++p) {
                const auto& noalpha_p = noalpha[p];
                for (int q = p; q < nirrep_; ++q) {
                    const auto& noalpha_q = noalpha[q];
                    for (int r = 0; r < nirrep_; ++r) {
                        int sp = p ^ q ^ r;
                        if (sp < r)
                            continue;

                        // Precompute index lists
                        const auto& nvalpha_r = nvalpha[r];
                        const auto& nvalpha_s = nvalpha[sp];

                        int max_i = noalpha_p.size();
                        int max_j = noalpha_q.size();
                        int max_a = nvalpha_r.size();
                        int max_b = nvalpha_s.size();

                        // Generate aa excitations
                        for (int i = 0; i < max_i; ++i) {
                            int ii = noalpha_p[i];
                            new_det.set_alfa_bit(ii, false);
                            for (int j = (p == q ? i + 1 : 0); j < max_j; ++j) {
                                int jj = noalpha_q[j];
                                new_det.set_alfa_bit(jj, false);
                                for (int a = 0; a < max_a; ++a) {
                                    int aa = nvalpha_r[a];
                                    new_det.set_alfa_bit(aa, true);
                                    for (int b = (r == sp ? a + 1 : 0); b < max_b; ++b) {
                                        int bb = nvalpha_s[b];
                                        new_det.set_alfa_bit(bb, true);
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                                A_b[new_det] +=
                                                    (HIJ * det.slater_sign_aaaa(ii, jj, aa, bb));
                                                //} else if (std::fabs(HIJ) >= 1e-12) {
                                                //    E_b[new_det] += HIJ * det.slater_sign_aaaa(ii,
                                                //    jj, aa, bb);
                                            }
                                        }
                                        new_det.set_alfa_bit(bb, false);
                                    }
                                    new_det.set_alfa_bit(aa, false);
                                }
                                new_det.set_alfa_bit(jj, true);
                            }
                            new_det.set_alfa_bit(ii, true);
                        }
                        // Generate bb excitations
                        const auto& nobeta_p = nobeta[p];
                        const auto& nobeta_q = nobeta[q];
                        const auto& nvbeta_r = nvbeta[r];
                        const auto& nvbeta_s = nvbeta[sp];

                        max_i = nobeta_p.size();
                        max_j = nobeta_q.size();
                        max_a = nvbeta_r.size();
                        max_b = nvbeta_s.size();

                        for (int i = 0; i < max_i; ++i) {
                            int ii = nobeta_p[i];
                            new_det.set_beta_bit(ii, false);
                            for (int j = (p == q ? i + 1 : 0); j < max_j; ++j) {
                                int jj = nobeta_q[j];
                                new_det.set_beta_bit(jj, false);
                                for (int a = 0; a < max_a; ++a) {
                                    int aa = nvbeta_r[a];
                                    new_det.set_beta_bit(aa, true);
                                    for (int b = (r == sp ? a + 1 : 0); b < max_b; ++b) {
                                        int bb = nvbeta_s[b];
                                        new_det.set_beta_bit(bb, true);
                                        // Check if the determinant goes in this bin
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                                A_b[new_det] +=
                                                    (HIJ * det.slater_sign_bbbb(ii, jj, aa, bb));
                                                //} else if (std::fabs(HIJ) >= 1e-12) {
                                                //    E_b[new_det] += HIJ * det.slater_sign_bbbb(ii,
                                                //    jj, aa, bb);
                                            }
                                        }
                                        new_det.set_beta_bit(bb, false);
                                    }
                                    new_det.set_beta_bit(aa, false);
                                }
                                new_det.set_beta_bit(jj, true);
                            }
                            new_det.set_beta_bit(ii, true);
                        }
                    }
                }
            }
            for (int p = 0; p < nirrep_; ++p) {
                const auto& noalpha_p = noalpha[p];
                for (int q = 0; q < nirrep_; ++q) {
                    const auto& nobeta_q = nobeta[q];
                    for (int r = 0; r < nirrep_; ++r) {
                        int sp = p ^ q ^ r;
                        const auto& nvalpha_r = nvalpha[r];
                        const auto& nvbeta_s = nvbeta[sp];
                        // Generate ab excitations
                        for (auto& ii : noalpha_p) {
                            new_det.set_alfa_bit(ii, false);
                            for (auto& aa : nvalpha_r) {
                                new_det.set_alfa_bit(aa, true);
                                for (auto& jj : nobeta_q) {
                                    new_det.set_beta_bit(jj, false);
                                    for (auto& bb : nvbeta_s) {
                                        new_det.set_beta_bit(bb, true);
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                                A_b[new_det] +=
                                                    (HIJ * new_det.slater_sign_aa(ii, aa) *
                                                     new_det.slater_sign_bb(jj, bb));
                                                //} else if (std::fabs(HIJ) >= 1e-12) {
                                                //    E_b[new_det] += HIJ *
                                                //    new_det.slater_sign_aa(ii, aa) *
                                                //    new_det.slater_sign_bb(jj, bb);
                                            }
                                        }
                                        new_det.set_beta_bit(bb, false);
                                    }
                                    new_det.set_beta_bit(jj, true);
                                }
                                new_det.set_alfa_bit(aa, false);
                            }
                            new_det.set_alfa_bit(ii, true);
                        }
                    }
                }
            }
        } // end loop over reference

        // outfile->Printf("\n  Added %zu dets", A_b.size());

        // Remove duplicates
        for (det_hashvec::iterator it = dets.begin(), endit = dets.end(); it != endit; ++it) {
            A_b_t[thread_id].erase(*it);
        }
        if (thread_id == 0)
            outfile->Printf("\n  Build: %1.6f", build.get());

        local_timer merge;
#pragma omp critical
        {
            for (auto& pair : A_b_t[thread_id]) {
                bin_f_space[pair.first] += pair.second;
            }
        }
        //#pragma omp critical
        //        {
        //            for (auto& pair : E_b_t[thread_id]) {
        //                bin_E_space[pair.first] += pair.second;
        //            }
        //        }

#pragma omp barrier
        if (thread_id == 0)
            outfile->Printf("\n  Merge: %1.6f", merge.get());
    } // close threads

    return bin_f_space;
}

std::pair<std::vector<std::vector<std::pair<Determinant, double>>>, std::vector<size_t>>
AdaptiveCI::get_bin_F_space_vecsort(int bin, int nbin, psi::SharedMatrix evecs,
                                    DeterminantHashVec& P_space) {

    det_hash<double> bin_f_space;
    local_timer build;
    const size_t n_dets = P_space.size();
    const det_hashvec& dets = P_space.wfn_hash();
    int nmo = fci_ints_->nmo();
    std::vector<int> act_mo = mo_space_info_->get_dimension("ACTIVE").blocks();

    std::vector<std::vector<std::pair<Determinant, double>>> vec_A_b_t;
    std::vector<size_t> dets_t;
#pragma omp parallel
    {
        size_t n_threads = omp_get_num_threads();
        size_t thread_id = omp_get_thread_num();

#pragma omp critical
        {
            vec_A_b_t.resize(n_threads);
            dets_t.resize(n_threads);
        }

        std::vector<std::pair<Determinant, double>>& vec_A_b = vec_A_b_t[thread_id];

        size_t bin_size = n_dets / n_threads;
        bin_size += (thread_id < (n_dets % n_threads)) ? 1 : 0;
        size_t start_idx = (thread_id < (n_dets % n_threads))
                            ? thread_id * bin_size
                            : (n_dets % n_threads) * (bin_size + 1) +
                                  (thread_id - (n_dets % n_threads)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        // Loop over P space determinants
        // size_t guess_a = nalpha_ * (ncmo_ - nalpha_);
        // size_t guess_b = nbeta_ * (ncmo_ - nbeta_);
        // size_t guess_aa = nalpha_ * (nalpha_ - 1) * (ncmo_ - nalpha_) * (ncmo_ - nalpha_ - 1) /
        // 4;
        // size_t guess_bb = nbeta_ * (nbeta_ - 1) * (ncmo_ - nbeta_) * (ncmo_ - nbeta_ - 1) / 4;
        // size_t guess_ab = guess_a * guess_b;

        // size_t guess = (n_dets / nbin) * (guess_a + guess_b + guess_aa + guess_bb + guess_ab);

        size_t nocc2 = nalpha_ * nalpha_;
        size_t nvir2 = (nmo - nalpha_) * (nmo - nalpha_);
        size_t guess = (n_dets / nbin) * nocc2 * nvir2;
        // outfile->Printf("\n Guessing %zu dets in bin %d", guess, bin);
        vec_A_b.reserve(guess);
        //        A_b.reserve(guess);
        for (size_t I = start_idx; I < end_idx; ++I) {
            double c_I = evecs->get(I, 0);
            const Determinant& det = dets[I];
            std::vector<std::vector<int>> noalpha = det.get_asym_occ(act_mo);
            std::vector<std::vector<int>> nobeta  = det.get_bsym_occ(act_mo);
            std::vector<std::vector<int>> nvalpha = det.get_asym_vir(act_mo);
            std::vector<std::vector<int>> nvbeta  = det.get_bsym_vir(act_mo);
            Determinant new_det(det);

            // Generate alpha excitations
            for (int h = 0; h < nirrep_; ++h) {

                // Precompute indices
                const auto& noalpha_h = noalpha[h];
                const auto& nvalpha_h = nvalpha[h];

                for (auto& ii : noalpha_h) {
                    new_det.set_alfa_bit(ii, false);
                    for (auto& aa : nvalpha_h) {
                        new_det.set_alfa_bit(aa, true);
                        size_t hash_val = Determinant::Hash()(new_det);
                        if ((hash_val % nbin) == bin) {
                            double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa) * c_I;
                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                vec_A_b.push_back(std::make_pair(new_det, HIJ));
                            }
                        }
                        new_det.set_alfa_bit(aa, false);
                    }
                    new_det.set_alfa_bit(ii, true);
                }
                // Generate beta excitations
                const auto& nobeta_h = nobeta[h];
                const auto& nvbeta_h = nvbeta[h];
                for (auto& ii : nobeta_h) {
                    new_det.set_beta_bit(ii, false);
                    for (auto& aa : nvbeta_h) {
                        new_det.set_beta_bit(aa, true);
                        size_t hash_val = Determinant::Hash()(new_det);
                        if ((hash_val % nbin) == bin) {
                            double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa) * c_I;
                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                vec_A_b.push_back(std::make_pair(new_det, HIJ));
                            }
                        }
                        new_det.set_beta_bit(aa, false);
                    }
                    new_det.set_beta_bit(ii, true);
                }
            }
            for (int p = 0; p < nirrep_; ++p) {
                const auto& noalpha_p = noalpha[p];
                for (int q = p; q < nirrep_; ++q) {
                    const auto& noalpha_q = noalpha[q];
                    for (int r = 0; r < nirrep_; ++r) {
                        int sp = p ^ q ^ r;
                        if (sp < r)
                            continue;

                        // Precompute index lists
                        const auto& nvalpha_r = nvalpha[r];
                        const auto& nvalpha_s = nvalpha[sp];

                        int max_i = noalpha_p.size();
                        int max_j = noalpha_q.size();
                        int max_a = nvalpha_r.size();
                        int max_b = nvalpha_s.size();

                        // Generate aa excitations
                        for (int i = 0; i < max_i; ++i) {
                            int ii = noalpha_p[i];
                            new_det.set_alfa_bit(ii, false);
                            for (int j = (p == q ? i + 1 : 0); j < max_j; ++j) {
                                int jj = noalpha_q[j];
                                new_det.set_alfa_bit(jj, false);
                                for (int a = 0; a < max_a; ++a) {
                                    int aa = nvalpha_r[a];
                                    new_det.set_alfa_bit(aa, true);
                                    for (int b = (r == sp ? a + 1 : 0); b < max_b; ++b) {
                                        int bb = nvalpha_s[b];
                                        new_det.set_alfa_bit(bb, true);
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                                vec_A_b.push_back(std::make_pair(
                                                    new_det,
                                                    (HIJ * det.slater_sign_aaaa(ii, jj, aa, bb))));
                                            }
                                        }
                                        new_det.set_alfa_bit(bb, false);
                                    }
                                    new_det.set_alfa_bit(aa, false);
                                }
                                new_det.set_alfa_bit(jj, true);
                            }
                            new_det.set_alfa_bit(ii, true);
                        }
                        // Generate bb excitations
                        const auto& nobeta_p = nobeta[p];
                        const auto& nobeta_q = nobeta[q];
                        const auto& nvbeta_r = nvbeta[r];
                        const auto& nvbeta_s = nvbeta[sp];

                        max_i = nobeta_p.size();
                        max_j = nobeta_q.size();
                        max_a = nvbeta_r.size();
                        max_b = nvbeta_s.size();

                        for (int i = 0; i < max_i; ++i) {
                            int ii = nobeta_p[i];
                            new_det.set_beta_bit(ii, false);
                            for (int j = (p == q ? i + 1 : 0); j < max_j; ++j) {
                                int jj = nobeta_q[j];
                                new_det.set_beta_bit(jj, false);
                                for (int a = 0; a < max_a; ++a) {
                                    int aa = nvbeta_r[a];
                                    new_det.set_beta_bit(aa, true);
                                    for (int b = (r == sp ? a + 1 : 0); b < max_b; ++b) {
                                        int bb = nvbeta_s[b];
                                        new_det.set_beta_bit(bb, true);
                                        // Check if the determinant goes in this bin
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                                vec_A_b.push_back(std::make_pair(
                                                    new_det,
                                                    (HIJ * det.slater_sign_bbbb(ii, jj, aa, bb))));
                                            }
                                        }
                                        new_det.set_beta_bit(bb, false);
                                    }
                                    new_det.set_beta_bit(aa, false);
                                }
                                new_det.set_beta_bit(jj, true);
                            }
                            new_det.set_beta_bit(ii, true);
                        }
                    }
                }
            }

            for (int p = 0; p < nirrep_; ++p) {
                const auto& noalpha_p = noalpha[p];
                for (int q = 0; q < nirrep_; ++q) {
                    const auto& nobeta_q = nobeta[q];
                    for (int r = 0; r < nirrep_; ++r) {
                        int sp = p ^ q ^ r;
                        const auto& nvalpha_r = nvalpha[r];
                        const auto& nvbeta_s = nvbeta[sp];
                        // Generate ab excitations
                        for (auto& ii : noalpha_p) {
                            new_det.set_alfa_bit(ii, false);
                            for (auto& aa : nvalpha_r) {
                                new_det.set_alfa_bit(aa, true);
                                for (auto& jj : nobeta_q) {
                                    new_det.set_beta_bit(jj, false);
                                    for (auto& bb : nvbeta_s) {
                                        new_det.set_beta_bit(bb, true);
                                        size_t hash_val = Determinant::Hash()(new_det);
                                        if ((hash_val % nbin) == bin) {
                                            double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb) * c_I;
                                            if ((std::fabs(HIJ) >= screen_thresh_)) {
                                                vec_A_b.push_back(std::make_pair(
                                                    new_det, (HIJ * new_det.slater_sign_aa(ii, aa) *
                                                              new_det.slater_sign_bb(jj, bb))));
                                            }
                                        }
                                        new_det.set_beta_bit(bb, false);
                                    }
                                    new_det.set_beta_bit(jj, true);
                                }
                                new_det.set_alfa_bit(aa, false);
                            }
                            new_det.set_alfa_bit(ii, true);
                        }
                    }
                }
            }
        } // end loop over reference

        size_t num_new_dets = vec_A_b.size();

        //        outfile->Printf("\n  Added %zu dets", A_b.size());
        //        if( thread_id == 0 ){
        //          outfile->Printf("\n  Time spent forming vec_A_b: %1.6f", build.get());
        //          outfile->Printf("\n  Generated %zu determinants out of %zu guessed",
        //          num_new_dets, guess);
        //      }

        // Sort the determinant contributions
        local_timer F_sort;
        //        std::sort(vec_A_b.begin(), vec_A_b.begin() + num_new_dets);

        std::sort(
            vec_A_b.begin(), vec_A_b.begin() + num_new_dets,
            [](const std::pair<Determinant, double>& a,
               const std::pair<Determinant, double>& b) -> bool { return a.first < b.first; });
        // if( thread_id == 0 ){
        //    outfile->Printf("\n  Time spent sorting vec_A_b: %1.6f", F_sort.get());
        //}
        //        for (int i = 0; i < num_new_dets; i++){
        //            auto& p = vec_A_b[i];
        //            outfile->Printf("\n%s %20.12f",p.first.str().c_str(),p.second);
        //          }

        // Combine identical contributions by looping over all elements
        local_timer Combine;
        Determinant d = vec_A_b[0].first;
        double sum = 0.0;
        size_t pos = 0;
        for (size_t I = 0; I < num_new_dets; I++) {
            const Determinant& dI = vec_A_b[I].first;
            const double val = vec_A_b[I].second;
            if (dI == d) {
                sum += val;
            } else {
                vec_A_b[pos] = std::make_pair(d, sum);
                sum = val;
                pos += 1;
                d = dI;
            }
        }
        vec_A_b[pos] = std::make_pair(d, sum);
        pos += (pos == 0) ? 0 : 1;

        //   #pragma omp critical
        //   outfile->Printf("\n  Time spent combining unique elements of vec_A_b: %1.6f",
        //                   Combine.get());

        // store the number of unique determinants
        dets_t[thread_id] = pos;

        if (thread_id == 0)
            outfile->Printf("\n  Build: %1.6f", build.get());

        local_timer merge;

        std::vector<Determinant> sorted_dets(dets.size());
        for (const Determinant& det : dets) {
            sorted_dets.push_back(det);
        }
        std::sort(sorted_dets.begin(), sorted_dets.end());

        // Remove duplicates and merge
        //  #pragma omp critical
        //  {

        size_t ref_size = sorted_dets.size();
        size_t ref_pos = 0;
        size_t det_size = pos;
        size_t det_pos = 0;

        Determinant ref_head = sorted_dets[0];
        Determinant det_head = vec_A_b[0].first;
        // run through the vectors
        while ((ref_pos < ref_size) and (det_pos < det_size)) {
            det_head = vec_A_b[det_pos].first;
            ref_head = sorted_dets[ref_pos];
            if (det_head < ref_head) {
                //                    bin_f_space[det_head] += vec_A_b[det_pos].second;
                det_pos += 1;
            } else if (ref_head < det_head) {
                ref_pos += 1;
            } else {
                // found a match
                vec_A_b[det_pos].second = 0.0;
                det_pos += 1;
                ref_pos += 1;
            }
        }

        //            for (size_t I = det_pos; I < det_size; I++) {
        //                bin_f_space[vec_A_b[I].first] += vec_A_b[I].second;
        //            }
        //     }

        if (thread_id == 0)
            outfile->Printf("\n  Merge: %1.6f", merge.get());
    } // close threads

    // Account for duplicates between threads by adding
    // to thread 0, and zeroing other threads
    local_timer merget;
    int ntd = vec_A_b_t.size();
    if (ntd > 1) {
        // auto& vec_ref = vec_A_b_t[0];
        // for( int td = 1; td < ntd; ++td ){
        for (int td1 = 0; td1 < ntd; ++td1) {
            auto& vec_ref = vec_A_b_t[td1];
            for (int td2 = td1 + 1; td2 < ntd; ++td2) {

                auto& vec_td = vec_A_b_t[td2];

                size_t ref_size = vec_ref.size();
                size_t ref_pos = 0;
                size_t td_size = vec_td.size();
                size_t td_pos = 0;

                Determinant ref_head = vec_ref[0].first;
                Determinant td_head = vec_td[0].first;
                // run through the vectors
                while ((ref_pos < ref_size) and (td_pos < td_size)) {
                    ref_head = vec_ref[ref_pos].first;
                    td_head = vec_td[td_pos].first;
                    if (ref_head < td_head) {
                        ref_pos += 1;
                    } else if (td_head < ref_head) {
                        td_pos += 1;
                    } else {
                        // found a match
                        double& V = vec_td[td_pos].second;

                        vec_ref[ref_pos].second += V;
                        // vec_td[ref_pos].second = 0;
                        V = 0.0;
                        td_pos += 1;
                        ref_pos += 1;
                    }
                }
            }
        }
    }
    outfile->Printf("\n  Inter-thread merge: %1.6f", merget.get());
    return std::make_pair(vec_A_b_t, dets_t);
}

} // namespace forte

