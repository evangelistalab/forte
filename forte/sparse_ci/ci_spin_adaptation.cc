/*
 * @BEGIN LICENSE
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

#include <set>
#include <cmath>
#include <cassert>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"

#include "helpers/timer.h"

#include "sparse_ci/determinant_hashvector.h"
#include "ci_spin_adaptation.h"

namespace forte {

/// @brief A flag to enable/disable debug messages
// constexpr bool DEBUG_SPIN_ADAPTATION = false;

// // #if DEBUG_SPIN_ADAPTATION
// // template <typename... Args> void debug(const std::string& format, Args... args) {
// //     std::string new_format = "[DEBUG] " + format;
// //     psi::outfile->Printf(new_format.c_str(), args...);
// // }
// // #else
// // template <typename... Args> void debug(const std::string& format, Args... args) {}
// // #endif

// Utility functions

/// @brief A function to compute the Clebsch-Gordan coefficient
/// @param twoS Twice the value of S
/// @param twoM Twice the value of M
/// @param dtwoS Twice the change in S
/// @param dtwoM Twice the change in M
double ClebschGordan(double twoS, double twoM, int dtwoS, double dtwoM) {
    if (dtwoS == 1)
        return std::sqrt(0.5 * (twoS + dtwoM * twoM) / twoS);
    if (dtwoS == -1)
        return -dtwoM * std::sqrt(0.5 * (twoS + 2. - dtwoM * twoM) / (twoS + 2.));
    return 0.0;
}

/// @brief A function to compute the overlap between a determinant and a CSF
/// @param N The number of unpaired electrons
/// @param spin_coupling The spin coupling of the CSF (up = 0, down = 1)
/// @param det_occ The spin occupation of the determinant (up = alpha = 0, down = beta = 1)
double overlap(int N, const String& spin_coupling, const String& det_occ) {
    double overlap = 1.0;
    int pi = 0;
    int qi = 0;
    for (int i = 0; i < N; i++) {
        const int dpi = spin_coupling[i];
        const int dqi = det_occ[i];
        int dtwoS = 1 - 2 * dpi;
        int dtwoM = 1 - 2 * dqi;
        pi += dtwoS;
        qi += dtwoM;
        if (std::abs(qi) > pi)
            return 0.0;
        overlap *= ClebschGordan(pi, qi, dtwoS, dtwoM);
    }
    return overlap;
}

// SpinAdapter class

SpinAdapter::SpinAdapter(int twoS, int twoMs, int norb)
    : twoS_(twoS), twoMs_(twoMs), norb_(norb), N_ncsf_(norb + 1, 0),
      N_to_det_occupations_(norb + 1), N_to_overlaps_(norb + 1), N_to_noverlaps_(norb + 1) {}

size_t SpinAdapter::ncsf() const { return ncsf_; }

size_t SpinAdapter::ndet() const { return ndet_; }

void SpinAdapter::det_C_to_csf_C(std::shared_ptr<psi::Vector>& det_C,
                                 std::shared_ptr<psi::Vector>& csf_C) {
    local_timer timer;
    csf_C->zero(); // zero the vector csf_C

    // loop over all the elements of csf_to_det_coeff_ and add the contribution to csf_C
    for (size_t i = 0; i < ncsf_; i++) {
        const auto& start = csf_to_det_bounds_[i];
        const auto& end = csf_to_det_bounds_[i + 1];
        for (size_t j = start; j < end; j++) {
            const auto& [det_idx, coeff] = csf_to_det_coeff_[j];
            csf_C->add(i, coeff * det_C->get(det_idx));
        }
    }
}

void SpinAdapter::csf_C_to_det_C(std::shared_ptr<psi::Vector>& csf_C,
                                 std::shared_ptr<psi::Vector>& det_C) {
    local_timer timer;
    det_C->zero(); // zero the vector det_C

    // loop over all the elements of csf_to_det_coeff_ and add the contribution to det_C
    for (size_t i = 0; i < ncsf_; i++) {
        const auto& start = csf_to_det_bounds_[i];
        const auto& end = csf_to_det_bounds_[i + 1];
        for (size_t j = start; j < end; j++) {
            const auto& [det_idx, coeff] = csf_to_det_coeff_[j];
            det_C->add(det_idx, coeff * csf_C->get(i));
        }
    }
}

auto SpinAdapter::compute_unique_couplings() {
    // compute the number of couplings and CSFs for each allowed value of N
    size_t ncoupling = 0;
    size_t ncsf = 0;
    for (size_t N = 0; N < N_ncsf_.size(); N++) {
        if (N_ncsf_[N] > 0) {
            const auto spin_couplings = make_spin_couplings(N, twoS_);
            const auto determinant_occ = make_determinant_occupations(N, twoMs_);

            std::vector<std::tuple<size_t, size_t, double>> overlaps;
            std::vector<size_t> noverlaps_;

            size_t ncoupling_N = 0;
            size_t ncsf_N = 0;
            for (const auto& spin_coupling : spin_couplings) {
                size_t ndet_N = 0;
                size_t nonzero_overlap = 0;
                for (const auto& det_occ : determinant_occ) {
                    auto o = overlap(N, spin_coupling, det_occ);
                    if (std::fabs(o) > 0.0) {
                        overlaps.push_back(std::make_tuple(ncsf_N, ndet_N, o));
                        nonzero_overlap++;
                    }
                    ndet_N++;
                }
                ncoupling_N += nonzero_overlap;
                noverlaps_.push_back(nonzero_overlap);
                ncsf_N++;
            }
            // save the spin couplings and the determinant occupations
            N_to_det_occupations_[N] = determinant_occ;
            N_to_overlaps_[N] = overlaps;
            N_to_noverlaps_[N] = noverlaps_;
            ncoupling += ncoupling_N * N_ncsf_[N];
            ncsf += ncsf_N * N_ncsf_[N];
        }
    }
    return std::pair(ncoupling, ncsf);
}

void SpinAdapter::prepare_couplings(const std::vector<Determinant>& dets) {
    psi::outfile->Printf("\n\n  ==> Spin Adapter <==\n\n");

    ndet_ = dets.size();
    // build the address of each determinant
    DeterminantHashVec det_hash(dets);

    // find all the configurations
    local_timer t1;
    std::set<Configuration> confs;
    for (const auto& d : dets) {
        confs.insert(Configuration(d));
    }

    // count the configurations with the same number of unpaired electrons (N)
    for (const auto& conf : confs) {
        // exclude configurations with more unpaired electrons than twoS
        if (const auto N = conf.count_socc(); N >= twoS_)
            N_ncsf_[N]++;
    }

    // compute the number of couplings and CSFs for each allowed value of N
    const auto [ncoupling, ncsf] = compute_unique_couplings();

    // allocate memory for the couplings and the starting index of each CSF
    csf_to_det_coeff_.resize(ncoupling);
    csf_to_det_bounds_.resize(ncsf + 1);

    psi::outfile->Printf("    Number of CSFs:                        %10zu\n", ncsf);
    psi::outfile->Printf("    Number of couplings:                   %10zu\n\n", ncoupling);

    confs_ = std::vector<Configuration>(confs.begin(), confs.end());
    psi::outfile->Printf("    Timing for identifying configurations: %10.4f\n", t1.get());

    // loop over all the configurations and find the CSFs
    local_timer t2;
    ncsf_ = 0;
    ncoupling_ = 0;
    for (const auto& conf : confs_) {
        if (conf.count_socc() >= twoS_) {
            conf_to_csfs(conf, det_hash);
        }
    }
    psi::outfile->Printf("    Timing for finding the CSFs:           %10.4f\n", t2.get());

    // check that the number of couplings and CSFs is correct
    assert(ncsf_ == ncsf);
    assert(ncoupling_ == ncoupling);
}

void SpinAdapter::conf_to_csfs(const Configuration& conf, DeterminantHashVec& det_hash) {
    // number of unpaired electrons
    const auto N = conf.count_socc();
    String docc = conf.get_docc_str();
    std::vector<int> socc_vec(norb_);
    conf.get_socc_vec(norb_, socc_vec);

    const auto& determinant_occ = N_to_det_occupations_[N];
    const auto& noverlaps = N_to_noverlaps_[N];

    csf_to_det_bounds_[0] = 0;
    Determinant det;

    size_t temp = ncoupling_;
    for (auto [i, j, o] : N_to_overlaps_[N]) {
        const auto& det_occ = determinant_occ[j];
        det.set_str(docc, docc);
        // keep track of the sign of the singly occupied orbitals
        for (int i = N - 1; i >= 0; i--) {
            if (det_occ.get_bit(i)) {
                o *= det.create_beta_bit(socc_vec[i]);
            } else {
                o *= det.create_alfa_bit(socc_vec[i]);
            }
        }
        csf_to_det_coeff_[ncoupling_].first = det_hash.get_idx(det);
        csf_to_det_coeff_[ncoupling_].second = o;
        ncoupling_ += 1;
    }
    for (const auto& n : noverlaps) {
        temp += n;
        ncsf_ += 1;
        csf_to_det_bounds_[ncsf_] = temp;
    }
}

auto SpinAdapter::make_spin_couplings(int N, int twoS) -> std::vector<String> {
    if (N == 0)
        return std::vector<String>(1, String());
    std::vector<String> couplings;
    auto nup = (N + twoS) / 2;
    String coupling;
    // up = false = 0, down = true = 1
    // The coupling should always start with up
    for (int i = 0; i < nup; i++)
        coupling[i] = false;
    for (int i = nup; i < N; i++)
        coupling[i] = true;
    /// Generate all permutations of the path
    do {
        // check if the path is valid (no negative spin)
        bool valid = true;
        for (int i = 0, p = 0; i < N; i++) {
            p += 1 - 2 * coupling[i];
            if (p < 0)
                valid = false;
        }
        if (valid)
            couplings.push_back(coupling);
        // to keep the first coupling as up we only permute starting from the second element
    } while (std::next_permutation(coupling.begin() + 1, coupling.begin() + N));

    return couplings;
}

auto SpinAdapter::make_determinant_occupations(int N, int twoMs) -> std::vector<String> {
    std::vector<String> det_occs;
    if (N == 0)
        return std::vector<String>(1, String());
    auto nup = (N + twoMs) / 2;
    String det_occ;
    // true = 1 = up, false = 0 = down
    // The det_occ should always start with up
    for (int i = 0; i < nup; i++)
        det_occ[i] = false;
    for (int i = nup; i < N; i++)
        det_occ[i] = true;
    /// Generate all permutations of the path
    do {
        det_occs.push_back(det_occ);
    } while (std::next_permutation(det_occ.begin(), det_occ.begin() + N));
    return det_occs;
}

} // namespace forte
